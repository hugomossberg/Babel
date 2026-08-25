"""
app/core/quota.py
=================
Provider-scoped & Model-scoped daily quota / RPD (Requests-Per-Day) awareness
and Circuit Breaker with Single-Flight Half-Open Probing for Babel.

Architecture & State Machine:
-----------------------------
State transitions:
  ACTIVE
    │  (External daily quota exhausted)
    ▼
  BLOCKED
    │  (now >= next_probe_at)
    ▼
  HALF_OPEN (Single-Flight Probe Lease)
    ├── Success -> ACTIVE (reset probe_attempt to 0, clear probe lease/state)
    └── Daily Quota again -> BLOCKED (probe_attempt += 1, adaptive backoff + jitter)

Design Invariants:
------------------
- All timestamps are UTC-aware (datetime with tzinfo=timezone.utc).
- No naive datetime objects anywhere in this module.
- Single-flight probe lease: exactly ONE real DEFERRED job is leased to probe the provider.
- All competing workers receive DailyQuotaExhaustedError and make 0 provider calls.
- Inactive / no deferred work: 0 dummy requests sent (no "ping" / "hello").
- Adaptive backoff steps for estimated resets:
    attempt 0: ~15m (900s)
    attempt 1: ~30m (1800s)
    attempt 2: ~1h (3600s)
    attempt 3: ~2h (7200s)
    attempt 4: ~4h (14400s)
    attempt 5+: ~6h (21600s max cap)
- Jitter: ±10% on estimated delay, deterministic & testable via fakeable RNG / override.
- Exact provider reset (Retry-After) has priority 1: exact delay + 5s safety margin, no jitter, no adaptive probing before exact time.
- Babels own daily request budget (local RPD) is separate: resets at 00:00 UTC, no adaptive probe.
- Auth / permanent config errors do NOT enter probe loop.
"""

from dataclasses import dataclass
import random
import sqlite3
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Optional, Any, Tuple, Dict, List

from app.core.db import DB_PATH

logger = logging.getLogger("babel.quota")

# ---------------------------------------------------------------------------
# Constants & Tunables
# ---------------------------------------------------------------------------

EXACT_RESET_SAFETY_MARGIN_SECONDS = 5
PROBE_LEASE_TIMEOUT_SECONDS = 300  # 5 minutes before expired probe lease recovers
MAX_BACKOFF_SECONDS = 21600        # 6 hours maximum backoff cap

# Adaptive backoff steps in seconds: [15m, 30m, 1h, 2h, 4h, 6h]
ADAPTIVE_BACKOFF_STEPS = [900, 1800, 3600, 7200, 14400, 21600]

# Global deterministic jitter override (None = random uniform -0.10 to +0.10)
_JITTER_OVERRIDE: Optional[float] = None

def set_jitter_override(ratio: Optional[float]) -> None:
    """Set a deterministic jitter ratio (e.g. 0.0 or 0.05) for testing."""
    global _JITTER_OVERRIDE
    _JITTER_OVERRIDE = ratio

def get_jitter_ratio() -> float:
    """Return jitter ratio in range [-0.10, +0.10]."""
    if _JITTER_OVERRIDE is not None:
        return _JITTER_OVERRIDE
    return random.uniform(-0.10, 0.10)

def calculate_adaptive_probe_delay(attempt: int, jitter_ratio: Optional[float] = None) -> float:
    """
    Calculate adaptive delay in seconds for an unknown/estimated quota reset.
    Guaranteed to be capped at MAX_BACKOFF_SECONDS (plus jitter).
    """
    idx = min(max(0, attempt), len(ADAPTIVE_BACKOFF_STEPS) - 1)
    base_delay = ADAPTIVE_BACKOFF_STEPS[idx]
    
    jr = jitter_ratio if jitter_ratio is not None else get_jitter_ratio()
    # Clamp jitter ratio to [-0.10, 0.10]
    jr = max(-0.10, min(0.10, jr))
    
    delay = base_delay * (1.0 + jr)
    return max(60.0, delay)

# ---------------------------------------------------------------------------
# Standardized Quota Signal & Scopes
# ---------------------------------------------------------------------------

@dataclass
class QuotaSignal:
    provider: str
    kind: str  # "DAILY_QUOTA_EXHAUSTED", "TRANSIENT_RPM", "AUTH_ERROR", "PERMANENT_REQUEST_ERROR", "PROVIDER_UNAVAILABLE", "UNKNOWN_ERROR"
    scope_type: str = "provider"  # "provider" | "model" | "credential"
    scope_id: Optional[str] = None
    model: Optional[str] = None
    retry_after_seconds: Optional[int] = None
    reset_type: str = "estimated"
    raw_message: str = ""

    def __eq__(self, other):
        if isinstance(other, str):
            return self.kind == other
        if isinstance(other, QuotaSignal):
            return self.kind == other.kind and self.provider == other.provider
        return super().__eq__(other)

    def __str__(self):
        return self.kind


def build_scope_key(
    provider: str,
    model: Optional[str] = None,
    scope_type: str = "provider",
    scope_id: Optional[str] = None,
    credential: Optional[str] = None,
) -> str:
    """
    Build a deterministic, canonical scope key for SQLite persistence.
    
    Canonical Formats:
    - Provider scope (default):             "{provider}" (e.g. "gemini", "openrouter", "openai")
    - Credential-scoped Model:              "{provider}:credential:{cred_id}:model:{model_id}" (e.g. "openrouter:credential:key-a:model:claude")
    - Provider-global Model:                "{provider}:model:{model_id}" (e.g. "openrouter:model:claude")
    - Credential scope:                     "{provider}:credential:{cred_id}" (e.g. "openrouter:credential:key-a")
    - Unknown scope:                        "{provider}" (conservative fallback to parent provider scope)
    """
    p = (provider or "").strip().lower()
    st = (scope_type or "provider").strip().lower()
    c_id = (credential or "").strip().lower()

    if st == "model":
        m_id = (scope_id or model or "").strip().lower()
        if m_id:
            if m_id.startswith(f"{p}:"):
                return m_id
            if c_id:
                return f"{p}:credential:{c_id}:model:{m_id}"
            return f"{p}:model:{m_id}"
        if c_id:
            return f"{p}:credential:{c_id}"
        return p

    elif st == "credential":
        c_id = (credential or scope_id or "default").strip().lower()
        if c_id.startswith(f"{p}:credential:"):
            return c_id
        return f"{p}:credential:{c_id}"

    # "provider", "unknown", or any unrecognized scope_type -> provider scope
    return p

# Canonical alias
resolve_scope_key = build_scope_key


def get_parent_scope_keys(
    provider: str,
    model: Optional[str] = None,
    scope_type: str = "provider",
    scope_id: Optional[str] = None,
    credential: Optional[str] = None,
) -> List[str]:
    """
    Return all parent scope keys that must be ACTIVE for a given request context.
    Hierarchical order:
      1. Provider scope: "{provider}"
      2. Credential scope: "{provider}:credential:{credential_id}" (if credential is provided)
      3. Provider-global model scope: "{provider}:model:{model_id}" (if model is specified with credential)
    """
    parents = []
    p = (provider or "").strip().lower()
    if not p:
        return parents

    provider_key = p
    target_key = build_scope_key(provider, model=model, scope_type=scope_type, scope_id=scope_id, credential=credential)

    # 1. Provider-level parent
    if target_key != provider_key and provider_key not in parents:
        parents.append(provider_key)

    # 2. Credential-level parent
    cred = (credential or (scope_id if scope_type == "credential" else None) or "").strip().lower()
    if cred:
        cred_key = f"{p}:credential:{cred}"
        if target_key != cred_key and cred_key not in parents:
            parents.append(cred_key)

    # 3. Provider-global model parent (if request is credential-scoped model)
    m_id = (scope_id or model or "").strip().lower()
    if m_id and cred:
        global_model_key = f"{p}:model:{m_id}"
        if target_key != global_model_key and global_model_key not in parents:
            parents.append(global_model_key)

    return parents


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class QuotaError(Exception):
    """Base for all quota-related exceptions."""

class DailyQuotaExhaustedError(QuotaError):
    """
    Raised when a provider has confirmed daily RPD quota is exhausted.
    The caller MUST NOT retry immediately; instead it should defer the job.
    """
    def __init__(self, provider: str, retry_after_seconds: Optional[int] = None, raw_message: str = ""):
        self.provider = provider
        self.retry_after_seconds = retry_after_seconds
        self.raw_message = raw_message
        super().__init__(f"Daily quota exhausted for provider '{provider}': {raw_message}")


class RequestBudgetExhaustedError(QuotaError):
    """
    Raised when the user-configured daily request budget has been reached
    for this provider. The job should be deferred, not failed.
    """
    def __init__(self, provider: str, used: int, budget: int):
        self.provider = provider
        self.used = used
        self.budget = budget
        super().__init__(
            f"Daily request budget reached for provider '{provider}': {used}/{budget} requests used today."
        )


# ---------------------------------------------------------------------------
# Error classification logic
# ---------------------------------------------------------------------------

_DAILY_QUOTA_PATTERNS = [
    r"quota_id.*per.*day",
    r"daily.*quota.*exceeded",
    r"quota.*per.*day.*exceeded",
    r"per.*day.*quota",
    r"daily_request_limit_exceeded",
    r"dailyrequestlimitexceeded",
    r"ratequotaexceeded",
    r"resource_exhausted.*per.*day",
    r"resource_exhausted.*day",
    r"quota.*for.*the.*day",
    r"exceeded.*your.*quota.*for",
    r"exceeded.*quota.*for.*day",
    r"your.*quota.*day",
    r"gemini.*free.*tier.*limit",
    r"free_tier.*quota",
    r"rate_limit_exceeded.*per.*day",
    r"requests_per_day",
    r"rpd",
    r"daily.*limit.*reached",
    r"day.*limit.*exceeded",
    r"per.*day.*limit",
    r"exceeded.*daily.*limit",
    r"exceeded.*day.*quota",
    r"quota.*day",
]

_DAILY_QUOTA_COMPILED = [re.compile(p, re.IGNORECASE) for p in _DAILY_QUOTA_PATTERNS]

_RPM_ONLY_PATTERNS = [
    r"per.*minute",
    r"requests.*per.*minute",
    r"rpm",
    r"minute.*quota",
    r"rate.*limit.*minute",
    r"too.*many.*requests.*minute",
    r"tpm",
    r"tokens.*per.*minute",
]

_RPM_ONLY_COMPILED = [re.compile(p, re.IGNORECASE) for p in _RPM_ONLY_PATTERNS]

_AUTH_PATTERNS = [
    r"\b401\b", r"\b403\b",
    r"unauthorized", r"forbidden",
    r"api.key.not.valid", r"invalid.api.key",
    r"permission_denied", r"api_key_invalid",
    r"authentication", r"unauthenticated",
    r"not.configured",
]

_AUTH_COMPILED = [re.compile(p, re.IGNORECASE) for p in _AUTH_PATTERNS]


def classify_provider_error(
    exc: Exception,
    provider: str,
    *,
    model: Optional[str] = None,
    retry_after_seconds: Optional[int] = None,
) -> QuotaSignal:
    """
    Classify a provider exception into a standardized QuotaSignal.
    """
    err_str = str(exc)
    err_lower = err_str.lower()

    # Determine scope if explicitly model-specific
    scope_type = "provider"
    scope_id = provider.lower()
    if model and any(p in err_lower for p in ["model_quota", "per_model", "quota for model"]):
        scope_type = "model"
        scope_id = f"{provider.lower()}:{model.lower()}"

    # 1. Auth errors
    for pat in _AUTH_COMPILED:
        if pat.search(err_str):
            logger.debug("classify_provider_error: AUTH_ERROR detected for %s: %s", provider, err_str[:120])
            return QuotaSignal(
                provider=provider,
                kind="AUTH_ERROR",
                scope_type=scope_type,
                scope_id=scope_id,
                model=model,
                raw_message=err_str,
            )

    # 2. Permanent request errors (400 / 404 / invalid model)
    permanent_patterns = [
        r"\b400\b", r"\b404\b",
        r"model_not_found", r"invalid.model",
        r"permission.denied", r"not.found",
        r"bad.request",
    ]
    for pat in permanent_patterns:
        if re.search(pat, err_str, re.IGNORECASE):
            if not any(p.search(err_lower) for p in _DAILY_QUOTA_COMPILED):
                logger.debug("classify_provider_error: PERMANENT_REQUEST_ERROR for %s", provider)
                return QuotaSignal(
                    provider=provider,
                    kind="PERMANENT_REQUEST_ERROR",
                    scope_type=scope_type,
                    scope_id=scope_id,
                    model=model,
                    raw_message=err_str,
                )

    # 3. Daily quota patterns
    has_daily = any(p.search(err_lower) for p in _DAILY_QUOTA_COMPILED)
    has_rpm_only = any(p.search(err_lower) for p in _RPM_ONLY_COMPILED)

    if has_daily and not has_rpm_only:
        reset_type = "exact" if (retry_after_seconds is not None and retry_after_seconds > 0) else "estimated"
        logger.info(
            "classify_provider_error: DAILY_QUOTA_EXHAUSTED for provider=%s (retry_after=%s, reset_type=%s)",
            provider, retry_after_seconds, reset_type,
        )
        return QuotaSignal(
            provider=provider,
            kind="DAILY_QUOTA_EXHAUSTED",
            scope_type=scope_type,
            scope_id=scope_id,
            model=model,
            retry_after_seconds=retry_after_seconds,
            reset_type=reset_type,
            raw_message=err_str,
        )

    # 4. Transient indicators (RPM / 5xx / timeouts)
    transient_indicators = [
        r"\b429\b", r"rate.limit", r"resource_exhausted",
        r"quota", r"too.many.requests",
        r"\b500\b", r"\b502\b", r"\b503\b", r"\b504\b",
        r"timeout", r"connection", r"service.unavailable", r"overloaded",
    ]
    for pat in transient_indicators:
        if re.search(pat, err_lower, re.IGNORECASE):
            logger.debug("classify_provider_error: TRANSIENT_RPM for provider=%s", provider)
            return QuotaSignal(
                provider=provider,
                kind="TRANSIENT_RPM",
                scope_type=scope_type,
                scope_id=scope_id,
                model=model,
                retry_after_seconds=retry_after_seconds,
                raw_message=err_str,
            )

    logger.debug("classify_provider_error: UNKNOWN_ERROR for provider=%s: %s", provider, err_str[:80])
    return QuotaSignal(
        provider=provider,
        kind="UNKNOWN_ERROR",
        scope_type=scope_type,
        scope_id=scope_id,
        model=model,
        raw_message=err_str,
    )


# ---------------------------------------------------------------------------
# DB schema helpers
# ---------------------------------------------------------------------------

def _ensure_quota_table(conn: sqlite3.Connection) -> None:
    """Create or migrate provider_quota and daily_request_counts tables."""
    conn.execute("""
    CREATE TABLE IF NOT EXISTS provider_quota (
        provider            TEXT PRIMARY KEY,
        state               TEXT NOT NULL DEFAULT 'ACTIVE',
        blocked             INTEGER NOT NULL DEFAULT 0,
        reason              TEXT,
        blocked_at          TEXT,
        blocked_until       TEXT,
        reset_type          TEXT DEFAULT 'estimated',
        probe_attempt       INTEGER NOT NULL DEFAULT 0,
        probe_lease_until   TEXT,
        probe_lease_owner   TEXT,
        last_probe_at       TEXT,
        scope_type          TEXT DEFAULT 'provider',
        scope_id            TEXT,
        updated_at          TEXT NOT NULL
    )
    """)

    # Column migrations for existing database files
    for col, col_type in [
        ("state", "TEXT NOT NULL DEFAULT 'ACTIVE'"),
        ("reset_type", "TEXT DEFAULT 'estimated'"),
        ("probe_attempt", "INTEGER NOT NULL DEFAULT 0"),
        ("probe_lease_until", "TEXT"),
        ("probe_lease_owner", "TEXT"),
        ("last_probe_at", "TEXT"),
        ("scope_type", "TEXT DEFAULT 'provider'"),
        ("scope_id", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE provider_quota ADD COLUMN {col} {col_type}")
        except Exception:
            pass

    conn.execute("""
    CREATE TABLE IF NOT EXISTS daily_request_counts (
        provider        TEXT NOT NULL,
        window_date     TEXT NOT NULL,
        request_count   INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (provider, window_date)
    )
    """)


# ---------------------------------------------------------------------------
# UTC Datetime helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(iso: Optional[str]) -> Optional[datetime]:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Single-Flight Dispatch Slot & Circuit Breaker Lease
# ---------------------------------------------------------------------------

def acquire_dispatch_slot(
    provider: str,
    model: Optional[str] = None,
    job_id: Optional[Any] = None,
    scope_type: str = "provider",
    scope_id: Optional[str] = None,
    credential: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Atomic single-flight quota & circuit breaker gate.
    
    Returns:
      (True, {"is_probe": bool, "state": str}) -> Slot acquired, proceed with provider request.
      (False, {"reason": str, "blocked_until": str, "reset_type": str, "state": str}) -> Blocked, do NOT make request.
    """
    scope_key = resolve_scope_key(provider, model, scope_type, scope_id, credential=credential)
    now = _utcnow()
    now_iso = now.isoformat()

    try:
        conn = sqlite3.connect(DB_PATH, timeout=15.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("BEGIN EXCLUSIVE")
            _ensure_quota_table(conn)

            # 1. Check all hierarchical parent scopes (provider, credential)
            parents = get_parent_scope_keys(provider, model=model, scope_type=scope_type, scope_id=scope_id, credential=credential)
            for parent_key in parents:
                p_row = conn.execute("""
                SELECT state, blocked, reason, blocked_until, reset_type, probe_lease_until
                FROM provider_quota WHERE provider = ?
                """, (parent_key,)).fetchone()
                if p_row and p_row[1] == 1:
                    p_until = _parse_utc(p_row[3])
                    p_lease = _parse_utc(p_row[5])
                    if (p_until and now < p_until) or (p_lease and now < p_lease):
                        conn.execute("COMMIT")
                        return False, {
                            "reason": p_row[2] or f"Parent scope '{parent_key}' is blocked",
                            "blocked_until": p_row[3],
                            "reset_type": p_row[4] or "estimated",
                            "state": p_row[0] or "BLOCKED",
                            "is_parent_blocked": True,
                            "parent_scope": parent_key,
                        }

            # 2. Check target scope state
            row = conn.execute("""
            SELECT state, blocked, reason, blocked_until, reset_type, probe_attempt, probe_lease_until, probe_lease_owner
            FROM provider_quota WHERE provider = ?
            """, (scope_key,)).fetchone()

            if not row or (row[0] == "ACTIVE" and row[1] == 0):
                # State is ACTIVE: check local request budget
                if not _try_consume_budget_conn(conn, provider):
                    conn.execute("COMMIT")
                    return False, {
                        "reason": "REQUEST_BUDGET_EXHAUSTED",
                        "state": "ACTIVE",
                    }
                conn.execute("COMMIT")
                return True, {"is_probe": False, "state": "ACTIVE"}

            state, blocked, reason, blocked_until_str, reset_type, probe_attempt, probe_lease_until_str, probe_lease_owner = row
            blocked_until = _parse_utc(blocked_until_str)

            # If next_probe_at (blocked_until) has NOT been reached yet:
            if blocked_until and now < blocked_until:
                conn.execute("COMMIT")
                return False, {
                    "reason": reason or "Daily quota exhausted",
                    "blocked_until": blocked_until_str,
                    "reset_type": reset_type or "estimated",
                    "state": state or "BLOCKED",
                    "probe_attempt": probe_attempt,
                }

            # next_probe_at has passed (now >= next_probe_at) -> Eligible for HALF_OPEN probe!
            probe_lease_until = _parse_utc(probe_lease_until_str)
            if probe_lease_until and now < probe_lease_until:
                # Another worker currently holds the single-flight probe lease!
                conn.execute("COMMIT")
                return False, {
                    "reason": "Probe request currently in flight",
                    "blocked_until": blocked_until_str,
                    "reset_type": reset_type or "estimated",
                    "state": "HALF_OPEN",
                    "probe_attempt": probe_attempt,
                }

            # NO active probe lease (or lease has expired) -> CLAIM THE LEASE!
            if not _try_consume_budget_conn(conn, provider):
                conn.execute("COMMIT")
                return False, {
                    "reason": "REQUEST_BUDGET_EXHAUSTED",
                    "state": "HALF_OPEN",
                }

            new_lease_expiry = now + timedelta(seconds=PROBE_LEASE_TIMEOUT_SECONDS)
            conn.execute("""
            UPDATE provider_quota
            SET state = 'HALF_OPEN',
                probe_lease_until = ?,
                probe_lease_owner = ?,
                last_probe_at = ?,
                updated_at = ?
            WHERE provider = ?
            """, (new_lease_expiry.isoformat(), str(job_id or ''), now_iso, now_iso, scope_key))
            conn.commit()

            logger.info("Claimed single-flight probe lease for scope '%s' (job_id=%s, lease_until=%s)",
                        scope_key, job_id, new_lease_expiry.isoformat())
            return True, {"is_probe": True, "state": "HALF_OPEN", "probe_attempt": probe_attempt}

        except Exception:
            try: conn.execute("ROLLBACK")
            except Exception: pass
            raise
        finally:
            conn.close()
    except Exception as e:
        logger.error("acquire_dispatch_slot DB error for %s: %s", provider, e)
        # Fail-open
        return True, {"is_probe": False, "state": "ACTIVE"}


def record_provider_success(
    provider: str,
    model: Optional[str] = None,
    scope_type: str = "provider",
    scope_id: Optional[str] = None,
    credential: Optional[str] = None,
) -> None:
    """
    Called upon successful response from provider.
    Transitions circuit breaker from HALF_OPEN / BLOCKED to ACTIVE.
    Resets probe_attempt and clears probe leases.
    """
    scope_key = resolve_scope_key(provider, model, scope_type, scope_id, credential=credential)
    now_iso = _utcnow().isoformat()
    try:
        with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
            _ensure_quota_table(conn)
            row = conn.execute("SELECT state, blocked FROM provider_quota WHERE provider = ?", (scope_key,)).fetchone()
            if row and (row[0] != "ACTIVE" or row[1] != 0):
                conn.execute("""
                UPDATE provider_quota
                SET state = 'ACTIVE',
                    blocked = 0,
                    reason = NULL,
                    blocked_until = NULL,
                    probe_attempt = 0,
                    probe_lease_until = NULL,
                    probe_lease_owner = NULL,
                    updated_at = ?
                WHERE provider = ?
                """, (now_iso, scope_key))
                conn.commit()
                logger.info("Circuit breaker for scope '%s' successfully reset to ACTIVE.", scope_key)
    except Exception as e:
        logger.error("record_provider_success error for %s: %s", scope_key, e)


def record_provider_quota_exhausted(
    provider: str,
    reason: str,
    *,
    retry_after_seconds: Optional[int] = None,
    model: Optional[str] = None,
    scope_type: str = "provider",
    scope_id: Optional[str] = None,
    credential: Optional[str] = None,
    jitter_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Called when a provider reports DAILY_QUOTA_EXHAUSTED.
    Transitions circuit breaker to BLOCKED and calculates next_probe_at.
    """
    scope_key = resolve_scope_key(provider, model, scope_type, scope_id, credential=credential)
    now = _utcnow()
    now_iso = now.isoformat()

    # Priority 1: Exact reset metadata from provider
    if retry_after_seconds is not None and retry_after_seconds > 0:
        delay_seconds = float(retry_after_seconds + EXACT_RESET_SAFETY_MARGIN_SECONDS)
        blocked_until = now + timedelta(seconds=delay_seconds)
        reset_type = "exact"
        next_attempt = 0
        logger.info(
            "Circuit breaker '%s' BLOCKED until %s (Retry-After: %ds + %ds safety margin, exact reset). Reason: %s",
            scope_key, blocked_until.isoformat(), retry_after_seconds, EXACT_RESET_SAFETY_MARGIN_SECONDS, reason,
        )
    else:
        # Priority 2: Adaptive backoff with jitter
        current_attempt = 0
        try:
            with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
                _ensure_quota_table(conn)
                r = conn.execute("SELECT probe_attempt FROM provider_quota WHERE provider = ?", (scope_key,)).fetchone()
                if r and r[0] is not None:
                    current_attempt = r[0]
        except Exception:
            pass

        delay_seconds = calculate_adaptive_probe_delay(current_attempt, jitter_ratio)
        blocked_until = now + timedelta(seconds=delay_seconds)
        reset_type = "estimated"
        next_attempt = current_attempt + 1
        logger.warning(
            "Circuit breaker '%s' BLOCKED until %s (adaptive probe attempt %d -> delay %ds). Reason: %s",
            scope_key, blocked_until.isoformat(), current_attempt, int(delay_seconds), reason,
        )

    blocked_until_iso = blocked_until.isoformat()

    try:
        with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
            _ensure_quota_table(conn)
            conn.execute("""
            INSERT INTO provider_quota (
                provider, state, blocked, reason, blocked_at, blocked_until,
                reset_type, probe_attempt, probe_lease_until, probe_lease_owner,
                scope_type, scope_id, updated_at
            )
            VALUES (?, 'BLOCKED', 1, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?)
            ON CONFLICT(provider) DO UPDATE SET
                state             = 'BLOCKED',
                blocked           = 1,
                reason            = excluded.reason,
                blocked_at        = excluded.blocked_at,
                blocked_until     = excluded.blocked_until,
                reset_type        = excluded.reset_type,
                probe_attempt     = excluded.probe_attempt,
                probe_lease_until = NULL,
                probe_lease_owner = NULL,
                scope_type        = excluded.scope_type,
                scope_id          = excluded.scope_id,
                updated_at        = excluded.updated_at
            """, (
                scope_key, reason, now_iso, blocked_until_iso,
                reset_type, next_attempt, scope_type, scope_id or model or credential or provider, now_iso
            ))
            conn.commit()
    except Exception as e:
        logger.error("record_provider_quota_exhausted DB error for %s: %s", scope_key, e)

    return {
        "provider": provider,
        "scope_key": scope_key,
        "blocked_until": blocked_until_iso,
        "reset_type": reset_type,
        "probe_attempt": next_attempt,
        "delay_seconds": delay_seconds,
    }


def block_provider(
    provider: str,
    reason: str,
    *,
    retry_after_seconds: Optional[int] = None,
    reset_type: Optional[str] = None,
    model: Optional[str] = None,
    scope_type: str = "provider",
    scope_id: Optional[str] = None,
    credential: Optional[str] = None,
    jitter_ratio: Optional[float] = None,
) -> None:
    """Backwards-compatible wrapper for record_provider_quota_exhausted."""
    record_provider_quota_exhausted(
        provider=provider,
        reason=reason,
        retry_after_seconds=retry_after_seconds,
        model=model,
        scope_type=scope_type,
        scope_id=scope_id,
        credential=credential,
        jitter_ratio=jitter_ratio,
    )


def unblock_provider(
    provider: str,
    model: Optional[str] = None,
    scope_type: str = "provider",
    scope_id: Optional[str] = None,
    credential: Optional[str] = None,
) -> None:
    """Manually unblock a provider or scope."""
    scope_key = resolve_scope_key(provider, model, scope_type, scope_id, credential=credential)
    try:
        with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
            _ensure_quota_table(conn)
            _unblock_provider_conn(conn, scope_key)
            conn.commit()
            logger.info("Scope '%s' manually unblocked.", scope_key)
    except Exception as e:
        logger.error("unblock_provider DB error for %s: %s", scope_key, e)


def _unblock_provider_conn(conn: sqlite3.Connection, scope_key: str) -> None:
    """Unblock scope using an existing connection (no commit)."""
    now_iso = _utcnow().isoformat()
    conn.execute("""
    UPDATE provider_quota
    SET state = 'ACTIVE',
        blocked = 0,
        blocked_until = NULL,
        reset_type = NULL,
        reason = NULL,
        blocked_at = NULL,
        probe_attempt = 0,
        probe_lease_until = NULL,
        probe_lease_owner = NULL,
        updated_at = ?
    WHERE provider = ?
    """, (now_iso, scope_key))


def is_provider_blocked(
    provider: str,
    model: Optional[str] = None,
    scope_type: str = "provider",
    scope_id: Optional[str] = None,
    credential: Optional[str] = None,
) -> bool:
    """
    Return True if provider is currently in daily-quota block AND NOT eligible for probing.
    
    Specifically:
    - Returns False if state is ACTIVE.
    - Returns True if state is BLOCKED and now < next_probe_at.
    - Returns True if state is HALF_OPEN and active probe lease is held (probe in flight).
    - Returns False if state is BLOCKED and now >= next_probe_at (eligible for a DEFERRED job to probe).
    """
    scope_key = build_scope_key(provider, model=model, scope_type=scope_type, scope_id=scope_id, credential=credential)
    now = _utcnow()
    try:
        with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
            _ensure_quota_table(conn)

            # 1. Check all hierarchical parent scopes (provider, credential)
            parents = get_parent_scope_keys(provider, model=model, scope_type=scope_type, scope_id=scope_id, credential=credential)
            for parent_key in parents:
                p_row = conn.execute(
                    "SELECT state, blocked, blocked_until, probe_lease_until FROM provider_quota WHERE provider = ?",
                    (parent_key,)
                ).fetchone()
                if p_row and p_row[1] == 1:
                    p_until = _parse_utc(p_row[2])
                    p_lease = _parse_utc(p_row[3])
                    if (p_until and now < p_until) or (p_lease and now < p_lease):
                        return True

            # 2. Check target scope
            row = conn.execute(
                "SELECT state, blocked, blocked_until, probe_lease_until FROM provider_quota WHERE provider = ?",
                (scope_key,)
            ).fetchone()

            if not row or (row[0] == "ACTIVE" and row[1] == 0):
                return False

            state, blocked, blocked_until_str, probe_lease_until_str = row
            blocked_until = _parse_utc(blocked_until_str)
            probe_lease_until = _parse_utc(probe_lease_until_str)

            # If blocked_until is in the future -> blocked
            if blocked_until and now < blocked_until:
                return True

            # If probe lease is active -> probe in flight, other callers are blocked
            if probe_lease_until and now < probe_lease_until:
                return True

            # now >= blocked_until and no active probe lease -> eligible for probe dispatch!
            return False
    except Exception as e:
        logger.error("is_provider_blocked error for %s: %s", scope_key, e)
        return False


def get_provider_block_info(
    provider: str,
    model: Optional[str] = None,
    scope_type: str = "provider",
    scope_id: Optional[str] = None,
    credential: Optional[str] = None,
) -> dict:
    """
    Return comprehensive block state dict for API/UI use.
    """
    scope_key = build_scope_key(provider, model=model, scope_type=scope_type, scope_id=scope_id, credential=credential)
    now = _utcnow()
    try:
        with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
            _ensure_quota_table(conn)

            # Check all parent scopes (provider, credential)
            parents = get_parent_scope_keys(provider, model=model, scope_type=scope_type, scope_id=scope_id, credential=credential)
            for parent_key in parents:
                p_row = conn.execute("""
                SELECT state, blocked, reason, blocked_at, blocked_until, reset_type, probe_attempt, probe_lease_until, updated_at
                FROM provider_quota WHERE provider = ?
                """, (parent_key,)).fetchone()
                if p_row and p_row[1] == 1:
                    p_until = _parse_utc(p_row[4])
                    p_lease = _parse_utc(p_row[7])
                    if (p_until and now < p_until) or (p_lease and now < p_lease):
                        return {
                            "provider": provider,
                            "scope_key": scope_key,
                            "state": "BLOCKED",
                            "blocked": True,
                            "reason": p_row[2] or f"Parent scope '{parent_key}' is blocked",
                            "blocked_at": p_row[3],
                            "blocked_until": p_row[4],
                            "reset_type": p_row[5],
                            "probe_attempt": p_row[6],
                            "probe_lease_active": bool(p_lease and now < p_lease),
                            "updated_at": p_row[8],
                            "is_parent_blocked": True,
                            "parent_scope": parent_key,
                        }

            row = conn.execute("""
            SELECT state, blocked, reason, blocked_at, blocked_until, reset_type, probe_attempt, probe_lease_until, updated_at
            FROM provider_quota WHERE provider = ?
            """, (scope_key,)).fetchone()
    except Exception as e:
        logger.error("get_provider_block_info error: %s", e)
        row = None

    if not row:
        return {
            "provider": provider,
            "scope_key": scope_key,
            "state": "ACTIVE",
            "blocked": False,
            "reason": None,
            "blocked_at": None,
            "blocked_until": None,
            "reset_type": None,
            "probe_attempt": 0,
            "probe_lease_active": False,
            "updated_at": None,
        }

    state, blocked, reason, blocked_at, blocked_until_str, reset_type, probe_attempt, probe_lease_until_str, updated_at = row
    blocked_until = _parse_utc(blocked_until_str)
    probe_lease_until = _parse_utc(probe_lease_until_str)

    probe_lease_active = bool(probe_lease_until and now < probe_lease_until)
    currently_blocked = bool(blocked) and (
        (blocked_until and now < blocked_until) or probe_lease_active
    )

    current_state = "ACTIVE"
    if probe_lease_active:
        current_state = "HALF_OPEN"
    elif currently_blocked:
        current_state = "BLOCKED"
    elif blocked_until and now >= blocked_until:
        current_state = "HALF_OPEN"

    return {
        "provider": provider,
        "scope_key": scope_key,
        "state": current_state,
        "blocked": currently_blocked,
        "reason": reason,
        "blocked_at": blocked_at,
        "blocked_until": blocked_until_str,
        "reset_type": reset_type if (currently_blocked or current_state == "HALF_OPEN") else None,
        "probe_attempt": probe_attempt or 0,
        "probe_lease_active": probe_lease_active,
        "updated_at": updated_at,
    }


# ---------------------------------------------------------------------------
# Daily request budget / counter (Local Babel RPD)
# ---------------------------------------------------------------------------

def _today_window(provider: str) -> str:
    """Return the UTC date string for current budget window (YYYY-MM-DD)."""
    return _utcnow().strftime("%Y-%m-%d")


def get_daily_budget(provider: str) -> Optional[int]:
    """Return configured daily budget or None if unlimited."""
    from app.core.db import get_setting
    raw = get_setting(f"daily_request_budget_{provider}", "0").strip()
    if not raw or raw == "0":
        return None
    try:
        v = int(raw)
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


def get_daily_requests_used(provider: str) -> int:
    """Return how many requests have been dispatched today for this provider."""
    window = _today_window(provider)
    try:
        with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
            _ensure_quota_table(conn)
            row = conn.execute(
                "SELECT request_count FROM daily_request_counts WHERE provider = ? AND window_date = ?",
                (provider, window)
            ).fetchone()
            return row[0] if row else 0
    except Exception as e:
        logger.error("get_daily_requests_used error: %s", e)
        return 0


def _try_consume_budget_conn(conn: sqlite3.Connection, provider: str) -> bool:
    """Helper to check and increment budget using an existing EXCLUSIVE connection."""
    budget = get_daily_budget(provider)
    if budget is None:
        return True

    window = _today_window(provider)
    row = conn.execute(
        "SELECT request_count FROM daily_request_counts WHERE provider = ? AND window_date = ?",
        (provider, window)
    ).fetchone()

    current = row[0] if row else 0
    if current >= budget:
        logger.warning(
            "Daily request budget exhausted for provider '%s': %d/%d used today.",
            provider, current, budget,
        )
        return False

    conn.execute("""
    INSERT INTO daily_request_counts (provider, window_date, request_count)
    VALUES (?, ?, 1)
    ON CONFLICT(provider, window_date) DO UPDATE SET
        request_count = request_count + 1
    """, (provider, window))
    return True


def try_consume_request_budget(provider: str) -> bool:
    """
    Atomically check and increment the daily request counter for *provider*.
    Uses SQLite EXCLUSIVE transaction.
    """
    budget = get_daily_budget(provider)
    if budget is None:
        return True

    try:
        conn = sqlite3.connect(DB_PATH, timeout=15.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("BEGIN EXCLUSIVE")
            _ensure_quota_table(conn)
            allowed = _try_consume_budget_conn(conn, provider)
            conn.execute("COMMIT")
            return allowed
        except Exception:
            try: conn.execute("ROLLBACK")
            except Exception: pass
            raise
        finally:
            conn.close()
    except Exception as e:
        logger.error("try_consume_request_budget error for %s: %s", provider, e)
        return True


def is_local_budget_available(provider: str) -> bool:
    """Return True if local request budget allows requests (unlimited or used < budget)."""
    budget = get_daily_budget(provider)
    if budget is None:
        return True
    used = get_daily_requests_used(provider)
    return used < budget


def should_retry_deferred_job(job: dict, now: Optional[datetime] = None) -> bool:
    """
    Determine if a DEFERRED job is eligible for retry in the scheduler pass.

    Rules:
    1. Use the job's PINNED provider (waiting_provider -> primary_provider -> global fallback).
       This prevents re-assignment when global settings change.
    2. If the pinned provider is currently blocked by external daily quota / circuit breaker:
       -> Do NOT retry yet (respect circuit breaker).
    3. If job has structured defer_reason == LOCAL_RPD or INSUFFICIENT_LOCAL_BUDGET,
       OR legacy text-pattern match:
       -> If local budget is currently available (user raised budget, set unlimited,
          or day rolled over), retry immediately without waiting for old midnight next_retry_at.
       -> If local budget is still exhausted, respect next_retry_at (midnight reset).
    4. For other DEFERRED jobs (e.g. external quota reset):
       -> Retry once now >= next_retry_at.
    """
    if job.get("status") != "DEFERRED":
        return False

    # Resolve pinned provider — job's own config, not global setting
    pinned_provider = (
        job.get("waiting_provider")
        or job.get("primary_provider")
        or None
    )
    if pinned_provider:
        pinned_provider = pinned_provider.strip().lower()
    else:
        # Legacy fallback: no pinned provider yet — use global setting conservatively
        from app.core.db import get_setting
        pinned_provider = get_setting("ai_provider", "gemini").lower()

    if is_provider_blocked(pinned_provider):
        return False

    if now is None:
        now = _utcnow()

    # Structured defer reason (new jobs) takes priority
    defer_reason = (job.get("defer_reason") or "").strip()
    LOCAL_BUDGET_REASONS = {"LOCAL_RPD", "INSUFFICIENT_LOCAL_BUDGET"}

    if defer_reason in LOCAL_BUDGET_REASONS:
        if is_local_budget_available(pinned_provider):
            return True
        if job.get("next_retry_at"):
            try:
                nra = datetime.fromisoformat(job["next_retry_at"])
                if nra.tzinfo is None:
                    nra = nra.replace(tzinfo=timezone.utc)
                return now >= nra
            except Exception:
                return False
        return False

    if not defer_reason:
        # Legacy text-pattern fallback for old jobs without defer_reason column
        err_text = f"{job.get('last_error', '')} {job.get('error_message', '')}".lower()
        is_local_budget_deferral = ("budget" in err_text or "requestbudgetexhausted" in err_text)
        if is_local_budget_deferral:
            if is_local_budget_available(pinned_provider):
                return True
            if job.get("next_retry_at"):
                try:
                    nra = datetime.fromisoformat(job["next_retry_at"])
                    if nra.tzinfo is None:
                        nra = nra.replace(tzinfo=timezone.utc)
                    return now >= nra
                except Exception:
                    return False
            return False

    # External quota / PROVIDER_QUOTA / QUEUE_BACKLOG / ESCALATION deferrals:
    # require now >= next_retry_at
    if job.get("next_retry_at"):
        try:
            nra = datetime.fromisoformat(job["next_retry_at"])
            if nra.tzinfo is None:
                nra = nra.replace(tzinfo=timezone.utc)
            return now >= nra
        except Exception:
            return False

    return False


def check_minimum_budget_admission(
    provider: str,
    num_cues: int,
    batch_size: int,
) -> dict:
    """
    Pre-flight check for a NEW primary translation job.
    Computes the minimum number of primary batch requests required (ceil(cues/batch))
    and checks whether the current remaining local budget can cover them.

    Returns a dict:
      {
        "admitted": bool,         # True = may proceed, False = must defer
        "estimated_minimum": int, # ceil(num_cues / batch_size)
        "available": int|None,    # remaining budget (None = unlimited)
        "reason": str,            # DeferReason constant if not admitted
      }

    Notes:
    - This is ONLY for primary translation. QA/repair/escalation are NOT estimated.
    - For resumed/partial jobs, callers should NOT use this (they know remaining work).
    - Budget 0 = Unlimited => always admitted.
    """
    import math
    budget = get_daily_budget(provider)
    estimated_minimum = math.ceil(num_cues / batch_size) if batch_size > 0 else num_cues

    if budget is None:
        # Unlimited
        return {
            "admitted": True,
            "estimated_minimum": estimated_minimum,
            "available": None,
            "reason": "",
        }

    used = get_daily_requests_used(provider)
    available = max(0, budget - used)

    if available < estimated_minimum:
        return {
            "admitted": False,
            "estimated_minimum": estimated_minimum,
            "available": available,
            "reason": "INSUFFICIENT_LOCAL_BUDGET",
        }

    return {
        "admitted": True,
        "estimated_minimum": estimated_minimum,
        "available": available,
        "reason": "",
    }


def get_quota_status_for_provider(provider: str, model: Optional[str] = None) -> dict:
    """
    Return a unified quota status dict for API/UI consumption.
    """
    block_info = get_provider_block_info(provider, model=model)
    budget = get_daily_budget(provider)
    used = get_daily_requests_used(provider)
    remaining = (budget - used) if budget is not None else None

    return {
        "provider": provider,
        "scope_key": block_info["scope_key"],
        "state": block_info["state"],
        "blocked": block_info["blocked"],
        "reason": block_info["reason"],
        "blocked_until": block_info["blocked_until"],
        "reset_type": block_info["reset_type"],
        "probe_attempt": block_info["probe_attempt"],
        "probe_lease_active": block_info["probe_lease_active"],
        "budget": budget,
        "requests_today": used,
        "requests_remaining": remaining,
    }


# ---------------------------------------------------------------------------
# Retry-After extraction helpers
# ---------------------------------------------------------------------------

def extract_retry_after_from_exception(exc: Exception) -> Optional[int]:
    """
    Extract a Retry-After value (in seconds) from a provider exception.
    """
    # 1. Direct attribute
    for attr in ("retry_after", "retry_delay", "retry_delay_seconds"):
        val = getattr(exc, attr, None)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass

    # 2. Response headers on the exception object
    for attr in ("response", "http_response", "_response"):
        resp = getattr(exc, attr, None)
        if resp is not None:
            headers = getattr(resp, "headers", None)
            if headers:
                ra = headers.get("Retry-After") or headers.get("retry-after")
                if ra:
                    try:
                        return int(ra)
                    except (TypeError, ValueError):
                        pass

    # 3. String parsing (last resort)
    err_str = str(exc)
    match = re.search(r"[Rr]etry[_\-][Aa]fter[:\s]+(\d+)", err_str)
    if match:
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            pass

    return None
