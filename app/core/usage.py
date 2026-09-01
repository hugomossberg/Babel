"""
app/core/usage.py
=================
Phase 2: AI Usage Ledger — Token Accounting, Cost Estimation, and Usage Summaries.

Design principles:
- One usage row per real provider dispatch attempt (success or failure).
- Zero rows for requests blocked before dispatch (RPD/quota/circuit breaker).
- request_uid UNIQUE constraint prevents accidental double-counting.
- Provider/model stored as plain strings — no enum constraints.
- All tokens/costs are nullable: NULL means unknown, 0 means actual zero.
- estimated_cost_usd is always an ESTIMATE, never invoice truth.
- Usage ledger is observability/accounting only. It does NOT replace quota enforcement.

Pricing registry:
- All prices verified against official provider documentation (audit date: 2026-08-24).
- Time-sensitive promotional pricing uses effective_until/effective_from semantics.
- Unknown model -> estimated_cost_usd = NULL (never a fake value).
- DeepL and Ollama: no token pricing, cost always NULL.

Token semantics (Gemini and OpenAI):
- input_tokens = TOTAL prompt tokens INCLUDING cached subset.
- cached_input_tokens = cached subset of input_tokens (billed at lower rate).
- uncached_input = max(input_tokens - cached_input_tokens, 0)
- output_tokens = candidate/completion output tokens EXCLUDING thinking tokens.
- thinking_tokens = reasoning/thinking tokens (Gemini: thoughts_token_count;
  OpenAI: stored separately if applicable). NULL = provider does not report thinking.
- billable_output = output_tokens + thinking_tokens (both billed at output rate).
- cost = uncached_input * input_rate + cached * cached_rate + billable_output * output_rate
- No double-counting: cached tokens NOT also billed at full input_rate.
"""

import sqlite3
import uuid
import logging
from datetime import datetime, timezone, date
from typing import Optional, Dict, Any, List

from app.core.db import DB_PATH

logger = logging.getLogger("babel.usage")


class UsageStage:
    PRIMARY = "PRIMARY"
    MICRO_REPAIR = "MICRO_REPAIR"
    RECOVERY = "RECOVERY"
    ESCALATION = "ESCALATION"
    CLASSIFIER = "CLASSIFIER"
    ENTITY_VERIFY = "ENTITY_VERIFY"
    SEMANTIC_AUDIT = "SEMANTIC_AUDIT"
    SUBTITLE_TRUST_AUDIT = "SUBTITLE_TRUST_AUDIT"


class UsageStatus:
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PENDING = "PENDING"


def _migrate_usage_schema(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS ai_usage_ledger (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        request_uid          TEXT    NOT NULL,
        job_id               INTEGER,
        provider             TEXT    NOT NULL,
        model                TEXT    NOT NULL,
        stage                TEXT    NOT NULL,
        status               TEXT    NOT NULL DEFAULT 'PENDING',
        input_tokens         INTEGER,
        cached_input_tokens  INTEGER,
        output_tokens        INTEGER,
        thinking_tokens      INTEGER,
        estimated_cost_usd   REAL,
        error_type           TEXT,
        created_at           TEXT    NOT NULL,
        completed_at         TEXT,
        FOREIGN KEY (job_id) REFERENCES jobs(id)
    )
    """)
    # Idempotent migration: add thinking_tokens column to existing databases
    try:
        conn.execute("ALTER TABLE ai_usage_ledger ADD COLUMN thinking_tokens INTEGER")
    except Exception:
        pass  # Column already exists
    conn.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS idx_ai_usage_request_uid
        ON ai_usage_ledger (request_uid)
    """)
    conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_ai_usage_job_id
        ON ai_usage_ledger (job_id)
    """)
    conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_ai_usage_created_at
        ON ai_usage_ledger (created_at)
    """)
    conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_ai_usage_provider_created
        ON ai_usage_ledger (provider, created_at)
    """)


def init_usage_schema():
    try:
        with sqlite3.connect(DB_PATH, timeout=30.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            _migrate_usage_schema(conn)
            conn.commit()
        logger.info("AI usage ledger schema initialized/verified.")
    except Exception as e:
        logger.error("Failed to initialize usage ledger schema: %s", e)
        raise


# ---------------------------------------------------------------------------
# Pricing Registry
# ---------------------------------------------------------------------------
# All prices are USD per 1,000,000 tokens.
# Verification date: 2026-08-24 against official provider documentation.
#
# Fields per entry:
#   provider:              str
#   model:                 str
#   input_per_1m:          float | None
#   cached_input_per_1m:   float | None
#   output_per_1m:         float | None
#   effective_from:        "YYYY-MM-DD" | None  (inclusive)
#   effective_until:       "YYYY-MM-DD" | None  (inclusive, None=indefinite)
#
# Models NOT in registry (get_pricing returns None -> NULL cost):
#   gemini-3.5-pro:   not GA as of 2026-08-24
#   gemini-4.0-flash: not released as of 2026-08-24
#   gpt-5.6-turbo:    not a valid OpenAI model ID
#   DeepL, Ollama:    no token pricing available from provider
# ---------------------------------------------------------------------------

_PRICING_TABLE = [
    # Gemini 3.5 Flash-Lite: $0.30/$0.03/$2.50 per 1M (verified 2026-08-24)
    {"provider": "gemini", "model": "gemini-3.5-flash-lite",
     "input_per_1m": 0.30, "cached_input_per_1m": 0.03, "output_per_1m": 2.50,
     "effective_from": None, "effective_until": None},
    {"provider": "gemini", "model": "gemini-3.5-flash-lite-001",
     "input_per_1m": 0.30, "cached_input_per_1m": 0.03, "output_per_1m": 2.50,
     "effective_from": None, "effective_until": None},

    # Gemini 3.5 Flash: $1.50/$0.15/$9.00 per 1M (verified 2026-08-24)
    {"provider": "gemini", "model": "gemini-3.5-flash",
     "input_per_1m": 1.50, "cached_input_per_1m": 0.15, "output_per_1m": 9.00,
     "effective_from": None, "effective_until": None},

    # Gemini 3.6 Flash: promotional $0.75/$0.075/$3.75 until 2026-12-31
    {"provider": "gemini", "model": "gemini-3.6-flash",
     "input_per_1m": 0.75, "cached_input_per_1m": 0.075, "output_per_1m": 3.75,
     "effective_from": None, "effective_until": "2026-12-31"},
    # Gemini 3.6 Flash: standard $1.50/$0.15/$7.50 from 2027-01-01
    {"provider": "gemini", "model": "gemini-3.6-flash",
     "input_per_1m": 1.50, "cached_input_per_1m": 0.15, "output_per_1m": 7.50,
     "effective_from": "2027-01-01", "effective_until": None},

    # Gemini 3.7 Flash: promotional $0.75/$0.075/$3.75 until 2026-12-31
    {"provider": "gemini", "model": "gemini-3.7-flash",
     "input_per_1m": 0.75, "cached_input_per_1m": 0.075, "output_per_1m": 3.75,
     "effective_from": None, "effective_until": "2026-12-31"},
    # Gemini 3.7 Flash: standard $1.50/$0.15/$7.50 from 2027-01-01
    {"provider": "gemini", "model": "gemini-3.7-flash",
     "input_per_1m": 1.50, "cached_input_per_1m": 0.15, "output_per_1m": 7.50,
     "effective_from": "2027-01-01", "effective_until": None},

    # Legacy Gemini models
    {"provider": "gemini", "model": "gemini-1.5-pro",
     "input_per_1m": 3.50, "cached_input_per_1m": 0.875, "output_per_1m": 10.50,
     "effective_from": None, "effective_until": None},
    {"provider": "gemini", "model": "gemini-1.5-flash",
     "input_per_1m": 0.075, "cached_input_per_1m": 0.01875, "output_per_1m": 0.30,
     "effective_from": None, "effective_until": None},

    # OpenAI gpt-4o-mini: $0.15/$0.075/$0.60 per 1M (verified 2026-08-24)
    {"provider": "openai", "model": "gpt-4o-mini",
     "input_per_1m": 0.15, "cached_input_per_1m": 0.075, "output_per_1m": 0.60,
     "effective_from": None, "effective_until": None},

    # OpenAI gpt-4o: $2.50/$1.25/$10.00 per 1M (verified 2026-08-24)
    {"provider": "openai", "model": "gpt-4o",
     "input_per_1m": 2.50, "cached_input_per_1m": 1.25, "output_per_1m": 10.00,
     "effective_from": None, "effective_until": None},
    {"provider": "openai", "model": "gpt-4o-2024-11-20",
     "input_per_1m": 2.50, "cached_input_per_1m": 1.25, "output_per_1m": 10.00,
     "effective_from": None, "effective_until": None},
    {"provider": "openai", "model": "gpt-4o-2024-08-06",
     "input_per_1m": 2.50, "cached_input_per_1m": 1.25, "output_per_1m": 10.00,
     "effective_from": None, "effective_until": None},

    # OpenAI o1-mini: $1.10/$0.55/$4.40 per 1M (verified 2026-08-24)
    {"provider": "openai", "model": "o1-mini",
     "input_per_1m": 1.10, "cached_input_per_1m": 0.55, "output_per_1m": 4.40,
     "effective_from": None, "effective_until": None},

    # OpenAI gpt-4-turbo: $10.00/None/$30.00 per 1M (no cached discount listed)
    {"provider": "openai", "model": "gpt-4-turbo",
     "input_per_1m": 10.00, "cached_input_per_1m": None, "output_per_1m": 30.00,
     "effective_from": None, "effective_until": None},
    {"provider": "openai", "model": "gpt-4-turbo-preview",
     "input_per_1m": 10.00, "cached_input_per_1m": None, "output_per_1m": 30.00,
     "effective_from": None, "effective_until": None},

    # OpenAI GPT-4.1 family (current production as of 2026)
    {"provider": "openai", "model": "gpt-4.1",
     "input_per_1m": 2.00, "cached_input_per_1m": 0.50, "output_per_1m": 8.00,
     "effective_from": None, "effective_until": None},
    {"provider": "openai", "model": "gpt-4.1-mini",
     "input_per_1m": 0.40, "cached_input_per_1m": 0.10, "output_per_1m": 1.60,
     "effective_from": None, "effective_until": None},
    {"provider": "openai", "model": "gpt-4.1-nano",
     "input_per_1m": 0.10, "cached_input_per_1m": 0.025, "output_per_1m": 0.40,
     "effective_from": None, "effective_until": None},

    # OpenAI o4-mini
    {"provider": "openai", "model": "o4-mini",
     "input_per_1m": 1.10, "cached_input_per_1m": 0.275, "output_per_1m": 4.40,
     "effective_from": None, "effective_until": None},
]


def _parse_pricing_date(s):
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except Exception:
        return None


def get_pricing(provider, model, at_date=None):
    """
    Look up pricing for a (provider, model) pair at a specific date.

    at_date: the date for time-sensitive pricing (UTC). Defaults to today UTC.
    Returns None if no matching entry — caller must use NULL cost.
    """
    if not provider or not model:
        return None

    p = provider.lower().strip()
    m = model.lower().strip()

    if at_date is None:
        at_date = datetime.now(timezone.utc).date()

    def _matches_date(entry):
        ef = _parse_pricing_date(entry.get("effective_from"))
        eu = _parse_pricing_date(entry.get("effective_until"))
        if ef is not None and at_date < ef:
            return False
        if eu is not None and at_date > eu:
            return False
        return True

    def _build(entry):
        return {
            "input_per_1m": entry.get("input_per_1m"),
            "cached_input_per_1m": entry.get("cached_input_per_1m"),
            "output_per_1m": entry.get("output_per_1m"),
        }

    # Pass 1: exact match
    for entry in _PRICING_TABLE:
        if entry["provider"] != p:
            continue
        if entry["model"] != m:
            continue
        if not _matches_date(entry):
            continue
        return _build(entry)

    # No fallback: unknown model/alias -> None (caller must use NULL cost).
    # All version variants we price are explicit in _PRICING_TABLE (e.g. gemini-3.5-flash-lite-001).
    # Prefix/suffix matching was removed because it incorrectly assigned sibling-model pricing
    # to unverified models (e.g. gemini-3.5-flash-turbo -> gemini-3.5-flash rates).
    return None


def calculate_estimated_cost(provider, model, input_tokens, cached_input_tokens, output_tokens,
                              thinking_tokens=None, at_date=None):
    """
    Calculate estimated cost in USD.

    Returns None if provider/model unknown or all tokens None.

    Cost formula:
      uncached_input  = max(input_tokens - cached_input_tokens, 0)
      billable_output = output_tokens + thinking_tokens  (both billed at output_rate)
      cost = uncached_input * input_rate + cached * cached_rate + billable_output * output_rate

    output_tokens:   candidate/completion output EXCLUDING thinking tokens.
    thinking_tokens: reasoning tokens (Gemini: thoughts_token_count). NULL = 0 for billing.
    No double-counting: cached tokens NOT also billed at full input_rate.
    """
    pricing = get_pricing(provider, model, at_date=at_date)
    if pricing is None:
        return None

    if input_tokens is None and output_tokens is None and thinking_tokens is None:
        return None

    input_rate = pricing.get("input_per_1m")
    cached_rate = pricing.get("cached_input_per_1m")
    output_rate = pricing.get("output_per_1m")

    total_cost = 0.0

    if input_tokens is not None and input_rate is not None:
        if cached_input_tokens and cached_rate is not None:
            non_cached = max(0, input_tokens - cached_input_tokens)
            total_cost += (non_cached / 1_000_000.0) * input_rate
            total_cost += (cached_input_tokens / 1_000_000.0) * cached_rate
        else:
            total_cost += (input_tokens / 1_000_000.0) * input_rate

    # billable_output = output_tokens + thinking_tokens (both at output rate)
    billable_output = (output_tokens or 0) + (thinking_tokens or 0)
    if billable_output > 0 and output_rate is not None:
        total_cost += (billable_output / 1_000_000.0) * output_rate
    elif output_tokens is None and thinking_tokens is None:
        pass  # no output tokens at all — don't add zero cost

    return total_cost


def extract_gemini_usage(response):
    """
    Extract token usage from Gemini SDK GenerateContentResponse.

    Token semantics:
      prompt_token_count:           total prompt tokens (includes cached subset)
      cached_content_token_count:   cached subset (billed at lower rate)
      candidates_token_count:       candidate output tokens EXCLUDING thinking tokens
      thoughts_token_count:         thinking/reasoning tokens (Gemini thinking models only)
                                    NULL/absent if model does not use thinking.

    billable_output = candidates_token_count + thoughts_token_count
    Both are billed at the same output_per_1m rate.

    cached_input_tokens=None if cached_content_token_count <= 0 (no caching used).
    thinking_tokens=None if thoughts_token_count is absent or zero (model has no thinking).
    """
    result = {"input_tokens": None, "cached_input_tokens": None, "output_tokens": None, "thinking_tokens": None}
    try:
        um = getattr(response, "usage_metadata", None)
        if um is None:
            return result
        pt = getattr(um, "prompt_token_count", None)
        if pt is not None:
            result["input_tokens"] = int(pt)
        cc = getattr(um, "cached_content_token_count", None)
        if cc is not None and int(cc) > 0:
            result["cached_input_tokens"] = int(cc)
        ct = getattr(um, "candidates_token_count", None)
        if ct is not None:
            result["output_tokens"] = int(ct)
        # thoughts_token_count is available in Gemini thinking models
        # (e.g. gemini-3.7-flash with thinking enabled). Absent in non-thinking models.
        tt = getattr(um, "thoughts_token_count", None)
        if tt is None:
            # Try camelCase variant used in some SDK versions
            tt = getattr(um, "thoughtsTokenCount", None)
        if tt is not None and int(tt) > 0:
            result["thinking_tokens"] = int(tt)
    except Exception as e:
        logger.debug("extract_gemini_usage: parse error: %s", e)
    return result


def extract_openai_usage(response):
    """
    Extract token usage from OpenAI SDK ChatCompletion or dict response.

    prompt_tokens: total prompt tokens (includes cached subset)
    completion_tokens: output tokens INCLUDING reasoning tokens
    prompt_tokens_details.cached_tokens / prompt_cache_hit_tokens: cached subset

    cached_input_tokens=None if cached_tokens <= 0 (no caching used).
    thinking_tokens: always None for OpenAI (reasoning is included in completion_tokens).
    """
    result = {"input_tokens": None, "cached_input_tokens": None, "output_tokens": None, "thinking_tokens": None}
    try:
        if isinstance(response, dict):
            usage = response.get("usage")
        else:
            usage = getattr(response, "usage", None)
        if usage is None:
            return result

        if isinstance(usage, dict):
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            ptd = usage.get("prompt_tokens_details") or {}
            cached = ptd.get("cached_tokens") if isinstance(ptd, dict) else getattr(ptd, "cached_tokens", None)
        else:
            pt = getattr(usage, "prompt_tokens", None)
            ct = getattr(usage, "completion_tokens", None)
            ptd = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(ptd, "cached_tokens", None) if ptd is not None else None

        if pt is not None:
            result["input_tokens"] = int(pt)
        if ct is not None:
            result["output_tokens"] = int(ct)
        if cached is not None and int(cached) > 0:
            result["cached_input_tokens"] = int(cached)
    except Exception as e:
        logger.debug("extract_openai_usage: parse error: %s", e)
    return result


def extract_anthropic_usage(response):
    """
    Extract token usage from Anthropic Messages API response (dict or object).

    input_tokens: total input prompt tokens
    output_tokens: generated completion tokens
    cache_read_input_tokens: cached prompt tokens (discounted)
    """
    result = {"input_tokens": None, "cached_input_tokens": None, "output_tokens": None, "thinking_tokens": None}
    try:
        if isinstance(response, dict):
            usage = response.get("usage")
        else:
            usage = getattr(response, "usage", None)
        if usage is None:
            return result

        if isinstance(usage, dict):
            it = usage.get("input_tokens")
            ot = usage.get("output_tokens")
            cached = usage.get("cache_read_input_tokens")
        else:
            it = getattr(usage, "input_tokens", None)
            ot = getattr(usage, "output_tokens", None)
            cached = getattr(usage, "cache_read_input_tokens", None)

        if it is not None:
            result["input_tokens"] = int(it)
        if ot is not None:
            result["output_tokens"] = int(ot)
        if cached is not None and int(cached) > 0:
            result["cached_input_tokens"] = int(cached)
    except Exception as e:
        logger.debug("extract_anthropic_usage: parse error: %s", e)
    return result



def extract_ollama_usage(response):
    result = {"input_tokens": None, "cached_input_tokens": None, "output_tokens": None, "thinking_tokens": None}
    try:
        if isinstance(response, dict):
            pt = response.get("prompt_eval_count")
            ct = response.get("eval_count")
            if pt is not None:
                result["input_tokens"] = int(pt)
            if ct is not None:
                result["output_tokens"] = int(ct)
    except Exception:
        pass
    return result

def extract_usage_from_response(provider, response):
    """Dispatch to correct provider usage adapter. Unknown providers return all-None."""
    p = (provider or "").lower().strip()
    if p == "gemini":
        return extract_gemini_usage(response)
    elif p in ("openai", "openrouter", "deepseek", "custom", "custom_openai"):
        return extract_openai_usage(response)
    elif p == "anthropic":
        return extract_anthropic_usage(response)
    elif p in ("ollama", "localai"):
        return extract_ollama_usage(response)
    else:
        return {"input_tokens": None, "cached_input_tokens": None, "output_tokens": None, "thinking_tokens": None}


def generate_request_uid():
    return str(uuid.uuid4())


def record_dispatch(request_uid, provider, model, stage, job_id=None, created_at=None):
    """
    Insert PENDING row BEFORE dispatching to provider.
    Returns True on success, False on DB error or duplicate uid.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    if created_at is None:
        created_at = now_iso
    try:
        with sqlite3.connect(DB_PATH, timeout=15.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.execute(
                """INSERT OR IGNORE INTO ai_usage_ledger
                   (request_uid, job_id, provider, model, stage, status, created_at)
                   VALUES (?, ?, ?, ?, ?, 'PENDING', ?)""",
                (request_uid, job_id, provider, model, stage, created_at),
            )
            conn.commit()
            if cursor.rowcount == 0:
                logger.warning("Usage ledger: duplicate request_uid=%s (idempotency guard)", request_uid)
                return False
            return True
    except Exception as e:
        logger.error("Usage ledger: record_dispatch failed for %s: %s", request_uid, e)
        return False


def complete_dispatch(request_uid, status, input_tokens=None, cached_input_tokens=None,
                      output_tokens=None, thinking_tokens=None, estimated_cost_usd=None, error_type=None):
    """Update PENDING row with final outcome and token metadata.

    thinking_tokens: reasoning/thinking token count (Gemini only); NULL if not applicable.
    estimated_cost_usd: computed from billable_output = output_tokens + thinking_tokens.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        with sqlite3.connect(DB_PATH, timeout=15.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.execute(
                """UPDATE ai_usage_ledger
                   SET status=?, input_tokens=?, cached_input_tokens=?, output_tokens=?,
                       thinking_tokens=?, estimated_cost_usd=?, error_type=?, completed_at=?
                   WHERE request_uid=?""",
                (status, input_tokens, cached_input_tokens, output_tokens,
                 thinking_tokens, estimated_cost_usd, error_type, now_iso, request_uid),
            )
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        logger.error("Usage ledger: complete_dispatch failed for %s: %s", request_uid, e)
        return False


def cancel_dispatch(request_uid):
    """
    Delete PENDING row when pre-SDK exception detected.
    Enforces: one row = one real provider request attempt.
    """
    try:
        with sqlite3.connect(DB_PATH, timeout=15.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.execute(
                "DELETE FROM ai_usage_ledger WHERE request_uid=? AND status='PENDING'",
                (request_uid,),
            )
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        logger.error("Usage ledger: cancel_dispatch failed for %s: %s", request_uid, e)
        return False


def _utc_today_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def get_job_usage_summary(job_id):
    try:
        with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT stage, provider, model, status,
                          input_tokens, cached_input_tokens, output_tokens, thinking_tokens, estimated_cost_usd
                   FROM ai_usage_ledger
                   WHERE job_id=? AND status!='PENDING'
                   ORDER BY id""",
                (job_id,),
            ).fetchall()
    except Exception as e:
        logger.error("get_job_usage_summary: DB error for job_id=%s: %s", job_id, e)
        return _empty_job_summary(job_id)

    if not rows:
        return _empty_job_summary(job_id)

    total_calls = len(rows)
    total_input = total_cached = total_output = total_thinking = total_cost = None
    by_stage = {}
    by_provider = {}
    by_provider_model = {}
    detail_dict = {}

    for row in rows:
        stage = row["stage"]
        provider = row["provider"]
        model = row["model"]
        pm_key = f"{provider}/{model}"
        detail_key = (stage, provider, model)

        for bucket_key, bucket_dict in [(stage, by_stage), (provider, by_provider), (pm_key, by_provider_model)]:
            if bucket_key not in bucket_dict:
                bucket_dict[bucket_key] = _empty_bucket()
            b = bucket_dict[bucket_key]
            b["calls"] += 1
            b["input_tokens"] = _nullable_add(b["input_tokens"], row["input_tokens"])
            b["cached_input_tokens"] = _nullable_add(b["cached_input_tokens"], row["cached_input_tokens"])
            b["output_tokens"] = _nullable_add(b["output_tokens"], row["output_tokens"])
            b["thinking_tokens"] = _nullable_add(b["thinking_tokens"], row["thinking_tokens"])
            b["estimated_cost_usd"] = _nullable_add(b["estimated_cost_usd"], row["estimated_cost_usd"])

        if detail_key not in detail_dict:
            detail_dict[detail_key] = {"stage": stage, "provider": provider, "model": model, **_empty_bucket()}
        d = detail_dict[detail_key]
        d["calls"] += 1
        d["input_tokens"] = _nullable_add(d["input_tokens"], row["input_tokens"])
        d["cached_input_tokens"] = _nullable_add(d["cached_input_tokens"], row["cached_input_tokens"])
        d["output_tokens"] = _nullable_add(d["output_tokens"], row["output_tokens"])
        d["thinking_tokens"] = _nullable_add(d["thinking_tokens"], row["thinking_tokens"])
        d["estimated_cost_usd"] = _nullable_add(d["estimated_cost_usd"], row["estimated_cost_usd"])

        total_input = _nullable_add(total_input, row["input_tokens"])
        total_cached = _nullable_add(total_cached, row["cached_input_tokens"])
        total_output = _nullable_add(total_output, row["output_tokens"])
        total_thinking = _nullable_add(total_thinking, row["thinking_tokens"])
        total_cost = _nullable_add(total_cost, row["estimated_cost_usd"])

    return {
        "job_id": job_id,
        "total_calls": total_calls,
        "total_input_tokens": total_input,
        "total_cached_input_tokens": total_cached,
        "total_output_tokens": total_output,
        "total_thinking_tokens": total_thinking,
        "total_estimated_cost_usd": total_cost,
        "breakdown": {
            "by_stage": by_stage,
            "by_provider": by_provider,
            "by_provider_model": by_provider_model,
            "detail": list(detail_dict.values()),
        },
        "raw_rows": total_calls,
    }


def get_today_usage_summary():
    today = _utc_today_str()
    day_start = today + "T00:00:00"
    day_end = today + "T23:59:59.999999"

    try:
        with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT provider, model,
                          COUNT(*) as calls,
                          SUM(input_tokens) as input_tokens,
                          SUM(cached_input_tokens) as cached_input_tokens,
                          SUM(output_tokens) as output_tokens,
                          SUM(estimated_cost_usd) as estimated_cost_usd
                   FROM ai_usage_ledger
                   WHERE created_at>=? AND created_at<=? AND status!='PENDING'
                   GROUP BY provider, model""",
                (day_start, day_end),
            ).fetchall()
    except Exception as e:
        logger.error("get_today_usage_summary: DB error: %s", e)
        return {"date_utc": today, "providers": {}, "total": _empty_today_total()}

    providers = {}
    total_calls = 0
    total_input = total_cached = total_output = total_cost = None

    for row in rows:
        p = row["provider"]
        m = row["model"]
        if p not in providers:
            providers[p] = {
                "calls_today": 0,
                "input_tokens_today": None,
                "cached_input_tokens_today": None,
                "output_tokens_today": None,
                "estimated_cost_today": None,
                "models": {}
            }

        # Add to provider totals
        providers[p]["calls_today"] += row["calls"] or 0
        providers[p]["input_tokens_today"] = _nullable_add(providers[p]["input_tokens_today"], row["input_tokens"])
        providers[p]["cached_input_tokens_today"] = _nullable_add(providers[p]["cached_input_tokens_today"], row["cached_input_tokens"])
        providers[p]["output_tokens_today"] = _nullable_add(providers[p]["output_tokens_today"], row["output_tokens"])
        providers[p]["estimated_cost_today"] = _nullable_add(providers[p]["estimated_cost_today"], row["estimated_cost_usd"])

        # Add to model breakdown
        providers[p]["models"][m] = {
            "calls_today": row["calls"] or 0,
            "input_tokens_today": row["input_tokens"],
            "cached_input_tokens_today": row["cached_input_tokens"],
            "output_tokens_today": row["output_tokens"],
            "estimated_cost_today": row["estimated_cost_usd"],
        }

        total_calls += row["calls"] or 0
        total_input = _nullable_add(total_input, row["input_tokens"])
        total_cached = _nullable_add(total_cached, row["cached_input_tokens"])
        total_output = _nullable_add(total_output, row["output_tokens"])
        total_cost = _nullable_add(total_cost, row["estimated_cost_usd"])

    return {
        "date_utc": today,
        "providers": providers,
        "total": {
            "calls_today": total_calls,
            "input_tokens_today": total_input,
            "cached_input_tokens_today": total_cached,
            "output_tokens_today": total_output,
            "estimated_cost_today": total_cost,
        },
    }


MIN_SAMPLE_THRESHOLD = 3


def get_historical_stats():
    """
    Historical usage statistics.

    Two separate concepts:
    A) ALL-TIME PROVIDER USAGE — all completed provider requests (SUCCESS + FAILED).
       These are what actually cost money.
       Excludes PENDING rows only (not yet completed).
       Fields: total_calls_all_time, total_estimated_cost_all_time

    B) TRANSLATED JOB STATISTICS — only jobs with status=TRANSLATED.
       Used for per-job averages (calls per translated job, cost per translated job).
       Fields: completed_jobs_with_ai, average_calls_per_job, average_estimated_cost_per_job

    NULL-job_id rows are included in all-time totals (non-attributed requests still cost money).
    """
    try:
        with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row

            # A) All-time provider usage: ALL completed rows (excluding PENDING)
            all_time_row = conn.execute(
                """SELECT
                       COUNT(id)              AS total_calls,
                       SUM(estimated_cost_usd) AS total_cost
                   FROM ai_usage_ledger
                   WHERE status != 'PENDING'"""
            ).fetchone()

            all_time_calls = (all_time_row["total_calls"] or 0) if all_time_row else 0
            all_time_cost = (all_time_row["total_cost"]) if all_time_row else None

            # B) Translated-job statistics: rows attributed to TRANSLATED jobs only
            translated_row = conn.execute(
                """SELECT
                       COUNT(DISTINCT u.job_id) AS completed_jobs_with_ai,
                       COUNT(u.id)              AS translated_calls,
                       SUM(u.estimated_cost_usd) AS translated_cost
                   FROM ai_usage_ledger u
                   INNER JOIN jobs j ON j.id = u.job_id
                   WHERE j.status = 'TRANSLATED'
                     AND u.status != 'PENDING'
                     AND u.job_id IS NOT NULL"""
            ).fetchone()

            n = (translated_row["completed_jobs_with_ai"] or 0) if translated_row else 0
            tc = (translated_row["translated_calls"] or 0) if translated_row else 0
            cost = (translated_row["translated_cost"]) if translated_row else None

            if n == 0 and all_time_calls == 0:
                return {
                    "completed_jobs_with_ai": 0,
                    "average_calls_per_job": None,
                    "average_estimated_cost_per_job": None,
                    "total_calls_all_time": 0,
                    "total_estimated_cost_all_time": None,
                    "has_sufficient_history": False,
                    "min_sample_threshold": MIN_SAMPLE_THRESHOLD,
                }

            return {
                "completed_jobs_with_ai": n,
                "average_calls_per_job": tc / n if n > 0 else None,
                "average_estimated_cost_per_job": (cost / n) if (cost is not None and n > 0) else None,
                "total_calls_all_time": all_time_calls,
                "total_estimated_cost_all_time": all_time_cost,
                "has_sufficient_history": n >= MIN_SAMPLE_THRESHOLD,
                "min_sample_threshold": MIN_SAMPLE_THRESHOLD,
            }

    except Exception as e:
        logger.error("get_historical_stats: DB error: %s", e)
        return {
            "completed_jobs_with_ai": 0,
            "average_calls_per_job": None,
            "average_estimated_cost_per_job": None,
            "total_calls_all_time": 0,
            "total_estimated_cost_all_time": None,
            "has_sufficient_history": False,
            "min_sample_threshold": MIN_SAMPLE_THRESHOLD,
        }


def _nullable_add(a, b):
    if a is None and b is None:
        return None
    return (a or 0.0) + (b or 0.0)


def _empty_bucket():
    return {"calls": 0, "input_tokens": None, "cached_input_tokens": None, "output_tokens": None, "thinking_tokens": None, "estimated_cost_usd": None}


def _empty_job_summary(job_id):
    return {
        "job_id": job_id,
        "total_calls": 0,
        "total_input_tokens": None,
        "total_cached_input_tokens": None,
        "total_output_tokens": None,
        "total_thinking_tokens": None,
        "total_estimated_cost_usd": None,
        "breakdown": {"by_stage": {}, "by_provider": {}, "by_provider_model": {}, "detail": []},
        "raw_rows": 0,
    }


def _empty_today_total():
    return {
        "calls_today": 0,
        "input_tokens_today": None,
        "cached_input_tokens_today": None,
        "output_tokens_today": None,
        "estimated_cost_today": None,
    }
