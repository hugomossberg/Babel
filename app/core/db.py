import sqlite3
import os
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Defer reason constants (centralised — used by quota.py, pipeline.py, tests)
# ---------------------------------------------------------------------------

class DeferReason:
    LOCAL_RPD                 = "LOCAL_RPD"
    PROVIDER_QUOTA            = "PROVIDER_QUOTA"
    QUEUE_BACKLOG             = "QUEUE_BACKLOG"
    INSUFFICIENT_LOCAL_BUDGET = "INSUFFICIENT_LOCAL_BUDGET"
    ESCALATION_LOCAL_RPD      = "ESCALATION_LOCAL_RPD"
    ESCALATION_PROVIDER_QUOTA = "ESCALATION_PROVIDER_QUOTA"

class DeferStage:
    PRIMARY    = "PRIMARY"
    ESCALATION = "ESCALATION"

# ---------------------------------------------------------------------------
# Central active-job status definition
# ---------------------------------------------------------------------------
# These statuses represent a job that is actively doing work for a video_path
# (or is waiting for its turn).  A new job must NOT be created for the same
# video_path while any of these statuses exists.
#
# Rules:
#  • QUEUED          — waiting for a processing slot
#  • TRANSLATING     — pipeline is actively running
#  • RECOVERING      — stale-queue recovery retry
#  • ESCALATING      — escalating to secondary provider
#  • WAITING_PROVIDER— blocked on provider quota, will retry
#  • RETRY_PENDING   — scheduled for retry
#  • PARTIAL         — partial result, finishing up
#  • WAITING_SOURCE  — waiting for source subtitle
#  • DEFERRED        — daily budget exhausted; will auto-resume same job
#
# Terminal statuses (TRANSLATED, FAILED, CANCELLED, ALREADY EXISTS) are NOT
# included — they do not block a new legitimate job or force-retranslate.
ACTIVE_JOB_STATUSES: tuple = (
    "QUEUED",
    "TRANSLATING",
    "RECOVERING",
    "ESCALATING",
    "WAITING_PROVIDER",
    "RETRY_PENDING",
    "PARTIAL",
    "WAITING_SOURCE",
    "DEFERRED",
)

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("BABEL_DB_PATH", "/app/data/babel.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH, timeout=30.0) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS translation_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            series_title TEXT NOT NULL,
            original_text TEXT NOT NULL,
            translated_text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(series_title, original_text)
        )
        ''')

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_path TEXT NOT NULL,
            title TEXT,
            status TEXT NOT NULL,
            event_source TEXT,
            reason TEXT,
            target_languages TEXT,
            ai_model TEXT,
            total_lines INTEGER DEFAULT 0,
            cleaned_sdh_lines INTEGER DEFAULT 0,
            dropped_lines INTEGER DEFAULT 0,
            sync_diff_ms INTEGER DEFAULT 0,
            output_files TEXT,
            error_message TEXT,
            logs TEXT,
            duration_seconds REAL DEFAULT 0.0,
            processed_lines INTEGER DEFAULT 0,
            current_batch TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            retry_count INTEGER DEFAULT 0,
            next_retry_at TEXT,
            last_error TEXT,
            -- Provider/model pinning (set on first real AI dispatch, immutable after)
            primary_provider        TEXT,
            primary_model           TEXT,
            escalation_enabled      INTEGER DEFAULT 0,
            escalation_provider     TEXT,
            escalation_model        TEXT,
            -- Deferred metadata
            defer_reason            TEXT,
            waiting_provider        TEXT,
            waiting_model           TEXT,
            defer_stage             TEXT,
            deferred_at             TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """)

        # Provider quota / RPD block state and Circuit Breaker
        cursor.execute("""
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
                cursor.execute(f"ALTER TABLE provider_quota ADD COLUMN {col} {col_type}")
            except Exception:
                pass

        # Clear any in-flight probe leases from prior session on restart
        cursor.execute("UPDATE provider_quota SET probe_lease_until = NULL, probe_lease_owner = NULL WHERE probe_lease_until IS NOT NULL")

        # Per-provider daily request counter (UTC calendar day window)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_request_counts (
            provider        TEXT NOT NULL,
            window_date     TEXT NOT NULL,
            request_count   INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (provider, window_date)
        )
        """)
        # ---------------------------------------------------------------
        # Add columns if not existing yet on old database files.
        # IMPORTANT: This must run BEFORE the UPDATE statements below
        # that reference these columns (e.g. error_message).
        # ---------------------------------------------------------------
        for _col, _col_type in [
            ("reason",             "TEXT"),
            ("target_languages",   "TEXT"),
            ("ai_model",           "TEXT"),
            ("total_lines",        "INTEGER DEFAULT 0"),
            ("cleaned_sdh_lines",  "INTEGER DEFAULT 0"),
            ("dropped_lines",      "INTEGER DEFAULT 0"),
            ("sync_diff_ms",       "INTEGER DEFAULT 0"),
            ("output_files",       "TEXT"),
            ("error_message",      "TEXT"),
            ("duration_seconds",   "REAL DEFAULT 0.0"),
            ("processed_lines",    "INTEGER DEFAULT 0"),
            ("current_batch",      "TEXT"),
            ("retry_count",        "INTEGER DEFAULT 0"),
            ("next_retry_at",      "TEXT"),
            ("last_error",         "TEXT"),
            # Provider pinning (Phase 1)
            ("primary_provider",   "TEXT"),
            ("primary_model",      "TEXT"),
            ("escalation_enabled", "INTEGER DEFAULT 0"),
            ("escalation_provider","TEXT"),
            ("escalation_model",   "TEXT"),
            # Deferred metadata (Phase 1)
            ("defer_reason",       "TEXT"),
            ("waiting_provider",   "TEXT"),
            ("waiting_model",      "TEXT"),
            ("defer_stage",        "TEXT"),
            ("deferred_at",        "TEXT"),
            # v2.3.42: Force-retranslate intent persistence (persists through DEFERRED/RETRY)
            ("force_retranslate",  "INTEGER DEFAULT 0"),
        ]:
            try:
                cursor.execute(f"ALTER TABLE jobs ADD COLUMN {_col} {_col_type}")
            except Exception:
                pass

        # Reset stuck jobs on restart
        cursor.execute("""
        UPDATE jobs
        SET status = 'FAILED', error_message = 'Interrupted by server restart'
        WHERE status IN ('PENDING', 'PROCESSING', 'RUNNING')
        """)

        cursor.execute("""
        UPDATE jobs
        SET status = 'RETRY_PENDING', error_message = 'Recovered from restart'
        WHERE status IN ('TRANSLATING', 'RECOVERING', 'ESCALATING', 'WAITING_PROVIDER', 'RETRY_PENDING')
        """)

        # DEFERRED jobs survive restart — they keep their next_retry_at and will
        # be picked up by the scheduler automatically after the quota resets.
        # No state change needed for DEFERRED on restart.


        # Check if notify_jellyfin already exists BEFORE defaults are inserted
        has_notify_jellyfin = cursor.execute("SELECT 1 FROM settings WHERE key = 'notify_jellyfin'").fetchone() is not None
        if not has_notify_jellyfin:
            # One-time migration: If legacy jellyfin_enabled exists, carry its value over
            row_jf = cursor.execute("SELECT value FROM settings WHERE key = 'jellyfin_enabled'").fetchone()
            if row_jf:
                cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('notify_jellyfin', ?)", (row_jf[0],))

        defaults = {
            "gemini_api_key": "",
            "gemini_model": "gemini-3.5-flash-lite",
            "batch_size": "150",
            "max_concurrent_jobs": "3",
            "batch_concurrency": "2",
            "glossary": "",
            "enable_bazarr_check": "true",
            "bazarr_url": "http://bazarr:6767",
            "bazarr_api_key": "",
            "wait_time_seconds": "15",
            "clean_sdh": "true",
            "notify_jellyfin": "false",
            "jellyfin_enabled": "false",
            "jellyfin_url": "http://jellyfin:8096",
            "jellyfin_api_key": "",
            "media_series_path": "/tv",
            "media_movies_path": "/movies",
            "qa_max_unresolved_cues": "3",
            "qa_max_unresolved_ratio": "0.01",
            "languages": json.dumps([
                {"name": "Swedish", "code": "sv", "enabled": True}
            ]),
            # Daily request budget defaults — "0" means Unlimited (backward compatible)
            "daily_request_budget_gemini": "0",
            "daily_request_budget_openai": "0",
            "daily_request_budget_deepl": "0",
            "daily_request_budget_ollama": "0",
        }
        for k, v in defaults.items():
            cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (k, v))

        # Webhook secret canonical resolution:
        # BABEL_WEBHOOK_SECRET env var (non-empty) always wins over any persisted DB value.
        # This ensures Docker users can change the secret in .env + container restart
        # without a stale DB value remaining active.
        # Semantics:
        #   - BABEL_WEBHOOK_SECRET non-empty → persist and use env value (UPDATE OR INSERT)
        #   - BABEL_WEBHOOK_SECRET empty/absent → keep whatever is in DB (INSERT OR IGNORE with "")
        _env_secret = os.getenv("BABEL_WEBHOOK_SECRET", "")
        if _env_secret:
            # Env explicitly set: overwrite DB regardless of existing value
            cursor.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES ('webhook_secret', ?)",
                (_env_secret,)
            )
        else:
            # Env absent/empty: preserve existing DB value; insert empty string only if no row exists
            cursor.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES ('webhook_secret', '')"
            )

        conn.commit()

    # Phase 2: Initialize AI usage ledger schema (idempotent, separate connection)
    try:
        from app.core.usage import init_usage_schema
        init_usage_schema()
    except Exception as _usage_err:
        logger.warning("Could not initialize usage ledger schema: %s", _usage_err)


def get_setting(key: str, default: str = "") -> str:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        row = cursor.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

def set_setting(key: str, value: str):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
        conn.commit()

def get_positive_int_setting(key: str, default: int) -> int:
    """
    Safely retrieves a strictly positive integer setting (>= 1).
    Handles empty strings, non-numeric strings, floats, and negative/zero numbers.
    """
    raw_value = get_setting(key, str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            f"Invalid integer setting {key}={raw_value!r}; using default {default}"
        )
        return max(1, default)
    return max(1, value)

def get_int_setting(key: str, default: int, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    """
    Safely retrieves an integer setting with optional min/max bounds.
    """
    raw_value = get_setting(key, str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            f"Invalid integer setting {key}={raw_value!r}; using default {default}"
        )
        value = default
    if min_val is not None:
        value = max(min_val, value)
    if max_val is not None:
        value = min(max_val, value)
    return value

def get_float_setting(key: str, default: float, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    """
    Safely retrieves a float setting with optional min/max bounds.
    """
    raw_value = get_setting(key, str(default))
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            f"Invalid float setting {key}={raw_value!r}; using default {default}"
        )
        value = default
    if min_val is not None:
        value = max(min_val, value)
    if max_val is not None:
        value = min(max_val, value)
    return value

def create_job(video_path: str, event_source: str = "MANUAL", title: Optional[str] = None) -> int:
    now = datetime.now(timezone.utc).isoformat()
    if not title:
        title = os.path.basename(video_path)

    provider = get_setting("ai_provider", "gemini").lower()
    if provider == "openai":
        active_model = f"OpenAI ({get_setting('openai_model', 'gpt-4o-mini')})"
    elif provider == "deepl":
        active_model = "DeepL Translate"
    elif provider in ["ollama", "localai"]:
        active_model = f"Ollama ({get_setting('ollama_model', 'llama3')})"
    else:
        active_model = f"Gemini ({get_setting('gemini_model', 'gemini-3.5-flash-lite')})"

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Add column if not existing yet on old database files
        try:
            cursor.execute("ALTER TABLE jobs ADD COLUMN ai_model TEXT")
        except Exception:
            pass
        cursor.execute("""
        INSERT INTO jobs (video_path, title, status, event_source, ai_model, created_at, updated_at, logs)
        VALUES (?, ?, 'QUEUED', ?, ?, ?, ?, ?)
        """, (video_path, title, event_source, active_model, now, now, json.dumps([f"Job created ({event_source}) for {os.path.basename(video_path)}"])))
        conn.commit()
        return cursor.lastrowid


def get_active_job_for_video(video_path: str) -> Optional[Dict[str, Any]]:
    """
    Return the most recent active (non-terminal) job for *video_path*, or None.

    Uses ACTIVE_JOB_STATUSES as the single source of truth.
    Reads under a shared connection (no EXCLUSIVE lock needed — read-only).
    """
    norm = os.path.normpath(video_path)
    placeholders = ",".join("?" * len(ACTIVE_JOB_STATUSES))
    try:
        with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                f"""SELECT * FROM jobs
                    WHERE video_path = ? AND status IN ({placeholders})
                    ORDER BY id DESC LIMIT 1""",
                (norm, *ACTIVE_JOB_STATUSES),
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            if d.get("logs"):
                try:
                    d["logs"] = json.loads(d["logs"])
                except Exception:
                    d["logs"] = []
            else:
                d["logs"] = []
            return d
    except Exception as e:
        logger.error("get_active_job_for_video error for %s: %s", video_path, e)
        return None


def create_job_if_no_active(
    video_path: str,
    event_source: str = "MANUAL",
    title: Optional[str] = None,
    force_retranslate: bool = False,
) -> Dict[str, Any]:
    """
    Atomically create a new QUEUED job for *video_path* — BUT ONLY IF no active
    (non-terminal) job already exists for the same path.

    Uses an SQLite EXCLUSIVE transaction so that two concurrent callers cannot
    both see "no active job" and both insert a duplicate.

    Returns a dict:
        {
          "job_id":        int,        # ID of job (new or existing)
          "created":       bool,       # True  = new job created
                                       # False = existing active job returned
          "existing_job":  dict|None,  # populated when created=False
        }

    Force semantics: if *force_retranslate* is True, the function still checks
    for an existing active job and returns it unchanged (created=False) if one
    is found.  Force only means "the old target subtitle may be replaced when
    the NEW translation is ready and QA-approved" — it does NOT create a second
    parallel active job for the same video_path.  The flag is persisted in the
    DB row so the pipeline can honour it after DEFERRED / RETRY cycles.

    Fail-closed: if the exclusive dedupe check fails due to a DB error, we do
    NOT fall back to an unconditional create_job().  The caller receives the
    exception and should return a retryable error to the client.
    """
    if not title:
        title = os.path.basename(video_path)
    norm = os.path.normpath(video_path)

    provider = get_setting("ai_provider", "gemini").lower()
    if provider == "openai":
        active_model = f"OpenAI ({get_setting('openai_model', 'gpt-4o-mini')})"
    elif provider == "deepl":
        active_model = "DeepL Translate"
    elif provider in ["ollama", "localai"]:
        active_model = f"Ollama ({get_setting('ollama_model', 'llama3')})"
    else:
        active_model = f"Gemini ({get_setting('gemini_model', 'gemini-3.5-flash-lite')})"

    now = datetime.now(timezone.utc).isoformat()
    placeholders = ",".join("?" * len(ACTIVE_JOB_STATUSES))

    # No outer try/except — we fail closed on DB errors (see docstring above).
    conn = sqlite3.connect(DB_PATH, timeout=15.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("BEGIN EXCLUSIVE")

        # Ensure new columns exist (idempotent on old DBs)
        for _col, _def in [
            ("ai_model", "TEXT"),
            ("force_retranslate", "INTEGER DEFAULT 0"),
        ]:
            try:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {_col} {_def}")
            except Exception:
                pass

        # Check for an existing active job inside the exclusive lock.
        # Force does NOT bypass this check.
        existing = conn.execute(
            f"""SELECT * FROM jobs
                WHERE video_path = ? AND status IN ({placeholders})
                ORDER BY id DESC LIMIT 1""",
            (norm, *ACTIVE_JOB_STATUSES),
        ).fetchone()
        if existing:
            conn.execute("ROLLBACK")
            d = dict(existing)
            if d.get("logs"):
                try:
                    d["logs"] = json.loads(d["logs"])
                except Exception:
                    d["logs"] = []
            else:
                d["logs"] = []
            return {"job_id": d["id"], "created": False, "existing_job": d}

        # No active job: create a new QUEUED job, storing the force flag.
        conn.execute(
            """INSERT INTO jobs
                   (video_path, title, status, event_source, ai_model,
                    force_retranslate, created_at, updated_at, logs)
               VALUES (?, ?, 'QUEUED', ?, ?, ?, ?, ?, ?)""",
            (norm, title, event_source, active_model,
             1 if force_retranslate else 0,
             now, now,
             json.dumps([f"Job created ({event_source}) for {os.path.basename(norm)}"]))
        )
        job_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("COMMIT")
        return {"job_id": job_id, "created": True, "existing_job": None}

    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.close()




def get_active_jobs_by_video_paths(video_paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Efficiently fetch active jobs for a list of video_paths.

    Returns a dict mapping normalized video_path → most-recent active job dict.
    Paths with no active job are absent from the returned dict.

    Design: Instead of building WHERE video_path IN (?, ?, ..., ?) with
    potentially thousands of paths (SQLite limit: 999 / SQLITE_MAX_VARIABLE_NUMBER),
    we fetch ALL rows whose status is in ACTIVE_JOB_STATUSES in one tiny query
    (active jobs are always far fewer than total library files) and then filter
    by the requested paths in Python.
    """
    if not video_paths:
        return {}

    norm_paths_set = {os.path.normpath(p) for p in video_paths}
    placeholders_status = ",".join("?" * len(ACTIVE_JOB_STATUSES))

    try:
        with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            # Small query — only filters on status index, not on path list.
            # Returns the most recent active job per video_path (MAX id = latest).
            rows = conn.execute(
                f"""SELECT j.*
                    FROM jobs j
                    INNER JOIN (
                        SELECT video_path, MAX(id) AS max_id
                        FROM jobs
                        WHERE status IN ({placeholders_status})
                        GROUP BY video_path
                    ) sub ON j.id = sub.max_id""",
                ACTIVE_JOB_STATUSES,
            ).fetchall()

        result: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            d = dict(row)
            norm = os.path.normpath(d.get("video_path", ""))
            # Filter in Python — only include paths the caller asked about
            if norm not in norm_paths_set:
                continue
            if d.get("logs"):
                try:
                    d["logs"] = json.loads(d["logs"])
                except Exception:
                    d["logs"] = []
            else:
                d["logs"] = []
            result[norm] = d
        return result
    except Exception as e:
        logger.error("get_active_jobs_by_video_paths error: %s", e)
        return {}

def append_job_log(job_id: int, message: str):
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    formatted = f"[{now}] {message}"
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("UPDATE jobs SET logs = json_insert(coalesce(logs, '[]'), '$[#]', ?) WHERE id = ?", (formatted, job_id))
        except sqlite3.OperationalError:
            row = cursor.execute("SELECT logs FROM jobs WHERE id = ?", (job_id,)).fetchone()
            logs = []
            if row and row[0]:
                try:
                    logs = json.loads(row[0])
                except Exception:
                    pass
            logs.append(formatted)
            cursor.execute("UPDATE jobs SET logs = ? WHERE id = ?", (json.dumps(logs), job_id))
        conn.commit()

def update_job(job_id: int, **kwargs):
    now = datetime.now(timezone.utc).isoformat()
    kwargs["updated_at"] = now

    set_clauses = [f"{k} = ?" for k in kwargs.keys()]
    values = list(kwargs.values()) + [job_id]

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(f"UPDATE jobs SET {', '.join(set_clauses)} WHERE id = ?", values)
        conn.commit()

def delete_job(job_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        conn.commit()

def clear_all_jobs():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Delete usage ledger rows first to avoid orphaned FK references.
        # Both tables cleared in the same transaction for consistency.
        try:
            cursor.execute("DELETE FROM ai_usage_ledger")
        except Exception:
            pass  # Table may not exist yet (schema migration pending)
        cursor.execute("DELETE FROM jobs")
        conn.commit()

def claim_job_for_retry(job_id: int) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE jobs SET status = 'QUEUED' WHERE id = ? AND status IN "
            "('WAITING_PROVIDER', 'RETRY_PENDING', 'RECOVERING', 'PARTIAL', 'WAITING_SOURCE', 'DEFERRED')",
            (job_id,)
        )
        conn.commit()
        return cursor.rowcount > 0


def get_job_by_id(job_id: int) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        row = cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            return None
        data = dict(row)
        if data.get("logs"):
            try:
                data["logs"] = json.loads(data["logs"])
            except Exception:
                data["logs"] = [data["logs"]]
        else:
            data["logs"] = []
        return data

def get_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("logs"):
                try:
                    d["logs"] = json.loads(d["logs"])
                except Exception:
                    d["logs"] = []
            else:
                d["logs"] = []
            results.append(d)
        return results

def get_job_stats() -> Dict[str, Any]:
    _status_placeholders = ",".join("?" * len(ACTIVE_JOB_STATUSES))
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        total = cursor.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        translated = cursor.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('TRANSLATED', 'SUCCESS')").fetchone()[0]
        healthy = cursor.execute("SELECT COUNT(*) FROM jobs WHERE status = 'HEALTHY'").fetchone()[0]
        repaired = cursor.execute("SELECT COUNT(*) FROM jobs WHERE status = 'REPAIRED'").fetchone()[0]
        failed = cursor.execute("SELECT COUNT(*) FROM jobs WHERE status = 'FAILED'").fetchone()[0]
        deferred = cursor.execute("SELECT COUNT(*) FROM jobs WHERE status = 'DEFERRED'").fetchone()[0]
        active = cursor.execute(
            f"SELECT COUNT(*) FROM jobs WHERE status IN ({_status_placeholders})",
            ACTIVE_JOB_STATUSES
        ).fetchone()[0]
        avg_dur = cursor.execute("SELECT AVG(duration_seconds) FROM jobs WHERE status IN ('TRANSLATED', 'REPAIRED', 'SUCCESS')").fetchone()[0] or 0.0
        return {
            "total": total,
            "translated": translated,
            "healthy": healthy,
            "repaired": repaired,
            "failed": failed,
            "deferred": deferred,
            "active_jobs": active,

            "avg_duration_seconds": round(avg_dur, 1)
        }

def get_jobs_by_status(statuses: list) -> list:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(statuses))
        cursor.execute(f"SELECT * FROM jobs WHERE status IN ({placeholders}) ORDER BY id ASC", tuple(statuses))
        rows = cursor.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("logs"):
                try: d["logs"] = json.loads(d["logs"])
                except Exception: d["logs"] = []
            else:
                d["logs"] = []
            results.append(d)
        return results

def save_translation_memory(series_title: str, original: str, translated: str):
    if not series_title or not original or not translated: return
    if len(original.split()) > 6: return
    save_translation_memory_bulk(series_title, [{"original": original, "translated": translated}])

def save_translation_memory_bulk(series_title: str, items: list):
    if not series_title or not items: return
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    valid_items = []
    for item in items:
        orig = item.get("original", "").strip()
        trans = item.get("translated", "").strip()
        if not orig or not trans: continue
        if len(orig.split()) > 6: continue
        valid_items.append((series_title, orig, trans, now))

    if not valid_items: return

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.executemany("INSERT OR REPLACE INTO translation_memory (series_title, original_text, translated_text, created_at) VALUES (?, ?, ?, ?)", valid_items)
        conn.commit()

def _is_legit_legacy_tm_key(target_title: str, candidate_title: str) -> bool:
    if candidate_title == target_title:
        return True
    import re
    pattern = re.compile(rf"^{re.escape(target_title)}\s*-\s*[sS]\d{{1,4}}[eE]\d{{1,4}}.*$")
    return bool(pattern.match(candidate_title))

def get_translation_memory(series_title: str, limit: int = 20) -> list:
    if not series_title: return []
    import random
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        escaped_title = series_title.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        cursor.execute(
            "SELECT series_title, original_text, translated_text FROM translation_memory WHERE series_title = ? OR series_title LIKE ? ESCAPE '\\' OR series_title LIKE ? ESCAPE '\\'",
            (series_title, f"{escaped_title} - S%", f"{escaped_title} - s%")
        )
        rows = cursor.fetchall()
        valid = [
            {"original": r["original_text"], "translated": r["translated_text"]}
            for r in rows
            if _is_legit_legacy_tm_key(series_title, r["series_title"])
        ]
        if len(valid) > limit:
            return random.sample(valid, limit)
        return valid

def recover_stale_queued_jobs():
    import sqlite3
    from datetime import datetime, timezone, timedelta
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now(timezone.utc)
    threshold = now - timedelta(minutes=15)
    now_iso = now.isoformat()
    threshold_iso = threshold.isoformat()

    cursor.execute('''
        UPDATE jobs
        SET status = 'RECOVERING', next_retry_at = ?, updated_at = ?
        WHERE status = 'QUEUED' AND updated_at < ?
    ''', (now_iso, now_iso, threshold_iso))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Phase 1: Provider/Model Pinning
# ---------------------------------------------------------------------------

def pin_job_provider(
    job_id: int,
    primary_provider: str,
    primary_model: str,
    escalation_enabled: bool = False,
    escalation_provider: Optional[str] = None,
    escalation_model: Optional[str] = None,
) -> None:
    """
    Persist the effective AI config for a job on first real dispatch.
    Once pinned, provider is immutable — global setting changes do not affect this job.
    Only writes if primary_provider is not already set (idempotent).
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Only pin if not already pinned
        row = cursor.execute(
            "SELECT primary_provider FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row and row[0]:
            return  # Already pinned — do not overwrite
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute("""
            UPDATE jobs
            SET primary_provider    = ?,
                primary_model       = ?,
                escalation_enabled  = ?,
                escalation_provider = ?,
                escalation_model    = ?,
                updated_at          = ?
            WHERE id = ?
        """, (
            primary_provider,
            primary_model,
            1 if escalation_enabled else 0,
            escalation_provider,
            escalation_model,
            now,
            job_id,
        ))
        conn.commit()


def update_deferred_metadata(
    job_id: int,
    defer_reason: str,
    waiting_provider: str,
    waiting_model: Optional[str],
    defer_stage: str = "PRIMARY",
) -> None:
    """
    Persist structured deferred metadata for a job.
    Called whenever a job transitions to DEFERRED status.
    """
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE jobs
            SET defer_reason     = ?,
                waiting_provider = ?,
                waiting_model    = ?,
                defer_stage      = ?,
                deferred_at      = ?,
                updated_at       = ?
            WHERE id = ?
        """, (defer_reason, waiting_provider, waiting_model, defer_stage, now, now, job_id))
        conn.commit()


# ---------------------------------------------------------------------------
# Phase 1: FIFO per provider + atomic claim
# ---------------------------------------------------------------------------

def get_eligible_deferred_jobs_for_provider(
    provider: str,
    stage: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return DEFERRED jobs that are waiting for *provider*, ordered FIFO.
    Ordering: deferred_at ASC (oldest first), then id ASC as tie-break.
    If deferred_at is NULL (legacy), falls back to created_at.
    stage: if set, filter by defer_stage (e.g. 'PRIMARY' or 'ESCALATION').
    """
    p = (provider or "").strip().lower()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if stage:
            rows = cursor.execute("""
                SELECT * FROM jobs
                WHERE status = 'DEFERRED'
                  AND lower(coalesce(waiting_provider, primary_provider, '')) = ?
                  AND defer_stage = ?
                ORDER BY coalesce(deferred_at, created_at) ASC, id ASC
            """, (p, stage)).fetchall()
        else:
            rows = cursor.execute("""
                SELECT * FROM jobs
                WHERE status = 'DEFERRED'
                  AND lower(coalesce(waiting_provider, primary_provider, '')) = ?
                ORDER BY coalesce(deferred_at, created_at) ASC, id ASC
            """, (p,)).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("logs"):
                try: d["logs"] = json.loads(d["logs"])
                except Exception: d["logs"] = []
            else:
                d["logs"] = []
            results.append(d)
        return results


def claim_fifo_job_for_retry(job_id: int) -> bool:
    """
    Atomically claim a DEFERRED job for retry using an EXCLUSIVE transaction.
    Returns True iff the job was successfully transitioned from DEFERRED -> QUEUED.
    Race-safe: only one winner for each job_id.
    """
    try:
        conn = sqlite3.connect(DB_PATH, timeout=15.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("BEGIN EXCLUSIVE")
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE jobs SET status = 'QUEUED', updated_at = ? "
                "WHERE id = ? AND status = 'DEFERRED'",
                (datetime.now(timezone.utc).isoformat(), job_id),
            )
            claimed = cursor.rowcount > 0
            conn.execute("COMMIT")
            return claimed
        except Exception:
            try: conn.execute("ROLLBACK")
            except Exception: pass
            raise
        finally:
            conn.close()
    except Exception as exc:
        logger.error("claim_fifo_job_for_retry error for job %s: %s", job_id, exc)
        return False


def has_older_eligible_deferred_backlog(
    provider: str,
    newer_than_created_at: str,
) -> bool:
    """
    Return True if there is any DEFERRED job for *provider* that is older than
    newer_than_created_at. Used to enforce FIFO: a new job must not cut the queue.
    """
    p = (provider or "").strip().lower()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        row = cursor.execute("""
            SELECT 1 FROM jobs
            WHERE status = 'DEFERRED'
              AND lower(coalesce(waiting_provider, primary_provider, '')) = ?
              AND coalesce(deferred_at, created_at) < ?
            LIMIT 1
        """, (p, newer_than_created_at)).fetchone()
        return row is not None
