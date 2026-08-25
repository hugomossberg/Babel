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
            "batch_size": "50",
            "max_concurrent_jobs": "1",
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
            "webhook_secret": os.getenv("BABEL_WEBHOOK_SECRET", ""),
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
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        total = cursor.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        translated = cursor.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('TRANSLATED', 'SUCCESS')").fetchone()[0]
        healthy = cursor.execute("SELECT COUNT(*) FROM jobs WHERE status = 'HEALTHY'").fetchone()[0]
        repaired = cursor.execute("SELECT COUNT(*) FROM jobs WHERE status = 'REPAIRED'").fetchone()[0]
        failed = cursor.execute("SELECT COUNT(*) FROM jobs WHERE status = 'FAILED'").fetchone()[0]
        deferred = cursor.execute("SELECT COUNT(*) FROM jobs WHERE status = 'DEFERRED'").fetchone()[0]
        active = cursor.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('EXTRACTING', 'TRANSLATING', 'QUEUED', 'WAITING_PROVIDER', 'RECOVERING', 'PARTIAL', 'WAITING_SOURCE')").fetchone()[0]
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
