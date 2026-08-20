import sqlite3
import os
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

DB_PATH = os.getenv("BABEL_DB_PATH", "/app/data/babel.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH, timeout=30.0) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
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
            updated_at TEXT NOT NULL
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """)
        
        # Reset stuck jobs on restart
        cursor.execute("""
        UPDATE jobs 
        SET status = 'FAILED', error_message = 'Interrupted by server restart' 
        WHERE status IN ('PENDING', 'PROCESSING', 'RUNNING', 'TRANSLATING', 'QUEUED')
        """)
        
        defaults = {
            "gemini_api_key": "",
            "gemini_model": "gemini-3.5-flash-lite",
            "batch_size": "50",
            "max_concurrency": "1",
            "max_concurrent_jobs": "1",
            "glossary": "",
            "enable_bazarr_check": "true",
            "bazarr_url": "http://dev-bazarr:6767",
            "bazarr_api_key": "",
            "wait_time_seconds": "15",
            "clean_sdh": "true",
            "notify_jellyfin": "true",
            "jellyfin_url": "http://dev-jellyfin:8096",
            "jellyfin_api_key": "devtestkey1234567890abcdef",
            "media_series_path": "/tv",
            "media_movies_path": "/movies",
            "languages": json.dumps([
                {"name": "Swedish", "code": "sv", "enabled": True}
            ])
        }
        for k, v in defaults.items():
            cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (k, v))
            
        conn.commit()

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
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        row = cursor.execute("SELECT logs FROM jobs WHERE id = ?", (job_id,)).fetchone()
        logs = []
        if row and row[0]:
            try:
                logs = json.loads(row[0])
            except Exception:
                pass
        now = datetime.now(timezone.utc).strftime("%H:%M:%S")
        logs.append(f"[{now}] {message}")
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
        avg_dur = cursor.execute("SELECT AVG(duration_seconds) FROM jobs WHERE status IN ('TRANSLATED', 'REPAIRED', 'SUCCESS')").fetchone()[0] or 0.0
        return {
            "total": total,
            "translated": translated,
            "healthy": healthy,
            "repaired": repaired,
            "failed": failed,
            "avg_duration_seconds": round(avg_dur, 1)
        }
