import pytest
from app.core.db import init_db, update_job, get_job_by_id, create_job, recover_stale_queued_jobs
import app.core.db as db
import sqlite3
from datetime import datetime, timezone, timedelta

def test_stale_vs_recent_queued():
    init_db()
    conn = sqlite3.connect(db.DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM jobs")
    conn.commit()
    conn.close()
    
    recent_id = create_job("/tmp/recent.mkv")
    stale_id = create_job("/tmp/stale.mkv")
    
    now = datetime.now(timezone.utc)
    recent_time = now.isoformat()
    stale_time = (now - timedelta(minutes=20)).isoformat()
    
    conn = sqlite3.connect(db.DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE jobs SET status = 'QUEUED', updated_at = ? WHERE id = ?", (recent_time, recent_id))
    cursor.execute("UPDATE jobs SET status = 'QUEUED', updated_at = ? WHERE id = ?", (stale_time, stale_id))
    conn.commit()
    conn.close()
    
    recover_stale_queued_jobs()
    
    r_job = get_job_by_id(recent_id)
    assert r_job["status"] == "QUEUED"
    
    s_job = get_job_by_id(stale_id)
    assert s_job["status"] == "RECOVERING"
