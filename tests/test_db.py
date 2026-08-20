import pytest
import sqlite3
import os
from app.core import db

def test_init_db_queued_recovery(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path))
    
    # First init to create tables
    db.init_db()
    
    # Insert some jobs
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO jobs (video_path, status, created_at, updated_at) 
        VALUES ('v1.mp4', 'QUEUED', '2023-01-01', '2023-01-01')
        """)
        cursor.execute("""
        INSERT INTO jobs (video_path, status, created_at, updated_at) 
        VALUES ('v2.mp4', 'RUNNING', '2023-01-01', '2023-01-01')
        """)
        conn.commit()
        
    # Re-init db, simulating a restart
    db.init_db()
    
    # Check statuses
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.cursor()
        queued_job = cursor.execute("SELECT status FROM jobs WHERE video_path = 'v1.mp4'").fetchone()
        running_job = cursor.execute("SELECT status FROM jobs WHERE video_path = 'v2.mp4'").fetchone()
        
        assert queued_job[0] == 'QUEUED'
        assert running_job[0] == 'FAILED'
