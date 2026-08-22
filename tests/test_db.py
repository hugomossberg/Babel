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

def test_get_positive_int_setting(tmp_path, monkeypatch):
    db_path = tmp_path / "test_settings.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path))
    db.init_db()

    # Default fallback
    assert db.get_positive_int_setting("non_existent_key", 50) == 50

    # Corrupt string values
    db.set_setting("batch_size", "")
    assert db.get_positive_int_setting("batch_size", 50) == 50

    db.set_setting("batch_size", "abc")
    assert db.get_positive_int_setting("batch_size", 50) == 50

    db.set_setting("batch_size", "3.5")
    assert db.get_positive_int_setting("batch_size", 50) == 50

    # Negative and zero
    db.set_setting("batch_size", "0")
    assert db.get_positive_int_setting("batch_size", 50) == 1

    db.set_setting("batch_size", "-5")
    assert db.get_positive_int_setting("batch_size", 50) == 1

    # Valid positive integer
    db.set_setting("batch_size", "25")
    assert db.get_positive_int_setting("batch_size", 50) == 25

    # max_concurrent_jobs and batch_concurrency
    db.set_setting("max_concurrent_jobs", "garbage")
    assert db.get_positive_int_setting("max_concurrent_jobs", 1) == 1

    db.set_setting("batch_concurrency", "")
    assert db.get_positive_int_setting("batch_concurrency", 3) == 3
