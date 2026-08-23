import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.core import db
import sqlite3
import os

@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "babel.db"
    db.DB_PATH = str(db_path)
    db.init_db()
    yield str(db_path)

def test_get_job_stats_includes_active_jobs(temp_db):
    stats = db.get_job_stats()
    assert "active_jobs" in stats
    assert stats["active_jobs"] == 0

    with sqlite3.connect(temp_db) as conn:
        conn.execute(
            """INSERT INTO jobs (
                video_path, target_languages, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)""",
            ("/tv/test.mkv", "sv", "TRANSLATING", "now", "now")
        )
    
    stats = db.get_job_stats()
    assert stats["active_jobs"] == 1

@pytest.mark.asyncio
async def test_api_stats_returns_active_jobs(temp_db):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "active_jobs" in data
        assert data["active_jobs"] == 0
