import pytest
import os
import tempfile
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from app.core.db import get_job_by_id, set_setting

def test_webhook_process_synchronous_job_creation(tmp_path):
    client = TestClient(app)
    
    # Create dummy media file in temp directory
    media_file = tmp_path / "My.Show.S01E01.mkv"
    media_file.write_text("dummy")
    
    # Set media path setting to include tmp_path
    set_setting("media_series_path", str(tmp_path))
    
    with patch("app.main.AUTH_USERNAME", ""), patch("app.main.AUTH_PASSWORD", ""), \
         patch("app.services.pipeline.pipeline.process_video_file") as mock_pipeline:
        
        resp = client.post("/webhook/process", json={"video_path": str(media_file), "force_retranslate": False})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert "job_id" in data
        
        job_id = data["job_id"]
        assert job_id is not None
        
        # Verify job exists immediately in the database
        job = get_job_by_id(job_id)
        assert job is not None
        assert job["video_path"] == str(media_file)
        assert job["status"] == "QUEUED"
        assert job["event_source"] == "MANUAL"
