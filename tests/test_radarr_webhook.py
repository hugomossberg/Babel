import pytest
import os
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from app.core.db import get_job_by_id, set_setting

def test_radarr_webhook_success(tmp_path):
    client = TestClient(app)

    # Create dummy media file in temp directory
    media_file = tmp_path / "My.Movie.2023.mkv"
    media_file.write_text("dummy")

    # Set media path setting to include tmp_path
    set_setting("media_movies_path", str(tmp_path))

    payload = {
        "eventType": "Download",
        "movie": {
            "title": "My Movie",
            "year": 2023,
            "folderPath": str(tmp_path)
        },
        "movieFile": {
            "relativePath": "My.Movie.2023.mkv",
            "path": str(media_file)
        }
    }

    with patch("app.main.AUTH_USERNAME", ""), patch("app.main.AUTH_PASSWORD", ""), \
         patch("app.services.pipeline.pipeline.process_video_file") as mock_pipeline:

        resp = client.post("/webhook/radarr", json=payload)
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
        assert job["event_source"] == "RADARR"
