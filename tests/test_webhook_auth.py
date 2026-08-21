import pytest
from app.main import app
from fastapi.testclient import TestClient
from unittest.mock import patch

def test_webhook_process_auth():
    client = TestClient(app)
    
    with patch("app.main.AUTH_USERNAME", "admin"), patch("app.main.AUTH_PASSWORD", "pass"):
        # Without auth
        resp1 = client.post("/webhook/process", json={"video_path": "/fake/video.mkv"})
        assert resp1.status_code == 401
        
        # With valid Basic Auth
        resp2 = client.post("/webhook/process", json={"video_path": "/fake/video.mkv"}, auth=("admin", "pass"))
        # Returns 422 because video_path doesn't exist, but bypasses 401
        assert resp2.status_code != 401
