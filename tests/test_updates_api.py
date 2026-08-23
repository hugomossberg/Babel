import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app, process_one_retry_pass
from app.services.updates_controller import updates_controller

@pytest.fixture
def client():
    return TestClient(app)

def test_updates_get(client):
    with patch("app.services.updates_controller.UpdatesController.get_update_info", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {"update_available": True, "latest_version": "v2.3.26-beta"}
        res = client.get("/api/updates")
        assert res.status_code == 200
        assert res.json()["latest_version"] == "v2.3.26-beta"

def test_updates_trigger_success(client):
    with patch("app.services.updates_controller.UpdatesController.trigger_update", new_callable=AsyncMock) as mock_trig:
        mock_trig.return_value = {"success": True, "message": "Update initiated"}
        res = client.post("/api/updates/trigger", json={"target_version": "v2.3.26-beta"})
        assert res.status_code == 200
        assert res.json()["success"] is True

def test_updates_trigger_fail(client):
    with patch("app.services.updates_controller.UpdatesController.trigger_update", new_callable=AsyncMock) as mock_trig:
        mock_trig.return_value = {"success": False, "message": "Failed to update"}
        res = client.post("/api/updates/trigger", json={"target_version": "v2.3.26-beta"})
        assert res.status_code == 400
        assert res.json()["detail"] == "Failed to update"

def test_webhooks_blocked_during_update_lock(client):
    with patch.object(updates_controller, "is_locked_for_update", return_value=True):
        # 1. Sonarr webhook
        res1 = client.post("/webhook/sonarr", json={"eventType": "Download"})
        assert res1.status_code == 503
        assert "maintenance mode" in res1.json()["detail"]

        # 2. Radarr webhook
        res2 = client.post("/webhook/radarr", json={"eventType": "Download"})
        assert res2.status_code == 503
        assert "maintenance mode" in res2.json()["detail"]

        # 3. Manual process webhook
        res3 = client.post("/webhook/process", json={"video_path": "/tv/Show/ep.mkv"})
        assert res3.status_code == 503
        assert "maintenance mode" in res3.json()["detail"]

@pytest.mark.asyncio
async def test_retry_pass_skipped_during_update_lock():
    with patch.object(updates_controller, "is_locked_for_update", return_value=True), \
         patch("app.core.db.get_jobs_by_status") as mock_db:
        tasks = []
        async for task in process_one_retry_pass():
            tasks.append(task)
        assert len(tasks) == 0
        mock_db.assert_not_called()
