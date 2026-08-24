import pytest
from unittest.mock import patch, AsyncMock, MagicMock
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
async def test_updates_controller_handles_409_as_ongoing_update():
    ctrl = updates_controller
    with patch.object(ctrl, "is_locked_for_update", return_value=False), \
         patch.object(ctrl, "get_real_updater_status", return_value=(True, "pulling")), \
         patch("app.core.db.get_jobs_by_status", return_value=[]), \
         patch.object(ctrl, "get_update_info", return_value={"update_available": True, "latest_version": "v2.3.99-beta"}), \
         patch("httpx.AsyncClient.post") as mock_post:

        mock_post.return_value = MagicMock(status_code=409, text='{"detail": "Update already in progress (pulling)"}')
        res = await ctrl.trigger_update("v2.3.99-beta")
        assert res["success"] is False
        assert "already in progress" in res["message"]
        assert ctrl.update_status == "updating"

@pytest.mark.asyncio
async def test_updates_controller_timeout_with_active_updater_treated_as_success():
    ctrl = updates_controller
    with patch.object(ctrl, "is_locked_for_update", return_value=False), \
         patch.object(ctrl, "get_real_updater_status", side_effect=[(True, "idle"), (True, "inspecting")]), \
         patch("app.core.db.get_jobs_by_status", return_value=[]), \
         patch.object(ctrl, "get_update_info", return_value={"update_available": True, "latest_version": "v2.3.99-beta"}), \
         patch("httpx.AsyncClient.post", side_effect=Exception("Read timeout")):

        res = await ctrl.trigger_update("v2.3.99-beta")
        assert res["success"] is True
        assert "already in progress" in res["message"] or "Update initiated" in res["message"]
        assert ctrl.update_status == "updating"
