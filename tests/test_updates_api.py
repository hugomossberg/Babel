import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app
from app.core.db import get_setting, set_setting

@pytest.fixture
def client():
    return TestClient(app)

def test_updates_get(client):
    with patch("app.services.updates_controller.UpdatesController.get_update_info", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {"update_available": True, "latest_version": "v2.3.25-beta"}
        res = client.get("/api/updates")
        assert res.status_code == 200
        assert res.json()["latest_version"] == "v2.3.25-beta"

def test_updates_trigger_success(client):
    with patch("app.services.updates_controller.UpdatesController.trigger_update", new_callable=AsyncMock) as mock_trig:
        mock_trig.return_value = {"success": True, "message": "OK"}
        res = client.post("/api/updates/trigger", json={"target_version": "v2.3.25-beta"})
        assert res.status_code == 200
        assert res.json()["success"] == True

def test_updates_trigger_fail(client):
    with patch("app.services.updates_controller.UpdatesController.trigger_update", new_callable=AsyncMock) as mock_trig:
        mock_trig.return_value = {"success": False, "message": "Failed"}
        res = client.post("/api/updates/trigger", json={"target_version": "v2.3.25-beta"})
        assert res.status_code == 400
        assert res.json()["detail"] == "Failed"
