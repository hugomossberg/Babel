import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from app.updater.main import app as updater_app, UPDATER_SECRET, rollback
import app.updater.main as updater_module

@pytest.fixture
def updater_client():
    return TestClient(updater_app)

def test_updater_health_and_status(updater_client):
    res = updater_client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "updater_healthy"

    with patch.object(updater_module, "UPDATER_SECRET", "token"):
        res = updater_client.get("/status", headers={"X-Updater-Token": "token"})
        assert res.status_code == 200
        assert "status" in res.json()

def test_updater_auth_enforcement(updater_client):
    # Set secret for test
    with patch.object(updater_module, "UPDATER_SECRET", "super_secret_test_token"):
        # 1. No token -> 401
        res = updater_client.post("/update", json={"target_version": "v2.3.26-beta"})
        assert res.status_code == 401

        # 2. Wrong token -> 401
        res = updater_client.post(
            "/update",
            json={"target_version": "v2.3.26-beta"},
            headers={"X-Updater-Token": "wrong_token"}
        )
        assert res.status_code == 401

        # 3. Valid X-Updater-Token -> passes auth (fails at docker call or validation)
        with patch("app.updater.main.call_docker", new_callable=AsyncMock) as mock_docker:
            mock_docker.return_value = MagicMock(status_code=500)
            res = updater_client.post(
                "/update",
                json={"target_version": "v2.3.26-beta"},
                headers={"X-Updater-Token": "super_secret_test_token"}
            )
            # Should reach inspect step (not 401)
            assert res.status_code != 401

        # 4. Valid Authorization Bearer -> passes auth
        with patch("app.updater.main.call_docker", new_callable=AsyncMock) as mock_docker:
            mock_docker.return_value = MagicMock(status_code=500)
            res = updater_client.post(
                "/update",
                json={"target_version": "v2.3.26-beta"},
                headers={"Authorization": "Bearer super_secret_test_token"}
            )
            assert res.status_code != 401

def test_updater_malformed_target_version(updater_client):
    with patch.object(updater_module, "UPDATER_SECRET", "token"):
        headers = {"X-Updater-Token": "token"}

        # Test malicious inputs
        res = updater_client.post("/update", json={"target_version": "v1.0; rm -rf /"}, headers=headers)
        assert res.status_code == 400

        res = updater_client.post("/update", json={"target_version": "../latest"}, headers=headers)
        assert res.status_code == 400

        res = updater_client.post("/update", json={"target_version": "other-repo:latest"}, headers=headers)
        assert res.status_code == 400

def test_updater_concurrency_conflict(updater_client):
    with patch.object(updater_module, "UPDATER_SECRET", "token"), \
         patch.object(updater_module, "UPDATER_STATE", "pulling"):
        res = updater_client.post(
            "/update",
            json={"target_version": "v2.3.26-beta"},
            headers={"X-Updater-Token": "token"}
        )
        assert res.status_code == 409
        assert "in progress" in res.json()["detail"]

@pytest.mark.asyncio
async def test_updater_rollback_execution():
    with patch("app.updater.main.call_docker", new_callable=AsyncMock) as mock_docker:
        mock_docker.return_value = MagicMock(status_code=200)
        await rollback("babel_rollback", "babel")
        assert updater_module.UPDATER_STATE == "rolled_back"
        assert mock_docker.call_count == 3

def test_updater_pull_failure(updater_client):
    with patch.object(updater_module, "UPDATER_SECRET", "token"), \
         patch.object(updater_module, "UPDATER_STATE", "idle"), \
         patch("app.updater.main.call_docker", new_callable=AsyncMock) as mock_docker:

        # 1. Inspect succeeds
        inspect_resp = MagicMock(status_code=200)
        inspect_resp.json.return_value = {"Config": {}, "HostConfig": {}, "NetworkSettings": {}}

        # 2. Pull fails
        pull_resp = MagicMock(status_code=500, text="Registry down")

        mock_docker.side_effect = [inspect_resp, pull_resp]

        res = updater_client.post(
            "/update",
            json={"target_version": "v2.3.26-beta"},
            headers={"X-Updater-Token": "token"}
        )
        assert res.status_code == 500
        assert "Failed to pull image" in res.json()["detail"]
        assert updater_module.UPDATER_STATE == "failed"

def test_updater_create_failure_triggers_rollback(updater_client):
    with patch.object(updater_module, "UPDATER_SECRET", "token"), \
         patch.object(updater_module, "UPDATER_STATE", "idle"), \
         patch("app.updater.main.rollback", new_callable=AsyncMock) as mock_rollback, \
         patch("app.updater.main.call_docker", new_callable=AsyncMock) as mock_docker:

        # 1. Inspect succeeds
        inspect_resp = MagicMock(status_code=200)
        inspect_resp.json.return_value = {"Config": {"Image": "babel:latest"}, "HostConfig": {}, "NetworkSettings": {}}

        # 2. Pull succeeds
        pull_resp = MagicMock(status_code=200)

        # 3. Retag succeeds
        retag_resp = MagicMock(status_code=201)

        # 4. Check leftover rollback container (none)
        leftover_resp = MagicMock(status_code=404)

        # 5. Stop old container
        stop_resp = MagicMock(status_code=204)

        # 6. Rename old container
        rename_resp = MagicMock(status_code=204)

        # 7. Create new container fails
        create_resp = MagicMock(status_code=500, text="Invalid volume mount")

        mock_docker.side_effect = [inspect_resp, pull_resp, retag_resp, leftover_resp, stop_resp, rename_resp, create_resp]

        res = updater_client.post(
            "/update",
            json={"target_version": "v2.3.26-beta"},
            headers={"X-Updater-Token": "token"}
        )
        assert res.status_code == 500
        assert "Failed to create new container" in res.json()["detail"]
        mock_rollback.assert_awaited_once_with("babel_rollback", "babel")

def test_updater_start_failure_triggers_rollback(updater_client):
    with patch.object(updater_module, "UPDATER_SECRET", "token"), \
         patch.object(updater_module, "UPDATER_STATE", "idle"), \
         patch("app.updater.main.rollback", new_callable=AsyncMock) as mock_rollback, \
         patch("app.updater.main.call_docker", new_callable=AsyncMock) as mock_docker:

        inspect_resp = MagicMock(status_code=200)
        inspect_resp.json.return_value = {"Config": {"Image": "babel:latest"}, "HostConfig": {}, "NetworkSettings": {}}
        pull_resp = MagicMock(status_code=200)
        retag_resp = MagicMock(status_code=201)
        leftover_resp = MagicMock(status_code=404)
        stop_resp = MagicMock(status_code=204)
        rename_resp = MagicMock(status_code=204)
        create_resp = MagicMock(status_code=201)
        start_resp = MagicMock(status_code=500, text="Port already allocated")

        mock_docker.side_effect = [inspect_resp, pull_resp, retag_resp, leftover_resp, stop_resp, rename_resp, create_resp, start_resp]

        res = updater_client.post(
            "/update",
            json={"target_version": "v2.3.26-beta"},
            headers={"X-Updater-Token": "token"}
        )
        assert res.status_code == 500
        assert "Failed to start new container" in res.json()["detail"]
        mock_rollback.assert_awaited_once_with("babel_rollback", "babel")
