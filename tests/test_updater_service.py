import pytest
import asyncio
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

        # 3. Valid X-Updater-Token -> passes auth (returns 200 started)
        with patch.object(updater_module, "UPDATER_STATE", "idle"), \
             patch("app.updater.main.run_update", new_callable=AsyncMock):
            res = updater_client.post(
                "/update",
                json={"target_version": "v2.3.26-beta"},
                headers={"X-Updater-Token": "super_secret_test_token"}
            )
            assert res.status_code == 200
            assert res.json()["status"] == "started"

        # 4. Valid Authorization Bearer -> passes auth
        with patch.object(updater_module, "UPDATER_STATE", "idle"), \
             patch("app.updater.main.run_update", new_callable=AsyncMock):
            res = updater_client.post(
                "/update",
                json={"target_version": "v2.3.26-beta"},
                headers={"Authorization": "Bearer super_secret_test_token"}
            )
            assert res.status_code == 200
            assert res.json()["status"] == "started"

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

@pytest.mark.asyncio
async def test_updater_post_returns_immediately_without_waiting_for_pull():
    """1, 2, 3: POST /update returns HTTP 200 immediately, background task continues execution."""
    pull_started = asyncio.Event()
    finish_pull = asyncio.Event()

    async def mock_call_docker(method, path, **kwargs):
        if "/images/create" in path:
            pull_started.set()
            await finish_pull.wait()
            return MagicMock(status_code=200)
        if "/containers/babel/json" in path:
            resp = MagicMock(status_code=200)
            resp.json.return_value = {"Config": {}, "HostConfig": {}, "NetworkSettings": {}}
            return resp
        return MagicMock(status_code=200)

    with patch.object(updater_module, "UPDATER_SECRET", "token"), \
         patch.object(updater_module, "UPDATER_STATE", "idle"), \
         patch("app.updater.main.call_docker", side_effect=mock_call_docker):

        client = TestClient(updater_app)
        res = client.post(
            "/update",
            json={"target_version": "v2.3.33-beta"},
            headers={"X-Updater-Token": "token"}
        )
        # 1. HTTP 200 returned immediately
        assert res.status_code == 200
        assert res.json()["status"] == "started"

        # 3. Background task is alive and running
        assert updater_module.UPDATE_TASK is not None
        await pull_started.wait()
        assert updater_module.UPDATER_STATE == "pulling"

        # Complete pull and wait for background task
        finish_pull.set()
        await asyncio.sleep(0.05)

@pytest.mark.asyncio
async def test_updater_background_success_when_healthy():
    """4: Background task completes with success when new container is healthy."""
    async def mock_call_docker(method, path, **kwargs):
        if method == "GET" and "/containers/babel/json" in path:
            resp = MagicMock(status_code=200)
            resp.json.return_value = {
                "Config": {"Image": "babel:latest"},
                "HostConfig": {},
                "NetworkSettings": {},
                "State": {"Health": {"Status": "healthy"}, "Status": "running"}
            }
            return resp
        if method == "GET" and "_rollback/json" in path:
            return MagicMock(status_code=404)
        if method == "POST" and "/containers/create" in path:
            return MagicMock(status_code=201)
        if method == "POST" and "/containers/babel/start" in path:
            return MagicMock(status_code=204)
        return MagicMock(status_code=200)

    with patch.object(updater_module, "UPDATER_STATE", "idle"), \
         patch("app.updater.main.call_docker", side_effect=mock_call_docker), \
         patch("asyncio.sleep", new_callable=AsyncMock):

        await updater_module.run_update("v2.3.33-beta")
        assert updater_module.UPDATER_STATE == "success"

@pytest.mark.asyncio
async def test_updater_pull_failure_sets_failed_state():
    """5: Background exception/failure before replacement => failed state."""
    async def mock_call_docker(method, path, **kwargs):
        if "/containers/babel/json" in path:
            resp = MagicMock(status_code=200)
            resp.json.return_value = {"Config": {}, "HostConfig": {}, "NetworkSettings": {}}
            return resp
        if "/images/create" in path:
            return MagicMock(status_code=500, text="Registry down")
        return MagicMock(status_code=200)

    with patch.object(updater_module, "UPDATER_STATE", "idle"), \
         patch("app.updater.main.call_docker", side_effect=mock_call_docker):

        await updater_module.run_update("v2.3.33-beta")
        assert updater_module.UPDATER_STATE == "failed"

@pytest.mark.asyncio
async def test_updater_create_failure_triggers_rollback():
    """6: Failure after replacement => rollback is executed."""
    with patch.object(updater_module, "UPDATER_STATE", "idle"), \
         patch("app.updater.main.rollback", new_callable=AsyncMock) as mock_rollback, \
         patch("app.updater.main.call_docker", new_callable=AsyncMock) as mock_docker:

        inspect_resp = MagicMock(status_code=200)
        inspect_resp.json.return_value = {"Config": {"Image": "babel:latest"}, "HostConfig": {}, "NetworkSettings": {}}
        pull_resp = MagicMock(status_code=200)
        retag_resp = MagicMock(status_code=201)
        leftover_resp = MagicMock(status_code=404)
        stop_resp = MagicMock(status_code=204)
        rename_resp = MagicMock(status_code=204)
        create_resp = MagicMock(status_code=500, text="Invalid volume mount")

        mock_docker.side_effect = [inspect_resp, pull_resp, retag_resp, leftover_resp, stop_resp, rename_resp, create_resp]

        await updater_module.run_update("v2.3.33-beta")
        mock_rollback.assert_awaited_once_with("babel_rollback", "babel")

@pytest.mark.asyncio
async def test_updater_start_failure_triggers_rollback():
    """6: Start failure after replacement => rollback is executed."""
    with patch.object(updater_module, "UPDATER_STATE", "idle"), \
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

        await updater_module.run_update("v2.3.33-beta")
        mock_rollback.assert_awaited_once_with("babel_rollback", "babel")
