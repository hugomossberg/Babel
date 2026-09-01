import os
import re
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.config import VERSION
from app.services.updates_controller import UpdatesController

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_github_200_returns_latest_version():
    controller = UpdatesController()
    mock_releases = [
        {
            "tag_name": "v99.0.0-beta",
            "html_url": "https://github.com/hugomossberg/Babel/releases/tag/v99.0.0-beta",
            "published_at": "2026-08-28T00:00:00Z",
            "body": "New feature release"
        }
    ]
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = mock_releases

    with patch.object(controller, "get_real_updater_status", return_value=(True, "idle")), \
         patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        info = await controller.get_update_info(force_refresh=True)

        assert info["latest_version"] == "v99.0.0-beta"
        assert info["update_available"] is True
        assert info["update_check_ok"] is True
        assert info["update_check_error"] is None
        assert info["metadata_stale"] is False
        assert info["current_version"] == VERSION

@pytest.mark.asyncio
async def test_github_403_without_cache():
    controller = UpdatesController()
    mock_resp = MagicMock(status_code=403)
    mock_resp.headers = {"X-RateLimit-Reset": str(int(datetime.now(timezone.utc).timestamp()) + 3600)}

    with patch.object(controller, "get_real_updater_status", return_value=(True, "idle")), \
         patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        info = await controller.get_update_info(force_refresh=True)

        assert info["update_check_ok"] is False
        assert info["update_check_error"] is not None
        assert "403" in info["update_check_error"] or "rate limit" in info["update_check_error"].lower()
        assert info["latest_version"] is None
        assert info["update_available"] is False
        assert info["metadata_stale"] is False

@pytest.mark.asyncio
async def test_github_403_with_existing_cache_preserves_metadata():
    controller = UpdatesController()
    now = datetime.now(timezone.utc).timestamp()
    controller.cached_release_metadata = {
        "latest_version": "v99.0.0-beta",
        "release_url": "https://github.com/...",
        "release_notes": "Existing notes",
        "published_at": "2026-08-28T00:00:00Z"
    }
    controller.cache_time = now - 1000  # Expired cache

    mock_resp = MagicMock(status_code=403)
    mock_resp.headers = {"X-RateLimit-Reset": str(int(now) + 3600)}

    with patch.object(controller, "get_real_updater_status", return_value=(True, "idle")), \
         patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        info = await controller.get_update_info(force_refresh=True)

        assert info["update_check_ok"] is False
        assert info["latest_version"] == "v99.0.0-beta"
        assert info["update_available"] is True
        assert info["metadata_stale"] is True
        assert info["update_check_error"] is not None

@pytest.mark.asyncio
async def test_rate_limit_cooldown_prevents_requests():
    controller = UpdatesController()
    now = datetime.now(timezone.utc).timestamp()
    controller.next_allowed_github_check = now + 600  # 10 minutes in future

    with patch.object(controller, "get_real_updater_status", return_value=(True, "idle")), \
         patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        info = await controller.get_update_info(force_refresh=False)
        mock_get.assert_not_called()
        assert info["update_check_ok"] is False

        # Even with force_refresh=True, active cooldown must NOT be bypassed
        info_forced = await controller.get_update_info(force_refresh=True)
        mock_get.assert_not_called()
        assert info_forced["update_check_ok"] is False

@pytest.mark.asyncio
async def test_runtime_status_does_not_call_github():
    controller = UpdatesController()
    with patch.object(controller, "get_real_updater_status", return_value=(True, "updating")), \
         patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        status = await controller.get_runtime_status()
        mock_get.assert_not_called()
        assert status["current_version"] == VERSION
        assert status["updater_status"] == "updating"
        assert status["one_click_update_available"] is True

def test_api_updates_runtime_endpoint(client):
    with patch("app.services.updates_controller.UpdatesController.get_runtime_status", new_callable=AsyncMock) as mock_runtime:
        mock_runtime.return_value = {
            "current_version": VERSION,
            "updater_status": "pulling",
            "one_click_update_available": True
        }
        res = client.get("/api/updates/runtime")
        assert res.status_code == 200
        data = res.json()
        assert data["current_version"] == VERSION
        assert data["updater_status"] == "pulling"
        assert data["one_click_update_available"] is True

def test_frontend_uses_runtime_endpoint_in_pollhealth():
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    poll_health_match = re.search(r"pollHealth\(expectedVersion\) \{.*?(?=formatDate)", html, re.DOTALL)
    assert poll_health_match, "pollHealth function not found"
    poll_func = poll_health_match.group(0)

    assert "/api/updates/runtime" in poll_func
    assert "/api/updates?force=true" not in poll_func

def test_frontend_init_does_not_force_check():
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    init_match = re.search(r"async init\(\) \{.*?(?=async manualCheckUpdates)", html, re.DOTALL)
    assert init_match, "init function not found"
    init_func = init_match.group(0)

    assert "await this.checkUpdates();" in init_func
    assert "await this.checkUpdates(true);" not in init_func

def test_frontend_manual_check_forces_refresh():
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    manual_match = re.search(r"async manualCheckUpdates\(\) \{.*?(?=async checkUpdates)", html, re.DOTALL)
    assert manual_match, "manualCheckUpdates function not found"
    manual_func = manual_match.group(0)

    assert "/api/updates?force=true" in manual_func

def test_frontend_update_error_priority_over_up_to_date():
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    # Priority 5 (Error) must come before Priority 7 (Up to date)
    pos_error = html.find('updateCheckError')
    pos_up_to_date = html.find('!updateCheckError && !updateData.update_available')

    assert pos_error != -1
    assert pos_up_to_date != -1
    assert pos_error < pos_up_to_date
    assert "Unable to check for updates" in html

def test_dockerfile_accepts_babel_version():
    with open("Dockerfile", "r", encoding="utf-8") as f:
        dockerfile = f.read()

    assert "ARG BABEL_VERSION" in dockerfile
    assert "ENV BABEL_VERSION=${BABEL_VERSION}" in dockerfile

def test_config_reads_env_babel_version(monkeypatch):
    import importlib
    import app.config

    original_env = os.getenv("BABEL_VERSION")
    try:
        # Determine baseline fallback version when env is missing
        monkeypatch.delenv("BABEL_VERSION", raising=False)
        importlib.reload(app.config)
        fallback_version = app.config.VERSION
        assert bool(fallback_version)
        assert isinstance(fallback_version, str)

        # Case 1: Custom version provided
        monkeypatch.setenv("BABEL_VERSION", "2.9.9-beta")
        importlib.reload(app.config)
        assert app.config.VERSION == "2.9.9-beta"

        # Case 2: Empty string in environment (e.g. untagged Docker build without build arg)
        monkeypatch.setenv("BABEL_VERSION", "")
        importlib.reload(app.config)
        assert app.config.VERSION == fallback_version

        # Case 3: Missing from environment
        monkeypatch.delenv("BABEL_VERSION", raising=False)
        importlib.reload(app.config)
        assert app.config.VERSION == fallback_version
    finally:
        if original_env is not None:
            monkeypatch.setenv("BABEL_VERSION", original_env)
        else:
            monkeypatch.delenv("BABEL_VERSION", raising=False)
        importlib.reload(app.config)

def test_docker_publish_workflow_version_extraction():
    with open(".github/workflows/docker-publish.yml", "r", encoding="utf-8") as f:
        wf = f.read()

    assert 'VERSION="${GITHUB_REF_NAME#v}"' in wf
    assert 'BABEL_VERSION=${{ steps.version.outputs.version }}' in wf
