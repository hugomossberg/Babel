import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from app.services.updates_controller import UpdatesController

@pytest.mark.asyncio
async def test_version_parsing():
    controller = UpdatesController()
    assert controller._parse_version("v2.3.25-beta") == (2, 3, 25)
    assert controller._parse_version("2.3.25-beta") == (2, 3, 25)
    assert controller._parse_version("v2.3.26") == (2, 3, 26)
    assert controller._parse_version("2.3.26") == (2, 3, 26)
    assert controller._parse_version("v2.3.9-beta") < controller._parse_version("v2.3.10-beta")
    assert controller._parse_version("junk") == (0, 0, 0)
    assert controller._parse_version("") == (0, 0, 0)
    assert controller._parse_version(None) == (0, 0, 0)

@pytest.mark.asyncio
async def test_cached_updates():
    from datetime import datetime, timezone
    controller = UpdatesController()
    controller.cached_info = {"latest_version": "v2.3.27-beta", "update_available": True}
    controller.cache_time = datetime.now(timezone.utc).timestamp()

    info = await controller.get_update_info(force_refresh=False)
    assert info["latest_version"] == "v2.3.27-beta"

@pytest.mark.asyncio
async def test_release_channel_and_notes_bounding():
    controller = UpdatesController()

    mock_releases = [
        {
            "tag_name": "v2.3.29-beta",
            "html_url": "https://github.com/hugomossberg/Babel/releases/tag/v2.3.29-beta",
            "published_at": "2026-08-23T00:00:00Z",
            "body": "X" * 1500
        },
        {
            "tag_name": "v3.0.0", # Non-beta should be ignored on beta channel
            "html_url": "https://github.com/hugomossberg/Babel/releases/tag/v3.0.0",
            "published_at": "2026-08-23T00:00:00Z",
            "body": "Major release"
        }
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_releases

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        info = await controller.get_update_info(force_refresh=True)
        assert info["update_available"] is True
        assert info["latest_version"] == "v2.3.29-beta"
        assert len(info["release_notes"]) <= 1040
        assert "[View full release on GitHub]" in info["release_notes"]

@pytest.mark.asyncio
async def test_trigger_update_validations():
    controller = UpdatesController()

    # 1. Blocked if locked
    controller.is_maintenance_locked = True
    res = await controller.trigger_update("v2.3.27-beta")
    assert res["success"] is False
    assert "already in progress" in res["message"]
    controller.is_maintenance_locked = False

    # 2. Blocked if updater unreachable
    with patch.object(controller, "get_real_updater_status", new_callable=AsyncMock) as mock_st:
        mock_st.return_value = (False, "idle")
        res = await controller.trigger_update("v2.3.27-beta")
        assert res["success"] is False
        assert "unreachable" in res["message"]
        assert controller.is_maintenance_locked is False

    # 3. Blocked if active jobs running
    with patch.object(controller, "get_real_updater_status", new_callable=AsyncMock) as mock_st, \
         patch("app.core.db.get_jobs_by_status", return_value=[{"id": 1, "status": "TRANSLATING"}]):
        mock_st.return_value = (True, "idle")
        res = await controller.trigger_update("v2.3.27-beta")
        assert res["success"] is False
        assert "Active jobs are running" in res["message"]
        assert controller.is_maintenance_locked is False

    # 4. Blocked if downgrade / same version
    with patch.object(controller, "get_real_updater_status", new_callable=AsyncMock) as mock_st, \
         patch("app.core.db.get_jobs_by_status", return_value=[]), \
         patch.object(controller, "get_update_info", new_callable=AsyncMock) as mock_info:
        mock_st.return_value = (True, "idle")
        mock_info.return_value = {"update_available": True, "latest_version": "v2.3.24-beta"}
        res = await controller.trigger_update("v2.3.24-beta")
        assert res["success"] is False
        assert "newer than current version" in res["message"]
        assert controller.is_maintenance_locked is False

    # 5. Blocked if target does not match latest release
    with patch.object(controller, "get_real_updater_status", new_callable=AsyncMock) as mock_st, \
         patch("app.core.db.get_jobs_by_status", return_value=[]), \
         patch.object(controller, "get_update_info", new_callable=AsyncMock) as mock_info:
        mock_st.return_value = (True, "idle")
        mock_info.return_value = {"update_available": True, "latest_version": "v2.3.29-beta"}
        res = await controller.trigger_update("v2.3.30-beta")
        assert res["success"] is False
        assert "does not match verified latest release" in res["message"]
        assert controller.is_maintenance_locked is False

    # 6. Success triggers updater
    with patch.object(controller, "get_real_updater_status", new_callable=AsyncMock) as mock_st, \
         patch("app.core.db.get_jobs_by_status", return_value=[]), \
         patch.object(controller, "get_update_info", new_callable=AsyncMock) as mock_info, \
         patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_st.return_value = (True, "idle")
        mock_info.return_value = {"update_available": True, "latest_version": "v2.3.29-beta"}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        res = await controller.trigger_update("v2.3.29-beta")
        assert res["success"] is True
        assert controller.update_status == "updating"

@pytest.mark.asyncio
async def test_get_real_updater_status_auth():
    controller = UpdatesController()

    # 1. Secret missing/wrong (auth fails, status returns 401)
    # Even if health returns 200, status returns 401, so available should be False
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        def side_effect(url, **kwargs):
            mock_resp = MagicMock()
            if url.endswith("/health"):
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"status": "updater_healthy"}
                return mock_resp
            if url.endswith("/status"):
                mock_resp.status_code = 401
                mock_resp.json.return_value = {"detail": "Invalid or missing updater token"}
                return mock_resp
        mock_get.side_effect = side_effect

        avail, status = await controller.get_real_updater_status()
        assert avail is False
        assert status == "idle"

    # 2. Secret correct (auth succeeds, status returns 200)
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        def side_effect(url, **kwargs):
            mock_resp = MagicMock()
            if url.endswith("/health"):
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"status": "updater_healthy"}
                return mock_resp
            if url.endswith("/status"):
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"status": "idle"}
                return mock_resp
        mock_get.side_effect = side_effect

        avail, status = await controller.get_real_updater_status()
        assert avail is True
        assert status == "idle"

@pytest.mark.asyncio
async def test_one_click_update_disabled_if_auth_fails():
    controller = UpdatesController()

    # When auth fails, one_click_update_available should evaluate to false
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        def side_effect(url, **kwargs):
            mock_resp = MagicMock()
            if url.endswith("/health"):
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"status": "updater_healthy"}
                return mock_resp
            if url.endswith("/status"):
                mock_resp.status_code = 401
                return mock_resp
            # For github releases
            mock_resp.status_code = 200
            mock_resp.json.return_value = []
            return mock_resp
        mock_get.side_effect = side_effect

        info = await controller.get_update_info(force_refresh=True)
        assert info["one_click_update_available"] is False

    # When auth succeeds, one_click_update_available should evaluate to true
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        def side_effect(url, **kwargs):
            mock_resp = MagicMock()
            if url.endswith("/health"):
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"status": "updater_healthy"}
                return mock_resp
            if url.endswith("/status"):
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"status": "idle"}
                return mock_resp
            # For github releases
            mock_resp.status_code = 200
            mock_resp.json.return_value = []
            return mock_resp
        mock_get.side_effect = side_effect

        info = await controller.get_update_info(force_refresh=True)
        assert info["one_click_update_available"] is True

