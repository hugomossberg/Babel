import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
import httpx

from app.main import app
from app.core.db import get_setting, set_setting
from app.services.jellyfin_notifier import check_jellyfin_connection, notify_jellyfin_library_refresh
from app.services.plex_notifier import (
    check_plex_connection,
    notify_plex_library_refresh,
    _map_to_plex_path,
    _path_is_within,
)

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture(autouse=True)
def clean_media_settings():
    old_jf_key = get_setting("jellyfin_api_key", "")
    old_jf_url = get_setting("jellyfin_url", "")
    old_jf_notify = get_setting("notify_jellyfin", "false")
    old_plex_token = get_setting("plex_token", "")
    old_plex_url = get_setting("plex_url", "")
    old_plex_notify = get_setting("notify_plex", "false")
    old_plex_babel_pref = get_setting("plex_path_babel_prefix", "")
    old_plex_plex_pref = get_setting("plex_path_plex_prefix", "")
    yield
    set_setting("jellyfin_api_key", old_jf_key)
    set_setting("jellyfin_url", old_jf_url)
    set_setting("notify_jellyfin", old_jf_notify)
    set_setting("plex_token", old_plex_token)
    set_setting("plex_url", old_plex_url)
    set_setting("notify_plex", old_plex_notify)
    set_setting("plex_path_babel_prefix", old_plex_babel_pref)
    set_setting("plex_path_plex_prefix", old_plex_plex_pref)

# 1. Independent toggles & simultaneous ON support
def test_1_independent_settings_and_both_enabled_simultaneously(client):
    """Jellyfin and Plex are independent toggles; both can be enabled or disabled simultaneously."""
    # Both ON
    resp = client.post("/api/settings/integrations", json={
        "enable_bazarr_check": True,
        "bazarr_url": "http://bazarr:6767",
        "bazarr_api_key": "",
        "bazarr_container_name": "bazarr",
        "notify_jellyfin": True,
        "jellyfin_url": "http://jellyfin:8096",
        "jellyfin_api_key": "",
        "notify_plex": True,
        "plex_url": "http://plex:32400",
        "plex_token": "",
        "plex_path_babel_prefix": "",
        "plex_path_plex_prefix": ""
    })
    assert resp.status_code == 200
    all_settings = client.get("/api/settings/all").json()
    assert all_settings["integrations"]["notify_jellyfin"] is True
    assert all_settings["integrations"]["notify_plex"] is True

    # Jellyfin ON, Plex OFF
    client.post("/api/settings/integrations", json={
        "enable_bazarr_check": True,
        "bazarr_url": "http://bazarr:6767",
        "bazarr_api_key": "",
        "bazarr_container_name": "bazarr",
        "notify_jellyfin": True,
        "jellyfin_url": "http://jellyfin:8096",
        "jellyfin_api_key": "",
        "notify_plex": False,
        "plex_url": "http://plex:32400",
        "plex_token": "",
        "plex_path_babel_prefix": "",
        "plex_path_plex_prefix": ""
    })
    all_settings = client.get("/api/settings/all").json()
    assert all_settings["integrations"]["notify_jellyfin"] is True
    assert all_settings["integrations"]["notify_plex"] is False

    # Jellyfin OFF, Plex ON
    client.post("/api/settings/integrations", json={
        "enable_bazarr_check": True,
        "bazarr_url": "http://bazarr:6767",
        "bazarr_api_key": "",
        "bazarr_container_name": "bazarr",
        "notify_jellyfin": False,
        "jellyfin_url": "http://jellyfin:8096",
        "jellyfin_api_key": "",
        "notify_plex": True,
        "plex_url": "http://plex:32400",
        "plex_token": "",
        "plex_path_babel_prefix": "",
        "plex_path_plex_prefix": ""
    })
    all_settings = client.get("/api/settings/all").json()
    assert all_settings["integrations"]["notify_jellyfin"] is False
    assert all_settings["integrations"]["notify_plex"] is True

    # Both OFF
    client.post("/api/settings/integrations", json={
        "enable_bazarr_check": True,
        "bazarr_url": "http://bazarr:6767",
        "bazarr_api_key": "",
        "bazarr_container_name": "bazarr",
        "notify_jellyfin": False,
        "jellyfin_url": "http://jellyfin:8096",
        "jellyfin_api_key": "",
        "notify_plex": False,
        "plex_url": "http://plex:32400",
        "plex_token": "",
        "plex_path_babel_prefix": "",
        "plex_path_plex_prefix": ""
    })
    all_settings = client.get("/api/settings/all").json()
    assert all_settings["integrations"]["notify_jellyfin"] is False
    assert all_settings["integrations"]["notify_plex"] is False

# 2. Plex path mapping optional & empty behavior
def test_2_plex_path_mapping_optional_and_empty():
    """Empty/unconfigured Plex path mapping returns input path unmodified."""
    set_setting("plex_path_babel_prefix", "")
    set_setting("plex_path_plex_prefix", "")
    assert _map_to_plex_path("/media/tv/Show/ep.srt") == "/media/tv/Show/ep.srt"
    assert _map_to_plex_path("/any/custom/path/file.mkv") == "/any/custom/path/file.mkv"

# 3. Plex test connection success
@pytest.mark.asyncio
async def test_3_plex_test_connection_success(client):
    """Plex test connection succeeds against a real read-only endpoint (/library/sections)."""
    sections_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <MediaContainer size="2">
      <Directory key="1" title="Movies" />
      <Directory key="2" title="TV" />
    </MediaContainer>"""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = httpx.Response(200, text=sections_xml)
        resp = client.post("/api/settings/test-plex", json={
            "plex_url": "http://plex:32400",
            "plex_token": "valid_token_123"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["sections_count"] == 2
        mock_get.assert_called_once()
        assert mock_get.call_args[0][0] == "http://plex:32400/library/sections"

# 4. Plex test connection invalid token
@pytest.mark.asyncio
async def test_4_plex_test_connection_invalid_token(client):
    """Plex test connection detects invalid auth tokens (401/403) cleanly."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = httpx.Response(401)
        resp = client.post("/api/settings/test-plex", json={
            "plex_url": "http://plex:32400",
            "plex_token": "secret_bad_token"
        })
        assert resp.status_code == 400
        assert "Invalid Plex Token" in resp.json()["detail"]
        assert "secret_bad_token" not in resp.text

# 5. Plex test connection unreachable & timeout
@pytest.mark.asyncio
async def test_5_plex_test_connection_unreachable_and_timeout(client):
    """Plex test connection handles connection errors and timeouts gracefully."""
    with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("Connection refused")):
        resp = client.post("/api/settings/test-plex", json={
            "plex_url": "http://unreachable:32400",
            "plex_token": "secret_token_123"
        })
        assert resp.status_code == 400
        assert "Unable to reach Plex server" in resp.json()["detail"]
        assert "secret_token_123" not in resp.text

    with patch("httpx.AsyncClient.get", side_effect=httpx.TimeoutException("Timeout")):
        resp = client.post("/api/settings/test-plex", json={
            "plex_url": "http://timeout:32400",
            "plex_token": "secret_token_123"
        })
        assert resp.status_code == 400
        assert "Connection timed out" in resp.json()["detail"]

# 6. Plex test connection uses saved key when masked
def test_6_plex_test_connection_uses_saved_key_when_masked(client):
    """Masked token resolves to stored DB secret during test connection."""
    set_setting("plex_token", "real_stored_plex_token_999")
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = httpx.Response(200, text="<MediaContainer size='1'><Directory key='1'/></MediaContainer>")
        resp = client.post("/api/settings/test-plex", json={
            "plex_url": "http://plex:32400",
            "plex_token": "••••••••0999"
        })
        assert resp.status_code == 200
        mock_get.assert_called_once()
        assert mock_get.call_args[1]["params"]["X-Plex-Token"] == "real_stored_plex_token_999"

# 7. Jellyfin test connection success
@pytest.mark.asyncio
async def test_7_jellyfin_test_connection_success(client):
    """Jellyfin test connection verifies authenticated access against /System/Info."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = httpx.Response(200, json={"Version": "10.8.13", "ServerName": "Homelab-Jellyfin"})
        resp = client.post("/api/settings/test-jellyfin", json={
            "jellyfin_url": "http://jellyfin:8096",
            "jellyfin_api_key": "valid_jf_token"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "v10.8.13"
        assert data["server_name"] == "Homelab-Jellyfin"
        assert mock_get.call_args[0][0] == "http://jellyfin:8096/System/Info"
        assert mock_get.call_args[1]["headers"]["X-Emby-Token"] == "valid_jf_token"

# 8. Jellyfin test connection invalid token
@pytest.mark.asyncio
async def test_8_jellyfin_test_connection_invalid_token(client):
    """Jellyfin test connection cleanly reports 401/403 unauthorized tokens."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = httpx.Response(401)
        resp = client.post("/api/settings/test-jellyfin", json={
            "jellyfin_url": "http://jellyfin:8096",
            "jellyfin_api_key": "secret_bad_jf_token"
        })
        assert resp.status_code == 400
        assert "Invalid Jellyfin API Token" in resp.json()["detail"]
        assert "secret_bad_jf_token" not in resp.text

# 9. Jellyfin test connection unreachable & timeout
@pytest.mark.asyncio
async def test_9_jellyfin_test_connection_unreachable_and_timeout(client):
    """Jellyfin test connection handles network unreachable and timeout without leaking tokens."""
    with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("Connection refused")):
        resp = client.post("/api/settings/test-jellyfin", json={
            "jellyfin_url": "http://unreachable:8096",
            "jellyfin_api_key": "secret_jf_token"
        })
        assert resp.status_code == 400
        assert "Unable to reach Jellyfin server" in resp.json()["detail"]
        assert "secret_jf_token" not in resp.text

    with patch("httpx.AsyncClient.get", side_effect=httpx.TimeoutException("Timeout")):
        resp = client.post("/api/settings/test-jellyfin", json={
            "jellyfin_url": "http://timeout:8096",
            "jellyfin_api_key": "secret_jf_token"
        })
        assert resp.status_code == 400
        assert "Connection timed out" in resp.json()["detail"]

# 10. Jellyfin test connection uses saved key when masked
def test_10_jellyfin_test_connection_uses_saved_key_when_masked(client):
    """Masked Jellyfin token resolves to stored DB secret during test connection."""
    set_setting("jellyfin_api_key", "real_stored_jf_key_777")
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = httpx.Response(200, json={"Version": "10.8.13"})
        resp = client.post("/api/settings/test-jellyfin", json={
            "jellyfin_url": "http://jellyfin:8096",
            "jellyfin_api_key": "••••••••0777"
        })
        assert resp.status_code == 200
        mock_get.assert_called_once()
        assert mock_get.call_args[1]["headers"]["X-Emby-Token"] == "real_stored_jf_key_777"

# 11. Plex boundary-safe mapping & longest prefix preference
def test_11_plex_boundary_safe_mapping_and_longest_prefix():
    """Boundary-safe path mapping prevents substring collisions (/tv vs /tv4k) and prefers longest prefix."""
    set_setting("plex_path_babel_prefix", "/tv")
    set_setting("plex_path_plex_prefix", "/mnt/media/tv")
    # /tv4k must not be mapped by /tv prefix rule
    assert _map_to_plex_path("/tv4k/Show/ep.srt") == "/tv4k/Show/ep.srt"
    # exact /tv prefix maps correctly
    assert _map_to_plex_path("/tv/Show/ep.srt") == "/mnt/media/tv/Show/ep.srt"

    # Overlapping prefixes: longest match wins
    set_setting("plex_path_babel_prefix", "/data,/data/media/tv4k")
    set_setting("plex_path_plex_prefix", "/mnt/data,/library/uhd")
    assert _map_to_plex_path("/data/media/tv4k/Show/ep.srt") == "/library/uhd/Show/ep.srt"
    assert _map_to_plex_path("/data/media/tv/Show/ep.srt") == "/mnt/data/media/tv/Show/ep.srt"

# 12. Plex notifier failure never breaks translation or raises
@pytest.mark.asyncio
async def test_12_plex_notifier_failure_is_non_fatal():
    """Plex notifier errors are logged as warnings and do not raise or fail jobs."""
    set_setting("plex_url", "http://failing-plex:32400")
    set_setting("plex_token", "some_token")
    with patch("httpx.AsyncClient.get", side_effect=Exception("Fatal Plex API explosion")):
        # Must execute cleanly without raising
        await notify_plex_library_refresh("/tv/Show/ep.srt")

# 13. Jellyfin notifier respects setting toggle
@pytest.mark.asyncio
async def test_13_jellyfin_notifier_respects_setting():
    """Jellyfin notifier does not make network calls when disabled."""
    set_setting("notify_jellyfin", "false")
    set_setting("jellyfin_url", "http://jellyfin:8096")
    set_setting("jellyfin_api_key", "test_key")
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        await notify_jellyfin_library_refresh()
        mock_post.assert_not_called()
