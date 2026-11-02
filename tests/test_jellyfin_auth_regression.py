"""Jellyfin auth compatibility and safe connection/notification failures (#20)."""
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.services import jellyfin_notifier


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["10.8.13", "12.2.0"])
async def test_connection_and_refresh_use_supported_authorization(version, monkeypatch):
    """Both endpoints work when legacy headers are disabled; no token in URLs."""
    requests = []

    def handle(request):
        requests.append(request)
        assert request.headers.get("Authorization") == 'MediaBrowser Token="test_api_key"'
        assert "X-Emby-Token" not in request.headers
        assert "test_api_key" not in str(request.url)
        if request.method == "GET":
            assert request.url.path == "/jellyfin/System/Info"
            return httpx.Response(200, json={"Version": version, "ServerName": "Test"})
        assert request.method == "POST"
        assert request.url.path == "/jellyfin/Library/Refresh"
        return httpx.Response(204)

    settings = {"notify_jellyfin": "true", "jellyfin_url": "http://server/jellyfin/", "jellyfin_api_key": "test_api_key"}
    monkeypatch.setattr(jellyfin_notifier, "get_setting", lambda key, default="": settings.get(key, default))
    client_factory = httpx.AsyncClient
    with patch.object(jellyfin_notifier.httpx, "AsyncClient", side_effect=lambda **kwargs: client_factory(
        transport=httpx.MockTransport(handle), **kwargs
    )):
        result = await jellyfin_notifier.check_jellyfin_connection(settings["jellyfin_url"], "test_api_key")
        await jellyfin_notifier.notify_jellyfin_library_refresh()
    assert result == {"connected": True, "version": f"v{version}", "server_name": "Test", "status": "ok"}
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_auth_value_cannot_inject_authorization_parameters():
    """Jellyfin's quoted auth values use percent encoding, not raw interpolation."""
    token = 'key", Client="injected\r\nX-Test: value'
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as get:
        get.return_value = httpx.Response(401)
        await jellyfin_notifier.check_jellyfin_connection("http://server", token)
    assert get.call_args.kwargs["headers"]["Authorization"] == (
        'MediaBrowser Token="key%22%2C%20Client%3D%22injected%0D%0AX-Test%3A%20value"'
    )
    assert token not in get.call_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("status, message", [
    (401, "Invalid Jellyfin API Token"),
    (403, "Invalid Jellyfin API Token"),
    (404, "endpoint not found"),
    (500, "HTTP 500"),
    (302, "HTTP 302"),
])
async def test_connection_http_errors_are_safe_and_do_not_retry(status, message):
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as get:
        get.return_value = httpx.Response(status, text="secret_token_from_error_body")
        result = await jellyfin_notifier.check_jellyfin_connection("http://server", "secret_token")
    assert result["connected"] is False
    assert message in result["message"]
    assert "secret_token" not in str(result)
    get.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [
    httpx.Response(200, text="<html>proxy login</html>"),
    httpx.Response(200, json=[]),
    httpx.Response(200, json="wrong endpoint"),
])
async def test_connection_rejects_invalid_success_response(response):
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=response):
        result = await jellyfin_notifier.check_jellyfin_connection("http://server", "secret_token")
    assert result["connected"] is False
    assert result["message"] == "Invalid response from Jellyfin server"


@pytest.mark.asyncio
@pytest.mark.parametrize("error, message", [
    (httpx.ReadTimeout("secret_token"), "Connection timed out"),
    (httpx.ConnectError("secret_token"), "Unable to reach Jellyfin server"),
    (RuntimeError("secret_token"), "Connection failed"),
])
async def test_connection_exceptions_do_not_expose_secret(error, message, caplog):
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=error):
        result = await jellyfin_notifier.check_jellyfin_connection("http://server", "secret_token")
    assert result == {"connected": False, "message": message}
    assert "secret_token" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [httpx.Response(401), httpx.Response(500), httpx.ReadTimeout("secret_token"), RuntimeError("secret_token")])
async def test_refresh_errors_are_nonfatal_and_do_not_expose_secrets(failure, monkeypatch, caplog):
    settings = {"notify_jellyfin": "true", "jellyfin_url": "http://server", "jellyfin_api_key": "secret_token"}
    monkeypatch.setattr(jellyfin_notifier, "get_setting", lambda key, default="": settings.get(key, default))
    kwargs = {"side_effect": failure} if isinstance(failure, Exception) else {"return_value": failure}
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, **kwargs) as post:
        await jellyfin_notifier.notify_jellyfin_library_refresh()
    post.assert_awaited_once()
    assert "secret_token" not in caplog.text
    assert "Jellyfin" in caplog.text
