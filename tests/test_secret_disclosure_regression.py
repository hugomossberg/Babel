"""Settings responses and routine HTTP logs must never expose API credentials."""
import logging
from unittest.mock import patch

import httpx
import pytest

from app.api import dashboard
from app.core import db
from app.core.security import is_masked_secret, mask_secret, resolve_secret_key


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path / 'security.db'))
    db.init_db()
    monkeypatch.setattr(dashboard, '_MODELS_CACHE', {})


@pytest.mark.parametrize('secret', ['x', 'short', '1234567', '12345678', 'longer_secret_abcd'])
def test_every_nonempty_secret_is_masked_and_roundtrips(secret):
    db.set_setting('gemini_api_key', secret)
    masked = mask_secret(secret)
    assert masked != secret
    assert is_masked_secret(masked)
    assert resolve_secret_key(masked, 'gemini_api_key') == secret
    if len(secret) < 8:
        assert secret not in masked
    else:
        assert masked.endswith(secret[-4:])
    assert mask_secret('') == ''


@pytest.mark.asyncio
async def test_settings_response_does_not_expose_short_credentials():
    for setting in ('gemini_api_key', 'bazarr_api_key', 'jellyfin_api_key', 'plex_token'):
        db.set_setting(setting, 's3crt')
    settings = await dashboard.api_get_all_settings()
    assert settings['ai']['gemini_api_key'] != 's3crt'
    for setting in ('bazarr_api_key', 'jellyfin_api_key', 'plex_token'):
        assert settings['integrations'][setting] != 's3crt'


@pytest.mark.asyncio
async def test_gemini_discovery_uses_header_and_never_logs_key_in_url(caplog):
    secret = 'fixture_secret_never_log'
    db.set_setting('gemini_api_key', secret)
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={'models': []})

    factory = httpx.AsyncClient
    with patch.object(dashboard.httpx, 'AsyncClient', side_effect=lambda **kwargs: factory(
        transport=httpx.MockTransport(handle), **kwargs
    )), caplog.at_level(logging.INFO, logger='httpx'):
        await dashboard.api_get_provider_models(provider='gemini', url=None, refresh=True)
    assert len(requests) == 1
    assert secret not in caplog.text
    assert secret not in str(requests[0].url)
    assert requests[0].headers.get('x-goog-api-key') == secret
