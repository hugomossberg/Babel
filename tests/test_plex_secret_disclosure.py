"""Plex authentication must never become part of a logged request URL."""
import logging
from unittest.mock import patch

import httpx
import pytest

from app.services import plex_notifier as plex

TOKEN = 'plex-regression-fake-secret'


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['connection', 'notification'])
@pytest.mark.parametrize('status', [200, 401, 500])
async def test_plex_token_is_header_only_on_success_and_http_failure(operation, status, caplog):
    caplog.set_level(logging.INFO, logger='httpx')
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(status, text='<MediaContainer><Directory key="1"><Location path="/movies"/></Directory></MediaContainer>')
    original = httpx.AsyncClient
    def client(**kwargs):
        return original(**kwargs, transport=httpx.MockTransport(handle))
    settings = {'plex_url':'http://plex-audit.invalid', 'plex_token':TOKEN}
    with patch.object(plex.httpx, 'AsyncClient', side_effect=client), patch.object(plex, 'get_setting', side_effect=lambda key, default='': settings.get(key,default)):
        if operation == 'connection':
            result = await plex.check_plex_connection(settings['plex_url'], TOKEN)
            assert result['connected'] is (status == 200)
        else:
            await plex.notify_plex_library_refresh('/movies/Fixture.mkv')
    assert requests
    assert all(request.headers.get('X-Plex-Token') == TOKEN for request in requests)
    assert all(TOKEN not in str(request.url) for request in requests)
    assert TOKEN not in caplog.text
    if operation == 'notification' and status == 200:
        assert len(requests) == 2
        assert requests[-1].url.path == '/library/sections/1/refresh'


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['connection', 'notification'])
async def test_plex_exception_text_cannot_disclose_token(operation, caplog):
    def fail(request):
        raise RuntimeError('Provider echoed ' + TOKEN)
    original = httpx.AsyncClient
    def client(**kwargs):
        return original(**kwargs, transport=httpx.MockTransport(fail))
    settings = {'plex_url':'http://plex-audit.invalid','plex_token':TOKEN}
    with patch.object(plex.httpx,'AsyncClient',side_effect=client), patch.object(plex,'get_setting',side_effect=lambda key,default='':settings.get(key,default)):
        if operation == 'connection':
            result=await plex.check_plex_connection(settings['plex_url'],TOKEN)
            assert result['connected'] is False
            assert TOKEN not in str(result)
        else:
            await plex.notify_plex_library_refresh('/movies/Fixture.mkv')
    assert TOKEN not in caplog.text
