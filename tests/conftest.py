import pytest
import json
import socket
import httpx
from unittest.mock import patch, AsyncMock
from app.services.updates_controller import updates_controller


@pytest.fixture(autouse=True)
def forbid_external_network(monkeypatch):
    """Catch missing mocks even when application error handling catches the error.

    In-process transports and loopback test servers still work. Tests must mock
    external HTTP/SDK contracts, never depend on DNS, credentials or live APIs.
    """
    attempts = []
    real_resolve = socket.getaddrinfo
    real_connect = socket.socket.connect

    def check(host):
        if host not in (None, 'localhost', '127.0.0.1', '::1', b'localhost', b'127.0.0.1', b'::1'):
            attempts.append(str(host))
            raise AssertionError(f'Unmocked external network access: {host}')

    def resolve(host, *args, **kwargs):
        check(host)
        return real_resolve(host, *args, **kwargs)

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            check(address[0])
        return real_connect(sock, address)

    monkeypatch.setattr(socket, 'getaddrinfo', resolve)
    monkeypatch.setattr(socket.socket, 'connect', connect)
    yield
    assert not attempts, f'Missing external-service mock(s): {attempts}'


@pytest.fixture
def empty_gemini_provider(monkeypatch):
    """Successful provider HTTP, but unusable translation/recovery content.

    Exercise the real SDK, parsing, retries, recovery and QA. Only the network
    transport is replaced; no QA decision or publication function is mocked.
    """
    state = {'calls': [], 'failure_contract': None, 'failure': None}

    def handle_request(transport, request):
        if request.url.host != 'generativelanguage.googleapis.com':
            pytest.fail(f'Unexpected provider in Gemini fixture: {request.url.host}')
        body = json.loads(request.content)
        schema = body.get('generationConfig', {}).get('responseSchema', {})
        properties = schema.get('properties', {})
        if not properties:
            pytest.fail('Every fixture completion must declare its contract')
        # Valid JSON and a successful SDK response are not proof of translation
        # quality: all the actual candidate/recovery values remain unusable.
        content = {key: [] if key in ('translations', 'results', 'batches') else ''
                   for key in properties}
        state['calls'].append({'body': body, 'properties': set(properties)})
        if state['failure_contract'] in properties:
            if state['failure'] == 'timeout':
                raise httpx.ReadTimeout('Deterministic recovery timeout', request=request)
            if state['failure'] == 'unauthorized':
                return httpx.Response(401, request=request, json={
                    'error': {'code': 401, 'status': 'UNAUTHENTICATED', 'message': 'Invalid fixture API key'},
                })
        return httpx.Response(200, request=request, json={
            'candidates': [{'content': {'role': 'model', 'parts': [{'text': json.dumps(content)}]},
                            'finishReason': 'STOP'}],
            'usageMetadata': {'promptTokenCount': 10, 'candidatesTokenCount': 5, 'totalTokenCount': 15},
        })

    monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', handle_request)
    return state


@pytest.fixture
def indexed_bazarr_http(tmp_path, monkeypatch):
    """Bazarr HTTP contracts for lifecycle tests; actual coordination still runs."""
    calls = []

    async def handle_request(transport, request):
        if request.url.host != 'bazarr':
            pytest.fail(f'Unexpected host in Bazarr fixture: {request.url.host}')
        calls.append((request.method, request.url.path))
        if request.method == 'GET' and request.url.path == '/api/system/jobs':
            data = {'data': []}
        elif request.method == 'GET' and request.url.path == '/api/movies':
            data = {'data': [{'radarrId': i + 1, 'path': str(path), 'title': path.stem}
                             for i, path in enumerate(sorted(tmp_path.rglob('*.mkv')))]}
        elif request.method == 'PATCH' and request.url.path == '/api/movies/subtitles':
            data = {'message': 'Fixture search accepted'}
        else:
            pytest.fail(f'Unexpected Bazarr HTTP contract: {request.method} {request.url.path}')
        return httpx.Response(200, request=request, json=data)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, 'handle_async_request', handle_request)
    return calls


@pytest.fixture(autouse=True)
def reset_updates_controller_state():
    updates_controller.is_maintenance_locked = False
    updates_controller.update_status = "idle"
    yield
    updates_controller.is_maintenance_locked = False
    updates_controller.update_status = "idle"


@pytest.fixture(autouse=True)
def hermetic_semantic_audit(request):
    """
    Dependency seam: ensures that audit_batch_semantic_integrity never makes
    live API calls during the test suite.

    For tests that do NOT explicitly mock audit_batch_semantic_integrity on their
    SubtitleTranslator instance, this autouse fixture provides a safe ALIGNED
    default response (class-level patch on SubtitleTranslator).

    Tests that set pipeline.translator.audit_batch_semantic_integrity = AsyncMock(...)
    (i.e. instance-level override) will use their own mock and this default will be
    shadowed, so the patch is fully transparent to test-specific mocks.

    Tests decorated with @pytest.mark.skip_hermetic_audit receive the REAL
    audit_batch_semantic_integrity implementation so they can test contract logic
    directly via _dispatch_llm_completion mocks.

    This guarantees:
    - 0 live paid API calls from pytest.
    - No fail-closed SUSPECT cascades in hermetic tests that don't care about semantics.
    - Explicit semantic audit tests control their own mock and are unaffected.
    - Contract hardening tests can test the real code via the marker.
    """
    if request.node.get_closest_marker("skip_hermetic_audit"):
        # Let the real implementation run — the test is responsible for
        # mocking _dispatch_llm_completion to prevent live API calls.
        yield
        return

    async def _default_audit(self, batch_payloads, **kwargs):
        return {
            bp["batch_id"]: {
                "batch_id": bp["batch_id"],
                "verdict": "ALIGNED",
                "confidence": "HIGH",
                "details": "hermetic_semantic_audit fixture default"
            }
            for bp in batch_payloads
        }

    with patch(
        "app.services.translator.SubtitleTranslator.audit_batch_semantic_integrity",
        new=_default_audit,
    ):
        yield
