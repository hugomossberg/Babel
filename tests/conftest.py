import asyncio
import pytest
import json
import socket
import ssl
import sys
import time
import httpx
from unittest.mock import patch, AsyncMock
from app.services.updates_controller import updates_controller


class _VirtualClock:
    """time.monotonic plus whatever idle time the virtual loop skipped.

    Everything else, time.sleep included, stays real: synchronous code may
    be pacing a real subprocess.
    """

    offset = 0.0

    def __getattr__(self, name):
        return getattr(time, name)

    def monotonic(self):
        return time.monotonic() + self.offset


_virtual_clock = _VirtualClock()


class _VirtualTimeLoop(asyncio.SelectorEventLoop):
    """Skip idle waits instead of sleeping through them.

    When nothing is ready, the loop jumps the clock to its next timer rather
    than blocking, so polls, backoffs and deadlines cost no wall time and do
    not race the machine's load. Executor jobs (provider SDK calls,
    to_thread) and open pipes or sockets (subprocesses, loopback servers)
    keep real time.
    """

    def __init__(self):
        super().__init__()
        self._executor_jobs = 0
        real_select = self._selector.select

        def select(timeout=None):
            # The loop's own self-pipe is always registered.
            busy = self._executor_jobs or len(self._selector.get_map()) > 1
            if not timeout or busy:
                return real_select(timeout)
            events = real_select(0)
            if not events:
                _virtual_clock.offset += timeout
            return events

        self._selector.select = select

    def time(self):
        return _virtual_clock.monotonic()

    def run_in_executor(self, executor, func, *args):
        future = super().run_in_executor(executor, func, *args)
        self._executor_jobs += 1

        def done(_):
            self._executor_jobs -= 1

        future.add_done_callback(done)
        return future

    async def shutdown_default_executor(self, timeout=None):
        self._executor_jobs += 1  # joining worker threads takes real time
        await super().shutdown_default_executor(timeout)


def pytest_asyncio_loop_factories(config, item):
    return {'virtual_clock': _VirtualTimeLoop}


@pytest.fixture(autouse=True)
def virtual_monotonic(monkeypatch):
    """App code measures its waits on the virtual loop's clock.

    Tests timing themselves use asyncio.get_running_loop().time().
    """
    for name, module in list(sys.modules.items()):
        if name.startswith('app.') and getattr(module, 'time', None) is time:
            monkeypatch.setattr(module, 'time', _virtual_clock)


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
    # No TLS happens, and the SDK builds a client (loading the CA bundle)
    # for every call.
    monkeypatch.setattr(ssl.SSLContext, 'load_verify_locations', lambda *a, **k: None)
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
