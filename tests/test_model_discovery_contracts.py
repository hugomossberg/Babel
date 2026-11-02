"""Offline contracts for model discovery; never performs inference or external I/O."""
import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.dashboard import _MODELS_CACHE
from app.core.db import set_setting, get_setting
from app.core.ai_providers import filter_gemini_models, filter_openai_models, filter_anthropic_models


@pytest.fixture(autouse=True)
def isolated_discovery():
    _MODELS_CACHE.clear()
    for provider in ("gemini", "openai", "anthropic", "deepseek", "custom_openai"):
        set_setting(provider + "_api_key", "")
    yield
    _MODELS_CACHE.clear()


def response(payload, status=200):
    return httpx.Response(status, json=payload, request=httpx.Request("GET", "https://provider.invalid/models"))


def fetch(provider, **params):
    return TestClient(app).get("/api/settings/models", params={"provider": provider, **params})


@pytest.mark.parametrize("provider,pages,cursor", [
    ("gemini", [{"models": [{"name": "models/Gemini-Custom", "supportedGenerationMethods": ["generateContent"]}], "nextPageToken": "page-two"}, {"models": [{"name": "models/Gemini-Next", "supportedGenerationMethods": ["generateContent"]}]}], "pageToken"),
    ("anthropic", [{"data": [{"id": "claude-custom"}], "has_more": True, "last_id": "claude-custom"}, {"data": [{"id": "claude-next"}], "has_more": False}], "after_id"),
])
def test_pagination_and_exact_ids(provider, pages, cursor):
    set_setting(provider + "_api_key", "test-secret")
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=[response(p) for p in pages]) as get:
        result = fetch(provider).json()
    assert result["source"] == "live"
    assert len(result["models"]) == 2
    assert get.call_count == 2
    assert cursor in get.call_args.kwargs["params"]
    if provider == "gemini":
        assert result["models"][0]["id"] == "Gemini-Custom"
    assert "test-secret" not in str(result)
    assert all("test-secret" not in str(c.args) for c in get.call_args_list)


def test_deepseek_live_endpoint_and_text_modalities():
    set_setting("deepseek_api_key", "secret")
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=response({"data": [
        {"id": "deepseek-flash", "input_modalities": ["text"], "output_modalities": ["text"]},
        {"id": "embed-only", "output_modalities": ["embedding"]},
    ]})) as get:
        result = fetch("deepseek").json()
    assert get.call_args.args[0] == "https://api.deepseek.com/models"
    assert result["source"] == "live"
    assert [m["id"] for m in result["models"]] == ["deepseek-flash"]


@pytest.mark.parametrize("provider,key", [("ollama", "models"), ("custom", "data"), ("openrouter", "data")])
def test_successful_empty_list_is_not_replaced_with_suggestions(provider, key):
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=response({key: []})):
        result = fetch(provider).json()
    assert result["models"] == []
    assert result["source"] == "live"
    assert result["error"] is None


@pytest.mark.parametrize("failure,code", [
    (httpx.ReadTimeout("secret-key https://user:secret@bad.invalid"), "timeout"),
    (httpx.ConnectError("secret-key"), "unavailable"),
    (response({}, 401), "authentication"),
    (response({}, 403), "authentication"),
    (response({}, 500), "unavailable"),
    (response({"data": None}), "invalid_response"),
])
def test_errors_are_safe_and_fallback_is_explicit(failure, code, caplog):
    set_setting("openai_api_key", "secret-key")
    kwargs = {"side_effect": failure} if isinstance(failure, Exception) else {"return_value": failure}
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, **kwargs):
        result = fetch("openai").json()
    assert result["source"] == "fallback"
    assert result["error"] == code
    assert result["listing_verified"] is False
    assert "secret-key" not in str(result) + caplog.text


def test_cache_refresh_and_failure_preserve_original_timestamp():
    good = response({"data": [{"id": "openai/gpt-4o-mini"}]})
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=good) as get:
        first = fetch("openrouter").json()
        cached = fetch("openrouter").json()
        refreshed = fetch("openrouter", refresh=True).json()
    assert get.call_count == 2
    assert first["source"] == refreshed["source"] == "live"
    assert cached["source"] == "cache"
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=httpx.ReadTimeout("secret")):
        stale = fetch("openrouter", refresh=True).json()
    assert stale["source"] == "cache" and stale["stale"] is True
    assert stale["fetched_at"] == refreshed["fetched_at"]
    assert stale["models"] == first["models"]


def test_credentials_and_configured_endpoint_scope_cache():
    set_setting("custom_openai_url", "http://first:8000/v1")
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=[response({"data": [{"id": str(n)}]}) for n in range(3)]) as get:
        assert fetch("custom").json()["models"][0]["id"] == "0"
        set_setting("custom_openai_url", "http://second:8000/v1")
        assert fetch("custom").json()["models"][0]["id"] == "1"
        set_setting("custom_openai_api_key", "new-secret")
        assert fetch("custom").json()["models"][0]["id"] == "2"
    assert get.call_count == 3
    assert "new-secret" not in str(_MODELS_CACHE)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://user:secret@localhost", "http://localhost?key=secret", "http://169.254.169.254/latest", "http://localhost/#secret", "http://localhost/\\evil"])
def test_unsafe_endpoint_never_requested(url):
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as get:
        result = fetch("custom", url=url)
    assert result.status_code == 400
    get.assert_not_called()
    assert "secret" not in result.text


def test_local_failure_never_presents_suggestions_as_installed():
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=httpx.ConnectError("offline")):
        result = fetch("ollama").json()
    assert result["models"] == []
    assert result["source"] == "fallback"
    assert result["listing_verified"] is False


def test_repeated_pagination_cursor_fails_closed():
    set_setting("gemini_api_key", "secret")
    payload = {"models": [{"name": "models/gemini-test", "supportedGenerationMethods": ["generateContent"]}], "nextPageToken": "repeated"}
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=response(payload)) as get:
        result = fetch("gemini").json()
    assert get.call_count == 2
    assert result["source"] == "fallback"
    assert result["error"] == "invalid_response"


def test_discovery_never_replaces_saved_model():
    set_setting("openai_model", "ft:my-exact/Legacy-Model")
    set_setting("openai_api_key", "secret")
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=response({"data": [{"id": "gpt-4o-mini"}]})):
        result = fetch("openai").json()
    assert result["source"] == "live"
    assert get_setting("openai_model") == "ft:my-exact/Legacy-Model"


def test_gemini_requires_generation_method_and_filters_image_live():
    raw = [{"name": "models/" + mid, "supportedGenerationMethods": methods} for mid, methods in [
        ("gemini-text", ["generateContent"]), ("gemini-no-method", []),
        ("gemini-nano-banana", ["generateContent"]), ("gemini-flash-image", ["generateContent"]),
        ("gemini-live", ["bidiGenerateContent"]),
    ]]
    assert [m["id"] for m in filter_gemini_models(raw)] == ["gemini-text"]


def test_openai_filter_preserves_snapshot_without_alias_and_excludes_responses_only():
    raw = [{"id": mid} for mid in ["gpt-4o-2024-08-06", "gpt-5-pro", "gpt-5-codex", "gpt-image-1", "not-gpt-random"]]
    assert [m["id"] for m in filter_openai_models(raw)] == ["gpt-4o-2024-08-06"]


def test_anthropic_lifecycle_retired_never_recommended():
    raw = [{"id": "claude-current", "lifecycle": "active"}, {"id": "claude-old", "lifecycle": "retired"}, {"id": "claude-winding-down", "lifecycle": "deprecated"}]
    assert [m["id"] for m in filter_anthropic_models(raw)] == ["claude-current"]


def test_custom_ids_are_not_restricted_by_catalog_names():
    from app.core.ai_providers import filter_custom_models, filter_ollama_models
    assert filter_custom_models([{"id": "my-embedding-translator"}])[0]["id"] == "my-embedding-translator"
    assert filter_ollama_models([{"name": "my-embed-translator:latest"}])[0]["id"] == "my-embed-translator:latest"


@pytest.mark.parametrize("payload", [{"data": [{"id": 42}]}, {"data": [None]}, {"data": "not-a-list"}])
def test_malformed_records_fail_closed(payload):
    set_setting("openai_api_key", "secret")
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=response(payload)):
        result = fetch("openai").json()
    assert result["source"] == "fallback"
    assert result["error"] == "invalid_response"


@pytest.mark.asyncio
async def test_parallel_listing_singleflight():
    from app.services.model_discovery import discover_models
    gate = asyncio.Event()
    async def get(*args, **kwargs):
        await gate.wait()
        return response({"data": [{"id": "openai/gpt-4o-mini"}]})
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=get) as mock_get:
        tasks = [asyncio.create_task(discover_models("openrouter", refresh=True)) for _ in range(5)]
        # Yield to all requests, without real waiting or a network connection.
        for _ in range(10):
            await asyncio.sleep(0)
        gate.set()
        results = await asyncio.gather(*tasks)
    assert mock_get.call_count == 1
    assert all(result["models"][0]["id"] == "openai/gpt-4o-mini" for result in results)


def test_negative_cache_expires_without_fake_live_status(monkeypatch):
    from app.services import model_discovery as discovery
    clock = [1000.0]
    monkeypatch.setattr(discovery.time, "time", lambda: clock[0])
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=httpx.ConnectError("offline")) as get:
        assert fetch("openrouter").json()["source"] == "fallback"
        assert fetch("openrouter").json()["source"] == "fallback"
        assert get.call_count == 1
        clock[0] += 16
        assert fetch("openrouter").json()["source"] == "fallback"
        assert get.call_count == 2


def test_local_cache_ttl_and_empty_list_refresh(monkeypatch):
    from app.services import model_discovery as discovery
    clock = [1000.0]
    monkeypatch.setattr(discovery.time, "time", lambda: clock[0])
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=[response({"models": [{"name": "local:Exact"}]}), response({"models": []})]) as get:
        assert fetch("ollama").json()["source"] == "live"
        clock[0] += 59
        assert fetch("ollama").json()["source"] == "cache"
        clock[0] += 2
        assert fetch("ollama").json()["models"] == []
        assert get.call_count == 2


def test_redirect_never_forwards_key_to_other_endpoint():
    set_setting("gemini_api_key", "secret")
    requests = []
    def transport(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": "https://other.invalid/steal"})
    original = httpx.AsyncClient
    def client(**kwargs):
        return original(**kwargs, transport=httpx.MockTransport(transport))
    with patch("app.services.model_discovery.httpx.AsyncClient", side_effect=client):
        result = fetch("gemini").json()
    assert len(requests) == 1
    assert requests[0].headers["x-goog-api-key"] == "secret"
    assert "secret" not in str(requests[0].url)
    assert result["source"] == "fallback"


@pytest.mark.parametrize("provider", ["openai", "openrouter"])
@pytest.mark.parametrize("model,supports_none", [("gpt-6.1-sol", False), ("gpt-6.1-terra", True), ("gpt-6-astra", False), ("gpt-6-luna", True)])
def test_current_openai_capabilities(provider, model, supports_none):
    from app.core.ai_providers import get_model_capabilities
    mid = "openai/" + model if provider == "openrouter" else model
    caps = get_model_capabilities(provider, mid)
    assert caps.structured_output and caps.json_object
    assert caps.temperature is False
    assert caps.temperature_requires_no_reasoning is supports_none
    assert caps.max_output_tokens == 8192


def test_deepseek_current_alias_preserves_non_thinking_contract():
    from app.core.ai_providers import get_model_capabilities, context_from_settings
    caps = get_model_capabilities("deepseek", "deepseek-flash")
    assert caps.json_object and caps.thinking_control and caps.temperature
    set_setting("deepseek_model", "deepseek-v4-flash")
    assert context_from_settings("deepseek").model == "deepseek-v4-flash"


def test_modern_anthropic_native_format_contract():
    from app.core.ai_providers import get_model_capabilities
    for model in ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-sonnet-4-6", "claude-opus-4-8"):
        assert get_model_capabilities("anthropic", model).native_output_config


def test_database_initialization_never_migrates_saved_models():
    from app.core.db import init_db
    for key, value in [("deepseek_model", "deepseek-v4-flash"), ("openrouter_model", "anthropic/claude-sonnet-5"), ("openai_model", "ft:Saved/Case-Sensitive")]:
        set_setting(key, value)
    init_db()
    assert get_setting("deepseek_model") == "deepseek-v4-flash"
    assert get_setting("openrouter_model") == "anthropic/claude-sonnet-5"
    assert get_setting("openai_model") == "ft:Saved/Case-Sensitive"


def test_gemini_deprecated_alias_is_labelled_without_rewriting_id():
    result = filter_gemini_models([{"name": "models/gemini-3.7-flash", "supportedGenerationMethods": ["generateContent"]}])
    assert result[0]["id"] == "gemini-3.7-flash"
    assert "deprecated" in result[0]["name"]


@pytest.mark.parametrize("provider", ["openai", "openrouter", "deepseek", "custom"])
def test_connection_probe_does_not_echo_sdk_secrets(provider):
    from unittest.mock import MagicMock
    sdk = MagicMock()
    sdk.chat.completions.create.side_effect = RuntimeError("secret-key https://secret@provider.invalid?api_key=secret-key")
    with patch("openai.OpenAI", return_value=sdk):
        result = TestClient(app).post('/api/settings/test-ai', json={"provider": provider, "api_key": "secret-key", "model": "Exact/Model", "url": "http://localhost:8000/v1"})
    assert result.status_code == 400
    assert "secret-key" not in result.text and "provider.invalid" not in result.text


def test_connection_probe_uses_reasoning_compatible_token_limit():
    from unittest.mock import MagicMock
    sdk = MagicMock()
    with patch("openai.OpenAI", return_value=sdk):
        result = TestClient(app).post('/api/settings/test-ai', json={"provider": "openai", "api_key": "fixture", "model": "gpt-6.1-sol"})
    assert result.status_code == 200
    kwargs = sdk.chat.completions.create.call_args.kwargs
    assert kwargs['model'] == 'gpt-6.1-sol'
    assert kwargs['max_completion_tokens'] == 32
    assert 'max_tokens' not in kwargs


def test_connection_probe_anthropic_error_body_is_not_public():
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=response({"error": {"message": "secret-key"}}, 401)):
        result = TestClient(app).post('/api/settings/test-ai', json={"provider": "anthropic", "api_key": "secret-key"})
    assert result.status_code == 400
    assert "secret-key" not in result.text


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,model,reasoning,temperature", [
    ("openai", "gpt-6.1-sol", None, False),
    ("openai", "gpt-6.1-terra", "none", True),
    ("openrouter", "openai/gpt-6-luna", "none", True),
    ("deepseek", "deepseek-flash", None, True),
])
async def test_current_model_real_dispatch_contract(provider, model, reasoning, temperature):
    from unittest.mock import MagicMock
    from app.services.translator import SubtitleTranslator
    sdk = MagicMock()
    sdk.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content='{"result":"valid"}'))], usage=None)
    with patch("openai.OpenAI", return_value=sdk):
        result = await SubtitleTranslator()._dispatch_llm_completion(provider=provider, model_name=model, system_prompt='sys', user_prompt='Return JSON', schema={'type':'object'}, reasoning_effort=reasoning)
    assert result == '{"result":"valid"}'
    kwargs = sdk.chat.completions.create.call_args.kwargs
    assert kwargs['model'] == model
    assert ('temperature' in kwargs) is temperature
    if provider == 'deepseek':
        assert kwargs['extra_body'] == {'thinking': {'type': 'disabled'}}
        assert kwargs['response_format']['type'] == 'json_object'
    else:
        assert kwargs['response_format']['json_schema']['strict'] is True


def test_refresh_bypasses_cache_even_with_frozen_clock(monkeypatch):
    from app.services import model_discovery as discovery
    monkeypatch.setattr(discovery.time, 'time', lambda: 1000.0)
    with patch('httpx.AsyncClient.get', new_callable=AsyncMock, return_value=response({'data': []})) as get:
        assert fetch('openrouter').json()['source'] == 'live'
        assert fetch('openrouter', refresh=True).json()['source'] == 'live'
    assert get.call_count == 2


def test_fresh_database_uses_central_defaults(monkeypatch, tmp_path):
    from app.core import db
    from app.core.ai_providers import get_default_model
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path / 'new.db'))
    db.init_db()
    assert db.get_setting('deepseek_model') == get_default_model('deepseek') == 'deepseek-flash'
    assert db.get_setting('openrouter_model') == get_default_model('openrouter')


def test_pagination_budget_never_exposes_incomplete_list():
    set_setting('gemini_api_key', 'secret')
    pages = [response({'models': [{'name': f'models/page-{n}', 'supportedGenerationMethods': ['generateContent']}], 'nextPageToken': str(n)}) for n in range(20)]
    with patch('httpx.AsyncClient.get', new_callable=AsyncMock, side_effect=pages) as get:
        result = fetch('gemini').json()
    assert get.call_count == 20
    assert result['source'] == 'fallback'
    assert result['error'] == 'invalid_response'
    assert not any(m['id'].startswith('page-') for m in result['models'])


def test_openrouter_explicit_endpoint_and_modality_contracts():
    from app.core.ai_providers import filter_openrouter_models
    models = [
        {'id': 'provider/text', 'architecture': {'input_modalities': ['text'], 'output_modalities': ['text']}, 'supported_endpoints': ['chat/completions']},
        {'id': 'provider/batch', 'supported_endpoints': ['batch']},
        {'id': 'provider/response-only', 'supported_endpoints': ['responses']},
        {'id': 'provider/speech', 'architecture': {'input_modalities': ['audio'], 'output_modalities': ['text']}},
    ]
    assert [m['id'] for m in filter_openrouter_models(models)] == ['provider/text']


def test_legacy_openai_probe_retains_legacy_token_parameter():
    from unittest.mock import MagicMock
    sdk = MagicMock()
    with patch('openai.OpenAI', return_value=sdk):
        result = TestClient(app).post('/api/settings/test-ai', json={'provider': 'openai', 'api_key': 'fixture', 'model': 'gpt-3.5-turbo'})
    assert result.status_code == 200
    kwargs = sdk.chat.completions.create.call_args.kwargs
    assert kwargs['model'] == 'gpt-3.5-turbo'
    assert kwargs['max_tokens'] == 10
    assert 'max_completion_tokens' not in kwargs


def test_slow_failure_still_gets_full_negative_cache_ttl(monkeypatch):
    from app.services import model_discovery as discovery
    clock = [1000.0]
    monkeypatch.setattr(discovery.time, 'time', lambda: clock[0])
    async def slow_failure(*args, **kwargs):
        clock[0] += 20
        raise httpx.ReadTimeout('fixture timeout')
    with patch('httpx.AsyncClient.get', new_callable=AsyncMock, side_effect=slow_failure) as get:
        assert fetch('openrouter').json()['error'] == 'timeout'
        assert fetch('openrouter').json()['source'] == 'fallback'
        assert get.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('model,expected_format', [
    ('gpt-3.5-turbo', 'json_object'),
    ('gpt-4-turbo', 'json_object'),
    ('gpt-4o-2024-05-13', 'json_object'),
    ('gpt-4', None),
    ('gpt-4o-2024-08-06', 'json_schema'),
])
async def test_legacy_openai_dispatch_uses_only_supported_json_format(model, expected_format):
    from unittest.mock import MagicMock
    from app.services.translator import SubtitleTranslator
    sdk = MagicMock()
    sdk.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content='{"result":"valid"}'))], usage=None)
    with patch('openai.OpenAI', return_value=sdk):
        result = await SubtitleTranslator()._dispatch_llm_completion(provider='openai', model_name=model, system_prompt='Return JSON', user_prompt='JSON result', schema={'type': 'object'})
    assert result == '{"result":"valid"}'
    kwargs = sdk.chat.completions.create.call_args.kwargs
    assert kwargs.get('response_format', {}).get('type') == expected_format
    assert kwargs['model'] == model


def test_custom_discovery_preserves_existing_dummy_auth_without_saved_key():
    with patch('httpx.AsyncClient.get', new_callable=AsyncMock, return_value=response({'data': [{'id': 'Local/Exact'}]})) as get:
        result = fetch('custom').json()
    assert result['source'] == 'live'
    assert get.call_args.kwargs['headers']['Authorization'] == 'Bearer dummy'
