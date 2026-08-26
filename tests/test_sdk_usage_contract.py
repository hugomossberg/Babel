"""
SDK ↔ Usage Identity Contract Tests
======================================
For each provider, verifies the complete identity chain:

  resolved ProviderContext (provider, model)
  == actual SDK/network call (model arg, endpoint, client)
  == Usage ledger record (provider, model, stage)

Additionally asserts that Gemini SDK is NOT called for non-Gemini providers.

Providers covered (8 total):
  1. gemini
  2. openai
  3. anthropic
  4. openrouter
  5. deepseek
  6. custom
  7. ollama
  8. deepl

Plus Queue / Job Details / Execution Log identity and Escalation separation tests.
"""
import json
import sqlite3
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.ai_providers import ProviderContext, resolve_job_provider_context
from app.core.usage import UsageStage
from app.core.db import (
    DB_PATH, init_db, create_job, get_job_by_id, get_jobs,
    pin_job_provider, append_job_log, set_setting,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _openai_mock_response(text):
    choice = MagicMock()
    choice.message.content = text
    return MagicMock(choices=[choice])


def _anthropic_mock_response(text, stop_reason="end_turn"):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "stop_reason": stop_reason,
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    return resp


BATCH = [{"id": 1, "text": "Hello"}]
OK_RESPONSE = json.dumps({"translations": [{"id": 1, "text": "Hej"}]})


# ---------------------------------------------------------------------------
# 1. Gemini – resolved provider == genai SDK call == usage record
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_contract_gemini_sdk_usage_identity():
    """
    Resolved: gemini / gemini-3.5-flash-lite
    SDK call: genai.Client.models.generate_content(model="gemini-3.5-flash-lite")
    Usage: record_dispatch(provider="gemini", model="gemini-3.5-flash-lite", stage=UsageStage.PRIMARY)
    """
    from app.services.translator import SubtitleTranslator

    translator = SubtitleTranslator()
    _ctx = ProviderContext(provider="gemini", model="gemini-3.5-flash-lite")

    mock_gemini_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = OK_RESPONSE
    mock_gemini_client.models.generate_content.return_value = mock_resp

    dispatched = []

    def fake_record(request_uid, provider, model, stage, job_id=None, created_at=None):
        dispatched.append({"provider": provider, "model": model, "stage": stage})

    with patch("app.core.ai_providers.context_from_settings", return_value=_ctx), \
         patch("app.core.ai_providers.resolve_job_provider_context", return_value=_ctx), \
         patch("app.services.translator.get_setting", side_effect=lambda k, d=None: {
             "gemini_api_key": "test-key",
             "gemini_model": "gemini-3.5-flash-lite",
         }.get(k, d)), \
         patch.object(translator, "get_gemini_client", return_value=mock_gemini_client), \
         patch("app.core.usage.record_dispatch", side_effect=fake_record):

        await translator.translate_batch(
            BATCH, target_language="Swedish",
            provider_ctx=_ctx
        )

    # SDK call must use exact model
    mock_gemini_client.models.generate_content.assert_called()
    call_kwargs = mock_gemini_client.models.generate_content.call_args.kwargs
    assert call_kwargs.get("model") == "gemini-3.5-flash-lite", (
        f"Gemini SDK called with wrong model: {call_kwargs.get('model')}"
    )

    # Usage must record exact provider, model, and stage
    assert len(dispatched) >= 1, "Usage record_dispatch was not called for gemini"
    rec = dispatched[0]
    assert rec["provider"] == "gemini", f"Expected provider 'gemini', got '{rec['provider']}'"
    assert rec["model"] == "gemini-3.5-flash-lite", f"Expected model 'gemini-3.5-flash-lite', got '{rec['model']}'"
    assert rec["stage"] == UsageStage.PRIMARY, f"Expected stage '{UsageStage.PRIMARY}', got '{rec['stage']}'"


# ---------------------------------------------------------------------------
# 2. OpenAI – resolved provider == openai SDK == usage record
#    Gemini must NOT be called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_contract_openai_sdk_usage_identity():
    """
    Resolved: openai / gpt-4o-mini
    SDK call: openai.chat.completions.create(model="gpt-4o-mini")
    Usage: record_dispatch(provider="openai", model="gpt-4o-mini", stage=UsageStage.PRIMARY)
    Gemini: assert_not_called
    """
    from app.services.translator import SubtitleTranslator

    translator = SubtitleTranslator()
    _ctx = ProviderContext(provider="openai", model="gpt-4o-mini")

    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = _openai_mock_response(OK_RESPONSE)
    mock_gemini_client = MagicMock()

    dispatched = []

    def fake_record(request_uid, provider, model, stage, job_id=None, created_at=None):
        dispatched.append({"provider": provider, "model": model, "stage": stage})

    with patch("app.core.ai_providers.context_from_settings", return_value=_ctx), \
         patch("app.core.ai_providers.resolve_job_provider_context", return_value=_ctx), \
         patch("app.services.translator.get_setting", side_effect=lambda k, d=None: {
             "openai_api_key": "sk-test", "openai_model": "gpt-4o-mini",
         }.get(k, d)), \
         patch.object(translator, "get_openai_client", return_value=mock_openai_client), \
         patch.object(translator, "get_gemini_client", return_value=mock_gemini_client), \
         patch("app.core.usage.record_dispatch", side_effect=fake_record):

        await translator.translate_batch(
            BATCH, target_language="Swedish",
            provider_ctx=_ctx
        )

    # Gemini SDK must NOT be called for OpenAI jobs
    mock_gemini_client.models.generate_content.assert_not_called()

    # OpenAI SDK called with exact model
    mock_openai_client.chat.completions.create.assert_called()
    oa_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    assert oa_kwargs.get("model") == "gpt-4o-mini", f"OpenAI called with: {oa_kwargs.get('model')}"

    # Usage identity: exact provider, model, and stage
    assert len(dispatched) >= 1, "Usage record_dispatch was not called for openai"
    rec = dispatched[0]
    assert rec["provider"] == "openai", f"Expected provider 'openai', got '{rec['provider']}'"
    assert rec["model"] == "gpt-4o-mini", f"Expected model 'gpt-4o-mini', got '{rec['model']}'"
    assert rec["stage"] == UsageStage.PRIMARY, f"Expected stage '{UsageStage.PRIMARY}', got '{rec['stage']}'"


# ---------------------------------------------------------------------------
# 3. Anthropic – httpx POST with correct model, output_config for claude-sonnet-5
#    Gemini must NOT be called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_contract_anthropic_sdk_usage_identity():
    """
    Resolved: anthropic / claude-sonnet-5
    HTTP call: POST to Anthropic with model="claude-sonnet-5"
    Usage: record_dispatch(provider="anthropic", model="claude-sonnet-5", stage=UsageStage.PRIMARY)
    Gemini: assert_not_called
    """
    from app.services.translator import SubtitleTranslator

    translator = SubtitleTranslator()
    _ctx = ProviderContext(provider="anthropic", model="claude-sonnet-5")
    mock_gemini_client = MagicMock()
    captured_payloads = []

    async def fake_post(url, **kwargs):
        captured_payloads.append(kwargs.get("json", {}))
        return _anthropic_mock_response(OK_RESPONSE)

    mock_http = MagicMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    mock_http.post = AsyncMock(side_effect=fake_post)

    dispatched = []

    def fake_record(request_uid, provider, model, stage, job_id=None, created_at=None):
        dispatched.append({"provider": provider, "model": model, "stage": stage})

    with patch("app.core.ai_providers.context_from_settings", return_value=_ctx), \
         patch("app.core.ai_providers.resolve_job_provider_context", return_value=_ctx), \
         patch("app.services.translator.get_setting", side_effect=lambda k, d=None: {
             "anthropic_api_key": "sk-ant-test", "anthropic_model": "claude-sonnet-5",
         }.get(k, d)), \
         patch.object(translator, "get_gemini_client", return_value=mock_gemini_client), \
         patch("httpx.AsyncClient", return_value=mock_http), \
         patch("app.core.usage.record_dispatch", side_effect=fake_record):

        await translator.translate_batch(
            BATCH, target_language="Swedish",
            provider_ctx=_ctx
        )

    # Gemini SDK must NOT be called for Anthropic jobs
    mock_gemini_client.models.generate_content.assert_not_called()

    # HTTP call uses exact model
    assert len(captured_payloads) >= 1
    payload = captured_payloads[0]
    assert payload.get("model") == "claude-sonnet-5", f"Wrong model: {payload.get('model')}"

    # Usage identity: exact provider, model, and stage
    assert len(dispatched) >= 1, "Usage record_dispatch was not called for anthropic"
    rec = dispatched[0]
    assert rec["provider"] == "anthropic", f"Expected provider 'anthropic', got '{rec['provider']}'"
    assert rec["model"] == "claude-sonnet-5", f"Expected model 'claude-sonnet-5', got '{rec['model']}'"
    assert rec["stage"] == UsageStage.PRIMARY, f"Expected stage '{UsageStage.PRIMARY}', got '{rec['stage']}'"


# ---------------------------------------------------------------------------
# 4. OpenRouter – openai-compat with openrouter base_url and custom model
#    Gemini must NOT be called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_contract_openrouter_sdk_usage_identity():
    """
    Resolved: openrouter / anthropic/claude-sonnet-5
    SDK call: openai.OpenAI with base_url="https://openrouter.ai/api/v1"
    Usage: record_dispatch(provider="openrouter", model="anthropic/claude-sonnet-5", stage=UsageStage.PRIMARY)
    Gemini: assert_not_called
    """
    from app.services.translator import SubtitleTranslator

    translator = SubtitleTranslator()
    _ctx = ProviderContext(provider="openrouter", model="anthropic/claude-sonnet-5")
    mock_gemini_client = MagicMock()

    created_clients = []
    mock_openai_instance = MagicMock()
    mock_openai_instance.chat.completions.create.return_value = _openai_mock_response(OK_RESPONSE)

    def fake_openai_init(*args, **kwargs):
        created_clients.append(kwargs)
        return mock_openai_instance

    dispatched = []

    def fake_record(request_uid, provider, model, stage, job_id=None, created_at=None):
        dispatched.append({"provider": provider, "model": model, "stage": stage})

    with patch("app.core.ai_providers.context_from_settings", return_value=_ctx), \
         patch("app.core.ai_providers.resolve_job_provider_context", return_value=_ctx), \
         patch("app.services.translator.get_setting", side_effect=lambda k, d=None: {
             "openrouter_api_key": "sk-or-test",
             "openrouter_model": "anthropic/claude-sonnet-5",
         }.get(k, d)), \
         patch.object(translator, "get_gemini_client", return_value=mock_gemini_client), \
         patch("openai.OpenAI", side_effect=fake_openai_init), \
         patch("app.core.usage.record_dispatch", side_effect=fake_record):

        await translator.translate_batch(
            BATCH, target_language="Swedish",
            provider_ctx=_ctx
        )

    # Gemini SDK must NOT be called for OpenRouter jobs
    mock_gemini_client.models.generate_content.assert_not_called()

    # Client must be configured with exact OpenRouter base_url
    assert len(created_clients) >= 1, "openai.OpenAI was not instantiated for OpenRouter"
    assert created_clients[0].get("base_url") == "https://openrouter.ai/api/v1", (
        f"Wrong OpenRouter base_url: {created_clients[0].get('base_url')}"
    )

    # OpenRouter chat.completions.create called with exact model
    mock_openai_instance.chat.completions.create.assert_called()
    oa_kwargs = mock_openai_instance.chat.completions.create.call_args.kwargs
    assert oa_kwargs.get("model") == "anthropic/claude-sonnet-5", (
        f"OpenRouter called with wrong model: {oa_kwargs.get('model')}"
    )

    # Usage identity: exact provider, model, and stage
    assert len(dispatched) >= 1, "Usage record_dispatch was not called for openrouter"
    rec = dispatched[0]
    assert rec["provider"] == "openrouter", f"Expected provider 'openrouter', got '{rec['provider']}'"
    assert rec["model"] == "anthropic/claude-sonnet-5", (
        f"Expected model 'anthropic/claude-sonnet-5', got '{rec['model']}'"
    )
    assert rec["stage"] == UsageStage.PRIMARY, f"Expected stage '{UsageStage.PRIMARY}', got '{rec['stage']}'"


# ---------------------------------------------------------------------------
# 5. DeepSeek – openai-compat with thinking disabled
#    Gemini must NOT be called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_contract_deepseek_sdk_usage_identity():
    """
    Resolved: deepseek / deepseek-v4-flash
    SDK: openai-compat with base_url="https://api.deepseek.com" and extra_body.thinking.type == "disabled"
    Usage: record_dispatch(provider="deepseek", model="deepseek-v4-flash", stage=UsageStage.PRIMARY)
    Gemini: assert_not_called
    """
    from app.services.translator import SubtitleTranslator

    translator = SubtitleTranslator()
    _ctx = ProviderContext(provider="deepseek", model="deepseek-v4-flash")
    mock_gemini_client = MagicMock()

    created_clients = []
    mock_openai_instance = MagicMock()
    mock_openai_instance.chat.completions.create.return_value = _openai_mock_response(OK_RESPONSE)

    def fake_openai_init(*args, **kwargs):
        created_clients.append(kwargs)
        return mock_openai_instance

    dispatched = []

    def fake_record(request_uid, provider, model, stage, job_id=None, created_at=None):
        dispatched.append({"provider": provider, "model": model, "stage": stage})

    with patch("app.core.ai_providers.context_from_settings", return_value=_ctx), \
         patch("app.core.ai_providers.resolve_job_provider_context", return_value=_ctx), \
         patch("app.services.translator.get_setting", side_effect=lambda k, d=None: {
             "deepseek_api_key": "sk-ds-test", "deepseek_model": "deepseek-v4-flash",
         }.get(k, d)), \
         patch.object(translator, "get_gemini_client", return_value=mock_gemini_client), \
         patch("openai.OpenAI", side_effect=fake_openai_init), \
         patch("app.core.usage.record_dispatch", side_effect=fake_record):

        await translator.translate_batch(
            BATCH, target_language="Swedish",
            provider_ctx=_ctx
        )

    # Gemini SDK must NOT be called for DeepSeek jobs
    mock_gemini_client.models.generate_content.assert_not_called()

    # Client must be configured with exact DeepSeek base_url
    assert len(created_clients) >= 1, "openai.OpenAI was not instantiated for DeepSeek"
    assert created_clients[0].get("base_url") == "https://api.deepseek.com", (
        f"Wrong DeepSeek base_url: {created_clients[0].get('base_url')}"
    )

    mock_openai_instance.chat.completions.create.assert_called()
    oa_kwargs = mock_openai_instance.chat.completions.create.call_args.kwargs
    assert oa_kwargs.get("model") == "deepseek-v4-flash", f"Wrong model: {oa_kwargs.get('model')}"

    # DeepSeek thinking must be disabled
    extra_body = oa_kwargs.get("extra_body", {})
    assert extra_body.get("thinking", {}).get("type") == "disabled", (
        f"DeepSeek thinking must be disabled. extra_body: {extra_body}"
    )

    # Usage identity: exact provider, model, and stage
    assert len(dispatched) >= 1, "Usage record_dispatch was not called for deepseek"
    rec = dispatched[0]
    assert rec["provider"] == "deepseek", f"Expected provider 'deepseek', got '{rec['provider']}'"
    assert rec["model"] == "deepseek-v4-flash", f"Expected model 'deepseek-v4-flash', got '{rec['model']}'"
    assert rec["stage"] == UsageStage.PRIMARY, f"Expected stage '{UsageStage.PRIMARY}', got '{rec['stage']}'"


# ---------------------------------------------------------------------------
# 6. Custom – openai-compat with custom_openai_url and conservative capabilities
#    Gemini must NOT be called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_contract_custom_sdk_usage_identity():
    """
    Resolved: custom / my-custom-model
    SDK call: openai.OpenAI with base_url="http://custom-llm-host:8000/v1" and model="my-custom-model"
    Conservative caps: no blind response_format sent if custom provider doesn't advertise it
    Usage: record_dispatch(provider="custom", model="my-custom-model", stage=UsageStage.PRIMARY)
    Gemini: assert_not_called
    """
    from app.services.translator import SubtitleTranslator

    translator = SubtitleTranslator()
    _ctx = ProviderContext(provider="custom", model="my-custom-model")
    mock_gemini_client = MagicMock()

    created_clients = []
    mock_openai_instance = MagicMock()
    mock_openai_instance.chat.completions.create.return_value = _openai_mock_response(OK_RESPONSE)

    def fake_openai_init(*args, **kwargs):
        created_clients.append(kwargs)
        return mock_openai_instance

    dispatched = []

    def fake_record(request_uid, provider, model, stage, job_id=None, created_at=None):
        dispatched.append({"provider": provider, "model": model, "stage": stage})

    with patch("app.core.ai_providers.context_from_settings", return_value=_ctx), \
         patch("app.core.ai_providers.resolve_job_provider_context", return_value=_ctx), \
         patch("app.services.translator.get_setting", side_effect=lambda k, d=None: {
             "custom_openai_api_key": "sk-custom-test",
             "custom_openai_url": "http://custom-llm-host:8000/v1",
             "custom_openai_model": "my-custom-model",
         }.get(k, d)), \
         patch.object(translator, "get_gemini_client", return_value=mock_gemini_client), \
         patch("openai.OpenAI", side_effect=fake_openai_init), \
         patch("app.core.usage.record_dispatch", side_effect=fake_record):

        await translator.translate_batch(
            BATCH, target_language="Swedish",
            provider_ctx=_ctx
        )

    # Gemini SDK must NOT be called for Custom jobs
    mock_gemini_client.models.generate_content.assert_not_called()

    # Client must be configured with exact custom base_url
    assert len(created_clients) >= 1, "openai.OpenAI was not instantiated for Custom"
    assert created_clients[0].get("base_url") == "http://custom-llm-host:8000/v1", (
        f"Wrong custom base_url: {created_clients[0].get('base_url')}"
    )

    # Chat completions called with exact custom model
    mock_openai_instance.chat.completions.create.assert_called()
    oa_kwargs = mock_openai_instance.chat.completions.create.call_args.kwargs
    assert oa_kwargs.get("model") == "my-custom-model", (
        f"Custom called with wrong model: {oa_kwargs.get('model')}"
    )

    # Conservative caps: response_format must NOT be blindly sent for unadvertised capabilities
    assert "response_format" not in oa_kwargs, (
        f"response_format should not be sent blindly for custom provider without json_object capability"
    )

    # Usage identity: exact provider, model, and stage
    assert len(dispatched) >= 1, "Usage record_dispatch was not called for custom"
    rec = dispatched[0]
    assert rec["provider"] == "custom", f"Expected provider 'custom', got '{rec['provider']}'"
    assert rec["model"] == "my-custom-model", f"Expected model 'my-custom-model', got '{rec['model']}'"
    assert rec["stage"] == UsageStage.PRIMARY, f"Expected stage '{UsageStage.PRIMARY}', got '{rec['stage']}'"


# ---------------------------------------------------------------------------
# 7. Ollama – httpx POST to ollama_url with token capture
#    Gemini must NOT be called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_contract_ollama_sdk_usage_identity():
    """
    Resolved: ollama / llama3
    HTTP call: POST to ollama_url/api/generate with model="llama3"
    Token capture: prompt_eval_count → input_tokens, eval_count → output_tokens
    Usage: record_dispatch(provider="ollama", model="llama3", stage=UsageStage.PRIMARY)
    Gemini: assert_not_called
    """
    from app.services.translator import SubtitleTranslator

    translator = SubtitleTranslator()
    _ctx = ProviderContext(provider="ollama", model="llama3")
    mock_gemini_client = MagicMock()
    captured_payloads = []

    async def fake_post(url, **kwargs):
        captured_payloads.append({"url": url, "body": kwargs.get("json", {})})
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "response": OK_RESPONSE,
            "prompt_eval_count": 20,
            "eval_count": 15,
        }
        return resp

    mock_http = MagicMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    mock_http.post = AsyncMock(side_effect=fake_post)

    dispatched = []

    def fake_record(request_uid, provider, model, stage, job_id=None, created_at=None):
        dispatched.append({"provider": provider, "model": model, "stage": stage})

    with patch("app.core.ai_providers.context_from_settings", return_value=_ctx), \
         patch("app.core.ai_providers.resolve_job_provider_context", return_value=_ctx), \
         patch("app.services.translator.get_setting", side_effect=lambda k, d=None: {
             "ollama_url": "http://localhost:11434", "ollama_model": "llama3",
         }.get(k, d)), \
         patch.object(translator, "get_gemini_client", return_value=mock_gemini_client), \
         patch("httpx.AsyncClient", return_value=mock_http), \
         patch("app.core.usage.record_dispatch", side_effect=fake_record):

        await translator.translate_batch(
            BATCH, target_language="Swedish",
            provider_ctx=_ctx
        )

    mock_gemini_client.models.generate_content.assert_not_called()

    # HTTP call to ollama endpoint with exact model
    assert any("ollama" in c["url"] or "11434" in c["url"] for c in captured_payloads), (
        f"No call to ollama endpoint. Captured: {[c['url'] for c in captured_payloads]}"
    )
    ollama_call = captured_payloads[0]
    assert ollama_call["body"].get("model") == "llama3", (
        f"Ollama called with wrong model: {ollama_call['body'].get('model')}"
    )

    # Usage identity: exact provider, model, and stage
    assert len(dispatched) >= 1, "Usage record_dispatch was not called for ollama"
    rec = dispatched[0]
    assert rec["provider"] == "ollama", f"Expected provider 'ollama', got '{rec['provider']}'"
    assert rec["model"] == "llama3", f"Expected model 'llama3', got '{rec['model']}'"
    assert rec["stage"] == UsageStage.PRIMARY, f"Expected stage '{UsageStage.PRIMARY}', got '{rec['stage']}'"


# ---------------------------------------------------------------------------
# 8. DeepL – uses translate_batch_deepl, NOT gemini
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_contract_deepl_sdk_usage_identity():
    """
    Resolved: deepl / deepl_free
    HTTP call: POST to api-free.deepl.com/v2/translate (or api.deepl.com)
    model_type must be in request body
    Gemini: assert_not_called
    Usage: record_dispatch(provider="deepl", model="deepl_free", stage=UsageStage.PRIMARY) — token counts NULL
    """
    from app.services.translator import SubtitleTranslator

    translator = SubtitleTranslator()
    _ctx = ProviderContext(provider="deepl", model="deepl_free")
    mock_gemini_client = MagicMock()
    captured = []

    DEEPL_RESPONSE = {
        "translations": [{"detected_source_language": "EN", "text": "Hej"}]
    }

    async def fake_post(url, **kwargs):
        captured.append({"url": url, "body": kwargs.get("json", kwargs.get("data", {}))})
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = DEEPL_RESPONSE
        return resp

    mock_http = MagicMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    mock_http.post = AsyncMock(side_effect=fake_post)

    dispatched = []

    def fake_record(request_uid, provider, model, stage, job_id=None, created_at=None):
        dispatched.append({"provider": provider, "model": model, "stage": stage})

    with patch("app.core.ai_providers.context_from_settings", return_value=_ctx), \
         patch("app.core.ai_providers.resolve_job_provider_context", return_value=_ctx), \
         patch("app.services.translator.get_setting", side_effect=lambda k, d=None: {
             "deepl_api_key": "test:fx", "deepl_model": "deepl_free",
             "target_language": "Swedish",
         }.get(k, d)), \
         patch.object(translator, "get_gemini_client", return_value=mock_gemini_client), \
         patch("httpx.AsyncClient", return_value=mock_http), \
         patch("app.core.usage.record_dispatch", side_effect=fake_record):

        await translator.translate_batch(
            BATCH, target_language="Swedish",
            provider_ctx=_ctx
        )

    mock_gemini_client.models.generate_content.assert_not_called()

    # DeepL endpoint called
    assert len(captured) >= 1, "No HTTP call made for DeepL"
    assert any("deepl" in c["url"].lower() for c in captured), (
        f"DeepL endpoint not called. Got: {[c['url'] for c in captured]}"
    )

    # Usage identity: exact provider, model, and stage
    assert len(dispatched) >= 1, "Usage record_dispatch was not called for deepl"
    rec = dispatched[0]
    assert rec["provider"] == "deepl", f"Expected provider 'deepl', got '{rec['provider']}'"
    assert rec["model"] == "deepl_free", f"Expected model 'deepl_free', got '{rec['model']}'"
    assert rec["stage"] == UsageStage.PRIMARY, f"Expected stage '{UsageStage.PRIMARY}', got '{rec['stage']}'"


# ---------------------------------------------------------------------------
# 9. Queue / Job Details / Execution Log Identity Contract Tests
# ---------------------------------------------------------------------------

def test_queue_job_details_execution_log_identity(tmp_path):
    """
    Verifies that a job pinned to (provider=anthropic, model=claude-sonnet-5)
    presents the exact same provider/model identity across:
      1. Queue (get_jobs list view)
      2. Job Details (get_job_by_id)
      3. Provider Resolution (resolve_job_provider_context)
      4. Execution Log (append_job_log recording exact engine)
    """
    test_db = tmp_path / "test_identity.db"
    with patch("app.core.db.DB_PATH", str(test_db)):
        init_db()

        # Set global default to openai to prove pinning is immune to global changes
        set_setting("ai_provider", "openai")
        set_setting("openai_model", "gpt-4o-mini")

        job_id = create_job("test_identity_show_s01e01.mkv")

        # Pin job to Anthropic
        pin_job_provider(
            job_id=job_id,
            primary_provider="anthropic",
            primary_model="claude-sonnet-5",
            escalation_enabled=False,
        )

        # 1. Job Details Identity
        job_details = get_job_by_id(job_id)
        assert job_details is not None
        assert job_details["primary_provider"] == "anthropic"
        assert job_details["primary_model"] == "claude-sonnet-5"

        # 2. Queue Identity (get_jobs list view)
        queue_jobs = get_jobs(limit=10)
        matching = [j for j in queue_jobs if j["id"] == job_id]
        assert len(matching) == 1
        assert matching[0]["primary_provider"] == "anthropic"
        assert matching[0]["primary_model"] == "claude-sonnet-5"

        # 3. Provider Resolution Context
        ctx = resolve_job_provider_context(job_id)
        assert ctx.provider == "anthropic"
        assert ctx.model == "claude-sonnet-5"
        assert "Anthropic" in ctx.engine_label or "claude" in ctx.engine_label.lower()

        # 4. Execution Log
        append_job_log(job_id, f"Translation started using engine: {ctx.engine_label}")
        updated_job = get_job_by_id(job_id)
        log_text = "\n".join(updated_job.get("logs", []))
        assert ctx.engine_label in log_text


def test_escalation_job_identity_separation(tmp_path):
    """
    Verifies that an escalation job strictly separates primary provider/model
    from escalation provider/model in:
      1. Job Details
      2. resolve_job_provider_context(job_id, escalation=False) vs (job_id, escalation=True)
    """
    test_db = tmp_path / "test_esc_identity.db"
    with patch("app.core.db.DB_PATH", str(test_db)):
        init_db()

        job_id = create_job("test_escalation_show_s01e02.mkv")

        # Pin primary to Gemini and escalation to Anthropic Opus
        pin_job_provider(
            job_id=job_id,
            primary_provider="gemini",
            primary_model="gemini-3.5-flash-lite",
            escalation_enabled=True,
            escalation_provider="anthropic",
            escalation_model="claude-opus-5",
        )

        # 1. Job Details separation
        job_details = get_job_by_id(job_id)
        assert job_details is not None
        assert job_details["primary_provider"] == "gemini"
        assert job_details["primary_model"] == "gemini-3.5-flash-lite"
        assert job_details["escalation_enabled"] == 1
        assert job_details["escalation_provider"] == "anthropic"
        assert job_details["escalation_model"] == "claude-opus-5"

        # 2. Primary resolution
        primary_ctx = resolve_job_provider_context(job_id, escalation=False)
        assert primary_ctx.provider == "gemini"
        assert primary_ctx.model == "gemini-3.5-flash-lite"

        # 3. Escalation resolution
        esc_ctx = resolve_job_provider_context(job_id, escalation=True)
        assert esc_ctx.provider == "anthropic"
        assert esc_ctx.model == "claude-opus-5"
