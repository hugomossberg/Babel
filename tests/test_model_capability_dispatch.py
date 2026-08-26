"""
Comprehensive regression and capability-driven dispatch tests for Babel AI providers and models.
Verifies that all request building is driven by centralized capabilities, exact model family mapping,
and documented default reasoning effort rules.
"""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock

from app.core.ai_providers import (
    ModelCapabilities,
    get_model_capabilities,
    resolve_job_provider_context,
    context_from_settings,
    normalize_provider,
    ProviderContext,
)
from app.services.translator import SubtitleTranslator
from app.core.db import create_job, pin_job_provider, set_setting, get_setting


# ============================================================================
# 1. CENTRAL CAPABILITY RESOLUTION & PRECEDENCE TESTS
# ============================================================================

def test_gemini_capabilities():
    # Legacy models support temperature
    caps_legacy = get_model_capabilities("gemini", "gemini-2.5-flash")
    assert caps_legacy.temperature is True
    assert caps_legacy.structured_output is True
    assert caps_legacy.json_object is True

    # Gemini 3.x models do not support custom temperature in Google GenAI SDK
    caps_3 = get_model_capabilities("gemini", "gemini-3.5-flash-lite")
    assert caps_3.temperature is False
    assert caps_3.structured_output is True
    assert caps_3.json_object is True

    caps_37 = get_model_capabilities("gemini", "gemini-3.7-flash")
    assert caps_37.temperature is False

    # Unknown future Gemini model -> safe conservative fallback
    caps_unknown = get_model_capabilities("gemini", "gemini-future-9.0")
    assert caps_unknown.temperature is False
    assert caps_unknown.structured_output is True
    assert caps_unknown.json_object is True


def test_openai_capabilities():
    # 1. Classical GPT models support temperature unconditionally without reasoning
    for m in ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4-turbo", "gpt-3.5-turbo"]:
        caps = get_model_capabilities("openai", m)
        assert caps.temperature is True, f"{m} must have temperature=True"
        assert caps.thinking_control is False, f"{m} must have thinking_control=False"
        assert caps.temperature_requires_no_reasoning is False
        assert caps.default_reasoning_effort is None
        assert caps.structured_output is True
        assert caps.json_object is True

    # 2. GPT-5.6 Flagship and GPT-5.5: default reasoning = "medium", supports reasoning "none"
    for m in ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6", "gpt-5.5"]:
        caps = get_model_capabilities("openai", m)
        assert caps.thinking_control is True, f"{m} must have thinking_control=True"
        assert caps.temperature is False, f"{m} default temperature must be False"
        assert caps.temperature_requires_no_reasoning is True, f"{m} must have temperature_requires_no_reasoning=True"
        assert caps.default_reasoning_effort == "medium", f"{m} default reasoning must be 'medium'"
        assert caps.structured_output is True
        assert caps.json_object is True

    # 3. GPT-5.1, GPT-5.2, GPT-5.4: default reasoning = "none"
    for m in ["gpt-5.1", "gpt-5.2", "gpt-5.4"]:
        caps = get_model_capabilities("openai", m)
        assert caps.thinking_control is True, f"{m} must have thinking_control=True"
        assert caps.temperature is False, f"{m} default temperature field is False"
        assert caps.temperature_requires_no_reasoning is True, f"{m} must have temperature_requires_no_reasoning=True"
        assert caps.default_reasoning_effort == "none", f"{m} default reasoning must be 'none'"
        assert caps.structured_output is True

    # 4. GPT-5.3-chat-latest: deprecated Instant/Chat model, non-reasoning, supports temperature & Structured Outputs
    caps_chat = get_model_capabilities("openai", "gpt-5.3-chat-latest")
    assert caps_chat.thinking_control is False, "gpt-5.3-chat-latest must NOT have thinking_control=True"
    assert caps_chat.temperature is True, "gpt-5.3-chat-latest supports temperature"
    assert caps_chat.temperature_requires_no_reasoning is False
    assert caps_chat.default_reasoning_effort is None
    assert caps_chat.structured_output is True
    assert caps_chat.json_object is True

    # 5. GPT-5.3-codex: does NOT support reasoning "none", default_reasoning_effort is None
    caps_codex = get_model_capabilities("openai", "gpt-5.3-codex")
    assert caps_codex.thinking_control is True
    assert caps_codex.temperature is False
    assert caps_codex.temperature_requires_no_reasoning is False, "gpt-5.3-codex must NOT have temperature_requires_no_reasoning=True"
    assert caps_codex.default_reasoning_effort is None
    assert caps_codex.structured_output is True

    # 6. Early/base GPT-5 models: reasoning supported, but custom temperature is not supported
    for m in ["gpt-5", "gpt-5-mini", "gpt-5-nano"]:
        caps = get_model_capabilities("openai", m)
        assert caps.thinking_control is True, f"{m} must have thinking_control=True"
        assert caps.temperature is False, f"{m} must not allow temperature"
        assert caps.temperature_requires_no_reasoning is False, f"{m} must have temperature_requires_no_reasoning=False"
        assert caps.default_reasoning_effort == "medium"

    # 7. OpenAI reasoning models (o1, o3, o4, etc.) reject temperature
    for m in ["o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4", "o4-mini"]:
        caps = get_model_capabilities("openai", m)
        assert caps.temperature is False, f"{m} must not allow temperature"
        assert caps.thinking_control is True, f"{m} must support thinking_control"
        assert caps.temperature_requires_no_reasoning is False
        assert caps.default_reasoning_effort == "medium"

    # 8. Unknown/future OpenAI model (e.g. gpt-5.9-future) -> safe conservative fallback
    caps_unknown = get_model_capabilities("openai", "gpt-5.9-future")
    assert caps_unknown.temperature is False
    assert caps_unknown.thinking_control is False
    assert caps_unknown.temperature_requires_no_reasoning is False
    assert caps_unknown.default_reasoning_effort is None
    assert caps_unknown.structured_output is False
    assert caps_unknown.json_object is False


def test_anthropic_capabilities():
    # Claude 5.x and 4.5/4.1 models: no temperature, native output_config supported
    caps_5 = get_model_capabilities("anthropic", "claude-sonnet-5")
    assert caps_5.temperature is False
    assert caps_5.native_output_config is True
    assert caps_5.structured_output is True

    caps_45 = get_model_capabilities("anthropic", "claude-sonnet-4-5")
    assert caps_45.temperature is False
    assert caps_45.native_output_config is True

    # Claude 3.5 models: temperature supported, native output_config False
    caps_35 = get_model_capabilities("anthropic", "claude-3-5-sonnet-latest")
    assert caps_35.temperature is True
    assert caps_35.native_output_config is False

    # Unknown Anthropic model -> safe conservative fallback
    caps_unknown = get_model_capabilities("anthropic", "claude-experimental-x")
    assert caps_unknown.temperature is False
    assert caps_unknown.native_output_config is False


def test_deepseek_capabilities():
    # Chat models support temperature and thinking_control
    caps_chat = get_model_capabilities("deepseek", "deepseek-v4-flash")
    assert caps_chat.temperature is True
    assert caps_chat.thinking_control is True
    assert caps_chat.json_object is True

    # Reasoner models
    caps_r1 = get_model_capabilities("deepseek", "deepseek-reasoner")
    assert caps_r1.temperature is False
    assert caps_r1.thinking_control is True

    # Unknown deepseek model -> safe conservative fallback
    caps_unknown = get_model_capabilities("deepseek", "deepseek-unknown-v9")
    assert caps_unknown.temperature is False
    assert caps_unknown.thinking_control is False


def test_openrouter_capabilities():
    # Known mapped models
    caps_gpt = get_model_capabilities("openrouter", "openai/gpt-4o")
    assert caps_gpt.temperature is True
    assert caps_gpt.json_object is True

    for m in ["openai/gpt-5.6-sol", "openai/gpt-5.6", "openai/gpt-5.5"]:
        caps_gpt56 = get_model_capabilities("openrouter", m)
        assert caps_gpt56.temperature is False
        assert caps_gpt56.thinking_control is True
        assert caps_gpt56.temperature_requires_no_reasoning is True
        assert caps_gpt56.default_reasoning_effort == "medium"
        assert caps_gpt56.json_object is True

    for m in ["openai/gpt-5.1", "openai/gpt-5.2", "openai/gpt-5.4"]:
        caps_gpt5_none = get_model_capabilities("openrouter", m)
        assert caps_gpt5_none.temperature is False
        assert caps_gpt5_none.thinking_control is True
        assert caps_gpt5_none.temperature_requires_no_reasoning is True
        assert caps_gpt5_none.default_reasoning_effort == "none"
        assert caps_gpt5_none.json_object is True

    caps_ds = get_model_capabilities("openrouter", "deepseek/deepseek-v4-flash")
    assert caps_ds.temperature is True
    assert caps_ds.thinking_control is True

    caps_gem = get_model_capabilities("openrouter", "google/gemini-3.7-flash")
    assert caps_gem.temperature is False

    # Unknown OpenRouter model (e.g. openai/gpt-5.9-future) -> safe conservative fallback
    caps_unknown = get_model_capabilities("openrouter", "openai/gpt-5.9-future")
    assert caps_unknown.temperature is False
    assert caps_unknown.thinking_control is False
    assert caps_unknown.structured_output is False
    assert caps_unknown.json_object is False
    assert caps_unknown.temperature_requires_no_reasoning is False
    assert caps_unknown.default_reasoning_effort is None


def test_custom_capabilities():
    # Custom endpoints always use safe conservative fallback
    caps = get_model_capabilities("custom", "any-local-model")
    assert caps.temperature is False
    assert caps.thinking_control is False
    assert caps.structured_output is False
    assert caps.json_object is False


def test_ollama_capabilities():
    caps_llama = get_model_capabilities("ollama", "llama3")
    assert caps_llama.temperature is True
    assert caps_llama.json_object is True

    caps_unknown = get_model_capabilities("ollama", "my-finetuned-weights")
    assert caps_unknown.temperature is False
    assert caps_unknown.json_object is True


def test_deepl_capabilities():
    caps = get_model_capabilities("deepl", "prefer_quality_optimized")
    assert caps.temperature is False
    assert caps.thinking_control is False
    assert caps.structured_output is False
    assert caps.json_object is False
    assert caps.semantic_audit is False


def test_unknown_provider_fallback():
    caps = get_model_capabilities("totally_unknown_provider", "some-model")
    assert caps.temperature is False
    assert caps.thinking_control is False
    assert caps.structured_output is False
    assert caps.json_object is False
    assert caps.semantic_audit is False


# ============================================================================
# 2. DISPATCH & REQUEST BUILDING BEHAVIOR TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_dispatch_gemini_temperature_controlled():
    """TEST 1 & 2: Gemini supported vs unsupported temperature."""
    translator = SubtitleTranslator()

    captured_configs = []

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = '{"result": "ok"}'
    mock_resp.usage_metadata = None

    def capture_generate_content(model, contents, config):
        captured_configs.append({"model": model, "config": config})
        return mock_resp

    mock_client.models.generate_content.side_effect = capture_generate_content

    with patch("google.genai.Client", return_value=mock_client):
        # 1. Gemini 3.5 (temperature unsupported -> temperature NOT passed in config)
        await translator._dispatch_llm_completion(
            provider="gemini",
            model_name="gemini-3.5-flash-lite",
            system_prompt="sys",
            user_prompt="user",
            temperature=0.1,
        )
        assert len(captured_configs) == 1
        cfg1 = captured_configs[0]["config"]
        assert cfg1.temperature is None, "Gemini 3.x must NOT send temperature"

        # 2. Gemini 2.5 (temperature supported -> temperature passed in config)
        await translator._dispatch_llm_completion(
            provider="gemini",
            model_name="gemini-2.5-flash",
            system_prompt="sys",
            user_prompt="user",
            temperature=0.1,
        )
        assert len(captured_configs) == 2
        cfg2 = captured_configs[1]["config"]
        assert cfg2.temperature == 0.1, "Gemini 2.5 must send temperature when supported"


@pytest.mark.asyncio
async def test_dispatch_openai_gpt_5_2_sends_temperature_with_default_reasoning_none():
    """TEST 1 & 2: GPT-5.2 without explicit reasoning uses documented default 'none' and sends temperature."""
    translator = SubtitleTranslator()

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content='{"translations": []}'))]
    mock_resp.usage = None

    captured_kwargs = []
    mock_client.chat.completions.create.side_effect = lambda **kwargs: captured_kwargs.append(kwargs) or mock_resp

    with patch("openai.OpenAI", return_value=mock_client):
        await translator._dispatch_llm_completion(
            provider="openai",
            model_name="gpt-5.2",
            system_prompt="sys",
            user_prompt="user",
            schema={"type": "object"},
            temperature=0.1,
        )
        assert len(captured_kwargs) == 1
        kw = captured_kwargs[0]
        assert kw["temperature"] == 0.1, "gpt-5.2 must send temperature by default since default reasoning is 'none'"
        assert kw["response_format"]["type"] == "json_schema"
        assert kw["response_format"]["json_schema"]["strict"] is True


@pytest.mark.asyncio
async def test_dispatch_openai_gpt_5_3_chat_latest_sends_temperature():
    """TEST: gpt-5.3-chat-latest is an instant chat model and sends temperature."""
    translator = SubtitleTranslator()

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content='{"translations": []}'))]
    mock_resp.usage = None

    captured_kwargs = []
    mock_client.chat.completions.create.side_effect = lambda **kwargs: captured_kwargs.append(kwargs) or mock_resp

    with patch("openai.OpenAI", return_value=mock_client):
        await translator._dispatch_llm_completion(
            provider="openai",
            model_name="gpt-5.3-chat-latest",
            system_prompt="sys",
            user_prompt="user",
            schema={"type": "object"},
            temperature=0.1,
        )
        assert len(captured_kwargs) == 1
        kw = captured_kwargs[0]
        assert kw["temperature"] == 0.1, "gpt-5.3-chat-latest must send temperature"
        assert kw["response_format"]["type"] == "json_schema"
        assert kw["response_format"]["json_schema"]["strict"] is True


@pytest.mark.asyncio
async def test_dispatch_openai_gpt_5_6_omits_temperature_with_default_reasoning_medium():
    """TEST 3 & 4: GPT-5.6 without explicit reasoning uses default 'medium' and OMITS temperature."""
    translator = SubtitleTranslator()

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content='{"translations": []}'))]
    mock_resp.usage = None

    for m in ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6", "gpt-5.5"]:
        captured_kwargs = []
        mock_client.chat.completions.create.side_effect = lambda **kwargs: captured_kwargs.append(kwargs) or mock_resp

        with patch("openai.OpenAI", return_value=mock_client):
            await translator._dispatch_llm_completion(
                provider="openai",
                model_name=m,
                system_prompt="sys",
                user_prompt="user",
                schema={"type": "object"},
                temperature=0.1,
            )
            assert len(captured_kwargs) == 1
            kw = captured_kwargs[0]
            assert "temperature" not in kw, f"{m} must NOT send temperature in default reasoning mode ('medium')"
            assert kw["response_format"]["type"] == "json_schema"
            assert kw["response_format"]["json_schema"]["strict"] is True


@pytest.mark.asyncio
async def test_dispatch_openai_gpt_5_6_allows_temperature_when_explicit_reasoning_none():
    """TEST 5: Explicit reasoning_effort='none' for GPT-5.6 sends temperature."""
    translator = SubtitleTranslator()

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content='{"translations": []}'))]
    mock_resp.usage = None

    captured_kwargs = []
    mock_client.chat.completions.create.side_effect = lambda **kwargs: captured_kwargs.append(kwargs) or mock_resp

    with patch("openai.OpenAI", return_value=mock_client):
        await translator._dispatch_llm_completion(
            provider="openai",
            model_name="gpt-5.6-sol",
            system_prompt="sys",
            user_prompt="user",
            schema={"type": "object"},
            temperature=0.1,
            reasoning_effort="none",
        )
        assert len(captured_kwargs) == 1
        kw = captured_kwargs[0]
        assert kw["reasoning_effort"] == "none"
        assert kw["temperature"] == 0.1, "gpt-5.6-sol must send temperature when reasoning_effort is explicitly 'none'"
        assert kw["response_format"]["type"] == "json_schema"
        assert kw["response_format"]["json_schema"]["strict"] is True


@pytest.mark.asyncio
async def test_dispatch_openai_gpt_5_3_codex_never_sends_temperature():
    """TEST 6: gpt-5.3-codex does not support reasoning 'none' and NEVER sends temperature."""
    translator = SubtitleTranslator()

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content='{"translations": []}'))]
    mock_resp.usage = None

    captured_kwargs = []
    mock_client.chat.completions.create.side_effect = lambda **kwargs: captured_kwargs.append(kwargs) or mock_resp

    with patch("openai.OpenAI", return_value=mock_client):
        await translator._dispatch_llm_completion(
            provider="openai",
            model_name="gpt-5.3-codex",
            system_prompt="sys",
            user_prompt="user",
            schema={"type": "object"},
            temperature=0.1,
        )
        assert len(captured_kwargs) == 1
        kw = captured_kwargs[0]
        assert "temperature" not in kw, "gpt-5.3-codex must NOT send temperature"


@pytest.mark.asyncio
async def test_dispatch_openai_future_model_conservative_fallback():
    """TEST 8 & 9: Unknown future model (gpt-5.9-future) uses conservative fallback."""
    translator = SubtitleTranslator()

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content='{"translations": []}'))]
    mock_resp.usage = None

    captured_kwargs = []
    mock_client.chat.completions.create.side_effect = lambda **kwargs: captured_kwargs.append(kwargs) or mock_resp

    with patch("openai.OpenAI", return_value=mock_client):
        await translator._dispatch_llm_completion(
            provider="openai",
            model_name="gpt-5.9-future",
            system_prompt="sys",
            user_prompt="user",
            schema={"type": "object"},
            temperature=0.1,
        )
        assert len(captured_kwargs) == 1
        kw = captured_kwargs[0]
        assert "temperature" not in kw, "Unknown model must not receive temperature"
        assert "response_format" not in kw, "Unknown model with structured_output=False must not receive response_format"


@pytest.mark.asyncio
async def test_dispatch_openai_classical_gpt_sends_temperature():
    """TEST 9: Classical GPT models (gpt-4o, gpt-4.1) send temperature."""
    translator = SubtitleTranslator()

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content='{"translations": []}'))]
    mock_resp.usage = None

    for m in ["gpt-4o", "gpt-4.1"]:
        captured_kwargs = []
        mock_client.chat.completions.create.side_effect = lambda **kwargs: captured_kwargs.append(kwargs) or mock_resp

        with patch("openai.OpenAI", return_value=mock_client):
            await translator._dispatch_llm_completion(
                provider="openai",
                model_name=m,
                system_prompt="sys",
                user_prompt="user",
                schema={"type": "object"},
                temperature=0.1,
            )
            assert len(captured_kwargs) == 1
            kw = captured_kwargs[0]
            assert kw["temperature"] == 0.1, f"{m} must send temperature=0.1"
            assert kw["response_format"]["type"] == "json_schema"
            assert kw["response_format"]["json_schema"]["strict"] is True


@pytest.mark.asyncio
async def test_dispatch_openai_reasoning_and_temperature():
    """TEST: OpenAI reasoning model (o3-mini)."""
    translator = SubtitleTranslator()

    captured_kwargs = []
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content='{"translations": []}'))]
    mock_resp.usage = None

    def capture_create(**kwargs):
        captured_kwargs.append(kwargs)
        return mock_resp

    mock_client.chat.completions.create.side_effect = capture_create

    with patch("openai.OpenAI", return_value=mock_client):
        await translator._dispatch_llm_completion(
            provider="openai",
            model_name="o3-mini",
            system_prompt="sys",
            user_prompt="user",
            schema={"type": "object"},
            temperature=0.1,
        )
        assert len(captured_kwargs) == 1
        kw = captured_kwargs[0]
        assert "temperature" not in kw, "o3-mini must NOT send temperature"
        assert kw["response_format"]["type"] == "json_schema"
        assert kw["response_format"]["json_schema"]["strict"] is True


@pytest.mark.asyncio
async def test_dispatch_deepseek_thinking_control():
    """TEST: DeepSeek thinking control disabled via capability."""
    translator = SubtitleTranslator()

    captured_kwargs = []
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content='{"translations": []}'))]
    mock_resp.usage = None

    def capture_create(**kwargs):
        captured_kwargs.append(kwargs)
        return mock_resp

    mock_client.chat.completions.create.side_effect = capture_create

    with patch("openai.OpenAI", return_value=mock_client):
        # DeepSeek V4 Flash: thinking_control is True -> disables thinking for translation
        await translator._dispatch_llm_completion(
            provider="deepseek",
            model_name="deepseek-v4-flash",
            system_prompt="sys",
            user_prompt="user",
            schema={"type": "object"},
            temperature=0.1,
        )
        assert len(captured_kwargs) == 1
        kw = captured_kwargs[0]
        assert kw.get("extra_body") == {"thinking": {"type": "disabled"}}
        assert kw["temperature"] == 0.1


@pytest.mark.asyncio
async def test_dispatch_custom_openai_safe_fallback():
    """TEST: Custom OpenAI endpoint safe minimal request."""
    translator = SubtitleTranslator()

    captured_kwargs = []
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content='{"translations": []}'))]
    mock_resp.usage = None

    def capture_create(**kwargs):
        captured_kwargs.append(kwargs)
        return mock_resp

    mock_client.chat.completions.create.side_effect = capture_create

    with patch("openai.OpenAI", return_value=mock_client):
        await translator._dispatch_llm_completion(
            provider="custom",
            model_name="my-custom-llm-v1",
            system_prompt="sys",
            user_prompt="user",
            schema={"type": "object"},
            temperature=0.7,
        )
        assert len(captured_kwargs) == 1
        kw = captured_kwargs[0]
        assert kw["model"] == "my-custom-llm-v1"
        assert "temperature" not in kw, "Custom provider must not send optional temperature"
        assert "response_format" not in kw, "Custom provider must not send optional response_format"
        assert "extra_body" not in kw


@pytest.mark.asyncio
async def test_dispatch_openrouter_preserves_model_id_with_safe_fallback():
    """TEST: OpenRouter preserves exact model ID and uses capability-driven fields."""
    translator = SubtitleTranslator()

    captured_kwargs = []
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content='{"translations": []}'))]
    mock_resp.usage = None

    def capture_create(**kwargs):
        captured_kwargs.append(kwargs)
        return mock_resp

    mock_client.chat.completions.create.side_effect = capture_create

    with patch("openai.OpenAI", return_value=mock_client):
        # Exact model ID preserved:
        await translator._dispatch_llm_completion(
            provider="openrouter",
            model_name="meta-llama/llama-3.3-70b-instruct",
            system_prompt="sys",
            user_prompt="user",
            schema={"type": "object"},
            temperature=0.1,
        )
        assert len(captured_kwargs) == 1
        kw = captured_kwargs[0]
        assert kw["model"] == "meta-llama/llama-3.3-70b-instruct"
        assert kw["temperature"] == 0.1
        assert "response_format" not in kw  # llama on openrouter doesn't force json_object


@pytest.mark.asyncio
async def test_dispatch_ollama_json_and_temperature_capability():
    """TEST: Ollama only sends format='json' and options when capabilities allow."""
    translator = SubtitleTranslator()

    captured_payloads = []
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": '{"translations": []}'}

    async def mock_post(url, json=None, **kwargs):
        captured_payloads.append(json)
        return mock_resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        # 1. Known Llama3 model on Ollama (supports temperature and json_object)
        await translator._dispatch_llm_completion(
            provider="ollama",
            model_name="llama3",
            system_prompt="sys",
            user_prompt="user",
            schema={"type": "object"},
            temperature=0.1,
        )
        assert len(captured_payloads) == 1
        p1 = captured_payloads[0]
        assert p1["format"] == "json"
        assert p1["options"] == {"temperature": 0.1}

        # 2. LocalAI transport alias: reuses Ollama endpoint transport, but does NOT inherit speculative capabilities
        await translator._dispatch_llm_completion(
            provider="localai",
            model_name="llama3",
            system_prompt="sys",
            user_prompt="user",
            schema={"type": "object"},
            temperature=0.1,
        )
        assert len(captured_payloads) == 2
        p2 = captured_payloads[1]
        assert "format" not in p2, "LocalAI must not inherit Ollama json_object format"
        assert "options" not in p2, "LocalAI must not inherit Ollama temperature options"


# ============================================================================
# 3. PRIMARY / GENERIC CONSISTENCY (TEST 10)
# ============================================================================

@pytest.mark.asyncio
async def test_primary_and_generic_consistency_openai():
    """TEST 10: Primary translate_batch_openai and generic dispatch use identical capability rules."""
    translator = SubtitleTranslator()

    captured_kwargs = []
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content='{"translations": [{"id": 0, "text": "hej"}]}'))]
    mock_resp.usage = None

    def capture_create(**kwargs):
        captured_kwargs.append(kwargs)
        return mock_resp

    mock_client.chat.completions.create.side_effect = capture_create

    with patch.object(translator, "get_openai_client", return_value=mock_client):
        # 1. Primary batch translation with GPT-5.6 Sol (default reasoning mode -> omit temperature)
        await translator.translate_batch_openai(
            items=[{"id": 0, "text": "hello"}],
            target_language="Swedish",
            model_name="gpt-5.6-sol",
        )
        assert len(captured_kwargs) == 1
        primary_kw = captured_kwargs[0]
        assert "temperature" not in primary_kw, "Primary translate_batch_openai must not send temperature to gpt-5.6-sol"
        assert primary_kw["response_format"]["type"] == "json_schema"
        assert primary_kw["response_format"]["json_schema"]["strict"] is True

    with patch("openai.OpenAI", return_value=mock_client):
        # 2. Generic dispatch with same model gpt-5.6-sol
        await translator._dispatch_llm_completion(
            provider="openai",
            model_name="gpt-5.6-sol",
            system_prompt="sys",
            user_prompt="user",
            schema={"type": "object"},
            temperature=0.1,
        )
        assert len(captured_kwargs) == 2
        generic_kw = captured_kwargs[1]
        assert "temperature" not in generic_kw, "Generic dispatch must not send temperature to gpt-5.6-sol"
        assert generic_kw["response_format"]["type"] == "json_schema"
        assert generic_kw["response_format"]["json_schema"]["strict"] is True


# ============================================================================
# 4. JOB PINNING PRESERVATION (TEST 11)
# ============================================================================

def test_pinned_job_capabilities_preserved():
    """TEST 11: Pinned job provider and model context resolves exact pinned model capabilities."""
    job_id = create_job("test_pinning_video.mkv")

    # Pin job to Gemini 2.5 (temperature supported)
    pin_job_provider(job_id, "gemini", "gemini-2.5-flash")

    # Change global settings to OpenAI o3-mini (temperature unsupported)
    set_setting("ai_provider", "openai")
    set_setting("openai_model", "o3-mini")

    # Resolve context for pinned job
    ctx = resolve_job_provider_context(job_id)
    assert ctx.provider == "gemini"
    assert ctx.model == "gemini-2.5-flash"

    # Capability resolution for pinned job
    caps = get_model_capabilities(ctx.provider, ctx.model)
    assert caps.temperature is True, "Pinned job capability must be resolved for gemini-2.5-flash (True)"

    # Conversely, unpinned context uses new settings
    unpinned_ctx = context_from_settings()
    assert unpinned_ctx.provider == "openai"
    assert unpinned_ctx.model == "o3-mini"
    unpinned_caps = get_model_capabilities(unpinned_ctx.provider, unpinned_ctx.model)
    assert unpinned_caps.temperature is False, "Unpinned settings capability must resolve for o3-mini (False)"


# ============================================================================
# 5. DEEPL REGRESSION PRESERVATION
# ============================================================================

@pytest.mark.asyncio
async def test_deepl_regression_not_affected_by_capabilities():
    """DeepL translation and QA bypass flows are completely unaffected."""
    translator = SubtitleTranslator()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"translations": [{"text": "Hej världen"}]}

    set_setting("deepl_api_key", "test-key")
    set_setting("deepl_model_type", "prefer_quality_optimized")

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        res = await translator.translate_batch_deepl(
            items=[{"id": 0, "text": "Hello world"}],
            target_language="Swedish",
            source_language="English",
        )
        assert len(res) == 1
        assert res[0]["text"] == "Hej världen"

        # Check posted JSON
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["text"] == ["Hello world"]
        assert call_kwargs["json"]["model_type"] == "prefer_quality_optimized"

    set_setting("deepl_api_key", "")
