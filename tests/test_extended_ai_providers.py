"""
Tests for extended AI providers (Anthropic, OpenRouter, DeepSeek, Custom OpenAI-compatible),
including dispatching, settings persistence, secret masking, connectivity tests,
contextual escalation, and model catalog caching/fallback.
"""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.core.db import get_setting, set_setting
from app.core.security import mask_secret, resolve_secret_key, is_masked_secret
from app.services.translator import SubtitleTranslator, get_provider_capabilities


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def translator():
    return SubtitleTranslator()


# ============================================================================
# 1. DISPATCH TESTS (Anthropic, OpenRouter, DeepSeek, Custom)
# ============================================================================

@pytest.mark.asyncio
async def test_anthropic_dispatch_mocked(translator):
    """Verify Anthropic Messages API dispatch using httpx."""
    set_setting("anthropic_api_key", "test-anthropic-key")
    set_setting("anthropic_model", "claude-sonnet-5")

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "content": [{"type": "text", "text": "Hej världen"}]
        }
        mock_post.return_value = mock_resp

        result = await translator._dispatch_llm_completion(
            provider="anthropic",
            model_name="claude-sonnet-5",
            system_prompt="Translate English to Swedish",
            user_prompt="Hello world",
        )

        assert result.strip() == "Hej världen"
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "https://api.anthropic.com/v1/messages" in args[0]
        assert kwargs["headers"]["x-api-key"] == "test-anthropic-key"
        assert kwargs["headers"]["anthropic-version"] == "2023-06-01"


@pytest.mark.asyncio
async def test_openrouter_dispatch_mocked(translator):
    """Verify OpenRouter dispatch using openai SDK compatibility layer with custom headers."""
    set_setting("openrouter_api_key", "test-openrouter-key")
    set_setting("openrouter_model", "anthropic/claude-3.5-sonnet")

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "God morgon"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_resp

        result = await translator._dispatch_llm_completion(
            provider="openrouter",
            model_name="anthropic/claude-3.5-sonnet",
            system_prompt="Translate English to Swedish",
            user_prompt="Good morning",
        )

        assert result.strip() == "God morgon"
        mock_openai_cls.assert_called_once()
        _, kwargs = mock_openai_cls.call_args
        assert kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert kwargs["api_key"] == "test-openrouter-key"
        assert "HTTP-Referer" in kwargs["default_headers"]
        assert "X-Title" in kwargs["default_headers"]


@pytest.mark.asyncio
async def test_deepseek_dispatch_mocked(translator):
    """Verify DeepSeek dispatch using official base URL."""
    set_setting("deepseek_api_key", "test-deepseek-key")
    set_setting("deepseek_model", "deepseek-v4-flash")

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "Tack så mycket"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_resp

        result = await translator._dispatch_llm_completion(
            provider="deepseek",
            model_name="deepseek-v4-flash",
            system_prompt="Translate English to Swedish",
            user_prompt="Thank you very much",
        )

        assert result.strip() == "Tack så mycket"
        mock_openai_cls.assert_called_once()
        _, kwargs = mock_openai_cls.call_args
        assert kwargs["base_url"] == "https://api.deepseek.com"
        assert kwargs["api_key"] == "test-deepseek-key"


@pytest.mark.asyncio
async def test_custom_openai_dispatch_mocked(translator):
    """Verify Custom OpenAI-compatible dispatch with user-provided URL and model."""
    set_setting("custom_openai_url", "http://192.168.1.50:8000/v1")
    set_setting("custom_openai_api_key", "custom-secret")
    set_setting("custom_openai_model", "my-custom-llm")

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "Välkommen"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_resp

        result = await translator._dispatch_llm_completion(
            provider="custom",
            model_name="my-custom-llm",
            system_prompt="Translate English to Swedish",
            user_prompt="Welcome",
        )

        assert result.strip() == "Välkommen"
        mock_openai_cls.assert_called_once()
        _, kwargs = mock_openai_cls.call_args
        assert kwargs["base_url"] == "http://192.168.1.50:8000/v1"
        assert kwargs["api_key"] == "custom-secret"


# ============================================================================
# 2. TEST CONNECTION FOR ALL 4 EXTENDED PROVIDERS
# ============================================================================

def test_test_ai_connection_anthropic(client):
    """Test /api/settings/test-ai for Anthropic."""
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        resp = client.post("/api/settings/test-ai", json={
            "provider": "anthropic",
            "api_key": "sk-ant-test-key",
            "model": "claude-sonnet-5"
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["response"] == "Connected"


def test_test_ai_connection_openrouter(client):
    """Test /api/settings/test-ai for OpenRouter."""
    with patch("openai.OpenAI") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_choice = MagicMock()
        mock_choice.message.content = "Connected"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_instance.chat.completions.create.return_value = mock_resp

        resp = client.post("/api/settings/test-ai", json={
            "provider": "openrouter",
            "api_key": "sk-or-test-key",
            "model": "anthropic/claude-3.5-sonnet"
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_test_ai_connection_deepseek(client):
    """Test /api/settings/test-ai for DeepSeek."""
    with patch("openai.OpenAI") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_choice = MagicMock()
        mock_choice.message.content = "Connected"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_instance.chat.completions.create.return_value = mock_resp

        resp = client.post("/api/settings/test-ai", json={
            "provider": "deepseek",
            "api_key": "sk-deepseek-test",
            "model": "deepseek-v4-flash"
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_test_ai_connection_custom_openai(client):
    """Test /api/settings/test-ai for Custom OpenAI."""
    with patch("openai.OpenAI") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_choice = MagicMock()
        mock_choice.message.content = "Connected"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_instance.chat.completions.create.return_value = mock_resp

        resp = client.post("/api/settings/test-ai", json={
            "provider": "custom",
            "api_key": "local-key",
            "model": "mistral-7b",
            "url": "http://localhost:8000/v1"
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ============================================================================
# 3. SETTINGS PERSISTENCE & SECRET MASKING
# ============================================================================

def test_settings_persistence_and_masking(client):
    """Verify settings save, persistence in DB, and masking on GET."""
    payload = {
        "ai_provider": "anthropic",
        "anthropic_api_key": "sk-ant-very-secret-key-12345",
        "anthropic_model": "claude-3-5-haiku-latest",
        "openrouter_api_key": "sk-or-very-secret-key-67890",
        "openrouter_model": "deepseek/deepseek-v4-flash",
        "deepseek_api_key": "sk-ds-very-secret-key-abcde",
        "deepseek_model": "deepseek-reasoner",
        "custom_openai_url": "http://10.0.0.1:8000/v1",
        "custom_openai_api_key": "sk-cust-very-secret-xyz",
        "custom_openai_model": "llama-3-8b",
        "escalation_provider": "deepseek",
        "escalation_model": "deepseek-reasoner",
        "escalate_to_pro": True,
        "batch_size": 50,
        "max_concurrent_jobs": 3,
        "batch_concurrency": 2,
    }

    save_resp = client.post("/api/settings/ai", json=payload)
    assert save_resp.status_code == 200
    assert save_resp.json()["status"] == "saved"

    # Verify DB contains actual values
    assert get_setting("ai_provider") == "anthropic"
    assert get_setting("anthropic_api_key") == "sk-ant-very-secret-key-12345"
    assert get_setting("anthropic_model") == "claude-3-5-haiku-latest"
    assert get_setting("openrouter_api_key") == "sk-or-very-secret-key-67890"
    assert get_setting("openrouter_model") == "deepseek/deepseek-v4-flash"
    assert get_setting("deepseek_api_key") == "sk-ds-very-secret-key-abcde"
    assert get_setting("deepseek_model") == "deepseek-reasoner"
    assert get_setting("custom_openai_url") == "http://10.0.0.1:8000/v1"
    assert get_setting("custom_openai_api_key") == "sk-cust-very-secret-xyz"
    assert get_setting("custom_openai_model") == "llama-3-8b"
    assert get_setting("escalation_provider") == "deepseek"
    assert get_setting("escalation_model") == "deepseek-reasoner"

    # Verify GET /api/settings/all masks the secrets
    get_resp = client.get("/api/settings/all")
    assert get_resp.status_code == 200
    ai_data = get_resp.json()["ai"]

    assert is_masked_secret(ai_data["anthropic_api_key"])
    assert is_masked_secret(ai_data["openrouter_api_key"])
    assert is_masked_secret(ai_data["deepseek_api_key"])
    assert is_masked_secret(ai_data["custom_openai_api_key"])
    assert ai_data["has_anthropic_key"] is True
    assert ai_data["has_openrouter_key"] is True
    assert ai_data["has_deepseek_key"] is True
    assert ai_data["has_custom_openai_key"] is True

    # Verify unmasking when saving with masked values
    save_masked_resp = client.post("/api/settings/ai", json={
        "anthropic_api_key": ai_data["anthropic_api_key"],
        "openrouter_api_key": ai_data["openrouter_api_key"],
    })
    assert save_masked_resp.status_code == 200
    # Original secrets should remain intact
    assert get_setting("anthropic_api_key") == "sk-ant-very-secret-key-12345"
    assert get_setting("openrouter_api_key") == "sk-or-very-secret-key-67890"


# ============================================================================
# 4. CONTEXTUAL ESCALATION FOR COMPATIBLE PROVIDERS
# ============================================================================

@pytest.mark.asyncio
async def test_contextual_escalation_with_extended_providers(translator):
    """Verify escalate_single_line dispatches to configured escalation provider."""
    # Test escalation to Anthropic
    with patch.object(translator, "_dispatch_llm_completion", new_callable=AsyncMock) as mock_dispatch:
        mock_dispatch.return_value = '{"translation": "Översatt rad"}'

        set_setting("ai_provider", "openai")
        set_setting("escalate_to_pro", "true")
        set_setting("escalation_provider", "anthropic")
        set_setting("escalation_model", "claude-3-opus-latest")
        set_setting("anthropic_api_key", "test-key")

        res = await translator.escalate_single_line(
            target_idx=1,
            target_text="Difficult dialogue",
            prev_text="Previous line",
            next_text="Next line",
            target_language="Swedish",
            show_title="Test Movie",
            source_language="English",
        )

        assert res == "Översatt rad"
        mock_dispatch.assert_called_once()
        args, kwargs = mock_dispatch.call_args
        assert kwargs["provider"] == "anthropic"
        assert kwargs["model_name"] == "claude-3-opus-latest"


# ============================================================================
# 5. DYNAMIC MODEL CATALOG, CACHING & FALLBACK
# ============================================================================

def test_model_catalog_endpoint_and_cache(client):
    """Verify /api/settings/models endpoint with caching and fallback."""
    from app.api.dashboard import _MODELS_CACHE
    _MODELS_CACHE.clear()
    set_setting("gemini_api_key", "")
    # Gemini models
    resp_gemini = client.get("/api/settings/models?provider=gemini")
    assert resp_gemini.status_code == 200
    assert len(resp_gemini.json()["models"]) >= 4

    # Anthropic models
    resp_ant = client.get("/api/settings/models?provider=anthropic")
    assert resp_ant.status_code == 200
    ant_models = [m["id"] for m in resp_ant.json()["models"]]
    assert "claude-sonnet-5" in ant_models

    # DeepSeek models
    resp_ds = client.get("/api/settings/models?provider=deepseek")
    assert resp_ds.status_code == 200
    ds_models = [m["id"] for m in resp_ds.json()["models"]]
    assert "deepseek-v4-flash" in ds_models

    # OpenRouter models (mocked or fallback)
    with patch("httpx.AsyncClient.get") as mock_async_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Llama 3.3 70B"},
                {"id": "mistralai/mistral-large-2411", "name": "Mistral Large"}
            ]
        }
        mock_async_get.return_value = mock_resp

        # Clear cache first if needed
        from app.api.dashboard import _MODELS_CACHE
        _MODELS_CACHE.pop("openrouter:", None)

        resp_or = client.get("/api/settings/models?provider=openrouter")
        assert resp_or.status_code == 200
        or_models = [m["id"] for m in resp_or.json()["models"]]
        assert "meta-llama/llama-3.3-70b-instruct" in or_models

        # Check cached on second call
        resp_or_cached = client.get("/api/settings/models?provider=openrouter")
        assert resp_or_cached.status_code == 200
        assert resp_or_cached.json()["cached"] is True


def test_saved_model_not_in_catalog_preserved():
    """Verify that a saved custom model is not wiped or reset if not in catalog."""
    set_setting("openrouter_model", "my-custom/special-fine-tuned-model:v2")
    assert get_setting("openrouter_model") == "my-custom/special-fine-tuned-model:v2"

    set_setting("custom_openai_model", "hf-internal/custom-llm")
    assert get_setting("custom_openai_model") == "hf-internal/custom-llm"
