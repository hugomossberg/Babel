"""
Comprehensive regression tests for Babel AI Model Catalog, Discovery, and Contextual Escalation.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.core.ai_providers import (
    PROVIDERS,
    DEFAULT_MODELS,
    PROVIDER_FALLBACK_CATALOGS,
    get_default_model,
    get_provider_catalog,
    get_provider_spec,
    normalize_provider,
    context_from_settings,
    resolve_job_provider_context,
    get_model_capabilities,
    ProviderContext,
)
from app.core.db import get_setting, set_setting, create_job, pin_job_provider


@pytest.fixture
def client():
    return TestClient(app)


# ============================================================================
# 1. CENTRAL MODEL CATALOG & SOURCE OF TRUTH
# ============================================================================

def test_central_catalog_consistency():
    """Verify that all supported providers have entries in PROVIDERS, DEFAULT_MODELS, and catalogs."""
    expected_providers = ["gemini", "openai", "anthropic", "openrouter", "deepseek", "custom", "ollama", "deepl"]
    for prov in expected_providers:
        assert prov in PROVIDERS
        assert prov in DEFAULT_MODELS
        assert prov in PROVIDER_FALLBACK_CATALOGS
        spec = get_provider_spec(prov)
        assert spec.default_model == DEFAULT_MODELS[prov]
        catalog = get_provider_catalog(prov)
        assert len(catalog) >= 1
        # Check that default model exists in catalog
        default_m = get_default_model(prov)
        assert any(m["id"] == default_m for m in catalog)


# ============================================================================
# 2. GOOGLE GEMINI CATALOG & DISCOVERY
# ============================================================================

def test_gemini_catalog_contains_current_models(client):
    """Verify Gemini catalog contains multiple current text models including Gemini 3.x series."""
    catalog = get_provider_catalog("gemini")
    model_ids = [m["id"] for m in catalog]

    assert "gemini-3.7-flash" in model_ids
    assert "gemini-3.6-flash" in model_ids
    assert "gemini-3.5-flash" in model_ids
    assert "gemini-3.5-flash-lite" in model_ids
    assert "gemini-3.1-flash-lite" in model_ids

    # Verify endpoint returns Gemini models
    resp = client.get("/api/settings/models?provider=gemini")
    assert resp.status_code == 200
    data = resp.json()
    resp_ids = [m["id"] for m in data["models"]]
    assert "gemini-3.7-flash" in resp_ids
    assert "gemini-3.5-flash-lite" in resp_ids
    assert data["default_model"] == "gemini-3.5-flash-lite"


@pytest.mark.asyncio
async def test_gemini_live_discovery_filters_non_text(client):
    """Verify live discovery for Gemini filters out image/audio/embedding models."""
    from app.api.dashboard import _MODELS_CACHE
    _MODELS_CACHE.pop("gemini:", None)

    mock_gemini_response = {
        "models": [
            {"name": "models/gemini-3.7-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-embedding-001", "supportedGenerationMethods": ["embedContent"]},
            {"name": "models/imagen-3.0-generate-002", "supportedGenerationMethods": ["generateImages"]},
            {"name": "models/gemini-future-text-2027", "supportedGenerationMethods": ["generateContent"]},
        ]
    }

    set_setting("gemini_api_key", "test-gemini-key")
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_gemini_response
        mock_get.return_value = mock_resp

        resp = client.get("/api/settings/models?provider=gemini")
        assert resp.status_code == 200
        ids = [m["id"] for m in resp.json()["models"]]
        assert "gemini-3.7-flash" in ids
        assert "gemini-future-text-2027" in ids
        assert "gemini-embedding-001" not in ids
        assert "imagen-3.0-generate-002" not in ids
    set_setting("gemini_api_key", "")
    _MODELS_CACHE.pop("gemini:", None)


# ============================================================================
# 3. OPENAI CATALOG & DISCOVERY (STRICT ISOLATION)
# ============================================================================

def test_openai_catalog_contains_current_gpt_models(client):
    """Verify direct OpenAI provider returns GPT models and excludes non-OpenAI models."""
    catalog = get_provider_catalog("openai")
    model_ids = [m["id"] for m in catalog]

    assert "gpt-5.6-sol" in model_ids
    assert "gpt-5.6-terra" in model_ids
    assert "gpt-5.6-luna" in model_ids
    assert "gpt-4o-mini" in model_ids
    assert "gpt-4o" in model_ids

    # Ensure NO Gemini or Claude models in OpenAI catalog
    assert not any("gemini" in m for m in model_ids)
    assert not any("claude" in m for m in model_ids)

    resp = client.get("/api/settings/models?provider=openai")
    assert resp.status_code == 200
    data = resp.json()
    resp_ids = [m["id"] for m in data["models"]]
    assert "gpt-5.6-sol" in resp_ids
    assert "gpt-4o-mini" in resp_ids
    assert not any("gemini" in m for m in resp_ids)


# ============================================================================
# 4. DEEPSEEK CATALOG & V4 IDS
# ============================================================================

def test_deepseek_catalog_uses_v4_ids(client):
    """Verify DeepSeek uses V4 IDs and does NOT use deepseek-chat as new default."""
    assert DEFAULT_MODELS["deepseek"] == "deepseek-v4-flash"

    catalog = get_provider_catalog("deepseek")
    model_ids = [m["id"] for m in catalog]
    assert "deepseek-v4-flash" in model_ids
    assert "deepseek-v4-pro" in model_ids
    assert "deepseek-chat" not in model_ids

    resp = client.get("/api/settings/models?provider=deepseek")
    assert resp.status_code == 200
    data = resp.json()
    assert data["default_model"] == "deepseek-v4-flash"
    resp_ids = [m["id"] for m in data["models"]]
    assert "deepseek-v4-flash" in resp_ids
    assert "deepseek-v4-pro" in resp_ids


def test_deepseek_legacy_saved_model_preserved():
    """Verify that a user who has legacy deepseek-chat saved in DB retains it."""
    set_setting("deepseek_model", "deepseek-chat")
    ctx = context_from_settings("deepseek")
    assert ctx.model == "deepseek-chat"


# ============================================================================
# 5. OPENROUTER DYNAMIC CATALOG & PROVIDER/MODEL FORMAT
# ============================================================================

def test_openrouter_live_discovery_and_id_format(client):
    """Verify OpenRouter returns dynamic model list preserving exact provider/model ID format."""
    from app.api.dashboard import _MODELS_CACHE
    _MODELS_CACHE.pop("openrouter:", None)

    mock_openrouter_data = {
        "data": [
            {"id": "google/gemini-3.7-flash", "name": "Google: Gemini 3.7 Flash"},
            {"id": "anthropic/claude-sonnet-5", "name": "Anthropic: Claude Sonnet 5"},
            {"id": "openai/gpt-4o-mini", "name": "OpenAI: GPT-4o-mini"},
            {"id": "deepseek/deepseek-v4-flash", "name": "DeepSeek: DeepSeek V4 Flash"},
        ]
    }

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_openrouter_data
        mock_get.return_value = mock_resp

        resp = client.get("/api/settings/models?provider=openrouter")
        assert resp.status_code == 200
        models = resp.json()["models"]
        assert len(models) == 4
        assert models[0]["id"] == "google/gemini-3.7-flash"
        assert "/" in models[0]["id"]
        assert models[1]["id"] == "anthropic/claude-sonnet-5"


def test_openrouter_fallback_catalog_has_provider_model_format():
    """Verify OpenRouter fallback catalog uses provider/model format."""
    catalog = get_provider_catalog("openrouter")
    for m in catalog:
        assert "/" in m["id"], f"OpenRouter model ID {m['id']} missing provider/ prefix"


# ============================================================================
# 6. CONTEXTUAL ESCALATION & PROVIDER-MODEL COUPLING
# ============================================================================

def test_contextual_escalation_same_as_primary():
    """When escalation_provider is 'none', escalation context matches primary."""
    set_setting("ai_provider", "gemini")
    set_setting("gemini_model", "gemini-3.5-flash-lite")
    set_setting("escalate_to_pro", "true")
    set_setting("escalation_provider", "none")
    set_setting("escalation_model", "")

    esc_ctx = context_from_settings(escalation=True)
    assert esc_ctx.provider == "gemini"
    assert esc_ctx.model == "gemini-3.5-flash-lite"


def test_contextual_escalation_provider_switch_anthropic():
    """When escalation_provider is anthropic, escalation context uses Anthropic."""
    set_setting("ai_provider", "gemini")
    set_setting("escalate_to_pro", "true")
    set_setting("escalation_provider", "anthropic")
    set_setting("escalation_model", "claude-sonnet-5")

    esc_ctx = context_from_settings(escalation=True)
    assert esc_ctx.provider == "anthropic"
    assert esc_ctx.model == "claude-sonnet-5"


def test_contextual_escalation_provider_switch_deepseek():
    """When escalation_provider is deepseek, escalation context uses DeepSeek V4."""
    set_setting("ai_provider", "openai")
    set_setting("escalate_to_pro", "true")
    set_setting("escalation_provider", "deepseek")
    set_setting("escalation_model", "deepseek-v4-flash")

    esc_ctx = context_from_settings(escalation=True)
    assert esc_ctx.provider == "deepseek"
    assert esc_ctx.model == "deepseek-v4-flash"


def test_contextual_escalation_disabled():
    """When escalate_to_pro is false, context_from_settings(escalation=True) returns primary."""
    set_setting("ai_provider", "openai")
    set_setting("openai_model", "gpt-4o-mini")
    set_setting("escalate_to_pro", "false")
    set_setting("escalation_provider", "anthropic")
    set_setting("escalation_model", "claude-sonnet-5")

    esc_ctx = context_from_settings(escalation=True)
    assert esc_ctx.provider == "openai"
    assert esc_ctx.model == "gpt-4o-mini"


def test_job_pinning_and_escalation_resolution():
    """Verify job provider pinning preserves primary and escalation independently."""
    job_id = create_job("test_escalation_pinning.mkv")

    pin_job_provider(
        job_id,
        primary_provider="gemini",
        primary_model="gemini-3.5-flash-lite",
        escalation_enabled=True,
        escalation_provider="anthropic",
        escalation_model="claude-opus-5",
    )

    primary_ctx = resolve_job_provider_context(job_id, escalation=False)
    assert primary_ctx.provider == "gemini"
    assert primary_ctx.model == "gemini-3.5-flash-lite"

    esc_ctx = resolve_job_provider_context(job_id, escalation=True)
    assert esc_ctx.provider == "anthropic"
    assert esc_ctx.model == "claude-opus-5"


# ============================================================================
# 7. FUTURE MODEL TEST (DYNAMIC INTEGRATION WITHOUT FRONTEND HARDCODING)
# ============================================================================

def test_future_model_id_dynamically_returned(client):
    """If backend discovery returns a brand new model ID, it is provided in API response."""
    from app.api.dashboard import _MODELS_CACHE
    _MODELS_CACHE.pop("custom:http://custom-ai:8000/v1", None)

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"id": "nextgen-ai-2028-hyper-translate", "name": "NextGen AI 2028"}
            ]
        }
        mock_get.return_value = mock_resp

        resp = client.get("/api/settings/models?provider=custom&url=http://custom-ai:8000/v1")
        assert resp.status_code == 200
        models = resp.json()["models"]
        assert any(m["id"] == "nextgen-ai-2028-hyper-translate" for m in models)


# ============================================================================
# 8. DEEPL HANDLING (NON-GENERAL LLM)
# ============================================================================

def test_deepl_provider_spec():
    """Verify DeepL is marked as general_llm=False and has strategy models."""
    spec = get_provider_spec("deepl")
    assert spec.general_llm is False
    catalog = get_provider_catalog("deepl")
    ids = [m["id"] for m in catalog]
    assert "prefer_quality_optimized" in ids
    assert "latency_optimized" in ids
