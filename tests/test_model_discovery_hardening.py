"""
tests/test_model_discovery_hardening.py
======================================
Regression tests for AI provider model discovery, capability-based filtering,
saved model display without '(saved)', batch deduplication, and DeepL strategies.
"""

import pytest
import re
from unittest.mock import patch, MagicMock
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
    filter_gemini_models,
    filter_openai_models,
    filter_anthropic_models,
    filter_openrouter_models,
    filter_ollama_models,
    filter_custom_models,
)
from app.core.db import get_setting, set_setting


@pytest.fixture
def client():
    return TestClient(app)


# ============================================================================
# 1. UI DISPLAY: NO '(saved)' OR '(batch)' IN MODEL OPTIONS
# ============================================================================

def test_html_template_has_no_saved_or_batch_literals():
    """Verify that index.html never renders '(saved)' or '(batch)' suffixes."""
    import os
    html_path = os.path.join(os.path.dirname(__file__), "..", "app", "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "(saved)" not in content, "Found literal '(saved)' in HTML template"
    assert " + ' (saved)'" not in content, "Found JS '(saved)' concatenation in HTML template"
    assert " + ' (batch)'" not in content, "Found JS '(batch)' concatenation in HTML template"


def test_saved_model_is_still_selected_when_not_in_live_catalog():
    """Saved model must be retained in context even if absent from catalog."""
    set_setting("gemini_model", "legacy-gemini-custom-snapshot")
    ctx = context_from_settings("gemini")
    assert ctx.model == "legacy-gemini-custom-snapshot"

    set_setting("openai_model", "custom-openai-fine-tune")
    ctx_oa = context_from_settings("openai")
    assert ctx_oa.model == "custom-openai-fine-tune"


# ============================================================================
# 2. GOOGLE GEMINI FILTERING & DISCOVERY
# ============================================================================

def test_filter_gemini_models_removes_non_text():
    """Gemini filter rejects image, audio, Lyria, computer-use, deep-research, antigravity, and embeddings."""
    raw_models = [
        {"name": "models/gemini-3.5-flash-lite", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-3.7-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-future-text-2028", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/imagen-3.0-generate-002", "supportedGenerationMethods": ["generateImages"]},
        {"name": "models/veo-2.0-generate-001", "supportedGenerationMethods": ["generateVideos"]},
        {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
        {"name": "models/lyria-realtime-001", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-computer-use-preview", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-deep-research-001", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/antigravity-preview-001", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/aqa-qa-001", "supportedGenerationMethods": ["generateContent"]},
    ]
    filtered = filter_gemini_models(raw_models)
    ids = [m["id"] for m in filtered]

    assert "gemini-3.5-flash-lite" in ids
    assert "gemini-3.7-flash" in ids
    assert "gemini-future-text-2028" in ids

    assert "imagen-3.0-generate-002" not in ids
    assert "veo-2.0-generate-001" not in ids
    assert "text-embedding-004" not in ids
    assert "lyria-realtime-001" not in ids
    assert "gemini-computer-use-preview" not in ids
    assert "gemini-deep-research-001" not in ids
    assert "antigravity-preview-001" not in ids
    assert "aqa-qa-001" not in ids


def test_gemini_endpoint_live_discovery(client):
    """GET /api/settings/models?provider=gemini uses capability filtering and preserves exact IDs."""
    from app.api.dashboard import _MODELS_CACHE
    _MODELS_CACHE.pop("gemini:", None)

    mock_data = {
        "models": [
            {"name": "models/gemini-3.5-flash-lite", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-3.7-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/imagen-3.0", "supportedGenerationMethods": ["generateImages"]},
            {"name": "models/gemini-deep-research", "supportedGenerationMethods": ["generateContent"]},
        ]
    }
    set_setting("gemini_api_key", "test-gemini-key")
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_data
        mock_get.return_value = mock_resp

        resp = client.get("/api/settings/models?provider=gemini&refresh=true")
        assert resp.status_code == 200
        data = resp.json()
        ids = [m["id"] for m in data["models"]]
        assert "gemini-3.5-flash-lite" in ids
        assert "gemini-3.7-flash" in ids
        assert "imagen-3.0" not in ids
        assert "gemini-deep-research" not in ids


# ============================================================================
# 3. OPENAI FILTERING & DISCOVERY
# ============================================================================

def test_filter_openai_models_removes_non_text_and_snapshots():
    """OpenAI filter retains standard text models and removes transcription, TTS, image, embeddings, and dated snapshots."""
    raw_models = [
        {"id": "gpt-4o"},
        {"id": "gpt-4o-mini"},
        {"id": "o1-mini"},
        {"id": "o3-mini"},
        {"id": "chatgpt-4o-latest"},
        {"id": "gpt-5.6-terra"},
        {"id": "whisper-1"},
        {"id": "tts-1"},
        {"id": "tts-1-hd"},
        {"id": "dall-e-3"},
        {"id": "text-embedding-3-small"},
        {"id": "text-moderation-latest"},
        {"id": "gpt-4o-realtime-preview"},
        {"id": "gpt-4o-audio-preview"},
        {"id": "gpt-4o-2024-08-06"},
        {"id": "o1-mini-2024-09-12"},
        {"id": "gpt-4-0613"},
        {"id": "ft:gpt-4o:my-org:custom-001"},
    ]
    filtered = filter_openai_models(raw_models)
    ids = [m["id"] for m in filtered]

    assert "gpt-4o" in ids
    assert "gpt-4o-mini" in ids
    assert "o1-mini" in ids
    assert "o3-mini" in ids
    assert "chatgpt-4o-latest" in ids
    assert "gpt-5.6-terra" in ids

    # Filtered categories
    assert "whisper-1" not in ids
    assert "tts-1" not in ids
    assert "tts-1-hd" not in ids
    assert "dall-e-3" not in ids
    assert "text-embedding-3-small" not in ids
    assert "text-moderation-latest" not in ids
    assert "gpt-4o-realtime-preview" not in ids
    assert "gpt-4o-audio-preview" not in ids
    assert "gpt-4o-2024-08-06" not in ids
    assert "o1-mini-2024-09-12" not in ids
    assert "gpt-4-0613" not in ids
    assert "ft:gpt-4o:my-org:custom-001" not in ids


# ============================================================================
# 4. OPENROUTER FILTERING, BATCH DEDUPLICATION & METADATA
# ============================================================================

def test_filter_openrouter_models_removes_batch_and_image_models():
    """OpenRouter filter eliminates :batch endpoints, image-only models, audio models, and preserves exact IDs."""
    raw_models = [
        {
            "id": "openai/gpt-4o",
            "name": "OpenAI: GPT-4o",
            "architecture": {"modality": "text->text", "output_modalities": ["text"]}
        },
        {
            "id": "openai/gpt-4o:batch",
            "name": "OpenAI: GPT-4o (batch)",
            "architecture": {"modality": "text->text", "output_modalities": ["text"]}
        },
        {
            "id": "anthropic/claude-3.5-sonnet",
            "name": "Anthropic: Claude 3.5 Sonnet",
            "architecture": {"modality": "text+image->text", "output_modalities": ["text"]}
        },
        {
            "id": "anthropic/claude-3.5-sonnet:batch",
            "name": "Anthropic: Claude 3.5 Sonnet (batch)",
            "architecture": {"modality": "text+image->text", "output_modalities": ["text"]}
        },
        {
            "id": "black-forest-labs/flux-1-schnell",
            "name": "FLUX 1 Schnell",
            "architecture": {"modality": "text->image", "output_modalities": ["image"]}
        },
        {
            "id": "openai/whisper-large-v3",
            "name": "Whisper Large V3",
            "architecture": {"modality": "audio->text", "output_modalities": ["text"]}
        },
        {
            "id": "cohere/embed-multilingual-v3.0",
            "name": "Cohere: Embed Multilingual",
            "architecture": {"modality": "text->embedding", "output_modalities": ["embedding"]}
        },
    ]
    filtered = filter_openrouter_models(raw_models)
    ids = [m["id"] for m in filtered]
    names = [m["name"] for m in filtered]

    assert "openai/gpt-4o" in ids
    assert "anthropic/claude-3.5-sonnet" in ids

    # Batch variants must be completely excluded
    assert "openai/gpt-4o:batch" not in ids
    assert "anthropic/claude-3.5-sonnet:batch" not in ids
    assert not any("(batch)" in n.lower() for n in names)

    # Incompatible modalities must be excluded
    assert "black-forest-labs/flux-1-schnell" not in ids
    assert "openai/whisper-large-v3" not in ids
    assert "cohere/embed-multilingual-v3.0" not in ids


def test_openrouter_exact_ids_preserved():
    """OpenRouter model IDs retain their full provider/model format for API requests."""
    raw = [
        {"id": "google/gemini-3.7-flash", "name": "Google: Gemini 3.7 Flash"},
        {"id": "deepseek/deepseek-v4-flash", "name": "DeepSeek: DeepSeek V4 Flash"},
    ]
    filtered = filter_openrouter_models(raw)
    assert filtered[0]["id"] == "google/gemini-3.7-flash"
    assert filtered[0]["name"] == "Google: Gemini 3.7 Flash"
    assert filtered[1]["id"] == "deepseek/deepseek-v4-flash"
    assert filtered[1]["name"] == "DeepSeek: DeepSeek V4 Flash"


# ============================================================================
# 5. ANTHROPIC, OLLAMA, AND CUSTOM FILTERING
# ============================================================================

def test_filter_anthropic_models_preserves_display_name_and_exact_id():
    """Anthropic discovery uses display_name for UI and exact ID internally."""
    raw = [
        {"id": "claude-3-7-sonnet-20250219", "display_name": "Claude 3.7 Sonnet"},
        {"id": "claude-3-5-haiku-20241022", "display_name": "Claude 3.5 Haiku"},
    ]
    filtered = filter_anthropic_models(raw)
    assert filtered[0]["id"] == "claude-3-7-sonnet-20250219"
    assert filtered[0]["name"] == "Claude 3.7 Sonnet"
    assert filtered[1]["id"] == "claude-3-5-haiku-20241022"
    assert filtered[1]["name"] == "Claude 3.5 Haiku"


def test_filter_ollama_and_custom_not_overfiltered():
    """Custom and Ollama models allow arbitrary user-defined models."""
    ollama_raw = [
        {"name": "my-local-custom-model:latest"},
        {"name": "llama3.3:70b"},
        {"name": "nomic-embed-text:latest"},  # Should be filtered
    ]
    filtered_ollama = filter_ollama_models(ollama_raw)
    ollama_ids = [m["id"] for m in filtered_ollama]
    assert "my-local-custom-model:latest" in ollama_ids
    assert "llama3.3:70b" in ollama_ids
    assert "nomic-embed-text:latest" not in ollama_ids

    custom_raw = [
        {"id": "custom-vllm-mistral-large"},
        {"id": "fine-tuned-translation-model-v2"},
        {"id": "text-embedding-ada-002"},  # Should be filtered
    ]
    filtered_custom = filter_custom_models(custom_raw)
    custom_ids = [m["id"] for m in filtered_custom]
    assert "custom-vllm-mistral-large" in custom_ids
    assert "fine-tuned-translation-model-v2" in custom_ids
    assert "text-embedding-ada-002" not in custom_ids


# ============================================================================
# 6. DEEPL MODEL STRATEGY MAPPING & BACKWARD COMPATIBILITY
# ============================================================================

def test_deepl_strategy_catalog_and_options(client):
    """DeepL offers Prefer Quality Optimized, Quality Optimized, and Latency Optimized."""
    catalog = get_provider_catalog("deepl")
    ids = [m["id"] for m in catalog]
    names = [m["name"] for m in catalog]

    assert "prefer_quality_optimized" in ids
    assert "quality_optimized" in ids
    assert "latency_optimized" in ids

    assert any("Prefer Quality Optimized" in n for n in names)
    assert any("Quality Optimized" in n for n in names)
    assert any("Latency Optimized" in n for n in names)

    # API endpoint returns DeepL strategies
    resp = client.get("/api/settings/models?provider=deepl")
    assert resp.status_code == 200
    data = resp.json()
    resp_ids = [m["id"] for m in data["models"]]
    assert "prefer_quality_optimized" in resp_ids
    assert "quality_optimized" in resp_ids
    assert "latency_optimized" in resp_ids
    assert data["default_model"] == "prefer_quality_optimized"


def test_deepl_backward_compatibility():
    """Existing saved deepl_model_type setting continues to be resolved correctly."""
    set_setting("deepl_model_type", "quality_optimized")
    ctx = context_from_settings("deepl")
    assert ctx.model == "quality_optimized"

    set_setting("deepl_model_type", "latency_optimized")
    ctx2 = context_from_settings("deepl")
    assert ctx2.model == "latency_optimized"

    set_setting("deepl_model_type", "prefer_quality_optimized")
    ctx3 = context_from_settings("deepl")
    assert ctx3.model == "prefer_quality_optimized"


# ============================================================================
# 7. REFRESH MODELS ENDPOINT
# ============================================================================

def test_refresh_models_bypasses_cache(client):
    """Passing refresh=true forces a new discovery and updates the cache."""
    import time
    from app.api.dashboard import _MODELS_CACHE
    _MODELS_CACHE["openrouter:"] = (
        time.time(),
        [{"id": "stale/model", "name": "Stale Model"}]
    )

    # Without refresh, returns cached data
    resp_cached = client.get("/api/settings/models?provider=openrouter")
    assert resp_cached.json()["cached"] is True
    assert resp_cached.json()["models"][0]["id"] == "stale/model"

    # With refresh=true, bypasses cache and queries live/fallback
    mock_data = {
        "data": [
            {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Meta: Llama 3.3 70B", "architecture": {"output_modalities": ["text"]}}
        ]
    }
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_data
        mock_get.return_value = mock_resp

        resp_fresh = client.get("/api/settings/models?provider=openrouter&refresh=true")
        assert resp_fresh.status_code == 200
        assert resp_fresh.json()["cached"] is False
        assert resp_fresh.json()["models"][0]["id"] == "meta-llama/llama-3.3-70b-instruct"
