import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
import srt
from datetime import timedelta

from app.services.translator import (
    SubtitleTranslator,
    get_system_instruction,
    validate_batch_translation_results,
    is_safe_keep_prefilter,
    build_translation_prompt,
    build_translation_output_schema,
)
from app.core.ai_providers import (
    get_model_capabilities,
    resolve_job_provider_context,
    context_from_settings,
    ProviderContext,
)
from app.core.db import create_job, set_setting, pin_job_provider


# ============================================================================
# 1. ID CONTRACT & PROMPT/SCHEMA BUILDER TESTS
# ============================================================================

def test_system_instruction_contains_strict_id_contract():
    instruction = get_system_instruction("Swedish", show_title="The Office (US)", source_language="English")
    assert "STRICT 1-TO-1 CUE SYNCHRONIZATION & ID CONTRACT" in instruction
    assert "integer \"id\"" in instruction
    assert "ascending numerical order" in instruction
    assert "NEVER merge two or more cues" in instruction


def test_build_translation_prompt_and_schema():
    items = [
        {"id": 100, "text": "First"},
        {"id": 101, "text": "Second"},
    ]
    prompt = build_translation_prompt(items, "Swedish", context_section="\n\nCtx")
    assert "INPUT CONTRACT: Input items have IDs from 100 to 101 (total 2 items)" in prompt
    assert "First" in prompt and "Second" in prompt

    schema = build_translation_output_schema(items, "Swedish")
    assert schema["type"] == "OBJECT"
    assert "translations" in schema["properties"]
    assert schema["properties"]["translations"]["type"] == "ARRAY"
    item_props = schema["properties"]["translations"]["items"]["properties"]
    assert "id" in item_props and "text" in item_props


def test_convert_to_openai_json_schema():
    translator = SubtitleTranslator()
    openapi_schema = {
        "type": "OBJECT",
        "properties": {
            "translations": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "id": {"type": "INTEGER", "description": "The cue ID"},
                        "text": {"type": "STRING", "description": "The translated text"}
                    },
                    "required": ["id", "text"]
                }
            }
        },
        "required": ["translations"]
    }
    openai_schema = translator._convert_to_openai_json_schema(openapi_schema)
    assert openai_schema["type"] == "object"
    assert openai_schema["additionalProperties"] is False
    assert openai_schema["properties"]["translations"]["type"] == "array"
    item_props = openai_schema["properties"]["translations"]["items"]
    assert item_props["type"] == "object"
    assert item_props["additionalProperties"] is False
    assert item_props["properties"]["id"]["type"] == "integer"
    assert item_props["properties"]["text"]["type"] == "string"
    assert "id" in item_props["required"] and "text" in item_props["required"]


# ============================================================================
# 2. CONTIGUOUS BATCH & SAFE-KEEP TESTS (PROOFS 1, 2, 3)
# ============================================================================

@pytest.mark.asyncio
async def test_safe_keep_items_preserve_contiguous_batch(monkeypatch):
    """PROOF 1: Safe-keep items do not create holes in a partially AI-processed batch."""
    translator = SubtitleTranslator()
    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=1), content="Line 1"),
        srt.Subtitle(index=2, start=timedelta(seconds=1), end=timedelta(seconds=2), content="<i></i>"),
        srt.Subtitle(index=3, start=timedelta(seconds=2), end=timedelta(seconds=3), content="Line 3"),
    ]

    received_batches = []

    async def mock_translate_batch(items, target_language="Swedish", **kwargs):
        received_batches.append(items)
        return [
            {"id": it["id"], "text": "Översatt" if it["text"] != "<i></i>" else "<i></i>"}
            for it in items
        ]

    monkeypatch.setattr(translator, "translate_batch", mock_translate_batch)

    res = await translator.translate_srt_content(subs, target_language="Swedish", batch_size=50)

    assert len(received_batches) == 1
    # Full batch with all 3 items (including <i></i>) was sent in continuous sequence
    assert len(received_batches[0]) == 3
    assert [it["id"] for it in received_batches[0]] == [0, 1, 2]
    assert res[0].content == "Översatt"
    assert res[1].content == "<i></i>"
    assert res[2].content == "Översatt"


@pytest.mark.asyncio
async def test_all_safe_keep_batch_bypasses_ai(monkeypatch):
    """PROOF 2: An all-safe-keep batch completely bypasses AI calls."""
    translator = SubtitleTranslator()
    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=1), content="<i></i>"),
        srt.Subtitle(index=2, start=timedelta(seconds=1), end=timedelta(seconds=2), content="1234"),
        srt.Subtitle(index=3, start=timedelta(seconds=2), end=timedelta(seconds=3), content="---"),
    ]

    called = False

    async def mock_translate_batch(items, **kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(translator, "translate_batch", mock_translate_batch)

    res = await translator.translate_srt_content(subs, target_language="Swedish", batch_size=50)

    assert not called, "AI translation must NOT be called for an all-safe-keep batch"
    assert res[0].content == "<i></i>"
    assert res[1].content == "1234"
    assert res[2].content == "---"


@pytest.mark.asyncio
async def test_gemini_primary_contiguous_ids_and_schema(monkeypatch):
    """PROOF 3 & 10: Gemini primary translation enforces contiguous IDs and native schema."""
    translator = SubtitleTranslator()
    items = [
        {"id": 450, "text": "Hello"},
        {"id": 451, "text": "World"},
        {"id": 452, "text": "<i></i>"},
    ]

    captured_prompt = None
    captured_config = None

    class MockModels:
        def generate_content(self, model, contents, config):
            nonlocal captured_prompt, captured_config
            captured_prompt = contents
            captured_config = config
            mock_resp = MagicMock()
            mock_resp.text = json.dumps({
                "translations": [
                    {"id": 450, "text": "Hej"},
                    {"id": 451, "text": "Värld"},
                    {"id": 452, "text": "<i></i>"}
                ]
            })
            mock_resp.usage_metadata = None
            return mock_resp

    class MockClient:
        def __init__(self):
            self.models = MockModels()

    monkeypatch.setattr(translator, "get_gemini_client", lambda: MockClient())

    res = await translator.translate_batch_gemini(
        items, target_language="Swedish", model_name="gemini-3.5-flash-lite", source_language="English"
    )

    assert len(res) == 3
    assert res[0]["id"] == 450
    assert res[0]["text"] == "Hej"
    assert res[2]["id"] == 452

    assert captured_prompt is not None
    assert "INPUT CONTRACT: Input items have IDs from 450 to 452 (total 3 items)" in captured_prompt
    assert "strictly preserving the exact matching integer \"id\"" in captured_prompt
    assert captured_config is not None
    assert captured_config.max_output_tokens == 8192
    assert captured_config.response_mime_type == "application/json"
    assert captured_config.response_schema is not None


# ============================================================================
# 3. EXACT-ID VALIDATION TESTS (PROOFS 4, 5, 6)
# ============================================================================

def test_exact_id_validator_rejects_missing_id():
    """PROOF 4: Exact-ID validation rejects missing IDs."""
    expected = [{"id": 0, "text": "Hello"}, {"id": 1, "text": "World"}]
    raw_results = [{"id": 0, "text": "Hej"}]  # Missing ID 1

    valid_map, report = validate_batch_translation_results(expected, raw_results)
    assert report["is_clean"] is False
    assert report["missing_ids"] == [1]
    assert 0 in valid_map
    assert 1 not in valid_map


def test_exact_id_validator_rejects_duplicate_id():
    """PROOF 5: Exact-ID validation rejects duplicate IDs."""
    expected = [{"id": 0, "text": "Hello"}, {"id": 1, "text": "World"}]
    raw_results = [
        {"id": 0, "text": "Hej"},
        {"id": 0, "text": "Hej igen"},  # Duplicate ID 0
        {"id": 1, "text": "Värld"},
    ]

    valid_map, report = validate_batch_translation_results(expected, raw_results)
    assert report["is_clean"] is False
    assert report["duplicate_ids"] == [0]


def test_exact_id_validator_rejects_extra_unknown_id():
    """PROOF 6: Exact-ID validation rejects unknown/extra IDs."""
    expected = [{"id": 0, "text": "Hello"}]
    raw_results = [
        {"id": 0, "text": "Hej"},
        {"id": 99, "text": "Extra"},  # Unknown ID 99
    ]

    valid_map, report = validate_batch_translation_results(expected, raw_results)
    assert report["is_clean"] is False
    assert report["unknown_ids"] == [99]
    assert 99 not in valid_map


# ============================================================================
# 4. PROVIDER STRUCTURED OUTPUT TESTS (PROOFS 7, 8, 9, 11, 12, 13, 14, 15, 16)
# ============================================================================

@pytest.mark.asyncio
async def test_openai_strict_capable_model_uses_strict_json_schema():
    """PROOF 7: OpenAI strict-capable models receive strict JSON Schema."""
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

    # 1. Primary batch
    with patch.object(translator, "get_openai_client", return_value=mock_client):
        await translator.translate_batch_openai(
            items=[{"id": 0, "text": "hello"}],
            target_language="Swedish",
            model_name="gpt-4o",
        )
        assert len(captured_kwargs) == 1
        rf = captured_kwargs[0]["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["strict"] is True
        assert rf["json_schema"]["name"] == "translation_response"
        assert rf["json_schema"]["schema"]["type"] == "object"
        assert rf["json_schema"]["schema"]["additionalProperties"] is False

    # 2. Generic dispatch
    with patch("openai.OpenAI", return_value=mock_client):
        await translator._dispatch_llm_completion(
            provider="openai",
            model_name="gpt-4o",
            system_prompt="sys",
            user_prompt="user",
            schema={"type": "OBJECT", "properties": {"res": {"type": "STRING"}}, "required": ["res"]},
        )
        assert len(captured_kwargs) == 2
        rf2 = captured_kwargs[1]["response_format"]
        assert rf2["type"] == "json_schema"
        assert rf2["json_schema"]["strict"] is True
        assert rf2["json_schema"]["name"] == "structured_output"
        assert rf2["json_schema"]["schema"]["type"] == "object"
        assert rf2["json_schema"]["schema"]["additionalProperties"] is False


@pytest.mark.asyncio
async def test_json_only_model_uses_json_object():
    """PROOF 8: JSON-only models (e.g. DeepSeek or OpenRouter with json_object) use json_object."""
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
            provider="deepseek",
            model_name="deepseek-v4-flash",
            system_prompt="sys",
            user_prompt="user",
            schema={"type": "OBJECT"},
        )
        assert len(captured_kwargs) == 1
        assert captured_kwargs[0]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_unknown_openai_model_has_no_speculative_response_format():
    """PROOF 9: Unknown future OpenAI models receive NO speculative response_format."""
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
            model_name="gpt-unknown-999",
            system_prompt="sys",
            user_prompt="user",
            schema={"type": "OBJECT"},
        )
        assert len(captured_kwargs) == 1
        assert "response_format" not in captured_kwargs[0]


@pytest.mark.asyncio
async def test_anthropic_native_and_fallback():
    """PROOF 11: Anthropic uses native output_config on modern models and prompt fallback on legacy models."""
    translator = SubtitleTranslator()

    # 1. Native output_config for Claude 5
    caps_5 = get_model_capabilities("anthropic", "claude-sonnet-5")
    assert caps_5.native_output_config is True
    assert caps_5.structured_output is True

    # 2. Legacy fallback for Claude 3.5
    caps_35 = get_model_capabilities("anthropic", "claude-3-5-sonnet-latest")
    assert caps_35.native_output_config is False


@pytest.mark.asyncio
async def test_custom_openai_is_conservative():
    """PROOF 12: Custom OpenAI provider remains conservative with no optional fields."""
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
            model_name="my-local-model",
            system_prompt="sys",
            user_prompt="user",
            schema={"type": "OBJECT"},
        )
        assert len(captured_kwargs) == 1
        assert "response_format" not in captured_kwargs[0]
        assert "temperature" not in captured_kwargs[0]


@pytest.mark.asyncio
async def test_deepl_remains_unaffected():
    """PROOF 13: DeepL translation path is completely untouched."""
    translator = SubtitleTranslator()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"translations": [{"text": "Hej"}]}

    set_setting("deepl_api_key", "test-key")
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        res = await translator.translate_batch_deepl(
            items=[{"id": 0, "text": "Hello"}],
            target_language="Swedish",
        )
        assert len(res) == 1
        assert res[0]["text"] == "Hej"
    set_setting("deepl_api_key", "")


@pytest.mark.asyncio
async def test_no_fallback_resends_on_clean_batch(monkeypatch):
    """PROOF 15: A clean translation batch never triggers retries or extra paid API calls."""
    translator = SubtitleTranslator()
    call_count = 0

    async def mock_translate_batch(items, **kwargs):
        nonlocal call_count
        call_count += 1
        return [{"id": it["id"], "text": f"Översatt {it['id']}"} for it in items]

    monkeypatch.setattr(translator, "translate_batch", mock_translate_batch)

    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=1), content="Hello"),
        srt.Subtitle(index=2, start=timedelta(seconds=1), end=timedelta(seconds=2), content="World"),
    ]

    res = await translator.translate_srt_content(subs, target_language="Swedish", batch_size=50)
    assert len(res) == 2
    assert call_count == 1, "Exactly 1 call should be made for a clean batch (0 retries)"


def test_job_pinning_governs_provider_model_capabilities():
    """PROOF 16: Pinned job provider and model context governs capabilities."""
    job_id = create_job("test_pinning_video_2.mkv")
    pin_job_provider(job_id, "openai", "gpt-4o")

    # Global setting points to different provider
    set_setting("ai_provider", "deepseek")
    set_setting("deepseek_model", "deepseek-v4-flash")

    ctx = resolve_job_provider_context(job_id)
    assert ctx.provider == "openai"
    assert ctx.model == "gpt-4o"

    caps = get_model_capabilities(ctx.provider, ctx.model)
    assert caps.structured_output is True
    assert caps.temperature is True


@pytest.mark.asyncio
async def test_ollama_and_localai_capability_separation():
    """PROOF: Ollama retains verified JSON mode, while LocalAI does NOT inherit capabilities."""
    # 1. Capability resolution separation
    ollama_caps = get_model_capabilities("ollama", "llama3")
    assert ollama_caps.json_object is True
    assert ollama_caps.temperature is True
    assert ollama_caps.structured_output is False

    localai_caps = get_model_capabilities("localai", "llama3")
    assert localai_caps.json_object is False
    assert localai_caps.temperature is False
    assert localai_caps.structured_output is False

    # 2. Dispatch payload separation
    translator = SubtitleTranslator()
    captured_payloads = []
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"response": '{"translations": []}'}

    async def mock_post(url, json=None, **kwargs):
        captured_payloads.append(json)
        return mock_resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        # Ollama sends format="json" and options={"temperature": 0.1}
        await translator._dispatch_llm_completion(
            provider="ollama",
            model_name="llama3",
            system_prompt="sys",
            user_prompt="user",
            schema={"type": "object"},
            temperature=0.1,
        )
        assert len(captured_payloads) == 1
        p_ollama = captured_payloads[0]
        assert p_ollama["format"] == "json"
        assert p_ollama["options"] == {"temperature": 0.1}

        # LocalAI reuses transport but sends NO speculative format or options
        await translator._dispatch_llm_completion(
            provider="localai",
            model_name="llama3",
            system_prompt="sys",
            user_prompt="user",
            schema={"type": "object"},
            temperature=0.1,
        )
        assert len(captured_payloads) == 2
        p_localai = captured_payloads[1]
        assert "format" not in p_localai
        assert "options" not in p_localai

    # 3. Batch translation separation
    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        # translate_batch with localai provider
        await translator.translate_batch(
            items=[{"id": 0, "text": "Hello"}],
            target_language="Swedish",
            provider_ctx=ProviderContext(provider="localai", model="llama3"),
        )
        assert len(captured_payloads) == 3
        p_batch_localai = captured_payloads[2]
        assert "format" not in p_batch_localai
        assert "options" not in p_batch_localai
