"""
Anthropic Structured Output – Payload Contract Tests
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "translations": {
            "type": "ARRAY",
            "items": {"type": "OBJECT",
                      "properties": {"id": {"type": "INTEGER"}, "text": {"type": "STRING"}},
                      "required": ["id", "text"]}
        }
    },
    "required": ["translations"]
}


def _make_mock_response(stop_reason="end_turn", text="{}"):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "stop_reason": stop_reason,
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    return resp


@pytest.mark.asyncio
async def test_anthropic_native_output_config_in_payload_for_supported_model():
    """claude-sonnet-5: output_config.format must be present in HTTP payload when schema given."""
    from app.services.translator import SubtitleTranslator
    translator = SubtitleTranslator()
    captured_payloads = []

    async def fake_post(url, **kwargs):
        captured_payloads.append(kwargs.get("json", {}))
        return _make_mock_response(text='{"translations": [{"id": 1, "text": "Hej"}]}')

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(side_effect=fake_post)

    with patch("app.services.translator.get_setting", side_effect=lambda k, d=None: {
             "anthropic_api_key": "sk-ant-test", "anthropic_model": "claude-sonnet-5"
         }.get(k, d)), \
         patch("httpx.AsyncClient", return_value=mock_client):

        await translator._dispatch_llm_completion(
            provider="anthropic",
            model_name="claude-sonnet-5",
            system_prompt="You are a translator.",
            user_prompt="Translate: Hello",
            schema=SCHEMA,
            temperature=0.1,
        )

    assert len(captured_payloads) == 1
    payload = captured_payloads[0]
    assert "output_config" in payload, (
        f"output_config missing for claude-sonnet-5+schema. Keys: {list(payload.keys())}"
    )
    assert payload["output_config"]["format"]["type"] == "json_schema"
    # The schema must be normalized to real JSON Schema, NOT forwarded verbatim.
    # SCHEMA is authored in Google GenAI's uppercase convention; Anthropic rejects
    # that with 400 "Invalid JSON Schema in output format".
    sent = payload["output_config"]["format"]["schema"]
    assert sent == {
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}, "text": {"type": "string"}},
                    "required": ["id", "text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["translations"],
        "additionalProperties": False,
    }
    assert "temperature" not in payload, "claude-sonnet-5 does not support temperature"
    assert payload["model"] == "claude-sonnet-5"


def _iter_types(node):
    """Yield every 'type' string value in a nested schema."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "type" and isinstance(v, str):
                yield v
            else:
                yield from _iter_types(v)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_types(item)


@pytest.mark.asyncio
async def test_anthropic_payload_never_contains_uppercase_types():
    """Regression: no uppercase type name may survive into the Anthropic payload.

    Anthropic returns 400 for any schema still using the Google GenAI convention.
    """
    from app.services.translator import SubtitleTranslator
    translator = SubtitleTranslator()
    captured_payloads = []

    async def fake_post(url, **kwargs):
        captured_payloads.append(kwargs.get("json", {}))
        return _make_mock_response(text='{"translations": []}')

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(side_effect=fake_post)

    with patch("app.services.translator.get_setting", side_effect=lambda k, d=None: {
             "anthropic_api_key": "sk-ant-test", "anthropic_model": "claude-sonnet-5"
         }.get(k, d)), \
         patch("httpx.AsyncClient", return_value=mock_client):

        await translator._dispatch_llm_completion(
            provider="anthropic",
            model_name="claude-sonnet-5",
            system_prompt="You are a translator.",
            user_prompt="Translate: Hello",
            schema=SCHEMA,
            temperature=0.1,
        )

    sent = captured_payloads[0]["output_config"]["format"]["schema"]
    bad = [t for t in _iter_types(sent) if t != t.lower()]
    assert not bad, f"uppercase JSON types leaked into Anthropic payload: {bad}"


@pytest.mark.asyncio
async def test_anthropic_preserves_optional_fields_in_required():
    """Anthropic must not have optional properties promoted into `required`.

    Anthropic only requires additionalProperties:false. Several audit schemas
    deliberately leave fields such as `details` optional; forcing them required
    would change model behavior.
    """
    from app.services.translator import SubtitleTranslator
    translator = SubtitleTranslator()
    captured_payloads = []

    partial_required = {
        "type": "OBJECT",
        "properties": {
            "verdict": {"type": "STRING"},
            "confidence": {"type": "STRING"},
            "details": {"type": "STRING"},
        },
        "required": ["verdict", "confidence"],
    }

    async def fake_post(url, **kwargs):
        captured_payloads.append(kwargs.get("json", {}))
        return _make_mock_response(text='{"verdict": "ok", "confidence": "high"}')

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(side_effect=fake_post)

    with patch("app.services.translator.get_setting", side_effect=lambda k, d=None: {
             "anthropic_api_key": "sk-ant-test", "anthropic_model": "claude-sonnet-5"
         }.get(k, d)), \
         patch("httpx.AsyncClient", return_value=mock_client):

        await translator._dispatch_llm_completion(
            provider="anthropic",
            model_name="claude-sonnet-5",
            system_prompt="Audit.",
            user_prompt="Audit this.",
            schema=partial_required,
            temperature=None,
        )

    sent = captured_payloads[0]["output_config"]["format"]["schema"]
    assert sent["required"] == ["verdict", "confidence"], (
        f"optional field was promoted to required: {sent['required']}"
    )
    assert sent["additionalProperties"] is False


def test_strict_schema_conversion_variants():
    """The shared converter: lowercase + additionalProperties always; required opt-in."""
    from app.services.translator import SubtitleTranslator
    translator = SubtitleTranslator()

    src = {
        "type": "OBJECT",
        "properties": {"a": {"type": "STRING"}, "b": {"type": "INTEGER"}},
        "required": ["a"],
    }

    # Anthropic flavor: preserve authored `required`
    anthropic = translator._convert_to_strict_json_schema(src)
    assert anthropic["type"] == "object"
    assert anthropic["properties"]["b"]["type"] == "integer"
    assert anthropic["additionalProperties"] is False
    assert anthropic["required"] == ["a"]

    # OpenAI flavor: strict mode promotes every property
    openai = translator._convert_to_openai_json_schema(src)
    assert openai["additionalProperties"] is False
    assert sorted(openai["required"]) == ["a", "b"]

    # Input must not be mutated
    assert src["type"] == "OBJECT"
    assert src["required"] == ["a"]


@pytest.mark.asyncio
async def test_anthropic_fallback_json_prompt_for_unsupported_model():
    """claude-3-opus: no output_config; schema injected into system prompt."""
    from app.services.translator import SubtitleTranslator
    translator = SubtitleTranslator()
    captured_payloads = []

    async def fake_post(url, **kwargs):
        captured_payloads.append(kwargs.get("json", {}))
        return _make_mock_response(text='{"translations": [{"id": 1, "text": "Hej"}]}')

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(side_effect=fake_post)

    with patch("app.services.translator.get_setting", side_effect=lambda k, d=None: {
             "anthropic_api_key": "sk-ant-test", "anthropic_model": "claude-3-opus-20240229"
         }.get(k, d)), \
         patch("httpx.AsyncClient", return_value=mock_client):

        await translator._dispatch_llm_completion(
            provider="anthropic",
            model_name="claude-3-opus-20240229",
            system_prompt="You are a translator.",
            user_prompt="Translate: Hello",
            schema=SCHEMA,
            temperature=0.1,
        )

    assert len(captured_payloads) == 1
    payload = captured_payloads[0]
    assert "output_config" not in payload, "claude-3 must NOT have output_config"
    assert "translations" in payload["system"], (
        "Schema properties must be injected into system prompt for fallback models"
    )
    assert "json" in payload["system"].lower(), "system prompt must mention JSON for fallback"
    # The inlined schema should also be real JSON Schema — describing types as
    # "OBJECT"/"STRING" to the model is inconsistent with the JSON it must emit.
    assert '"OBJECT"' not in payload["system"] and '"STRING"' not in payload["system"], (
        "uppercase Google GenAI types leaked into the fallback system prompt"
    )
    assert '"object"' in payload["system"]


@pytest.mark.asyncio
async def test_anthropic_max_tokens_stop_reason_returns_none():
    """stop_reason=max_tokens → dispatch returns None (truncated response)."""
    from app.services.translator import SubtitleTranslator
    translator = SubtitleTranslator()

    async def fake_post(url, **kwargs):
        return _make_mock_response(stop_reason="max_tokens", text="partial...")

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(side_effect=fake_post)

    with patch("app.services.translator.get_setting", side_effect=lambda k, d=None: {
             "anthropic_api_key": "sk-ant-test", "anthropic_model": "claude-sonnet-5"
         }.get(k, d)), \
         patch("httpx.AsyncClient", return_value=mock_client):

        result = await translator._dispatch_llm_completion(
            provider="anthropic",
            model_name="claude-sonnet-5",
            system_prompt="Translate.",
            user_prompt="Hello",
            schema=None,
            temperature=None,
        )

    assert result is None, f"stop_reason=max_tokens must return None, got: {result!r}"


@pytest.mark.asyncio
async def test_anthropic_no_schema_no_output_config():
    """schema=None → no output_config in payload (passthrough mode)."""
    from app.services.translator import SubtitleTranslator
    translator = SubtitleTranslator()
    captured_payloads = []

    async def fake_post(url, **kwargs):
        captured_payloads.append(kwargs.get("json", {}))
        return _make_mock_response(text="Free text response")

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(side_effect=fake_post)

    with patch("app.services.translator.get_setting", side_effect=lambda k, d=None: {
             "anthropic_api_key": "sk-ant-test", "anthropic_model": "claude-sonnet-5"
         }.get(k, d)), \
         patch("httpx.AsyncClient", return_value=mock_client):

        result = await translator._dispatch_llm_completion(
            provider="anthropic",
            model_name="claude-sonnet-5",
            system_prompt="Translate.",
            user_prompt="Hello",
            schema=None,
            temperature=None,
        )

    payload = captured_payloads[0]
    assert "output_config" not in payload, "No output_config when schema=None"
    assert result == "Free text response"


def test_anthropic_native_output_config_model_capabilities():
    """ModelCapabilities.native_output_config flag matches Anthropic API support matrix."""
    from app.core.ai_providers import get_model_capabilities

    supported = [
        "claude-sonnet-5", "claude-opus-5", "claude-fable-5",
        "claude-haiku-4-5-20251001", "claude-sonnet-4-5", "claude-opus-4-1",
    ]
    for m in supported:
        caps = get_model_capabilities("anthropic", m)
        assert caps.native_output_config is True, f"{m}: expected native_output_config=True"

    unsupported = ["claude-3-opus-20240229", "claude-3-5-sonnet-20241022", "claude-2.1"]
    for m in unsupported:
        caps = get_model_capabilities("anthropic", m)
        assert caps.native_output_config is False, f"{m}: expected native_output_config=False"
