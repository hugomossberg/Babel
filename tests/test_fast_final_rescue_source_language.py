"""
Regression tests for Fast Final Rescue source_language propagation.

Scenarios from v2.3.43 hardening prompt:
  I. Fast Final Rescue: en -> sv — no NameError
  J. Fast Final Rescue: fr -> sv — correct source language used in prompt
  K. Fast Final Rescue: de -> en — correct source language used in prompt

Also tests:
  - first_pass_micro_repair_batch receives source_language (not hardcoded English)
  - qa_gate source_language_name parameter shows correct language in messages
"""
import pytest
import asyncio
import json
from datetime import timedelta
from unittest.mock import patch, AsyncMock, MagicMock

import srt

from app.services.translator import SubtitleTranslator
from app.services.pipeline import qa_gate


def make_srt_subs(texts, start_s=0):
    return [
        srt.Subtitle(
            index=i + 1,
            start=timedelta(seconds=start_s + i),
            end=timedelta(seconds=start_s + i + 1),
            content=t,
        )
        for i, t in enumerate(texts)
    ]


MOCK_SETTINGS = {
    "ai_provider": "gemini",
    "gemini_api_key": "dummy_key",
    "gemini_model": "gemini-3.5-flash-lite",
    "escalate_to_pro": "false",
    "escalation_provider": "none",
    "escalation_model": "",
    "batch_size": "50",
}


def mock_get_setting(key, default=""):
    return MOCK_SETTINGS.get(key, default)


def make_gemini_mock(response_text):
    """Return a (mock_client, captured_prompts) tuple. captured_prompts is filled on each call."""
    captured = {"system_prompts": []}

    def generate_content(model, contents, config):
        captured["system_prompts"].append(config.system_instruction)
        resp = MagicMock()
        resp.text = response_text
        resp.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=10)
        return resp

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = generate_content
    return mock_client, captured


# ─── I. en->sv — no NameError ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_I_fast_final_rescue_en_sv_no_name_error():
    """I. Fast Final Rescue en->sv must not raise NameError for source_language."""
    translator = SubtitleTranslator()
    rescue_items = [{"id": 0, "target": "Hello, how are you today?",
                     "context_before": "(none)", "context_after": "Good morning."}]

    mock_client, captured = make_gemini_mock('{"results": [{"id": 0, "text": "Hej, hur mar du idag?"}]}')

    from app.core.ai_providers import ProviderContext
    _gemini_ctx = ProviderContext(provider="gemini", model="gemini-3.5-flash-lite")
    with patch("app.services.translator.get_setting", side_effect=mock_get_setting), \
         patch("app.services.translator.append_job_log"), \
         patch("app.core.ai_providers.context_from_settings", return_value=_gemini_ctx):
        import google.genai as genai_mod
        with patch.object(genai_mod, "Client", return_value=mock_client):
            # MUST NOT raise NameError
            results = await translator.fast_final_rescue_batch(
                rescue_items,
                target_language="Swedish",
                source_language="English",
                show_title="Test Show",
                attempt=1,
                job_id=None,
            )

    assert isinstance(results, list), "fast_final_rescue_batch must return a list"
    assert len(captured["system_prompts"]) >= 1
    prompt = captured["system_prompts"][0]
    assert "English" in prompt, f"Prompt must contain 'English' source lang. Got: {prompt[:200]}"
    assert "Swedish" in prompt, f"Prompt must contain 'Swedish' target lang. Got: {prompt[:200]}"


# ─── J. fr->sv — correct source language in prompt ────────────────────────

@pytest.mark.asyncio
async def test_J_fast_final_rescue_fr_sv_correct_source_language():
    """J. Fast Final Rescue fr->sv must say 'French' in prompt, not hardcode 'English'."""
    translator = SubtitleTranslator()
    rescue_items = [{"id": 0, "target": "Bonjour, comment vas-tu?",
                     "context_before": "(none)", "context_after": "(none)"}]

    mock_client, captured = make_gemini_mock('{"results": [{"id": 0, "text": "Hej, hur mar du?"}]}')

    from app.core.ai_providers import ProviderContext
    _gemini_ctx = ProviderContext(provider="gemini", model="gemini-3.5-flash-lite")
    with patch("app.services.translator.get_setting", side_effect=mock_get_setting), \
         patch("app.services.translator.append_job_log"), \
         patch("app.core.ai_providers.context_from_settings", return_value=_gemini_ctx):
        import google.genai as genai_mod
        with patch.object(genai_mod, "Client", return_value=mock_client):
            await translator.fast_final_rescue_batch(
                rescue_items,
                target_language="Swedish",
                source_language="French",
                attempt=1,
                job_id=None,
            )

    assert len(captured["system_prompts"]) >= 1
    prompt = captured["system_prompts"][0]
    assert "French" in prompt, \
        f"Prompt must reference 'French' as source language. Got: {prompt[:300]}"
    assert "from the English source" not in prompt, \
        f"Prompt must NOT hardcode 'English source' for fr->sv. Got: {prompt[:300]}"


# ─── K. de->en — correct source language ──────────────────────────────────

@pytest.mark.asyncio
async def test_K_fast_final_rescue_de_en_correct_source_language():
    """K. Fast Final Rescue de->en must say 'German' as source, 'English' as target."""
    translator = SubtitleTranslator()
    rescue_items = [{"id": 0, "target": "Guten Morgen, wie geht es dir?",
                     "context_before": "(none)", "context_after": "(none)"}]

    mock_client, captured = make_gemini_mock('{"results": [{"id": 0, "text": "Good morning, how are you?"}]}')

    from app.core.ai_providers import ProviderContext
    _gemini_ctx = ProviderContext(provider="gemini", model="gemini-3.5-flash-lite")
    with patch("app.services.translator.get_setting", side_effect=mock_get_setting), \
         patch("app.services.translator.append_job_log"), \
         patch("app.core.ai_providers.context_from_settings", return_value=_gemini_ctx):
        import google.genai as genai_mod
        with patch.object(genai_mod, "Client", return_value=mock_client):
            await translator.fast_final_rescue_batch(
                rescue_items,
                target_language="English",
                source_language="German",
                attempt=1,
                job_id=None,
            )

    assert len(captured["system_prompts"]) >= 1
    prompt = captured["system_prompts"][0]
    assert "German" in prompt, \
        f"Prompt must reference 'German' as source language. Got: {prompt[:300]}"
    assert "English" in prompt, \
        f"Prompt must reference 'English' as target language. Got: {prompt[:300]}"


# ─── Attempt 2 also uses dynamic source language ──────────────────────────

@pytest.mark.asyncio
async def test_rescue_attempt2_source_language_dynamic_fr():
    """Attempt 2 prompt must not hardcode 'English' when source_language='French'."""
    translator = SubtitleTranslator()
    rescue_items = [{"id": 0, "target": "Bonjour", "context_before": "(none)", "context_after": "(none)"}]

    mock_client, captured = make_gemini_mock('{"results": [{"id": 0, "text": "Hej"}]}')

    from app.core.ai_providers import ProviderContext
    _gemini_ctx = ProviderContext(provider="gemini", model="gemini-3.5-flash-lite")
    with patch("app.services.translator.get_setting", side_effect=mock_get_setting), \
         patch("app.services.translator.append_job_log"), \
         patch("app.core.ai_providers.context_from_settings", return_value=_gemini_ctx):
        import google.genai as genai_mod
        with patch.object(genai_mod, "Client", return_value=mock_client):
            await translator.fast_final_rescue_batch(
                rescue_items,
                target_language="Swedish",
                source_language="French",
                attempt=2,
                job_id=None,
            )

    assert len(captured["system_prompts"]) >= 1
    prompt = captured["system_prompts"][0]
    assert "French" in prompt, \
        f"Attempt 2 prompt must contain 'French'. Got: {prompt[:300]}"
    # The old hardcoded string should not appear
    assert "from the English source" not in prompt, \
        f"Attempt 2 prompt must not hardcode 'English source' for fr->sv. Got: {prompt[:300]}"


# ─── first_pass_micro_repair_batch uses source_language ──────────────────

@pytest.mark.asyncio
async def test_micro_repair_source_language_fr():
    """first_pass_micro_repair_batch must use source_language='French' in its prompt."""
    translator = SubtitleTranslator()
    repair_items = [{"id": 0, "target": "Bonjour", "context_before": "(none)", "context_after": "(none)"}]

    mock_client, captured = make_gemini_mock('{"results": [{"id": 0, "text": "Hej"}]}')

    from app.core.ai_providers import ProviderContext
    _gemini_ctx = ProviderContext(provider="gemini", model="gemini-3.5-flash-lite")
    with patch("app.services.translator.get_setting", side_effect=mock_get_setting), \
         patch("app.services.translator.append_job_log"), \
         patch("app.core.ai_providers.context_from_settings", return_value=_gemini_ctx):
        import google.genai as genai_mod
        with patch.object(genai_mod, "Client", return_value=mock_client):
            await translator.first_pass_micro_repair_batch(
                repair_items,
                target_language="Swedish",
                source_language="French",
                show_title="",
                job_id=None,
            )

    assert len(captured["system_prompts"]) >= 1
    prompt = captured["system_prompts"][0]
    assert "French" in prompt, \
        f"Micro repair prompt must use 'French' as source lang. Got: {prompt[:300]}"
    assert "translating English dialogue" not in prompt, \
        f"Micro repair must not hardcode 'English dialogue'. Got: {prompt[:300]}"


# ─── qa_gate source_language_name in messages ─────────────────────────────

def test_qa_gate_source_language_name_french_in_warnings():
    """qa_gate PASS_WITH_WARNINGS message must say 'French', not 'English'."""
    # One line remains as French (unresolved)
    src_texts = [
        "Bonjour tout le monde.",
        "Comment allez-vous aujourd hui?",
        "Au revoir.",
    ]
    tgt_texts = [
        "Hej allihopa.",
        "Comment allez-vous aujourd hui?",  # unresolved — same as source
        "Hej da.",
    ]
    source_subs = make_srt_subs(src_texts)
    translated_subs = make_srt_subs(tgt_texts)

    result = qa_gate(
        source_subs,
        translated_subs,
        target_lang_code="sv",
        source_language_name="French",
        allow_warnings=True,
        max_unresolved_count=2,
        max_unresolved_ratio=0.5,
    )

    all_messages = result.get("warnings", []) + result.get("issues", [])
    french_in_msg = any("French" in m for m in all_messages)
    english_in_msg = any("English" in m for m in all_messages)

    assert french_in_msg, \
        f"qa_gate must mention 'French' as source language. Got: {all_messages}"
    assert not english_in_msg, \
        f"qa_gate must NOT say 'English' when source_language_name='French'. Got: {all_messages}"


def test_qa_gate_default_source_language_name_no_english_hardcode():
    """qa_gate default (source_language_name='source') must not hardcode 'English'."""
    src_texts = ["Hello there, how are you?", "Goodbye my friend."]
    tgt_texts = ["Hello there, how are you?", "Goodbye my friend."]

    source_subs = make_srt_subs(src_texts)
    translated_subs = make_srt_subs(tgt_texts)

    result = qa_gate(
        source_subs,
        translated_subs,
        target_lang_code="sv",
        # No source_language_name → default "source"
        allow_warnings=True,
        max_unresolved_count=5,
        max_unresolved_ratio=1.0,
    )

    all_messages = result.get("warnings", []) + result.get("issues", [])
    english_hardcoded = any(
        "unresolved English" in m or "original English" in m for m in all_messages
    )
    assert not english_hardcoded, \
        f"Default qa_gate must not hardcode 'English' in messages. Got: {all_messages}"
