import pytest
import os
import srt
import json
from datetime import timedelta
from unittest.mock import patch, MagicMock

import app.services.translator
from app.services.translator import (
    SubtitleTranslator,
    is_meaningful_translation,
    is_usable_translation
)
from app.services.pipeline import SubtitlePipeline
from app.core.db import get_job_by_id
import google.genai


def test_is_meaningful_translation_unit():
    """Test 1: Unit tests for central recovery acceptance helper."""
    # Normalized echoes must be rejected (False)
    assert not is_meaningful_translation("Hello?", "Hello!")
    assert not is_meaningful_translation("<i>Hello</i>", "hello.")
    assert not is_meaningful_translation("HELLO", "hello")
    assert not is_meaningful_translation("Hello...", "<i>Hello!</i>")
    assert not is_meaningful_translation("What?", "What!")
    assert not is_meaningful_translation("Come on, man.", "come on man")
    assert not is_meaningful_translation("Good morning.", "good morning!")

    # Meaningful translations must be accepted (True)
    assert is_meaningful_translation("Hello", "Hej")
    assert is_meaningful_translation("What?", "Vad?")
    assert is_meaningful_translation("Come on, man.", "Kom igen.")
    assert is_meaningful_translation("Good morning.", "God morgon.")

    # Blank / unusable candidate translations must fail-closed (False)
    assert not is_meaningful_translation("Hello", "")
    assert not is_meaningful_translation("Hello", None)
    assert not is_meaningful_translation("Hello", "   ")
    assert not is_meaningful_translation("Hello", "<i></i>")


@pytest.mark.asyncio
async def test_escalation_normalized_echo(monkeypatch):
    """Test 2: Escalation rejects normalized echo in attempt 1, accepts valid in attempt 2."""
    translator = SubtitleTranslator()

    monkeypatch.setattr(app.services.translator, "get_setting", lambda k, d="": "gemini" if k == "ai_provider" else "false")

    call_count = 0

    class MockResponse:
        def __init__(self, text):
            self.text = text

    def mock_generate_content(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MockResponse('{"translation": "Hello!"}')  # Normalized echo of Hello?
        else:
            return MockResponse('{"translation": "Hej!"}')   # Valid translation

    class MockModels:
        def generate_content(self, *args, **kwargs):
            return mock_generate_content(*args, **kwargs)

    class MockClient:
        def __init__(self, *args, **kwargs):
            self.models = MockModels()

    monkeypatch.setattr(google.genai, "Client", MockClient)

    exhausted = set()
    result = await translator.escalate_single_line(
        target_idx=100,
        target_text="Hello?",
        prev_text="Hi",
        next_text="Hey",
        target_language="Swedish",
        show_title="Test",
        is_real_untranslated=True,
        job_id=1,
        exhausted_strategies=exhausted
    )

    assert call_count == 2
    assert result == "Hej!"
    # The first attempt was exhausted
    assert len(exhausted) == 1


@pytest.mark.asyncio
async def test_escalation_all_normalized_echoes_exhaust(monkeypatch):
    """Test 3: Escalation exhausts all attempts when all return normalized echoes."""
    translator = SubtitleTranslator()

    monkeypatch.setattr(app.services.translator, "get_setting", lambda k, d="": "gemini" if k == "ai_provider" else "false")

    call_count = 0

    class MockResponse:
        def __init__(self, text):
            self.text = text

    def mock_generate_content(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MockResponse('{"translation": "Hello!"}')
        elif call_count == 2:
            return MockResponse('{"translation": "<i>Hello.</i>"}')
        else:
            return MockResponse('{"translation": "HELLO?"}')

    class MockModels:
        def generate_content(self, *args, **kwargs):
            return mock_generate_content(*args, **kwargs)

    class MockClient:
        def __init__(self, *args, **kwargs):
            self.models = MockModels()

    monkeypatch.setattr(google.genai, "Client", MockClient)

    exhausted = set()
    result = await translator.escalate_single_line(
        target_idx=100,
        target_text="Hello?",
        prev_text="Hi",
        next_text="Hey",
        target_language="Swedish",
        show_title="Test",
        is_real_untranslated=True,
        job_id=1,
        exhausted_strategies=exhausted
    )

    assert call_count == 3
    assert result is None
    # All 3 strategies must be recorded as exhausted
    assert len(exhausted) == 3


@pytest.mark.asyncio
async def test_targeted_recovery_rejects_normalized_echo(tmp_path, monkeypatch):
    """Test 4: Targeted recovery does not count false success for normalized echoes."""
    def mock_get_setting(key, default=""):
        settings = {
            "ai_provider": "gemini",
            "gemini_api_key": "dummy_api_key",
            "gemini_model": "gemini-3.5-flash-lite",
            "escalate_to_pro": "true",
            "escalation_provider": "gemini",
            "escalation_model": "gemini-3.5-flash-lite",
            "batch_size": "50",
            "max_concurrent_jobs": "1",
            "wait_time_seconds": "0",
            "extract_target_embedded": "false",
            "extract_source_embedded": "false",
            "languages": json.dumps([{"code": "sv", "name": "Swedish", "enabled": True}]),
        }
        return settings.get(key, default)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)
    monkeypatch.setattr("app.services.translator.get_setting", mock_get_setting)

    video_path = tmp_path / "TestShow.S01E01.mkv"
    video_path.touch()
    en_srt = tmp_path / "TestShow.S01E01.en.srt"

    subs = []
    for i in range(25):
        subs.append(srt.Subtitle(index=i+1, start=timedelta(seconds=i), end=timedelta(seconds=i+1), content=f"Dialogue line {i}"))
    # Add stubborn cue as line 26 (index 25)
    subs.append(srt.Subtitle(index=26, start=timedelta(seconds=25), end=timedelta(seconds=26), content="Hello?"))

    with open(en_srt, "w") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()

    translate_calls = 0
    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        nonlocal translate_calls
        translate_calls += 1
        results = []
        for item in items:
            idx = item["id"]
            if idx == 25:
                if translate_calls == 1:
                    # Initial pass returns original text
                    results.append({"id": idx, "text": "Hello?"})
                else:
                    # Targeted recovery returns normalized echo
                    results.append({"id": idx, "text": "Hello!"})
            else:
                results.append({"id": idx, "text": f"Svensk rad {idx}"})
        return results

    async def mock_classify(items, lang, title):
        # Classify all as translate
        return [{"id": item["id"], "action": "translate", "reason": "none", "text": ""} for item in items]

    # Escalation eventually recovers with real Swedish
    async def mock_escalate_single_line(self, target_idx, target_text, prev_text, next_text, target_language, show_title, is_real_untranslated=False, job_id=None, exhausted_strategies=None, **kwargs):
        return "Hej?"

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(SubtitleTranslator, "escalate_single_line", mock_escalate_single_line)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
    assert res["status"] != "failed"

    job = get_job_by_id(res["job_id"])
    assert job["status"] == "TRANSLATED"

    # Verify that Targeted Recovery logged translated 0/1
    combined_logs = "\n".join(job["logs"])
    assert "Targeted Recovery: translated 0/1" in combined_logs
    assert "Escalation: Translated line 25 using dialogue context" in combined_logs


@pytest.mark.asyncio
async def test_pipeline_escalation_defense_in_depth(tmp_path, monkeypatch):
    """Test 5: Pipeline escalation wrapper rejects normalized echoes even if returned by translator."""
    def mock_get_setting(key, default=""):
        settings = {
            "ai_provider": "gemini",
            "gemini_api_key": "dummy_api_key",
            "gemini_model": "gemini-3.5-flash-lite",
            "escalate_to_pro": "true",
            "escalation_provider": "gemini",
            "escalation_model": "gemini-3.5-flash-lite",
            "batch_size": "50",
            "max_concurrent_jobs": "1",
            "wait_time_seconds": "0",
            "extract_target_embedded": "false",
            "extract_source_embedded": "false",
            "languages": json.dumps([{"code": "sv", "name": "Swedish", "enabled": True}]),
        }
        return settings.get(key, default)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)
    monkeypatch.setattr("app.services.translator.get_setting", mock_get_setting)

    video_path = tmp_path / "TestShow.S01E02.mkv"
    video_path.touch()
    en_srt = tmp_path / "TestShow.S01E02.en.srt"

    subs = []
    for i in range(25):
        subs.append(srt.Subtitle(index=i+1, start=timedelta(seconds=i), end=timedelta(seconds=i+1), content=f"Dialogue line {i}"))
    subs.append(srt.Subtitle(index=26, start=timedelta(seconds=25), end=timedelta(seconds=26), content="Hello?"))

    with open(en_srt, "w") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()

    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        results = []
        for item in items:
            idx = item["id"]
            if idx == 25:
                results.append({"id": idx, "text": "Hello?"})
            else:
                results.append({"id": idx, "text": f"Svensk rad {idx}"})
        return results

    async def mock_classify(items, lang, title):
        return [{"id": item["id"], "action": "translate", "reason": "none", "text": ""} for item in items]

    # Mock escalation to improperly return normalized echo
    async def mock_escalate_single_line(self, target_idx, target_text, prev_text, next_text, target_language, show_title, is_real_untranslated=False, job_id=None, exhausted_strategies=None, **kwargs):
        return "Hello!"

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(SubtitleTranslator, "escalate_single_line", mock_escalate_single_line)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")

    job = get_job_by_id(res["job_id"])
    # The pipeline must fail closed because escalation's returned echo was rejected
    assert job["status"] == "FAILED" or job["status"] == "RECOVERING"
    combined_logs = "\n".join(job["logs"])
    assert "Escalation: Rejected identical fallback for line 25" in combined_logs
    # Final sv.srt must NOT exist
    sv_srt = tmp_path / "TestShow.S01E02.sv.srt"
    assert not sv_srt.exists()
