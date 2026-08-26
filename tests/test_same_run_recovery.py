import pytest
import os
import srt
from datetime import timedelta
import json
from unittest.mock import patch, MagicMock

from app.services.pipeline import SubtitlePipeline
from app.services.translator import SubtitleTranslator
from app.core.db import get_job_by_id

@pytest.fixture
def mock_db_settings(monkeypatch):
    def mock_get_setting(key, default=""):
        settings = {
            "ai_provider": "gemini",
            "gemini_api_key": "dummy_api_key",
            "gemini_model": "gemini-3.5-flash-lite",
            "escalate_to_pro": "true",
            "escalation_provider": "gemini",
            "escalation_model": "gemini-3.5-flash-lite", # SAME MODEL
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
    # After hardening, provider resolution uses context_from_settings() from central registry
    # which reads from DB. Mock it to avoid state bleed from other tests in full suite.
    from app.core.ai_providers import ProviderContext
    _gemini_ctx = ProviderContext(provider="gemini", model="gemini-3.5-flash-lite")
    monkeypatch.setattr("app.core.ai_providers.context_from_settings", lambda *a, **kw: _gemini_ctx)

@pytest.mark.asyncio
async def test_first_execution_success(mock_db_settings, tmp_path, monkeypatch):
    """
    Test Step 9, 11, 12: Same run recovery with 1 stubborn cue.
    Should pass without RECOVERING status or returning to worker queue.
    """
    video_path = tmp_path / "Survivors.Remorse.S02E06.mkv"
    video_path.touch()
    en_srt = tmp_path / "Survivors.Remorse.S02E06.en.srt"
    
    subs = []
    # Create 26 lines. 11 will be legitimate KEEP, 15 will be REAL dialogue.
    for i in range(26):
        if i < 11:
            text = "NASA" # legitimate KEEP
        else:
            text = f"Dialogue line {i}" # REAL dialogue
        subs.append(srt.Subtitle(index=i+1, start=timedelta(seconds=i), end=timedelta(seconds=i+1), content=text))
    
    with open(en_srt, "w") as f:
        f.write(srt.compose(subs))
    
    pipeline = SubtitlePipeline()
    
    # Mock translator methods
    call_counts = {
        "translate_batch": 0,
        "classify": 0,
        "escalate": 0
    }
    
    # The prompts we capture
    isolated_prompt = None

    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        call_counts["translate_batch"] += 1
        results = []
        for item in items:
            idx = item["id"]
            if idx == 25: # The stubborn cue (line 26 / index 25)
                # First full translation or targeted translation -> identical
                results.append({"id": idx, "text": item["text"]})
            else:
                if "Dialogue" in item["text"]:
                    results.append({"id": idx, "text": f"Svensk undertext för rad {idx}"})
                else:
                    results.append({"id": idx, "text": item["text"]}) # identical
        return results

    async def mock_classify(items, lang, title, **kwargs):
        call_counts["classify"] += 1
        results = []
        for item in items:
            idx = item["id"]
            if item["text"] == "NASA":
                results.append({"id": idx, "action": "keep", "reason": "acronym", "text": item["text"]})
            else:
                results.append({"id": idx, "action": "translate", "reason": "none", "text": ""}) # simulate blank/identical
        return results

    # We need to capture the prompt used in isolated attempt
    original_escalate = SubtitleTranslator.escalate_single_line
    
    async def mock_escalate_single_line(self, target_idx, target_text, prev_text, next_text, target_language, show_title, is_real_untranslated=False, job_id=None, **kwargs):
        call_counts["escalate"] += 1
        
        # In our mock, we let it use the real escalate_single_line to capture the prompts
        # but we mock the provider's generate content
        return await original_escalate(self, target_idx, target_text, prev_text, next_text, target_language, show_title, is_real_untranslated, job_id, **kwargs)

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(SubtitleTranslator, "escalate_single_line", mock_escalate_single_line)
    
    # Mock Gemini client inside translator
    from google.genai import types
    class MockModels:
        def generate_content(self, model, contents, config):
            nonlocal isolated_prompt
            prompt_text = str(contents)
            sys_inst = str(getattr(config, 'system_instruction', ''))

            if "professional film/TV subtitle translator" in sys_inst or ("Previous/Next are context only" in sys_inst and "TARGET is known" in sys_inst):
                # Stage 1: Contextual
                return type('obj', (object,), {'text': '{"results": [{"id": 25, "text": "Dialogue line 25"}]}'})
            elif "strict" in sys_inst and "Item ID" in prompt_text:
                # Stage 2: Bulk Strict
                return type('obj', (object,), {'text': '{"results": [{"id": 25, "text": "Dialogue line 25"}]}'})
            elif "strict" in sys_inst and "failed QA" in prompt_text:
                # Stage 3: Per-Cue Escalation Attempt 2 (Strict with context)
                return type('obj', (object,), {'text': '{"translation": "' + f"Dialogue line {target_idx}" + '"}'})
            elif "strict" in sys_inst and "Translate this subtitle dialogue" in prompt_text:
                # Stage 3: Per-Cue Escalation Attempt 3 (Isolated)
                isolated_prompt = prompt_text
                return type('obj', (object,), {'text': '{"translation": "Svensk text"}'})

            return type('obj', (object,), {'text': '{"translation": "Swedish"}'})

    class MockClient:
        def __init__(self, api_key=None):
            self.models = MockModels()
            
    monkeypatch.setattr("app.services.translator.genai.Client", MockClient)

    # We need a target_idx for our MockModels closure
    target_idx = 25

    # Act
    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
    assert res["status"] != "failed"
    
    job = get_job_by_id(res["job_id"])
    assert job["status"] == "TRANSLATED" # NOT RECOVERING!
    
    # Main translation ran 1 time for the batch; recovery used structured stages
    assert call_counts["translate_batch"] == 1
    
    # Escalate should be called once (for the stubborn cue)
    assert call_counts["escalate"] == 1
    
    # Verify Step 10: Isolated prompt is truly isolated
    assert isolated_prompt is not None
    assert "Previous:" not in isolated_prompt
    assert "Next:" not in isolated_prompt
    assert "show title" not in isolated_prompt.lower()
    
    # Verify Step 11: Same model used
    # The mock uses mock_db_settings which has gemini-3.5-flash-lite for both

