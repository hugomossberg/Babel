import pytest
import os
import json
import srt
from datetime import timedelta
from unittest.mock import patch, AsyncMock, MagicMock

from app.core.cleaner import subs_to_srt_string
from app.services.pipeline import SubtitlePipeline, qa_gate
from app.services.translator import (
    SubtitleTranslator,
    is_strictly_valid_entity_candidate,
    is_valid_shared_or_entity_keep,
    is_meaningful_translation,
    validate_classifier_output
)
from app.core.db import init_db, get_job_by_id, create_job, DB_PATH


@pytest.fixture
def mock_db_settings(monkeypatch, tmp_path):
    test_db = str(tmp_path / "test_v2_3_43_hardening.db")
    monkeypatch.setattr("app.core.db.DB_PATH", test_db)
    init_db()

    def mock_get_setting(key, default=""):
        settings = {
            "ai_provider": "gemini",
            "gemini_api_key": "dummy_api_key",
            "gemini_model": "gemini-3.5-flash-lite",
            "escalate_to_pro": "false",
            "escalation_provider": "none",
            "escalation_model": "",
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
    monkeypatch.setattr("app.services.translator.SubtitleTranslator.first_pass_micro_repair_batch", AsyncMock(return_value=[]))


# =========================================================================
# 1. BORGO ZERO-DURATION & STRUCTURAL INTEGRITY TESTS
# =========================================================================

def test_borgo_zero_duration_cue_preserved_in_subs_to_srt_string():
    """
    Verify Borgo cue 1094 (start == end == 01:08:23.326) is 100% preserved
    and never dropped by srt.compose(reindex=False).
    """
    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=3), content="Normal cue 1"),
        srt.Subtitle(index=2, start=timedelta(seconds=5), end=timedelta(seconds=5), content="Zero duration cue 2"),
        srt.Subtitle(index=3, start=timedelta(seconds=6), end=timedelta(seconds=8), content="Normal cue 3")
    ]
    srt_str = subs_to_srt_string(subs)
    parsed = list(srt.parse(srt_str))
    assert len(parsed) == 3
    assert parsed[1].content == "Zero duration cue 2"
    assert parsed[1].start == parsed[1].end == timedelta(seconds=5)


def test_empty_cue_preserved_with_invisible_placeholder():
    """
    Verify completely empty cues are preserved with <i></i> placeholder.
    """
    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content=""),
        srt.Subtitle(index=2, start=timedelta(seconds=3), end=timedelta(seconds=4), content="Hello")
    ]
    srt_str = subs_to_srt_string(subs)
    parsed = list(srt.parse(srt_str))
    assert len(parsed) == 2
    assert parsed[0].content == "<i></i>"


# =========================================================================
# 2. MULTI-NAME LIST & SHARED CROSS-LINGUAL COGNATE VALIDATION
# =========================================================================

def test_romanian_multi_name_lists_classified_as_valid_entity_candidates():
    """
    Verify multi-name lists with particles (e.g. R.M.N. credits) pass
    is_strictly_valid_entity_candidate up to 16 tokens.
    """
    assert is_strictly_valid_entity_candidate("Németh Zsolt, Kelemen Tibor")
    assert is_strictly_valid_entity_candidate("Ioan von Weber, Maria de Silva")
    assert is_strictly_valid_entity_candidate("Dr. Radu Popescu")
    # Sentences with common English words must fail
    assert not is_strictly_valid_entity_candidate("This is a long sentence with many words")


def test_shared_cross_lingual_words_accepted_in_swedish():
    """
    Verify legitimate cross-lingual words (German 'Ja', Italian 'Mamma', 'Nej', 'OK')
    are accepted by is_valid_shared_or_entity_keep.
    """
    assert is_valid_shared_or_entity_keep("Ja", "Ja", target_lang="sv")
    assert is_valid_shared_or_entity_keep("Mamma", "Mamma", target_lang="sv")
    assert is_valid_shared_or_entity_keep("Nej", "Nej", target_lang="sv")
    assert is_valid_shared_or_entity_keep("OK", "OK", target_lang="sv")
    assert is_valid_shared_or_entity_keep("Bravo", "Bravo", target_lang="sv")
    assert is_valid_shared_or_entity_keep("Taxi", "Taxi", target_lang="sv")
    # Non-shared English dialogue must NOT pass
    assert not is_valid_shared_or_entity_keep("Hello", "Hello", target_lang="sv")
    assert not is_valid_shared_or_entity_keep("Where are you?", "Where are you?", target_lang="sv")


# =========================================================================
# 3. CALL COUNT REDUCTION & ELIMINATION OF N x 3 ESCALATION
# =========================================================================

@pytest.mark.asyncio
async def test_4_unresolved_cues_rmn_bulk_recovery_call_reduction(mock_db_settings, tmp_path, monkeypatch):
    """
    Verify 4 unresolved cues (R.M.N. scenario) trigger exactly 1 bulk call
    instead of 12 per-cue escalation calls (4 * 3).
    """
    video_path = tmp_path / "RMN.2022.mkv"
    video_path.touch()
    en_srt = tmp_path / "RMN.2022.en.srt"

    cues_text = [f"Dialogue line {i}" for i in range(20)]
    subs = [
        srt.Subtitle(index=i+1, start=timedelta(seconds=i*2), end=timedelta(seconds=i*2+1), content=cues_text[i])
        for i in range(len(cues_text))
    ]
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()

    # Pass 0: 4 cues (index 2, 5, 8, 12) return untranslated
    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        results = []
        for item in items:
            idx = item["id"]
            if idx in (2, 5, 8, 12):
                results.append({"id": idx, "text": item["text"]})
            else:
                results.append({"id": idx, "text": f"Svenska {idx}"})
        return results

    async def mock_classify(items, lang, title, **kwargs):
        return [{"id": item["id"], "action": "translate", "reason": "none", "text": ""} for item in items]

    call_stats = {"bulk_recovery": 0, "escalation_single": 0}

    async def mock_rescue_batch(items, target_language, source_language="Romanian", show_title="", attempt=1, job_id=None):
        call_stats["bulk_recovery"] += 1
        # Bulk recovery translates all 4 in 1 single structured call
        return [{"id": it["id"], "text": f"Återställd svenska {it['id']}"} for it in items]

    async def mock_escalate_single(*args, **kwargs):
        call_stats["escalation_single"] += 1
        return "Escalated"

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(pipeline.translator, "fast_final_rescue_batch", mock_rescue_batch)
    monkeypatch.setattr(pipeline.translator, "escalate_single_line", mock_escalate_single)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
    assert res["status"] != "failed"

    job = get_job_by_id(res["job_id"])
    assert job["status"] == "TRANSLATED"

    # Exactly 1 bulk recovery call made; 0 single-line escalation calls made!
    assert call_stats["bulk_recovery"] == 1
    assert call_stats["escalation_single"] == 0


@pytest.mark.asyncio
async def test_9_unresolved_cues_teachers_lounge_bulk_recovery_call_reduction(mock_db_settings, tmp_path, monkeypatch):
    """
    Verify 9 unresolved cues (Teachers' Lounge scenario) trigger exactly 1 bulk call
    instead of 27 per-cue escalation calls (9 * 3).
    """
    video_path = tmp_path / "TeachersLounge.2023.mkv"
    video_path.touch()
    en_srt = tmp_path / "TeachersLounge.2023.en.srt"

    cues_text = [f"Dialogue line {i}" for i in range(30)]
    subs = [
        srt.Subtitle(index=i+1, start=timedelta(seconds=i*2), end=timedelta(seconds=i*2+1), content=cues_text[i])
        for i in range(len(cues_text))
    ]
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()
    stubborn_ids = {1, 3, 5, 7, 9, 11, 13, 15, 17}

    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        results = []
        for item in items:
            idx = item["id"]
            if idx in stubborn_ids:
                results.append({"id": idx, "text": item["text"]})
            else:
                results.append({"id": idx, "text": f"Detta är en svensk dialograd {idx}"})
        return results

    async def mock_classify(items, lang, title, **kwargs):
        return [{"id": item["id"], "action": "translate", "reason": "none", "text": ""} for item in items]

    call_stats = {"bulk_recovery": 0, "escalation_single": 0}

    async def mock_rescue_batch(items, target_language, source_language="German", show_title="", attempt=1, job_id=None):
        call_stats["bulk_recovery"] += 1
        assert len(items) == 9
        return [{"id": it["id"], "text": f"Detta är svensk översättning {it['id']}"} for it in items]

    async def mock_escalate_single(*args, **kwargs):
        call_stats["escalation_single"] += 1
        return "Escalated"

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(pipeline.translator, "fast_final_rescue_batch", mock_rescue_batch)
    monkeypatch.setattr(pipeline.translator, "escalate_single_line", mock_escalate_single)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
    assert res["status"] != "failed"

    job = get_job_by_id(res["job_id"])
    assert job["status"] == "TRANSLATED"

    # Exactly 1 bulk recovery call for all 9 items; 0 escalation calls!
    assert call_stats["bulk_recovery"] == 1
    assert call_stats["escalation_single"] == 0


@pytest.mark.asyncio
async def test_20_unresolved_cues_danish_bulk_recovery_call_reduction(mock_db_settings, tmp_path, monkeypatch):
    """
    Verify 20 unresolved cues (Danish media scenario) trigger exactly 1 bulk call
    instead of 60 per-cue escalation calls (20 * 3).
    """
    video_path = tmp_path / "DanishShow.S01E01.mkv"
    video_path.touch()
    da_srt = tmp_path / "DanishShow.S01E01.en.srt"

    cues_text = [f"Dansk replik {i}" for i in range(50)]
    subs = [
        srt.Subtitle(index=i+1, start=timedelta(seconds=i*2), end=timedelta(seconds=i*2+1), content=cues_text[i])
        for i in range(len(cues_text))
    ]
    with open(da_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()
    stubborn_ids = set(range(0, 20))

    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        results = []
        for item in items:
            idx = item["id"]
            if idx in stubborn_ids:
                results.append({"id": idx, "text": item["text"]})
            else:
                results.append({"id": idx, "text": f"Svenska {idx}"})
        return results

    async def mock_classify(items, lang, title, **kwargs):
        return [{"id": item["id"], "action": "translate", "reason": "none", "text": ""} for item in items]

    call_stats = {"bulk_recovery": 0, "escalation_single": 0}

    async def mock_rescue_batch(items, target_language, source_language="Danish", show_title="", attempt=1, job_id=None):
        call_stats["bulk_recovery"] += 1
        assert len(items) == 20
        return [{"id": it["id"], "text": f"Dansk-svensk översättning {it['id']}"} for it in items]

    async def mock_escalate_single(*args, **kwargs):
        call_stats["escalation_single"] += 1
        return "Escalated"

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(pipeline.translator, "fast_final_rescue_batch", mock_rescue_batch)
    monkeypatch.setattr(pipeline.translator, "escalate_single_line", mock_escalate_single)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
    assert res["status"] != "failed"

    job = get_job_by_id(res["job_id"])
    assert job["status"] == "TRANSLATED"

    # Exactly 1 bulk recovery call for all 20 items; 0 escalation calls!
    assert call_stats["bulk_recovery"] == 1
    assert call_stats["escalation_single"] == 0
