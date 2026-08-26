import pytest
import os
import json
import srt
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.cleaner import clean_subtitle_text, sanitize_srt_content, EMPTY_PLACEHOLDER
from app.core.extractor import (
    get_cached_embedded_srt,
    save_cached_embedded_srt,
    invalidate_cached_embedded_srt,
    extract_embedded_srt,
)
from app.services.translator import (
    SubtitleTranslator,
    is_safe_keep_prefilter,
    is_meaningful_translation,
)
from app.services.pipeline import SubtitlePipeline, qa_gate
import app.core.db


@pytest.fixture
def mock_db_env(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    os.makedirs(db_path.parent, exist_ok=True)
    monkeypatch.setattr("app.core.db.DB_PATH", str(db_path))
    app.core.db.init_db()

    def mock_get_setting(key, default=""):
        settings = {
            "ai_provider": "gemini",
            "gemini_api_key": "test-key-123",
            "gemini_model": "gemini-3.5-flash-lite",
            "escalate_to_pro": "false",
            "escalation_provider": "none",
            "escalation_model": "",
            "batch_size": "50",
            "max_concurrent_jobs": "1",
            "wait_time_seconds": "0",
            "extract_target_embedded": "false",
            "extract_source_embedded": "true",
            "clean_sdh": "true",
            "languages": json.dumps([{"code": "sv", "name": "Swedish", "enabled": True}]),
        }
        return settings.get(key, default)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)
    monkeypatch.setattr("app.services.translator.get_setting", mock_get_setting)
    monkeypatch.setattr("app.core.db.get_setting", mock_get_setting)
    return db_path


# =========================================================================
# 1. Structural SDH & Speaker Prefix Tests
# =========================================================================

def test_sdh_structural_brackets_removed():
    """English source SDH noise patterns with trailing adverbs are stripped."""
    assert clean_subtitle_text("[sighs]") == EMPTY_PLACEHOLDER
    assert clean_subtitle_text("[door closes]") == EMPTY_PLACEHOLDER
    assert clean_subtitle_text("[chuckles softly]") == EMPTY_PLACEHOLDER
    assert clean_subtitle_text("[laughs faintly]") == EMPTY_PLACEHOLDER
    assert clean_subtitle_text("[dramatic music playing]") == EMPTY_PLACEHOLDER
    assert clean_subtitle_text("[in Spanish]") == EMPTY_PLACEHOLDER


def test_sdh_mixed_cue_preserves_dialogue():
    """Mixed cues strip SDH while preserving real dialogue."""
    assert clean_subtitle_text("[door closes] Where are you?") == "Where are you?"
    assert clean_subtitle_text("Hello John! [laughs softly]") == "Hello John!"
    assert clean_subtitle_text("[sighs] I don't know what to say.") == "I don't know what to say."
    res = clean_subtitle_text("Wait... (gasps) Look at that!")
    assert "Wait..." in res and "Look at that!" in res


def test_sdh_real_dialogue_in_brackets_and_parentheses_preserved():
    """Spoken dialogue within brackets and parentheses is preserved."""
    assert clean_subtitle_text("[I know what you did.]") == "[I know what you did.]"
    assert clean_subtitle_text("[Take cover!]") == "[Take cover!]"
    assert clean_subtitle_text("(Come here)") == "(Come here)"
    assert clean_subtitle_text("(I mean it.)") == "(I mean it.)"
    assert clean_subtitle_text("(Don't do that.)") == "(Don't do that.)"
    assert clean_subtitle_text("This is (very) nice.") == "This is (very) nice."


def test_sdh_speaker_prefix_stripping():
    """Structural speaker prefixes like 'MAN: Tibby.' become 'Tibby.' without wordlists."""
    assert clean_subtitle_text("MAN: Tibby.") == "Tibby."
    assert clean_subtitle_text("MAN:Tibby.") == "Tibby."
    assert clean_subtitle_text("WOMAN: [sobbing] Please stay.") == "Please stay."
    assert clean_subtitle_text("OFFICER 1: Freeze!") == "Freeze!"
    assert clean_subtitle_text("JOHN: Where are you?") == "Where are you?"
    assert clean_subtitle_text("NARRATOR: In the beginning...") == "In the beginning..."
    assert clean_subtitle_text("HOMME: Bonjour Pierre!") == "Bonjour Pierre!"



# =========================================================================
# 2. Persistent Extraction Cache Tests
# =========================================================================

def test_extraction_cache_hit_and_invalidation(tmp_path, mock_db_env):
    """Extraction cache returns cached content on identical file, misses on mtime/size change."""
    fake_video = str(tmp_path / "movie.mkv")
    with open(fake_video, "wb") as f:
        f.write(b"RIFF" + b"\x00" * 1000)

    sample_srt = "1\n00:00:01,000 --> 00:00:03,000\nHello World\n"
    save_cached_embedded_srt(fake_video, track_id=2, lang="eng", content=sample_srt)

    # 1. Exact Hit
    cached = get_cached_embedded_srt(fake_video, track_id=2, lang="en")
    assert cached == sample_srt

    # 2. Invalidation by modification (size change)
    with open(fake_video, "wb") as f:
        f.write(b"RIFF" + b"\x00" * 2000)

    missed = get_cached_embedded_srt(fake_video, track_id=2, lang="en")
    assert missed is None

    # 3. Explicit invalidation
    save_cached_embedded_srt(fake_video, track_id=2, lang="en", content=sample_srt)
    assert get_cached_embedded_srt(fake_video, track_id=2, lang="en") is not None
    invalidate_cached_embedded_srt(fake_video)
    assert get_cached_embedded_srt(fake_video, track_id=2, lang="en") is None


def test_extract_embedded_srt_uses_cache_without_subprocess(tmp_path, mock_db_env):
    """extract_embedded_srt reuses cache and executes zero subprocess commands."""
    fake_video = str(tmp_path / "movie.mkv")
    with open(fake_video, "wb") as f:
        f.write(b"MKV_HEADER" + b"\x00" * 500)

    sample_srt = "1\n00:00:01,000 --> 00:00:03,000\nCached subtitle text\n"
    save_cached_embedded_srt(fake_video, track_id=1, lang="eng", content=sample_srt)

    out_srt = str(tmp_path / "movie.en.srt")
    tracks_info = {
        "subtitles": [
            {"id": 1, "language": "eng", "codec": "SubRip/SRT", "forced": False, "title": "English"}
        ]
    }

    with patch("app.core.extractor.subprocess.run") as mock_run:
        ok = extract_embedded_srt(fake_video, out_srt, preferred_lang="eng", tracks_info=tracks_info)
        assert ok is True
        assert os.path.exists(out_srt)
        with open(out_srt, "r", encoding="utf-8") as f:
            assert f.read() == sample_srt
        # ZERO subprocess runs should happen on cache hit
        mock_run.assert_not_called()


# =========================================================================
# 3. Consolidated Global Micro Repair Tests
# =========================================================================

@pytest.mark.asyncio
async def test_consolidated_global_micro_repair_reduces_roundtrips(mock_db_env, monkeypatch):
    """Multiple batches with failed cues trigger consolidated bulk micro-repair instead of N per-batch calls."""
    translator = SubtitleTranslator()

    # Create 6 cues (2 batches of 3 cues)
    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="Hello"),
        srt.Subtitle(index=2, start=timedelta(seconds=2), end=timedelta(seconds=3), content="Batch 1 Fail"),
        srt.Subtitle(index=3, start=timedelta(seconds=3), end=timedelta(seconds=4), content="Good morning"),
        srt.Subtitle(index=4, start=timedelta(seconds=4), end=timedelta(seconds=5), content="Thank you"),
        srt.Subtitle(index=5, start=timedelta(seconds=5), end=timedelta(seconds=6), content="Batch 2 Fail"),
        srt.Subtitle(index=6, start=timedelta(seconds=6), end=timedelta(seconds=7), content="Goodbye"),
    ]

    micro_repair_call_count = 0

    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        res = []
        for it in items:
            t = it["text"]
            if "Fail" in t:
                # Return echo (untranslated) to trigger micro repair
                res.append({"id": it["id"], "text": t})
            else:
                res.append({"id": it["id"], "text": f"Oversatt {it['id']}"})
        return res

    async def mock_first_pass_micro_repair_batch(items, target_language, show_title="", **kwargs):
        nonlocal micro_repair_call_count
        micro_repair_call_count += 1
        return [
            {"id": it["id"], "text": f"Reparerad {it['id']}"}
            for it in items
        ]

    monkeypatch.setattr(translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(translator, "first_pass_micro_repair_batch", mock_first_pass_micro_repair_batch)

    res = await translator.translate_srt_content(subs, target_language="Swedish", batch_size=3)

    # Exactly 1 bulk call for all failed cues across both batches
    assert micro_repair_call_count == 1
    assert res[1].content == "Reparerad 1"
    assert res[4].content == "Reparerad 4"
    assert len(res) == 6
