import pytest
import os
import srt
import asyncio
from datetime import timedelta
from unittest.mock import patch, AsyncMock, MagicMock

from app.services.bazarr_checker import find_external_subtitle
from app.core.validator import check_language_representative, detect_language_heuristics, evaluate_subtitle_health
from app.core.cleaner import clean_subtitle_text, EMPTY_PLACEHOLDER
from app.services.translator import (
    SubtitleTranslator,
    ProviderUnavailableError,
    ProviderConfigurationError
)
from app.services.pipeline import SubtitlePipeline, QA_STATUS_FAIL, QA_STATUS_PASS_WITH_WARNINGS


# ===========================================================================
# 1. BUG A: CROSS-EPISODE SUBTITLE MATCHING ISOLATION
# ===========================================================================
def test_bug_a_cross_episode_isolation(tmp_path):
    """Verify find_external_subtitle strictly matches episode boundaries (e.g. S01E01 vs S01E010)."""
    video = str(tmp_path / "MyShow - S01E01.mkv")
    with open(video, "w") as f:
        f.write("dummy video content")

    # Only S01E010 exists on disk
    sub_ep10 = str(tmp_path / "MyShow - S01E010.sv.srt")
    with open(sub_ep10, "w", encoding="utf-8") as f:
        f.write("1\n00:00:01,000 --> 00:00:03,000\nDetta är avsnitt 10 med svensk text som är tillräckligt lång för att överstiga 100 bytes.")

    # Should NOT match episode 10 for episode 1
    assert find_external_subtitle(video, "sv") is None

    # Now create valid variants for S01E01
    sub_forced = str(tmp_path / "MyShow - S01E01.forced.sv.srt")
    with open(sub_forced, "w", encoding="utf-8") as f:
        f.write("1\n00:00:01,000 --> 00:00:03,000\nForced undertext som ska ignoreras av find_external_subtitle.")
    assert find_external_subtitle(video, "sv") is None

    sub_swe = str(tmp_path / "MyShow - S01E01.swedish.srt")
    with open(sub_swe, "w", encoding="utf-8") as f:
        f.write("1\n00:00:01,000 --> 00:00:03,000\nDetta är avsnitt 1 med svensk text som är tillräckligt lång för att överstiga 100 bytes.")
    assert find_external_subtitle(video, "sv") == sub_swe

    # Swedish locale variant
    os.remove(sub_swe)
    sub_locale = str(tmp_path / "MyShow - S01E01.sv-SE.srt")
    with open(sub_locale, "w", encoding="utf-8") as f:
        f.write("1\n00:00:01,000 --> 00:00:03,000\nDetta är avsnitt 1 med svensk text som är tillräckligt lång för att överstiga 100 bytes.")
    assert find_external_subtitle(video, "sv") == sub_locale


# ===========================================================================
# 2. BUG B: CONCURRENT TARGET RACE & EMBEDDED EXTRACTION PRESERVATION
# ===========================================================================
@pytest.mark.asyncio
async def test_bug_b_concurrent_target_race_preservation(tmp_path, monkeypatch):
    """If external target subtitle appears concurrently during embedded extraction, preserve it."""
    video = str(tmp_path / "RaceShow - S01E01.mkv")
    with open(video, "w") as f:
        f.write("dummy video")

    target_path = str(tmp_path / "RaceShow - S01E01.sv.srt")

    pipeline = SubtitlePipeline()

    # Simulate extract_embedded_srt creating a temp file, but concurrently writing the external target
    def mock_extract(vpath, outpath, preferred_lang="eng"):
        with open(outpath, "w", encoding="utf-8") as f:
            f.write("1\n00:00:01,000 --> 00:00:03,000\nInbäddad svensk text som är helt frisk och komplett.\n")
        # Concurrently create external target
        with open(target_path, "w", encoding="utf-8") as f:
            f.write("1\n00:00:01,000 --> 00:00:03,000\nExtern Bazarr-nedladdad svensk text som skapades under extraktionen.\n")
        return True

    monkeypatch.setattr("app.services.pipeline.extract_embedded_srt", mock_extract)
    monkeypatch.setattr("app.services.pipeline.get_setting", lambda k, d="": "true" if k in ["extract_target_embedded", "auto_repair_unhealthy"] else ("false" if k in ["notify_jellyfin", "enable_bazarr_check"] else d))
    monkeypatch.setattr("app.services.pipeline.notify_jellyfin_library_refresh", AsyncMock())
    monkeypatch.setattr(pipeline, "get_configured_languages", lambda: [{"name": "Swedish", "code": "sv", "enabled": True}])

    res = await pipeline.process_video_file(video, force_retranslate=False)

    # Target file should contain the external Bazarr text, not overwritten
    with open(target_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Extern Bazarr-nedladdad" in content


# ===========================================================================
# 3. BUG C: LANGUAGE QA DETECTS ENGLISH WITH 'MAN'/'MEN' ACCURATELY
# ===========================================================================
def test_bug_c_english_with_man_men_flagged_as_english():
    """English subtitles containing words like 'man' and 'men' must be detected as English and fail Swedish QA."""
    subs = []
    for i in range(100):
        content = f"The man said to all the other men that they should go home right now, line {i}."
        subs.append(srt.Subtitle(i + 1, timedelta(seconds=i * 2), timedelta(seconds=i * 2 + 1), content))

    res = check_language_representative(subs, target_lang_code="sv")
    assert res["confident_wrong_language"] is True
    assert res["detected_lang"] == "en"
    assert res["confidence"] > 0.85


# ===========================================================================
# 4. BUG D: CLEANER PRESERVES PARENTHETICAL DIALOGUE & STRIPS SDH
# ===========================================================================
def test_bug_d_cleaner_dialogue_and_sdh_handling():
    """Real dialogue in parentheses must be preserved; pure SDH and music notes must be cleaned."""
    # Dialogue cases to preserve
    assert clean_subtitle_text("(Come here)") == "(Come here)"
    assert clean_subtitle_text("(Please wait)") == "(Please wait)"
    assert clean_subtitle_text("(Help me)") == "(Help me)"
    assert clean_subtitle_text("(Stay back)") == "(Stay back)"
    assert clean_subtitle_text("(Good morning)") == "(Good morning)"
    assert clean_subtitle_text("[whispering] Don't make a sound.") == "Don't make a sound."

    # SDH cases to clean
    assert clean_subtitle_text("[laughing]") == EMPTY_PLACEHOLDER
    assert clean_subtitle_text("[door closes]") == EMPTY_PLACEHOLDER
    assert clean_subtitle_text("[phone ringing]") == EMPTY_PLACEHOLDER
    assert clean_subtitle_text("[door closes]") == EMPTY_PLACEHOLDER
    assert clean_subtitle_text("[SCREAMING]") == EMPTY_PLACEHOLDER
    assert clean_subtitle_text("♪ Never gonna give you up ♪") == "Never gonna give you up"
    assert clean_subtitle_text("♪♪♪") == EMPTY_PLACEHOLDER


# ===========================================================================
# 5. BUG E: PROVIDER ERRORS IN MICRO-REPAIR ARE RE-RAISED
# ===========================================================================
@pytest.mark.asyncio
async def test_bug_e_micro_repair_reraises_provider_errors(monkeypatch):
    """When first_pass_micro_repair_batch encounters ProviderUnavailableError, it is re-raised."""
    translator = SubtitleTranslator()

    # Create dummy subs with 1 line
    subs = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Hello world")
    ]

    # Mock translate_batch to return identical text triggering micro repair
    async def mock_translate_batch(batch, **kwargs):
        return [{"id": 0, "text": "Hello world"}]

    # Mock first_pass_micro_repair_batch to raise ProviderUnavailableError
    async def mock_micro_repair(items, **kwargs):
        raise ProviderUnavailableError("Simulated 429 rate limit in micro repair")

    monkeypatch.setattr(translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(translator, "first_pass_micro_repair_batch", mock_micro_repair)

    # Calling translate_srt_content must re-raise ProviderUnavailableError
    with pytest.raises(ProviderUnavailableError):
        await translator.translate_srt_content(
            subs,
            target_language="Swedish",
            job_id=None
        )
