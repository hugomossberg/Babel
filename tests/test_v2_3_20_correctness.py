import pytest
import os
import tempfile
import sqlite3
import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, patch, MagicMock
import srt

from app.core.cleaner import clean_subtitle_text, sanitize_srt_content
from app.core.db import init_db, save_translation_memory_bulk, get_translation_memory
from app.services.bazarr_checker import find_external_subtitle
from app.services.scanner import is_target_language_subtitle, is_subtitle_for_video
from app.services.translator import (
    SubtitleTranslator,
    ProviderUnavailableError,
    ProviderConfigurationError
)


# ============================================================================
# BUG A: SDH CLEANER DIALOGUE PRESERVATION vs SDH NOISE CLEANING
# ============================================================================

def test_sdh_preserves_real_dialogue():
    """Verify that real dialogue inside parentheses or brackets is never stripped."""
    dialogue_cases = [
        ("(Come here)", "(Come here)"),
        ("(Please wait)", "(Please wait)"),
        ("(Music is everything to me.)", "(Music is everything to me.)"),
        ("(Speaking of which, let us go.)", "(Speaking of which, let us go.)"),
        ("(Water is all we have.)", "(Water is all we have.)"),
        ("(Car trouble again?)", "(Car trouble again?)"),
        ("(COME HERE)", "(COME HERE)"),
        ("[I know what you did.]", "[I know what you did.]"),
        ("[Take cover!]", "[Take cover!]"),
        ("I told him (and I mean this) to leave.", "I told him (and I mean this) to leave."),
        ("This is (very) nice.", "This is (very) nice."),
    ]
    for raw, expected in dialogue_cases:
        assert clean_subtitle_text(raw) == expected, f"Failed preserving: {raw}"


def test_sdh_cleans_sound_effects_and_music_notes():
    """Verify that sound effects, reactions, music cues and notes are cleaned to <i></i>."""
    sdh_cases = [
        ("(door slams)", "<i></i>"),
        ("(phone ringing)", "<i></i>"),
        ("(gunshots)", "<i></i>"),
        ("(laughing)", "<i></i>"),
        ("(sighs)", "<i></i>"),
        ("[MUSIC PLAYING]", "<i></i>"),
        ("[DOOR SLAMS]", "<i></i>"),
        ("[Dramatic music playing]", "<i></i>"),
        ("(SCREAMING)", "<i></i>"),
        ("♪ music ♪", "<i></i>"),
        ("♪♪♪", "<i></i>"),
    ]
    for raw, expected in sdh_cases:
        assert clean_subtitle_text(raw) == expected, f"Failed cleaning SDH: {raw}"


def test_sdh_mixed_lines_and_lyrics():
    """Verify that mixed lines strip SDH cues and lyrics strip music notes."""
    assert clean_subtitle_text("[door opens] Hello John! (sighs)") == "Hello John!"
    assert clean_subtitle_text("(whispering) Don't make a sound.") == "Don't make a sound."
    assert clean_subtitle_text("♪ Never gonna give you up ♪") == "Never gonna give you up"


# ============================================================================
# BUG B: TRANSLATION MEMORY CROSS-SERIES ISOLATION
# ============================================================================

def test_tm_cross_series_isolation(tmp_path):
    """Verify that TM queries for 'Foo' do not leak entries from 'Foo - Bar' or 'Foo - Special'."""
    db_file = tmp_path / "test_tm.db"
    with patch("app.core.db.DB_PATH", str(db_file)):
        init_db()
        save_translation_memory_bulk("Foo", [{"original": "exact match", "translated": "exakt match"}])
        save_translation_memory_bulk("Foo - S01E01 - Pilot", [{"original": "pilot line", "translated": "pilot rad"}])
        save_translation_memory_bulk("Foo - S02E15 - Finale", [{"original": "finale line", "translated": "final rad"}])
        save_translation_memory_bulk("Foo - Bar", [{"original": "bar line", "translated": "bar rad"}])
        save_translation_memory_bulk("Foo - Special", [{"original": "special line", "translated": "special rad"}])
        save_translation_memory_bulk("Foo_2", [{"original": "foo2 line", "translated": "foo2 rad"}])
        save_translation_memory_bulk("Foo % Bar", [{"original": "percent show", "translated": "procent serie"}])
        save_translation_memory_bulk("Foo % Bar - S01E01", [{"original": "percent ep", "translated": "procent avsnitt"}])
        save_translation_memory_bulk("Foo % Bar - Extra", [{"original": "percent extra", "translated": "procent extra"}])

        # Test query for 'Foo'
        tm_foo = get_translation_memory("Foo", limit=50)
        origs_foo = {r["original"] for r in tm_foo}
        assert "exact match" in origs_foo
        assert "pilot line" in origs_foo
        assert "finale line" in origs_foo
        assert "bar line" not in origs_foo
        assert "special line" not in origs_foo
        assert "foo2 line" not in origs_foo
        assert "percent show" not in origs_foo

        # Test query for 'Foo % Bar'
        tm_percent = get_translation_memory("Foo % Bar", limit=50)
        origs_percent = {r["original"] for r in tm_percent}
        assert "percent show" in origs_percent
        assert "percent ep" in origs_percent
        assert "percent extra" not in origs_percent
        assert "exact match" not in origs_percent


# ============================================================================
# BUG C: FORCED SUBTITLES REJECTION IN BAZARR_CHECKER & SCANNER
# ============================================================================

def test_bazarr_checker_rejects_forced_subtitles(tmp_path):
    """Verify find_external_subtitle rejects all forced/signs/songs variants."""
    video = tmp_path / "Show.S01E01.mkv"
    video.write_text("fake video content")

    # Create dummy files
    files_to_create = [
        "Show.S01E01.sv.srt",
        "Show.S01E01.sv.full.srt",
        "Show.S01E01.sv.forced.srt",
        "Show.S01E01.forced.sv.srt",
        "Show.S01E01.sv-forced.srt",
        "Show.S01E01.sv_forced.srt",
        "Show.S01E010.sv.srt",
    ]
    for fn in files_to_create:
        p = tmp_path / fn
        p.write_text("1\n00:00:01,000 --> 00:00:02,000\nDummy text content for testing purposes.\n")

    # When sv-forced is searched, find_external_subtitle should find full .sv.srt or .sv.full.srt, never .forced
    # Test individual file checks by removing full ones
    (tmp_path / "Show.S01E01.sv.srt").unlink()
    (tmp_path / "Show.S01E01.sv.full.srt").unlink()

    # Now only forced/unrelated files remain
    assert find_external_subtitle(str(video), "sv") is None


def test_scanner_rejects_forced_subtitles():
    """Verify is_target_language_subtitle rejects forced, signs, songs across delimiters."""
    sv_aliases = ["sv", "swe", "swedish"]
    en_aliases = ["en", "eng", "english"]

    # Full subtitles -> True
    assert is_target_language_subtitle("Show.S01E01.sv.srt", sv_aliases) is True
    assert is_target_language_subtitle("Show.S01E01.sv.full.srt", sv_aliases) is True
    assert is_target_language_subtitle("Movie.en.srt", en_aliases) is True

    # Forced subtitles -> False
    assert is_target_language_subtitle("Show.S01E01.sv.forced.srt", sv_aliases) is False
    assert is_target_language_subtitle("Show.S01E01.forced.sv.srt", sv_aliases) is False
    assert is_target_language_subtitle("Show.S01E01.sv-forced.srt", sv_aliases) is False
    assert is_target_language_subtitle("Show.S01E01.sv_forced.srt", sv_aliases) is False
    assert is_target_language_subtitle("Movie.en.forced.srt", en_aliases) is False
    assert is_target_language_subtitle("Movie.forced.en.srt", en_aliases) is False
    assert is_target_language_subtitle("Movie.en-forced.srt", en_aliases) is False
    assert is_target_language_subtitle("Movie.en_forced.srt", en_aliases) is False


def test_scanner_is_subtitle_for_video_boundaries():
    """Verify boundary checks in scanner."""
    assert is_subtitle_for_video("Show.S01E01", "Show.S01E01.sv.srt") is True
    assert is_subtitle_for_video("Show.S01E01", "Show.S01E01.sv-forced.srt") is True
    assert is_subtitle_for_video("Show.S01E01", "Show.S01E010.sv.srt") is False


# ============================================================================
# BUG D: PROVIDER EXCEPTION BUBBLING IN ENTITY VERIFICATION
# ============================================================================

@pytest.mark.asyncio
async def test_entity_verification_bubbles_provider_unavailable():
    """Verify ProviderUnavailableError is re-raised from entity verification."""
    t = SubtitleTranslator()
    items = [{"id": 0, "text": "Arthur"}]
    source_subs = [srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Arthur")]
    translated_subs = [srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Arthur")]

    t.verify_single_occurrence_entities = AsyncMock(side_effect=ProviderUnavailableError("simulated outage"))

    with patch("app.services.translator.get_setting") as mock_s, patch("google.genai.Client") as mock_c:
        mock_s.side_effect = lambda k, d="": "gemini" if k == "ai_provider" else ("dummy_key" if k == "gemini_api_key" else d)
        mock_c.return_value.models.generate_content.return_value = MagicMock(
            text='{"results":[{"id":0,"action":"keep","reason":"proper_noun","text":"Arthur"}]}'
        )
        with pytest.raises(ProviderUnavailableError):
            await t.classify_and_recover_identical(
                items, "Swedish", "Show", source_subs=source_subs, translated_subs=translated_subs
            )


@pytest.mark.asyncio
async def test_entity_verification_bubbles_provider_configuration_error():
    """Verify ProviderConfigurationError is re-raised from entity verification."""
    t = SubtitleTranslator()
    items = [{"id": 0, "text": "Arthur"}]
    source_subs = [srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Arthur")]
    translated_subs = [srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Arthur")]

    t.verify_single_occurrence_entities = AsyncMock(side_effect=ProviderConfigurationError("simulated bad key"))

    with patch("app.services.translator.get_setting") as mock_s, patch("google.genai.Client") as mock_c:
        mock_s.side_effect = lambda k, d="": "gemini" if k == "ai_provider" else ("dummy_key" if k == "gemini_api_key" else d)
        mock_c.return_value.models.generate_content.return_value = MagicMock(
            text='{"results":[{"id":0,"action":"keep","reason":"proper_noun","text":"Arthur"}]}'
        )
        with pytest.raises(ProviderConfigurationError):
            await t.classify_and_recover_identical(
                items, "Swedish", "Show", source_subs=source_subs, translated_subs=translated_subs
            )


@pytest.mark.asyncio
async def test_entity_verification_handles_local_parsing_errors():
    """Verify generic non-provider errors during entity verification are logged and handled locally."""
    t = SubtitleTranslator()
    items = [{"id": 0, "text": "Arthur"}]
    source_subs = [srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Arthur")]
    translated_subs = [srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Arthur")]

    t.verify_single_occurrence_entities = AsyncMock(side_effect=ValueError("bad json from entity LLM"))

    with patch("app.services.translator.get_setting") as mock_s, patch("google.genai.Client") as mock_c:
        mock_s.side_effect = lambda k, d="": "gemini" if k == "ai_provider" else ("dummy_key" if k == "gemini_api_key" else d)
        mock_c.return_value.models.generate_content.return_value = MagicMock(
            text='{"results":[{"id":0,"action":"keep","reason":"proper_noun","text":"Arthur"}]}'
        )
        res = await t.classify_and_recover_identical(
            items, "Swedish", "Show", source_subs=source_subs, translated_subs=translated_subs
        )
        assert len(res) == 1
        assert res[0]["action"] == "translate"
