"""
tests/test_source_resolver_root_cause_audit.py

Regression test suite for Source Resolver & Extractor Root Cause Audit:
1. Highest-ranked healthy candidate stops further extraction immediately.
2. Audio-match prioritization.
3. Sequential candidate testing only on actual failure.
4. Subprocess pipe deadlock immunity (large stdout/stderr output).
5. Bounded timeout / mkvextract timeout skips slow fallback.
6. Cancelled subprocess group terminates cleanly with no orphans.
7. Extraction cache hits skip subprocess invocations.
8. Forced Translate preserves fast single-source selection.
9. Language-agnostic behavior for arbitrary language pairs.
"""

import asyncio
import os
import srt
import time
import threading
import subprocess
import pytest
from datetime import timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from app.core.extractor import (
    _run_cancellable_cmd,
    extract_embedded_srt,
    get_cached_embedded_srt,
    save_cached_embedded_srt,
    DEFAULT_EXTRACTION_TIMEOUT,
)
from app.services.source_resolver import SourceResolver, SourceOrigin
from app.services.pipeline import SubtitlePipeline
from app.core.db import create_job, get_job_by_id


def _make_valid_srt(num_cues=30, lang="en", duration_span=None):
    cues = []
    texts = {
        "en": "This is an English dialogue sentence for test cue number",
        "spa": "Esta es una frase de diálogo en español para la prueba número",
        "es": "Esta es una frase de diálogo en español para la prueba número",
        "ja": "これはテスト用の日本語字幕のセリフです番号",
        "jpn": "これはテスト用の日本語字幕のセリフです番号",
        "sv": "Detta är en svensk dialogmening för testspår nummer",
    }
    base_text = texts.get(lang, "This is a standard subtitle dialogue cue number")
    for i in range(num_cues):
        if duration_span:
            start_sec = (i / num_cues) * (duration_span * 0.9) + 5.0
        else:
            start_sec = i * 4.0
        end_sec = start_sec + 2.5
        cues.append(srt.Subtitle(
            index=i + 1,
            start=timedelta(seconds=start_sec),
            end=timedelta(seconds=end_sec),
            content=f"{base_text} {i + 1}."
        ))
    return srt.compose(cues)


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    import app.core.db as db_mod
    orig = db_mod.DB_PATH
    db_mod.DB_PATH = str(tmp_path / "test_audit.db")
    db_mod.init_db()
    yield tmp_path
    db_mod.DB_PATH = orig


@pytest.mark.asyncio
async def test_highest_ranked_healthy_candidate_stops_further_extraction(tmp_path):
    """
    Sisterhood-case invariant:
    When track 2 (EN) is healthy and extracted, SourceResolver stops immediately.
    Tracks 3 (ES), 4 (FI), 5 (FR), 6 (IT), 7 (NL), 8 (PL), 9 (PT) must NEVER be extracted.
    """
    video = tmp_path / "Sisterhood.2005.mkv"
    video.touch()

    tracks_info = {
        "audio": [{"id": 1, "language": "eng"}],
        "duration": 7200.0,
        "subtitles": [
            {"id": 2, "language": "eng", "codec": "SubRip/SRT", "forced": False, "title": "English"},
            {"id": 3, "language": "spa", "codec": "SubRip/SRT", "forced": False, "title": "Spanish"},
            {"id": 4, "language": "fin", "codec": "SubRip/SRT", "forced": False, "title": "Finnish"},
            {"id": 5, "language": "fre", "codec": "SubRip/SRT", "forced": False, "title": "French"},
            {"id": 6, "language": "ita", "codec": "SubRip/SRT", "forced": False, "title": "Italian"},
            {"id": 7, "language": "nld", "codec": "SubRip/SRT", "forced": False, "title": "Dutch"},
            {"id": 8, "language": "pol", "codec": "SubRip/SRT", "forced": False, "title": "Polish"},
            {"id": 9, "language": "por", "codec": "SubRip/SRT", "forced": False, "title": "Portuguese"},
            {"id": 10, "language": "swe", "codec": "SubRip/SRT", "forced": False, "title": "Swedish"},
        ]
    }

    extracted_languages = []

    def mock_extract(vpath, outpath, preferred_lang="eng", tracks_info=None, cancel_event=None):
        extracted_languages.append(preferred_lang)
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(_make_valid_srt(50, lang=preferred_lang, duration_span=7200.0))
        return True

    resolver = SourceResolver(
        video_path=str(video),
        container_tracks=tracks_info,
        primary_audio_lang="en",
        target_languages=["sv"],
        bazarr_url="",
        bazarr_api_key="",
        enable_bazarr=False,
        extract_source_embedded=True,
        source_search_deadline=0.0,
    )

    with patch("app.services.pipeline._safe_extract_embedded_srt", side_effect=mock_extract):
        source = await resolver.resolve()

    assert source is not None
    assert source.origin == SourceOrigin.EMBEDDED
    assert source.language == "en"
    # Exactly one extraction attempted (en track 2)
    assert extracted_languages == ["en"], f"Expected only ['en'], but extracted: {extracted_languages}"


@pytest.mark.asyncio
async def test_audio_match_prioritizes_candidate_languages(tmp_path):
    """
    When primary audio is Spanish ('es'), Spanish embedded track must be evaluated
    before English or other languages.
    """
    video = tmp_path / "SpanishFilm.mkv"
    video.touch()

    tracks_info = {
        "audio": [{"id": 1, "language": "spa"}],
        "duration": 5000.0,
        "subtitles": [
            {"id": 2, "language": "eng", "codec": "SubRip/SRT", "forced": False, "title": "English"},
            {"id": 3, "language": "spa", "codec": "SubRip/SRT", "forced": False, "title": "Spanish"},
        ]
    }

    extracted_languages = []

    def mock_extract(vpath, outpath, preferred_lang="spa", tracks_info=None, cancel_event=None):
        extracted_languages.append(preferred_lang)
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(_make_valid_srt(40, lang=preferred_lang, duration_span=5000.0))
        return True

    resolver = SourceResolver(
        video_path=str(video),
        container_tracks=tracks_info,
        primary_audio_lang="es",
        target_languages=["sv"],
        bazarr_url="",
        bazarr_api_key="",
        enable_bazarr=False,
        extract_source_embedded=True,
        source_search_deadline=0.0,
    )

    with patch("app.services.pipeline._safe_extract_embedded_srt", side_effect=mock_extract):
        source = await resolver.resolve()

    assert source is not None
    assert source.origin == SourceOrigin.EMBEDDED
    assert source.language == "es"
    assert extracted_languages == ["es"], f"Expected ['es'] first and only, got: {extracted_languages}"


@pytest.mark.asyncio
async def test_next_candidate_only_tested_on_failure(tmp_path):
    """
    Sequential fallback: Track 2 (EN) fails extraction or is corrupt/empty.
    Only then is the next candidate (ES) attempted and selected.
    """
    video = tmp_path / "FallbackTest.mkv"
    video.touch()

    tracks_info = {
        "audio": [{"id": 1, "language": "eng"}],
        "duration": 5000.0,
        "subtitles": [
            {"id": 2, "language": "eng", "codec": "SubRip/SRT", "forced": False, "title": "English"},
            {"id": 3, "language": "spa", "codec": "SubRip/SRT", "forced": False, "title": "Spanish"},
        ]
    }

    extracted_languages = []

    def mock_extract(vpath, outpath, preferred_lang="eng", tracks_info=None, cancel_event=None):
        extracted_languages.append(preferred_lang)
        if preferred_lang == "en":
            # Return corrupt/empty file
            with open(outpath, "w", encoding="utf-8") as f:
                f.write("corrupt")
            return True
        else:
            with open(outpath, "w", encoding="utf-8") as f:
                f.write(_make_valid_srt(40, lang=preferred_lang, duration_span=5000.0))
            return True

    resolver = SourceResolver(
        video_path=str(video),
        container_tracks=tracks_info,
        primary_audio_lang="en",
        target_languages=["sv"],
        bazarr_url="",
        bazarr_api_key="",
        enable_bazarr=False,
        extract_source_embedded=True,
        source_search_deadline=0.0,
    )

    with patch("app.services.pipeline._safe_extract_embedded_srt", side_effect=mock_extract):
        source = await resolver.resolve()

    assert source is not None
    assert source.origin == SourceOrigin.EMBEDDED
    assert source.language == "es"
    assert extracted_languages == ["en", "es"]


def test_subprocess_pipe_deadlock_immunity():
    """
    _run_cancellable_cmd must not deadlock when child process writes huge stderr/stdout data (>128 KB).
    """
    cancel_ev = threading.Event()
    # Script outputs 200 KB of text to stderr
    cmd = ["python3", "-c", "import sys; sys.stderr.write('x' * 200000); sys.stderr.flush()"]

    t0 = time.monotonic()
    ret = _run_cancellable_cmd(cmd, timeout=10.0, cancel_event=cancel_ev)
    elapsed = time.monotonic() - t0

    assert ret == 0
    assert elapsed < 5.0, f"Process took {elapsed:.2f}s, indicating potential pipe deadlock!"


def test_cancelled_subprocess_terminates_cleanly():
    """
    _run_cancellable_cmd kills the process group immediately when cancel_event is set.
    """
    cancel_ev = threading.Event()
    cmd = ["sleep", "30"]

    # Trigger cancellation after 0.1s
    timer = threading.Timer(0.1, cancel_ev.set)
    timer.start()

    t0 = time.monotonic()
    ret = _run_cancellable_cmd(cmd, timeout=30.0, cancel_event=cancel_ev)
    elapsed = time.monotonic() - t0

    assert ret == -1
    assert elapsed < 1.0


def test_mkvextract_timeout_skips_ffmpeg_fallback(tmp_path):
    """
    If mkvextract times out on a large MKV, ffmpeg fallback should be skipped
    to avoid doubling the timeout latency on containers where ffmpeg is strictly slower.
    """
    video = tmp_path / "large_movie.mkv"
    video.touch()
    out_srt = tmp_path / "out.srt"

    tracks_info = {
        "duration": 5000.0,
        "subtitles": [
            {"id": 2, "language": "eng", "codec": "SubRip/SRT", "forced": False, "title": "English"}
        ]
    }

    call_log = []

    def mock_run_cancellable(cmd, timeout=10.0, cancel_event=None):
        call_log.append(cmd[0])
        if cmd[0] == "mkvextract":
            raise subprocess.TimeoutExpired(cmd, timeout)
        return 0

    with patch("app.core.extractor._run_cancellable_cmd", side_effect=mock_run_cancellable):
        res = extract_embedded_srt(
            video_path=str(video),
            output_srt_path=str(out_srt),
            preferred_lang="eng",
            tracks_info=tracks_info,
            timeout=5.0,
        )

    assert res is False
    assert call_log == ["mkvextract"], f"ffmpeg should NOT be called after mkvextract timeout: {call_log}"


def test_cached_source_skips_subprocess(tmp_path):
    """
    If track is already cached in embedded_extraction_cache, extract_embedded_srt
    returns True immediately without executing any subprocess.
    """
    video = tmp_path / "cached_movie.mkv"
    video.touch()
    out_srt = tmp_path / "out.srt"
    srt_content = _make_valid_srt(30, lang="en")

    save_cached_embedded_srt(str(video), 2, "eng", srt_content)

    tracks_info = {
        "duration": 5000.0,
        "subtitles": [
            {"id": 2, "language": "eng", "codec": "SubRip/SRT", "forced": False, "title": "English"}
        ]
    }

    with patch("app.core.extractor._run_cancellable_cmd") as mock_cmd:
        res = extract_embedded_srt(
            video_path=str(video),
            output_srt_path=str(out_srt),
            preferred_lang="eng",
            tracks_info=tracks_info,
        )

    assert res is True
    assert out_srt.exists()
    assert mock_cmd.call_count == 0, "Subprocess must not be called on cache hit"


@pytest.mark.asyncio
async def test_forced_translate_preserves_single_source_selection(tmp_path):
    """
    Forced Translate (force_retranslate=True) must not extract all embedded languages;
    it should extract only the single highest-ranked valid source track.
    """
    video = tmp_path / "ForcedMovie.mkv"
    video.touch()

    # Pre-existing target file
    sv_srt = tmp_path / "ForcedMovie.sv.srt"
    sv_srt.write_text(_make_valid_srt(20, lang="sv"), encoding="utf-8")

    tracks_info = {
        "audio": [{"id": 1, "language": "eng"}],
        "duration": 6000.0,
        "subtitles": [
            {"id": 2, "language": "eng", "codec": "SubRip/SRT", "forced": False, "title": "English"},
            {"id": 3, "language": "spa", "codec": "SubRip/SRT", "forced": False, "title": "Spanish"},
            {"id": 4, "language": "fre", "codec": "SubRip/SRT", "forced": False, "title": "French"},
        ]
    }

    extracted_languages = []

    def fake_extract(vp, outpath, preferred_lang="eng", tracks_info=None, cancel_event=None):
        extracted_languages.append(preferred_lang)
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(_make_valid_srt(30, lang="en", duration_span=6000.0))
        return True

    async def fake_translate(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        return [srt.Subtitle(index=i+1, start=s.start, end=s.end, content=f"Svenska {i}") for i, s in enumerate(subs)]

    pipeline = SubtitlePipeline()
    settings = {
        "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
        "extract_source_embedded": "true",
        "enable_bazarr_check": "false",
    }

    with patch("app.services.pipeline.get_setting", side_effect=lambda k, d="": settings.get(k, d)), \
         patch("app.services.pipeline.inspect_mkv_tracks", return_value=tracks_info), \
         patch("app.services.pipeline._safe_extract_embedded_srt", side_effect=fake_extract), \
         patch.object(pipeline, "trigger_bazarr_search", AsyncMock()), \
         patch.object(pipeline.translator, "translate_srt_content", fake_translate):
        job_id = create_job(str(video))
        res = await pipeline.process_video_file(str(video), job_id=job_id, force_retranslate=True)

    assert res["status"] == "translated"
    assert extracted_languages == ["en"], f"Expected only ['en'] extraction, got {extracted_languages}"


@pytest.mark.asyncio
async def test_language_agnostic_embedded_source_resolution(tmp_path):
    """
    Verify language-agnostic behavior for arbitrary language pairs (e.g. Japanese source, German target).
    """
    video = tmp_path / "AnimeEpisode.mkv"
    video.touch()

    tracks_info = {
        "audio": [{"id": 1, "language": "jpn"}],
        "duration": 1400.0,
        "subtitles": [
            {"id": 2, "language": "jpn", "codec": "SubRip/SRT", "forced": False, "title": "Japanese"},
        ]
    }

    def fake_extract(vp, outpath, preferred_lang="ja", tracks_info=None, cancel_event=None):
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(_make_valid_srt(30, lang="ja", duration_span=1400.0))
        return True

    resolver = SourceResolver(
        video_path=str(video),
        container_tracks=tracks_info,
        primary_audio_lang="ja",
        target_languages=["de"],
        bazarr_url="",
        bazarr_api_key="",
        enable_bazarr=False,
        extract_source_embedded=True,
        source_search_deadline=0.0,
    )

    with patch("app.services.pipeline._safe_extract_embedded_srt", side_effect=fake_extract):
        source = await resolver.resolve()

    assert source is not None
    assert source.origin == SourceOrigin.EMBEDDED
    assert source.language == "ja"
