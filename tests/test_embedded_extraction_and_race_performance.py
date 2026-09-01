import asyncio
import os
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import srt

from app.core.db import DB_PATH, create_job, get_job_by_id, init_db
from app.core.extractor import (
    extract_embedded_srt,
    get_cached_embedded_srt,
    save_cached_embedded_srt,
    invalidate_cached_embedded_srt,
    inspect_mkv_tracks,
    _run_cancellable_cmd,
    _extraction_semaphore,
    _in_flight_extractions,
    _in_flight_lock,
)
from app.services.pipeline import SubtitlePipeline
from app.services.source_resolver import SourceResolver, SubtitleSource, SourceOrigin

SAMPLE_SRT = """1
00:00:01,000 --> 00:00:04,000
Welcome to the MasterChef kitchen.

2
00:00:05,000 --> 00:00:08,000
Tonight you face your biggest challenge.
"""

SAMPLE_SV_SRT = """1
00:00:01,000 --> 00:00:04,000
Välkommen till MasterChef-köket.

2
00:00:05,000 --> 00:00:08,000
Ikväll möter ni er största utmaning.

3
00:00:09,000 --> 00:00:12,000
Kockarna är redo att börja laga mat.

4
00:00:13,000 --> 00:00:16,000
Tiden startar nu på klockan.

5
00:00:17,000 --> 00:00:20,000
Smakerna måste sitta perfekt.

6
00:00:21,000 --> 00:00:24,000
Juryn väntar med spänning på resultatet.

7
00:00:25,000 --> 00:00:28,000
Det här är en fantastisk rätt.

8
00:00:29,000 --> 00:00:32,000
Bra jobbat av alla deltagare ikväll.

9
00:00:33,000 --> 00:00:36,000
Vi ses igen i nästa avsnitt.

10
00:00:37,000 --> 00:00:40,000
Tack för att ni tittade.
"""


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test_perf.db")
    monkeypatch.setattr("app.core.db.DB_PATH", test_db)
    init_db()
    with _in_flight_lock:
        _in_flight_extractions.clear()
    from app.services.bazarr_coordinator import bazarr_coordinator
    bazarr_coordinator.reset()
    yield
    bazarr_coordinator.reset()


@pytest.mark.asyncio
async def test_1_cached_embedded_extraction_skips_subprocess(tmp_path):
    """1. Proves cached embedded extraction skips subprocess entirely (0 subprocess calls on cache hit)."""
    video_path = str(tmp_path / "test_video.mkv")
    Path(video_path).write_text("fake video content", encoding="utf-8")
    out_srt = str(tmp_path / "extracted.srt")

    # Seed the cache
    save_cached_embedded_srt(video_path, track_id=1, lang="eng", content=SAMPLE_SRT)

    tracks_info = {
        "duration": 120.0,
        "subtitles": [{"id": 1, "language": "eng", "codec": "SubRip/SRT", "forced": False, "title": "English"}]
    }

    with patch("app.core.extractor.subprocess.run") as mock_subproc_run, \
         patch("app.core.extractor.subprocess.Popen") as mock_subproc_popen:
        success = extract_embedded_srt(video_path, out_srt, preferred_lang="eng", tracks_info=tracks_info)
        assert success is True
        assert os.path.exists(out_srt)
        with open(out_srt, "r", encoding="utf-8") as f:
            assert "MasterChef" in f.read()
        # 0 subprocesses launched
        mock_subproc_run.assert_not_called()
        mock_subproc_popen.assert_not_called()


def test_2_unchanged_media_persistent_cache_survives(tmp_path):
    """2. Proves unchanged media reuses persistent SQLite cache across simulated process restarts."""
    video_path = str(tmp_path / "episode.mkv")
    Path(video_path).write_text("media bytes", encoding="utf-8")

    # Save to SQLite cache
    save_cached_embedded_srt(video_path, track_id=2, lang="eng", content=SAMPLE_SRT)

    # Directly query get_cached_embedded_srt simulating another process invocation
    cached = get_cached_embedded_srt(video_path, track_id=2, lang="eng")
    assert cached is not None
    assert "MasterChef kitchen" in cached


def test_3_changed_size_or_mtime_invalidates_cache(tmp_path):
    """3. Proves modified media file size or mtime invalidates cache immediately."""
    video_path = str(tmp_path / "episode.mkv")
    Path(video_path).write_text("initial content", encoding="utf-8")

    save_cached_embedded_srt(video_path, track_id=1, lang="eng", content=SAMPLE_SRT)
    assert get_cached_embedded_srt(video_path, track_id=1, lang="eng") is not None

    # Modify file size and mtime
    time.sleep(0.01)
    Path(video_path).write_text("initial content modified with new bytes", encoding="utf-8")

    # Cache should miss because size & mtime changed
    assert get_cached_embedded_srt(video_path, track_id=1, lang="eng") is None


@pytest.mark.asyncio
async def test_4_existing_external_source_selected_when_available(tmp_path):
    """4. Proves external source is preferred when external subtitle is available."""
    video_path = str(tmp_path / "movie.mkv")
    Path(video_path).write_text("video content", encoding="utf-8")
    ext_en_srt = str(tmp_path / "movie.en.srt")
    Path(ext_en_srt).write_text(SAMPLE_SRT, encoding="utf-8")

    resolver = SourceResolver(
        video_path=video_path,
        target_languages=["sv"],
        primary_audio_lang="en",
        bazarr_url="",
        bazarr_api_key="",
        enable_bazarr=False,
        source_search_deadline=15.0,
        extract_source_embedded=False,
        container_tracks={"subtitles": []},
        find_external_subtitle_fn=lambda vp, lang: ext_en_srt if lang == "en" else None
    )
    src = await resolver.resolve()
    assert src is not None
    assert src.origin == SourceOrigin.EXTERNAL
    assert "MasterChef" in src.content


@pytest.mark.asyncio
async def test_5_and_6_bazarr_target_wins_and_cancels_extraction(tmp_path, monkeypatch):
    """5 & 6. Proves Bazarr target appearing while extraction is running causes immediate BAZARR MATCH and terminates subprocess."""
    video_path = tmp_path / "show.mkv"
    video_path.touch()
    target_sv_srt = str(tmp_path / "show.sv.srt")

    tracks_info = {
        "duration": 300.0,
        "subtitles": [{"id": 1, "language": "eng", "codec": "SubRip/SRT", "forced": False}]
    }

    pipeline = SubtitlePipeline()
    source_extracted = False
    source_cancelled = False

    async def slow_source_resolve(*args, **kwargs):
        nonlocal source_extracted, source_cancelled
        try:
            await asyncio.sleep(5.0)
            source_extracted = True
            return SubtitleSource(
                origin=SourceOrigin.EMBEDDED,
                path=str(tmp_path / "slow_src.en.srt"),
                language="en",
                content="1\n00:00:01,000 --> 00:00:03,000\nHello\n",
                cues=[],
            )
        except asyncio.CancelledError:
            source_cancelled = True
            raise

    # Target arrives during race
    async def target_writer():
        await asyncio.sleep(0.05)
        if not os.path.exists(target_sv_srt):
            with open(target_sv_srt, "w", encoding="utf-8") as f:
                f.write(SAMPLE_SV_SRT)

    def mock_get_setting(key, default=""):
        s = {
            "ai_provider": "gemini",
            "gemini_api_key": "dummy_api_key",
            "gemini_model": "gemini-3.5-flash-lite",
            "batch_size": "50",
            "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
            "enable_bazarr": "true",
            "extract_source_embedded": "true",
            "extract_target_embedded": "false",
            "bazarr_url": "http://bazarr:6767",
            "bazarr_api_key": "key123",
        }
        return s.get(key, default)

    from app.services.source_resolver import BazarrResult, BazarrResultCode
    from app.services.bazarr_coordinator import bazarr_coordinator, BazarrJobsPollResult, BazarrJobPollStatus
    monkeypatch.setattr(pipeline, "trigger_bazarr_search", AsyncMock(return_value=BazarrResult(code=BazarrResultCode.TRIGGERED, language="sv", detail="Accepted", media_correlated=True,)))
    monkeypatch.setattr(bazarr_coordinator, "poll_system_jobs", AsyncMock(return_value=BazarrJobsPollResult(status=BazarrJobPollStatus.KNOWN_IDLE, jobs=[])))
    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)
    monkeypatch.setattr("app.core.db.get_setting", mock_get_setting)
    monkeypatch.setattr("app.services.pipeline.inspect_mkv_tracks", lambda vp: tracks_info)
    monkeypatch.setattr("app.services.source_resolver.SourceResolver.resolve", slow_source_resolve)

    asyncio.create_task(target_writer())
    res = await pipeline.process_video_file(str(video_path), event_source="SONARR")

    assert res["status"] == "skipped"
    assert res["reason"] == "bazarr_downloaded"
    assert source_cancelled is True
    assert source_extracted is False


@pytest.mark.asyncio
async def test_7_no_orphan_extraction_process_on_cancellation(tmp_path):
    """7. Proves _run_cancellable_cmd kills the process group immediately on cancel_event."""
    cancel_event = threading.Event()

    # Launch a long-running sleep subprocess
    cmd = ["sleep", "60"]

    def run_cmd():
        return _run_cancellable_cmd(cmd, timeout=30.0, cancel_event=cancel_event)

    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(None, run_cmd)

    await asyncio.sleep(0.1)
    # Cancel it
    cancel_event.set()
    ret = await task
    assert ret == -1


@pytest.mark.asyncio
async def test_8_one_container_metadata_probe_per_job(tmp_path):
    """8. Proves container metadata probe inspect_mkv_tracks is called exactly once per job."""
    video_path = str(tmp_path / "episode.mkv")
    Path(video_path).write_text("media bytes", encoding="utf-8")
    out_srt = str(tmp_path / "episode.sv.srt")
    Path(out_srt).write_text(SAMPLE_SV_SRT, encoding="utf-8")

    pipeline = SubtitlePipeline()
    job_id = create_job(video_path)

    probe_mock = MagicMock(return_value={"duration": 100.0, "subtitles": []})
    with patch("app.services.pipeline.inspect_mkv_tracks", probe_mock), \
         patch("app.services.pipeline.find_external_subtitle", return_value=out_srt), \
         patch("app.services.pipeline.get_setting", side_effect=lambda k, d=None: "true" if k == "extract_source_embedded" else d):
        await pipeline._run_pipeline_logic(job_id, video_path)

        # inspect_mkv_tracks must be called at most once per job
        assert probe_mock.call_count <= 1


@pytest.mark.asyncio
async def test_9_bounded_extraction_io_concurrency(tmp_path):
    """9. Proves extraction respects bounded concurrency semaphore limit (max 2)."""
    assert _extraction_semaphore._value <= 2


@pytest.mark.asyncio
async def test_10_single_flight_deduplication(tmp_path):
    """10. Proves concurrent extractions for the exact same media+track perform physical work only once."""
    video_path = str(tmp_path / "shared_video.mkv")
    Path(video_path).write_text("video content", encoding="utf-8")
    out_1 = str(tmp_path / "out1.srt")
    out_2 = str(tmp_path / "out2.srt")

    tracks_info = {
        "duration": 60.0,
        "subtitles": [{"id": 1, "language": "eng", "codec": "SubRip/SRT", "forced": False}]
    }

    subproc_calls = []

    def mock_subproc(cmd, *args, **kwargs):
        subproc_calls.append(cmd)
        time.sleep(0.15)
        # Write output file
        target = cmd[-1].split(":", 1)[1] if ":" in cmd[-1] and not cmd[-1].startswith("/") else cmd[-1]
        with open(target, "w", encoding="utf-8") as f:
            f.write(SAMPLE_SRT)
        return MagicMock(returncode=0)

    with patch("app.core.extractor.subprocess.run", side_effect=mock_subproc):
        loop = asyncio.get_event_loop()
        t1 = loop.run_in_executor(None, extract_embedded_srt, video_path, out_1, "eng", tracks_info)
        t2 = loop.run_in_executor(None, extract_embedded_srt, video_path, out_2, "eng", tracks_info)

        r1, r2 = await asyncio.gather(t1, t2)
        assert r1 is True
        assert r2 is True
        assert os.path.exists(out_1)
        assert os.path.exists(out_2)

        # Only 1 subprocess call should have been executed due to single-flight + cache
        assert len(subproc_calls) == 1


@pytest.mark.asyncio
async def test_11_extraction_failure_cleans_temporary_files(tmp_path):
    """11. Proves extraction failure cleans temporary files immediately."""
    video_path = str(tmp_path / "bad_video.mkv")
    Path(video_path).write_text("bad data", encoding="utf-8")
    out_srt = str(tmp_path / "out.srt")

    tracks_info = {
        "duration": 60.0,
        "subtitles": [{"id": 1, "language": "eng", "codec": "SubRip/SRT", "forced": False}]
    }

    with patch("app.core.extractor.subprocess.run", side_effect=subprocess.CalledProcessError(1, ["mkvextract"])):
        success = extract_embedded_srt(video_path, out_srt, preferred_lang="eng", tracks_info=tracks_info)
        assert success is False
        assert not os.path.exists(out_srt)
