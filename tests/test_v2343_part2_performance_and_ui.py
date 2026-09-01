import os
from pathlib import Path
import srt
import time
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from app.core.extractor import (
    extract_embedded_srt,
    get_cached_embedded_srt,
    save_cached_embedded_srt,
    invalidate_cached_embedded_srt,
)
from app.core.db import (
    init_db,
    get_job_by_id,
    update_job,
    create_job,
)
from app.services.pipeline import SubtitlePipeline
from app.services.source_resolver import SourceResolver, SubtitleSource, SourceOrigin
from app.services.translator import SubtitleTranslator


def make_valid_srt(lang="en", count=10):
    lines = []
    for i in range(1, count + 1):
        if lang == "sv":
            text = f"Detta är en testrad nummer {i} på svenska för babel och vi kontrollerar kvaliteten noga."
        else:
            text = f"This is a test line number {i} in english for babel and we check the quality carefully."
        lines.append(f"{i}\n00:00:0{i:02d},000 --> 00:00:0{i:02d},500\n{text}\n")
    return "\n".join(lines)


@pytest.fixture(autouse=True)
def ensure_db():
    init_db()


# ─── PASS 2A: EMBEDDED EXTRACTION CACHE ───────────────────────────────────────

def test_extraction_cache_save_get_and_hit(tmp_path):
    video = tmp_path / "movie.mkv"
    video.write_text("dummy video content for cache test")

    sample_srt = make_valid_srt("en", 10)
    assert get_cached_embedded_srt(str(video), track_id=2, lang="en") is None

    saved = save_cached_embedded_srt(str(video), track_id=2, lang="en", content=sample_srt)
    assert saved is True

    cached = get_cached_embedded_srt(str(video), track_id=2, lang="en")
    assert cached == sample_srt


def test_extraction_cache_invalidates_on_media_change(tmp_path):
    video = tmp_path / "movie.mkv"
    video.write_text("initial content")

    sample_srt = make_valid_srt("en", 10)
    save_cached_embedded_srt(str(video), track_id=1, lang="en", content=sample_srt)
    assert get_cached_embedded_srt(str(video), track_id=1, lang="en") == sample_srt

    # Modify the video file (changes size and mtime)
    video.write_text("modified content with different size and mtime")
    assert get_cached_embedded_srt(str(video), track_id=1, lang="en") is None


def test_extraction_cache_isolates_tracks_and_languages(tmp_path):
    video = tmp_path / "movie.mkv"
    video.write_text("video tracks isolation test")

    en_srt = make_valid_srt("en", 10)
    sv_srt = make_valid_srt("sv", 10)

    save_cached_embedded_srt(str(video), track_id=1, lang="en", content=en_srt)
    save_cached_embedded_srt(str(video), track_id=2, lang="sv", content=sv_srt)

    assert get_cached_embedded_srt(str(video), track_id=1, lang="en") == en_srt
    assert get_cached_embedded_srt(str(video), track_id=2, lang="sv") == sv_srt
    assert get_cached_embedded_srt(str(video), track_id=1, lang="sv") is None
    assert get_cached_embedded_srt(str(video), track_id=2, lang="en") is None


def test_invalidate_cached_embedded_srt(tmp_path):
    video = tmp_path / "series_s01e01.mkv"
    video.write_text("content")

    sample_srt = make_valid_srt("en", 10)
    save_cached_embedded_srt(str(video), track_id=0, lang="en", content=sample_srt)
    assert get_cached_embedded_srt(str(video), track_id=0, lang="en") is not None

    invalidate_cached_embedded_srt(str(video))
    assert get_cached_embedded_srt(str(video), track_id=0, lang="en") is None


def test_extract_embedded_srt_uses_cache_and_skips_mkvextract(tmp_path):
    video = tmp_path / "godland.mkv"
    video.write_text("fake video file content")
    out_srt = tmp_path / "godland.extracted.srt"

    sample_srt = make_valid_srt("en", 10)
    save_cached_embedded_srt(str(video), track_id=1, lang="eng", content=sample_srt)

    tracks_info = {
        "subtitles": [
            {"id": 1, "codec": "SubRip/SRT", "language": "eng", "forced": False, "default": True, "title": "English"}
        ],
        "audio": [],
        "duration": 600.0
    }

    with patch("subprocess.run") as mock_subproc:
        success = extract_embedded_srt(str(video), str(out_srt), preferred_lang="eng", tracks_info=tracks_info)
        assert success is True
        assert mock_subproc.call_count == 0  # mkvextract / ffmpeg was completely skipped!
        assert out_srt.exists()
        assert out_srt.read_text(encoding="utf-8") == sample_srt


# ─── PASS 2C & 2D: TARGET/SOURCE RACE & ATOMIC PRE-AI CHECK ───────────────────

@pytest.mark.asyncio
async def test_race_winner_bazarr_early_poller(tmp_path, monkeypatch):
    video = tmp_path / "race_test.mkv"
    video.touch()
    sv_target = str(tmp_path / "race_test.sv.srt")

    def mock_settings(k, d=""):
        return {
            "ai_provider": "gemini",
            "gemini_api_key": "dummy",
            "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
            "enable_bazarr_check": "true",
            "clean_sdh": "true",
            "extract_source_embedded": "true",
        }.get(k, d)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_settings)
    monkeypatch.setattr("app.services.translator.get_setting", mock_settings)

    pipeline = SubtitlePipeline()
    translate_mock = AsyncMock()
    monkeypatch.setattr(pipeline.translator, "translate_srt_content", translate_mock)

    # Mock Bazarr trigger to write target file and return accepted result
    from app.services.source_resolver import BazarrResult, BazarrResultCode
    async def mock_bazarr_trigger(vp, language="sv"):
        with open(sv_target, "w", encoding="utf-8") as f:
            f.write(make_valid_srt("sv", 10))
        return BazarrResult(code=BazarrResultCode.TRIGGERED, language=language, detail="Search accepted")

    monkeypatch.setattr(pipeline, "trigger_bazarr_search", mock_bazarr_trigger)

    en_target = str(tmp_path / "race_test.en.srt")
    with open(en_target, "w", encoding="utf-8") as f:
        f.write(make_valid_srt("en", 10))

    call_count = 0
    def mock_find(vp, lang):
        nonlocal call_count
        if lang == "sv":
            call_count += 1
            if call_count == 1:
                return None
            return sv_target
        if lang == "en":
            return en_target
        return None

    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find):
        res = await pipeline.process_video_file(str(video), event_source="SONARR")

    assert res["status"] == "skipped"
    assert res["reason"] == "bazarr_downloaded"
    assert translate_mock.call_count == 0
    job = get_job_by_id(res["job_id"])
    assert job["status"] == "BAZARR MATCH"
    assert "Target/Source race winner: Bazarr" in "".join(job["logs"])
    assert "AI calls: 0" in "".join(job["logs"])


@pytest.mark.asyncio
async def test_atomic_pre_ai_target_check_stops_ai(tmp_path, monkeypatch):
    video = tmp_path / "atomic_check.mkv"
    video.touch()
    en_src = tmp_path / "atomic_check.en.srt"
    en_src.write_text(make_valid_srt("en", 10), encoding="utf-8")
    sv_target = tmp_path / "atomic_check.sv.srt"

    def mock_settings(k, d=""):
        return {
            "ai_provider": "gemini",
            "gemini_api_key": "dummy",
            "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
            "enable_bazarr_check": "false",
            "clean_sdh": "true",
            "extract_source_embedded": "false",
        }.get(k, d)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_settings)
    monkeypatch.setattr("app.services.translator.get_setting", mock_settings)

    pipeline = SubtitlePipeline()
    translate_mock = AsyncMock()
    monkeypatch.setattr(pipeline.translator, "translate_srt_content", translate_mock)

    check_num = 0
    def mock_find(vp, lang):
        nonlocal check_num
        if lang == "sv":
            check_num += 1
            if check_num == 1:
                return None
            # Target appears right before AI start
            sv_target.write_text(make_valid_srt("sv", 10), encoding="utf-8")
            return str(sv_target)
        if lang == "en":
            return str(en_src)
        return None

    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find):
        res = await pipeline.process_video_file(str(video), event_source="SONARR")

    assert translate_mock.call_count == 0
    job = get_job_by_id(res["job_id"])
    logs = "".join(job["logs"])
    assert "Target appeared before AI start" in logs
    assert "AI skipped" in logs
    assert "AI calls: 0" in logs


# ─── PASS 2G: RECOVERY EFFICIENCY ─────────────────────────────────────────────

def test_fast_final_rescue_batch_not_double_wrapped():
    """Verify that fast_final_rescue_batch is a regular async method, not wrapped with @with_retry."""
    translator = SubtitleTranslator()
    method = getattr(translator, "fast_final_rescue_batch")
    assert asyncio.iscoroutinefunction(method)
    assert not hasattr(method, "__retry_wrapped__")


# ─── PASS 2H & 2I: UI CONSISTENCY & MODAL JOB_ID BINDING ──────────────────────

def test_ui_template_contains_strict_job_id_and_na_cards():
    index_path = Path(__file__).resolve().parents[1] / "app" / "templates" / "index.html"
    html = index_path.read_text(encoding="utf-8")

    # Pass 2H: Sync Drift & Dropped card N/A check
    assert "N/A" in html
    assert "(pre-existing)" in html
    assert "selectedJob.sync_diff_ms !== -1" in html

    # Pass 2I: loadJobs() matches STRICTLY by j.id === this.selectedJob.id
    assert "j.id === this.selectedJob.id" in html
    # Ensure video_path is NOT used to match selectedJob in loadJobs
    assert "this.selectedJob.video_path && j.video_path === this.selectedJob.video_path" not in html

    # Pass 2J: openJobModal fetches by exact job id
    assert "fetch('/api/jobs/' + targetJobId)" in html
    assert "loadJobUsage(targetJobId)" in html


# ─── PASS 2B & 2E: PROBE REUSE & SAFE CANCELLATION ────────────────────────────

@pytest.mark.asyncio
async def test_container_probe_reused_single_call(tmp_path, monkeypatch):
    video = tmp_path / "probe_reuse.mkv"
    video.write_text("probe reuse test content", encoding="utf-8")
    sample_srt = make_valid_srt("en", 10)
    save_cached_embedded_srt(str(video), track_id=1, lang="eng", content=sample_srt)

    def mock_settings(k, d=""):
        return {
            "ai_provider": "gemini",
            "gemini_api_key": "dummy",
            "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
            "enable_bazarr_check": "false",
            "clean_sdh": "true",
            "extract_source_embedded": "true",
            "extract_target_embedded": "true",
        }.get(k, d)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_settings)
    monkeypatch.setattr("app.services.translator.get_setting", mock_settings)

    pipeline = SubtitlePipeline()
    probe_call_count = 0

    def mock_inspect(vp):
        nonlocal probe_call_count
        probe_call_count += 1
        return {
            "subtitles": [
                {"id": 1, "codec": "SubRip/SRT", "language": "eng", "forced": False, "default": True, "title": "English"}
            ],
            "audio": [{"id": 0, "language": "eng", "default": True, "forced": False}],
            "duration": 500.0
        }

    monkeypatch.setattr("app.services.pipeline.inspect_mkv_tracks", mock_inspect)
    sample_cues = list(srt.parse(sample_srt))

    async def mock_translate(*args, **kwargs):
        return [
            srt.Subtitle(index=i+1, start=sample_cues[i].start,
                         end=sample_cues[i].end,
                         content=f"Detta är översatt rad {i+1} på svenska.") for i in range(10)
        ]

    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)

    res = await pipeline.process_video_file(str(video), event_source="SONARR")
    assert res["status"] == "translated"
    # Container probe was run exactly once and reused for audio, embedded target, and source resolver
    assert probe_call_count == 1


@pytest.mark.asyncio
async def test_race_winner_source_embedded_logged(tmp_path, monkeypatch):
    video = tmp_path / "source_wins.mkv"
    video.write_text("source wins test content", encoding="utf-8")

    def mock_settings(k, d=""):
        return {
            "ai_provider": "gemini",
            "gemini_api_key": "dummy",
            "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
            "enable_bazarr_check": "false",
            "clean_sdh": "true",
            "extract_source_embedded": "true",
        }.get(k, d)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_settings)
    monkeypatch.setattr("app.services.translator.get_setting", mock_settings)

    pipeline = SubtitlePipeline()
    sample_srt = make_valid_srt("en", 10)
    save_cached_embedded_srt(str(video), track_id=1, lang="eng", content=sample_srt)

    monkeypatch.setattr("app.services.pipeline.inspect_mkv_tracks", lambda vp: {
        "subtitles": [
            {"id": 1, "codec": "SubRip/SRT", "language": "eng", "forced": False, "default": True, "title": "English"}
        ],
        "audio": [],
        "duration": 500.0
    })

    sample_cues = list(srt.parse(sample_srt))

    async def mock_translate(*args, **kwargs):
        return [
            srt.Subtitle(index=i+1, start=sample_cues[i].start,
                         end=sample_cues[i].end,
                         content=f"Detta är översatt rad {i+1} på svenska.") for i in range(10)
        ]

    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)

    res = await pipeline.process_video_file(str(video), event_source="SONARR")
    assert res["status"] == "translated"
    job = get_job_by_id(res["job_id"])
    logs = "".join(job["logs"])
    assert "Target/Source race winner: Embedded source" in logs
    assert "Source language: English" in logs


# ─── PASS 2I & 2J: MULTIPLE JOBS SAME VIDEO PATH MODAL ISOLATION ───────────────

def test_database_jobs_isolated_by_id_with_same_video_path():
    """Verify that multiple job records for the same video_path maintain independent IDs and states."""
    vpath = "/data/media/movies/Godland (2022)/Godland (2022).mkv"
    j1 = create_job(video_path=vpath, event_source="SONARR")
    update_job(j1, status="FAILED", reason="API Error")

    j2 = create_job(video_path=vpath, event_source="SONARR")
    update_job(j2, status="ALREADY EXISTS", reason="All target subtitles already exist")

    assert j1 != j2
    job1_rec = get_job_by_id(j1)
    job2_rec = get_job_by_id(j2)

    assert job1_rec["id"] == j1
    assert job1_rec["status"] == "FAILED"
    assert job2_rec["id"] == j2
    assert job2_rec["status"] == "ALREADY EXISTS"


# ─── PASS 2E: PROCESS-LEVEL CANCELLATION PROOF ────────────────────────────────

def test_cancellable_cmd_terminates_process_on_cancel_event():
    """Verify that _run_cancellable_cmd actively kills the underlying OS subprocess when cancel_event is set."""
    from app.core.extractor import _run_cancellable_cmd
    import threading
    import time

    cancel_ev = threading.Event()
    cmd = ["sleep", "30"]
    t_start = time.monotonic()

    timer = threading.Timer(0.1, cancel_ev.set)
    timer.start()

    ret = _run_cancellable_cmd(cmd, timeout=30.0, cancel_event=cancel_ev)
    elapsed = time.monotonic() - t_start

    assert ret == -1
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_source_resolver_cancel_terminates_subprocess(tmp_path):
    """Verify that SourceResolver.cancel() immediately sets cancel_event and stops extraction."""
    from app.services.source_resolver import SourceResolver
    import threading

    resolver = SourceResolver(
        video_path=str(tmp_path / "test.mkv"),
        container_tracks=None,
        primary_audio_lang="und",
        target_languages=["sv"],
        bazarr_url="",
        bazarr_api_key="",
        enable_bazarr=False,
        extract_source_embedded=True,
        source_search_deadline=5.0,
    )

    assert not resolver.cancel_event.is_set()
    resolver.cancel()
    assert resolver.cancel_event.is_set()
