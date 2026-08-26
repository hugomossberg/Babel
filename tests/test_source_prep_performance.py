import pytest
import os
import srt
import asyncio
from datetime import timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from app.core.extractor import extract_embedded_srt, inspect_mkv_tracks
from app.services.pipeline import SubtitlePipeline
from app.core.db import get_job_by_id

def make_srt_string(cues):
    subs = []
    for i, (start_s, end_s, text) in enumerate(cues):
        subs.append(srt.Subtitle(
            index=i + 1,
            start=timedelta(seconds=start_s),
            end=timedelta(seconds=end_s),
            content=text
        ))
    return srt.compose(subs)

@pytest.fixture
def base_settings(monkeypatch):
    def mock_get_setting(key, default=""):
        settings = {
            "ai_provider": "gemini",
            "gemini_api_key": "test_key",
            "gemini_model": "gemini-3.5-flash-lite",
            "batch_size": "50",
            "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
            "enable_bazarr_check": "true",
            "clean_sdh": "true",
            "extract_source_embedded": "true",
            "extract_target_embedded": "true",
            "auto_repair_unhealthy": "true",
            # original_language_guard removed in v2.3.43 — kept for backward compat only
        }
        return settings.get(key, default)
    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)
    monkeypatch.setattr("app.services.translator.get_setting", mock_get_setting)

def test_1_normal_mkv_embedded_extraction(tmp_path):
    """1. Normal MKV embedded English extraction: mkvextract is now the primary tool
    for MKV files (uses seek table, faster than ffmpeg for large files)."""
    out_srt = str(tmp_path / "out.srt")
    tracks_info = {
        "subtitles": [{"id": 2, "language": "eng", "codec": "SubRip/SRT", "forced": False, "title": "English"}]
    }

    def mock_run(cmd, *args, **kwargs):
        # mkvextract writes to "track_id:path" — extract just the path
        target = cmd[-1]
        if ":" in target and not target.startswith("/"):
            target = target.split(":", 1)[1]
        with open(target, "w", encoding="utf-8") as f:
            f.write(make_srt_string([(1, 2, "Hello world")]))
        return MagicMock(returncode=0)

    with patch("app.core.extractor.subprocess.run", side_effect=mock_run) as mock_subprocess:
        res = extract_embedded_srt("movie.mkv", out_srt, preferred_lang="eng", tracks_info=tracks_info)
        assert res is True
        mock_subprocess.assert_called_once()
        cmd = mock_subprocess.call_args[0][0]
        # For MKV: mkvextract is now the primary tool
        assert cmd[0] == "mkvextract"
        # mkvextract format: mkvextract tracks <video> <track_id>:<output_path>
        assert "2:" in cmd[-1]  # track id 2 with output path

def test_2_source_track_selection_priority(tmp_path):
    """2 & 3. Source track selection: normal dialogue preferred over SDH, forced skipped.
    Track 4 (Normal Dialogue, default=True) scores highest: 100 + 20 (dialogue) + 10 (default) = 130
    Track 3 (SDH) scores: 100 + 20 (sdh) = 120. Track 1 is forced and skipped. Track 2 is wrong lang.
    For MKV files mkvextract is used as the primary tool, format: mkvextract tracks <video> <track_id>:<output_path>
    """
    out_srt = str(tmp_path / "out.srt")
    tracks_info = {
        "subtitles": [
            {"id": 1, "language": "eng", "codec": "SubRip/SRT", "forced": True, "title": "English Forced"},
            {"id": 2, "language": "dan", "codec": "SubRip/SRT", "forced": False, "title": "Danish"},
            {"id": 3, "language": "eng", "codec": "SubRip/SRT", "forced": False, "title": "English (SDH)"},
            {"id": 4, "language": "eng", "codec": "SubRip/SRT", "forced": False, "title": "English (Normal Dialogue)", "default": True}
        ]
    }

    def mock_run(cmd, *args, **kwargs):
        target = cmd[-1]
        if ":" in target and not target.startswith("/"):
            target = target.split(":", 1)[1]
        with open(target, "w", encoding="utf-8") as f:
            f.write(make_srt_string([(1, 2, "Test")]))
        return MagicMock(returncode=0)

    with patch("app.core.extractor.subprocess.run", side_effect=mock_run) as mock_subprocess:
        res = extract_embedded_srt("movie.mkv", out_srt, preferred_lang="eng", tracks_info=tracks_info)
        assert res is True
        cmd = mock_subprocess.call_args[0][0]
        # For MKV: mkvextract, track id=4 (highest score) in format "4:<output_path>"
        assert cmd[0] == "mkvextract"
        assert cmd[-1].startswith("4:")

def test_4_mp4_embedded_subtitles(tmp_path):
    """4 & 5. MP4 embedded subtitles support via ffmpeg."""
    out_srt = str(tmp_path / "out.srt")
    tracks_info = {
        "subtitles": [{"id": 0, "language": "eng", "codec": "mov_text", "forced": False, "title": "CC"}]
    }
    with patch("app.core.extractor.subprocess.run") as mock_run, \
         patch("app.core.extractor.os.path.exists", return_value=True), \
         patch("app.core.extractor.os.path.getsize", return_value=500), \
         patch("builtins.open", MagicMock(return_value=MagicMock(__enter__=lambda s: MagicMock(read=lambda: make_srt_string([(1, 2, "MP4 text")]))))):
        
        mock_run.return_value = MagicMock(returncode=0)
        res = extract_embedded_srt("video.mp4", out_srt, preferred_lang="eng", tracks_info=tracks_info)
        assert res is True
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"

@pytest.mark.asyncio
async def test_6_one_probe_per_job(base_settings, tmp_path, monkeypatch):
    """11 & 12. Verify ONE probe per job and no redundant extractions."""
    video = tmp_path / "test.mkv"
    video.touch()
    
    pipeline = SubtitlePipeline()
    monkeypatch.setattr(pipeline.translator, "translate_srt_content", AsyncMock(return_value=[]))
    monkeypatch.setattr(pipeline, "trigger_bazarr_search", AsyncMock())
    monkeypatch.setattr("app.services.pipeline.qa_gate", lambda *args, **kwargs: {"passed": True, "score": 100})
    
    probe_call_count = 0
    def mock_inspect(vp):
        nonlocal probe_call_count
        probe_call_count += 1
        return {
            "subtitles": [{"id": 2, "language": "eng", "codec": "SubRip/SRT", "forced": False}],
            "audio": [{"id": 1, "language": "eng", "default": True, "forced": False}],
            "duration": 120.0
        }
    monkeypatch.setattr("app.services.pipeline.inspect_mkv_tracks", mock_inspect)
    monkeypatch.setattr("app.core.extractor.inspect_mkv_tracks", mock_inspect)
    
    extract_call_count = 0
    def mock_extract(vp, out, preferred_lang, tracks_info=None):
        nonlocal extract_call_count
        extract_call_count += 1
        if preferred_lang == "sv":
            return False  # Target sv not embedded
        with open(out, "w", encoding="utf-8") as f:
            f.write(make_srt_string([(1, 3, "Line 1"), (4, 6, "Line 2")]))
        return True
    monkeypatch.setattr("app.services.pipeline.extract_embedded_srt", mock_extract)
    
    await pipeline._run_pipeline_logic(1, str(video), wait_seconds=0)
    
    # inspect_mkv_tracks must be called EXACTLY ONCE
    assert probe_call_count == 1, f"Expected exactly 1 container probe, got {probe_call_count}"
    # Target check (sv) + Source check (eng) -> target failed fast, source extracted once
    assert extract_call_count == 2

@pytest.mark.asyncio
async def test_13_source_equals_target_skips_ai(base_settings, tmp_path, monkeypatch):
    """13. SOURCE==TARGET shortcut: if source IS the target language, publish directly — no AI needed.
    
    v2.3.43: OLG removed as a blocker. The SOURCE==TARGET invariant enforces that
    source_language != target_language for every AI dispatch. If source IS English
    and target IS English, the subtitle is published directly without translation.
    """
    video = tmp_path / "english_audio.mkv"
    video.touch()

    # Configure target language to English
    def mock_get_setting(key, default=""):
        if key == "languages":
            return '[{"code": "en", "name": "English", "enabled": true}]'
        # original_language_guard is ignored in v2.3.43
        return default
    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)

    pipeline = SubtitlePipeline()
    translate_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(pipeline.translator, "translate_srt_content", translate_mock)

    def mock_inspect(vp):
        return {
            "subtitles": [{"id": 2, "language": "eng", "codec": "SubRip/SRT", "forced": False}],
            "audio": [{"id": 1, "language": "eng", "default": True, "forced": False}],
            "duration": 120.0
        }
    monkeypatch.setattr("app.services.pipeline.inspect_mkv_tracks", mock_inspect)
    monkeypatch.setattr("app.core.extractor.inspect_mkv_tracks", mock_inspect)

    # Use realistic English content so language detection returns "en"
    ENGLISH_SRT = make_srt_string([
        (0, 2, "Hello, welcome to the show."),
        (2, 4, "We are glad to have you here."),
        (4, 6, "The weather is nice today."),
        (6, 8, "Please take a seat."),
        (8, 10, "Thank you for joining us."),
        (10, 12, "We will start shortly."),
        (12, 14, "Stay tuned for more."),
        (14, 16, "This is an important announcement."),
        (16, 18, "Please listen carefully."),
        (18, 20, "We appreciate your patience."),
    ])

    def mock_extract(vp, out, preferred_lang, tracks_info=None):
        with open(out, "w", encoding="utf-8") as f:
            f.write(ENGLISH_SRT)
        return True
    monkeypatch.setattr("app.services.pipeline.extract_embedded_srt", mock_extract)

    res = await pipeline._run_pipeline_logic(1, str(video), wait_seconds=0)

    # SOURCE==TARGET shortcut: source is English, target is English
    # → subtitle published directly, AI translation skipped
    assert translate_mock.call_count == 0, (
        f"AI was called {translate_mock.call_count} times but source IS the target language"
    )

def test_14_mkvextract_corrupt_output_triggers_ffmpeg_fallback(tmp_path):
    """v2.3.43: For MKV files mkvextract is now the PRIMARY tool.
    If mkvextract produces corrupt/unparseable output, ffmpeg is tried as fallback."""
    out_srt = str(tmp_path / "fallback.srt")
    tracks_info = {
        "subtitles": [{"id": 7, "language": "eng", "codec": "SubRip/SRT", "forced": False, "title": "English"}]
    }

    call_history = []
    def mock_subprocess(cmd, *args, **kwargs):
        call_history.append(cmd[0])
        if cmd[0] == "mkvextract":
            # Simulate mkvextract creating a corrupted/unparseable file
            with open(out_srt, "w", encoding="utf-8") as f:
                f.write("THIS IS NOT VALID SRT CONTENT AT ALL AND CANNOT BE PARSED")
            return MagicMock(returncode=0)
        elif cmd[0] == "ffmpeg":
            # Simulate ffmpeg successfully creating valid SRT as fallback
            with open(out_srt, "w", encoding="utf-8") as f:
                f.write(make_srt_string([(1, 4, "Clean extracted subtitle from ffmpeg fallback")]))
            return MagicMock(returncode=0)
        return MagicMock(returncode=0)

    with patch("app.core.extractor.subprocess.run", side_effect=mock_subprocess):
        res = extract_embedded_srt("corrupt_stream.mkv", out_srt, preferred_lang="eng", tracks_info=tracks_info)
        assert res is True
        # New order: mkvextract first (primary for MKV), ffmpeg second (fallback)
        assert call_history == ["mkvextract", "ffmpeg"]
        with open(out_srt, "r", encoding="utf-8") as f:
            subs = list(srt.parse(f.read()))
        assert len(subs) == 1
        assert subs[0].content == "Clean extracted subtitle from ffmpeg fallback"
