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
            "original_language_guard": "true"
        }
        return settings.get(key, default)
    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)
    monkeypatch.setattr("app.services.translator.get_setting", mock_get_setting)

def test_1_normal_mkv_embedded_extraction(tmp_path):
    """1. Normal MKV embedded English extraction with ffmpeg."""
    out_srt = str(tmp_path / "out.srt")
    tracks_info = {
        "subtitles": [{"id": 2, "language": "eng", "codec": "SubRip/SRT", "forced": False, "title": "English"}]
    }
    with patch("app.core.extractor.subprocess.run") as mock_run, \
         patch("app.core.extractor.os.path.exists", return_value=True), \
         patch("app.core.extractor.os.path.getsize", return_value=500), \
         patch("builtins.open", MagicMock(return_value=MagicMock(__enter__=lambda s: MagicMock(read=lambda: make_srt_string([(1, 2, "Hello world")]))))):
        
        mock_run.return_value = MagicMock(returncode=0)
        res = extract_embedded_srt("movie.mkv", out_srt, preferred_lang="eng", tracks_info=tracks_info)
        assert res is True
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-map" in cmd
        assert "0:s:0" in cmd

def test_2_source_track_selection_priority(tmp_path):
    """2 & 3. Source track selection: normal dialogue preferred over SDH, forced skipped."""
    out_srt = str(tmp_path / "out.srt")
    tracks_info = {
        "subtitles": [
            {"id": 1, "language": "eng", "codec": "SubRip/SRT", "forced": True, "title": "English Forced"},
            {"id": 2, "language": "dan", "codec": "SubRip/SRT", "forced": False, "title": "Danish"},
            {"id": 3, "language": "eng", "codec": "SubRip/SRT", "forced": False, "title": "English (SDH)"},
            {"id": 4, "language": "eng", "codec": "SubRip/SRT", "forced": False, "title": "English (Normal Dialogue)", "default": True}
        ]
    }
    with patch("app.core.extractor.subprocess.run") as mock_run, \
         patch("app.core.extractor.os.path.exists", return_value=True), \
         patch("app.core.extractor.os.path.getsize", return_value=500), \
         patch("builtins.open", MagicMock(return_value=MagicMock(__enter__=lambda s: MagicMock(read=lambda: make_srt_string([(1, 2, "Test")]))))):
        
        mock_run.return_value = MagicMock(returncode=0)
        res = extract_embedded_srt("movie.mkv", out_srt, preferred_lang="eng", tracks_info=tracks_info)
        assert res is True
        # Track 4 is at index 3 in subtitles list (highest score: score 100 + 20 dialogue + 10 default = 130 vs SDH 120)
        cmd = mock_run.call_args[0][0]
        assert "0:s:3" in cmd

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
async def test_13_audio_guard_uses_cached_tracks(base_settings, tmp_path, monkeypatch):
    """13. Audio language guard skips translation to English when primary audio is English."""
    video = tmp_path / "english_audio.mkv"
    video.touch()

    # Configure target language to English
    def mock_get_setting(key, default=""):
        if key == "languages":
            return '[{"code": "en", "name": "English", "enabled": true}]'
        if key == "original_language_guard":
            return "true"
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

    def mock_extract(vp, out, preferred_lang, tracks_info=None):
        with open(out, "w", encoding="utf-8") as f:
            f.write(make_srt_string([(1, 3, "Dialogue")]))
        return True
    monkeypatch.setattr("app.services.pipeline.extract_embedded_srt", mock_extract)

    res = await pipeline._run_pipeline_logic(1, str(video), wait_seconds=0)

    # Guard should skip translation because primary audio is English
    assert translate_mock.call_count == 0

def test_14_ffmpeg_corrupt_output_triggers_mkvextract_fallback(tmp_path):
    """5. Verify ffmpeg invalid/empty output triggers fallback to mkvextract."""
    out_srt = str(tmp_path / "fallback.srt")
    tracks_info = {
        "subtitles": [{"id": 7, "language": "eng", "codec": "SubRip/SRT", "forced": False, "title": "English"}]
    }

    call_history = []
    def mock_subprocess(cmd, *args, **kwargs):
        call_history.append(cmd[0])
        if cmd[0] == "ffmpeg":
            # Simulate ffmpeg creating a corrupted/unparseable file
            with open(out_srt, "w", encoding="utf-8") as f:
                f.write("THIS IS NOT VALID SRT CONTENT AT ALL AND CANNOT BE PARSED")
            return MagicMock(returncode=0)
        elif cmd[0] == "mkvextract":
            # Simulate mkvextract successfully creating valid SRT
            with open(out_srt, "w", encoding="utf-8") as f:
                f.write(make_srt_string([(1, 4, "Clean extracted subtitle from fallback")]))
            return MagicMock(returncode=0)
        return MagicMock(returncode=0)

    with patch("app.core.extractor.subprocess.run", side_effect=mock_subprocess):
        res = extract_embedded_srt("corrupt_stream.mkv", out_srt, preferred_lang="eng", tracks_info=tracks_info)
        assert res is True
        assert call_history == ["ffmpeg", "mkvextract"]
        with open(out_srt, "r", encoding="utf-8") as f:
            subs = list(srt.parse(f.read()))
        assert len(subs) == 1
        assert subs[0].content == "Clean extracted subtitle from fallback"
