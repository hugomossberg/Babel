import pytest
import os
import subprocess
from unittest.mock import patch, MagicMock
from app.core.extractor import extract_embedded_srt

def _create_mock_srt(out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("1\n00:00:01,000 --> 00:00:03,000\nHello\n")

@patch("app.core.extractor.inspect_mkv_tracks")
def test_track_selection_ffmpeg(mock_inspect, tmp_path):
    tracks_info = {
        "subtitles": [
            {
                "id": 1,
                "language": "eng",
                "codec": "SubRip/SRT",
                "forced": False,
                "title": "English Signs & Songs"
            },
            {
                "id": 2,
                "language": "eng",
                "codec": "SubRip/SRT",
                "forced": False,
                "title": "English Full SDH"
            }
        ]
    }
    mock_inspect.return_value = tracks_info
    out_srt = str(tmp_path / "out.srt")

    def mock_subprocess(cmd, *args, **kwargs):
        _create_mock_srt(cmd[-1])
        return MagicMock(returncode=0)

    with patch("app.core.extractor.subprocess.run", side_effect=mock_subprocess) as mock_run:
        res = extract_embedded_srt("fake.mkv", out_srt, preferred_lang="eng")
        assert res is True
        # check that ffmpeg was called with stream index 1 (-map 0:s:1)
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "ffmpeg"
        assert "-map" in args
        map_idx = args.index("-map")
        assert args[map_idx + 1] == "0:s:1"
        assert out_srt in args

@patch("app.core.extractor.inspect_mkv_tracks")
def test_vtt_fast_ffmpeg_conversion(mock_inspect, tmp_path):
    tracks_info = {
        "subtitles": [
            {
                "id": 1,
                "language": "eng",
                "codec": "S_TEXT/WEBVTT",
                "forced": False,
                "title": "English"
            }
        ]
    }
    mock_inspect.return_value = tracks_info
    out_srt = str(tmp_path / "out.srt")

    def mock_subprocess(cmd, *args, **kwargs):
        _create_mock_srt(cmd[-1])
        return MagicMock(returncode=0)

    with patch("app.core.extractor.subprocess.run", side_effect=mock_subprocess) as mock_run:
        res = extract_embedded_srt("fake.mkv", out_srt, preferred_lang="eng")
        assert res is True
        # ffmpeg extracts and converts in one step
        mock_run.assert_called_once()
        ffmpeg_cmd = mock_run.call_args[0][0]
        assert ffmpeg_cmd[0] == "ffmpeg"
        assert ffmpeg_cmd[1] == "-y"
        assert "-map" in ffmpeg_cmd
        assert "-c:s" in ffmpeg_cmd
        assert "srt" in ffmpeg_cmd

@patch("app.core.extractor.inspect_mkv_tracks")
def test_mkvextract_fallback_on_ffmpeg_failure(mock_inspect, tmp_path):
    tracks_info = {
        "subtitles": [
            {
                "id": 5,
                "language": "eng",
                "codec": "SubRip/SRT",
                "forced": False,
                "title": "English"
            }
        ]
    }
    mock_inspect.return_value = tracks_info
    out_srt = str(tmp_path / "out.srt")

    # First call (ffmpeg) fails, second call (mkvextract) succeeds
    def mock_subprocess(cmd, *args, **kwargs):
        if cmd[0] == "ffmpeg":
            raise subprocess.CalledProcessError(1, cmd)
        elif cmd[0] == "mkvextract":
            _create_mock_srt(out_srt)
            return MagicMock(returncode=0)
        return MagicMock(returncode=0)

    with patch("app.core.extractor.subprocess.run", side_effect=mock_subprocess) as mock_run:
        res = extract_embedded_srt("fake.mkv", out_srt, preferred_lang="eng")
        assert res is True
        assert mock_run.call_count == 2
        ffmpeg_cmd = mock_run.call_args_list[0][0][0]
        mkv_cmd = mock_run.call_args_list[1][0][0]
        assert ffmpeg_cmd[0] == "ffmpeg"
        assert mkv_cmd[0] == "mkvextract"
        assert f"5:{out_srt}" in mkv_cmd[3]

def test_tracks_info_caching_skips_probe(tmp_path):
    tracks_info = {
        "subtitles": [
            {
                "id": 1,
                "language": "eng",
                "codec": "SubRip/SRT",
                "forced": False,
                "title": "English"
            }
        ]
    }
    out_srt = str(tmp_path / "out.srt")

    def mock_subprocess(cmd, *args, **kwargs):
        _create_mock_srt(cmd[-1])
        return MagicMock(returncode=0)

    with patch("app.core.extractor.inspect_mkv_tracks") as mock_inspect:
        with patch("app.core.extractor.subprocess.run", side_effect=mock_subprocess) as mock_run:
            res = extract_embedded_srt("fake.mkv", out_srt, preferred_lang="eng", tracks_info=tracks_info)
            assert res is True
            # inspect_mkv_tracks must NOT be called when tracks_info is provided
            mock_inspect.assert_not_called()
            mock_run.assert_called_once()
