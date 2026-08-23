import pytest
import os
import subprocess
from unittest.mock import patch, MagicMock
from app.core.extractor import extract_embedded_srt

@patch("app.core.extractor.inspect_mkv_tracks")
@patch("app.core.extractor.subprocess.run")
@patch("app.core.extractor.os.path.exists")
@patch("app.core.extractor.os.path.getsize")
def test_track_selection_ffmpeg(mock_getsize, mock_exists, mock_run, mock_inspect):
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
    
    mock_exists.return_value = True
    mock_getsize.return_value = 500
    
    extract_embedded_srt("fake.mkv", "out.srt", preferred_lang="eng")
    
    # check that ffmpeg was called with stream index 1 (-map 0:s:1)
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args[0] == "ffmpeg"
    assert "-map" in args
    map_idx = args.index("-map")
    assert args[map_idx + 1] == "0:s:1"
    assert "out.srt" in args

@patch("app.core.extractor.inspect_mkv_tracks")
@patch("app.core.extractor.subprocess.run")
@patch("app.core.extractor.os.path.exists")
@patch("app.core.extractor.os.path.getsize")
def test_vtt_fast_ffmpeg_conversion(mock_getsize, mock_exists, mock_run, mock_inspect):
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
    
    mock_exists.return_value = True
    mock_getsize.return_value = 500
    
    class MockProcess:
        returncode = 0
    mock_run.return_value = MockProcess()
    
    extract_embedded_srt("fake.mkv", "out.srt", preferred_lang="eng")
    
    # ffmpeg extracts and converts in one step
    mock_run.assert_called_once()
    ffmpeg_cmd = mock_run.call_args[0][0]
    assert ffmpeg_cmd[0] == "ffmpeg"
    assert ffmpeg_cmd[1] == "-y"
    assert "-map" in ffmpeg_cmd
    assert "-c:s" in ffmpeg_cmd
    assert "srt" in ffmpeg_cmd

@patch("app.core.extractor.inspect_mkv_tracks")
@patch("app.core.extractor.subprocess.run")
@patch("app.core.extractor.os.path.exists")
@patch("app.core.extractor.os.path.getsize")
def test_mkvextract_fallback_on_ffmpeg_failure(mock_getsize, mock_exists, mock_run, mock_inspect):
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

    # First call (ffmpeg) fails, second call (mkvextract) succeeds
    def mock_subprocess(cmd, *args, **kwargs):
        if cmd[0] == "ffmpeg":
            raise subprocess.CalledProcessError(1, cmd)
        return MagicMock(returncode=0)
    mock_run.side_effect = mock_subprocess

    mock_exists.return_value = True
    mock_getsize.return_value = 500

    res = extract_embedded_srt("fake.mkv", "out.srt", preferred_lang="eng")

    assert mock_run.call_count == 2
    ffmpeg_cmd = mock_run.call_args_list[0][0][0]
    mkv_cmd = mock_run.call_args_list[1][0][0]
    assert ffmpeg_cmd[0] == "ffmpeg"
    assert mkv_cmd[0] == "mkvextract"
    assert "5:out.srt" in mkv_cmd[3]

def test_tracks_info_caching_skips_probe():
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
    with patch("app.core.extractor.inspect_mkv_tracks") as mock_inspect:
        with patch("app.core.extractor.subprocess.run") as mock_run:
            with patch("app.core.extractor.os.path.exists", return_value=True), \
                 patch("app.core.extractor.os.path.getsize", return_value=500):
                mock_run.return_value = MagicMock(returncode=0)
                extract_embedded_srt("fake.mkv", "out.srt", preferred_lang="eng", tracks_info=tracks_info)

                # inspect_mkv_tracks must NOT be called when tracks_info is provided
                mock_inspect.assert_not_called()
                mock_run.assert_called_once()
