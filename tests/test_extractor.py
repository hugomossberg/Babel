import pytest
import os
import subprocess
from unittest.mock import patch
from app.core.extractor import extract_embedded_srt

@patch("app.core.extractor.inspect_mkv_tracks")
@patch("app.core.extractor.subprocess.run")
@patch("app.core.extractor.os.path.exists")
@patch("app.core.extractor.os.path.getsize")
def test_track_selection(mock_getsize, mock_exists, mock_run, mock_inspect):
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
    
    # Mock os.path.exists and os.path.getsize
    mock_exists.return_value = True
    mock_getsize.return_value = 500
    
    extract_embedded_srt("fake.mkv", "out.srt", preferred_lang="eng")
    
    # check that mkvextract was called with track 2
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert "2:out.srt" in args[3]

@patch("app.core.extractor.inspect_mkv_tracks")
@patch("app.core.extractor.subprocess.run")
@patch("app.core.extractor.os.path.exists")
@patch("app.core.extractor.os.path.getsize")
@patch("app.core.extractor.os.rename")
@patch("app.core.extractor.os.remove")
def test_vtt_normalization(mock_remove, mock_rename, mock_getsize, mock_exists, mock_run, mock_inspect):
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
    
    # mkvextract then ffmpeg
    assert mock_run.call_count == 2
    ffmpeg_cmd = mock_run.call_args_list[1][0][0]
    assert ffmpeg_cmd[0] == "ffmpeg"
    assert ffmpeg_cmd[1] == "-y"
    assert "srt" in ffmpeg_cmd

