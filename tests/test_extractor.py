import pytest
import os
import subprocess
from unittest.mock import patch, MagicMock, call
from app.core.extractor import extract_embedded_srt

def _create_mock_srt(out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("1\n00:00:01,000 --> 00:00:03,000\nHello\n")

@patch("app.core.extractor.inspect_mkv_tracks")
def test_track_selection_mkv_primary(mock_inspect, tmp_path):
    """For .mkv files mkvextract is now the PRIMARY tool (faster via seek table).
    After our v2.3.43 fix the first subprocess call for a .mkv must be mkvextract,
    not ffmpeg."""
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
        # mkvextract format: "track_id:output_path" — extract just the path
        target = cmd[-1]
        if ":" in target and not target.startswith("/"):
            target = target.split(":", 1)[1]
        _create_mock_srt(target)
        return MagicMock(returncode=0)

    with patch("app.core.extractor.subprocess.run", side_effect=mock_subprocess) as mock_run:
        res = extract_embedded_srt("fake.mkv", out_srt, preferred_lang="eng")
        assert res is True
        # For .mkv, the first call must be mkvextract (primary)
        # With a valid output, there should be only one call (mkvextract succeeded)
        first_cmd = mock_run.call_args_list[0][0][0]
        assert first_cmd[0] == "mkvextract"


@patch("app.core.extractor.inspect_mkv_tracks")
def test_track_selection_non_mkv_uses_ffmpeg_first(mock_inspect, tmp_path):
    """For non-MKV containers (.mp4, .avi) ffmpeg is the primary tool."""
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
    mock_inspect.return_value = tracks_info
    out_srt = str(tmp_path / "out.srt")

    def mock_subprocess(cmd, *args, **kwargs):
        _create_mock_srt(cmd[-1])
        return MagicMock(returncode=0)

    with patch("app.core.extractor.subprocess.run", side_effect=mock_subprocess) as mock_run:
        res = extract_embedded_srt("fake.mp4", out_srt, preferred_lang="eng")
        assert res is True
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        # Non-MKV → ffmpeg is primary
        assert args[0] == "ffmpeg"
        assert "-map" in args
        assert "0:s:0" in args


@patch("app.core.extractor.inspect_mkv_tracks")
def test_vtt_mkv_mkvextract_then_ffmpeg_conversion(mock_inspect, tmp_path):
    """VTT codec in MKV: mkvextract extracts raw VTT, then ffmpeg converts to SRT.
    So we expect 2 calls: mkvextract first, ffmpeg second for conversion."""
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

    call_count = [0]

    def mock_subprocess(cmd, *args, **kwargs):
        call_count[0] += 1
        # Both mkvextract and ffmpeg conversion succeed
        _create_mock_srt(out_srt)
        return MagicMock(returncode=0)

    with patch("app.core.extractor.subprocess.run", side_effect=mock_subprocess) as mock_run:
        res = extract_embedded_srt("fake.mkv", out_srt, preferred_lang="eng")
        assert res is True
        # Call 1: mkvextract (primary for .mkv)
        # Call 2: ffmpeg to convert VTT → SRT
        assert mock_run.call_count == 2
        first_cmd = mock_run.call_args_list[0][0][0]
        second_cmd = mock_run.call_args_list[1][0][0]
        assert first_cmd[0] == "mkvextract"
        assert second_cmd[0] == "ffmpeg"


@patch("app.core.extractor.inspect_mkv_tracks")
def test_mkvextract_primary_ffmpeg_fallback_on_failure(mock_inspect, tmp_path):
    """For .mkv: when mkvextract fails, ffmpeg is attempted as fallback."""
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

    def mock_subprocess(cmd, *args, **kwargs):
        if cmd[0] == "mkvextract":
            raise subprocess.CalledProcessError(1, cmd)
        elif cmd[0] == "ffmpeg":
            _create_mock_srt(out_srt)
            return MagicMock(returncode=0)
        return MagicMock(returncode=0)

    with patch("app.core.extractor.subprocess.run", side_effect=mock_subprocess) as mock_run:
        res = extract_embedded_srt("fake.mkv", out_srt, preferred_lang="eng")
        assert res is True
        assert mock_run.call_count == 2
        first_cmd = mock_run.call_args_list[0][0][0]
        second_cmd = mock_run.call_args_list[1][0][0]
        assert first_cmd[0] == "mkvextract"
        assert second_cmd[0] == "ffmpeg"
        assert "-map" in second_cmd


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
        # Handle mkvextract "track_id:output_path" format
        target = cmd[-1]
        if ":" in target and not target.startswith("/"):
            target = target.split(":", 1)[1]
        _create_mock_srt(target)
        return MagicMock(returncode=0)

    with patch("app.core.extractor.inspect_mkv_tracks") as mock_inspect:
        with patch("app.core.extractor.subprocess.run", side_effect=mock_subprocess) as mock_run:
            res = extract_embedded_srt("fake.mkv", out_srt, preferred_lang="eng", tracks_info=tracks_info)
            assert res is True
            # inspect_mkv_tracks must NOT be called when tracks_info is provided
            mock_inspect.assert_not_called()
            # For .mkv: mkvextract called first; if it succeeds there is 1 call
            assert mock_run.call_count >= 1
            first_cmd = mock_run.call_args_list[0][0][0]
            assert first_cmd[0] == "mkvextract"
