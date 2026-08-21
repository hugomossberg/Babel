import pytest
import os
from unittest.mock import patch
from app.services.scanner import scan_library_folders

def test_library_discovery(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.scanner._get_target_lang_codes", lambda: ["sv"])
    
    # Create movies dir
    movies_dir = tmp_path / "movies"
    movies_dir.mkdir()
    
    # Movie 1 with matching subtitle
    m1_vid = movies_dir / "Movie.1.mkv"
    m1_vid.touch()
    m1_sub = movies_dir / "Movie.1.sv.srt"
    m1_sub.touch()
    
    # Movie 1 Extended with matching subtitle (should not cross-pollinate)
    m1x_vid = movies_dir / "Movie.1.Extended.mkv"
    m1x_vid.touch()
    m1x_sub = movies_dir / "Movie.1.Extended.sv.srt"
    m1x_sub.touch()
    
    # Another format
    m2_vid = movies_dir / "Movie.2.m4v"
    m2_vid.touch()
    m2_sub = movies_dir / "Movie.2.en.srt"
    m2_sub.touch()
    
    res = scan_library_folders(str(movies_dir), category="movies")
    
    assert len(res) == 3
    
    # Check m1
    m1_item = next(r for r in res if r["filename"] == "Movie.1.mkv")
    assert m1_item["has_target_sub"] is True
    assert len(m1_item["subtitles"]) == 1
    assert m1_item["subtitles"][0]["filename"] == "Movie.1.sv.srt"
    
    # Check m1 extended
    m1x_item = next(r for r in res if r["filename"] == "Movie.1.Extended.mkv")
    assert m1x_item["has_target_sub"] is True
    assert len(m1x_item["subtitles"]) == 1
    assert m1x_item["subtitles"][0]["filename"] == "Movie.1.Extended.sv.srt"
    
    # Check m2
    m2_item = next(r for r in res if r["filename"] == "Movie.2.m4v")
    assert m2_item["has_target_sub"] is False
    assert len(m2_item["subtitles"]) == 1
    assert m2_item["subtitles"][0]["filename"] == "Movie.2.en.srt"
