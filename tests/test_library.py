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

def test_scanner_aliases_and_ignored_tags(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.scanner._get_target_lang_codes", lambda: ["sv"])
    movies_dir = tmp_path / "movies2"
    movies_dir.mkdir()

    # Create video
    vid = movies_dir / "Test.Movie.2026.mkv"
    vid.touch()

    # Test all valid aliases
    aliases = ["sv", "swe", "sve", "swedish", "svenska"]
    for alias in aliases:
        sub = movies_dir / f"Test.Movie.2026.{alias}.srt"
        sub.touch()
        res = scan_library_folders(str(movies_dir), category="movies")
        item = next(r for r in res if r["filename"] == "Test.Movie.2026.mkv")
        assert item["has_target_sub"] is True, f"Failed to match alias {alias}"
        sub.unlink()

    # Test forced / signs / songs must NOT count as regular target subtitle
    ignored_variants = ["sv.forced.srt", "swe.signs.srt", "swedish.songs.srt", "forced.srt"]
    for inv in ignored_variants:
        sub = movies_dir / f"Test.Movie.2026.{inv}"
        sub.touch()
        res = scan_library_folders(str(movies_dir), category="movies")
        item = next(r for r in res if r["filename"] == "Test.Movie.2026.mkv")
        assert item["has_target_sub"] is False, f"Should have ignored {inv}"
        sub.unlink()

def test_scanner_unrelated_tokens_ignored(tmp_path, monkeypatch):
    from app.services.scanner import is_subtitle_for_video

    # True matches
    assert is_subtitle_for_video("Movie", "Movie.sv.srt") is True
    assert is_subtitle_for_video("Movie", "Movie.swe.srt") is True
    assert is_subtitle_for_video("Movie", "Movie.sve.srt") is True
    assert is_subtitle_for_video("Movie", "Movie.swedish.srt") is True
    assert is_subtitle_for_video("Movie", "Movie.svenska.srt") is True
    assert is_subtitle_for_video("Movie", "Movie.en.srt") is True
    assert is_subtitle_for_video("Movie", "Movie.forced.srt") is True

    # False matches (arbitrary tokens or prefixes that don't match video base)
    assert is_subtitle_for_video("Movie", "Movie.foo.srt") is False
    assert is_subtitle_for_video("Movie", "Movie.cut.srt") is False
    assert is_subtitle_for_video("Movie", "Movie.Extended.sv.srt") is False
    assert is_subtitle_for_video("Movie", "OtherMovie.sv.srt") is False
    assert is_subtitle_for_video("Movie", "Movie.tmp.srt") is False
    assert is_subtitle_for_video("Movie", "Movie.sv.babel-replaced.123.srt") is False
