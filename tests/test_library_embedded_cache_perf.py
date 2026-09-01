import os
import time
import json
import pytest
from unittest.mock import patch, MagicMock

from app.services.scanner import (
    scan_library_folders,
    embedded_prober,
    is_embedded_probing_active,
    _EMBEDDED_TRACKS_CACHE,
)
from app.core.db import (
    get_cached_embedded_subtitle_tracks,
    set_cached_embedded_subtitle_tracks,
    bulk_get_cached_embedded_subtitle_tracks,
)


def test_library_scan_non_blocking_and_caching(tmp_path, monkeypatch):
    """
    Verify that scanning uncached media files does NOT block,
    returns embedded_status_known=False initially,
    schedules background probing, and populates persistent cache.
    """
    monkeypatch.setattr("app.services.scanner._get_target_lang_codes", lambda: ["sv"])

    movies_dir = tmp_path / "movies"
    movies_dir.mkdir()

    vid_path = movies_dir / "UncachedMovie.2026.mkv"
    vid_path.write_bytes(b"dummy video data")

    # Mock inspect_mkv_tracks to return Swedish embedded subtitle
    mock_tracks = {
        "subtitles": [
            {"id": 0, "codec": "subrip", "language": "swe", "title": "Swedish SDH", "forced": False}
        ],
        "audio": []
    }

    with patch("app.core.extractor.inspect_mkv_tracks", return_value=mock_tracks) as mock_inspect:
        # First scan: cache miss -> returns immediately, embedded_status_known=False
        res = scan_library_folders(str(movies_dir), category="movies")
        assert len(res) == 1
        item = res[0]
        assert item["filename"] == "UncachedMovie.2026.mkv"
        assert item["embedded_status_known"] is False
        assert item["has_target_sub"] is False

        # Wait for background prober to complete
        embedded_prober.wait_completion(timeout=5.0)

        # inspect_mkv_tracks should have been called in background
        assert mock_inspect.call_count >= 1

    # Second scan: cache hit from SQLite persistent cache!
    # Mock inspect_mkv_tracks to fail if called, ensuring it does NOT probe again
    with patch("app.core.extractor.inspect_mkv_tracks", side_effect=RuntimeError("Should not be called!")):
        res2 = scan_library_folders(str(movies_dir), category="movies")
        assert len(res2) == 1
        item2 = res2[0]
        assert item2["embedded_status_known"] is True
        assert item2["has_embedded_target"] is True
        assert item2["has_target_sub"] is True
        assert item2["target_sub_source"] == "embedded"


def test_persistent_cache_survives_l1_cache_clear(tmp_path, monkeypatch):
    """
    Verify persistent SQLite cache survives when in-memory L1 cache is cleared.
    """
    monkeypatch.setattr("app.services.scanner._get_target_lang_codes", lambda: ["sv"])
    from app.services.scanner import _EMBEDDED_TRACKS_CACHE

    movies_dir = tmp_path / "movies_persist"
    movies_dir.mkdir()
    vid_path = movies_dir / "PersistMovie.mkv"
    vid_path.write_bytes(b"content")

    st = os.stat(str(vid_path))
    tracks = [{"id": 1, "codec": "srt", "language": "sv", "title": "Swedish", "forced": False}]

    # Set directly in persistent DB cache
    set_cached_embedded_subtitle_tracks(str(vid_path), int(st.st_size), getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)), tracks)

    # Clear in-memory L1 cache
    _EMBEDDED_TRACKS_CACHE.clear()

    # Scan should read from SQLite and populate item correctly without probing
    with patch("app.core.extractor.inspect_mkv_tracks", side_effect=RuntimeError("Blocked")):
        res = scan_library_folders(str(movies_dir), category="movies")
        assert len(res) == 1
        item = res[0]
        assert item["embedded_status_known"] is True
        assert item["has_embedded_target"] is True
        assert item["has_target_sub"] is True


def test_cache_invalidation_on_file_modification(tmp_path, monkeypatch):
    """
    Verify that updating mtime / size invalidates cache and schedules a re-probe.
    """
    monkeypatch.setattr("app.services.scanner._get_target_lang_codes", lambda: ["sv"])

    movies_dir = tmp_path / "movies_mod"
    movies_dir.mkdir()
    vid_path = movies_dir / "ModMovie.mkv"
    vid_path.write_bytes(b"original data")

    mock_tracks_1 = {"subtitles": [{"id": 0, "codec": "subrip", "language": "swe", "title": "Swedish"}], "audio": []}
    mock_tracks_2 = {"subtitles": [{"id": 0, "codec": "subrip", "language": "eng", "title": "English"}], "audio": []}

    with patch("app.core.extractor.inspect_mkv_tracks", return_value=mock_tracks_1):
        scan_library_folders(str(movies_dir), category="movies")
        embedded_prober.wait_completion(timeout=5.0)

    # File has Swedish sub
    res = scan_library_folders(str(movies_dir), category="movies")
    assert res[0]["has_target_sub"] is True

    # Modify file: append data (changes size and mtime)
    vid_path.write_bytes(b"original data with more bytes")

    with patch("app.core.extractor.inspect_mkv_tracks", return_value=mock_tracks_2):
        # Immediate scan: cache invalidated, status unknown
        res2 = scan_library_folders(str(movies_dir), category="movies")
        assert res2[0]["embedded_status_known"] is False

        embedded_prober.wait_completion(timeout=5.0)

        # After re-probe, updated to English-only (no Swedish target sub)
        res3 = scan_library_folders(str(movies_dir), category="movies")
        assert res3[0]["embedded_status_known"] is True
        assert res3[0]["has_embedded_target"] is False
        assert res3[0]["has_target_sub"] is False


def test_probe_failure_is_handled_gracefully(tmp_path, monkeypatch):
    """
    Verify that if inspect_mkv_tracks raises an exception, the background prober
    records a failed status so Babel does NOT falsely claim known absence (embedded_status_known remains False),
    and does NOT re-probe in an infinite loop.
    """
    monkeypatch.setattr("app.services.scanner._get_target_lang_codes", lambda: ["sv"])

    movies_dir = tmp_path / "movies_fail"
    movies_dir.mkdir()
    vid_path = movies_dir / "CorruptMovie.mkv"
    vid_path.write_bytes(b"corrupted")

    with patch("app.core.extractor.inspect_mkv_tracks", side_effect=Exception("Corrupt container")) as mock_probe:
        scan_library_folders(str(movies_dir), category="movies")
        embedded_prober.wait_completion(timeout=5.0)
        assert mock_probe.call_count >= 1

    # Subsequent scan: failure is cached, so it does NOT re-probe, and embedded_status_known is False (unknown/uninspected)
    with patch("app.core.extractor.inspect_mkv_tracks", side_effect=RuntimeError("Must not be probed again!")):
        res = scan_library_folders(str(movies_dir), category="movies")
        assert len(res) == 1
        assert res[0]["embedded_status_known"] is False
        assert res[0]["has_embedded_target"] is False
        assert res[0]["has_target_sub"] is False
        assert is_embedded_probing_active() is False


def test_external_target_short_circuits_probing(tmp_path, monkeypatch):
    """
    Verify that media files with existing external target subtitles:
    - Return immediately as Complete (has_target_sub=True, embedded_status_known=True, target_sub_source='external')
    - NEVER call inspect_mkv_tracks (patch raises if called)
    - Schedule zero background probes (get_pending_count() == 0, is_embedded_probing_active() is False)
    - Repeated scans continue to schedule zero background probes
    """
    monkeypatch.setattr("app.services.scanner._get_target_lang_codes", lambda: ["sv"])
    _EMBEDDED_TRACKS_CACHE.clear()

    movies_dir = tmp_path / "movies_external_strict"
    movies_dir.mkdir()
    vid_path = movies_dir / "CompleteMovie.2026.mkv"
    vid_path.write_bytes(b"some media data")
    sub_path = movies_dir / "CompleteMovie.2026.sv.srt"
    sub_path.write_text("1\n00:00:01,000 --> 00:00:03,000\nHej\n", encoding="utf-8")

    # Prober inspect_mkv_tracks must NEVER be invoked for external-complete media
    with patch("app.core.extractor.inspect_mkv_tracks", side_effect=RuntimeError("Must NEVER probe external-complete media!")):
        res = scan_library_folders(str(movies_dir), category="movies")
        assert len(res) == 1
        assert res[0]["filename"] == "CompleteMovie.2026.mkv"
        assert res[0]["has_target_sub"] is True
        assert res[0]["embedded_status_known"] is True
        assert res[0]["target_sub_source"] == "external"
        assert embedded_prober.get_pending_count() == 0
        assert is_embedded_probing_active() is False

        # Repeated scan should also schedule zero probes
        res2 = scan_library_folders(str(movies_dir), category="movies")
        assert len(res2) == 1
        assert res2[0]["has_target_sub"] is True
        assert embedded_prober.get_pending_count() == 0
        assert is_embedded_probing_active() is False


def test_existing_embedded_cache_used_without_probe(tmp_path, monkeypatch):
    """
    Verify that if valid embedded cache exists for an external-complete media,
    it reports target_sub_source='both' without calling inspect_mkv_tracks.
    """
    monkeypatch.setattr("app.services.scanner._get_target_lang_codes", lambda: ["sv"])
    _EMBEDDED_TRACKS_CACHE.clear()

    movies_dir = tmp_path / "movies_cached_both"
    movies_dir.mkdir()
    vid_path = movies_dir / "BothMovie.2026.mkv"
    vid_path.write_bytes(b"media content")
    sub_path = movies_dir / "BothMovie.2026.sv.srt"
    sub_path.write_text("1\n00:00:01,000 --> 00:00:03,000\nHej\n", encoding="utf-8")

    st = os.stat(str(vid_path))
    set_cached_embedded_subtitle_tracks(
        str(vid_path),
        int(st.st_size),
        getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)),
        [{"id": 0, "language": "swe", "codec": "SubRip/SRT", "forced": False, "title": "Swedish"}]
    )

    with patch("app.core.extractor.inspect_mkv_tracks", side_effect=RuntimeError("Must not probe cached media")):
        res = scan_library_folders(str(movies_dir), category="movies")
        assert len(res) == 1
        assert res[0]["has_target_sub"] is True
        assert res[0]["has_embedded_target"] is True
        assert res[0]["target_sub_source"] == "both"
        assert embedded_prober.get_pending_count() == 0
        assert is_embedded_probing_active() is False


def test_delete_external_sub_lazily_triggers_probing(tmp_path, monkeypatch):
    """
    Verify that an uncached media file with external sub is initially not probed,
    but deleting the external sub causes ONLY that media item to be scheduled for probing.
    """
    monkeypatch.setattr("app.services.scanner._get_target_lang_codes", lambda: ["sv"])
    _EMBEDDED_TRACKS_CACHE.clear()

    movies_dir = tmp_path / "movies_delete_lazy"
    movies_dir.mkdir()
    vid_path = movies_dir / "LazyMovie.2026.mkv"
    vid_path.write_bytes(b"media content")
    sub_path = movies_dir / "LazyMovie.2026.sv.srt"
    sub_path.write_text("1\n00:00:01,000 --> 00:00:03,000\nHej\n", encoding="utf-8")

    # Step 1: Initial scan with external sub -> zero probes
    with patch("app.core.extractor.inspect_mkv_tracks", side_effect=RuntimeError("Must not probe")):
        res1 = scan_library_folders(str(movies_dir), category="movies")
        assert res1[0]["has_target_sub"] is True
        assert res1[0]["embedded_status_known"] is True
        assert res1[0]["target_sub_source"] == "external"
        assert embedded_prober.get_pending_count() == 0

    # Step 2: Delete external subtitle file
    os.remove(str(sub_path))

    # Step 3: Next scan sees no external sub and no cache -> schedules lazy background probe
    mock_probe = MagicMock(return_value={
        "subtitles": [{"id": 0, "language": "swe", "codec": "SubRip/SRT", "forced": False}],
        "audio": []
    })
    with patch("app.core.extractor.inspect_mkv_tracks", mock_probe):
        res2 = scan_library_folders(str(movies_dir), category="movies")
        # Immediate response: status unknown, queued
        assert res2[0]["has_target_sub"] is False
        assert res2[0]["embedded_status_known"] is False

        # Wait for prober to complete
        embedded_prober.wait_completion(timeout=5.0)
        assert mock_probe.call_count == 1

    # Step 4: Scan after probing -> now known Complete via embedded!
    with patch("app.core.extractor.inspect_mkv_tracks", side_effect=RuntimeError("Must not probe cached")):
        res3 = scan_library_folders(str(movies_dir), category="movies")
        assert res3[0]["has_target_sub"] is True
        assert res3[0]["has_embedded_target"] is True
        assert res3[0]["embedded_status_known"] is True
        assert res3[0]["target_sub_source"] == "embedded"
