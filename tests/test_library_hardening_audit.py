import os
import json
import time
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
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


@pytest.fixture
def client():
    return TestClient(app)


def test_media_files_status_endpoint_contract(client):
    """
    Verify GET /api/media-files/status returns scanning_embedded and pending count without disk scan.
    """
    res = client.get("/api/media-files/status")
    assert res.status_code == 200
    data = res.json()
    assert "scanning_embedded" in data
    assert "pending" in data
    assert isinstance(data["scanning_embedded"], bool)
    assert isinstance(data["pending"], int)


def test_failed_probe_distinguished_from_no_subtitles(tmp_path, monkeypatch):
    """
    Verify that probe failures are explicitly cached as FAILED,
    resulting in embedded_status_known=False (not treated as definitive absence).
    """
    monkeypatch.setattr("app.services.scanner._get_target_lang_codes", lambda: ["sv"])
    _EMBEDDED_TRACKS_CACHE.clear()

    series_dir = tmp_path / "series"
    series_dir.mkdir()
    show_dir = series_dir / "BrokenShow"
    show_dir.mkdir()
    ep_path = show_dir / "BrokenShow.S01E01.mkv"
    ep_path.write_bytes(b"broken content")

    with patch("app.core.extractor.inspect_mkv_tracks", side_effect=Exception("Corrupted container")):
        scan_library_folders(str(series_dir), category="series")
        embedded_prober.wait_completion(timeout=5.0)

    # Verify persistent DB record contains failed status
    st = os.stat(str(ep_path))
    cached = get_cached_embedded_subtitle_tracks(str(ep_path), int(st.st_size), getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
    assert cached is not None
    assert cached.get("status") == "failed"

    # Second scan: returns embedded_status_known=False
    res = scan_library_folders(str(series_dir), category="series")
    assert len(res) == 1
    ep = res[0]["episodes"][0]
    assert ep["embedded_status_known"] is False
    assert ep["has_embedded_target"] is False
    assert ep["has_target_sub"] is False


def test_delete_subtitles_transitions_state_cleanly(tmp_path, monkeypatch):
    """
    Verify that deleting external subtitle from an episode where embedded target sub exists
    transitions source from 'both' (or 'external') to 'embedded' with has_target_sub=True.
    """
    monkeypatch.setattr("app.services.scanner._get_target_lang_codes", lambda: ["sv"])
    _EMBEDDED_TRACKS_CACHE.clear()

    series_dir = tmp_path / "series_del"
    series_dir.mkdir()
    show_dir = series_dir / "DualShow"
    show_dir.mkdir()
    ep_path = show_dir / "DualShow.S01E01.mkv"
    ep_path.write_bytes(b"media with embedded swedish")
    sub_path = show_dir / "DualShow.S01E01.sv.srt"
    sub_path.write_text("1\n00:00:01,000 --> 00:00:02,000\nHej\n", encoding="utf-8")

    st = os.stat(str(ep_path))
    tracks = [{"id": 0, "codec": "subrip", "language": "swe", "title": "Swedish", "forced": False}]
    set_cached_embedded_subtitle_tracks(str(ep_path), int(st.st_size), getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)), tracks)

    # Initial scan with external sub present
    res1 = scan_library_folders(str(series_dir), category="series")
    assert res1[0]["episodes"][0]["has_target_sub"] is True
    assert res1[0]["episodes"][0]["target_sub_source"] == "both"

    # Delete external subtitle
    os.remove(str(sub_path))

    # Scan after deletion: embedded track keeps item Complete!
    res2 = scan_library_folders(str(series_dir), category="series")
    ep2 = res2[0]["episodes"][0]
    assert ep2["has_target_sub"] is True
    assert ep2["has_embedded_target"] is True
    assert ep2["target_sub_source"] == "embedded"


def test_large_library_bulk_lookup_performance(tmp_path, monkeypatch):
    """
    Verify scanning a large library (1,000 episodes across 50 shows)
    uses bulk cache queries efficiently and completes in under 2 seconds.
    """
    monkeypatch.setattr("app.services.scanner._get_target_lang_codes", lambda: ["sv"])
    _EMBEDDED_TRACKS_CACHE.clear()

    series_dir = tmp_path / "large_series"
    series_dir.mkdir()

    items_to_cache = []
    for s_idx in range(50):
        show_name = f"Show_{s_idx:02d}"
        show_dir = series_dir / show_name
        show_dir.mkdir()
        for ep_idx in range(20):
            ep_file = show_dir / f"{show_name}.S01E{ep_idx:02d}.mkv"
            ep_file.write_bytes(b"dummy")
            st = os.stat(str(ep_file))
            items_to_cache.append((
                str(ep_file),
                int(st.st_size),
                getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)),
                [{"id": 0, "codec": "subrip", "language": "swe", "title": "Swedish"}]
            ))

    # Pre-populate persistent SQLite cache in bulk
    for path, size, mtime_ns, tracks in items_to_cache:
        set_cached_embedded_subtitle_tracks(path, size, mtime_ns, tracks)

    # Time the full library scan
    start_time = time.time()
    res = scan_library_folders(str(series_dir), category="series")
    elapsed = time.time() - start_time

    assert len(res) == 50
    total_episodes = sum(len(show["episodes"]) for show in res)
    assert total_episodes == 1000
    assert elapsed < 2.0, f"Large library scan took {elapsed:.2f}s (expected < 2.0s)"
    assert is_embedded_probing_active() is False
