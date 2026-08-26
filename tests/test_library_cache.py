import pytest
import asyncio
import time
from unittest.mock import patch
from app.services.scanner import scan_library_folders, _fast_count_subtitle_lines, _SUB_LINE_CACHE
from app.api.dashboard import api_get_media_files, invalidate_media_cache

@pytest.fixture
def clean_db(tmp_path):
    import app.core.db as db_mod
    db_path = str(tmp_path / "test.db")
    with patch("app.core.db.DB_PATH", db_path):
        db_mod.DB_PATH = db_path
        db_mod.init_db()
    return db_path

@pytest.mark.asyncio
async def test_fast_count_subtitle_lines_caching(tmp_path):
    sub = tmp_path / "test.srt"
    sub.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n\n2\n00:00:03,000 --> 00:00:04,000\nWorld\n", encoding="utf-8")

    _SUB_LINE_CACHE.clear()
    count1 = _fast_count_subtitle_lines(str(sub))
    assert count1 == 2
    assert str(sub) in _SUB_LINE_CACHE

    # Second call should hit cache
    count2 = _fast_count_subtitle_lines(str(sub))
    assert count2 == 2

@pytest.mark.asyncio
async def test_library_caching_and_force(tmp_path, clean_db):
    movies_dir = tmp_path / "movies"
    movies_dir.mkdir()
    m1 = movies_dir / "Movie.1.mkv"
    m1.touch()
    sub1 = movies_dir / "Movie.1.sv.srt"
    sub1.write_text("1\n00:00:01,000 --> 00:00:02,000\nHej\n", encoding="utf-8")

    tv_dir = tmp_path / "tv"
    tv_dir.mkdir()

    with patch("app.core.db.DB_PATH", clean_db):
        import app.core.db as db_mod
        db_mod.set_setting("media_movies_path", str(movies_dir))
        db_mod.set_setting("media_series_path", str(tv_dir))
        db_mod.set_setting("languages", '[{"name": "Swedish", "code": "sv", "enabled": true}]')

        invalidate_media_cache()

        # First fetch: populates cache
        res1 = await api_get_media_files(force=False)
        assert len(res1["movies"]) == 1
        assert res1["movies"][0]["has_target_sub"] is True

        # Add a new movie on disk without force=True (should return cached data)
        m2 = movies_dir / "Movie.2.mkv"
        m2.touch()

        res2 = await api_get_media_files(force=False)
        assert len(res2["movies"]) == 1  # Served from cache

        # Fetch with force=True (should re-scan and find Movie 2)
        res3 = await api_get_media_files(force=True)
        assert len(res3["movies"]) == 2

@pytest.mark.asyncio
async def test_concurrent_media_files_requests_no_429(tmp_path, clean_db):
    movies_dir = tmp_path / "movies"
    movies_dir.mkdir()
    m1 = movies_dir / "Movie.1.mkv"
    m1.touch()
    tv_dir = tmp_path / "tv"
    tv_dir.mkdir()

    with patch("app.core.db.DB_PATH", clean_db):
        import app.core.db as db_mod
        db_mod.set_setting("media_movies_path", str(movies_dir))
        db_mod.set_setting("media_series_path", str(tv_dir))

        invalidate_media_cache()

        # 10 concurrent requests
        results = await asyncio.gather(*(api_get_media_files() for _ in range(10)))
        assert len(results) == 10
        for r in results:
            assert "movies" in r
            assert len(r["movies"]) == 1
