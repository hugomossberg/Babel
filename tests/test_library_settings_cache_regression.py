"""A cached empty/stale library must not survive folder or language changes."""
import asyncio

import pytest

from app.api import dashboard


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", ["folders", "languages"])
async def test_setting_save_invalidates_library_snapshot(setting, monkeypatch):
    settings = {"media_series_path": "/old/tv", "media_movies_path": "/old/movies"}
    monkeypatch.setattr(dashboard, "get_setting", lambda key, default="": settings.get(key, default))
    monkeypatch.setattr(dashboard, "set_setting", lambda key, value: settings.__setitem__(key, value))
    monkeypatch.setattr(dashboard, "_scan_lock", asyncio.Lock())
    monkeypatch.setattr(dashboard, "_media_cache", {})
    monkeypatch.setattr(dashboard, "_media_cache_paths", [])
    monkeypatch.setattr(dashboard, "_media_cache_time", 0.0)
    scans = []

    def scan(path, category):
        scans.append((path, category))
        return []

    monkeypatch.setattr(dashboard, "scan_library_folders", scan)
    await dashboard.api_get_media_files()
    assert len(scans) == 2
    await dashboard.api_get_media_files()
    assert len(scans) == 2  # Ordinary load uses the cached snapshot.
    if setting == "folders":
        await dashboard.api_save_folders(dashboard.MediaFoldersSettingsRequest(
            media_series_path="/new/tv", media_movies_path="/new/movies",
        ))
    else:
        await dashboard.api_save_languages(dashboard.LanguagesSettingsRequest(
            languages=[dashboard.TargetLangItem(name="Spanish", code="es", enabled=True)],
        ))
    await dashboard.api_get_media_files()
    assert len(scans) == 4
    if setting == "folders":
        assert scans[-2:] == [("/new/tv", "series"), ("/new/movies", "movies")]


@pytest.mark.asyncio
@pytest.mark.parametrize("waiting_request", [False, True])
async def test_old_scan_cannot_restore_cache_after_folder_change(monkeypatch, waiting_request):
    settings = {"media_series_path": "/old/tv", "media_movies_path": "/old/movies"}
    monkeypatch.setattr(dashboard, "get_setting", lambda key, default="": settings.get(key, default))
    monkeypatch.setattr(dashboard, "set_setting", lambda key, value: settings.__setitem__(key, value))
    monkeypatch.setattr(dashboard, "_scan_lock", asyncio.Lock())
    monkeypatch.setattr(dashboard, "_media_cache", {})
    monkeypatch.setattr(dashboard, "_media_cache_paths", [])
    monkeypatch.setattr(dashboard, "_media_cache_time", 0.0)
    started, release = asyncio.Event(), asyncio.Event()
    scans = []
    original_to_thread = asyncio.to_thread

    def scan(path, category):
        scans.append((path, category))
        return []

    async def delayed_thread(function, *args, **kwargs):
        if function is scan and not started.is_set():
            started.set()
            await release.wait()
        return await original_to_thread(function, *args, **kwargs)

    monkeypatch.setattr(dashboard, "scan_library_folders", scan)
    monkeypatch.setattr(dashboard.asyncio, "to_thread", delayed_thread)
    old_scan = asyncio.create_task(dashboard.api_get_media_files())
    await asyncio.wait_for(started.wait(), timeout=2)
    waiter = asyncio.create_task(dashboard.api_get_media_files()) if waiting_request else None
    await asyncio.sleep(0)
    await dashboard.api_save_folders(dashboard.MediaFoldersSettingsRequest(
        media_series_path="/new/tv", media_movies_path="/new/movies",
    ))
    release.set()
    await asyncio.wait_for(old_scan, timeout=2)
    if waiter:
        await asyncio.wait_for(waiter, timeout=2)
    await dashboard.api_get_media_files()
    assert scans[-2:] == [("/new/tv", "series"), ("/new/movies", "movies")]
