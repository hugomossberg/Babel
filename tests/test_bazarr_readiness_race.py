import pytest
import asyncio
import time
import httpx
from unittest.mock import patch, MagicMock

from app.services.bazarr_coordinator import (
    BazarrCoordinator,
    BazarrResultCode,
    BazarrLifecycleState,
)
from app.services.source_resolver import (
    SourceResolver,
    BazarrResult,
    BazarrResultCode,
    trigger_bazarr_search,
)


@pytest.mark.asyncio
async def test_bazarr_readiness_retry_eventually_succeeds():
    """
    Test that when media is not yet indexed in Bazarr (e.g. immediately after Radarr import),
    the coordinator enters a bounded readiness retry loop and succeeds once indexed.
    """
    coordinator = BazarrCoordinator()
    coordinator.reset()

    call_count = 0
    video_path = "/data/media/movies/The Fast and the Furious Tokyo Drift (2006)/Tokyo Drift.mkv"

    def mock_handler(request: httpx.Request):
        nonlocal call_count
        url = str(request.url)
        if "/api/system/jobs" in url:
            return httpx.Response(200, json=[])
        elif "/api/movies" in url and request.method == "GET":
            call_count += 1
            if call_count < 3:
                # Initially empty (Bazarr hasn't indexed the movie yet)
                return httpx.Response(200, json={"data": []})
            else:
                # Now indexed in Bazarr
                return httpx.Response(200, json={"data": [{
                    "id": 42,
                    "radarrId": 101,
                    "title": "The Fast and the Furious: Tokyo Drift",
                    "year": 2006,
                    "path": "/movies/The Fast and the Furious Tokyo Drift (2006)/Tokyo Drift.mkv"
                }]})
        elif "/api/movies/subtitles" in url and request.method == "PATCH":
            return httpx.Response(200, json={"status": "queued"})
        return httpx.Response(404)

    orig_client = httpx.AsyncClient
    transport = httpx.MockTransport(mock_handler)

    logs = []
    with patch("httpx.AsyncClient", lambda **kwargs: orig_client(transport=transport, **{k: v for k, v in kwargs.items() if k != "transport"})), \
         patch("app.core.db.append_job_log", lambda j_id, msg: logs.append(msg)):

        t0 = time.monotonic()
        res = await coordinator.trigger_or_attach_target_search(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="secret",
            job_id=99,
            readiness_timeout=3.0,
        )
        elapsed = time.monotonic() - t0

    assert res.code == BazarrResultCode.TRIGGERED
    assert res.was_accepted is True
    assert call_count >= 3
    assert any("WAITING_FOR_MEDIA" in log for log in logs)
    assert any("media matched" in log for log in logs)

    # Verify cached media lookup happens immediately on second call
    call_count_before = call_count
    with patch("httpx.AsyncClient", lambda **kwargs: orig_client(transport=transport, **{k: v for k, v in kwargs.items() if k != "transport"})):
        res2 = await coordinator.trigger_or_attach_target_search(
            video_path=video_path,
            target_lang="en",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="secret",
            job_id=99,
            readiness_timeout=3.0,
        )
    assert res2.code == BazarrResultCode.TRIGGERED


@pytest.mark.asyncio
async def test_bazarr_readiness_retry_timeout_fails_cleanly():
    """
    Test that if Bazarr never indexes the media within the readiness window,
    the coordinator returns MEDIA_NOT_FOUND without hanging or crashing.
    """
    coordinator = BazarrCoordinator()
    coordinator.reset()

    video_path = "/data/media/movies/NonExistent (2020)/NonExistent.mkv"

    def mock_handler(request: httpx.Request):
        url = str(request.url)
        if "/api/system/jobs" in url:
            return httpx.Response(200, json=[])
        elif "/api/movies" in url:
            return httpx.Response(200, json={"data": []})
        elif "/api/series" in url:
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404)

    orig_client = httpx.AsyncClient
    transport = httpx.MockTransport(mock_handler)

    logs = []
    with patch("httpx.AsyncClient", lambda **kwargs: orig_client(transport=transport, **{k: v for k, v in kwargs.items() if k != "transport"})), \
         patch("app.core.db.append_job_log", lambda j_id, msg: logs.append(msg)):

        t0 = time.monotonic()
        res = await coordinator.trigger_or_attach_target_search(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="secret",
            job_id=100,
            readiness_timeout=1.0,
        )
        elapsed = time.monotonic() - t0

    assert res.code == BazarrResultCode.MEDIA_NOT_FOUND
    assert elapsed >= 0.9
    assert any("WAITING_FOR_MEDIA" in log for log in logs)


@pytest.mark.asyncio
async def test_already_indexed_no_delay():
    """
    Test that if media is already indexed in Bazarr, no readiness sleep occurs.
    """
    coordinator = BazarrCoordinator()
    coordinator.reset()

    video_path = "/data/media/movies/IndexedMovie (2021)/IndexedMovie.mkv"

    def mock_handler(request: httpx.Request):
        url = str(request.url)
        if "/api/system/jobs" in url:
            return httpx.Response(200, json=[])
        elif "/api/movies" in url and request.method == "GET":
            return httpx.Response(200, json={"data": [{
                "id": 10,
                "radarrId": 505,
                "title": "Indexed Movie",
                "year": 2021,
                "path": "/movies/IndexedMovie (2021)/IndexedMovie.mkv"
            }]})
        elif "/api/movies/subtitles" in url and request.method == "PATCH":
            return httpx.Response(200, json={"status": "queued"})
        return httpx.Response(404)

    orig_client = httpx.AsyncClient
    transport = httpx.MockTransport(mock_handler)

    t0 = time.monotonic()
    with patch("httpx.AsyncClient", lambda **kwargs: orig_client(transport=transport, **{k: v for k, v in kwargs.items() if k != "transport"})):
        res = await coordinator.trigger_or_attach_target_search(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="secret",
            readiness_timeout=5.0,
        )
    elapsed = time.monotonic() - t0

    assert res.code == BazarrResultCode.TRIGGERED
    assert elapsed < 0.5  # Immediate match, no waiting


@pytest.mark.asyncio
async def test_arr_unindexed_returns_waiting_for_media():
    """
    Test that an ARR import whose media is not yet indexed returns WAITING_FOR_MEDIA,
    whereas a manual job returns MEDIA_NOT_FOUND.
    """
    coordinator = BazarrCoordinator()
    coordinator.reset()

    video_path = "/data/media/movies/Unindexed (2022)/Unindexed.mkv"

    def mock_handler(request: httpx.Request):
        url = str(request.url)
        if "/api/system/jobs" in url:
            return httpx.Response(200, json=[])
        elif "/api/movies" in url:
            return httpx.Response(200, json={"data": []})
        elif "/api/series" in url:
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404)

    orig_client = httpx.AsyncClient
    transport = httpx.MockTransport(mock_handler)

    with patch("httpx.AsyncClient", lambda **kwargs: orig_client(transport=transport, **{k: v for k, v in kwargs.items() if k != "transport"})):
        # Radarr / ARR import -> WAITING_FOR_MEDIA
        res_arr = await coordinator.trigger_or_attach_target_search(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="secret",
            event_source="RADARR",
            readiness_timeout=0.1,
        )
        assert res_arr.code == BazarrResultCode.WAITING_FOR_MEDIA

        # Manual trigger without ARR source or IDs -> MEDIA_NOT_FOUND
        res_manual = await coordinator.trigger_or_attach_target_search(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="secret",
            event_source="MANUAL",
            readiness_timeout=0.1,
        )
        assert res_manual.code == BazarrResultCode.MEDIA_NOT_FOUND


def test_cross_language_sync_isolation():
    """
    Verify that Bazarr sync jobs strictly match the requested target language.
    A German or English sync job must NOT correlate with or block a Swedish target search/sync.
    """
    coordinator = BazarrCoordinator()
    coordinator.reset()

    video_path = "/data/media/movies/The French Connection (1971)/The French Connection.mkv"

    from app.services.bazarr_coordinator import BazarrJobInfo
    jobs = [
        BazarrJobInfo(
            job_id="1",
            status="running",
            job_name="Syncing subtitles for The French Connection (German)",
            progress_message="Syncing subtitles for The French Connection (German)",
            job_type="sync",
            matched_language="de",
        ),
        BazarrJobInfo(
            job_id="2",
            status="running",
            job_name="Syncing subtitles for The French Connection (English)",
            progress_message="Syncing subtitles for The French Connection (English)",
            job_type="sync",
            matched_language="en",
        ),
        BazarrJobInfo(
            job_id="3",
            status="running",
            job_name="Syncing subtitles for The French Connection (Swedish)",
            progress_message="Syncing subtitles for The French Connection (Swedish)",
            job_type="sync",
            matched_language="sv",
        ),
    ]

    # Target = Swedish (sv)
    search_jobs, sync_jobs = coordinator.classify_jobs_for_target(jobs, video_path, "sv")
    assert len(sync_jobs) == 1
    assert "Swedish" in sync_jobs[0].job_name

    # Target = English (en)
    search_jobs, sync_jobs = coordinator.classify_jobs_for_target(jobs, video_path, "en")
    assert len(sync_jobs) == 1
    assert "English" in sync_jobs[0].job_name

    # Target = German (de)
    search_jobs, sync_jobs = coordinator.classify_jobs_for_target(jobs, video_path, "de")
    assert len(sync_jobs) == 1
    assert "German" in sync_jobs[0].job_name


@pytest.mark.asyncio
async def test_source_resolver_halts_immediately_on_waiting_for_media():
    """
    Verify that when Bazarr returns WAITING_FOR_MEDIA (media-level unindexed state),
    SourceResolver stops immediately without querying subsequent fallback languages.
    Exposes is_waiting_for_media=True and returns None.
    """
    video_path = "/data/media/movies/UnindexedMovie (2023)/UnindexedMovie.mkv"

    # Container with Japanese audio, target is Swedish -> fallback source order: ja, en, ...
    container_tracks = {
        "audio": [{"index": 1, "language": "ja"}],
        "subtitles": [],
        "duration": 3600.0,
    }

    resolver = SourceResolver(
        video_path=video_path,
        container_tracks=container_tracks,
        primary_audio_lang="ja",
        target_languages=["sv"],
        bazarr_url="http://bazarr:6767",
        bazarr_api_key="secret",
        enable_bazarr=True,
        extract_source_embedded=False,
        source_search_deadline=30.0,
        event_source="RADARR",
        find_external_subtitle_fn=lambda *args, **kwargs: None,
    )

    calls = []

    async def mock_trigger(v_path, lang, url, key, **kwargs):
        calls.append(lang)
        return BazarrResult(
            code=BazarrResultCode.WAITING_FOR_MEDIA,
            language=lang,
            detail="Media not yet indexed in Bazarr",
        )

    with patch("app.services.source_resolver.trigger_bazarr_search", side_effect=mock_trigger):
        src = await resolver.resolve()

    assert src is None
    assert resolver.is_waiting_for_media is True
    # Crucial invariant: only the first candidate language was attempted before halting immediately
    assert len(calls) == 1
    assert calls[0] == "ja"
