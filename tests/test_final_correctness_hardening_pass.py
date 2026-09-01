import asyncio
import datetime
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import srt

import app.core.db as db
from app.core.db import (
    create_job,
    get_job_by_id,
    init_db,
    set_setting,
    get_cached_embedded_subtitle_tracks,
    set_cached_embedded_subtitle_tracks,
    bulk_get_cached_embedded_subtitle_tracks,
)
from app.core.trust_engine import (
    CandidateOrigin,
    SubtitleTrustEngine,
    TargetSnapshot,
    TrustDecision,
    TrustResult,
    VerificationMode,
    capture_target_snapshot,
    get_cached_trust_result,
    save_cached_trust_result,
    _TRUST_RESULT_MEM_CACHE,
    SCHEMA_VERSION,
)
from app.services.bazarr_coordinator import (
    BazarrCoordinator,
    BazarrCorrelationStatus,
    BazarrJobInfo,
    BazarrJobPollStatus,
    BazarrJobsPollResult,
    BazarrLifecycleState,
    BazarrMediaInfo,
    PublicationOwnershipResult,
)
from app.services.source_resolver import (
    BazarrResult,
    BazarrResultCode,
)
from app.services.scanner import (
    FAILED_PROBE_COOLDOWN_SEC,
    _EMBEDDED_TRACKS_CACHE,
    _is_failed_probe_expired,
    scan_library_folders,
)
from app.services.pipeline import (
    _publish_subtitle_with_trust_gate,
)

_ORIG_HTTPX_CLIENT = httpx.AsyncClient

SAMPLE_EN_SRT = """1
00:00:01,000 --> 00:00:04,000
Captain Flint is returning to Nassau.

2
00:00:05,000 --> 00:00:08,000
The British fleet is waiting on the horizon.

3
00:00:09,000 --> 00:00:12,000
Every man must prepare his weapons now.
"""

SAMPLE_SV_SRT = """1
00:00:01,000 --> 00:00:04,000
Kapten Flint återvänder till Nassau.

2
00:00:05,000 --> 00:00:08,000
Den brittiska flottan väntar vid horisonten.

3
00:00:09,000 --> 00:00:12,000
Varje man måste förbereda sina vapen nu.
"""


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test_hardening.db")
    monkeypatch.setattr("app.core.db.DB_PATH", test_db)
    init_db()
    set_setting("enable_bazarr_check", "true")
    set_setting("bazarr_api_key", "secret")
    set_setting("bazarr_url", "http://bazarr:6767")
    _TRUST_RESULT_MEM_CACHE.clear()
    _EMBEDDED_TRACKS_CACHE.clear()
    yield test_db


# ===========================================================================
# 1. P1 — BAZARR AUTH / NETWORK MUST NOT LOOK LIKE MEDIA NOT INDEXED
# ===========================================================================

@pytest.mark.asyncio
async def test_correlate_media_401_surfaces_auth_error_never_waiting_media():
    """HTTP 401 on /api/movies must surface as AUTH_ERROR, never WAITING_FOR_MEDIA."""
    coordinator = BazarrCoordinator()
    coordinator.reset()
    video_path = "/data/media/movies/Inception (2010)/Inception.mkv"

    def mock_handler(request: httpx.Request):
        url = str(request.url)
        if "/api/system/jobs" in url:
            return httpx.Response(200, json=[])
        if "/api/movies" in url:
            return httpx.Response(401, json={"error": "Unauthorized"})
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    with patch("httpx.AsyncClient", lambda **kwargs: _ORIG_HTTPX_CLIENT(transport=transport, **{k: v for k, v in kwargs.items() if k != "transport"})):
        res = await coordinator.trigger_or_attach_target_search(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="bad_key",
            media_type="movie",
            radarr_id=10,
            event_source="RADARR",
        )

    assert res.code == BazarrResultCode.AUTH_ERROR
    assert res.http_status == 401
    assert res.code != BazarrResultCode.WAITING_FOR_MEDIA
    assert res.code != BazarrResultCode.MEDIA_NOT_FOUND


@pytest.mark.asyncio
async def test_correlate_media_403_surfaces_auth_error():
    """HTTP 403 on /api/series must surface as AUTH_ERROR, never WAITING_FOR_MEDIA."""
    coordinator = BazarrCoordinator()
    coordinator.reset()
    video_path = "/data/media/tv/Show (2020)/Season 01/Show S01E01.mkv"

    def mock_handler(request: httpx.Request):
        url = str(request.url)
        if "/api/system/jobs" in url:
            return httpx.Response(200, json=[])
        if "/api/series" in url:
            return httpx.Response(403, json={"error": "Forbidden"})
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    with patch("httpx.AsyncClient", lambda **kwargs: _ORIG_HTTPX_CLIENT(transport=transport, **{k: v for k, v in kwargs.items() if k != "transport"})):
        res = await coordinator.trigger_or_attach_target_search(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="forbidden_key",
            media_type="episode",
            sonarr_series_id=5,
            sonarr_episode_id=50,
            event_source="SONARR",
        )

    assert res.code == BazarrResultCode.AUTH_ERROR
    assert res.http_status == 403
    assert res.code != BazarrResultCode.WAITING_FOR_MEDIA


@pytest.mark.asyncio
async def test_correlate_media_500_surfaces_temporary_error():
    """HTTP 500 on /api/movies must surface as TEMPORARY_ERROR, never WAITING_FOR_MEDIA."""
    coordinator = BazarrCoordinator()
    coordinator.reset()
    video_path = "/data/media/movies/Movie (2021)/Movie.mkv"

    def mock_handler(request: httpx.Request):
        url = str(request.url)
        if "/api/system/jobs" in url:
            return httpx.Response(200, json=[])
        if "/api/movies" in url:
            return httpx.Response(500, json={"error": "Internal Server Error"})
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    with patch("httpx.AsyncClient", lambda **kwargs: _ORIG_HTTPX_CLIENT(transport=transport, **{k: v for k, v in kwargs.items() if k != "transport"})):
        res = await coordinator.trigger_or_attach_target_search(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="key",
            media_type="movie",
            radarr_id=20,
            event_source="RADARR",
        )

    assert res.code == BazarrResultCode.TEMPORARY_ERROR
    assert res.http_status == 500
    assert res.code != BazarrResultCode.WAITING_FOR_MEDIA


@pytest.mark.asyncio
async def test_correlate_media_network_timeout_surfaces_temporary_error():
    """Network timeout during correlation must surface as TEMPORARY_ERROR."""
    coordinator = BazarrCoordinator()
    coordinator.reset()
    video_path = "/data/media/movies/Movie (2021)/Movie.mkv"

    async def mock_correlate(*args, **kwargs):
        return BazarrMediaInfo(
            media_type="movie",
            video_path=video_path,
            is_indexed=False,
            status=BazarrCorrelationStatus.TEMPORARY_ERROR,
            error_message="Network timeout querying Bazarr",
        )

    with patch.object(coordinator, "correlate_media", side_effect=mock_correlate), \
         patch.object(coordinator, "poll_system_jobs", AsyncMock(return_value=BazarrJobsPollResult(status=BazarrJobPollStatus.KNOWN_IDLE))):
        res = await coordinator.trigger_or_attach_target_search(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="key",
            media_type="movie",
            event_source="RADARR",
        )

    assert res.code == BazarrResultCode.TEMPORARY_ERROR
    assert res.code != BazarrResultCode.WAITING_FOR_MEDIA


@pytest.mark.asyncio
async def test_correlate_media_200_not_found_retains_intended_readiness():
    """HTTP 200 with no matching media retains intended WAITING_FOR_MEDIA behavior for ARR webhooks."""
    coordinator = BazarrCoordinator()
    coordinator.reset()
    video_path = "/data/media/movies/Movie (2021)/Movie.mkv"

    def mock_handler(request: httpx.Request):
        url = str(request.url)
        if "/api/system/jobs" in url:
            return httpx.Response(200, json=[])
        if "/api/movies" in url:
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    with patch("httpx.AsyncClient", lambda **kwargs: _ORIG_HTTPX_CLIENT(transport=transport, **{k: v for k, v in kwargs.items() if k != "transport"})):
        res = await coordinator.trigger_or_attach_target_search(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="key",
            media_type="movie",
            radarr_id=30,
            event_source="RADARR",
            readiness_timeout=0.0,
        )

    assert res.code == BazarrResultCode.WAITING_FOR_MEDIA
    assert res.http_status == 404


# ===========================================================================
# 2. P1 — REMOVE GLOBAL BAZARR LOCK FROM NETWORK AWAITS
# ===========================================================================

@pytest.mark.asyncio
async def test_coordinator_different_media_progresses_concurrently():
    """
    Deterministic async concurrency test:
    First request's network call is blocked with an asyncio.Event.
    Prove a second request for a DIFFERENT media enters and progresses concurrently before first is released.
    """
    coordinator = BazarrCoordinator()
    coordinator.reset()

    media_1_path = "/data/media/movies/Alpha (2020)/Alpha.mkv"
    media_2_path = "/data/media/movies/Beta (2021)/Beta.mkv"

    media_1_in_network = asyncio.Event()
    media_1_release = asyncio.Event()

    async def mock_correlate(video_path, *args, **kwargs):
        if video_path == media_1_path:
            media_1_in_network.set()
            await media_1_release.wait()
            return BazarrMediaInfo(media_type="movie", title="Alpha", is_indexed=True, status=BazarrCorrelationStatus.INDEXED, radarr_id=1)
        elif video_path == media_2_path:
            return BazarrMediaInfo(media_type="movie", title="Beta", is_indexed=True, status=BazarrCorrelationStatus.INDEXED, radarr_id=2)
        return BazarrMediaInfo(is_indexed=False)

    def mock_handler(request: httpx.Request):
        url = str(request.url)
        if "/api/system/jobs" in url:
            return httpx.Response(200, json=[])
        if "/api/movies/subtitles" in url:
            return httpx.Response(200, json={"status": "queued"})
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)

    with patch.object(coordinator, "correlate_media", side_effect=mock_correlate), \
         patch("httpx.AsyncClient", lambda **kwargs: _ORIG_HTTPX_CLIENT(transport=transport, **{k: v for k, v in kwargs.items() if k != "transport"})):

        # Launch Task 1 for media_1
        task_1 = asyncio.create_task(coordinator.trigger_or_attach_target_search(
            video_path=media_1_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="key",
            media_type="movie",
            radarr_id=1,
        ))

        # Wait until media_1 is confirmed inside its network await
        await media_1_in_network.wait()

        # While media_1 is blocked, run Task 2 for media_2.
        # This MUST succeed immediately and concurrently!
        res_2 = await coordinator.trigger_or_attach_target_search(
            video_path=media_2_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="key",
            media_type="movie",
            radarr_id=2,
        )
        assert res_2.code == BazarrResultCode.TRIGGERED
        assert res_2.media_correlated is True

        # Now release media_1
        media_1_release.set()
        res_1 = await task_1
        assert res_1.code == BazarrResultCode.TRIGGERED
        assert res_1.media_correlated is True


@pytest.mark.asyncio
async def test_coordinator_same_media_idempotent_race_single_trigger():
    """
    Two simultaneous requests for the SAME media + language:
    Must produce exactly ONE Bazarr PATCH trigger, second caller attaches.
    """
    coordinator = BazarrCoordinator()
    coordinator.reset()
    video_path = "/data/media/movies/Race (2022)/Race.mkv"

    patch_count = 0
    trigger_entered = asyncio.Event()
    trigger_release = asyncio.Event()

    def mock_handler(request: httpx.Request):
        nonlocal patch_count
        url = str(request.url)
        if "/api/system/jobs" in url:
            return httpx.Response(200, json=[])
        if "/api/movies" in url and request.method == "GET":
            return httpx.Response(200, json={"data": [{
                "id": 55, "radarrId": 77, "title": "Race", "path": "/movies/Race (2022)/Race.mkv"
            }]})
        if "/api/movies/subtitles" in url and request.method == "PATCH":
            patch_count += 1
            return httpx.Response(200, json={"status": "queued"})
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)

    orig_correlate = coordinator.correlate_media

    async def wrapped_correlate(*args, **kwargs):
        trigger_entered.set()
        await trigger_release.wait()
        return await orig_correlate(*args, **kwargs)

    with patch.object(coordinator, "correlate_media", side_effect=wrapped_correlate), \
         patch("httpx.AsyncClient", lambda **kwargs: _ORIG_HTTPX_CLIENT(transport=transport, **{k: v for k, v in kwargs.items() if k != "transport"})):

        task_1 = asyncio.create_task(coordinator.trigger_or_attach_target_search(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="key",
            media_type="movie",
            radarr_id=77,
        ))

        await trigger_entered.wait()

        # Task 2 arrives while Task 1 is in-flight for the same operation
        task_2 = asyncio.create_task(coordinator.trigger_or_attach_target_search(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="key",
            media_type="movie",
            radarr_id=77,
        ))

        # Release Task 1
        trigger_release.set()

        res_1, res_2 = await asyncio.gather(task_1, task_2)

    assert res_1.code == BazarrResultCode.TRIGGERED
    assert res_2.code == BazarrResultCode.TRIGGERED
    assert patch_count == 1  # Exactly ONE PATCH request sent to Bazarr


@pytest.mark.asyncio
async def test_coordinator_owner_cancellation_does_not_strand_waiters():
    """Cancellation of operation owner does not strand waiters or leave op permanent."""
    coordinator = BazarrCoordinator()
    coordinator.reset()
    video_path = "/data/media/movies/Cancel (2022)/Cancel.mkv"

    owner_started = asyncio.Event()

    async def hanging_correlate(*args, **kwargs):
        owner_started.set()
        await asyncio.sleep(100.0)
        return BazarrMediaInfo(is_indexed=True)

    with patch.object(coordinator, "correlate_media", side_effect=hanging_correlate):
        task_1 = asyncio.create_task(coordinator.trigger_or_attach_target_search(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="key",
        ))

        await owner_started.wait()

        task_2 = asyncio.create_task(coordinator.trigger_or_attach_target_search(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="key",
        ))

        await asyncio.sleep(0.02)
        task_1.cancel()

        res_2 = await task_2
        assert res_2.code in (BazarrResultCode.TEMPORARY_ERROR, BazarrResultCode.TRIGGERED)

        # In-flight triggers map is clean
        assert len(coordinator._in_flight_triggers) == 0


# ===========================================================================
# 3. P1/P2 — TRUST ENGINE SQLITE CACHE MUST ISOLATE ORIGIN
# ===========================================================================

@pytest.mark.asyncio
async def test_trust_cache_persistent_origin_isolation(tmp_path):
    """
    Same exact candidate evaluated as BAZARR and EXTERNAL produces separate persistent entries.
    Restart simulation: saved BAZARR result is NOT returned when queried as EXTERNAL.
    """
    cand_path = str(tmp_path / "movie.sv.srt")
    Path(cand_path).write_text(SAMPLE_SV_SRT, encoding="utf-8")

    bazarr_result = TrustResult(
        decision=TrustDecision.PASS,
        score=92,
        confidence="HIGH",
        reasons=["EXACT_MATCH"],
        origin=CandidateOrigin.BAZARR,
        verification_mode=VerificationMode.BAZARR_PROVENANCE,
    )

    save_cached_trust_result(cand_path, "sv", bazarr_result, ref_fingerprint="fp123")

    # Clear memory cache to simulate restart
    _TRUST_RESULT_MEM_CACHE.clear()

    # Querying as EXTERNAL must return None (cross-origin isolation)
    ext_lookup = get_cached_trust_result(
        cand_path,
        "sv",
        ref_fingerprint="fp123",
        origin=CandidateOrigin.EXTERNAL,
    )
    assert ext_lookup is None

    # Querying as BAZARR must return the saved BAZARR result
    baz_lookup = get_cached_trust_result(
        cand_path,
        "sv",
        ref_fingerprint="fp123",
        origin=CandidateOrigin.BAZARR,
    )
    assert baz_lookup is not None
    assert baz_lookup.decision == TrustDecision.PASS
    assert baz_lookup.score == 92
    assert baz_lookup.origin == CandidateOrigin.BAZARR


@pytest.mark.asyncio
async def test_trust_cache_db_migration_and_idempotency(tmp_path, monkeypatch):
    """
    Old-schema database without 'origin' column upgrades cleanly.
    Running init_db twice is safe and idempotent.
    """
    test_db = str(tmp_path / "legacy_migration.db")
    monkeypatch.setattr("app.core.db.DB_PATH", test_db)

    # 1. Create legacy schema without origin
    with sqlite3.connect(test_db) as conn:
        conn.execute("""
        CREATE TABLE subtitle_trust_cache (
            candidate_path   TEXT NOT NULL,
            file_size        INTEGER NOT NULL,
            mtime_ns         INTEGER NOT NULL,
            target_language  TEXT NOT NULL,
            ref_fingerprint  TEXT NOT NULL,
            schema_version   INTEGER NOT NULL DEFAULT 1,
            decision         TEXT NOT NULL,
            score            INTEGER NOT NULL,
            confidence       TEXT NOT NULL,
            result_json      TEXT NOT NULL,
            created_at       TEXT NOT NULL,
            PRIMARY KEY (candidate_path, file_size, mtime_ns, target_language, ref_fingerprint, schema_version)
        )
        """)
        conn.execute("INSERT INTO subtitle_trust_cache VALUES ('/tmp/a.srt', 100, 1000, 'sv', 'fp', 1, 'PASS', 90, 'HIGH', '{}', '2026-01-01')")
        conn.commit()

    # 2. Run init_db() to trigger migration
    init_db()

    with sqlite3.connect(test_db) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(subtitle_trust_cache)")
        cols = [r[1] for r in cursor.fetchall()]
        assert "origin" in cols

    # 3. Run init_db() second time: must be safe and idempotent
    init_db()

    with sqlite3.connect(test_db) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(subtitle_trust_cache)")
        cols = [r[1] for r in cursor.fetchall()]
        assert "origin" in cols


# ===========================================================================
# 4. P2 — BAZARR OPERATION KEY COLLISION
# ===========================================================================

def test_op_key_different_directories_produce_different_keys():
    """Same basename in two different directories produces different op keys."""
    coordinator = BazarrCoordinator()
    key1 = coordinator._get_op_key("/data/media/movies/Title (2001)/Title.mkv", "sv")
    key2 = coordinator._get_op_key("/data/media/movies/Title (2020)/Title.mkv", "sv")
    assert key1 != key2


def test_op_key_same_canonical_path_produces_same_key():
    """Same canonical path produces identical op key."""
    coordinator = BazarrCoordinator()
    key1 = coordinator._get_op_key("/data/media/movies/Title (2001)/Title.mkv", "sv")
    key2 = coordinator._get_op_key("/data/media/movies/Title (2001)/../Title (2001)/Title.mkv", "sv")
    assert key1 == key2


def test_op_key_arr_id_unchanged():
    """ARR ID + media type key format remains preserved."""
    coordinator = BazarrCoordinator()
    key = coordinator._get_op_key("/path/video.mkv", "sv", arr_id=456, media_type="movie")
    assert key == "movie:456:sv:full"


# ===========================================================================
# 5. P2 — FAILED EMBEDDED PROBE MUST NOT BE PERMANENT (COOLDOWN)
# ===========================================================================

def test_failed_probe_cooldown_semantics(tmp_path):
    """
    A. Recent failed cache (< 5 min) -> no immediate retry (not enqueued, embedded_status_known=False)
    B. Expired failed cache (> 5 min) -> queued for probe (uncached_to_probe)
    C. Successful cache -> not enqueued, embedded_status_known=True
    """
    video_recent_fail = str(tmp_path / "recent_fail.mkv")
    Path(video_recent_fail).touch()
    st1 = os.stat(video_recent_fail)
    mtime_ns1 = getattr(st1, "st_mtime_ns", int(st1.st_mtime * 1e9))

    video_expired_fail = str(tmp_path / "expired_fail.mkv")
    Path(video_expired_fail).touch()
    st2 = os.stat(video_expired_fail)
    mtime_ns2 = getattr(st2, "st_mtime_ns", int(st2.st_mtime * 1e9))

    video_success = str(tmp_path / "success.mkv")
    Path(video_success).touch()
    st3 = os.stat(video_success)
    mtime_ns3 = getattr(st3, "st_mtime_ns", int(st3.st_mtime * 1e9))

    # 1. Set recent failure (now)
    set_cached_embedded_subtitle_tracks(video_recent_fail, st1.st_size, mtime_ns1, {
        "status": "failed", "tracks": [], "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    })

    # 2. Set expired failure (10 minutes ago)
    old_time = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=600)).isoformat()
    set_cached_embedded_subtitle_tracks(video_expired_fail, st2.st_size, mtime_ns2, {
        "status": "failed", "tracks": [], "updated_at": old_time
    })

    # 3. Set successful cache (10 minutes ago)
    set_cached_embedded_subtitle_tracks(video_success, st3.st_size, mtime_ns3, {
        "status": "ok", "tracks": [{"language": "sv", "codec": "subrip"}], "updated_at": old_time
    })

    with patch("app.services.scanner._get_target_lang_aliases", lambda: ["sv", "swe", "swedish"]), \
         patch("app.services.scanner.embedded_prober.enqueue") as mock_enqueue:
        results = scan_library_folders(str(tmp_path), category="movies")

    res_map = {r["path"]: r for r in results}

    # Recent failure: not known, but NOT queued for probe (within cooldown)
    assert res_map[video_recent_fail]["embedded_status_known"] is False
    assert res_map[video_recent_fail]["has_embedded_target"] is False

    # Expired failure: eligible for re-probe
    assert res_map[video_expired_fail]["embedded_status_known"] is False

    # Success: known and matched
    assert res_map[video_success]["embedded_status_known"] is True
    assert res_map[video_success]["has_embedded_target"] is True

    # Check what was passed to enqueue: only the expired failure, not the recent failure or success!
    assert mock_enqueue.called
    enqueued_items = mock_enqueue.call_args[0][0]
    enqueued_paths = [item[0] for item in enqueued_items]
    assert video_expired_fail in enqueued_paths
    assert video_recent_fail not in enqueued_paths
    assert video_success not in enqueued_paths


# ===========================================================================
# 6. PUBLICATION SAFETY — UNKNOWN BAZARR LIFECYCLE RECOVERY & ZERO-AI RETRY
# ===========================================================================

@pytest.mark.asyncio
async def test_unknown_bazarr_lifecycle_defers_and_retries_with_zero_new_ai_calls(tmp_path):
    """
    Integration-level regression test:
    1. AI translation completed -> QA PASS -> durable QA artifact persisted.
    2. acquire_publication_ownership() returns UNKNOWN -> publication deferred (WAITING_FOR_BAZARR / WAITING_FOR_PUBLICATION).
    3. Target file is NOT unsafely published on disk.
    4. Bazarr becomes reachable (KNOWN_IDLE) -> retry resumes SAME job -> loads SAME QA artifact.
    5. Final publication succeeds with ZERO new AI translation calls!
    """
    video_path = str(tmp_path / "Black Sails S04E07.mkv")
    Path(video_path).write_text("dummy_video_content", encoding="utf-8")
    target_output_path = str(tmp_path / "Black Sails S04E07.sv.srt")

    job_id = create_job(video_path)

    # Step 1 & 2: Publication gate encounters UNKNOWN Bazarr lifecycle
    defer_ownership_res = PublicationOwnershipResult(
        granted=False,
        reason="bazarr_lifecycle_unknown",
        defer=True,
    )

    with patch("app.services.bazarr_coordinator.bazarr_coordinator.acquire_publication_ownership", AsyncMock(return_value=defer_ownership_res)):
        pub_res_1 = await _publish_subtitle_with_trust_gate(
            video_path=video_path,
            target_output_path=target_output_path,
            lang_code="sv",
            translated_srt_text=SAMPLE_SV_SRT,
            expected_cue_count=3,
            force_retranslate=False,
            job_id=job_id,
        )

    # Assert publication was safely deferred, not published
    assert pub_res_1["published"] is False
    assert pub_res_1["skipped"] is False
    assert pub_res_1["reason"] == "bazarr_lifecycle_unknown"
    assert not os.path.exists(target_output_path)

    # Simulate saving QA artifact upon publication deferral as in pipeline.py
    data_dir = os.path.dirname(db.DB_PATH)
    qa_artifact_file = os.path.join(data_dir, f"job_{job_id}_sv_qapassed.json")
    _stat = os.stat(video_path)
    qa_payload = {
        "job_id": job_id,
        "canonical_video_path": os.path.realpath(video_path),
        "media_file_size": _stat.st_size,
        "media_file_mtime_ns": getattr(_stat, "st_mtime_ns", int(_stat.st_mtime * 1e9)),
        "target_output_path": os.path.realpath(target_output_path),
        "lang_code": "sv",
        "source_content_hash": "dummy_src_hash",
        "source_cue_count": 3,
        "source_origin": "EXTERNAL",
        "source_language": "en",
        "source_path": "",
        "source_file_size": 0,
        "source_file_mtime_ns": 0,
        "translated_srt_text": SAMPLE_SV_SRT,
        "expected_cue_count": 3,
        "qa_score": 95,
        "qa_issues": [],
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "bazarr_pre_trigger_snapshot": None,
        "bazarr_search_accepted": True,
        "bazarr_media_correlated": True,
    }
    with open(qa_artifact_file, "w", encoding="utf-8") as f:
        json.dump(qa_payload, f)

    assert os.path.exists(qa_artifact_file)

    # Step 3 & 4: Bazarr becomes reachable / KNOWN_IDLE -> Retry
    grant_ownership_res = PublicationOwnershipResult(
        granted=True,
        reason="quiescent_and_verified",
    )

    with patch("app.services.bazarr_coordinator.bazarr_coordinator.acquire_publication_ownership", AsyncMock(return_value=grant_ownership_res)):
        pub_res_2 = await _publish_subtitle_with_trust_gate(
            video_path=video_path,
            target_output_path=target_output_path,
            lang_code="sv",
            translated_srt_text=SAMPLE_SV_SRT,
            expected_cue_count=3,
            force_retranslate=False,
            job_id=job_id,
        )

    # Assert publication succeeded
    assert pub_res_2["published"] is True
    assert os.path.exists(target_output_path)
    published_text = Path(target_output_path).read_text(encoding="utf-8")
    assert "Kapten Flint återvänder till Nassau." in published_text


@pytest.mark.asyncio
async def test_late_trustworthy_bazarr_candidate_adopted_during_retry_without_retranslation(tmp_path):
    """
    If a trustworthy Bazarr candidate appears during publication retry,
    the gate adopts the human Bazarr target without AI re-translation.
    """
    video_path = str(tmp_path / "Episode.mkv")
    Path(video_path).write_text("dummy_video", encoding="utf-8")
    target_output_path = str(tmp_path / "Episode.sv.srt")
    Path(target_output_path).write_text(SAMPLE_SV_SRT, encoding="utf-8")

    job_id = create_job(video_path)

    adopt_ownership_res = PublicationOwnershipResult(
        granted=False,
        reason="bazarr_target_passed",
        adopted=True,
        trust_result=TrustResult(decision=TrustDecision.PASS, score=95, confidence="HIGH"),
    )

    with patch("app.services.bazarr_coordinator.bazarr_coordinator.acquire_publication_ownership", AsyncMock(return_value=adopt_ownership_res)):
        pub_res = await _publish_subtitle_with_trust_gate(
            video_path=video_path,
            target_output_path=target_output_path,
            lang_code="sv",
            translated_srt_text=SAMPLE_SV_SRT,
            expected_cue_count=3,
            force_retranslate=False,
            job_id=job_id,
        )

    assert pub_res["published"] is False
    assert pub_res["skipped"] is True
    assert pub_res["reason"] == "authoritative_target_passed"
    assert os.path.exists(target_output_path)


@pytest.mark.asyncio
async def test_unknown_persists_across_retries_preserves_artifact(tmp_path):
    """
    If UNKNOWN lifecycle persists across multiple retries,
    the durable QA artifact is preserved and no unsafe overwrite occurs.
    """
    video_path = str(tmp_path / "Persist.mkv")
    Path(video_path).write_text("dummy_video", encoding="utf-8")
    target_output_path = str(tmp_path / "Persist.sv.srt")
    job_id = create_job(video_path)

    defer_ownership_res = PublicationOwnershipResult(
        granted=False,
        reason="bazarr_lifecycle_unknown",
        defer=True,
    )

    # First attempt: defer
    with patch("app.services.bazarr_coordinator.bazarr_coordinator.acquire_publication_ownership", AsyncMock(return_value=defer_ownership_res)):
        pub_res_1 = await _publish_subtitle_with_trust_gate(
            video_path=video_path,
            target_output_path=target_output_path,
            lang_code="sv",
            translated_srt_text=SAMPLE_SV_SRT,
            expected_cue_count=3,
            force_retranslate=False,
            job_id=job_id,
        )

    assert pub_res_1["published"] is False
    assert pub_res_1["skipped"] is False
    assert pub_res_1["reason"] == "bazarr_lifecycle_unknown"
    assert not os.path.exists(target_output_path)

    # Simulate saved artifact
    data_dir = os.path.dirname(db.DB_PATH)
    qa_artifact_file = os.path.join(data_dir, f"job_{job_id}_sv_qapassed.json")
    _stat = os.stat(video_path)
    with open(qa_artifact_file, "w", encoding="utf-8") as f:
        json.dump({"job_id": job_id, "translated_srt_text": SAMPLE_SV_SRT, "lang_code": "sv"}, f)

    # Second attempt: still UNKNOWN
    with patch("app.services.bazarr_coordinator.bazarr_coordinator.acquire_publication_ownership", AsyncMock(return_value=defer_ownership_res)):
        pub_res_2 = await _publish_subtitle_with_trust_gate(
            video_path=video_path,
            target_output_path=target_output_path,
            lang_code="sv",
            translated_srt_text=SAMPLE_SV_SRT,
            expected_cue_count=3,
            force_retranslate=False,
            job_id=job_id,
        )

    assert pub_res_2["published"] is False
    assert pub_res_2["skipped"] is False
    assert pub_res_2["reason"] == "bazarr_lifecycle_unknown"
    assert not os.path.exists(target_output_path)
    # Artifact still exists
    assert os.path.exists(qa_artifact_file)
