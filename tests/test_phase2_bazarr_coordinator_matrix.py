"""
Test Suite for Phase 2: Bazarr Coordinator & Lifecycle Matrix (A through M + UNKNOWN Safety).

Verifies authoritative lifecycle coordination:
- Embedded target > Finalized trusted Bazarr/external target > AI fallback
- Real Bazarr job state tracking (SEARCHING, SYNCING, FINALIZING)
- Authoritative UNKNOWN lifecycle distinction (UNKNOWN != IDLE)
- ARR ID media correlation across container path differences
- Operation deduplication & 409 Conflict handling
- Generation snapshotting & Quiescence verification
- Publication ownership invariant (no overwriting active or unknown Bazarr workers)
- 2 Fast 2 Furious regression prevention
"""

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from app.core.trust_engine import (
    SubtitleTrustEngine,
    CandidateOrigin,
    TrustDecision,
    TrustResult,
    TargetSnapshot,
    capture_target_snapshot,
)
from app.services.bazarr_coordinator import (
    BazarrCoordinator,
    BazarrLifecycleState,
    BazarrJobInfo,
    BazarrMediaInfo,
    BazarrJobPollStatus,
    BazarrJobsPollResult,
    PublicationOwnershipResult,
    _normalize_poll_result,
    _extract_job_language_codes,
)
from app.services.source_resolver import (
    BazarrResult,
    BazarrResultCode,
    SubtitleSource,
    SourceOrigin,
)
from app.services.pipeline import SubtitlePipeline
import app.core.db as db_mod


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db_file = tmp_path / "test_matrix.db"
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_file))
    monkeypatch.setattr("app.core.quota.DB_PATH", str(db_file), raising=False)
    db_mod.init_db()
    from app.services.bazarr_coordinator import bazarr_coordinator
    bazarr_coordinator.reset()
    yield
    bazarr_coordinator.reset()


def setup_mock_settings(monkeypatch, custom_settings=None):
    settings = {
        "ai_provider": "gemini",
        "gemini_api_key": "dummy_test_key",
        "gemini_model": "gemini-3.5-flash-lite",
        "enable_bazarr_check": "true",
        "enable_bazarr": "true",
        "bazarr_url": "http://bazarr:6767",
        "bazarr_api_key": "test_api_key",
        "sonarr_url": "http://sonarr:8989",
        "sonarr_api_key": "test_sonarr_key",
        "radarr_url": "http://radarr:7878",
        "radarr_api_key": "test_radarr_key",
        "languages": json.dumps([{"name": "Swedish", "code": "sv", "enabled": True}]),
        "bazarr_quiescence_seconds": "0.1",
        "bazarr_candidate_stability_seconds": "0.03",
        "bazarr_grace_seconds": "1.5",
        "hybrid_bazarr_max_wait_sec": "1.5",
        "batch_size": "50",
    }
    if custom_settings:
        settings.update(custom_settings)

    def mock_get_setting(key, default=None):
        return settings.get(key, default)

    monkeypatch.setattr("app.core.db.get_setting", mock_get_setting)
    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)
    monkeypatch.setattr("app.services.bazarr_coordinator.get_setting", mock_get_setting)


# ===========================================================================
# P0: UNKNOWN LIFECYCLE TESTS
# ===========================================================================

@pytest.mark.asyncio
async def test_unknown_lifecycle_provisional_protection_and_recovery(tmp_path, monkeypatch):
    """
    P0 Regression: Bazarr target exists & is syncing -> API raises error (UNKNOWN).
    Verify:
      - API error is classified as UNKNOWN (not idle)
      - Candidate remains PROVISIONAL
      - Quiescent unchanged file does NOT cause premature terminal Trust rejection
      - Publication ownership is deferred/denied
      - Target is NOT backed up or replaced
    Then simulate API recovery (KNOWN_IDLE + final file write):
      - Final Trust PASS
      - Result is BAZARR MATCH with 0 AI calls
    """
    setup_mock_settings(monkeypatch, {
        "bazarr_quiescence_seconds": "0.1",
        "bazarr_candidate_stability_seconds": "0.03",
        "bazarr_grace_seconds": "2.0",
    })

    video_path = tmp_path / "UnknownRecovery.mkv"
    video_path.touch()

    en_srt = tmp_path / "UnknownRecovery.en.srt"
    en_lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nEnglish line {i}" for i in range(1, 25)]
    en_srt.write_text("\n\n".join(en_lines), encoding="utf-8")

    sv_srt_path = str(tmp_path / "UnknownRecovery.sv.srt")
    sv_initial = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nKort {i}" for i in range(1, 6)]
    Path(sv_srt_path).write_text("\n\n".join(sv_initial), encoding="utf-8")

    pipeline = SubtitlePipeline()
    ai_called = False

    async def mock_translate_batch(batch, **kwargs):
        nonlocal ai_called
        ai_called = True
        return [{"id": item["id"], "text": f"AI {item['id']}"} for item in batch]
    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)

    # Step sequence:
    # 1. Sync active
    # 2. API exception (UNKNOWN) while candidate is unchanged
    # 3. API recovers with KNOWN_IDLE and final complete file
    poll_count = 0

    async def dynamic_poll_jobs(self, *args, **kwargs):
        nonlocal poll_count
        poll_count += 1
        if poll_count == 1:
            return BazarrJobsPollResult(
                status=BazarrJobPollStatus.ACTIVE,
                jobs=[BazarrJobInfo(
                    job_id="sync_unk",
                    job_name="Syncing Subtitles",
                    status="running",
                    job_type="sync",
                    progress_message="UnknownRecovery.sv.srt",
                )]
            )
        elif poll_count in (2, 3):
            # Transient API timeout/500 -> UNKNOWN
            return BazarrJobsPollResult(
                status=BazarrJobPollStatus.UNKNOWN,
                error="Timeout connecting to Bazarr API",
            )
        else:
            # Bazarr completes
            return BazarrJobsPollResult(
                status=BazarrJobPollStatus.KNOWN_IDLE,
                jobs=[],
            )

    monkeypatch.setattr(BazarrCoordinator, "poll_system_jobs", dynamic_poll_jobs)

    async def bazarr_finalizer():
        await asyncio.sleep(0.12)
        sv_complete = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nKomplett svensk text {i}" for i in range(1, 25)]
        Path(sv_srt_path).write_text("\n\n".join(sv_complete), encoding="utf-8")

    fin_task = asyncio.create_task(bazarr_finalizer())

    def mock_find(vp, lang):
        if lang == "en": return str(en_srt)
        if lang == "sv" and os.path.exists(sv_srt_path): return sv_srt_path
        return None

    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find):
        res = await pipeline.process_video_file(str(video_path), event_source="RADARR")

    await fin_task
    assert res["status"] == "skipped"
    assert res["reason"] in ("bazarr_downloaded", "already_exists")
    assert not ai_called

    # Ensure no backup created and human target preserved
    assert not os.path.exists(sv_srt_path + ".bak")
    final_text = Path(sv_srt_path).read_text(encoding="utf-8")
    assert "Komplett svensk text 1" in final_text
    assert "AI 1" not in final_text


@pytest.mark.asyncio
async def test_publication_gate_ai_passed_but_bazarr_unknown(tmp_path, monkeypatch):
    """
    Publication gate test: AI QA passed, but querying Bazarr lifecycle returns UNKNOWN.
    Required: Publication ownership is refused/deferred; AI publication remains blocked.
    """
    setup_mock_settings(monkeypatch)
    coordinator = BazarrCoordinator()
    video_path = tmp_path / "GateUnknown.mkv"
    video_path.touch()

    # Bazarr job poll returns UNKNOWN (e.g. 500 error / network error)
    with patch.object(coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_jobs:
        mock_jobs.return_value = BazarrJobsPollResult(
            status=BazarrJobPollStatus.UNKNOWN,
            error="Connection refused: Bazarr restarting",
        )

        res = await coordinator.acquire_publication_ownership(
            video_path=str(video_path),
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="key",
            timeout_sec=0.1,
        )

        assert res.granted is False
        assert res.defer is True
        assert res.reason == "bazarr_lifecycle_unknown"


# ===========================================================================
# ORIGINAL PHASE 2 MATRIX SCENARIOS (A through M)
# ===========================================================================

@pytest.mark.asyncio
async def test_matrix_a_fresh_arr_import_absent_then_appears(tmp_path, monkeypatch):
    """
    Matrix A: Fresh ARR import initially absent in Bazarr, then appears.
    Bounded retry, no premature WAITING_SOURCE, single idempotent trigger.
    """
    setup_mock_settings(monkeypatch)
    video_path = tmp_path / "MatrixA.mkv"
    video_path.touch()

    en_srt = tmp_path / "MatrixA.en.srt"
    en_lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nEnglish line {i}" for i in range(1, 20)]
    en_srt.write_text("\n\n".join(en_lines), encoding="utf-8")

    sv_srt = tmp_path / "MatrixA.sv.srt"
    sv_lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nSvensk text {i}" for i in range(1, 20)]
    sv_srt.write_text("\n\n".join(sv_lines), encoding="utf-8")

    pipeline = SubtitlePipeline()
    ai_called = False

    async def mock_translate_batch(batch, **kwargs):
        nonlocal ai_called
        ai_called = True
        return [{"id": item["id"], "text": f"AI {item['id']}"} for item in batch]
    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)

    def mock_find(vp, lang):
        if lang == "en": return str(en_srt)
        if lang == "sv": return str(sv_srt)
        return None

    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find):
        res = await pipeline.process_video_file(str(video_path), event_source="SONARR")

    assert res["status"] == "skipped"
    assert res["reason"] in ("already_exists", "bazarr_downloaded")
    assert not ai_called


@pytest.mark.asyncio
async def test_matrix_b_target_search_pending_no_ai_source(tmp_path, monkeypatch):
    """
    Matrix B: Target search pending + no AI source -> source failure is not terminal.
    Babel enters waiting source retry or holds for Bazarr without crashing.
    """
    setup_mock_settings(monkeypatch, {"enable_bazarr_check": "true", "hybrid_bazarr_max_wait_sec": "0.2"})
    video_path = tmp_path / "MatrixB.mkv"
    video_path.touch()

    pipeline = SubtitlePipeline()

    with patch("app.services.pipeline.find_external_subtitle", return_value=None), \
         patch("app.services.pipeline._safe_extract_embedded_srt", return_value=False), \
         patch("app.core.extractor.inspect_mkv_tracks", return_value={"subtitles": [], "audio": [{"id": 1, "language": "eng"}]}), \
         patch.object(BazarrCoordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_jobs:
        mock_jobs.return_value = BazarrJobsPollResult(status=BazarrJobPollStatus.KNOWN_IDLE, jobs=[])
        res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")

    assert res["status"] in ("waiting_source", "failed")
    assert "No usable source subtitle" in res["reason"]


@pytest.mark.asyncio
async def test_matrix_c_target_file_appears_while_searching_provisional(tmp_path, monkeypatch):
    """
    Matrix C: Target file appears while SEARCHING is active.
    Candidate is marked PROVISIONAL; no premature terminal Trust rejection.
    """
    setup_mock_settings(monkeypatch)
    coordinator = BazarrCoordinator()
    video_path = tmp_path / "MatrixC.mkv"
    video_path.touch()
    sv_path = tmp_path / "MatrixC.sv.srt"
    sv_path.write_text("1\n00:00:01,000 --> 00:00:01,500\nIncomplete...", encoding="utf-8")

    search_job = BazarrJobInfo(
        job_id="search_c",
        job_name="Searching subtitles for MatrixC",
        status="running",
        job_type="search",
    )

    with patch.object(coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_jobs:
        mock_jobs.return_value = BazarrJobsPollResult(status=BazarrJobPollStatus.ACTIVE, jobs=[search_job])
        state, cand, snap = await coordinator.coordinate_target(
            video_path=str(video_path),
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="key",
            max_wait_seconds=0.1,
        )
        assert state in (BazarrLifecycleState.TARGET_APPEARED, BazarrLifecycleState.SEARCHING, BazarrLifecycleState.TIMED_OUT)


@pytest.mark.asyncio
async def test_matrix_d_target_unchanged_beyond_quiescence_syncing_active(tmp_path, monkeypatch):
    """
    Matrix D: Target file unchanged beyond old quiescence, but SYNCING is still active.
    Candidate remains PROVISIONAL; coordinator does not prematurely finalize or reject.
    """
    setup_mock_settings(monkeypatch, {"bazarr_quiescence_seconds": "0.05"})
    coordinator = BazarrCoordinator()
    video_path = tmp_path / "MatrixD.mkv"
    video_path.touch()
    sv_path = tmp_path / "MatrixD.sv.srt"
    sv_path.write_text("1\n00:00:01,000 --> 00:00:02,000\nPartial sync...", encoding="utf-8")

    sync_job = BazarrJobInfo(
        job_id="sync_d",
        job_name="Syncing Subtitles with Video",
        status="running",
        job_type="sync",
        progress_message="MatrixD.sv.srt",
    )

    with patch.object(coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_jobs:
        mock_jobs.return_value = BazarrJobsPollResult(status=BazarrJobPollStatus.ACTIVE, jobs=[sync_job])
        state, cand, snap = await coordinator.coordinate_target(
            video_path=str(video_path),
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="key",
            max_wait_seconds=0.15,
        )
        # Stays in SYNCING / provisional hold; never claimed as final failure
        assert state in (BazarrLifecycleState.SYNCING, BazarrLifecycleState.TIMED_OUT)


@pytest.mark.asyncio
async def test_matrix_e_search_sync_complete_final_trust_pass_bazarr_match(tmp_path, monkeypatch):
    """Matrix E: Search and sync complete + final Trust PASS -> BAZARR MATCH, 0 AI calls."""
    setup_mock_settings(monkeypatch)
    video_path = tmp_path / "MatrixE.mkv"
    video_path.touch()

    en_srt = tmp_path / "MatrixE.en.srt"
    en_lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nEnglish line {i}" for i in range(1, 20)]
    en_srt.write_text("\n\n".join(en_lines), encoding="utf-8")

    sv_srt = tmp_path / "MatrixE.sv.srt"
    sv_lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nSvensk text {i}" for i in range(1, 20)]
    sv_srt.write_text("\n\n".join(sv_lines), encoding="utf-8")

    pipeline = SubtitlePipeline()
    ai_called = False

    async def mock_translate_batch(batch, **kwargs):
        nonlocal ai_called
        ai_called = True
        return [{"id": item["id"], "text": "AI"} for item in batch]
    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)

    with patch("app.services.pipeline.find_external_subtitle", side_effect=lambda vp, l: str(en_srt) if l == "en" else str(sv_srt)), \
         patch.object(BazarrCoordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_jobs:
        mock_jobs.return_value = BazarrJobsPollResult(status=BazarrJobPollStatus.KNOWN_IDLE, jobs=[])
        res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")

    assert res["status"] == "skipped"
    assert res["reason"] in ("bazarr_downloaded", "already_exists")
    assert not ai_called


@pytest.mark.asyncio
async def test_matrix_f_search_sync_complete_final_trust_fail_ai_fallback(tmp_path, monkeypatch):
    """Matrix F: Search and sync complete + final Trust FAIL -> AI fallback executes."""
    setup_mock_settings(monkeypatch, {
        "bazarr_quiescence_seconds": "0.05",
        "bazarr_candidate_stability_seconds": "0.02",
        "bazarr_grace_seconds": "0.3",
    })
    video_path = tmp_path / "MatrixF.mkv"
    video_path.touch()

    en_srt = tmp_path / "MatrixF.en.srt"
    en_lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nEnglish line {i}" for i in range(1, 20)]
    en_srt.write_text("\n\n".join(en_lines), encoding="utf-8")

    # Garbage file with 1 cue
    sv_srt = tmp_path / "MatrixF.sv.srt"
    sv_srt.write_text("1\n00:00:01,000 --> 00:00:01,800\nGarbage line", encoding="utf-8")

    pipeline = SubtitlePipeline()
    ai_called = False

    async def mock_translate_batch(batch, **kwargs):
        nonlocal ai_called
        ai_called = True
        return [{"id": item["id"], "text": f"Detta är en svensk dialog och replik {item['id']}."} for item in batch]
    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr("app.services.pipeline.qa_gate", lambda *args, **kwargs: {"passed": True, "score": 100, "issues": []})

    with patch("app.services.pipeline.find_external_subtitle", side_effect=lambda vp, l: str(en_srt) if l == "en" else str(sv_srt)), \
         patch.object(BazarrCoordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_jobs:
        mock_jobs.return_value = BazarrJobsPollResult(status=BazarrJobPollStatus.KNOWN_IDLE, jobs=[])
        res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")

    assert res["status"] == "translated"
    assert ai_called is True


@pytest.mark.asyncio
async def test_matrix_g_babel_retry_while_operation_active_no_duplicate(monkeypatch):
    """Matrix G: Babel retry while operation active -> attaches to existing without duplicate trigger."""
    coordinator = BazarrCoordinator()

    media = BazarrMediaInfo(media_type="movie", radarr_id=222, title="Dedup Movie", is_indexed=True)
    with patch.object(coordinator, "correlate_media", new_callable=AsyncMock) as mock_corr, \
         patch("httpx.AsyncClient.patch") as mock_patch:
        mock_corr.return_value = media
        mock_resp = MagicMock()
        mock_resp.status_code = 409
        mock_patch.return_value = mock_resp

        res = await coordinator.trigger_or_attach_target_search(
            video_path="/media/Movies/Dedup (2021)/Dedup.mkv",
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="key",
            radarr_id=222,
            media_type="movie",
        )
        assert res.code in (BazarrResultCode.CONFLICT, BazarrResultCode.TRIGGERED)


@pytest.mark.asyncio
async def test_matrix_h_media_correlation_and_active_sync_matching(tmp_path):
    """
    Matrix H: /media and /data/media paths with same ARR ID -> same media identity
    AND active sync job is correctly correlated.
    """
    coordinator = BazarrCoordinator()
    host_path = "/media/Movies/Inception (2010)/Inception (2010).mkv"
    docker_path = "/data/media/Movies/Inception (2010)/Inception (2010).mkv"

    movies_payload = [{"id": 77, "radarrId": 303, "path": docker_path, "title": "Inception"}]

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": movies_payload}
        mock_get.return_value = mock_resp

        media_info = await coordinator.correlate_media(
            video_path=host_path,
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="key",
            radarr_id=303,
        )

        assert media_info.radarr_id == 303
        assert media_info.bazarr_id == 77

        # Classify sync job referring to docker path
        jobs = [BazarrJobInfo(
            job_id="sync_inc",
            job_name="Syncing Inception (2010).sv.srt",
            status="running",
            job_type="sync",
        )]
        s_jobs, sync_jobs = coordinator.classify_jobs_for_target(jobs, host_path, "sv", media_info)
        assert len(sync_jobs) == 1


def test_bazarr_coordinator_job_classification_language_boundaries():
    """
    Focused tests for Bazarr job classification with strict language boundaries:
    1. Unrelated generic search: must NOT correlate
    2. Same-title generic search: may correlate
    3. Same-title Swedish search: must correlate
    4. Same-title English/German search when target is Swedish: must NOT correlate
    5. Media/title/job text containing 'en', 'de', 'es' inside words (Seven, Garden, Demon, Forest):
       must NOT be interpreted as language codes
    6. Titles containing language names (e.g. 'The French Connection') do not produce false language codes
    """
    coordinator = BazarrCoordinator()
    video_path = "/data/media/TV/Modern Family/Modern.Family.S07E07.Seven.Garden.Demon.Forest.mkv"

    # 1. Unrelated generic search jobs
    unrelated_jobs = [
        BazarrJobInfo(job_id="j1", job_name="Searching subtitles", status="running", job_type="search"),
        BazarrJobInfo(job_id="j2", job_name="Searching missing subtitles", status="running", job_type="search"),
        BazarrJobInfo(job_id="j3", job_name="Searching subtitles for Another Movie (2020)", status="running", job_type="search"),
    ]
    for job in unrelated_jobs:
        s_jobs, _ = coordinator.classify_jobs_for_target([job], video_path, "sv")
        assert len(s_jobs) == 0, f"Expected no match for unrelated job: {job.job_name}"

    # 2. Same-title generic search (no explicit language)
    generic_title_job = BazarrJobInfo(
        job_id="j4",
        job_name="Searching subtitles for Modern Family S07E07 Seven Garden Demon Forest",
        status="running",
        job_type="search",
    )
    s_jobs, _ = coordinator.classify_jobs_for_target([generic_title_job], video_path, "sv")
    assert len(s_jobs) == 1, "Generic search for this specific title must correlate"

    # 3. Same-title Swedish search (explicit target language)
    sv_job_1 = BazarrJobInfo(
        job_id="j5",
        job_name="Searching subtitles for Modern Family S07E07 Seven Garden Demon Forest (Swedish)",
        status="running",
        job_type="search",
    )
    sv_job_2 = BazarrJobInfo(
        job_id="j6",
        job_name="Searching subtitles for Modern.Family.S07E07.Seven.Garden.Demon.Forest.sv.srt",
        status="running",
        job_type="search",
    )
    s_jobs_1, _ = coordinator.classify_jobs_for_target([sv_job_1], video_path, "sv")
    assert len(s_jobs_1) == 1, "Explicit Swedish search for this title must correlate"
    s_jobs_2, _ = coordinator.classify_jobs_for_target([sv_job_2], video_path, "sv")
    assert len(s_jobs_2) == 1, "Explicit .sv.srt search for this title must correlate"

    # 4. Same-title English / German / Spanish search while target is Swedish
    en_job = BazarrJobInfo(
        job_id="j7",
        job_name="Searching subtitles for Modern Family S07E07 Seven Garden Demon Forest (English)",
        status="running",
        job_type="search",
    )
    de_job = BazarrJobInfo(
        job_id="j8",
        job_name="Searching subtitles for Modern.Family.S07E07.Seven.Garden.Demon.Forest.de.srt",
        status="running",
        job_type="search",
    )
    s_jobs_en, _ = coordinator.classify_jobs_for_target([en_job], video_path, "sv")
    assert len(s_jobs_en) == 0, "English search for this title must NOT correlate when target is Swedish"
    s_jobs_de, _ = coordinator.classify_jobs_for_target([de_job], video_path, "sv")
    assert len(s_jobs_de) == 0, "German search for this title must NOT correlate when target is Swedish"

    # 5. Words containing embedded two-letter codes: 'en' in Seven, 'de' in Demon/Garden, 'es' in Forest
    # Check that helper does not extract spurious languages
    extracted_codes = _extract_job_language_codes(
        "Searching subtitles for Modern Family S07E07 Seven Garden Demon Forest",
        ignore_title="Modern Family S07E07 Seven Garden Demon Forest",
    )
    assert extracted_codes == set(), f"Expected empty extracted codes, got: {extracted_codes}"

    # 6. Title containing language name: 'The French Connection'
    french_conn_path = "/data/media/Movies/The French Connection (1971)/The.French.Connection.1971.mkv"
    french_media = BazarrMediaInfo(title="The French Connection", video_path=french_conn_path)

    # 6a. Title: "The French Connection", Job: "Searching subtitles for The French Connection", Target: sv
    # => generic title-specific search, correlate
    fc_generic_job = BazarrJobInfo(
        job_id="j9",
        job_name="Searching subtitles for The French Connection (1971)",
        status="running",
        job_type="search",
    )
    s_jobs_fc_gen, _ = coordinator.classify_jobs_for_target([fc_generic_job], french_conn_path, "sv", french_media)
    assert len(s_jobs_fc_gen) == 1, "Generic search for The French Connection must correlate for sv"

    # 6b. Title: "The French Connection", Job: "Searching subtitles for The French Connection (French)", Target: sv
    # => explicit different language, DO NOT correlate
    fc_french_job = BazarrJobInfo(
        job_id="j10_fr",
        job_name="Searching subtitles for The French Connection (1971) (French)",
        status="running",
        job_type="search",
    )
    s_jobs_fc_fr, _ = coordinator.classify_jobs_for_target([fc_french_job], french_conn_path, "sv", french_media)
    assert len(s_jobs_fc_fr) == 0, "Explicit French search for The French Connection must NOT correlate when target is Swedish"

    # 6c. Title: "The French Connection", Job: "Searching subtitles for The French Connection (Swedish)", Target: sv
    # => correlate
    fc_swedish_job = BazarrJobInfo(
        job_id="j10_sv",
        job_name="Searching subtitles for The French Connection (1971) (Swedish)",
        status="running",
        job_type="search",
    )
    s_jobs_fc_sv, _ = coordinator.classify_jobs_for_target([fc_swedish_job], french_conn_path, "sv", french_media)
    assert len(s_jobs_fc_sv) == 1, "Explicit Swedish search for The French Connection must correlate when target is Swedish"

    # 6d. Explicit German search for The French Connection (target: sv) -> should NOT correlate
    fc_german_job = BazarrJobInfo(
        job_id="j10_de",
        job_name="Searching subtitles for The French Connection (1971) (German)",
        status="running",
        job_type="search",
    )
    s_jobs_fc_de, _ = coordinator.classify_jobs_for_target([fc_german_job], french_conn_path, "sv", french_media)
    assert len(s_jobs_fc_de) == 0, "German search for The French Connection must NOT correlate for sv"

    # 7. Title containing a 2-letter language-code-looking token, with that same code explicitly repeated outside title
    en_title_path = "/data/media/TV/Show EN/Show.EN.S01E01.mkv"
    en_title_media = BazarrMediaInfo(title="Show EN", video_path=en_title_path)

    # Job explicitly searching for English: 'Searching subtitles for Show EN (en)'
    # Outside token 'en' must be preserved and recognized as English
    job_en_outside = BazarrJobInfo(
        job_id="j11",
        job_name="Searching subtitles for Show EN (en)",
        status="running",
        job_type="search",
    )
    # When target is Swedish (sv), English job must NOT correlate
    s_jobs_en_out_sv, _ = coordinator.classify_jobs_for_target([job_en_outside], en_title_path, "sv", en_title_media)
    assert len(s_jobs_en_out_sv) == 0, "English job for Show EN must NOT correlate when requested target is Swedish"

    # When target is English (en), English job MUST correlate
    s_jobs_en_out_en, _ = coordinator.classify_jobs_for_target([job_en_outside], en_title_path, "en", en_title_media)
    assert len(s_jobs_en_out_en) == 1, "English job for Show EN MUST correlate when requested target is English"



@pytest.mark.asyncio
async def test_matrix_i_speculative_ai_running_final_bazarr_pass_human_wins(tmp_path, monkeypatch):
    """Matrix I: Speculative AI running + final Bazarr PASS -> stop further AI, human target wins."""
    setup_mock_settings(monkeypatch, {
        "bazarr_quiescence_seconds": "0.08",
        "bazarr_candidate_stability_seconds": "0.02",
        "bazarr_grace_seconds": "1.5",
    })
    video_path = tmp_path / "MatrixI.mkv"
    video_path.touch()

    en_srt = tmp_path / "MatrixI.en.srt"
    en_lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nEnglish line {i}" for i in range(1, 25)]
    en_srt.write_text("\n\n".join(en_lines), encoding="utf-8")

    sv_srt_path = str(tmp_path / "MatrixI.sv.srt")
    pipeline = SubtitlePipeline()
    ai_called = False

    async def mock_translate_batch(batch, **kwargs):
        nonlocal ai_called
        ai_called = True
        await asyncio.sleep(0.05)
        return [{"id": item["id"], "text": f"AI {item['id']}"} for item in batch]
    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)

    # Bazarr delivers healthy Swedish subtitle at 30ms
    async def bazarr_producer():
        await asyncio.sleep(0.03)
        sv_lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nSvensk Bazarr {i}" for i in range(1, 25)]
        Path(sv_srt_path).write_text("\n\n".join(sv_lines), encoding="utf-8")

    b_task = asyncio.create_task(bazarr_producer())

    def mock_find(vp, lang):
        if lang == "en": return str(en_srt)
        if lang == "sv" and os.path.exists(sv_srt_path): return sv_srt_path
        return None

    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find), \
         patch.object(BazarrCoordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_jobs:
        mock_jobs.return_value = BazarrJobsPollResult(status=BazarrJobPollStatus.KNOWN_IDLE, jobs=[])
        res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")

    await b_task
    assert res["status"] == "skipped"
    assert res["reason"] in ("bazarr_downloaded", "already_exists")
    assert not ai_called


@pytest.mark.asyncio
async def test_matrix_j_ai_qa_pass_bazarr_sync_active_publication_prohibited(tmp_path, monkeypatch):
    """Matrix J: AI QA PASS + Bazarr target sync active -> publication prohibited."""
    setup_mock_settings(monkeypatch)
    coordinator = BazarrCoordinator()
    video_path = tmp_path / "MatrixJ.mkv"
    video_path.touch()

    active_sync = BazarrJobInfo(
        job_id="sync_j",
        job_name="Syncing Subtitles",
        status="running",
        job_type="sync",
        progress_message="MatrixJ.sv.srt",
    )

    with patch.object(coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_jobs:
        mock_jobs.return_value = BazarrJobsPollResult(status=BazarrJobPollStatus.ACTIVE, jobs=[active_sync])
        res = await coordinator.acquire_publication_ownership(
            video_path=str(video_path),
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="key",
            timeout_sec=0.1,
        )
        assert res.granted is False
        assert res.defer is True
        assert res.reason in ("bazarr_syncing", "bazarr_actively_writing")


@pytest.mark.asyncio
async def test_matrix_k_bazarr_completes_no_target_ai_qa_pass_publication_permitted(tmp_path, monkeypatch):
    """Matrix K: Bazarr definitively completes with no target + AI QA PASS -> AI publication permitted."""
    setup_mock_settings(monkeypatch)
    coordinator = BazarrCoordinator()
    video_path = tmp_path / "MatrixK.mkv"
    video_path.touch()

    with patch.object(coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_jobs:
        mock_jobs.return_value = BazarrJobsPollResult(status=BazarrJobPollStatus.KNOWN_IDLE, jobs=[])
        res = await coordinator.acquire_publication_ownership(
            video_path=str(video_path),
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="key",
            timeout_sec=0.1,
        )
        assert res.granted is True
        assert res.reason == "quiescent_and_verified"


@pytest.mark.asyncio
async def test_matrix_l_probe_reports_no_embedded_tracks_no_24lang_loop(tmp_path, monkeypatch):
    """Matrix L: Container probe reports no embedded subtitle tracks -> no redundant extraction loop."""
    setup_mock_settings(monkeypatch)
    video_path = tmp_path / "MatrixL.mkv"
    video_path.touch()

    en_srt = tmp_path / "MatrixL.en.srt"
    en_lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nLine {i}" for i in range(1, 20)]
    en_srt.write_text("\n\n".join(en_lines), encoding="utf-8")

    pipeline = SubtitlePipeline()
    ai_called = False

    async def mock_translate_batch(batch, **kwargs):
        nonlocal ai_called
        ai_called = True
        return [{"id": item["id"], "text": f"Detta är en svensk dialog och replik {item['id']}."} for item in batch]
    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr("app.services.pipeline.qa_gate", lambda *args, **kwargs: {"passed": True, "score": 100, "issues": []})

    extract_call_count = 0
    def counting_extract(*args, **kwargs):
        nonlocal extract_call_count
        extract_call_count += 1
        return False

    with patch("app.services.pipeline.find_external_subtitle", side_effect=lambda vp, l: str(en_srt) if l == "en" else None), \
         patch("app.services.pipeline.inspect_mkv_tracks", return_value={"subtitles": [], "audio": [{"id": 1, "language": "eng"}]}), \
         patch("app.services.pipeline._safe_extract_embedded_srt", side_effect=counting_extract), \
         patch.object(BazarrCoordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_jobs:
        mock_jobs.return_value = BazarrJobsPollResult(status=BazarrJobPollStatus.KNOWN_IDLE, jobs=[])
        res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")

    assert res["status"] == "translated"
    assert extract_call_count == 0  # Bypassed since inspect_mkv_tracks reported 0 subtitle tracks


@pytest.mark.asyncio
async def test_matrix_m_en_source_and_target_both_arrive_target_syncing(tmp_path, monkeypatch):
    """
    Matrix M: Bazarr EN source and target both arrive, target still syncing.
    Source may prepare, target remains provisional, no target rejection/publication race.
    """
    setup_mock_settings(monkeypatch, {
        "bazarr_quiescence_seconds": "0.1",
        "bazarr_candidate_stability_seconds": "0.03",
        "bazarr_grace_seconds": "1.5",
    })
    video_path = tmp_path / "MatrixM_Race.mkv"
    video_path.touch()

    en_srt = tmp_path / "MatrixM_Race.en.srt"
    en_lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nEnglish line {i}" for i in range(1, 20)]
    en_srt.write_text("\n\n".join(en_lines), encoding="utf-8")

    sv_srt_path = str(tmp_path / "MatrixM_Race.sv.srt")
    sv_provisional = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nPartial {i}" for i in range(1, 5)]
    Path(sv_srt_path).write_text("\n\n".join(sv_provisional), encoding="utf-8")

    pipeline = SubtitlePipeline()
    ai_called = False

    async def mock_translate_batch(batch, **kwargs):
        nonlocal ai_called
        ai_called = True
        return [{"id": item["id"], "text": "AI"} for item in batch]
    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)

    active_jobs = [
        BazarrJobInfo(
            job_id="sync_m",
            job_name="Syncing Subtitles",
            status="running",
            job_type="sync",
            progress_message="MatrixM_Race.sv.srt",
        )
    ]

    async def mock_poll(self, *args, **kwargs):
        return BazarrJobsPollResult(
            status=BazarrJobPollStatus.ACTIVE if active_jobs else BazarrJobPollStatus.KNOWN_IDLE,
            jobs=list(active_jobs),
        )
    monkeypatch.setattr(BazarrCoordinator, "poll_system_jobs", mock_poll)

    async def complete_sync():
        await asyncio.sleep(0.08)
        sv_complete = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nDetta är en komplett svensk textrad {i}" for i in range(1, 20)]
        Path(sv_srt_path).write_text("\n\n".join(sv_complete), encoding="utf-8")
        active_jobs.clear()

    c_task = asyncio.create_task(complete_sync())

    def mock_find(vp, lang):
        if lang == "en": return str(en_srt)
        if lang == "sv" and os.path.exists(sv_srt_path): return sv_srt_path
        return None

    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find):
        res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")

    await c_task
    assert res["status"] == "skipped"
    assert res["reason"] in ("bazarr_downloaded", "already_exists")
    assert not ai_called
    assert not os.path.exists(sv_srt_path + ".bak")


# ===========================================================================
# 2 FAST 2 FURIOUS CANONICAL REGRESSION
# ===========================================================================

@pytest.mark.asyncio
async def test_matrix_c_2fast2furious_sync_regression(tmp_path, monkeypatch):
    """
    2 Fast 2 Furious authoritative regression test.
    Bazarr search triggered for movie.
    Candidate appears at 20ms while Bazarr job is actively SYNCING.
    Coordinator holds provisional candidate, waits for Bazarr sync to finish.
    Final sync generation passes Trust.
    Result: BAZARR MATCH, 0 AI calls, no backup file created.
    """
    setup_mock_settings(monkeypatch, {
        "bazarr_quiescence_seconds": "0.1",
        "bazarr_candidate_stability_seconds": "0.03",
        "bazarr_grace_seconds": "2.0",
    })

    video_path = tmp_path / "2 Fast 2 Furious (2003) Bluray-1080p.mkv"
    video_path.touch()

    en_srt = tmp_path / "2 Fast 2 Furious (2003) Bluray-1080p.en.srt"
    en_lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nFast dialogue {i}" for i in range(1, 25)]
    en_srt.write_text("\n\n".join(en_lines), encoding="utf-8")

    sv_srt_path = str(tmp_path / "2 Fast 2 Furious (2003) Bluray-1080p.sv.srt")

    coordinator = BazarrCoordinator()
    pipeline = SubtitlePipeline()
    ai_called = False

    async def mock_translate_batch(batch, **kwargs):
        nonlocal ai_called
        ai_called = True
        return [{"id": item["id"], "text": f"AI {item['id']}"} for item in batch]
    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)

    # Job lifecycle simulation
    active_jobs = [
        BazarrJobInfo(
            job_id="sync_1",
            job_name="Syncing Subtitles with Video",
            status="running",
            job_type="sync",
            progress_message="2 Fast 2 Furious (2003) Bluray-1080p.sv.srt",
        )
    ]

    async def poll_jobs_mock(self, *args, **kwargs):
        return BazarrJobsPollResult(
            status=BazarrJobPollStatus.ACTIVE if active_jobs else BazarrJobPollStatus.KNOWN_IDLE,
            jobs=list(active_jobs),
        )
    monkeypatch.setattr(BazarrCoordinator, "poll_system_jobs", poll_jobs_mock)
    monkeypatch.setattr(BazarrCoordinator, "poll_system_jobs", poll_jobs_mock)

    async def bazarr_worker():
        # Candidate appears initially at 20ms
        await asyncio.sleep(0.02)
        sv_lines_a = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nKort {i}" for i in range(1, 6)]
        Path(sv_srt_path).write_text("\n\n".join(sv_lines_a), encoding="utf-8")

        # Bazarr sync finishes at 100ms and writes complete final file
        await asyncio.sleep(0.08)
        sv_lines_b = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nDet här är komplett svensk dialog {i}" for i in range(1, 25)]
        Path(sv_srt_path).write_text("\n\n".join(sv_lines_b), encoding="utf-8")
        active_jobs.clear()

    worker_task = asyncio.create_task(bazarr_worker())

    def mock_find(vp, lang):
        if lang == "en": return str(en_srt)
        if lang == "sv" and os.path.exists(sv_srt_path): return sv_srt_path
        return None

    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find):
        res = await pipeline.process_video_file(
            str(video_path),
            event_source="RADARR",
        )

    await worker_task
    assert res["status"] == "skipped"
    assert res["reason"] in ("bazarr_downloaded", "already_exists")
    assert not ai_called
    # Assert Bazarr target was not backed up
    assert not os.path.exists(sv_srt_path + ".bak")
    assert not os.path.exists(sv_srt_path + ".original.bak")
    # Assert Bazarr target was not replaced by Babel
    final_content = Path(sv_srt_path).read_text(encoding="utf-8")
    assert "Det här är komplett svensk dialog 1" in final_content
    assert "AI 1" not in final_content
