"""
Regression tests for Bazarr late-lifecycle provenance and global offset repair.

Matrix covered:
A. Late finalized current-run Bazarr candidate receives truthful strong provenance
   and safe repair/adoption (AI calls = 0, Babel AI output not published).
B. search_accepted=False => not strong provenance (no repair on LOW_COVERAGE).
C. media_correlated=False => not strong provenance.
D. pre-existing same generation => not strong provenance (is_strong_current_run is False).
E. ACTIVE / UNKNOWN lifecycle => no repair and publication remains deferred.
F. Repair/revalidation FAIL => candidate not adopted; normal Babel fallback remains available.
G. Final conflict uses BAZARR origin/provenance only for provably current-run candidate.
H. Unrelated external candidate remains EXTERNAL / fail-closed.
"""

import asyncio
import datetime
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import srt

import app.core.db as db_mod
from app.core.trust_engine import (
    BazarrProvenance,
    CandidateOrigin,
    SubtitleTrustEngine,
    TargetSnapshot,
    TrustDecision,
    align_subtitle_timelines,
    capture_target_snapshot,
    SyncErrorType,
)
from app.services.bazarr_coordinator import (
    BazarrCoordinator,
    BazarrJobInfo,
    BazarrJobPollStatus,
    BazarrJobsPollResult,
    BazarrLifecycleState,
    PublicationOwnershipResult,
    bazarr_coordinator,
)
from app.services.pipeline import SubtitlePipeline, _publish_subtitle_with_trust_gate
from app.services.source_resolver import SubtitleSource, SourceOrigin
from tests.test_bazarr_global_offset_repair import _build_black_sails_fixture, _make_cues, _make_srt


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db_file = tmp_path / "test_late_provenance.db"
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_file))
    monkeypatch.setattr("app.core.quota.DB_PATH", str(db_file), raising=False)
    db_mod.init_db()
    bazarr_coordinator.reset()
    yield
    bazarr_coordinator.reset()


# ==============================================================================
# A. Late finalized current-run Bazarr candidate receives truthful strong provenance
#    and safe repair/adoption (AI calls = 0, Babel AI output not published).
# ==============================================================================
@pytest.mark.asyncio
async def test_case_a_late_finalized_bazarr_candidate_repaired_and_adopted(tmp_path):
    """
    Case A: Late finalized current-run Bazarr candidate appears on disk.
    - search_accepted=True, media_correlated=True, pre_trigger_snapshot is absent (new file).
    - Bazarr is KNOWN_IDLE and candidate is quiescent.
    - Truthful BazarrProvenance is constructed.
    - Existing safe repair shifts timestamps and revalidation passes Trust.
    - Candidate is adopted (adopted=True, granted=False).
    - File on disk is updated with shifted timestamps and identical text.
    """
    ref_specs, target_specs, ref_cues, tgt_cues = _build_black_sails_fixture()

    video_path = str(tmp_path / "Black.Sails.S03E04.1080p.mkv")
    Path(video_path).touch()

    # Write Swedish candidate with +20.45s offset (LOW_COVERAGE)
    sv_file = tmp_path / "Black.Sails.S03E04.sv.srt"
    sv_file.write_text(srt.compose(tgt_cues), encoding="utf-8")

    en_file = tmp_path / "Black.Sails.S03E04.en.srt"
    en_file.write_text(srt.compose(ref_cues), encoding="utf-8")
    ref_source = SubtitleSource(
        origin=SourceOrigin.EMBEDDED,
        path=str(en_file),
        language="en",
        content=srt.compose(ref_cues),
        cues=ref_cues,
    )

    pre_snap = TargetSnapshot(path="", exists=False, size=0, mtime_ns=0)
    coordinator = BazarrCoordinator()

    def find_sv(vp, lang):
        if lang == "sv" and os.path.exists(str(sv_file)):
            return str(sv_file)
        return None

    with patch.object(coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_poll:
        mock_poll.return_value = BazarrJobsPollResult(
            status=BazarrJobPollStatus.KNOWN_IDLE, jobs=[]
        )

        result = await coordinator.acquire_publication_ownership(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="test",
            provided_source=ref_source,
            job_id=None,
            timeout_sec=5.0,
            find_external_subtitle_fn=find_sv,
            pre_trigger_snapshot=pre_snap,
            search_accepted=True,
            media_correlated=True,
        )

    assert result.adopted is True, f"Expected adopted=True, got: {result}"
    assert result.granted is False
    assert result.trust_result is not None
    assert result.trust_result.passed is True
    assert result.trust_result.repair is not None
    assert abs(result.trust_result.repair["applied_shift_sec"] - (-20.45)) < 0.1

    # Verify candidate file on disk was shifted
    repaired_cues = list(srt.parse(sv_file.read_text(encoding="utf-8")))
    assert len(repaired_cues) == len(target_specs)
    expected_first_start = target_specs[0][0] - 20.45
    assert abs(repaired_cues[0].start.total_seconds() - expected_first_start) < 0.1
    for i, c in enumerate(repaired_cues):
        assert c.content == target_specs[i][2]


# ==============================================================================
# B. search_accepted=False => not strong provenance (no repair on LOW_COVERAGE)
# ==============================================================================
@pytest.mark.asyncio
async def test_case_b_no_search_accepted_no_provenance(tmp_path):
    """
    Case B: search_accepted is False.
    - No BazarrProvenance is constructed.
    - LOW_COVERAGE candidate is NOT repaired.
    - Candidate fails Trust and is NOT adopted.
    - Ownership is granted to Babel (granted=True, adopted=False).
    """
    _, target_specs, ref_cues, tgt_cues = _build_black_sails_fixture()

    video_path = str(tmp_path / "Show.S01E01.mkv")
    Path(video_path).touch()

    sv_file = tmp_path / "Show.S01E01.sv.srt"
    sv_file.write_text(srt.compose(tgt_cues), encoding="utf-8")

    ref_source = SubtitleSource(
        origin=SourceOrigin.EMBEDDED,
        path=str(tmp_path / "Show.S01E01.en.srt"),
        language="en",
        content=srt.compose(ref_cues),
        cues=ref_cues,
    )

    pre_snap = TargetSnapshot(path="", exists=False, size=0, mtime_ns=0)
    coordinator = BazarrCoordinator()

    def find_sv(vp, lang):
        return str(sv_file) if lang == "sv" else None

    with patch.object(coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_poll:
        mock_poll.return_value = BazarrJobsPollResult(
            status=BazarrJobPollStatus.KNOWN_IDLE, jobs=[]
        )

        result = await coordinator.acquire_publication_ownership(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="test",
            provided_source=ref_source,
            job_id=None,
            timeout_sec=3.0,
            find_external_subtitle_fn=find_sv,
            pre_trigger_snapshot=pre_snap,
            search_accepted=False,  # <— NOT accepted
            media_correlated=True,
        )

    assert result.adopted is False
    assert result.granted is True


# ==============================================================================
# C. media_correlated=False => not strong provenance
# ==============================================================================
@pytest.mark.asyncio
async def test_case_c_no_media_correlated_no_provenance(tmp_path):
    """
    Case C: media_correlated is False.
    - No BazarrProvenance is constructed.
    - Candidate fails Trust and is NOT adopted.
    - Ownership is granted to Babel (granted=True, adopted=False).
    """
    _, target_specs, ref_cues, tgt_cues = _build_black_sails_fixture()

    video_path = str(tmp_path / "Movie.mkv")
    Path(video_path).touch()

    sv_file = tmp_path / "Movie.sv.srt"
    sv_file.write_text(srt.compose(tgt_cues), encoding="utf-8")

    ref_source = SubtitleSource(
        origin=SourceOrigin.EMBEDDED,
        path=str(tmp_path / "Movie.en.srt"),
        language="en",
        content=srt.compose(ref_cues),
        cues=ref_cues,
    )

    pre_snap = TargetSnapshot(path="", exists=False, size=0, mtime_ns=0)
    coordinator = BazarrCoordinator()

    def find_sv(vp, lang):
        return str(sv_file) if lang == "sv" else None

    with patch.object(coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_poll:
        mock_poll.return_value = BazarrJobsPollResult(
            status=BazarrJobPollStatus.KNOWN_IDLE, jobs=[]
        )

        result = await coordinator.acquire_publication_ownership(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="test",
            provided_source=ref_source,
            job_id=None,
            timeout_sec=3.0,
            find_external_subtitle_fn=find_sv,
            pre_trigger_snapshot=pre_snap,
            search_accepted=True,
            media_correlated=False,  # <— NOT correlated
        )

    assert result.adopted is False
    assert result.granted is True


# ==============================================================================
# D. pre-existing same generation => not strong provenance
# ==============================================================================
@pytest.mark.asyncio
async def test_case_d_preexisting_same_generation_not_strong(tmp_path):
    """
    Case D: Candidate existed before the Bazarr trigger with identical generation.
    - is_strong_current_run() evaluates to False (generation_id unchanged).
    - Candidate fails Trust and is NOT adopted.
    - Ownership is granted to Babel.
    """
    _, target_specs, ref_cues, tgt_cues = _build_black_sails_fixture()

    video_path = str(tmp_path / "Preexisting.mkv")
    Path(video_path).touch()

    sv_file = tmp_path / "Preexisting.sv.srt"
    sv_file.write_text(srt.compose(tgt_cues), encoding="utf-8")

    pre_snap = capture_target_snapshot(str(sv_file))
    assert pre_snap.exists

    ref_source = SubtitleSource(
        origin=SourceOrigin.EMBEDDED,
        path=str(tmp_path / "Preexisting.en.srt"),
        language="en",
        content=srt.compose(ref_cues),
        cues=ref_cues,
    )

    coordinator = BazarrCoordinator()

    def find_sv(vp, lang):
        return str(sv_file) if lang == "sv" else None

    with patch.object(coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_poll:
        mock_poll.return_value = BazarrJobsPollResult(
            status=BazarrJobPollStatus.KNOWN_IDLE, jobs=[]
        )

        result = await coordinator.acquire_publication_ownership(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="test",
            provided_source=ref_source,
            job_id=None,
            timeout_sec=3.0,
            find_external_subtitle_fn=find_sv,
            pre_trigger_snapshot=pre_snap,  # <— same generation as candidate
            search_accepted=True,
            media_correlated=True,
        )

    assert result.adopted is False
    assert result.granted is True


# ==============================================================================
# E. ACTIVE / UNKNOWN lifecycle => no repair and publication remains deferred
# ==============================================================================
@pytest.mark.asyncio
async def test_case_e1_unknown_lifecycle_defers_publication(tmp_path):
    """
    Case E1: Bazarr lifecycle is UNKNOWN (e.g. API error).
    - Publication must be deferred (defer=True, granted=False, adopted=False).
    """
    video_path = str(tmp_path / "Unknown.mkv")
    Path(video_path).touch()

    coordinator = BazarrCoordinator()

    with patch.object(coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_poll:
        mock_poll.return_value = BazarrJobsPollResult(
            status=BazarrJobPollStatus.UNKNOWN, jobs=[], error="Connection error"
        )

        result = await coordinator.acquire_publication_ownership(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="test",
            job_id=None,
            timeout_sec=0.5,
            pre_trigger_snapshot=TargetSnapshot(path="", exists=False, size=0, mtime_ns=0),
            search_accepted=True,
            media_correlated=True,
        )

    assert result.granted is False
    assert result.defer is True
    assert result.adopted is False
    assert "unknown" in result.reason.lower()


@pytest.mark.asyncio
async def test_case_e2_active_jobs_denies_ownership(tmp_path):
    """
    Case E2: Correlated Bazarr search/sync job is actively running.
    - If active jobs do not finish within timeout, publication is deferred.
    """
    video_path = str(tmp_path / "Active.mkv")
    Path(video_path).touch()

    coordinator = BazarrCoordinator()
    active_job = BazarrJobInfo(
        job_id="job-1",
        job_name="Sync subtitle for Active",
        status="Running",
        job_type="sync",
    )

    with patch.object(coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_poll, \
         patch.object(coordinator, "classify_jobs_for_target", return_value=([], [active_job])):

        mock_poll.return_value = BazarrJobsPollResult(
            status=BazarrJobPollStatus.ACTIVE, jobs=[active_job]
        )

        result = await coordinator.acquire_publication_ownership(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="test",
            job_id=None,
            timeout_sec=0.5,
            pre_trigger_snapshot=TargetSnapshot(path="", exists=False, size=0, mtime_ns=0),
            search_accepted=True,
            media_correlated=True,
        )

    assert result.granted is False
    assert result.defer is True
    assert result.adopted is False
    assert "writing" in result.reason.lower() or "active" in result.reason.lower()


# ==============================================================================
# F. Repair/revalidation FAIL => candidate not adopted; normal Babel fallback remains available
# ==============================================================================
@pytest.mark.asyncio
async def test_case_f_unrepairable_candidate_not_adopted_fallback_available(tmp_path):
    """
    Case F: Candidate has progressive drift (FPS mismatch) — unrepairable.
    - search_accepted=True, media_correlated=True, pre_trigger_snapshot is absent.
    - Trust Engine correctly rejects progressive drift (FAIL).
    - Candidate is NOT adopted (adopted=False).
    - Ownership is granted to Babel (granted=True) so normal AI translation can publish.
    """
    video_path = str(tmp_path / "Drift.mkv")
    Path(video_path).touch()

    ref_specs = [
        (10.0 + i * 10.0, 13.0 + i * 10.0, f"English dialogue line {i}")
        for i in range(60)
    ]
    ref_source = SubtitleSource(
        origin=SourceOrigin.EMBEDDED,
        path=str(tmp_path / "Drift.en.srt"),
        language="en",
        content=_make_srt(ref_specs),
        cues=_make_cues(ref_specs),
    )

    # Candidate with 3.5s progressive drift across 10 minutes
    cand_specs = [
        (10.0 + i * 10.0 + (i / 60.0) * 3.5, 13.0 + i * 10.0 + (i / 60.0) * 3.5, f"Svensk replik {i}")
        for i in range(60)
    ]
    cand_file = tmp_path / "Drift.sv.srt"
    cand_file.write_text(_make_srt(cand_specs), encoding="utf-8")

    pre_snap = TargetSnapshot(path="", exists=False, size=0, mtime_ns=0)
    coordinator = BazarrCoordinator()

    def find_sv(vp, lang):
        return str(cand_file) if lang == "sv" else None

    with patch.object(coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_poll:
        mock_poll.return_value = BazarrJobsPollResult(
            status=BazarrJobPollStatus.KNOWN_IDLE, jobs=[]
        )

        result = await coordinator.acquire_publication_ownership(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="test",
            provided_source=ref_source,
            job_id=None,
            timeout_sec=3.0,
            find_external_subtitle_fn=find_sv,
            pre_trigger_snapshot=pre_snap,
            search_accepted=True,
            media_correlated=True,
        )

    # Progressive drift cannot be repaired -> not adopted -> ownership granted for AI fallback
    assert result.adopted is False
    assert result.granted is True


# ==============================================================================
# G. Final conflict uses BAZARR origin/provenance only for provably current-run candidate
# ==============================================================================
@pytest.mark.asyncio
async def test_case_g_pipeline_trust_gate_uses_bazarr_origin_for_current_run(tmp_path):
    """
    Case G: _publish_subtitle_with_trust_gate receives current-run Bazarr facts.
    - Candidate generation is proven quiescent by acquire_publication_ownership().
    - Current candidate matches proven snapshot -> evaluates as CandidateOrigin.BAZARR with truthful BazarrProvenance.
    """
    _, target_specs, ref_cues, tgt_cues = _build_black_sails_fixture()

    video_path = str(tmp_path / "G_Test.mkv")
    Path(video_path).touch()

    sv_file = tmp_path / "G_Test.sv.srt"
    sv_file.write_text(srt.compose(tgt_cues), encoding="utf-8")

    pre_snap = TargetSnapshot(path="", exists=False, size=0, mtime_ns=0)
    snap_a = capture_target_snapshot(str(sv_file))

    prov_a = BazarrProvenance(
        video_path=video_path,
        target_lang="sv",
        search_accepted=True,
        pre_trigger_snapshot=pre_snap,
        is_finalized=True,
        is_quiescent=True,
        media_correlated=True,
        poll_state=BazarrJobPollStatus.KNOWN_IDLE,
        candidate_snapshot=snap_a,
    )

    ownership_mock_res = PublicationOwnershipResult(
        granted=True,
        reason="quiescent_and_verified",
        proven_bazarr_provenance=prov_a,
        proven_candidate_snapshot=snap_a,
    )

    captured_calls = []

    async def _mock_evaluate(self_inner, *, candidate_path, origin, bazarr_provenance=None, **kwargs):
        captured_calls.append({
            "candidate_path": candidate_path,
            "origin": origin,
            "has_provenance": bazarr_provenance is not None,
            "poll_state": bazarr_provenance.poll_state if bazarr_provenance else None,
        })
        return TrustResult(
            decision=TrustDecision.PASS,
            score=92,
            confidence="HIGH",
            reasons=["Mocked PASS"],
            origin=origin,
            verification_mode=VerificationMode.REFERENCE if bazarr_provenance else VerificationMode.STANDALONE,
        )

    from app.core.trust_engine import VerificationMode

    settings = {"enable_bazarr_check": "true", "bazarr_api_key": "test"}

    with patch("app.services.pipeline.get_setting", side_effect=lambda k, d=None: settings.get(k, d)), \
         patch("app.services.pipeline.find_external_subtitle", side_effect=lambda vp, lang: str(sv_file) if lang == "sv" else None), \
         patch.object(bazarr_coordinator, "acquire_publication_ownership", new_callable=AsyncMock, return_value=ownership_mock_res), \
         patch.object(SubtitleTrustEngine, "_evaluate_candidate_internal", _mock_evaluate):

        result = await _publish_subtitle_with_trust_gate(
            video_path=video_path,
            target_output_path=str(sv_file),
            lang_code="sv",
            translated_srt_text=_make_srt([(0, 1, "AI translation")]),
            expected_cue_count=1,
            job_id=None,
            bazarr_pre_trigger_snapshot=pre_snap,
            bazarr_search_accepted=True,
            bazarr_media_correlated=True,
        )

    assert len(captured_calls) >= 1
    call = captured_calls[0]
    assert call["origin"] == CandidateOrigin.BAZARR
    assert call["has_provenance"] is True
    assert str(call["poll_state"]) == "KNOWN_IDLE" or getattr(call["poll_state"], "value", "") == "KNOWN_IDLE"


@pytest.mark.asyncio
async def test_case_g2_mutation_after_ownership_drops_provenance(tmp_path):
    """
    Case G2: Invariant check for post-ownership candidate mutation.
    - acquire_publication_ownership() proves generation A quiescent and returns proven snapshot A.
    - Candidate mutates to generation B BEFORE final-conflict evaluation.
    - Final conflict discovers generation B != snapshot A.
    - Gate MUST NOT synthesize is_quiescent=True or grant strong Bazarr provenance to generation B.
    - Generation B must be evaluated as CandidateOrigin.EXTERNAL with bazarr_provenance=None.
    """
    video_path = str(tmp_path / "G2_Test.mkv")
    Path(video_path).touch()

    sv_file = tmp_path / "G2_Test.sv.srt"
    sv_file.write_text(_make_srt([(0, 2, "Generation A content")]), encoding="utf-8")

    pre_snap = TargetSnapshot(path="", exists=False, size=0, mtime_ns=0)
    snap_a = capture_target_snapshot(str(sv_file))

    prov_a = BazarrProvenance(
        video_path=video_path,
        target_lang="sv",
        search_accepted=True,
        pre_trigger_snapshot=pre_snap,
        is_finalized=True,
        is_quiescent=True,
        media_correlated=True,
        poll_state=BazarrJobPollStatus.KNOWN_IDLE,
        candidate_snapshot=snap_a,
    )

    # Ownership proved generation A
    ownership_mock_res = PublicationOwnershipResult(
        granted=True,
        reason="quiescent_and_verified",
        proven_bazarr_provenance=prov_a,
        proven_candidate_snapshot=snap_a,
    )

    captured_calls = []

    async def _mock_ownership(*args, **kwargs):
        # When ownership check runs, mutate candidate to generation B immediately after
        sv_file.write_text(_make_srt([(0, 2, "Mutated Generation B content - unproven")]), encoding="utf-8")
        # Ensure mtime or size differs to guarantee new generation ID
        os.utime(str(sv_file), (time_now := snap_a.mtime_ns / 1e9 + 10.0, time_now))
        return ownership_mock_res

    async def _mock_evaluate(self_inner, *, candidate_path, origin, bazarr_provenance=None, **kwargs):
        captured_calls.append({
            "candidate_path": candidate_path,
            "origin": origin,
            "has_provenance": bazarr_provenance is not None,
            "provenance": bazarr_provenance,
        })
        return TrustResult(
            decision=TrustDecision.PASS,
            score=92,
            confidence="HIGH",
            reasons=["Mocked PASS"],
            origin=origin,
            verification_mode=VerificationMode.STANDALONE,
        )

    from app.core.trust_engine import VerificationMode

    settings = {"enable_bazarr_check": "true", "bazarr_api_key": "test"}

    with patch("app.services.pipeline.get_setting", side_effect=lambda k, d=None: settings.get(k, d)), \
         patch("app.services.pipeline.find_external_subtitle", side_effect=lambda vp, lang: str(sv_file) if lang == "sv" else None), \
         patch.object(bazarr_coordinator, "acquire_publication_ownership", side_effect=_mock_ownership), \
         patch.object(SubtitleTrustEngine, "_evaluate_candidate_internal", _mock_evaluate):

        result = await _publish_subtitle_with_trust_gate(
            video_path=video_path,
            target_output_path=str(sv_file),
            lang_code="sv",
            translated_srt_text=_make_srt([(0, 1, "AI translation")]),
            expected_cue_count=1,
            job_id=None,
            bazarr_pre_trigger_snapshot=pre_snap,
            bazarr_search_accepted=True,
            bazarr_media_correlated=True,
        )

    assert len(captured_calls) >= 1, "Trust Engine must have been called during final conflict"
    call = captured_calls[0]
    # Invariant verified: generation B MUST NOT inherit strong Bazarr provenance or quiescence
    assert call["origin"] == CandidateOrigin.EXTERNAL, (
        f"Expected origin=EXTERNAL for mutated generation B, but got {call['origin']}"
    )
    assert call["has_provenance"] is False, (
        "Mutated generation B must NOT receive bazarr_provenance (provenance bound strictly to snapshot A)"
    )


# ==============================================================================
# H. Unrelated external candidate remains EXTERNAL / fail-closed
# ==============================================================================
@pytest.mark.asyncio
async def test_case_h_pipeline_trust_gate_uses_external_without_provenance(tmp_path):
    """
    Case H: _publish_subtitle_with_trust_gate is called WITHOUT Bazarr provenance facts.
    - Candidate must be classified as CandidateOrigin.EXTERNAL with bazarr_provenance=None.
    """
    video_path = str(tmp_path / "H_Test.mkv")
    Path(video_path).touch()

    sv_file = tmp_path / "H_Test.sv.srt"
    sv_file.write_text(_make_srt([(0, 1, "Existing external subtitle")]), encoding="utf-8")

    captured_calls = []

    async def _mock_evaluate(self_inner, *, candidate_path, origin, bazarr_provenance=None, **kwargs):
        captured_calls.append({
            "origin": origin,
            "has_provenance": bazarr_provenance is not None,
        })
        return TrustResult(
            decision=TrustDecision.PASS,
            score=92,
            confidence="HIGH",
            reasons=["Mocked PASS"],
            origin=origin,
            verification_mode=VerificationMode.STANDALONE,
        )

    from app.core.trust_engine import VerificationMode

    settings = {"enable_bazarr_check": "false", "bazarr_api_key": ""}

    with patch("app.services.pipeline.get_setting", side_effect=lambda k, d=None: settings.get(k, d)), \
         patch("app.services.pipeline.find_external_subtitle", side_effect=lambda vp, lang: str(sv_file) if lang == "sv" else None), \
         patch.object(SubtitleTrustEngine, "_evaluate_candidate_internal", _mock_evaluate):

        await _publish_subtitle_with_trust_gate(
            video_path=video_path,
            target_output_path=str(sv_file),
            lang_code="sv",
            translated_srt_text=_make_srt([(0, 1, "New AI translation")]),
            expected_cue_count=1,
            job_id=None,
        )

    assert len(captured_calls) >= 1
    call = captured_calls[0]
    assert call["origin"] == CandidateOrigin.EXTERNAL
    assert call["has_provenance"] is False


# ==============================================================================
# I. CACHE TRANSITION TEST:
#    Stale non-passing cache entry from weak/provisional evaluation does NOT
#    prevent safe global-offset repair when candidate gains strong provenance.
# ==============================================================================
@pytest.mark.asyncio
async def test_cache_transition_stale_fail_bypassed_for_strong_provenance_repair(tmp_path):
    """
    1. Candidate generation A on disk with constant +20.45s offset.
    2. First evaluation: Bazarr origin with weak/provisional provenance
       (search_accepted=False or poll_state=ACTIVE).
       -> Real Trust Engine evaluates candidate, alignment yields LOW_COVERAGE.
       -> Since provenance is not strong, safe repair is NOT attempted.
       -> Trust returns FAIL and saves FAIL in real cache.
    3. Second evaluation: EXACT SAME generation A on disk evaluated with
       strong current-run BazarrProvenance + auto_repair=True.
       -> Cached FAIL is bypassed because candidate now has strong Bazarr provenance
          and auto_repair/allow_global_offset_repair is True.
       -> Safe global-offset repair executes in scratch.
       -> Scratch revalidation returns PASS.
       -> Repaired candidate is adopted and written to disk.
    """
    ref_specs, target_specs, ref_cues, tgt_cues = _build_black_sails_fixture()

    video_path = str(tmp_path / "Black.Sails.S04E07.1080p.mkv")
    Path(video_path).touch()

    # Write Swedish candidate with +20.45s offset (LOW_COVERAGE)
    sv_file = tmp_path / "Black.Sails.S04E07.sv.srt"
    sv_file.write_text(srt.compose(tgt_cues), encoding="utf-8")

    en_file = tmp_path / "Black.Sails.S04E07.en.srt"
    en_file.write_text(srt.compose(ref_cues), encoding="utf-8")
    ref_source = SubtitleSource(
        origin=SourceOrigin.EMBEDDED,
        path=str(en_file),
        language="en",
        content=srt.compose(ref_cues),
        cues=ref_cues,
    )

    engine = SubtitleTrustEngine()
    snap_initial = capture_target_snapshot(str(sv_file))

    # --- Phase 1: Weak/Provisional Evaluation ---
    weak_prov = BazarrProvenance(
        video_path=video_path,
        target_lang="sv",
        search_accepted=False,  # Not strong
        pre_trigger_snapshot=None,
        is_finalized=False,
        is_quiescent=False,
        media_correlated=False,
        poll_state=BazarrJobPollStatus.ACTIVE,
        candidate_snapshot=snap_initial,
    )

    res1 = await engine.evaluate_candidate(
        video_path=video_path,
        candidate_path=str(sv_file),
        target_lang="sv",
        origin=CandidateOrigin.BAZARR,
        provided_source=ref_source,
        job_id=None,
        auto_repair=True,
        allow_ai_audit=False,
        bazarr_provenance=weak_prov,
    )

    # Verify initial evaluation failed and was cached
    assert res1.decision == TrustDecision.FAIL
    assert res1.passed is False
    assert res1.repair is None
    # File on disk must remain unchanged at original +20.45s offset
    assert capture_target_snapshot(str(sv_file)).generation_id == snap_initial.generation_id

    # --- Phase 2: Strong Provenance Evaluation on EXACT SAME Generation ---
    strong_prov = BazarrProvenance(
        video_path=video_path,
        target_lang="sv",
        search_accepted=True,
        pre_trigger_snapshot=TargetSnapshot(path="", exists=False, size=0, mtime_ns=0),
        is_finalized=True,
        is_quiescent=True,
        media_correlated=True,
        poll_state=BazarrJobPollStatus.KNOWN_IDLE,
        candidate_snapshot=snap_initial,
    )

    res2 = await engine.evaluate_candidate(
        video_path=video_path,
        candidate_path=str(sv_file),
        target_lang="sv",
        origin=CandidateOrigin.BAZARR,
        provided_source=ref_source,
        job_id=None,
        auto_repair=True,
        allow_global_offset_repair=True,
        allow_ai_audit=False,
        bazarr_provenance=strong_prov,
    )

    # Invariant verified: Cached FAIL was bypassed, safe repair executed and passed!
    assert res2.passed is True, f"Expected PASS after strong provenance repair, got: {res2}"
    assert res2.decision in (TrustDecision.PASS, TrustDecision.PASS_WITH_WARNINGS)
    assert res2.repair is not None
    assert abs(res2.repair["applied_shift_sec"] - (-20.45)) < 0.1

    # Verify candidate file on disk was shifted and text preserved
    repaired_cues = list(srt.parse(sv_file.read_text(encoding="utf-8")))
    assert len(repaired_cues) == len(target_specs)
    expected_first_start = target_specs[0][0] - 20.45
    assert abs(repaired_cues[0].start.total_seconds() - expected_first_start) < 0.1
    for i, c in enumerate(repaired_cues):
        assert c.content == target_specs[i][2]


# ==============================================================================
# J. GLOBAL UNRELATED/UNCLASSIFIED PUBLICATION TEST (Scenario B):
#    search_accepted=True, poll status ACTIVE, no classified target jobs,
#    active job = generic "Downloading Subtitles" (ambiguous/unclassified elsewhere).
#    Expected:
#      - publication ownership is NOT blocked merely by global ACTIVE
#      - target has no correlated active work / write risk
#      - publication is granted (granted=True, defer=False).
# ==============================================================================
@pytest.mark.asyncio
async def test_active_unclassified_jobs_allows_publication_ownership(tmp_path):
    """
    search_accepted=True, poll status ACTIVE, no classified target jobs,
    active job = generic "Downloading Subtitles" (ambiguous/unclassified globally).
    Expected:
      - publication ownership is NOT blocked merely by global ACTIVE
      - publication is granted without delay (granted=True, defer=False).
    """
    video_path = str(tmp_path / "Black.Sails.S04E07.mkv")
    Path(video_path).touch()

    coordinator = BazarrCoordinator()
    # Ambiguous generic subtitle job elsewhere in Bazarr
    ambiguous_job = BazarrJobInfo(
        job_id="job-unclass-9",
        job_name="Downloading Subtitles",
        status="Running",
        job_type="download_subtitles",
        progress_message="Working on queue...",
    )

    with patch.object(coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_poll:
        mock_poll.return_value = BazarrJobsPollResult(
            status=BazarrJobPollStatus.ACTIVE, jobs=[ambiguous_job]
        )

        result = await coordinator.acquire_publication_ownership(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="test",
            job_id=None,
            timeout_sec=0.5,
            pre_trigger_snapshot=TargetSnapshot(path="", exists=False, size=0, mtime_ns=0),
            search_accepted=True,
            media_correlated=True,
        )

    assert result.granted is True
    assert result.defer is False
    assert result.reason == "quiescent_and_verified"


# ==============================================================================
# K. ACTIVE BUT CONCLUSIVELY UNRELATED JOBS TEST:
#    search_accepted=True, poll status ACTIVE,
#    all active jobs are conclusively unrelated (e.g. backup, health, or other show)
#    -> target progresses according to authoritative-idle policy
#    -> strong current-run provenance established
#    -> offset-repaired candidate adopted (adopted=True, granted=False).
# ==============================================================================
@pytest.mark.asyncio
async def test_active_conclusively_unrelated_jobs_allows_target_idle_and_adoption(tmp_path):
    """
    search_accepted=True, poll status ACTIVE,
    active jobs = ["Database Backup", "Sync subtitle for Breaking Bad S01E01"].
    All active jobs are conclusively unrelated.
    Expected:
      - unrelated work does not block this target
      - target progresses according to authoritative-idle policy
      - strong current-run provenance established
      - offset-repaired candidate adopted (adopted=True, granted=False).
    """
    ref_specs, target_specs, ref_cues, tgt_cues = _build_black_sails_fixture()

    video_path = str(tmp_path / "Black.Sails.S04E07.1080p.mkv")
    Path(video_path).touch()

    # Swedish candidate on disk with +20.45s offset
    sv_file = tmp_path / "Black.Sails.S04E07.sv.srt"
    sv_file.write_text(srt.compose(tgt_cues), encoding="utf-8")

    en_file = tmp_path / "Black.Sails.S04E07.en.srt"
    en_file.write_text(srt.compose(ref_cues), encoding="utf-8")
    ref_source = SubtitleSource(
        origin=SourceOrigin.EMBEDDED,
        path=str(en_file),
        language="en",
        content=srt.compose(ref_cues),
        cues=ref_cues,
    )

    pre_snap = TargetSnapshot(path="", exists=False, size=0, mtime_ns=0)
    coordinator = BazarrCoordinator()

    unrelated_jobs = [
        BazarrJobInfo(job_id="bak-1", job_name="Backup Database", status="Running", job_type="backup"),
        BazarrJobInfo(job_id="diff-1", job_name="Sync subtitle for Breaking.Bad.S01E01.mkv", status="Running", job_type="sync"),
        BazarrJobInfo(job_id="diff-2", job_name="Downloading French Subtitles", status="Running", job_type="download", matched_language="fr"),
    ]

    def find_sv(vp, lang):
        return str(sv_file) if lang == "sv" and os.path.exists(str(sv_file)) else None

    with patch.object(coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_poll:
        mock_poll.return_value = BazarrJobsPollResult(
            status=BazarrJobPollStatus.ACTIVE, jobs=unrelated_jobs
        )

        result = await coordinator.acquire_publication_ownership(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="test",
            provided_source=ref_source,
            job_id=None,
            timeout_sec=5.0,
            find_external_subtitle_fn=find_sv,
            pre_trigger_snapshot=pre_snap,
            search_accepted=True,
            media_correlated=True,
        )

    assert result.adopted is True, f"Expected adopted=True, got: {result}"
    assert result.granted is False
    assert result.trust_result is not None
    assert result.trust_result.passed is True
    assert result.trust_result.repair is not None
    assert abs(result.trust_result.repair["applied_shift_sec"] - (-20.45)) < 0.1

    # Verify candidate file on disk was shifted
    repaired_cues = list(srt.parse(sv_file.read_text(encoding="utf-8")))
    assert len(repaired_cues) == len(target_specs)
    expected_first_start = target_specs[0][0] - 20.45
    assert abs(repaired_cues[0].start.total_seconds() - expected_first_start) < 0.1


# ==============================================================================
# L. S04E08-STYLE HYBRID RACE REGRESSION (Scenario A):
#    search accepted, media correlated, Bazarr poll = ACTIVE (generic/unclassified job),
#    target appears during Hybrid window, candidate is stable and healthy ->
#    candidate is NOT ignored, evaluated, and BAZARR candidate is adopted (AI skipped).
# ==============================================================================
@pytest.mark.asyncio
async def test_scenario_a_s04e08_hybrid_race_unclassified_jobs_does_not_block_candidate(tmp_path):
    """
    Scenario A: S04E08 pattern where Bazarr has an unclassified job active globally.
    A candidate appears on disk during the Hybrid coordination window.
    Expected:
      - candidate is detected immediately
      - Trust Engine evaluates candidate: PASS (score >= 85)
      - coordinate_target returns FINALIZED_WITH_TARGET with Trust PASS
      - AI is not required (AI calls: 0).
    """
    video_path = str(tmp_path / "Black.Sails.S04E08.1080p.mkv")
    Path(video_path).touch()

    ref_specs, target_specs, ref_cues, _ = _build_black_sails_fixture()
    # Aligned Swedish cues matching reference timing
    aligned_tgt_cues = [
        srt.Subtitle(
            index=i + 1,
            start=datetime.timedelta(seconds=target_specs[i][0] - 20.45),
            end=datetime.timedelta(seconds=target_specs[i][1] - 20.45),
            content=target_specs[i][2],
        )
        for i in range(len(target_specs))
    ]

    sv_file = tmp_path / "Black.Sails.S04E08.sv.srt"
    sv_file.write_text(srt.compose(aligned_tgt_cues), encoding="utf-8")

    en_file = tmp_path / "Black.Sails.S04E08.en.srt"
    en_file.write_text(srt.compose(ref_cues), encoding="utf-8")
    ref_source = SubtitleSource(
        origin=SourceOrigin.EMBEDDED,
        path=str(en_file),
        language="en",
        content=srt.compose(ref_cues),
        cues=ref_cues,
    )

    coordinator = BazarrCoordinator()
    unclassified_job = BazarrJobInfo(
        job_id="generic-123",
        job_name="Downloading Subtitles",
        status="Running",
        job_type="download_subtitles",
        progress_message="Queue worker active...",
    )

    def find_sv(vp, lang):
        return str(sv_file) if lang == "sv" and os.path.exists(str(sv_file)) else None

    with patch.object(coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_poll:
        mock_poll.return_value = BazarrJobsPollResult(
            status=BazarrJobPollStatus.ACTIVE, jobs=[unclassified_job]
        )

        state, path, trust_res = await coordinator.coordinate_target(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="test",
            max_wait_seconds=4.0,
            candidate_stability_sec=0.05,
            quiescence_sec=0.1,
            provided_source=ref_source,
            find_external_subtitle_fn=find_sv,
            pre_trigger_snapshot=TargetSnapshot(path="", exists=False, size=0, mtime_ns=0),
            search_accepted=True,
            media_correlated=True,
        )

    assert state == BazarrLifecycleState.FINALIZED_WITH_TARGET
    assert path == str(sv_file)
    assert trust_res is not None
    assert trust_res.passed is True


# ==============================================================================
# M. REAL CORRELATED TARGET WRITER SAFETY (Scenario C):
#    Video has explicit correlated search/sync jobs in Bazarr ->
#    acquire_publication_ownership blocks with defer=True, reason="bazarr_actively_writing".
# ==============================================================================
@pytest.mark.asyncio
async def test_scenario_c_correlated_target_writer_blocks_publication(tmp_path):
    """
    Scenario C: Video has an active search/sync job explicitly matching its filename/title.
    Expected:
      - acquire_publication_ownership blocks (granted=False, defer=True)
      - reason is "bazarr_actively_writing"
      - Babel does NOT overwrite target.
    """
    video_path = str(tmp_path / "Black.Sails.S04E07.1080p.mkv")
    Path(video_path).touch()

    coordinator = BazarrCoordinator()
    correlated_job = BazarrJobInfo(
        job_id="sync-s04e07",
        job_name="Sync subtitle for Black.Sails.S04E07.1080p.mkv",
        status="Running",
        job_type="sync",
        progress_message="Syncing Swedish subtitles...",
        matched_language="sv",
    )

    with patch.object(coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_poll:
        mock_poll.return_value = BazarrJobsPollResult(
            status=BazarrJobPollStatus.ACTIVE, jobs=[correlated_job]
        )

        result = await coordinator.acquire_publication_ownership(
            video_path=video_path,
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="test",
            job_id=None,
            timeout_sec=0.2,
            pre_trigger_snapshot=TargetSnapshot(path="", exists=False, size=0, mtime_ns=0),
            search_accepted=True,
            media_correlated=True,
        )

    assert result.granted is False
    assert result.defer is True
    assert result.reason in ("bazarr_actively_writing", "bazarr_target_syncing")


# ==============================================================================
# N. PUBLICATION PENDING REUSE TEST (Scenario D):
#    AI translation completes once, QA PASS, real correlated writer blocks publication.
#    Expected first pass:
#      - status = WAITING_FOR_BAZARR
#      - translation artifact saved with complete payload
#    Expected retry pass:
#      - publication gate arbitrates only
#      - AI translation calls = 0 (primary translation not rerun)
#      - Babel output published when Bazarr is clear.
# ==============================================================================
@pytest.mark.asyncio
async def test_scenario_d_publication_pending_reuse_without_retranslation(tmp_path, monkeypatch):
    """
    Scenario D: Full pipeline execution where publication is initially deferred.
    First pass sets status WAITING_FOR_BAZARR.
    Retry pass reuses the QA-passed artifact and performs 0 AI calls.
    """
    video_path = tmp_path / "Black.Sails.S03E05.1080p.mkv"
    video_path.touch()

    target_srt = tmp_path / "Black.Sails.S03E05.1080p.sv.srt"
    src_srt = tmp_path / "Black.Sails.S03E05.1080p.en.srt"
    cues_data = [
        (1.0, 3.0, "Line 1 in English"),
        (4.0, 6.0, "Line 2 in English"),
        (7.0, 9.0, "Line 3 in English"),
        (10.0, 12.0, "Line 4 in English"),
        (13.0, 15.0, "Line 5 in English"),
    ]
    en_cues = [srt.Subtitle(index=i+1, start=datetime.timedelta(seconds=s), end=datetime.timedelta(seconds=e), content=t) for i, (s, e, t) in enumerate(cues_data)]
    src_srt.write_text(srt.compose(en_cues), encoding="utf-8")

    pipeline = SubtitlePipeline()

    ai_call_counter = [0]
    async def mock_ai_translate(*args, **kwargs):
        ai_call_counter[0] += 1
        return [
            srt.Subtitle(index=i+1, start=c.start, end=c.end, content=f"Rad {i+1} på svenska")
            for i, c in enumerate(en_cues)
        ]

    # Patch settings
    monkeypatch.setattr("app.services.pipeline.get_setting", lambda k, default="": {
        "enable_bazarr_check": "true",
        "bazarr_url": "http://bazarr:6767",
        "bazarr_api_key": "testkey",
        "target_languages": "sv",
        "languages": '[{"name": "Swedish", "code": "sv", "enabled": true}]',
        "auto_repair_unhealthy": "true",
        "extract_target_embedded": "false",
        "extract_source_embedded": "false",
    }.get(k, default))

    # Pass 1: Correlated sync job blocks publication ownership
    active_sync_job = BazarrJobInfo(
        job_id="sync-s03e05",
        job_name="Sync subtitle for Black.Sails.S03E05.1080p.mkv",
        status="Running",
        job_type="sync",
        matched_language="sv",
    )

    job_id = db_mod.create_job(video_path=str(video_path), event_source="MANUAL")

    with (
        patch.object(pipeline.translator, "translate_srt_content", side_effect=mock_ai_translate),
        patch.object(bazarr_coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_poll,
        patch.object(pipeline, "trigger_bazarr_search", new_callable=AsyncMock, return_value=True),
    ):
        mock_poll.return_value = BazarrJobsPollResult(status=BazarrJobPollStatus.ACTIVE, jobs=[active_sync_job])

        res1 = await pipeline.process_video_file(
            video_path=str(video_path),
            job_id=job_id,
            wait_seconds=0,
        )

    # Verify Pass 1 outcome: WAITING_FOR_BAZARR status, 1 AI call made, qapassed artifact created
    job_record_1 = db_mod.get_job_by_id(job_id)
    assert job_record_1["status"] == "WAITING_FOR_BAZARR"
    assert ai_call_counter[0] == 1
    assert not target_srt.exists()

    data_dir = os.path.dirname(db_mod.DB_PATH)
    qapassed_path = os.path.join(data_dir, f"job_{job_id}_sv_qapassed.json")
    assert os.path.exists(qapassed_path)

    # Pass 2: Retry with Bazarr now KNOWN_IDLE (writer clear)
    with (
        patch.object(pipeline.translator, "translate_srt_content", side_effect=mock_ai_translate),
        patch.object(bazarr_coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_poll2,
        patch.object(pipeline, "trigger_bazarr_search", new_callable=AsyncMock, return_value=True),
    ):
        mock_poll2.return_value = BazarrJobsPollResult(status=BazarrJobPollStatus.KNOWN_IDLE, jobs=[])

        res2 = await pipeline.process_video_file(
            video_path=str(video_path),
            job_id=job_id,
            event_source="RETRY",
            wait_seconds=0,
        )

    # Verify Pass 2 outcome: TRANSLATED status, EXACTLY 0 new AI calls (total remains 1), file published
    job_record_2 = db_mod.get_job_by_id(job_id)
    assert job_record_2["status"] == "TRANSLATED"
    assert ai_call_counter[0] == 1, f"Expected AI calls to remain exactly 1, but got {ai_call_counter[0]}"
    assert target_srt.exists()
    assert not os.path.exists(qapassed_path)


# ==============================================================================
# O. LATE BAZARR ADOPTION DURING PUBLICATION WAIT (Scenario E):
#    AI translation already QA PASSED and saved, publication was waiting.
#    A current-run Bazarr candidate appears on disk with +20.45s offset (LOW_COVERAGE).
#    On retry, safe offset repair -> exact Trust PASS -> Bazarr human target adopted.
#    Pending Babel output discarded/skipped, 0 new AI calls.
# ==============================================================================
@pytest.mark.asyncio
async def test_scenario_e_late_bazarr_adoption_during_publication_wait(tmp_path, monkeypatch):
    """
    Scenario E: Late repairable human Bazarr candidate appears while publication is deferred.
    Expected:
      - Human candidate is repaired and adopted (BAZARR MATCH)
      - Stored Babel AI output is discarded
      - ZERO new AI calls made.
    """
    ref_specs, target_specs, ref_cues, tgt_cues = _build_black_sails_fixture()

    video_path = tmp_path / "Black.Sails.S04E07.1080p.mkv"
    video_path.touch()

    target_srt = tmp_path / "Black.Sails.S04E07.1080p.sv.srt"
    src_srt = tmp_path / "Black.Sails.S04E07.1080p.en.srt"
    src_srt.write_text(srt.compose(ref_cues), encoding="utf-8")

    pipeline = SubtitlePipeline()

    ai_call_counter = [0]
    async def mock_ai_translate(*args, **kwargs):
        ai_call_counter[0] += 1
        return [
            srt.Subtitle(index=i+1, start=c.start, end=c.end, content=f"AI översatt {i+1}")
            for i, c in enumerate(ref_cues)
        ]

    monkeypatch.setattr("app.services.pipeline.get_setting", lambda k, default="": {
        "enable_bazarr_check": "true",
        "bazarr_url": "http://bazarr:6767",
        "bazarr_api_key": "testkey",
        "target_languages": "sv",
        "languages": '[{"name": "Swedish", "code": "sv", "enabled": true}]',
        "auto_repair_unhealthy": "true",
        "extract_target_embedded": "false",
        "extract_source_embedded": "false",
    }.get(k, default))

    # Initial Pass: sync job active, blocks publication
    active_sync_job = BazarrJobInfo(
        job_id="sync-s04e07",
        job_name="Sync subtitle for Black.Sails.S04E07.1080p.mkv",
        status="Running",
        job_type="sync",
        matched_language="sv",
    )

    job_id = db_mod.create_job(video_path=str(video_path), event_source="MANUAL")

    with (
        patch.object(pipeline.translator, "translate_srt_content", side_effect=mock_ai_translate),
        patch.object(bazarr_coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_poll,
        patch.object(pipeline, "trigger_bazarr_search", new_callable=AsyncMock, return_value=True),
    ):
        mock_poll.return_value = BazarrJobsPollResult(status=BazarrJobPollStatus.ACTIVE, jobs=[active_sync_job])

        res1 = await pipeline.process_video_file(
            video_path=str(video_path),
            job_id=job_id,
            wait_seconds=0,
        )

    job_record_1 = db_mod.get_job_by_id(job_id)
    assert job_record_1["status"] == "WAITING_FOR_BAZARR"
    assert ai_call_counter[0] == 1

    # While waiting, Bazarr finishes writing Swedish target candidate with +20.45s offset
    target_srt.write_text(srt.compose(tgt_cues), encoding="utf-8")

    # Retry pass: Bazarr is now idle
    with (
        patch.object(pipeline.translator, "translate_srt_content", side_effect=mock_ai_translate),
        patch.object(bazarr_coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_poll2,
        patch.object(pipeline, "trigger_bazarr_search", new_callable=AsyncMock, return_value=True),
    ):
        mock_poll2.return_value = BazarrJobsPollResult(status=BazarrJobPollStatus.KNOWN_IDLE, jobs=[])

        res2 = await pipeline.process_video_file(
            video_path=str(video_path),
            job_id=job_id,
            event_source="RETRY",
            wait_seconds=0,
        )

    job_record_2 = db_mod.get_job_by_id(job_id)
    assert job_record_2["status"] == "BAZARR MATCH"
    assert ai_call_counter[0] == 1, "AI re-translation must not have been executed on retry"

    # Verify repaired human subtitle is on disk
    repaired_cues = list(srt.parse(target_srt.read_text(encoding="utf-8")))
    assert len(repaired_cues) == len(target_specs)
    expected_first_start = target_specs[0][0] - 20.45
    assert abs(repaired_cues[0].start.total_seconds() - expected_first_start) < 0.1
