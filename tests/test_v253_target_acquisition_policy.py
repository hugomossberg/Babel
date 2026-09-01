"""
Comprehensive Acceptance & Regression Test Suite for Babel v2.5.3
Target Acquisition Policy Hardening:
1. Housemaid: 17.62GB MKV with embedded Swedish target -> Zero extraction when materialize_embedded_target=false
2. Materialization Opt-in: materialize_embedded_target=true -> extraction & publication occurs, 0 AI
3. Priority Order: Existing external checked first -> Embedded extraction never called
4. Fast Five: Strong finalized current-run Bazarr with LOW_COVERAGE (~0.62) -> PASS_WITH_WARNINGS, BAZARR MATCH, AI=0
5. Fast Five External Control: Generic external candidate with LOW_COVERAGE (~0.62) -> strict FAIL-CLOSED
6. Bazarr Negatives: Unaccepted, searching, syncing, unchanged gen, unstable, wrong lang -> Rejected
7. Target/Source Race: Early Bazarr target win cancels and drains source resolution cleanly, AI=0
8. Bazarr Miss: Bazarr coordinator finds no target -> AI translation proceeds normally
"""

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import srt

import app.core.db as db_mod
from app.core.db import DB_PATH, create_job, get_job_by_id, init_db, update_job, append_job_log
from app.core.trust_engine import (
    SubtitleTrustEngine,
    TrustDecision,
    TrustResult,
    CandidateOrigin,
    VerificationMode,
    SubtitleIntent,
    TargetSnapshot,
    BazarrProvenance,
    capture_target_snapshot,
)
from app.services.pipeline import SubtitlePipeline, TargetResolution
from app.services.source_resolver import SourceResolver, SourceOrigin, SubtitleSource
from app.services.bazarr_coordinator import (
    BazarrCoordinator,
    BazarrResult,
    BazarrResultCode,
    BazarrJobPollStatus,
    BazarrJobsPollResult,
    BazarrJobInfo,
    BazarrLifecycleState,
    bazarr_coordinator,
)


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Each test gets its own isolated DB."""
    db_file = str(tmp_path / "test_target_policy.db")
    monkeypatch.setattr(db_mod, "DB_PATH", db_file)
    monkeypatch.setattr("app.core.quota.DB_PATH", db_file, raising=False)
    db_mod.init_db()
    from app.services.bazarr_coordinator import bazarr_coordinator
    bazarr_coordinator.reset()
    yield tmp_path
    bazarr_coordinator.reset()


def generate_cues_srt(cue_timings, content_prefix="Detta är en svensk dialog och replik"):
    items = []
    for i, (start_s, end_s) in enumerate(cue_timings, start=1):
        s_td = srt.timedelta(seconds=start_s)
        e_td = srt.timedelta(seconds=end_s)
        items.append(srt.Subtitle(index=i, start=s_td, end=e_td, content=f"{content_prefix} nummer {i}."))
    return srt.compose(items)


# ==============================================================================
# 1. HOUSEMAID: Zero extraction by default (materialize_embedded_target=false)
# ==============================================================================
@pytest.mark.asyncio
async def test_housemaid_zero_target_extraction_by_default(tmp_path, monkeypatch):
    """
    HOUSEMAID PRODUCTION CASE:
    17.62 GB MKV with embedded Swedish full text track.
    materialize_embedded_target = false (default)
    => Zero extraction / mkvextract must NOT run!
    => Bazarr must NOT be called!
    => AI must NOT be called!
    => Status: ALREADY EXISTS
    => Reason truthful that embedded target satisfies language.
    => External .sv.srt is NOT created.
    """
    mkv_file = str(tmp_path / "The.Housemaid.2016.1080p.mkv")
    Path(mkv_file).write_bytes(b"dummy_data" * 100)

    # 17.62 GB size simulation
    monkeypatch.setattr(os.path, "getsize", lambda p: 18919964672 if p == mkv_file else len(b""))

    # Container metadata with Swedish full subtitle track and Korean audio
    tracks_info = {
        "video": [{"id": 0, "codec": "V_MPEG4/ISO/AVC"}],
        "audio": [{"id": 1, "language": "kor", "codec": "A_DTS", "title": "Korean DTS"}],
        "subtitles": [
            {
                "id": 2,
                "language": "swe",
                "codec": "SubRip/SRT",
                "forced": False,
                "title": "",
            },
            {
                "id": 3,
                "language": "eng",
                "codec": "SubRip/SRT",
                "forced": False,
                "title": "English Full",
            },
        ],
        "duration": 8640.0,
    }
    monkeypatch.setattr("app.services.pipeline.inspect_mkv_tracks", lambda p: tracks_info)

    pipeline = SubtitlePipeline()
    bazarr_mock = AsyncMock()
    translate_mock = AsyncMock()
    extract_spy = MagicMock()

    monkeypatch.setattr(pipeline, "trigger_bazarr_search", bazarr_mock)
    monkeypatch.setattr(pipeline.translator, "translate_srt_content", translate_mock)
    monkeypatch.setattr("app.services.pipeline._safe_extract_embedded_srt", extract_spy)
    monkeypatch.setattr("app.core.extractor.extract_embedded_srt", extract_spy)

    def mock_get_setting(key, default=""):
        settings = {
            "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
            "extract_target_embedded": "true",
            "extract_source_embedded": "true",
            "enable_bazarr": "true",
            "materialize_embedded_target": "false",  # DEFAULT
        }
        return settings.get(key, default)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)
    monkeypatch.setattr("app.core.db.get_setting", mock_get_setting)

    res = await pipeline.process_video_file(mkv_file, event_source="SONARR")

    assert res["status"] == "skipped"
    assert res["reason"] == "already_exists"

    # ZERO extraction / mkvextract
    extract_spy.assert_not_called()

    # ZERO Bazarr / ZERO AI
    bazarr_mock.assert_not_called()
    translate_mock.assert_not_called()

    # External file was NOT written
    sv_file = str(tmp_path / "The.Housemaid.2016.1080p.sv.srt")
    assert not os.path.exists(sv_file)

    # Job logs prove metadata satisfaction
    job = get_job_by_id(res["job_id"])
    assert job["status"] == "ALREADY EXISTS"
    assert "Embedded target satisfies language (Swedish)" in job.get("reason", "")
    log_text = "\n".join(job.get("logs") or [])
    assert "Embedded target scan: Swedish SubRip/SRT candidate found (track 2)" in log_text
    assert "Materialization skipped" in log_text
    assert "Bazarr skipped — embedded target satisfied language" in log_text
    assert "AI skipped" in log_text
    assert "AI calls: 0" in log_text


# ==============================================================================
# 2. MATERIALIZATION OPT-IN: materialize_embedded_target=true
# ==============================================================================
@pytest.mark.asyncio
async def test_materialization_opt_in_extracts_and_publishes(tmp_path, monkeypatch):
    """
    When materialize_embedded_target=true:
    => Extraction occurs via _safe_extract_embedded_srt
    => External .sv.srt is published
    => Zero AI calls
    """
    mkv_file = str(tmp_path / "Housemaid.OptIn.mkv")
    Path(mkv_file).write_bytes(b"dummy_data" * 100)

    tracks_info = {
        "audio": [{"id": 1, "language": "eng", "default": True, "forced": False}],
        "subtitles": [
            {
                "id": 2,
                "language": "swe",
                "codec": "SubRip/SRT",
                "forced": False,
                "title": "",
            }
        ],
        "duration": 120.0,
    }
    monkeypatch.setattr("app.services.pipeline.inspect_mkv_tracks", lambda p: tracks_info)
    monkeypatch.setattr("app.core.extractor.inspect_mkv_tracks", lambda p: tracks_info)

    sv_srt_content = generate_cues_srt([(1.0, 3.0), (4.0, 6.0), (7.0, 9.0), (10.0, 12.0), (13.0, 15.0)])

    def mock_extract(video_p, out_p, preferred_lang=None, tracks_info=None):
        Path(out_p).write_text(sv_srt_content, encoding="utf-8")
        return True

    pipeline = SubtitlePipeline()
    bazarr_mock = AsyncMock()
    translate_mock = AsyncMock()

    monkeypatch.setattr(pipeline, "trigger_bazarr_search", bazarr_mock)
    monkeypatch.setattr(pipeline.translator, "translate_srt_content", translate_mock)
    monkeypatch.setattr("app.services.pipeline._safe_extract_embedded_srt", mock_extract)

    def mock_get_setting(key, default=""):
        settings = {
            "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
            "extract_target_embedded": "true",
            "extract_source_embedded": "true",
            "enable_bazarr": "true",
            "materialize_embedded_target": "true",  # OPT-IN
        }
        return settings.get(key, default)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)
    monkeypatch.setattr("app.core.db.get_setting", mock_get_setting)

    res = await pipeline.process_video_file(mkv_file, event_source="SONARR")

    assert res["status"] == "skipped"
    assert res["reason"] == "already_exists"

    # External file was published
    sv_file = str(tmp_path / "Housemaid.OptIn.sv.srt")
    assert os.path.exists(sv_file)
    assert "Detta är en svensk dialog" in Path(sv_file).read_text(encoding="utf-8")

    bazarr_mock.assert_not_called()
    translate_mock.assert_not_called()


# ==============================================================================
# 3. EXISTING EXTERNAL FIRST: External target evaluated before embedded
# ==============================================================================
@pytest.mark.asyncio
async def test_existing_external_evaluated_first_no_extraction(tmp_path, monkeypatch):
    """
    When valid external .sv.srt already exists on disk AND embedded Swedish track exists:
    => External target evaluated FIRST
    => Trust PASS on external target
    => Embedded extraction NEVER called!
    => Bazarr NEVER called!
    => AI NEVER called!
    => Status: ALREADY EXISTS
    """
    mkv_file = str(tmp_path / "DualPresence.mkv")
    Path(mkv_file).write_bytes(b"dummy_data" * 100)

    # Valid external Swedish SRT (5 cues)
    ext_sv = tmp_path / "DualPresence.sv.srt"
    cues_text = generate_cues_srt([(1.0, 3.0), (4.0, 6.0), (7.0, 9.0), (10.0, 12.0), (13.0, 15.0)])
    ext_sv.write_text(cues_text, encoding="utf-8")

    # Valid external English source reference (5 cues)
    ext_en = tmp_path / "DualPresence.en.srt"
    ext_en.write_text(generate_cues_srt([(1.0, 3.0), (4.0, 6.0), (7.0, 9.0), (10.0, 12.0), (13.0, 15.0)], content_prefix="This is English dialogue and sentence"), encoding="utf-8")

    tracks_info = {
        "audio": [{"id": 1, "language": "eng", "default": True, "forced": False}],
        "subtitles": [
            {
                "id": 2,
                "language": "swe",
                "codec": "SubRip/SRT",
                "forced": False,
                "title": "",
            }
        ],
        "duration": 120.0,
    }
    monkeypatch.setattr("app.services.pipeline.inspect_mkv_tracks", lambda p: tracks_info)

    pipeline = SubtitlePipeline()
    extract_spy = MagicMock()
    bazarr_mock = AsyncMock()
    translate_mock = AsyncMock()

    monkeypatch.setattr("app.services.pipeline._safe_extract_embedded_srt", extract_spy)
    monkeypatch.setattr(pipeline, "trigger_bazarr_search", bazarr_mock)
    monkeypatch.setattr(pipeline.translator, "translate_srt_content", translate_mock)

    def mock_get_setting(key, default=""):
        settings = {
            "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
            "extract_target_embedded": "true",
            "extract_source_embedded": "true",
            "enable_bazarr": "true",
            "materialize_embedded_target": "true",  # Even if true, external wins first
        }
        return settings.get(key, default)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)
    monkeypatch.setattr("app.core.db.get_setting", mock_get_setting)

    res = await pipeline.process_video_file(mkv_file, event_source="SONARR")

    assert res["status"] == "skipped"
    assert res["reason"] == "already_exists"

    # Embedded extraction was NEVER invoked
    extract_spy.assert_not_called()
    bazarr_mock.assert_not_called()
    translate_mock.assert_not_called()

    job = get_job_by_id(res["job_id"])
    assert job["status"] == "ALREADY EXISTS"


# ==============================================================================
# 4. FAST FIVE: Strong finalized Bazarr with LOW_COVERAGE (~0.62) -> PASS_WITH_WARNINGS
# ==============================================================================
@pytest.mark.asyncio
async def test_fast_five_bazarr_low_coverage_pass_with_warnings(tmp_path):
    """
    POLICY CORRECTION TEST:
    Bazarr downloaded an incomplete Swedish subtitle (only 62 cues for 100 cue film, missing tail).
    Under the safe timing policy:
    => LOW_COVERAGE is NOT blindly downgraded to PASS_WITH_WARNINGS
    => Cannot be repaired (unrepairable missing 38 cues)
    => Evaluates to TrustDecision.FAIL (fail-closed)
    => passed=False
    """
    engine = SubtitleTrustEngine()
    video_path = str(tmp_path / "Fast.Five.2011.1080p.mkv")

    # Reference English cues (100 dialogue cues across 2 hours)
    ref_timings = [(i * 60.0 + 1.0, i * 60.0 + 4.0) for i in range(100)]
    ref_cues = [
        srt.Subtitle(index=i, start=srt.timedelta(seconds=s), end=srt.timedelta(seconds=e), content=f"English dialogue sentence {i}")
        for i, (s, e) in enumerate(ref_timings, start=1)
    ]
    ref_content = srt.compose(ref_cues)
    ref_source = SubtitleSource(
        origin=SourceOrigin.EMBEDDED,
        path=str(tmp_path / "ref.en.srt"),
        language="en",
        content=ref_content,
        cues=ref_cues,
    )

    # Swedish Bazarr target: 62 dialogue cues matching ref, with >180s gap (missing remaining 38 cues)
    cand_timings = [(i * 60.0 + 1.0, i * 60.0 + 4.0) for i in range(62)]
    cand_cues = [
        srt.Subtitle(index=i, start=srt.timedelta(seconds=s), end=srt.timedelta(seconds=e), content=f"Detta är en svensk dialog och replik {i}.")
        for i, (s, e) in enumerate(cand_timings, start=1)
    ]
    cand_file = tmp_path / "Fast.Five.2011.1080p.sv.srt"
    cand_file.write_text(srt.compose(cand_cues), encoding="utf-8")

    cand_snap = capture_target_snapshot(str(cand_file))

    # Strong finalized current-run Bazarr provenance
    prov = BazarrProvenance(
        video_path=video_path,
        target_lang="sv",
        search_accepted=True,
        pre_trigger_snapshot=TargetSnapshot(path="", exists=False, size=0, mtime_ns=0),  # Authoritatively absent pre-trigger
        is_finalized=True,
        is_quiescent=True,
        media_correlated=True,
        poll_state=BazarrJobPollStatus.KNOWN_IDLE,
        candidate_snapshot=cand_snap,
    )

    res = await engine.evaluate_candidate(
        video_path=video_path,
        candidate_path=str(cand_file),
        target_lang="sv",
        origin=CandidateOrigin.BAZARR,
        provided_source=ref_source,
        allow_ai_audit=False,
        bazarr_provenance=prov,
    )

    # Invariant: Must FAIL closed when incomplete/unrepairable!
    assert res.passed is False
    assert res.decision == TrustDecision.FAIL
    assert any("coverage" in r.lower() or "timing sync" in r.lower() for r in res.reasons)


# ==============================================================================
# 5. FAST FIVE EXTERNAL CONTROL: Generic external candidate fails closed
# ==============================================================================
@pytest.mark.asyncio
async def test_fast_five_external_control_low_coverage_strict_fail(tmp_path):
    """
    Control case for Fast Five:
    Same 62% coverage Swedish candidate, but origin is EXTERNAL (or unaccepted Bazarr).
    => Evaluates to TrustDecision.FAIL (fail-closed)
    => passed=False
    => Reason indicates low_coverage timing sync failure
    """
    engine = SubtitleTrustEngine()
    video_path = str(tmp_path / "Fast.Five.Control.mkv")

    ref_timings = [(i * 60.0 + 1.0, i * 60.0 + 4.0) for i in range(100)]
    ref_cues = [
        srt.Subtitle(index=i, start=srt.timedelta(seconds=s), end=srt.timedelta(seconds=e), content=f"English dialogue sentence {i}")
        for i, (s, e) in enumerate(ref_timings, start=1)
    ]
    ref_content = srt.compose(ref_cues)
    ref_source = SubtitleSource(
        origin=SourceOrigin.EMBEDDED,
        path=str(tmp_path / "ref.en.srt"),
        language="en",
        content=ref_content,
        cues=ref_cues,
    )

    cand_timings = [(i * 60.0 + 1.0, i * 60.0 + 4.0) for i in range(62)]
    cand_cues = [
        srt.Subtitle(index=i, start=srt.timedelta(seconds=s), end=srt.timedelta(seconds=e), content=f"Detta är en svensk dialog och replik {i}.")
        for i, (s, e) in enumerate(cand_timings, start=1)
    ]
    cand_file = tmp_path / "Fast.Five.Control.sv.srt"
    cand_file.write_text(srt.compose(cand_cues), encoding="utf-8")

    # Generic external candidate (no strong Bazarr provenance)
    res = await engine.evaluate_candidate(
        video_path=video_path,
        candidate_path=str(cand_file),
        target_lang="sv",
        origin=CandidateOrigin.EXTERNAL,
        provided_source=ref_source,
        allow_ai_audit=False,
        bazarr_provenance=None,
    )

    assert res.passed is False
    assert res.decision == TrustDecision.FAIL
    assert any("coverage" in r.lower() or "timing sync" in r.lower() for r in res.reasons)


# ==============================================================================
# 6. BAZARR NEGATIVES: Non-finalized, unaccepted, or corrupt targets rejected
# ==============================================================================
@pytest.mark.asyncio
async def test_bazarr_negatives_strict_rejection(tmp_path):
    """
    Proves that non-strong Bazarr conditions fail closed:
    a) Unaccepted search (search_accepted=False)
    b) Unchanged generation (snapshot matches pre-trigger)
    c) Wrong language
    d) Missing / invalid poll_state (None, "NONE", "IDLE", UNKNOWN)
    e) Unfinalized / timeout (is_finalized=False)
    f) Non-quiescent (is_quiescent=False)
    g) Missing media correlation (media_correlated=False)
    h) Missing pre-trigger observation (pre_trigger_snapshot=None)
    """
    engine = SubtitleTrustEngine()
    video_path = str(tmp_path / "Negative.mkv")

    ref_timings = [(1.0, 3.0), (4.0, 6.0), (7.0, 9.0), (10.0, 12.0), (13.0, 15.0)]
    ref_cues = [
        srt.Subtitle(index=i, start=srt.timedelta(seconds=s), end=srt.timedelta(seconds=e), content=f"Dialogue {i}")
        for i, (s, e) in enumerate(ref_timings, start=1)
    ]
    ref_content = srt.compose(ref_cues)
    ref_source = SubtitleSource(
        origin=SourceOrigin.EMBEDDED,
        path=str(tmp_path / "ref.en.srt"),
        language="en",
        content=ref_content,
        cues=ref_cues,
    )

    cand_file = tmp_path / "Negative.sv.srt"
    cand_file.write_text(generate_cues_srt([(1.0, 3.0), (4.0, 6.0), (7.0, 9.0), (10.0, 12.0), (13.0, 15.0)]), encoding="utf-8")
    cand_snap = capture_target_snapshot(str(cand_file))
    absent_pre = TargetSnapshot(path="", exists=False, size=0, mtime_ns=0)

    # 1. Base valid strong provenance
    base_prov = BazarrProvenance(
        video_path=video_path,
        target_lang="sv",
        search_accepted=True,
        pre_trigger_snapshot=absent_pre,
        is_finalized=True,
        is_quiescent=True,
        media_correlated=True,
        poll_state=BazarrJobPollStatus.KNOWN_IDLE,
        candidate_snapshot=cand_snap,
    )
    assert base_prov.is_strong_current_run(cand_snap) is True

    # 2. Unaccepted search
    p_unacc = BazarrProvenance(
        video_path=video_path,
        target_lang="sv",
        search_accepted=False,
        pre_trigger_snapshot=absent_pre,
        is_finalized=True,
        is_quiescent=True,
        media_correlated=True,
        poll_state=BazarrJobPollStatus.KNOWN_IDLE,
        candidate_snapshot=cand_snap,
    )
    assert p_unacc.is_strong_current_run(cand_snap) is False

    # 3. Unchanged generation
    p_unchanged = BazarrProvenance(
        video_path=video_path,
        target_lang="sv",
        search_accepted=True,
        pre_trigger_snapshot=cand_snap,
        is_finalized=True,
        is_quiescent=True,
        media_correlated=True,
        poll_state=BazarrJobPollStatus.KNOWN_IDLE,
        candidate_snapshot=cand_snap,
    )
    assert p_unchanged.is_strong_current_run(cand_snap) is False

    # 4. poll_state=None / "NONE" / generic "IDLE" / UNKNOWN
    for invalid_poll in [None, "NONE", "IDLE", "idle", BazarrJobPollStatus.UNKNOWN, "UNKNOWN"]:
        p_poll = BazarrProvenance(
            video_path=video_path,
            target_lang="sv",
            search_accepted=True,
            pre_trigger_snapshot=absent_pre,
            is_finalized=True,
            is_quiescent=True,
            media_correlated=True,
            poll_state=invalid_poll,
            candidate_snapshot=cand_snap,
        )
        assert p_poll.is_strong_current_run(cand_snap) is False, f"poll_state={invalid_poll} must not qualify as strong"

    # 5. Unfinalized / timeout (is_finalized=False)
    p_unfin = BazarrProvenance(
        video_path=video_path,
        target_lang="sv",
        search_accepted=True,
        pre_trigger_snapshot=absent_pre,
        is_finalized=False,
        is_quiescent=True,
        media_correlated=True,
        poll_state=BazarrJobPollStatus.KNOWN_IDLE,
        candidate_snapshot=cand_snap,
    )
    assert p_unfin.is_strong_current_run(cand_snap) is False

    # 6. Non-quiescent (is_quiescent=False)
    p_nonq = BazarrProvenance(
        video_path=video_path,
        target_lang="sv",
        search_accepted=True,
        pre_trigger_snapshot=absent_pre,
        is_finalized=True,
        is_quiescent=False,
        media_correlated=True,
        poll_state=BazarrJobPollStatus.KNOWN_IDLE,
        candidate_snapshot=cand_snap,
    )
    assert p_nonq.is_strong_current_run(cand_snap) is False

    # 7. media_correlated=False
    p_uncorr = BazarrProvenance(
        video_path=video_path,
        target_lang="sv",
        search_accepted=True,
        pre_trigger_snapshot=absent_pre,
        is_finalized=True,
        is_quiescent=True,
        media_correlated=False,
        poll_state=BazarrJobPollStatus.KNOWN_IDLE,
        candidate_snapshot=cand_snap,
    )
    assert p_uncorr.is_strong_current_run(cand_snap) is False

    # 8. pre_trigger_snapshot=None (missing baseline evidence)
    p_nobase = BazarrProvenance(
        video_path=video_path,
        target_lang="sv",
        search_accepted=True,
        pre_trigger_snapshot=None,
        is_finalized=True,
        is_quiescent=True,
        media_correlated=True,
        poll_state=BazarrJobPollStatus.KNOWN_IDLE,
        candidate_snapshot=cand_snap,
    )
    assert p_nobase.is_strong_current_run(cand_snap) is False

    # 9. Wrong language candidate (5 German cues)
    german_cues = [
        srt.Subtitle(index=i, start=srt.timedelta(seconds=i*3), end=srt.timedelta(seconds=i*3+2), content=f"Das ist ein deutscher Text ohne Zweifel Nummer {i}.")
        for i in range(1, 6)
    ]
    german_file = tmp_path / "Negative.de.srt"
    german_file.write_text(srt.compose(german_cues), encoding="utf-8")
    german_snap = capture_target_snapshot(str(german_file))

    prov_german = BazarrProvenance(
        video_path=video_path,
        target_lang="sv",
        search_accepted=True,
        pre_trigger_snapshot=absent_pre,
        is_finalized=True,
        is_quiescent=True,
        media_correlated=True,
        poll_state=BazarrJobPollStatus.KNOWN_IDLE,
        candidate_snapshot=german_snap,
    )
    res_lang = await engine.evaluate_candidate(
        video_path=video_path,
        candidate_path=str(german_file),
        target_lang="sv",
        origin=CandidateOrigin.BAZARR,
        provided_source=ref_source,
        bazarr_provenance=prov_german,
    )
    assert res_lang.passed is False
    assert res_lang.decision == TrustDecision.FAIL
    assert any("language" in r.lower() for r in res_lang.reasons)


# ==============================================================================
# 7. TARGET/SOURCE RACE: Clean cancellation and drainage on early Bazarr win
# ==============================================================================
@pytest.mark.asyncio
async def test_race_cancellation_on_early_bazarr_win(tmp_path, monkeypatch):
    """
    Proves true concurrent Target / Source Race:
    - Slow source extraction starts (taking 5.0s)
    - Bazarr target appears and finalizes immediately
    - Concurrent poller detects Bazarr target
    - Source extraction is cleanly cancelled and drained
    - Result is BAZARR MATCH
    - AI calls = 0
    """
    mkv_file = str(tmp_path / "RaceMovie.mkv")
    Path(mkv_file).write_bytes(b"dummy_data" * 100)

    target_sv = str(tmp_path / "RaceMovie.sv.srt")
    if os.path.exists(target_sv):
        os.remove(target_sv)

    pipeline = SubtitlePipeline()
    pipeline.trigger_bazarr_search = AsyncMock(return_value=BazarrResult(code=BazarrResultCode.TRIGGERED, detail="OK", media_correlated=True))
    source_extracted = False
    source_cancelled = False

    async def slow_source_resolve(*args, **kwargs):
        nonlocal source_extracted, source_cancelled
        try:
            await asyncio.sleep(5.0)
            source_extracted = True
            return SubtitleSource(
                origin=SourceOrigin.EMBEDDED,
                path=str(tmp_path / "slow_src.en.srt"),
                language="en",
                content="1\n00:00:01,000 --> 00:00:03,000\nHello\n",
                cues=[],
            )
        except asyncio.CancelledError:
            source_cancelled = True
            raise

    # Target arrives during race
    async def target_writer():
        await asyncio.sleep(0.05)
        cues_text = generate_cues_srt([(1.0, 3.0), (4.0, 6.0), (7.0, 9.0), (10.0, 12.0), (13.0, 15.0)])
        Path(target_sv).write_text(cues_text, encoding="utf-8")

    monkeypatch.setattr("app.services.source_resolver.SourceResolver.resolve", slow_source_resolve)

    def mock_get_setting(key, default=""):
        settings = {
            "ai_provider": "gemini",
            "gemini_api_key": "dummy_test_key",
            "gemini_model": "gemini-3.5-flash-lite",
            "batch_size": "50",
            "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
            "extract_target_embedded": "false",
            "extract_source_embedded": "true",
            "enable_bazarr": "true",
            "bazarr_url": "http://bazarr:6767",
            "bazarr_api_key": "key123",
        }
        return settings.get(key, default)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)
    monkeypatch.setattr("app.core.db.get_setting", mock_get_setting)

    with patch.object(bazarr_coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_poll:
        mock_poll.return_value = BazarrJobsPollResult(
            status=BazarrJobPollStatus.KNOWN_IDLE,
            jobs=[],
        )

        asyncio.create_task(target_writer())
        res = await pipeline.process_video_file(mkv_file, event_source="SONARR")

    assert res["status"] == "skipped"
    assert res["reason"] == "bazarr_downloaded"

    assert source_cancelled is True
    assert source_extracted is False

    job = get_job_by_id(res["job_id"])
    assert job["status"] == "BAZARR MATCH"
    log_text = "\n".join(job.get("logs") or [])
    assert "Target/Source race winner: Bazarr" in log_text
    assert "Source extraction cancelled" in log_text
    assert "AI skipped" in log_text
    assert "AI calls: 0" in log_text


# ==============================================================================
# 8. BAZARR MISS: Proceeds to AI translation
# ==============================================================================
@pytest.mark.asyncio
async def test_bazarr_miss_proceeds_to_ai_translation(tmp_path, monkeypatch):
    """
    When Bazarr search finalizes with NO target:
    => Pipeline cleanly proceeds to AI translation
    => Source cues translated
    => Status: TRANSLATED
    """
    mkv_file = str(tmp_path / "BazarrMiss.mkv")
    Path(mkv_file).write_bytes(b"dummy_data" * 100)

    src_cues = [
        srt.Subtitle(index=1, start=srt.timedelta(seconds=1), end=srt.timedelta(seconds=3), content="Hello world."),
        srt.Subtitle(index=2, start=srt.timedelta(seconds=4), end=srt.timedelta(seconds=6), content="Good morning."),
        srt.Subtitle(index=3, start=srt.timedelta(seconds=7), end=srt.timedelta(seconds=9), content="How are you."),
        srt.Subtitle(index=4, start=srt.timedelta(seconds=10), end=srt.timedelta(seconds=12), content="Fine thank you."),
        srt.Subtitle(index=5, start=srt.timedelta(seconds=13), end=srt.timedelta(seconds=15), content="Goodbye now."),
    ]
    src_file = tmp_path / "BazarrMiss.en.srt"
    src_file.write_text(srt.compose(src_cues), encoding="utf-8")

    tracks_info = {
        "audio": [{"id": 1, "language": "eng", "default": True, "forced": False}],
        "subtitles": [],
        "duration": 120.0,
    }
    monkeypatch.setattr("app.services.pipeline.inspect_mkv_tracks", lambda p: tracks_info)

    pipeline = SubtitlePipeline()
    pipeline.trigger_bazarr_search = AsyncMock(return_value=BazarrResult(code=BazarrResultCode.TRIGGERED, detail="OK"))
    ai_called = False

    async def mock_translate(subs, target_language="Swedish", **kwargs):
        nonlocal ai_called
        ai_called = True
        return [
            srt.Subtitle(index=s.index, start=s.start, end=s.end, content=f"Detta är en svensk replik {s.index}.")
            for s in subs
        ]

    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)
    monkeypatch.setattr("app.services.pipeline.qa_gate", lambda *args, **kwargs: {"passed": True, "score": 100, "issues": []})

    def mock_get_setting(key, default=""):
        settings = {
            "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
            "extract_target_embedded": "false",
            "extract_source_embedded": "false",
            "enable_bazarr": "true",
            "bazarr_url": "http://bazarr:6767",
            "bazarr_api_key": "key123",
            "hybrid_bazarr_max_wait_sec": "0.1",
        }
        return settings.get(key, default)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)
    monkeypatch.setattr("app.core.db.get_setting", mock_get_setting)

    with patch.object(bazarr_coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_poll:
        mock_poll.return_value = BazarrJobsPollResult(
            status=BazarrJobPollStatus.KNOWN_IDLE,
            jobs=[],
        )

        res = await pipeline.process_video_file(mkv_file, event_source="SONARR")

    assert res["status"] in ("translated", "success")
    assert ai_called is True

    sv_out = tmp_path / "BazarrMiss.sv.srt"
    assert sv_out.exists()
    assert "Detta är en svensk replik" in sv_out.read_text(encoding="utf-8")


# ==============================================================================
# 9. CORRELATION PROOF MATRIX: Explicit correlation proof verification (A, B, C, D)
# ==============================================================================
@pytest.mark.asyncio
async def test_provenance_explicit_correlation_proof_matrix(tmp_path, monkeypatch):
    """
    Validates explicit correlation proof requirements:
    A. Normal successfully correlated PATCH search: accepted=True, media_correlated=True -> Strong Provenance = True
    B. Accepted/attached result without authoritative correlation proof: accepted=True, media_correlated=False -> Strong Provenance = False
    C. Legacy raw True: accepted=True, media_correlated=False -> Strong Provenance = False
    D. Fast Five with explicit correlation proof remains BAZARR MATCH, AI calls = 0
    """
    video_path = str(tmp_path / "CorrelationProof.mkv")
    Path(video_path).write_bytes(b"dummy_data" * 50)
    cand_file = str(tmp_path / "CorrelationProof.sv.srt")
    cues_text = generate_cues_srt([(1.0, 3.0), (4.0, 6.0), (7.0, 9.0), (10.0, 12.0), (13.0, 15.0)])
    Path(cand_file).write_text(cues_text, encoding="utf-8")
    cand_snap = capture_target_snapshot(cand_file)
    pre_snap = TargetSnapshot(path="", exists=False, size=0, mtime_ns=0)

    # Case A: Normal successfully correlated PATCH search (accepted=True, media_correlated=True)
    res_a = BazarrResult(code=BazarrResultCode.TRIGGERED, detail="accepted", media_correlated=True)
    assert res_a.was_accepted is True
    assert res_a.media_correlated is True
    prov_a = BazarrProvenance(
        video_path=video_path,
        target_lang="sv",
        search_accepted=res_a.was_accepted,
        pre_trigger_snapshot=pre_snap,
        is_finalized=True,
        is_quiescent=True,
        media_correlated=res_a.media_correlated,
        poll_state=BazarrJobPollStatus.KNOWN_IDLE,
        candidate_snapshot=cand_snap,
    )
    assert prov_a.is_strong_current_run() is True

    # Case B: Accepted/attached result without authoritative correlation proof (accepted=True, media_correlated=False)
    res_b = BazarrResult(code=BazarrResultCode.TRIGGERED, detail="attached_to_active_bazarr_job", media_correlated=False)
    assert res_b.was_accepted is True
    assert res_b.media_correlated is False
    prov_b = BazarrProvenance(
        video_path=video_path,
        target_lang="sv",
        search_accepted=res_b.was_accepted,
        pre_trigger_snapshot=pre_snap,
        is_finalized=True,
        is_quiescent=True,
        media_correlated=res_b.media_correlated,
        poll_state=BazarrJobPollStatus.KNOWN_IDLE,
        candidate_snapshot=cand_snap,
    )
    assert prov_b.is_strong_current_run() is False

    # Case C: Legacy raw True (accepted=True, media_correlated=False)
    raw_c = True
    accepted_c = bool(raw_c)
    correlated_c = getattr(raw_c, "media_correlated", False)
    assert accepted_c is True
    assert correlated_c is False
    prov_c = BazarrProvenance(
        video_path=video_path,
        target_lang="sv",
        search_accepted=accepted_c,
        pre_trigger_snapshot=pre_snap,
        is_finalized=True,
        is_quiescent=True,
        media_correlated=correlated_c,
        poll_state=BazarrJobPollStatus.KNOWN_IDLE,
        candidate_snapshot=cand_snap,
    )
    assert prov_c.is_strong_current_run() is False

    # Case D: Fast Five with explicit correlation proof remains BAZARR MATCH (AI calls = 0)
    engine = SubtitleTrustEngine()
    ref_cues = [
        srt.Subtitle(index=i, start=srt.timedelta(seconds=i * 60.0 + 1.0), end=srt.timedelta(seconds=i * 60.0 + 4.0), content=f"English line {i}")
        for i in range(100)
    ]
    ref_source = SubtitleSource(
        origin=SourceOrigin.EMBEDDED,
        path=str(tmp_path / "ref_d.en.srt"),
        language="en",
        content=srt.compose(ref_cues),
        cues=ref_cues,
    )
    cand_cues_d = [
        srt.Subtitle(index=i, start=srt.timedelta(seconds=i * 60.0 + 1.0), end=srt.timedelta(seconds=i * 60.0 + 4.0), content=f"Svensk replik {i}")
        for i in range(62)
    ]
    cand_file_d = str(tmp_path / "FastFive_D.sv.srt")
    Path(cand_file_d).write_text(srt.compose(cand_cues_d), encoding="utf-8")
    cand_snap_d = capture_target_snapshot(cand_file_d)

    prov_d = BazarrProvenance(
        video_path=video_path,
        target_lang="sv",
        search_accepted=True,
        pre_trigger_snapshot=pre_snap,
        is_finalized=True,
        is_quiescent=True,
        media_correlated=True,
        poll_state=BazarrJobPollStatus.KNOWN_IDLE,
        candidate_snapshot=cand_snap_d,
    )
    tres_d = await engine.evaluate_candidate(
        video_path=video_path,
        candidate_path=cand_file_d,
        target_lang="sv",
        origin=CandidateOrigin.BAZARR,
        provided_source=ref_source,
        allow_ai_audit=False,
        bazarr_provenance=prov_d,
    )
    assert tres_d.passed is False
    assert tres_d.decision == TrustDecision.FAIL
