"""
Focused regression tests for Safe Global Offset Repair for Bazarr targets.

Test matrix covering:
A. Black-Sails-shaped case: complete target, constant +20.45s offset
   -> repair succeeds -> final Trust passes -> candidate text unchanged -> AI not required
B. Progressive drift:
   -> MUST NOT global-shift repair
C. Real missing 3-minute section / partial subtitle:
   -> MUST NOT repair
D. Sudden discontinuity / different cut:
   -> MUST NOT repair
E. Already good Bazarr subtitle:
   -> no repair attempted
"""

import asyncio
import datetime
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import srt

from app.core.trust_engine import (
    SubtitleTrustEngine,
    TrustDecision,
    CandidateOrigin,
    BazarrProvenance,
    TargetSnapshot,
    capture_target_snapshot,
    estimate_global_offset,
    align_subtitle_timelines,
    SyncErrorType,
)
from app.services.source_resolver import (
    SubtitleSource,
    SourceOrigin,
)
from app.services.bazarr_coordinator import (
    BazarrJobPollStatus,
    BazarrJobsPollResult,
    BazarrCoordinator,
)
from app.services.pipeline import SubtitlePipeline
import app.core.db as db_mod


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db_file = tmp_path / "test_offset_repair.db"
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_file))
    monkeypatch.setattr("app.core.quota.DB_PATH", str(db_file), raising=False)
    db_mod.init_db()
    from app.services.bazarr_coordinator import bazarr_coordinator
    bazarr_coordinator.reset()
    yield
    bazarr_coordinator.reset()


def _make_cues(specs):
    """Helper to create list of Subtitle cues from [(start_sec, end_sec, text), ...]"""
    return [
        srt.Subtitle(
            index=i + 1,
            start=datetime.timedelta(seconds=s),
            end=datetime.timedelta(seconds=e),
            content=txt,
        )
        for i, (s, e, txt) in enumerate(specs)
    ]


def _make_srt(specs):
    return srt.compose(_make_cues(specs))


def _build_black_sails_fixture():
    """
    Deterministic forensic reproduction of Black Sails S03E04 scenario:
    - 75 English reference cues across 2400s (dialogue + SDH sound effects)
    - 52 Swedish target cues across 2400s (consolidated dialogue, pure speech)
    - Delayed by constant +20.45s offset
    - Raw unshifted alignment classifies as SyncErrorType.LOW_COVERAGE with raw coverage ~45-55%
    - Safe global offset repair estimates approximately +20.45s
    - Shifted candidate passes Trust with ~88-95% score, 0ms drift, and 0 AI calls
    """
    import random
    ref_specs = []

    # 0 to 188.4s: Prologue in English reference
    ref_specs.append((10.0, 14.0, "[WAVES CRASHING ON THE REEF]"))
    ref_specs.append((25.0, 30.0, "Flint: Look to the horizon and tell me what you see."))
    ref_specs.append((50.0, 54.0, "[THUNDER RUMBLES]"))
    ref_specs.append((85.0, 91.0, "Silver: They are coming for Nassau."))
    ref_specs.append((120.0, 124.0, "[CANNON FIRE]"))
    ref_specs.append((145.0, 151.0, "Flint: All hands to battle stations!"))
    ref_specs.append((175.0, 180.0, "Silver: Fire on my mark!"))
    ref_specs.append((188.4, 192.0, "[PIRATES CHEERING]"))

    # Swedish prologue dialogue:
    sv_cues = [
        (25.0, 30.0, "Flint: Titta mot horisonten och säg vad du ser."),
        (85.0, 91.0, "Silver: De är på väg mot Nassau."),
        (145.0, 151.0, "Flint: Alla man på sina poster!"),
        (175.0, 180.0, "Silver: Skjut på min signal!"),
    ]

    # Main episode (195s to 2400s): 48 dialogue scenes
    rng = random.Random(42)
    t = 195.0
    for i in range(48):
        dur = rng.uniform(3.5, 4.8)
        gap = 16.0 if i % 2 == 0 else 34.0

        # English reference dialogue line
        en_text = f"Captain Flint main dialogue line {i}."
        ref_specs.append((t, t + dur, en_text))

        # SDH in English reference (~7% SDH)
        if i in (6, 14, 22, 30, 38, 46):
            ref_specs.append((t + dur + 0.8, t + dur + 3.0, f"[SOUND EFFECT {i}]"))

        # Swedish target dialogue (pure dialogue, no SDH)
        sv_text = f"Kapten Flint svensk replik {i} i Nassau."
        sv_cues.append((t, t + dur, sv_text))

        t += dur + gap + (3.0 if i in (6, 14, 22, 30, 38, 46) else 0.0)

    # Delay all Swedish candidate cues by constant +20.45s offset
    target_specs = [(s + 20.45, e + 20.45, txt) for s, e, txt in sv_cues]

    ref_cues = _make_cues(ref_specs)
    tgt_cues = _make_cues(target_specs)

    return ref_specs, target_specs, ref_cues, tgt_cues


# ==============================================================================
# A. Black-Sails-shaped case:
#    complete target, constant +20.45s offset
#    -> repair succeeds
#    -> final Trust passes
#    -> candidate text unchanged
#    -> AI not required (AI calls: 0)
# ==============================================================================
@pytest.mark.asyncio
async def test_case_a_black_sails_global_offset_repair_succeeds(tmp_path):
    """
    Case A: Black-Sails-shaped production forensic scenario.
    Raw Swedish Bazarr candidate has constant +20.45s offset.
    - Initial un-shifted alignment: low coverage / large initial gap -> LOW_COVERAGE
    - Safe global offset repair estimates +20.45s
    - Repair shifts timestamps by -20.45s
    - Revalidation receives Trust PASS
    - File on disk is updated with exact shifted timestamps and 100% identical text
    - AI calls = 0
    """
    engine = SubtitleTrustEngine()
    video_path = str(tmp_path / "Black.Sails.S03E04.1080p.mkv")
    Path(video_path).touch()

    ref_specs, target_specs, ref_cues, tgt_cues = _build_black_sails_fixture()

    # Verify raw un-shifted alignment classifies as LOW_COVERAGE
    raw_align = align_subtitle_timelines(tgt_cues, ref_cues)
    assert raw_align.sync_error_type == SyncErrorType.LOW_COVERAGE
    assert raw_align.ref_coverage < 0.70

    ref_content = srt.compose(ref_cues)
    ref_source = SubtitleSource(
        origin=SourceOrigin.EMBEDDED,
        path=str(tmp_path / "Black.Sails.S03E04.en.srt"),
        language="en",
        content=ref_content,
        cues=ref_cues,
    )

    cand_file = tmp_path / "Black.Sails.S03E04.sv.srt"
    cand_file.write_text(srt.compose(tgt_cues), encoding="utf-8")
    cand_snap = capture_target_snapshot(str(cand_file))

    # Strong current-run Bazarr provenance
    prov = BazarrProvenance(
        video_path=video_path,
        target_lang="sv",
        search_accepted=True,
        pre_trigger_snapshot=TargetSnapshot(path="", exists=False, size=0, mtime_ns=0),
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
        auto_repair=True,
    )

    # Assertions
    assert res.passed is True
    assert res.decision == TrustDecision.PASS
    assert res.score >= 88
    assert res.repair is not None
    assert abs(res.repair["original_offset_sec"] - 20.45) < 0.1
    assert abs(res.repair["applied_shift_sec"] - (-20.45)) < 0.1
    assert res.ai_used is False
    assert res.ai_calls == 0

    disk_cues = list(srt.parse(cand_file.read_text(encoding="utf-8")))
    assert len(disk_cues) == len(target_specs)
    for i, cue in enumerate(disk_cues):
        assert cue.content == target_specs[i][2]
        expected_start = target_specs[i][0] - 20.45
        assert abs(cue.start.total_seconds() - expected_start) < 0.1


# ==============================================================================
# B. Progressive drift:
#    -> MUST NOT global-shift repair
# ==============================================================================
@pytest.mark.asyncio
async def test_case_b_progressive_drift_must_not_repair(tmp_path):
    """
    Case B: Candidate has progressive timing drift (e.g. 23.976 vs 25 FPS mismatch).
    -> Must NOT apply global offset repair.
    -> Must FAIL closed.
    """
    engine = SubtitleTrustEngine()
    video_path = str(tmp_path / "DriftMovie.mkv")
    Path(video_path).touch()

    ref_specs = [
        (10.0 + i * 10.0, 13.0 + i * 10.0, f"English dialogue line {i}")
        for i in range(60)
    ]
    ref_source = SubtitleSource(
        origin=SourceOrigin.EMBEDDED,
        path=str(tmp_path / "DriftMovie.en.srt"),
        language="en",
        content=_make_srt(ref_specs),
        cues=_make_cues(ref_specs),
    )

    # Progressive drift stretching by 3.5s across the 10 minutes
    cand_specs = [
        (10.0 + i * 10.0 + (i / 60.0) * 3.5, 13.0 + i * 10.0 + (i / 60.0) * 3.5, f"Svensk replik nummer {i}")
        for i in range(60)
    ]
    cand_file = tmp_path / "DriftMovie.sv.srt"
    cand_file.write_text(_make_srt(cand_specs), encoding="utf-8")
    cand_snap = capture_target_snapshot(str(cand_file))

    prov = BazarrProvenance(
        video_path=video_path,
        target_lang="sv",
        search_accepted=True,
        pre_trigger_snapshot=TargetSnapshot(path="", exists=False, size=0, mtime_ns=0),
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
        auto_repair=True,
    )

    assert res.passed is False
    assert res.decision == TrustDecision.FAIL
    assert res.repair is None
    assert any("drift" in r.lower() or "fps" in r.lower() or "timing" in r.lower() for r in res.reasons)


# ==============================================================================
# C. Real missing 3-minute section / partial subtitle:
#    -> MUST NOT repair
# ==============================================================================
@pytest.mark.asyncio
async def test_case_c_missing_section_partial_must_not_repair(tmp_path):
    """
    Case C: Candidate has a 3-minute missing dialogue section (uncovered span > 180s).
    -> Must NOT apply global offset repair.
    -> Must FAIL closed.
    """
    engine = SubtitleTrustEngine()
    video_path = str(tmp_path / "MissingSection.mkv")
    Path(video_path).touch()

    ref_specs = [
        (10.0 + i * 5.0, 13.0 + i * 5.0, f"English dialogue line {i}")
        for i in range(100)
    ]
    ref_source = SubtitleSource(
        origin=SourceOrigin.EMBEDDED,
        path=str(tmp_path / "MissingSection.en.srt"),
        language="en",
        content=_make_srt(ref_specs),
        cues=_make_cues(ref_specs),
    )

    # Missing dialogue between t=100s and t=350s (250s missing section)
    cand_specs = [
        (10.0 + i * 5.0, 13.0 + i * 5.0, f"Svensk replik nummer {i}")
        for i in range(100)
        if (10.0 + i * 5.0 < 100.0 or 10.0 + i * 5.0 > 350.0)
    ]
    cand_file = tmp_path / "MissingSection.sv.srt"
    cand_file.write_text(_make_srt(cand_specs), encoding="utf-8")
    cand_snap = capture_target_snapshot(str(cand_file))

    prov = BazarrProvenance(
        video_path=video_path,
        target_lang="sv",
        search_accepted=True,
        pre_trigger_snapshot=TargetSnapshot(path="", exists=False, size=0, mtime_ns=0),
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
        auto_repair=True,
    )

    assert res.passed is False
    assert res.decision == TrustDecision.FAIL
    assert res.repair is None
    assert any("coverage" in r.lower() or "missing" in r.lower() or "gap" in r.lower() for r in res.reasons)


# ==============================================================================
# D. Sudden discontinuity / different cut:
#    -> MUST NOT repair
# ==============================================================================
@pytest.mark.asyncio
async def test_case_d_sudden_discontinuity_different_cut_must_not_repair(tmp_path):
    """
    Case D: First 20 cues in sync (0s offset), cues 20-40 jump by +2.5s (different cut / scene inserted).
    -> Must NOT apply single global offset repair.
    -> Must FAIL closed.
    """
    engine = SubtitleTrustEngine()
    video_path = str(tmp_path / "DifferentCut.mkv")
    Path(video_path).touch()

    ref_specs = [
        (10.0 + i * 15.0, 13.0 + i * 15.0, f"English dialogue line {i}")
        for i in range(40)
    ]
    ref_source = SubtitleSource(
        origin=SourceOrigin.EMBEDDED,
        path=str(tmp_path / "DifferentCut.en.srt"),
        language="en",
        content=_make_srt(ref_specs),
        cues=_make_cues(ref_specs),
    )

    cand_specs = []
    for i in range(40):
        shift = 0.0 if i < 20 else 2.5
        cand_specs.append((10.0 + i * 15.0 + shift, 13.0 + i * 15.0 + shift, f"Svensk replik {i}"))

    cand_file = tmp_path / "DifferentCut.sv.srt"
    cand_file.write_text(_make_srt(cand_specs), encoding="utf-8")
    cand_snap = capture_target_snapshot(str(cand_file))

    prov = BazarrProvenance(
        video_path=video_path,
        target_lang="sv",
        search_accepted=True,
        pre_trigger_snapshot=TargetSnapshot(path="", exists=False, size=0, mtime_ns=0),
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
        auto_repair=True,
    )

    assert res.passed is False
    assert res.decision == TrustDecision.FAIL
    assert res.repair is None
    assert any("discontinuity" in r.lower() or "mismatch" in r.lower() or "cut" in r.lower() or "timing" in r.lower() for r in res.reasons)


# ==============================================================================
# E. Already good Bazarr subtitle:
#    -> no repair attempted
# ==============================================================================
@pytest.mark.asyncio
async def test_case_e_already_good_bazarr_no_repair_attempted(tmp_path):
    """
    Case E: Subtitle is already in perfect sync (< 0.05s variance).
    -> Must PASS directly on first evaluation.
    -> res.repair is None (no repair attempted or needed).
    """
    engine = SubtitleTrustEngine()
    video_path = str(tmp_path / "GoodSync.mkv")
    Path(video_path).touch()

    ref_specs = [
        (10.0 + i * 10.0, 13.0 + i * 10.0, f"English dialogue line {i}")
        for i in range(30)
    ]
    ref_source = SubtitleSource(
        origin=SourceOrigin.EMBEDDED,
        path=str(tmp_path / "GoodSync.en.srt"),
        language="en",
        content=_make_srt(ref_specs),
        cues=_make_cues(ref_specs),
    )

    cand_specs = [
        (10.02 + i * 10.0, 13.02 + i * 10.0, f"Svensk replik {i}")
        for i in range(30)
    ]
    cand_file = tmp_path / "GoodSync.sv.srt"
    cand_file.write_text(_make_srt(cand_specs), encoding="utf-8")
    cand_snap = capture_target_snapshot(str(cand_file))

    prov = BazarrProvenance(
        video_path=video_path,
        target_lang="sv",
        search_accepted=True,
        pre_trigger_snapshot=TargetSnapshot(path="", exists=False, size=0, mtime_ns=0),
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
        auto_repair=True,
    )

    assert res.passed is True
    assert res.decision == TrustDecision.PASS
    assert res.score >= 90
    assert res.repair is None  # No repair was needed or applied
    assert res.ai_used is False
    assert res.ai_calls == 0


# ==============================================================================
# Pipeline Integration: Black Sails Scenario with AI Skipped (AI Calls: 0)
# ==============================================================================
@pytest.mark.asyncio
async def test_pipeline_integration_black_sails_repaired_and_ai_skipped(tmp_path, monkeypatch):
    """
    Full pipeline test:
    Bazarr target with +20.45s offset arrives.
    SubtitlePipeline coordinates and evaluates candidate.
    Safe global offset repair shifts candidate.
    Pipeline accepts repaired target as BAZARR MATCH.
    AI translator is NOT invoked (AI calls = 0).
    """
    settings = {
        "ai_provider": "gemini",
        "gemini_api_key": "dummy_key",
        "gemini_model": "gemini-3.5-flash-lite",
        "enable_bazarr_check": "true",
        "enable_bazarr": "true",
        "bazarr_url": "http://bazarr:6767",
        "bazarr_api_key": "test_key",
        "languages": '[{"name": "Swedish", "code": "sv", "enabled": true}]',
        "bazarr_quiescence_seconds": "0.1",
        "bazarr_candidate_stability_seconds": "0.03",
        "bazarr_grace_seconds": "1.0",
    }
    monkeypatch.setattr("app.core.db.get_setting", lambda k, d=None: settings.get(k, d))
    monkeypatch.setattr("app.services.pipeline.get_setting", lambda k, d=None: settings.get(k, d))
    monkeypatch.setattr("app.services.bazarr_coordinator.get_setting", lambda k, d=None: settings.get(k, d))

    video_path = tmp_path / "Black.Sails.S03E04.mkv"
    video_path.touch()

    ref_specs, target_specs, ref_cues, tgt_cues = _build_black_sails_fixture()

    en_srt = tmp_path / "Black.Sails.S03E04.en.srt"
    en_srt.write_text(srt.compose(ref_cues), encoding="utf-8")

    sv_srt = tmp_path / "Black.Sails.S03E04.sv.srt"

    pipeline = SubtitlePipeline()
    ai_called = False

    async def mock_translate_batch(batch, **kwargs):
        nonlocal ai_called
        ai_called = True
        return [{"id": item["id"], "text": "AI"} for item in batch]

    from app.services.source_resolver import BazarrResult, BazarrResultCode

    async def mock_bazarr_search(*args, **kwargs):
        sv_srt.write_text(srt.compose(tgt_cues), encoding="utf-8")
        return BazarrResult(code=BazarrResultCode.TRIGGERED, detail="Accepted", media_correlated=True)

    monkeypatch.setattr(pipeline, "trigger_bazarr_search", mock_bazarr_search)

    def mock_find(vp, lang):
        if lang == "en": return str(en_srt)
        if lang == "sv" and os.path.exists(str(sv_srt)): return str(sv_srt)
        return None

    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find), \
         patch.object(BazarrCoordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_jobs, \
         patch.object(BazarrCoordinator, "correlate_media", new_callable=AsyncMock) as mock_corr:
        mock_jobs.return_value = BazarrJobsPollResult(status=BazarrJobPollStatus.KNOWN_IDLE, jobs=[])
        mock_corr.return_value = MagicMock(is_indexed=True, radarr_id=None, sonarr_episode_id=404)

        res = await pipeline.process_video_file(str(video_path), event_source="SONARR")

    assert res["status"] in ("skipped", "completed", "bazarr_downloaded")
    assert res["reason"] in ("bazarr_downloaded", "already_exists")
    assert ai_called is False  # AI translation was bypassed

    # Verify candidate file on disk was shifted and text preserved
    repaired_cues = list(srt.parse(sv_srt.read_text(encoding="utf-8")))
    assert len(repaired_cues) == len(target_specs)
    expected_first_start = target_specs[0][0] - 20.45
    assert abs(repaired_cues[0].start.total_seconds() - expected_first_start) < 0.1
    for i, c in enumerate(repaired_cues):
        assert c.content == target_specs[i][2]
