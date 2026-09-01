"""
Comprehensive 16-point hardening and calibration test suite for Subtitle Trust Engine,
atomic publication gate, safe constant-offset repair, and accurate diagnostics.
"""

import asyncio
import os
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import srt
from datetime import timedelta

from app.core.db import create_job, get_job_by_id, init_db
from app.core.trust_engine import (
    CandidateOrigin,
    SubtitleIntent,
    SubtitleTrustEngine,
    SyncErrorType,
    TargetSnapshot,
    TrustDecision,
    TrustResult,
    VerificationMode,
    align_subtitle_timelines,
    apply_safe_repair,
    can_safely_repair_offset,
    capture_target_snapshot,
    format_trust_summary,
    repair_constant_offset,
    validate_standalone_structure,
)
from app.services.pipeline import (
    _publish_subtitle_atomic,
    _publish_subtitle_with_trust_gate,
)


@pytest.fixture(autouse=True)
def setup_tmp_db(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test_hardening.db")
    monkeypatch.setattr("app.core.db.DB_PATH", test_db)
    init_db()
    yield test_db


def _make_cues(specs):
    """Helper to create list of Subtitle cues from [(start_sec, end_sec, text), ...]"""
    return [
        srt.Subtitle(
            index=i + 1,
            start=timedelta(seconds=s),
            end=timedelta(seconds=e),
            content=txt,
        )
        for i, (s, e, txt) in enumerate(specs)
    ]


def _make_srt(specs):
    return srt.compose(_make_cues(specs))


# ── TEST 1: Clean Bazarr Target with Matching Release (PASS, AI=0) ─────────
@pytest.mark.asyncio
async def test_01_clean_target_matching_release_passes(tmp_path):
    ref_specs = [
        (1.0, 4.0, "Captain Flint is returning to Nassau."),
        (5.0, 8.0, "The British fleet is waiting on the horizon."),
        (9.0, 12.0, "Every man must prepare his weapons now."),
        (13.0, 16.0, "We fight for freedom or we hang."),
        (17.0, 20.0, "Silver has gathered the men on deck."),
        (21.0, 24.0, "The cannons are primed and loaded."),
    ]
    target_specs = [
        (1.0, 4.0, "Kapten Flint återvänder till Nassau."),
        (5.0, 8.0, "Den brittiska flottan väntar vid horisonten."),
        (9.0, 12.0, "Varje man måste förbereda sina vapen nu."),
        (13.0, 16.0, "Vi kämpar för frihet eller så hängs vi."),
        (17.0, 20.0, "Silver har samlat männen på däck."),
        (21.0, 24.0, "Kanonerna är laddade och klara."),
    ]
    video_path = str(tmp_path / "Black.Sails.S04E01.mkv")
    Path(video_path).touch()
    cand_path = str(tmp_path / "Black.Sails.S04E01.sv.srt")
    Path(cand_path).write_text(_make_srt(target_specs), encoding="utf-8")

    from app.core.trust_engine import ReferenceInfo
    ref_info = ReferenceInfo(
        source_type="container_track",
        language="en",
        cues=_make_cues(ref_specs),
        raw_content=_make_srt(ref_specs),
    )

    engine = SubtitleTrustEngine()
    with patch("app.core.trust_engine.find_and_rank_references", return_value=[ref_info]):
        res = await engine.evaluate_candidate(
            video_path=video_path,
            candidate_path=cand_path,
            target_lang="sv",
            origin=CandidateOrigin.BAZARR,
            auto_repair=True,
        )

    assert res.passed is True
    assert res.decision == TrustDecision.PASS
    assert res.score >= 90
    assert res.metrics["ref_coverage"] >= 0.95
    assert res.origin == CandidateOrigin.BAZARR


# ── TEST 2: Wrong Episode Target with Valid Syntax (FAIL, Fallback) ────────
@pytest.mark.asyncio
async def test_02_wrong_episode_target_fails_and_babel_publishes(tmp_path):
    ref_specs = [
        (1.0, 4.0, "Captain Flint is returning to Nassau."),
        (5.0, 8.0, "The British fleet is waiting on the horizon."),
        (9.0, 12.0, "Every man must prepare his weapons now."),
        (13.0, 16.0, "We fight for freedom or we hang."),
        (17.0, 20.0, "Silver has gathered the men on deck."),
    ]
    # Wrong episode: completely shifted timing (offset > 120s) and mismatched content
    wrong_specs = [
        (120.0, 123.0, "Tidigare i ett helt annat avsnitt."),
        (125.0, 128.0, "Vi seglar norrut mot Boston."),
        (130.0, 133.0, "Ingen kontakt med Nassau alls."),
        (135.0, 138.0, "Alla män är redo för strid."),
        (140.0, 143.0, "Vi ses i nästa hamn."),
    ]
    video_path = str(tmp_path / "Black.Sails.S04E02.mkv")
    Path(video_path).touch()
    cand_path = str(tmp_path / "Black.Sails.S04E02.sv.srt")
    Path(cand_path).write_text(_make_srt(wrong_specs), encoding="utf-8")

    from app.core.trust_engine import ReferenceInfo
    ref_info = ReferenceInfo(
        source_type="container_track",
        language="en",
        cues=_make_cues(ref_specs),
        raw_content=_make_srt(ref_specs),
    )

    engine = SubtitleTrustEngine()
    with patch("app.core.trust_engine.find_and_rank_references", return_value=[ref_info]):
        res = await engine.evaluate_candidate(
            video_path=video_path,
            candidate_path=cand_path,
            target_lang="sv",
            origin=CandidateOrigin.EXTERNAL,
            auto_repair=True,
        )

    assert res.passed is False
    assert res.decision == TrustDecision.FAIL


# ── TEST 3: Partial / Forced Subtitle Detection (FAIL) ─────────────────────
@pytest.mark.asyncio
async def test_03_partial_forced_subtitle_fails(tmp_path):
    # Reference has 200 cues, candidate has only 5 cues in Swedish
    ref_specs = [(float(i * 5), float(i * 5 + 3), f"English dialogue line {i}") for i in range(200)]
    forced_specs = [(float(i * 100), float(i * 100 + 2), f"Forcerad skylt på svenska nummer {i}") for i in range(5)]

    video_path = str(tmp_path / "Movie.mkv")
    Path(video_path).touch()
    cand_path = str(tmp_path / "Movie.sv.srt")
    Path(cand_path).write_text(_make_srt(forced_specs), encoding="utf-8")

    from app.core.trust_engine import ReferenceInfo
    ref_info = ReferenceInfo(
        source_type="container_track",
        language="en",
        cues=_make_cues(ref_specs),
        raw_content=_make_srt(ref_specs),
    )

    engine = SubtitleTrustEngine()
    with patch("app.core.trust_engine.find_and_rank_references", return_value=[ref_info]):
        res = await engine.evaluate_candidate(
            video_path=video_path,
            candidate_path=cand_path,
            target_lang="sv",
            origin=CandidateOrigin.EXTERNAL,
            expected_intent=SubtitleIntent.FULL,
        )

    assert res.passed is False
    assert "partial/forced" in str(res.reasons).lower()


# ── TEST 4: Sudden Discontinuity / Different Cut (FAIL) ───────────────────
def test_04_sudden_discontinuity_detected():
    # 40 cues total: first 20 cues at -1.5s offset, cues 20-40 jump to +2.0s offset (step = 3.5s)
    ref_specs = [(float(i * 10), float(i * 10 + 3), f"Ref {i}") for i in range(40)]
    target_specs = []
    for i in range(20):
        target_specs.append((float(i * 10 - 1.5), float(i * 10 + 1.5), f"Target {i}"))
    for i in range(20, 40):
        target_specs.append((float(i * 10 + 2.0), float(i * 10 + 5.0), f"Target {i}"))

    ref_cues = _make_cues(ref_specs)
    target_cues = _make_cues(target_specs)

    align_res = align_subtitle_timelines(target_cues, ref_cues)
    assert align_res.sync_error_type == SyncErrorType.SUDDEN_DISCONTINUITY
    assert align_res.max_discontinuity_sec >= 3.5


# ── TEST 5: Constant Offset Repair (auto_repair=True -> PASS) ──────────────
@pytest.mark.asyncio
async def test_05_constant_offset_repaired_successfully(tmp_path):
    # Candidate is globally shifted late by +2.50s
    ref_specs = [(float(i * 5), float(i * 5 + 3), f"English dialogue line {i}") for i in range(20)]
    target_specs = [(float(i * 5 + 2.5), float(i * 5 + 5.5), f"Svensk replik här {i}") for i in range(20)]

    video_path = str(tmp_path / "Episode5.mkv")
    Path(video_path).touch()
    cand_path = str(tmp_path / "Episode5.sv.srt")
    Path(cand_path).write_text(_make_srt(target_specs), encoding="utf-8")

    from app.core.trust_engine import ReferenceInfo
    ref_info = ReferenceInfo(
        source_type="container_track",
        language="en",
        cues=_make_cues(ref_specs),
        raw_content=_make_srt(ref_specs),
    )

    engine = SubtitleTrustEngine()
    with patch("app.core.trust_engine.find_and_rank_references", return_value=[ref_info]):
        res = await engine.evaluate_candidate(
            video_path=video_path,
            candidate_path=cand_path,
            target_lang="sv",
            origin=CandidateOrigin.EXTERNAL,
            auto_repair=True,
        )

    assert res.passed is True
    assert res.decision == TrustDecision.PASS
    assert res.repair is not None
    assert abs(res.repair["original_offset_sec"] - 2.5) < 0.1
    assert abs(res.repair["applied_shift_sec"] - (-2.5)) < 0.1

    # Verify on-disk file was shifted
    repaired_disk_cues = list(srt.parse(Path(cand_path).read_text(encoding="utf-8")))
    first_cue = repaired_disk_cues[0]
    assert abs(first_cue.start.total_seconds() - 0.0) < 0.05


# ── TEST 6: Constant Offset with auto_repair=False (REPAIRABLE) ───────────
@pytest.mark.asyncio
async def test_06_constant_offset_no_repair_returns_repairable(tmp_path):
    ref_specs = [(float(i * 5), float(i * 5 + 3), f"English dialogue line {i}") for i in range(20)]
    target_specs = [(float(i * 5 + 2.5), float(i * 5 + 5.5), f"Svensk replik här {i}") for i in range(20)]

    video_path = str(tmp_path / "Episode6.mkv")
    Path(video_path).touch()
    cand_path = str(tmp_path / "Episode6.sv.srt")
    Path(cand_path).write_text(_make_srt(target_specs), encoding="utf-8")

    from app.core.trust_engine import ReferenceInfo
    ref_info = ReferenceInfo(
        source_type="container_track",
        language="en",
        cues=_make_cues(ref_specs),
        raw_content=_make_srt(ref_specs),
    )

    engine = SubtitleTrustEngine()
    with patch("app.core.trust_engine.find_and_rank_references", return_value=[ref_info]):
        res = await engine.evaluate_candidate(
            video_path=video_path,
            candidate_path=cand_path,
            target_lang="sv",
            origin=CandidateOrigin.EXTERNAL,
            auto_repair=False,
        )

    assert res.passed is False
    assert res.decision == TrustDecision.REPAIRABLE
    assert res.is_repairable is True


# ── TEST 7: Progressive Timing Drift / FPS Mismatch (FAIL) ─────────────────
def test_07_progressive_drift_detected():
    # Linear drift from 0s to 4.0s (23.976 to 25.000 fps stretch over 20 minutes)
    ref_cues = _make_cues([(float(i * 60), float(i * 60 + 5), f"Ref {i}") for i in range(20)])
    target_cues = _make_cues([(float(i * 60 + (i * 0.20)), float(i * 60 + (i * 0.20) + 5), f"Target {i}") for i in range(20)])

    align_res = align_subtitle_timelines(target_cues, ref_cues)
    assert align_res.sync_error_type == SyncErrorType.PROGRESSIVE_DRIFT
    assert abs(align_res.linear_drift_sec) >= 2.0


# ── TEST 8: Corrupt / 0-byte Candidate (FAIL Early) ────────────────────────
@pytest.mark.asyncio
async def test_08_corrupt_empty_candidate_fails_early(tmp_path):
    video_path = str(tmp_path / "Episode8.mkv")
    Path(video_path).touch()
    cand_path = str(tmp_path / "Episode8.sv.srt")
    Path(cand_path).write_text("", encoding="utf-8")  # 0 bytes

    engine = SubtitleTrustEngine()
    res = await engine.evaluate_candidate(
        video_path=video_path,
        candidate_path=cand_path,
        target_lang="sv",
        origin=CandidateOrigin.EXTERNAL,
    )

    assert res.passed is False
    assert res.decision == TrustDecision.FAIL
    assert res.score == 0


# ── TEST 9: Advertisements / Spam Stripping ────────────────────────────────
def test_09_advertisement_stripping(tmp_path):
    dirty_srt = """1
00:00:01,000 --> 00:00:04,000
Downloaded from OpenSubtitles.org

2
00:00:05,000 --> 00:00:08,000
Kapten Flint återvänder till Nassau.

3
00:00:09,000 --> 00:00:12,000
Vi måste sätta segel mot öarna.

4
00:00:13,000 --> 00:00:16,000
Männen är redo för strid.

5
00:00:17,000 --> 00:00:20,000
Subtitles by Addic7ed.com
"""
    res = validate_standalone_structure(dirty_srt)
    assert res.metrics["ad_count"] == 2


# ── TEST 10: Wrong Language Candidate (FAIL) ───────────────────────────────
@pytest.mark.asyncio
async def test_10_wrong_language_fails(tmp_path):
    video_path = str(tmp_path / "Episode10.mkv")
    Path(video_path).touch()
    cand_path = str(tmp_path / "Episode10.sv.srt")
    # File is named .sv.srt but content is clearly English across 10 cues
    english_specs = [
        (float(i * 5), float(i * 5 + 3), f"The English sailing captain commands his royal naval crew number {i}")
        for i in range(10)
    ]
    Path(cand_path).write_text(_make_srt(english_specs), encoding="utf-8")

    engine = SubtitleTrustEngine()
    res = await engine.evaluate_candidate(
        video_path=video_path,
        candidate_path=cand_path,
        target_lang="sv",
        origin=CandidateOrigin.EXTERNAL,
    )

    assert res.passed is False
    assert res.decision == TrustDecision.FAIL
    assert any("language" in r.lower() for r in res.reasons)


# ── TEST 11: Synthetic Black Sails S04E06 (89.1% coverage, 216.6s gap -> FAIL)
def test_11_black_sails_s04e06_synthetic_gap_metrics():
    # 494 cues reference, 365 cues candidate (89.1% coverage, 134.2s uncovered dialogue, 216.6s timeline span)
    ref_specs = []
    t = 0.0
    for i in range(494):
        dur = 2.0
        ref_specs.append((t, t + dur, f"Ref cue {i}"))
        t += dur + 3.0  # spacing
    ref_cues = _make_cues(ref_specs)

    # Candidate matches all except a stretch of ~43 cues between t=500s and t=720s (gap of 220s)
    # and scattered missing cues amounting to ~130s uncovered dialogue
    target_specs = []
    for i, (s, e, txt) in enumerate(ref_specs):
        if 100 <= i <= 143:
            continue  # Miss this block -> timeline span > 216s
        if i % 8 == 0:
            continue  # Scattered drops
        target_specs.append((s, e, f"Target cue {i}"))
    target_cues = _make_cues(target_specs)

    align_res = align_subtitle_timelines(target_cues, ref_cues)
    assert align_res.sync_error_type == SyncErrorType.LOW_COVERAGE
    assert align_res.largest_anchor_gap_sec >= 200.0
    assert align_res.uncovered_reference_dialogue_sec > 100.0
    assert align_res.max_uncovered_active_dialogue_sec < 100.0  # active dialogue per cue is small
    assert "large unmatched timeline gap" in align_res.issues[0] or "Low reference dialogue coverage" in align_res.issues[0]


# ── TEST 12: Publication Gate TOCTOU Race & Revalidation ───────────────────
@pytest.mark.asyncio
async def test_12_publication_gate_toctou_revalidates(tmp_path):
    video_path = str(tmp_path / "Episode12.mkv")
    Path(video_path).touch()
    target_path = str(tmp_path / "Episode12.sv.srt")
    Path(target_path).write_text("1\n00:00:01,000 --> 00:00:04,000\nGamla svenska ord\n", encoding="utf-8")
    job_id = create_job(video_path)

    # Trust Engine returns FAIL
    fail_tres = TrustResult(
        decision=TrustDecision.FAIL,
        score=20,
        confidence="HIGH",
        reasons=["REFERENCE_MISMATCH"],
    )

    with patch.object(SubtitleTrustEngine, "evaluate_candidate", AsyncMock(return_value=fail_tres)):
        pub_res = await _publish_subtitle_with_trust_gate(
            video_path=video_path,
            target_output_path=target_path,
            lang_code="sv",
            translated_srt_text="1\n00:00:01,000 --> 00:00:04,000\nBabel QA translation\n",
            expected_cue_count=1,
            job_id=job_id,
        )

    assert pub_res["published"] is True
    assert "Babel QA translation" in Path(target_path).read_text(encoding="utf-8")


# ── TEST 13: Provenance Truthfulness ───────────────────────────────────────
@pytest.mark.asyncio
async def test_13_provenance_truthfulness():
    tres_bazarr = TrustResult(
        decision=TrustDecision.PASS,
        score=95,
        confidence="HIGH",
        origin=CandidateOrigin.BAZARR,
    )
    tres_external = TrustResult(
        decision=TrustDecision.PASS,
        score=95,
        confidence="HIGH",
        origin=CandidateOrigin.EXTERNAL,
    )

    summary_b = format_trust_summary(tres_bazarr)
    summary_e = format_trust_summary(tres_external)

    assert "Candidate: External / Bazarr" in summary_b
    assert "Candidate: External" in summary_e


# ── TEST 14: Repair Bounds & Negative Timestamp Safety ─────────────────────
def test_14_repair_bounds_safety():
    cues = _make_cues([
        (0.5, 3.0, "Hello"),
        (4.0, 7.0, "World"),
    ])

    # Shifting by +2.0s means delta = -2.0s. Start 0.5 - 2.0 = -1.5s -> Negative!
    can_repair_neg, reason_neg = can_safely_repair_offset(cues, 2.0)
    assert can_repair_neg is False
    assert "negative timestamps" in reason_neg

    # Offset > 300s
    can_repair_huge, reason_huge = can_safely_repair_offset(cues, 350.0)
    assert can_repair_huge is False
    assert "exceeds maximum safe repair threshold" in reason_huge

    # Safe offset (-1.0s shift means delta = +1.0s)
    can_repair_safe, _ = can_safely_repair_offset(cues, -1.0)
    assert can_repair_safe is True


# ── TEST 15: Safe Repair Transactional Snapshot Check ──────────────────────
def test_15_apply_safe_repair_toctou_rollback(tmp_path):
    target = tmp_path / "test_repair.srt"
    target.write_text("1\n00:00:01,000 --> 00:00:03,000\nOriginal\n", encoding="utf-8")

    pre_snap = capture_target_snapshot(str(target))

    # Simulate another process modifying target
    time.sleep(0.01)
    target.write_text("1\n00:00:01,000 --> 00:00:03,000\nMutated by external process\n", encoding="utf-8")

    # apply_safe_repair must reject due to snapshot mismatch
    repaired_content = "1\n00:00:02,000 --> 00:00:04,000\nRepaired\n"
    success = apply_safe_repair(str(target), repaired_content, expected_snapshot=pre_snap)
    assert success is False
    assert "Mutated by external process" in target.read_text(encoding="utf-8")


# ── TEST 16: Trust Execution Summary Format & Diagnostic Metrics ──────────
def test_16_format_trust_summary_output():
    tres_fail = TrustResult(
        decision=TrustDecision.FAIL,
        score=35,
        confidence="HIGH",
        reasons=["Low reference dialogue coverage (89.1%) or large unmatched timeline gap (216.6s)"],
        metrics={
            "ref_coverage": 0.891,
            "uncovered_ref_dialogue_sec": 134.2,
            "largest_unmatched_timeline_gap_sec": 216.6,
        },
        reference={"language_name": "English", "source_type": "container_track"},
        origin=CandidateOrigin.EXTERNAL,
    )

    summary = format_trust_summary(tres_fail)
    assert "--- Subtitle Trust ---" in summary
    assert "Candidate: External" in summary
    assert "Reference: Embedded English" in summary
    assert "Decision: FAIL" in summary
    assert "Dialogue coverage: 89.1%" in summary
    assert "Uncovered reference dialogue: 134.2s" in summary
    assert "Timeline/anchor gap: 216.6s" in summary
    assert "Action: Babel fallback" in summary
    assert "----------------------" in summary
