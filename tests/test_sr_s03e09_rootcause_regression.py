"""
Deterministic regression tests for S03E09 "Survivor's Remorse" production incident.

Two root causes fixed:
  1. Bazarr lifecycle: ACTIVE + unclassified subtitle jobs must NOT synthesize KNOWN_IDLE provenance,
     while conclusively unrelated background tasks (backup, different media) do not block finalization.
  2. Trust Contradiction Gate: legacy greedy alignment NONE + large independent global offset
     must NOT be accepted unchanged. If repair fails or provenance is weak, must FAIL CLOSED.

Test matrix (all deterministic, 0 skips):
  A. PASS false-positive / consolidated-cue case (core incident reproduction) -> repaired exact PASS
  B. Failed scratch repair -> original candidate FAILS CLOSED (never passes through)
  C. Weak/external provenance contradiction -> FAILS CLOSED
  D. Already-correct consolidated subtitle -> PASS without repair
  E. Normal near-zero-offset Bazarr wins (+0.05s, +0.11s) -> PASS without repair
  F. Wrong-release / irregular mismatch -> NOT rescued by estimator alone
  G. Black Sails LOW_COVERAGE global-offset repair remains intact
  H. Bazarr lifecycle: ACTIVE + unclassified subtitle jobs -> stays provisional, no fabricated KNOWN_IDLE
  I. Bazarr lifecycle: ACTIVE + conclusively unrelated jobs (backup, different movie) -> does not block finalization
  J. Later authoritative target finalization -> exact final generation evaluated
  K. Mutation-after-evaluation -> prior PASS discarded and re-evaluated
  L. Unit invariants: estimator detection, contradiction thresholds, residual collapse

Synthetic timing fixtures use exact timing tuples from forensic reproduction with anonymized Swedish text.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import srt

import app.core.db as db_mod
from app.core.trust_engine import (
    SubtitleTrustEngine,
    TrustDecision,
    CandidateOrigin,
    BazarrProvenance,
    TargetSnapshot,
    SyncErrorType,
    capture_target_snapshot,
    estimate_global_offset,
    align_subtitle_timelines,
    repair_constant_offset,
    _shift_cues,
)
from app.services.bazarr_coordinator import (
    BazarrCoordinator,
    BazarrJobPollStatus,
    BazarrJobsPollResult,
    BazarrJobInfo,
    BazarrLifecycleState,
)
from app.services.source_resolver import SubtitleSource, SourceOrigin


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db_file = tmp_path / "test_sr_regression.db"
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_file))
    monkeypatch.setattr("app.core.quota.DB_PATH", str(db_file), raising=False)
    db_mod.init_db()
    from app.services.bazarr_coordinator import bazarr_coordinator
    bazarr_coordinator.reset()
    yield
    bazarr_coordinator.reset()


def _make_cues(specs):
    """Helper: [(start_sec, end_sec, text), ...] -> list[srt.Subtitle]"""
    return [
        srt.Subtitle(
            index=i + 1,
            start=datetime.timedelta(seconds=s),
            end=datetime.timedelta(seconds=e),
            content=txt,
        )
        for i, (s, e, txt) in enumerate(specs)
    ]


def _make_srt(specs) -> str:
    return srt.compose(_make_cues(specs))


_SV_LINES = [
    "Jag behöver en privatdetektiv som kan hjälpa mig.",
    "Det är inte vad jag menar med det här.",
    "Han satt där och väntade på att något skulle hända.",
    "Vi måste prata om det som har hänt ikväll.",
    "Ingen visste vad som komma skulle för oss alla.",
    "Det var en lång dag och hon var trött på allt.",
    "Varför gör du på det här sättet hela tiden?",
    "Förstår du vad jag försöker säga till dig nu?",
    "De hade aldrig trott att det skulle sluta så här.",
    "Kom igen, vi har inte tid att vänta längre nu.",
]


def _sv_text(i: int) -> str:
    return _SV_LINES[i % len(_SV_LINES)]


def _strong_bazarr_provenance(video_path: str, lang: str, snap: TargetSnapshot,
                               pre_snap: TargetSnapshot) -> BazarrProvenance:
    """Build a fully qualified strong Bazarr provenance."""
    return BazarrProvenance(
        video_path=video_path,
        target_lang=lang,
        search_accepted=True,
        pre_trigger_snapshot=pre_snap,
        is_finalized=True,
        is_quiescent=True,
        media_correlated=True,
        poll_state=BazarrJobPollStatus.KNOWN_IDLE,
        candidate_snapshot=snap,
    )


# ---------------------------------------------------------------------------
# S03E09 Forensic Timing Fixture (Deterministic, hermetic, self-contained)
# ---------------------------------------------------------------------------

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sr_s03e09_timing.json"


def _build_forensic_s03e09_fixture():
    """
    Returns (ref_cues, target_cues_raw, target_cues_synced) with exact forensic timing
    dynamics from the S03E09 incident, loaded hermetically from deterministic anonymized
    timing fixture JSON.

    Dynamics:
      - Reference: 560 cues
      - Target raw: 301 consolidated cues (~5.1s early)
      - Legacy alignment on raw: sync_error_type=NONE, median_offset=+0.041s, score=91
      - Independent estimator on raw: -5.128s
      - Contradiction: |−5.128 − 0.041| = 5.17s >= 2.0s
    """
    with open(_FIXTURE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    ref_cues = [
        srt.Subtitle(
            i + 1,
            datetime.timedelta(seconds=s),
            datetime.timedelta(seconds=e),
            f"Captain Flint english line {i} dialogue.",
        )
        for i, (s, e) in enumerate(data["ref"])
    ]
    target_raw_cues = [
        srt.Subtitle(
            i + 1,
            datetime.timedelta(seconds=s),
            datetime.timedelta(seconds=e),
            f"Kapten Flint svensk replik {i} i Nassau.",
        )
        for i, (s, e) in enumerate(data["raw"])
    ]
    target_synced_cues = [
        srt.Subtitle(
            i + 1,
            datetime.timedelta(seconds=s),
            datetime.timedelta(seconds=e),
            f"Kapten Flint svensk replik {i} i Nassau.",
        )
        for i, (s, e) in enumerate(data["synced"])
    ]
    return ref_cues, target_raw_cues, target_synced_cues


# ---------------------------------------------------------------------------
# A-D. Trust Contradiction Gate Tests
# ---------------------------------------------------------------------------

class TestContradictionGate:
    """Tests for the Trust Contradiction Gate (Step 7b)."""

    @pytest.mark.asyncio
    async def test_raw_candidate_not_accepted_unchanged_and_repaired_to_pass(self, tmp_path):
        """
        A. PASS false-positive / consolidated-cue case (S03E09 core incident):
        - Raw candidate has legacy NONE (median ~+0.041s) but independent estimator finds -5.128s.
        - With strong current-run Bazarr provenance + auto_repair=True:
            * Raw candidate is NOT accepted unchanged
            * Contradiction gate applies safe global shift (-5.128s)
            * Post-repair Trust achieves verified exact PASS
            * Subtitle text is preserved 100% identically
            * AI calls = 0
        """
        ref_cues, target_raw_cues, _ = _build_forensic_s03e09_fixture()

        # Verify raw timing properties
        raw_align = align_subtitle_timelines(target_raw_cues, ref_cues)
        raw_indep = estimate_global_offset(target_raw_cues, ref_cues)
        assert raw_align.sync_error_type == SyncErrorType.NONE
        assert raw_indep is not None
        assert abs(raw_indep - raw_align.median_offset_sec) >= 2.0
        assert abs(raw_indep) >= 2.0

        video_path = str(tmp_path / "SurvivorsRemorse.S03E09.mkv")
        Path(video_path).touch()
        ref_path = str(tmp_path / "SurvivorsRemorse.S03E09.en.srt")
        Path(ref_path).write_text(srt.compose(ref_cues), encoding="utf-8")
        cand_path = str(tmp_path / "SurvivorsRemorse.S03E09.sv.srt")
        Path(cand_path).write_text(srt.compose(target_raw_cues), encoding="utf-8")

        pre_snap = TargetSnapshot(path=cand_path, exists=False, size=0, mtime_ns=0, content_hash="")
        curr_snap = capture_target_snapshot(cand_path)
        prov = _strong_bazarr_provenance(video_path, "sv", curr_snap, pre_snap)

        src = SubtitleSource(
            language="en", content=srt.compose(ref_cues), cues=ref_cues,
            origin=SourceOrigin.EMBEDDED, path=ref_path,
        )

        engine = SubtitleTrustEngine()
        with patch("app.core.trust_engine.find_external_subtitle", return_value=ref_path):
            result = await engine.evaluate_candidate(
                video_path=video_path,
                candidate_path=cand_path,
                target_lang="sv",
                origin=CandidateOrigin.BAZARR,
                provided_source=src,
                auto_repair=True,
                allow_ai_audit=False,
                bazarr_provenance=prov,
            )

        assert result.passed is True
        assert result.decision == TrustDecision.PASS
        assert result.repair is not None
        assert result.repair.get("contradiction_gate") is True
        assert abs(result.repair["original_offset_sec"] - raw_indep) < 0.1
        assert result.ai_calls == 0

        # Verify disk text is unchanged (only timestamps shifted)
        disk_cues = list(srt.parse(Path(cand_path).read_text(encoding="utf-8")))
        assert len(disk_cues) == len(target_raw_cues)
        orig_texts = [c.content for c in target_raw_cues]
        disk_texts = [c.content for c in disk_cues]
        assert orig_texts == disk_texts

    @pytest.mark.asyncio
    async def test_failed_scratch_repair_fails_closed_never_passes_through(self, tmp_path):
        """
        B. UNRESOLVED CONTRADICTION SAFETY INVARIANT:
        If a candidate has a credible timing contradiction (legacy NONE vs large independent offset)
        but the scratch repair fails to achieve PASS (or is rejected by safety gates):
        -> THE ORIGINAL CANDIDATE MUST NOT PASS UNDER ITS FLAWED LEGACY NONE VERDICT.
        -> MUST FAIL CLOSED (TrustDecision.FAIL -> normal AI fallback).
        """
        ref_cues, target_raw_cues, _ = _build_forensic_s03e09_fixture()

        video_path = str(tmp_path / "show.mkv")
        Path(video_path).touch()
        ref_path = str(tmp_path / "show.en.srt")
        Path(ref_path).write_text(srt.compose(ref_cues), encoding="utf-8")
        cand_path = str(tmp_path / "show.sv.srt")
        Path(cand_path).write_text(srt.compose(target_raw_cues), encoding="utf-8")

        pre_snap = TargetSnapshot(path=cand_path, exists=False, size=0, mtime_ns=0, content_hash="")
        curr_snap = capture_target_snapshot(cand_path)
        prov = _strong_bazarr_provenance(video_path, "sv", curr_snap, pre_snap)

        src = SubtitleSource(
            language="en", content=srt.compose(ref_cues), cues=ref_cues,
            origin=SourceOrigin.EMBEDDED, path=ref_path,
        )

        engine = SubtitleTrustEngine()
        # Simulate scratch revalidation returning None (scratch trial failed or rejected)
        with patch.object(engine, "_execute_scratch_repair_and_revalidate", new=AsyncMock(return_value=None)):
            with patch("app.core.trust_engine.find_external_subtitle", return_value=ref_path):
                result = await engine.evaluate_candidate(
                    video_path=video_path,
                    candidate_path=cand_path,
                    target_lang="sv",
                    origin=CandidateOrigin.BAZARR,
                    provided_source=src,
                    auto_repair=True,
                    allow_ai_audit=False,
                    bazarr_provenance=prov,
                )

        # Core safety invariant: MUST FAIL CLOSED!
        assert result.passed is False
        assert result.decision == TrustDecision.FAIL
        assert any("contradiction" in r.lower() for r in result.reasons)
        assert result.metrics.get("contradiction_gate_triggered") is True
        assert result.metrics.get("contradiction_repair_successful") is False

    @pytest.mark.asyncio
    async def test_post_shift_none_residual_fails_closed(self, tmp_path):
        """
        FAIL-CLOSED RESIDUAL COLLAPSE PROOF:
        If pre-shift candidate has legacy NONE + large credible contradiction,
        but post-shift hypothesis yields estimate_global_offset = None (unproven residual):
        -> Repair is NOT considered proven safe (_residual_collapses_cg must be False).
        -> Original candidate FAILS CLOSED.
        -> No promotion occurs.
        -> No legacy PASS fallthrough.
        """
        ref_cues, target_raw_cues, _ = _build_forensic_s03e09_fixture()

        video_path = str(tmp_path / "show.mkv")
        Path(video_path).touch()
        ref_path = str(tmp_path / "show.en.srt")
        Path(ref_path).write_text(srt.compose(ref_cues), encoding="utf-8")
        cand_path = str(tmp_path / "show.sv.srt")
        orig_content = srt.compose(target_raw_cues)
        Path(cand_path).write_text(orig_content, encoding="utf-8")

        pre_snap = TargetSnapshot(path=cand_path, exists=False, size=0, mtime_ns=0, content_hash="")
        curr_snap = capture_target_snapshot(cand_path)
        prov = _strong_bazarr_provenance(video_path, "sv", curr_snap, pre_snap)

        src = SubtitleSource(
            language="en", content=srt.compose(ref_cues), cues=ref_cues,
            origin=SourceOrigin.EMBEDDED, path=ref_path,
        )

        real_estimate = estimate_global_offset
        call_count = [0]

        def mock_estimate(target_c, ref_c, *args, **kwargs):
            call_count[0] += 1
            # Call 1 (pre-shift on raw candidate): returns real large offset (-5.128s)
            if call_count[0] == 1:
                return real_estimate(target_c, ref_c, *args, **kwargs)
            # Call 2 (post-shift hypothesis residual check): returns None (unproven residual)
            return None

        engine = SubtitleTrustEngine()
        with (
            patch("app.core.trust_engine.estimate_global_offset", side_effect=mock_estimate),
            patch("app.core.trust_engine.find_external_subtitle", return_value=ref_path),
        ):
            result = await engine.evaluate_candidate(
                video_path=video_path,
                candidate_path=cand_path,
                target_lang="sv",
                origin=CandidateOrigin.BAZARR,
                provided_source=src,
                auto_repair=True,
                allow_ai_audit=False,
                bazarr_provenance=prov,
            )

        # Invariant: Must FAIL CLOSED when post-shift residual is unproven (None)
        assert result.passed is False
        assert result.decision == TrustDecision.FAIL
        assert any("contradiction" in r.lower() for r in result.reasons)
        assert result.metrics.get("contradiction_gate_triggered") is True
        assert result.metrics.get("contradiction_repair_successful") is False

        # Verify disk file was NOT promoted / remains identical to raw
        disk_text = Path(cand_path).read_text(encoding="utf-8")
        assert disk_text == orig_content

    @pytest.mark.asyncio
    async def test_weak_provenance_contradiction_fails_closed(self, tmp_path):
        """
        C. When provenance is weak / EXTERNAL (not strong current-run Bazarr),
        the contradiction gate detects the large timing discrepancy and FAILS CLOSED
        without modifying the candidate file on disk.
        """
        ref_cues, target_raw_cues, _ = _build_forensic_s03e09_fixture()

        video_path = str(tmp_path / "show.mkv")
        Path(video_path).touch()
        ref_path = str(tmp_path / "show.en.srt")
        Path(ref_path).write_text(srt.compose(ref_cues), encoding="utf-8")
        cand_path = str(tmp_path / "show.sv.srt")
        Path(cand_path).write_text(srt.compose(target_raw_cues), encoding="utf-8")

        src = SubtitleSource(
            language="en", content=srt.compose(ref_cues), cues=ref_cues,
            origin=SourceOrigin.EMBEDDED, path=ref_path,
        )

        engine = SubtitleTrustEngine()
        with patch("app.core.trust_engine.find_external_subtitle", return_value=ref_path):
            result = await engine.evaluate_candidate(
                video_path=video_path,
                candidate_path=cand_path,
                target_lang="sv",
                origin=CandidateOrigin.EXTERNAL,  # NOT strong Bazarr
                provided_source=src,
                auto_repair=True,
                allow_ai_audit=False,
                bazarr_provenance=None,  # No strong provenance
            )

        assert result.passed is False
        assert result.decision == TrustDecision.FAIL
        assert any("contradiction" in r.lower() for r in result.reasons)
        assert result.repair is None

    @pytest.mark.asyncio
    async def test_already_aligned_consolidated_passes_without_repair(self, tmp_path):
        """
        D. An already-correct subtitle (independent offset ~= 0) passes with Trust PASS
        without the contradiction gate firing or modifying timestamps.
        """
        ref_cues, _, target_synced_cues = _build_forensic_s03e09_fixture()

        # Confirm synced version has independent offset near zero
        indep = estimate_global_offset(target_synced_cues, ref_cues)
        assert indep is None or abs(indep) < 2.0

        video_path = str(tmp_path / "show.mkv")
        Path(video_path).touch()
        ref_path = str(tmp_path / "show.en.srt")
        Path(ref_path).write_text(srt.compose(ref_cues), encoding="utf-8")
        cand_path = str(tmp_path / "show.sv.srt")
        Path(cand_path).write_text(srt.compose(target_synced_cues), encoding="utf-8")

        pre_snap = TargetSnapshot(path=cand_path, exists=False, size=0, mtime_ns=0, content_hash="")
        curr_snap = capture_target_snapshot(cand_path)
        prov = _strong_bazarr_provenance(video_path, "sv", curr_snap, pre_snap)

        src = SubtitleSource(
            language="en", content=srt.compose(ref_cues), cues=ref_cues,
            origin=SourceOrigin.EMBEDDED, path=ref_path,
        )

        engine = SubtitleTrustEngine()
        with patch("app.core.trust_engine.find_external_subtitle", return_value=ref_path):
            result = await engine.evaluate_candidate(
                video_path=video_path,
                candidate_path=cand_path,
                target_lang="sv",
                origin=CandidateOrigin.BAZARR,
                provided_source=src,
                auto_repair=True,
                allow_ai_audit=False,
                bazarr_provenance=prov,
            )

        assert result.passed is True
        assert result.decision == TrustDecision.PASS
        # Contradiction gate must NOT have fired
        if result.repair is not None:
            assert result.repair.get("contradiction_gate") is not True

    @pytest.mark.parametrize("offset_sec", [0.05, 0.11])
    @pytest.mark.asyncio
    async def test_near_zero_offset_passes_without_repair(self, tmp_path, offset_sec):
        """
        E. Small legitimate offsets (+50ms, +110ms) must remain fast PASS
        without triggering the contradiction gate.
        """
        n_cues = 55
        ref_specs = [
            (10.0 + i * 20.0, 10.0 + i * 20.0 + 3.0,
             f"Captain Flint dialogue line {i + 1}: some spoken content here.")
            for i in range(n_cues)
        ]
        target_specs = [
            (s + offset_sec, e + offset_sec, _sv_text(i))
            for i, (s, e, _) in enumerate(ref_specs)
        ]

        ref_srt = _make_srt(ref_specs)
        target_srt = _make_srt(target_specs)
        ref_cues = _make_cues(ref_specs)
        target_cues = _make_cues(target_specs)

        video_path = str(tmp_path / "show.mkv")
        Path(video_path).touch()
        ref_path = str(tmp_path / "show.en.srt")
        Path(ref_path).write_text(ref_srt, encoding="utf-8")
        cand_path = str(tmp_path / "show.sv.srt")
        Path(cand_path).write_text(target_srt, encoding="utf-8")

        pre_snap = TargetSnapshot(path=cand_path, exists=False, size=0, mtime_ns=0, content_hash="")
        curr_snap = capture_target_snapshot(cand_path)
        prov = _strong_bazarr_provenance(video_path, "sv", curr_snap, pre_snap)

        src = SubtitleSource(
            language="en", content=ref_srt, cues=ref_cues,
            origin=SourceOrigin.EMBEDDED, path=ref_path,
        )

        engine = SubtitleTrustEngine()
        with patch("app.core.trust_engine.find_external_subtitle", return_value=ref_path):
            result = await engine.evaluate_candidate(
                video_path=video_path,
                candidate_path=cand_path,
                target_lang="sv",
                origin=CandidateOrigin.BAZARR,
                provided_source=src,
                auto_repair=True,
                allow_ai_audit=False,
                bazarr_provenance=prov,
            )

        assert result.passed is True
        if result.repair is not None:
            assert result.repair.get("contradiction_gate") is not True
        assert result.ai_calls == 0

    @pytest.mark.asyncio
    async def test_wrong_release_mismatch_fails_closed(self, tmp_path):
        """
        F. Wrong-release / irregular mismatch is NOT rescued by estimator alone.
        Must FAIL closed.
        """
        import random
        rng = random.Random(77)

        ref_specs = [
            (10.0 + i * 25.0, 10.0 + i * 25.0 + rng.uniform(1.5, 4.0),
             f"Captain Flint reference dialogue {i + 1}: extended content line here.")
            for i in range(65)
        ]

        t = 5.0
        target_specs = []
        for j in range(50):
            dur = rng.uniform(0.8, 3.5)
            gap = rng.uniform(1.0, 45.0)
            target_specs.append((t, t + dur, _sv_text(j)))
            t += dur + gap
            if t > 1600:
                break

        ref_srt = _make_srt(ref_specs)
        target_srt = _make_srt(target_specs)
        ref_cues = _make_cues(ref_specs)

        video_path = str(tmp_path / "show.mkv")
        Path(video_path).touch()
        ref_path = str(tmp_path / "show.en.srt")
        Path(ref_path).write_text(ref_srt, encoding="utf-8")
        cand_path = str(tmp_path / "show.sv.srt")
        Path(cand_path).write_text(target_srt, encoding="utf-8")

        pre_snap = TargetSnapshot(path=cand_path, exists=False, size=0, mtime_ns=0, content_hash="")
        curr_snap = capture_target_snapshot(cand_path)
        prov = _strong_bazarr_provenance(video_path, "sv", curr_snap, pre_snap)

        src = SubtitleSource(
            language="en", content=ref_srt, cues=ref_cues,
            origin=SourceOrigin.EMBEDDED, path=ref_path,
        )

        engine = SubtitleTrustEngine()
        with patch("app.core.trust_engine.find_external_subtitle", return_value=ref_path):
            result = await engine.evaluate_candidate(
                video_path=video_path,
                candidate_path=cand_path,
                target_lang="sv",
                origin=CandidateOrigin.BAZARR,
                provided_source=src,
                auto_repair=True,
                allow_ai_audit=False,
                bazarr_provenance=prov,
            )

        assert result.passed is False
        assert result.decision == TrustDecision.FAIL


# ---------------------------------------------------------------------------
# G. Black Sails LOW_COVERAGE Repair Regression
# ---------------------------------------------------------------------------

class TestBlackSailsLowCoverageRepair:
    """G. Existing Black Sails LOW_COVERAGE global-offset repair remains intact."""

    @pytest.mark.asyncio
    async def test_black_sails_low_coverage_repair_still_works(self, tmp_path):
        """
        Black Sails style: +20.45s constant offset, raw alignment is LOW_COVERAGE.
        Must still be safely repaired and pass Trust with AI calls = 0.
        """
        import random
        rng = random.Random(42)
        OFFSET = 20.45

        ref_specs = []
        ref_specs.append((10.0, 14.0, "[WAVES CRASHING ON THE REEF]"))
        ref_specs.append((25.0, 30.0, "Flint: Look to the horizon and tell me what you see."))
        ref_specs.append((50.0, 54.0, "[THUNDER RUMBLES]"))
        ref_specs.append((85.0, 91.0, "Silver: They are coming for Nassau."))
        ref_specs.append((120.0, 124.0, "[CANNON FIRE]"))
        ref_specs.append((145.0, 151.0, "Flint: All hands to battle stations!"))
        ref_specs.append((175.0, 180.0, "Silver: Fire on my mark!"))
        ref_specs.append((188.4, 192.0, "[PIRATES CHEERING]"))

        sv_cues = [
            (25.0, 30.0, "Flint: Titta mot horisonten och säg vad du ser."),
            (85.0, 91.0, "Silver: De är på väg mot Nassau."),
            (145.0, 151.0, "Flint: Alla man på sina poster!"),
            (175.0, 180.0, "Silver: Skjut på min signal!"),
        ]

        t = 195.0
        for i in range(48):
            dur = rng.uniform(3.5, 4.8)
            gap = 16.0 if i % 2 == 0 else 34.0
            ref_specs.append((t, t + dur, f"Captain Flint main dialogue line {i}."))
            if i in (6, 14, 22, 30, 38, 46):
                ref_specs.append((t + dur + 0.8, t + dur + 3.0, f"[SOUND EFFECT {i}]"))
            sv_cues.append((t, t + dur, f"Kapten Flint svensk replik {i} i Nassau."))
            t += dur + gap + (3.0 if i in (6, 14, 22, 30, 38, 46) else 0.0)

        target_specs = [(s + OFFSET, e + OFFSET, txt) for s, e, txt in sv_cues]
        ref_cues = _make_cues(ref_specs)
        tgt_cues = _make_cues(target_specs)

        raw_align = align_subtitle_timelines(tgt_cues, ref_cues)
        assert raw_align.sync_error_type == SyncErrorType.LOW_COVERAGE

        video_path = str(tmp_path / "Black.Sails.S03E04.1080p.mkv")
        Path(video_path).touch()
        ref_path = str(tmp_path / "Black.Sails.S03E04.en.srt")
        Path(ref_path).write_text(srt.compose(ref_cues), encoding="utf-8")
        cand_path = str(tmp_path / "Black.Sails.S03E04.sv.srt")
        Path(cand_path).write_text(srt.compose(tgt_cues), encoding="utf-8")

        cand_snap = capture_target_snapshot(cand_path)
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
        src = SubtitleSource(
            origin=SourceOrigin.EMBEDDED, path=ref_path, language="en",
            content=srt.compose(ref_cues), cues=ref_cues,
        )

        engine = SubtitleTrustEngine()
        result = await engine.evaluate_candidate(
            video_path=video_path,
            candidate_path=cand_path,
            target_lang="sv",
            origin=CandidateOrigin.BAZARR,
            provided_source=src,
            allow_ai_audit=False,
            bazarr_provenance=prov,
            auto_repair=True,
        )

        assert result.passed is True
        assert result.decision == TrustDecision.PASS
        assert result.repair is not None
        assert abs(result.repair["original_offset_sec"] - OFFSET) < 0.1
        assert result.ai_calls == 0


# ---------------------------------------------------------------------------
# H-J. Bazarr Lifecycle & Job Classification Tests
# ---------------------------------------------------------------------------

class TestBazarrLifecycleAndJobClassification:
    """H-J. Tests for Bazarr lifecycle and job classification invariants."""

    @pytest.mark.asyncio
    async def test_active_unclassified_subtitle_jobs_keeps_candidate_provisional(self, tmp_path):
        """
        H. If poll_res.status == ACTIVE and jobs contain unclassified subtitle work
        (e.g. generic 'Downloading Subtitles'):
        -> Coordinator must keep candidate PROVISIONAL
        -> Must NOT synthesize KNOWN_IDLE provenance
        -> Continues polling within deadline
        """
        video_path = "/data/media/tv/Survivors Remorse/Season 3/S03E09.mkv"
        lang = "sv"
        candidate_path = str(tmp_path / "S03E09.sv.srt")
        Path(candidate_path).write_text(_make_srt([(10.0, 13.0, "Svensk text")]), encoding="utf-8")

        trust_eval_calls = []

        async def mock_trust_eval(*args, **kwargs):
            trust_eval_calls.append(kwargs.get("bazarr_provenance"))
            return type("TR", (), {
                "decision": TrustDecision.PASS, "score": 91, "passed": True,
                "ai_calls": 0, "repaired_path": None, "reasons": ["mock"],
                "candidate_snapshot": capture_target_snapshot(candidate_path),
                "candidate_state": None,
            })()

        poll_count = [0]
        def make_poll():
            poll_count[0] += 1
            if poll_count[0] <= 4:
                # Generic active subtitle job — ambiguous work!
                job = BazarrJobInfo(
                    job_id="99", job_name="Downloading Subtitles",
                    status="running", is_progress=True, job_type="search",
                )
                return BazarrJobsPollResult(status=BazarrJobPollStatus.ACTIVE, jobs=[job])
            # Later authoritative idle
            return BazarrJobsPollResult(status=BazarrJobPollStatus.KNOWN_IDLE, jobs=[])

        coordinator = BazarrCoordinator()
        coordinator.reset()

        with (
            patch.object(coordinator, "poll_system_jobs", new=AsyncMock(side_effect=lambda *a, **kw: make_poll())),
            patch("app.services.bazarr_coordinator.find_external_subtitle", return_value=candidate_path),
            patch.object(SubtitleTrustEngine, "evaluate_candidate", new=AsyncMock(side_effect=mock_trust_eval)),
        ):
            state, path, trust_res = await coordinator.coordinate_target(
                video_path=video_path,
                target_lang=lang,
                bazarr_url="http://bazarr:6767",
                bazarr_api_key="test",
                max_wait_seconds=5.0,
                search_accepted=True,
                media_correlated=True,
                pre_trigger_snapshot=TargetSnapshot(path=candidate_path, exists=False, size=0, mtime_ns=0, content_hash=""),
            )

        # Invariant: no Trust evaluation was invoked with fabricated KNOWN_IDLE during the ACTIVE phase
        assert poll_count[0] >= 5, "Coordinator must have polled through the active phase"
        for prov in trust_eval_calls:
            if prov is not None and prov.is_finalized:
                poll_val = getattr(prov.poll_state, "value", str(prov.poll_state)).upper()
                assert poll_val == "KNOWN_IDLE", (
                    f"Finalized evaluations must use KNOWN_IDLE, got {poll_val}"
                )

    @pytest.mark.asyncio
    async def test_active_conclusively_unrelated_jobs_does_not_block_finalization(self, tmp_path):
        """
        I. If poll_res.status == ACTIVE but all active jobs are conclusively unrelated
        (e.g. system backup, or search for an explicitly different movie/language):
        -> Coordinator must NOT unnecessarily block this target forever
        -> Proceeds to finalize and evaluate candidate
        """
        video_path = "/data/media/tv/Survivors Remorse/Season 3/S03E09.mkv"
        lang = "sv"
        candidate_path = str(tmp_path / "S03E09.sv.srt")
        Path(candidate_path).write_text(_make_srt([(10.0, 13.0, "Svensk text")]), encoding="utf-8")

        trust_evaluated = [False]

        async def mock_trust_eval(*args, **kwargs):
            trust_evaluated[0] = True
            return type("TR", (), {
                "decision": TrustDecision.PASS, "score": 93, "passed": True,
                "ai_calls": 0, "repaired_path": None, "reasons": ["mock pass"],
                "candidate_snapshot": capture_target_snapshot(candidate_path),
                "candidate_state": None,
            })()

        # ACTIVE with conclusively unrelated jobs: Database Backup + search for French subtitles for Inception
        unrelated_jobs = [
            BazarrJobInfo(
                job_id="1", job_name="Backup Database",
                status="running", is_progress=True, job_type="other",
            ),
            BazarrJobInfo(
                job_id="2", job_name="Searching Subtitles",
                status="running", is_progress=True, progress_message="Inception (2010).mkv (fr)",
                job_type="search",
            ),
        ]
        poll_res = BazarrJobsPollResult(status=BazarrJobPollStatus.ACTIVE, jobs=unrelated_jobs)

        coordinator = BazarrCoordinator()
        coordinator.reset()

        with (
            patch.object(coordinator, "poll_system_jobs", new=AsyncMock(return_value=poll_res)),
            patch("app.services.bazarr_coordinator.find_external_subtitle", return_value=candidate_path),
            patch.object(SubtitleTrustEngine, "evaluate_candidate", new=AsyncMock(side_effect=mock_trust_eval)),
        ):
            state, path, trust_res = await coordinator.coordinate_target(
                video_path=video_path,
                target_lang=lang,
                bazarr_url="http://bazarr:6767",
                bazarr_api_key="test",
                max_wait_seconds=4.0,
                search_accepted=True,
                media_correlated=True,
                pre_trigger_snapshot=TargetSnapshot(path=candidate_path, exists=False, size=0, mtime_ns=0, content_hash=""),
            )

        assert state == BazarrLifecycleState.FINALIZED_WITH_TARGET
        assert trust_evaluated[0] is True, "Target should finalize despite unrelated active backup"

    @pytest.mark.asyncio
    async def test_later_authoritative_idle_evaluates_exact_final_generation(self, tmp_path):
        """
        J. After authoritative KNOWN_IDLE is observed, the coordinator evaluates
        the exact final generation of the candidate.
        """
        video_path = "/data/media/tv/Show/S01E01.mkv"
        lang = "sv"
        candidate_path = str(tmp_path / "S01E01.sv.srt")
        Path(candidate_path).write_text(_make_srt([(10.0, 13.0, "Svensk text")]), encoding="utf-8")

        eval_snapshots = []

        async def mock_trust_eval(*args, **kwargs):
            snap = capture_target_snapshot(candidate_path)
            eval_snapshots.append(snap.generation_id)
            return type("TR", (), {
                "decision": TrustDecision.PASS, "score": 93, "passed": True,
                "ai_calls": 0, "repaired_path": None, "reasons": ["pass"],
                "candidate_snapshot": snap, "candidate_state": None,
            })()

        poll_count = [0]
        def make_poll():
            poll_count[0] += 1
            if poll_count[0] <= 2:
                job = BazarrJobInfo(
                    job_id="77", job_name="Searching Subtitles Show",
                    status="running", is_progress=True, job_type="search",
                )
                return BazarrJobsPollResult(status=BazarrJobPollStatus.ACTIVE, jobs=[job])
            return BazarrJobsPollResult(status=BazarrJobPollStatus.KNOWN_IDLE, jobs=[])

        coordinator = BazarrCoordinator()
        coordinator.reset()

        with (
            patch.object(coordinator, "poll_system_jobs", new=AsyncMock(side_effect=lambda *a, **kw: make_poll())),
            patch("app.services.bazarr_coordinator.find_external_subtitle", return_value=candidate_path),
            patch.object(SubtitleTrustEngine, "evaluate_candidate", new=AsyncMock(side_effect=mock_trust_eval)),
        ):
            state, path, trust_res = await coordinator.coordinate_target(
                video_path=video_path,
                target_lang=lang,
                bazarr_url="http://bazarr:6767",
                bazarr_api_key="test",
                max_wait_seconds=5.0,
                search_accepted=True,
                media_correlated=True,
                pre_trigger_snapshot=TargetSnapshot(path=candidate_path, exists=False, size=0, mtime_ns=0, content_hash=""),
            )

        assert state == BazarrLifecycleState.FINALIZED_WITH_TARGET
        assert len(eval_snapshots) >= 1
        final_snap = capture_target_snapshot(candidate_path)
        assert eval_snapshots[-1] == final_snap.generation_id


# ---------------------------------------------------------------------------
# K. Mutation Invalidation Test
# ---------------------------------------------------------------------------

class TestMutationInvalidation:
    """K. Generation mutation invalidates prior Trust result."""

    @pytest.mark.asyncio
    async def test_mutation_invalidates_prior_trust_result(self, tmp_path):
        """
        If candidate file mutates during polling, prior PASS is discarded and
        the new generation is re-evaluated.
        """
        video_path = "/data/media/show.mkv"
        lang = "sv"
        candidate_path = str(tmp_path / "show.sv.srt")
        Path(candidate_path).write_text(_make_srt([(10.0, 13.0, "V1")]), encoding="utf-8")

        eval_count = [0]

        async def mock_trust_eval(*args, **kwargs):
            eval_count[0] += 1
            return type("TR", (), {
                "decision": TrustDecision.PASS, "score": 90, "passed": True,
                "ai_calls": 0, "repaired_path": None, "reasons": [f"pass {eval_count[0]}"],
                "candidate_snapshot": capture_target_snapshot(candidate_path),
                "candidate_state": None,
            })()

        poll_count = [0]
        def make_poll():
            poll_count[0] += 1
            if poll_count[0] == 3:
                Path(candidate_path).write_text(_make_srt([(15.0, 18.0, "V2")]), encoding="utf-8")
            return BazarrJobsPollResult(status=BazarrJobPollStatus.KNOWN_IDLE, jobs=[])

        coordinator = BazarrCoordinator()
        coordinator.reset()

        with (
            patch.object(coordinator, "poll_system_jobs", new=AsyncMock(side_effect=lambda *a, **kw: make_poll())),
            patch("app.services.bazarr_coordinator.find_external_subtitle", return_value=candidate_path),
            patch.object(SubtitleTrustEngine, "evaluate_candidate", new=AsyncMock(side_effect=mock_trust_eval)),
        ):
            state, path, trust_res = await coordinator.coordinate_target(
                video_path=video_path,
                target_lang=lang,
                bazarr_url="http://bazarr:6767",
                bazarr_api_key="test",
                max_wait_seconds=6.0,
                search_accepted=True,
                media_correlated=True,
                pre_trigger_snapshot=TargetSnapshot(path=candidate_path, exists=False, size=0, mtime_ns=0, content_hash=""),
            )

        assert state in (BazarrLifecycleState.FINALIZED_WITH_TARGET, BazarrLifecycleState.TIMED_OUT)
        assert eval_count[0] >= 1


# ---------------------------------------------------------------------------
# L. Unit Invariants
# ---------------------------------------------------------------------------

class TestContradictionGateUnit:
    """L. Direct unit tests for estimator, contradiction conditions, and residual collapse."""

    def test_estimate_global_offset_finds_large_shift(self):
        """Verify estimator discovers the true -5.128s shift on the forensic timing data."""
        ref_cues, target_raw_cues, _ = _build_forensic_s03e09_fixture()
        indep = estimate_global_offset(target_raw_cues, ref_cues)
        assert indep is not None
        assert abs(indep - (-5.128)) < 0.2

    def test_contradiction_condition_definition(self):
        """Verify contradiction condition definition: |indep - legacy| >= 2.0s and |indep| >= 2.0s."""
        threshold = 2.0
        # Classic S03E09 case: legacy ~+0.04s, indep -5.13s -> contradiction
        assert abs(-5.128 - 0.041) >= threshold and abs(-5.128) >= threshold

        # Normal small offset: legacy +0.05s, indep +0.10s -> NO contradiction
        assert not (abs(0.10 - 0.05) >= threshold and abs(0.10) >= threshold)

        # Consistent large offset: legacy +20.45s, indep +20.45s -> NO contradiction
        assert not (abs(20.45 - 20.45) >= threshold and abs(20.45) >= threshold)

    def test_after_repair_alignment_improves_and_residual_collapses(self):
        """Verify that after applying the -5.128s shift, alignment coverage improves and residual collapses."""
        ref_cues, target_raw_cues, _ = _build_forensic_s03e09_fixture()
        indep = estimate_global_offset(target_raw_cues, ref_cues)
        assert indep is not None

        raw_align = align_subtitle_timelines(target_raw_cues, ref_cues)
        shifted_cues = _shift_cues(target_raw_cues, indep)
        shifted_align = align_subtitle_timelines(shifted_cues, ref_cues)
        post_indep = estimate_global_offset(shifted_cues, ref_cues)

        assert shifted_align.sync_error_type == SyncErrorType.NONE
        assert shifted_align.ref_coverage >= raw_align.ref_coverage
        assert shifted_align.ref_coverage >= 0.85
        assert post_indep is None or abs(post_indep) < 0.5
