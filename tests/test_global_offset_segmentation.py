"""Timing estimation must not depend on how a release splits dialogue cues."""
from datetime import timedelta
import random
from unittest.mock import patch

import pytest
import srt

from app.core.trust_engine import estimate_global_offset, _shift_cues


def consolidated_dialogue(offset, inner_gap=0.05, groups=60, seed=42):
    rng = random.Random(seed)
    reference, target = [], []
    second = abs(offset) + 20
    for i in range(groups):
        first, last = rng.uniform(1.8, 3.8), rng.uniform(1.8, 3.8)
        end = second + first + inner_gap + last
        reference.extend([
            srt.Subtitle(2 * i + 1, timedelta(seconds=second), timedelta(seconds=second + first), 'First sentence.'),
            srt.Subtitle(2 * i + 2, timedelta(seconds=second + first + inner_gap), timedelta(seconds=end), 'Second sentence.'),
        ])
        target.append(srt.Subtitle(i + 1, timedelta(seconds=second + offset), timedelta(seconds=end + offset),
                                   'Första meningen. Andra meningen.'))
        second = end + rng.uniform(3, 12)
    return reference, target


@pytest.mark.parametrize('offset', [-20.45, -5.128, 0.0, 5.128, 20.45])
@pytest.mark.parametrize('inner_gap', [0.0, 0.05, 0.5])
def test_consolidated_cues_have_unbiased_offset_and_proven_zero_residual(offset, inner_gap):
    reference, target = consolidated_dialogue(offset, inner_gap)
    actual = estimate_global_offset(target, reference)
    assert actual is not None
    assert actual == pytest.approx(offset, abs=0.002)
    shifted = _shift_cues(target, actual)
    assert estimate_global_offset(shifted, reference) == pytest.approx(0, abs=0.002)
    assert [cue.content for cue in shifted] == [cue.content for cue in target]
    assert len(shifted) == len(target)
    assert [cue.end - cue.start for cue in shifted] == [cue.end - cue.start for cue in target]


def test_long_silent_gaps_inside_consolidations_do_not_get_a_timing_estimate():
    # Even perfect alignment cannot cover 70% of this target's activity: most
    # target display time is reference silence. The existing gate must reject it.
    reference, target = consolidated_dialogue(-5.128, inner_gap=5)
    assert estimate_global_offset(target, reference) is None


def test_search_window_is_respected():
    reference, target = consolidated_dialogue(25)
    assert estimate_global_offset(target, reference, max_search_offset_sec=2) is None


def test_empty_or_too_short_activity_cannot_prove_an_offset():
    reference, target = consolidated_dialogue(5.128, groups=2)
    assert estimate_global_offset(target, reference) is None
    assert estimate_global_offset([], reference) is None


def contiguous_dialogue(duration=5.0):
    reference = [srt.Subtitle(i + 1, timedelta(seconds=20 + i * 5),
                              timedelta(seconds=20 + i * 5 + duration),
                              f'English dialogue line {i}: we must discuss what happened last night.')
                 for i in range(60)]
    target = _shift_cues(reference, 5.128)
    for i, cue in enumerate(target):
        cue.content = f'Vi måste prata om det som har hänt ikväll {i}.'
    return reference, target


@pytest.mark.parametrize('duration', [5.0, 5.1, 5.128])
def test_contiguous_and_overlapping_activity_retains_offset_evidence(duration):
    reference, target = contiguous_dialogue(duration)
    assert estimate_global_offset(target, reference) == pytest.approx(-5.128, abs=0.002)
    assert estimate_global_offset(_shift_cues(target, -5.128), reference) == pytest.approx(0, abs=0.002)


@pytest.fixture
def trust_database(tmp_path, monkeypatch):
    from app.core import db, trust_engine
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path / 'trust.db'))
    monkeypatch.setattr(trust_engine, '_TRUST_RESULT_MEM_CACHE', {})
    db.init_db()


def trusted_source_files(tmp_path):
    from app.services.source_resolver import SourceOrigin, SubtitleSource
    reference, target = contiguous_dialogue()
    video, ref_path, candidate = tmp_path / 'episode.mkv', tmp_path / 'episode.en.srt', tmp_path / 'episode.sv.srt'
    video.touch()
    ref_path.write_text(srt.compose(reference))
    raw = srt.compose(target)
    candidate.write_text(raw)
    source = SubtitleSource(language='en', origin=SourceOrigin.EMBEDDED,
                            path=str(ref_path), cues=reference, content=srt.compose(reference))
    return video, ref_path, candidate, source, raw


@pytest.mark.asyncio
@pytest.mark.parametrize('strong_provenance', [False, True])
async def test_contiguous_false_legacy_pass_is_rejected_or_verified_repaired(tmp_path, trust_database, strong_provenance):
    from app.core.trust_engine import (BazarrProvenance, CandidateOrigin, SubtitleTrustEngine,
                                       TargetSnapshot, capture_target_snapshot)
    from app.services.bazarr_coordinator import BazarrJobPollStatus
    video, ref_path, candidate, source, raw = trusted_source_files(tmp_path)
    provenance = BazarrProvenance(
        video_path=str(video), target_lang='sv', search_accepted=True,
        pre_trigger_snapshot=TargetSnapshot(path=str(candidate), exists=False, size=0, mtime_ns=0),
        is_finalized=True, is_quiescent=True, media_correlated=True,
        poll_state=BazarrJobPollStatus.KNOWN_IDLE, candidate_snapshot=capture_target_snapshot(str(candidate)),
    ) if strong_provenance else None
    with patch('app.core.trust_engine.find_external_subtitle', return_value=str(ref_path)):
        result = await SubtitleTrustEngine().evaluate_candidate(
            video_path=str(video), candidate_path=str(candidate), target_lang='sv',
            origin=CandidateOrigin.BAZARR if strong_provenance else CandidateOrigin.EXTERNAL,
            provided_source=source, auto_repair=True, allow_ai_audit=False, bazarr_provenance=provenance,
        )
    assert result.passed == strong_provenance
    if strong_provenance:
        assert result.repair['contradiction_gate'] is True
        before, after = list(srt.parse(raw)), list(srt.parse(candidate.read_text()))
        assert len(after) == len(before)
        for original, repaired in zip(before, after):
            assert repaired.content == original.content
            assert repaired.start - original.start == timedelta(seconds=5.128)
            assert repaired.end - original.end == timedelta(seconds=5.128)
    else:
        assert result.metrics['contradiction_gate_triggered'] is True
        assert candidate.read_text() == raw


@pytest.mark.asyncio
async def test_old_policy_pass_is_recomputed_but_current_policy_result_is_reused(tmp_path, trust_database):
    from app.core import trust_engine as te
    video, ref_path, candidate, source, raw = trusted_source_files(tmp_path)
    with patch.object(te, 'find_external_subtitle', return_value=str(ref_path)):
        refs = te.find_and_rank_references(video_path=str(video), target_lang='sv', provided_source=source)
        reference = next(ref for ref in refs if ref.cues)
        fingerprint = te.compute_reference_fingerprint(
            reference=reference, expected_intent=te.SubtitleIntent.FULL,
            origin=te.CandidateOrigin.EXTERNAL, container_tracks=None, target_lang='sv')
        te.save_cached_trust_result(str(candidate), 'sv', te.TrustResult(
            decision=te.TrustDecision.PASS, score=100, confidence='HIGH', reasons=['Old policy'],
            origin=te.CandidateOrigin.EXTERNAL, verification_mode=te.VerificationMode.REFERENCE,
        ), fingerprint, schema_version=3)
        te._TRUST_RESULT_MEM_CACHE.clear()
        engine = te.SubtitleTrustEngine()
        kwargs = dict(video_path=str(video), candidate_path=str(candidate), target_lang='sv',
                      origin=te.CandidateOrigin.EXTERNAL, provided_source=source,
                      auto_repair=True, allow_ai_audit=False)
        result = await engine.evaluate_candidate(**kwargs)
        assert result.passed is False
        assert result.metrics['contradiction_gate_triggered'] is True
        te._TRUST_RESULT_MEM_CACHE.clear()
        with patch.object(te, 'estimate_global_offset', side_effect=AssertionError('Current policy cache must be used')):
            cached = await engine.evaluate_candidate(**kwargs)
        assert cached.passed is False
        assert cached.metrics['contradiction_gate_triggered'] is True
    assert candidate.read_text() == raw
