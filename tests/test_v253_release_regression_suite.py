"""
Comprehensive Regression Test Suite for Babel v2.5.3-beta
Validating:
1. Joker A/B (Standalone and Reference-guided robust timing validation with dialogue pauses)
2. Wrong Episode (Fail-closed rejection of mismatched / systematically shifted external subtitles)
3. American Pie (Publication-only deferred retry without AI re-translation)
4. Source Budget Starvation (Bounded per-language polling without starvation)
5. Multi-Target Bazarr Correlation (Independent per-language provenance tracking)
6. Stale Bazarr Operation Attachment (Verification of active jobs before attachment)
"""

import os
import json
import time
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
import srt

from app.core.trust_engine import (
    SubtitleTrustEngine, TrustDecision, CandidateOrigin,
    VerificationMode, SubtitleIntent, TargetSnapshot,
    BazarrProvenance, align_subtitle_timelines, capture_target_snapshot
)
from app.services.pipeline import SubtitlePipeline
from app.services.source_resolver import SourceResolver, SourceOrigin, SubtitleSource
from app.services.bazarr_coordinator import (
    BazarrCoordinator, BazarrResult, BazarrResultCode,
    BazarrJobPollStatus, BazarrJobsPollResult, BazarrJobInfo,
    BazarrLifecycleState, BazarrMediaInfo
)
from app.core.db import (
    create_job, get_job_by_id, update_job, append_job_log,
    DB_PATH
)


# Sample SRT Generators
def generate_srt(cues, shift_ms=0, content_prefix="Line"):
    items = []
    for i, (start_s, end_s) in enumerate(cues, start=1):
        s_ms = int(start_s * 1000) + shift_ms
        e_ms = int(end_s * 1000) + shift_ms
        start_td = srt.timedelta(milliseconds=s_ms)
        end_td = srt.timedelta(milliseconds=e_ms)
        items.append(srt.Subtitle(index=i, start=start_td, end=end_td, content=f"{content_prefix} {i}"))
    return srt.compose(items)


# ===========================================================================
# 1. JOKER A/B TESTS: Robust Discontinuity & Pause Tolerance
# ===========================================================================

@pytest.mark.asyncio
async def test_joker_case_a_standalone_pause_tolerance(tmp_path):
    """
    Joker Case A (Standalone / Bazarr Provenance):
    A human subtitle has a 5.63s dialogue pause/gap (discontinuity) between scenes,
    but median pace and cadence match speech audio / video duration.
    Trust Engine must PASS with score >= 85 and NOT falsely reject due to natural pauses.
    """
    video_path = tmp_path / "Joker.2019.mkv"
    video_path.touch()

    # 50 cues with a 6-second pause between cue 25 and 26
    cues = []
    t = 10.0
    for i in range(1, 51):
        if i == 26:
            t += 6.0  # 6-second pause
        else:
            t += 2.5
        cues.append((t, t + 1.8))

    sv_srt_path = str(tmp_path / "Joker.2019.sv.srt")
    Path(sv_srt_path).write_text(generate_srt(cues, content_prefix="Det här är en svensk dialograd nummer"), encoding="utf-8")

    trust_engine = SubtitleTrustEngine()
    cand_snap = capture_target_snapshot(sv_srt_path)
    provenance = BazarrProvenance(
        video_path=str(video_path),
        target_lang="sv",
        search_accepted=True,
        pre_trigger_snapshot=TargetSnapshot(path="", exists=False, size=0, mtime_ns=0),
        is_finalized=True,
        is_quiescent=True,
        media_correlated=True,
        poll_state=BazarrJobPollStatus.KNOWN_IDLE,
        candidate_snapshot=cand_snap,
    )

    result = await trust_engine.evaluate_candidate(
        video_path=str(video_path),
        candidate_path=sv_srt_path,
        target_lang="sv",
        origin=CandidateOrigin.BAZARR,
        container_tracks={"duration": t + 10.0, "subtitles": []},
        allow_ai_audit=False,
        bazarr_provenance=provenance,
    )

    assert result.passed is True
    assert result.score >= 85
    assert result.decision == TrustDecision.PASS


@pytest.mark.asyncio
async def test_joker_case_b_reference_guided_pause_tolerance(tmp_path):
    """
    Joker Case B (Reference-Guided Alignment):
    English reference and Swedish target both have natural conversational pauses.
    Statistical timeline alignment (median / MAD) must maintain near-zero residual drift
    and pass evaluation without false rejection.
    """
    cues_en = []
    t = 5.0
    for i in range(1, 60):
        if i in (15, 35):
            t += 8.0  # Dramatic pauses
        else:
            t += 3.0
        cues_en.append((t, t + 2.0))

    # Swedish target matches closely with tiny normal variance (< 100ms)
    cues_sv = [(s + 0.05, e + 0.05) for s, e in cues_en]

    en_subs = list(srt.parse(generate_srt(cues_en, content_prefix="This is an English reference cue number")))
    sv_subs = list(srt.parse(generate_srt(cues_sv, content_prefix="Detta är en svensk motsvarande rad nummer")))

    alignment = align_subtitle_timelines(sv_subs, en_subs)
    assert abs(alignment.median_offset_sec) < 0.2
    assert alignment.score >= 85
    assert alignment.ref_coverage >= 0.95


# ===========================================================================
# 2. WRONG EPISODE / FAIL-CLOSED TESTS
# ===========================================================================

@pytest.mark.asyncio
async def test_wrong_episode_systematic_drift_rejected(tmp_path):
    """
    Wrong Episode / Stale external subtitle:
    A candidate subtitle from a different episode has mismatched cue pacing and large
    cumulative drift (>3500ms offset from reference).
    Trust Engine must FAIL-CLOSED and reject candidate.
    """
    video_path = tmp_path / "Series.S01E02.mkv"
    video_path.touch()

    # Episode 2 reference (40 cues at 3s intervals)
    cues_ref = [(float(i * 3), float(i * 3 + 2)) for i in range(1, 41)]
    en_srt_path = str(tmp_path / "Series.S01E02.en.srt")
    Path(en_srt_path).write_text(generate_srt(cues_ref, content_prefix="This is episode two dialogue line"), encoding="utf-8")

    # Stale Episode 1 target with completely different timing and cadence (e.g. 7s interval cues)
    cues_wrong = [(float(i * 7 + 10), float(i * 7 + 12)) for i in range(1, 25)]
    sv_srt_path = str(tmp_path / "Series.S01E02.sv.srt")
    Path(sv_srt_path).write_text(generate_srt(cues_wrong, content_prefix="Det här är fel avsnitt svensk text rad"), encoding="utf-8")

    trust_engine = SubtitleTrustEngine()
    source_sub = SubtitleSource(
        path=en_srt_path,
        language="en",
        origin=SourceOrigin.EXTERNAL,
        content=Path(en_srt_path).read_text(encoding="utf-8"),
        cues=list(srt.parse(Path(en_srt_path).read_text(encoding="utf-8")))
    )

    result = await trust_engine.evaluate_candidate(
        video_path=str(video_path),
        candidate_path=sv_srt_path,
        target_lang="sv",
        origin=CandidateOrigin.EXTERNAL,
        container_tracks={"duration": 200.0, "subtitles": []},
        provided_source=source_sub,
        allow_ai_audit=False,
    )

    assert result.passed is False
    assert result.decision == TrustDecision.FAIL


# ===========================================================================
# 3. AMERICAN PIE: Publication Deferral & Resumption without Re-Translation
# ===========================================================================

@pytest.mark.asyncio
async def test_american_pie_deferred_publication_resumes_without_ai(tmp_path, monkeypatch):
    """
    American Pie Scenario:
    1. AI translation completes and passes QA.
    2. At publication time, Bazarr is actively writing or holding file lock -> publication is deferred.
    3. The QA-passed translated text is atomically cached.
    4. On subsequent retry pass, Babel resumes publication directly, successfully publishing
       without re-invoking the AI translator.
    """
    video_path = tmp_path / "AmericanPie.mkv"
    video_path.touch()
    en_srt = tmp_path / "AmericanPie.en.srt"
    cues_30 = [(float(i * 3), float(i * 3 + 2)) for i in range(1, 31)]
    en_srt.write_text(generate_srt(cues_30, content_prefix="English dialogue line"), encoding="utf-8")

    pipeline = SubtitlePipeline()
    job_id = create_job(str(video_path))

    ai_call_count = 0
    async def mock_translate(*args, **kwargs):
        nonlocal ai_call_count
        ai_call_count += 1
        return [
            srt.Subtitle(index=i, start=srt.timedelta(seconds=s), end=srt.timedelta(seconds=e), content=f"Svenska dialograd {i}")
            for i, (s, e) in enumerate(cues_30, 1)
        ]

    def mock_get_setting(k, d=""):
        if k == "languages":
            return json.dumps([{"code": "sv", "name": "Swedish", "enabled": True}])
        if k == "enable_bazarr":
            return "true"
        return d

    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)
    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)
    monkeypatch.setattr("app.services.pipeline.find_external_subtitle", lambda vp, l: str(en_srt) if l == "en" else None)
    monkeypatch.setattr(pipeline, "trigger_bazarr_search", AsyncMock(return_value=BazarrResult(code=BazarrResultCode.TRIGGERED, detail="OK")))

    # Pass 1: Simulate publication blocked because Bazarr is actively writing
    with patch("app.services.pipeline._publish_subtitle_with_trust_gate",
               AsyncMock(return_value={"published": False, "skipped": False, "reason": "bazarr_actively_writing"})):
        res1 = await pipeline._run_pipeline_logic(job_id, str(video_path), wait_seconds=0)

    assert res1["status"] in ("waiting_for_bazarr", "waiting_for_publication", "recovering", "partial")
    assert ai_call_count == 1

    # Verify cached artifact was saved
    import app.core.db
    data_dir = os.path.dirname(app.core.db.DB_PATH)
    qapassed_file = os.path.join(data_dir, f"job_{job_id}_sv_qapassed.json")
    assert os.path.exists(qapassed_file)

    # Pass 2: Retry pass — Bazarr write completed, publication gate succeeds
    with patch("app.services.pipeline._publish_subtitle_with_trust_gate",
               AsyncMock(return_value={"published": True, "skipped": False, "reason": "verified"})):
        res2 = await pipeline._run_pipeline_logic(job_id, str(video_path), wait_seconds=0)

    assert res2["status"] == "translated"
    # AI MUST NOT have been called a second time!
    assert ai_call_count == 1
    # Cached artifact must be cleaned up
    assert not os.path.exists(qapassed_file)


# ===========================================================================
# 4. SOURCE SEARCH BUDGET STARVATION TEST
# ===========================================================================

@pytest.mark.asyncio
async def test_source_resolver_budget_prevents_starvation(tmp_path, monkeypatch):
    """
    Source Resolver Starvation Test:
    When searching for source subtitles across multiple candidate languages (e.g. en, es),
    if 'en' search times out or hangs, the bounded polling budget ensures it does not
    consume the entire pipeline deadline, allowing fallback languages to be checked.
    """
    video_path = str(tmp_path / "Movie.mkv")
    Path(video_path).touch()

    es_srt_path = str(tmp_path / "Movie.es.srt")
    cues_25 = [(float(i * 3), float(i * 3 + 2)) for i in range(1, 26)]
    Path(es_srt_path).write_text(generate_srt(cues_25, content_prefix="Hola buenos días cómo estás amigo mío número"), encoding="utf-8")

    resolver = SourceResolver(
        video_path=video_path,
        container_tracks={"duration": 120.0, "subtitles": []},
        primary_audio_lang="en",
        target_languages=["sv"],
        bazarr_url="http://bazarr:6767",
        bazarr_api_key="mock_key",
        enable_bazarr=True,
        extract_source_embedded=False,
        source_search_deadline=2.0,
        source_poll_interval=0.5,
    )

    poll_calls = []
    async def mock_poll_jobs(*args, **kwargs):
        poll_calls.append(time.monotonic())
        return BazarrJobsPollResult(
            status=BazarrJobPollStatus.ACTIVE,
            jobs=[BazarrJobInfo(job_id="1", job_name="Search en", status="running", job_type="search", matched_language="en")]
        )

    monkeypatch.setattr(BazarrCoordinator, "poll_system_jobs", mock_poll_jobs)
    monkeypatch.setattr("app.services.source_resolver.find_external_subtitle",
                        lambda vp, l: es_srt_path if l == "es" else None)

    source = await resolver.resolve()
    # Starvation prevented: Spanish source found despite English search being active
    assert source is not None
    assert source.language == "es"


# ===========================================================================
# 5. MULTI-TARGET BAZARR CORRELATION TEST
# ===========================================================================

@pytest.mark.asyncio
async def test_multi_target_independent_correlation(tmp_path, monkeypatch):
    """
    Multi-Target Bazarr Correlation:
    When target languages are ['sv', 'no']:
    - Bazarr succeeds for 'sv' -> 'sv' gets BAZARR origin and is accepted.
    - Bazarr fails / returns nothing for 'no' -> 'no' does NOT inherit false Bazarr state,
      and is translated via AI.
    """
    video_path = tmp_path / "DualTarget.mkv"
    video_path.touch()

    cues_30 = [(float(i * 3), float(i * 3 + 2)) for i in range(1, 31)]
    en_srt = tmp_path / "DualTarget.en.srt"
    en_srt.write_text(generate_srt(cues_30, content_prefix="This is English dialogue number"), encoding="utf-8")

    sv_srt = tmp_path / "DualTarget.sv.srt"
    Path(sv_srt).write_text(generate_srt(cues_30, content_prefix="Detta är svensk text och dialog nummer"), encoding="utf-8")

    pipeline = SubtitlePipeline()
    job_id = create_job(str(video_path))

    translated_targets = []
    async def mock_translate(subs, target_language, *args, **kwargs):
        translated_targets.append(target_language)
        return [
            srt.Subtitle(index=i, start=srt.timedelta(seconds=s), end=srt.timedelta(seconds=e), content=f"Dette er en flott norsk oversettelse for replikk {i}")
            for i, (s, e) in enumerate(cues_30, 1)
        ]

    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)
    monkeypatch.setattr(pipeline, "get_configured_languages", lambda: [
        {"code": "sv", "name": "Swedish", "enabled": True},
        {"code": "no", "name": "Norwegian", "enabled": True}
    ])
    monkeypatch.setattr("app.services.pipeline.get_setting", lambda k, d="": "true" if k == "enable_bazarr" else d)

    def mock_find(vp, lang):
        if lang == "en": return str(en_srt)
        if lang == "sv": return str(sv_srt)
        return None

    monkeypatch.setattr("app.services.pipeline.find_external_subtitle", mock_find)

    async def mock_bazarr_trigger(vp, language="sv", **kwargs):
        if language == "sv":
            return BazarrResult(code=BazarrResultCode.TRIGGERED, detail="Accepted")
        return BazarrResult(code=BazarrResultCode.MEDIA_NOT_FOUND, detail="Not found")

    monkeypatch.setattr(pipeline, "trigger_bazarr_search", mock_bazarr_trigger)

    with patch("app.services.pipeline._publish_subtitle_with_trust_gate",
               AsyncMock(return_value={"published": True, "skipped": False, "reason": "verified"})):
        res = await pipeline._run_pipeline_logic(job_id, str(video_path), wait_seconds=0)

    # Swedish matched via Bazarr/existing target, Norwegian translated via AI
    assert "Norwegian" in translated_targets
    assert "Swedish" not in translated_targets
    assert res["status"] in ("translated", "partial")


# ===========================================================================
# 6. STALE BAZARR OPERATION ATTACHMENT TEST
# ===========================================================================

@pytest.mark.asyncio
async def test_stale_bazarr_operation_not_attached(tmp_path, monkeypatch):
    """
    Bazarr Coordinator Stale Operation Attachment:
    If a cached operation ID in state is already finished / no longer in active_jobs,
    trigger_or_attach_target_search must NOT attach to the stale operation, but instead
    trigger a fresh search.
    """
    coordinator = BazarrCoordinator()
    from app.services.bazarr_coordinator import BazarrOperation
    coordinator._operations["Movie.mkv:sv:full"] = BazarrOperation(
        op_key="Movie.mkv:sv:full",
        video_path="Movie.mkv",
        target_lang="sv",
        is_search_triggered=True,
        trigger_time=time.monotonic() - 100.0
    )

    # Mock active jobs as KNOWN_IDLE (stale operation is gone)
    monkeypatch.setattr(coordinator, "poll_system_jobs", AsyncMock(return_value=BazarrJobsPollResult(
        status=BazarrJobPollStatus.KNOWN_IDLE,
        jobs=[]
    )))

    # Mock media correlation and HTTP search trigger
    monkeypatch.setattr(coordinator, "correlate_media", AsyncMock(return_value=BazarrMediaInfo(
        is_indexed=True, radarr_id=456, media_type="movie", title="Movie"
    )))

    with patch("httpx.AsyncClient.patch", AsyncMock(return_value=MagicMock(status_code=200, json=lambda: {"data": "ok"}))):
        result = await coordinator.trigger_or_attach_target_search(
            video_path="Movie.mkv",
            target_lang="sv",
            bazarr_url="http://bazarr:6767",
            bazarr_api_key="mock_key"
        )

    assert result.code == BazarrResultCode.TRIGGERED
    assert result.detail != "attached_to_existing"


# ===========================================================================
# 7. BAZARR UNKNOWN LIFECYCLE REGRESSION TESTS (CASES A, B, C, D)
# ===========================================================================

@pytest.mark.asyncio
async def test_bazarr_api_unknown_case_a_no_target_remains_unknown(tmp_path, monkeypatch):
    """
    Case A: Bazarr API is UNKNOWN (outage / network error) and no target exists on disk.
    Lifecycle must remain UNKNOWN / INDETERMINATE, NEVER falsely marked FINALIZED_NO_TARGET.
    """
    coordinator = BazarrCoordinator()
    monkeypatch.setattr(coordinator, "poll_system_jobs", AsyncMock(return_value=BazarrJobsPollResult(
        status=BazarrJobPollStatus.UNKNOWN,
        jobs=[],
        error="502 Bad Gateway"
    )))

    state, cand, tres = await coordinator.coordinate_target(
        video_path=str(tmp_path / "MovieA.mkv"),
        target_lang="sv",
        bazarr_url="http://bazarr:6767",
        bazarr_api_key="key",
        max_wait_seconds=0.1,
        find_external_subtitle_fn=lambda vp, l: None
    )

    assert state in (BazarrLifecycleState.UNKNOWN, BazarrLifecycleState.INDETERMINATE)
    assert state != BazarrLifecycleState.FINALIZED_NO_TARGET
    assert cand is None


@pytest.mark.asyncio
async def test_bazarr_api_unknown_case_b_valid_target_provisional_not_finalized(tmp_path, monkeypatch):
    """
    Case B: Bazarr API is UNKNOWN, but a stable valid target exists on disk.
    Candidate may be provisionally inspected/validated, but lifecycle is NOT
    falsely manufactured as FINALIZED_WITH_TARGET solely from API UNKNOWN.
    """
    coordinator = BazarrCoordinator()
    monkeypatch.setattr(coordinator, "poll_system_jobs", AsyncMock(return_value=BazarrJobsPollResult(
        status=BazarrJobPollStatus.UNKNOWN,
        jobs=[],
        error="Connection reset by peer"
    )))

    target_path = tmp_path / "MovieB.sv.srt"
    target_path.write_text(generate_srt([(float(i*3), float(i*3+2)) for i in range(1, 25)], content_prefix="Det här är en svensk dialograd nummer"), encoding="utf-8")
    ref_path = tmp_path / "MovieB.en.srt"
    ref_path.write_text(generate_srt([(float(i*3), float(i*3+2)) for i in range(1, 25)], content_prefix="English dialogue line"), encoding="utf-8")
    ref_content_b = ref_path.read_text(encoding="utf-8")
    prov_src_b = SubtitleSource(
        path=str(ref_path),
        origin=SourceOrigin.EXTERNAL,
        language="en",
        content=ref_content_b,
        cues=list(srt.parse(ref_content_b))
    )

    state, cand, tres = await coordinator.coordinate_target(
        video_path=str(tmp_path / "MovieB.mkv"),
        target_lang="sv",
        bazarr_url="http://bazarr:6767",
        bazarr_api_key="key",
        max_wait_seconds=0.1,
        provided_source=prov_src_b,
        find_external_subtitle_fn=lambda vp, l: str(target_path)
    )

    assert state in (BazarrLifecycleState.UNKNOWN, BazarrLifecycleState.INDETERMINATE)
    assert state != BazarrLifecycleState.FINALIZED_WITH_TARGET
    assert cand == str(target_path)
    assert tres is not None
    assert tres.passed is True


@pytest.mark.asyncio
async def test_bazarr_api_unknown_case_c_publication_fails_closed(tmp_path, monkeypatch):
    """
    Case C: Bazarr API is UNKNOWN, candidate exists, Bazarr may still be writing.
    Babel publication ownership gate must fail closed and NOT overwrite the candidate.
    """
    coordinator = BazarrCoordinator()
    monkeypatch.setattr(coordinator, "poll_system_jobs", AsyncMock(return_value=BazarrJobsPollResult(
        status=BazarrJobPollStatus.UNKNOWN,
        jobs=[],
        error="Timeout querying Bazarr"
    )))

    target_path = tmp_path / "MovieC.sv.srt"
    target_path.write_text(generate_srt([(1, 4)], content_prefix="Det här är text"), encoding="utf-8")

    ownership_res = await coordinator.acquire_publication_ownership(
        video_path=str(tmp_path / "MovieC.mkv"),
        target_lang="sv",
        bazarr_url="http://bazarr:6767",
        bazarr_api_key="key",
        timeout_sec=0.1,
        find_external_subtitle_fn=lambda vp, l: str(target_path)
    )

    assert ownership_res.granted is False
    assert ownership_res.defer is True
    assert ownership_res.reason == "bazarr_lifecycle_unknown"


@pytest.mark.asyncio
async def test_bazarr_api_unknown_case_d_recovers_to_finalized(tmp_path, monkeypatch):
    """
    Case D: API recovers later and reports KNOWN_IDLE / completed.
    Candidate can then become authoritative normally (FINALIZED_WITH_TARGET).
    """
    coordinator = BazarrCoordinator()
    target_path = tmp_path / "MovieD.sv.srt"
    target_path.write_text(generate_srt([(float(i*3), float(i*3+2)) for i in range(1, 25)], content_prefix="Det här är en svensk dialograd nummer"), encoding="utf-8")
    ref_path_d = tmp_path / "MovieD.en.srt"
    ref_path_d.write_text(generate_srt([(float(i*3), float(i*3+2)) for i in range(1, 25)], content_prefix="English dialogue line"), encoding="utf-8")
    ref_content_d = ref_path_d.read_text(encoding="utf-8")
    prov_src_d = SubtitleSource(
        path=str(ref_path_d),
        origin=SourceOrigin.EXTERNAL,
        language="en",
        content=ref_content_d,
        cues=list(srt.parse(ref_content_d))
    )

    # API is now KNOWN_IDLE
    monkeypatch.setattr(coordinator, "poll_system_jobs", AsyncMock(return_value=BazarrJobsPollResult(
        status=BazarrJobPollStatus.KNOWN_IDLE,
        jobs=[]
    )))

    state, cand, tres = await coordinator.coordinate_target(
        video_path=str(tmp_path / "MovieD.mkv"),
        target_lang="sv",
        bazarr_url="http://bazarr:6767",
        bazarr_api_key="key",
        max_wait_seconds=0.5,
        quiescence_sec=0.0,
        provided_source=prov_src_d,
        find_external_subtitle_fn=lambda vp, l: str(target_path)
    )

    assert state == BazarrLifecycleState.FINALIZED_WITH_TARGET
    assert cand == str(target_path)
    assert tres is not None
    assert tres.passed is True


# ===========================================================================
# 8. TRUST ENGINE: NON-DIALOGUE GAP VS GENUINE MISSING DIALOGUE
# ===========================================================================

def test_trust_engine_non_dialogue_long_gap_passes():
    """
    Non-dialogue long gap:
    A film has 250s of dialogue-free scene (e.g. music/action between t=100s and t=350s).
    Both reference and candidate have zero speech during this gap.
    Overall dialogue coverage is >95%.
    Trust Engine MUST NOT fail solely because raw unmatched timeline span > 200s.
    """
    # 50 cues before t=100s, then 250s silent gap, then 50 cues after t=350s
    ref_cues = []
    t = 1.0
    idx = 1
    for i in range(50):
        ref_cues.append(srt.Subtitle(index=idx, start=srt.timedelta(seconds=t), end=srt.timedelta(seconds=t+1.8), content=f"Ref cue {i}"))
        idx += 1
        t += 2.0

    t = 350.0  # 250s jump with no dialogue in reference either
    for i in range(50, 100):
        ref_cues.append(srt.Subtitle(index=idx, start=srt.timedelta(seconds=t), end=srt.timedelta(seconds=t+1.8), content=f"Ref cue {i}"))
        idx += 1
        t += 2.0

    target_cues = [
        srt.Subtitle(index=c.index, start=c.start, end=c.end, content=f"Target sv {c.content}")
        for c in ref_cues
    ]

    from app.core.trust_engine import align_subtitle_timelines, SyncErrorType
    align_res = align_subtitle_timelines(target_cues, ref_cues)
    assert align_res.sync_error_type == SyncErrorType.NONE
    assert align_res.uncovered_reference_dialogue_sec == 0.0
    assert align_res.ref_coverage >= 0.95


def test_trust_engine_genuine_missing_dialogue_fails():
    """
    Genuine missing dialogue:
    Candidate drops >90 seconds of active reference dialogue.
    Trust Engine MUST classify as LOW_COVERAGE and fail.
    """
    ref_cues = []
    t = 1.0
    for i in range(100):
        ref_cues.append(srt.Subtitle(index=i+1, start=srt.timedelta(seconds=t), end=srt.timedelta(seconds=t+2.0), content=f"Ref dialogue {i}"))
        t += 3.0

    # Target misses cues 30 to 75 (45 cues * 2.0s = 90.0s of active dialogue missing)
    target_cues = []
    for i, c in enumerate(ref_cues):
        if 30 <= i <= 75:
            continue
        target_cues.append(srt.Subtitle(index=len(target_cues)+1, start=c.start, end=c.end, content=f"Target sv {c.content}"))

    from app.core.trust_engine import align_subtitle_timelines, SyncErrorType
    align_res = align_subtitle_timelines(target_cues, ref_cues)
    assert align_res.sync_error_type == SyncErrorType.LOW_COVERAGE
    assert align_res.uncovered_reference_dialogue_sec >= 90.0


# ===========================================================================
# 9. DEFERRED QA ARTIFACT IDENTITY INVALIDATION TEST
# ===========================================================================

@pytest.mark.asyncio
async def test_deferred_qa_artifact_invalidated_on_media_file_change(tmp_path, monkeypatch):
    """
    Identity Safety:
    If a cached QA-passed translation artifact exists for a job, but the video file
    was modified / replaced (mtime or size changed), Babel MUST invalidate the artifact
    and rerun translation rather than publishing a stale subtitle.
    """
    import app.core.db
    monkeypatch.setattr(app.core.db, "DB_PATH", str(tmp_path / "test.db"))
    app.core.db.init_db()

    video_path = tmp_path / "MovieQA.mkv"
    video_path.write_bytes(b"initial_media_content_12345")

    source_srt = tmp_path / "MovieQA.en.srt"
    cues_2 = [(1.0, 3.0), (4.0, 6.0)]
    source_srt.write_text(generate_srt(cues_2, content_prefix="Original source"), encoding="utf-8")

    job_id = create_job(str(video_path))
    update_job(job_id, status="PENDING")

    # Write a QA artifact with the initial file size & mtime
    stat = os.stat(video_path)
    qapassed_file = tmp_path / f"job_{job_id}_sv_qapassed.json"
    qapassed_payload = {
        "job_id": job_id,
        "canonical_video_path": os.path.realpath(str(video_path)),
        "media_file_size": stat.st_size,
        "media_file_mtime_ns": getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)),
        "target_output_path": str(tmp_path / "MovieQA.sv.srt"),
        "lang_code": "sv",
        "source_cue_count": 2,
        "translated_srt_text": "1\n00:00:01,000 --> 00:00:03,000\nStale translation\n\n2\n00:00:04,000 --> 00:00:06,000\nStale rad 2\n",
        "expected_cue_count": 2,
        "qa_score": 95,
        "qa_issues": [],
    }
    qapassed_file.write_text(json.dumps(qapassed_payload), encoding="utf-8")

    # Now modify the video file (e.g. new release / remux downloaded)
    video_path.write_bytes(b"new_replaced_media_content_different_length_999999999")

    pipeline = SubtitlePipeline()
    monkeypatch.setattr("app.services.pipeline.inspect_mkv_tracks", lambda vp: {
        "subtitles": [{"id": 2, "language": "eng", "codec": "SubRip/SRT", "forced": False}],
        "audio": [{"id": 1, "language": "eng", "default": True, "forced": False}],
        "duration": 60.0
    })
    monkeypatch.setattr("app.services.pipeline.find_external_subtitle", lambda vp, l: str(source_srt) if l == "en" else None)

    ai_translated = False
    async def mock_translate(*args, **kwargs):
        nonlocal ai_translated
        ai_translated = True
        return [
            srt.Subtitle(index=1, start=srt.timedelta(seconds=1), end=srt.timedelta(seconds=3), content="Det här är ny svensk text"),
            srt.Subtitle(index=2, start=srt.timedelta(seconds=4), end=srt.timedelta(seconds=6), content="Det här är rad två")
        ]

    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)
    monkeypatch.setattr("app.services.pipeline._publish_subtitle_with_trust_gate",
                        AsyncMock(return_value={"published": True, "skipped": False, "reason": "verified"}))

    await pipeline._run_pipeline_logic(job_id, str(video_path), wait_seconds=0)

    # Cached stale artifact must be invalidated and AI translation rerun
    assert not qapassed_file.exists()
    assert ai_translated is True


@pytest.mark.asyncio
async def test_deferred_qa_artifact_invalidated_on_source_change_with_same_cue_count(tmp_path, monkeypatch):
    """
    Source Identity Safety:
    Media file remains byte-identical, but the source subtitle file is replaced with a new version
    having the EXACT SAME cue count but different timing/content.
    Babel MUST reject the cached QA artifact via source fingerprint mismatch (content hash/mtime),
    never publish the stale translation, and regenerate from the new source.
    """
    from datetime import datetime, timezone
    import app.core.db
    monkeypatch.setattr(app.core.db, "DB_PATH", str(tmp_path / "test.db"))
    app.core.db.init_db()

    video_path = tmp_path / "MovieSameMedia.mkv"
    video_bytes = b"constant_media_content_exact_same_bytes_12345"
    video_path.write_bytes(video_bytes)

    source_srt = tmp_path / "MovieSameMedia.en.srt"

    # Source A has 2 cues
    source_a_cues = [
        srt.Subtitle(index=1, start=srt.timedelta(seconds=1.0), end=srt.timedelta(seconds=3.0), content="Source A line 1"),
        srt.Subtitle(index=2, start=srt.timedelta(seconds=4.0), end=srt.timedelta(seconds=6.0), content="Source A line 2"),
    ]
    source_srt.write_text(srt.compose(source_a_cues), encoding="utf-8")

    job_id = create_job(str(video_path))
    update_job(job_id, status="PENDING")

    # Capture Source A identity
    from app.services.pipeline import compute_source_fingerprint
    from app.services.source_resolver import SubtitleSource, SourceOrigin

    source_a_obj = SubtitleSource(
        path=str(source_srt),
        language="en",
        origin=SourceOrigin.EXTERNAL,
        content=srt.compose(source_a_cues),
        cues=source_a_cues,
    )
    src_a_fingerprint = compute_source_fingerprint(source_a_obj, subs=source_a_cues, video_path=str(video_path))

    # Write QA artifact corresponding to Source A
    stat = os.stat(video_path)
    qapassed_file = tmp_path / f"job_{job_id}_sv_qapassed.json"
    qapassed_payload = {
        "job_id": job_id,
        "canonical_video_path": os.path.realpath(str(video_path)),
        "media_file_size": stat.st_size,
        "media_file_mtime_ns": getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)),
        "target_output_path": str(tmp_path / "MovieSameMedia.sv.srt"),
        "lang_code": "sv",
        **src_a_fingerprint,
        "translated_srt_text": "1\n00:00:01,000 --> 00:00:03,000\nStale Source A Svensk text\n\n2\n00:00:04,000 --> 00:00:06,000\nStale Source A rad 2\n",
        "expected_cue_count": 2,
        "qa_score": 95,
        "qa_issues": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    qapassed_file.write_text(json.dumps(qapassed_payload), encoding="utf-8")

    # Crucial Step: Replace Source A with Source B
    # Source B has the EXACT SAME cue count (2 cues) but DIFFERENT timings and content!
    # Media file is completely untouched (byte-identical).
    source_b_cues = [
        srt.Subtitle(index=1, start=srt.timedelta(seconds=10.0), end=srt.timedelta(seconds=12.0), content="Source B new text 1"),
        srt.Subtitle(index=2, start=srt.timedelta(seconds=14.0), end=srt.timedelta(seconds=16.0), content="Source B new text 2"),
    ]
    source_srt.write_text(srt.compose(source_b_cues), encoding="utf-8")

    pipeline = SubtitlePipeline()
    monkeypatch.setattr("app.services.pipeline.inspect_mkv_tracks", lambda vp: {
        "subtitles": [{"id": 2, "language": "eng", "codec": "SubRip/SRT", "forced": False}],
        "audio": [{"id": 1, "language": "eng", "default": True, "forced": False}],
        "duration": 60.0
    })
    monkeypatch.setattr("app.services.pipeline.find_external_subtitle", lambda vp, l: str(source_srt) if l == "en" else None)

    published_texts = []
    async def mock_publish(video_path, target_output_path, lang_code, translated_srt_text, expected_cue_count, **kwargs):
        published_texts.append(translated_srt_text)
        return {"published": True, "skipped": False, "reason": "verified"}

    ai_translated = False
    async def mock_translate(*args, **kwargs):
        nonlocal ai_translated
        ai_translated = True
        return [
            srt.Subtitle(index=1, start=srt.timedelta(seconds=10), end=srt.timedelta(seconds=12), content="Ny Source B svensk text 1"),
            srt.Subtitle(index=2, start=srt.timedelta(seconds=14), end=srt.timedelta(seconds=16), content="Ny Source B svensk text 2")
        ]

    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)
    monkeypatch.setattr("app.services.pipeline._publish_subtitle_with_trust_gate", mock_publish)

    await pipeline._run_pipeline_logic(job_id, str(video_path), wait_seconds=0)

    # 1. Cached stale QA artifact MUST be rejected and deleted
    assert not qapassed_file.exists()
    # 2. AI translation MUST be rerun for the new source
    assert ai_translated is True
    # 3. Stale translation from Source A must NEVER have been published
    assert len(published_texts) == 1
    assert "Stale Source A" not in published_texts[0]
    assert "Ny Source B svensk text" in published_texts[0]
