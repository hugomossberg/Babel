import asyncio
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import srt

from app.core.trust_engine import (
    SubtitleTrustEngine,
    TrustDecision,
    CandidateOrigin,
    CandidateState,
    TargetSnapshot,
    capture_target_snapshot,
    get_candidate_state,
    wait_for_file_stability,
    wait_for_candidate_quiescence,
    is_file_stable,
    DEFAULT_CANDIDATE_STABILITY_SEC,
    DEFAULT_BAZARR_QUIESCENCE_SEC,
    DEFAULT_HYBRID_BAZARR_MAX_WAIT_SEC
)
from app.services.pipeline import SubtitlePipeline
from app.core.db import get_job_by_id


def setup_mock_settings(monkeypatch, overrides=None):
    settings = {
        "ai_provider": "gemini",
        "gemini_api_key": "dummy_api_key",
        "gemini_model": "gemini-3.5-flash-lite",
        "batch_size": "50",
        "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
        "enable_bazarr_check": "true",
        "bazarr_grace_seconds": "2.0",
        "bazarr_quiescence_seconds": "1.2",
        "bazarr_candidate_stability_seconds": "0.15",
    }
    if overrides:
        settings.update(overrides)
    def mock_get_setting(key, default=""):
        return settings.get(key, default)
    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)
    monkeypatch.setattr("app.services.translator.get_setting", mock_get_setting)
    from app.services.source_resolver import BazarrResult, BazarrResultCode
    async def mock_bazarr_trigger(*args, language="sv", **kwargs):
        return BazarrResult(code=BazarrResultCode.TRIGGERED, language=language, detail="Search accepted")
    monkeypatch.setattr("app.services.pipeline.SubtitlePipeline.trigger_bazarr_search", mock_bazarr_trigger)


@pytest.fixture
def mock_db_settings(monkeypatch):
    setup_mock_settings(monkeypatch)


# ============================================================================
# 1. TargetSnapshot and CandidateState Verification
# ============================================================================

def test_target_snapshot_and_candidate_state(tmp_path):
    fpath = tmp_path / "test.sv.srt"

    # Absent
    snap_absent = capture_target_snapshot(str(fpath))
    assert not snap_absent.exists
    assert snap_absent.generation_id == "absent"
    assert get_candidate_state(snap_absent) == CandidateState.ABSENT

    # Written
    fpath.write_text("1\n00:00:01,000 --> 00:00:04,000\nHej världen!\n", encoding="utf-8")
    snap1 = capture_target_snapshot(str(fpath))
    assert snap1.exists
    assert snap1.size > 0
    assert snap1.content_hash != ""
    assert get_candidate_state(snap1) == CandidateState.PENDING

    # Modified
    fpath.write_text("1\n00:00:01,000 --> 00:00:05,000\nHej världen modifierad!\n", encoding="utf-8")
    snap2 = capture_target_snapshot(str(fpath))
    assert snap2.exists
    assert snap2.generation_id != snap1.generation_id


@pytest.mark.asyncio
async def test_file_stability_check(tmp_path):
    fpath = tmp_path / "stream_write.sv.srt"

    # Not existing
    assert not await wait_for_file_stability(str(fpath), min_stability_sec=0.1, timeout_sec=0.2)
    assert not is_file_stable(str(fpath), min_stability_sec=0.1)

    # Stable valid file (> 100 bytes)
    full_content = "\n\n".join([
        f"{i}\n00:00:0{i},000 --> 00:00:0{i},500\nDetta är en stabil och bra svensk text rad {i}"
        for i in range(1, 10)
    ])
    fpath.write_text(full_content, encoding="utf-8")
    assert await wait_for_file_stability(str(fpath), min_stability_sec=0.1, timeout_sec=0.5)


@pytest.mark.asyncio
async def test_sustained_stability_and_generation_changes(tmp_path):
    """
    Test scenario:
    A) Appears (Gen A)
    B) Stays unchanged for 50ms (< min_stability_sec)
    C) Changes to Gen B, pauses 60ms (< min_stability_sec)
    D) Changes to Gen C (final settled version, 15 lines) and settles for > 150ms
    Only generation D may be declared stable and authoritatively evaluated.
    """
    fpath = tmp_path / "multi_gen.sv.srt"

    async def writer():
        # Gen A
        fpath.write_text("1\n00:00:01,000 --> 00:00:02,000\nGen A line", encoding="utf-8")
        await asyncio.sleep(0.06)
        # Gen B
        fpath.write_text("1\n00:00:01,000 --> 00:00:02,000\nGen B line\n2\n00:00:03,000 --> 00:00:04,000\nLine 2", encoding="utf-8")
        await asyncio.sleep(0.06)
        # Gen C (settled)
        lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nGen C settled line {i}" for i in range(1, 15)]
        fpath.write_text("\n\n".join(lines), encoding="utf-8")

    w_task = asyncio.create_task(writer())
    # Require 0.12s of continuous stability
    stable = await wait_for_file_stability(str(fpath), min_stability_sec=0.12, timeout_sec=1.5, interval_sec=0.02)
    await w_task

    assert stable is True
    snap = capture_target_snapshot(str(fpath))
    assert snap.exists
    # Verify the settled snapshot is Gen C
    content = fpath.read_text(encoding="utf-8")
    assert "Gen C settled line 14" in content


# ============================================================================
# 2. Black Sails S04E06 Candidate Hardening (Bad candidate correctly rejected)
# ============================================================================

@pytest.mark.asyncio
async def test_black_sails_s04e06_bad_candidate_rejected(tmp_path):
    video_path = str(tmp_path / "Black.Sails.S04E06.mkv")
    Path(video_path).touch()

    # 494 cue English reference (~55 mins)
    en_cues = []
    for i in range(1, 495):
        start_sec = i * 6.5
        end_sec = start_sec + 3.0
        m_s, s_s = divmod(int(start_sec), 60)
        h_s, m_s = divmod(m_s, 60)
        m_e, s_e = divmod(int(end_sec), 60)
        h_e, m_e = divmod(m_e, 60)
        en_cues.append(f"{i}\n{h_s:02d}:{m_s:02d}:{s_s:02d},000 --> {h_e:02d}:{m_e:02d}:{s_e:02d},000\nEnglish reference dialogue cue {i}")

    en_path = str(tmp_path / "Black.Sails.S04E06.en.srt")
    Path(en_path).write_text("\n\n".join(en_cues), encoding="utf-8")

    # Incomplete Swedish candidate (only 365 cues, ~89.1% coverage)
    sv_cues = []
    for i in range(1, 366):
        start_sec = i * 6.5
        end_sec = start_sec + 3.0
        m_s, s_s = divmod(int(start_sec), 60)
        h_s, m_s = divmod(m_s, 60)
        m_e, s_e = divmod(int(end_sec), 60)
        h_e, m_e = divmod(m_e, 60)
        sv_cues.append(f"{i}\n{h_s:02d}:{m_s:02d}:{s_s:02d},000 --> {h_e:02d}:{m_e:02d}:{s_e:02d},000\nSvensk undertext dialog rad {i}")

    sv_path = str(tmp_path / "Black.Sails.S04E06.sv.srt")
    Path(sv_path).write_text("\n\n".join(sv_cues), encoding="utf-8")

    engine = SubtitleTrustEngine()
    res = await engine.evaluate_candidate(
        video_path=video_path,
        candidate_path=sv_path,
        target_lang="sv",
        origin=CandidateOrigin.BAZARR,
    )

    # Incomplete candidate MUST be rejected (passed = False)
    assert res.passed is False
    assert res.decision == TrustDecision.FAIL
    assert res.candidate_state == CandidateState.REJECTED


# ============================================================================
# 3. Bazarr Adaptive Grace Check (Early exit on rejection & immediate win on PASS)
# ============================================================================

@pytest.mark.asyncio
async def test_bazarr_adaptive_grace_rejects_and_proceeds_to_ai(mock_db_settings, tmp_path, monkeypatch):
    video_path = tmp_path / "Show.S01E01.mkv"
    video_path.touch()

    # English reference
    en_srt = tmp_path / "Show.S01E01.en.srt"
    en_lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nValid English line {i}" for i in range(1, 25)]
    en_srt.write_text("\n\n".join(en_lines), encoding="utf-8")

    # Incomplete Swedish candidate on disk (only 5 lines vs 24)
    sv_srt = tmp_path / "Show.S01E01.sv.srt"
    sv_lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nKort svensk rad {i}" for i in range(1, 6)]
    sv_srt.write_text("\n\n".join(sv_lines), encoding="utf-8")

    pipeline = SubtitlePipeline()
    ai_called = False

    async def mock_translate_srt(*args, **kwargs):
        nonlocal ai_called
        ai_called = True
        subs = kwargs.get("subs", [])
        return [
            srt.Subtitle(index=s.index, start=s.start, end=s.end, content=f"Översatt rad {s.index}")
            for s in subs
        ]
    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate_srt)

    def mock_find(vp, lang):
        if lang == "en": return str(en_srt)
        if lang == "sv": return str(sv_srt)
        return None

    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find):
        res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")

    assert res["status"] in ["translated", "completed"]
    assert ai_called is True  # AI was properly invoked after bad Bazarr candidate was rejected


# ============================================================================
# 4. Late Bazarr Target Arrival After AI Translation Starts
# ============================================================================

@pytest.mark.asyncio
async def test_late_bazarr_target_after_ai_start_rescues(mock_db_settings, tmp_path, monkeypatch):
    video_path = tmp_path / "LateArrival.mkv"
    video_path.touch()

    en_srt = tmp_path / "LateArrival.en.srt"
    en_lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nValid English line {i}" for i in range(1, 15)]
    en_srt.write_text("\n\n".join(en_lines), encoding="utf-8")

    sv_srt_path = str(tmp_path / "LateArrival.sv.srt")

    pipeline = SubtitlePipeline()
    translate_called = False

    async def mock_translate_srt(*args, **kwargs):
        nonlocal translate_called
        translate_called = True
        # Simulate Bazarr writing a healthy file during AI translation
        sv_lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nPerfekt svensk undertext rad {i}" for i in range(1, 15)]
        Path(sv_srt_path).write_text("\n\n".join(sv_lines), encoding="utf-8")
        subs = kwargs.get("subs", [])
        return [
            srt.Subtitle(index=s.index, start=s.start, end=s.end, content=f"AI rad {s.index}")
            for s in subs
        ]
    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate_srt)

    find_count = 0
    def mock_find(vp, lang):
        nonlocal find_count
        find_count += 1
        if lang == "en": return str(en_srt)
        if lang == "sv":
            if os.path.exists(sv_srt_path):
                return sv_srt_path
            return None
        return None

    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find):
        res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")

    assert res["status"] in ["translated", "bazarr_downloaded", "skipped", "completed", "bazarr match"]
    job = get_job_by_id(res["job_id"])
    assert "Bazarr candidate arrived after AI start" in "".join(job["logs"]) or "verified" in "".join(job["logs"]) or "Using verified external target" in "".join(job["logs"])


# ============================================================================
# 5. Late Bazarr Target QA Rescue (Bad AI output rescued by healthy external target)
# ============================================================================

@pytest.mark.asyncio
async def test_qa_failure_rescued_by_late_healthy_bazarr_target(mock_db_settings, tmp_path, monkeypatch):
    video_path = tmp_path / "RescueTest.mkv"
    video_path.touch()

    en_srt = tmp_path / "RescueTest.en.srt"
    en_lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nValid English line {i}" for i in range(1, 20)]
    en_srt.write_text("\n\n".join(en_lines), encoding="utf-8")

    sv_srt_path = str(tmp_path / "RescueTest.sv.srt")

    pipeline = SubtitlePipeline()

    translation_done = False

    async def mock_translate_srt(*args, **kwargs):
        nonlocal translation_done
        # AI returns empty list, simulating catastrophic QA failure / dropped lines
        translation_done = True
        return []
    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate_srt)

    async def mock_escalate(*args, **kwargs):
        return None
    monkeypatch.setattr(pipeline.translator, "escalate_single_line", mock_escalate)

    # Place a verified Swedish file on disk just before QA finishes
    sv_lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nFrisk svensk undertext rad {i}" for i in range(1, 20)]
    Path(sv_srt_path).write_text("\n\n".join(sv_lines), encoding="utf-8")

    def mock_find(vp, lang):
        if lang == "en": return str(en_srt)
        if lang == "sv":
            if translation_done:
                return sv_srt_path
            return None
        return None

    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find):
        res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")

    job = get_job_by_id(res["job_id"])
    assert job["status"] in ["BAZARR MATCH", "TRANSLATED", "COMPLETED"]
    logs = "".join(job["logs"])
    assert "Rescue verification passed" in logs or "Late candidate verified" in logs or "Using verified external target" in logs


# ============================================================================
# 6. Mid-Translation Early Stop (Stops subsequent batch scheduling safely)
# ============================================================================

@pytest.mark.asyncio
async def test_mid_translation_candidate_stops_subsequent_batches(mock_db_settings, tmp_path, monkeypatch):
    """
    Deterministic test proving that a candidate arriving during multi-batch translation
    prevents subsequent batches from being dispatched.
    """
    from app.services.translator import SubtitleTranslator

    translator = SubtitleTranslator()
    batches_executed = []

    cues = [
        srt.Subtitle(index=i, start=srt.timedelta(seconds=i), end=srt.timedelta(seconds=i+1), content=f"Source dialogue line {i}")
        for i in range(1, 101)
    ]

    candidate_file = tmp_path / "mid_trans.sv.srt"

    async def mock_translate_batch(batch, **kwargs):
        batch_ids = [item["id"] for item in batch]
        batches_executed.append(batch_ids)
        # After batch 1 executes, a healthy settled external file arrives on disk
        if len(batches_executed) == 1:
            lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nExternal human translated line {i}" for i in range(1, 101)]
            candidate_file.write_text("\n\n".join(lines), encoding="utf-8")
        await asyncio.sleep(0.05)
        return [{"id": item["id"], "text": f"Översatt {item['text']}"} for item in batch]

    monkeypatch.setattr(translator, "translate_batch", mock_translate_batch)

    async def mock_early_stop():
        if candidate_file.exists():
            return await wait_for_file_stability(str(candidate_file), min_stability_sec=0.05, timeout_sec=0.3)
        return False

    # Batch size of 20 with 100 cues = 5 batches total
    monkeypatch.setattr("app.services.translator.get_positive_int_setting", lambda k, d: 1)  # concurrency=1 to test serial early stop
    res = await translator.translate_srt_content(
        subs=cues,
        target_language="Swedish",
        batch_size=20,
        early_stop_check=mock_early_stop
    )

    # Prove that not all 5 batches ran — early stop halted subsequent batches
    assert len(batches_executed) < 5
    assert len(batches_executed) >= 1


# ============================================================================
# 7. Truthful Provenance Testing (BAZARR vs EXTERNAL)
# ============================================================================

@pytest.mark.asyncio
async def test_truthful_candidate_origin_provenance(mock_db_settings, tmp_path, monkeypatch):
    """
    Ensures that pre-existing target on disk is labeled EXTERNAL even when Bazarr is enabled,
    while late targets triggered and fetched by Bazarr are labeled BAZARR.
    """
    video_path = tmp_path / "ProvTest.mkv"
    video_path.touch()

    # Pre-existing external file on disk before pipeline runs
    sv_existing = tmp_path / "ProvTest.sv.srt"
    lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nExisterande svensk text {i}" for i in range(1, 20)]
    sv_existing.write_text("\n\n".join(lines), encoding="utf-8")

    en_existing = tmp_path / "ProvTest.en.srt"
    en_lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nExisting english text {i}" for i in range(1, 20)]
    en_existing.write_text("\n\n".join(en_lines), encoding="utf-8")

    pipeline = SubtitlePipeline()
    evaluated_origins = []

    real_eval = SubtitleTrustEngine.evaluate_candidate

    async def mock_eval(self, *args, **kwargs):
        origin = kwargs.get("origin")
        if origin:
            evaluated_origins.append(origin)
        return await real_eval(self, *args, **kwargs)

    monkeypatch.setattr(SubtitleTrustEngine, "evaluate_candidate", mock_eval)

    def mock_find(vp, lang):
        if lang == "sv": return str(sv_existing)
        if lang == "en": return str(en_existing)
        return None

    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find):
        res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")

    assert res["status"] in ["already_exists", "skipped", "completed"]
    # Pre-existing file on disk must be evaluated as EXTERNAL, not falsely claimed as BAZARR
    assert CandidateOrigin.EXTERNAL in evaluated_origins


# ============================================================================
# 8. Deterministic Quiescence & Candidate Finalization Test Cases (A - H)
# ============================================================================

@pytest.mark.asyncio
async def test_case_a_short_pause_must_not_finalize(tmp_path):
    """Case A: Generation A unchanged 50ms -> must not finalize."""
    fpath = tmp_path / "case_a.sv.srt"
    fpath.write_text("1\n00:00:01,000 --> 00:00:02,000\nHej", encoding="utf-8")

    # Require 0.2s quiescence, with timeout of 0.08s
    quiescent, snap = await wait_for_candidate_quiescence(str(fpath), quiescence_sec=0.2, timeout_sec=0.08, interval_sec=0.02)
    assert not quiescent, "50ms pause must NOT be declared quiescent when 200ms is required"


@pytest.mark.asyncio
async def test_case_b_stable_to_read_is_not_finalized(tmp_path):
    """
    Case B: Generation A unchanged 80ms (stable to read at stability=40ms)
    MUST NOT be assumed final. Writer changes to Gen B, which then becomes final.
    """
    fpath = tmp_path / "case_b.sv.srt"

    async def writer():
        # Gen A
        fpath.write_text("1\n00:00:01,000 --> 00:00:02,000\nGen A provisional", encoding="utf-8")
        await asyncio.sleep(0.08)
        # Gen B (after 80ms)
        fpath.write_text("1\n00:00:01,000 --> 00:00:02,000\nGen B finalized replacement", encoding="utf-8")

    w_task = asyncio.create_task(writer())
    # Quiescence window of 0.15s, timeout 0.6s
    quiescent, snap = await wait_for_candidate_quiescence(str(fpath), quiescence_sec=0.15, timeout_sec=0.6, interval_sec=0.02)
    await w_task

    assert quiescent is True
    assert snap.exists
    content = fpath.read_text(encoding="utf-8")
    assert "Gen B finalized replacement" in content


@pytest.mark.asyncio
async def test_case_c_sync_replacement_old_verdict_not_terminal(tmp_path, monkeypatch):
    """
    Case C: Generation A arrives -> evaluated and rejected (5 lines vs 20).
    Bazarr-style sync replaces with Generation B (20 lines).
    Old verdict must not be terminal; pipeline accepts Gen B as BAZARR MATCH.
    """
    setup_mock_settings(monkeypatch, {
        "bazarr_quiescence_seconds": "0.12",
        "bazarr_candidate_stability_seconds": "0.04",
        "bazarr_grace_seconds": "1.5",
    })

    video_path = tmp_path / "CaseC.mkv"
    video_path.touch()

    en_srt = tmp_path / "CaseC.en.srt"
    en_lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nEnglish dialogue {i}" for i in range(1, 21)]
    en_srt.write_text("\n\n".join(en_lines), encoding="utf-8")

    sv_srt_path = str(tmp_path / "CaseC.sv.srt")

    pipeline = SubtitlePipeline()
    ai_called = False

    async def mock_translate_batch(batch, **kwargs):
        nonlocal ai_called
        ai_called = True
        return [{"id": item["id"], "text": f"Svensk rad {item['id']}"} for item in batch]
    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)

    async def bazarr_lifecycle():
        # Gen A: arrives at 20ms (incomplete, 5 lines)
        await asyncio.sleep(0.02)
        sv_lines_a = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nKort rad {i}" for i in range(1, 6)]
        Path(sv_srt_path).write_text("\n\n".join(sv_lines_a), encoding="utf-8")

        # Gen B: sync/post-processing replaces at 90ms with full 20 lines
        await asyncio.sleep(0.07)
        sv_lines_b = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nKomplett svensk text {i}" for i in range(1, 21)]
        Path(sv_srt_path).write_text("\n\n".join(sv_lines_b), encoding="utf-8")

    lifecycle_task = asyncio.create_task(bazarr_lifecycle())

    def mock_find(vp, lang):
        if lang == "en": return str(en_srt)
        if lang == "sv" and os.path.exists(sv_srt_path): return sv_srt_path
        return None

    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find):
        res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")

    await lifecycle_task
    assert res["status"] == "skipped"
    assert ai_called is False
    job = get_job_by_id(res["job_id"])
    assert job["status"] == "BAZARR MATCH"


@pytest.mark.asyncio
async def test_case_d_replacement_does_not_poison_new_generation(tmp_path, monkeypatch):
    """
    Case D: Generation A evaluated and rejected.
    Replacement generation B arrives later.
    Old rejection must not poison Gen B.
    """
    setup_mock_settings(monkeypatch, {
        "bazarr_quiescence_seconds": "0.12",
        "bazarr_candidate_stability_seconds": "0.04",
        "bazarr_grace_seconds": "1.5",
    })

    video_path = tmp_path / "CaseD.mkv"
    video_path.touch()

    en_srt = tmp_path / "CaseD.en.srt"
    en_lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nEnglish line {i}" for i in range(1, 16)]
    en_srt.write_text("\n\n".join(en_lines), encoding="utf-8")

    sv_srt_path = str(tmp_path / "CaseD.sv.srt")

    pipeline = SubtitlePipeline()
    ai_called = False

    async def mock_translate_batch(batch, **kwargs):
        nonlocal ai_called
        ai_called = True
        return [{"id": item["id"], "text": f"Svensk rad {item['id']}"} for item in batch]
    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)

    async def bazarr_lifecycle():
        # Gen A: 3 lines at 20ms
        await asyncio.sleep(0.02)
        Path(sv_srt_path).write_text("1\n00:00:01,000 --> 00:00:01,800\nRad 1\n\n2\n00:00:02,000 --> 00:00:02,800\nRad 2", encoding="utf-8")

        # Gen B: 15 lines at 90ms
        await asyncio.sleep(0.07)
        full_lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nFrisk komplett text {i}" for i in range(1, 16)]
        Path(sv_srt_path).write_text("\n\n".join(full_lines), encoding="utf-8")

    lifecycle_task = asyncio.create_task(bazarr_lifecycle())

    def mock_find(vp, lang):
        if lang == "en": return str(en_srt)
        if lang == "sv" and os.path.exists(sv_srt_path): return sv_srt_path
        return None

    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find):
        res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")

    await lifecycle_task
    assert res["status"] == "skipped"
    assert ai_called is False
    job = get_job_by_id(res["job_id"])
    assert job["status"] == "BAZARR MATCH"


@pytest.mark.asyncio
async def test_case_e_gen_a_fail_then_gen_b_pass_bazarr_match(tmp_path, monkeypatch):
    """
    Case E: Gen A FAIL -> wait -> Gen B PASS -> final BAZARR MATCH -> AI 0 calls.
    """
    setup_mock_settings(monkeypatch, {
        "bazarr_quiescence_seconds": "0.12",
        "bazarr_candidate_stability_seconds": "0.04",
        "bazarr_grace_seconds": "1.5",
    })

    video_path = tmp_path / "CaseE.mkv"
    video_path.touch()

    en_srt = tmp_path / "CaseE.en.srt"
    en_lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nEng {i}" for i in range(1, 15)]
    en_srt.write_text("\n\n".join(en_lines), encoding="utf-8")

    sv_srt_path = str(tmp_path / "CaseE.sv.srt")

    pipeline = SubtitlePipeline()
    ai_dispatched = False

    async def mock_translate_batch(batch, **kwargs):
        nonlocal ai_dispatched
        ai_dispatched = True
        return [{"id": item["id"], "text": f"Svensk rad {item['id']}"} for item in batch]
    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)

    async def bazarr_lifecycle():
        # Gen A: bad candidate at 20ms
        await asyncio.sleep(0.02)
        Path(sv_srt_path).write_text("1\n00:00:01,000 --> 00:00:01,800\nBad Gen A", encoding="utf-8")

        # Gen B: healthy replacement at 90ms
        await asyncio.sleep(0.07)
        full_lines = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},800\nHealthy Gen B {i}" for i in range(1, 15)]
        Path(sv_srt_path).write_text("\n\n".join(full_lines), encoding="utf-8")

    lifecycle_task = asyncio.create_task(bazarr_lifecycle())

    def mock_find(vp, lang):
        if lang == "en": return str(en_srt)
        if lang == "sv" and os.path.exists(sv_srt_path): return sv_srt_path
        return None

    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find):
        res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")

    await lifecycle_task
    assert res["status"] == "skipped"
    assert ai_dispatched is False
    job = get_job_by_id(res["job_id"])
    assert job["status"] == "BAZARR MATCH"


@pytest.mark.asyncio
async def test_case_f_gen_a_pass_invalidated_by_bad_gen_b(tmp_path, monkeypatch):
    """
    Case F: Gen A PASS -> Gen B replaces it (incomplete/bad) before quiescence ->
    Gen A PASS is invalidated -> Gen B evaluated and fails -> AI fallback.
    """
    setup_mock_settings(monkeypatch, {
        "bazarr_quiescence_seconds": "0.15",
        "bazarr_candidate_stability_seconds": "0.03",
        "bazarr_grace_seconds": "1.5",
    })

    video_path = tmp_path / "CaseF.mkv"
    video_path.touch()

    en_srt = tmp_path / "CaseF.en.srt"
    en_lines = [f"{i}\n00:00:0{i},000 --> 00:00:0{i},800\nThis is good english line {i}" for i in range(1, 10)]
    en_srt.write_text("\n\n".join(en_lines), encoding="utf-8")

    sv_srt_path = str(tmp_path / "CaseF.sv.srt")
    # Initial Gen A is valid
    full_lines = [f"{i}\n00:00:0{i},000 --> 00:00:0{i},800\nDetta ar bra svensk rad {i}" for i in range(1, 10)]
    Path(sv_srt_path).write_text("\n\n".join(full_lines), encoding="utf-8")

    pipeline = SubtitlePipeline()
    ai_called = False

    async def mock_translate_batch(batch, **kwargs):
        nonlocal ai_called
        ai_called = True
        return [{"id": item["id"], "text": f"Detta ar oversatt svensk text {item['id']}"} for item in batch]
    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)

    async def mock_escalate(*args, **kwargs):
        return "Detta ar oversatt svensk text"
    monkeypatch.setattr(pipeline.translator, "escalate_single_line", mock_escalate)

    # While Gen A is in provisional quiescence window, writer truncates/corrupts file to bad Gen B
    async def corrupt_file():
        await asyncio.sleep(0.04)
        Path(sv_srt_path).write_text("1\n00:00:01,000 --> 00:00:01,800\nCorrupt Gen B", encoding="utf-8")

    c_task = asyncio.create_task(corrupt_file())

    def mock_find(vp, lang):
        if lang == "en": return str(en_srt)
        if lang == "sv" and os.path.exists(sv_srt_path): return sv_srt_path
        return None

    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find):
        res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")

    await c_task
    # Because Gen B was corrupt, pipeline must not have blindly accepted Gen A's early pass, and must have run AI fallback
    assert ai_called is True
    assert res["status"] in ["translated", "completed"]


@pytest.mark.asyncio
async def test_case_g_gen_a_fail_no_replacement_reaches_quiescence_ai_fallback(tmp_path, monkeypatch):
    """
    Case G: Gen A FAIL -> no replacement -> finalization/quiescence reached -> AI fallback.
    """
    setup_mock_settings(monkeypatch, {
        "bazarr_quiescence_seconds": "0.10",
        "bazarr_candidate_stability_seconds": "0.03",
        "bazarr_grace_seconds": "1.0",
    })

    video_path = tmp_path / "CaseG.mkv"
    video_path.touch()

    en_srt = tmp_path / "CaseG.en.srt"
    en_lines = [f"{i}\n00:00:0{i},000 --> 00:00:0{i},800\nThis is good english line {i}" for i in range(1, 10)]
    en_srt.write_text("\n\n".join(en_lines), encoding="utf-8")

    sv_srt_path = str(tmp_path / "CaseG.sv.srt")
    # Bad candidate that stays on disk unchanged
    Path(sv_srt_path).write_text("1\n00:00:01,000 --> 00:00:01,800\nBad Candidate Single Line", encoding="utf-8")

    pipeline = SubtitlePipeline()
    ai_called = False

    async def mock_translate_batch(batch, **kwargs):
        nonlocal ai_called
        ai_called = True
        return [{"id": item["id"], "text": f"Detta ar oversatt svensk text {item['id']}"} for item in batch]
    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)

    async def mock_escalate(*args, **kwargs):
        return "Detta ar oversatt svensk text"
    monkeypatch.setattr(pipeline.translator, "escalate_single_line", mock_escalate)

    def mock_find(vp, lang):
        if lang == "en": return str(en_srt)
        if lang == "sv" and os.path.exists(sv_srt_path): return sv_srt_path
        return None

    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find):
        res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")

    assert ai_called is True
    assert res["status"] in ["translated", "completed"]


@pytest.mark.asyncio
async def test_case_h_candidate_keeps_changing_until_deadline_ai_fallback(tmp_path, monkeypatch):
    """
    Case H: Candidate keeps changing until hard deadline -> no blind acceptance -> AI fallback.
    """
    setup_mock_settings(monkeypatch, {
        "bazarr_quiescence_seconds": "0.25",
        "bazarr_candidate_stability_seconds": "0.03",
        "bazarr_grace_seconds": "0.25",
    })

    video_path = tmp_path / "CaseH.mkv"
    video_path.touch()

    en_srt = tmp_path / "CaseH.en.srt"
    en_lines = [f"{i}\n00:00:0{i},000 --> 00:00:0{i},800\nThis is good english line {i}" for i in range(1, 10)]
    en_srt.write_text("\n\n".join(en_lines), encoding="utf-8")

    sv_srt_path = str(tmp_path / "CaseH.sv.srt")

    pipeline = SubtitlePipeline()
    ai_called = False

    async def mock_translate_batch(batch, **kwargs):
        nonlocal ai_called
        ai_called = True
        return [{"id": item["id"], "text": f"Detta ar oversatt svensk text {item['id']}"} for item in batch]
    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)

    async def mock_escalate(*args, **kwargs):
        return "Detta ar oversatt svensk text"
    monkeypatch.setattr(pipeline.translator, "escalate_single_line", mock_escalate)

    # Continuous mutations every 15ms
    stop_mutator = asyncio.Event()
    async def mutator():
        counter = 0
        while not stop_mutator.is_set():
            counter += 1
            Path(sv_srt_path).write_text(f"1\n00:00:01,000 --> 00:00:01,800\nMutating line {counter}", encoding="utf-8")
            await asyncio.sleep(0.015)

    m_task = asyncio.create_task(mutator())

    def mock_find(vp, lang):
        if lang == "en": return str(en_srt)
        if lang == "sv" and os.path.exists(sv_srt_path): return sv_srt_path
        return None

    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find):
        res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")

    stop_mutator.set()
    await m_task

    assert ai_called is True
    assert res["status"] in ["translated", "completed"]
