import asyncio
import os
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import srt
from datetime import timedelta

import app.core.db as db
from app.core.db import create_job, get_job_by_id, init_db
from app.core.trust_engine import (
    CandidateOrigin,
    SubtitleIntent,
    SubtitleTrustEngine,
    TargetSnapshot,
    TrustDecision,
    TrustResult,
    VerificationMode,
    capture_target_snapshot,
)
from app.services.pipeline import (
    SubtitlePipeline,
    _publish_subtitle_atomic,
    _publish_subtitle_with_trust_gate,
)

SAMPLE_EN_SRT = """1
00:00:01,000 --> 00:00:04,000
Captain Flint is returning to Nassau.

2
00:00:05,000 --> 00:00:08,000
The British fleet is waiting on the horizon.

3
00:00:09,000 --> 00:00:12,000
Every man must prepare his weapons now.

4
00:00:13,000 --> 00:00:16,000
We fight for freedom or we hang.

5
00:00:17,000 --> 00:00:20,000
Silver has gathered the men on deck.

6
00:00:21,000 --> 00:00:24,000
The cannons are primed and loaded.

7
00:00:25,000 --> 00:00:28,000
Take your positions immediately.

8
00:00:29,000 --> 00:00:32,000
No quarter will be given today.
"""

SAMPLE_SV_CORRECT_SRT = """1
00:00:01,000 --> 00:00:04,000
Kapten Flint återvänder till Nassau.

2
00:00:05,000 --> 00:00:08,000
Den brittiska flottan väntar vid horisonten.

3
00:00:09,000 --> 00:00:12,000
Varje man måste förbereda sina vapen nu.

4
00:00:13,000 --> 00:00:16,000
Vi kämpar för frihet eller så hängs vi.

5
00:00:17,000 --> 00:00:20,000
Silver har samlat männen på däck.

6
00:00:21,000 --> 00:00:24,000
Kanonerna är laddade och klara.

7
00:00:25,000 --> 00:00:28,000
Inta era positioner omedelbart.

8
00:00:29,000 --> 00:00:32,000
Ingen nåd kommer att ges idag.
"""

SAMPLE_SV_WRONG_EPISODE_SRT = """1
00:00:01,000 --> 00:00:04,000
Tidigare i förra säsongen av serien.

2
00:00:05,000 --> 00:00:08,000
Vi seglade mot en helt annan ö i söder.

3
00:00:09,000 --> 00:00:12,000
Detta är fel avsnitt och fel textning.

4
00:00:13,000 --> 00:00:16,000
Ingen matchning mot ljudspåret överhuvudtaget.

5
00:00:17,000 --> 00:00:20,000
Strukturen är giltig men innehållet är helt fel.

6
00:00:21,000 --> 00:00:24,000
Gamla hälsokontroller trodde detta var grönt.

7
00:00:25,000 --> 00:00:28,000
Men Trust Engine måste underkänna detta.

8
00:00:29,000 --> 00:00:32,000
Och Babel måste publicera rätt textning.
"""


@pytest.fixture(autouse=True)
def setup_tmp_db(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test_trust_gate.db")
    monkeypatch.setattr("app.core.db.DB_PATH", test_db)
    init_db()
    yield test_db


@pytest.mark.asyncio
async def test_black_sails_s04e03_wrong_target_rejected_and_babel_publishes(tmp_path):
    """
    Real reproduction test: Black Sails S04E03.
    A wrong target subtitle exists on disk with valid SRT structure (legacy health GREEN).
    Subtitle Trust Engine rejects it (FAIL: reference mismatch / wrong content).
    Two-phase publication gate backs up the wrong target and publishes QA-passed Babel subtitle.
    """
    video_path = str(tmp_path / "Black Sails - S04E03 - XXXI WEBRip-1080p.mkv")
    Path(video_path).touch()
    target_output_path = str(tmp_path / "Black Sails - S04E03 - XXXI WEBRip-1080p.sv.srt")

    # Place wrong external subtitle on disk
    Path(target_output_path).write_text(SAMPLE_SV_WRONG_EPISODE_SRT, encoding="utf-8")
    job_id = create_job(video_path)

    # Mock SubtitleTrustEngine to reject the candidate with FAIL (as in real production)
    fail_trust_result = TrustResult(
        decision=TrustDecision.FAIL,
        score=15,
        confidence="HIGH",
        reasons=["CUE_ALIGNMENT_MISMATCH", "REFERENCE_OVERLAP_LOW"],
        origin=CandidateOrigin.EXTERNAL,
        verification_mode=VerificationMode.REFERENCE,
    )

    with patch.object(SubtitleTrustEngine, "evaluate_candidate", AsyncMock(return_value=fail_trust_result)):
        pub_res = await _publish_subtitle_with_trust_gate(
            video_path=video_path,
            target_output_path=target_output_path,
            lang_code="sv",
            translated_srt_text=SAMPLE_SV_CORRECT_SRT,
            expected_cue_count=8,
            force_retranslate=False,
            job_id=job_id,
        )

    # Must be published
    assert pub_res["published"] is True
    assert pub_res["skipped"] is False
    assert pub_res["reason"] == "published"

    # Verify target file contains Babel's correct Swedish translation
    published_content = Path(target_output_path).read_text(encoding="utf-8")
    assert "Kapten Flint återvänder till Nassau." in published_content
    assert "Tidigare i förra säsongen" not in published_content

    # Verify rejected target was backed up
    parent_files = [f.name for f in tmp_path.iterdir()]
    backup_files = [f for f in parent_files if "babel-replaced" in f]
    assert len(backup_files) == 1
    backup_path = tmp_path / backup_files[0]
    assert "Tidigare i förra säsongen" in backup_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_late_good_target_authoritative_pass_preserves_external(tmp_path):
    """
    Late good target arrives and passes SubtitleTrustEngine authoritative evaluation (score >= 70).
    Publication gate detects authoritative PASS, preserves the external file, and skips publishing Babel output.
    """
    video_path = str(tmp_path / "Episode.mkv")
    Path(video_path).touch()
    target_output_path = str(tmp_path / "Episode.sv.srt")
    Path(target_output_path).write_text(SAMPLE_SV_CORRECT_SRT, encoding="utf-8")
    job_id = create_job(video_path)

    pass_trust_result = TrustResult(
        decision=TrustDecision.PASS,
        score=95,
        confidence="HIGH",
        reasons=["EXACT_CUE_REFERENCE_MATCH"],
        origin=CandidateOrigin.EXTERNAL,
        verification_mode=VerificationMode.REFERENCE,
    )

    with patch.object(SubtitleTrustEngine, "evaluate_candidate", AsyncMock(return_value=pass_trust_result)):
        pub_res = await _publish_subtitle_with_trust_gate(
            video_path=video_path,
            target_output_path=target_output_path,
            lang_code="sv",
            translated_srt_text=SAMPLE_SV_CORRECT_SRT,
            expected_cue_count=8,
            force_retranslate=False,
            job_id=job_id,
        )

    # Must be skipped, not published
    assert pub_res["published"] is False
    assert pub_res["skipped"] is True
    assert pub_res["reason"] == "authoritative_target_passed"

    # No backup files created
    parent_files = [f.name for f in tmp_path.iterdir()]
    backup_files = [f for f in parent_files if "babel-replaced" in f]
    assert len(backup_files) == 0


@pytest.mark.asyncio
async def test_toctou_mutation_detected_and_revalidated(tmp_path):
    """
    TOCTOU safety: external file mutates between Trust preflight and atomic compare-and-publish.
    Gate detects snapshot mismatch (target_mutated) and loops back for revalidation.
    """
    video_path = str(tmp_path / "Episode2.mkv")
    Path(video_path).touch()
    target_output_path = str(tmp_path / "Episode2.sv.srt")
    Path(target_output_path).write_text(SAMPLE_SV_WRONG_EPISODE_SRT, encoding="utf-8")
    job_id = create_job(video_path)

    call_count = 0
    fail_trust_result = TrustResult(
        decision=TrustDecision.FAIL,
        score=20,
        confidence="HIGH",
        reasons=["BAD_ALIGNMENT"],
        origin=CandidateOrigin.EXTERNAL,
        verification_mode=VerificationMode.REFERENCE,
    )

    async def dynamic_evaluate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return fail_trust_result

    # First attempt: simulate a race by mutating file between snapshot and publish
    original_publish_atomic = _publish_subtitle_atomic
    atomic_call_count = 0

    def simulated_race_publish(*args, **kwargs):
        nonlocal atomic_call_count
        atomic_call_count += 1
        if atomic_call_count == 1:
            # Simulate mutation by changing mtime/size
            time.sleep(0.01)
            Path(target_output_path).write_text(SAMPLE_SV_WRONG_EPISODE_SRT + "\n# mutated", encoding="utf-8")
        return original_publish_atomic(*args, **kwargs)

    with patch.object(SubtitleTrustEngine, "evaluate_candidate", side_effect=dynamic_evaluate), \
         patch("app.services.pipeline._publish_subtitle_atomic", side_effect=simulated_race_publish):
        pub_res = await _publish_subtitle_with_trust_gate(
            video_path=video_path,
            target_output_path=target_output_path,
            lang_code="sv",
            translated_srt_text=SAMPLE_SV_CORRECT_SRT,
            expected_cue_count=8,
            force_retranslate=False,
            job_id=job_id,
        )

    # After mutation detection and retry, publish succeeded
    assert pub_res["published"] is True
    assert call_count >= 2
    assert "Kapten Flint återvänder" in Path(target_output_path).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_unknown_unstable_candidate_fails_closed(tmp_path):
    """
    Candidate evaluation yields UNKNOWN (unverifiable without reference).
    Gate refuses to blindly overwrite or blindly accept; fails closed.
    """
    video_path = str(tmp_path / "Episode3.mkv")
    Path(video_path).touch()
    target_output_path = str(tmp_path / "Episode3.sv.srt")
    Path(target_output_path).write_text(SAMPLE_SV_WRONG_EPISODE_SRT, encoding="utf-8")
    job_id = create_job(video_path)

    unknown_trust_result = TrustResult(
        decision=TrustDecision.UNKNOWN,
        score=50,
        confidence="LOW",
        reasons=["NO_REFERENCE_AVAILABLE"],
        origin=CandidateOrigin.EXTERNAL,
        verification_mode=VerificationMode.STANDALONE,
    )

    with patch.object(SubtitleTrustEngine, "evaluate_candidate", AsyncMock(return_value=unknown_trust_result)):
        pub_res = await _publish_subtitle_with_trust_gate(
            video_path=video_path,
            target_output_path=target_output_path,
            lang_code="sv",
            translated_srt_text=SAMPLE_SV_CORRECT_SRT,
            expected_cue_count=8,
            force_retranslate=False,
            job_id=job_id,
            max_conflict_retries=2,
        )

    assert pub_res["published"] is False
    assert pub_res["skipped"] is False
    assert pub_res["reason"] == "target_unverified_conflict"


@pytest.mark.asyncio
async def test_legacy_health_cannot_bypass_without_authoritative_trust_gate(tmp_path):
    """
    Verify that _publish_subtitle_atomic with default allow_legacy_health=False
    rejects unverified target conflicts and requires snapshot-aware trust decisions.
    """
    video_path = str(tmp_path / "Episode4.mkv")
    Path(video_path).touch()
    target_output_path = str(tmp_path / "Episode4.sv.srt")
    Path(target_output_path).write_text(SAMPLE_SV_WRONG_EPISODE_SRT, encoding="utf-8")
    job_id = create_job(video_path)

    # Calling _publish_subtitle_atomic without trust_gate_snapshot and allow_legacy_health=False
    res = _publish_subtitle_atomic(
        video_path=video_path,
        target_output_path=target_output_path,
        lang_code="sv",
        translated_srt_text=SAMPLE_SV_CORRECT_SRT,
        expected_cue_count=8,
        force_retranslate=False,
        job_id=job_id,
        allow_legacy_health=False,
    )

    # Must NOT skip as "existing_healthy"
    assert res.get("reason") != "existing_healthy"
    assert res.get("published") is False
    assert res.get("reason") == "target_mutated"
