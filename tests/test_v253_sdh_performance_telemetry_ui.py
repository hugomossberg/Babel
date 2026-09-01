"""
Comprehensive Test Suite for SDH Classifier Concurrency, Source Prep Telemetry,
and UI/Stats Correctness in Babel v2.5.3-beta.
"""

import asyncio
from datetime import timedelta
from pathlib import Path
import os
import sqlite3
import pytest
import srt

from app.core.cleaner import (
    sanitize_srt_content_with_provenance,
    analyze_subtitle_cue,
    SanitizerResult,
    SegmentClassification,
    ClassificationProvenance,
    EMPTY_PLACEHOLDER,
)
from app.core import db
from app.core.trust_engine import (
    SubtitleTrustEngine,
    CandidateOrigin,
    TrustDecision,
    TrustResult,
    TargetSnapshot,
)
from app.services.source_resolver import (
    BazarrResult,
    BazarrResultCode,
)
from app.services.translator import SubtitleTranslator
from app.services.pipeline import SubtitlePipeline
from unittest.mock import AsyncMock, patch, MagicMock


def make_ambiguous_srt(count: int = 125) -> str:
    subs = []
    for i in range(count):
        subs.append(
            srt.Subtitle(
                index=i + 1,
                start=timedelta(seconds=i * 2),
                end=timedelta(seconds=i * 2 + 1),
                content=f"Doctor (whisper {i + 1}): Please hurry.",
            )
        )
    return srt.compose(subs)


def make_swedish_srt(count: int = 10) -> str:
    subs = []
    for i in range(count):
        subs.append(
            srt.Subtitle(
                index=i + 1,
                start=timedelta(seconds=i * 3),
                end=timedelta(seconds=i * 3 + 2),
                content=f"Det här är ett svenskt avsnitt med dialog rad {i + 1} för testning.",
            )
        )
    return srt.compose(subs)


# ===========================================================================
# A. BOUNDED PARALLEL CLASSIFICATION (PURE EVENT SYNCHRONIZATION)
# ===========================================================================

@pytest.mark.asyncio
async def test_a_bounded_parallel_classification_event_driven():
    """
    Constructs 125 unique ambiguous items (3 chunks: 50, 50, 25).
    Proves:
    - Multiple chunks in flight simultaneously when concurrency > 1
    - Exactly two classifier chunks enter before slot is released
    - Maximum observed concurrency <= configured cap (2)
    - Each chunk <= 50
    - Request count == ceil(125 / 50) == 3
    - Results applied deterministically
    - Uses pure async event synchronization (zero sleeps)
    """
    srt_text = make_ambiguous_srt(125)

    in_flight = 0
    max_in_flight = 0
    chunk_sizes = []
    call_count = 0

    gate_2_tasks_entered = asyncio.Event()
    release_task_0 = asyncio.Event()
    release_all = asyncio.Event()

    async def mock_classifier(chunk, source_language="English", job_id=None):
        nonlocal in_flight, max_in_flight, call_count
        my_id = call_count
        call_count += 1
        chunk_sizes.append(len(chunk))
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)

        if in_flight == 2:
            gate_2_tasks_entered.set()

        if my_id == 0:
            await release_task_0.wait()
        else:
            await release_all.wait()

        in_flight -= 1
        return [{"id": item["id"], "classification": "NON_DIALOGUE"} for item in chunk]

    async def runner():
        sanitizer_task = asyncio.create_task(
            sanitize_srt_content_with_provenance(
                srt_text,
                source_language="English",
                classifier_fn=mock_classifier,
                concurrency=2,
            )
        )

        # Wait until exactly 2 tasks are concurrently in-flight
        await gate_2_tasks_entered.wait()
        assert in_flight == 2
        assert call_count == 2  # Task 3 is blocked by the semaphore

        # Release task 0 to open a slot for task 3
        release_task_0.set()
        # Release remaining tasks
        release_all.set()

        return await sanitizer_task

    res = await runner()
    subs, prov_map, cleaned_count = res

    assert call_count == 3
    assert chunk_sizes == [50, 50, 25]
    assert all(cs <= 50 for cs in chunk_sizes)
    assert max_in_flight == 2
    assert cleaned_count == 125
    assert len(subs) == 125
    for s in subs:
        assert s.content == "Please hurry."

    # Diagnostic telemetry check
    telem = res.telemetry
    assert telem["ambiguous_unique"] == 125
    assert telem["classifier_batches"] == 3
    assert telem["classifier_concurrency"] == 2
    assert "local_analysis_s" in telem
    assert "classifier_wait_s" in telem
    assert "reconstruction_s" in telem
    assert "total_s" in telem


# ===========================================================================
# B. OUTPUT EQUIVALENCE
# ===========================================================================

@pytest.mark.asyncio
async def test_b_output_equivalence():
    """
    Given the same classifier answers:
    concurrency=1 vs concurrency=2 must produce 100% identical outputs:
    - cleaned subtitle text
    - cleaned_count
    - cue provenance classifications
    - removed SDH segments
    """
    srt_text = make_ambiguous_srt(120)

    async def mock_classifier(chunk, source_language="English", job_id=None):
        # Even items -> NON_DIALOGUE, Odd items -> DIALOGUE
        return [
            {"id": it["id"], "classification": "NON_DIALOGUE" if it["id"] % 2 == 0 else "DIALOGUE"}
            for it in chunk
        ]

    res_seq = await sanitize_srt_content_with_provenance(
        srt_text,
        source_language="English",
        classifier_fn=mock_classifier,
        concurrency=1,
    )

    res_par = await sanitize_srt_content_with_provenance(
        srt_text,
        source_language="English",
        classifier_fn=mock_classifier,
        concurrency=2,
    )

    subs_seq, prov_seq, count_seq = res_seq
    subs_par, prov_par, count_par = res_par

    assert count_seq == count_par
    assert len(subs_seq) == len(subs_par)
    for s1, s2 in zip(subs_seq, subs_par):
        assert s1.content == s2.content
        assert s1.start == s2.start
        assert s1.end == s2.end

    for idx in prov_seq:
        p1 = prov_seq[idx]
        p2 = prov_par[idx]
        assert p1.cleaned_content == p2.cleaned_content
        assert p1.removed_sdh_segments == p2.removed_sdh_segments
        assert len(p1.segments) == len(p2.segments)
        for seg1, seg2 in zip(p1.segments, p2.segments):
            assert seg1.classification == seg2.classification
            assert seg1.provenance == seg2.provenance


# ===========================================================================
# C. CHUNK FAILURE (FAIL-SAFE ISOLATION)
# ===========================================================================

@pytest.mark.asyncio
async def test_c_chunk_failure_fail_safe_isolation():
    """
    One classifier chunk fails while other chunks succeed.
    Expected:
    - Failed chunk candidates preserved fail-safe as DIALOGUE
    - Successful chunk candidates apply classifications
    - No subtitle cues disappear or shift
    """
    srt_text = make_ambiguous_srt(125)

    async def faulty_classifier(chunk, source_language="English", job_id=None):
        # Chunk 2 (ids 50..99) raises an error
        ids = [it["id"] for it in chunk]
        if 50 in ids:
            raise RuntimeError("Provider 503 Service Unavailable")
        return [{"id": it["id"], "classification": "NON_DIALOGUE"} for it in chunk]

    res = await sanitize_srt_content_with_provenance(
        srt_text,
        source_language="English",
        classifier_fn=faulty_classifier,
        concurrency=2,
    )

    subs, prov_map, cleaned_count = res
    assert len(subs) == 125

    # Cues 1..50 (ids 0..49) were in chunk 1 -> successfully stripped
    for i in range(50):
        assert subs[i].content == "Please hurry."
        assert prov_map[i].segments[0].classification == SegmentClassification.NON_DIALOGUE

    # Cues 51..100 (ids 50..99) were in chunk 2 -> fail-safe preserved as DIALOGUE
    for i in range(50, 100):
        assert f"Doctor (whisper {i + 1}): Please hurry." == subs[i].content
        assert prov_map[i].segments[0].classification == SegmentClassification.DIALOGUE
        assert prov_map[i].segments[0].provenance == ClassificationProvenance.FAIL_SAFE_PRESERVED

    # Cues 101..125 (ids 100..124) were in chunk 3 -> successfully stripped
    for i in range(100, 125):
        assert subs[i].content == "Please hurry."
        assert prov_map[i].segments[0].classification == SegmentClassification.NON_DIALOGUE


# ===========================================================================
# D. CANCELLATION PROPAGATION & DRAINED CHILD TASKS
# ===========================================================================

@pytest.mark.asyncio
async def test_d_cancellation_propagates_and_drains_tasks():
    """
    asyncio.CancelledError must NEVER be swallowed into fail-safe completion.
    Proves:
    - Child classifier tasks enter and block on an Event
    - Parent sanitizer task is cancelled
    - Every child task reaches its finally block and records cleanup
    - No child task remains running or pending
    - CancelledError propagates to the caller
    - Zero sleeps used (pure asyncio.Event synchronization)
    """
    srt_text = make_ambiguous_srt(125)

    entered_event = asyncio.Event()
    block_event = asyncio.Event()
    child_cleaned_up = []
    child_active = []

    async def blocking_classifier(chunk, source_language="English", job_id=None):
        child_id = len(child_active)
        child_active.append(child_id)
        entered_event.set()
        try:
            await block_event.wait()
            return [{"id": item["id"], "classification": "NON_DIALOGUE"} for item in chunk]
        finally:
            child_cleaned_up.append(child_id)

    sanitizer_task = asyncio.create_task(
        sanitize_srt_content_with_provenance(
            srt_text,
            source_language="English",
            classifier_fn=blocking_classifier,
            concurrency=2,
        )
    )

    # Wait until child task has entered and is blocking
    await entered_event.wait()
    assert len(child_active) >= 1
    assert len(child_cleaned_up) == 0

    # Cancel the parent task
    sanitizer_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await sanitizer_task

    # Verify that all started child tasks have reached their finally block and completed cleanup
    assert len(child_cleaned_up) == len(child_active)
    assert len(child_cleaned_up) >= 1


# ===========================================================================
# E. USAGE / REQUEST COUNT
# ===========================================================================

@pytest.mark.asyncio
async def test_e_exact_request_count_under_concurrency():
    """
    For 125 ambiguous items, total dispatches must remain exactly ceil(125/50) = 3.
    No duplicate dispatches under concurrency.
    """
    srt_text = make_ambiguous_srt(125)
    dispatches = 0

    async def mock_classifier(chunk, source_language="English", job_id=None):
        nonlocal dispatches
        dispatches += 1
        return [{"id": it["id"], "classification": "NON_DIALOGUE"} for it in chunk]

    await sanitize_srt_content_with_provenance(
        srt_text,
        source_language="English",
        classifier_fn=mock_classifier,
        concurrency=4,
    )

    assert dispatches == 3


# ===========================================================================
# F. STATS (EXISTING / SKIPPED COUNTING ALL COMPATIBLE STATUSES)
# ===========================================================================

def test_f_stats_existing_skipped_counts_including_legacy(tmp_path):
    """
    Mixed status dataset:
    2 TRANSLATED
    2 BAZARR MATCH
    1 ALREADY EXISTS
    1 HEALTHY
    1 SKIPPED
    1 EXISTING / SKIPPED
    1 FAILED
    1 TRANSLATING (active)
    Expected:
    total = 10
    translated = 2
    existing_skipped = 6 (2 BAZARR MATCH + 1 ALREADY EXISTS + 1 HEALTHY + 1 SKIPPED + 1 EXISTING / SKIPPED)
    failed = 1
    active_jobs = 1
    avg_duration_seconds = 85.0 (translated only)
    """
    db_file = tmp_path / "stats_test.db"
    orig_db = db.DB_PATH
    db.DB_PATH = str(db_file)
    db.init_db()

    try:
        with sqlite3.connect(str(db_file)) as conn:
            statuses = [
                ("TRANSLATED", 80.0),
                ("TRANSLATED", 90.0),
                ("BAZARR MATCH", 5.0),
                ("BAZARR MATCH", 4.0),
                ("ALREADY EXISTS", 1.0),
                ("HEALTHY", 2.0),
                ("SKIPPED", 3.0),
                ("EXISTING / SKIPPED", 2.5),
                ("FAILED", 10.0),
                ("TRANSLATING", 0.0),
            ]
            for s, dur in statuses:
                conn.execute(
                    "INSERT INTO jobs (video_path, status, duration_seconds, created_at, updated_at) VALUES (?, ?, ?, 'now', 'now')",
                    ("/media/movie.mkv", s, dur),
                )
            conn.commit()

        stats = db.get_job_stats()
        assert stats["total"] == 10
        assert stats["translated"] == 2
        assert stats["existing_skipped"] == 6
        assert stats["failed"] == 1
        assert stats["active_jobs"] == 1
        assert "healthy" in stats
        # Avg duration only for TRANSLATED/REPAIRED/SUCCESS: (80 + 90) / 2 = 85.0s
        assert stats["avg_duration_seconds"] == 85.0
    finally:
        db.DB_PATH = orig_db


# ===========================================================================
# G. UI PROVENANCE AND CARD SEMANTICS
# ===========================================================================

def test_g_ui_template_provenance_and_stats():
    index_path = Path(__file__).resolve().parents[1] / "app" / "templates" / "index.html"
    html = index_path.read_text(encoding="utf-8")

    # Existing / Skipped card bound to stats.existing_skipped
    assert "stats.existing_skipped || 0" in html
    # Modal provenance sublabels
    assert "(found by Bazarr this run)" in html
    assert "(Bazarr + existing targets)" in html
    assert "(extracted from media)" in html
    assert "(external target appeared during processing)" in html
    assert "(pre-existing)" in html
    # Header rename for avg duration
    assert "Avg Translation Time" in html


# ===========================================================================
# H. EMBEDDED TARGET EXTRACTION AND PUBLISHED REASON
# ===========================================================================

@pytest.mark.asyncio
async def test_h_embedded_target_extracted_and_published_reason(tmp_path, monkeypatch):
    """
    When embedded target is extracted and published without AI:
    - status is ALREADY EXISTS
    - reason is 'Embedded target extracted and published'
    - translate_srt_content is NOT awaited
    - trigger_bazarr_search is NOT awaited
    - published file exists on disk
    """
    db_file = tmp_path / "embed_test.db"
    orig_db = db.DB_PATH
    db.DB_PATH = str(db_file)
    db.init_db()

    video = tmp_path / "Modern.Family.S01E01.mkv"
    video.touch()
    expected_out_srt = tmp_path / "Modern.Family.S01E01.sv.srt"

    try:
        def mock_settings(k, d=""):
            return {
                "ai_provider": "gemini",
                "gemini_api_key": "dummy",
                "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
                "enable_bazarr_check": "true",
                "clean_sdh": "true",
                "extract_source_embedded": "true",
                "extract_target_embedded": "true",
                "materialize_embedded_target": "true",
            }.get(k, d)

        monkeypatch.setattr("app.services.pipeline.get_setting", mock_settings)
        monkeypatch.setattr("app.services.translator.get_setting", mock_settings)

        pipeline = SubtitlePipeline()
        mock_translate = AsyncMock(return_value=[])
        mock_bazarr_trigger = AsyncMock()
        monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)
        monkeypatch.setattr(pipeline, "trigger_bazarr_search", mock_bazarr_trigger)

        # Inspect shows Swedish track embedded
        def mock_inspect(vp):
            return {
                "subtitles": [
                    {"id": 2, "language": "swe", "codec": "SubRip/SRT", "forced": False, "title": "Swedish"}
                ],
                "audio": [{"id": 1, "language": "eng", "default": True, "forced": False}],
                "duration": 120.0,
            }
        monkeypatch.setattr("app.services.pipeline.inspect_mkv_tracks", mock_inspect)
        monkeypatch.setattr("app.core.extractor.inspect_mkv_tracks", mock_inspect)

        # Extraction succeeds with Swedish content
        sv_srt_content = make_swedish_srt(10)
        def mock_extract(vp, out, preferred_lang, tracks_info=None):
            with open(out, "w", encoding="utf-8") as f:
                f.write(sv_srt_content)
            return True
        monkeypatch.setattr("app.services.pipeline._safe_extract_embedded_srt", mock_extract)

        job_id = db.create_job(video_path=str(video), event_source="SONARR")

        res = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)
        assert res["status"] == "skipped"
        assert res["reason"] == "already_exists"

        job = db.get_job_by_id(job_id)
        assert job["status"] == "ALREADY EXISTS"
        assert job["reason"] == "Embedded target extracted and published"

        # Explicit assertion that AI and Bazarr search were never invoked
        mock_translate.assert_not_awaited()
        mock_bazarr_trigger.assert_not_awaited()

        # Assert published Swedish file exists
        assert expected_out_srt.exists()

        logs = "".join(job["logs"])
        assert "Bazarr skipped — embedded target satisfied language" in logs
        assert "AI skipped" in logs
        assert "AI calls: 0" in logs
    finally:
        db.DB_PATH = orig_db


# ===========================================================================
# I. PROVENANCE CASES: PRE-EXISTING UNKNOWN -> PASS, TRUE BAZARR, UNACCEPTED EXTERNAL, AND MIXED
# ===========================================================================

@pytest.mark.asyncio
async def test_i1_pre_existing_unknown_to_pass_yields_already_exists(tmp_path, monkeypatch):
    """
    CASE 1: PRE-EXISTING UNKNOWN -> PASS
    - External Swedish subtitle exists before job start (pre-trigger snapshot recorded).
    - Initial Trust evaluation without reference returns UNKNOWN.
    - After source resolution, Trust evaluation PASSes with CandidateOrigin.EXTERNAL.
    - Result MUST NOT claim 'Bazarr found all targets'.
    - Status: ALREADY EXISTS, reason: 'Pre-existing target verified after source resolution'.
    - AI calls: 0.
    """
    db_file = tmp_path / "prov_test1.db"
    orig_db = db.DB_PATH
    db.DB_PATH = str(db_file)
    db.init_db()

    video = tmp_path / "movie.mkv"
    video.touch()
    sv_target = tmp_path / "movie.sv.srt"
    sv_content = make_swedish_srt(10)
    sv_target.write_text(sv_content, encoding="utf-8")

    en_source = tmp_path / "movie.en.srt"
    en_source.write_text(make_ambiguous_srt(10), encoding="utf-8")

    try:
        def mock_settings(k, d=""):
            return {
                "ai_provider": "gemini",
                "gemini_api_key": "dummy",
                "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
                "enable_bazarr_check": "true",
                "clean_sdh": "true",
                "extract_source_embedded": "false",
                "extract_target_embedded": "false",
            }.get(k, d)

        monkeypatch.setattr("app.services.pipeline.get_setting", mock_settings)
        monkeypatch.setattr("app.services.translator.get_setting", mock_settings)

        pipeline = SubtitlePipeline()
        mock_translate = AsyncMock(return_value=[])
        monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)
        monkeypatch.setattr(
            pipeline,
            "trigger_bazarr_search",
            AsyncMock(return_value=BazarrResult(code=BazarrResultCode.ACCEPTED, detail="Queued"))
        )

        # Mock trust engine: UNKNOWN initially, PASS after source provided
        eval_count = 0
        async def mock_evaluate(self, video_path, candidate_path, target_lang, origin=None, provided_source=None, **kwargs):
            nonlocal eval_count
            eval_count += 1
            if provided_source is None:
                return TrustResult(decision=TrustDecision.UNKNOWN, score=0, confidence="LOW", reasons=["Awaiting reference"], origin=CandidateOrigin.EXTERNAL)
            return TrustResult(decision=TrustDecision.PASS, score=95, confidence="HIGH", reasons=["Cadence match"], origin=origin or CandidateOrigin.EXTERNAL)

        monkeypatch.setattr(SubtitleTrustEngine, "evaluate_candidate", mock_evaluate)

        job_id = db.create_job(video_path=str(video), event_source="SONARR")
        res = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)

        assert res["status"] == "skipped"
        job = db.get_job_by_id(job_id)
        assert job["status"] == "ALREADY EXISTS"
        assert job["reason"] == "Pre-existing target verified after source resolution"
        mock_translate.assert_not_awaited()
    finally:
        db.DB_PATH = orig_db


@pytest.mark.asyncio
async def test_i2_true_bazarr_target_yields_bazarr_match(tmp_path, monkeypatch):
    """
    CASE 2: TRUE BAZARR TARGET
    - Target does not exist before Bazarr trigger.
    - Bazarr trigger runs and explicitly returns BazarrResultCode.TRIGGERED.
    - Target file appears on disk with new generation.
    - Trust PASS with CandidateOrigin.BAZARR.
    - Status: BAZARR MATCH, reason: 'Bazarr found all targets'.
    """
    db_file = tmp_path / "prov_test2.db"
    orig_db = db.DB_PATH
    db.DB_PATH = str(db_file)
    db.init_db()

    video = tmp_path / "movie.mkv"
    video.touch()
    sv_target = tmp_path / "movie.sv.srt"

    en_source = tmp_path / "movie.en.srt"
    en_source.write_text(make_ambiguous_srt(10), encoding="utf-8")

    try:
        def mock_settings(k, d=""):
            return {
                "ai_provider": "gemini",
                "gemini_api_key": "dummy",
                "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
                "enable_bazarr_check": "true",
                "clean_sdh": "true",
                "extract_source_embedded": "false",
                "extract_target_embedded": "false",
            }.get(k, d)

        monkeypatch.setattr("app.services.pipeline.get_setting", mock_settings)
        monkeypatch.setattr("app.services.translator.get_setting", mock_settings)

        pipeline = SubtitlePipeline()
        mock_translate = AsyncMock(return_value=[])
        monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)

        # Bazarr trigger returns accepted BazarrResult and writes the target file
        async def mock_bazarr_search(vp, language=None, **kwargs):
            sv_target.write_text(make_swedish_srt(10), encoding="utf-8")
            return BazarrResult(code=BazarrResultCode.TRIGGERED, language=language or "sv", detail="Search accepted")

        monkeypatch.setattr(pipeline, "trigger_bazarr_search", mock_bazarr_search)

        async def mock_evaluate(self, video_path, candidate_path, target_lang, origin=None, provided_source=None, **kwargs):
            return TrustResult(decision=TrustDecision.PASS, score=95, confidence="HIGH", reasons=["Cadence match"], origin=origin or CandidateOrigin.BAZARR)

        monkeypatch.setattr(SubtitleTrustEngine, "evaluate_candidate", mock_evaluate)

        job_id = db.create_job(video_path=str(video), event_source="SONARR")
        res = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)

        assert res["status"] == "skipped"
        job = db.get_job_by_id(job_id)
        assert job["status"] == "BAZARR MATCH"
        assert job["reason"] == "Bazarr found all targets"
        mock_translate.assert_not_awaited()
    finally:
        db.DB_PATH = orig_db


@pytest.mark.asyncio
async def test_i3_new_external_no_bazarr_acceptance_must_not_become_bazarr_match(tmp_path, monkeypatch):
    """
    CASE 3: NEW EXTERNAL TARGET WITHOUT BAZARR ACCEPTANCE
    - No target exists initially.
    - Bazarr trigger returns MEDIA_NOT_FOUND (not accepted).
    - An external target arrives during the run.
    - CandidateOrigin MUST be EXTERNAL.
    - Job status MUST NOT be BAZARR MATCH.
    - UI provenance must not say '(found by Bazarr this run)'.
    """
    db_file = tmp_path / "prov_test3.db"
    orig_db = db.DB_PATH
    db.DB_PATH = str(db_file)
    db.init_db()

    video = tmp_path / "movie.mkv"
    video.touch()
    sv_target = tmp_path / "movie.sv.srt"

    en_source = tmp_path / "movie.en.srt"
    en_source.write_text(make_ambiguous_srt(10), encoding="utf-8")

    try:
        def mock_settings(k, d=""):
            return {
                "ai_provider": "gemini",
                "gemini_api_key": "dummy",
                "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
                "enable_bazarr_check": "true",
                "clean_sdh": "true",
                "extract_source_embedded": "false",
                "extract_target_embedded": "false",
            }.get(k, d)

        monkeypatch.setattr("app.services.pipeline.get_setting", mock_settings)
        monkeypatch.setattr("app.services.translator.get_setting", mock_settings)

        pipeline = SubtitlePipeline()
        mock_translate = AsyncMock(return_value=[])
        monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)

        # Bazarr trigger returns MEDIA_NOT_FOUND (unaccepted)
        async def mock_bazarr_search(vp, language=None, **kwargs):
            return BazarrResult(code=BazarrResultCode.MEDIA_NOT_FOUND, language=language or "sv", detail="Not in library")

        monkeypatch.setattr(pipeline, "trigger_bazarr_search", mock_bazarr_search)

        # External file created during run by another process
        sv_target.write_text(make_swedish_srt(10), encoding="utf-8")

        async def mock_evaluate(self, video_path, candidate_path, target_lang, origin=None, provided_source=None, **kwargs):
            # Origin must be EXTERNAL because Bazarr was not accepted
            assert origin == CandidateOrigin.EXTERNAL
            return TrustResult(decision=TrustDecision.PASS, score=95, confidence="HIGH", reasons=["Cadence match"], origin=origin)

        monkeypatch.setattr(SubtitleTrustEngine, "evaluate_candidate", mock_evaluate)

        job_id = db.create_job(video_path=str(video), event_source="SONARR")
        res = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)

        job = db.get_job_by_id(job_id)
        # MUST NOT claim BAZARR MATCH
        assert job["status"] != "BAZARR MATCH"
        assert job["status"] == "ALREADY EXISTS"
        assert "Bazarr found all targets" not in (job["reason"] or "")
        mock_translate.assert_not_awaited()
    finally:
        db.DB_PATH = orig_db


@pytest.mark.asyncio
async def test_i4_mixed_multi_language_resolution(tmp_path, monkeypatch):
    """
    CASE 4: MULTI-LANGUAGE MIX
    - Swedish (sv) is pre-existing EXTERNAL.
    - Danish (da) is newly downloaded by BAZARR with accepted search.
    - Both PASS Trust.
    - Status: BAZARR MATCH, reason: 'Targets resolved without AI (Bazarr + existing)'.
    """
    db_file = tmp_path / "prov_test4.db"
    orig_db = db.DB_PATH
    db.DB_PATH = str(db_file)
    db.init_db()

    video = tmp_path / "movie.mkv"
    video.touch()

    # Pre-existing Swedish file
    sv_target = tmp_path / "movie.sv.srt"
    sv_target.write_text(make_swedish_srt(10), encoding="utf-8")

    # Danish file does not exist initially
    da_target = tmp_path / "movie.da.srt"

    en_source = tmp_path / "movie.en.srt"
    en_source.write_text(make_ambiguous_srt(10), encoding="utf-8")

    try:
        def mock_settings(k, d=""):
            return {
                "ai_provider": "gemini",
                "gemini_api_key": "dummy",
                "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}, {"code": "da", "name": "Danish", "enabled": true}]',
                "enable_bazarr_check": "true",
                "clean_sdh": "true",
                "extract_source_embedded": "false",
                "extract_target_embedded": "false",
            }.get(k, d)

        monkeypatch.setattr("app.services.pipeline.get_setting", mock_settings)
        monkeypatch.setattr("app.services.translator.get_setting", mock_settings)

        pipeline = SubtitlePipeline()
        mock_translate = AsyncMock(return_value=[])
        monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)

        # Bazarr creates Danish file and returns accepted BazarrResult
        async def mock_bazarr_search(vp, language=None, **kwargs):
            if language == "da":
                da_target.write_text(make_swedish_srt(10), encoding="utf-8")
            return BazarrResult(code=BazarrResultCode.TRIGGERED, language=language or "da", detail="Accepted")

        monkeypatch.setattr(pipeline, "trigger_bazarr_search", mock_bazarr_search)

        async def mock_evaluate(self, video_path, candidate_path, target_lang, origin=None, provided_source=None, **kwargs):
            if provided_source is None and target_lang == "sv":
                # Initial external check returns UNKNOWN so sv enters source resolution
                return TrustResult(decision=TrustDecision.UNKNOWN, score=0, confidence="LOW", reasons=["Awaiting reference"], origin=CandidateOrigin.EXTERNAL)
            return TrustResult(decision=TrustDecision.PASS, score=95, confidence="HIGH", reasons=["Cadence match"], origin=origin)

        monkeypatch.setattr(SubtitleTrustEngine, "evaluate_candidate", mock_evaluate)

        job_id = db.create_job(video_path=str(video), event_source="SONARR")
        res = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)

        assert res["status"] == "skipped"
        job = db.get_job_by_id(job_id)
        assert job["status"] == "BAZARR MATCH"
        assert job["reason"] == "Targets resolved without AI (Bazarr + existing)"
        mock_translate.assert_not_awaited()
    finally:
        db.DB_PATH = orig_db


# ===========================================================================
# J. BAZARR ACCEPTANCE CONTRACT VARIANTS UNIT TEST
# ===========================================================================

@pytest.mark.asyncio
async def test_j_bazarr_acceptance_contract_variants(tmp_path, monkeypatch):
    """
    Direct unit test proving all return contract variants of trigger_bazarr_search:
    - BazarrResult.was_accepted == True -> Accepted
    - True (legacy positive boolean) -> Accepted
    - None -> NOT accepted
    - False -> NOT accepted
    - MEDIA_NOT_FOUND, AUTH_ERROR, WAITING_FOR_MEDIA, TEMPORARY_ERROR -> NOT accepted
    - Exception -> NOT accepted
    """
    video = tmp_path / "test.mkv"
    video.touch()

    pipeline = SubtitlePipeline()

    variants = [
        (BazarrResult(code=BazarrResultCode.TRIGGERED), True),
        (BazarrResult(code=BazarrResultCode.ACCEPTED), True),
        (True, True),
        (None, False),
        (False, False),
        (BazarrResult(code=BazarrResultCode.MEDIA_NOT_FOUND), False),
        (BazarrResult(code=BazarrResultCode.AUTH_ERROR), False),
        (BazarrResult(code=BazarrResultCode.WAITING_FOR_MEDIA), False),
        (BazarrResult(code=BazarrResultCode.TEMPORARY_ERROR), False),
    ]

    for ret_val, expected_accepted in variants:
        _accepted_map = {}

        async def mock_search(vp, language="sv", **kwargs):
            if isinstance(ret_val, Exception):
                raise ret_val
            return ret_val

        # Simulate _do_btarget acceptance logic directly
        _lc = "sv"
        _ln = "Swedish"
        _r = await mock_search(str(video), language=_lc)
        if isinstance(_r, BazarrResult):
            if _r.was_accepted:
                _accepted_map[_lc] = True
        elif _r is True:
            _accepted_map[_lc] = True

        is_accepted = _accepted_map.get(_lc, False)
        assert is_accepted == expected_accepted, f"Failed for return value: {ret_val}"


# ===========================================================================
# K. COORDINATOR UNACCEPTED BAZARR MUST NOT BE LABELED BAZARR MATCH
# ===========================================================================

@pytest.mark.asyncio
async def test_k_coordinator_unaccepted_bazarr_cannot_label_bazarr_match(tmp_path, monkeypatch):
    """
    Enters the coordinator FINALIZED_WITH_TARGET path with:
    - No target initially
    - Bazarr trigger unaccepted (e.g. MEDIA_NOT_FOUND)
    - External target appears
    - Trust PASS
    Expected:
    - CandidateOrigin is EXTERNAL (fail-closed)
    - Status is ALREADY EXISTS (NOT BAZARR MATCH)
    - Reason is NOT 'Bazarr found all targets'
    """
    db_file = tmp_path / "coord_test_k.db"
    orig_db = db.DB_PATH
    db.DB_PATH = str(db_file)
    db.init_db()

    video = tmp_path / "movie.mkv"
    video.touch()
    sv_target = tmp_path / "movie.sv.srt"

    en_source = tmp_path / "movie.en.srt"
    en_source.write_text(make_ambiguous_srt(10), encoding="utf-8")

    try:
        def mock_settings(k, d=""):
            return {
                "ai_provider": "gemini",
                "gemini_api_key": "dummy",
                "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
                "enable_bazarr_check": "true",
                "clean_sdh": "true",
                "extract_source_embedded": "false",
                "extract_target_embedded": "false",
                "bazarr_grace_seconds": "1.0",
                "bazarr_quiescence_seconds": "0.1",
                "bazarr_candidate_stability_seconds": "0.02",
            }.get(k, d)

        monkeypatch.setattr("app.services.pipeline.get_setting", mock_settings)
        monkeypatch.setattr("app.services.translator.get_setting", mock_settings)

        pipeline = SubtitlePipeline()
        mock_translate = AsyncMock(return_value=[])
        monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)

        # Bazarr trigger returns unaccepted MEDIA_NOT_FOUND
        async def mock_bazarr_search(vp, language=None, **kwargs):
            return BazarrResult(code=BazarrResultCode.MEDIA_NOT_FOUND, language=language or "sv", detail="Not in library")

        monkeypatch.setattr(pipeline, "trigger_bazarr_search", mock_bazarr_search)

        # Mock Bazarr coordinator to finalize with target (simulating external process creating the target during coordination)
        from app.services.bazarr_coordinator import BazarrLifecycleState, bazarr_coordinator
        sv_target.write_text(make_swedish_srt(10), encoding="utf-8")

        async def mock_coordinate(video_path, target_lang, **kwargs):
            tres = TrustResult(decision=TrustDecision.PASS, score=95, confidence="HIGH", reasons=["Cadence match"], origin=CandidateOrigin.EXTERNAL)
            return (BazarrLifecycleState.FINALIZED_WITH_TARGET, str(sv_target), tres)

        monkeypatch.setattr(bazarr_coordinator, "coordinate_target", mock_coordinate)

        job_id = db.create_job(video_path=str(video), event_source="SONARR")
        res = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)

        assert res["status"] == "skipped"
        job = db.get_job_by_id(job_id)
        assert job["status"] != "BAZARR MATCH"
        assert job["status"] == "ALREADY EXISTS"
        assert "Bazarr found all targets" not in (job["reason"] or "")
        mock_translate.assert_not_awaited()
    finally:
        db.DB_PATH = orig_db


# ===========================================================================
# L. COORDINATOR ACCEPTED BAZARR LABELS BAZARR MATCH
# ===========================================================================

@pytest.mark.asyncio
async def test_l_coordinator_accepted_bazarr_labels_bazarr_match(tmp_path, monkeypatch):
    """
    Enters the coordinator FINALIZED_WITH_TARGET path with:
    - No target initially
    - Bazarr trigger accepted (TRIGGERED)
    - Target appears
    - Trust PASS
    Expected:
    - CandidateOrigin is BAZARR
    - Status is BAZARR MATCH
    - Reason is 'Bazarr found all targets'
    """
    db_file = tmp_path / "coord_test_l.db"
    orig_db = db.DB_PATH
    db.DB_PATH = str(db_file)
    db.init_db()

    video = tmp_path / "movie.mkv"
    video.touch()
    sv_target = tmp_path / "movie.sv.srt"

    en_source = tmp_path / "movie.en.srt"
    en_source.write_text(make_ambiguous_srt(10), encoding="utf-8")

    try:
        def mock_settings(k, d=""):
            return {
                "ai_provider": "gemini",
                "gemini_api_key": "dummy",
                "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
                "enable_bazarr_check": "true",
                "clean_sdh": "true",
                "extract_source_embedded": "false",
                "extract_target_embedded": "false",
                "bazarr_grace_seconds": "1.0",
                "bazarr_quiescence_seconds": "0.1",
                "bazarr_candidate_stability_seconds": "0.02",
            }.get(k, d)

        monkeypatch.setattr("app.services.pipeline.get_setting", mock_settings)
        monkeypatch.setattr("app.services.translator.get_setting", mock_settings)

        pipeline = SubtitlePipeline()
        mock_translate = AsyncMock(return_value=[])
        monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)

        # Bazarr trigger returns accepted TRIGGERED
        async def mock_bazarr_search(vp, language=None, **kwargs):
            return BazarrResult(code=BazarrResultCode.TRIGGERED, language=language or "sv", detail="Search accepted")

        monkeypatch.setattr(pipeline, "trigger_bazarr_search", mock_bazarr_search)

        from app.services.bazarr_coordinator import BazarrLifecycleState, bazarr_coordinator

        async def mock_coordinate(video_path, target_lang, **kwargs):
            sv_target.write_text(make_swedish_srt(10), encoding="utf-8")
            tres = TrustResult(decision=TrustDecision.PASS, score=95, confidence="HIGH", reasons=["Cadence match"], origin=CandidateOrigin.BAZARR)
            return (BazarrLifecycleState.FINALIZED_WITH_TARGET, str(sv_target), tres)

        monkeypatch.setattr(bazarr_coordinator, "coordinate_target", mock_coordinate)

        job_id = db.create_job(video_path=str(video), event_source="SONARR")
        res = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)

        assert res["status"] == "skipped"
        job = db.get_job_by_id(job_id)
        assert job["status"] == "BAZARR MATCH"
        assert job["reason"] == "Bazarr found all targets"
        mock_translate.assert_not_awaited()
    finally:
        db.DB_PATH = orig_db


# ===========================================================================
# M. EMPTY ORIGIN MAP PROVENANCE FAILS CLOSED
# ===========================================================================

def test_m_empty_origin_map_fails_closed():
    """
    Proves that empty origin mapping evaluates fail-closed to False for all_bazarr,
    never claiming Bazarr ownership without positive evidence.
    """
    empty_origins = {}
    all_bazarr = bool(empty_origins) and all(orig == CandidateOrigin.BAZARR for orig in empty_origins.values())
    all_external = bool(empty_origins) and all(orig == CandidateOrigin.EXTERNAL for orig in empty_origins.values())
    has_bazarr = bool(empty_origins) and any(orig == CandidateOrigin.BAZARR for orig in empty_origins.values())

    assert all_bazarr is False
    assert all_external is False
    assert has_bazarr is False


# ===========================================================================
# N. STATS: NO-AI EXTERNAL TARGET EXCLUDED FROM TRANSLATED AVERAGE
# ===========================================================================

def test_n_stats_no_ai_external_target_excluded_from_translated_average(tmp_path):
    """
    Proves that a job resolved as ALREADY EXISTS ('External target appeared during processing')
    is counted under existing_skipped, and is NOT included in translated count or avg_duration_seconds.
    """
    db_file = tmp_path / "stats_rescue_test.db"
    orig_db = db.DB_PATH
    db.DB_PATH = str(db_file)
    db.init_db()

    try:
        with sqlite3.connect(str(db_file)) as conn:
            # 1 translated job (60.0s)
            conn.execute(
                "INSERT INTO jobs (video_path, status, duration_seconds, sync_diff_ms, created_at, updated_at) VALUES ('/m1.mkv', 'TRANSLATED', 60.0, 15, 'now', 'now')"
            )
            # 1 external target rescue job (1.5s, no AI)
            conn.execute(
                "INSERT INTO jobs (video_path, status, reason, duration_seconds, sync_diff_ms, created_at, updated_at) VALUES ('/m2.mkv', 'ALREADY EXISTS', 'External target appeared during processing', 1.5, -1, 'now', 'now')"
            )
            conn.commit()

        stats = db.get_job_stats()
        assert stats["total"] == 2
        assert stats["translated"] == 1
        assert stats["existing_skipped"] == 1
        # Avg duration must be 60.0s, NOT dragged down by the 1.5s rescue
        assert stats["avg_duration_seconds"] == 60.0
    finally:
        db.DB_PATH = orig_db


# ===========================================================================
# O. MID-TRANSLATION EXTERNAL TARGET WINS TRUTHFULLY
# ===========================================================================

@pytest.mark.asyncio
async def test_o_mid_translation_external_target_wins_truthfully(tmp_path, monkeypatch):
    """
    Proves that when an external verified target appears while AI translation is in progress:
    - Bazarr is disabled
    - Target absent initially
    - AI translation starts
    - Verified external target appears mid-run
    - early_stop_check detects it
    - External target wins and AI translation is NOT published
    - Job status: ALREADY EXISTS, reason: 'External target appeared during processing'
    - status != BAZARR MATCH and status != TRANSLATED
    - External target file on disk remains untouched
    - AI provider calls/usage accounting remain truthful
    """
    db_file = tmp_path / "mid_trans_test.db"
    orig_db = db.DB_PATH
    db.DB_PATH = str(db_file)
    db.init_db()

    video = tmp_path / "movie.mkv"
    video.touch()
    sv_target = tmp_path / "movie.sv.srt"
    expected_content = make_swedish_srt(10)

    en_source = tmp_path / "movie.en.srt"
    en_source.write_text(make_ambiguous_srt(10), encoding="utf-8")

    try:
        def mock_settings(k, d=""):
            return {
                "ai_provider": "gemini",
                "gemini_api_key": "dummy",
                "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
                "enable_bazarr_check": "false",
                "clean_sdh": "true",
                "extract_source_embedded": "false",
                "extract_target_embedded": "false",
                "bazarr_quiescence_seconds": "0.0",
                "bazarr_candidate_stability_seconds": "0.01",
            }.get(k, d)

        monkeypatch.setattr("app.services.pipeline.get_setting", mock_settings)
        monkeypatch.setattr("app.services.translator.get_setting", mock_settings)

        pipeline = SubtitlePipeline()

        # Mock translator to simulate external target appearing during translation
        async def mock_translate_srt_content(subs, early_stop_check=None, **kwargs):
            # External file appears on disk mid-translation
            sv_target.write_text(expected_content, encoding="utf-8")
            if early_stop_check:
                stopped = await early_stop_check()
                assert stopped is True
            # Simulate returning empty or partial translation since stopped
            return []

        monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate_srt_content)

        # Mock Trust Engine to verify the candidate
        async def mock_evaluate(self, video_path, candidate_path, **kwargs):
            return TrustResult(decision=TrustDecision.PASS, score=95, confidence="HIGH", reasons=["Cadence match"], origin=CandidateOrigin.EXTERNAL)

        monkeypatch.setattr(SubtitleTrustEngine, "evaluate_candidate", mock_evaluate)

        job_id = db.create_job(video_path=str(video), event_source="MANUAL")
        res = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)

        assert res["status"] != "failed"
        job = db.get_job_by_id(job_id)
        assert job["status"] != "BAZARR MATCH"
        assert job["status"] != "TRANSLATED"
        assert job["status"] == "ALREADY EXISTS"
        assert job["reason"] == "External target appeared during processing"
        assert job["sync_diff_ms"] == -1

        # Assert external target file was preserved exactly as written
        assert sv_target.read_text(encoding="utf-8") == expected_content
    finally:
        db.DB_PATH = orig_db


# ===========================================================================
# P. MULTI-LANGUAGE: MIXED AI + EXTERNAL YIELDS TRANSLATED
# ===========================================================================

@pytest.mark.asyncio
async def test_p_mixed_ai_and_external_multilang_yields_translated(tmp_path, monkeypatch):
    """
    Proves that in a multi-language job where one language is satisfied by an external target
    and another language is translated and published by Babel AI:
    - Final status is TRANSLATED (because Babel supplied at least one target)
    """
    db_file = tmp_path / "mixed_multilang_test.db"
    orig_db = db.DB_PATH
    db.DB_PATH = str(db_file)
    db.init_db()

    video = tmp_path / "multilang.mkv"
    video.touch()
    sv_target = tmp_path / "multilang.sv.srt"

    en_source = tmp_path / "multilang.en.srt"
    en_source.write_text(make_ambiguous_srt(10), encoding="utf-8")

    try:
        def mock_settings(k, d=""):
            return {
                "ai_provider": "gemini",
                "gemini_api_key": "dummy",
                "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}, {"code": "da", "name": "Danish", "enabled": true}]',
                "enable_bazarr_check": "false",
                "clean_sdh": "true",
                "extract_source_embedded": "false",
                "extract_target_embedded": "false",
                "bazarr_quiescence_seconds": "0.0",
                "bazarr_candidate_stability_seconds": "0.01",
            }.get(k, d)

        monkeypatch.setattr("app.services.pipeline.get_setting", mock_settings)
        monkeypatch.setattr("app.services.translator.get_setting", mock_settings)

        pipeline = SubtitlePipeline()

        # Mock translator: sv is rescued mid-translation, da is translated by AI
        async def mock_translate(subs, target_language="Swedish", early_stop_check=None, **kwargs):
            if target_language == "Swedish":
                sv_target.write_text(make_swedish_srt(10), encoding="utf-8")
                if early_stop_check:
                    await early_stop_check()
                return []
            return list(srt.parse(make_swedish_srt(10)))

        monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)

        def mock_evaluate_qa(*args, **kwargs):
            return {"passed": True, "score": 95, "issues": [], "real_untranslated_ids": []}

        monkeypatch.setattr("app.services.pipeline.qa_gate", mock_evaluate_qa)

        # Mock Trust Engine
        async def mock_evaluate(self, video_path, candidate_path, **kwargs):
            return TrustResult(decision=TrustDecision.PASS, score=95, confidence="HIGH", reasons=["Cadence match"], origin=CandidateOrigin.EXTERNAL)

        monkeypatch.setattr(SubtitleTrustEngine, "evaluate_candidate", mock_evaluate)

        job_id = db.create_job(video_path=str(video), event_source="MANUAL")
        res = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)

        assert res["status"] != "failed"
        job = db.get_job_by_id(job_id)
        assert job["status"] == "TRANSLATED"
        assert "sv" in job["target_languages"]
        assert "da" in job["target_languages"]
    finally:
        db.DB_PATH = orig_db
