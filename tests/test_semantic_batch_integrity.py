import pytest
import datetime
import srt
from unittest.mock import AsyncMock, patch, MagicMock

from app.core.validator import (
    BatchSemanticState,
    PrimaryBatchInfo,
    SemanticIncidentTracker,
    extract_batch_alignment_samples,
    IncidentState,
    AlignmentIncident,
)
from app.core.usage import UsageStage
from app.services.translator import (
    validate_batch_translation_results,
    SubtitleTranslator,
)
from app.services.pipeline import SubtitlePipeline


def _make_dummy_subs(count: int, start_time_sec: int = 10) -> list[srt.Subtitle]:
    subs = []
    for i in range(count):
        s_sec = start_time_sec + (i * 3)
        e_sec = s_sec + 2
        start_td = datetime.timedelta(seconds=s_sec)
        end_td = datetime.timedelta(seconds=e_sec)
        subs.append(
            srt.Subtitle(
                index=i + 1,
                start=start_td,
                end=end_td,
                content=f"Line {i + 1} original text dialogue."
            )
        )
    return subs


def _make_dummy_sv_subs(count: int, start_time_sec: int = 10) -> list[srt.Subtitle]:
    subs = []
    for i in range(count):
        s_sec = start_time_sec + (i * 3)
        e_sec = s_sec + 2
        start_td = datetime.timedelta(seconds=s_sec)
        end_td = datetime.timedelta(seconds=e_sec)
        subs.append(
            srt.Subtitle(
                index=i + 1,
                start=start_td,
                end=end_td,
                content=f"Rad {i + 1} svensk text dialog."
            )
        )
    return subs


# ==============================================================================
# 1. STRUCTURAL VALIDATOR CONTRACT TESTS
# ==============================================================================

def test_structural_clean_does_not_imply_semantic_alignment_trust():
    """
    Simulate The Office scenario: A batch returns integer IDs 301..450,
    so structurally it is clean, but the content inside is shifted.
    validate_batch_translation_results must report is_clean=True and is_structurally_clean=True,
    which downstream code knows verifies ONLY structural syntax, NOT semantic correspondence.
    """
    expected_items = [{"id": i, "text": f"English line {i}"} for i in range(301, 451)]
    # Provider shifted the text by +1 (ID 301 has English line 302's translation, etc.)
    shifted_results = [{"id": i, "text": f"Svensk rad {i + 1}"} for i in range(301, 450)]
    shifted_results.append({"id": 450, "text": "Svensk rad 450"})

    valid_map, report = validate_batch_translation_results(expected_items, shifted_results)

    assert report["is_clean"] is True
    assert report["is_structurally_clean"] is True
    assert len(valid_map) == 150
    assert report["missing_ids"] == []
    assert report["unknown_ids"] == []
    assert report["duplicate_ids"] == []
    assert report["malformed_count"] == 0


def test_structural_missing_unknown_and_duplicates():
    """Verify that structural exact-ID validator rejects corrupt IDs."""
    expected_items = [{"id": i, "text": f"Line {i}"} for i in range(1, 6)]
    raw_results = [
        {"id": 1, "text": "Rad 1"},
        {"id": 2, "text": "Rad 2"},
        {"id": 2, "text": "Rad 2 duplicate"},
        {"id": 99, "text": "Unknown ID"},
        {"id": "invalid", "text": "Malformed"},
    ]

    valid_map, report = validate_batch_translation_results(expected_items, raw_results)

    assert report["is_clean"] is False
    assert report["is_structurally_clean"] is False
    assert 2 in report["duplicate_ids"]
    assert 99 in report["unknown_ids"]
    assert 3 in report["missing_ids"]
    assert 4 in report["missing_ids"]
    assert 5 in report["missing_ids"]
    assert report["malformed_count"] == 1
    assert len(valid_map) == 2


# ==============================================================================
# 2. SEMANTIC INCIDENT TRACKER & PRIMARY BATCH PROVENANCE TESTS
# ==============================================================================

def test_semantic_incident_tracker_initializes_primary_batches():
    """Verify dynamic initialization of primary batches for arbitrary cue counts."""
    tracker = SemanticIncidentTracker(total_cues=620, batch_size=150)
    batches = tracker.get_batches()

    assert len(batches) == 5
    assert batches[0].start_idx == 0 and batches[0].end_idx == 149 and batches[0].cue_count == 150
    assert batches[1].start_idx == 150 and batches[1].end_idx == 299 and batches[1].cue_count == 150
    assert batches[2].start_idx == 300 and batches[2].end_idx == 449 and batches[2].cue_count == 150
    assert batches[3].start_idx == 450 and batches[3].end_idx == 599 and batches[3].cue_count == 150
    assert batches[4].start_idx == 600 and batches[4].end_idx == 619 and batches[4].cue_count == 20

    # Batch IDs (1-based)
    assert batches[0].start_id == 1 and batches[0].end_id == 150
    assert batches[2].start_id == 301 and batches[2].end_id == 450
    assert batches[4].start_id == 601 and batches[4].end_id == 620


def test_findings_within_same_batch_coalesce_to_single_batch_incident():
    """
    Multiple micro-findings inside cues 301-450 (e.g. at 320-323, 336-341, 386-389, 445-450)
    must map to the single primary batch (batch_idx = 2).
    """
    tracker = SemanticIncidentTracker(total_cues=620, batch_size=150)

    # Findings in batch 2 (cues 300..449, 1-based 301..450)
    inc1 = AlignmentIncident(start_idx=319, end_idx=322, verdict="SHIFT_PLUS_1", confidence="HIGH", details="Shift at 320")
    inc2 = AlignmentIncident(start_idx=335, end_idx=340, verdict="SHIFT_PLUS_1", confidence="HIGH", details="Shift at 336")
    inc3 = AlignmentIncident(start_idx=385, end_idx=388, verdict="SHIFT_PLUS_1", confidence="HIGH", details="Shift at 386")
    inc4 = AlignmentIncident(start_idx=444, end_idx=449, verdict="SHIFT_PLUS_1", confidence="HIGH", details="Shift at 445")

    tracker.register_or_merge([inc1, inc2, inc3, inc4])

    # Primary batch 2 must be marked SUSPECT
    b2 = tracker.find_batch_by_index(2)
    assert b2 is not None
    assert b2.state == BatchSemanticState.SUSPECT
    assert b2.start_id == 301
    assert b2.end_id == 450

    # Other primary batches must remain UNVERIFIED
    assert tracker.find_batch_by_index(0).state == BatchSemanticState.UNVERIFIED
    assert tracker.find_batch_by_index(1).state == BatchSemanticState.UNVERIFIED
    assert tracker.find_batch_by_index(3).state == BatchSemanticState.UNVERIFIED
    assert tracker.find_batch_by_index(4).state == BatchSemanticState.UNVERIFIED


def test_stratified_sample_extraction():
    """Verify that extract_batch_alignment_samples extracts beginning, middle, and end samples."""
    source_subs = _make_dummy_subs(620)
    translated_subs = _make_dummy_sv_subs(620)

    samples = extract_batch_alignment_samples(
        source_subs,
        translated_subs,
        start_idx=300,
        end_idx=449,
        max_pairs=8
    )

    assert len(samples) <= 8
    assert len(samples) > 0
    # Samples must be within batch 301..450 (1-based IDs)
    for s in samples:
        assert 301 <= s["id"] <= 450
        assert "original text" in s["source"]
        assert "svensk text" in s["target"]

    # Must cover start, middle, and end
    ids = [s["id"] for s in samples]
    assert min(ids) <= 305
    assert max(ids) >= 445
    assert any(350 <= i <= 400 for i in ids)


# ==============================================================================
# 3. CONSOLIDATED BATCH AUDIT & STAGE DETERMINATION TESTS
# ==============================================================================

def test_infer_usage_stage_mappings():
    """Verify that batch semantic methods are properly categorized into UsageStage."""
    from app.services.translator import _infer_usage_stage

    assert _infer_usage_stage("audit_batch_semantic_integrity", {}) == UsageStage.SEMANTIC_AUDIT
    assert _infer_usage_stage("confirm_batch_semantic_integrity", {}) == UsageStage.SEMANTIC_AUDIT
    assert _infer_usage_stage("verify_repaired_batch_integrity", {}) == UsageStage.SEMANTIC_AUDIT
    assert _infer_usage_stage("translate_batch", {}) == UsageStage.PRIMARY
    assert _infer_usage_stage("repair_alignment_region", {}) == UsageStage.RECOVERY


@pytest.mark.asyncio
async def test_consolidated_batch_audit_clean_job_single_call():
    """
    For a clean 620-cue job (5 batches), consolidated audit sends all 5 batches in 1 AI call,
    and all return ALIGNED.
    """
    pipeline = SubtitlePipeline()
    source_subs = _make_dummy_subs(620)
    translated_subs = _make_dummy_sv_subs(620)

    tracker = SemanticIncidentTracker(total_cues=620, batch_size=150)

    mock_audit = AsyncMock(return_value={
        1: {"batch_id": 1, "verdict": "ALIGNED", "confidence": "HIGH", "details": "All 1-to-1"},
        2: {"batch_id": 2, "verdict": "ALIGNED", "confidence": "HIGH", "details": "All 1-to-1"},
        3: {"batch_id": 3, "verdict": "ALIGNED", "confidence": "HIGH", "details": "All 1-to-1"},
        4: {"batch_id": 4, "verdict": "ALIGNED", "confidence": "HIGH", "details": "All 1-to-1"},
        5: {"batch_id": 5, "verdict": "ALIGNED", "confidence": "HIGH", "details": "All 1-to-1"},
    })
    pipeline.translator.audit_batch_semantic_integrity = mock_audit

    report = await pipeline.check_semantic_cue_alignment(
        source_subs=source_subs,
        translated_subs=translated_subs,
        target_language="Swedish",
        source_language="English",
        batch_size=150,
        incident_tracker=tracker
    )

    # Exactly 1 consolidated AI audit call
    assert mock_audit.call_count == 1
    assert len(report["issues"]) == 0
    assert len(report["affected_indices"]) == 0

    # All batches in tracker must be ALIGNED
    for b in tracker.get_batches():
        assert b.state == BatchSemanticState.ALIGNED


# ==============================================================================
# 4. SINGLE CORRUPT BATCH RECOVERY & BOUNDED CALL COUNT TESTS (THE OFFICE SIMULATION)
# ==============================================================================

@pytest.mark.asyncio
async def test_the_office_benchmark_simulation_bounded_calls_and_atomic_commit():
    """
    Simulate The Office S02E19 run (620 cues, batch_size=150):
    - Primary batches 1, 2, 4, 5 are clean.
    - Primary batch 3 (cues 301..450) is semantically corrupt (+1 shift, tail missing).
    - Pipeline must execute:
        1. Consolidated audit (1 call) -> detects batch 3 suspect.
        2. Confirmation call for batch 3 (1 call) -> confirms batch 3 CORRUPT.
        3. Source recovery for batch 3 -> partitions 150 cues into 2 sub-batches (75 + 75 cues) -> 2 primary translate_batch calls.
        4. Post-repair verification (1 call) -> returns ALIGNED.
        5. Atomically commits repaired text.
        6. Total recovery dispatches = 4 (1 confirm + 2 recovery + 1 verify).
        7. 0 ms timestamp drift, 0 dropped cues.
    """
    pipeline = SubtitlePipeline()
    source_subs = _make_dummy_subs(620)
    translated_subs = _make_dummy_sv_subs(620)

    # Corrupt batch 3 translated cues (simulate shift)
    for i in range(300, 449):
        translated_subs[i].content = f"Rad {i + 2} svensk text dialog (shifted +1)"
    translated_subs[449].content = "Rad 449 svensk text dialog"

    tracker = SemanticIncidentTracker(total_cues=620, batch_size=150)

    # 1. Consolidated audit returns batch 3 SUSPECT, others ALIGNED
    mock_audit = AsyncMock(return_value={
        1: {"batch_id": 1, "verdict": "ALIGNED", "confidence": "HIGH", "details": ""},
        2: {"batch_id": 2, "verdict": "ALIGNED", "confidence": "HIGH", "details": ""},
        3: {"batch_id": 3, "verdict": "SUSPECT", "confidence": "HIGH", "details": "Sequential shift detected at cue 320"},
        4: {"batch_id": 4, "verdict": "ALIGNED", "confidence": "HIGH", "details": ""},
        5: {"batch_id": 5, "verdict": "ALIGNED", "confidence": "HIGH", "details": ""},
    })
    pipeline.translator.audit_batch_semantic_integrity = mock_audit

    # 2. Confirmation call for batch 3 returns CORRUPT
    mock_confirm = AsyncMock(return_value={
        "verdict": "CORRUPT",
        "confidence": "HIGH",
        "details": "Confirmed +1 shift across cues 301-450"
    })
    pipeline.translator.confirm_batch_semantic_integrity = mock_confirm

    # 3. Recovery translation produces clean 1-to-1 translations from canonical source
    async def mock_translate_batch(items, **kwargs):
        return [{"id": it["id"], "text": f"Rad {it['id']} korrekt reparerad svensk text."} for it in items]

    pipeline.translator.translate_batch = AsyncMock(side_effect=mock_translate_batch)

    # 4. Post-repair verification returns ALIGNED
    mock_verify = AsyncMock(return_value={
        "verdict": "ALIGNED",
        "confidence": "HIGH",
        "details": "Repaired batch is 1-to-1 aligned"
    })
    pipeline.translator.verify_repaired_batch_integrity = mock_verify

    # Step A: Audit
    audit_report = await pipeline.check_semantic_cue_alignment(
        source_subs=source_subs,
        translated_subs=translated_subs,
        target_language="Swedish",
        source_language="English",
        batch_size=150,
        incident_tracker=tracker
    )

    assert mock_audit.call_count == 1
    assert len(audit_report["affected_indices"]) == 150
    assert audit_report["affected_indices"][0] == 300
    assert audit_report["affected_indices"][-1] == 449

    # Step B & C & D: Repair
    mutated_indices = set()
    def _apply_mutation(idx: int, new_text: str):
        translated_subs[idx].content = new_text
        mutated_indices.add(idx)

    success = await pipeline._repair_semantic_alignment_incidents(
        subs=source_subs,
        translated_subs=translated_subs,
        incidents=audit_report["incidents"],
        target_language="Swedish",
        source_language="English",
        apply_mutation_fn=_apply_mutation,
        incident_tracker=tracker
    )

    assert success is True
    assert mock_confirm.call_count == 1
    # 150 cues partitioned into 2 sub-batches (75 + 75 cues)
    assert pipeline.translator.translate_batch.call_count == 2
    assert mock_verify.call_count == 1

    # Batch 3 in tracker is now REPAIRED
    b3 = tracker.find_batch_by_index(2)
    assert b3.state == BatchSemanticState.REPAIRED

    # Check that mutated cues are all within cues 300..449
    assert len(mutated_indices) == 150
    assert min(mutated_indices) == 300
    assert max(mutated_indices) == 449
    for i in range(300, 450):
        assert "korrekt reparerad" in translated_subs[i].content

    # Invariants: 0ms drift, exact cue count
    assert len(translated_subs) == len(source_subs)
    for i in range(len(source_subs)):
        assert translated_subs[i].start == source_subs[i].start
        assert translated_subs[i].end == source_subs[i].end


# ==============================================================================
# 5. CONTRADICTORY CONFIRMATION & FALSE POSITIVE TESTS (FAIL CLOSED)
# ==============================================================================

@pytest.mark.asyncio
async def test_false_positive_audit_discarded_by_confirmation_zero_mutation():
    """
    If consolidated audit flags a batch as SUSPECT, but focused confirmation returns
    ALIGNED with HIGH confidence, it must be discarded as a false positive with 0 mutations.
    """
    pipeline = SubtitlePipeline()
    source_subs = _make_dummy_subs(100)
    translated_subs = _make_dummy_sv_subs(100)

    tracker = SemanticIncidentTracker(total_cues=100, batch_size=50)

    # Batch 1 flagged SUSPECT by audit
    pipeline.translator.audit_batch_semantic_integrity = AsyncMock(return_value={
        1: {"batch_id": 1, "verdict": "SUSPECT", "confidence": "LOW", "details": "Possible split sentence shift"},
        2: {"batch_id": 2, "verdict": "ALIGNED", "confidence": "HIGH", "details": ""},
    })

    # Confirmation proves it is ALIGNED (false positive discarded)
    pipeline.translator.confirm_batch_semantic_integrity = AsyncMock(return_value={
        "verdict": "ALIGNED",
        "confidence": "HIGH",
        "details": "Natural split sentence across cues 12-13, 1-to-1 alignment preserved"
    })

    pipeline.translator.translate_batch = AsyncMock()

    audit_report = await pipeline.check_semantic_cue_alignment(
        source_subs=source_subs,
        translated_subs=translated_subs,
        target_language="Swedish",
        source_language="English",
        batch_size=50,
        incident_tracker=tracker
    )

    mutations = []
    success = await pipeline._repair_semantic_alignment_incidents(
        subs=source_subs,
        translated_subs=translated_subs,
        incidents=audit_report["incidents"],
        target_language="Swedish",
        source_language="English",
        apply_mutation_fn=lambda idx, txt: mutations.append(idx),
        incident_tracker=tracker
    )

    assert success is True
    # Zero recovery translation calls and zero mutations
    assert pipeline.translator.translate_batch.call_count == 0
    assert len(mutations) == 0
    assert tracker.find_batch_by_index(0).state == BatchSemanticState.ALIGNED


@pytest.mark.asyncio
async def test_contradictory_confirmation_fails_closed_zero_mutation():
    """
    If confirmation returns UNCERTAIN or contradictory evidence, the pipeline
    must fail closed without making blind speculative mutations.
    """
    pipeline = SubtitlePipeline()
    source_subs = _make_dummy_subs(100)
    translated_subs = _make_dummy_sv_subs(100)

    tracker = SemanticIncidentTracker(total_cues=100, batch_size=50)

    pipeline.translator.audit_batch_semantic_integrity = AsyncMock(return_value={
        1: {"batch_id": 1, "verdict": "SUSPECT", "confidence": "LOW", "details": "Shift suspected"},
        2: {"batch_id": 2, "verdict": "ALIGNED", "confidence": "HIGH", "details": ""},
    })

    # Confirmation returns UNCERTAIN
    pipeline.translator.confirm_batch_semantic_integrity = AsyncMock(return_value={
        "verdict": "UNCERTAIN",
        "confidence": "LOW",
        "details": "Non-verbal sound cues make alignment ambiguous"
    })

    pipeline.translator.translate_batch = AsyncMock()

    audit_report = await pipeline.check_semantic_cue_alignment(
        source_subs=source_subs,
        translated_subs=translated_subs,
        target_language="Swedish",
        source_language="English",
        batch_size=50,
        incident_tracker=tracker
    )

    mutations = []
    success = await pipeline._repair_semantic_alignment_incidents(
        subs=source_subs,
        translated_subs=translated_subs,
        incidents=audit_report["incidents"],
        target_language="Swedish",
        source_language="English",
        apply_mutation_fn=lambda idx, txt: mutations.append(idx),
        incident_tracker=tracker
    )

    assert success is True  # No confirmed corrupt batches to repair
    assert pipeline.translator.translate_batch.call_count == 0
    assert len(mutations) == 0
    assert tracker.find_batch_by_index(0).state == BatchSemanticState.FAILED_REPAIR


# ==============================================================================
# 6. POST-REPAIR VERIFICATION FAILURE & ATOMIC REJECTION TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_post_repair_verification_failure_rejects_candidate_and_prevents_cascade():
    """
    If re-translation is performed but post-repair verification fails (returns CORRUPT/UNCERTAIN),
    the candidate patch must be discarded atomically and the batch marked FAILED_REPAIR.
    It must NOT enter an infinite repair loop.
    """
    pipeline = SubtitlePipeline()
    source_subs = _make_dummy_subs(50)
    translated_subs = _make_dummy_sv_subs(50)
    orig_content = [s.content for s in translated_subs]

    tracker = SemanticIncidentTracker(total_cues=50, batch_size=50)

    pipeline.translator.audit_batch_semantic_integrity = AsyncMock(return_value={
        1: {"batch_id": 1, "verdict": "SUSPECT", "confidence": "HIGH", "details": "Shift"}
    })

    pipeline.translator.confirm_batch_semantic_integrity = AsyncMock(return_value={
        "verdict": "CORRUPT",
        "confidence": "HIGH",
        "details": "Confirmed shift"
    })

    pipeline.translator.translate_batch = AsyncMock(return_value=[
        {"id": i + 1, "text": f"Re-translated {i + 1}"} for i in range(50)
    ])

    # Post-repair verify FAILS
    pipeline.translator.verify_repaired_batch_integrity = AsyncMock(return_value={
        "verdict": "CORRUPT",
        "confidence": "HIGH",
        "details": "Re-translation still has alignment drift"
    })

    audit_report = await pipeline.check_semantic_cue_alignment(
        source_subs=source_subs,
        translated_subs=translated_subs,
        target_language="Swedish",
        source_language="English",
        batch_size=50,
        incident_tracker=tracker
    )

    mutations = []
    success = await pipeline._repair_semantic_alignment_incidents(
        subs=source_subs,
        translated_subs=translated_subs,
        incidents=audit_report["incidents"],
        target_language="Swedish",
        source_language="English",
        apply_mutation_fn=lambda idx, txt: mutations.append(idx),
        incident_tracker=tracker
    )

    assert success is False
    assert len(mutations) == 0
    assert tracker.find_batch_by_index(0).state == BatchSemanticState.FAILED_REPAIR
    # Ensure translated_subs was not modified
    for i in range(50):
        assert translated_subs[i].content == orig_content[i]


# ==============================================================================
# 7. LARGE FILE DYNAMIC CHUNKING TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_large_file_audit_chunking():
    """
    For a very large file (e.g. 15 batches), consolidated audit must be chunked into
    slices of 8 batches, resulting in exactly ceil(15/8) = 2 AI calls.
    """
    pipeline = SubtitlePipeline()
    source_subs = _make_dummy_subs(750)
    translated_subs = _make_dummy_sv_subs(750)

    tracker = SemanticIncidentTracker(total_cues=750, batch_size=50)
    assert len(tracker.get_batches()) == 15

    call_chunks = []
    async def mock_audit(batch_payloads, **kwargs):
        call_chunks.append(len(batch_payloads))
        return {
            bp["batch_id"]: {"batch_id": bp["batch_id"], "verdict": "ALIGNED", "confidence": "HIGH", "details": ""}
            for bp in batch_payloads
        }

    pipeline.translator.audit_batch_semantic_integrity = AsyncMock(side_effect=mock_audit)

    report = await pipeline.check_semantic_cue_alignment(
        source_subs=source_subs,
        translated_subs=translated_subs,
        target_language="Swedish",
        source_language="English",
        batch_size=50,
        incident_tracker=tracker
    )

    assert len(call_chunks) == 2
    assert call_chunks[0] == 8
    assert call_chunks[1] == 7
    assert len(report["issues"]) == 0


# ==============================================================================
# 8. P0 FAIL-CLOSED REGRESSION TESTS — NO EVIDENCE != ALIGNED
# ==============================================================================

@pytest.mark.asyncio
async def test_fail_closed_audit_raises_exception_never_aligned():
    """
    P0 REGRESSION: If audit_batch_semantic_integrity raises an exception,
    the affected batches MUST be marked SUSPECT (fail-closed), never ALIGNED.
    """
    pipeline = SubtitlePipeline()
    source_subs = _make_dummy_subs(100)
    translated_subs = _make_dummy_sv_subs(100)
    tracker = SemanticIncidentTracker(total_cues=100, batch_size=50)

    # Audit raises an exception (e.g. network failure, invalid API key)
    pipeline.translator.audit_batch_semantic_integrity = AsyncMock(
        side_effect=Exception("API_KEY_INVALID: 400 Bad Request")
    )

    report = await pipeline.check_semantic_cue_alignment(
        source_subs=source_subs,
        translated_subs=translated_subs,
        target_language="Swedish",
        source_language="English",
        batch_size=50,
        incident_tracker=tracker
    )

    # No batch may be ALIGNED when audit raised an exception
    for b in tracker.get_batches():
        assert b.state != BatchSemanticState.ALIGNED, (
            f"Batch {b.batch_idx} was marked ALIGNED after audit exception — fail-open regression!"
        )


@pytest.mark.asyncio
async def test_fail_closed_audit_returns_empty_dict_never_aligned():
    """
    P0 REGRESSION: If audit_batch_semantic_integrity returns {} (e.g. translator-level
    exception already caught), the pipeline must NOT silently mark batches ALIGNED.
    """
    pipeline = SubtitlePipeline()
    source_subs = _make_dummy_subs(100)
    translated_subs = _make_dummy_sv_subs(100)
    tracker = SemanticIncidentTracker(total_cues=100, batch_size=50)

    # Translator-level exception → returns {}
    pipeline.translator.audit_batch_semantic_integrity = AsyncMock(return_value={})

    report = await pipeline.check_semantic_cue_alignment(
        source_subs=source_subs,
        translated_subs=translated_subs,
        target_language="Swedish",
        source_language="English",
        batch_size=50,
        incident_tracker=tracker
    )

    for b in tracker.get_batches():
        assert b.state != BatchSemanticState.ALIGNED, (
            f"Batch {b.batch_idx} was marked ALIGNED when audit returned {{}} — fail-open regression!"
        )


@pytest.mark.asyncio
async def test_fail_closed_missing_batch_id_in_response_is_uncertain():
    """
    P0 REGRESSION: If the AI returns results for batch 1, 2, 4, 5 but omits batch 3,
    batch 3 must be marked SUSPECT (UNCERTAIN), never ALIGNED.
    """
    pipeline = SubtitlePipeline()
    source_subs = _make_dummy_subs(250)
    translated_subs = _make_dummy_sv_subs(250)
    tracker = SemanticIncidentTracker(total_cues=250, batch_size=50)
    # 5 batches expected: ids 1..5

    async def mock_audit_missing_batch_3(batch_payloads, **kwargs):
        # Returns results for all except batch_id 3
        return {
            bp["batch_id"]: {"batch_id": bp["batch_id"], "verdict": "ALIGNED", "confidence": "HIGH", "details": ""}
            for bp in batch_payloads
            if bp["batch_id"] != 3
        }

    pipeline.translator.audit_batch_semantic_integrity = AsyncMock(
        side_effect=mock_audit_missing_batch_3
    )

    report = await pipeline.check_semantic_cue_alignment(
        source_subs=source_subs,
        translated_subs=translated_subs,
        target_language="Swedish",
        source_language="English",
        batch_size=50,
        incident_tracker=tracker
    )

    b3 = tracker.find_batch_by_index(2)  # batch_idx=2 → batch_id=3
    assert b3 is not None
    assert b3.state != BatchSemanticState.ALIGNED, (
        "Batch 3 was marked ALIGNED when it was missing from AI response — fail-open regression!"
    )
    # Batches 1, 2, 4, 5 must be ALIGNED (they had explicit ALIGNED results)
    assert tracker.find_batch_by_index(0).state == BatchSemanticState.ALIGNED
    assert tracker.find_batch_by_index(1).state == BatchSemanticState.ALIGNED
    assert tracker.find_batch_by_index(3).state == BatchSemanticState.ALIGNED
    assert tracker.find_batch_by_index(4).state == BatchSemanticState.ALIGNED


@pytest.mark.asyncio
async def test_fail_closed_invalid_verdict_clamped_to_uncertain():
    """
    P0 REGRESSION: If the AI returns an unrecognized verdict string,
    it must be treated as UNCERTAIN (fail-closed), never silently accepted as ALIGNED.
    The batch must not be marked ALIGNED.
    """
    pipeline = SubtitlePipeline()
    source_subs = _make_dummy_subs(50)
    translated_subs = _make_dummy_sv_subs(50)
    tracker = SemanticIncidentTracker(total_cues=50, batch_size=50)

    # AI returns a gibberish verdict
    pipeline.translator.audit_batch_semantic_integrity = AsyncMock(return_value={
        1: {"batch_id": 1, "verdict": "PROBABLY_FINE", "confidence": "HIGH", "details": "Looks okay"}
    })

    report = await pipeline.check_semantic_cue_alignment(
        source_subs=source_subs,
        translated_subs=translated_subs,
        target_language="Swedish",
        source_language="English",
        batch_size=50,
        incident_tracker=tracker
    )

    b = tracker.find_batch_by_index(0)
    assert b.state != BatchSemanticState.ALIGNED, (
        "Batch was marked ALIGNED with invalid verdict 'PROBABLY_FINE' — fail-open regression!"
    )


@pytest.mark.asyncio
async def test_fail_closed_provider_unavailable_marks_suspect():
    """
    P0 REGRESSION: A provider unavailability (ProviderError or similar) must NOT
    silently mark batches ALIGNED. Batches must be SUSPECT (fail-closed).
    """
    pipeline = SubtitlePipeline()
    source_subs = _make_dummy_subs(100)
    translated_subs = _make_dummy_sv_subs(100)
    tracker = SemanticIncidentTracker(total_cues=100, batch_size=50)

    # Simulate provider unavailable
    class ProviderUnavailableError(Exception):
        pass

    pipeline.translator.audit_batch_semantic_integrity = AsyncMock(
        side_effect=ProviderUnavailableError("503 Service Unavailable")
    )

    report = await pipeline.check_semantic_cue_alignment(
        source_subs=source_subs,
        translated_subs=translated_subs,
        target_language="Swedish",
        source_language="English",
        batch_size=50,
        incident_tracker=tracker
    )

    for b in tracker.get_batches():
        assert b.state != BatchSemanticState.ALIGNED, (
            f"Batch {b.batch_idx} was ALIGNED after provider unavailable — fail-open regression!"
        )


@pytest.mark.asyncio
async def test_explicit_aligned_high_confidence_is_accepted():
    """
    P0 POSITIVE: An explicit ALIGNED verdict with HIGH confidence from audit
    MUST set batch state to ALIGNED. This verifies the happy path works.
    """
    pipeline = SubtitlePipeline()
    source_subs = _make_dummy_subs(100)
    translated_subs = _make_dummy_sv_subs(100)
    tracker = SemanticIncidentTracker(total_cues=100, batch_size=50)

    pipeline.translator.audit_batch_semantic_integrity = AsyncMock(return_value={
        1: {"batch_id": 1, "verdict": "ALIGNED", "confidence": "HIGH", "details": "Perfect 1-to-1"},
        2: {"batch_id": 2, "verdict": "ALIGNED", "confidence": "HIGH", "details": "Perfect 1-to-1"},
    })

    report = await pipeline.check_semantic_cue_alignment(
        source_subs=source_subs,
        translated_subs=translated_subs,
        target_language="Swedish",
        source_language="English",
        batch_size=50,
        incident_tracker=tracker
    )

    assert len(report["issues"]) == 0
    assert tracker.find_batch_by_index(0).state == BatchSemanticState.ALIGNED
    assert tracker.find_batch_by_index(1).state == BatchSemanticState.ALIGNED


@pytest.mark.asyncio
async def test_fail_closed_uncertain_verdict_never_aligned():
    """
    P0 REGRESSION: UNCERTAIN verdict must NEVER produce BatchSemanticState.ALIGNED.
    """
    pipeline = SubtitlePipeline()
    source_subs = _make_dummy_subs(50)
    translated_subs = _make_dummy_sv_subs(50)
    tracker = SemanticIncidentTracker(total_cues=50, batch_size=50)

    pipeline.translator.audit_batch_semantic_integrity = AsyncMock(return_value={
        1: {"batch_id": 1, "verdict": "UNCERTAIN", "confidence": "LOW", "details": "Ambiguous non-verbal cues"}
    })

    report = await pipeline.check_semantic_cue_alignment(
        source_subs=source_subs,
        translated_subs=translated_subs,
        target_language="Swedish",
        source_language="English",
        batch_size=50,
        incident_tracker=tracker
    )

    b = tracker.find_batch_by_index(0)
    assert b.state != BatchSemanticState.ALIGNED, (
        "Batch was ALIGNED on UNCERTAIN verdict — fail-open regression!"
    )


@pytest.mark.asyncio
async def test_fail_closed_aligned_low_confidence_is_not_aligned():
    """
    P0 REGRESSION: ALIGNED verdict with LOW confidence must NOT set ALIGNED.
    Low-confidence ALIGNED is indistinguishable from UNCERTAIN and must fail-closed.
    """
    pipeline = SubtitlePipeline()
    source_subs = _make_dummy_subs(50)
    translated_subs = _make_dummy_sv_subs(50)
    tracker = SemanticIncidentTracker(total_cues=50, batch_size=50)

    pipeline.translator.audit_batch_semantic_integrity = AsyncMock(return_value={
        1: {"batch_id": 1, "verdict": "ALIGNED", "confidence": "LOW", "details": "Weak match"}
    })

    report = await pipeline.check_semantic_cue_alignment(
        source_subs=source_subs,
        translated_subs=translated_subs,
        target_language="Swedish",
        source_language="English",
        batch_size=50,
        incident_tracker=tracker
    )

    b = tracker.find_batch_by_index(0)
    assert b.state != BatchSemanticState.ALIGNED, (
        "Batch was ALIGNED with LOW confidence ALIGNED — fail-open regression!"
    )


# ==============================================================================
# 9. P1 SAMPLING COVERAGE TESTS — CORRUPTION BETWEEN STANDARD SAMPLES
# ==============================================================================

def test_sampling_covers_batch_boundaries_always():
    """
    P1: extract_batch_alignment_samples MUST always include the first and last
    valid cue in the batch (batch boundaries), regardless of max_pairs.
    """
    source_subs = _make_dummy_subs(620)
    translated_subs = _make_dummy_sv_subs(620)

    samples = extract_batch_alignment_samples(
        source_subs, translated_subs,
        start_idx=300, end_idx=449, max_pairs=8
    )
    ids = [s["id"] for s in samples]
    # First valid cue in batch (0-indexed 300 → id=301) must be present
    assert 301 in ids, f"Batch start boundary (id=301) missing from samples: {ids}"
    # Last valid cue in batch (0-indexed 449 → id=450) must be present
    assert 450 in ids, f"Batch end boundary (id=450) missing from samples: {ids}"


def test_sampling_detects_corruption_between_standard_samples():
    """
    P1: Corruption that falls between standard start/mid/end sample positions
    (e.g. at cue 380 in batch 301..450) must still be picked up via split-sentence
    hotspot detection or anomaly_indices hint.

    This test injects a split-sentence at position 375 (no terminal punctuation),
    which means position 376 is a hotspot. anomaly_indices=[379] also guarantees
    that the corruption neighborhood is included.
    """
    import srt, datetime

    # Build 150-cue batch with a split-sentence at index 375 (batch cue 76 of 150)
    source_subs = []
    for i in range(620):
        start = datetime.timedelta(seconds=10 + i * 3)
        end = start + datetime.timedelta(seconds=2)
        if i == 374:
            # Split sentence — no terminal punctuation → next cue (375) is a hotspot
            content = "And by the time I arrived"
        else:
            content = f"Line {i + 1} original text dialogue."
        source_subs.append(srt.Subtitle(index=i + 1, start=start, end=end, content=content))

    translated_subs = _make_dummy_sv_subs(620)

    # With anomaly_indices hint for the corruption zone
    samples = extract_batch_alignment_samples(
        source_subs, translated_subs,
        start_idx=300, end_idx=449,
        max_pairs=8,
        anomaly_indices=[379]  # Hint: corruption around cue 380
    )
    ids = [s["id"] for s in samples]

    # The anomaly at 379 (id=380) OR the split-sentence hotspot at 375 (id=376)
    # must be included in the samples
    hotspot_covered = any(374 <= (sid - 1) <= 382 for sid in ids)
    assert hotspot_covered, (
        f"Corruption zone (cues 375-382) not covered by samples: {ids}. "
        f"Sampling must include split-sentence hotspots and anomaly hints."
    )


# ==============================================================================
# 10. CONTRACT HARDENING — DUPLICATE / UNKNOWN BATCH_ID REGRESSION TESTS
# ==============================================================================

import json as _json


def _make_payloads(*batch_ids):
    """Helper: make minimal batch_payloads list for given batch_ids."""
    return [{"batch_id": bid, "start_id": bid * 10, "end_id": bid * 10 + 9, "samples": []} for bid in batch_ids]


def _llm_response(*items):
    """Helper: serialise item dicts into the JSON format audit_batch_semantic_integrity expects."""
    return _json.dumps({"batches": list(items)})


@pytest.mark.skip_hermetic_audit
@pytest.mark.asyncio
async def test_contract_duplicate_batch_id_never_aligned():
    """
    CONTRACT: If the AI returns batch_id 3 twice in the same response,
    batch 3 MUST be UNCERTAIN/LOW — never ALIGNED via last-write-wins.
    All requested batches in the chunk become UNCERTAIN (tainted response).
    """
    translator = SubtitleTranslator()
    payloads = _make_payloads(1, 2, 3)

    # AI returns batch 3 twice — one ALIGNED and one CORRUPT duplicate
    response = _llm_response(
        {"batch_id": 1, "verdict": "ALIGNED", "confidence": "HIGH", "details": "ok"},
        {"batch_id": 2, "verdict": "ALIGNED", "confidence": "HIGH", "details": "ok"},
        {"batch_id": 3, "verdict": "ALIGNED", "confidence": "HIGH", "details": "first"},
        {"batch_id": 3, "verdict": "CORRUPT", "confidence": "HIGH", "details": "duplicate"},
    )

    with patch.object(translator, "_dispatch_llm_completion", AsyncMock(return_value=response)):
        with patch("app.core.ai_providers.context_from_settings",
                   return_value=type("ctx", (), {"provider": "gemini", "model": "test"})()),  \
             patch("app.core.ai_providers.resolve_job_provider_context",
                   return_value=type("ctx", (), {"provider": "gemini", "model": "test"})()):
            results = await translator.audit_batch_semantic_integrity(payloads)

    assert 3 in results
    assert results[3]["verdict"] != "ALIGNED", (
        f"Duplicate batch_id 3 must NOT be ALIGNED — got {results[3]['verdict']}"
    )
    assert results[3]["verdict"] == "UNCERTAIN"
    assert results[3]["confidence"] == "LOW"
    # All batches in tainted chunk are UNCERTAIN
    for bid in (1, 2, 3):
        assert results[bid]["verdict"] == "UNCERTAIN", (
            f"Batch {bid} should be UNCERTAIN after duplicate contract violation, got {results[bid]['verdict']}"
        )


@pytest.mark.skip_hermetic_audit
@pytest.mark.asyncio
async def test_contract_unknown_batch_id_causes_chunk_fail_closed():
    """
    CONTRACT: If the AI returns batch_id=99 (not in requested={1,2,3}),
    all requested batches must become UNCERTAIN (tainted response).
    The unknown ID must NOT cause any requested batch to be ALIGNED.
    """
    translator = SubtitleTranslator()
    payloads = _make_payloads(1, 2, 3)

    # AI returns valid 1+2+3 AND an extra unknown 99
    response = _llm_response(
        {"batch_id": 1, "verdict": "ALIGNED", "confidence": "HIGH", "details": "ok"},
        {"batch_id": 2, "verdict": "ALIGNED", "confidence": "HIGH", "details": "ok"},
        {"batch_id": 3, "verdict": "ALIGNED", "confidence": "HIGH", "details": "ok"},
        {"batch_id": 99, "verdict": "ALIGNED", "confidence": "HIGH", "details": "hallucinated"},
    )

    with patch.object(translator, "_dispatch_llm_completion", AsyncMock(return_value=response)):
        with patch("app.core.ai_providers.context_from_settings",
                   return_value=type("ctx", (), {"provider": "gemini", "model": "test"})()),  \
             patch("app.core.ai_providers.resolve_job_provider_context",
                   return_value=type("ctx", (), {"provider": "gemini", "model": "test"})()):
            results = await translator.audit_batch_semantic_integrity(payloads)

    # No requested batch should be ALIGNED — entire chunk is tainted
    for bid in (1, 2, 3):
        assert results[bid]["verdict"] == "UNCERTAIN", (
            f"Batch {bid} should be UNCERTAIN after unknown_id contract violation, got {results[bid]['verdict']}"
        )
    # Unknown id 99 must not appear in results at all
    assert 99 not in results


@pytest.mark.skip_hermetic_audit
@pytest.mark.asyncio
async def test_contract_duplicate_with_otherwise_aligned_payload_fails_closed():
    """
    CONTRACT: Even if all OTHER batches would be ALIGNED,
    a single duplicate batch_id causes the entire chunk to fail-closed.
    """
    translator = SubtitleTranslator()
    payloads = _make_payloads(1, 2, 3, 4, 5)

    # Batches 1–5 all look ALIGNED, but batch 2 appears twice
    response = _llm_response(
        {"batch_id": 1, "verdict": "ALIGNED", "confidence": "HIGH", "details": "ok"},
        {"batch_id": 2, "verdict": "ALIGNED", "confidence": "HIGH", "details": "first"},
        {"batch_id": 2, "verdict": "ALIGNED", "confidence": "HIGH", "details": "duplicate"},
        {"batch_id": 3, "verdict": "ALIGNED", "confidence": "HIGH", "details": "ok"},
        {"batch_id": 4, "verdict": "ALIGNED", "confidence": "HIGH", "details": "ok"},
        {"batch_id": 5, "verdict": "ALIGNED", "confidence": "HIGH", "details": "ok"},
    )

    with patch.object(translator, "_dispatch_llm_completion", AsyncMock(return_value=response)):
        with patch("app.core.ai_providers.context_from_settings",
                   return_value=type("ctx", (), {"provider": "gemini", "model": "test"})()),  \
             patch("app.core.ai_providers.resolve_job_provider_context",
                   return_value=type("ctx", (), {"provider": "gemini", "model": "test"})()):
            results = await translator.audit_batch_semantic_integrity(payloads)

    # Every batch must be UNCERTAIN — no ALIGNED sneaks through
    for bid in (1, 2, 3, 4, 5):
        assert results[bid]["verdict"] == "UNCERTAIN", (
            f"Batch {bid} must be UNCERTAIN when any duplicate exists in response, got {results[bid]['verdict']}"
        )
    assert results[2]["confidence"] == "LOW"


@pytest.mark.skip_hermetic_audit
@pytest.mark.asyncio
async def test_contract_fully_valid_response_accepted_as_aligned():
    """
    CONTRACT POSITIVE: A fully valid response with correct batch_ids,
    no duplicates, no unknowns, and ALIGNED HIGH verdicts MUST be accepted.
    """
    translator = SubtitleTranslator()
    payloads = _make_payloads(1, 2, 3)

    response = _llm_response(
        {"batch_id": 1, "verdict": "ALIGNED", "confidence": "HIGH", "details": "perfect"},
        {"batch_id": 2, "verdict": "ALIGNED", "confidence": "HIGH", "details": "perfect"},
        {"batch_id": 3, "verdict": "ALIGNED", "confidence": "HIGH", "details": "perfect"},
    )

    with patch.object(translator, "_dispatch_llm_completion", AsyncMock(return_value=response)):
        with patch("app.core.ai_providers.context_from_settings",
                   return_value=type("ctx", (), {"provider": "gemini", "model": "test"})()),  \
             patch("app.core.ai_providers.resolve_job_provider_context",
                   return_value=type("ctx", (), {"provider": "gemini", "model": "test"})()):
            results = await translator.audit_batch_semantic_integrity(payloads)

    for bid in (1, 2, 3):
        assert results[bid]["verdict"] == "ALIGNED", (
            f"Clean response batch {bid} must be ALIGNED, got {results[bid]['verdict']}"
        )
        assert results[bid]["confidence"] == "HIGH"
