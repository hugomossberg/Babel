import pytest
import srt
from datetime import timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from app.core.validator import (
    SemanticIncidentTracker,
    AlignmentIncident,
    IncidentState,
    BatchSemanticState,
    PrimaryBatchInfo,
)
from app.services.pipeline import SubtitlePipeline, qa_gate
from app.services.translator import (
    SubtitleTranslator,
    validate_recovery_batch_results,
)


def _make_dummy_subs(count: int, prefix: str = "Cue"):
    return [
        srt.Subtitle(
            index=i + 1,
            start=timedelta(seconds=i * 2),
            end=timedelta(seconds=i * 2 + 1),
            content=f"{prefix} {i + 1}"
        )
        for i in range(count)
    ]


# ===========================================================================
# TEST A: Structural failure diagnostics
# ===========================================================================
def test_a_structural_failure_diagnostics():
    """Verifies that validate_recovery_batch_results produces explicit diagnostic
    fields on missing IDs, duplicate IDs, non-integer IDs, and malformed items,
    and fails closed."""
    expected_items = [
        {"id": 1, "text": "First dialogue line"},
        {"id": 2, "text": "Second dialogue line"},
        {"id": 3, "text": "Named Entity Item"},
        {"id": 4, "text": "Fourth dialogue line"}
    ]

    # Raw results with missing ID 4, duplicate ID 2, unknown ID 99, malformed non-dict and bad ID
    raw_results = [
        {"id": 1, "text": "Translated first dialogue line"},
        {"id": 2, "text": "Translated second dialogue line"},
        {"id": 2, "text": "Duplicate second dialogue line"},
        {"id": 99, "text": "Unknown cue translation"},
        {"id": "invalid_int_id", "text": "Malformed ID cue"},
        "malformed_non_dict_item",
        {"id": 3, "text": None}  # Non-string text
    ]

    valid_map, report = validate_recovery_batch_results(expected_items, raw_results)

    assert report["is_clean"] is False
    assert report["is_structurally_clean"] is False
    assert 4 in report["missing_ids"]
    assert 2 in report["duplicate_ids"]
    assert 99 in report["unknown_ids"]
    assert report["malformed_count"] == 3  # non-integer string id, non-dict item, non-string text
    assert len(valid_map) < len(expected_items)


# ===========================================================================
# TEST B: Identical valid cues are not rejected as structural failure
# ===========================================================================
def test_b_identical_valid_cues_not_rejected_as_structural_failure():
    """Verifies that sub-batch recovery translation containing legitimate identical
    strings (e.g. proper names, titles, entity keeps) is accepted as structurally clean."""
    expected_items = [
        {"id": 1, "text": "START OF BROADCAST SEGMENT."},
        {"id": 2, "text": "GUEST SPEAKER ALPHA IS PRESENT."},
        {"id": 3, "text": "PROPER NOUN BETA."},
        {"id": 4, "text": "MUSICAL PERFORMANCE GAMMA."}
    ]

    # Model preserves identical proper names
    raw_results = [
        {"id": 1, "text": "TARGET START OF BROADCAST SEGMENT."},
        {"id": 2, "text": "TARGET GUEST SPEAKER ALPHA IS PRESENT."},
        {"id": 3, "text": "PROPER NOUN BETA."},
        {"id": 4, "text": "TARGET MUSICAL PERFORMANCE GAMMA."}
    ]

    valid_map, report = validate_recovery_batch_results(expected_items, raw_results)

    assert report["is_clean"] is True
    assert report["missing_ids"] == []
    assert report["unknown_ids"] == []
    assert report["duplicate_ids"] == []
    assert report["malformed_count"] == 0
    assert len(valid_map) == 4
    assert valid_map[3] == "PROPER NOUN BETA."


# ===========================================================================
# TEST C: New SUSPECT after ALIGNED is not automatically CORRUPT
# ===========================================================================
@pytest.mark.asyncio
async def test_c_new_suspect_after_aligned_is_not_automatically_corrupt():
    """Verifies that an ALIGNED batch audited again and flagged as SUSPECT does not
    block QA without confirmation. When Stage B confirmation says ALIGNED, the batch
    remains non-blocking and QA passes."""
    pipeline = SubtitlePipeline()
    pipeline.translator = SubtitleTranslator()

    source_subs = _make_dummy_subs(150, prefix="Source dialogue sequence")
    target_subs = _make_dummy_subs(150, prefix="Target translated sequence")

    tracker = SemanticIncidentTracker(total_cues=150, batch_size=50)

    # Initial state: batch 0 was previously confirmed ALIGNED
    batch_0 = tracker.find_batch_by_index(0)
    batch_0.state = BatchSemanticState.ALIGNED

    # Later audit triggers and yields SUSPECT finding for batch 0
    suspect_inc = AlignmentIncident(start_idx=0, end_idx=49, verdict="SHIFT_MINUS_1", confidence="MEDIUM", details="Suspicious drift")
    tracker.register_or_merge([suspect_inc])

    # SUSPECT in tracker is not confirmed corruption
    assert batch_0.state == BatchSemanticState.SUSPECT
    assert len(tracker.get_active_corrupt_batches()) == 0
    assert len(tracker.get_all_active_issues()) == 0

    # Stage B Confirmation runs and confirms ALIGNED
    async def mock_confirm(**kwargs):
        return {"verdict": "ALIGNED", "confidence": "HIGH"}

    with patch.object(pipeline.translator, "confirm_batch_semantic_integrity", new_callable=AsyncMock, side_effect=mock_confirm):
        success = await pipeline._repair_semantic_alignment_incidents(
            subs=source_subs,
            translated_subs=target_subs,
            incidents=[suspect_inc],
            target_language="synthetic_target",
            source_language="synthetic_source",
            incident_tracker=tracker
        )

    assert success is True
    assert batch_0.state == BatchSemanticState.ALIGNED
    assert suspect_inc.state == IncidentState.VERIFIED
    assert len(tracker.get_all_active_issues()) == 0

    # QA Gate passes cleanly with isolated language check
    with patch("app.services.pipeline.check_language_representative", return_value={"confident_wrong_language": False, "detected_lang": "synthetic_target", "confidence": 1.0, "section": "none", "wrong_language_cue_ids": [], "legit_foreign_cue_ids": []}):
        qa_res = qa_gate(
            source_subs,
            target_subs,
            target_lang_code="synthetic_target",
            source_language_name="synthetic_source",
            semantic_alignment_issues=tracker.get_all_active_issues(),
            allow_warnings=True
        )
    assert qa_res["passed"] is True


# ===========================================================================
# TEST D: New SUSPECT can still become real corruption when confirmed
# ===========================================================================
@pytest.mark.asyncio
async def test_d_new_suspect_becomes_real_corruption_when_confirmed():
    """Verifies that when a re-audited SUSPECT batch is confirmed as CORRUPT by
    Stage B confirmation and recovery fails, it becomes an active QA-blocking issue."""
    pipeline = SubtitlePipeline()
    pipeline.translator = SubtitleTranslator()

    source_subs = _make_dummy_subs(150, prefix="Source dialogue sequence")
    target_subs = _make_dummy_subs(150, prefix="Target translated sequence")

    tracker = SemanticIncidentTracker(total_cues=150, batch_size=50)
    batch_0 = tracker.find_batch_by_index(0)
    batch_0.state = BatchSemanticState.ALIGNED

    # New audit finding
    suspect_inc = AlignmentIncident(start_idx=0, end_idx=49, verdict="SHIFT_MINUS_1", confidence="HIGH", details="Real shift")
    tracker.register_or_merge([suspect_inc])

    # Confirmation says CORRUPT
    async def mock_confirm(**kwargs):
        return {"verdict": "SHIFT_MINUS_1", "confidence": "HIGH"}

    # Recovery fails
    async def mock_translate_batch(sub_payload, **kwargs):
        return [{"id": 9999, "text": "Malformed"}]

    with patch.object(pipeline.translator, "confirm_batch_semantic_integrity", new_callable=AsyncMock, side_effect=mock_confirm), \
         patch.object(pipeline.translator, "translate_batch", new_callable=AsyncMock, side_effect=mock_translate_batch):
        success = await pipeline._repair_semantic_alignment_incidents(
            subs=source_subs,
            translated_subs=target_subs,
            incidents=[suspect_inc],
            target_language="synthetic_target",
            source_language="synthetic_source",
            incident_tracker=tracker
        )

    assert success is False
    assert batch_0.state == BatchSemanticState.FAILED_REPAIR
    active_issues = tracker.get_all_active_issues()
    assert len(active_issues) > 0
    assert any("SHIFT_MINUS_1" in iss for iss in active_issues)

    # QA Gate fails closed with isolated language check
    with patch("app.services.pipeline.check_language_representative", return_value={"confident_wrong_language": False, "detected_lang": "synthetic_target", "confidence": 1.0, "section": "none", "wrong_language_cue_ids": [], "legit_foreign_cue_ids": []}):
        qa_res = qa_gate(
            source_subs,
            target_subs,
            target_lang_code="synthetic_target",
            source_language_name="synthetic_source",
            semantic_alignment_issues=active_issues,
            allow_warnings=True
        )
    assert qa_res["passed"] is False
    assert any("Semantic alignment corruption" in iss for iss in qa_res["issues"])


# ===========================================================================
# TEST E: Recovery failure preserves fail-closed behavior without target mutation
# ===========================================================================
@pytest.mark.asyncio
async def test_e_recovery_failure_preserves_fail_closed_without_target_mutation():
    """Verifies that when recovery attempts fail or post-repair verification is not ALIGNED,
    no partial mutations are committed to target subtitles and QA fails closed."""
    pipeline = SubtitlePipeline()
    pipeline.translator = SubtitleTranslator()

    source_subs = _make_dummy_subs(100, prefix="Source dialogue sequence")
    target_subs = _make_dummy_subs(100, prefix="Original target translation")

    tracker = SemanticIncidentTracker(total_cues=100, batch_size=50)
    batch_0 = tracker.find_batch_by_index(0)
    batch_0.state = BatchSemanticState.CONFIRMED_CORRUPT

    inc = AlignmentIncident(start_idx=0, end_idx=49, verdict="SHIFT_MINUS_1", confidence="HIGH")
    tracker.register_or_merge([inc])
    batch_0.state = BatchSemanticState.CONFIRMED_CORRUPT

    # Sub-batch returns valid structure, but post-repair verification detects persistent misalignment
    async def mock_translate_batch(sub_payload, **kwargs):
        return [{"id": p["id"], "text": f"Candidate {p['id']}"} for p in sub_payload]

    async def mock_verify_repaired(**kwargs):
        return {"verdict": "SHIFT_MINUS_1", "confidence": "HIGH"}

    async def mock_confirm(**kwargs):
        return {"verdict": "ALIGNED", "confidence": "HIGH"}

    with patch.object(pipeline.translator, "confirm_batch_semantic_integrity", new_callable=AsyncMock, side_effect=mock_confirm), \
         patch.object(pipeline.translator, "translate_batch", new_callable=AsyncMock, side_effect=mock_translate_batch), \
         patch.object(pipeline.translator, "verify_repaired_batch_integrity", new_callable=AsyncMock, side_effect=mock_verify_repaired):
        success = await pipeline._repair_semantic_alignment_incidents(
            subs=source_subs,
            translated_subs=target_subs,
            incidents=[inc],
            target_language="synthetic_target",
            source_language="synthetic_source",
            incident_tracker=tracker
        )

    assert success is False
    assert batch_0.state == BatchSemanticState.FAILED_REPAIR
    # Original target content must remain completely untouched
    for i in range(50):
        assert target_subs[i].content == f"Original target translation {i + 1}"


# ===========================================================================
# TEST F: Successful recovery atomically repairs and clears QA blocker
# ===========================================================================
@pytest.mark.asyncio
async def test_f_successful_recovery_atomically_repairs_and_clears_qa_blocker():
    """Verifies that when a confirmed corrupt batch is recovered and post-repair verification
    confirms ALIGNED, translation commits atomically, state transitions to REPAIRED,
    and QA gate passes cleanly with 0 dropped cues and 0ms drift."""
    pipeline = SubtitlePipeline()
    pipeline.translator = SubtitleTranslator()

    source_subs = _make_dummy_subs(100, prefix="Source dialogue sequence")
    target_subs = _make_dummy_subs(100, prefix="Target translated sequence")

    tracker = SemanticIncidentTracker(total_cues=100, batch_size=50)
    batch_0 = tracker.find_batch_by_index(0)
    batch_0.state = BatchSemanticState.CONFIRMED_CORRUPT

    inc = AlignmentIncident(start_idx=0, end_idx=49, verdict="SHIFT_MINUS_1", confidence="HIGH")
    tracker.register_or_merge([inc])
    batch_0.state = BatchSemanticState.CONFIRMED_CORRUPT

    # Sub-batch returns repaired text (including invariant proper names)
    async def mock_translate_batch(sub_payload, **kwargs):
        return [
            {"id": p["id"], "text": "PROPER NOUN BETA" if p["id"] == 10 else f"Repaired target cue {p['id']}"}
            for p in sub_payload
        ]

    async def mock_verify_repaired(**kwargs):
        return {"verdict": "ALIGNED", "confidence": "HIGH"}

    async def mock_confirm(**kwargs):
        return {"verdict": "ALIGNED", "confidence": "HIGH"}

    with patch.object(pipeline.translator, "confirm_batch_semantic_integrity", new_callable=AsyncMock, side_effect=mock_confirm), \
         patch.object(pipeline.translator, "translate_batch", new_callable=AsyncMock, side_effect=mock_translate_batch), \
         patch.object(pipeline.translator, "verify_repaired_batch_integrity", new_callable=AsyncMock, side_effect=mock_verify_repaired):
        success = await pipeline._repair_semantic_alignment_incidents(
            subs=source_subs,
            translated_subs=target_subs,
            incidents=[inc],
            target_language="synthetic_target",
            source_language="synthetic_source",
            incident_tracker=tracker
        )

    assert success is True
    assert batch_0.state == BatchSemanticState.REPAIRED
    assert inc.state == IncidentState.VERIFIED
    assert len(tracker.get_all_active_issues()) == 0

    # Verify atomic update of target_subs
    assert target_subs[0].content == "Repaired target cue 1"
    assert target_subs[9].content == "PROPER NOUN BETA"
    assert target_subs[49].content == "Repaired target cue 50"

    # Strict timing preservation
    for i in range(100):
        assert target_subs[i].start == source_subs[i].start
        assert target_subs[i].end == source_subs[i].end

    # QA Gate passes cleanly with isolated language check
    with patch("app.services.pipeline.check_language_representative", return_value={"confident_wrong_language": False, "detected_lang": "synthetic_target", "confidence": 1.0, "section": "none", "wrong_language_cue_ids": [], "legit_foreign_cue_ids": []}):
        qa_res = qa_gate(
            source_subs,
            target_subs,
            target_lang_code="synthetic_target",
            source_language_name="synthetic_source",
            safe_ids=[9],  # cue 10 (0-indexed 9) is entity keep
            semantic_alignment_issues=tracker.get_all_active_issues(),
            allow_warnings=True
        )
    assert qa_res["passed"] is True
