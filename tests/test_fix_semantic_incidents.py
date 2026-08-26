import pytest
from unittest.mock import patch, AsyncMock
from app.core.validator import SemanticIncidentTracker, AlignmentIncident, IncidentState, BatchSemanticState, PrimaryBatchInfo
from app.services.pipeline import SubtitlePipeline
from app.services.translator import SubtitleTranslator
import srt
import datetime

def _make_dummy_subs(count, prefix="Sub"):
    subs = []
    for i in range(count):
        start = datetime.timedelta(seconds=i*5)
        end = datetime.timedelta(seconds=i*5+4)
        subs.append(srt.Subtitle(i+1, start, end, f"{prefix} {i+1}"))
    return subs

@pytest.mark.asyncio
async def test_a_real_primary_batch_attempt_1_fail_attempt_2_success():
    pipeline = SubtitlePipeline()
    pipeline.translator = SubtitleTranslator()

    source_subs = _make_dummy_subs(100, prefix="Source")
    target_subs = _make_dummy_subs(100, prefix="Target")

    tracker = SemanticIncidentTracker(total_cues=100, batch_size=50)
    batch = tracker.find_batch_by_index(0)
    batch.state = BatchSemanticState.CONFIRMED_CORRUPT

    # Setup incident that overlaps
    inc = AlignmentIncident(start_idx=10, end_idx=20, verdict="SHIFT_MINUS_1", confidence="HIGH")
    tracker.register_or_merge([inc])
    batch.state = BatchSemanticState.CONFIRMED_CORRUPT # Re-set as it gets overwritten to SUSPECT

    call_count = 0
    async def mock_translate_batch(sub_payload, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # structural failure on attempt 1 (duplicate id)
            return [{"id": sub_payload[0]["id"], "text": "Dup"}, {"id": sub_payload[0]["id"], "text": "Dup2"}]
        else:
            # structurally valid on attempt 2
            return [{"id": p["id"], "text": f"Repaired {p['id']}"} for p in sub_payload]

    async def mock_verify_repaired_batch_integrity(**kwargs):
        return {"verdict": "ALIGNED", "confidence": "HIGH"}

    async def mock_confirm(batch_id, **kwargs):
        return {"verdict": "ALIGNED", "confidence": "HIGH"}

    with patch.object(pipeline.translator, "translate_batch", new_callable=AsyncMock, side_effect=mock_translate_batch), \
         patch.object(pipeline.translator, "verify_repaired_batch_integrity", new_callable=AsyncMock, side_effect=mock_verify_repaired_batch_integrity), \
         patch.object(pipeline.translator, "confirm_batch_semantic_integrity", new_callable=AsyncMock, side_effect=mock_confirm):
        
        success = await pipeline._repair_semantic_alignment_incidents(
            subs=source_subs,
            translated_subs=target_subs,
            incidents=[inc],
            target_language="sv",
            source_language="en",
            incident_tracker=tracker
        )
    
    # Assert exact 2 attempts
    assert batch.repair_attempts == 2
    assert batch.state == BatchSemanticState.REPAIRED
    # Assert candidate 1 never mutates target, but candidate 2 commits atomically
    assert target_subs[10].content == "Repaired 11"
    # Assert incident resolved
    assert inc.state == IncidentState.VERIFIED
    # active semantic issues = 0
    issues = tracker.get_all_active_issues()
    assert len(issues) == 0

@pytest.mark.asyncio
async def test_b_real_primary_batch_attempt_1_fail_attempt_2_fail():
    pipeline = SubtitlePipeline()
    pipeline.translator = SubtitleTranslator()

    source_subs = _make_dummy_subs(100, prefix="Source")
    target_subs = _make_dummy_subs(100, prefix="Target")

    tracker = SemanticIncidentTracker(total_cues=100, batch_size=50)
    batch = tracker.find_batch_by_index(0)
    batch.state = BatchSemanticState.CONFIRMED_CORRUPT
    inc = AlignmentIncident(start_idx=10, end_idx=20, verdict="SHIFT_MINUS_1", confidence="HIGH")
    tracker.register_or_merge([inc])
    batch.state = BatchSemanticState.CONFIRMED_CORRUPT

    async def mock_translate_batch(sub_payload, **kwargs):
        # structural failure every time
        return [{"id": 9999, "text": "Bad"}]

    async def mock_confirm(batch_id, **kwargs):
        return {"verdict": "ALIGNED", "confidence": "HIGH"}

    with patch.object(pipeline.translator, "translate_batch", new_callable=AsyncMock, side_effect=mock_translate_batch), \
         patch.object(pipeline.translator, "confirm_batch_semantic_integrity", new_callable=AsyncMock, side_effect=mock_confirm):
        success = await pipeline._repair_semantic_alignment_incidents(
            subs=source_subs,
            translated_subs=target_subs,
            incidents=[inc],
            target_language="sv",
            source_language="en",
            incident_tracker=tracker
        )
    
    assert batch.repair_attempts == 2
    assert batch.state == BatchSemanticState.FAILED_REPAIR
    assert target_subs[10].content == "Target 11" # NO mutation
    assert inc.state == IncidentState.DISCOVERED # incident active
    assert len(tracker.get_all_active_issues()) > 0
    assert not success

@pytest.mark.asyncio
async def test_c_initial_suspect_incident_confirmation_aligned_high():
    # TEST C: initial SUSPECT incident, confirmation ALIGNED HIGH -> stale incident inte når final QA
    pipeline = SubtitlePipeline()
    pipeline.translator = SubtitleTranslator()

    source_subs = _make_dummy_subs(50, prefix="Source")
    target_subs = _make_dummy_subs(50, prefix="Target")

    tracker = SemanticIncidentTracker(total_cues=50, batch_size=50)
    inc = AlignmentIncident(start_idx=0, end_idx=49, verdict="SHIFT_MINUS_1", confidence="HIGH")
    tracker.register_or_merge([inc])

    async def mock_confirm(**kwargs):
        return {"verdict": "ALIGNED", "confidence": "HIGH"}

    with patch.object(pipeline.translator, "confirm_batch_semantic_integrity", new_callable=AsyncMock, side_effect=mock_confirm):
        success = await pipeline._repair_semantic_alignment_incidents(
            subs=source_subs,
            translated_subs=target_subs,
            incidents=[inc],
            target_language="sv",
            source_language="en",
            incident_tracker=tracker
        )

    batch = tracker.find_batch_by_index(0)
    assert batch.state == BatchSemanticState.ALIGNED
    assert inc.state == IncidentState.VERIFIED
    assert len(tracker.get_all_active_issues()) == 0

@pytest.mark.asyncio
async def test_d_separate_unconfirmed_incident_remains_active():
    # TEST D: en separat/unconfirmed incident finns samtidigt, får INTE resolve unrelated incident
    pipeline = SubtitlePipeline()
    pipeline.translator = SubtitleTranslator()

    source_subs = _make_dummy_subs(100, prefix="Source")
    target_subs = _make_dummy_subs(100, prefix="Target")

    tracker = SemanticIncidentTracker(total_cues=100, batch_size=50)
    inc_a = AlignmentIncident(start_idx=0, end_idx=49, verdict="SHIFT_MINUS_1", confidence="HIGH")
    inc_b = AlignmentIncident(start_idx=60, end_idx=99, verdict="SHIFT_PLUS_1", confidence="HIGH")
    tracker.register_or_merge([inc_a, inc_b])

    # Only batch 0 gets confirmed ALIGNED. Batch 1 gets confirmed CORRUPT.
    async def mock_confirm(batch_id, **kwargs):
        if batch_id == 1:
            return {"verdict": "ALIGNED", "confidence": "HIGH"}
        else:
            return {"verdict": "SHIFT_PLUS_1", "confidence": "HIGH"}

    # Mock the actual repair out so we only test confirmation and resolution
    async def mock_translate_batch(sub_payload, **kwargs):
        return [{"id": 9999, "text": "Bad"}]

    with patch.object(pipeline.translator, "confirm_batch_semantic_integrity", new_callable=AsyncMock, side_effect=mock_confirm), \
         patch.object(pipeline.translator, "translate_batch", new_callable=AsyncMock, side_effect=mock_translate_batch):
        success = await pipeline._repair_semantic_alignment_incidents(
            subs=source_subs,
            translated_subs=target_subs,
            incidents=[inc_a, inc_b],
            target_language="sv",
            source_language="en",
            incident_tracker=tracker
        )

    batch_a = tracker.find_batch_by_index(0)
    batch_b = tracker.find_batch_by_index(1)
    
    assert batch_a.state == BatchSemanticState.ALIGNED
    assert batch_b.state == BatchSemanticState.FAILED_REPAIR
    
    assert inc_a.state == IncidentState.VERIFIED
    assert inc_b.state == IncidentState.DISCOVERED
    issues = tracker.get_all_active_issues()
    assert len(issues) > 0
    assert any("SHIFT_PLUS_1" in iss for iss in issues)

