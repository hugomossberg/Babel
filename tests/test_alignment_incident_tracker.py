import pytest
import srt
from datetime import timedelta
from unittest.mock import AsyncMock, patch, MagicMock
from app.core.validator import (
    AlignmentIncident,
    AlignmentRegion,
    IncidentState,
    SemanticIncidentTracker,
    cluster_alignment_findings,
)
from app.services.pipeline import SubtitlePipeline
from app.services.translator import SubtitleTranslator


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


def test_incident_state_machine_initialization():
    inc = AlignmentIncident(
        start_idx=10,
        end_idx=15,
        verdict="SHIFT_PLUS_1",
        confidence="HIGH"
    )
    assert inc.state == IncidentState.DISCOVERED
    assert inc.repair_attempts == 0
    assert inc.incident_id == "inc_10_15"


def test_semantic_incident_tracker_register_and_merge():
    tracker = SemanticIncidentTracker()
    inc1 = AlignmentIncident(start_idx=10, end_idx=15, verdict="SHIFT_PLUS_1", confidence="HIGH", details="First finding")
    registered = tracker.register_or_merge([inc1])
    assert len(registered) == 1
    assert registered[0] is inc1

    # Overlapping finding should merge into existing incident rather than duplicating
    inc2 = AlignmentIncident(start_idx=12, end_idx=17, verdict="SHIFT_PLUS_1", confidence="HIGH", details="Second finding")
    merged = tracker.register_or_merge([inc2])
    assert len(merged) == 1
    assert merged[0] is inc1
    assert inc1.start_idx == 10
    assert inc1.end_idx == 17
    assert "Second finding" in inc1.details


def test_semantic_incident_tracker_terminal_states():
    tracker = SemanticIncidentTracker()
    inc = AlignmentIncident(start_idx=10, end_idx=15, verdict="SHIFT_PLUS_1", confidence="HIGH")
    tracker.register_or_merge([inc])

    # Mark REPAIRED
    inc.state = IncidentState.REPAIRED
    repairable = tracker.get_repairable_incidents([inc])
    assert len(repairable) == 0

    # Mark FAILED_REPAIR
    inc.state = IncidentState.FAILED_REPAIR
    repairable = tracker.get_repairable_incidents([inc])
    assert len(repairable) == 0

    # Exhausted attempts
    inc.state = IncidentState.DISCOVERED
    inc.repair_attempts = 2
    repairable = tracker.get_repairable_incidents([inc])
    assert len(repairable) == 0
    assert inc.state == IncidentState.FAILED_REPAIR


def test_semantic_incident_tracker_active_issues():
    tracker = SemanticIncidentTracker()
    inc1 = AlignmentIncident(start_idx=5, end_idx=8, verdict="SHIFT_PLUS_1", confidence="HIGH", details="Shift detected")
    inc2 = AlignmentIncident(start_idx=20, end_idx=25, verdict="MERGED", confidence="HIGH", details="Merged lines")
    tracker.register_or_merge([inc1, inc2])

    inc1.state = IncidentState.REPAIRED
    inc2.state = IncidentState.FAILED_REPAIR

    active = tracker.get_all_active_issues()
    assert len(active) == 1
    assert "MERGED at cues 21-26" in active[0]


@pytest.mark.asyncio
async def test_hard_source_remap_prompt_isolation():
    translator = SubtitleTranslator()
    source_ctx = [
        {"id": 1, "text": "Hello"},
        {"id": 2, "text": "World"},
        {"id": 3, "text": "This is a test"},
        {"id": 4, "text": "Goodbye"}
    ]
    target_ctx = [
        {"id": 1, "text": "Hej"},
        {"id": 2, "text": "Världen"},
        {"id": 3, "text": "Felaktig rad"},
        {"id": 4, "text": "Adjö"}
    ]

    with patch.object(translator, "_dispatch_llm_completion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = '{"translations": [{"id": 2, "text": "Världen"}, {"id": 3, "text": "Detta är ett test"}]}'
        with patch("app.services.translator.get_setting", return_value="gemini"):
            res = await translator.repair_alignment_region(
                repair_cue_ids=[2, 3],
                source_context_items=source_ctx,
                target_context_items=target_ctx,
                target_language="Swedish",
                source_language="English",
                verdict="SHIFT_PLUS_1"
            )

            assert len(res) == 2
            # Verify system prompt contains HARD SOURCE REMAP contract
            call_kwargs = mock_llm.call_args.kwargs
            system_prompt = call_kwargs["system_prompt"]
            user_prompt = call_kwargs["user_prompt"]

            assert "HARD SOURCE REMAP" in system_prompt
            assert "SOURCE OF TRUTH TO TRANSLATE" in user_prompt
            # Target cues for repair IDs (2, 3) must NOT be passed as authoritative target items
            assert "LEFT CONTEXT (REFERENCE ONLY / DO NOT COPY / DO NOT REMAP)" in user_prompt
            assert "RIGHT CONTEXT (REFERENCE ONLY / DO NOT COPY / DO NOT REMAP)" in user_prompt


@pytest.mark.asyncio
async def test_repair_incidents_bounded_to_two_attempts():
    pipeline = SubtitlePipeline()
    pipeline.translator = SubtitleTranslator()

    source_subs = _make_dummy_subs(20, prefix="Source")
    target_subs = _make_dummy_subs(20, prefix="Target")

    inc = AlignmentIncident(
        start_idx=5,
        end_idx=8,
        verdict="SHIFT_PLUS_1",
        confidence="HIGH",
        confirmation_required=False
    )
    tracker = SemanticIncidentTracker()
    tracker.register_or_merge([inc])

    # Mock repair_alignment_region to return a candidate, but audit_cue_alignment_window always says UNCERTAIN (fails verify)
    with patch.object(pipeline.translator, "repair_alignment_region", new_callable=AsyncMock) as mock_repair, \
         patch.object(pipeline.translator, "audit_cue_alignment_window", new_callable=AsyncMock) as mock_verify:

        mock_repair.return_value = [
            {"id": j + 1, "text": f"Repaired {j + 1}"} for j in range(20)
        ]
        mock_verify.return_value = {"alignment_verdict": "UNCERTAIN", "confidence": "LOW"}

        success = await pipeline._repair_semantic_alignment_incidents(
            subs=source_subs,
            translated_subs=target_subs,
            incidents=[inc],
            target_language="Swedish",
            source_language="English",
            incident_tracker=tracker
        )

        assert not success
        # Exactly 2 attempts must have been made
        assert inc.repair_attempts == 2
        assert inc.state == IncidentState.FAILED_REPAIR
        # Target subs must remain unmodified (atomic commit prevented)
        for j in range(5, 9):
            assert target_subs[j].content == f"Target {j + 1}"


@pytest.mark.asyncio
async def test_repair_incidents_atomic_commit_on_success():
    pipeline = SubtitlePipeline()
    pipeline.translator = SubtitleTranslator()

    source_subs = _make_dummy_subs(20, prefix="Source")
    target_subs = _make_dummy_subs(20, prefix="Target")

    inc = AlignmentIncident(
        start_idx=5,
        end_idx=8,
        verdict="SHIFT_PLUS_1",
        confidence="HIGH",
        confirmation_required=False
    )
    tracker = SemanticIncidentTracker()
    tracker.register_or_merge([inc])

    with patch.object(pipeline.translator, "repair_alignment_region", new_callable=AsyncMock) as mock_repair, \
         patch.object(pipeline.translator, "audit_cue_alignment_window", new_callable=AsyncMock) as mock_verify:

        mock_repair.side_effect = lambda repair_cue_ids, **kwargs: [{"id": cid, "text": f"Repaired {cid}"} for cid in repair_cue_ids]
        mock_verify.return_value = {"alignment_verdict": "ALIGNED", "confidence": "HIGH"}

        success = await pipeline._repair_semantic_alignment_incidents(
            subs=source_subs,
            translated_subs=target_subs,
            incidents=[inc],
            target_language="Swedish",
            source_language="English",
            incident_tracker=tracker
        )

        assert success
        assert inc.repair_attempts == 1
        assert inc.state == IncidentState.REPAIRED
        # Target subs must be atomically committed with repaired text
        for j in range(5, 9):
            assert target_subs[j].content == f"Repaired {j + 1}"


@pytest.mark.asyncio
async def test_mutation_dirtying_no_re_repair_of_terminal_incident():
    pipeline = SubtitlePipeline()
    pipeline.translator = SubtitleTranslator()

    tracker = SemanticIncidentTracker()
    inc = AlignmentIncident(
        start_idx=5,
        end_idx=8,
        verdict="SHIFT_PLUS_1",
        confidence="HIGH",
        state=IncidentState.FAILED_REPAIR,
        repair_attempts=2
    )
    tracker.register_or_merge([inc])

    repairable = tracker.get_repairable_incidents([inc])
    assert len(repairable) == 0

    # Even if check_semantic_cue_alignment finds raw findings overlapping cues 5-8
    raw_findings = [{"start_idx": 5, "end_idx": 8, "verdict": "SHIFT_PLUS_1", "confidence": "HIGH", "details": ""}]
    clustered = cluster_alignment_findings(raw_findings, total_cues=20)
    merged = tracker.register_or_merge(clustered)
    repairable_after = tracker.get_repairable_incidents(merged)

    # Must STILL be 0 repairable incidents to prevent infinite cascade loops
    assert len(repairable_after) == 0


@pytest.mark.asyncio
async def test_boundary_guards_verification_window():
    """Verify that local verification window covers candidate patch plus 1-cue boundary guards."""
    pipeline = SubtitlePipeline()
    pipeline.translator = SubtitleTranslator()

    source_subs = _make_dummy_subs(20, prefix="Source")
    target_subs = _make_dummy_subs(20, prefix="Target")

    # Repair cues 5-8 (0-indexed indices 4..7)
    inc = AlignmentIncident(
        start_idx=4,
        end_idx=7,
        verdict="SHIFT_PLUS_1",
        confidence="HIGH",
        confirmation_required=False
    )
    tracker = SemanticIncidentTracker()
    tracker.register_or_merge([inc])

    recorded_verify_source = []
    recorded_verify_target = []

    async def mock_verify(source_items, target_items, **kwargs):
        nonlocal recorded_verify_source, recorded_verify_target
        recorded_verify_source = source_items
        recorded_verify_target = target_items
        return {"alignment_verdict": "ALIGNED", "confidence": "HIGH"}

    with patch.object(pipeline.translator, "repair_alignment_region", new_callable=AsyncMock) as mock_repair, \
         patch.object(pipeline.translator, "audit_cue_alignment_window", side_effect=mock_verify):

        mock_repair.side_effect = lambda repair_cue_ids, **kwargs: [{"id": cid, "text": f"Repaired {cid}"} for cid in repair_cue_ids]

        success = await pipeline._repair_semantic_alignment_incidents(
            subs=source_subs,
            translated_subs=target_subs,
            incidents=[inc],
            target_language="Swedish",
            source_language="English",
            incident_tracker=tracker
        )

        assert success
        # Check that verify window has left guard (index 3 = cue 4), repaired cues (indices 4..7 = cues 5..8), and right guard (index 8 = cue 9)
        verify_ids = [item["id"] for item in recorded_verify_source]
        assert verify_ids == [4, 5, 6, 7, 8, 9]

        # Verify candidate target contains left guard (Target 4), repaired cues (Repaired 5..8), and right guard (Target 9)
        assert recorded_verify_target[0]["text"] == "Target 4"
        for idx, cid in enumerate(range(5, 9)):
            assert recorded_verify_target[idx + 1]["text"] == f"Repaired {cid}"
        assert recorded_verify_target[-1]["text"] == "Target 9"


def test_qa_gate_fails_closed_on_semantic_alignment_issues_even_with_allow_warnings_true():
    """Verify that semantic alignment corruption fails closed even when allow_warnings=True."""
    from app.services.pipeline import qa_gate
    source_subs = _make_dummy_subs(5, prefix="Source")
    target_subs = _make_dummy_subs(5, prefix="Target")

    res = qa_gate(
        source_subs,
        target_subs,
        target_lang_code="sv",
        allow_warnings=True,
        semantic_alignment_issues=["SHIFT_PLUS_1 at cues 2-4: target cue 2 translates source cue 3"]
    )
    assert res["passed"] is False
    assert res["status"] == "FAIL"
    assert any("Semantic alignment corruption" in issue for issue in res["issues"])


@pytest.mark.asyncio
async def test_all_confirmed_incidents_repaired_without_arbitrary_slice_cap():
    """Verify that more than 4 confirmed incidents are all processed without truncation."""
    pipeline = SubtitlePipeline()
    pipeline.translator = SubtitleTranslator()

    source_subs = _make_dummy_subs(100, prefix="Source")
    target_subs = _make_dummy_subs(100, prefix="Target")

    # Create 6 distinct incidents spaced 12 cues apart (tolerance=4)
    incidents = [
        AlignmentIncident(start_idx=i * 12, end_idx=i * 12 + 2, verdict="SHIFT_PLUS_1", confidence="HIGH", confirmation_required=False)
        for i in range(6)
    ]
    tracker = SemanticIncidentTracker()
    tracker.register_or_merge(incidents)

    repaired_cue_ids = []

    async def mock_repair(repair_cue_ids, **kwargs):
        nonlocal repaired_cue_ids
        repaired_cue_ids.extend(repair_cue_ids)
        return [{"id": cid, "text": f"Repaired {cid}"} for cid in repair_cue_ids]

    with patch.object(pipeline.translator, "repair_alignment_region", side_effect=mock_repair), \
         patch.object(pipeline.translator, "audit_cue_alignment_window", return_value={"alignment_verdict": "ALIGNED", "confidence": "HIGH"}):

        success = await pipeline._repair_semantic_alignment_incidents(
            subs=source_subs,
            translated_subs=target_subs,
            incidents=incidents,
            target_language="Swedish",
            source_language="English",
            incident_tracker=tracker
        )

        assert success
        # All 6 incidents (each 3 cues = 18 total cues) must have been repaired
        assert len(repaired_cue_ids) == 18
        for inc in incidents:
            assert inc.state == IncidentState.REPAIRED


def test_generic_identical_word_no_cancer_special_case():
    """Verify that 'cancer' is not hardcoded and handled via standard QA policy."""
    from app.services.translator import SHARED_CROSS_LINGUAL_WORDS
    assert "cancer" not in SHARED_CROSS_LINGUAL_WORDS


@pytest.mark.asyncio
async def test_core_aligned_with_untracked_boundary_shift_is_rejected():
    """Verify that core ALIGNED + boundary SHIFT on supposedly clean baseline is rejected."""
    pipeline = SubtitlePipeline()
    pipeline.translator = SubtitleTranslator()

    source_subs = _make_dummy_subs(20, prefix="Source")
    target_subs = _make_dummy_subs(20, prefix="Target")

    inc = AlignmentIncident(
        start_idx=5,
        end_idx=8,
        verdict="SHIFT_PLUS_1",
        confidence="HIGH",
        confirmation_required=False
    )
    tracker = SemanticIncidentTracker()
    tracker.register_or_merge([inc])

    # Mock: Guarded window returns SHIFT_PLUS_1, Core-only window returns ALIGNED
    async def mock_audit(source_items, target_items, **kwargs):
        if len(source_items) > 4:  # Guarded window (includes left/right guards)
            return {"alignment_verdict": "SHIFT_PLUS_1", "confidence": "HIGH"}
        else:  # Core-only window
            return {"alignment_verdict": "ALIGNED", "confidence": "HIGH"}

    with patch.object(pipeline.translator, "repair_alignment_region", return_value=[{"id": cid, "text": f"Repaired {cid}"} for cid in range(6, 10)]), \
         patch.object(pipeline.translator, "audit_cue_alignment_window", side_effect=mock_audit):

        success = await pipeline._repair_semantic_alignment_incidents(
            subs=source_subs,
            translated_subs=target_subs,
            incidents=[inc],
            target_language="Swedish",
            source_language="English",
            incident_tracker=tracker
        )

        # Because guard cues are NOT tracked as separate incidents, boundary shift must NOT be ignored
        assert not success
        assert inc.state == IncidentState.FAILED_REPAIR
        # Target subs must NOT have been mutated
        for j in range(5, 9):
            assert target_subs[j].content == f"Target {j + 1}"


@pytest.mark.asyncio
async def test_core_aligned_with_separately_tracked_neighboring_incident_is_accepted_and_separated():
    """Verify that core ALIGNED is accepted if boundary discrepancy belongs to a separate known incident."""
    pipeline = SubtitlePipeline()
    pipeline.translator = SubtitleTranslator()

    source_subs = _make_dummy_subs(20, prefix="Source")
    target_subs = _make_dummy_subs(20, prefix="Target")

    # Inc 1: cues 6-9 (indices 5-8)
    inc1 = AlignmentIncident(
        start_idx=5,
        end_idx=8,
        verdict="SHIFT_PLUS_1",
        confidence="HIGH",
        confirmation_required=False
    )
    # Inc 2: cues 10-13 (indices 9-12) - neighboring incident
    inc2 = AlignmentIncident(
        start_idx=9,
        end_idx=12,
        verdict="SHIFT_MINUS_1",
        confidence="HIGH",
        confirmation_required=False
    )
    tracker = SemanticIncidentTracker()
    tracker._incidents[inc1.incident_id] = inc1
    tracker._incidents[inc2.incident_id] = inc2

    # Mock: Cue 10 is part of inc2 (shifted), but cues 5-9 are clean/repaired
    async def mock_audit(source_items, target_items, **kwargs):
        if any(it.get("id") == 10 for it in source_items):
            return {"alignment_verdict": "SHIFT_PLUS_1", "confidence": "HIGH"}
        return {"alignment_verdict": "ALIGNED", "confidence": "HIGH"}

    with patch.object(pipeline.translator, "repair_alignment_region", return_value=[{"id": cid, "text": f"Repaired {cid}"} for cid in range(6, 10)]), \
         patch.object(pipeline.translator, "audit_cue_alignment_window", side_effect=mock_audit):

        success = await pipeline._repair_semantic_alignment_incidents(
            subs=source_subs,
            translated_subs=target_subs,
            incidents=[inc1],
            target_language="Swedish",
            source_language="English",
            incident_tracker=tracker
        )

        # Because right guard is covered by known incident inc2, inc1 core is safely committed
        assert success
        assert inc1.state == IncidentState.REPAIRED
        for j in range(5, 9):
            assert target_subs[j].content == f"Repaired {j + 1}"
        # inc2 remains separate and not dropped
        assert inc2.state == IncidentState.DISCOVERED


@pytest.mark.asyncio
async def test_primary_batch_clean_id_contract_with_identical_content_does_not_discard_or_split():
    """TEST 1: Verify that 100% ID coverage with identical/content-invalid items does NOT trigger discard/split."""
    translator = SubtitleTranslator()
    source_subs = _make_dummy_subs(6, prefix="Source")

    call_count = 0
    received_payloads = []

    async def mock_translate_batch(payload, **kwargs):
        nonlocal call_count, received_payloads
        call_count += 1
        received_payloads.append(list(payload))
        # Returns all 6 IDs exactly once (missing=0, unknown=0, dup=0, malformed=0),
        # but IDs 2 and 3 return identical English text ("Source 3", "Source 4")
        return [
            {"id": 0, "text": "Svenska 1"},
            {"id": 1, "text": "Svenska 2"},
            {"id": 2, "text": "Source 3"},   # Identical / content invalid
            {"id": 3, "text": "Source 4"},   # Identical / content invalid
            {"id": 4, "text": "Svenska 5"},
            {"id": 5, "text": "Svenska 6"},
        ]

    with patch.object(translator, "translate_batch", side_effect=mock_translate_batch), \
         patch.object(translator, "first_pass_micro_repair_batch", return_value=[]):

        translated = await translator.translate_srt_content(
            subs=source_subs,
            target_language="Swedish",
            source_language="English",
            batch_size=10
        )

        # 1. Verify NO atomic discard or recursive split occurred (exactly 1 AI batch call)
        assert call_count == 1

        # 2. Verify valid results were committed
        assert translated[0].content == "Svenska 1"
        assert translated[1].content == "Svenska 2"
        assert translated[4].content == "Svenska 5"
        assert translated[5].content == "Svenska 6"

        # 3. Content-invalid items remain unresolved (for downstream micro repair / QA recovery)
        assert translated[2].content == "Source 3"
        assert translated[3].content == "Source 4"


@pytest.mark.asyncio
async def test_primary_batch_true_structural_id_missing_triggers_atomic_discard_and_source_retry():
    """TEST 2: Verify that a true structural ID anomaly (missing expected ID) discards the candidate batch and retries."""
    translator = SubtitleTranslator()
    source_subs = _make_dummy_subs(6, prefix="Source")

    call_count = 0
    received_payloads = []

    async def mock_translate_batch(payload, **kwargs):
        nonlocal call_count, received_payloads
        call_count += 1
        received_payloads.append(list(payload))

        if call_count == 1:
            # Flawed output: drops cue ID 2 (index 2), shifts cue 3 into ID 2, etc. (missing ID 5)
            # IDs returned: 0, 1, 2, 3, 4 -> missing ID 5 (TRUE structural anomaly)
            return [
                {"id": 0, "text": "Svenska 1"},
                {"id": 1, "text": "Svenska 2"},
                {"id": 2, "text": "Svenska 4 (shifted)"},
                {"id": 3, "text": "Svenska 5 (shifted)"},
                {"id": 4, "text": "Svenska 6 (shifted)"},
            ]
        else:
            # Retry output: exact 1-to-1 clean translation
            return [
                {"id": p["id"], "text": f"Svenska {p['id'] + 1} (clean)"}
                for p in payload
            ]

    with patch.object(translator, "translate_batch", side_effect=mock_translate_batch), \
         patch.object(translator, "first_pass_micro_repair_batch", return_value=[]):

        translated = await translator.translate_srt_content(
            subs=source_subs,
            target_language="Swedish",
            source_language="English",
            batch_size=10
        )

        # 1. Verify flawed output was NOT committed (no shifted text in translated output)
        for sub in translated:
            assert "(shifted)" not in sub.content

        # 2. Verify retry used original source of truth
        assert call_count >= 2
        assert len(received_payloads[1]) == 6
        for idx, p in enumerate(received_payloads[1]):
            assert p["id"] == idx
            assert p["text"] == f"Source {idx + 1}"

        # 3. Verify complete retry was committed atomically with exact 1-to-1 matching
        assert len(translated) == 6
        for idx in range(6):
            assert translated[idx].content == f"Svenska {idx + 1} (clean)"
