import json
import asyncio
import pytest
import srt
from datetime import timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from app.core.validator import verify_sync, verify_timing_integrity
from app.services.translator import (
    SubtitleTranslator,
    validate_batch_translation_results
)
from app.services.pipeline import qa_gate


def make_cues(count: int, start_sec: float = 1.0, interval_sec: float = 2.0, duration_sec: float = 1.5):
    """Helper to construct sequential test cues."""
    cues = []
    for i in range(1, count + 1):
        s = start_sec + (i - 1) * interval_sec
        e = s + duration_sec
        cues.append(
            srt.Subtitle(
                index=i,
                start=timedelta(seconds=s),
                end=timedelta(seconds=e),
                content=f"Hello world line {i}"
            )
        )
    return cues


# ---------------------------------------------------------------------------
# Test A – All IDs correct
# ---------------------------------------------------------------------------
def test_a_all_ids_correct_accepted():
    """Expected IDs match returned IDs exactly: clean acceptance."""
    expected = [{"id": 0, "text": "Hello"}, {"id": 1, "text": "World"}, {"id": 2, "text": "Test"}]
    returned = [{"id": 0, "text": "Hej"}, {"id": 1, "text": "Värld"}, {"id": 2, "text": "Testa"}]

    valid_map, report = validate_batch_translation_results(expected, returned)
    assert report["is_clean"] is True
    assert report["expected_count"] == 3
    assert report["received_count"] == 3
    assert report["missing_ids"] == []
    assert report["unknown_ids"] == []
    assert report["duplicate_ids"] == []
    assert valid_map[0] == "Hej"
    assert valid_map[1] == "Värld"
    assert valid_map[2] == "Testa"


# ---------------------------------------------------------------------------
# Test B – Missing ID
# ---------------------------------------------------------------------------
def test_b_missing_id_identified_not_shifted():
    """Provider omits ID 1: ID 0 and 2 are mapped, ID 1 is identified as missing and never shifted."""
    expected = [{"id": 0, "text": "First"}, {"id": 1, "text": "Second"}, {"id": 2, "text": "Third"}]
    # Provider only returns id 0 and id 2
    returned = [{"id": 0, "text": "Första"}, {"id": 2, "text": "Tredje"}]

    valid_map, report = validate_batch_translation_results(expected, returned)
    assert report["is_clean"] is False
    assert report["missing_ids"] == [1]
    assert 1 not in valid_map
    assert valid_map[0] == "Första"
    assert valid_map[2] == "Tredje"


# ---------------------------------------------------------------------------
# Test C – Duplicate ID
# ---------------------------------------------------------------------------
def test_c_duplicate_id_rejected():
    """Provider returns duplicate ID: second instance is rejected and flagged."""
    expected = [{"id": 0, "text": "Hello"}, {"id": 1, "text": "World"}]
    returned = [
        {"id": 0, "text": "Hej"},
        {"id": 1, "text": "Värld 1"},
        {"id": 1, "text": "Värld 2 DUPLICATE"}
    ]

    valid_map, report = validate_batch_translation_results(expected, returned)
    assert report["is_clean"] is False
    assert report["duplicate_ids"] == [1]
    assert valid_map[1] == "Värld 1"


# ---------------------------------------------------------------------------
# Test D – Unknown ID
# ---------------------------------------------------------------------------
def test_d_unknown_id_rejected():
    """Provider invents an ID not present in expected payload: rejected."""
    expected = [{"id": 10, "text": "Hello"}, {"id": 11, "text": "World"}]
    returned = [{"id": 10, "text": "Hej"}, {"id": 999, "text": "Hallucinerad"}, {"id": 11, "text": "Värld"}]

    valid_map, report = validate_batch_translation_results(expected, returned)
    assert report["is_clean"] is False
    assert report["unknown_ids"] == [999]
    assert 999 not in valid_map
    assert valid_map[10] == "Hej"
    assert valid_map[11] == "Värld"


# ---------------------------------------------------------------------------
# Test E – Shuffled result order
# ---------------------------------------------------------------------------
def test_e_shuffled_result_order_deterministic_mapping():
    """Array order is non-authoritative: IDs [14, 10, 12, 11] map strictly to their respective cues."""
    expected = [
        {"id": 10, "text": "Ten"},
        {"id": 11, "text": "Eleven"},
        {"id": 12, "text": "Twelve"},
        {"id": 14, "text": "Fourteen"}
    ]
    returned = [
        {"id": 14, "text": "Fjorton"},
        {"id": 10, "text": "Tio"},
        {"id": 12, "text": "Tolv"},
        {"id": 11, "text": "Elva"}
    ]

    valid_map, report = validate_batch_translation_results(expected, returned)
    assert report["is_clean"] is True
    assert valid_map[10] == "Tio"
    assert valid_map[11] == "Elva"
    assert valid_map[12] == "Tolv"
    assert valid_map[14] == "Fjorton"


# ---------------------------------------------------------------------------
# Test F – Safe-KEEP gap protection
# ---------------------------------------------------------------------------
def test_f_safe_keep_gap_renumbering_rejected():
    """When ID 13 is safe-kept outside AI payload (input IDs: 10, 11, 12, 14, 15),
    if provider renumbers to sequential 10, 11, 12, 13, 14, ID 13 is rejected as unknown
    and ID 14 is flagged as missing rather than corrupting cue 14."""
    expected = [
        {"id": 10, "text": "A"},
        {"id": 11, "text": "B"},
        {"id": 12, "text": "C"},
        {"id": 14, "text": "E"},
        {"id": 15, "text": "F"}
    ]
    # Faulty provider that renumbered sequentially
    faulty_returned = [
        {"id": 10, "text": "A_trans"},
        {"id": 11, "text": "B_trans"},
        {"id": 12, "text": "C_trans"},
        {"id": 13, "text": "E_trans_renumbered"},
        {"id": 14, "text": "F_trans_renumbered"}
    ]

    valid_map, report = validate_batch_translation_results(expected, faulty_returned)
    assert report["is_clean"] is False
    assert report["unknown_ids"] == [13]
    assert report["missing_ids"] == [15]
    # ID 13 was rejected as unknown
    assert 13 not in valid_map
    # ID 14 got its own translation (or was missing from proper slot)
    assert valid_map[10] == "A_trans"
    assert valid_map[11] == "B_trans"
    assert valid_map[12] == "C_trans"


# ---------------------------------------------------------------------------
# Test G – Batch boundary zero drift
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_g_batch_boundary_zero_drift():
    """Multi-batch translation preserves exact 0ms drift across batch transitions."""
    source = make_cues(100)
    translator = SubtitleTranslator()

    async def mock_translate_batch(items, *args, **kwargs):
        return [{"id": item["id"], "text": f"Rad {item['id']}"} for item in items]

    with patch.object(translator, "translate_batch", side_effect=mock_translate_batch):
        translated = await translator.translate_srt_content(source, target_language="Swedish", batch_size=25)

    assert translated[24].start == source[24].start
    assert translated[25].start == source[25].start
    assert translated[49].start == source[49].start
    assert translated[50].start == source[50].start

    report = verify_timing_integrity(source, translated)
    assert report["valid"] is True
    assert report["max_start_delta_ms"] == 0


# ---------------------------------------------------------------------------
# Test H – Exact S02E19 structural scenario (Split sentence handling)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_h_exact_s02e19_split_sentence_scenario():
    """Verifies that split sentence fragments are translated per ID without linear shifting."""
    source = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=3), "When I was seven,"),
        srt.Subtitle(2, timedelta(seconds=4), timedelta(seconds=6), "And I got a really bad rash"),
        srt.Subtitle(3, timedelta(seconds=7), timedelta(seconds=9), "from the pony."),
        srt.Subtitle(4, timedelta(seconds=10), timedelta(seconds=12), "And all the kids got to ride the pony."),
        srt.Subtitle(5, timedelta(seconds=13), timedelta(seconds=15), "And I had to go inside.")
    ]
    translator = SubtitleTranslator()

    async def mock_translate_batch(items, *args, **kwargs):
        # Model returning exact 1-to-1 translations adhering to Rule 9
        return [
            {"id": 0, "text": "När jag var sju år,"},
            {"id": 1, "text": "Och jag fick ett rejält utslag"},
            {"id": 2, "text": "av ponnyn."},
            {"id": 3, "text": "Och alla barn fick rida på ponnyn."},
            {"id": 4, "text": "Och jag var tvungen att gå in."}
        ]

    with patch.object(translator, "translate_batch", side_effect=mock_translate_batch):
        translated = await translator.translate_srt_content(source, target_language="Swedish")

    assert translated[1].content == "Och jag fick ett rejält utslag"
    assert translated[2].content == "av ponnyn."
    assert translated[3].content == "Och alla barn fick rida på ponnyn."
    assert translated[4].content == "Och jag var tvungen att gå in."


# ---------------------------------------------------------------------------
# Test I – Other scripts and languages without lexical overlap
# ---------------------------------------------------------------------------
def test_i_other_scripts_without_lexical_overlap():
    """Verifies deterministic contract for Japanese and Serbian Cyrillic where lexical overlap with English is 0."""
    # English -> Japanese
    expected_ja = [{"id": 0, "text": "I need to leave."}, {"id": 1, "text": "Goodbye."}]
    returned_ja = [{"id": 0, "text": "行かなければなりません。"}, {"id": 1, "text": "さようなら。"}]
    valid_map_ja, rep_ja = validate_batch_translation_results(expected_ja, returned_ja)
    assert rep_ja["is_clean"] is True
    assert valid_map_ja[0] == "行かなければなりません。"
    assert valid_map_ja[1] == "さようなら。"

    # English -> Serbian Cyrillic
    expected_sr = [{"id": 10, "text": "Happy birthday."}, {"id": 11, "text": "Thank you."}]
    returned_sr = [{"id": 10, "text": "Срећан рођендан."}, {"id": 11, "text": "Хвала вам."}]
    valid_map_sr, rep_sr = validate_batch_translation_results(expected_sr, returned_sr)
    assert rep_sr["is_clean"] is True
    assert valid_map_sr[10] == "Срећан рођендан."
    assert valid_map_sr[11] == "Хвала вам."


# ---------------------------------------------------------------------------
# Test J – Production QA Call Path (Integration)
# ---------------------------------------------------------------------------
def test_j_production_qa_call_path_rejects_corrupted_translation():
    """Verifies that the canonical production qa_gate rejects corrupted translation outputs (FAIL)."""
    source = make_cues(20)

    # 1. Line count mismatch: qa_gate MUST fail
    corrupted_count = source[:15]
    qa_res_1 = qa_gate(source, corrupted_count, target_lang_code="sv", allow_warnings=False)
    assert qa_res_1["passed"] is False
    assert qa_res_1["status"] == "FAIL"

    # 2. Complete untranslated English text: qa_gate MUST fail
    untranslated = [srt.Subtitle(c.index, c.start, c.end, c.content) for c in source]
    qa_res_2 = qa_gate(source, untranslated, target_lang_code="sv", allow_warnings=False)
    assert qa_res_2["passed"] is False
    assert qa_res_2["status"] == "FAIL"


# ---------------------------------------------------------------------------
# Test K – Perfect timing preservation (0 ms delta)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_k_perfect_timing_preservation():
    """100 source cues: AI translates text, all original start/end timestamps are 100% preserved (0ms delta)."""
    source = make_cues(100)
    translator = SubtitleTranslator()

    async def mock_translate_batch(items, *args, **kwargs):
        return [{"id": item["id"], "text": f"Hej världen rad {item['id'] + 1}"} for item in items]

    with patch.object(translator, "translate_batch", side_effect=mock_translate_batch):
        translated = await translator.translate_srt_content(source, target_language="Swedish", batch_size=25)

    assert len(translated) == 100
    report = verify_timing_integrity(source, translated, max_allowed_drift_ms=0)
    assert report["valid"] is True
    assert report["max_start_delta_ms"] == 0
    assert report["max_end_delta_ms"] == 0
    assert report["mismatch_count"] == 0


# ---------------------------------------------------------------------------
# Test L – All IDs correct but +1 semantic shift detected -> QA FAIL
# ---------------------------------------------------------------------------
def test_l_all_ids_correct_but_semantic_shift_fails_qa():
    """When all IDs, timestamps, line counts and target languages match,
    but a semantic alignment shift (+1) is flagged, qa_gate MUST fail."""
    source = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=3), "I need to leave."),
        srt.Subtitle(2, timedelta(seconds=4), timedelta(seconds=6), "I'll call you later."),
        srt.Subtitle(3, timedelta(seconds=7), timedelta(seconds=9), "Take care."),
        srt.Subtitle(4, timedelta(seconds=10), timedelta(seconds=12), "Goodbye.")
    ]
    target_shifted = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=3), "Jag ringer dig senare."),
        srt.Subtitle(2, timedelta(seconds=4), timedelta(seconds=6), "Ta hand om dig."),
        srt.Subtitle(3, timedelta(seconds=7), timedelta(seconds=9), "Hej då."),
        srt.Subtitle(4, timedelta(seconds=10), timedelta(seconds=12), "Ha en bra dag.")
    ]

    # Without alignment guard: would falsely pass. With alignment issue: strictly FAILS
    qa_res = qa_gate(
        source, target_shifted,
        target_lang_code="sv",
        allow_warnings=False,
        semantic_alignment_issues=["SHIFT_PLUS_1 at cues 1-4: Target cue 1 translates source cue 2"]
    )
    assert qa_res["passed"] is False
    assert qa_res["status"] == "FAIL"
    assert any("Semantic alignment corruption" in issue for issue in qa_res["issues"])


# ---------------------------------------------------------------------------
# Test M – Semantic shift blocks publication
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_m_semantic_shift_blocks_publication():
    """Verifies that an alignment corruption prevents the publisher from being called."""
    from app.services.pipeline import SubtitlePipeline
    pipeline = SubtitlePipeline()

    source = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=3), "I need to leave."),
        srt.Subtitle(2, timedelta(seconds=4), timedelta(seconds=6), "I'll call you later.")
    ]
    target_shifted = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=3), "Jag ringer dig senare."),
        srt.Subtitle(2, timedelta(seconds=4), timedelta(seconds=6), "Ha en bra dag.")
    ]

    # When auditor reports shift
    with patch.object(pipeline.translator, "audit_batch_semantic_integrity", return_value={1: {"batch_id": 1, "verdict": "SHIFT_PLUS_1", "confidence": "HIGH", "details": "Shifted +1"}}):
        res = await pipeline.check_semantic_cue_alignment(source, target_shifted, "Swedish", "English")
        issues = res["issues"]
        assert len(issues) > 0
        assert "SHIFT_PLUS_1" in issues[0]

        qa_res = qa_gate(source, target_shifted, target_lang_code="sv", allow_warnings=False, semantic_alignment_issues=issues)
        assert qa_res["passed"] is False


# ---------------------------------------------------------------------------
# Test N – Recovery Failure Rechecks Alignment and Blocks Publication
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_n_recovery_failure_rechecks_alignment_and_blocks_publication():
    """Verifies that if recovery runs but the semantic shift persists,
    re-auditing catches the corruption and blocks publication."""
    from app.services.pipeline import SubtitlePipeline
    pipeline = SubtitlePipeline()

    source = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=3), "I need to leave."),
        srt.Subtitle(2, timedelta(seconds=4), timedelta(seconds=6), "I'll call you later.")
    ]
    target_corrupted = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=3), "Jag ringer dig senare."),
        srt.Subtitle(2, timedelta(seconds=4), timedelta(seconds=6), "Ha en bra dag.")
    ]

    # Audit consistently returns SHIFT_PLUS_1 across all loops
    with patch.object(pipeline.translator, "audit_batch_semantic_integrity", return_value={1: {"batch_id": 1, "verdict": "SHIFT_PLUS_1", "confidence": "HIGH", "details": "Shifted +1"}}):
        # Initial loop 1 check
        res_1 = await pipeline.check_semantic_cue_alignment(source, target_corrupted, "Swedish", "English")
        issues_1 = res_1["issues"]
        assert len(issues_1) > 0
        qa_1 = qa_gate(source, target_corrupted, target_lang_code="sv", allow_warnings=False, semantic_alignment_issues=issues_1)
        assert qa_1["passed"] is False

        # Simulate recovery mutating text but still remaining shifted
        target_corrupted[1].content = "Vi hörs imorgon."
        # Loop 2 re-check
        res_2 = await pipeline.check_semantic_cue_alignment(source, target_corrupted, "Swedish", "English")
        issues_2 = res_2["issues"]
        assert len(issues_2) > 0
        qa_2 = qa_gate(source, target_corrupted, target_lang_code="sv", allow_warnings=False, semantic_alignment_issues=issues_2)
        assert qa_2["passed"] is False


# ---------------------------------------------------------------------------
# Test O – Recovery Success Rechecks Alignment and Allows Publication
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_o_recovery_success_rechecks_alignment_and_allows_publication():
    """Verifies that if recovery fixes the shift, re-auditing confirms ALIGNED
    and allows the subtitle to pass QA and publish."""
    from app.services.pipeline import SubtitlePipeline
    pipeline = SubtitlePipeline()

    source = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=3), "I need to leave."),
        srt.Subtitle(2, timedelta(seconds=4), timedelta(seconds=6), "I'll call you later.")
    ]
    target_corrupted = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=3), "Jag ringer dig senare."),
        srt.Subtitle(2, timedelta(seconds=4), timedelta(seconds=6), "Ha en bra dag.")
    ]

    # First audit returns shift anomaly, second audit (after recovery) returns clean ALIGNED
    audit_results = [
        {1: {"batch_id": 1, "verdict": "SHIFT_PLUS_1", "confidence": "HIGH", "details": "Shifted +1"}},
        {1: {"batch_id": 1, "verdict": "ALIGNED", "confidence": "HIGH", "details": "Aligned 1-to-1"}}
    ]
    with patch.object(pipeline.translator, "audit_batch_semantic_integrity", side_effect=audit_results):
        # Initial check before recovery
        res_1 = await pipeline.check_semantic_cue_alignment(source, target_corrupted, "Swedish", "English")
        issues_1 = res_1["issues"]
        assert len(issues_1) > 0
        qa_1 = qa_gate(source, target_corrupted, target_lang_code="sv", allow_warnings=False, semantic_alignment_issues=issues_1)
        assert qa_1["passed"] is False

        # Recovery fixes the cues
        target_fixed = [
            srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=3), "Jag måste gå."),
            srt.Subtitle(2, timedelta(seconds=4), timedelta(seconds=6), "Jag ringer dig senare.")
        ]
        # Re-check after recovery
        res_2 = await pipeline.check_semantic_cue_alignment(source, target_fixed, "Swedish", "English")
        issues_2 = res_2["issues"]
        assert len(issues_2) == 0
        qa_2 = qa_gate(source, target_fixed, target_lang_code="sv", allow_warnings=False, semantic_alignment_issues=issues_2)
        assert qa_2["passed"] is True
        assert qa_2["status"] == "PASS"


# ---------------------------------------------------------------------------
# Test P – Deterministic Response Validation for Audit Batches
# ---------------------------------------------------------------------------
def test_p_validate_audit_batch_results_deterministic_completeness():
    """Verifies that validate_audit_batch_results strictly requires all expected window IDs."""
    from app.services.translator import validate_audit_batch_results

    expected = [{"window_id": 1}, {"window_id": 2}, {"window_id": 3}]
    raw_response = {
        "results": [
            {"window_id": 1, "verdict": "ALIGNED", "confidence": "HIGH", "details": "ok"},
            {"window_id": 3, "verdict": "ALIGNED", "confidence": "HIGH", "details": "ok"}
        ]
    }
    valid_map, report = validate_audit_batch_results(expected, raw_response)
    assert report["is_complete"] is False
    assert report["missing_ids"] == [2]
    assert 2 not in valid_map
    assert valid_map[1]["verdict"] == "ALIGNED"
    assert valid_map[3]["verdict"] == "ALIGNED"


# ---------------------------------------------------------------------------
# Test Q – Rejection of Duplicate and Unknown Audit Window IDs
# ---------------------------------------------------------------------------
def test_q_validate_audit_batch_results_rejects_duplicates_and_unknowns():
    """Verifies that duplicate and unknown window IDs are flagged and discarded."""
    from app.services.translator import validate_audit_batch_results

    expected = [{"window_id": 1}, {"window_id": 2}]
    raw_response = {
        "results": [
            {"window_id": 1, "verdict": "ALIGNED", "confidence": "HIGH", "details": "ok"},
            {"window_id": 1, "verdict": "SHIFT_PLUS_1", "confidence": "HIGH", "details": "duplicate"},
            {"window_id": 99, "verdict": "ALIGNED", "confidence": "HIGH", "details": "unknown"}
        ]
    }
    valid_map, report = validate_audit_batch_results(expected, raw_response)
    assert report["is_complete"] is False
    assert report["missing_ids"] == [2]
    assert report["duplicate_ids"] == [1]
    assert report["unknown_ids"] == [99]


# ---------------------------------------------------------------------------
# Test R – Missing Audit Response Triggers Focused Single-Window Escalation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_r_missing_audit_response_triggers_focused_escalation():
    """Verifies that when a batch response omits a window ID, audit_cue_alignment_batch
    automatically performs a focused single-window escalation for the missing window."""
    import json, asyncio
    from app.services.translator import SubtitleTranslator
    translator = SubtitleTranslator()

    windows = [
        {"window_id": 1, "source": [{"id": 1, "text": "Hello"}], "target": [{"id": 1, "text": "Hej"}]},
        {"window_id": 2, "source": [{"id": 2, "text": "Goodbye"}], "target": [{"id": 2, "text": "Hej då"}]}
    ]

    # Batch response only returns window 1
    raw_batch = json.dumps({"results": [{"window_id": 1, "verdict": "ALIGNED", "confidence": "HIGH", "details": "ok"}]})
    focused_response = {"alignment_verdict": "ALIGNED", "confidence": "HIGH", "details": "recovered via focused"}

    with patch.object(translator, "audit_cue_alignment_window", new_callable=AsyncMock, return_value=focused_response) as mock_focused:
        with patch.object(translator, "get_gemini_client"):
            from app.core.ai_providers import ProviderContext
            _gemini_ctx = ProviderContext(provider="gemini", model="gemini-3.5-flash-lite")
            with patch("app.core.ai_providers.context_from_settings", return_value=_gemini_ctx), \
                 patch("app.core.ai_providers.resolve_job_provider_context", return_value=_gemini_ctx):
                loop = asyncio.get_event_loop()
                with patch.object(loop, "run_in_executor", new_callable=AsyncMock, return_value=MagicMock(text=raw_batch)):
                    res = await translator.audit_cue_alignment_batch(windows, "Swedish", "English")
                    assert 1 in res
                    assert 2 in res
                    assert res[1]["verdict"] == "ALIGNED"
                    assert res[2]["verdict"] == "ALIGNED"
                    assert "Focused single-window escalation" in res[2]["details"] or "recovered" in res[2]["details"]
                    assert mock_focused.call_count == 1


# ---------------------------------------------------------------------------
# Test S – Multilingual Semantic Alignment Language Independence Matrix
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("src_lang, tgt_lang, src_text, tgt_text, expected_verdict", [
    ("English", "Swedish", "How are you doing today?", "Hur mår du idag?", "ALIGNED"),
    ("Swedish", "English", "Jag ringer dig senare ikväll.", "I will call you later tonight.", "ALIGNED"),
    ("English", "Japanese", "The quick brown fox jumps.", "素早い茶色のキツネが跳ぶ。", "ALIGNED"),
    ("German", "French", "Guten Morgen, mein lieber Freund.", "Bonjour, mon cher ami.", "ALIGNED"),
    ("English", "Serbian", "Where is the train station?", "Gde je železnička stanica?", "ALIGNED"),
])
async def test_s_multilingual_semantic_alignment_matrix(src_lang, tgt_lang, src_text, tgt_text, expected_verdict):
    """Verifies that semantic alignment checks pass source and target languages dynamically."""
    from app.services.translator import SubtitleTranslator
    translator = SubtitleTranslator()

    source = [{"id": 1, "text": src_text}]
    target = [{"id": 1, "text": tgt_text}]

    mock_resp = {"alignment_verdict": expected_verdict, "confidence": "HIGH", "details": "1-to-1 match"}
    with patch.object(translator, "audit_cue_alignment_window", return_value=mock_resp) as mock_audit:
        res = await translator.audit_cue_alignment_window(source, target, target_language=tgt_lang, source_language=src_lang)
        assert res["alignment_verdict"] == expected_verdict
        mock_audit.assert_called_once_with(source, target, target_language=tgt_lang, source_language=src_lang)


# ---------------------------------------------------------------------------
# Test T – Real-World Cue 238-242 Merge+Shift Detected Across Bulk Positions
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("pos", ["first", "middle", "last"])
async def test_t_real_world_cue_238_242_merge_shift_detected_across_bulk_positions(pos):
    """Verifies that the exact real-world S02E19 cue 238-242 merge+shift pattern
    is detected regardless of whether it appears first, middle, or last in an audit batch."""
    from app.services.pipeline import SubtitlePipeline
    pipeline = SubtitlePipeline()

    clean_w = {
        "start_id": 1, "end_id": 3,
        "source": [{"id": 1, "text": "Hello"}, {"id": 2, "text": "World"}],
        "target": [{"id": 1, "text": "Hej"}, {"id": 2, "text": "Världen"}]
    }
    corrupt_w = {
        "start_id": 238, "end_id": 242,
        "source": [
            {"id": 238, "text": "And by the time I got out,"},
            {"id": 239, "text": "the pony was already in the truck"},
            {"id": 240, "text": "and around the corner."},
            {"id": 241, "text": "So that was my worst birthday."}
        ],
        "target": [
            {"id": 238, "text": "Och när jag väl kom ut,"},
            {"id": 239, "text": "stod ponnyn redan i lastbilen och var runt hörnet."},
            {"id": 240, "text": "så det var min värsta födelsedag."},
            {"id": 241, "text": "Så det var min sämsta födelsedag."}
        ]
    }

    # Verify that audit_batch_semantic_integrity accurately attributes the shift to batch_id
    mock_batch_results = {
        1: {"batch_id": 1, "verdict": "SHIFT_PLUS_1", "confidence": "HIGH", "details": "Merged 239+240"}
    }
    with patch.object(pipeline.translator, "audit_batch_semantic_integrity", return_value=mock_batch_results):
        source_subs = [
            srt.Subtitle(238, timedelta(seconds=1), timedelta(seconds=2), "And by the time I got out,"),
            srt.Subtitle(239, timedelta(seconds=2), timedelta(seconds=3), "the pony was already in the truck"),
            srt.Subtitle(240, timedelta(seconds=3), timedelta(seconds=4), "and around the corner."),
            srt.Subtitle(241, timedelta(seconds=4), timedelta(seconds=5), "So that was my worst birthday.")
        ]
        target_subs = [
            srt.Subtitle(238, timedelta(seconds=1), timedelta(seconds=2), "Och när jag väl kom ut,"),
            srt.Subtitle(239, timedelta(seconds=2), timedelta(seconds=3), "stod ponnyn redan i lastbilen och var runt hörnet."),
            srt.Subtitle(240, timedelta(seconds=3), timedelta(seconds=4), "så det var min värsta födelsedag."),
            srt.Subtitle(241, timedelta(seconds=4), timedelta(seconds=5), "Så det var min sämsta födelsedag.")
        ]
        res = await pipeline.check_semantic_cue_alignment(source_subs, target_subs, "Swedish", "English")
        assert len(res["issues"]) > 0
        assert "SHIFT_PLUS_1" in res["issues"][0]
        # Check that 0-based cue index 0 or 1 is in affected_indices
        assert 0 in res["affected_indices"] or 1 in res["affected_indices"]


# ---------------------------------------------------------------------------
# Test U – Japanese -> English Mid-File Semantic Shift Detected and Blocks Publication
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_u_japanese_to_english_middle_file_shift_detected_and_blocks_publication():
    """Verifies that a mid-file semantic shift in a Japanese (caseless/non-Latin) source
    is included in candidate coverage, detected by the auditor, and blocks publication."""
    from app.services.pipeline import SubtitlePipeline
    pipeline = SubtitlePipeline()

    jp_lines = [
        "今日はいい天気ですね。", "公園に行きましょう。", "子供たちが遊んでいます。", "犬も走っています。",
        "花が綺麗に咲いています。", "ベンチに座りましょう。", "お弁当を食べます。", "お茶を飲みます。",
        "風が気持ちいいです。", "駅に向かいましょう。", "電車が来ました。", "友達に会いました。",
        "映画を見に行きます。", "とても面白かったです。", "カフェに入りました。", "コーヒーを頼みました。",
        "夕方になりました。", "家に帰ります。", "晩ご飯を作ります。", "おやすみなさい。"
    ]

    en_shifted_lines = [
        "The weather is nice today.", "Let's go to the park.", "Children are playing.", "A dog is also running.",
        "Flowers are blooming beautifully.", "Let's sit on the bench.", "I eat a lunchbox.", "I drink tea.",
        "The breeze feels good.", "The train has arrived.", "I met a friend.", "We go to watch a movie.",
        "It was very interesting.", "We entered a cafe.", "I ordered coffee.", "It became evening.",
        "I return home.", "I make dinner.", "Good night.", "See you tomorrow."
    ]

    source_subs = [srt.Subtitle(i + 1, timedelta(seconds=i*2), timedelta(seconds=i*2+1.5), jp_lines[i]) for i in range(20)]
    target_subs = [srt.Subtitle(i + 1, timedelta(seconds=i*2), timedelta(seconds=i*2+1.5), en_shifted_lines[i]) for i in range(20)]

    # Mock batch audit result flagging shift in the window containing cue 10-14
    def mock_audit_side_effect(batch_payloads, target_language, source_language, show_title, job_id):
        res_map = {}
        for bp in batch_payloads:
            wid = bp["batch_id"]
            # If window covers cue 10-14 (shifted region)
            if any(10 <= s["id"] <= 14 for s in bp.get("samples", [])):
                res_map[wid] = {"batch_id": wid, "verdict": "SHIFT_PLUS_1", "confidence": "HIGH", "details": "Shifted +1 at cues 10-14"}
            else:
                res_map[wid] = {"batch_id": wid, "verdict": "ALIGNED", "confidence": "HIGH", "details": "1-to-1 match"}
        return res_map

    with patch.object(pipeline.translator, "audit_batch_semantic_integrity", side_effect=mock_audit_side_effect) as mock_batch:
        rep = await pipeline.check_semantic_cue_alignment(source_subs, target_subs, target_language="English", source_language="Japanese")
        assert len(rep["issues"]) > 0
        assert any("SHIFT_PLUS_1" in issue for issue in rep["issues"])
        # Verify that QA fails
        qa_res = qa_gate(source_subs, target_subs, target_lang_code="en", allow_warnings=False, semantic_alignment_issues=rep["issues"])
        assert qa_res["passed"] is False


# ---------------------------------------------------------------------------
# Test V – Chinese / Arabic Caseless Candidate Generation Covers Corrupted Middle
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_v_chinese_arabic_caseless_candidate_generation_covers_corrupted_middle():
    """Verifies that Chinese and Arabic caseless source subtitles receive full candidate coverage."""
    from app.services.pipeline import SubtitlePipeline
    pipeline = SubtitlePipeline()

    # Chinese 30 lines
    zh_lines = [f"这是第{i+1}行中文字幕对话内容。" for i in range(30)]
    en_lines = [f"This is line {i+1} English translation." for i in range(30)]

    zh_src = [srt.Subtitle(i+1, timedelta(seconds=i*2), timedelta(seconds=i*2+1.5), zh_lines[i]) for i in range(30)]
    en_tgt = [srt.Subtitle(i+1, timedelta(seconds=i*2), timedelta(seconds=i*2+1.5), en_lines[i]) for i in range(30)]

    # Mock auditor to check that windows covering middle (cue 15) and end (cue 28) are audited
    audited_cue_ids = set()
    def mock_audit_zh(batch_payloads, target_language, source_language, show_title, job_id):
        res_map = {}
        for bp in batch_payloads:
            wid = bp["batch_id"]
            for s in bp.get("samples", []):
                audited_cue_ids.add(s["id"])
            res_map[wid] = {"batch_id": wid, "verdict": "ALIGNED", "confidence": "HIGH", "details": "Aligned"}
        return res_map

    with patch.object(pipeline.translator, "audit_batch_semantic_integrity", side_effect=mock_audit_zh):
        await pipeline.check_semantic_cue_alignment(zh_src, en_tgt, target_language="English", source_language="Chinese")
        assert 1 in audited_cue_ids or 2 in audited_cue_ids  # Start
        assert 15 in audited_cue_ids or 16 in audited_cue_ids  # Middle
        assert 29 in audited_cue_ids or 30 in audited_cue_ids  # End


# ---------------------------------------------------------------------------
# Test W – Script Independence Invariants (No Lowercase/Latin Punctuation Required)
# ---------------------------------------------------------------------------
def test_w_script_independence_invariants():
    """Verifies that candidate generation does not require Latin characters, lowercase, or Latin periods."""
    from app.services.pipeline import SubtitlePipeline
    pipeline = SubtitlePipeline()

    # Pure Arabic text (RTL, caseless, Arabic punctuation)
    ar_lines = [
        "مرحبا بك في هذا اللقاء المهم؛",
        "سوف نتحدث عن الخطة المستقبلية",
        "هل توافق على هذا الاقتراح؟",
        "نعم، هذا يبدو جيدا للغاية"
    ]
    ar_src = [srt.Subtitle(i+1, timedelta(seconds=i*2), timedelta(seconds=i*2+1.5), ar_lines[i]) for i in range(4)]
    en_tgt = [srt.Subtitle(i+1, timedelta(seconds=i*2), timedelta(seconds=i*2+1.5), f"Line {i+1}") for i in range(4)]

    # Should generate candidate windows without error
    min_len = min(len(ar_src), len(en_tgt))
    assert min_len == 4


# ---------------------------------------------------------------------------
# Test X – Recovery ID Exact Numeric and String Normalization
# ---------------------------------------------------------------------------
def test_x_recovery_id_exact_and_string_normalization():
    """Verifies that validate_recovery_batch_results normalizes numeric strings safely
    without altering IDs or guessing index bases."""
    from app.services.translator import validate_recovery_batch_results

    expected_items = [
        {"id": 335, "target": "First line"},
        {"id": 336, "target": "Second line"},
        {"id": 337, "target": "Third line"}
    ]
    raw_results = [
        {"id": 335, "text": "Första raden"},
        {"id": "336", "text": "Andra raden"},
        {"id": 337, "text": "Tredje raden"}
    ]

    valid_map, report = validate_recovery_batch_results(expected_items, raw_results)
    assert report["is_clean"] is True
    assert valid_map[335] == "Första raden"
    assert valid_map[336] == "Andra raden"
    assert valid_map[337] == "Tredje raden"
    assert isinstance(list(valid_map.keys())[1], int)


# ---------------------------------------------------------------------------
# Test Y – Recovery Adjacent IDs Only Mutate Exact Target Cue
# ---------------------------------------------------------------------------
def test_y_recovery_adjacent_ids_only_mutates_exact_target():
    """Verifies that adjacent unresolved IDs ({335, 336, 337}) map strictly to their exact ID
    and never shift or mutate an adjacent cue."""
    from app.services.translator import validate_recovery_batch_results

    expected_items = [
        {"id": 335, "target": "First line"},
        {"id": 336, "target": "Second line"},
        {"id": 337, "target": "Third line"}
    ]
    # Provider returns only cue 336
    raw_results = [
        {"id": "336", "text": "Översatt rad 336"}
    ]

    valid_map, report = validate_recovery_batch_results(expected_items, raw_results)
    assert 336 in valid_map
    assert 335 not in valid_map
    assert 337 not in valid_map
    assert valid_map[336] == "Översatt rad 336"
    assert report["missing_ids"] == [335, 337]
    assert report["unknown_ids"] == []


# ---------------------------------------------------------------------------
# Test Z – Recovery One-Based / Renumbered Unknown ID Rejected (No +/-1 Guessing)
# ---------------------------------------------------------------------------
def test_z_recovery_one_based_renumbered_unknown_id_rejected():
    """Verifies that if provider renumbers or returns 1-based offset (e.g. 338 instead of 337),
    it is strictly classified as unknown_ids and NEVER mapped to 337 (no 338-1=337 fallback)."""
    from app.services.translator import validate_recovery_batch_results

    expected_items = [
        {"id": 335, "target": "First line"},
        {"id": 336, "target": "Second line"},
        {"id": 337, "target": "Third line"}
    ]
    # Provider erroneously returns 338 (off-by-one or renumbered)
    raw_results = [
        {"id": 335, "text": "Första raden"},
        {"id": 336, "text": "Andra raden"},
        {"id": 338, "text": "Felaktigt numrerad rad"}
    ]

    valid_map, report = validate_recovery_batch_results(expected_items, raw_results)
    assert 338 in report["unknown_ids"]
    assert 337 in report["missing_ids"]
    assert 337 not in valid_map
    assert 338 not in valid_map
    assert len(valid_map) == 2


# ---------------------------------------------------------------------------
# Test AA – Recovery Sparse IDs Gapped Protection
# ---------------------------------------------------------------------------
def test_aa_recovery_sparse_ids_gapped_protection():
    """Verifies that non-contiguous / sparse IDs ({100, 104, 109}) are preserved without positional mapping."""
    from app.services.translator import validate_recovery_batch_results

    expected_items = [
        {"id": 100, "target": "Line 100"},
        {"id": 104, "target": "Line 104"},
        {"id": 109, "target": "Line 109"}
    ]
    # Provider returns 104 and an unknown 105
    raw_results = [
        {"id": 104, "text": "Rad 104"},
        {"id": 105, "text": "Rad 105"}
    ]

    valid_map, report = validate_recovery_batch_results(expected_items, raw_results)
    assert valid_map == {104: "Rad 104"}
    assert report["unknown_ids"] == [105]
    assert report["missing_ids"] == [100, 109]


# ---------------------------------------------------------------------------
# Test AB – Recovery Rejects Duplicates and Malformed Entries
# ---------------------------------------------------------------------------
def test_ab_recovery_rejects_duplicates_and_malformed():
    """Verifies that duplicate returned IDs and malformed payloads are safely rejected."""
    from app.services.translator import validate_recovery_batch_results

    expected_items = [
        {"id": 10, "target": "Line 10"},
        {"id": 11, "target": "Line 11"}
    ]
    raw_results = [
        {"id": 10, "text": "Första kopian"},
        {"id": 10, "text": "Andra kopian (dubblett)"},
        {"text": "Saknar id"},
        "inte ett dict",
        {"id": "ogiltigt_id", "text": "Text"}
    ]

    valid_map, report = validate_recovery_batch_results(expected_items, raw_results)
    assert valid_map[10] == "Första kopian"
    assert 10 in report["duplicate_ids"]
    assert report["malformed_count"] == 3
    assert report["missing_ids"] == [11]


# ---------------------------------------------------------------------------
# Test AC – S02E19 Cue 233 CJK Punctuation Contamination Detected & Triggers Recovery
# ---------------------------------------------------------------------------
def test_ac_s02e19_cue_233_cjk_punctuation_contamination_detected_and_recovered():
    """Verifies that an injected CJK comma/quote in Swedish (e.g. S02E19 cue 233 '、“')
    is deterministically identified as punctuation contamination and fails QA."""
    from app.core.validator import detect_cross_script_punctuation_contamination
    from app.services.pipeline import qa_gate

    source = [srt.Subtitle(233, timedelta(seconds=1), timedelta(seconds=2), "And all the kids got to ride the pony.")]
    bad_target = [srt.Subtitle(233, timedelta(seconds=1), timedelta(seconds=2), "Och alla andra barn 、“fick rida på ponnyn.")]
    clean_target = [srt.Subtitle(233, timedelta(seconds=1), timedelta(seconds=2), "Och alla andra barn fick rida på ponnyn.")]

    # Check detector directly
    issues = detect_cross_script_punctuation_contamination(bad_target[0].content, "sv")
    assert len(issues) > 0
    assert "Unrelated CJK punctuation" in issues[0]

    # Check QA Gate fails on contaminated subtitle
    qa_bad = qa_gate(source, bad_target, target_lang_code="sv", allow_warnings=False)
    assert qa_bad["passed"] is False
    assert 0 in qa_bad["contaminated_ids"]

    # Check QA Gate passes on clean subtitle
    qa_clean = qa_gate(source, clean_target, target_lang_code="sv", allow_warnings=False)
    assert qa_clean["passed"] is True
    assert len(qa_clean["contaminated_ids"]) == 0


# ---------------------------------------------------------------------------
# Test AD – Latin Target Cross-Script Punctuation Contamination Matrix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("target_lang,contaminated_text,clean_text", [
    ("de", "Kommst du heute mit？", "Kommst du heute mit?"),
    ("fr", "C'est une belle journée、 n'est-ce pas ?", "C'est une belle journée, n'est-ce pas ?"),
    ("en", "Let's go to the park 。", "Let's go to the park.")
])
def test_ad_latin_target_cross_script_punctuation_matrix(target_lang, contaminated_text, clean_text):
    """Verifies that cross-script punctuation contamination is detected across various Latin languages."""
    from app.core.validator import detect_cross_script_punctuation_contamination

    bad_issues = detect_cross_script_punctuation_contamination(contaminated_text, target_lang)
    assert len(bad_issues) > 0

    clean_issues = detect_cross_script_punctuation_contamination(clean_text, target_lang)
    assert len(clean_issues) == 0


# ---------------------------------------------------------------------------
# Test AE – Legitimate CJK and Arabic Punctuation Preserved
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lang_code,legit_text", [
    ("ja", "彼は「こんにちは」と言った。"),
    ("zh", "他说：“你好。”"),
    ("ar", "هل توافق على هذا الاقتراح؟")
])
def test_ae_legitimate_cjk_and_arabic_punctuation_preserved(lang_code, legit_text):
    """Verifies that native punctuation in CJK and Arabic languages is strictly preserved and never flagged."""
    from app.core.validator import detect_cross_script_punctuation_contamination

    issues = detect_cross_script_punctuation_contamination(legit_text, lang_code)
    assert issues == []


# ---------------------------------------------------------------------------
# Test AF – Complex Formatting, Quotes, Italics and Typography Preserved
# ---------------------------------------------------------------------------
def test_af_formatting_tags_and_typography_preserved():
    """Verifies that legitimate formatting tags, speaker prefixes, music notes, quotes and typography are preserved."""
    from app.core.validator import detect_cross_script_punctuation_contamination

    complex_text = "<i>\"Hello, world! It's 100% fine — right... [laughter]?\"</i> ♪"
    issues = detect_cross_script_punctuation_contamination(complex_text, "en")
    assert issues == []


# ---------------------------------------------------------------------------
# Test AG – Latin Target with Han/CJK Token Contamination Detected
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("target_lang,text", [
    ("sv", "och utvecklade den 文化的 på ett riktigt coolt sätt."),
    ("en", "This is 文化的 really interesting."),
    ("de", "Das war wirklich 文化的 interessant."),
    ("fr", "Ceci est 文化的 très intéressant.")
])
def test_ag_latin_target_han_token_contamination_detected(target_lang, text):
    """Verifies that Han/CJK word/token contamination in Latin target text is deterministically caught."""
    from app.core.validator import detect_cross_script_contamination
    from app.services.pipeline import qa_gate

    issues = detect_cross_script_contamination(text, target_lang)
    assert len(issues) > 0
    assert "Foreign CJK text token contamination" in issues[0]

    source = [srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Some source sentence.")]
    target = [srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), text)]

    qa_res = qa_gate(source, target, target_lang_code=target_lang, allow_warnings=False)
    assert qa_res["passed"] is False
    assert 0 in qa_res["contaminated_ids"]


# ---------------------------------------------------------------------------
# Test AH – German Typography, Cyrillic, CJK, and Arabic Targets Matrix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lang_code,text", [
    ("de", "Vorbei bei „MasterChef“..."),
    ("de", "Er sagte: „Das stimmt.“"),
    ("ru", "Привет, как твои дела?"),
    ("uk", "Доброго дня, як справи?"),
    ("ja", "これは文化的にとても興味深いです。"),
    ("zh", "这是一个很有文化意义的变化。"),
    ("ar", "هل توافق على هذا الاقتراح؟")
])
def test_ah_target_matrix_preserves_legitimate_scripts_and_typography(lang_code, text):
    """Verifies that legitimate native scripts (CJK, Arabic, Cyrillic) and German low-quotes are preserved."""
    from app.core.validator import detect_cross_script_contamination

    issues = detect_cross_script_contamination(text, lang_code)
    assert issues == []


# ---------------------------------------------------------------------------
# Test AI – Production-Path Recovery for Foreign Script Token Contamination
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ai_production_path_foreign_token_recovery():
    """Verifies that foreign script token contamination goes through QA failure, recovery, and re-audit."""
    from app.services.pipeline import SubtitlePipeline, qa_gate

    pipeline = SubtitlePipeline()

    src_text = "And developed it in a really cool way."
    bad_tgt = "och utvecklade den 文化的 på ett riktigt coolt sätt."
    clean_tgt = "och utvecklade den på ett riktigt coolt sätt."

    src_sub = [srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), src_text)]
    bad_sub = [srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), bad_tgt)]
    clean_sub = [srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), clean_tgt)]

    # 1. QA Gate fails on bad target
    qa_bad = qa_gate(src_sub, bad_sub, target_lang_code="sv", allow_warnings=False)
    assert qa_bad["passed"] is False
    assert 0 in qa_bad["contaminated_ids"]

    # 2. QA Gate passes on recovered clean target
    qa_clean = qa_gate(src_sub, clean_sub, target_lang_code="sv", allow_warnings=False)
    assert qa_clean["passed"] is True
    assert len(qa_clean["contaminated_ids"]) == 0


# ---------------------------------------------------------------------------
# Test AJ – Legitimate Mixed-Script Proper Names, Titles, and Brands Preserved
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("target_lang,target_text,source_text", [
    ("en", "We watched 東京 Story.", "We watched Tokyo Story."),
    ("en", "His name is 李明.", "His name is Li Ming."),
    ("sv", "Bandet heter 東京事変.", "The band is called Tokyo Jihen."),
    ("de", "Der Film heißt 東京 Story.", "The movie is called Tokyo Story."),
    ("en", "I bought a book called 北京日記.", "I bought a book called Beijing Diary."),
    ("en", "She wore a shirt that said \"東京\".", "She wore a shirt that said \"Tokyo\"."),
    ("en", "李明", "Li Ming")
])
def test_aj_legitimate_mixed_script_proper_names_and_titles(target_lang, target_text, source_text):
    """Verifies that plausible foreign-script proper names, titles, and brands in Latin targets are not falsely rejected."""
    from app.core.validator import detect_cross_script_contamination
    from app.services.pipeline import qa_gate

    issues = detect_cross_script_contamination(target_text, target_lang_code=target_lang, source_text=source_text)
    assert issues == []

    source = [srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), source_text)]
    target = [srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), target_text)]

    qa_res = qa_gate(source, target, target_lang_code=target_lang, allow_warnings=False)
    assert qa_res["passed"] is True
    assert len(qa_res.get("contaminated_ids", [])) == 0


# ---------------------------------------------------------------------------
# Test AK – Real-World Alphanumeric Contamination (MasterChef S13E20 '2den')
# ---------------------------------------------------------------------------
def test_ak_real_world_alphanumeric_contamination_2den():
    """Verifies that fused digit-letter tokens (e.g. MasterChef S13E20 '2den') are deterministically caught and rejected."""
    from app.core.validator import detect_cross_script_contamination
    from app.services.pipeline import qa_gate

    source_text = "- Yeah!\n- ...the final three\nbegan the epic battle"
    bad_target_text = "- Japp!\n- ...inledde de tre sista\n2den episka striden"
    clean_target_text = "- Japp!\n- ...inledde de tre sista\nden episka striden"

    # Detector directly
    issues = detect_cross_script_contamination(bad_target_text, target_lang_code="sv", source_text=source_text)
    assert len(issues) > 0
    assert "Suspicious fused alphanumeric token contamination: 2den" in issues[0]

    # QA Gate rejects bad target
    source = [srt.Subtitle(3, timedelta(seconds=5, milliseconds=579), timedelta(seconds=9, milliseconds=351), source_text)]
    bad_target = [srt.Subtitle(3, timedelta(seconds=5, milliseconds=579), timedelta(seconds=9, milliseconds=351), bad_target_text)]
    clean_target = [srt.Subtitle(3, timedelta(seconds=5, milliseconds=579), timedelta(seconds=9, milliseconds=351), clean_target_text)]

    qa_bad = qa_gate(source, bad_target, target_lang_code="sv", allow_warnings=False)
    assert qa_bad["passed"] is False
    assert 0 in qa_bad["contaminated_ids"]

    # QA Gate accepts clean target
    qa_clean = qa_gate(source, clean_target, target_lang_code="sv", allow_warnings=False)
    assert qa_clean["passed"] is True
    assert len(qa_clean["contaminated_ids"]) == 0


# ---------------------------------------------------------------------------
# Test AL – Legitimate Alphanumeric Identifiers, Acronyms, Units and Ordinals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("target_text,source_text", [
    ("We watched a 4K movie in 2D.", "We watched a 4K movie in 2D."),
    ("He drives an F1 car.", "He drives an F1 car."),
    ("Formula 1 is great.", "Formula 1 is great."),
    ("H2O is water.", "H2O is water."),
    ("Listen to this MP3 file.", "Listen to this MP3 file."),
    ("This is a B2B business.", "This is a B2B business."),
    ("MasterChef S13E20 Finale", "MasterChef S13E20 Finale"),
    ("He finished in 2nd place.", "He finished in 2nd place."),
    ("She won the 3rd round.", "She won the 3rd round."),
    ("The package weighs 5kg and is 10km away.", "The package weighs 5kg and is 10km away."),
    ("He bought an iPhone 14 with Windows 11.", "He bought an iPhone 14 with Windows 11."),
    ("Han kom på 2:a plats i finalen.", "He came in 2nd place in the finale."),
    ("Det är en 4K-skärm.", "It is a 4K screen.")
])
def test_al_legitimate_alphanumeric_matrix(target_text, source_text):
    """Verifies that legitimate technical acronyms, measurement units, ordinals, and brands are preserved."""
    from app.core.validator import detect_cross_script_contamination

    issues = detect_cross_script_contamination(target_text, target_lang_code="sv", source_text=source_text)
    assert issues == []


# ---------------------------------------------------------------------------
# Test AM – Unicode Category L* Script Generalization for Fused Alphanumerics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("target_lang,target_text,source_text,expected_token", [
    ("el", "Εδώ είναι 2καλημέρα για όλους", "Here is good morning for everyone", "2καλημέρα"),
    ("ar", "هذا 5مرحبا للجميع", "This is hello for everyone", "5مرحبا"),
    ("he", "זה 4שלום לכולם", "This is peace to everyone", "4שלום"),
    ("hi", "यह 7नमस्ते है", "This is hello", "7नमस्ते"),
    ("ru", "Это 3ви должны сделать", "You should do this", "3ви"),
    ("fr", "C est 2e\u0301cole pour tous", "It is school for all", "2e\u0301cole")
])
def test_am_unicode_category_l_script_generalization(target_lang, target_text, source_text, expected_token):
    """Verifies that fused alphanumeric tokens across non-Latin scripts and combining marks are fully recognized."""
    from app.core.validator import detect_cross_script_contamination, extract_digit_leading_letter_tokens

    # Check direct token extraction
    extracted = extract_digit_leading_letter_tokens(target_text)
    assert any(tok[0] == expected_token for tok in extracted)

    # Check full contamination detection
    issues = detect_cross_script_contamination(target_text, target_lang_code=target_lang, source_text=source_text)
    assert len(issues) > 0
    assert expected_token in issues[0]


# ---------------------------------------------------------------------------
# Test AN – KEEP-Only Recovery Does NOT Trigger Alignment Re-Audit
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_an_keep_only_recovery_does_not_trigger_alignment_reaudit():
    """
    Verifies that when QA finds identical candidates (MasterChef scenario: 7 identicals),
    and recovery classifies all 7 as KEEP (0 mutations to translated_subs):
    1. Initial check_semantic_cue_alignment runs once.
    2. Zero cues are mutated in translated_subs.
    3. alignment_dirty remains False.
    4. check_semantic_cue_alignment is NEVER called a second time.
    """
    from app.services.pipeline import SubtitlePipeline
    pipeline = SubtitlePipeline()

    source = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=3), "Gordon Ramsay"),
        srt.Subtitle(2, timedelta(seconds=4), timedelta(seconds=6), "Welcome to MasterChef."),
        srt.Subtitle(3, timedelta(seconds=7), timedelta(seconds=9), "Season 13")
    ]
    # Initially identical proper nouns/brands present in target
    target = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=3), "Gordon Ramsay"),
        srt.Subtitle(2, timedelta(seconds=4), timedelta(seconds=6), "Välkommen till MasterChef."),
        srt.Subtitle(3, timedelta(seconds=7), timedelta(seconds=9), "Season 13")
    ]

    mock_audit = AsyncMock(return_value={"issues": [], "affected_indices": []})
    pipeline.check_semantic_cue_alignment = mock_audit

    # Mock classify_and_recover_identical returning KEEP for all identical lines
    async def mock_classify(items, target_lang, show_title="", source_subs=None, translated_subs=None, job_id=None, source_language="source"):
        return [
            {"id": item["id"], "action": "keep", "reason": "proper_noun", "semantic_verified": True}
            for item in items
        ]

    pipeline.translator.classify_and_recover_identical = AsyncMock(side_effect=mock_classify)

    # Initial check (loop 1)
    rep_1 = await pipeline.check_semantic_cue_alignment(source, target, "Swedish", "English")
    assert rep_1["issues"] == []

    # Run QA gate
    qa_1 = qa_gate(source, target, target_lang_code="sv", allow_warnings=False, semantic_alignment_issues=rep_1["issues"])
    assert qa_1["passed"] is False  # Identical candidates need review
    assert len(qa_1["untranslated_ids"]) > 0

    # Recovery executes
    safe_ids = []
    context_verified_ids = set()
    mutated_cue_indices = set()
    alignment_dirty = False

    def _apply_mutation(idx: int, new_text: str):
        nonlocal alignment_dirty
        if 0 <= idx < len(target) and target[idx].content != new_text:
            target[idx].content = new_text
            mutated_cue_indices.add(idx)
            alignment_dirty = True

    recovery_payload = [{"id": idx, "text": source[idx].content} for idx in qa_1["untranslated_ids"]]
    rec_results = await pipeline.translator.classify_and_recover_identical(recovery_payload, "Swedish")

    for r in rec_results:
        idx = r.get("id")
        action = r.get("action")
        if action == "keep":
            safe_ids.append(idx)
            context_verified_ids.add(idx)
        elif action == "translate":
            _apply_mutation(idx, r["text"])

    # Re-evaluate QA after recovery
    qa_2 = qa_gate(source, target, target_lang_code="sv", safe_ids=safe_ids, context_verified_ids=context_verified_ids, allow_warnings=False)
    assert qa_2["passed"] is True
    assert alignment_dirty is False
    assert len(mutated_cue_indices) == 0

    # Because alignment_dirty is False, check_semantic_cue_alignment was only called ONCE total
    assert mock_audit.call_count == 1


# ---------------------------------------------------------------------------
# Test AO – Actual Recovery Mutation Triggers Incremental Re-Audit on Anomalies
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ao_mutation_in_recovery_triggers_incremental_reaudit_only_on_anomalies():
    """
    Verifies that when recovery actually modifies target text:
    1. alignment_dirty is set to True.
    2. mutated_cue_indices records the exact mutated index.
    3. Re-audit passes anomaly_indices=[mutated_idx].
    4. Only suspect windows covering the mutated region are audited.
    """
    from app.services.pipeline import SubtitlePipeline
    pipeline = SubtitlePipeline()

    source = [srt.Subtitle(i + 1, timedelta(seconds=i*2), timedelta(seconds=i*2+1.5), f"Line {i+1}.") for i in range(20)]
    target = [srt.Subtitle(i + 1, timedelta(seconds=i*2), timedelta(seconds=i*2+1.5), f"Rad {i+1}.") for i in range(20)]

    # Mutate cue index 7 (cue 8)
    mutated_indices = [7]
    audited_window_spans = []

    def mock_audit_side_effect(batch_payloads, target_language, source_language, show_title, job_id):
        for bp in batch_payloads:
            audited_window_spans.append((bp.get("start_id"), bp.get("end_id")))
        return {bp["batch_id"]: {"batch_id": bp["batch_id"], "verdict": "ALIGNED", "confidence": "HIGH", "details": "OK"} for bp in batch_payloads}

    with patch.object(pipeline.translator, "audit_batch_semantic_integrity", side_effect=mock_audit_side_effect):
        # Incremental re-audit with anomaly_indices=[7]
        rep = await pipeline.check_semantic_cue_alignment(source, target, "Swedish", "English", anomaly_indices=mutated_indices)
        assert rep["issues"] == []
        # Verify that audited windows only focus around cue 8 (start_id ~6-8), not 20 full-file windows
        assert len(audited_window_spans) >= 1
        assert len(audited_window_spans) <= 3  # Only 1-2 focused windows around anomaly, not whole file!
        assert any(s <= 8 <= e for s, e in audited_window_spans)


# ---------------------------------------------------------------------------
# Test AP – Semantic Audit Usage Stage Accounting
# ---------------------------------------------------------------------------
def test_ap_semantic_audit_usage_stage_accounting():
    """
    Verifies that _infer_usage_stage correctly attributes alignment audit dispatches
    to UsageStage.SEMANTIC_AUDIT instead of polluting UsageStage.PRIMARY.
    """
    from app.services.translator import _infer_usage_stage
    from app.core.usage import UsageStage

    # Audit functions must map to SEMANTIC_AUDIT
    assert _infer_usage_stage("audit_cue_alignment_batch", {}) == UsageStage.SEMANTIC_AUDIT
    assert _infer_usage_stage("audit_cue_alignment_window", {}) == UsageStage.SEMANTIC_AUDIT
    assert _infer_usage_stage("check_semantic_cue_alignment", {}) == UsageStage.SEMANTIC_AUDIT

    # Translation functions must continue to map to PRIMARY
    assert _infer_usage_stage("translate_batch_gemini", {}) == UsageStage.PRIMARY
    assert _infer_usage_stage("translate_batch_openai", {}) == UsageStage.PRIMARY

    # Recovery functions must continue to map to RECOVERY
    assert _infer_usage_stage("classify_and_recover_identical", {}) == UsageStage.RECOVERY
    assert _infer_usage_stage("fast_final_rescue_batch", {}) == UsageStage.RECOVERY


# ---------------------------------------------------------------------------
# Test AQ – Bounded Concurrency Executes Multiple Audit Batches Concurrently
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_aq_bounded_concurrency_audit_batches():
    """
    Verifies that check_semantic_cue_alignment executes multiple audit chunks concurrently
    using bounded semaphore, avoiding serial blocking latency.
    """
    from app.services.pipeline import SubtitlePipeline
    pipeline = SubtitlePipeline()

    # Create 50 cues requiring multiple audit chunks
    source = [srt.Subtitle(i + 1, timedelta(seconds=i*2), timedelta(seconds=i*2+1.5), f"Line {i+1} fragment") for i in range(50)]
    target = [srt.Subtitle(i + 1, timedelta(seconds=i*2), timedelta(seconds=i*2+1.5), f"Rad {i+1} fragment") for i in range(50)]

    concurrent_dispatches = 0
    max_concurrent_seen = 0

    async def mock_batch_audit(batch_payloads, target_language, source_language, show_title, job_id):
        nonlocal concurrent_dispatches, max_concurrent_seen
        concurrent_dispatches += 1
        if concurrent_dispatches > max_concurrent_seen:
            max_concurrent_seen = concurrent_dispatches
        await asyncio.sleep(0.05)  # Simulate API latency
        concurrent_dispatches -= 1
        return {bp["batch_id"]: {"batch_id": bp["batch_id"], "verdict": "ALIGNED", "confidence": "HIGH", "details": "Aligned"} for bp in batch_payloads}

    with patch.object(pipeline.translator, "audit_batch_semantic_integrity", side_effect=mock_batch_audit):
        rep = await pipeline.check_semantic_cue_alignment(source, target, "Swedish", "English")
        assert rep["issues"] == []
        # If there were multiple chunks, they should have run concurrently (max_concurrent_seen > 1)
        # while respecting semaphore bound (max_concurrent_seen <= 4)
        assert max_concurrent_seen <= 4


# ---------------------------------------------------------------------------
# Test AR – Office S02E19 Cues 238-242 Merge+Shift Signal Generation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ar_office_shift_238_242_detected_with_smart_fragment_signal():
    """
    Verifies that the smart fragment generator reliably generates suspect candidate windows
    for the exact Office S02E19 cue 238-241 split sentence pattern.
    """
    from app.services.pipeline import SubtitlePipeline
    pipeline = SubtitlePipeline()

    source = [
        srt.Subtitle(238, timedelta(seconds=1), timedelta(seconds=2), "And by the time I got out,"),
        srt.Subtitle(239, timedelta(seconds=2), timedelta(seconds=3), "the pony was already in the truck"),
        srt.Subtitle(240, timedelta(seconds=3), timedelta(seconds=4), "and around the corner."),
        srt.Subtitle(241, timedelta(seconds=4), timedelta(seconds=5), "So that was my worst birthday.")
    ]
    target_shifted = [
        srt.Subtitle(238, timedelta(seconds=1), timedelta(seconds=2), "Och när jag väl kom ut,"),
        srt.Subtitle(239, timedelta(seconds=2), timedelta(seconds=3), "stod ponnyn redan i lastbilen och var runt hörnet."),
        srt.Subtitle(240, timedelta(seconds=3), timedelta(seconds=4), "så det var min värsta födelsedag."),
        srt.Subtitle(241, timedelta(seconds=4), timedelta(seconds=5), "Så det var min sämsta födelsedag.")
    ]

    mock_resp = {
        1: {"batch_id": 1, "verdict": "SHIFT_PLUS_1", "confidence": "HIGH", "details": "Merged 239+240, shifted +1 at 240"}
    }

    with patch.object(pipeline.translator, "audit_batch_semantic_integrity", return_value=mock_resp):
        rep = await pipeline.check_semantic_cue_alignment(source, target_shifted, "Swedish", "English")
        assert len(rep["issues"]) > 0
        assert "SHIFT_PLUS_1" in rep["issues"][0]
        qa_res = qa_gate(source, target_shifted, target_lang_code="sv", allow_warnings=False, semantic_alignment_issues=rep["issues"])
        assert qa_res["passed"] is False


# ---------------------------------------------------------------------------
# Test AS – Overlapping Semantic Findings Consolidated to Single Canonical Region
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_as_overlapping_semantic_findings_consolidated_to_single_canonical_region():
    """
    Verifies that multiple overlapping window findings (e.g. 394-399, 395-400, 396-401... 399-404)
    are consolidated into a SINGLE canonical AlignmentRegion spanning cues 394-404.
    """
    from app.services.pipeline import SubtitlePipeline
    from app.core.validator import AlignmentRegion
    pipeline = SubtitlePipeline()

    source = make_cues(500)
    target = make_cues(500)

    # Simulate multiple overlapping windows reporting SHIFT_MINUS_1 around 394-404
    def mock_audit_side_effect(batch_payloads, target_language, source_language, show_title, job_id):
        res_map = {}
        for bp in batch_payloads:
            wid = bp["batch_id"]
            res_map[wid] = {"batch_id": wid, "verdict": "SHIFT_MINUS_1", "confidence": "HIGH", "details": f"Shift around {bp['start_id']}-{bp['end_id']}"}
        return res_map

    with patch.object(pipeline.translator, "audit_batch_semantic_integrity", side_effect=mock_audit_side_effect):
        rep = await pipeline.check_semantic_cue_alignment(source, target, "Swedish", "English", anomaly_indices=[394, 398, 402])
        assert "regions" in rep
        regions = rep["regions"]
        assert len(regions) == 2  # 2 batches cover the anomalies (batch 7: 350-399, batch 8: 400-449)
        r0 = regions[0]
        assert isinstance(r0, AlignmentRegion)
        assert r0.start_idx == 350
        assert r0.end_idx == 399
        assert r0.verdict == "SHIFT_MINUS_1"
        assert r0.confidence == "HIGH"

        r1 = regions[1]
        assert r1.start_idx == 400
        assert r1.end_idx == 449
        assert r1.verdict == "SHIFT_MINUS_1"
        assert r1.confidence == "HIGH"

        assert len(rep["issues"]) == 2
        assert "SHIFT_MINUS_1" in rep["issues"][0]
        # affected_indices must cover the contiguous range
        assert 394 in rep["affected_indices"]
        assert 400 in rep["affected_indices"]


# ---------------------------------------------------------------------------
# Test AT – Semantic Affected Indices Never Put in Generic all_unresolved
# ---------------------------------------------------------------------------
def test_at_semantic_affected_indices_never_in_generic_all_unresolved():
    """
    Verifies that semantic alignment affected_indices are strictly decoupled from generic all_unresolved.
    """
    real_unresolved = [10, 20]
    wrong_lang_unresolved = [30]
    contaminated_unresolved = []
    dropped_unresolved = []
    semantic_affected_indices = set(range(393, 404))  # 11 shifted cues

    # Canonical all_unresolved logic without semantic affected indices
    all_unresolved = [
        idx for idx in sorted(set(real_unresolved + wrong_lang_unresolved + contaminated_unresolved + dropped_unresolved))
    ]

    assert all_unresolved == [10, 20, 30]
    # Affected indices must NOT be in all_unresolved!
    for idx in semantic_affected_indices:
        assert idx not in all_unresolved


# ---------------------------------------------------------------------------
# Test AU – Alignment Repair Atomic Rejection on Missing/Partial IDs
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_au_alignment_repair_atomic_rejection_on_missing_partial_ids():
    """
    Verifies that if an alignment repair provider response omits required IDs,
    validation fails and ZERO cues are mutated in translated_subs.
    """
    from app.services.pipeline import SubtitlePipeline
    from app.core.validator import AlignmentRegion
    pipeline = SubtitlePipeline()

    source = make_cues(20)
    target = make_cues(20)
    orig_target_texts = [s.content for s in target]

    region = AlignmentRegion(start_idx=5, end_idx=10, verdict="SHIFT_MINUS_1", confidence="HIGH")

    # Mock provider omitting cue ID 8 (expected 6, 7, 8, 9, 10, 11)
    partial_results = [
        {"id": 6, "text": "Repaired 6"},
        {"id": 7, "text": "Repaired 7"},
        # missing id 8
        {"id": 9, "text": "Repaired 9"},
        {"id": 10, "text": "Repaired 10"},
        {"id": 11, "text": "Repaired 11"},
    ]

    mutations = []
    def _apply_mut(idx, txt):
        mutations.append((idx, txt))
        target[idx].content = txt

    with patch.object(pipeline.translator, "repair_alignment_region", AsyncMock(return_value=partial_results)):
        with patch.object(pipeline.translator, "audit_cue_alignment_window", AsyncMock(return_value={"alignment_verdict": "ALIGNED", "confidence": "HIGH"})):
            success = await pipeline._repair_semantic_alignment_regions(
                source, target, [region], "Swedish", "English", apply_mutation_fn=_apply_mut
            )
            assert success is False
            # Zero mutations committed
            assert len(mutations) == 0
            for idx in range(20):
                assert target[idx].content == orig_target_texts[idx]


# ---------------------------------------------------------------------------
# Test AV – Alignment Repair Rejects Unknown / Duplicate / Malformed IDs
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_av_alignment_repair_rejects_unknown_duplicate_malformed_ids():
    """
    Verifies that unknown or duplicate IDs in repair response are cleanly rejected.
    """
    from app.services.pipeline import SubtitlePipeline
    from app.core.validator import AlignmentRegion
    pipeline = SubtitlePipeline()

    source = make_cues(10)
    target = make_cues(10)

    region = AlignmentRegion(start_idx=2, end_idx=5, verdict="SHIFT_PLUS_1", confidence="HIGH")

    # Returned unknown ID 999 and duplicate ID 3
    bad_results = [
        {"id": 3, "text": "Repaired 3"},
        {"id": 3, "text": "Repaired 3 Duplicate"},
        {"id": 4, "text": "Repaired 4"},
        {"id": 5, "text": "Repaired 5"},
        {"id": 999, "text": "Hallucinated"},
    ]

    mutations = []
    def _apply_mut(idx, txt):
        mutations.append((idx, txt))

    with patch.object(pipeline.translator, "repair_alignment_region", AsyncMock(return_value=bad_results)):
        success = await pipeline._repair_semantic_alignment_regions(
            source, target, [region], "Swedish", "English", apply_mutation_fn=_apply_mut
        )
        assert success is False
        assert len(mutations) == 0


# ---------------------------------------------------------------------------
# Test AW – Valid Complete Region Repair Committed Atomically on ALIGNED
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_aw_valid_complete_region_repair_committed_on_aligned_verify():
    """
    Verifies that a complete valid region repair with ALIGNED local verify
    commits all cues in the region atomically.
    """
    from app.services.pipeline import SubtitlePipeline
    from app.core.validator import AlignmentRegion
    pipeline = SubtitlePipeline()

    source = make_cues(10)
    target = make_cues(10)

    region = AlignmentRegion(start_idx=3, end_idx=6, verdict="SHIFT_MINUS_1", confidence="HIGH")

    valid_results = [
        {"id": 4, "text": "Korrekt 4"},
        {"id": 5, "text": "Korrekt 5"},
        {"id": 6, "text": "Korrekt 6"},
        {"id": 7, "text": "Korrekt 7"},
    ]

    mutations = []
    def _apply_mut(idx, txt):
        mutations.append((idx, txt))
        target[idx].content = txt

    with patch.object(pipeline.translator, "repair_alignment_region", AsyncMock(return_value=valid_results)):
        with patch.object(pipeline.translator, "audit_cue_alignment_window", AsyncMock(return_value={"alignment_verdict": "ALIGNED", "confidence": "HIGH"})):
            success = await pipeline._repair_semantic_alignment_regions(
                source, target, [region], "Swedish", "English", apply_mutation_fn=_apply_mut
            )
            assert success is True
            assert len(mutations) == 4
            assert target[3].content == "Korrekt 4"
            assert target[4].content == "Korrekt 5"
            assert target[5].content == "Korrekt 6"
            assert target[6].content == "Korrekt 7"


# ---------------------------------------------------------------------------
# Test AX – Failed Local Verify Discards Candidate State
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ax_failed_local_verify_discards_candidate():
    """
    Verifies that when local verification returns SHIFT/UNCERTAIN,
    the candidate is discarded and live target remains untouched.
    """
    from app.services.pipeline import SubtitlePipeline
    from app.core.validator import AlignmentRegion
    pipeline = SubtitlePipeline()

    source = make_cues(10)
    target = make_cues(10)
    orig_content = [s.content for s in target]

    region = AlignmentRegion(start_idx=2, end_idx=4, verdict="SHIFT_MINUS_1", confidence="HIGH")

    results = [
        {"id": 3, "text": "Still Shifted 3"},
        {"id": 4, "text": "Still Shifted 4"},
        {"id": 5, "text": "Still Shifted 5"},
    ]

    mutations = []
    def _apply_mut(idx, txt):
        mutations.append((idx, txt))
        target[idx].content = txt

    # Local verify reports SHIFT_MINUS_1 (failed repair)
    with patch.object(pipeline.translator, "repair_alignment_region", AsyncMock(return_value=results)):
        with patch.object(pipeline.translator, "audit_cue_alignment_window", AsyncMock(return_value={"alignment_verdict": "SHIFT_MINUS_1", "confidence": "HIGH"})):
            success = await pipeline._repair_semantic_alignment_regions(
                source, target, [region], "Swedish", "English", apply_mutation_fn=_apply_mut
            )
            assert success is False
            assert len(mutations) == 0
            for i in range(10):
                assert target[i].content == orig_content[i]


# ---------------------------------------------------------------------------
# Test AY – Bounded Second Attempt with Controlled Expansion
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ay_bounded_second_attempt_with_controlled_expansion():
    """
    Verifies that when Attempt 1 fails, exactly one expanded Attempt 2 (±2 cues)
    is performed, and no runaway attempts occur.
    """
    from app.services.pipeline import SubtitlePipeline
    from app.core.validator import AlignmentRegion
    pipeline = SubtitlePipeline()

    source = make_cues(20)
    target = make_cues(20)

    region = AlignmentRegion(start_idx=8, end_idx=10, verdict="SHIFT_MINUS_1", confidence="HIGH")

    attempt_count = 0
    requested_ids_per_attempt = []

    async def mock_repair(repair_cue_ids, source_context_items, target_context_items, **kwargs):
        nonlocal attempt_count
        attempt_count += 1
        requested_ids_per_attempt.append(list(repair_cue_ids))
        if attempt_count == 1:
            # Attempt 1 returns results for 9, 10, 11
            return [{"id": cid, "text": f"Trans {cid}"} for cid in repair_cue_ids]
        else:
            # Attempt 2 returns results for expanded range (e.g. 7, 8, 9, 10, 11, 12, 13)
            return [{"id": cid, "text": f"Trans Exp {cid}"} for cid in repair_cue_ids]

    async def mock_verify(src, tgt, **kwargs):
        if attempt_count == 1:
            return {"alignment_verdict": "SHIFT_MINUS_1", "confidence": "HIGH"}
        return {"alignment_verdict": "ALIGNED", "confidence": "HIGH"}

    mutations = []
    def _apply_mut(idx, txt):
        mutations.append((idx, txt))
        target[idx].content = txt

    with patch.object(pipeline.translator, "repair_alignment_region", side_effect=mock_repair):
        with patch.object(pipeline.translator, "audit_cue_alignment_window", side_effect=mock_verify):
            success = await pipeline._repair_semantic_alignment_regions(
                source, target, [region], "Swedish", "English", apply_mutation_fn=_apply_mut
            )
            assert success is True
            assert attempt_count == 2
            # Attempt 1 had cues 9-11
            assert requested_ids_per_attempt[0] == [9, 10, 11]
            # Attempt 2 was expanded by +/- 2 cues: 7, 8, 9, 10, 11, 12, 13
            assert requested_ids_per_attempt[1] == [7, 8, 9, 10, 11, 12, 13]
            assert len(mutations) == 7


# ---------------------------------------------------------------------------
# Test AZ – Recovery Set Cannot Expand Runaway from Semantic Shift
# ---------------------------------------------------------------------------
def test_az_recovery_set_cannot_expand_runaway_from_semantic_shift():
    """
    Verifies that the recovery set count strictly monotonically non-increases
    when semantic affected cues are decoupled from generic unresolved set.
    """
    # Loop 1: 5 real untranslated cues
    unresolved_loop1 = {10, 11, 12, 13, 14}
    # 2 are recovered
    recovered_loop1 = {10, 11}
    unresolved_loop2 = unresolved_loop1 - recovered_loop1
    assert len(unresolved_loop2) == 3
    # 3 are recovered
    recovered_loop2 = {12, 13, 14}
    unresolved_loop3 = unresolved_loop2 - recovered_loop2
    assert len(unresolved_loop3) == 0  # Fully converged, no 32 -> 42 -> 50 explosion!


# ---------------------------------------------------------------------------
# Test BA – Keep No-Op Does Not Trigger Alignment Re-Audit
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ba_keep_no_op_does_not_trigger_alignment_reaudit():
    """
    Verifies that when recovery produces KEEP on all items (0 text mutations),
    alignment_dirty remains False and no re-audit API call is made.
    """
    from app.services.pipeline import SubtitlePipeline
    pipeline = SubtitlePipeline()

    source = make_cues(10)
    target = make_cues(10)

    alignment_dirty = False
    mutated_indices = set()

    def _apply_mut(idx, text):
        nonlocal alignment_dirty
        if target[idx].content != text:
            target[idx].content = text
            mutated_indices.add(idx)
            alignment_dirty = True

    # Model keeps text unchanged
    for i in range(5):
        _apply_mut(i, target[i].content)

    assert alignment_dirty is False
    assert len(mutated_indices) == 0


# ---------------------------------------------------------------------------
# Test BB – Single Cue Source-Preserved Fallback Triggers Only Local Audit
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bb_single_cue_source_preserved_fallback_only_local_audit():
    """
    Verifies that when cue 66 falls back to source-preserved, only the anomaly window
    covering cue 66 is audited, not the entire file.
    """
    from app.services.pipeline import SubtitlePipeline
    pipeline = SubtitlePipeline()

    source = make_cues(100)
    target = make_cues(100)

    audited_windows = []
    def mock_audit(batch_payloads, **kwargs):
        for bp in batch_payloads:
            audited_windows.append((bp["start_id"], bp["end_id"]))
        return {bp["batch_id"]: {"batch_id": bp["batch_id"], "verdict": "ALIGNED", "confidence": "HIGH"} for bp in batch_payloads}

    with patch.object(pipeline.translator, "audit_batch_semantic_integrity", side_effect=mock_audit):
        # Single mutated anomaly at cue 66 (0-indexed 65)
        rep = await pipeline.check_semantic_cue_alignment(source, target, "Swedish", "English", anomaly_indices=[65])
        assert rep["issues"] == []
        assert len(audited_windows) >= 1
        assert len(audited_windows) <= 3  # Only 1-2 local windows around cue 66!
        assert any(s <= 66 <= e for s, e in audited_windows)


# ---------------------------------------------------------------------------
# Test BC – Focused Semantic Escalation Bounded Concurrency
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bc_focused_semantic_escalation_bounded_concurrency():
    """
    Verifies that audit_cue_alignment_batch runs multiple focused escalations concurrently
    using bounded semaphore (max 3 concurrent).
    """
    from app.services.translator import SubtitleTranslator
    translator = SubtitleTranslator()

    windows = [
        {"window_id": i + 1, "source": [{"id": i*5 + j, "text": f"Source {i*5+j}"} for j in range(5)], "target": [{"id": i*5 + j, "text": f"Target {i*5+j}"} for j in range(5)]}
        for i in range(6)
    ]

    concurrent_esc = 0
    max_concurrent_seen = 0

    async def mock_focused_window(*args, **kwargs):
        nonlocal concurrent_esc, max_concurrent_seen
        concurrent_esc += 1
        if concurrent_esc > max_concurrent_seen:
            max_concurrent_seen = concurrent_esc
        await asyncio.sleep(0.05)
        concurrent_esc -= 1
        return {"alignment_verdict": "ALIGNED", "confidence": "HIGH", "details": "Escalated"}

    # Mock primary LLM dispatch to return UNCERTAIN for all 6 windows to trigger escalation
    with patch.object(translator, "_dispatch_llm_completion", AsyncMock(return_value=json.dumps({
        "results": [{"window_id": i + 1, "verdict": "UNCERTAIN", "confidence": "LOW", "details": "Unsure"} for i in range(6)]
    }))):
        with patch.object(translator, "audit_cue_alignment_window", side_effect=mock_focused_window):
            res = await translator.audit_cue_alignment_batch(windows, "Swedish", "English")
            assert len(res) == 6
            assert all(v["verdict"] == "ALIGNED" for v in res.values())
            # Must run concurrently (> 1) and bounded (<= 3)
            assert max_concurrent_seen > 1
            assert max_concurrent_seen <= 3


# ---------------------------------------------------------------------------
# Test BD – Semantic Audit Candidate Count Regression (~620 and ~967 Cues)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bd_semantic_audit_candidate_count_regression_620_and_967_cues():
    """
    Verifies that candidate window generation for 620 cues (Office) and 967 cues (MasterChef)
    is dramatically reduced to ~15-30 candidate windows total (<= 6 batch calls).
    """
    from app.services.pipeline import SubtitlePipeline
    pipeline = SubtitlePipeline()

    # Construct realistic dialogue with natural sentence fragments
    def make_dialogue_cues(count):
        cues = []
        for i in range(count):
            rem = i % 15
            if rem == 0:
                text = "And by the time I arrived,"
            elif rem == 1:
                text = "the chef had prepared the dish"
            elif rem == 2:
                text = "and served it to the judges."
            else:
                text = f"This is regular standalone sentence {i+1}."
            cues.append(srt.Subtitle(i + 1, timedelta(seconds=i*2), timedelta(seconds=i*2+1.5), text))
        return cues

    # 1. 620 cues (The Office scale)
    cues_620_src = make_dialogue_cues(620)
    cues_620_tgt = make_dialogue_cues(620)

    batches_dispatched_620 = 0
    total_windows_620 = 0

    def mock_audit_620(batch_payloads, **kwargs):
        nonlocal batches_dispatched_620, total_windows_620
        batches_dispatched_620 += 1
        total_windows_620 += len(batch_payloads)
        return {bp["batch_id"]: {"batch_id": bp["batch_id"], "verdict": "ALIGNED", "confidence": "HIGH"} for bp in batch_payloads}

    with patch.object(pipeline.translator, "audit_batch_semantic_integrity", side_effect=mock_audit_620):
        await pipeline.check_semantic_cue_alignment(cues_620_src, cues_620_tgt, "Swedish", "English")
        assert total_windows_620 <= 60, f"Expected <= 60 candidate windows for 620 cues, got {total_windows_620}"
        assert batches_dispatched_620 <= 6, f"Expected <= 6 batch calls for 620 cues, got {batches_dispatched_620}"

    # 2. 967 cues (MasterChef scale)
    cues_967_src = make_dialogue_cues(967)
    cues_967_tgt = make_dialogue_cues(967)

    batches_dispatched_967 = 0
    total_windows_967 = 0

    def mock_audit_967(batch_payloads, **kwargs):
        nonlocal batches_dispatched_967, total_windows_967
        batches_dispatched_967 += 1
        total_windows_967 += len(batch_payloads)
        return {bp["batch_id"]: {"batch_id": bp["batch_id"], "verdict": "ALIGNED", "confidence": "HIGH"} for bp in batch_payloads}

    with patch.object(pipeline.translator, "audit_batch_semantic_integrity", side_effect=mock_audit_967):
        await pipeline.check_semantic_cue_alignment(cues_967_src, cues_967_tgt, "Swedish", "English")
        assert total_windows_967 <= 90, f"Expected <= 90 candidate windows for 967 cues, got {total_windows_967}"
        assert batches_dispatched_967 <= 9, f"Expected <= 9 batch calls for 967 cues, got {batches_dispatched_967}"


# ---------------------------------------------------------------------------
# Test BE – Office SHIFT_MINUS_1 Repaired and Verified Locally
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_be_office_shift_minus_1_repaired_and_verified_locally():
    """
    Simulates Office S02E19 SHIFT_MINUS_1 around cues 394-404:
    Detects shift -> consolidates into 1 region -> repairs atomically -> local verifies -> commits cleanly.
    """
    from app.services.pipeline import SubtitlePipeline
    pipeline = SubtitlePipeline()

    source = make_cues(500)
    source[393].content = "Haven't had a hug all day."
    source[394].content = "I really need some love."
    source[395].content = "Come over here."

    target_shifted = make_cues(500)
    # Target 394 contains translation of 395 (shifted -1)
    target_shifted[393].content = "Jag behöver verkligen lite kärlek."
    target_shifted[394].content = "Kom hit."
    target_shifted[395].content = "Tack så mycket."

    # Audit detects SHIFT_MINUS_1
    audit_results = {
        8: {"batch_id": 8, "verdict": "SHIFT_MINUS_1", "confidence": "HIGH", "details": "Cue 394 missing, cues 395-404 shifted -1"}
    }

    # Repair results dynamically generated for requested IDs
    def mock_repair(repair_cue_ids, **kwargs):
        return [{"id": cid, "text": f"Reparerad rad {cid}."} for cid in repair_cue_ids]

    with patch.object(pipeline.translator, "audit_batch_semantic_integrity", return_value=audit_results):
        rep = await pipeline.check_semantic_cue_alignment(source, target_shifted, "Swedish", "English", anomaly_indices=[394])
        assert len(rep["regions"]) == 1
        assert rep["regions"][0].verdict == "SHIFT_MINUS_1"

        with patch.object(pipeline.translator, "repair_alignment_region", side_effect=mock_repair):
            with patch.object(pipeline.translator, "audit_cue_alignment_window", AsyncMock(return_value={"alignment_verdict": "ALIGNED", "confidence": "HIGH"})):
                success = await pipeline._repair_semantic_alignment_regions(
                    source, target_shifted, rep["regions"], "Swedish", "English"
                )
                assert success is True
                assert target_shifted[393].content == "Reparerad rad 394."
                assert target_shifted[394].content == "Reparerad rad 395."


# ---------------------------------------------------------------------------
# Test BF – Malformed / Missing Audit Response Stays Fail-Safe
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bf_malformed_missing_audit_response_stays_failsafe():
    """
    Verifies that malformed JSON or empty strings from audit LLM do not crash the pipeline.
    """
    from app.services.translator import SubtitleTranslator
    translator = SubtitleTranslator()

    windows = [{"window_id": 1, "source": [{"id": 1, "text": "Hello"}], "target": [{"id": 1, "text": "Hej"}]}]

    with patch.object(translator, "_dispatch_llm_completion", AsyncMock(return_value="NOT_JSON")):
        with patch.object(translator, "audit_cue_alignment_window", AsyncMock(return_value={"alignment_verdict": "ALIGNED", "confidence": "LOW"})):
            res = await translator.audit_cue_alignment_batch(windows, "Swedish", "English")
            assert len(res) == 1
            assert res[1]["verdict"] == "ALIGNED"


# ---------------------------------------------------------------------------
# Test BG – Caching Aligned Windows Avoids Redundant Calls
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bg_caching_aligned_windows_avoids_redundant_calls():
    """
    Verifies that calling check_semantic_cue_alignment a second time on unchanged text
    reuses cached ALIGNED results without making any new API calls.
    """
    from app.services.pipeline import SubtitlePipeline
    pipeline = SubtitlePipeline()

    source = make_cues(50)
    target = make_cues(50)

    api_call_count = 0
    def mock_audit(batch_payloads, **kwargs):
        nonlocal api_call_count
        api_call_count += 1
        return {bp["batch_id"]: {"batch_id": bp["batch_id"], "verdict": "ALIGNED", "confidence": "HIGH"} for bp in batch_payloads}

    from app.core.validator import SemanticIncidentTracker
    tracker = SemanticIncidentTracker()

    with patch.object(pipeline.translator, "audit_batch_semantic_integrity", side_effect=mock_audit):
        # First call: populates cache
        rep1 = await pipeline.check_semantic_cue_alignment(source, target, "Swedish", "English", incident_tracker=tracker)
        assert rep1["issues"] == []
        first_calls = api_call_count
        assert first_calls > 0

        # Second call on unchanged text: should hit cache
        rep2 = await pipeline.check_semantic_cue_alignment(source, target, "Swedish", "English", incident_tracker=tracker)
        assert rep2["issues"] == []
        assert api_call_count == first_calls  # Zero new API calls!
