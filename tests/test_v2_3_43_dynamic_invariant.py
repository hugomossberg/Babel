"""
Unit and regression tests for Pass 1 & 1.1: Dynamic invariant / identical-output correctness.

Verifies:
1. Pure structural invariants (numbers, timestamps, punctuation, markup, SDH brackets).
2. Dynamic AI-classified invariants across multiple language pairs (target English, German, French, Spanish, Swedish).
3. Completely unseen lexical words and Unicode scripts that exist in NO static word lists.
4. Fail-closed language-agnostic sanity checks (rejecting long dialogue sentences, echoing translations, missing items).
5. QA gate verification for non-Swedish targets.
"""
import pytest
import json
import srt
import unittest.mock
from datetime import timedelta
from app.services.translator import (
    is_valid_shared_or_entity_keep,
    is_pure_structural_invariant,
    is_meaningful_translation,
    validate_classifier_output,
    validate_semantic_invariant_verification_output,
    SubtitleTranslator
)
from app.services.pipeline import qa_gate


def test_pure_structural_invariants_deterministic():
    """Verifies that purely non-lexical / structural cues are recognized deterministically."""
    # Numbers, percentages, timecodes, symbols
    assert is_pure_structural_invariant("100%")
    assert is_pure_structural_invariant("0153...")
    assert is_pure_structural_invariant("12:30")
    assert is_pure_structural_invariant("$50.00")
    assert is_pure_structural_invariant("... 4.")
    assert is_pure_structural_invariant("...")
    assert is_pure_structural_invariant("---")
    assert is_pure_structural_invariant("<i></i>")
    assert is_pure_structural_invariant("")

    # Multi-line structural cues
    assert is_pure_structural_invariant("- 100%\n- 50%")
    assert is_pure_structural_invariant("- ...\n- ---")

    # Lexical and sound-effect dialog containing letters is NOT a pure structural invariant (requires AI or lexical evaluation)
    assert not is_pure_structural_invariant("Hello world")
    assert not is_pure_structural_invariant("(Klick)")
    assert not is_pure_structural_invariant("[suckar]")
    assert not is_pure_structural_invariant("Where are you going?")
    assert not is_pure_structural_invariant("Det er et smukt hus.")


def test_real_media_8_invariant_patterns():
    """Verifies all 8 verified real-media invariant patterns evaluate to True via semantic verifier."""
    # Pure structural cues evaluate to True deterministically
    assert is_pure_structural_invariant("... 4.")
    assert is_pure_structural_invariant("---")

    # All 8 verified real-media cues validated via independent batch semantic invariant verification
    raw_verifier_resp = json.dumps({
        "results": [
            {"id": 1, "invariant_in_target": True, "explanation": "Structural number with shared affirmative"},
            {"id": 2, "invariant_in_target": True, "explanation": "Non-verbal acoustic sound effect"},
            {"id": 3, "invariant_in_target": True, "explanation": "Proper name Lukas with shared affirmative"},
            {"id": 4, "invariant_in_target": True, "explanation": "Proper name and fish noun"},
            {"id": 5, "invariant_in_target": True, "explanation": "Shared noun query"},
            {"id": 6, "invariant_in_target": True, "explanation": "Shared imperative"},
            {"id": 7, "invariant_in_target": True, "explanation": "Shared religious affirmation"},
            {"id": 8, "invariant_in_target": True, "explanation": "Character name and action imperative"}
        ]
    })
    candidates = [
        {"id": 1, "target": "... 4. Ja."},
        {"id": 2, "target": "<i>(Klick)</i>"},
        {"id": 3, "target": "Ja, Lukas."},
        {"id": 4, "target": "Torsk, Þorskur."},
        {"id": 5, "target": "Torsk?"},
        {"id": 6, "target": "Se."},
        {"id": 7, "target": "- Amen.\n- Amen."},
        {"id": 8, "target": "Minato! Passa!"}
    ]
    verified_ids = validate_semantic_invariant_verification_output(raw_verifier_resp, candidates)
    assert verified_ids == {1, 2, 3, 4, 5, 6, 7, 8}

    # Verify that raw classifier output does NOT grant KEEP authority on its own (fail-closed trust boundary)
    raw_classifier_resp = json.dumps({
        "results": [
            {"id": 3, "action": "keep", "reason": "proper_noun", "text": "Ja, Lukas."}
        ]
    })
    clf_res = validate_classifier_output(raw_classifier_resp, [{"id": 3, "text": "Ja, Lukas."}])
    assert len(clf_res) == 1
    assert clf_res[0]["action"] == "translate"
    assert clf_res[0]["reason"] == "needs_semantic_verification"


def test_multilingual_unseen_invariants_across_language_pairs():
    """
    Tests dynamic invariants across various non-Swedish language pairs
    with completely unseen words (not present in any static lists) including non-Latin scripts.
    """
    # 1. Target: English (from Japanese / French)
    raw_en = json.dumps({
        "results": [
            {"id": 10, "invariant_in_target": True, "explanation": "Proper entity"},
            {"id": 11, "invariant_in_target": True, "explanation": "Universal loanword"},
            {"id": 12, "invariant_in_target": True, "explanation": "Brand"},
            {"id": 13, "invariant_in_target": True, "explanation": "Universal loanword"}
        ]
    })
    candidates_en = [
        {"id": 10, "target": "Kamehameha!"},
        {"id": 11, "target": "Sayonara."},
        {"id": 12, "target": "AeroZoid-9000"},
        {"id": 13, "target": "Cliché."}
    ]
    res_en = validate_semantic_invariant_verification_output(raw_en, candidates_en)
    assert res_en == {10, 11, 12, 13}

    # 2. Target: German (from Spanish / Italian)
    raw_de = json.dumps({
        "results": [
            {"id": 20, "invariant_in_target": True, "explanation": "Proper noun"},
            {"id": 21, "invariant_in_target": True, "explanation": "Interjection"},
            {"id": 22, "invariant_in_target": True, "explanation": "Brand + number"},
            {"id": 23, "invariant_in_target": True, "explanation": "Interjection"}
        ]
    })
    candidates_de = [
        {"id": 20, "target": "Don Quijote"},
        {"id": 21, "target": "¡Olé!"},
        {"id": 22, "target": "Quax 700"},
        {"id": 23, "target": "Bravissimo!"}
    ]
    res_de = validate_semantic_invariant_verification_output(raw_de, candidates_de)
    assert res_de == {20, 21, 22, 23}

    # 3. Target: French (from Romanian / Greek / Italian)
    raw_fr = json.dumps({
        "results": [
            {"id": 30, "invariant_in_target": True, "explanation": "Name"},
            {"id": 31, "invariant_in_target": True, "explanation": "Name"},
            {"id": 32, "invariant_in_target": True, "explanation": "Greeting + Name"}
        ]
    })
    candidates_fr = [
        {"id": 30, "target": "Zorba"},
        {"id": 31, "target": "Xylofonix"},
        {"id": 32, "target": "Ciao, Pierre."}
    ]
    res_fr = validate_semantic_invariant_verification_output(raw_fr, candidates_fr)
    assert res_fr == {30, 31, 32}

    # 4. Target: English from Non-Latin Scripts (Japanese & Arabic)
    raw_non_latin = json.dumps({
        "results": [
            {"id": 40, "invariant_in_target": True, "explanation": "Recognized loanword"},
            {"id": 41, "invariant_in_target": True, "explanation": "Recognized universal greeting"}
        ]
    })
    candidates_non_latin = [
        {"id": 40, "target": "さようなら"},
        {"id": 41, "target": "سلام"}
    ]
    res_non_latin = validate_semantic_invariant_verification_output(raw_non_latin, candidates_non_latin)
    assert res_non_latin == {40, 41}


def test_language_agnostic_sanity_checks_fail_closed():
    """
    Verifies that language-agnostic sanity checks fail-closed against:
    - Long conversational dialogue mistakenly labeled KEEP by LLM
    - Translations echoing source text
    - Missing IDs in classifier response
    """
    # 1. Long dialogue sentence labeled KEEP -> downgraded to TRANSLATE with empty text
    long_sentence = "I think we should probably head to the train station right now before it gets too dark outside."
    raw_bad_keep = json.dumps({
        "results": [
            {"id": 1, "action": "keep", "reason": "shared_word", "text": long_sentence}
        ]
    })
    items = [{"id": 1, "text": long_sentence}]
    res = validate_classifier_output(raw_bad_keep, items)
    assert len(res) == 1
    assert res[0]["action"] == "translate"
    assert res[0]["text"] == ""

    # 2. Translate action with echo text -> cleared to empty string
    raw_echo = json.dumps({
        "results": [
            {"id": 2, "action": "translate", "reason": "translate", "text": "Where are you going?"}
        ]
    })
    items_echo = [{"id": 2, "text": "Where are you going?"}]
    res_echo = validate_classifier_output(raw_echo, items_echo)
    assert len(res_echo) == 1
    assert res_echo[0]["action"] == "translate"
    assert res_echo[0]["text"] == ""

    # 3. Missing item in classifier response -> failsafe generates TRANSLATE with empty text
    raw_missing = json.dumps({
        "results": [
            {"id": 1, "action": "keep", "reason": "symbol", "text": "100%"}
        ]
    })
    items_missing = [{"id": 1, "text": "100%"}, {"id": 2, "text": "Unseen cue"}]
    res_missing = validate_classifier_output(raw_missing, items_missing)
    assert len(res_missing) == 2
    assert res_missing[0]["action"] == "keep"
    assert res_missing[1]["action"] == "translate"
    assert res_missing[1]["text"] == ""
    assert res_missing[1]["reason"] == "malformed_fallback"


def test_qa_gate_multilingual_non_swedish_targets():
    """
    Verifies that qa_gate correctly accepts verified safe_ids for non-Swedish target languages
    (e.g. target English from French, target German from Spanish) without Swedish assumptions.
    """
    # French to English run
    src_fr = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="100%"),
        srt.Subtitle(index=2, start=timedelta(seconds=3), end=timedelta(seconds=4), content="<i>(Bip)</i>"),
        srt.Subtitle(index=3, start=timedelta(seconds=5), end=timedelta(seconds=6), content="Dr. Moreau"),
        srt.Subtitle(index=4, start=timedelta(seconds=7), end=timedelta(seconds=8), content="Bonjour, comment allez-vous?"),
    ]
    trans_en = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="100%"),
        srt.Subtitle(index=2, start=timedelta(seconds=3), end=timedelta(seconds=4), content="<i>(Bip)</i>"),
        srt.Subtitle(index=3, start=timedelta(seconds=5), end=timedelta(seconds=6), content="Dr. Moreau"),
        srt.Subtitle(index=4, start=timedelta(seconds=7), end=timedelta(seconds=8), content="Hello, how are you doing?"),
    ]
    safe_ids = [0, 1, 2]
    res = qa_gate(
        source_subs=src_fr,
        translated_subs=trans_en,
        target_lang_code="en",
        source_language_name="French",
        safe_ids=safe_ids
    )
    assert res["passed"] is True
    assert res["score"] == 100
    assert len(res["real_untranslated_ids"]) == 0


def test_adversarial_classifier_dialogue_hallucinations_fail_closed():
    """
    Adversarial testing across various language pairs and scripts where the classifier
    deliberately hallucinates KEEP on conversational dialogue sentences.
    Babel MUST reject all of them fail-closed.
    """
    adversarial_payload = json.dumps({
        "results": [
            {"id": 1, "action": "keep", "reason": "shared_word", "text": "Come here."},
            {"id": 2, "action": "keep", "reason": "cognate", "text": "I need help."},
            {"id": 3, "action": "keep", "reason": "shared_word", "text": "What happened?"},
            {"id": 4, "action": "keep", "reason": "shared_word", "text": "Wait for me."},
            {"id": 5, "action": "keep", "reason": "shared_word", "text": "This is mine."},
            {"id": 6, "action": "keep", "reason": "shared_word", "text": "Was ist los?"},
            {"id": 7, "action": "keep", "reason": "shared_word", "text": "Attends un instant."},
            {"id": 8, "action": "keep", "reason": "shared_word", "text": "No puedo más."}
        ]
    })
    items = [
        {"id": 1, "text": "Come here."},
        {"id": 2, "text": "I need help."},
        {"id": 3, "text": "What happened?"},
        {"id": 4, "text": "Wait for me."},
        {"id": 5, "text": "This is mine."},
        {"id": 6, "text": "Was ist los?"},
        {"id": 7, "text": "Attends un instant."},
        {"id": 8, "text": "No puedo más."}
    ]
    res = validate_classifier_output(adversarial_payload, items)
    assert len(res) == 8
    for item_res in res:
        assert item_res["action"] == "translate", f"Failed to downgrade adversarial item: {item_res}"
        assert item_res["text"] == "", f"Failed to clear text for adversarial item: {item_res}"


def test_alphabetic_non_verbal_positive_and_negative_flows():
    """
    Tests both positive and negative paths for alphabetic non-verbal / SDH cues:
    1. Positive: AI classifier identifies legitimate identical sound representations across language pairs.
       -> Allowed as safe_id / KEEP without recovery.
    2. Negative: Language-dependent descriptive SDH cues are correctly translated or intercepted if misclassified.
    """
    # 1. POSITIVE: Cross-lingual onomatopoeic invariants verified by AI
    pos_payload = json.dumps({
        "results": [
            {"id": 1, "invariant_in_target": True, "explanation": "Acoustic click sound effect"},
            {"id": 2, "invariant_in_target": True, "explanation": "Acoustic beep sound effect"},
            {"id": 3, "invariant_in_target": True, "explanation": "Acoustic laugh onomatopoeia"},
            {"id": 4, "invariant_in_target": True, "explanation": "Acoustic chime sound effect"}
        ]
    })
    pos_items = [
        {"id": 1, "target": "<i>(Klick)</i>"},
        {"id": 2, "target": "(Bip)"},
        {"id": 3, "target": "(Haha)"},
        {"id": 4, "target": "(Ding-Dong)"}
    ]
    pos_res = validate_semantic_invariant_verification_output(pos_payload, pos_items)
    assert pos_res == {1, 2, 3, 4}

    # Verify pipeline / QA gate accepts these in safe_ids with 0 dropped cues
    src_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="<i>(Klick)</i>"),
        srt.Subtitle(index=2, start=timedelta(seconds=3), end=timedelta(seconds=4), content="(Bip)"),
        srt.Subtitle(index=3, start=timedelta(seconds=5), end=timedelta(seconds=6), content="(Haha)"),
        srt.Subtitle(index=4, start=timedelta(seconds=7), end=timedelta(seconds=8), content="(Ding-Dong)")
    ]
    trans_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="<i>(Klick)</i>"),
        srt.Subtitle(index=2, start=timedelta(seconds=3), end=timedelta(seconds=4), content="(Bip)"),
        srt.Subtitle(index=3, start=timedelta(seconds=5), end=timedelta(seconds=6), content="(Haha)"),
        srt.Subtitle(index=4, start=timedelta(seconds=7), end=timedelta(seconds=8), content="(Ding-Dong)")
    ]
    safe_ids = [0, 1, 2, 3]
    context_verified_ids = {0, 1, 2, 3}
    qa_check = qa_gate(
        source_subs=src_subs,
        translated_subs=trans_subs,
        target_lang_code="sv",
        source_language_name="German",
        safe_ids=safe_ids,
        context_verified_ids=context_verified_ids
    )
    assert qa_check["passed"] is True
    assert qa_check["score"] == 100
    assert len(qa_check["real_untranslated_ids"]) == 0

    # 2. NEGATIVE: Descriptive language-dependent SDH cues must be TRANSLATED
    neg_payload = json.dumps({
        "results": [
            {"id": 10, "action": "translate", "reason": "translate", "text": "[suckar]"},
            {"id": 11, "action": "translate", "reason": "translate", "text": "(dörren stängs)"},
            {"id": 12, "action": "translate", "reason": "translate", "text": "(applaudissements)"}
        ]
    })
    neg_items = [
        {"id": 10, "text": "[SIGHING]"},
        {"id": 11, "text": "(door closes)"},
        {"id": 12, "text": "(applåder)"}
    ]
    neg_res = validate_classifier_output(neg_payload, neg_items)
    assert len(neg_res) == 3
    assert neg_res[0]["action"] == "translate"
    assert neg_res[0]["text"] == "[suckar]"
    assert neg_res[1]["action"] == "translate"
    assert neg_res[1]["text"] == "(dörren stängs)"
    assert neg_res[2]["action"] == "translate"
    assert neg_res[2]["text"] == "(applaudissements)"

    # 3. NEGATIVE INTERCEPT: Hallucinated KEEP on multi-word descriptive SDH phrases is rejected
    hallucinated_payload = json.dumps({
        "results": [
            {"id": 20, "action": "keep", "reason": "non_verbal", "text": "(door closes)"},
            {"id": 21, "action": "keep", "reason": "non_verbal", "text": "[music playing]"},
            {"id": 22, "action": "keep", "reason": "non_verbal", "text": "[whispering softly]"}
        ]
    })
    hallucinated_items = [
        {"id": 20, "text": "(door closes)"},
        {"id": 21, "text": "[music playing]"},
        {"id": 22, "text": "[whispering softly]"}
    ]
    hal_res = validate_classifier_output(hallucinated_payload, hallucinated_items)
    assert len(hal_res) == 3
    for r in hal_res:
        assert r["action"] == "translate", f"Expected downgrade to TRANSLATE for descriptive SDH: {r}"
        assert r["text"] == "", f"Expected cleared text for descriptive SDH: {r}"


def test_adversarial_sighing_fail_closed_while_klick_passes():
    """
    Explicit regression test for adversarial single-token bracketed cues:
    - [SIGHING] falsely mocked as KEEP / non_verbal MUST fail closed (downgraded to TRANSLATE with empty text).
    - (Klick) with positive invariant classification MUST be allowed as KEEP.
    """
    # 1. Adversarial mock: [SIGHING] claimed as KEEP/non_verbal
    bad_sighing_payload = json.dumps({
        "results": [
            {"id": 1, "action": "keep", "reason": "non_verbal", "text": "[SIGHING]"}
        ]
    })
    res_sighing = validate_classifier_output(bad_sighing_payload, [{"id": 1, "text": "[SIGHING]"}])
    assert len(res_sighing) == 1
    assert res_sighing[0]["action"] == "translate"
    assert res_sighing[0]["text"] == ""

    # 2. Legitimate invariant: (Klick) claimed as KEEP/non_verbal with positive invariant verification
    good_klick_payload = json.dumps({
        "results": [
            {"id": 2, "invariant_in_target": True, "explanation": "Identical onomatopoeia"}
        ]
    })
    res_klick = validate_semantic_invariant_verification_output(good_klick_payload, [{"id": 2, "target": "<i>(Klick)</i>"}])
    assert res_klick == {2}


@pytest.mark.asyncio
async def test_two_stage_semantic_verification_adversarial_sighing_and_batched_efficiency():
    """
    Tests the complete 2-stage verification flow on SubtitleTranslator:
    1. First classifier proposes KEEP on [SIGHING] (hallucination), (Klick), and 'Torsk, Þorskur.'.
    2. Stage 2 (Semantic Verifier) is called in EXACTLY 1 BATCH CALL for all ambiguous candidates.
    3. Semantic Verifier rejects [SIGHING] (invariant_in_target=False) and approves (Klick) + 'Torsk, Þorskur.'.
    4. Result: [SIGHING] fails closed to TRANSLATE with empty text.
       (Klick) and 'Torsk, Þorskur.' are approved as verified KEEP.
    5. Verifies batch call count is EXACTLY 1 for N candidates (not N calls).
    """
    translator = SubtitleTranslator()

    source_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="[SIGHING]"),
        srt.Subtitle(index=2, start=timedelta(seconds=3), end=timedelta(seconds=4), content="<i>(Klick)</i>"),
        srt.Subtitle(index=3, start=timedelta(seconds=5), end=timedelta(seconds=6), content="Torsk, Þorskur."),
        srt.Subtitle(index=4, start=timedelta(seconds=7), end=timedelta(seconds=8), content="Where are you going?"),
    ]
    translated_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="[SIGHING]"),
        srt.Subtitle(index=2, start=timedelta(seconds=3), end=timedelta(seconds=4), content="<i>(Klick)</i>"),
        srt.Subtitle(index=3, start=timedelta(seconds=5), end=timedelta(seconds=6), content="Torsk, Þorskur."),
        srt.Subtitle(index=4, start=timedelta(seconds=7), end=timedelta(seconds=8), content="Where are you going?"),
    ]

    items = [
        {"id": 0, "text": "[SIGHING]"},
        {"id": 1, "text": "<i>(Klick)</i>"},
        {"id": 2, "text": "Torsk, Þorskur."},
        {"id": 3, "text": "Where are you going?"},
    ]

    stage1_response = json.dumps({
        "results": [
            {"id": 0, "action": "keep", "reason": "non_verbal", "text": "[SIGHING]"},
            {"id": 1, "action": "keep", "reason": "non_verbal", "text": "<i>(Klick)</i>"},
            {"id": 2, "action": "keep", "reason": "proper_noun", "text": "Torsk, Þorskur."},
            {"id": 3, "action": "translate", "reason": "translate", "text": "Vart är du på väg?"}
        ]
    })

    verify_call_count = 0
    candidate_batch_size = 0

    async def mock_verify(candidates, target_language, show_title="", job_id=None, source_language="English"):
        nonlocal verify_call_count, candidate_batch_size
        verify_call_count += 1
        candidate_batch_size = len(candidates)
        raw_verifier_resp = json.dumps({
            "results": [
                {"id": 0, "invariant_in_target": False, "explanation": "English descriptive SDH requires translation"},
                {"id": 1, "invariant_in_target": True, "explanation": "Onomatopoeic sound identical in Swedish"},
                {"id": 2, "invariant_in_target": True, "explanation": "Character names invariant in Swedish"}
            ]
        })
        return validate_semantic_invariant_verification_output(raw_verifier_resp, candidates, show_title=show_title)

    translator.verify_single_occurrence_entities = mock_verify
    translator.verify_alphabetic_invariants_batch = mock_verify

    def mock_get_setting(k, default=None):
        if k == "ai_provider":
            return "gemini"
        if k == "gemini_api_key":
            return "mock_key"
        return default or "mock_value"

    with unittest.mock.patch("google.genai.Client"), \
         unittest.mock.patch("app.services.translator.get_setting", side_effect=mock_get_setting), \
         unittest.mock.patch("asyncio.get_event_loop") as mock_loop:
        mock_loop_inst = unittest.mock.MagicMock()
        mock_resp = unittest.mock.MagicMock()
        mock_resp.text = stage1_response
        mock_loop_inst.run_in_executor = unittest.mock.AsyncMock(return_value=mock_resp)
        mock_loop.return_value = mock_loop_inst

        results = await translator.classify_and_recover_identical(
            items=items,
            target_language="Swedish",
            show_title="Test Movie",
            source_subs=source_subs,
            translated_subs=translated_subs,
            source_language="English"
        )

    assert verify_call_count == 1, f"Expected exactly 1 batched verification call, got {verify_call_count}"
    assert candidate_batch_size == 3, f"Expected batch size of 3 candidates, got {candidate_batch_size}"

    results_map = {r["id"]: r for r in results}
    assert results_map[0]["action"] == "translate"
    assert results_map[0]["text"] == ""
    assert results_map[0]["reason"] == "unverified_invariant"

    assert results_map[1]["action"] == "keep"
    assert results_map[1]["text"] == "<i>(Klick)</i>"
    assert results_map[1]["reason"] == "verified_non_verbal"

    assert results_map[2]["action"] == "keep"
    assert results_map[2]["text"] == "Torsk, Þorskur."
    assert results_map[2]["reason"] == "verified_proper_noun"

    assert results_map[3]["action"] == "translate"
    assert results_map[3]["text"] == "Vart är du på väg?"


@pytest.mark.asyncio
async def test_alphabetic_non_verbal_cannot_bypass_semantic_verification_and_5_behaviors():
    """
    Explicit regression test proving:
    1. An alphabetic non_verbal that the legacy helper would call safe (e.g. '(Bip)', 'Mmm.', '(Ding)')
       MUST NOT bypass semantic verification in the new invariant path.
    2. Pure structural cues (e.g. '100%', '12:30', '♪ ♪', '---') result in 0 extra verification calls.
    3. Three ambiguous alphabetic KEEP candidates result in EXACTLY 1 batch verification call.
    4. Semantic verifier returning False results in TRANSLATE/recovery (empty text).
    5. Semantic verifier returning True results in safe_id / KEEP.
    """
    # 1. Prove legacy-safe alphabetic sounds cannot bypass validate_classifier_output
    raw_legacy_sound = json.dumps({
        "results": [
            {"id": 1, "action": "keep", "reason": "non_verbal", "text": "Mmm."},
            {"id": 2, "action": "keep", "reason": "non_verbal", "text": "(Bip)"},
            {"id": 3, "action": "keep", "reason": "non_verbal", "text": "(Ding)"}
        ]
    })
    items_sound = [
        {"id": 1, "text": "Mmm."},
        {"id": 2, "text": "(Bip)"},
        {"id": 3, "text": "(Ding)"}
    ]
    val_res = validate_classifier_output(raw_legacy_sound, items_sound)
    assert len(val_res) == 3
    for r in val_res:
        assert r["action"] == "translate", f"Alphabetic sound must not bypass to KEEP: {r}"
        assert r["reason"] == "needs_semantic_verification", f"Expected needs_semantic_verification: {r}"
        assert r["text"] == "", f"Expected cleared text: {r}"

    # 2. Pure structural cues -> 0 extra verification calls
    translator = SubtitleTranslator()
    struct_source = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "100%"),
        srt.Subtitle(2, timedelta(seconds=3), timedelta(seconds=4), "12:30"),
        srt.Subtitle(3, timedelta(seconds=5), timedelta(seconds=6), "♪ ♪"),
        srt.Subtitle(4, timedelta(seconds=7), timedelta(seconds=8), "---")
    ]
    struct_items = [
        {"id": 0, "text": "100%"},
        {"id": 1, "text": "12:30"},
        {"id": 2, "text": "♪ ♪"},
        {"id": 3, "text": "---"}
    ]
    stage1_struct_resp = json.dumps({
        "results": [
            {"id": 0, "action": "keep", "reason": "symbol", "text": "100%"},
            {"id": 1, "action": "keep", "reason": "number", "text": "12:30"},
            {"id": 2, "action": "keep", "reason": "symbol", "text": "♪ ♪"},
            {"id": 3, "action": "keep", "reason": "symbol", "text": "---"}
        ]
    })
    verify_call_count = 0
    async def mock_verify(candidates, target_language, show_title="", job_id=None, source_language="English"):
        nonlocal verify_call_count
        verify_call_count += 1
        return set()
    translator.verify_single_occurrence_entities = mock_verify
    translator.verify_alphabetic_invariants_batch = mock_verify

    def mock_get_setting(k, default=None):
        if k == "ai_provider": return "gemini"
        if k == "gemini_api_key": return "mock_key"
        return default or "mock_value"

    with unittest.mock.patch("google.genai.Client"), \
         unittest.mock.patch("app.services.translator.get_setting", side_effect=mock_get_setting), \
         unittest.mock.patch("asyncio.get_event_loop") as mock_loop:
        mock_loop_inst = unittest.mock.MagicMock()
        mock_resp = unittest.mock.MagicMock()
        mock_resp.text = stage1_struct_resp
        mock_loop_inst.run_in_executor = unittest.mock.AsyncMock(return_value=mock_resp)
        mock_loop.return_value = mock_loop_inst

        struct_res = await translator.classify_and_recover_identical(
            items=struct_items,
            target_language="Swedish",
            show_title="Test Struct",
            source_subs=struct_source,
            translated_subs=struct_source,
            source_language="English"
        )
    assert verify_call_count == 0, f"Pure structural cues must require 0 verification calls, got {verify_call_count}"
    assert len(struct_res) == 4
    for r in struct_res:
        assert r["action"] == "keep"

    # 3, 4, 5. Three ambiguous candidates -> Exactly 1 batch call, True->KEEP, False->TRANSLATE
    ambig_source = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "(Bip)"),
        srt.Subtitle(2, timedelta(seconds=3), timedelta(seconds=4), "[SIGHING]"),
        srt.Subtitle(3, timedelta(seconds=5), timedelta(seconds=6), "Torsk, Þorskur.")
    ]
    ambig_items = [
        {"id": 0, "text": "(Bip)"},
        {"id": 1, "text": "[SIGHING]"},
        {"id": 2, "text": "Torsk, Þorskur."}
    ]
    stage1_ambig_resp = json.dumps({
        "results": [
            {"id": 0, "action": "keep", "reason": "non_verbal", "text": "(Bip)"},
            {"id": 1, "action": "keep", "reason": "non_verbal", "text": "[SIGHING]"},
            {"id": 2, "action": "keep", "reason": "proper_noun", "text": "Torsk, Þorskur."}
        ]
    })
    batch_verify_calls = 0
    batch_cand_size = 0
    async def mock_batch_verify(candidates, target_language, show_title="", job_id=None, source_language="English"):
        nonlocal batch_verify_calls, batch_cand_size
        batch_verify_calls += 1
        batch_cand_size = len(candidates)
        raw_resp = json.dumps({
            "results": [
                {"id": 0, "invariant_in_target": True, "explanation": "Valid invariant beep sound"},
                {"id": 1, "invariant_in_target": False, "explanation": "English descriptive SDH"},
                {"id": 2, "invariant_in_target": True, "explanation": "Proper noun"}
            ]
        })
        return validate_semantic_invariant_verification_output(raw_resp, candidates, show_title=show_title)

    translator.verify_single_occurrence_entities = mock_batch_verify
    translator.verify_alphabetic_invariants_batch = mock_batch_verify

    with unittest.mock.patch("google.genai.Client"), \
         unittest.mock.patch("app.services.translator.get_setting", side_effect=mock_get_setting), \
         unittest.mock.patch("asyncio.get_event_loop") as mock_loop:
        mock_loop_inst = unittest.mock.MagicMock()
        mock_resp = unittest.mock.MagicMock()
        mock_resp.text = stage1_ambig_resp
        mock_loop_inst.run_in_executor = unittest.mock.AsyncMock(return_value=mock_resp)
        mock_loop.return_value = mock_loop_inst

        ambig_res = await translator.classify_and_recover_identical(
            items=ambig_items,
            target_language="Swedish",
            show_title="Test Ambig",
            source_subs=ambig_source,
            translated_subs=ambig_source,
            source_language="English"
        )

    assert batch_verify_calls == 1, f"Expected exactly 1 batch call for 3 candidates, got {batch_verify_calls}"
    assert batch_cand_size == 3, f"Expected batch size 3, got {batch_cand_size}"
    res_map = {r["id"]: r for r in ambig_res}
    assert res_map[0]["action"] == "keep"
    assert res_map[0]["reason"] == "verified_non_verbal"
    assert res_map[1]["action"] == "translate"
    assert res_map[1]["text"] == ""
    assert res_map[2]["action"] == "keep"
    assert res_map[2]["reason"] == "verified_proper_noun"


@pytest.mark.asyncio
async def test_acronym_and_brand_must_go_through_semantic_verification_batch():
    """
    Verifies that 'acronym' and 'brand' with alphabetic content CANNOT bypass Stage 2.
    They must be routed to the same semantic verification batch.
    - If semantic verifier returns False -> TRANSLATE (empty text)
    - If semantic verifier returns True -> KEEP (verified)
    - Pure numbers and symbols still require 0 extra verification calls.
    """
    translator = SubtitleTranslator()

    source_subs = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "NASA"),
        srt.Subtitle(2, timedelta(seconds=3), timedelta(seconds=4), "Bluetooth"),
        srt.Subtitle(3, timedelta(seconds=5), timedelta(seconds=6), "FakeBrand")
    ]
    items = [
        {"id": 0, "text": "NASA"},
        {"id": 1, "text": "Bluetooth"},
        {"id": 2, "text": "FakeBrand"}
    ]
    stage1_resp = json.dumps({
        "results": [
            {"id": 0, "action": "keep", "reason": "acronym", "text": "NASA"},
            {"id": 1, "action": "keep", "reason": "brand", "text": "Bluetooth"},
            {"id": 2, "action": "keep", "reason": "brand", "text": "FakeBrand"}
        ]
    })

    batch_calls = 0
    batch_cand_size = 0
    async def mock_batch_verify(candidates, target_language, show_title="", job_id=None, source_language="English"):
        nonlocal batch_calls, batch_cand_size
        batch_calls += 1
        batch_cand_size = len(candidates)
        raw_resp = json.dumps({
            "results": [
                {"id": 0, "invariant_in_target": True, "explanation": "Valid space agency acronym"},
                {"id": 1, "invariant_in_target": True, "explanation": "Valid tech standard brand"},
                {"id": 2, "invariant_in_target": False, "explanation": "Unrecognized brand, should translate contextually"}
            ]
        })
        return validate_semantic_invariant_verification_output(raw_resp, candidates, show_title=show_title)

    translator.verify_single_occurrence_entities = mock_batch_verify
    translator.verify_alphabetic_invariants_batch = mock_batch_verify

    def mock_get_setting(k, default=None):
        if k == "ai_provider": return "gemini"
        if k == "gemini_api_key": return "mock_key"
        return default or "mock_value"

    with unittest.mock.patch("google.genai.Client"), \
         unittest.mock.patch("app.services.translator.get_setting", side_effect=mock_get_setting), \
         unittest.mock.patch("asyncio.get_event_loop") as mock_loop:
        mock_loop_inst = unittest.mock.MagicMock()
        mock_resp = unittest.mock.MagicMock()
        mock_resp.text = stage1_resp
        mock_loop_inst.run_in_executor = unittest.mock.AsyncMock(return_value=mock_resp)
        mock_loop.return_value = mock_loop_inst

        results = await translator.classify_and_recover_identical(
            items=items,
            target_language="Swedish",
            show_title="Test Brands",
            source_subs=source_subs,
            translated_subs=source_subs,
            source_language="English"
        )

    assert batch_calls == 1, f"Expected exactly 1 batch verification call for acronyms/brands, got {batch_calls}"
    assert batch_cand_size == 3, f"Expected 3 candidates in batch, got {batch_cand_size}"

    res_map = {r["id"]: r for r in results}
    # NASA: verifier True -> KEEP
    assert res_map[0]["action"] == "keep"
    assert res_map[0]["reason"] == "verified_acronym"

    # Bluetooth: verifier True -> KEEP
    assert res_map[1]["action"] == "keep"
    assert res_map[1]["reason"] == "verified_brand"

    # FakeBrand: verifier False -> TRANSLATE (empty text)
    assert res_map[2]["action"] == "translate"
    assert res_map[2]["text"] == ""





