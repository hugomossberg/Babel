import pytest
from datetime import timedelta
import srt

from app.services.translator import has_entity_evidence, validate_classifier_output, is_deterministically_safe_keep
from app.services.pipeline import qa_gate

def make_subs(pairs):
    src_subs = []
    trans_subs = []
    for idx, (src, trans) in enumerate(pairs, 1):
        src_subs.append(srt.Subtitle(idx, timedelta(seconds=idx), timedelta(seconds=idx+1), src))
        trans_subs.append(srt.Subtitle(idx, timedelta(seconds=idx), timedelta(seconds=idx+1), trans))
    return src_subs, trans_subs

def test_has_entity_evidence_basic():
    # Episode context: Boston appears in genuinely translated dialogue
    pairs = [
        ("Will it be Boston or Atlanta?", "Blir det Boston eller Atlanta?"),
        ("I had to attend a funeral in Boston.", "Jag var tvungen att gå på en begravning i Boston."),
        ("Fuck Boston.", "Dra åt helvete Boston."),
        ("Boston.", "Boston."),  # target candidate
    ]
    src_subs, trans_subs = make_subs(pairs)

    # Candidate "Boston." (idx 3) should have evidence
    assert has_entity_evidence("Boston.", src_subs, trans_subs, target_idx=3) is True

def test_has_entity_evidence_multi_token_proper_nouns():
    # Candidate "Cam, Reggie."
    pairs = [
        ("Cam is playing tonight.", "Cam spelar ikväll."),
        ("Reggie told me the news.", "Reggie berättade nyheterna för mig."),
        ("Cam, Reggie.", "Cam, Reggie."), # target candidate
    ]
    src_subs, trans_subs = make_subs(pairs)

    assert has_entity_evidence("Cam, Reggie.", src_subs, trans_subs, target_idx=2) is True

def test_has_entity_evidence_rejects_common_english_words():
    # Common English words like "Party?", "Red Alert", "Help" must NEVER be kept via evidence
    pairs = [
        ("We are going to the party.", "Vi ska gå på festen."),
        ("Party?", "Party?"), # common word
    ]
    src_subs, trans_subs = make_subs(pairs)

    assert has_entity_evidence("Party?", src_subs, trans_subs, target_idx=1) is False
    assert has_entity_evidence("Red Alert", src_subs, trans_subs, target_idx=1) is False
    assert has_entity_evidence("Help", src_subs, trans_subs, target_idx=1) is False

def test_has_entity_evidence_no_evidence_in_run():
    # "Morgan?" has NO occurrences in other lines in the episode
    pairs = [
        ("Are you ready?", "Är du redo?"),
        ("Let's go.", "Nu drar vi."),
        ("Morgan?", "Morgan?"), # target
    ]
    src_subs, trans_subs = make_subs(pairs)

    assert has_entity_evidence("Morgan?", src_subs, trans_subs, target_idx=2) is False

def test_has_entity_evidence_requires_meaningful_translation_for_evidence():
    # If the other lines containing the word are ALSO untranslated / source copies,
    # they cannot serve as evidence.
    pairs = [
        ("Boston is great.", "Boston is great."), # identical / not translated
        ("Boston.", "Boston."), # target candidate
    ]
    src_subs, trans_subs = make_subs(pairs)

    assert has_entity_evidence("Boston.", src_subs, trans_subs, target_idx=1) is False

def test_has_entity_evidence_word_boundary_isolation():
    # Ensure token boundary \b is respected: "Cat" in "Catching" or "Catalog" should NOT provide evidence
    pairs = [
        ("I am catching the ball.", "Jag fångar bollen."),
        ("Look at the catalog.", "Titta i katalogen."),
        ("Cat.", "Cat."), # target
    ]
    src_subs, trans_subs = make_subs(pairs)

    assert has_entity_evidence("Cat.", src_subs, trans_subs, target_idx=2) is False

def test_validate_classifier_output_with_evidence():
    pairs = [
        ("Boston or Atlanta?", "Boston eller Atlanta?"),
        ("Boston.", "Boston."),
    ]
    src_subs, trans_subs = make_subs(pairs)

    raw_json = '{"results": [{"id": 1, "action": "keep", "reason": "proper_noun", "text": "Boston."}]}'
    items = [{"id": 1, "text": "Boston."}]

    results = validate_classifier_output(raw_json, items, show_title="Test Show", source_subs=src_subs, translated_subs=trans_subs)
    assert len(results) == 1
    assert results[0]["action"] == "keep"
    assert "evidence" in results[0]["reason"]

def test_validate_classifier_output_without_evidence_downgrades():
    pairs = [
        ("Hello there.", "Hej där."),
        ("Morgan?", "Morgan?"),
    ]
    src_subs, trans_subs = make_subs(pairs)

    raw_json = '{"results": [{"id": 1, "action": "keep", "reason": "proper_noun", "text": "Morgan?"}]}'
    items = [{"id": 1, "text": "Morgan?"}]

    results = validate_classifier_output(raw_json, items, show_title="Test Show", source_subs=src_subs, translated_subs=trans_subs)
    assert len(results) == 1
    # Without evidence, proper_noun KEEP is downgraded to TRANSLATE
    assert results[0]["action"] == "translate"
    assert results[0]["text"] == ""

def test_qa_gate_with_evidence_based_safe_id():
    pairs = [
        ("Welcome to Boston.", "Välkommen till Boston."),
        ("Boston.", "Boston."),
    ]
    src_subs, trans_subs = make_subs(pairs)

    # Without safe_ids, line 1 is untranslated
    gate_res_no_safe = qa_gate(src_subs, trans_subs, target_lang_code="sv", safe_ids=[])
    assert gate_res_no_safe["real_untranslated_ids"] == [1]

    # With safe_ids containing 1, evidence allows it through
    gate_res_safe = qa_gate(src_subs, trans_subs, target_lang_code="sv", safe_ids=[1])
    assert gate_res_safe["real_untranslated_ids"] == []
    assert gate_res_safe["passed"] is True
