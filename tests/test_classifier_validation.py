import pytest
from app.services.translator import validate_classifier_output

def test_classifier_correct_dict_root():
    items = [{"id": 1, "text": "Hello"}]
    raw = '{"results": [{"id": 1, "action": "keep", "reason": "proper_noun", "text": "Hello"}]}'
    out = validate_classifier_output(raw, items)
    assert len(out) == 1
    assert out[0]["id"] == 1
    assert out[0]["action"] == "keep"

def test_classifier_list_root():
    items = [{"id": 1, "text": "Hello"}]
    raw = '[{"id": 1, "action": "translate", "text": "Hej"}]'
    out = validate_classifier_output(raw, items)
    assert len(out) == 1
    assert out[0]["action"] == "translate"

def test_classifier_list_of_strings():
    items = [{"id": 1, "text": "Hello"}]
    raw = '["translate"]'
    out = validate_classifier_output(raw, items)
    assert len(out) == 1
    assert out[0]["action"] == "translate"
    assert out[0]["reason"] == "malformed_fallback"

def test_classifier_missing_result_id():
    items = [{"id": 1, "text": "Hello"}, {"id": 2, "text": "World"}]
    raw = '{"results": [{"id": 1, "action": "keep", "reason": "proper_noun"}]}'
    out = validate_classifier_output(raw, items)
    assert len(out) == 2
    
    id2 = next(x for x in out if x["id"] == 2)
    assert id2["action"] == "translate"
    assert id2["reason"] == "malformed_fallback"

def test_classifier_invalid_action():
    items = [{"id": 1, "text": "Hello"}]
    raw = '{"results": [{"id": 1, "action": "ignore"}]}'
    out = validate_classifier_output(raw, items)
    assert len(out) == 1
    assert out[0]["action"] == "translate"
    assert out[0]["reason"] == "malformed_fallback"

def test_classifier_keep_invalid_reason():
    items = [{"id": 1, "text": "Hello"}]
    raw = '{"results": [{"id": 1, "action": "keep", "reason": "because_i_said_so"}]}'
    out = validate_classifier_output(raw, items)
    assert len(out) == 1
    assert out[0]["action"] == "translate"

def test_classifier_markdown_wrapping():
    items = [{"id": 1, "text": "Hello"}]
    raw = "```json\n{\"results\": [{\"id\": 1, \"action\": \"keep\", \"reason\": \"proper_noun\"}]}\n```"
    out = validate_classifier_output(raw, items)
    assert len(out) == 1
    assert out[0]["action"] == "keep"
