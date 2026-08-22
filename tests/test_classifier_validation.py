import pytest
from app.services.translator import validate_classifier_output

def test_classifier_correct_dict_root():
    items = [{"id": 1, "text": "WiFi"}]
    raw = '{"results": [{"id": 1, "action": "keep", "reason": "brand", "text": "WiFi"}]}'
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
    items = [{"id": 1, "text": "WiFi"}, {"id": 2, "text": "World"}]
    raw = '{"results": [{"id": 1, "action": "keep", "reason": "brand"}]}'
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
    items = [{"id": 1, "text": "WiFi"}]
    raw = "```json\n{\"results\": [{\"id\": 1, \"action\": \"keep\", \"reason\": \"brand\"}]}\n```"
    out = validate_classifier_output(raw, items)
    assert len(out) == 1
    assert out[0]["action"] == "keep"

def test_classifier_keep_none():
    items = [{"id": 1, "text": "Hello"}]
    raw = '{"results": [{"id": 1, "action": "keep", "reason": "none"}]}'
    out = validate_classifier_output(raw, items)
    assert out[0]["action"] == "translate"

def test_classifier_keep_proper_noun_sentence():
    items = [{"id": 1, "text": "This is a very long sentence that is not a proper noun"}]
    raw = '{"results": [{"id": 1, "action": "keep", "reason": "proper_noun"}]}'
    out = validate_classifier_output(raw, items)
    assert out[0]["action"] == "translate"

def test_classifier_keep_number_no_digits():
    items = [{"id": 1, "text": "Hello"}]
    raw = '{"results": [{"id": 1, "action": "keep", "reason": "number"}]}'
    out = validate_classifier_output(raw, items)
    assert out[0]["action"] == "translate"

def test_classifier_translate_echoes_source():
    items = [{"id": 1, "text": "Hello"}]
    raw = '{"results": [{"id": 1, "action": "translate", "text": "Hello"}]}'
    out = validate_classifier_output(raw, items)
    assert out[0]["action"] == "translate"
    assert out[0]["text"] == ""

def test_classifier_translate_valid():
    items = [{"id": 1, "text": "Hello"}]
    raw = '{"results": [{"id": 1, "action": "translate", "text": "Hej"}]}'
    out = validate_classifier_output(raw, items)
    assert out[0]["action"] == "translate"
    assert out[0]["text"] == "Hej"

def test_validate_classifier_output_strict_keep():
    items = [
        {"id": 1, "text": "Get out!"},
        {"id": 2, "text": "911"},
        {"id": 3, "text": "NASA"},
        {"id": 4, "text": "LeBron James"},
        {"id": 5, "text": "New York"},
        {"id": 6, "text": "Come here."},
        {"id": 7, "text": "What happened?"}
    ]
    import json
    raw_text = json.dumps([
        {"id": 1, "action": "keep", "reason": "proper_noun", "text": "Get out!"},
        {"id": 2, "action": "keep", "reason": "number", "text": "911"},
        {"id": 3, "action": "keep", "reason": "acronym", "text": "NASA"},
        {"id": 4, "action": "keep", "reason": "proper_noun", "text": "LeBron James"},
        {"id": 5, "action": "keep", "reason": "proper_noun", "text": "New York"},
        {"id": 6, "action": "keep", "reason": "proper_noun", "text": "Come here."},
        {"id": 7, "action": "keep", "reason": "proper_noun", "text": "What happened?"}
    ])
    
    results = validate_classifier_output(raw_text, items)
    results_map = {r["id"]: r["action"] for r in results}
    
    assert results_map[1] == "translate"
    assert results_map[2] == "keep"
    assert results_map[3] == "keep"
    assert results_map[4] == "translate"
    assert results_map[5] == "translate"
    assert results_map[6] == "translate"
    assert results_map[7] == "translate"
