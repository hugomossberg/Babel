import pytest
from app.services.translator import validate_classifier_output
import json

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
    assert results_map[4] == "keep"
    assert results_map[5] == "keep"
    assert results_map[6] == "translate"
    assert results_map[7] == "translate"

