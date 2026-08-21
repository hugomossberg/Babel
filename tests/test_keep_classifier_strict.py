import pytest
from app.services.translator import validate_classifier_output

def test_strict_non_verbal():
    # Should downgrade
    raw1 = '{"results": [{"id": 1, "action": "keep", "reason": "non_verbal"}]}'
    items1 = [{"id": 1, "text": "Come here"}]
    res1 = validate_classifier_output(raw1, items1)
    assert res1[0]["text"] == ""

    # Should downgrade
    items2 = [{"id": 1, "text": "Thank you"}]
    res2 = validate_classifier_output(raw1, items2)
    assert res2[0]["text"] == ""
    
    # Should allow
    items3 = [{"id": 1, "text": "[SIGHING]"}]
    res3 = validate_classifier_output(raw1, items3)
    assert res3[0]["action"] == "keep"

    # Should allow
    items4 = [{"id": 1, "text": "(door closes)"}]
    res4 = validate_classifier_output(raw1, items4)
    assert res4[0]["action"] == "keep"
