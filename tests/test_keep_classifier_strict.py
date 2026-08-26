import pytest
from app.services.translator import validate_classifier_output

def test_strict_non_verbal():
    # Should downgrade (contains dialogue or letters)
    raw1 = '{"results": [{"id": 1, "action": "keep", "reason": "non_verbal"}]}'
    items1 = [{"id": 1, "text": "Come here"}]
    res1 = validate_classifier_output(raw1, items1)
    assert res1[0]["text"] == ""

    # Should downgrade
    items2 = [{"id": 1, "text": "Thank you"}]
    res2 = validate_classifier_output(raw1, items2)
    assert res2[0]["text"] == ""

    # Should downgrade (contains descriptive verbal participle)
    items3 = [{"id": 1, "text": "[SIGHING]"}]
    res3 = validate_classifier_output(raw1, items3)
    assert res3[0]["text"] == ""

    # Should downgrade (multi-word descriptive SDH phrase)
    items4 = [{"id": 1, "text": "(door closes)"}]
    res4 = validate_classifier_output(raw1, items4)
    assert res4[0]["text"] == ""

    # Should downgrade (contains letters)
    items5 = [{"id": 1, "text": "[I love you]"}]
    res5 = validate_classifier_output(raw1, items5)
    assert res5[0]["text"] == ""

    # Pure symbols/music allowed as keep
    items6 = [{"id": 1, "text": "♪ ♪"}]
    res6 = validate_classifier_output(raw1, items6)
    assert res6[0]["action"] == "keep"

def test_strict_proper_nouns():
    raw_proper = '{"results": [{"id": 1, "action": "keep", "reason": "proper_noun"}]}'

    # Proper nouns fail closed to TRANSLATE (downgraded) so the model evaluates context and localization
    for candidate in [
        "John Smith", "New York", "Los Angeles", "Harry Potter",
        "Red Alert", "Green Light", "Dead Body", "Black Car", "White House",
        "Good Morning", "Happy Birthday", "Big Problem", "Last Chance",
        "First Time", "New Plan", "Bad Idea", "Blue Moon",
        "Hello", "This Is My House", "I Love You", "Netflix", "Apple", "John"
    ]:
        res = validate_classifier_output(raw_proper, [{"id": 1, "text": candidate}])
        assert res[0]["action"] == "translate"
        assert res[0]["text"] == ""  # Downgraded to empty text (TRANSLATE)
