import pytest
from app.core.validator import detect_language_heuristics

def test_language_detection_short_sample():
    # Short ambiguous sample should return unknown/low conf
    res = detect_language_heuristics("Hej")
    assert res["lang"] == "unknown"

def test_language_detection_long_sample():
    # Long clear sample should return lang with high conf
    text_de = "Guten Morgen. Wie geht es dir heute? Das ist ein Test."
    res = detect_language_heuristics(text_de)
    assert res["lang"] == "de"
    assert res["confidence"] > 0.8
    
    text_fr = "Bonjour. Comment allez-vous aujourd'hui? C'est un test."
    res2 = detect_language_heuristics(text_fr)
    assert res2["lang"] == "fr"
    assert res2["confidence"] > 0.8
