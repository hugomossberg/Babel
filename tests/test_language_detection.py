import pytest
from app.core.validator import detect_language_heuristics, check_language_representative
from app.core.languages import normalize_language_code, get_language, LANGUAGES

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

def test_language_detection_new_supported_languages():
    samples = {
        "bg": "Здравейте, как сте днес? Това е тест на български език.",
        "cs": "Ahoj, jak se máš? Toto je test v českém jazyce.",
        "ro": "Bună ziua, ce mai faci? Acesta este un test în limba română.",
        "hu": "Jó napot kívánok, hogy vagy? Ez egy teszt magyar nyelven.",
        "tr": "Merhaba, nasılsın? Bu Türkçe dilinde bir testtir.",
        "el": "Γεια σας, πώς είστε σήμερα; Αυτή είναι μια δοκιμή στα ελληνικά."
    }
    for code, text in samples.items():
        res = detect_language_heuristics(text, expected_language=code)
        assert res["lang"] == code
        assert res["confidence"] > 0.8

def test_language_normalization():
    test_cases = [
        ("Swedish", "sv"), ("Svenska", "sv"), ("swe", "sv"), ("sv", "sv"),
        ("Bulgarian", "bg"), ("български", "bg"), ("bul", "bg"), ("bg", "bg"),
        ("Czech", "cs"), ("čeština", "cs"), ("ces", "cs"), ("cze", "cs"), ("cs", "cs"),
        ("Romanian", "ro"), ("română", "ro"), ("ron", "ro"), ("rum", "ro"), ("ro", "ro"),
        ("Hungarian", "hu"), ("magyar", "hu"), ("hun", "hu"), ("hu", "hu"),
        ("Turkish", "tr"), ("türkçe", "tr"), ("tur", "tr"), ("tr", "tr"),
        ("Greek", "el"), ("ελληνικά", "el"), ("ell", "el"), ("gre", "el"), ("el", "el"),
    ]
    for raw, expected in test_cases:
        assert normalize_language_code(raw) == expected

def test_scandinavian_disambiguation():
    # Danish sentence should not be misclassified as Swedish when expected_language is 'da'
    da_text = "Hej, hvordan har du det i dag? Dette er en test på dansk sprog."
    res_da = detect_language_heuristics(da_text, expected_language="da")
    assert res_da["lang"] == "da"

    # Norwegian sentence should not be misclassified as Swedish when expected_language is 'no'
    no_text = "Hei, hvordan har du det i dag? Dette er en test på norsk språk."
    res_no = detect_language_heuristics(no_text, expected_language="no")
    assert res_no["lang"] == "no"
