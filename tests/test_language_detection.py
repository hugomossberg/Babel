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
        ("Serbian", "sr"), ("српски", "sr"), ("srpski", "sr"), ("srp", "sr"), ("scc", "sr"), ("sr", "sr"),
        ("Croatian", "hr"), ("hrvatski", "hr"), ("hrv", "hr"), ("scr", "hr"), ("hr", "hr"),
        ("Bosnian", "bs"), ("bosanski", "bs"), ("bos", "bs"), ("bs", "bs"),
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

def test_serbian_cyrillic_and_latin_detection():
    # Serbian Cyrillic
    sr_cyr = "Ово је сасвим природна српска реченица коју користимо за тестирање превода и детекције језика у систему."
    res_cyr = detect_language_heuristics(sr_cyr, expected_language="sr")
    assert res_cyr["lang"] == "sr"
    assert res_cyr["confidence"] > 0.8

    # Serbian Latin (detects as hr/bcs family, normalized/compatible)
    sr_lat = "Ovo je sasvim prirodna srpska rečenica koju koristimo za testiranje prevoda i detekcije jezika u sistemu."
    res_lat = detect_language_heuristics(sr_lat, expected_language="sr")
    assert res_lat["lang"] in ["sr", "hr", "bs"]
    assert res_lat["confidence"] > 0.8


def test_polish_english_word_collisions_retains_polish():
    """Issue #2: Polish text with English collision words (a, on, to) must detect as pl with high confidence."""
    polish_text = (
        "To nie jest to, co myslisz. Musimy stad uciekac, zanim oni tu dotra.\n"
        "A co jesli nas znajda? Wtedy bedziemy walczyc. Nie mamy innego wyboru.\n"
        "On powiedzial, ze nigdy nie wolno mi sie poddawac, bez wzgledu na wszystko."
    )
    res_with_expected = detect_language_heuristics(polish_text, expected_language="pl")
    assert res_with_expected["lang"] == "pl"
    assert res_with_expected["confidence"] > 0.90

    res_without_expected = detect_language_heuristics(polish_text)
    assert res_without_expected["lang"] == "pl"
    assert res_without_expected["confidence"] > 0.90


def test_finnish_english_word_collisions_retains_finnish():
    """Issue #2: Finnish text with English collision words (he, on) must detect as fi with high confidence."""
    finnish_text = (
        "En uskonut etta nain kavisi. Sanoin sinulle ettet menisi yksin ulos.\n"
        "Meidan on loydettava keino paasta pois taalta ennen kuin he saapuvat.\n"
        "Mita jos he loytavat meidat? Sitten taistelemme.\n"
        "Meilla ei ole vaihtoehtoa."
    )
    res_with_expected = detect_language_heuristics(finnish_text, expected_language="fi")
    assert res_with_expected["lang"] == "fi"
    assert res_with_expected["confidence"] > 0.90

    res_without_expected = detect_language_heuristics(finnish_text)
    assert res_without_expected["lang"] == "fi"
    assert res_without_expected["confidence"] > 0.90


def test_generic_registered_language_collision_resilience():
    """Any registered language with high confidence detector result must not be overwritten by few English words."""
    # German sentence containing English common words ("in", "so", "and", "on")
    de_text = "Dies ist ein ganz normaler deutscher Text mit ein paar Wörtern wie in and so on."
    res_de = detect_language_heuristics(de_text, expected_language="de")
    assert res_de["lang"] == "de"
    assert res_de["confidence"] > 0.90

    # Italian sentence containing English common words ("in", "me", "come")
    it_text = "Questa è una frase completa in italiano per me e per tutti quelli che vogliono imparare come fare."
    res_it = detect_language_heuristics(it_text, expected_language="it")
    assert res_it["lang"] == "it"
    assert res_it["confidence"] > 0.90


def test_short_english_rescue_unregistered_detector_noise():
    """Short English phrases misidentified by langdetect as unregistered codes (e.g. 'cy', 'af') are rescued to 'en'."""
    # 'What did you do?' raw langdetect returns Welsh ('cy') @ ~1.0 confidence
    res_cy = detect_language_heuristics("What did you do?")
    assert res_cy["lang"] == "en"
    assert res_cy["confidence"] >= 0.95

    # 'Did you get what you want?' raw langdetect returns Afrikaans ('af') @ ~0.85 confidence
    res_af = detect_language_heuristics("Did you get what you want?")
    assert res_af["lang"] == "en"
    assert res_af["confidence"] >= 0.95


def test_serbian_cyrillic_not_vetoed_by_shared_bcs_words():
    """Issue #5: words that exist in BOTH Serbian and Macedonian must not veto the Serbian assist.

    'нема', 'треба', 'овој' and 'ова' are ordinary Serbian. Previously any one of them
    was treated as positive Macedonian evidence, so genuine Serbian Cyrillic containing
    conclusive markers (ђ, ћ, није, шта) was still rejected as 'mk'.
    """
    shared_word_samples = [
        # 'нема' - identical in Serbian and Macedonian
        "Није било лако, али нема шта да се ради. Ђорђе је рекао да ће доћи сутра увече.",
        # 'треба' - identical in Serbian and Macedonian
        "Шта треба да урадимо сада? Ђаво је однео шалу, али ово није крај приче.",
        # 'овој' - Serbian feminine locative of 'овај'
        "У овој кући је увек било топло. Није важно шта други мисле о нама и о ђацима.",
        # 'ова' - ordinary Serbian plural
        "Ова деца су највише уживала. Шта је било, није важно, ђак је ћутао цело вече.",
    ]
    for text in shared_word_samples:
        res = detect_language_heuristics(text, expected_language="sr")
        assert res["lang"] == "sr", f"Serbian text falsely vetoed as {res['lang']}: {text[:40]}"


def test_macedonian_vs_serbian_cyrillic_differentiation():
    """Genuine Macedonian with expected_language='sr' must not be falsely converted to exact Serbian."""
    # Standard Macedonian sample with common markers
    mk_text = "Ова е македонски текст за проверка на јазикот во системот. Здраво, како сте денес? Ова е тест на македонски јазик."
    res_mk = detect_language_heuristics(mk_text, expected_language="sr")
    assert res_mk["lang"] == "mk"

    # Macedonian without common marker words (proving absence of MK words does NOT fabricate Serbian)
    mk_subtle = "Слушам музика во собата додека читам книга со приказни за светот."
    res_subtle = detect_language_heuristics(mk_subtle, expected_language="sr")
    assert res_subtle["lang"] == "mk"

    # Neutral Cyrillic without positive Serbian evidence must NOT be fabricated as sr
    neutral_cyr = "Гледаме филм за историјата на светот со интерес."
    res_neutral = detect_language_heuristics(neutral_cyr, expected_language="sr")
    assert res_neutral["lang"] != "sr"
