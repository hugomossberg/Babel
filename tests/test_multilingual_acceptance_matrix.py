import pytest
import os
import re
import srt
import json
import sqlite3
from datetime import timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from app.core.languages import LANGUAGES, get_language, normalize_language_code
from app.core.validator import (
    detect_language_heuristics,
    classify_cue_language_mismatch,
    check_language_representative,
    evaluate_subtitle_health,
    verify_sync,
    check_dropped_lines
)
from app.services.pipeline import (
    SubtitlePipeline,
    qa_gate,
    QA_STATUS_PASS,
    QA_STATUS_PASS_WITH_WARNINGS,
    QA_STATUS_FAIL
)
from app.core.db import init_db, get_job_by_id, create_job, DB_PATH
from app.services.translator import SubtitleTranslator, get_system_instruction


# ===========================================================================
# 1. TEST MATRIX FOR ALL REGISTERED TARGET LANGUAGES
# ===========================================================================
ALL_LANGUAGE_SAMPLES = {
    "sv": ("Swedish", "swe", "Detta är en fullständigt naturlig svensk mening som vi använder för att verifiera språkdetektering i systemet."),
    "en": ("English", "eng", "This is a completely natural English sentence that we use to verify language detection in the system."),
    "de": ("German", "deu", "Dies ist ein vollkommen natürlicher deutscher Satz, den wir verwenden, um die Spracherkennung im System zu überprüfen."),
    "fr": ("French", "fra", "Ceci est une phrase française tout à fait naturelle que nous utilisons pour vérifier la détection de la langue dans le système."),
    "es": ("Spanish", "spa", "Esta es una frase en español completamente natural que utilizamos para verificar la detección de idioma en el sistema."),
    "it": ("Italian", "ita", "Questa è una frase italiana completamente naturale che utilizziamo per verificare il rilevamento della lingua nel sistema."),
    "nl": ("Dutch", "nld", "Dit is een volkomen natuurlijke Nederlandse zin die we gebruiken om de taaldetectie in het systeem te verifiëren."),
    "pl": ("Polish", "pol", "To jest całkowicie naturalne polskie zdanie, którego używamy do weryfikacji wykrywania języka w systemie."),
    "pt": ("Portuguese", "por", "Esta é uma frase em português completamente natural que usamos para verificar a deteção de idioma no sistema."),
    "ru": ("Russian", "rus", "Это совершенно естественное русское предложение, которое мы используем для проверки определения языка в системе."),
    "ja": ("Japanese", "jpn", "これはシステム内の言語検出を検証するために使用する完全に自然な日本語の文章です。"),
    "zh": ("Chinese", "zho", "这是一个完全自然的中文句子，我们用它来验证系统中的语言检测。"),
    "ko": ("Korean", "kor", "이것은 시스템에서 언어 감지를 확인하기 위해 사용하는 완전히 natural한 한국어 문장입니다."),
    "fi": ("Finnish", "fin", "Tämä on täysin luonnollinen suomenkielinen lause, jota käytämme kielen tunnistuksen tarkistamiseen järjestelmässä."),
    "da": ("Danish", "dan", "Dette er en fuldstændig naturlig dansk sætning, som vi bruger til at verificere sprogregistrering i systemet."),
    "no": ("Norwegian", "nor", "Dette er en fullstendig naturlig norsk setning som vi bruker til å verifisere språkdeteksjon i systemet."),
    "bg": ("Bulgarian", "bul", "Това е напълно естествено изречение на български език, което използваме за проверка на езика в системата."),
    "cs": ("Czech", "ces", "Toto je zcela přirozená česká věta, kterou používáme k ověření detekce jazyka v systému."),
    "ro": ("Romanian", "ron", "Aceasta este o propoziție complet naturală în limba română pe care o folosim pentru a verifica detectarea limbii în sistem."),
    "hu": ("Hungarian", "hun", "Ez egy teljesen természetes magyar mondat, amelyet a rendszer nyelvi felismerésének ellenőrzésére használunk."),
    "tr": ("Turkish", "tur", "Bu, sistemdeki dil algılamasını doğrulamak için kullandığımız tamamen doğal bir Türkçe cümledir."),
    "el": ("Greek", "ell", "Αυτή είναι μια εντελώς φυσική πρόταση στα ελληνικά που χρησιμοποιούμε για την επαλήθευση της ανίχνευσης γλώσσας στο σύστημα."),
    "sr": ("Serbian", "srp", "Ово је сасвим природна српска реченица коју користимо за тестирање превода и детекције језика у систему."),
    "hr": ("Croatian", "hrv", "Ovo je potpuno prirodna hrvatska rečenica koju koristimo za provjeru prijevoda i detekcije jezika u sustavu."),
    "bs": ("Bosnian", "bos", "Ovo je sasvim prirodna bosanska rečenica koju koristimo za provjeru prevoda i detekcije jezika u sistemu.")
}

@pytest.mark.parametrize("code,data", list(ALL_LANGUAGE_SAMPLES.items()))
def test_all_registered_languages_registry_normalization_and_detection(code, data):
    name, alias, sample_text = data
    
    # 1. Registry lookup
    lang_by_code = get_language(code)
    assert lang_by_code is not None, f"Language {code} missing from registry"
    assert lang_by_code.code == code
    assert lang_by_code.display_name == name

    # 2. Normalization
    assert normalize_language_code(code) == code
    assert normalize_language_code(name) == code
    assert normalize_language_code(alias) == code

    # 3. Detection with expected_language
    det = detect_language_heuristics(sample_text, expected_language=code)
    from app.core.validator import are_languages_compatible
    assert are_languages_compatible(det["lang"], code), f"Detection mismatch for {code}: expected {code}, got {det['lang']}"
    assert det["confidence"] >= 0.75, f"Confidence too low for {code}: {det['confidence']}"

    # 4. Stratified representative sample check
    subs = [
        srt.Subtitle(index=i+1, start=timedelta(seconds=i*3), end=timedelta(seconds=i*3+2), content=f"{sample_text} #{i+1}")
        for i in range(15)
    ]
    rep = check_language_representative(subs, target_lang_code=code)
    assert rep["confident_wrong_language"] is False, f"Representative check falsely failed for {code}: {rep}"
    assert are_languages_compatible(rep["detected_lang"], code), f"Representative check detected wrong lang for {code}: {rep['detected_lang']}"


# ===========================================================================
# 2. BULGARIAN END-TO-END REGRESSION & QA TESTS
# ===========================================================================
def test_bulgarian_registry_and_aliases():
    """Verify Bulgarian is in registry with canonical aliases."""
    bg = get_language("bg")
    assert bg is not None
    assert bg.code == "bg"
    assert bg.display_name == "Bulgarian"
    assert bg.deepl_code == "BG"
    assert "bul" in bg.aliases
    assert "bulgarian" in bg.aliases
    assert "български" in bg.aliases

    assert normalize_language_code("Bulgarian") == "bg"
    assert normalize_language_code("bul") == "bg"
    assert normalize_language_code("български") == "bg"
    assert normalize_language_code("BG") == "bg"


def test_bulgarian_qa_accepts_clean_translation_and_rejects_leftover_or_russian():
    """Verify Bulgarian QA passes valid Bulgarian, flags English leftover, and rejects Russian hallucination."""
    source_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=3), content="Hello, how are you doing today, Captain?"),
        srt.Subtitle(index=2, start=timedelta(seconds=4), end=timedelta(seconds=6), content="We are preparing the ship for departure immediately."),
        srt.Subtitle(index=3, start=timedelta(seconds=7), end=timedelta(seconds=9), content="The crew is ready and awaiting your command."),
    ]
    
    # 1. Clean Bulgarian translation -> PASS
    clean_bg_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=3), content="Здравейте, как сте днес, капитане?"),
        srt.Subtitle(index=2, start=timedelta(seconds=4), end=timedelta(seconds=6), content="Подготвяме кораба за отплаване незабавно."),
        srt.Subtitle(index=3, start=timedelta(seconds=7), end=timedelta(seconds=9), content="Екипажът е готов и чака вашата заповед."),
    ]
    res_clean = qa_gate(source_subs, clean_bg_subs, target_lang_code="bg")
    assert res_clean["passed"] is True, f"Clean Bulgarian QA failed: {res_clean}"
    assert res_clean["dropped_count"] == 0
    assert len(res_clean["untranslated_ids"]) == 0

    # 2. English leftover in Bulgarian target -> Real untranslated flagged
    leftover_bg_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=3), content="Здравейте, как сте днес, капитане?"),
        srt.Subtitle(index=2, start=timedelta(seconds=4), end=timedelta(seconds=6), content="We are preparing the ship for departure immediately."),
        srt.Subtitle(index=3, start=timedelta(seconds=7), end=timedelta(seconds=9), content="Екипажът е готов и чака вашата заповед."),
    ]
    res_leftover = qa_gate(source_subs, leftover_bg_subs, target_lang_code="bg", allow_warnings=False)
    assert res_leftover["passed"] is False
    assert 1 in res_leftover["untranslated_ids"] or 1 in res_leftover.get("real_untranslated_ids", [])

    # 3. High-confidence Russian translation instead of Bulgarian -> Confident wrong language mismatch FAIL
    ru_subs = [
        srt.Subtitle(index=i+1, start=timedelta(seconds=i*3), end=timedelta(seconds=i*3+2),
                     content=f"Здравствуйте товарищ генерал, наши солдаты готовы к выполнению боевого приказа номер {i+1}.")
        for i in range(25)
    ]
    src_ru_subs = [
        srt.Subtitle(index=i+1, start=timedelta(seconds=i*3), end=timedelta(seconds=i*3+2),
                     content=f"Hello general, our soldiers are ready to execute the combat order number {i+1}.")
        for i in range(25)
    ]
    rep_ru = check_language_representative(ru_subs, target_lang_code="bg", source_sub_blocks=src_ru_subs)
    assert rep_ru["confident_wrong_language"] is True
    assert rep_ru["detected_lang"] == "ru"


@pytest.mark.asyncio
async def test_bulgarian_full_pipeline_hermetic_e2e(tmp_path):
    """
    Hermetic end-to-end pipeline test for Bulgarian:
    English source -> AI Translation to Bulgarian -> REAL QA gate PASS -> .bg.srt atomic publish -> 0ms sync lock.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.cursor().execute("DELETE FROM jobs")
    conn.commit()
    conn.close()

    with patch("google.genai.Client"):
        pipeline = SubtitlePipeline()

    video_path = tmp_path / "movie.mkv"
    video_path.touch()

    source_cues = [
        "Welcome to our city, we hope you enjoy your stay.",
        "The meeting will begin in ten minutes at the main hall.",
        "Please make sure to bring all required documents.",
        "Everything has been prepared according to the schedule."
    ]
    source_subs = [
        srt.Subtitle(index=i+1, start=timedelta(seconds=i*4), end=timedelta(seconds=i*4+3), content=source_cues[i])
        for i in range(len(source_cues))
    ]
    source_srt_path = str(video_path).replace(".mkv", ".en.srt")
    with open(source_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(source_subs))

    bg_translations = [
        "Добре дошли в нашия град, надяваме се да се насладите на престоя си.",
        "Срещата ще започне след десет минути в главната зала.",
        "Моля, уверете се, че носите всички необходими документи.",
        "Всичко е подготвено според предварителния график."
    ]

    def fake_get_setting(key, default=None):
        if key == "languages":
            return '[{"name": "Bulgarian", "code": "bg", "enabled": true}]'
        if key == "auto_repair_unhealthy": return "false"
        if key == "extract_target_embedded": return "false"
        if key == "extract_source_embedded": return "false"
        if key == "enable_bazarr_check": return "false"
        return default

    async def fake_translate(subs, target_language="Bulgarian", *args, **kwargs):
        return [
            srt.Subtitle(index=sub.index, start=sub.start, end=sub.end, content=bg_translations[sub.index - 1])
            for sub in subs
        ]

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting), \
         patch.object(pipeline, "trigger_bazarr_search"), \
         patch.object(pipeline.translator, "translate_srt_content", side_effect=fake_translate):

        job_id = create_job(str(video_path))
        res = await pipeline._run_pipeline_logic(job_id, str(video_path), wait_seconds=0)

        assert res["status"] in ["success", "translated"]
        job = get_job_by_id(job_id)
        assert job["status"] in ["COMPLETED", "TRANSLATED"]

        # Verify output file existence and naming
        bg_out_path = str(video_path).replace(".mkv", ".bg.srt")
        assert os.path.exists(bg_out_path), f"Bulgarian output file {bg_out_path} was not published"

        with open(bg_out_path, "r", encoding="utf-8") as f:
            out_content = f.read()

        out_subs = list(srt.parse(out_content))
        assert len(out_subs) == len(source_subs)

        # Verify sync integrity (0ms drift)
        sync_report = verify_sync(source_subs, out_subs)
        assert sync_report["valid"] is True
        assert sync_report["start_diff_ms"] == 0
        assert sync_report["end_diff_ms"] == 0

        # Verify dropped lines = 0
        dropped_count, _ = check_dropped_lines(source_subs, out_subs)
        assert dropped_count == 0


# ===========================================================================
# 3. PIPELINE GENERALITY FOR ALL 6 NEW TARGET LANGUAGES + SV
# ===========================================================================
@pytest.mark.asyncio
@pytest.mark.parametrize("target_code,target_name,sample_translation", [
    ("sv", "Swedish", "Välkommen till vår stad, vi hoppas att du trivs."),
    ("bg", "Bulgarian", "Добре дошли в нашия град, надяваме се да се насладите на престоя си."),
    ("cs", "Czech", "Vítejte v našem městě, doufáme, že se vám pobyt bude líbit."),
    ("ro", "Romanian", "Bun venit în orașul nostru, sperăm să vă bucurați de ședere."),
    ("hu", "Hungarian", "Üdvözöljük városunkban, reméljük, kellemesen fogja tölteni az idejét."),
    ("tr", "Turkish", "Şehrimize hoş geldiniz, umarız konaklamanızdan keyif alırsınız."),
    ("el", "Greek", "Καλώς ήρθατε στην πόλη μας, ελπίζουμε να απολαύσετε τη διαμονή σας.")
])
async def test_full_pipeline_across_new_languages(tmp_path, target_code, target_name, sample_translation):
    """Verify entire pipeline (Source -> Translate -> REAL QA -> Publish .<lang>.srt) works for new languages."""
    init_db()
    with patch("google.genai.Client"):
        pipeline = SubtitlePipeline()

    video_path = tmp_path / f"video_{target_code}.mkv"
    video_path.touch()

    source_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=4), content="Welcome to our city, we hope you enjoy your stay."),
        srt.Subtitle(index=2, start=timedelta(seconds=5), end=timedelta(seconds=8), content="Please contact our support desk if you need any assistance.")
    ]
    source_srt_path = str(video_path).replace(f"_{target_code}.mkv", f"_{target_code}.en.srt")
    with open(source_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(source_subs))

    def fake_get_setting(key, default=None):
        if key == "languages":
            return json.dumps([{"name": target_name, "code": target_code, "enabled": True}])
        if key == "auto_repair_unhealthy": return "false"
        if key == "extract_target_embedded": return "false"
        if key == "extract_source_embedded": return "false"
        if key == "enable_bazarr_check": return "false"
        return default

    async def fake_translate(subs, target_language=target_name, *args, **kwargs):
        return [
            srt.Subtitle(index=1, start=subs[0].start, end=subs[0].end, content=sample_translation),
            srt.Subtitle(index=2, start=subs[1].start, end=subs[1].end, content=f"{sample_translation} (2)")
        ]

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting), \
         patch.object(pipeline, "trigger_bazarr_search"), \
         patch.object(pipeline.translator, "translate_srt_content", side_effect=fake_translate):

        job_id = create_job(str(video_path))
        res = await pipeline._run_pipeline_logic(job_id, str(video_path), wait_seconds=0)

        assert res["status"] in ["success", "translated"]
        out_path = str(video_path).replace(f"_{target_code}.mkv", f"_{target_code}.{target_code}.srt")
        assert os.path.exists(out_path), f"Output subtitle {out_path} was not created for {target_code}"


# ===========================================================================
# 4. MULTI-TARGET TESTS: (A) REAL QA ALL-SUCCESS & (B) PARTIAL ISOLATION
# ===========================================================================
@pytest.mark.asyncio
async def test_real_multi_target_qa_all_success_sv_bg_cs(tmp_path):
    """
    1. REAL MULTI-TARGET QA TEST (NO MOCKED QA GATE):
    English source -> targets: sv, bg, cs.
    All 3 receive natural, high-quality target translations.
    Verify:
    - Real QA gate executes for all 3 targets and cleanly PASSES
    - Output files movie.sv.srt, movie.bg.srt, movie.cs.srt all created atomically
    - Cue count matches source exactly (no dropped cues)
    - 0ms timestamp drift across all 3
    - No cross-target language state contamination or temp collisions
    - Final job status is TRANSLATED with target_languages='sv,bg,cs'
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.cursor().execute("DELETE FROM jobs")
    conn.commit()
    conn.close()

    with patch("google.genai.Client"):
        pipeline = SubtitlePipeline()

    video_path = tmp_path / "movie.mkv"
    video_path.touch()

    source_cues = [
        "Welcome everyone to tonight's special presentation.",
        "We are glad to have all our international guests with us.",
        "The conference will start in ten minutes in the grand auditorium.",
        "Please silence all mobile devices before the program begins."
    ]
    source_subs = [
        srt.Subtitle(index=i+1, start=timedelta(seconds=i*5), end=timedelta(seconds=i*5+4), content=source_cues[i])
        for i in range(len(source_cues))
    ]
    source_srt_path = str(video_path).replace(".mkv", ".en.srt")
    with open(source_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(source_subs))

    target_translations = {
        "Swedish": [
            "Välkomna alla till kvällens specialpresentation.",
            "Vi är glada att ha alla våra internationella gäster hos oss.",
            "Konferensen börjar om tio minuter i den stora auditoriet.",
            "Vänligen stäng av alla mobila enheter innan programmet börjar."
        ],
        "Bulgarian": [
            "Добре дошли на всички на тазвечершното специално представяне.",
            "Радваме се, че всички наши международни гости са с нас.",
            "Конференцията ще започне след десет минути в голямата зала.",
            "Моля, изключете звука на всички мобилни устройства преди началото."
        ],
        "Czech": [
            "Vítejte všichni na dnešní speciální prezentaci.",
            "Jsme rádi, že máme všechny naše mezinárodní hosty s námi.",
            "Konference začne za deset minut ve velkém auditoriu.",
            "Před začátkem programu prosím ztište všechna mobilní zařízení."
        ]
    }

    def fake_get_setting(key, default=None):
        if key == "languages":
            return json.dumps([
                {"name": "Swedish", "code": "sv", "enabled": True},
                {"name": "Bulgarian", "code": "bg", "enabled": True},
                {"name": "Czech", "code": "cs", "enabled": True}
            ])
        if key == "auto_repair_unhealthy": return "false"
        if key == "extract_target_embedded": return "false"
        if key == "extract_source_embedded": return "false"
        if key == "enable_bazarr_check": return "false"
        return default

    async def fake_translate(subs, target_language="Swedish", *args, **kwargs):
        lines = target_translations.get(target_language)
        if not lines:
            lang_obj = get_language(target_language)
            if lang_obj:
                lines = target_translations.get(lang_obj.display_name, [])
        return [
            srt.Subtitle(index=sub.index, start=sub.start, end=sub.end, content=lines[sub.index - 1])
            for sub in subs
        ]

    # Notice: qa_gate is NOT mocked here! Real QA gate evaluates each target.
    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting), \
         patch.object(pipeline, "trigger_bazarr_search"), \
         patch.object(pipeline.translator, "translate_srt_content", side_effect=fake_translate):

        job_id = create_job(str(video_path))
        res = await pipeline._run_pipeline_logic(job_id, str(video_path), wait_seconds=0)

        assert res["status"] in ["success", "translated"]
        job = get_job_by_id(job_id)
        assert job["status"] in ["COMPLETED", "TRANSLATED"]
        assert "sv" in job["target_languages"]
        assert "bg" in job["target_languages"]
        assert "cs" in job["target_languages"]

        # Verify all 3 output files exist
        sv_file = str(video_path).replace(".mkv", ".sv.srt")
        bg_file = str(video_path).replace(".mkv", ".bg.srt")
        cs_file = str(video_path).replace(".mkv", ".cs.srt")

        for p, code, expected_kw in [
            (sv_file, "sv", "Välkomna alla"),
            (bg_file, "bg", "Добре дошли на всички"),
            (cs_file, "cs", "Vítejte všichni")
        ]:
            assert os.path.exists(p), f"Target output {p} missing"
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
            assert expected_kw in content
            parsed = list(srt.parse(content))
            assert len(parsed) == len(source_subs)

            # Strict 0ms sync verification
            sync_res = verify_sync(source_subs, parsed)
            assert sync_res["valid"] is True
            assert sync_res["start_diff_ms"] == 0
            assert sync_res["end_diff_ms"] == 0

            # Dropped lines check
            dropped, _ = check_dropped_lines(source_subs, parsed)
            assert dropped == 0


@pytest.mark.asyncio
async def test_multi_target_isolation_sv_bg_cs_partial_failure(tmp_path):
    """
    Multi-target isolation test where one language fails QA:
    - Swedish and Bulgarian succeed
    - Czech fails QA
    - Verifies that Swedish and Bulgarian are still published while Czech is blocked,
      and final status is PARTIAL.
    """
    init_db()
    with patch("google.genai.Client"):
        pipeline = SubtitlePipeline()

    video_path = tmp_path / "multitarget_show.mkv"
    video_path.touch()

    source_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=4), content="This is an important multi-target test sentence."),
        srt.Subtitle(index=2, start=timedelta(seconds=5), end=timedelta(seconds=8), content="All languages must be processed cleanly and independently.")
    ]
    source_srt_path = str(video_path).replace(".mkv", ".en.srt")
    with open(source_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(source_subs))

    def fake_get_setting(key, default=None):
        if key == "languages":
            return json.dumps([
                {"name": "Swedish", "code": "sv", "enabled": True},
                {"name": "Bulgarian", "code": "bg", "enabled": True},
                {"name": "Czech", "code": "cs", "enabled": True}
            ])
        if key == "auto_repair_unhealthy": return "false"
        if key == "extract_target_embedded": return "false"
        if key == "extract_source_embedded": return "false"
        if key == "enable_bazarr_check": return "false"
        return default

    async def fake_translate(subs, target_language="Swedish", *args, **kwargs):
        if target_language in ["Swedish", "sv"]:
            return [
                srt.Subtitle(1, subs[0].start, subs[0].end, "Detta är en viktig mening för test."),
                srt.Subtitle(2, subs[1].start, subs[1].end, "Alla språk måste behandlas rent och oberoende.")
            ]
        elif target_language in ["Bulgarian", "bg"]:
            return [
                srt.Subtitle(1, subs[0].start, subs[0].end, "Това е важно изречение за тест."),
                srt.Subtitle(2, subs[1].start, subs[1].end, "Всички езици трябва да се обработват чисто и независимо.")
            ]
        else: # Czech
            return [
                srt.Subtitle(1, subs[0].start, subs[0].end, "Bad translation"),
                srt.Subtitle(2, subs[1].start, subs[1].end, "Bad translation 2")
            ]

    def fake_qa_gate(*args, **kwargs):
        lang = kwargs.get("target_lang_code", "")
        if lang in ["sv", "bg"]:
            return {"passed": True, "status": "PASS", "score": 100, "issues": [], "dropped_count": 0, "untranslated_ids": [], "real_untranslated_ids": [], "dropped_details": [], "sync_diff_ms": 0}
        return {"passed": False, "status": "FAIL", "score": 20, "issues": ["Language mismatch"], "dropped_count": 0, "untranslated_ids": [0, 1], "real_untranslated_ids": [0, 1], "wrong_language_ids": [0, 1], "dropped_details": [], "sync_diff_ms": 0}

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting), \
         patch.object(pipeline, "trigger_bazarr_search"), \
         patch.object(pipeline.translator, "translate_srt_content", side_effect=fake_translate), \
         patch.object(pipeline.translator, "escalate_single_line", return_value=None), \
         patch.object(pipeline.translator, "classify_and_recover_identical", return_value=[]), \
         patch.object(pipeline.translator, "translate_batch", return_value=[]), \
         patch("app.services.pipeline.qa_gate", side_effect=fake_qa_gate):

        job_id = create_job(str(video_path))
        res = await pipeline._run_pipeline_logic(job_id, str(video_path), wait_seconds=0)

        assert res["status"] == "partial"
        job = get_job_by_id(job_id)
        assert job["status"] == "PARTIAL"

        sv_file = str(video_path).replace(".mkv", ".sv.srt")
        bg_file = str(video_path).replace(".mkv", ".bg.srt")
        cs_file = str(video_path).replace(".mkv", ".cs.srt")

        assert os.path.exists(sv_file), "Swedish file should exist"
        assert os.path.exists(bg_file), "Bulgarian file should exist"
        assert not os.path.exists(cs_file), "Czech file should not be published on QA fail"


# ===========================================================================
# 5. SCANDINAVIAN REGRESSION TESTS (sv, no, da)
# ===========================================================================
def test_scandinavian_overlapping_words_disambiguation():
    """
    Ensure overlapping Scandinavian words ('og', 'en', 'vi', 'skal', 'gå', 'hjem', 'nå', 'nu')
    are never hijacked to Swedish when expected_language is 'no' or 'da'.
    """
    no_text = "Dette er en viktig norsk setning og vi skal gå hjem nå."
    da_text = "Dette er en vigtig dansk sætning og vi skal gå hjem nu."
    sv_text = "Detta är en viktig svensk mening och vi ska gå hem nu."

    # When expected=no, Norwegian MUST remain 'no'
    res_no = detect_language_heuristics(no_text, expected_language="no")
    assert res_no["lang"] == "no", f"Expected 'no', got {res_no['lang']}"

    # When expected=da, Danish MUST remain 'da'
    res_da = detect_language_heuristics(da_text, expected_language="da")
    assert res_da["lang"] == "da", f"Expected 'da', got {res_da['lang']}"

    # When expected=sv, Swedish MUST remain 'sv'
    res_sv = detect_language_heuristics(sv_text, expected_language="sv")
    assert res_sv["lang"] == "sv", f"Expected 'sv', got {res_sv['lang']}"

    # Norwegian representative check with expected 'no'
    no_subs = [srt.Subtitle(i+1, timedelta(seconds=i*2), timedelta(seconds=i*2+1), f"{no_text} {i}") for i in range(10)]
    rep_no = check_language_representative(no_subs, target_lang_code="no")
    assert rep_no["confident_wrong_language"] is False
    assert rep_no["detected_lang"] == "no"

    # Danish representative check with expected 'da'
    da_subs = [srt.Subtitle(i+1, timedelta(seconds=i*2), timedelta(seconds=i*2+1), f"{da_text} {i}") for i in range(10)]
    rep_da = check_language_representative(da_subs, target_lang_code="da")
    assert rep_da["confident_wrong_language"] is False
    assert rep_da["detected_lang"] == "da"


# ===========================================================================
# 6. PROVIDER DISPATCH VERIFICATION (Gemini, OpenAI, DeepL, Ollama)
# ===========================================================================
@pytest.mark.asyncio
async def test_provider_dispatch_gemini():
    """Verify Gemini provider dispatch uses configured model and target language in system prompt."""
    translator = SubtitleTranslator()
    
    def fake_settings(key, default=None):
        if key == "ai_provider": return "gemini"
        if key == "gemini_model": return "gemini-3.5-flash-lite"
        if key == "gemini_api_key": return "dummy-gemini-key"
        return default

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({"translations": [{"id": 1, "text": "Здравейте"}]})
    mock_client.models.generate_content.return_value = mock_response

    with patch("app.services.translator.get_setting", side_effect=fake_settings), \
         patch.object(translator, "get_gemini_client", return_value=mock_client):
        
        batch = [{"id": 1, "text": "Hello"}]
        res = await translator.translate_batch(
            batch,
            target_language="Bulgarian",
            show_title="Test Show"
        )
        
        assert len(res) == 1
        assert res[0]["text"] == "Здравейте"
        
        # Verify SDK client call arguments
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-3.5-flash-lite"
        assert "Bulgarian" in call_kwargs["config"].system_instruction


@pytest.mark.asyncio
@pytest.mark.parametrize("target_lang,expected_keyword", [
    ("Bulgarian", "Bulgarian"),
    ("Czech", "Czech")
])
async def test_provider_dispatch_openai(target_lang, expected_keyword):
    """Verify OpenAI provider dispatch routes with correct model and target language."""
    translator = SubtitleTranslator()

    def fake_settings(key, default=None):
        if key == "ai_provider": return "openai"
        if key == "openai_model": return "gpt-4o-mini"
        if key == "openai_api_key": return "dummy-openai-key"
        return default

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({"translations": [{"id": 1, "text": "Test"}]})
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    with patch("app.services.translator.get_setting", side_effect=fake_settings), \
         patch.object(translator, "get_openai_client", return_value=mock_client):
        
        batch = [{"id": 1, "text": "Hello"}]
        res = await translator.translate_batch(
            batch,
            target_language=target_lang,
            show_title="Test Show"
        )
        
        assert len(res) == 1
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        messages = call_kwargs["messages"]
        system_msg = next(m["content"] for m in messages if m["role"] == "system")
        assert f"English to {expected_keyword}" in system_msg


@pytest.mark.asyncio
@pytest.mark.parametrize("target_lang,expected_deepl_code", [
    ("Bulgarian", "BG"),
    ("Norwegian", "NB"),
    ("Swedish", "SV"),
    ("Czech", "CS"),
    ("Greek", "EL"),
    ("Portuguese", "PT-PT"),
    ("Serbian", "SR"),
    ("Croatian", "HR"),
    ("Bosnian", "BS")
])
async def test_provider_dispatch_deepl(target_lang, expected_deepl_code):
    """Verify DeepL provider maps target languages to canonical DeepL codes via registry without hardcoding."""
    translator = SubtitleTranslator()

    def fake_settings(key, default=None):
        if key == "ai_provider": return "deepl"
        if key == "deepl_api_key": return "dummy-deepl-key"
        return default

    mock_post = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"translations": [{"text": "Oversatt tekst"}]}
    mock_post.return_value = mock_resp

    with patch("app.services.translator.get_setting", side_effect=fake_settings), \
         patch("httpx.AsyncClient.post", mock_post):
        
        batch = [{"id": 1, "text": "Hello"}]
        res = await translator.translate_batch(
            batch,
            target_language=target_lang
        )

        assert len(res) == 1
        assert res[0]["text"] == "Oversatt tekst"
        
        # Verify target_lang in DeepL payload matches canonical registry code
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["json"]["target_lang"] == expected_deepl_code
        assert call_kwargs["json"]["source_lang"] == "EN"


@pytest.mark.asyncio
async def test_provider_dispatch_ollama():
    """Verify Ollama provider dispatch routes correctly with target language, custom url and model."""
    translator = SubtitleTranslator()

    def fake_settings(key, default=None):
        if key == "ai_provider": return "ollama"
        if key == "ollama_url": return "http://ollama-host:11434"
        if key == "ollama_model": return "llama3.2:latest"
        return default

    mock_post = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": json.dumps({"translations": [{"id": 1, "text": "Översatt"}]})}
    mock_post.return_value = mock_resp

    with patch("app.services.translator.get_setting", side_effect=fake_settings), \
         patch("httpx.AsyncClient.post", mock_post):
        
        batch = [{"id": 1, "text": "Hello"}]
        res = await translator.translate_batch(
            batch,
            target_language="Swedish",
            show_title="Test Show"
        )

        assert len(res) == 1
        mock_post.assert_called_once()
        url = mock_post.call_args.args[0] if mock_post.call_args.args else mock_post.call_args.kwargs.get("url")
        assert "http://ollama-host:11434" in url
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "llama3.2:latest"
        assert "English to Swedish" in payload["prompt"]


# ===========================================================================
# 7. QA STRICTNESS IN MULTI-TARGET CONTEXT
# ===========================================================================
def test_qa_strictness_high_confidence_wrong_language_fails():
    """High-confidence wrong language translation must strictly FAIL QA."""
    src = [srt.Subtitle(i+1, timedelta(seconds=i*3), timedelta(seconds=i*3+2), f"This is an English line number {i+1} for testing.") for i in range(20)]
    de_target = [srt.Subtitle(i+1, timedelta(seconds=i*3), timedelta(seconds=i*3+2), f"Das ist eine deutsche Zeile Nummer {i+1} für diesen Test.") for i in range(20)]
    
    qa = qa_gate(src, de_target, target_lang_code="sv")
    assert qa["passed"] is False, "High confidence German instead of Swedish must fail QA"
    assert any("language mismatch" in str(issue).lower() or "wrong_language" in str(issue).lower() for issue in qa.get("issues", []))


def test_qa_strictness_short_ambiguous_does_not_falsely_fail():
    """Short ambiguous names/dialogues must not falsely fail with confident wrong language."""
    src = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Yes."),
        srt.Subtitle(2, timedelta(seconds=3), timedelta(seconds=4), "Okay."),
        srt.Subtitle(3, timedelta(seconds=5), timedelta(seconds=6), "Hi.")
    ]
    target = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Ja."),
        srt.Subtitle(2, timedelta(seconds=3), timedelta(seconds=4), "Okej."),
        srt.Subtitle(3, timedelta(seconds=5), timedelta(seconds=6), "Hej.")
    ]
    qa = qa_gate(src, target, target_lang_code="sv")
    assert qa["passed"] is True, f"Short valid dialogues should pass QA, got {qa}"


def test_qa_strictness_dropped_cues_strictly_fails():
    """Dropped cues must strictly fail QA across any target language."""
    src = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Hello."),
        srt.Subtitle(2, timedelta(seconds=3), timedelta(seconds=4), "How are you?"),
        srt.Subtitle(3, timedelta(seconds=5), timedelta(seconds=6), "Goodbye.")
    ]
    target_bg = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Здравейте."),
        srt.Subtitle(2, timedelta(seconds=5), timedelta(seconds=6), "Довиждане.")
    ]
    qa = qa_gate(src, target_bg, target_lang_code="bg")
    assert qa["passed"] is False, "Dropped cue must strictly fail QA"
    assert qa["dropped_count"] >= 1


# ===========================================================================
# 8. SERBIAN, CROATIAN & BOSNIAN QA SUITE
# ===========================================================================
def test_bcs_languages_qa_gate_passes():
    """BCS translations (sr Cyrillic/Latin, hr, bs) must pass QA without detector confusion."""
    src = [
        srt.Subtitle(i+1, timedelta(seconds=i*3), timedelta(seconds=i*3+2), f"This is an important English line number {i+1} for testing.")
        for i in range(15)
    ]

    # Serbian Cyrillic
    sr_cyr_target = [
        srt.Subtitle(i+1, timedelta(seconds=i*3), timedelta(seconds=i*3+2), f"Ово је важна српска реченица број {i+1} за тестирање превода.")
        for i in range(15)
    ]
    qa_sr_cyr = qa_gate(src, sr_cyr_target, target_lang_code="sr")
    assert qa_sr_cyr["passed"] is True, f"Serbian Cyrillic failed QA: {qa_sr_cyr}"

    # Croatian
    hr_target = [
        srt.Subtitle(i+1, timedelta(seconds=i*3), timedelta(seconds=i*3+2), f"Ovo je važna hrvatska rečenica broj {i+1} za provjeru prijevoda.")
        for i in range(15)
    ]
    qa_hr = qa_gate(src, hr_target, target_lang_code="hr")
    assert qa_hr["passed"] is True, f"Croatian failed QA: {qa_hr}"

    # Bosnian
    bs_target = [
        srt.Subtitle(i+1, timedelta(seconds=i*3), timedelta(seconds=i*3+2), f"Ovo je važna bosanska rečenica broj {i+1} za provjeru prevoda.")
        for i in range(15)
    ]
    qa_bs = qa_gate(src, bs_target, target_lang_code="bs")
    assert qa_bs["passed"] is True, f"Bosnian failed QA: {qa_bs}"


def test_bcs_languages_qa_gate_fails_on_unrelated_language():
    """BCS target languages must strictly fail QA when translated into an unrelated language like German or English."""
    src = [
        srt.Subtitle(i+1, timedelta(seconds=i*3), timedelta(seconds=i*3+2), f"This is an English line number {i+1} for testing.")
        for i in range(20)
    ]
    de_target = [
        srt.Subtitle(i+1, timedelta(seconds=i*3), timedelta(seconds=i*3+2), f"Das ist eine deutsche Zeile Nummer {i+1} für diesen Test.")
        for i in range(20)
    ]
    en_untranslated_target = [
        srt.Subtitle(i+1, timedelta(seconds=i*3), timedelta(seconds=i*3+2), f"This is an English line number {i+1} for testing.")
        for i in range(20)
    ]
    for bcs_lang in ["sr", "hr", "bs"]:
        qa_de = qa_gate(src, de_target, target_lang_code=bcs_lang)
        assert qa_de["passed"] is False, f"German translation for {bcs_lang} target must strictly fail QA"

        qa_en = qa_gate(src, en_untranslated_target, target_lang_code=bcs_lang)
        assert qa_en["passed"] is False, f"Untranslated English for {bcs_lang} target must strictly fail QA"


def test_polish_qa_gate_passes_and_rejects_untranslated_english(tmp_path):
    """Issue #2: Realistic Polish subtitle must pass QA gate and subtitle health, but untranslated English must fail."""
    src = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=4), "This is not what you think. We have to run away before they arrive."),
        srt.Subtitle(2, timedelta(seconds=5), timedelta(seconds=8), "And what if they find us? Then we will fight. We have no other choice."),
        srt.Subtitle(3, timedelta(seconds=9), timedelta(seconds=12), "He said that I should never give up, no matter what happens."),
        srt.Subtitle(4, timedelta(seconds=13), timedelta(seconds=16), "We need to prepare everything carefully for tomorrow morning."),
        srt.Subtitle(5, timedelta(seconds=17), timedelta(seconds=20), "Everything will be fine as long as we stay together here.")
    ]
    pl_cues = [
        "To nie jest to, co myslisz. Musimy stad uciekac, zanim oni tu dotra.",
        "A co jesli nas znajda? Wtedy bedziemy walczyc. Nie mamy innego wyboru.",
        "On powiedzial, ze nigdy nie wolno mi sie poddawac, bez wzgledu na wszystko.",
        "Musimy przygotowac wszystko bardzo uwaznie na jutrzejszy poranek.",
        "Wszystko bedzie dobrze, dopoki jestesmy tutaj razem."
    ]
    pl_target = [
        srt.Subtitle(i+1, src[i].start, src[i].end, pl_cues[i])
        for i in range(len(pl_cues))
    ]

    qa_pl = qa_gate(src, pl_target, target_lang_code="pl")
    assert qa_pl["passed"] is True, f"Polish translation unexpectedly failed QA: {qa_pl}"

    pl_file = tmp_path / "polish.pl.srt"
    with open(pl_file, "w", encoding="utf-8") as f:
        f.write(srt.compose(pl_target))
    health_pl = evaluate_subtitle_health(str(pl_file), target_lang_code="pl")
    assert health_pl["status"] == "GREEN", f"Polish health score unexpected: {health_pl}"

    # English unchanged for target pl must fail QA
    qa_fail = qa_gate(src, src, target_lang_code="pl")
    assert qa_fail["passed"] is False, "Untranslated English must strictly fail QA when target is Polish"


def test_finnish_qa_gate_passes_and_rejects_untranslated_english(tmp_path):
    """Issue #2: Realistic Finnish subtitle must pass QA gate and subtitle health, but untranslated English must fail."""
    src = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=4), "I did not think this would happen. I told you not to go out alone."),
        srt.Subtitle(2, timedelta(seconds=5), timedelta(seconds=8), "We must find a way to get out of here before they arrive."),
        srt.Subtitle(3, timedelta(seconds=9), timedelta(seconds=12), "What if they find us? Then we fight. We have no other choice."),
        srt.Subtitle(4, timedelta(seconds=13), timedelta(seconds=16), "We will stay together until the morning comes."),
        srt.Subtitle(5, timedelta(seconds=17), timedelta(seconds=20), "Everything is going to work out fine in the end.")
    ]
    fi_cues = [
        "En uskonut etta nain kavisi. Sanoin sinulle ettet menisi yksin ulos.",
        "Meidan on loydettava keino paasta pois taalta ennen kuin he saapuvat.",
        "Mita jos he loytavat meidat? Sitten taistelemme. Meilla ei ole vaihtoehtoa.",
        "Pysymme yhdessa aina siihen asti kunnes aamu saapuu.",
        "Kaikki tulee jarjestymaan aivan hyvin loppujen lopuksi."
    ]
    fi_target = [
        srt.Subtitle(i+1, src[i].start, src[i].end, fi_cues[i])
        for i in range(len(fi_cues))
    ]

    qa_fi = qa_gate(src, fi_target, target_lang_code="fi")
    assert qa_fi["passed"] is True, f"Finnish translation unexpectedly failed QA: {qa_fi}"

    fi_file = tmp_path / "finnish.fi.srt"
    with open(fi_file, "w", encoding="utf-8") as f:
        f.write(srt.compose(fi_target))
    health_fi = evaluate_subtitle_health(str(fi_file), target_lang_code="fi")
    assert health_fi["status"] == "GREEN", f"Finnish health score unexpected: {health_fi}"

    # English unchanged for target fi must fail QA
    qa_fail = qa_gate(src, src, target_lang_code="fi")
    assert qa_fail["passed"] is False, "Untranslated English must strictly fail QA when target is Finnish"


# ===========================================================================
# 9. BCS FULL PIPELINE HERMETIC E2E (SR, HR, BS & MULTI-TARGET)
# ===========================================================================
@pytest.mark.asyncio
@pytest.mark.parametrize("target_code,target_name,translations", [
    ("sr", "Serbian", [
        "Ово је сасвим природна српска реченица број један за тестирање превода.",
        "Морамо пронаћи решење за овај проблем пре него што почне састанак.",
        "Хвала вам пуно што сте дошли данас да нам помогнете у раду.",
        "Све је спремно према плану и можемо одмах почети."
    ]),
    ("hr", "Croatian", [
        "Ovo je potpuno prirodna hrvatska rečenica broj jedan za provjeru prijevoda.",
        "Moramo pronaći rješenje za ovaj problem prije nego što počne sastanak.",
        "Hvala vam puno što ste došli danas kako biste nam pomogli u radu.",
        "Sve je spremno prema planu i možemo odmah započeti."
    ]),
    ("bs", "Bosnian", [
        "Ovo je sasvim prirodna bosanska rečenica broj jedan za provjeru prevoda.",
        "Moramo pronaći rješenje za ovaj problem prije nego što počne sastanak.",
        "Hvala vam puno što ste došli danas kako biste nam pomogli u radu.",
        "Sve je spremno prema planu i možemo odmah početi."
    ])
])
async def test_bcs_individual_full_pipeline_hermetic_e2e(tmp_path, target_code, target_name, translations):
    """
    Hermetic E2E test for each BCS target individually:
    English source -> AI Translation -> REAL QA gate PASS -> .<lang>.srt atomic publish -> 0ms sync lock.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.cursor().execute("DELETE FROM jobs")
    conn.commit()
    conn.close()

    with patch("google.genai.Client"):
        pipeline = SubtitlePipeline()

    video_path = tmp_path / f"bcs_video_{target_code}.mkv"
    video_path.touch()

    source_cues = [
        "This is an important English sentence number one for testing translation.",
        "We must find a solution to this problem before the meeting begins.",
        "Thank you very much for coming today to help us with the work.",
        "Everything is ready according to the plan and we can start immediately."
    ]
    source_subs = [
        srt.Subtitle(index=i+1, start=timedelta(seconds=i*4), end=timedelta(seconds=i*4+3), content=source_cues[i])
        for i in range(len(source_cues))
    ]
    source_srt_path = str(video_path).replace(f"_{target_code}.mkv", f"_{target_code}.en.srt")
    with open(source_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(source_subs))

    def fake_get_setting(key, default=None):
        if key == "languages":
            return json.dumps([{"name": target_name, "code": target_code, "enabled": True}])
        if key == "auto_repair_unhealthy": return "false"
        if key == "extract_target_embedded": return "false"
        if key == "extract_source_embedded": return "false"
        if key == "enable_bazarr_check": return "false"
        return default

    async def fake_translate(subs, target_language=target_name, *args, **kwargs):
        return [
            srt.Subtitle(index=sub.index, start=sub.start, end=sub.end, content=translations[sub.index - 1])
            for sub in subs
        ]

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting), \
         patch.object(pipeline, "trigger_bazarr_search"), \
         patch.object(pipeline.translator, "translate_srt_content", side_effect=fake_translate):

        job_id = create_job(str(video_path))
        res = await pipeline._run_pipeline_logic(job_id, str(video_path), wait_seconds=0)

        assert res["status"] in ["success", "translated"]
        job = get_job_by_id(job_id)
        assert job["status"] in ["COMPLETED", "TRANSLATED"]

        out_path = str(video_path).replace(f"_{target_code}.mkv", f"_{target_code}.{target_code}.srt")
        assert os.path.exists(out_path), f"Output file {out_path} was not published"

        with open(out_path, "r", encoding="utf-8") as f:
            out_content = f.read()

        out_subs = list(srt.parse(out_content))
        assert len(out_subs) == len(source_subs)

        # 0ms sync lock & 0 dropped cues
        sync_report = verify_sync(source_subs, out_subs)
        assert sync_report["valid"] is True
        assert sync_report["start_diff_ms"] == 0
        assert sync_report["end_diff_ms"] == 0

        dropped_count, _ = check_dropped_lines(source_subs, out_subs)
        assert dropped_count == 0


@pytest.mark.asyncio
async def test_bcs_multi_target_real_qa_all_success_sr_hr_bs(tmp_path):
    """
    REAL multi-target E2E pipeline for sr + hr + bs concurrently:
    - Real QA gate executes for all 3 targets and cleanly PASSES
    - Output files movie.sr.srt, movie.hr.srt, movie.bs.srt all created atomically
    - Cue count matches source exactly (0 dropped cues)
    - 0ms timestamp drift across all 3
    - No cross-target language state contamination or temp collisions
    - Final job status is TRANSLATED with target_languages='sr,hr,bs'
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.cursor().execute("DELETE FROM jobs")
    conn.commit()
    conn.close()

    with patch("google.genai.Client"):
        pipeline = SubtitlePipeline()

    video_path = tmp_path / "bcs_multi_movie.mkv"
    video_path.touch()

    source_cues = [
        "Welcome everyone to tonight's special presentation.",
        "We are glad to have all our international guests with us.",
        "The conference will start in ten minutes in the grand auditorium.",
        "Please silence all mobile devices before the program begins."
    ]
    source_subs = [
        srt.Subtitle(index=i+1, start=timedelta(seconds=i*5), end=timedelta(seconds=i*5+4), content=source_cues[i])
        for i in range(len(source_cues))
    ]
    source_srt_path = str(video_path).replace(".mkv", ".en.srt")
    with open(source_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(source_subs))

    target_translations = {
        "Serbian": [
            "Добродошли сви на вечерашњу специјалну презентацију.",
            "Драго нам је што су сви наши међународни гости са нама.",
            "Конференција ће почети за десет минута у великој сали.",
            "Молимо вас да утишате све мобилне уређаје пре почетка."
        ],
        "Croatian": [
            "Dobrodošli svi na večerašnju posebnu prezentaciju.",
            "Drago nam je što su svi naši međunarodni gosti s nama.",
            "Konferencija će započeti za deset minuta u velikoj dvorani.",
            "Molimo vas da utišate sve mobilne uređaje prije početka."
        ],
        "Bosnian": [
            "Dobrodošli svi na večerašnju posebnu prezentaciju.",
            "Drago nam je što su svi naši međunarodni gosti sa nama.",
            "Konferencija će početi za deset minuta u velikoj dvorani.",
            "Molimo vas da utišate sve mobilne uređaje prije početka."
        ]
    }

    def fake_get_setting(key, default=None):
        if key == "languages":
            return json.dumps([
                {"name": "Serbian", "code": "sr", "enabled": True},
                {"name": "Croatian", "code": "hr", "enabled": True},
                {"name": "Bosnian", "code": "bs", "enabled": True}
            ])
        if key == "auto_repair_unhealthy": return "false"
        if key == "extract_target_embedded": return "false"
        if key == "extract_source_embedded": return "false"
        if key == "enable_bazarr_check": return "false"
        return default

    async def fake_translate(subs, target_language="Serbian", *args, **kwargs):
        trans_list = target_translations.get(target_language, [])
        return [
            srt.Subtitle(index=sub.index, start=sub.start, end=sub.end, content=trans_list[sub.index - 1])
            for sub in subs
        ]

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting), \
         patch.object(pipeline, "trigger_bazarr_search"), \
         patch.object(pipeline.translator, "translate_srt_content", side_effect=fake_translate):

        job_id = create_job(str(video_path))
        res = await pipeline._run_pipeline_logic(job_id, str(video_path), wait_seconds=0)

        assert res["status"] in ["success", "translated"]
        job = get_job_by_id(job_id)
        assert job["status"] in ["COMPLETED", "TRANSLATED"]

        for code in ["sr", "hr", "bs"]:
            out_file = str(video_path).replace(".mkv", f".{code}.srt")
            assert os.path.exists(out_file), f"Output file for {code} missing: {out_file}"
            with open(out_file, "r", encoding="utf-8") as f:
                parsed = list(srt.parse(f.read()))
            assert len(parsed) == len(source_subs)

            sync_report = verify_sync(source_subs, parsed)
            assert sync_report["valid"] is True
            assert sync_report["start_diff_ms"] == 0
            assert sync_report["end_diff_ms"] == 0

            dropped, _ = check_dropped_lines(source_subs, parsed)
            assert dropped == 0
