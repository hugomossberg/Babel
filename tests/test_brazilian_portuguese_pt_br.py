import pytest
import os
import json
import sqlite3
import srt
from datetime import timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from app.core.languages import (
    LANGUAGES,
    get_language,
    normalize_language_code,
    get_bazarr_language_code,
    get_deepl_source_code,
    get_deepl_target_code,
)
from app.core.validator import (
    are_languages_compatible,
    detect_language_heuristics,
    check_language_representative,
    evaluate_subtitle_health,
)
from app.services.bazarr_checker import find_external_subtitle
from app.services.bazarr_coordinator import _language_matches_job_text, _extract_job_language_codes
from app.services.scanner import is_target_language_subtitle, is_qualifying_embedded_subtitle_track
from app.services.translator import (
    get_system_instruction,
    build_translation_prompt,
    build_translation_output_schema,
    SubtitleTranslator,
)
from app.services.pipeline import SubtitlePipeline, qa_gate
from app.core.db import init_db, create_job, get_job_by_id, DB_PATH
from app.services.source_resolver import SubtitleSource, SourceOrigin


# ==============================================================================
# 1. LANGUAGE REGISTRY & NORMALIZATION
# ==============================================================================

def test_pt_and_pt_br_normalize_to_different_canonical_languages():
    """Verify pt and pt-BR are distinct first-class languages in registry and normalization."""
    # Canonical codes
    assert normalize_language_code("pt") == "pt"
    assert normalize_language_code("pt-BR") == "pt-BR"
    assert normalize_language_code("pt") != normalize_language_code("pt-BR")

    # Objects
    pt = get_language("pt")
    pt_br = get_language("pt-BR")

    assert pt is not None
    assert pt_br is not None
    assert pt.code == "pt"
    assert pt_br.code == "pt-BR"
    assert pt.display_name == "Portuguese"
    assert pt_br.display_name == "Brazilian Portuguese"


def test_pt_br_aliases_resolve_correctly():
    """Verify common Brazilian Portuguese aliases resolve to pt-BR."""
    aliases = [
        "pt-BR",
        "pt-br",
        "pt_br",
        "pt_BR",
        "por-BR",
        "por-br",
        "por_br",
        "por_BR",
        "pob",
        "pb",
        "brazilian portuguese",
        "brazilian",
        "português brasileiro",
        "portugues brasileiro",
        "portugues-brasileiro",
        "português (brasil)",
        "portugues (brasil)",
        "portuguese (brazil)",
        "portuguese (brazilian)",
        "pt-brazil",
        "pt (br)",
    ]
    for alias in aliases:
        lang = get_language(alias)
        assert lang is not None, f"Failed to get_language for alias: {alias}"
        assert lang.code == "pt-BR", f"Alias '{alias}' resolved to '{lang.code}', expected 'pt-BR'"
        assert normalize_language_code(alias) == "pt-BR", f"Alias '{alias}' normalized to '{normalize_language_code(alias)}', expected 'pt-BR'"


def test_existing_portuguese_aliases_resolve_to_pt():
    """Verify existing Portuguese / European Portuguese aliases continue to resolve to pt."""
    pt_aliases = [
        "pt",
        "por",
        "portuguese",
        "português",
        "portugues",
        "pt-pt",
        "pt-PT",
        "pt_pt",
        "pt_PT",
        "por-pt",
        "por_pt",
        "european portuguese",
        "portuguese (portugal)",
        "português (portugal)",
        "portugues (portugal)",
    ]
    for alias in pt_aliases:
        lang = get_language(alias)
        assert lang is not None, f"Failed to get_language for alias: {alias}"
        assert lang.code == "pt", f"Alias '{alias}' resolved to '{lang.code}', expected 'pt'"
        assert normalize_language_code(alias) == "pt", f"Alias '{alias}' normalized to '{normalize_language_code(alias)}', expected 'pt'"


def test_unrelated_dialects_and_ambiguous_strings():
    """
    Verify dialect handling does not become overly permissive:
    - Unsupported Portuguese dialects fall back to base 'pt' (NOT 'pt-BR')
    - Standard dialects of other languages fall back to their base code
    - Completely unknown strings return None / default
    """
    # Unsupported Portuguese dialects fall back to base 'pt', NEVER 'pt-BR'
    assert normalize_language_code("pt-AO") == "pt"
    assert normalize_language_code("pt-foobar") == "pt"
    assert normalize_language_code("pt-MZ") == "pt"

    # Other supported language dialects fall back to their base
    assert normalize_language_code("en-US") == "en"
    assert normalize_language_code("en-GB") == "en"
    assert normalize_language_code("en-foobar") == "en"
    assert normalize_language_code("sv-SE") == "sv"
    assert normalize_language_code("zh-CN") == "zh"
    assert normalize_language_code("de-AT") == "de"
    assert normalize_language_code("es-MX") == "es"

    # Unknown strings do not match any language
    assert get_language("unknown_random_string") is None
    assert normalize_language_code("unknown_random_string") == "unknown_random_string"
    assert normalize_language_code("unknown_random_string", default="unknown") == "unknown"
    assert get_language("") is None
    assert normalize_language_code("") == "unknown"


def test_display_language_name_helper():
    """Verify get_display_language_name correctly resolves human-readable names."""
    from app.core.languages import get_display_language_name

    assert get_display_language_name("pt-BR") == "Brazilian Portuguese"
    assert get_display_language_name("pt_br") == "Brazilian Portuguese"
    assert get_display_language_name("por-BR") == "Brazilian Portuguese"
    assert get_display_language_name("pt") == "Portuguese"
    assert get_display_language_name("por") == "Portuguese"
    assert get_display_language_name("sv") == "Swedish"
    assert get_display_language_name("en") == "English"
    assert get_display_language_name("") == "unknown"


def test_language_compatibility_pt_vs_pt_br():
    """Verify are_languages_compatible treats pt and pt-BR as distinct, non-identical languages."""
    assert are_languages_compatible("pt-BR", "pt-BR") is True
    assert are_languages_compatible("pt-BR", "pt_br") is True
    assert are_languages_compatible("pt", "pt") is True
    assert are_languages_compatible("pt", "por") is True

    # pt and pt-BR must NOT be compatible
    assert are_languages_compatible("pt", "pt-BR") is False
    assert are_languages_compatible("pt-BR", "pt") is False
    assert are_languages_compatible("pt-PT", "pt-BR") is False
    assert are_languages_compatible("por", "por-br") is False


# ==============================================================================
# 2. DEEPL & BAZARR CODE MAPPING
# ==============================================================================

def test_deepl_code_mapping_pt_and_pt_br():
    """Verify DeepL target code mapping routes pt-BR -> PT-BR and pt -> PT-PT."""
    # Target codes
    assert get_deepl_target_code("pt-BR") == "PT-BR"
    assert get_deepl_target_code("pt_br") == "PT-BR"
    assert get_deepl_target_code("Brazilian Portuguese") == "PT-BR"
    assert get_deepl_target_code("pt") == "PT-PT"
    assert get_deepl_target_code("Portuguese") == "PT-PT"
    assert get_deepl_target_code("por") == "PT-PT"

    # Source codes (DeepL uses generic PT for source)
    assert get_deepl_source_code("pt-BR") == "PT"
    assert get_deepl_source_code("pt") == "PT"
    assert get_deepl_source_code("Brazilian Portuguese") == "PT"
    assert get_deepl_source_code("Portuguese") == "PT"


def test_bazarr_code_mapping_pt_and_pt_br():
    """Verify Bazarr language code helper returns pt-BR for Brazilian Portuguese and pt for Portuguese."""
    assert get_bazarr_language_code("pt-BR") == "pt-BR"
    assert get_bazarr_language_code("pt_br") == "pt-BR"
    assert get_bazarr_language_code("Brazilian Portuguese") == "pt-BR"
    assert get_bazarr_language_code("pt") == "pt"
    assert get_bazarr_language_code("Portuguese") == "pt"
    assert get_bazarr_language_code("por") == "pt"


# ==============================================================================
# 3. TRANSLATION PROMPTS & DISPLAY NAMES
# ==============================================================================

def test_translation_prompts_for_pt_br_and_pt():
    """Verify translation prompts explicitly mention Brazilian Portuguese when pt-BR is targeted."""
    # 1. System instruction
    prompt_pt_br = get_system_instruction(target_language="Brazilian Portuguese")
    assert "Brazilian Portuguese" in prompt_pt_br
    assert "from English to Brazilian Portuguese" in prompt_pt_br

    prompt_pt_br_code = get_system_instruction(target_language="pt-BR")
    assert "Brazilian Portuguese" in prompt_pt_br_code

    prompt_pt = get_system_instruction(target_language="Portuguese")
    assert "from English to Portuguese" in prompt_pt
    assert "Brazilian Portuguese" not in prompt_pt

    prompt_pt_code = get_system_instruction(target_language="pt")
    assert "from English to Portuguese" in prompt_pt_code

    # 2. Build translation prompt
    batch = [{"id": 1, "text": "Good morning."}]
    p_br = build_translation_prompt(batch, target_language="Brazilian Portuguese")
    assert "into Brazilian Portuguese" in p_br

    p_br_code = build_translation_prompt(batch, target_language="pt-BR")
    assert "into Brazilian Portuguese" in p_br_code

    p_pt = build_translation_prompt(batch, target_language="Portuguese")
    assert "into Portuguese" in p_pt

    # 3. Output schema
    s_br = build_translation_output_schema(batch, target_language="Brazilian Portuguese")
    assert "Brazilian Portuguese" in s_br["properties"]["translations"]["items"]["properties"]["text"]["description"]

    s_br_code = build_translation_output_schema(batch, target_language="pt-BR")
    assert "Brazilian Portuguese" in s_br_code["properties"]["translations"]["items"]["properties"]["text"]["description"]


# ==============================================================================
# 4. SUBTITLE FILENAME DETECTION & LOOKUP
# ==============================================================================

def test_find_external_subtitle_pt_br_vs_pt_isolation(tmp_path):
    """
    Verify:
    1. Looking up pt-BR does NOT accept a plain .pt.srt or .por.srt file.
    2. Looking up pt does NOT accept a .pt-br.srt or .pt_br.srt file.
    3. Looking up pt-BR matches .pt-BR.srt, .pt-br.srt, .pt_br.srt, .por-br.srt, .pob.srt.
    4. Looking up pt matches .pt.srt, .por.srt, .pt-pt.srt.
    """
    video = tmp_path / "movie.mkv"
    video.touch()

    # Case A: Only movie.pt.srt exists
    pt_file = tmp_path / "movie.pt.srt"
    with open(pt_file, "w", encoding="utf-8") as f:
        f.write("1\n00:00:01,000 --> 00:00:03,000\nOlá, como vai você? Este é um arquivo de teste longo o suficiente para passar no tamanho mínimo.\n")

    # pt lookup finds it
    assert find_external_subtitle(str(video), "pt") == str(pt_file)
    assert find_external_subtitle(str(video), "Portuguese") == str(pt_file)

    # pt-BR lookup MUST NOT find it!
    assert find_external_subtitle(str(video), "pt-BR") is None
    assert find_external_subtitle(str(video), "Brazilian Portuguese") is None

    # Case B: Remove pt_file and create movie.pt-br.srt
    pt_file.unlink()
    pt_br_file = tmp_path / "movie.pt-br.srt"
    with open(pt_br_file, "w", encoding="utf-8") as f:
        f.write("1\n00:00:01,000 --> 00:00:03,000\nOi, tudo bem com você? Este é um arquivo de teste longo o suficiente para passar no tamanho mínimo.\n")

    # pt-BR lookup finds it
    assert find_external_subtitle(str(video), "pt-BR") == str(pt_br_file)
    assert find_external_subtitle(str(video), "Brazilian Portuguese") == str(pt_br_file)
    assert find_external_subtitle(str(video), "pt_br") == str(pt_br_file)

    # pt lookup MUST NOT find it!
    assert find_external_subtitle(str(video), "pt") is None
    assert find_external_subtitle(str(video), "Portuguese") is None


def test_scanner_is_target_language_subtitle_pt_br():
    """Verify scanner subtitle matching distinguishes pt-br from pt."""
    pt_aliases = ["pt", "por", "portuguese"]
    pt_br_aliases = ["pt-br", "pt_br", "por-br", "brazilian portuguese", "pob"]

    # movie.pt-br.srt is target for pt-BR, but NOT for pt
    assert is_target_language_subtitle("movie.pt-br.srt", pt_br_aliases) is True
    assert is_target_language_subtitle("movie.pt-br.srt", pt_aliases) is False

    # movie.pt_br.srt is target for pt-BR, but NOT for pt
    assert is_target_language_subtitle("movie.pt_br.srt", pt_br_aliases) is True
    assert is_target_language_subtitle("movie.pt_br.srt", pt_aliases) is False

    # movie.pt.srt is target for pt, but NOT for pt-BR
    assert is_target_language_subtitle("movie.pt.srt", pt_aliases) is True
    assert is_target_language_subtitle("movie.pt.srt", pt_br_aliases) is False

    # movie.por.srt is target for pt, but NOT for pt-BR
    assert is_target_language_subtitle("movie.por.srt", pt_aliases) is True
    assert is_target_language_subtitle("movie.por.srt", pt_br_aliases) is False


def test_scanner_is_qualifying_embedded_subtitle_track_pt_br():
    """Verify embedded track qualification distinguishes pt-br from pt."""
    pt_aliases = ["pt", "por", "portuguese"]
    pt_br_aliases = ["pt-br", "pt_br", "por-br", "brazilian portuguese", "pob"]

    track_pt_br = {"language": "pt-BR", "codec": "SubRip/SRT", "forced": False}
    track_pt = {"language": "por", "codec": "SubRip/SRT", "forced": False}

    assert is_qualifying_embedded_subtitle_track(track_pt_br, pt_br_aliases) is True
    assert is_qualifying_embedded_subtitle_track(track_pt_br, pt_aliases) is False

    assert is_qualifying_embedded_subtitle_track(track_pt, pt_aliases) is True
    assert is_qualifying_embedded_subtitle_track(track_pt, pt_br_aliases) is False


# ==============================================================================
# 5. BAZARR JOB STRING CORRELATION
# ==============================================================================

def test_bazarr_job_matching_pt_and_pt_br():
    """Verify Bazarr job matching correctly isolates [pt-BR] and [pt] jobs."""
    job_pt_br = "Search subtitles for The Matrix (1999) [pt-BR]"
    job_pt = "Search subtitles for The Matrix (1999) [pt]"
    job_pt_name = "Search subtitles for The Matrix (1999) Portuguese"
    job_br_name = "Search subtitles for The Matrix (1999) Brazilian Portuguese"

    # pt-BR matching
    assert _language_matches_job_text("pt-BR", job_pt_br) is True
    assert _language_matches_job_text("pt-BR", job_br_name) is True
    assert _language_matches_job_text("pt-BR", job_pt) is False
    assert _language_matches_job_text("pt-BR", job_pt_name) is False

    # pt matching
    assert _language_matches_job_text("pt", job_pt) is True
    assert _language_matches_job_text("pt", job_pt_name) is True
    assert _language_matches_job_text("pt", job_pt_br) is False
    assert _language_matches_job_text("pt", job_br_name) is False

    # Extract job codes
    assert "pt-BR" in _extract_job_language_codes(job_pt_br)
    assert "pt" not in _extract_job_language_codes(job_pt_br)

    assert "pt" in _extract_job_language_codes(job_pt)
    assert "pt-BR" not in _extract_job_language_codes(job_pt)


# ==============================================================================
# 6. LANGUAGE DETECTION & QA GATE
# ==============================================================================

def test_detect_language_heuristics_pt_br():
    """Verify language detection with expected pt-BR returns pt-BR."""
    sample_pt_br = "Oi, tudo bem? Nós estamos muito felizes em recebê-los aqui hoje na nossa cidade para o evento."
    res_br = detect_language_heuristics(sample_pt_br, expected_language="pt-BR")
    assert res_br["lang"] == "pt-BR"
    assert res_br["confidence"] >= 0.8

    res_pt = detect_language_heuristics(sample_pt_br, expected_language="pt")
    assert res_pt["lang"] == "pt"


def test_qa_gate_passes_clean_pt_br():
    """Verify QA gate passes clean Brazilian Portuguese translations."""
    source_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=4), content="Welcome everyone to tonight's presentation."),
        srt.Subtitle(index=2, start=timedelta(seconds=5), end=timedelta(seconds=8), content="We are delighted to have our international guests.")
    ]
    target_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=4), content="Sejam todos bem-vindos à apresentação desta noite."),
        srt.Subtitle(index=2, start=timedelta(seconds=5), end=timedelta(seconds=8), content="Estamos muito felizes em receber nossos convidados internacionais.")
    ]

    qa_res = qa_gate(source_subs, target_subs, target_lang_code="pt-BR")
    assert qa_res["passed"] is True, f"Clean PT-BR translation failed QA: {qa_res}"
    assert qa_res["dropped_count"] == 0


# ==============================================================================
# 7. SOURCE == TARGET SHORT-CIRCUIT
# ==============================================================================

@pytest.mark.asyncio
async def test_existing_target_pt_br_bypasses_ai_translation(tmp_path):
    """
    Verify that if a healthy PT-BR target subtitle already exists on disk,
    the pipeline skips AI translation.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.cursor().execute("DELETE FROM jobs")
    conn.commit()
    conn.close()

    with patch("google.genai.Client"):
        pipeline = SubtitlePipeline()

    video_path = tmp_path / "movie_br.mkv"
    video_path.touch()

    # Create reference English source and healthy PT-BR target with >= 5 cues
    cues_en = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=4), content="Hello, welcome to the conference everyone."),
        srt.Subtitle(index=2, start=timedelta(seconds=5), end=timedelta(seconds=8), content="We hope that you have an excellent stay with us."),
        srt.Subtitle(index=3, start=timedelta(seconds=9), end=timedelta(seconds=12), content="Our speakers have prepared exceptional talks."),
        srt.Subtitle(index=4, start=timedelta(seconds=13), end=timedelta(seconds=16), content="Please take your seats in the auditorium."),
        srt.Subtitle(index=5, start=timedelta(seconds=17), end=timedelta(seconds=20), content="The first session will begin shortly."),
        srt.Subtitle(index=6, start=timedelta(seconds=21), end=timedelta(seconds=24), content="Thank you for joining us today.")
    ]
    cues_br = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=4), content="Olá, bem-vindos a todos para a conferência."),
        srt.Subtitle(index=2, start=timedelta(seconds=5), end=timedelta(seconds=8), content="Esperamos que vocês tenham uma excelente estadia conosco."),
        srt.Subtitle(index=3, start=timedelta(seconds=9), end=timedelta(seconds=12), content="Nossos palestrantes prepararam palestras excepcionais."),
        srt.Subtitle(index=4, start=timedelta(seconds=13), end=timedelta(seconds=16), content="Por favor, tomem seus assentos no auditório."),
        srt.Subtitle(index=5, start=timedelta(seconds=17), end=timedelta(seconds=20), content="A primeira sessão começará em breve."),
        srt.Subtitle(index=6, start=timedelta(seconds=21), end=timedelta(seconds=24), content="Obrigado por se juntarem a nós hoje.")
    ]
    with open(str(video_path).replace(".mkv", ".en.srt"), "w", encoding="utf-8") as f:
        f.write(srt.compose(cues_en))
    with open(str(video_path).replace(".mkv", ".pt-br.srt"), "w", encoding="utf-8") as f:
        f.write(srt.compose(cues_br))

    def fake_get_setting(key, default=None):
        if key == "languages":
            return json.dumps([{"name": "Brazilian Portuguese", "code": "pt-BR", "enabled": True}])
        if key == "auto_repair_unhealthy": return "false"
        if key == "extract_target_embedded": return "false"
        if key == "extract_source_embedded": return "false"
        if key == "enable_bazarr_check": return "false"
        return default

    translate_called = False

    async def fake_translate(*args, **kwargs):
        nonlocal translate_called
        translate_called = True
        return cues_br

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting), \
         patch.object(pipeline, "trigger_bazarr_search"), \
         patch.object(pipeline.translator, "translate_srt_content", side_effect=fake_translate):

        job_id = create_job(str(video_path))
        res = await pipeline._run_pipeline_logic(job_id, str(video_path), wait_seconds=0)

        assert res["status"] in ["skipped", "success", "translated"]
        assert translate_called is False, "AI translation should NOT be called when healthy target already exists"


@pytest.mark.asyncio
async def test_pt_source_does_not_satisfy_pt_br_target_and_dispatches_ai(tmp_path):
    """
    Verify that if source is European Portuguese (.pt.srt) and target is Brazilian Portuguese (pt-BR),
    Babel recognizes they are distinct languages and dispatches AI translation to pt-BR.
    """
    init_db()
    with patch("google.genai.Client"):
        pipeline = SubtitlePipeline()

    video_path = tmp_path / "movie_cross.mkv"
    video_path.touch()

    # Source is European Portuguese with >= 5 cues
    cues_pt = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=4), content="Olá a todos vós, sejam muito bem-vindos à conferência."),
        srt.Subtitle(index=2, start=timedelta(seconds=5), end=timedelta(seconds=8), content="Estou a planear apresentar os resultados da pesquisa."),
        srt.Subtitle(index=3, start=timedelta(seconds=9), end=timedelta(seconds=12), content="A equipa trabalhou arduamente nos últimos meses."),
        srt.Subtitle(index=4, start=timedelta(seconds=13), end=timedelta(seconds=16), content="Esperamos que os dados sejam úteis para vós."),
        srt.Subtitle(index=5, start=timedelta(seconds=17), end=timedelta(seconds=20), content="Podem fazer perguntas no fecho da sessão."),
        srt.Subtitle(index=6, start=timedelta(seconds=21), end=timedelta(seconds=24), content="Muito obrigado pela vossa atenção.")
    ]
    with open(str(video_path).replace(".mkv", ".pt.srt"), "w", encoding="utf-8") as f:
        f.write(srt.compose(cues_pt))

    cues_pt_br = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=4), content="Olá a todos vocês, sejam muito bem-vindos à conferência."),
        srt.Subtitle(index=2, start=timedelta(seconds=5), end=timedelta(seconds=8), content="Estou planejando apresentar os resultados da pesquisa."),
        srt.Subtitle(index=3, start=timedelta(seconds=9), end=timedelta(seconds=12), content="A equipe trabalhou arduamente nos últimos meses."),
        srt.Subtitle(index=4, start=timedelta(seconds=13), end=timedelta(seconds=16), content="Esperamos que os dados sejam úteis para vocês."),
        srt.Subtitle(index=5, start=timedelta(seconds=17), end=timedelta(seconds=20), content="Podem fazer perguntas no encerramento da sessão."),
        srt.Subtitle(index=6, start=timedelta(seconds=21), end=timedelta(seconds=24), content="Muito obrigado pela atenção de vocês.")
    ]

    def fake_get_setting(key, default=None):
        if key == "languages":
            return json.dumps([{"name": "Brazilian Portuguese", "code": "pt-BR", "enabled": True}])
        if key == "auto_repair_unhealthy": return "false"
        if key == "extract_target_embedded": return "false"
        if key == "extract_source_embedded": return "false"
        if key == "enable_bazarr_check": return "false"
        return default

    translate_called_with = []

    async def fake_translate(subs, target_language="Brazilian Portuguese", *args, **kwargs):
        translate_called_with.append(target_language)
        return [
            srt.Subtitle(index=sub.index, start=sub.start, end=sub.end, content=cues_pt_br[sub.index - 1].content)
            for sub in subs
        ]

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting), \
         patch.object(pipeline, "trigger_bazarr_search"), \
         patch.object(pipeline.translator, "classify_and_recover_identical", return_value=[]), \
         patch.object(pipeline.translator, "translate_srt_content", side_effect=fake_translate):

        job_id = create_job(str(video_path))
        res = await pipeline._run_pipeline_logic(job_id, str(video_path), wait_seconds=0)

        assert res["status"] in ["success", "translated"]
        assert len(translate_called_with) == 1
        assert translate_called_with[0] in ["Brazilian Portuguese", "pt-BR"]

        out_path = str(video_path).replace(".mkv", ".pt-BR.srt")
        assert os.path.exists(out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "planejando" in content


# ==============================================================================
# 8. FULL PIPELINE TRANSLATION TO PT-BR
# ==============================================================================

@pytest.mark.asyncio
async def test_full_pipeline_translation_to_pt_br(tmp_path):
    """
    Verify full end-to-end translation pipeline for Brazilian Portuguese:
    English source -> AI translation to PT-BR -> QA gate PASS -> .pt-BR.srt published.
    """
    init_db()
    with patch("google.genai.Client"):
        pipeline = SubtitlePipeline()

    video_path = tmp_path / "show_pt_br.mkv"
    video_path.touch()

    source_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=4), content="Welcome to the conference, everyone."),
        srt.Subtitle(index=2, start=timedelta(seconds=5), end=timedelta(seconds=8), content="We are excited to share our latest research results.")
    ]
    source_srt = str(video_path).replace(".mkv", ".en.srt")
    with open(source_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(source_subs))

    pt_br_translations = [
        "Bem-vindos à conferência, pessoal.",
        "Estamos animados para compartilhar os resultados da nossa pesquisa recente."
    ]

    def fake_get_setting(key, default=None):
        if key == "languages":
            return json.dumps([{"name": "Brazilian Portuguese", "code": "pt-BR", "enabled": True}])
        if key == "auto_repair_unhealthy": return "false"
        if key == "extract_target_embedded": return "false"
        if key == "extract_source_embedded": return "false"
        if key == "enable_bazarr_check": return "false"
        return default

    async def fake_translate(subs, target_language="Brazilian Portuguese", *args, **kwargs):
        assert target_language in ["Brazilian Portuguese", "pt-BR"]
        return [
            srt.Subtitle(index=sub.index, start=sub.start, end=sub.end, content=pt_br_translations[sub.index - 1])
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

        out_path = str(video_path).replace(".mkv", ".pt-BR.srt")
        assert os.path.exists(out_path), f"Output {out_path} was not published"

        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()

        out_cues = list(srt.parse(content))
        assert len(out_cues) == len(source_subs)
        assert "Bem-vindos à conferência" in out_cues[0].content
