import pytest
import os
import srt
from datetime import timedelta
from unittest.mock import patch, AsyncMock, MagicMock

from app.core.validator import (
    detect_language_heuristics,
    classify_cue_language_mismatch,
    check_language_representative,
    evaluate_subtitle_health,
    extract_representative_dialogue_samples
)
from app.services.pipeline import (
    qa_gate,
    SubtitlePipeline,
    QA_STATUS_PASS,
    QA_STATUS_PASS_WITH_WARNINGS,
    QA_STATUS_FAIL
)


# ===========================================================================
# 1. PRODUCTION REPRODUCTION: TONIGHT SHOW JOBS #281 & #282
# ===========================================================================
def test_production_reproduction_tonight_show_job_281():
    """Verify Tonight Show Job #281 cues (ALL-CAPS with speaker prefixes) are recognized as Swedish, not German."""
    job_281_cues = [
        ">> Jimmy: JAG TROR ATT HAN SA TILL MIG ATT HAN SKULLE GÖRA DET HÄR.",
        ">> Jimmy: DET FINNS INGET BÄTTRE ÄN ATT VARA HÄR IKVÄLL MED ER ALLA.",
        ">> Jimmy: VI HAR EN FANTASTISK SHOW IKVÄLL MED MYCKET ROLIGA GÄSTER.",
        ">> Jimmy: VAD TYCKER DU OM DET SOM HÄNDE I GÅR KVÄLL PÅ NYHETERNA?",
        ">> Jimmy: TACK SÅ MYCKET FÖR ATT NI TITTAR PÅ OSS IKVÄLL IGEN.",
        ">> Jimmy: DET ÄR HELT OTROLIGT HUR SNABBT TIDEN GÅR NÄR MAN HAR ROLIGT.",
        ">> Jimmy: LÅT OSS VÄLKOMNA VÅR NÄSTA GÄST TILL PROGRAMMET IKVÄLL.",
        ">> Jimmy: JAG KAN INTE FÖRSTÅ ATT DET REDAN HAR GÅTT ETT HELT ÅR SEDAN SIST.",
        ">> Jimmy: DET HÄR ÄR NÅGOT SOM VI ALDRIG KOMMER ATT GLÖMMA.",
        ">> Jimmy: HUR MÅR DU IKVÄLL? DET ÄR SÅ ROLIGT ATT SE DIG HÄR HOS OSS.",
    ]
    # In v2.3.20, detect_language_heuristics on this combined text classified as 'de' with 86% confidence.
    combined = "\n".join(job_281_cues)
    res = detect_language_heuristics(combined)
    assert res["lang"] == "sv", f"Expected 'sv', got {res['lang']} (confidence={res['confidence']})"


def test_production_reproduction_tonight_show_job_282():
    """Verify Tonight Show Job #282 cues (ALL-CAPS with speaker prefixes) pass stratified language check."""
    # Build 100 realistic cues resembling Job #282
    subs = []
    for i in range(100):
        if i % 2 == 0:
            content = f">> Jimmy: DET HÄR ÄR ETT TEST MED ALL-CAPS RAD {i} OCH DET ÄR MYCKET BRA."
        else:
            content = f">> Jimmy: VI HAR ROLIGT IKVÄLL MED VÅRA GÄSTER PÅ SHOWEN RAD {i}."
        subs.append(srt.Subtitle(i + 1, timedelta(seconds=i * 2), timedelta(seconds=i * 2 + 1), content))

    check = check_language_representative(subs, "sv")
    assert not check["confident_wrong_language"], f"Falsely flagged wrong language: {check}"
    assert check["detected_lang"] == "sv"


# ===========================================================================
# 2. CASE C: LOWERCASE NORMALIZATION & HEURISTICS ROBUSTNESS
# ===========================================================================
def test_all_caps_scandinavian_text_not_misclassified_as_german():
    """ALL-CAPS Swedish dialogue with common particles must detect as Swedish."""
    samples = [
        "JAG VET INTE VAD SOM HÄNDER HÄR IKVÄLL",
        "DET ÄR INTE SÅ FARLIGT SOM DU TROR",
        "KAN DU HJÄLPA MIG MED DET HÄR I MORGON",
        "VI MÅSTE GÅ HEM NU INNAN DET BLIR FÖR SENT",
        "VAD GÖR DU HÄR? JAG TRODDE ATT DU VAR I STOCKHOLM",
    ]
    for s in samples:
        res = detect_language_heuristics(s)
        assert res["lang"] in {"sv", "no", "da", "unknown"}, f"String '{s}' classified as {res['lang']}"
        assert res["lang"] != "de", f"String '{s}' falsely classified as German!"


def test_swedish_keyword_rescue_handles_german_collision():
    """If langdetect outputs 'de' on short text but >= 2 Swedish markers exist, rescue to 'sv'."""
    text = "Och han sa att det inte var bra men vi kommer nu"
    res = detect_language_heuristics(text)
    assert res["lang"] == "sv"


# ===========================================================================
# 3. CUE CLASSIFIER UNIT TESTS (classify_cue_language_mismatch)
# ===========================================================================
def test_classify_cue_language_mismatch_types():
    """Verify all classifications from classify_cue_language_mismatch."""
    # Invariant
    assert classify_cue_language_mismatch("<i></i>", "<i></i>", "sv", "en")["status"] == "SAFE_INVARIANT"
    assert classify_cue_language_mismatch("123", "123", "sv", "en")["status"] == "SAFE_INVARIANT"

    # Correct target
    assert classify_cue_language_mismatch("Hej, hur mår du idag min vän?", "Hello, how are you today my friend?", "sv", "en")["status"] == "CORRECT_TARGET"

    # Legitimate foreign preserved (Source German -> Target German)
    src_de = "Guten Tag, Herr Müller. Wie geht es Ihnen heute?"
    tgt_de = "Guten Tag, Herr Müller. Wie geht es Ihnen heute?"
    assert classify_cue_language_mismatch(tgt_de, src_de, "sv", "en")["status"] == "LEGIT_FOREIGN_PRESERVED"

    # Legitimate foreign preserved (Source French -> Target French)
    src_fr = "Bonjour monsieur, bienvenue à Paris."
    tgt_fr = "Bonjour monsieur, bienvenue à Paris."
    assert classify_cue_language_mismatch(tgt_fr, src_fr, "sv", "en")["status"] == "LEGIT_FOREIGN_PRESERVED"

    # Wrong target language (Source English -> Target German when expected sv)
    src_en = "We have to leave this building immediately before the police arrive."
    tgt_de_hallucination = "Wir müssen dieses Gebäude sofort verlassen bevor die Polizei kommt."
    assert classify_cue_language_mismatch(tgt_de_hallucination, src_en, "sv", "en")["status"] == "WRONG_TARGET_LANGUAGE"


# ===========================================================================
# 4. CASE A: LEGITIMATE SOURCE FOREIGN DIALOGUE PASSES QA
# ===========================================================================
def test_legitimate_foreign_segment_in_source_passes_qa():
    """
    Source movie has 80 cues of English and a 20-cue German dialogue scene in the middle.
    Target translates English to Swedish and preserves the German dialogue.
    QA gate must PASS and not fail on language mismatch.
    """
    source_subs = []
    target_subs = []

    # 1. Start: English -> Swedish (40 cues)
    for i in range(40):
        src_c = f"This is an important English dialogue sentence number {i}."
        tgt_c = f"Detta är en viktig svensk dialogmening nummer {i}."
        source_subs.append(srt.Subtitle(i + 1, timedelta(seconds=i * 2), timedelta(seconds=i * 2 + 1), src_c))
        target_subs.append(srt.Subtitle(i + 1, timedelta(seconds=i * 2), timedelta(seconds=i * 2 + 1), tgt_c))

    # 2. Middle: German in source -> Preserved German in target (20 cues)
    for i in range(40, 60):
        src_c = f"Guten Tag Herr General, die Soldaten sind bereit für den Angriff, Befehl {i}."
        tgt_c = f"Guten Tag Herr General, die Soldaten sind bereit für den Angriff, Befehl {i}."
        source_subs.append(srt.Subtitle(i + 1, timedelta(seconds=i * 2), timedelta(seconds=i * 2 + 1), src_c))
        target_subs.append(srt.Subtitle(i + 1, timedelta(seconds=i * 2), timedelta(seconds=i * 2 + 1), tgt_c))

    # 3. End: English -> Swedish (40 cues)
    for i in range(60, 100):
        src_c = f"We have won the battle and now we can return home safely, cue {i}."
        tgt_c = f"Vi har vunnit slaget och nu kan vi återvända hem säkert, replik {i}."
        source_subs.append(srt.Subtitle(i + 1, timedelta(seconds=i * 2), timedelta(seconds=i * 2 + 1), src_c))
        target_subs.append(srt.Subtitle(i + 1, timedelta(seconds=i * 2), timedelta(seconds=i * 2 + 1), tgt_c))

    # Check language representative with source_sub_blocks
    check = check_language_representative(target_subs, "sv", source_sub_blocks=source_subs)
    assert not check["confident_wrong_language"], f"Case A failed: flagged as confident wrong language: {check}"

    # QA gate evaluation
    res = qa_gate(source_subs, target_subs, target_lang_code="sv")
    assert res["passed"] is True, f"Case A QA Gate failed: {res['issues']}"
    assert res["policy_details"]["confident_wrong_language"] is False


# ===========================================================================
# 5. CASE B: UNWANTED MODEL TRANSLATION TO WRONG TARGET LANGUAGE FAILS QA
# ===========================================================================
def test_unwanted_translation_to_german_fails_qa_and_flags_cues():
    """
    Source is pure English. Model erroneously translated middle segment (20 cues) into German.
    QA gate must catch this as confident_wrong_language and identify the wrong_language_ids.
    """
    source_subs = []
    target_subs = []

    # 1. Start: English -> Swedish (40 cues)
    for i in range(40):
        src_c = f"This is an important English dialogue sentence number {i}."
        tgt_c = f"Detta är en viktig svensk dialogmening nummer {i}."
        source_subs.append(srt.Subtitle(i + 1, timedelta(seconds=i * 2), timedelta(seconds=i * 2 + 1), src_c))
        target_subs.append(srt.Subtitle(i + 1, timedelta(seconds=i * 2), timedelta(seconds=i * 2 + 1), tgt_c))

    # 2. Middle: English in source -> German hallucination in target (20 cues)
    for i in range(40, 60):
        src_c = f"We must proceed carefully through the dangerous territory, instruction {i}."
        tgt_c = f"Wir müssen sehr vorsichtig durch das gefährliche Gebiet gehen, Anweisung {i}."
        source_subs.append(srt.Subtitle(i + 1, timedelta(seconds=i * 2), timedelta(seconds=i * 2 + 1), src_c))
        target_subs.append(srt.Subtitle(i + 1, timedelta(seconds=i * 2), timedelta(seconds=i * 2 + 1), tgt_c))

    # 3. End: English -> Swedish (40 cues)
    for i in range(60, 100):
        src_c = f"We have completed the mission and everything is fine now, cue {i}."
        tgt_c = f"Vi har slutfört uppdraget och allting är bra nu, replik {i}."
        source_subs.append(srt.Subtitle(i + 1, timedelta(seconds=i * 2), timedelta(seconds=i * 2 + 1), src_c))
        target_subs.append(srt.Subtitle(i + 1, timedelta(seconds=i * 2), timedelta(seconds=i * 2 + 1), tgt_c))

    # Check language representative with source_sub_blocks
    check = check_language_representative(target_subs, "sv", source_sub_blocks=source_subs)
    assert check["confident_wrong_language"] is True, "Case B failed: should have caught wrong language!"
    assert len(check["wrong_language_cue_ids"]) > 0, "Case B failed: no wrong_language_cue_ids returned"

    # Verify that wrong_language_cue_ids fall within range 40..59
    for cid in check["wrong_language_cue_ids"]:
        assert 40 <= cid < 60

    # QA gate evaluation
    res = qa_gate(source_subs, target_subs, target_lang_code="sv")
    assert res["passed"] is False, "Case B QA Gate must FAIL for hallucinated German"
    assert res["policy_details"]["confident_wrong_language"] is True
    assert len(res["wrong_language_ids"]) > 0


# ===========================================================================
# 6. EVALUATE SUBTITLE HEALTH WITH SOURCE REFERENCE
# ===========================================================================
def test_evaluate_subtitle_health_with_reference(tmp_path):
    """Health check must pass for subtitles containing preserved foreign dialogue if reference is given."""
    sub_file = str(tmp_path / "movie.sv.srt")

    subs_target = []
    subs_source = []

    for i in range(100):
        if 40 <= i < 60:
            content_src = f"Bonjour monsieur, comment allez-vous aujourd'hui {i}?"
            content_tgt = f"Bonjour monsieur, comment allez-vous aujourd'hui {i}?"
        else:
            content_src = f"Hello world, this is an English sentence {i}."
            content_tgt = f"Hej världen, detta är en svensk mening {i}."

        sub_s = srt.Subtitle(i + 1, timedelta(seconds=i * 2), timedelta(seconds=i * 2 + 1), content_src)
        sub_t = srt.Subtitle(i + 1, timedelta(seconds=i * 2), timedelta(seconds=i * 2 + 1), content_tgt)
        subs_source.append(sub_s)
        subs_target.append(sub_t)

    with open(sub_file, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs_target))

    health = evaluate_subtitle_health(sub_file, target_lang_code="sv", reference_sub_blocks=subs_source)
    assert health["status"] == "GREEN"
    assert health["health_score"] >= 90


# ===========================================================================
# 7. END-TO-END PIPELINE TARGETED RECOVERY OF WRONG LANGUAGE CUES
# ===========================================================================
@pytest.mark.asyncio
async def test_pipeline_targeted_recovery_recovers_wrong_language_cues(tmp_path, monkeypatch):
    """Pipeline QA recovery loop identifies wrong_language_ids and translates them to target language."""
    video = str(tmp_path / "GermanHallucinationShow - S01E01.mkv")
    with open(video, "w") as f:
        f.write("dummy video")

    target_path = str(tmp_path / "GermanHallucinationShow - S01E01.sv.srt")

    pipeline = SubtitlePipeline()

    # Create 50 source cues
    source_subs = []
    for i in range(50):
        content = f"English dialogue sentence number {i} for testing."
        source_subs.append(srt.Subtitle(i + 1, timedelta(seconds=i * 2), timedelta(seconds=i * 2 + 1), content))

    # Mock extract_embedded_srt to output source_subs
    def mock_extract(vpath, outpath, preferred_lang="eng"):
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(srt.compose(source_subs))
        return True

    # Mock translator: on first pass, batches 20-30 return German; on recovery/targeted, returns Swedish
    call_count = 0
    async def mock_translate_batch(batch, target_language="Swedish", show_title="", tm_key="", *args, **kwargs):
        nonlocal call_count
        call_count += 1
        results = []
        for item in batch:
            cid = item["id"]
            if 20 <= cid < 30 and call_count == 1:
                # First pass hallucination to German
                results.append({"id": cid, "text": f"Guten Tag, das ist ein deutscher Satz Nummer {cid}."})
            else:
                # Correct Swedish
                results.append({"id": cid, "text": f"Detta är en svensk mening nummer {cid}."})
        return results

    async def mock_escalate_single_line(idx, text, prev_text="", next_text="", target_language="Swedish", show_title="", tm_key="", is_real_untranslated=True, job_id=None, exhausted_strategies=None, *args, **kwargs):
        return f"Detta är en svensk mening nummer {idx}."

    async def mock_fast_final_rescue(items, target_language="Swedish", show_title="", tm_key="", attempt=1, job_id=None, *args, **kwargs):
        return [{"id": it["id"], "text": f"Detta är en svensk mening nummer {it['id']}."} for it in items]

    monkeypatch.setattr("app.services.pipeline.extract_embedded_srt", mock_extract)
    monkeypatch.setattr("app.services.pipeline.get_setting", lambda k, d="": "true" if k in ["extract_embedded"] else ("false" if k in ["notify_jellyfin", "enable_bazarr_check"] else d))
    monkeypatch.setattr("app.services.pipeline.notify_jellyfin_library_refresh", AsyncMock())
    monkeypatch.setattr(pipeline, "get_configured_languages", lambda: [{"name": "Swedish", "code": "sv", "enabled": True}])
    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "escalate_single_line", mock_escalate_single_line)
    monkeypatch.setattr(pipeline.translator, "fast_final_rescue_batch", mock_fast_final_rescue)

    res = await pipeline.process_video_file(video, force_retranslate=True)

    assert res["status"] == "translated"
    assert os.path.exists(target_path)
    with open(target_path, "r", encoding="utf-8") as f:
        final_content = f.read()
    assert "svensk mening" in final_content
    assert "Guten Tag" not in final_content
