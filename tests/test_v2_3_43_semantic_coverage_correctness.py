"""
Unit and regression tests for Pass 1.2: Semantic Coverage & Residual Lexical Verification.

Verifies:
1. POSITIVE 1: Multi-token Unicode entity sequences ("Héðinn, Beinir, Ari, Kári.", "<i>Bárður, Þórður, Böðvar, Lúðvík,</i>") KEEP when all spans are verified.
2. POSITIVE 2: Entity + valid shared lexical residual ("Ja, Lukas.", "Minato! Passa!") KEEP when entity is protected and residual is valid target rendering.
3. POSITIVE 3: Non-verbal acoustic invariant ("<i>(Klick)</i>", "(Bip)") KEEP with verified acoustic invariant evidence.
4. POSITIVE 4: Repeated shared dialogue ("- Amen.\n- Amen.") KEEP with positive evidence.
5. POSITIVE 5: Mixed punctuation, formatting, and numbers with shared words ("... 4. Ja.", "<i>12:30. OK.</i>").
6. POSITIVE 6: Non-Latin / Unicode scripts (Japanese さようなら, Arabic سلام, Korean, Chinese) without Latin/TitleCase assumptions.
7. NEGATIVE 7: Complete untranslated source sentences ("Ich komme morgen.", "Come here.") fail closed to TRANSLATE/recovery.
8. NEGATIVE 8: Entity + translatable source dialogue ("Lukas, ich komme morgen.") fails closed to TRANSLATE/recovery despite verified entity span.
9. NEGATIVE 9: False friend / invalid identical token fails closed to TRANSLATE/recovery.
10. NEGATIVE 10: Unknown / uncertain semantic content fails closed to TRANSLATE/recovery.
11. End-to-end pipeline recovery integration: Verified KEEP cues pass cleanly without 'Rejected unsafe KEEP' or false semantic deadlocks.
"""
import pytest
import json
import srt
from datetime import timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.translator import (
    SubtitleTranslator,
    validate_classifier_output,
    validate_semantic_invariant_verification_output,
    is_pure_structural_invariant,
    is_meaningful_translation,
    is_deterministically_safe_keep,
    has_entity_evidence
)
from app.services.pipeline import SubtitlePipeline, qa_gate


def test_positive_1_multi_token_unicode_entity_sequences():
    """Positive Control 1: Ren entity-sekvens med flera Unicode-namn."""
    raw_verifier_resp = json.dumps({
        "results": [
            {
                "id": 1,
                "invariant_in_target": True,
                "explanation": "Icelandic/Old Norse personal names list unchanged in Swedish"
            },
            {
                "id": 2,
                "invariant_in_target": True,
                "explanation": "Icelandic personal names list with HTML markup unchanged in Swedish"
            }
        ]
    })
    candidates = [
        {"id": 1, "target": "Héðinn, Beinir, Ari, Kári."},
        {"id": 2, "target": "<i>Bárður, Þórður, Böðvar, Lúðvík,</i>"}
    ]
    verified_ids = validate_semantic_invariant_verification_output(raw_verifier_resp, candidates, show_title="Godland")
    assert verified_ids == {1, 2}

    # Verify that in classifier validation with positive verification, they are accepted as KEEP
    raw_clf = json.dumps({
        "results": [
            {"id": 1, "action": "keep", "reason": "proper_noun", "invariant_in_target": True, "explanation": "Names", "text": "Héðinn, Beinir, Ari, Kári."},
            {"id": 2, "action": "keep", "reason": "proper_noun", "invariant_in_target": True, "explanation": "Names", "text": "<i>Bárður, Þórður, Böðvar, Lúðvík,</i>"}
        ]
    })
    items = [
        {"id": 1, "text": "Héðinn, Beinir, Ari, Kári."},
        {"id": 2, "text": "<i>Bárður, Þórður, Böðvar, Lúðvík,</i>"}
    ]
    clf_res = validate_classifier_output(raw_clf, items)
    assert len(clf_res) == 2
    # Fail-closed trust boundary: Raw classifier cannot grant KEEP without semantic verifier
    assert clf_res[0]["action"] == "translate"
    assert clf_res[0]["reason"] == "needs_semantic_verification"
    assert clf_res[1]["action"] == "translate"
    assert clf_res[1]["reason"] == "needs_semantic_verification"


def test_positive_2_entity_plus_valid_shared_lexical_residual():
    """Positive Control 2: Entity + legitim shared lexical residual ('Ja, Lukas.', 'Minato! Passa!')."""
    raw_verifier_resp = json.dumps({
        "results": [
            {
                "id": 814,
                "invariant_in_target": True,
                "explanation": "Proper name Lukas combined with legitimate shared affirmative particle Ja in German->Swedish context"
            },
            {
                "id": 815,
                "invariant_in_target": True,
                "explanation": "Character name Minato with legitimate loanword/action imperative Passa in context"
            }
        ]
    })
    candidates = [
        {"id": 814, "target": "Ja, Lukas.", "context_before": "Guten Morgen", "context_after": "Danke"},
        {"id": 815, "target": "Minato! Passa!", "context_before": "(ball bounces)", "context_after": "Hier!"}
    ]
    verified_ids = validate_semantic_invariant_verification_output(raw_verifier_resp, candidates, show_title="The Teachers' Lounge")
    assert verified_ids == {814, 815}


def test_positive_3_non_verbal_acoustic_sound_effect():
    """Positive Control 3: Non-verbal / acoustic sound effect ('<i>(Klick)</i>', '(Bip)')."""
    raw_verifier_resp = json.dumps({
        "results": [
            {
                "id": 420,
                "invariant_in_target": True,
                "explanation": "Acoustic mechanical click sound effect identical across German and Swedish"
            },
            {
                "id": 421,
                "invariant_in_target": True,
                "explanation": "Acoustic electronic beep onomatopoeia identical across language pairs"
            }
        ]
    })
    candidates = [
        {"id": 420, "target": "<i>(Klick)</i>"},
        {"id": 421, "target": "(Bip)"}
    ]
    verified_ids = validate_semantic_invariant_verification_output(raw_verifier_resp, candidates, show_title="The Teachers' Lounge")
    assert verified_ids == {420, 421}


def test_positive_4_repeated_shared_dialogue():
    """Positive Control 4: Repeated shared dialogue ('- Amen.\\n- Amen.')."""
    raw_verifier_resp = json.dumps({
        "results": [
            {
                "id": 390,
                "invariant_in_target": True,
                "explanation": "Universal religious affirmation Amen repeated across dialogue turns identical in Danish and Swedish"
            }
        ]
    })
    candidates = [
        {"id": 390, "target": "- Amen.\n- Amen."}
    ]
    verified_ids = validate_semantic_invariant_verification_output(raw_verifier_resp, candidates, show_title="Godland")
    assert verified_ids == {390}


def test_positive_5_mixed_punctuation_formatting_and_shared_numbers():
    """Positive Control 5: Mixed punctuation, formatting, and numeric compounds ('... 4. Ja.', '<i>12:30. OK.</i>')."""
    # Pure structural numbers & symbols evaluate deterministically
    assert is_pure_structural_invariant("... 4.")
    assert is_pure_structural_invariant("12:30")

    # Mixed with shared lexical residual evaluates cleanly via semantic verification
    raw_verifier_resp = json.dumps({
        "results": [
            {"id": 13, "invariant_in_target": True, "explanation": "Structural leading ellipsis and number with shared affirmative particle"},
            {"id": 14, "invariant_in_target": True, "explanation": "Timecode with universal affirmative OK in formatting tags"}
        ]
    })
    candidates = [
        {"id": 13, "target": "... 4. Ja."},
        {"id": 14, "target": "<i>12:30. OK.</i>"}
    ]
    verified = validate_semantic_invariant_verification_output(raw_verifier_resp, candidates)
    assert verified == {13, 14}


def test_positive_6_unicode_non_latin_scripts_without_latin_titlecase_assumptions():
    """Positive Control 6: Non-Latin / Unicode scripts (Japanese, Arabic, Cyrillic, Chinese)."""
    raw_verifier_resp = json.dumps({
        "results": [
            {"id": 101, "invariant_in_target": True, "explanation": "Japanese loanword phrase preserved unchanged in target localization"},
            {"id": 102, "invariant_in_target": True, "explanation": "Arabic universal greeting preserved in context"},
            {"id": 103, "invariant_in_target": True, "explanation": "Cyrillic brand / proper entity preserved in target"}
        ]
    })
    candidates = [
        {"id": 101, "target": "さようなら"},
        {"id": 102, "target": "سلام"},
        {"id": 103, "target": "Яндекс"}
    ]
    verified = validate_semantic_invariant_verification_output(raw_verifier_resp, candidates)
    assert verified == {101, 102, 103}


def test_negative_7_complete_untranslated_source_sentence_fails_closed():
    """Negative Control 7: Vanlig komplett oöversatt source-mening ('Ich komme morgen.')."""
    raw_verifier_resp = json.dumps({
        "results": [
            {
                "id": 501,
                "invariant_in_target": False,
                "explanation": "German conversational dialogue sentence requiring translation to Swedish ('Jag kommer imorgon.')"
            },
            {
                "id": 502,
                "invariant_in_target": False,
                "explanation": "English conversational command requiring translation ('Kom hit.')"
            }
        ]
    })
    candidates = [
        {"id": 501, "target": "Ich komme morgen."},
        {"id": 502, "target": "Come here right now."}
    ]
    verified = validate_semantic_invariant_verification_output(raw_verifier_resp, candidates)
    # Must fail closed: neither ID is in verified_ids
    assert verified == set()


def test_negative_8_entity_plus_translatable_source_dialogue_fails_closed():
    """Negative Control 8: Entity + translatable source dialogue ('Lukas, ich komme morgen.')."""
    raw_verifier_resp = json.dumps({
        "results": [
            {
                "id": 601,
                "invariant_in_target": False,
                "explanation": "Proper noun Lukas is an entity, but the residual clause 'ich komme morgen' is translatable German dialogue that must be localized"
            }
        ]
    })
    candidates = [
        {"id": 601, "target": "Lukas, ich komme morgen."}
    ]
    verified = validate_semantic_invariant_verification_output(raw_verifier_resp, candidates)
    # Must fail closed: residual is translatable, so entire line is rejected from invariant set
    assert verified == set()


def test_negative_9_false_friend_and_invalid_identical_token_fails_closed():
    """Negative Control 9: False friend where identical spelling is NOT a valid target rendering."""
    raw_verifier_resp = json.dumps({
        "results": [
            {
                "id": 701,
                "invariant_in_target": False,
                "explanation": "German 'Gift' means poison, not gift/present, and must be translated"
            },
            {
                "id": 702,
                "invariant_in_target": False,
                "explanation": "French 'pain' means bread and cannot be left untranslated as pain"
            }
        ]
    })
    candidates = [
        {"id": 701, "target": "Das ist Gift."},
        {"id": 702, "target": "Un pain."}
    ]
    verified = validate_semantic_invariant_verification_output(raw_verifier_resp, candidates)
    assert verified == set()


def test_negative_10_unknown_or_uncertain_semantic_content_fails_closed():
    """Negative Control 10: Unknown / uncertain semantic content / schema errors fail closed."""
    # 1. Missing explanation
    raw_bad_1 = json.dumps({
        "results": [{"id": 801, "invariant_in_target": True, "explanation": ""}]
    })
    assert validate_semantic_invariant_verification_output(raw_bad_1, [{"id": 801, "target": "Mystery Word"}]) == set()

    # 2. Invariant is False
    raw_bad_2 = json.dumps({
        "results": [{"id": 802, "invariant_in_target": False, "explanation": "Uncertain whether this is a name"}]
    })
    assert validate_semantic_invariant_verification_output(raw_bad_2, [{"id": 802, "target": "Dubious Word"}]) == set()

    # 3. Corrupt JSON / unparseable output
    raw_corrupt = "This is not json at all."
    assert validate_semantic_invariant_verification_output(raw_corrupt, [{"id": 803, "target": "Corrupt"}]) == set()


@pytest.mark.asyncio
async def test_end_to_end_classify_and_recover_identical_preserves_verified_cues():
    """
    End-to-end integration test of classify_and_recover_identical with semantic verification:
    - Verifies that verified entity sequences, non-verbal sounds, and entity+shared residuals are returned as 'keep' with 'verified_' reason.
    - Verifies that translatable dialogue is returned as 'translate' with cleared text.
    """
    translator = SubtitleTranslator()

    # Mock initial classifier response: flags ambiguous lines for verification
    mock_clf_resp = json.dumps({
        "results": [
            {"id": 1, "action": "keep", "reason": "proper_noun", "text": "Héðinn, Beinir, Ari, Kári."},
            {"id": 2, "action": "keep", "reason": "non_verbal", "text": "<i>(Klick)</i>"},
            {"id": 3, "action": "keep", "reason": "proper_noun", "text": "Ja, Lukas."},
            {"id": 4, "action": "keep", "reason": "proper_noun", "text": "Lukas, ich komme morgen."}
        ]
    })

    # Mock batch verifier response: accepts items 1, 2, 3 but rejects item 4 (translatable residual)
    mock_verifier_resp = {1, 2, 3}

    items = [
        {"id": 1, "text": "Héðinn, Beinir, Ari, Kári."},
        {"id": 2, "text": "<i>(Klick)</i>"},
        {"id": 3, "text": "Ja, Lukas."},
        {"id": 4, "text": "Lukas, ich komme morgen."}
    ]
    source_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="Héðinn, Beinir, Ari, Kári."),
        srt.Subtitle(index=2, start=timedelta(seconds=3), end=timedelta(seconds=4), content="<i>(Klick)</i>"),
        srt.Subtitle(index=3, start=timedelta(seconds=5), end=timedelta(seconds=6), content="Ja, Lukas."),
        srt.Subtitle(index=4, start=timedelta(seconds=7), end=timedelta(seconds=8), content="Lukas, ich komme morgen.")
    ]
    trans_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="Héðinn, Beinir, Ari, Kári."),
        srt.Subtitle(index=2, start=timedelta(seconds=3), end=timedelta(seconds=4), content="<i>(Klick)</i>"),
        srt.Subtitle(index=3, start=timedelta(seconds=5), end=timedelta(seconds=6), content="Ja, Lukas."),
        srt.Subtitle(index=4, start=timedelta(seconds=7), end=timedelta(seconds=8), content="Lukas, ich komme morgen.")
    ]

    # Patch verify_alphabetic_invariants_batch to return mock_verifier_resp
    with patch.object(translator, "verify_alphabetic_invariants_batch", new=AsyncMock(return_value=mock_verifier_resp)):
        with patch("app.services.translator.get_setting", return_value="gemini"):
            with patch("google.genai.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_resp = MagicMock()
                mock_resp.text = mock_clf_resp
                mock_resp.usage_metadata = None
                mock_client.models.generate_content.return_value = mock_resp
                mock_client_cls.return_value = mock_client

                results = await translator.classify_and_recover_identical(
                    items,
                    target_language="Swedish",
                    show_title="Regression Suite",
                    source_subs=source_subs,
                    translated_subs=trans_subs,
                    source_language="German"
                )

                res_map = {r["id"]: r for r in results}
                assert res_map[1]["action"] == "keep"
                assert "verified" in res_map[1]["reason"]
                assert res_map[2]["action"] == "keep"
                assert "verified" in res_map[2]["reason"]
                assert res_map[3]["action"] == "keep"
                assert "verified" in res_map[3]["reason"]
                assert res_map[4]["action"] == "translate"
                assert res_map[4]["text"] == ""


def test_pipeline_qa_gate_accepts_verified_invariants_and_rejects_untranslated():
    """
    Verifies that qa_gate correctly accepts safe_ids with context_verified_ids
    for Godland names, Teachers (Klick), and Teachers 'Ja, Lukas.', while failing
    untranslated German dialogue 'Lukas, ich komme morgen.'
    """
    src_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="<i>Bárður, Þórður, Böðvar, Lúðvík,</i>"),
        srt.Subtitle(index=2, start=timedelta(seconds=3), end=timedelta(seconds=4), content="<i>(Klick)</i>"),
        srt.Subtitle(index=3, start=timedelta(seconds=5), end=timedelta(seconds=6), content="Ja, Lukas."),
        srt.Subtitle(index=4, start=timedelta(seconds=7), end=timedelta(seconds=8), content="Lukas, ich komme morgen.")
    ]
    trans_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="<i>Bárður, Þórður, Böðvar, Lúðvík,</i>"),
        srt.Subtitle(index=2, start=timedelta(seconds=3), end=timedelta(seconds=4), content="<i>(Klick)</i>"),
        srt.Subtitle(index=3, start=timedelta(seconds=5), end=timedelta(seconds=6), content="Ja, Lukas."),
        srt.Subtitle(index=4, start=timedelta(seconds=7), end=timedelta(seconds=8), content="Lukas, ich komme morgen.")
    ]
    safe_ids = [0, 1, 2]
    context_verified_ids = {0, 1, 2}

    qa_res = qa_gate(
        source_subs=src_subs,
        translated_subs=trans_subs,
        target_lang_code="sv",
        source_language_name="German",
        safe_ids=safe_ids,
        context_verified_ids=context_verified_ids,
        allow_warnings=False
    )

    # Cue index 3 (4th cue) is untranslated dialogue and MUST be caught
    assert qa_res["passed"] is False
    assert len(qa_res["real_untranslated_ids"]) == 1
    assert qa_res["real_untranslated_ids"][0] == 3


@pytest.mark.asyncio
async def test_adversarial_reason_prefix_cannot_grant_authority_without_semantic_provenance(tmp_path, monkeypatch):
    """
    Krav 5 (Full Production Execution):
    Negativt integrationstest som kör den faktiska SubtitlePipeline.process_video_file.
    När primary classifier returnerar:
        action="keep", reason="verified_proper_noun", semantic_verified=False
    för verkligt oöversatt dialog ("Ich komme morgen."),
    MÅSTE den verkliga produktions-recovery-logiken i pipeline.py avvisa KEEP,
    logga "QA Recovery: Rejected unsafe KEEP...", och misslyckas i QA.
    """
    test_db = str(tmp_path / "test_adv_provenance.db")
    monkeypatch.setattr("app.core.db.DB_PATH", test_db)
    from app.core.db import init_db, get_job_by_id
    init_db()

    def mock_get_setting(key, default=""):
        settings = {
            "ai_provider": "gemini",
            "gemini_api_key": "dummy_key",
            "gemini_model": "gemini-3.5-flash-lite",
            "batch_size": "50",
            "max_concurrent_jobs": "1",
            "wait_time_seconds": "0",
            "extract_target_embedded": "false",
            "extract_source_embedded": "false",
            "languages": json.dumps([{"code": "sv", "name": "Swedish", "enabled": True}]),
        }
        return settings.get(key, default)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)
    monkeypatch.setattr("app.services.translator.get_setting", mock_get_setting)

    video_path = tmp_path / "AdversarialShow.mkv"
    video_path.touch()
    en_srt = tmp_path / "AdversarialShow.en.srt"

    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="Ich komme morgen."),
        srt.Subtitle(index=2, start=timedelta(seconds=3), end=timedelta(seconds=4), content="This is a second line of dialogue for testing size."),
        srt.Subtitle(index=3, start=timedelta(seconds=5), end=timedelta(seconds=6), content="And a third line to ensure the file exceeds 100 bytes.")
    ]
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()

    # 1. Main translation returns identical text for cue 0, and translated text for 1 and 2
    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        return [
            {"id": 0, "text": "Ich komme morgen."},
            {"id": 1, "text": "Detta är en andra dialograd för testning."},
            {"id": 2, "text": "Och en tredje rad för att säkerställa storlek."}
        ]

    # 2. Recovery classifier tries to spoof authority with reason="verified_proper_noun" but NO semantic_verified flag
    async def mock_classify(items, lang, title, **kwargs):
        return [{
            "id": 0,
            "action": "keep",
            "reason": "verified_proper_noun",
            "semantic_verified": False,
            "text": "Ich komme morgen."
        }]

    # 3. Fast rescue returns nothing
    async def mock_rescue_batch(items, target_language, source_language="English", show_title="", attempt=1, job_id=None):
        return []

    # 4. Escalate returns identical
    async def mock_escalate_single(*args, **kwargs):
        return "Ich komme morgen."

    monkeypatch.setattr(pipeline.translator, "first_pass_micro_repair_batch", AsyncMock(return_value=[]))
    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(pipeline.translator, "fast_final_rescue_batch", mock_rescue_batch)
    monkeypatch.setattr(pipeline.translator, "escalate_single_line", mock_escalate_single)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")

    job = get_job_by_id(res["job_id"])
    logs = [l.get("message", "") if isinstance(l, dict) else str(l) for l in job.get("logs", [])]

    # Prove production recovery gate rejected unsafe KEEP and logged it:
    assert any("Rejected unsafe KEEP for cue 1 ('Ich komme morgen.'). Forcing translation." in l for l in logs), \
        f"Expected rejection log not found in job logs: {logs}"

    # Prove job was NOT marked as successfully TRANSLATED (fails QA gate)
    assert job["status"] in {"QA_FAILED", "TRANSLATED_WITH_WARNINGS", "FAILED"}


@pytest.mark.asyncio
async def test_legitimate_semantic_provenance_grants_safe_status(tmp_path, monkeypatch):
    """
    Krav 6 (Full Production Execution):
    Positivt integrationstest som kör den faktiska SubtitlePipeline.process_video_file.
    När den faktiska semantic verifiern godkänner cuen och sätter semantic_verified=True,
    MÅSTE den verkliga produktions-recovery-logiken i pipeline.py acceptera cuen som
    (Semantic-verified Invariant), lägga den i safe_ids/context_verified_ids, och passera QA.
    """
    test_db = str(tmp_path / "test_legit_provenance.db")
    monkeypatch.setattr("app.core.db.DB_PATH", test_db)
    from app.core.db import init_db, get_job_by_id
    init_db()

    def mock_get_setting(key, default=""):
        settings = {
            "ai_provider": "gemini",
            "gemini_api_key": "dummy_key",
            "gemini_model": "gemini-3.5-flash-lite",
            "batch_size": "50",
            "max_concurrent_jobs": "1",
            "wait_time_seconds": "0",
            "extract_target_embedded": "false",
            "extract_source_embedded": "false",
            "languages": json.dumps([{"code": "sv", "name": "Swedish", "enabled": True}]),
        }
        return settings.get(key, default)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)
    monkeypatch.setattr("app.services.translator.get_setting", mock_get_setting)

    video_path = tmp_path / "GodlandShow.mkv"
    video_path.touch()
    en_srt = tmp_path / "GodlandShow.en.srt"

    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="Héðinn, Beinir, Ari, Kári."),
        srt.Subtitle(index=2, start=timedelta(seconds=3), end=timedelta(seconds=4), content="This is a second line of dialogue for testing size."),
        srt.Subtitle(index=3, start=timedelta(seconds=5), end=timedelta(seconds=6), content="And a third line to ensure the file exceeds 100 bytes.")
    ]
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()

    # 1. Main translation returns identical name list for cue 0, and translated text for 1 and 2
    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        return [
            {"id": 0, "text": "Héðinn, Beinir, Ari, Kári."},
            {"id": 1, "text": "Detta är en andra dialograd för testning."},
            {"id": 2, "text": "Och en tredje rad för att säkerställa storlek."}
        ]

    # 2. Mock classify to return raw candidate flagged for verification
    # and mock verify_single_occurrence_entities to return verified ID 0
    async def mock_classify_and_recover(items, target_language, show_title="", source_subs=None, translated_subs=None, **kwargs):
        # Calls the real verification logic or returns structured proven result
        return [{
            "id": 0,
            "action": "keep",
            "reason": "verified_proper_noun",
            "semantic_verified": True,
            "text": "Héðinn, Beinir, Ari, Kári."
        }]

    monkeypatch.setattr(pipeline.translator, "first_pass_micro_repair_batch", AsyncMock(return_value=[]))
    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify_and_recover)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")

    job = get_job_by_id(res["job_id"])
    logs = [l.get("message", "") if isinstance(l, dict) else str(l) for l in job.get("logs", [])]

    # Prove production recovery gate accepted legitimate semantic verification and logged it:
    assert any("Semantic-verified Invariant: 'Héðinn, Beinir, Ari, Kári.'" in l for l in logs), \
        f"Expected acceptance log not found in job logs: {logs}"

    # Prove job passed QA and completed as TRANSLATED
    assert job["status"] == "TRANSLATED"
