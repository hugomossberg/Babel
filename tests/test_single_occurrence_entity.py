import pytest
import os
import json
import srt
from datetime import timedelta, datetime, timezone
from unittest.mock import patch, MagicMock

import app.core.db as db
from app.core.db import init_db, create_job, get_job_by_id, update_job, set_setting
from app.services.pipeline import SubtitlePipeline, qa_gate
from app.services.translator import (
    SubtitleTranslator,
    is_strictly_valid_entity_candidate,
    validate_entity_verification_output,
    validate_classifier_output,
    has_entity_evidence,
    is_meaningful_translation,
    ProviderUnavailableError
)


@pytest.fixture(autouse=True)
def setup_teardown_db(tmp_path):
    orig_db = db.DB_PATH
    test_db = str(tmp_path / "test_single_entity.db")
    db.DB_PATH = test_db
    init_db()
    set_setting("target_language", "sv")
    set_setting("languages", json.dumps([{"code": "sv", "name": "Swedish", "enabled": True}]))
    set_setting("gemini_api_key", "mock-gemini-key")
    yield tmp_path
    db.clear_all_jobs()
    db.DB_PATH = orig_db


def test_strictly_valid_entity_candidate_adversarial():
    """Adversarial check: single-occurrence candidates must pass strict lexical rules."""
    # Production case Morgan & other valid single entity names
    assert is_strictly_valid_entity_candidate("Morgan?") is True
    assert is_strictly_valid_entity_candidate("Morgan") is True
    assert is_strictly_valid_entity_candidate("Kowalski") is True
    assert is_strictly_valid_entity_candidate("Jessica") is True
    assert is_strictly_valid_entity_candidate("Cam, Reggie.") is True

    # Common words MUST be rejected deterministically
    assert is_strictly_valid_entity_candidate("May?") is False
    assert is_strictly_valid_entity_candidate("Will?") is False
    assert is_strictly_valid_entity_candidate("Rose?") is False
    assert is_strictly_valid_entity_candidate("Bear?") is False
    assert is_strictly_valid_entity_candidate("Help?") is False
    assert is_strictly_valid_entity_candidate("Office?") is False
    assert is_strictly_valid_entity_candidate("Friends?") is False
    assert is_strictly_valid_entity_candidate("Good.") is False
    assert is_strictly_valid_entity_candidate("Bows. We're bowing.") is False

    # Lowercase, empty, placeholders
    assert is_strictly_valid_entity_candidate("morgan") is False
    assert is_strictly_valid_entity_candidate("") is False
    assert is_strictly_valid_entity_candidate("<i></i>") is False
    assert is_strictly_valid_entity_candidate("A") is False # single char


def test_validate_entity_verification_output_fail_closed():
    """Structured AI classifier output validation must be strictly fail-closed."""
    candidates = [{"id": 344, "target": "Morgan?"}]

    # Valid HIGH confidence named entity
    valid_json = json.dumps({
        "results": [{
            "id": 344,
            "verdict": "NAMED_ENTITY",
            "entity_type": "PERSON_NAME",
            "confidence": "HIGH",
            "explanation": "Reporter addressed at press conference"
        }]
    })
    assert validate_entity_verification_output(valid_json, candidates) == {344}

    # Medium or low confidence -> rejected
    med_json = json.dumps({
        "results": [{
            "id": 344,
            "verdict": "NAMED_ENTITY",
            "entity_type": "PERSON_NAME",
            "confidence": "MEDIUM"
        }]
    })
    assert validate_entity_verification_output(med_json, candidates) == set()

    # Translatable text verdict -> rejected
    trans_json = json.dumps({
        "results": [{
            "id": 344,
            "verdict": "TRANSLATABLE_TEXT",
            "entity_type": "NOT_AN_ENTITY",
            "confidence": "HIGH"
        }]
    })
    assert validate_entity_verification_output(trans_json, candidates) == set()

    # Ambiguous -> rejected
    amb_json = json.dumps({
        "results": [{
            "id": 344,
            "verdict": "AMBIGUOUS",
            "entity_type": "NOT_AN_ENTITY",
            "confidence": "LOW"
        }]
    })
    assert validate_entity_verification_output(amb_json, candidates) == set()

    # Adversarial: model hallucinating high confidence on common English word
    bad_candidate = [{"id": 10, "target": "Help?"}]
    adversarial_json = json.dumps({
        "results": [{
            "id": 10,
            "verdict": "NAMED_ENTITY",
            "entity_type": "PERSON_NAME",
            "confidence": "HIGH"
        }]
    })
    # Deterministic candidate check blocks it fail-closed
    assert validate_entity_verification_output(adversarial_json, bad_candidate) == set()


def test_same_run_evidence_and_substring_rejection():
    """Same-run entity evidence requires translated context and rejects substring/source-copy."""
    source_subs = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Boston."),
        srt.Subtitle(2, timedelta(seconds=5), timedelta(seconds=6), "I love Boston very much.")
    ]
    # Valid same-run evidence: cue 2 is meaningfully translated into Swedish mentioning Boston
    trans_valid = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Boston."),
        srt.Subtitle(2, timedelta(seconds=5), timedelta(seconds=6), "Jag älskar Boston väldigt mycket.")
    ]
    assert has_entity_evidence("Boston.", source_subs, trans_valid, target_idx=0) is True

    # Source-copy: cue 2 untranslated in English -> cannot create evidence
    trans_untranslated = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Boston."),
        srt.Subtitle(2, timedelta(seconds=5), timedelta(seconds=6), "I love Boston very much.")
    ]
    assert has_entity_evidence("Boston.", source_subs, trans_untranslated, target_idx=0) is False

    # Substring evidence rejection: "Cat" inside "Catch" must not count as evidence
    source_cat = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Cat."),
        srt.Subtitle(2, timedelta(seconds=5), timedelta(seconds=6), "Catch the ball.")
    ]
    trans_cat = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Cat."),
        srt.Subtitle(2, timedelta(seconds=5), timedelta(seconds=6), "Fånga bollen.")
    ]
    assert has_entity_evidence("Cat.", source_cat, trans_cat, target_idx=0) is False


@pytest.mark.asyncio
async def test_morgan_production_case_e2e(tmp_path):
    """
    E2E test matching Survivor's Remorse S03E02 cue 344 'Morgan?'.
    Context:
    341: Good. [SV: Bra.]
    342: ♪ (becomes <i></i>)
    343: [reporters clamoring] (becomes <i></i>)
    344: Morgan?
    345: Cam, 51 points.
    346: 10 out of 12 three-pointers.
    347: Tell us what you were feeling.
    """
    video_path = tmp_path / "Survivors.Remorse.S03E02.mkv"
    video_path.touch()

    subs = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Good."),
        srt.Subtitle(2, timedelta(seconds=2), timedelta(seconds=3), "♪"),
        srt.Subtitle(3, timedelta(seconds=3), timedelta(seconds=4), "[reporters clamoring]"),
        srt.Subtitle(4, timedelta(seconds=4), timedelta(seconds=5), "Morgan?"),
        srt.Subtitle(5, timedelta(seconds=5), timedelta(seconds=6), "Cam, 51 points."),
        srt.Subtitle(6, timedelta(seconds=6), timedelta(seconds=7), "10 out of 12 three-pointers."),
        srt.Subtitle(7, timedelta(seconds=7), timedelta(seconds=8), "Tell us what you were feeling.")
    ]
    en_srt_path = str(video_path).replace(".mkv", ".en.srt")
    with open(en_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()

    # Pass 1 translates dialogue, but keeps Morgan? identical
    pass1_translated = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Bra."),
        srt.Subtitle(2, timedelta(seconds=2), timedelta(seconds=3), "<i></i>"),
        srt.Subtitle(3, timedelta(seconds=3), timedelta(seconds=4), "<i></i>"),
        srt.Subtitle(4, timedelta(seconds=4), timedelta(seconds=5), "Morgan?"),
        srt.Subtitle(5, timedelta(seconds=5), timedelta(seconds=6), "Cam, 51 poäng."),
        srt.Subtitle(6, timedelta(seconds=6), timedelta(seconds=7), "10 av 12 trepoängare."),
        srt.Subtitle(7, timedelta(seconds=7), timedelta(seconds=8), "Berätta vad du kände.")
    ]

    async def mock_translate_srt(*args, **kwargs):
        import copy
        return copy.deepcopy(pass1_translated)

    # Primary classifier proposes keep proper_noun
    # SubtitleTranslator.verify_single_occurrence_entities verifies Morgan as PERSON_NAME HIGH confidence
    async def mock_classify_identical(items, target_lang, show_title="", source_subs=None, translated_subs=None):
        return [{"id": 3, "action": "keep", "reason": "context_verified_proper_noun", "text": "Morgan?"}]

    with patch("app.services.pipeline.find_external_subtitle", return_value=en_srt_path), \
         patch.object(pipeline.translator, "translate_srt_content", side_effect=mock_translate_srt), \
         patch.object(pipeline.translator, "classify_and_recover_identical", side_effect=mock_classify_identical), \
         patch.object(pipeline, "trigger_bazarr_search"):

        job_id = create_job(str(video_path))
        res = await pipeline.process_video_file(str(video_path), job_id=job_id, force_retranslate=True)

        assert res["status"] == "translated"
        job = get_job_by_id(job_id)
        assert job["status"] == "TRANSLATED"
        sv_srt = str(video_path).replace(".mkv", ".sv.srt")
        assert os.path.exists(sv_srt)

        # Invariant checks: 0 dropped, 0 drift
        with open(sv_srt, "r", encoding="utf-8") as f:
            out_subs = list(srt.parse(f.read()))
        assert len(out_subs) == len(subs)
        assert out_subs[3].content == "Morgan?"


@pytest.mark.asyncio
async def test_semantic_deadlock_fails_closed_no_worker_retry(tmp_path):
    """
    Semantic Deadlock:
    When provider calls succeed but bounded recovery is exhausted with 0 progress,
    the job must FAIL CLOSED (status = FAILED), not published, next_retry_at = None,
    and must not trigger worker retry loops.
    """
    video_path = tmp_path / "deadlock_test.mkv"
    video_path.touch()

    subs = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Hello"),
        srt.Subtitle(2, timedelta(seconds=2), timedelta(seconds=3), "What did you do?"),
    ]
    en_srt_path = str(video_path).replace(".mkv", ".en.srt")
    with open(en_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()

    # Pass 1 leaves line 2 untranslated
    pass1_translated = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Hej"),
        srt.Subtitle(2, timedelta(seconds=2), timedelta(seconds=3), "What did you do?"),
    ]

    async def mock_translate_srt(*args, **kwargs):
        import copy
        return copy.deepcopy(pass1_translated)

    async def mock_classify(*args, **kwargs):
        return [{"id": 1, "action": "translate", "reason": "none", "text": ""}]

    async def mock_batch(*args, **kwargs):
        # Stubbornly returns original English
        return [{"id": 1, "text": "What did you do?"}]

    async def mock_rescue(*args, **kwargs):
        return [{"id": 1, "text": "What did you do?"}]

    with patch("app.services.pipeline.find_external_subtitle", return_value=en_srt_path), \
         patch.object(pipeline.translator, "translate_srt_content", side_effect=mock_translate_srt), \
         patch.object(pipeline.translator, "classify_and_recover_identical", side_effect=mock_classify), \
         patch.object(pipeline.translator, "translate_batch", side_effect=mock_batch), \
             patch.object(pipeline.translator, "escalate_single_line", return_value="What did you do?"), \
         patch.object(pipeline.translator, "fast_final_rescue_batch", side_effect=mock_rescue), \
         patch.object(pipeline, "trigger_bazarr_search"):

        job_id = create_job(str(video_path))
        res = await pipeline.process_video_file(str(video_path), job_id=job_id, force_retranslate=True)

        # Must fail closed as FAILED (not RECOVERING)
        assert res["status"] == "failed"
        job = get_job_by_id(job_id)
        assert job["status"] == "FAILED"
        assert job["next_retry_at"] is None
        assert not os.path.exists(str(video_path).replace(".mkv", ".sv.srt"))


@pytest.mark.asyncio
async def test_transient_provider_error_sets_waiting_provider_with_retry(tmp_path):
    """
    Transient Provider Error:
    When provider raises ProviderUnavailableError (e.g. 429, timeout, network error),
    job must enter WAITING_PROVIDER with next_retry_at set for worker pickup.
    """
    video_path = tmp_path / "transient_test.mkv"
    video_path.touch()

    subs = [srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Hello")]
    en_srt_path = str(video_path).replace(".mkv", ".en.srt")
    with open(en_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()

    async def mock_translate_transient(*args, **kwargs):
        raise ProviderUnavailableError("Gemini API rate limited: HTTP 429")

    with patch("app.services.pipeline.find_external_subtitle", return_value=en_srt_path), \
         patch.object(pipeline.translator, "translate_srt_content", side_effect=mock_translate_transient), \
         patch.object(pipeline, "trigger_bazarr_search"):

        job_id = create_job(str(video_path))
        res = await pipeline.process_video_file(str(video_path), job_id=job_id, force_retranslate=True)

        assert res["status"] == "waiting_provider"
        job = get_job_by_id(job_id)
        assert job["status"] == "WAITING_PROVIDER"
        assert job["next_retry_at"] is not None
        assert job["retry_count"] == 1
