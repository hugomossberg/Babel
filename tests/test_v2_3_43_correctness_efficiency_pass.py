"""
Targeted regression test suite for Babel v2.3.43 correctness + efficiency pass.

Covers:
1. Legit identical invariant DE -> SV with name + lexical content ("Ja, Lukas.").
2. Legit identical invariant IT -> SV with mixed entity/lexical content ("Minato! Passa!").
3. Negative DE -> SV translatable dialogue ("Ich komme morgen.").
4. Negative IT -> SV translatable dialogue ("Perché sei qui?").
5. Other language pairs (ES -> EN, NL -> SV).
6. Non-Latin language pairs (Cyrillic -> EN, Japanese/CJK -> EN).
7. Escalation attempt 1 identical + verifier TRUE: skips attempts 2 and 3.
8. Escalation attempt 1 identical + verifier FALSE: continues to attempt 2.
9. Verifier malformed output: fail closed.
10. Verifier provider error: fail closed.
11. Fake reason/provenance: no authority without explicit verifier provenance.
12. UI drift metric 0 when all source/target timestamps are identical.
13. UI drift metric non-zero when timestamp is shifted.
14. Dropped > 0 when target subtitle is missing / empty for real source dialogue.
15. Modal metrics isolation by job_id in database.
"""
import pytest
import json
import srt
from datetime import timedelta

from app.core.validator import verify_sync, check_dropped_lines
from app.services.translator import (
    SubtitleTranslator,
    validate_semantic_invariant_verification_output,
)
from app.core.db import init_db, create_job, update_job, get_job_by_id


# =========================================================================
# 1-6. SEMANTIC INVARIANT VERIFICATION TESTS (POSITIVE & NEGATIVE)
# =========================================================================

def test_1_legit_identical_invariant_de_to_sv_name_and_lexical():
    """1. Legit identical invariant DE -> SV with name + lexical content ('Ja, Lukas.')."""
    candidates = [{"id": 42, "target": "Ja, Lukas."}]
    verifier_output = json.dumps({
        "results": [
            {
                "id": 42,
                "invariant_in_target": True,
                "explanation": "Proper name 'Lukas' and affirmative interjection 'Ja' are identically valid and natural in Swedish."
            }
        ]
    })
    verified_ids = validate_semantic_invariant_verification_output(verifier_output, candidates)
    assert verified_ids == {42}


def test_2_legit_identical_invariant_it_to_sv_mixed_entity_lexical():
    """2. Legit identical invariant IT -> SV with mixed entity/lexical content ('Minato! Passa!')."""
    candidates = [{"id": 88, "target": "Minato! Passa!"}]
    verifier_output = json.dumps({
        "results": [
            {
                "id": 88,
                "invariant_in_target": True,
                "explanation": "Character name 'Minato' and sports imperative 'Passa!' (pass the ball) are identically valid in Swedish."
            }
        ]
    })
    verified_ids = validate_semantic_invariant_verification_output(verifier_output, candidates)
    assert verified_ids == {88}


def test_2b_legit_identical_invariant_universal_single_word_call():
    """2b. Legit identical invariant FR -> EN single-word call / loanword ('Taxi!')."""
    candidates = [{"id": 401, "target": "Taxi!"}]
    verifier_output = json.dumps({
        "results": [
            {
                "id": 401,
                "invariant_in_target": True,
                "explanation": "Universal loanword and exclamation identically valid in English."
            }
        ]
    })
    verified_ids = validate_semantic_invariant_verification_output(verifier_output, candidates)
    assert verified_ids == {401}


def test_2c_conservative_rejection_of_ambiguous_single_word_fails_closed():
    """2c. Conservative verifier rejection of ambiguous standalone dialogue ('Passa!') fails closed."""
    candidates = [{"id": 1037, "target": "Passa!"}]
    verifier_output = json.dumps({
        "results": [
            {
                "id": 1037,
                "invariant_in_target": False,
                "explanation": "Ambiguous Italian verb without clear sports context treated as untranslated dialogue."
            }
        ]
    })
    verified_ids = validate_semantic_invariant_verification_output(verifier_output, candidates)
    assert verified_ids == set()


def test_3_negative_de_to_sv_translatable_dialogue():
    """3. Negative DE -> SV translatable dialogue ('Ich komme morgen.')."""
    candidates = [{"id": 101, "target": "Ich komme morgen."}]
    # Even if the verifier returns False or model returned identical text, it must be rejected
    verifier_output = json.dumps({
        "results": [
            {
                "id": 101,
                "invariant_in_target": False,
                "explanation": "Translatable German sentence that must be rendered as 'Jag kommer imorgon.'"
            }
        ]
    })
    verified_ids = validate_semantic_invariant_verification_output(verifier_output, candidates)
    assert verified_ids == set()


def test_4_negative_it_to_sv_translatable_dialogue():
    """4. Negative IT -> SV translatable dialogue ('Perché sei qui?')."""
    candidates = [{"id": 102, "target": "Perché sei qui?"}]
    verifier_output = json.dumps({
        "results": [
            {
                "id": 102,
                "invariant_in_target": False,
                "explanation": "Translatable Italian question that must be rendered as 'Varför är du här?'"
            }
        ]
    })
    verified_ids = validate_semantic_invariant_verification_output(verifier_output, candidates)
    assert verified_ids == set()


def test_5_other_language_pair_positive_and_negative():
    """5. Other language pairs (ES -> EN and NL -> SV)."""
    # ES -> EN
    es_candidates = [
        {"id": 1, "target": "San Francisco, California."},
        {"id": 2, "target": "¿Dónde está la biblioteca?"}
    ]
    es_output = json.dumps({
        "results": [
            {"id": 1, "invariant_in_target": True, "explanation": "Geographical location invariant in English"},
            {"id": 2, "invariant_in_target": False, "explanation": "Spanish question must translate to 'Where is the library?'"}
        ]
    })
    assert validate_semantic_invariant_verification_output(es_output, es_candidates) == {1}

    # NL -> SV
    nl_candidates = [
        {"id": 10, "target": "Amsterdam Centraal"},
        {"id": 11, "target": "Ik begrijp het niet."}
    ]
    nl_output = json.dumps({
        "results": [
            {"id": 10, "invariant_in_target": True, "explanation": "Station name invariant in Swedish"},
            {"id": 11, "invariant_in_target": False, "explanation": "Dutch dialogue must translate to 'Jag förstår inte.'"}
        ]
    })
    assert validate_semantic_invariant_verification_output(nl_output, nl_candidates) == {10}


def test_6_non_latin_language_pair():
    """6. Non-Latin language pairs (Cyrillic -> EN, Japanese/CJK -> EN)."""
    # Cyrillic (Russian) -> EN
    cyr_candidates = [
        {"id": 20, "target": "Анна Каренина"},
        {"id": 21, "target": "Я не знаю."}
    ]
    cyr_output = json.dumps({
        "results": [
            {"id": 20, "invariant_in_target": True, "explanation": "Literary title / proper name invariant"},
            {"id": 21, "invariant_in_target": False, "explanation": "Russian dialogue must translate to 'I don't know.'"}
        ]
    })
    assert validate_semantic_invariant_verification_output(cyr_output, cyr_candidates) == {20}

    # Japanese / CJK -> EN
    cjk_candidates = [
        {"id": 30, "target": "Tokyo Tower"},
        {"id": 31, "target": "おはようございます"}
    ]
    cjk_output = json.dumps({
        "results": [
            {"id": 30, "invariant_in_target": True, "explanation": "Landmark proper noun invariant"},
            {"id": 31, "invariant_in_target": False, "explanation": "Japanese greeting must translate to 'Good morning'"}
        ]
    })
    assert validate_semantic_invariant_verification_output(cjk_output, cjk_candidates) == {30}


# =========================================================================
# 7-10. ESCALATION EARLY VERIFICATION & FAIL-CLOSED TESTS
# =========================================================================

@pytest.mark.asyncio
async def test_7_escalation_attempt_1_identical_verifier_true_skips_attempt_2_3(monkeypatch):
    """
    7. Escalation attempt 1 identical + verifier TRUE:
       Returns candidate directly and skips attempts 2 and 3 (saving 2 LLM calls).
    """
    translator = SubtitleTranslator()
    exhausted = set()

    def mock_settings(k, d=""):
        if k == "ai_provider": return "gemini"
        return d

    monkeypatch.setattr("app.services.translator.get_setting", mock_settings)

    call_count = 0
    async def mock_exec_call(**kwargs):
        nonlocal call_count
        call_count += 1
        return json.dumps({"translation": "Ja, Lukas."})

    verifier_call_count = 0
    async def mock_verify_invariants(candidates, **kwargs):
        nonlocal verifier_call_count
        verifier_call_count += 1
        assert candidates[0]["target"] == "Ja, Lukas."
        return {candidates[0]["id"]} # Verified as True

    monkeypatch.setattr(translator, "_execute_single_escalation_call", mock_exec_call)
    monkeypatch.setattr(translator, "verify_alphabetic_invariants_batch", mock_verify_invariants)

    res = await translator.escalate_single_line(
        target_idx=5,
        target_text="Ja, Lukas.",
        prev_text="Hallo",
        next_text="Tschüss",
        target_language="Swedish",
        show_title="Teachers Lounge",
        is_real_untranslated=True,
        exhausted_strategies=exhausted,
        source_language="German"
    )

    assert res == "Ja, Lukas."
    assert call_count == 1, "Only Attempt 1 should execute; Attempts 2 and 3 must be skipped"
    assert verifier_call_count == 1, "Verifier called exactly once for Attempt 1 candidate"


@pytest.mark.asyncio
async def test_8_escalation_attempt_1_identical_verifier_false_continues_to_attempt_2(monkeypatch):
    """
    8. Escalation attempt 1 identical + verifier FALSE:
       Attempt 1 rejected -> continues to Attempt 2 (strict) which translates successfully.
    """
    translator = SubtitleTranslator()
    exhausted = set()

    def mock_settings(k, d=""):
        if k == "ai_provider": return "gemini"
        return d

    monkeypatch.setattr("app.services.translator.get_setting", mock_settings)

    call_count = 0
    async def mock_exec_call(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return json.dumps({"translation": "Ich komme morgen."}) # Identical (untranslated)
        else:
            return json.dumps({"translation": "Jag kommer imorgon."}) # Valid translation on Attempt 2

    verifier_call_count = 0
    async def mock_verify_invariants(candidates, **kwargs):
        nonlocal verifier_call_count
        verifier_call_count += 1
        return set() # Verifier says FALSE

    monkeypatch.setattr(translator, "_execute_single_escalation_call", mock_exec_call)
    monkeypatch.setattr(translator, "verify_alphabetic_invariants_batch", mock_verify_invariants)

    res = await translator.escalate_single_line(
        target_idx=5,
        target_text="Ich komme morgen.",
        prev_text="Hallo",
        next_text="Tschüss",
        target_language="Swedish",
        show_title="Test Show",
        is_real_untranslated=True,
        exhausted_strategies=exhausted,
        source_language="German"
    )

    assert res == "Jag kommer imorgon."
    assert call_count == 2, "Must proceed to Attempt 2 after Attempt 1 verification rejection"
    assert verifier_call_count == 1, "Verifier called once for Attempt 1"


@pytest.mark.asyncio
async def test_9_verifier_malformed_fails_closed(monkeypatch):
    """9. Verifier malformed output -> fails closed, cue is not verified."""
    candidates = [{"id": 7, "target": "Some dialogue"}]

    # 1. Non-JSON string
    assert validate_semantic_invariant_verification_output("INVALID JSON", candidates) == set()
    # 2. Empty string
    assert validate_semantic_invariant_verification_output("", candidates) == set()
    # 3. Missing results key
    assert validate_semantic_invariant_verification_output('{"other": 123}', candidates) == set()
    # 4. Invariant is True but missing explanation
    assert validate_semantic_invariant_verification_output('{"results": [{"id": 7, "invariant_in_target": true}]}', candidates) == set()


@pytest.mark.asyncio
async def test_10_verifier_provider_error_fails_closed(monkeypatch):
    """10. Verifier provider error -> caught fail-closed, does not verify cue."""
    translator = SubtitleTranslator()
    exhausted = set()

    def mock_settings(k, d=""):
        if k == "ai_provider": return "gemini"
        return d

    monkeypatch.setattr("app.services.translator.get_setting", mock_settings)

    async def mock_exec_call(**kwargs):
        return json.dumps({"translation": "Ich komme morgen."})

    async def mock_verify_invariants(candidates, **kwargs):
        raise RuntimeError("Transient 500 error in verifier")

    monkeypatch.setattr(translator, "_execute_single_escalation_call", mock_exec_call)
    monkeypatch.setattr(translator, "verify_alphabetic_invariants_batch", mock_verify_invariants)

    res = await translator.escalate_single_line(
        target_idx=5,
        target_text="Ich komme morgen.",
        prev_text="Hallo",
        next_text="Tschüss",
        target_language="Swedish",
        show_title="Test Show",
        is_real_untranslated=True,
        exhausted_strategies=exhausted,
        source_language="German"
    )

    # Provider error during verification must fail closed and return None
    assert res is None
    assert len(exhausted) == 3


def test_11_fake_reason_provenance_no_authority():
    """11. Fake reason strings (e.g. 'verified_invariant') alone never grant authority."""
    candidates = [{"id": 99, "target": "Unverified dialogue"}]
    # Even if reason claims to be verified, if invariant_in_target is False, it MUST NOT be approved
    fake_payload = json.dumps({
        "results": [
            {
                "id": 99,
                "invariant_in_target": False,
                "explanation": "verified_invariant proper_noun override"
            }
        ]
    })
    verified_ids = validate_semantic_invariant_verification_output(fake_payload, candidates)
    assert verified_ids == set()


# =========================================================================
# 12-15. SYNC DRIFT, DROPPED & UI METRIC AUDIT TESTS
# =========================================================================

def test_12_ui_drift_metric_zero_when_timestamps_identical():
    """12. UI drift metric 0 when all source/target timestamps are identical."""
    source_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1.0), end=timedelta(seconds=3.0), content="Hello"),
        srt.Subtitle(index=2, start=timedelta(seconds=4.5), end=timedelta(seconds=6.2), content="World"),
    ]
    target_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1.0), end=timedelta(seconds=3.0), content="Hej"),
        srt.Subtitle(index=2, start=timedelta(seconds=4.5), end=timedelta(seconds=6.2), content="Värld"),
    ]

    report = verify_sync(source_subs, target_subs)
    assert report["valid"] is True
    assert report["start_diff_ms"] == 0
    assert report["end_diff_ms"] == 0
    assert max(report["start_diff_ms"], report["end_diff_ms"]) == 0


def test_13_ui_drift_metric_non_zero_when_timestamp_shifted():
    """13. UI drift metric non-zero in a synthetic test where timestamp is shifted."""
    source_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1.000), end=timedelta(seconds=3.000), content="Hello"),
        srt.Subtitle(index=2, start=timedelta(seconds=4.000), end=timedelta(seconds=6.000), content="World"),
    ]
    target_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1.150), end=timedelta(seconds=3.000), content="Hej"), # 150ms start drift
        srt.Subtitle(index=2, start=timedelta(seconds=4.000), end=timedelta(seconds=6.250), content="Värld"), # 250ms end drift
    ]

    report = verify_sync(source_subs, target_subs)
    assert report["valid"] is False
    assert report["start_diff_ms"] == 150
    assert report["end_diff_ms"] == 250
    max_drift = max(report["start_diff_ms"], report["end_diff_ms"])
    assert max_drift == 250


def test_14_dropped_greater_than_zero_when_target_missing_cue():
    """14. Dropped > 0 when target subtitle is empty/tag for real source dialogue."""
    source_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=3), content="Real spoken line 1"),
        srt.Subtitle(index=2, start=timedelta(seconds=4), end=timedelta(seconds=6), content="Real spoken line 2"),
        srt.Subtitle(index=3, start=timedelta(seconds=7), end=timedelta(seconds=9), content="Real spoken line 3"),
    ]
    target_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=3), content="Översatt rad 1"),
        srt.Subtitle(index=2, start=timedelta(seconds=4), end=timedelta(seconds=6), content=""), # Dropped
        srt.Subtitle(index=3, start=timedelta(seconds=7), end=timedelta(seconds=9), content="<i></i>"), # Dropped (invisible tag for real dialogue)
    ]

    dropped_count, dropped_details = check_dropped_lines(source_subs, target_subs)
    assert dropped_count == 2
    assert len(dropped_details) == 2
    assert dropped_details[0]["index"] == 2
    assert dropped_details[0]["original"] == "Real spoken line 2"
    assert dropped_details[1]["index"] == 3
    assert dropped_details[1]["original"] == "Real spoken line 3"


def test_15_modal_metrics_job_id_isolation(tmp_path, monkeypatch):
    """15. Modal metrics: Exact job-specific values isolated by job_id in DB."""
    test_db = str(tmp_path / "test_jobs_isolation.db")
    monkeypatch.setattr("app.core.db.DB_PATH", test_db)
    init_db()

    job1_id = create_job(video_path="/path/to/movie_a.mkv", title="Movie A")
    job2_id = create_job(video_path="/path/to/movie_b.mkv", title="Movie B")

    # Job 1: Perfect 0ms drift, 0 dropped
    update_job(
        job_id=job1_id,
        status="TRANSLATED",
        sync_diff_ms=0,
        dropped_lines=0,
        duration_seconds=42.5,
        total_lines=1000
    )

    # Job 2: 350ms drift, 4 dropped
    update_job(
        job_id=job2_id,
        status="FAILED",
        sync_diff_ms=350,
        dropped_lines=4,
        duration_seconds=12.1,
        total_lines=850
    )

    j1 = get_job_by_id(job1_id)
    j2 = get_job_by_id(job2_id)

    assert j1["sync_diff_ms"] == 0
    assert j1["dropped_lines"] == 0
    assert j1["duration_seconds"] == 42.5
    assert j1["status"] == "TRANSLATED"

    assert j2["sync_diff_ms"] == 350
    assert j2["dropped_lines"] == 4
    assert j2["duration_seconds"] == 12.1
    assert j2["status"] == "FAILED"
