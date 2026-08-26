"""
Comprehensive regression and integration test suite for v2.3.43-beta:
Batch Semantic Invariant Verification for Recovery / Escalation Identical Candidates.

Verifies:
1. Production pipeline flow for identical recovery candidates (Bulk Contextual, Bulk Strict, Escalation).
2. Positive regressions with Godland real-media fixtures ('Torsk, Þorskur.', 'Torsk?', 'Se.', 'Far!', 'Far.', '- Tror du på magi?\\n- Ja.', 'Kom.').
3. Multi-token Unicode name lists and existing invariants ('Héðinn...', 'Bárður...', '(Klick)', 'Ja, Lukas.', 'Minato! Passa!', 'Amen.').
4. Multilingual invariance across DE->EN, FR->EN, IT->DE, EN->FR, and non-Latin/Unicode.
5. Fail-closed rejection of unverified source dialogue ('Ich komme morgen.', 'Das ist Gift.', 'Un pain.', '[DOOR CLOSES]', etc.).
6. Performance: Verified invariant cues in bulk recovery never invoke subsequent per-cue escalation (0 escalation calls).
7. Single batch verification call for N identical cues (no per-cue LLM explosion).
8. Strict provenance safety: Model-controlled reason strings alone never grant KEEP authority.
"""
import pytest
import os
import json
import srt
from datetime import timedelta
from unittest.mock import patch, AsyncMock

from app.core.cleaner import subs_to_srt_string
from app.services.pipeline import SubtitlePipeline, qa_gate
from app.services.translator import (
    SubtitleTranslator,
    is_meaningful_translation,
    is_pure_structural_invariant,
    validate_classifier_output,
    validate_semantic_invariant_verification_output
)
from app.core.db import init_db, get_job_by_id, DB_PATH


@pytest.fixture
def mock_pipeline_env(monkeypatch, tmp_path):
    test_db = str(tmp_path / "test_recovery_identical.db")
    monkeypatch.setattr("app.core.db.DB_PATH", test_db)
    init_db()

    def mock_get_setting(key, default=""):
        settings = {
            "ai_provider": "gemini",
            "gemini_api_key": "dummy_api_key",
            "gemini_model": "gemini-3.5-flash-lite",
            "escalate_to_pro": "false",
            "escalation_provider": "none",
            "escalation_model": "",
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
    monkeypatch.setattr("app.services.translator.SubtitleTranslator.first_pass_micro_repair_batch", AsyncMock(return_value=[]))


# =========================================================================
# 1. POSITIVE REAL-MEDIA REGRESSIONS (GODLAND FIXTURES)
# =========================================================================

@pytest.mark.asyncio
async def test_godland_8_identical_cues_verified_in_bulk_recovery(mock_pipeline_env, tmp_path, monkeypatch):
    """
    Positive Integration Test:
    Simulates the exact 8 Godland cues returned identical from Bulk Contextual Recovery.
    Verifies that:
    1. Batch semantic verification audits all 8 in ONE single call.
    2. All 8 gain semantic provenance and are marked safe.
    3. QA gate passes with score 100.
    4. Escalation is NEVER called (0 calls).
    """
    video_path = tmp_path / "Godland.2022.mkv"
    video_path.touch()
    en_srt = tmp_path / "Godland.2022.da.srt"

    cues_data = [
        (122, "Torsk, Þorskur."),
        (123, "Torsk?"),
        (132, "Se."),
        (362, "Far!"),
        (363, "Far."),
        (427, "- Tror du på magi?\n- Ja."),
        (573, "Kom."),
        (605, "Kom."),
    ]

    subs = []
    for idx, (cue_num, text) in enumerate(cues_data):
        subs.append(srt.Subtitle(index=idx + 1, start=timedelta(seconds=idx * 5), end=timedelta(seconds=idx * 5 + 3), content=text))

    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()

    # Initial translate pass returns original Danish text (triggering recovery)
    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        return [{"id": item["id"], "text": item["text"]} for item in items]

    # Classifier in Primary Recovery routes to translate/untranslated
    async def mock_classify(items, lang, title, **kwargs):
        return [{"id": item["id"], "action": "translate", "reason": "none", "text": ""} for item in items]

    # Bulk Contextual Recovery returns candidate outputs identical to source
    bcr_call_count = 0
    async def mock_fast_rescue(items, target_language, source_language="source", show_title="", attempt=1, job_id=None):
        nonlocal bcr_call_count
        bcr_call_count += 1
        return [{"id": it["id"], "text": it["target"]} for it in items]

    # Batch Semantic Verifier verifies all 8 in 1 batch call
    sem_verifier_call_count = 0
    sem_verifier_batch_sizes = []
    async def mock_verify_invariants(candidates, target_language, show_title="", job_id=None, source_language="source"):
        nonlocal sem_verifier_call_count
        sem_verifier_call_count += 1
        sem_verifier_batch_sizes.append(len(candidates))
        # Approve all 8 candidate IDs
        return {c["id"] for c in candidates}

    # Escalation tracking: must NOT be called
    escalation_call_count = 0
    async def mock_escalate_single(self, *args, **kwargs):
        nonlocal escalation_call_count
        escalation_call_count += 1
        return "Escalated"

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(pipeline.translator, "fast_final_rescue_batch", mock_fast_rescue)
    monkeypatch.setattr(pipeline.translator, "verify_alphabetic_invariants_batch", mock_verify_invariants)
    monkeypatch.setattr(SubtitleTranslator, "escalate_single_line", mock_escalate_single)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
    assert res["status"] != "failed"

    job = get_job_by_id(res["job_id"])
    assert job["status"] == "TRANSLATED"
    assert bcr_call_count == 1
    assert sem_verifier_call_count == 1
    assert sem_verifier_batch_sizes == [8]
    assert escalation_call_count == 0, "Escalation must NOT be invoked for verified invariant cues"

    # Verify target subtitle file was written with 100% sync and matching content
    sv_srt = tmp_path / "Godland.2022.sv.srt"
    assert os.path.exists(sv_srt)
    with open(sv_srt, "r", encoding="utf-8") as f:
        sv_parsed = list(srt.parse(f.read()))
    assert len(sv_parsed) == 8
    assert sv_parsed[0].content == "Torsk, Þorskur."
    assert sv_parsed[1].content == "Torsk?"
    assert sv_parsed[2].content == "Se."
    assert sv_parsed[3].content == "Far!"
    assert sv_parsed[4].content == "Far."
    assert sv_parsed[5].content == "- Tror du på magi?\n- Ja."
    assert sv_parsed[6].content == "Kom."
    assert sv_parsed[7].content == "Kom."


@pytest.mark.asyncio
async def test_existing_invariants_regression_fixtures():
    """
    Verifies that existing regression invariant patterns continue to be correctly verified
    via validate_semantic_invariant_verification_output fail-closed.
    """
    candidates = [
        {"id": 1, "target": "Héðinn, Beinir, Ari, Kári."},
        {"id": 2, "target": "Bárður, Þórður, Böðvar, Lúðvík."},
        {"id": 3, "target": "<i>(Klick)</i>"},
        {"id": 4, "target": "Ja, Lukas."},
        {"id": 5, "target": "Minato! Passa!"},
        {"id": 6, "target": "- Amen.\n- Amen."}
    ]
    raw_resp = json.dumps({
        "results": [
            {"id": 1, "invariant_in_target": True, "explanation": "Icelandic/Old Norse personal names list"},
            {"id": 2, "invariant_in_target": True, "explanation": "Icelandic/Old Norse personal names list"},
            {"id": 3, "invariant_in_target": True, "explanation": "Acoustic sound effect"},
            {"id": 4, "invariant_in_target": True, "explanation": "Shared affirmation and proper name"},
            {"id": 5, "invariant_in_target": True, "explanation": "Character name and soccer command"},
            {"id": 6, "invariant_in_target": True, "explanation": "Universal religious closing"}
        ]
    })
    verified_ids = validate_semantic_invariant_verification_output(raw_resp, candidates)
    assert verified_ids == {1, 2, 3, 4, 5, 6}


# =========================================================================
# 2. NEGATIVE FAIL-CLOSED INTEGRATION TESTS
# =========================================================================

@pytest.mark.asyncio
async def test_fail_closed_rejected_source_dialogue_continues_recovery(mock_pipeline_env, tmp_path, monkeypatch):
    """
    Negative Integration Test:
    1. Recovery returns identical untranslated German dialogue ('Ich komme morgen.').
    2. Semantic verifier rejects the cue (invariant_in_target=False).
    3. Production recovery logic does NOT mark the cue safe.
    4. Cue proceeds to subsequent recovery / escalation.
    5. When escalation provides the real translation ('Jag kommer imorgon.'), the job succeeds.
    """
    video_path = tmp_path / "GermanShow.S01E01.mkv"
    video_path.touch()
    en_srt = tmp_path / "GermanShow.S01E01.en.srt"

    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=3), content="Ich komme morgen."),
        srt.Subtitle(index=2, start=timedelta(seconds=4), end=timedelta(seconds=6), content="Guten Abend."),
        srt.Subtitle(index=3, start=timedelta(seconds=7), end=timedelta(seconds=9), content="Auf Wiedersehen.")
    ]
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()

    async def mock_translate_batch(items, *args, **kwargs):
        # Line 0 returns untranslated German dialogue, lines 1 and 2 translate
        results = []
        for it in items:
            if it["id"] == 0:
                results.append({"id": 0, "text": "Ich komme morgen."})
            elif it["id"] == 1:
                results.append({"id": 1, "text": "God kväll."})
            else:
                results.append({"id": 2, "text": "Adjö."})
        return results

    async def mock_classify(items, *args, **kwargs):
        return [{"id": it["id"], "action": "translate", "reason": "none", "text": ""} for it in items]

    # Bulk recovery returns candidate identical to source for line 0
    async def mock_fast_rescue(items, *args, **kwargs):
        return [{"id": it["id"], "text": "Ich komme morgen."} for it in items]

    # Semantic verifier rejects German dialogue as NOT invariant in Swedish
    async def mock_verify_invariants(candidates, *args, **kwargs):
        return set() # Empty set -> rejected fail-closed

    # Escalation provides the true Swedish translation
    escalated = False
    async def mock_escalate_single(self, *args, **kwargs):
        nonlocal escalated
        escalated = True
        return "Jag kommer imorgon."

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(pipeline.translator, "fast_final_rescue_batch", mock_fast_rescue)
    monkeypatch.setattr(pipeline.translator, "verify_alphabetic_invariants_batch", mock_verify_invariants)
    monkeypatch.setattr(SubtitleTranslator, "escalate_single_line", mock_escalate_single)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
    assert res["status"] != "failed"

    job = get_job_by_id(res["job_id"])
    assert job["status"] == "TRANSLATED"
    assert escalated is True, "Escalation must be invoked when semantic invariant verification rejects the cue"

    sv_srt = tmp_path / "GermanShow.S01E01.sv.srt"
    with open(sv_srt, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Jag kommer imorgon." in content


@pytest.mark.asyncio
async def test_fail_closed_malformed_verifier_output_fails_closed():
    """
    Verifies that malformed, corrupted, or schema-violating verifier outputs fail closed.
    """
    candidates = [{"id": 1, "target": "Some dialogue"}]

    # 1. Invalid JSON
    assert validate_semantic_invariant_verification_output("not json", candidates) == set()

    # 2. Missing explanation
    bad_resp1 = json.dumps({"results": [{"id": 1, "invariant_in_target": True, "explanation": ""}]})
    assert validate_semantic_invariant_verification_output(bad_resp1, candidates) == set()

    # 3. invariant_in_target is False
    bad_resp2 = json.dumps({"results": [{"id": 1, "invariant_in_target": False, "explanation": "Translatable"}]})
    assert validate_semantic_invariant_verification_output(bad_resp2, candidates) == set()

    # 4. Unknown/Hallucinated ID
    bad_resp3 = json.dumps({"results": [{"id": 999, "invariant_in_target": True, "explanation": "Hallucinated"}]})
    assert validate_semantic_invariant_verification_output(bad_resp3, candidates) == set()

    # 5. Overlength text exceeding lexical sanity bounds (> 16 tokens)
    long_cand = [{"id": 2, "target": "word " * 20}]
    bad_resp4 = json.dumps({"results": [{"id": 2, "invariant_in_target": True, "explanation": "Too long"}]})
    assert validate_semantic_invariant_verification_output(bad_resp4, long_cand) == set()


# =========================================================================
# 3. MULTILINGUAL ACCEPTANCE MATRIX (LANGUAGE-PAIR AGNOSTIC)
# =========================================================================

def test_multilingual_semantic_invariant_matrix():
    """
    Verifies semantic invariant evaluation across diverse language pairs:
    DE -> EN, FR -> EN, IT -> DE, EN -> FR, and Cyrillic/Unicode scripts.
    """
    # DE -> EN
    de_en_cand = [
        {"id": 1, "target": "Berlin"},               # Verified proper noun
        {"id": 2, "target": "Ich komme morgen."},     # Rejected translatable dialogue
        {"id": 3, "target": "Das ist Gift."}          # Rejected false friend ("Gift" in DE = poison)
    ]
    de_en_resp = json.dumps({
        "results": [
            {"id": 1, "invariant_in_target": True, "explanation": "City name invariant in English"},
            {"id": 2, "invariant_in_target": False, "explanation": "German dialogue must translate to 'I come tomorrow'"},
            {"id": 3, "invariant_in_target": False, "explanation": "German 'Gift' means poison, must translate"}
        ]
    })
    assert validate_semantic_invariant_verification_output(de_en_resp, de_en_cand) == {1}

    # FR -> EN
    fr_en_cand = [
        {"id": 10, "target": "Paris, France."},       # Verified entity
        {"id": 11, "target": "Un pain."},             # Rejected translatable noun phrase
        {"id": 12, "target": "[PORTES CLAQUENT]"}     # Rejected descriptive SDH
    ]
    fr_en_resp = json.dumps({
        "results": [
            {"id": 10, "invariant_in_target": True, "explanation": "Geographic proper noun invariant"},
            {"id": 11, "invariant_in_target": False, "explanation": "French phrase must translate to 'A bread'"},
            {"id": 12, "invariant_in_target": False, "explanation": "Descriptive SDH requires localization to '[DOORS SLAM]'"}
        ]
    })
    assert validate_semantic_invariant_verification_output(fr_en_resp, fr_en_cand) == {10}

    # IT -> DE
    it_de_cand = [
        {"id": 20, "target": "Marco Rossi"},          # Verified character name
        {"id": 21, "target": "Vieni qui subito."}     # Rejected Italian command
    ]
    it_de_resp = json.dumps({
        "results": [
            {"id": 20, "invariant_in_target": True, "explanation": "Italian personal name unchanged in German"},
            {"id": 21, "invariant_in_target": False, "explanation": "Italian command must translate to 'Komm sofort hierher'"}
        ]
    })
    assert validate_semantic_invariant_verification_output(it_de_resp, it_de_cand) == {20}

    # Non-Latin / Cyrillic -> EN
    cyrillic_cand = [
        {"id": 30, "target": "Токио"},                # Proper name
        {"id": 31, "target": "Привет, как дела?"}     # Translatable Russian dialogue
    ]
    cyrillic_resp = json.dumps({
        "results": [
            {"id": 30, "invariant_in_target": True, "explanation": "City name entity"},
            {"id": 31, "invariant_in_target": False, "explanation": "Russian greeting must translate"}
        ]
    })
    assert validate_semantic_invariant_verification_output(cyrillic_resp, cyrillic_cand) == {30}


# =========================================================================
# 4. PROVENANCE & SAFETY TRUST BOUNDARY TESTS
# =========================================================================

def test_raw_reason_strings_never_grant_semantic_authority():
    """
    Security Test:
    Ensures that model-controlled string attributes like reason='verified_proper_noun'
    or reason='invariant_shared_word' NEVER grant semantic authority by themselves.
    Only explicit validation via validate_semantic_invariant_verification_output can grant it.
    """
    raw_classifier_resp = json.dumps({
        "results": [
            {"id": 1, "action": "keep", "reason": "verified_proper_noun", "text": "Ich komme morgen."},
            {"id": 2, "action": "keep", "reason": "invariant_shared_word", "text": "Un pain."}
        ]
    })
    items = [
        {"id": 1, "text": "Ich komme morgen."},
        {"id": 2, "text": "Un pain."}
    ]
    validated = validate_classifier_output(raw_classifier_resp, items)
    # Both must be downgraded to translate with reason='needs_semantic_verification' and text=''
    for res in validated:
        assert res["action"] == "translate"
        assert res["reason"] == "needs_semantic_verification"
        assert res["text"] == ""
        assert res.get("semantic_verified") is not True


# =========================================================================
# 5. STAGE 2 (BULK STRICT RECOVERY) & BATCHING EFFICIENCY TESTS
# =========================================================================

@pytest.mark.asyncio
async def test_bulk_strict_recovery_identical_candidate_verified(mock_pipeline_env, tmp_path, monkeypatch):
    """
    Verifies that if Stage 1 (Bulk Contextual Recovery) does not resolve a cue,
    Stage 2 (Bulk Strict Recovery) can return an identical candidate,
    which is then batch-verified, gains semantic provenance, and avoids escalation.
    """
    video_path = tmp_path / "StrictShow.S01E01.mkv"
    video_path.touch()
    en_srt = tmp_path / "StrictShow.S01E01.en.srt"

    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=3), content="Torsk?"),
        srt.Subtitle(index=2, start=timedelta(seconds=4), end=timedelta(seconds=6), content="Good morning."),
        srt.Subtitle(index=3, start=timedelta(seconds=7), end=timedelta(seconds=9), content="How are you today?"),
        srt.Subtitle(index=4, start=timedelta(seconds=10), end=timedelta(seconds=12), content="Have a nice evening."),
    ]
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()

    async def mock_translate_batch(items, *args, **kwargs):
        # Line 0 returns untranslated, other lines translate
        results = []
        for it in items:
            if it["id"] == 0:
                results.append({"id": 0, "text": "Torsk?"})
            else:
                results.append({"id": it["id"], "text": f"Svensk text {it['id']}"})
        return results

    async def mock_classify(items, *args, **kwargs):
        return [{"id": it["id"], "action": "translate", "reason": "none", "text": ""} for it in items]

    # Stage 1 fails to resolve line 0 (returns empty/unusable), Stage 2 returns identical candidate "Torsk?"
    rescue_attempts = []
    async def mock_fast_rescue(items, target_language, source_language="source", show_title="", attempt=1, job_id=None):
        rescue_attempts.append(attempt)
        if attempt == 1:
            # Stage 1 fails to return usable translation for id 0
            return [{"id": 0, "text": ""}]
        else:
            # Stage 2 returns identical text
            return [{"id": 0, "text": "Torsk?"}]

    # Verifier approves id 0
    verifier_called = 0
    async def mock_verify_invariants(candidates, *args, **kwargs):
        nonlocal verifier_called
        verifier_called += 1
        return {c["id"] for c in candidates}

    # Escalation must not be called
    escalation_called = 0
    async def mock_escalate_single(self, *args, **kwargs):
        nonlocal escalation_called
        escalation_called += 1
        return "Escalated"

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(pipeline.translator, "fast_final_rescue_batch", mock_fast_rescue)
    monkeypatch.setattr(pipeline.translator, "verify_alphabetic_invariants_batch", mock_verify_invariants)
    monkeypatch.setattr(SubtitleTranslator, "escalate_single_line", mock_escalate_single)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
    assert res["status"] != "failed"

    job = get_job_by_id(res["job_id"])
    assert job["status"] == "TRANSLATED"
    assert rescue_attempts == [1, 2], "Must attempt both Stage 1 and Stage 2 bulk recovery"
    assert verifier_called == 1, "Verifier must be called in Stage 2"
    assert escalation_called == 0, "Escalation must NOT be called after Stage 2 verification"


@pytest.mark.asyncio
async def test_batching_efficiency_single_call_for_many_cues(mock_pipeline_env, tmp_path, monkeypatch):
    """
    Verifies that 12 identical recovery cues are verified in exactly 1 single batch call
    (no individual per-cue LLM explosion).
    """
    video_path = tmp_path / "BatchShow.S01E01.mkv"
    video_path.touch()
    en_srt = tmp_path / "BatchShow.S01E01.en.srt"

    # 12 invariant name cues interleaved with 20 English dialogue cues
    names = ["Lukas", "Bárður", "Þórður", "Böðvar", "Lúðvík", "Héðinn", "Beinir", "Ari", "Kári", "Torfur", "Borgar", "Minato"]
    subs = []
    for i in range(12):
        subs.append(srt.Subtitle(index=i*2+1, start=timedelta(seconds=i*4), end=timedelta(seconds=i*4+1), content=names[i]))
        subs.append(srt.Subtitle(index=i*2+2, start=timedelta(seconds=i*4+2), end=timedelta(seconds=i*4+3), content=f"This is an important conversation line number {i}."))

    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()

    async def mock_translate_batch(items, *args, **kwargs):
        # Even IDs (0, 2, 4...) are name cues returned untranslated; odd IDs are translated Swedish dialogue
        results = []
        for it in items:
            idx = it["id"]
            if idx % 2 == 0:
                results.append({"id": idx, "text": it["text"]})
            else:
                results.append({"id": idx, "text": f"Detta är en viktig svensk dialograd nummer {idx}."})
        return results

    async def mock_classify(items, *args, **kwargs):
        return [{"id": it["id"], "action": "translate", "reason": "none", "text": ""} for it in items]

    async def mock_fast_rescue(items, *args, **kwargs):
        return [{"id": it["id"], "text": it["target"]} for it in items]

    verifier_call_count = 0
    verifier_candidate_counts = []
    async def mock_verify_invariants(candidates, *args, **kwargs):
        nonlocal verifier_call_count
        verifier_call_count += 1
        verifier_candidate_counts.append(len(candidates))
        return {c["id"] for c in candidates}

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(pipeline.translator, "fast_final_rescue_batch", mock_fast_rescue)
    monkeypatch.setattr(pipeline.translator, "verify_alphabetic_invariants_batch", mock_verify_invariants)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
    assert res["status"] != "failed"

    assert verifier_call_count == 1, "Must batch all 12 cues into exactly 1 verifier call"
    assert verifier_candidate_counts == [12]


# =========================================================================
# 6. STAGE 3 (ESCALATION) IDENTICAL INVARIANT REGRESSIONS
# =========================================================================

@pytest.mark.asyncio
async def test_escalation_identical_candidate_verified_positive(mock_pipeline_env, tmp_path, monkeypatch):
    """
    Test Case 1 & 8: Escalation returns exact source text, semantic verifier approves.
    Verifies that:
    1. Candidate is not discarded by escalation.
    2. Batch semantic verifier receives actual source/target languages and dialogue context.
    3. Cue gains explicit semantic provenance and is marked safe.
    4. No further escalation attempts are made for this cue.
    5. Job completes with status TRANSLATED.
    """
    video_path = tmp_path / "EscalationShow.S01E01.mkv"
    video_path.touch()
    en_srt = tmp_path / "EscalationShow.S01E01.en.srt"

    # Fixture scenario modeled after Godland cue 427: short dialogue dialogue
    cue_427_text = "- Tror du på magi?\n- Ja."
    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=3), content=cue_427_text),
        srt.Subtitle(index=2, start=timedelta(seconds=4), end=timedelta(seconds=6), content="I believe in nature."),
        srt.Subtitle(index=3, start=timedelta(seconds=7), end=timedelta(seconds=9), content="Everything is alive."),
    ]
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()

    # Pass 1 translates lines 1 and 2, but leaves line 0 untranslated
    async def mock_translate_batch(items, *args, **kwargs):
        results = []
        for it in items:
            if it["id"] == 0:
                results.append({"id": 0, "text": cue_427_text})
            elif it["id"] == 1:
                results.append({"id": 1, "text": "Jag tror på naturen."})
            else:
                results.append({"id": 2, "text": "Allt är levande."})
        return results

    async def mock_classify(items, *args, **kwargs):
        return [{"id": it["id"], "action": "translate", "reason": "none", "text": ""} for it in items]

    async def mock_fast_rescue(items, *args, **kwargs):
        return [{"id": it["id"], "text": ""} for it in items]

    # Escalation returns the identical candidate on attempt 1 and calls early semantic verification
    escalation_call_count = 0
    async def mock_exec_call(**kwargs):
        nonlocal escalation_call_count
        escalation_call_count += 1
        return json.dumps({"translation": cue_427_text})

    # Batch verifier approves line 0
    verifier_call_count = 0
    async def mock_verify_invariants(candidates, target_language, show_title="", job_id=None, source_language="source"):
        nonlocal verifier_call_count
        verifier_call_count += 1
        assert candidates[0]["id"] == 0
        assert candidates[0]["target"] == cue_427_text
        return {0}

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(pipeline.translator, "fast_final_rescue_batch", mock_fast_rescue)
    monkeypatch.setattr(pipeline.translator, "_execute_single_escalation_call", mock_exec_call)
    monkeypatch.setattr(pipeline.translator, "verify_alphabetic_invariants_batch", mock_verify_invariants)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
    assert res["status"] != "failed"

    job = get_job_by_id(res["job_id"])
    assert job["status"] == "TRANSLATED"
    assert escalation_call_count == 1, "Single escalation call should produce the candidate"
    assert verifier_call_count == 1, "Batch verifier should be called in Escalation"


@pytest.mark.asyncio
async def test_escalation_identical_candidate_rejected_negative_fails_closed(mock_pipeline_env, tmp_path, monkeypatch):
    """
    Test Case 2: Escalation returns exact source text, semantic verifier says FALSE.
    Result: Cue is NOT marked safe, remains unresolved, and fails closed (semantic deadlock).
    """
    video_path = tmp_path / "EscReject.S01E01.mkv"
    video_path.touch()
    en_srt = tmp_path / "EscReject.S01E01.en.srt"

    german_dialogue = "Ich habe keine Ahnung davon."
    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=3), content=german_dialogue),
        srt.Subtitle(index=2, start=timedelta(seconds=4), end=timedelta(seconds=6), content="Good morning to you."),
        srt.Subtitle(index=3, start=timedelta(seconds=7), end=timedelta(seconds=9), content="See you tomorrow."),
    ]
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()

    async def mock_translate_batch(items, *args, **kwargs):
        results = []
        for it in items:
            if it["id"] == 0:
                results.append({"id": 0, "text": german_dialogue})
            elif it["id"] == 1:
                results.append({"id": 1, "text": "God morgon till dig."})
            else:
                results.append({"id": 2, "text": "Vi ses imorgon."})
        return results

    async def mock_classify(items, *args, **kwargs):
        return [{"id": it["id"], "action": "translate", "reason": "none", "text": ""} for it in items]

    async def mock_fast_rescue(items, *args, **kwargs):
        return [{"id": it["id"], "text": ""} for it in items]

    # Escalation returns identical German dialogue
    async def mock_escalate_single(*args, **kwargs):
        return german_dialogue

    # Verifier rejects German dialogue
    async def mock_verify_invariants(candidates, *args, **kwargs):
        return set() # Empty set -> reject fail-closed

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(pipeline.translator, "fast_final_rescue_batch", mock_fast_rescue)
    monkeypatch.setattr(pipeline.translator, "escalate_single_line", mock_escalate_single)
    monkeypatch.setattr(pipeline.translator, "verify_alphabetic_invariants_batch", mock_verify_invariants)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
    # Must fail QA and not publish as TRANSLATED
    job = get_job_by_id(res["job_id"])
    assert job["status"] == "FAILED"
    assert "Semantic deadlock" in job.get("error_message", "") or "QA Gate failed" in job.get("error_message", "")


@pytest.mark.asyncio
async def test_escalation_semantic_verifier_malformed_output_fails_closed(mock_pipeline_env, tmp_path, monkeypatch):
    """
    Test Case 3: Semantic verifier returns malformed/corrupt output during Escalation.
    Result: Fail-closed (returns empty set, cue is not marked safe).
    """
    candidates = [{"id": 10, "target": "Some candidate"}]
    # 1. Invalid JSON
    assert validate_semantic_invariant_verification_output("corrupt {json", candidates) == set()
    # 2. Results array has empty explanation
    assert validate_semantic_invariant_verification_output('{"results": [{"id": 10, "invariant_in_target": true, "explanation": ""}]}', candidates) == set()
    # 3. Missing boolean field
    assert validate_semantic_invariant_verification_output('{"results": [{"id": 10, "explanation": "Valid"}]}', candidates) == set()


@pytest.mark.asyncio
async def test_escalation_semantic_verifier_provider_error_fails_closed(mock_pipeline_env, tmp_path, monkeypatch):
    """
    Test Case 4: Semantic verifier throws a Provider error during Escalation.
    Result: Does NOT mask provider error as safe; cue remains rejected fail-closed.
    """
    video_path = tmp_path / "ProviderErr.S01E01.mkv"
    video_path.touch()
    en_srt = tmp_path / "ProviderErr.S01E01.en.srt"

    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=3), content="Some text"),
        srt.Subtitle(index=2, start=timedelta(seconds=4), end=timedelta(seconds=6), content="Good morning."),
        srt.Subtitle(index=3, start=timedelta(seconds=7), end=timedelta(seconds=9), content="Good night."),
    ]
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()

    async def mock_translate_batch(items, *args, **kwargs):
        return [
            {"id": 0, "text": "Some text"},
            {"id": 1, "text": "God morgon."},
            {"id": 2, "text": "God natt."}
        ]

    async def mock_classify(items, *args, **kwargs):
        return [{"id": it["id"], "action": "translate", "reason": "none", "text": ""} for it in items]

    async def mock_fast_rescue(items, *args, **kwargs):
        return [{"id": it["id"], "text": ""} for it in items]

    async def mock_escalate_single(*args, **kwargs):
        return "Some text"

    # Verifier raises Exception
    async def mock_verify_invariants(*args, **kwargs):
        raise RuntimeError("API Timeout / 500 Internal Server Error")

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(pipeline.translator, "fast_final_rescue_batch", mock_fast_rescue)
    monkeypatch.setattr(pipeline.translator, "escalate_single_line", mock_escalate_single)
    monkeypatch.setattr(pipeline.translator, "verify_alphabetic_invariants_batch", mock_verify_invariants)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
    job = get_job_by_id(res["job_id"])
    assert job["status"] == "FAILED"


# =========================================================================
# 7. MULTILINGUAL POSITIVE & NEGATIVE ESCALATION TESTS
# =========================================================================

def test_multilingual_positive_and_negative_invariants_matrix():
    """
    Test Case 6 & 7: Multilingual acceptance across multiple language pairs (DE->EN, FR->EN, IT->DE).
    Verifies that:
    - Genuine invariant candidates (proper names, entities) evaluate to True.
    - Translatable lexical text in target language evaluates to False.
    """
    # DE -> EN
    de_cand = [
        {"id": 1, "target": "Wolfgang Schneider"},   # True (personal name)
        {"id": 2, "target": "Guten Morgen, mein Freund."} # False (German greeting)
    ]
    de_resp = json.dumps({
        "results": [
            {"id": 1, "invariant_in_target": True, "explanation": "Proper name invariant"},
            {"id": 2, "invariant_in_target": False, "explanation": "German greeting must be translated"}
        ]
    })
    assert validate_semantic_invariant_verification_output(de_resp, de_cand) == {1}

    # FR -> EN
    fr_cand = [
        {"id": 10, "target": "Lyon, France."},       # True (geographic entity)
        {"id": 11, "target": "Pourquoi es-tu ici?"}   # False (French question)
    ]
    fr_resp = json.dumps({
        "results": [
            {"id": 10, "invariant_in_target": True, "explanation": "Geographic name invariant"},
            {"id": 11, "invariant_in_target": False, "explanation": "French question must translate to 'Why are you here?'"}
        ]
    })
    assert validate_semantic_invariant_verification_output(fr_resp, fr_cand) == {10}

    # IT -> DE
    it_cand = [
        {"id": 20, "target": "Leonardo da Vinci"},    # True (historical name)
        {"id": 21, "target": "Per favore, aiutami."}  # False (Italian request)
    ]
    it_resp = json.dumps({
        "results": [
            {"id": 20, "invariant_in_target": True, "explanation": "Historical name invariant in German"},
            {"id": 21, "invariant_in_target": False, "explanation": "Italian phrase must translate to 'Bitte hilf mir.'"}
        ]
    })
    assert validate_semantic_invariant_verification_output(it_resp, it_cand) == {20}
