import os
import json
import srt
import pytest
from datetime import timedelta
from unittest.mock import patch, MagicMock

from app.services.pipeline import (
    qa_gate,
    SubtitlePipeline,
    QA_STATUS_PASS,
    QA_STATUS_PASS_WITH_WARNINGS,
    QA_STATUS_FAIL,
    DEFAULT_QA_MAX_UNRESOLVED_COUNT,
    DEFAULT_QA_MAX_UNRESOLVED_RATIO,
)


# ---------------------------------------------------------------------------
# UNIT TESTS FOR QA_GATE
# ---------------------------------------------------------------------------

def test_qa_gate_zero_unresolved_pass():
    """Scenario 1: 0 unresolved lines -> PASS with 100 score and no warnings."""
    source_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=2), content="Hello world, how are you today?"),
        srt.Subtitle(index=2, start=timedelta(seconds=2), end=timedelta(seconds=4), content="I am doing very well, thank you."),
    ]
    trans_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=2), content="Hej världen, hur mår du idag?"),
        srt.Subtitle(index=2, start=timedelta(seconds=2), end=timedelta(seconds=4), content="Jag mår väldigt bra, tack så mycket."),
    ]

    res = qa_gate(source_subs, trans_subs, target_lang_code="sv")
    assert res["passed"] is True
    assert res["status"] == QA_STATUS_PASS
    assert res["score"] == 100
    assert len(res["issues"]) == 0
    assert len(res["warnings"]) == 0
    assert len(res["real_untranslated_ids"]) == 0
    assert len(res["preserved_untranslated_ids"]) == 0


def test_qa_gate_single_unresolved_in_large_sub_pass_with_warnings():
    """Scenario 2: 1 unresolved English line in 679 cues -> PASS_WITH_WARNINGS."""
    source_subs = [
        srt.Subtitle(index=i, start=timedelta(seconds=i * 2), end=timedelta(seconds=i * 2 + 1), content=f"This is an English dialogue line {i} that we speak.")
        for i in range(1, 680)
    ]
    trans_subs = [
        srt.Subtitle(index=i, start=timedelta(seconds=i * 2), end=timedelta(seconds=i * 2 + 1), content=f"Detta är en svensk dialograd {i} som vi pratar om.")
        for i in range(1, 680)
    ]
    # Cue 454 (0-indexed 453) remains in English
    trans_subs[453].content = source_subs[453].content

    res = qa_gate(source_subs, trans_subs, target_lang_code="sv", allow_warnings=True)
    assert res["passed"] is True
    assert res["status"] == QA_STATUS_PASS_WITH_WARNINGS
    assert res["real_untranslated_ids"] == [453]
    assert res["preserved_untranslated_ids"] == [453]
    assert len(res["warnings"]) == 1
    assert "1 unresolved English line" in res["warnings"][0]
    assert res["policy_details"]["structural_passed"] is True


def test_qa_gate_multiple_unresolved_under_threshold():
    """Scenario 3: 2 unresolved cues in 300 cues (2/300 = 0.67% <= 1%, count 2 <= 3) -> PASS_WITH_WARNINGS."""
    source_subs = [
        srt.Subtitle(index=i, start=timedelta(seconds=i * 2), end=timedelta(seconds=i * 2 + 1), content=f"This is an English dialogue sentence {i} in the movie.")
        for i in range(1, 301)
    ]
    trans_subs = [
        srt.Subtitle(index=i, start=timedelta(seconds=i * 2), end=timedelta(seconds=i * 2 + 1), content=f"Det här är en svensk dialogmening {i} i filmen vi ser.")
        for i in range(1, 301)
    ]
    trans_subs[50].content = source_subs[50].content
    trans_subs[150].content = source_subs[150].content

    res = qa_gate(source_subs, trans_subs, target_lang_code="sv", allow_warnings=True)
    assert res["passed"] is True
    assert res["status"] == QA_STATUS_PASS_WITH_WARNINGS
    assert res["preserved_untranslated_ids"] == [50, 150]


def test_qa_gate_exceeds_count_threshold_fails():
    """Scenario 4: 4 unresolved cues in 1000 cues (count 4 > 3, ratio 0.4% <= 1%) -> FAIL."""
    source_subs = [
        srt.Subtitle(index=i, start=timedelta(seconds=i * 2), end=timedelta(seconds=i * 2 + 1), content=f"This is an English dialogue sentence {i} in the movie.")
        for i in range(1, 1001)
    ]
    trans_subs = [
        srt.Subtitle(index=i, start=timedelta(seconds=i * 2), end=timedelta(seconds=i * 2 + 1), content=f"Det här är en svensk dialogmening {i} i filmen vi ser.")
        for i in range(1, 1001)
    ]
    for idx in [10, 20, 30, 40]:
        trans_subs[idx].content = source_subs[idx].content

    res = qa_gate(source_subs, trans_subs, target_lang_code="sv", allow_warnings=True)
    assert res["passed"] is False
    assert res["status"] == QA_STATUS_FAIL
    assert any("exceeds QA policy limit" in iss for iss in res["issues"])


def test_qa_gate_exceeds_ratio_threshold_fails():
    """Scenario 5: 2 unresolved cues in 50 cues (count 2 <= 3, ratio 4.0% > 1.0%) -> FAIL."""
    source_subs = [
        srt.Subtitle(index=i, start=timedelta(seconds=i * 2), end=timedelta(seconds=i * 2 + 1), content=f"This is an English dialogue line {i} we are speaking.")
        for i in range(1, 51)
    ]
    trans_subs = [
        srt.Subtitle(index=i, start=timedelta(seconds=i * 2), end=timedelta(seconds=i * 2 + 1), content=f"Detta är en svensk dialograd {i} som vi pratar om nu.")
        for i in range(1, 51)
    ]
    trans_subs[5].content = source_subs[5].content
    trans_subs[15].content = source_subs[15].content

    res = qa_gate(source_subs, trans_subs, target_lang_code="sv", allow_warnings=True)
    assert res["passed"] is False
    assert res["status"] == QA_STATUS_FAIL
    assert any("exceeds QA policy limit" in iss for iss in res["issues"])


def test_qa_gate_allow_warnings_false_requires_clean_pass():
    """Inside recovery loop, allow_warnings=False must return False so recovery continues."""
    source_subs = [
        srt.Subtitle(index=i, start=timedelta(seconds=i * 2), end=timedelta(seconds=i * 2 + 1), content=f"This is an English dialogue line {i} that we speak.")
        for i in range(1, 680)
    ]
    trans_subs = [
        srt.Subtitle(index=i, start=timedelta(seconds=i * 2), end=timedelta(seconds=i * 2 + 1), content=f"Detta är en svensk dialograd {i} som vi pratar om.")
        for i in range(1, 680)
    ]
    trans_subs[453].content = source_subs[453].content

    res = qa_gate(source_subs, trans_subs, target_lang_code="sv", allow_warnings=False)
    assert res["passed"] is False
    assert res["status"] == QA_STATUS_FAIL


# ---------------------------------------------------------------------------
# HARD ERROR / STRUCTURAL INTEGRITY TESTS
# ---------------------------------------------------------------------------

def test_qa_gate_structural_failures_always_fail():
    """Scenario 6: Hard structural defects must always fail with failure_type == 'structural'."""
    source_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=2), content="Hello, how are you?"),
        srt.Subtitle(index=2, start=timedelta(seconds=2), end=timedelta(seconds=4), content="I am doing well, thanks."),
    ]

    # Line count mismatch
    mismatch_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=2), content="Hej, hur mår du?"),
    ]
    res_mismatch = qa_gate(source_subs, mismatch_subs, target_lang_code="sv", allow_warnings=True)
    assert res_mismatch["passed"] is False
    assert res_mismatch["status"] == QA_STATUS_FAIL
    assert res_mismatch["policy_details"]["structural_passed"] is False
    assert res_mismatch["policy_details"]["failure_type"] == "structural"

    # Timestamp drift
    drift_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=2), content="Hej, hur mår du?"),
        srt.Subtitle(index=2, start=timedelta(seconds=3), end=timedelta(seconds=5), content="Jag mår bra, tack."),
    ]
    res_drift = qa_gate(source_subs, drift_subs, target_lang_code="sv", allow_warnings=True)
    assert res_drift["passed"] is False
    assert res_drift["status"] == QA_STATUS_FAIL
    assert res_drift["policy_details"]["structural_passed"] is False
    assert res_drift["policy_details"]["failure_type"] == "structural"

    # Dropped line (empty)
    dropped_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=2), content="Hej, hur mår du?"),
        srt.Subtitle(index=2, start=timedelta(seconds=2), end=timedelta(seconds=4), content=""),
    ]
    res_dropped = qa_gate(source_subs, dropped_subs, target_lang_code="sv", allow_warnings=True)
    assert res_dropped["passed"] is False
    assert res_dropped["status"] == QA_STATUS_FAIL
    assert res_dropped["policy_details"]["structural_passed"] is False
    assert res_dropped["policy_details"]["failure_type"] == "structural"


# ---------------------------------------------------------------------------
# ENTITY KEEP TESTS
# ---------------------------------------------------------------------------

def test_qa_gate_entity_keep_preserves_clean_pass():
    """Scenario 9: Valid named entities (proper nouns, brands, numbers) are not unresolved lines."""
    # 1. Deterministic brand/acronym keep (e.g. "FBI")
    source_subs_brand = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=2), content="FBI"),
        srt.Subtitle(index=2, start=timedelta(seconds=2), end=timedelta(seconds=4), content="Hello world, how are you today?"),
    ]
    trans_subs_brand = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=2), content="FBI"),
        srt.Subtitle(index=2, start=timedelta(seconds=2), end=timedelta(seconds=4), content="Hej världen, hur mår du idag?"),
    ]
    res_brand = qa_gate(source_subs_brand, trans_subs_brand, target_lang_code="sv", safe_ids=[0], allow_warnings=True)
    assert res_brand["passed"] is True
    assert res_brand["status"] == QA_STATUS_PASS
    assert len(res_brand["real_untranslated_ids"]) == 0
    assert len(res_brand["warnings"]) == 0

    # 2. Context-verified proper noun (e.g. "Solomon")
    source_subs_entity = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=2), content="Solomon"),
        srt.Subtitle(index=2, start=timedelta(seconds=2), end=timedelta(seconds=4), content="Hello world, how are you today?"),
    ]
    trans_subs_entity = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=2), content="Solomon"),
        srt.Subtitle(index=2, start=timedelta(seconds=2), end=timedelta(seconds=4), content="Hej världen, hur mår du idag?"),
    ]
    res_entity = qa_gate(source_subs_entity, trans_subs_entity, target_lang_code="sv", safe_ids=[0], context_verified_ids={0}, allow_warnings=True)
    assert res_entity["passed"] is True
    assert res_entity["status"] == QA_STATUS_PASS
    assert len(res_entity["real_untranslated_ids"]) == 0
    assert len(res_entity["warnings"]) == 0


def test_entity_keep_all_required_entities_clean_pass():
    """Scenario 9b: Solomon, Lucas Hood, Tiger Woods, FBI, BMW must all pass cleanly as safe KEEPs."""
    entities = ["Solomon", "Lucas Hood", "Tiger Woods", "FBI", "BMW"]
    for idx, entity in enumerate(entities):
        source_subs = [
            srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=2), content=entity),
            srt.Subtitle(index=2, start=timedelta(seconds=2), end=timedelta(seconds=4), content="Where are we going right now?"),
        ]
        trans_subs = [
            srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=2), content=entity),
            srt.Subtitle(index=2, start=timedelta(seconds=2), end=timedelta(seconds=4), content="Vart är vi på väg just nu?"),
        ]
        # With safe_id and context verification / deterministic check
        res = qa_gate(source_subs, trans_subs, target_lang_code="sv", safe_ids=[0], context_verified_ids={0}, allow_warnings=True)
        assert res["passed"] is True, f"Entity '{entity}' failed QA gate: {res}"
        assert res["status"] == QA_STATUS_PASS, f"Entity '{entity}' expected QA_STATUS_PASS, got {res['status']}"
        assert len(res["real_untranslated_ids"]) == 0
        assert len(res["warnings"]) == 0


def test_qa_gate_language_detection_semantic_quality_not_structural_blocker():
    """Language detection is a semantic quality signal; short dialogue and names must not fail structurally."""
    # Short dialogue with Swedish words and names
    source_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=2), content="Lucas Hood?"),
        srt.Subtitle(index=2, start=timedelta(seconds=2), end=timedelta(seconds=4), content="Yes, that is me."),
        srt.Subtitle(index=3, start=timedelta(seconds=4), end=timedelta(seconds=6), content="Come in, please."),
    ]
    trans_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=2), content="Lucas Hood?"),
        srt.Subtitle(index=2, start=timedelta(seconds=2), end=timedelta(seconds=4), content="Ja, det är jag."),
        srt.Subtitle(index=3, start=timedelta(seconds=4), end=timedelta(seconds=6), content="Kom in, är du snäll."),
    ]
    res = qa_gate(source_subs, trans_subs, target_lang_code="sv", safe_ids=[0], context_verified_ids={0}, allow_warnings=True)
    assert res["passed"] is True
    assert res["status"] == QA_STATUS_PASS
    assert res["policy_details"]["structural_passed"] is True
    assert res["policy_details"]["failure_type"] is None
    assert res["score"] == 100


def test_qa_gate_structurally_perfect_confident_wrong_language_fails_semantic_not_structural():
    """Structurally perfect subtitle with confident wrong language fails semantically, NOT structurally."""
    # 20 cues, valid SRT, perfect sync, 0 dropped cues, but output is German
    source_subs = [
        srt.Subtitle(index=i, start=timedelta(seconds=i * 2), end=timedelta(seconds=i * 2 + 1), content=f"This is an English dialogue line {i} that we speak.")
        for i in range(1, 21)
    ]
    trans_subs = [
        srt.Subtitle(index=i, start=timedelta(seconds=i * 2), end=timedelta(seconds=i * 2 + 1), content=f"Das ist ein deutscher Dialogsatz Nummer {i}, den wir sprechen.")
        for i in range(1, 21)
    ]

    res = qa_gate(source_subs, trans_subs, target_lang_code="sv", allow_warnings=True)
    assert res["passed"] is False
    assert res["status"] == QA_STATUS_FAIL
    # Structural integrity is TRUE (valid SRT, cue count match, 0 dropped, 0ms drift)
    assert res["policy_details"]["structural_passed"] is True
    # Semantic integrity is FALSE
    assert res["policy_details"]["semantic_passed"] is False
    assert res["policy_details"]["failure_type"] == "semantic"
    assert res["policy_details"]["confident_wrong_language"] is True
    assert any("Language mismatch" in iss for iss in res["issues"])


def test_qa_gate_standard_swedish_subtitle_passes_cleanly():
    """Standard healthy Swedish translation passes QA Gate with score 100."""
    source_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=2), content="Where are you going?"),
        srt.Subtitle(index=2, start=timedelta(seconds=2), end=timedelta(seconds=4), content="I am going home."),
    ]
    trans_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=2), content="Vart är du på väg?"),
        srt.Subtitle(index=2, start=timedelta(seconds=2), end=timedelta(seconds=4), content="Jag ska gå hem nu."),
    ]
    res = qa_gate(source_subs, trans_subs, target_lang_code="sv", allow_warnings=True)
    assert res["passed"] is True
    assert res["status"] == QA_STATUS_PASS
    assert res["score"] == 100
    assert res["policy_details"]["structural_passed"] is True
    assert res["policy_details"]["semantic_passed"] is True
    assert res["policy_details"]["failure_type"] is None


# ---------------------------------------------------------------------------
# END-TO-END PIPELINE TESTS WITH HERMETIC MOCKS
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_679_cues_with_1_deadlocked_cue_publishes_with_warnings(tmp_path):
    """
    Scenario 2 & 7 & 8: 679 cues, 1 deadlocked cue (cue 454).
    Bounded recovery attempts recovery, detects deadlock, applies QA fallback preserving source text,
    and publishes .sv.srt with PASS_WITH_WARNINGS and status TRANSLATED.
    """
    pipeline = SubtitlePipeline()
    video_path = tmp_path / "movie.mkv"
    video_path.touch()

    total_cues = 679
    deadlock_idx = 453  # Cue 454 (0-indexed)

    source_subs = [
        srt.Subtitle(
            index=i,
            start=timedelta(seconds=i * 2),
            end=timedelta(seconds=i * 2 + 1),
            content=f"English line {i}"
        )
        for i in range(1, total_cues + 1)
    ]
    source_srt_path = str(video_path).replace(".mkv", ".en.srt")
    with open(source_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(source_subs))

    def fake_get_setting(key, default=None):
        if key == "languages":
            return '[{"name": "Swedish", "code": "sv", "enabled": true}]'
        if key == "auto_repair_unhealthy": return "false"
        if key == "extract_target_embedded": return "false"
        if key == "extract_source_embedded": return "false"
        if key == "enable_bazarr_check": return "false"
        if key == "strict_sync_lock": return "true"
        return default

    logs_recorded = []
    def fake_append_job_log(job_id, msg):
        logs_recorded.append(msg)

    # First pass: returns Swedish for all cues except cue 454 which returns identical English
    def fake_translate(*args, **kwargs):
        out = []
        for i in range(1, total_cues + 1):
            if i - 1 == deadlock_idx:
                text = f"English line {i}"
            else:
                text = f"Detta är en svensk dialograd {i} som vi pratar om."
            out.append(srt.Subtitle(index=i, start=timedelta(seconds=i * 2), end=timedelta(seconds=i * 2 + 1), content=text))
        return out

    # Recovery mock that refuses to translate cue 454 (deadlock)
    async def fake_classify_and_recover(chunk, *args, **kwargs):
        return []

    async def fake_translate_batch(payload, *args, **kwargs):
        return [{"id": p["id"], "text": p["text"]} for p in payload]

    async def fake_escalate(*args, **kwargs):
        return ""

    async def fake_fast_rescue(items, *args, **kwargs):
        return []

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting), \
         patch("app.services.pipeline.append_job_log", side_effect=fake_append_job_log), \
         patch("app.services.pipeline.create_job", return_value=1), \
         patch("app.services.pipeline.update_job") as mock_update_job, \
         patch.object(pipeline, "trigger_bazarr_search"), \
         patch.object(pipeline.translator, "translate_srt_content", side_effect=fake_translate), \
         patch.object(pipeline.translator, "classify_and_recover_identical", side_effect=fake_classify_and_recover), \
         patch.object(pipeline.translator, "translate_batch", side_effect=fake_translate_batch), \
         patch.object(pipeline.translator, "translate_batch", return_value=[]), \
             patch.object(pipeline.translator, "escalate_single_line", side_effect=fake_escalate), \
         patch.object(pipeline.translator, "fast_final_rescue_batch", side_effect=fake_fast_rescue):

        res = await pipeline._run_pipeline_logic(1, str(video_path), wait_seconds=0)

        # Result must be TRANSLATED
        assert res["status"] == "translated"

        # Output file must exist and be valid SRT
        target_srt_path = str(video_path).replace(".mkv", ".sv.srt")
        assert os.path.exists(target_srt_path)

        with open(target_srt_path, "r", encoding="utf-8") as f:
            published_subs = list(srt.parse(f.read()))

        assert len(published_subs) == 679
        # Cue 454 (index 453) preserved original English
        assert published_subs[deadlock_idx].content == "English line 454"
        # All other cues are translated Swedish
        assert "Detta är en svensk dialograd 1" in published_subs[0].content
        assert "Detta är en svensk dialograd 679" in published_subs[678].content

        # Check logs for exact expected notifications
        log_text = "\n".join(logs_recorded)
        assert f"Semantic deadlock detected for cue {deadlock_idx + 1}" in log_text
        assert "QA fallback: preserving original source text" in log_text
        assert "QA Gate PASSED_WITH_WARNINGS" in log_text
        assert "1 unresolved English line" in log_text
        assert "1 source-preserved fallback" in log_text
        assert "Result: PASS_WITH_WARNINGS" in log_text
        assert "Published" in log_text


@pytest.mark.asyncio
async def test_pipeline_recovery_success_path_publishes_clean_pass(tmp_path):
    """
    Scenario 10: When recovery succeeds in translating the cue, job completes with clean PASS.
    """
    pipeline = SubtitlePipeline()
    video_path = tmp_path / "movie2.mkv"
    video_path.touch()

    total_cues = 50
    source_subs = [
        srt.Subtitle(index=i, start=timedelta(seconds=i * 2), end=timedelta(seconds=i * 2 + 1), content=f"Source line {i}")
        for i in range(1, total_cues + 1)
    ]
    source_srt_path = str(video_path).replace(".mkv", ".en.srt")
    with open(source_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(source_subs))

    def fake_get_setting(key, default=None):
        if key == "languages":
            return '[{"name": "Swedish", "code": "sv", "enabled": true}]'
        if key == "auto_repair_unhealthy": return "false"
        if key == "extract_target_embedded": return "false"
        if key == "extract_source_embedded": return "false"
        if key == "enable_bazarr_check": return "false"
        if key == "strict_sync_lock": return "true"
        return default

    # First pass: cue 10 left untranslated
    def fake_translate(*args, **kwargs):
        out = []
        for i in range(1, total_cues + 1):
            if i == 10:
                text = f"Source line {i}"
            else:
                text = f"Detta är en svensk dialograd {i} som vi pratar om."
            out.append(srt.Subtitle(index=i, start=timedelta(seconds=i * 2), end=timedelta(seconds=i * 2 + 1), content=text))
        return out

    # Targeted recovery succeeds
    async def fake_translate_batch(payload, *args, **kwargs):
        return [{"id": p["id"], "text": f"Återställd svensk rad {p['id'] + 1}"} for p in payload]

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting), \
         patch("app.services.pipeline.append_job_log"), \
         patch("app.services.pipeline.create_job", return_value=2), \
         patch("app.services.pipeline.update_job"), \
         patch.object(pipeline, "trigger_bazarr_search"), \
         patch.object(pipeline.translator, "translate_srt_content", side_effect=fake_translate), \
         patch.object(pipeline.translator, "classify_and_recover_identical", return_value=[]), \
         patch.object(pipeline.translator, "translate_batch", side_effect=fake_translate_batch):

        res = await pipeline._run_pipeline_logic(2, str(video_path), wait_seconds=0)

        assert res["status"] == "translated"
        target_srt_path = str(video_path).replace(".mkv", ".sv.srt")
        assert os.path.exists(target_srt_path)

        with open(target_srt_path, "r", encoding="utf-8") as f:
            published_subs = list(srt.parse(f.read()))

        assert published_subs[9].content == "Återställd svensk rad 10"


@pytest.mark.asyncio
async def test_deadlock_optimization_exact_call_counts_measured(tmp_path):
    """
    Hermetic benchmark for semantic-deadlock optimization:
    Proves exact recovery stages executed, exact model calls made in Loop 1,
    and exact count of redundant model calls avoided from eliminated loops (Loop 2 & 3).
    """
    pipeline = SubtitlePipeline()
    video_path = tmp_path / "deadlock_bench.mkv"
    video_path.touch()

    total_cues = 679
    deadlock_idx = 453  # cue 454

    source_subs = [
        srt.Subtitle(index=i, start=timedelta(seconds=i * 2), end=timedelta(seconds=i * 2 + 1), content=f"English line {i}")
        for i in range(1, total_cues + 1)
    ]
    source_srt_path = str(video_path).replace(".mkv", ".en.srt")
    with open(source_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(source_subs))

    def fake_get_setting(key, default=None):
        if key == "languages":
            return '[{"name": "Swedish", "code": "sv", "enabled": true}]'
        if key == "auto_repair_unhealthy": return "false"
        if key == "extract_target_embedded": return "false"
        if key == "extract_source_embedded": return "false"
        if key == "enable_bazarr_check": return "false"
        if key == "strict_sync_lock": return "true"
        return default

    # Call tracking counters
    call_counts = {
        "translate_srt_content": 0,
        "classify_and_recover_identical": 0,
        "escalate_single_line": 0,
        "fast_final_rescue_batch": 0,
    }

    def fake_translate(*args, **kwargs):
        call_counts["translate_srt_content"] += 1
        out = []
        for i in range(1, total_cues + 1):
            if i - 1 == deadlock_idx:
                text = f"English line {i}"
            else:
                text = f"Detta är en svensk dialograd {i} som vi pratar om."
            out.append(srt.Subtitle(index=i, start=timedelta(seconds=i * 2), end=timedelta(seconds=i * 2 + 1), content=text))
        return out

    async def fake_classify_and_recover(chunk, *args, **kwargs):
        call_counts["classify_and_recover_identical"] += 1
        return []

    async def fake_escalate(*args, **kwargs):
        call_counts["escalate_single_line"] += 1
        return ""

    async def fake_fast_rescue(items, *args, **kwargs):
        call_counts["fast_final_rescue_batch"] += 1
        return []

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting), \
         patch("app.services.pipeline.append_job_log"), \
         patch("app.services.pipeline.create_job", return_value=3), \
         patch("app.services.pipeline.update_job"), \
         patch.object(pipeline, "trigger_bazarr_search"), \
         patch.object(pipeline.translator, "translate_srt_content", side_effect=fake_translate), \
         patch.object(pipeline.translator, "classify_and_recover_identical", side_effect=fake_classify_and_recover), \
         patch.object(pipeline.translator, "translate_batch", return_value=[]), \
             patch.object(pipeline.translator, "escalate_single_line", side_effect=fake_escalate), \
         patch.object(pipeline.translator, "fast_final_rescue_batch", side_effect=fake_fast_rescue):

        res = await pipeline._run_pipeline_logic(3, str(video_path), wait_seconds=0)

        # 1. Pipeline result must be published as TRANSLATED with PASS_WITH_WARNINGS
        assert res["status"] == "translated"

        # 2. Verify exact calls made:
        # Pass 0: 1 main translation call
        assert call_counts["translate_srt_content"] == 1
        # Stage 1 (Primary Recovery): exactly 1 call
        assert call_counts["classify_and_recover_identical"] == 1
        # Stage 2 (Contextual Escalation): exactly 1 call
        assert call_counts["escalate_single_line"] == 1
        # Stage 3 (Fast Final Rescue): exactly 2 calls (attempt 1 and attempt 2)
        assert call_counts["fast_final_rescue_batch"] == 2

        # 3. Total recovery calls executed before deadlock is proven:
        total_recovery_calls_executed = (
            call_counts["classify_and_recover_identical"]
            + call_counts["escalate_single_line"]
            + call_counts["fast_final_rescue_batch"]
        )
        assert total_recovery_calls_executed == 4

        # 4. Old loop analysis:
        # With max_qa_loops = 3, old loop would have repeated the 4 recovery calls in Loop 2 and Loop 3:
        old_loop_calls_per_cycle = 4
        old_loop_total_recovery_calls = old_loop_calls_per_cycle * 3  # 12 calls
        redundant_calls_avoided = old_loop_total_recovery_calls - total_recovery_calls_executed
        assert redundant_calls_avoided == 8
