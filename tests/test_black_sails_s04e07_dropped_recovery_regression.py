import pytest
import os
from datetime import timedelta
from unittest.mock import patch
import srt

from app.services.pipeline import (
    SubtitlePipeline,
    qa_gate,
    QA_STATUS_PASS,
    QA_STATUS_PASS_WITH_WARNINGS,
    QA_STATUS_FAIL,
    DEFAULT_QA_MAX_UNRESOLVED_COUNT,
    DEFAULT_QA_MAX_UNRESOLVED_RATIO,
)
from app.core.validator import check_dropped_lines


@pytest.mark.asyncio
async def test_scenario_a_772_cues_with_2_unrecoverable_dropped_cues(tmp_path):
    """
    Scenario A:
    772-like translation with exactly 2 unrecoverable empty real-dialogue cues
    after all recovery attempts:
      -> source text preserved for those exact cues
      -> dropped_count == 0
      -> cue count and timestamps unchanged
      -> source_preserved count == 2
      -> final policy result follows existing bounded unresolved policy (PASS_WITH_WARNINGS)
      -> file is published
    """
    pipeline = SubtitlePipeline()
    video_path = tmp_path / "Black.Sails.S04E07.XXXV.WEBRip-1080p.mkv"
    video_path.touch()

    total_cues = 772
    dropped_idx_1 = 337  # Cue 338: "Sir." (0-indexed)
    dropped_idx_2 = 715  # Cue 716: "Flint." (0-indexed)

    source_subs = []
    for i in range(1, total_cues + 1):
        if i - 1 == dropped_idx_1:
            content = "Sir."
        elif i - 1 == dropped_idx_2:
            content = "Flint."
        else:
            content = f"English dialogue line {i} about pirate adventures."
        source_subs.append(
            srt.Subtitle(
                index=i,
                start=timedelta(seconds=i * 2),
                end=timedelta(seconds=i * 2 + 1, milliseconds=500),
                content=content
            )
        )

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

    # First pass: returns Swedish for all cues except the 2 dropped cues which are empty
    def fake_translate(*args, **kwargs):
        out = []
        for i in range(1, total_cues + 1):
            if i - 1 in (dropped_idx_1, dropped_idx_2):
                text = ""  # Empty translation (dropped by model)
            else:
                text = f"Detta är svensk dialograd {i} som piraterna talar om."
            out.append(
                srt.Subtitle(
                    index=i,
                    start=timedelta(seconds=i * 2),
                    end=timedelta(seconds=i * 2 + 1, milliseconds=500),
                    content=text
                )
            )
        return out

    # Recovery mock: all recovery attempts fail/exhaust for the 2 dropped cues
    async def fake_classify_and_recover(*args, **kwargs):
        return []

    async def fake_translate_batch(payload, *args, **kwargs):
        return []

    async def fake_escalate(*args, **kwargs):
        return ""

    async def fake_fast_rescue(*args, **kwargs):
        return []

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting), \
         patch("app.services.pipeline.append_job_log", side_effect=fake_append_job_log), \
         patch("app.services.pipeline.create_job", return_value=545), \
         patch("app.services.pipeline.update_job"), \
         patch.object(pipeline, "trigger_bazarr_search"), \
         patch.object(pipeline.translator, "translate_srt_content", side_effect=fake_translate), \
         patch.object(pipeline.translator, "classify_and_recover_identical", side_effect=fake_classify_and_recover), \
         patch.object(pipeline.translator, "translate_batch", side_effect=fake_translate_batch), \
         patch.object(pipeline.translator, "escalate_single_line", side_effect=fake_escalate), \
         patch.object(pipeline.translator, "fast_final_rescue_batch", side_effect=fake_fast_rescue):

        res = await pipeline._run_pipeline_logic(545, str(video_path), wait_seconds=0)

        # 1. Pipeline succeeds under bounded unresolved policy
        assert res["status"] == "translated"

        # 2. Output file must exist and be valid SRT
        target_srt_path = str(video_path).replace(".mkv", ".sv.srt")
        assert os.path.exists(target_srt_path)

        with open(target_srt_path, "r", encoding="utf-8") as f:
            published_subs = list(srt.parse(f.read()))

        # 3. Cue count must match exactly
        assert len(published_subs) == 772

        # 4. Timestamps must match exactly (0ms drift)
        for i in range(total_cues):
            assert published_subs[i].start == source_subs[i].start
            assert published_subs[i].end == source_subs[i].end

        # 5. Dropped count must be 0
        dropped_count, dropped_details = check_dropped_lines(source_subs, published_subs)
        assert dropped_count == 0
        assert len(dropped_details) == 0

        # 6. Source text preserved for the exact two dropped cues
        assert published_subs[dropped_idx_1].content == "Sir."
        assert published_subs[dropped_idx_2].content == "Flint."

        # 7. Other cues remain translated Swedish
        assert "Detta är svensk dialograd 1" in published_subs[0].content
        assert "Detta är svensk dialograd 772" in published_subs[771].content

        # 8. Check logs and QA summary
        log_text = "\n".join(logs_recorded)
        assert f"Semantic deadlock detected for cue {dropped_idx_1 + 1}" in log_text
        assert f"Semantic deadlock detected for cue {dropped_idx_2 + 1}" in log_text
        assert "QA fallback: preserving original source text" in log_text
        assert "QA Gate PASSED_WITH_WARNINGS" in log_text
        assert "2 unresolved" in log_text
        assert "2 source-preserved fallbacks" in log_text
        assert "Result: PASS_WITH_WARNINGS" in log_text
        assert "Published" in log_text


@pytest.mark.asyncio
async def test_scenario_b_too_many_unrecoverable_dropped_cues_fails_policy(tmp_path):
    """
    Scenario B:
    Too many dropped/unrecoverable cues:
      -> must NOT accidentally become a clean PASS or PASS_WITH_WARNINGS
      -> existing unresolved count/ratio safety policy still applies (FAIL)
      -> file is NOT published
    """
    pipeline = SubtitlePipeline()
    video_path = tmp_path / "movie_too_many_dropped.mkv"
    video_path.touch()

    total_cues = 100
    # 5 dropped cues out of 100 = 5.0% (> limit of 3 cues and > 1.0% limit)
    dropped_indices = {10, 20, 30, 40, 50}

    source_subs = [
        srt.Subtitle(
            index=i,
            start=timedelta(seconds=i * 2),
            end=timedelta(seconds=i * 2 + 1),
            content=f"English line {i} dialogue"
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

    def fake_translate(*args, **kwargs):
        out = []
        for i in range(1, total_cues + 1):
            if (i - 1) in dropped_indices:
                text = ""  # Empty translation
            else:
                text = f"Detta är svensk text {i}."
            out.append(srt.Subtitle(index=i, start=timedelta(seconds=i * 2), end=timedelta(seconds=i * 2 + 1), content=text))
        return out

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting), \
         patch("app.services.pipeline.append_job_log", side_effect=fake_append_job_log), \
         patch("app.services.pipeline.create_job", return_value=546), \
         patch("app.services.pipeline.update_job"), \
         patch.object(pipeline, "trigger_bazarr_search"), \
         patch.object(pipeline.translator, "translate_srt_content", side_effect=fake_translate), \
         patch.object(pipeline.translator, "classify_and_recover_identical", return_value=[]), \
         patch.object(pipeline.translator, "translate_batch", return_value=[]), \
         patch.object(pipeline.translator, "escalate_single_line", return_value=""), \
         patch.object(pipeline.translator, "fast_final_rescue_batch", return_value=[]):

        res = await pipeline._run_pipeline_logic(546, str(video_path), wait_seconds=0)

        # 1. Must FAIL because 5 > max_unresolved_count (3)
        assert res["status"] == "failed"

        # 2. Must NOT be published
        target_srt_path = str(video_path).replace(".mkv", ".sv.srt")
        assert not os.path.exists(target_srt_path)

        # 3. Log must show failure and NOT clean PASS
        log_text = "\n".join(logs_recorded)
        assert "QA Gate FAILED" in log_text
        assert "QA Gate PASSED (Score" not in log_text
        assert "QA Gate PASSED_WITH_WARNINGS" not in log_text
        assert "File NOT published" in log_text


@pytest.mark.asyncio
async def test_scenario_c_empty_structural_placeholder_not_converted_to_fake_dialogue(tmp_path):
    """
    Scenario C:
    Empty/structural source placeholder:
      -> must not be converted into fake dialogue fallback
      -> must not be source-preserved as real dialogue
      -> clean pass when real dialogue is translated
    """
    pipeline = SubtitlePipeline()
    video_path = tmp_path / "movie_structural_placeholder.mkv"
    video_path.touch()

    total_cues = 20
    # Cues with empty/structural placeholders
    placeholder_idx_1 = 4   # "<i></i>"
    placeholder_idx_2 = 12  # "<i></i>"

    source_subs = []
    for i in range(1, total_cues + 1):
        if i - 1 in (placeholder_idx_1, placeholder_idx_2):
            content = "<i></i>"
        else:
            content = f"English dialogue line {i}"
        source_subs.append(
            srt.Subtitle(
                index=i,
                start=timedelta(seconds=i * 2),
                end=timedelta(seconds=i * 2 + 1),
                content=content
            )
        )

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

    # Real dialogue is translated; placeholders remain empty / "<i></i>"
    def fake_translate(*args, **kwargs):
        out = []
        for i in range(1, total_cues + 1):
            if i - 1 == placeholder_idx_1:
                text = "<i></i>"
            elif i - 1 == placeholder_idx_2:
                text = ""
            else:
                text = f"Svensk replik {i} här."
            out.append(srt.Subtitle(index=i, start=timedelta(seconds=i * 2), end=timedelta(seconds=i * 2 + 1), content=text))
        return out

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting), \
         patch("app.services.pipeline.append_job_log", side_effect=fake_append_job_log), \
         patch("app.services.pipeline.create_job", return_value=547), \
         patch("app.services.pipeline.update_job"), \
         patch.object(pipeline, "trigger_bazarr_search"), \
         patch.object(pipeline.translator, "translate_srt_content", side_effect=fake_translate), \
         patch.object(pipeline.translator, "classify_and_recover_identical", return_value=[]), \
         patch.object(pipeline.translator, "translate_batch", return_value=[]), \
         patch.object(pipeline.translator, "escalate_single_line", return_value=""), \
         patch.object(pipeline.translator, "fast_final_rescue_batch", return_value=[]):

        res = await pipeline._run_pipeline_logic(547, str(video_path), wait_seconds=0)

        # 1. Clean PASS
        assert res["status"] == "translated"

        target_srt_path = str(video_path).replace(".mkv", ".sv.srt")
        assert os.path.exists(target_srt_path)

        with open(target_srt_path, "r", encoding="utf-8") as f:
            published_subs = list(srt.parse(f.read()))

        # 2. Structural placeholders were NOT converted to fake dialogue fallback
        log_text = "\n".join(logs_recorded)
        assert f"Semantic deadlock detected for cue {placeholder_idx_1 + 1}" not in log_text
        assert f"Semantic deadlock detected for cue {placeholder_idx_2 + 1}" not in log_text
        assert "0 source-preserved fallbacks" in log_text

        # 3. Dropped count is 0
        dropped_count, _ = check_dropped_lines(source_subs, published_subs)
        assert dropped_count == 0


@pytest.mark.asyncio
async def test_scenario_d_existing_successful_recovery_path_unchanged(tmp_path):
    """
    Scenario D:
    Existing successful recovery path:
      -> when targeted recovery successfully translates the dropped cue
      -> normal clean PASS without needing source-preservation fallback
      -> source_preserved_count == 0
    """
    pipeline = SubtitlePipeline()
    video_path = tmp_path / "movie_recovery_success.mkv"
    video_path.touch()

    total_cues = 40
    dropped_idx = 14  # Cue 15 initially empty

    source_subs = [
        srt.Subtitle(
            index=i,
            start=timedelta(seconds=i * 2),
            end=timedelta(seconds=i * 2 + 1),
            content=f"English line {i} spoken here"
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

    # First pass: cue 15 dropped (empty)
    def fake_translate(*args, **kwargs):
        out = []
        for i in range(1, total_cues + 1):
            if i - 1 == dropped_idx:
                text = ""  # Dropped
            else:
                text = f"Detta är svensk dialog {i}."
            out.append(srt.Subtitle(index=i, start=timedelta(seconds=i * 2), end=timedelta(seconds=i * 2 + 1), content=text))
        return out

    # Targeted recovery successfully translates cue 15
    async def fake_rescue_batch(items, *args, **kwargs):
        return [{"id": it["id"], "text": f"Återställd svensk replik {it['id'] + 1}."} for it in items]

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting), \
         patch("app.services.pipeline.append_job_log", side_effect=fake_append_job_log), \
         patch("app.services.pipeline.create_job", return_value=548), \
         patch("app.services.pipeline.update_job"), \
         patch.object(pipeline, "trigger_bazarr_search"), \
         patch.object(pipeline.translator, "translate_srt_content", side_effect=fake_translate), \
         patch.object(pipeline.translator, "classify_and_recover_identical", return_value=[]), \
         patch.object(pipeline.translator, "fast_final_rescue_batch", side_effect=fake_rescue_batch):

        res = await pipeline._run_pipeline_logic(548, str(video_path), wait_seconds=0)

        # 1. Clean PASS
        assert res["status"] == "translated"

        target_srt_path = str(video_path).replace(".mkv", ".sv.srt")
        assert os.path.exists(target_srt_path)

        with open(target_srt_path, "r", encoding="utf-8") as f:
            published_subs = list(srt.parse(f.read()))

        assert len(published_subs) == 40
        # Cue 15 was successfully recovered as Swedish, NOT source-preserved English
        assert published_subs[dropped_idx].content == "Återställd svensk replik 15."

        # 2. No source-preserved fallback
        log_text = "\n".join(logs_recorded)
        assert "QA Gate PASSED (Score: 100/100)" in log_text
        assert "0 source-preserved fallbacks" in log_text
