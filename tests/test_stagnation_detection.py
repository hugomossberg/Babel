import pytest
from datetime import timedelta
import srt
from unittest.mock import patch

from app.services.pipeline import SubtitlePipeline
import app.core.db

@pytest.mark.asyncio
async def test_stagnation_detection_breaks_loop_early(tmp_path):
    app.core.db.init_db()
    pipeline = SubtitlePipeline()
    video_path = tmp_path / "stagnation_test.mkv"
    video_path.touch()

    # Create 3 cues
    source_subs = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Hello"),
        srt.Subtitle(2, timedelta(seconds=2), timedelta(seconds=3), "World"),
        srt.Subtitle(3, timedelta(seconds=3), timedelta(seconds=4), "Morgan?"),
    ]
    source_srt_path = str(video_path).replace(".mkv", ".en.srt")
    with open(source_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(source_subs))

    def fake_get_setting(key, default=None):
        if key == "languages":
            return '[{"name": "Swedish", "code": "sv", "enabled": true}]'
        return default

    # First pass translates 1 and 2, but leaves 3 as "Morgan?"
    first_pass_subs = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Hej"),
        srt.Subtitle(2, timedelta(seconds=2), timedelta(seconds=3), "Värld"),
        srt.Subtitle(3, timedelta(seconds=3), timedelta(seconds=4), "Morgan?"),
    ]

    async def fake_translate_srt_content(*args, **kwargs):
        import copy
        return copy.deepcopy(first_pass_subs)

    # Classify attempts keep proper_noun, but without evidence it gets downgraded to translate with empty text
    async def fake_classify(*args, **kwargs):
        return [{"id": 2, "action": "translate", "text": ""}]

    # All recovery stages fail to translate Morgan
    async def fake_translate_batch(payload, *args, **kwargs):
        return [{"id": 2, "text": "Morgan?"}]

    async def fake_escalate(*args, **kwargs):
        return "Morgan?"

    with patch("google.genai.Client"):
        with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting):
            with patch.object(pipeline, "trigger_bazarr_search"), \
                 patch.object(pipeline.translator, "translate_srt_content", side_effect=fake_translate_srt_content), \
                 patch.object(pipeline.translator, "classify_and_recover_identical", side_effect=fake_classify), \
                 patch.object(pipeline.translator, "translate_batch", side_effect=fake_translate_batch), \
                 patch.object(pipeline.translator, "escalate_single_line", side_effect=fake_escalate):

                job_id = app.core.db.create_job(str(video_path))
                res = await pipeline._run_pipeline_logic(job_id, str(video_path), wait_seconds=0)

                # QA should fail and job enter failed / recovering
                assert res["status"] in ["recovering", "failed"]

                job = app.core.db.get_job_by_id(job_id)
                logs = job["logs"] if isinstance(job["logs"], list) else []

                # Verify stagnation detection was logged
                stagnation_log = next((l for l in logs if "Stagnation detected" in l), None)
                assert stagnation_log is not None, f"Expected stagnation detection log, got: {logs}"
                assert "Breaking QA recovery loop" in stagnation_log
