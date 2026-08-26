import pytest
from datetime import timedelta
import srt
import json
import os
from unittest.mock import patch, AsyncMock

from app.services.pipeline import pipeline
from app.services.translator import is_usable_translation
import app.core.db

def test_is_usable_translation():
    assert not is_usable_translation(None)
    assert not is_usable_translation("")
    assert not is_usable_translation("   ")
    assert not is_usable_translation("<i></i>")
    assert is_usable_translation("Kom hit")
    assert is_usable_translation("  Valid  ")

@pytest.fixture(autouse=True)
def setup_teardown_db(tmp_path):
    original_db = app.core.db.DB_PATH
    app.core.db.DB_PATH = "/tmp/test_blank_babel.db"
    app.core.db.init_db()
    app.core.db.clear_all_jobs()

    video = tmp_path / "test_blank.mkv"
    video.touch()

    yield str(video)

    app.core.db.clear_all_jobs()
    app.core.db.DB_PATH = original_db

@pytest.mark.asyncio
async def test_blank_recovery_regression(setup_teardown_db):
    video_path = setup_teardown_db
    job_id = app.core.db.create_job(video_path, "MANUAL", "test title")

    # Create 3 cues
    source_subs = []
    for i in range(1, 4):
        source_subs.append(srt.Subtitle(i, timedelta(seconds=i), timedelta(seconds=i+1), f"Source Dialogue {i}"))

    srt_path = video_path.replace(".mkv", ".en.srt")
    with open(srt_path, "w") as f:
        f.write(srt.compose(source_subs))

    first_pass_subs = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Svensk 1"),
        srt.Subtitle(2, timedelta(seconds=2), timedelta(seconds=3), "Source Dialogue 2"), # Identical -> Unresolved
        srt.Subtitle(3, timedelta(seconds=3), timedelta(seconds=4), "Svensk 3"),
    ]

    async def mock_translate_srt_content(*args, **kwargs):
        import copy
        return copy.deepcopy(first_pass_subs)

    async def mock_classify_and_recover_identical(items, target_language, show_title, **kwargs):
        # Return blank translation
        return [
            {"id": 1, "action": "translate", "text": "   "}
        ]

    async def mock_escalate_single_line(target_idx, target_text, prev_text, next_text, target_language, show_title, is_real_untranslated=False, **kwargs):
        # Escalation will save the day
        return "Escalated 2"

    with patch("app.services.pipeline.SubtitlePipeline.get_configured_languages", return_value=[{"name": "Swedish", "code": "sv", "enabled": True}]):
        with patch("app.services.translator.SubtitleTranslator.translate_srt_content", side_effect=mock_translate_srt_content):
            with patch("app.services.translator.SubtitleTranslator.classify_and_recover_identical", side_effect=mock_classify_and_recover_identical):
                with patch("app.services.translator.SubtitleTranslator.escalate_single_line", side_effect=mock_escalate_single_line):
                    with patch("app.services.translator.SubtitleTranslator.translate_batch", return_value=[]), \
                         patch("app.services.translator.SubtitleTranslator.fast_final_rescue_batch", return_value=[]):
                        with patch("app.services.pipeline.find_external_subtitle", return_value=srt_path):
                            with patch("os.rename"):
                                result = await pipeline.process_video_file(video_path, job_id=job_id, force_retranslate=True)

    assert result["status"] == "translated"
    job = app.core.db.get_job_by_id(job_id)

    # Blank translation from primary recovery should have been rejected
    logs = job["logs"] if isinstance(job["logs"], list) else []

    rejected_log = next((log for log in logs if "QA Recovery: Rejected blank/invalid translation for cue 2" in log), None)
    assert rejected_log is not None

    esc_log = next((log for log in logs if "Escalation: Translated cue 2 using dialogue context" in log), None)
    assert esc_log is not None

    # Line 1 is actually index 1 in Python since it's 0-indexed. ID 2 -> cue 2.
    assert job["dropped_lines"] == 0
