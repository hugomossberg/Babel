import pytest
from datetime import timedelta
import srt
import json
from unittest.mock import patch, MagicMock

from app.services.pipeline import pipeline
from app.core.db import DB_PATH
import app.core.db
import os

@pytest.fixture(autouse=True)
def setup_teardown_db(tmp_path):
    original_db = app.core.db.DB_PATH
    app.core.db.DB_PATH = "/tmp/test_dropped_babel.db"
    app.core.db.init_db()
    app.core.db.clear_all_jobs()
    
    video = tmp_path / "test_dropped.mkv"
    video.touch()
    
    yield str(video)
    
    app.core.db.clear_all_jobs()
    app.core.db.DB_PATH = original_db

@pytest.mark.asyncio
async def test_dropped_line_recovery(setup_teardown_db):
    video_path = setup_teardown_db
    job_id = app.core.db.create_job(video_path, "MANUAL", "test title")
    
    # Create 675 source cues
    source_subs = []
    for i in range(1, 676):
        source_subs.append(srt.Subtitle(i, timedelta(seconds=i), timedelta(seconds=i+1), f"Source English Dialogue {i}"))
        
    srt_path = video_path.replace(".mkv", ".en.srt")
    with open(srt_path, "w") as f:
        f.write(srt.compose(source_subs))
        
    # First pass: returns 675 cues, but 2 are dropped
    first_pass_subs = []
    for i in range(1, 676):
        if i in [100, 500]:
            first_pass_subs.append(srt.Subtitle(i, timedelta(seconds=i), timedelta(seconds=i+1), ""))
        else:
            first_pass_subs.append(srt.Subtitle(i, timedelta(seconds=i), timedelta(seconds=i+1), f"Svensk översättning dialog {i}"))
            
    async def mock_translate_srt_content(*args, **kwargs):
        import copy
        return copy.deepcopy(first_pass_subs)
        
    async def mock_escalate_single_line(target_idx, target_text, prev_text, next_text, target_language, show_title):
        assert target_idx in [99, 499] # 0-indexed
        if target_idx == 99:
            return "Reparerad svensk text 100"
        if target_idx == 499:
            return "Reparerad svensk text 500"
        return "Unknown"

    with patch("app.services.pipeline.SubtitlePipeline.get_configured_languages", return_value=[{"name": "Swedish", "code": "sv", "enabled": True}]):
        with patch("app.services.translator.SubtitleTranslator.translate_srt_content", side_effect=mock_translate_srt_content):
            with patch("app.services.translator.SubtitleTranslator.escalate_single_line", side_effect=mock_escalate_single_line):
                with patch("app.services.pipeline.find_external_subtitle", return_value=srt_path):
                    with patch("os.rename"):
                        result = await pipeline.process_video_file(video_path, job_id=job_id, force_retranslate=True)
                    
    assert result["status"] == "translated"
    
    job = app.core.db.get_job_by_id(job_id)
    assert job["status"] == "TRANSLATED"
    assert job["dropped_lines"] == 0
    assert job["total_lines"] == 675
    
    # Check that 2 were recovered
    logs = job["logs"] if isinstance(job["logs"], list) else []
    
    # Verify the summary line
    summary_line = next((log for log in logs if "2 translated on recovery" in log), None)
    assert summary_line is not None, "Bookkeeping for recovered_count failed"

    # Verify escalation was called and succeeded
    esc_log = next((log for log in logs if "Escalating using Primary Model (Contextual Mode)" in log), None)
    assert esc_log is not None
    assert "2 lines still unresolved" in esc_log
