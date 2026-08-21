import pytest
import os
import srt
from datetime import timedelta
from unittest.mock import patch, MagicMock
from app.services.pipeline import SubtitlePipeline
from app.core.db import get_job_by_id, init_db, DB_PATH

@pytest.mark.asyncio
async def test_multi_language_partial_e2e(tmp_path):
    # Use real DB for job state verification
    init_db()
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM jobs")
    conn.commit()
    conn.close()

    with patch("google.genai.Client"):
        pipeline = SubtitlePipeline()
    video_path = tmp_path / "video.mkv"
    video_path.touch()
    
    # Create REAL source subtitle on disk
    source_subs = [srt.Subtitle(index=i, start=timedelta(seconds=i), end=timedelta(seconds=i+1), content=f"Source {i} with a lot of text to make it larger than 100 bytes so it passes the check") for i in range(1, 10)]
    source_srt_path = str(video_path).replace(".mkv", ".en.srt")
    with open(source_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(source_subs))

    # Mock settings to return 2 languages: de and fr
    def fake_get_setting(key, default=None):
        if key == "languages":
            return '[{"name": "German", "code": "de", "enabled": true}, {"name": "French", "code": "fr", "enabled": true}]'
        if key == "auto_repair_unhealthy": return "false"
        if key == "extract_target_embedded": return "false"
        if key == "extract_source_embedded": return "false"
        if key == "enable_bazarr_check": return "false"
        return default

    translate_calls = {"de": 0, "fr": 0}
    
    async def fake_translate(subs, target_language="English", batch_size=50, job_id=None, show_title=None, *args, **kwargs):
        # We need to map target_language back to a code (e.g. 'German' -> 'de', 'French' -> 'fr') for translate_calls
        lang_map = {"German": "de", "French": "fr"}
        lang_code = lang_map.get(target_language, target_language)
        translate_calls[lang_code] += 1
        
        # Return matched length of translated subs
        return [srt.Subtitle(index=sub.index, start=sub.start, end=sub.end, content=f"Target {lang_code} {sub.index}") for sub in subs]

    qa_pass_fr = False
    
    def fake_qa_gate(*args, **kwargs):
        lang = kwargs.get("target_lang_code", "")
        if lang == "de":
            return {"passed": True, "score": 100, "issues": [], "dropped_count": 0, "untranslated_ids": [], "real_untranslated_ids": [], "dropped_details": [], "sync_diff_ms": 0}
        
        if qa_pass_fr:
            return {"passed": True, "score": 100, "issues": [], "dropped_count": 0, "untranslated_ids": [], "real_untranslated_ids": [], "dropped_details": [], "sync_diff_ms": 0}
            
        return {"passed": False, "score": 50, "issues": ["dropped line"], "dropped_count": 1, "untranslated_ids": [1], "real_untranslated_ids": [1], "dropped_details": [{"id": 1}], "sync_diff_ms": 0}

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting):
         with patch.object(pipeline, "trigger_bazarr_search"), \
              patch.object(pipeline.translator, "translate_srt_content", side_effect=fake_translate), \
              patch.object(pipeline.translator, "escalate_single_line", return_value="Escalated"), \
              patch.object(pipeline.translator, "classify_and_recover_identical", return_value=[{"id": 1, "action": "translate", "text": "Recovered text"}]), \
              patch("app.services.pipeline.qa_gate", side_effect=fake_qa_gate):
              
              from app.core.db import create_job
              job_id = create_job(str(video_path))

              # FIRST RUN
              res = await pipeline._run_pipeline_logic(job_id, str(video_path), wait_seconds=0)
              
              # Should be PARTIAL because 'de' passed, but 'fr' failed
              print("ERROR:", res.get("error"))
              assert res["status"] == "partial"
              
              job = get_job_by_id(job_id)
              assert job["status"] == "PARTIAL"
              assert job["next_retry_at"] is not None
              
              # 'de' and 'fr' should have been translated once
              assert translate_calls["de"] == 1
              assert translate_calls["fr"] == 1
              
              # A-output exists and we snapshot it
              out_de_srt = str(video_path).replace(".mkv", ".de.srt")
              assert os.path.exists(out_de_srt)
              with open(out_de_srt, "r", encoding="utf-8") as f:
                  de_content_pass_1 = f.read()
              
              # Now simulate the worker picking it up for a retry pass
              qa_pass_fr = True
              from app.core.db import update_job
              update_job(job_id, status="RECOVERING")
              
              # SECOND RUN (Worker claim)
              res2 = await pipeline._run_pipeline_logic(job_id, str(video_path), wait_seconds=0)
              
              assert res2["status"] == "translated"
              job2 = get_job_by_id(job_id)
              assert job2["status"] == "TRANSLATED"
              
              # 'de' should NOT have been translated again
              assert translate_calls["de"] == 1
              # 'fr' should have been translated a second time
              assert translate_calls["fr"] == 2
              
              # 'de' output should be byte-identical
              with open(out_de_srt, "r", encoding="utf-8") as f:
                  de_content_pass_2 = f.read()
              assert de_content_pass_1 == de_content_pass_2
              
              # 'fr' output should now exist
              out_fr_srt = str(video_path).replace(".mkv", ".fr.srt")
              assert os.path.exists(out_fr_srt)

