import pytest
from unittest.mock import patch, MagicMock
from app.services.pipeline import SubtitlePipeline
import os

@pytest.mark.asyncio
async def test_embedded_extraction_status_handling(tmp_path):
    pipeline = SubtitlePipeline()
    video = tmp_path / "video.mkv"
    video.touch()
    
    # Mock settings so Auto Repair is OFF
    def fake_get_setting(key, default):
        if key == "auto_repair_unhealthy": return "false"
        if key == "extract_target_embedded": return "true"
        if key == "enable_bazarr_check": return "false"
        if key == "extract_source_embedded": return "false"
        return default

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting):
        with patch("app.services.pipeline.extract_embedded_srt") as mock_extract:
            
            with patch("app.services.pipeline.evaluate_subtitle_health") as mock_health:
                with patch("app.services.pipeline.create_job", return_value=1), \
                     patch("app.services.pipeline.update_job"), \
                     patch("app.services.pipeline.append_job_log"), \
                     patch("os.replace") as mock_replace, \
                     patch("os.remove") as mock_remove:
                     
                     with patch.object(pipeline, "trigger_bazarr_search"), \
                          patch.object(pipeline.translator, "translate_srt_content", return_value=[]), \
                          patch("app.services.pipeline.qa_gate", return_value={"passed": True, "score": 100}):
                          
                          # Test RED status
                          def fake_extract_red(vid, out, preferred_lang):
                              open(out, "w").close()
                              return True
                          mock_extract.side_effect = fake_extract_red
                          mock_health.return_value = {"status": "RED", "reason": "Bad"}
                          
                          await pipeline._run_pipeline_logic(1, str(video), wait_seconds=0)
                          mock_health.assert_called()
                          mock_replace.assert_not_called()
                          mock_remove.assert_called()
                          
                          mock_health.reset_mock()
                          mock_replace.reset_mock()
                          mock_remove.reset_mock()
                          
                          # Test YELLOW status
                          def fake_extract_yellow(vid, out, preferred_lang):
                              open(out, "w").close()
                              return True
                          mock_extract.side_effect = fake_extract_yellow
                          mock_health.return_value = {"status": "YELLOW", "reason": "Warning"}
                          
                          await pipeline._run_pipeline_logic(1, str(video), wait_seconds=0)
                          mock_health.assert_called()
                          mock_replace.assert_not_called()
                          mock_remove.assert_called()
                          
                          mock_health.reset_mock()
                          mock_replace.reset_mock()
                          mock_remove.reset_mock()
                          
                          # Test GREEN status
                          def fake_extract_green(vid, out, preferred_lang):
                              open(out, "w").close()
                              return True
                          mock_extract.side_effect = fake_extract_green
                          mock_health.return_value = {"status": "GREEN", "reason": "Good"}
                          
                          await pipeline._run_pipeline_logic(1, str(video), wait_seconds=0)
                          mock_health.assert_called()
                          mock_replace.assert_called()
                          mock_remove.assert_not_called()
