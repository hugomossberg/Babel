import pytest
from unittest.mock import patch, MagicMock
from app.services.pipeline import SubtitlePipeline
import os

@pytest.mark.asyncio
async def test_embedded_extraction_always_validates(tmp_path):
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
            # Pretend we successfully extracted a file
            def fake_extract(vid, out, preferred_lang):
                with open(out, "w") as f:
                    f.write("1\n00:00:01,000 --> 00:00:02,000\nHello")
                return True
            mock_extract.side_effect = fake_extract
            
            with patch("app.services.pipeline.evaluate_subtitle_health") as mock_health:
                # We return RED, so the file should be rejected even though auto repair is OFF
                mock_health.return_value = {"status": "RED", "reason": "Bad embedded sub"}
                
                with patch("app.services.pipeline.create_job", return_value=1), \
                     patch("app.services.pipeline.update_job"), \
                     patch("app.services.pipeline.append_job_log"):
                     
                     with patch.object(pipeline, "trigger_bazarr_search"), \
                          patch.object(pipeline.translator, "translate_srt_content", return_value=[]), \
                          patch("app.services.pipeline.qa_gate", return_value={"passed": True, "score": 100}):
                          
                          # This will fail later in the pipeline because there's no source, 
                          # but it should pass the target extraction phase
                          await pipeline._run_pipeline_logic(1, str(video), wait_seconds=0)
                          
                # Verify health check was called!
                mock_health.assert_called()
