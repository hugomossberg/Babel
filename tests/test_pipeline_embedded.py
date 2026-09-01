import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.pipeline import SubtitlePipeline
from app.core.trust_engine import TrustResult, TrustDecision, CandidateOrigin, VerificationMode
import os

@pytest.mark.asyncio
async def test_embedded_extraction_status_handling(tmp_path):
    pipeline = SubtitlePipeline()
    video = tmp_path / "video.mkv"
    video.touch()
    
    # Mock settings so Auto Repair is OFF
    def fake_get_setting(key, default):
        if key == "languages": return '[{"name": "Swedish", "code": "sv", "enabled": true}]'
        if key == "auto_repair_unhealthy": return "false"
        if key == "extract_target_embedded": return "true"
        if key == "enable_bazarr_check": return "false"
        if key == "extract_source_embedded": return "false"
        return default

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting):
        with patch("app.services.pipeline.extract_embedded_srt") as mock_extract:
            
            with patch("app.services.pipeline.SubtitleTrustEngine.evaluate_candidate", new_callable=AsyncMock) as mock_trust:
                with patch("app.services.pipeline.create_job", return_value=1), \
                     patch("app.services.pipeline.update_job"), \
                     patch("app.services.pipeline.append_job_log"), \
                     patch("app.services.pipeline._publish_subtitle_atomic") as mock_publish, \
                     patch("os.remove") as mock_remove:
                     
                     with patch.object(pipeline, "trigger_bazarr_search"), \
                          patch.object(pipeline.translator, "translate_srt_content", return_value=[]), \
                          patch("app.services.pipeline.qa_gate", return_value={"passed": True, "score": 100}):
                          
                          # Test FAIL status (RED)
                          def fake_extract_red(vid, out, preferred_lang):
                              with open(out, "w", encoding="utf-8") as f:
                                  f.write("1\n00:00:01,000 --> 00:00:02,000\nBad\n\n")
                              return True
                          mock_extract.side_effect = fake_extract_red
                          mock_trust.return_value = TrustResult(
                              decision=TrustDecision.FAIL,
                              score=20,
                              confidence="HIGH",
                              reasons=["Structural issues"],
                              origin=CandidateOrigin.EMBEDDED,
                              verification_mode=VerificationMode.EMBEDDED_PROVENANCE
                          )
                          
                          await pipeline._run_pipeline_logic(1, str(video), wait_seconds=0)
                          mock_trust.assert_called()
                          mock_publish.assert_not_called()
                          mock_remove.assert_called()
                          
                          mock_trust.reset_mock()
                          mock_publish.reset_mock()
                          mock_remove.reset_mock()
                          
                          # Test REPAIRABLE status (YELLOW when auto_repair is off -> reject)
                          def fake_extract_yellow(vid, out, preferred_lang):
                              with open(out, "w", encoding="utf-8") as f:
                                  f.write("1\n00:00:01,000 --> 00:00:02,000\nWarning\n\n")
                              return True
                          mock_extract.side_effect = fake_extract_yellow
                          mock_trust.return_value = TrustResult(
                              decision=TrustDecision.REPAIRABLE,
                              score=60,
                              confidence="HIGH",
                              reasons=["Offset detected"],
                              origin=CandidateOrigin.EMBEDDED,
                              verification_mode=VerificationMode.EMBEDDED_PROVENANCE
                          )
                          
                          await pipeline._run_pipeline_logic(1, str(video), wait_seconds=0)
                          mock_trust.assert_called()
                          mock_publish.assert_not_called()
                          mock_remove.assert_called()
                          
                          mock_trust.reset_mock()
                          mock_publish.reset_mock()
                          mock_remove.reset_mock()
                          
                          # Test PASS status (GREEN)
                          def fake_extract_green(vid, out, preferred_lang):
                              with open(out, "w", encoding="utf-8") as f:
                                  f.write("1\n00:00:01,000 --> 00:00:02,000\nGood\n\n")
                              return True
                          mock_extract.side_effect = fake_extract_green
                          mock_trust.return_value = TrustResult(
                              decision=TrustDecision.PASS,
                              score=95,
                              confidence="HIGH",
                              reasons=["Passed embedded provenance"],
                              origin=CandidateOrigin.EMBEDDED,
                              verification_mode=VerificationMode.EMBEDDED_PROVENANCE
                          )
                          mock_publish.return_value = {"published": True, "skipped": False, "reason": "published"}
                          
                          await pipeline._run_pipeline_logic(1, str(video), wait_seconds=0)
                          mock_trust.assert_called()
                          mock_publish.assert_called()
                          mock_remove.assert_not_called()
