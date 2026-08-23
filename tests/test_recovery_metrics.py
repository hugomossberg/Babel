import pytest
from unittest.mock import patch, AsyncMock
from app.services.pipeline import pipeline
from app.core.db import create_job, get_job_by_id
import asyncio
import os

@pytest.mark.asyncio
async def test_recovery_metrics_count(tmp_path):
    # Setup
    video_path = tmp_path / "test.mkv"
    video_path.touch()
    en_srt_path = tmp_path / "test.en.srt"
    
    job_id = create_job(str(video_path))
    
    # We will simulate a scenario where source has 2 lines.
    # Initially: Line 0 translated OK, Line 1 returned as English (QA fail)
    # Targeted Recovery: Line 1 translated OK
    # We expect "1 translated on recovery" in logs.
    
    mock_source_srt = "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n2\n00:00:03,000 --> 00:00:04,000\nWorld\n"
    
    with open(en_srt_path, "w") as f:
        f.write(mock_source_srt)
        
    call_count = 0
    async def mock_translate_batch(batch, **kwargs):
        nonlocal call_count
        call_count += 1
        res = []
        if call_count == 1:
            # First pass: translate "Hello" -> "Hej", leave "World" as "World"
            for item in batch:
                if item["text"] == "Hello": res.append({"id": item["id"], "text": "Hej"})
                else: res.append({"id": item["id"], "text": "World"})
        else:
            # Recovery pass: translate "World" -> "Värld"
            for item in batch:
                res.append({"id": item["id"], "text": "Värld"})
        return res

    with patch("app.services.pipeline.find_external_subtitle", side_effect=lambda vp, lang: str(en_srt_path) if lang == "en" else None), \
         patch("app.services.pipeline.extract_embedded_srt", return_value=False), \
         patch("app.services.pipeline.get_setting", side_effect=lambda k, d=None: {"extract_source_embedded": "false", "languages": '[{"name": "Swedish", "code": "sv", "enabled": true}]', "qa_threshold_dropped": "0", "qa_threshold_sync": "100"}.get(k, d)), \
         patch("app.services.pipeline.SubtitlePipeline.trigger_bazarr_search", new_callable=AsyncMock), \
         patch("app.services.pipeline.SubtitlePipeline._get_semaphore", return_value=asyncio.Semaphore(1)), \
         patch("app.services.translator.SubtitleTranslator.translate_batch", side_effect=mock_translate_batch), \
         patch("asyncio.sleep", new_callable=AsyncMock):
         
         await pipeline.process_video_file(str(video_path), job_id=job_id)
         
    job = get_job_by_id(job_id)
    logs = "".join(job["logs"])
    assert "1 translated on recovery" in logs or "recovered 1/1" in logs
