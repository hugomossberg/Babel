import pytest
import os
import srt
from datetime import timedelta
import asyncio
from unittest.mock import patch, MagicMock

from app.services.pipeline import SubtitlePipeline
from app.core.db import get_job_by_id

@pytest.fixture
def mock_db_settings(monkeypatch):
    def mock_get_setting(key, default=""):
        settings = {
            "ai_provider": "gemini",
            "gemini_api_key": "dummy_api_key",
            "gemini_model": "gemini-3.5-flash-lite",
            "batch_size": "50",
            "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
            "enable_bazarr_check": "false",
        }
        return settings.get(key, default)
    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)
    monkeypatch.setattr("app.services.translator.get_setting", mock_get_setting)

@pytest.mark.asyncio
async def test_a_target_appears_after_extraction(mock_db_settings, tmp_path, monkeypatch):
    video_path = tmp_path / "test.mkv"
    video_path.touch()
    en_srt = tmp_path / "test.en.srt"
    with open(en_srt, "w") as f:
        f.write("1\n00:00:00,000 --> 00:00:01,000\nHello\n")
    
    pipeline = SubtitlePipeline()
    
    translate_calls = 0
    async def mock_translate_batch(*args, **kwargs):
        nonlocal translate_calls
        translate_calls += 1
        return [{"id": 0, "text": "Hej"}]
    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    
    async def mock_escalate(*args, **kwargs):
        return "Hej"
    monkeypatch.setattr(pipeline.translator, "escalate_single_line", mock_escalate)

    # We mock find_external_subtitle to return None the FIRST time (before extraction)
    # and the sv.srt path the SECOND time (mid-job check).
    sv_srt_path = str(tmp_path / "test.sv.srt")
    
    call_count = 0
    def mock_find(vp, lang):
        nonlocal call_count
        call_count += 1
        if lang == "sv":
            if call_count == 1:
                return None
            else:
                # create a healthy sv.srt to be found (>200 bytes, >5 lines)
                with open(sv_srt_path, "w") as f:
                    lines = []
                    for i in range(1, 10):
                        lines.append(f"{i}\n00:00:0{i},000 --> 00:00:0{i},500\nDetta är en mycket bra svensk text rad {i}\n")
                    f.write("\n".join(lines))
                return sv_srt_path
        if lang == "en":
            return str(en_srt)
        return None
        
    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find):
        res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
        
    assert res["status"] != "failed"
    assert translate_calls == 0  # AI translation skipped
    
    job = get_job_by_id(res["job_id"])
    assert job["status"] == "TRANSLATED" or job["status"] == "ALREADY_EXISTS" or job["status"] == "COMPLETED"
    
@pytest.mark.asyncio
async def test_b_target_appears_before_publish(mock_db_settings, tmp_path, monkeypatch):
    video_path = tmp_path / "test.mkv"
    video_path.touch()
    en_srt = tmp_path / "test.en.srt"
    with open(en_srt, "w") as f:
        f.write("1\n00:00:00,000 --> 00:00:01,000\nHello\n")
        
    pipeline = SubtitlePipeline()
    
    translate_calls = 0
    async def mock_translate_batch(*args, **kwargs):
        nonlocal translate_calls
        translate_calls += 1
        return [{"id": 0, "text": "Hej"}]
    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    
    async def mock_escalate(*args, **kwargs):
        return "Hej"
    monkeypatch.setattr(pipeline.translator, "escalate_single_line", mock_escalate)

    sv_srt_path = str(tmp_path / "test.sv.srt")
    sv_call_count = 0
    def mock_find(vp, lang):
        nonlocal sv_call_count
        if lang == "sv":
            sv_call_count += 1
            if sv_call_count <= 2:
                return None
            else:
                # Before publish (3rd sv call), it finds it
                with open(sv_srt_path, "w") as f:
                    lines = []
                    for i in range(1, 10):
                        lines.append(f"{i}\n00:00:0{i},000 --> 00:00:0{i},500\nSvensk text extern rad {i}\n")
                    f.write("\n".join(lines))
                return sv_srt_path
        if lang == "en":
            return str(en_srt)
        return None
        
    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find):
        res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
        
    assert res["status"] != "failed"
    assert translate_calls == 1  # AI translation happened
    
    # Check that the external target was NOT overwritten
    with open(sv_srt_path, "r") as f:
        content = f.read()
    assert "Svensk text extern" in content
    assert "Hej" not in content

@pytest.mark.asyncio
async def test_c_qa_fail_no_target_created(mock_db_settings, tmp_path, monkeypatch):
    video_path = tmp_path / "test.mkv"
    video_path.touch()
    en_srt = tmp_path / "test.en.srt"
    # Create 100 lines so QA can fail if translated poorly
    with open(en_srt, "w") as f:
        lines = []
        for i in range(1, 101):
            lines.append(f"{i}\n00:00:0{i%10},000 --> 00:00:0{i%10},500\nHello {i}\n")
        f.write("\n".join(lines))
        
    pipeline = SubtitlePipeline()
    
    async def mock_translate_batch(*args, **kwargs):
        # Return empty list to simulate total failure/dropped lines -> QA FAIL
        return []
    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    
    async def mock_escalate(*args, **kwargs):
        return None  # Fail escalation too, so QA fails
    monkeypatch.setattr(pipeline.translator, "escalate_single_line", mock_escalate)

    def mock_find(vp, lang):
        if lang == "en": return str(en_srt)
        return None
        
    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find):
        res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
        
    assert res["status"] in ["failed", "recovering"]
    
    sv_srt_path = tmp_path / "test.sv.srt"
    assert not sv_srt_path.exists()
