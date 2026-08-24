import pytest
import os
import srt
from datetime import timedelta
from unittest.mock import patch

from app.core.db import init_db, create_job, get_job_by_id
from app.services.pipeline import SubtitlePipeline
from app.core.quota import block_provider, unblock_provider

@pytest.fixture(autouse=True)
def setup_teardown_db(tmp_path):
    db_file = str(tmp_path / "babel_bazarr_quota_test.db")
    import app.core.db as db_module
    import app.core.quota as quota_module

    original_db = db_module.DB_PATH
    db_module.DB_PATH = db_file
    quota_module.DB_PATH = db_file

    db_module.init_db()
    unblock_provider("gemini")
    yield tmp_path
    unblock_provider("gemini")

    db_module.DB_PATH = original_db
    quota_module.DB_PATH = original_db

@pytest.mark.asyncio
async def test_a_target_subtitle_already_exists(setup_teardown_db):
    tmp_path = setup_teardown_db
    block_provider("gemini", "TEST_REASON")

    video_path = tmp_path / "test_a.mkv"
    video_path.touch()
    
    en_srt = tmp_path / "test_a.en.srt"
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write("1\n00:00:01,000 --> 00:00:02,000\nThis English text is much much much much longer so that we get more than 100 bytes in size. Otherwise Babel misses that the file is valid. That would be quite tragic\n")

    target_srt = tmp_path / "test_a.sv.srt"
    with open(target_srt, "w", encoding="utf-8") as f:
        f.write("1\n00:00:01,000 --> 00:00:02,000\nDenna svenska text ar mycket mycket mycket mycket langre sa att vi far mer an 100 bytes i storlek. Annars missar Babel att filen ar giltig. Det vore ju lite val tragiskt från Bazarr\n")

    pipeline = SubtitlePipeline()
    job_id = create_job(str(video_path))

    def fake_get_setting(key, default=None):
        if key == "languages": return '[{"name": "Swedish", "code": "sv", "enabled": true}]'
        if key == "extract_target_embedded": return "false"
        if key == "extract_source_embedded": return "false"
        if key == "enable_bazarr_check": return "false"
        if key == "auto_repair_unhealthy": return "false"
        return default

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting):
         res = await pipeline.process_video_file(str(video_path), job_id=job_id)

    assert res["status"] == "skipped"

@pytest.mark.asyncio
async def test_b_hybrid_bazarr_fulfills_job(setup_teardown_db):
    tmp_path = setup_teardown_db
    block_provider("gemini", "TEST_REASON")

    video_path = tmp_path / "test_b.mkv"
    video_path.touch()

    source_srt = tmp_path / "test_b.en.srt"
    with open(source_srt, "w", encoding="utf-8") as f:
        f.write("1\n00:00:01,000 --> 00:00:02,000\nThis English text is much much much much longer so that we get more than 100 bytes in size. Otherwise Babel misses that the file is valid. That would be quite tragic\n")

    pipeline = SubtitlePipeline()
    job_id = create_job(str(video_path))

    def fake_get_setting(key, default=None):
        if key == "languages": return '[{"name": "Swedish", "code": "sv", "enabled": true}]'
        if key == "extract_target_embedded": return "false"
        if key == "extract_source_embedded": return "false"
        if key == "enable_bazarr_check": return "true"
        if key == "auto_repair_unhealthy": return "false"
        return default

    # We mock find_external_subtitle to simulate Bazarr having downloaded the file by the time the pipeline checks for it
    original_find = __import__("app.services.bazarr_checker").services.bazarr_checker.find_external_subtitle
    call_count = {"sv": 0}
    def mock_find(path, code):
        if code == "sv":
            call_count["sv"] += 1
            if call_count["sv"] == 1:
                # First check (before Bazarr) -> miss
                return None
            else:
                # Second check (after Bazarr) -> hit!
                target_srt = tmp_path / "test_b.sv.srt"
                with open(target_srt, "w", encoding="utf-8") as f:
                    f.write("1\n00:00:01,000 --> 00:00:02,000\nDenna svenska text ar mycket mycket mycket mycket langre sa att vi far mer an 100 bytes i storlek. Annars missar Babel att filen ar giltig. Det vore ju lite val tragiskt från Bazarr\n")
                return str(target_srt)
        return original_find(path, code)

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting), \
         patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find):
         res = await pipeline.process_video_file(str(video_path), job_id=job_id)

    assert res["status"] == "skipped"
    assert res["reason"] == "bazarr_downloaded"

@pytest.mark.asyncio
async def test_c_hybrid_bazarr_miss_defers_job(setup_teardown_db):
    tmp_path = setup_teardown_db
    block_provider("gemini", "TEST_REASON")

    video_path = tmp_path / "test_c.mkv"
    video_path.touch()

    source_srt = tmp_path / "test_c.en.srt"
    with open(source_srt, "w", encoding="utf-8") as f:
        f.write("1\n00:00:01,000 --> 00:00:02,000\nThis English text is much much much much longer so that we get more than 100 bytes in size. Otherwise Babel misses that the file is valid. That would be quite tragic\n")

    pipeline = SubtitlePipeline()
    job_id = create_job(str(video_path))

    def fake_get_setting(key, default=None):
        if key == "languages": return '[{"name": "Swedish", "code": "sv", "enabled": true}]'
        if key == "extract_target_embedded": return "false"
        if key == "extract_source_embedded": return "false"
        if key == "enable_bazarr_check": return "true"
        if key == "auto_repair_unhealthy": return "false"
        return default

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting):
         res = await pipeline.process_video_file(str(video_path), job_id=job_id)

    assert res["status"] == "deferred"

