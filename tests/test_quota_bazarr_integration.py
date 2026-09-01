import pytest
import os
import srt
from datetime import timedelta
from unittest.mock import patch

from app.core.db import init_db, create_job, get_job_by_id
from app.services.pipeline import SubtitlePipeline
from app.core.quota import block_provider, unblock_provider

HEALTHY_EN_SRT = """1
00:00:01,000 --> 00:00:04,000
Welcome to the kitchen tonight.

2
00:00:05,000 --> 00:00:08,000
We have a major challenge ahead of us.

3
00:00:09,000 --> 00:00:12,000
All the ingredients are ready on the counter.

4
00:00:13,000 --> 00:00:16,000
You have sixty minutes to cook.

5
00:00:17,000 --> 00:00:20,000
Your time starts right now.

6
00:00:21,000 --> 00:00:24,000
Good luck to all of you.
"""

HEALTHY_SV_SRT = """1
00:00:01,000 --> 00:00:04,000
Välkommen till köket ikväll.

2
00:00:05,000 --> 00:00:08,000
Vi har en stor utmaning framför oss.

3
00:00:09,000 --> 00:00:12,000
Alla ingredienser är redo på bänken.

4
00:00:13,000 --> 00:00:16,000
Ni har sextio minuter på er att laga mat.

5
00:00:17,000 --> 00:00:20,000
Er tid börjar precis nu.

6
00:00:21,000 --> 00:00:24,000
Lycka till allihop.
"""

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
        f.write(HEALTHY_EN_SRT)

    target_srt = tmp_path / "test_a.sv.srt"
    with open(target_srt, "w", encoding="utf-8") as f:
        f.write(HEALTHY_SV_SRT)

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
        f.write(HEALTHY_EN_SRT)

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
                    f.write(HEALTHY_SV_SRT)
                return str(target_srt)
        return original_find(path, code)

    from app.services.source_resolver import BazarrResult, BazarrResultCode
    async def mock_bazarr_trigger(*args, language="sv", **kwargs):
        return BazarrResult(code=BazarrResultCode.TRIGGERED, language=language, detail="Search accepted")
    pipeline.trigger_bazarr_search = mock_bazarr_trigger

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
        f.write(HEALTHY_EN_SRT)

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

