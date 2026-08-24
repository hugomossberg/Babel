import pytest
import os
import asyncio
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
import srt
import json

from app.core import db
from app.core.db import init_db, create_job, get_job_by_id, update_job, clear_all_jobs, set_setting
from app.services.pipeline import pipeline
from app.main import retry_waiting_jobs

@pytest.fixture(autouse=True)
def setup_teardown_db(tmp_path):
    original_db = db.DB_PATH
    test_db = str(tmp_path / "test_babel_recovery_loop.db")
    db.DB_PATH = test_db
    db.init_db()
    db.set_setting("target_language", "sv")
    db.set_setting("languages", json.dumps([{"code": "sv", "name": "Swedish", "enabled": True}]))

    video = tmp_path / "test.mkv"
    video.touch()

    yield str(video)

    db.clear_all_jobs()
    db.DB_PATH = original_db

def make_srt_mock(texts):
    subs = []
    for i, t in enumerate(texts, 1):
        subs.append(srt.Subtitle(i, timedelta(seconds=i), timedelta(seconds=i+1), t))
    return subs

@pytest.mark.asyncio
async def test_never_give_up_recovers_in_loop(setup_teardown_db):
    """A) Primary translation 3 lines, line 2 empty. Recovery round 1 repairs -> QA PASS -> TRANSLATED."""
    video_path = setup_teardown_db
    job_id = create_job(video_path, "MANUAL")

    en_srt_content = "1\n00:00:00,000 --> 00:00:01,000\nLine 1\n\n2\n00:00:01,000 --> 00:00:02,000\nLine 2\n\n3\n00:00:02,000 --> 00:00:03,000\nLine 3\n"

    call_count = {"count": 0}

    async def mock_translate(*args, **kwargs):
        call_count["count"] += 1
        return make_srt_mock(["Detta är svenskt", "", "Mer svensk text"])

    async def mock_escalate(*args, **kwargs):
        return "Ny svensk rad"

    en_srt_path = str(video_path).replace(".mkv", ".en.srt")
    with open(en_srt_path, "w", encoding="utf-8") as f:
        f.write(en_srt_content)

    with patch("app.services.pipeline.find_external_subtitle", return_value=en_srt_path), \
         patch("app.services.translator.SubtitleTranslator.translate_srt_content", side_effect=mock_translate), \
         patch("app.services.translator.SubtitleTranslator.classify_and_recover_identical", return_value=[]), \
             patch("app.services.translator.SubtitleTranslator.translate_batch", return_value=[]), \
         patch("app.services.translator.SubtitleTranslator.escalate_single_line", side_effect=mock_escalate):

        result = await pipeline.process_video_file(video_path, job_id=job_id, force_retranslate=True)

    assert result["status"] == "translated"
    assert call_count["count"] == 1
    job = get_job_by_id(job_id)
    assert job["status"] == "TRANSLATED"

    # Verify the repaired line actually exists in the final subtitle
    sv_srt_path = str(video_path).replace(".mkv", ".sv.srt")
    assert os.path.exists(sv_srt_path)
    with open(sv_srt_path, "r", encoding="utf-8") as f:
        parsed = list(srt.parse(f.read()))

    assert len(parsed) == 3
    assert parsed[0].content == "Detta är svenskt"
    assert parsed[1].content == "Ny svensk rad"
    assert parsed[2].content == "Mer svensk text"

@pytest.mark.asyncio
async def test_never_give_up_fails_all_loops(setup_teardown_db):
    """C) All 3 internal QA loops fail -> status RECOVERING. Background worker picks it up."""
    video_path = setup_teardown_db
    job_id = create_job(video_path, "MANUAL")
    en_srt_content = "1\n00:00:00,000 --> 00:00:01,000\nLine 1\n\n2\n00:00:01,000 --> 00:00:02,000\nLine 2\n\n3\n00:00:02,000 --> 00:00:03,000\nLine 3\n\n4\n00:00:03,000 --> 00:00:04,000\nLine 4\n"

    async def mock_translate(*args, **kwargs):
        return make_srt_mock(["", "", ""])

    en_srt_path = str(video_path).replace(".mkv", ".en.srt")
    with open(en_srt_path, "w", encoding="utf-8") as f:
        f.write(en_srt_content)

    with patch("app.services.pipeline.find_external_subtitle", return_value=en_srt_path), \
         patch("app.services.translator.SubtitleTranslator.translate_srt_content", side_effect=mock_translate), \
         patch("app.services.translator.SubtitleTranslator.classify_and_recover_identical", return_value=[]), \
             patch("app.services.translator.SubtitleTranslator.translate_batch", return_value=[]), \
         patch("app.services.translator.SubtitleTranslator.escalate_single_line", return_value=""):

        result = await pipeline.process_video_file(video_path, job_id=job_id, force_retranslate=True)

    assert result["status"] == "failed"
    job = get_job_by_id(job_id)
    assert job["status"] == "FAILED"
    assert job["next_retry_at"] is None

    past_time = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    update_job(job_id, status="RECOVERING", next_retry_at=past_time)

    with patch("app.services.pipeline.SubtitlePipeline.process_video_file") as mock_process:
        task = asyncio.create_task(retry_waiting_jobs())
        await asyncio.sleep(0.1)
        task.cancel()

        mock_process.assert_called_once()
        job = get_job_by_id(job_id)
        assert job["status"] == "QUEUED"

@pytest.mark.asyncio
async def test_recovering_is_not_dead_state(setup_teardown_db):
    """Test RECOVERING -> worker picks same job_id -> processing resumes -> QA PASS -> TRANSLATED"""
    video_path = setup_teardown_db
    job_id = create_job(video_path, "MANUAL")
    past_time = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    update_job(job_id, status="RECOVERING", next_retry_at=past_time)

    en_srt_content = "1\n00:00:00,000 --> 00:00:01,000\nLine 1\n\n2\n00:00:01,000 --> 00:00:02,000\nLine 2\n\n3\n00:00:02,000 --> 00:00:03,000\nLine 3\n\n4\n00:00:03,000 --> 00:00:04,000\nLine 4\n"
    async def mock_translate_good(*args, **kwargs):
        return make_srt_mock(["Detta är svensk 1", "Detta är svensk 2", "Detta är svensk 3", "Detta är svensk 4"])

    en_srt_path = str(video_path).replace(".mkv", ".en.srt")
    with open(en_srt_path, "w", encoding="utf-8") as f:
        f.write(en_srt_content)

    with patch("app.services.pipeline.find_external_subtitle", return_value=en_srt_path), \
         patch("app.services.translator.SubtitleTranslator.translate_srt_content", side_effect=mock_translate_good), \
         patch("app.services.translator.SubtitleTranslator.classify_and_recover_identical", return_value=[]), \
             patch("app.services.translator.SubtitleTranslator.translate_batch", return_value=[]), \
         patch("app.services.translator.SubtitleTranslator.escalate_single_line", return_value=""):

        from app.main import process_one_retry_pass
        tasks = [t async for t in process_one_retry_pass()]
        await asyncio.gather(*tasks)

    job = get_job_by_id(job_id)
    assert job["status"] == "TRANSLATED"

@pytest.mark.asyncio
async def test_generic_exception_classification(setup_teardown_db):
    """Ensure generic exceptions are classified correctly."""
    video_path = setup_teardown_db
    job_id = create_job(video_path, "MANUAL")
    en_srt_content = "1\n00:00:00,000 --> 00:00:01,000\nLine 1\n\n2\n00:00:01,000 --> 00:00:02,000\nLine 2\n\n3\n00:00:02,000 --> 00:00:03,000\nLine 3\n\n4\n00:00:03,000 --> 00:00:04,000\nLine 4\n"

    async def mock_translate_timeout(*args, **kwargs):
        raise Exception("Request timeout after 30s")

    en_srt_path = str(video_path).replace(".mkv", ".en.srt")
    with open(en_srt_path, "w", encoding="utf-8") as f:
        f.write(en_srt_content)

    with patch("app.services.pipeline.find_external_subtitle", return_value=en_srt_path), \
         patch("app.services.translator.SubtitleTranslator.translate_srt_content", side_effect=mock_translate_timeout):
        await pipeline.process_video_file(video_path, job_id=job_id, force_retranslate=True)

    job = get_job_by_id(job_id)
    assert job["status"] == "WAITING_PROVIDER"

    async def mock_translate_terminal(*args, **kwargs):
        raise Exception("Permission denied: /path/to/file")

    with patch("app.services.pipeline.find_external_subtitle", return_value=en_srt_path), \
         patch("app.services.translator.SubtitleTranslator.translate_srt_content", side_effect=mock_translate_terminal):
        await pipeline.process_video_file(video_path, job_id=job_id, force_retranslate=True)

    job = get_job_by_id(job_id)
    assert job["status"] == "FAILED"

    async def mock_translate_generic(*args, **kwargs):
        raise Exception("Something weird happened in JSON parsing")

    with patch("app.services.pipeline.find_external_subtitle", return_value=en_srt_path), \
         patch("app.services.translator.SubtitleTranslator.translate_srt_content", side_effect=mock_translate_generic):
        await pipeline.process_video_file(video_path, job_id=job_id, force_retranslate=True)

    job = get_job_by_id(job_id)
    assert job["status"] == "RETRY_PENDING"

@pytest.mark.asyncio
async def test_persistent_retry_count_and_backoff(setup_teardown_db):
    """Ensure retry_count and backoff increment across exhausted attempts and survive restart."""
    video_path = setup_teardown_db
    job_id = create_job(video_path, "MANUAL")
    en_srt_content = "1\n00:00:00,000 --> 00:00:01,000\nLine 1\n\n2\n00:00:01,000 --> 00:00:02,000\nLine 2\n\n3\n00:00:02,000 --> 00:00:03,000\nLine 3\n\n4\n00:00:03,000 --> 00:00:04,000\nLine 4\n"

    from app.services.translator import ProviderUnavailableError
    async def mock_translate_fail(*args, **kwargs):
        raise ProviderUnavailableError("Temporary upstream 503 error")

    en_srt_path = str(video_path).replace(".mkv", ".en.srt")
    with open(en_srt_path, "w", encoding="utf-8") as f:
        f.write(en_srt_content)

    # Attempt 0 -> exhaustion -> retry 1, ~2 min
    with patch("app.services.pipeline.find_external_subtitle", return_value=en_srt_path), \
         patch("app.services.translator.SubtitleTranslator.translate_srt_content", side_effect=mock_translate_fail):

        await pipeline.process_video_file(video_path, job_id=job_id, force_retranslate=True)

    job = get_job_by_id(job_id)
    assert job["status"] == "WAITING_PROVIDER"
    assert job["retry_count"] == 1
    t1 = datetime.fromisoformat(job["next_retry_at"])
    now = datetime.now(timezone.utc)
    diff_mins = (t1 - now).total_seconds() / 60
    assert 1.5 <= diff_mins <= 2.5

    # Simulate restart
    init_db()

    # Attempt 1 -> exhaustion -> retry 2, ~5 min
    with patch("app.services.pipeline.find_external_subtitle", return_value=en_srt_path), \
         patch("app.services.translator.SubtitleTranslator.translate_srt_content", side_effect=mock_translate_fail):

        await pipeline.process_video_file(video_path, job_id=job_id, force_retranslate=True)

    job = get_job_by_id(job_id)
    assert job["status"] == "WAITING_PROVIDER"
    assert job["retry_count"] == 2
    t2 = datetime.fromisoformat(job["next_retry_at"])
    now = datetime.now(timezone.utc)
    diff_mins = (t2 - now).total_seconds() / 60
    assert 4 <= diff_mins <= 5.5

    # Simulate restart
    init_db()

    # Attempt 2 -> exhaustion -> retry 3, ~15 min
    with patch("app.services.pipeline.find_external_subtitle", return_value=en_srt_path), \
         patch("app.services.translator.SubtitleTranslator.translate_srt_content", side_effect=mock_translate_fail):

        await pipeline.process_video_file(video_path, job_id=job_id, force_retranslate=True)

    job = get_job_by_id(job_id)
    assert job["status"] == "WAITING_PROVIDER"
    assert job["retry_count"] == 3
    t3 = datetime.fromisoformat(job["next_retry_at"])
    now = datetime.now(timezone.utc)
    diff_mins = (t3 - now).total_seconds() / 60
    assert 14 <= diff_mins <= 15.5
