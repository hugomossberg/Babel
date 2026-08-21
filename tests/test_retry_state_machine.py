import pytest
import os
import asyncio
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
from app.core.db import init_db, create_job, get_job_by_id, update_job, clear_all_jobs, claim_job_for_retry, DB_PATH
from app.services.pipeline import pipeline
from app.services.translator import ProviderUnavailableError
from app.main import retry_waiting_jobs
import sqlite3

@pytest.fixture(autouse=True)
def setup_teardown_db(tmp_path):
    # Use a test db
    import app.core.db
    original_db = app.core.db.DB_PATH
    app.core.db.DB_PATH = "/tmp/test_babel_retry.db"
    app.core.db.init_db()
    app.core.db.clear_all_jobs()

    video = tmp_path / "test.mkv"
    video.touch()

    yield str(video)

    app.core.db.clear_all_jobs()
    app.core.db.DB_PATH = original_db

@pytest.mark.asyncio
async def test_provider_error_creates_waiting_state(setup_teardown_db):
    video_path = setup_teardown_db
    job_id = create_job(video_path, "MANUAL", "test title")

    with patch("app.services.translator.SubtitleTranslator.translate_srt_content", side_effect=ProviderUnavailableError("API rate limit exceeded")):
        # Mock so it thinks it found English subs
        with patch("app.services.pipeline.find_external_subtitle", side_effect=lambda p, l: video_path if l == "en" else None):
                with patch("builtins.open", MagicMock(side_effect=lambda *args, **kwargs: MagicMock(__enter__=lambda *a: MagicMock(read=lambda: "1\n00:00:00,000 --> 00:00:01,000\nHello\n"), __exit__=lambda *a: None))):
                    result = await pipeline.process_video_file(video_path, job_id=job_id, force_retranslate=True)
                    print(f"RESULT: {result}")

    assert result["status"] == "waiting_provider"
    assert result["job_id"] == job_id

    job = get_job_by_id(job_id)
    assert job["status"] == "WAITING_PROVIDER"
    assert job["retry_count"] == 1
    assert "API rate limit" in job["error_message"]
    assert job["next_retry_at"] is not None

@pytest.mark.asyncio
async def test_retry_loop_picks_up_waiting_job():
    job_id = create_job("test2.mkv", "MANUAL", "test title")
    # Simulate a job that failed in the past
    past_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    update_job(job_id, status="WAITING_PROVIDER", next_retry_at=past_time)

    with patch("app.services.pipeline.SubtitlePipeline.process_video_file") as mock_process:
        # Run one iteration of the loop
        task = asyncio.create_task(retry_waiting_jobs())
        await asyncio.sleep(0.1)  # Let it run
        task.cancel()

        mock_process.assert_called_once()
        args, kwargs = mock_process.call_args
        assert args[0] == "test2.mkv"
        assert kwargs["job_id"] == job_id

        job = get_job_by_id(job_id)
        assert job["status"] == "QUEUED"

@pytest.mark.asyncio
async def test_restart_recovery_changes_state():
    job_id1 = create_job("test3.mkv", "MANUAL")
    job_id2 = create_job("test4.mkv", "MANUAL")
    update_job(job_id1, status="TRANSLATING")
    update_job(job_id2, status="WAITING_PROVIDER")

    import app.core.db
    app.core.db.init_db()

    job1 = get_job_by_id(job_id1)
    job2 = get_job_by_id(job_id2)

    assert job1["status"] == "RETRY_PENDING"
    assert job2["status"] == "RETRY_PENDING"

def test_atomic_claim():
    job_id = create_job("test5.mkv", "MANUAL")
    update_job(job_id, status="WAITING_PROVIDER")

    success1 = claim_job_for_retry(job_id)
    success2 = claim_job_for_retry(job_id)

    assert success1 is True
    assert success2 is False  # Cannot claim twice

    job = get_job_by_id(job_id)
    assert job["status"] == "QUEUED"

def test_claim_job_for_retry_partial():
    # Regression test: PARTIAL jobs should be claimable
    job_id = create_job("test_partial.mkv")
    update_job(job_id, status="PARTIAL")

    success1 = claim_job_for_retry(job_id)
    success2 = claim_job_for_retry(job_id)

    assert success1 is True
    assert success2 is False  # Cannot claim twice

    job = get_job_by_id(job_id)
    assert job["status"] == "QUEUED"

def test_waiting_source_backoff(tmp_path):
    # Test missing source goes to WAITING_SOURCE with bounded backoff and eventually fails
    from app.services.pipeline import pipeline
    from unittest.mock import patch, AsyncMock
    import asyncio

    video_path = tmp_path / "no_source.mkv"
    video_path.touch()

    job_id = create_job(str(video_path))

    with patch("app.services.pipeline.find_external_subtitle", return_value=None), \
         patch("app.services.pipeline.extract_embedded_srt", return_value=False), \
         patch("app.services.pipeline.get_setting", side_effect=lambda k, d=None: {"extract_source_embedded": "true", "languages": '[{"name": "Swedish", "code": "sv", "enabled": true}]'}.get(k, d)), \
         patch("app.services.pipeline.SubtitlePipeline.trigger_bazarr_search", new_callable=AsyncMock), \
         patch("asyncio.sleep", new_callable=AsyncMock):

        # 1st try -> 1 min backoff
        asyncio.run(pipeline.process_video_file(str(video_path), job_id=job_id))
        job = get_job_by_id(job_id)
        assert job["status"] == "WAITING_SOURCE"
        assert job["retry_count"] == 1

        # 2nd try -> 5 min backoff
        asyncio.run(pipeline.process_video_file(str(video_path), job_id=job_id))
        job = get_job_by_id(job_id)
        assert job["status"] == "WAITING_SOURCE"
        assert job["retry_count"] == 2

        # 3rd try -> 15 min backoff
        asyncio.run(pipeline.process_video_file(str(video_path), job_id=job_id))
        job = get_job_by_id(job_id)
        assert job["status"] == "WAITING_SOURCE"
        assert job["retry_count"] == 3

        # 4th try -> 30 min backoff
        asyncio.run(pipeline.process_video_file(str(video_path), job_id=job_id))
        job = get_job_by_id(job_id)
        assert job["status"] == "WAITING_SOURCE"
        assert job["retry_count"] == 4

        # 5th try -> FAILED
        asyncio.run(pipeline.process_video_file(str(video_path), job_id=job_id))
        job = get_job_by_id(job_id)
        assert job["status"] == "FAILED"
        assert "No English subtitle source found" in job["error_message"]

@pytest.mark.asyncio
async def test_waiting_source_claimed_and_retried():
    from app.main import process_one_retry_pass
    job_id = create_job("due_source.mkv")

    # Set status to WAITING_SOURCE, next_retry_at to past
    now = datetime.now(timezone.utc)
    past = (now - timedelta(minutes=1)).isoformat()
    update_job(job_id, status="WAITING_SOURCE", next_retry_at=past)

    # Mock pipeline so we can verify it's called
    with patch("app.main.pipeline.process_video_file") as mock_process:
        async for _ in process_one_retry_pass():
            pass

    # Should be claimed (QUEUED) and pipeline called
    job = get_job_by_id(job_id)
    assert job["status"] == "QUEUED"
    mock_process.assert_called_once_with("due_source.mkv", event_source="RETRY", title="due_source.mkv", job_id=job_id)
