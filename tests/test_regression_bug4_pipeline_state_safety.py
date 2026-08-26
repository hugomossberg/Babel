import os
import asyncio
import pytest
from unittest.mock import patch, MagicMock
from app.services.pipeline import SubtitlePipeline
from app.core import db


@pytest.fixture(autouse=True)
def setup_tmp_db(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test_pipe_safety.db")
    monkeypatch.setattr("app.core.db.DB_PATH", test_db)
    db.init_db()
    yield test_db


@pytest.mark.asyncio
async def test_pipeline_crash_early_stage_sets_scheduled_or_failed_status(tmp_path):
    """Verify that an unexpected exception in early pipeline logic transitions job from TRANSLATING to a non-active terminal or scheduled retry state."""
    job_id = db.create_job("/media/test_video.mkv")
    assert job_id is not None

    pipeline = SubtitlePipeline()
    video_path = str(tmp_path / "test_video.mkv")
    with open(video_path, "w") as f:
        f.write("dummy")

    # Inject early exception during container track inspection
    with patch("app.services.pipeline.inspect_mkv_tracks", side_effect=RuntimeError("Container probe corrupted")):
        with patch("app.services.pipeline.get_setting") as mock_gs:
            mock_gs.side_effect = lambda k, d="": '[{"name": "Swedish", "code": "sv", "enabled": true}]' if k == "languages" else d

            # Run pipeline
            res = await pipeline._run_pipeline_logic(job_id, video_path, wait_seconds=0)

            # Check job state in DB
            job_data = db.get_job_by_id(job_id)
            assert job_data["status"] not in ("TRANSLATING", "ESCALATING")
            # Must be a valid terminal or scheduled recovery state with next_retry_at or FAILED
            assert job_data["status"] in ("RETRY_PENDING", "FAILED", "WAITING_PROVIDER", "WAITING_SOURCE")


@pytest.mark.asyncio
async def test_process_video_file_outer_boundary_catches_unhandled_crash(tmp_path):
    """Verify that if _execute_process_video crashes with an unhandled exception, the outer boundary marks job as FAILED and cleans up active tasks."""
    video_path = str(tmp_path / "crash_video.mkv")
    with open(video_path, "w") as f:
        f.write("dummy")

    job_id = db.create_job(video_path)
    db.update_job(job_id, status="TRANSLATING")

    pipeline = SubtitlePipeline()

    with patch.object(pipeline, "_execute_process_video", side_effect=ZeroDivisionError("Fatal unhandled calculation error")):
        with pytest.raises(ZeroDivisionError):
            await pipeline.process_video_file(video_path, job_id=job_id)

        # Assert job was transitioned to FAILED by the outer safety boundary
        job_data = db.get_job_by_id(job_id)
        assert job_data["status"] == "FAILED"
        assert "Unhandled pipeline crash" in job_data["error_message"]

        # Assert task cleanup occurred
        assert job_id not in pipeline._active_tasks
        assert os.path.normpath(video_path) not in pipeline._active_video_paths


@pytest.mark.asyncio
async def test_cancelled_job_leaves_no_lingering_active_state(tmp_path):
    """Verify that cancelled jobs transition to CANCELLED and clean up all resources."""
    video_path = str(tmp_path / "cancel_video.mkv")
    with open(video_path, "w") as f:
        f.write("dummy")

    job_id = db.create_job(video_path)
    pipeline = SubtitlePipeline()

    async def slow_execution(*args, **kwargs):
        await asyncio.sleep(10)

    with patch.object(pipeline, "_execute_process_video", side_effect=slow_execution):
        task = asyncio.create_task(pipeline.process_video_file(video_path, job_id=job_id))
        await asyncio.sleep(0.05)

        # Cancel job
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        job_data = db.get_job_by_id(job_id)
        assert job_data["status"] == "CANCELLED"
        assert job_id not in pipeline._active_tasks
        assert os.path.normpath(video_path) not in pipeline._active_video_paths
