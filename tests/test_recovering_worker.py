import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from app.core.db import create_job, update_job, get_job_by_id, DB_PATH
import app.core.db
from app.main import process_one_retry_pass
from app.services.pipeline import pipeline

@pytest.fixture(autouse=True)
def setup_teardown_db(tmp_path):
    app.core.db.DB_PATH = "/tmp/test_recovering.db"
    app.core.db.init_db()
    app.core.db.clear_all_jobs()
    
    video = tmp_path / "test_recovering.mkv"
    video.touch()
    
    yield str(video)
    
    app.core.db.clear_all_jobs()
    import os
    if os.path.exists("/tmp/test_recovering.db"):
        os.remove("/tmp/test_recovering.db")

@pytest.mark.asyncio
async def test_recovering_worker_chain(setup_teardown_db, monkeypatch):
    video_path = setup_teardown_db
    job_id = create_job(video_path, "MANUAL", "test title")
    
    # 1. Job is RECOVERING and next_retry_at is in the past
    past_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    update_job(job_id, status="RECOVERING", next_retry_at=past_time)
    
    # ensure it's not in active paths
    pipeline._active_video_paths.discard(video_path)
    
    # Mock _execute_process_video to just return RECOVERING again
    async def mock_execute(*args, **kwargs):
        update_job(job_id, status="RECOVERING")
        return {"status": "recovering", "job_id": job_id}
    
    monkeypatch.setattr(pipeline, "_execute_process_video", mock_execute)
    
    # 2. Run process_one_retry_pass
    tasks = []
    async for task in process_one_retry_pass():
        tasks.append(task)
        
    assert len(tasks) == 1
    
    # 3. Wait for task to finish
    await asyncio.gather(*tasks)
    
    job = get_job_by_id(job_id)
    assert job["status"] == "RECOVERING"
    
    # Verify it is no longer in active paths
    assert video_path not in pipeline._active_video_paths
