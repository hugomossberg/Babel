import pytest
import asyncio
from unittest.mock import patch
from app.services.pipeline import qa_gate
from app.core.validator import check_dropped_lines
from app.services.translator import validate_classifier_output
import srt
from datetime import timedelta, datetime, timezone
from app.main import app, process_one_retry_pass
from fastapi.testclient import TestClient

def test_qa_dropped_counts():
    source = [srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="Hello")]
    target_0 = [srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="Hej")]
    target_1 = [srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="")]
    
    assert qa_gate(source, target_0, target_lang_code="sv")["passed"] is True
    res1 = qa_gate(source, target_1, target_lang_code='sv')
    assert res1['passed'] is False
    assert any('dropped' in i for i in res1['issues'])

def test_dropped_dialogue_with_i():
    source = [srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="I need to tell you something.")]
    target = [srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="<i></i>")]
    
    count, details = check_dropped_lines(source, target)
    assert count == 1
    
    res = qa_gate(source, target, target_lang_code="sv")
    assert res['passed'] is False

def test_sdh_i_not_dropped():
    source = [srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="<i></i>")]
    target = [srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="<i></i>")]
    
    count, details = check_dropped_lines(source, target)
    assert count == 0
    assert qa_gate(source, target, target_lang_code="sv")["passed"] is True

def test_adversarial_keep():
    raw = '{"results": [{"id": 1, "action": "keep", "reason": "non_verbal"}]}'
    items = [{"id": 1, "text": "I don't know what you're talking about."}]
    res = validate_classifier_output(raw, items)
    assert res[0]["text"] == ""

    raw2 = '{"results": [{"id": 1, "action": "keep", "reason": "proper_noun"}]}'
    items2 = [{"id": 1, "text": "The quick brown fox jumps over the lazy dog"}]
    res2 = validate_classifier_output(raw2, items2)
    assert res2[0]["text"] == ""

def test_675_valid_cues(tmp_path):
    source = [srt.Subtitle(index=i, start=timedelta(seconds=i), end=timedelta(seconds=i+1), content=f"Line {i}") for i in range(1, 676)]
    target = [srt.Subtitle(index=i, start=timedelta(seconds=i), end=timedelta(seconds=i+1), content=f"Rad {i}") for i in range(1, 676)]
    
    assert len(source) == 675
    res = qa_gate(source, target, target_lang_code="sv")
    assert res['passed'] is False
    
    from app.core.cleaner import subs_to_srt_string
    text = subs_to_srt_string(target)
    temp_srt = tmp_path / "test_675.srt"
    temp_srt.write_text(text, encoding="utf-8")
        
    parsed = list(srt.parse(temp_srt.read_text(encoding="utf-8")))
    assert len(parsed) == 675

@pytest.mark.asyncio
async def test_retry_pending_timestamp():
    now = datetime.now(timezone.utc)
    future = now + timedelta(minutes=10)
    past = now - timedelta(minutes=10)
    
    jobs = [
        {"id": 1, "status": "RETRY_PENDING", "next_retry_at": future.isoformat(), "updated_at": past.isoformat()},
        {"id": 2, "status": "RECOVERING", "next_retry_at": past.isoformat(), "updated_at": past.isoformat()},
        {"id": 3, "status": "WAITING_PROVIDER", "updated_at": (now - timedelta(minutes=10)).isoformat()},
        {"id": 4, "status": "PARTIAL", "next_retry_at": past.isoformat(), "updated_at": past.isoformat()},
        {"id": 5, "status": "PARTIAL", "next_retry_at": future.isoformat(), "updated_at": past.isoformat()}
    ]
    
    with patch("app.main.get_jobs_by_status", return_value=jobs):
        with patch("app.core.db.claim_job_for_retry", return_value=True) as mock_claim:
            with patch("app.main.pipeline.process_video_file") as mock_process:
                async for t in process_one_retry_pass():
                    pass
                claims = [call[0][0] for call in mock_claim.call_args_list]
                assert 1 not in claims
                assert 2 in claims
                assert 3 in claims
                assert 4 in claims
                assert 5 not in claims

def test_webhook_unauthorized():
    client = TestClient(app)
    with patch("app.main.AUTH_USERNAME", "admin"), patch("app.main.AUTH_PASSWORD", "pass"):
        resp = client.post("/webhook/process", json={"video_path": "/fake/video.mkv"})
        assert resp.status_code == 401

@pytest.mark.asyncio
async def test_active_video_race():
    from app.services.pipeline import pipeline
    pipeline._active_video_paths.add("/tmp/race.mkv")
    
    from app.core.db import create_job, get_job_by_id, init_db, update_job
    import sqlite3
    init_db()
    from app.core.db import DB_PATH
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM jobs")
    conn.commit()
    conn.close()

    job_id = create_job("/tmp/race.mkv")
    update_job(job_id, status="QUEUED")
    
    res = await pipeline.process_video_file("/tmp/race.mkv", job_id=job_id)
    assert res["status"] == "skipped"
    job = get_job_by_id(job_id)
    assert job["status"] == "RECOVERING"
    pipeline._active_video_paths.discard("/tmp/race.mkv")
