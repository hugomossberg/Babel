import pytest
import os
import json
from unittest.mock import AsyncMock, patch
import srt
from datetime import timedelta
from app.services.translator import SubtitleTranslator, ProviderUnavailableError

@pytest.mark.asyncio
async def test_resume_translation(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    # Create the db directory
    os.makedirs(db_path.parent, exist_ok=True)
    monkeypatch.setattr("app.services.translator.DB_PATH", str(db_path))
    
    # Mock update_job to track progress
    job_id = 999
    progress_calls = []
    def mock_update_job(jid, **kwargs):
        if "processed_lines" in kwargs:
            progress_calls.append(kwargs["processed_lines"])
    monkeypatch.setattr("app.services.translator.update_job", mock_update_job)
    
    # Mock translate_batch
    translator = SubtitleTranslator()
    
    # Generate 400 subs
    subs = []
    for i in range(400):
        subs.append(srt.Subtitle(i, start=timedelta(seconds=i), end=timedelta(seconds=i+1), content=f"Line {i}"))
        
    call_count = {"count": 0}
    translated_batches = []
    
    async def mock_translate_batch(payload, **kwargs):
        translated_batches.append(payload)
        call_count["count"] += 1
        # Each batch is 50 lines. 
        # Fail after 3 batches (150 lines)
        if call_count["count"] > 3:
            raise ProviderUnavailableError("Provider failed")
            
        results = []
        for p in payload:
            results.append({"id": p["id"], "text": f"Translated {p['id']}"})
        return results

    with patch.object(translator, "translate_batch", new=AsyncMock(side_effect=mock_translate_batch)):
        try:
            await translator.translate_srt_content(subs, job_id=job_id, batch_size=50)
        except ProviderUnavailableError:
            pass
            
    # Check partial file exists
    data_dir = os.path.dirname(str(db_path))
    partial_file = os.path.join(data_dir, f"job_{job_id}_partial.json")
    assert os.path.exists(partial_file)
    
    with open(partial_file, "r") as f:
        partial_data = json.load(f)
    assert len(partial_data.get("lines", {})) == 150
    assert progress_calls[-1] == 150
    
    # NOW RESTART
    call_count["count"] = 0
    translated_batches.clear()
    progress_calls.clear()
    
    async def mock_translate_batch_resume(payload, **kwargs):
        translated_batches.append(payload)
        results = []
        for p in payload:
            results.append({"id": p["id"], "text": f"Translated {p['id']}"})
        return results
        
    with patch.object(translator, "translate_batch", new=AsyncMock(side_effect=mock_translate_batch_resume)):
        final_subs = await translator.translate_srt_content(subs, job_id=job_id, batch_size=50)
        
    # Check that it did NOT call translate_batch for the first 150 lines
    assert len(translated_batches) == 5  # 400 - 150 = 250 lines -> 5 batches
    assert translated_batches[0][0]["id"] == 150
    
    # Check progress bar calls
    assert progress_calls == [50, 100, 150, 200, 250, 300, 350, 400]
    
    # Check final result
    assert len(final_subs) == 400
    for i in range(400):
        assert final_subs[i].content == f"Translated {i}"
