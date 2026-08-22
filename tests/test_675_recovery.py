import pytest
import os
import srt
from datetime import timedelta
from unittest.mock import patch
from app.services.pipeline import SubtitlePipeline

@pytest.mark.asyncio
async def test_675_targeted_recovery(tmp_path):
    pipeline = SubtitlePipeline()
    video_path = tmp_path / "video.mkv"
    video_path.touch()
    
    # Create REAL source subtitle on disk
    source_subs = [srt.Subtitle(index=i, start=timedelta(seconds=i), end=timedelta(seconds=i+1), content=f"Source {i}") for i in range(1, 676)]
    source_srt_path = str(video_path).replace(".mkv", ".en.srt")
    with open(source_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(source_subs))
    
    # Configure target language to be 'sv' so it generates video.sv.srt
    def fake_get_setting(key, default=None):
        if key == "languages":
            return '[{"name": "Swedish", "code": "sv", "enabled": true}]'
        if key == "auto_repair_unhealthy": return "false"
        if key == "extract_target_embedded": return "false"
        if key == "extract_source_embedded": return "false"
        if key == "enable_bazarr_check": return "false"
        return default
        
    call_count = {"translate": 0, "escalate": 0}

    def fake_translate(*args, **kwargs):
        call_count["translate"] += 1
        # Generate 675 cues. Drop index 10 and 20.
        out_subs = []
        for i in range(1, 676):
            if i == 10 or i == 20:
                text = ""
            else:
                text = f"Detta är ett svenskt test {i}"
            out_subs.append(srt.Subtitle(index=i, start=timedelta(seconds=i), end=timedelta(seconds=i+1), content=text))
        return out_subs

    async def fake_escalate(text, *args, **kwargs):
        call_count["escalate"] += 1
        return "Recovered text"
        
    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting):
        with patch("app.services.pipeline.create_job", return_value=1), \
             patch("app.services.pipeline.update_job"), \
             patch("app.services.pipeline.append_job_log"):
             
             with patch.object(pipeline, "trigger_bazarr_search"), \
                  patch.object(pipeline.translator, "translate_srt_content", side_effect=fake_translate), \
                  patch.object(pipeline.translator, "escalate_single_line", side_effect=fake_escalate):
                  
                  res = await pipeline._run_pipeline_logic(1, str(video_path), wait_seconds=0)
                  
                  # Should be translated because the 2 dropped lines are escalated
                  assert res["status"] == "translated"
                  
                  # We should have called translate once, and escalate exactly twice
                  assert call_count["translate"] == 1
                  assert call_count["escalate"] == 2
                  
                  # Check output file (should be .sv.srt since target is sv)
                  out_srt = str(video_path).replace(".mkv", ".sv.srt")
                  assert os.path.exists(out_srt)
                  
                  with open(out_srt, "r", encoding="utf-8") as f:
                      parsed = list(srt.parse(f.read()))
                  
                  assert len(parsed) == 675
                  # Check that 10 and 20 were recovered (0-indexed in array)
                  assert parsed[9].content == "Recovered text"
                  assert parsed[19].content == "Recovered text"

                  # Verify that all other 673 cues were NOT mutated and match first-pass exactly
                  for i, sub in enumerate(parsed, 1):
                      if i not in (10, 20):
                          assert sub.content == f"Detta är ett svenskt test {i}"

@pytest.mark.asyncio
async def test_675_worker_targeted_retry(tmp_path):
    from app.core.db import init_db, get_job_by_id, update_job
    init_db()
    pipeline = SubtitlePipeline()
    video_path = tmp_path / "video2.mkv"
    video_path.touch()
    
    source_subs = [srt.Subtitle(index=i, start=timedelta(seconds=i), end=timedelta(seconds=i+1), content=f"Source {i}") for i in range(1, 676)]
    source_srt_path = str(video_path).replace(".mkv", ".en.srt")
    with open(source_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(source_subs))
    
    def fake_get_setting(key, default=None):
        if key == "languages":
            return '[{"name": "Swedish", "code": "sv", "enabled": true}]'
        if key == "auto_repair_unhealthy": return "false"
        return default

    translate_calls = []
    run_count = {"calls": 0}

    async def fake_translate_batch(payload, **kwargs):
        run_count["calls"] += 1
        ids = [p["id"] for p in payload]
        translate_calls.append(ids)
        results = []
        for p in payload:
            if run_count["calls"] <= 15 and p["id"] in (10, 20):
                pass
            else:
                results.append({"id": p["id"], "text": f"Svenska {p['id']}"})
        return results
    
    with patch("google.genai.Client"):
        with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting):
            with patch.object(pipeline, "trigger_bazarr_search"), \
                 patch.object(pipeline.translator, "translate_batch", side_effect=fake_translate_batch), \
                 patch.object(pipeline.translator, "classify_and_recover_identical", return_value=[]), \
                 patch.object(pipeline.translator, "escalate_single_line", return_value=""):
                 
                from app.core.db import create_job
                job_id = create_job(str(video_path))
                
                res = await pipeline._run_pipeline_logic(job_id, str(video_path), wait_seconds=0)
                assert res["status"] in ["recovering", "failed"]
                
                # Verify exactly 14 full-pass batches + 0 internal + 1 targeted Smart Recovery batch = 15 (stagnation guard prevents 2 redundant loops)
                assert len(translate_calls) == 15
                
                full_pass_ids = []
                recovery_ids = []
                for call_ids in translate_calls:
                    if call_ids == [10, 20]:
                        recovery_ids.append(call_ids)
                    else:
                        full_pass_ids.extend(call_ids)
                        
                assert len(recovery_ids) == 1, "Expected 0 internal Smart Recovery + 1 pipeline Targeted Recovery batch for [10, 20] (stagnation guard breaks early)"
                assert len(full_pass_ids) == 675
                assert sorted(full_pass_ids) == list(range(0, 675))
                
                translate_calls.clear()
                
                # --- AFTER FIRST RUN PROOF ---
                from app.core.db import DB_PATH
                import os, json, hashlib
                data_dir = os.path.dirname(DB_PATH)
                lang_code = "sv"
                partial_file = os.path.join(data_dir, f"job_{job_id}_{lang_code}_partial.json")
                assert os.path.exists(partial_file), "Partial file does not exist after FIRST RUN!"
                
                with open(partial_file, "r", encoding="utf-8") as f:
                    partial_data = json.load(f)
                    
                expected_fingerprint = hashlib.md5("".join(s.content for s in source_subs).encode("utf-8")).hexdigest() + "_" + lang_code
                assert partial_data["fingerprint"] == expected_fingerprint, "Fingerprint mismatch!"
                
                lines_data = partial_data["lines"]
                assert len(lines_data) == 673, f"Expected 673 lines in partial_dict, got {len(lines_data)}"
                
                missing_in_lines = [i for i in range(0, 675) if str(i) not in lines_data]
                assert missing_in_lines == [10, 20], f"Expected missing IDs [10, 20], got {missing_in_lines}"
                
                # --- BEFORE SECOND RUN PROOF ---
                assert job_id == job_id
                assert os.path.exists(partial_file), "Partial file was deleted before SECOND RUN!"
                
                update_job(job_id, status="RECOVERING")
                res2 = await pipeline._run_pipeline_logic(job_id, str(video_path), wait_seconds=0)
                assert res2["status"] == "translated"
                
                # Verify it ONLY sent the 2 missing lines to the provider
                assert len(translate_calls) == 1
                assert translate_calls[0] == [10, 20]
