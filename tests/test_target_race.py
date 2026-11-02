import pytest
import os
import srt
from datetime import timedelta
import asyncio
from unittest.mock import patch, MagicMock

from app.services.pipeline import SubtitlePipeline
from app.core.db import get_job_by_id
from app.core import db

@pytest.fixture(autouse=True)
def mock_db_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path / 'race.db'))
    db.init_db()
    settings = {
        "ai_provider": "gemini",
        "gemini_api_key": "dummy_api_key",
        "gemini_model": "gemini-3.5-flash-lite",
        "batch_size": "50",
        "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
        "enable_bazarr_check": "false",
        "extract_source_embedded": "false",
        "extract_target_embedded": "false",
    }
    for key, value in settings.items():
        db.set_setting(key, value)

@pytest.mark.asyncio
async def test_a_target_appears_after_extraction(mock_db_settings, tmp_path, monkeypatch):
    video_path = tmp_path / "test.mkv"
    video_path.touch()
    en_srt = tmp_path / "test.en.srt"
    with open(en_srt, "w") as f:
        lines = []
        for i in range(1, 10):
            lines.append(f"{i}\n00:00:0{i},000 --> 00:00:0{i},500\nThis is a very good English text line {i}\n")
        f.write("\n".join(lines))

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
    assert job["status"] != "BAZARR MATCH"
    assert job["status"] != "TRANSLATED"
    assert job["status"] == "ALREADY EXISTS"
    assert job["reason"] == "External target appeared during processing"

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
async def test_c_qa_fail_no_target_created(mock_db_settings, empty_gemini_provider, tmp_path, monkeypatch):
    video_path = tmp_path / "test.mkv"
    video_path.touch()
    en_srt = tmp_path / "test.en.srt"
    # Valid source timing ensures this fails for unusable provider output,
    # rather than malformed or repeating fixture timestamps.
    en_srt.write_text(srt.compose([
        srt.Subtitle(i + 1, timedelta(seconds=i * 3), timedelta(seconds=i * 3 + 2),
                     f'We should discuss the missing map before the journey {i}.')
        for i in range(100)
    ]))

    pipeline = SubtitlePipeline()

    def mock_find(vp, lang):
        if lang == "en": return str(en_srt)
        return None

    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find):
        res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")

    assert res["status"] == 'failed'
    job = get_job_by_id(res['job_id'])
    assert job['status'] == 'FAILED'
    assert job['primary_provider'] == 'gemini'
    assert job['next_retry_at'] is None
    assert 'QA Gate FAILED' in '\n'.join(job['logs'])
    assert any('translations' in call['properties'] for call in empty_gemini_provider['calls'])
    assert any('results' in call['properties'] for call in empty_gemini_provider['calls'])

    sv_srt_path = tmp_path / "test.sv.srt"
    assert not sv_srt_path.exists()

@pytest.mark.asyncio
async def test_d_final_no_clobber_race(mock_db_settings, tmp_path, monkeypatch):
    video_path = tmp_path / "test.mkv"
    video_path.touch()
    en_srt = tmp_path / "test.en.srt"
    with open(en_srt, "w") as f:
        lines = []
        for i in range(1, 10):
            lines.append(f"{i}\n00:00:0{i},000 --> 00:00:0{i},500\nThis is a very good English text line {i}\n")
        f.write("\n".join(lines))

    pipeline = SubtitlePipeline()

    translate_calls = 0
    async def mock_translate_batch(*args, **kwargs):
        nonlocal translate_calls
        translate_calls += 1
        return [{"id": i, "text": f"Svensk rad {i+1}"} for i in range(9)]
    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)

    sv_srt_path = str(tmp_path / "test.sv.srt")
    def mock_find(vp, lang):
        if lang == "en": return str(en_srt)
        return None

    # We mock os.link to raise FileExistsError, simulating Bazarr creating the file EXACTLY when we publish
    original_link = os.link

    def mock_link(src, dst):
        if dst == sv_srt_path:
            # Create the file just before we fail, simulating race condition
            with open(sv_srt_path, "w") as f:
                lines = []
                for i in range(1, 10):
                    lines.append(f"{i}\n00:00:0{i},000 --> 00:00:0{i},500\nDetta är en mycket bra svensk text rad {i}\n")
                f.write("\n".join(lines))
            raise FileExistsError(f"File exists: {dst}")
        return original_link(src, dst)

    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find), \
         patch("os.link", side_effect=mock_link):
        res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")

    assert res["status"] != "failed"
    assert translate_calls == 1  # AI translation happened

    # Check that the external target was NOT overwritten
    with open(sv_srt_path, "r") as f:
        content = f.read()
    assert "Detta är en mycket bra svensk text" in content

    job = get_job_by_id(res["job_id"])
    logs = "".join(job["logs"])
    assert "Preserving verified external target" in logs or "Subtitle Trust Engine: PASS" in logs
