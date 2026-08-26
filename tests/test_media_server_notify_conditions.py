import pytest
import os
import srt
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from app.services.pipeline import SubtitlePipeline
from app.services.source_resolver import SubtitleSource, SourceOrigin
from app.core.db import get_setting, set_setting
from app.main import app

def make_valid_srt(lang="en", count=10):
    lines = []
    for i in range(1, count + 1):
        if lang == "sv":
            text = f"Detta är en mycket bra svensk undertext rad {i} och vi gillar film"
        else:
            text = f"Hello world this is a very good english subtitle line {i} for testing"
        lines.append(f"{i}\n00:00:0{i:02d},000 --> 00:00:0{i:02d},500\n{text}\n")
    return "\n".join(lines)


@pytest.fixture
def pipeline_settings(monkeypatch):
    settings = {
        "ai_provider": "gemini",
        "gemini_api_key": "dummy_api_key",
        "gemini_model": "gemini-3.5-flash-lite",
        "batch_size": "50",
        "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
        "enable_bazarr_check": "false",
        "clean_sdh": "true",
        "extract_source_embedded": "false",
        "extract_target_embedded": "false",
        "auto_repair_unhealthy": "true",
        "wait_time_seconds": "0",
        "notify_jellyfin": "true",
        "notify_plex": "true",
    }
    def mock_get_setting(key, default=""):
        return settings.get(key, default)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)
    monkeypatch.setattr("app.services.translator.get_setting", mock_get_setting)
    return settings


# Test A — successful publish
@pytest.mark.asyncio
async def test_successful_publish_notifies_media_servers(pipeline_settings, tmp_path, monkeypatch):
    """When subtitle is successfully published, _notify_media_servers() is called."""
    video_path = tmp_path / "successful_publish.mkv"
    video_path.touch()
    en_srt = tmp_path / "successful_publish.en.srt"
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(make_valid_srt(lang="en", count=10))

    pipeline = SubtitlePipeline()

    async def mock_translate(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        return [
            srt.Subtitle(index=i + 1, start=sub.start, end=sub.end, content=f"Hej världen {i + 1} detta är en svensk text")
            for i, sub in enumerate(subs)
        ]

    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)

    notify_mock = AsyncMock()
    monkeypatch.setattr(pipeline, "_notify_media_servers", notify_mock)

    res = await pipeline.process_video_file(str(video_path), event_source="SONARR")

    assert res["status"] == "translated"
    assert len(res["output_files"]) == 1
    assert notify_mock.call_count == 1
    assert notify_mock.call_args[0][0] == str(video_path)


# Test B — QA FAILED / no publish
@pytest.mark.asyncio
async def test_qa_failed_no_publish_does_not_notify(pipeline_settings, tmp_path, monkeypatch):
    """When a job fails QA and no subtitle is published, _notify_media_servers() is NOT called."""
    video_path = tmp_path / "qa_failed.mkv"
    video_path.touch()
    en_srt = tmp_path / "qa_failed.en.srt"
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(make_valid_srt(lang="en", count=10))

    pipeline = SubtitlePipeline()

    # Translator returns empty / dropped text causing QA fail
    async def mock_translate_dropped(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        return [
            srt.Subtitle(index=i + 1, start=sub.start, end=sub.end, content="")
            for i, sub in enumerate(subs)
        ]

    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate_dropped)

    notify_mock = AsyncMock()
    monkeypatch.setattr(pipeline, "_notify_media_servers", notify_mock)

    # Force QA recovery attempts to be exhausted by setting high retry count
    from app.core.db import create_job, update_job
    job_id = create_job(str(video_path), "SONARR")
    update_job(job_id, retry_count=5)

    res = await pipeline._run_pipeline_logic(
        job_id=job_id,
        video_path=str(video_path),
        event_source="SONARR"
    )

    assert res["status"] == "failed"
    assert len(res.get("output_files", [])) == 0
    # Must NOT notify
    assert notify_mock.call_count == 0


# Test C — recovering / no publish
@pytest.mark.asyncio
async def test_recovering_no_publish_does_not_notify(pipeline_settings, tmp_path, monkeypatch):
    """When a job enters RECOVERING state without publishing, _notify_media_servers() is NOT called."""
    video_path = tmp_path / "recovering_job.mkv"
    video_path.touch()
    en_srt = tmp_path / "recovering_job.en.srt"
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(make_valid_srt(lang="en", count=10))

    pipeline = SubtitlePipeline()

    # QA fail with low retry_count leads to RECOVERING
    async def mock_translate_bad(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        return [
            srt.Subtitle(index=i + 1, start=sub.start, end=sub.end, content="")
            for i, sub in enumerate(subs)
        ]

    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate_bad)

    notify_mock = AsyncMock()
    monkeypatch.setattr(pipeline, "_notify_media_servers", notify_mock)

    from app.core.db import create_job, update_job
    job_id = create_job(str(video_path), "SONARR")
    update_job(job_id, retry_count=0)

    res = await pipeline._run_pipeline_logic(
        job_id=job_id,
        video_path=str(video_path),
        event_source="SONARR"
    )

    assert res["status"] in ["recovering", "failed"]
    assert len(res.get("output_files", [])) == 0
    assert notify_mock.call_count == 0


# Test D — delete endpoint notifies
def test_delete_subtitles_notifies_media_servers(tmp_path, monkeypatch):
    """Deleting target subtitles notifies Jellyfin and Plex."""
    client = TestClient(app)
    video_path = tmp_path / "delete_target.mkv"
    video_path.touch()
    sv_srt = tmp_path / "delete_target.sv.srt"
    with open(sv_srt, "w", encoding="utf-8") as f:
        f.write(make_valid_srt(lang="sv", count=5))

    set_setting("media_movies_path", str(tmp_path))
    set_setting("languages", '[{"code": "sv", "name": "Swedish", "enabled": true}]')
    set_setting("notify_jellyfin", "true")
    set_setting("notify_plex", "true")

    with patch("app.api.dashboard.notify_jellyfin_library_refresh", new_callable=AsyncMock) as mock_jf, \
         patch("app.api.dashboard.notify_plex_library_refresh", new_callable=AsyncMock) as mock_plex:

        resp = client.request("DELETE", "/api/subtitles", json={"video_path": str(video_path)})
        assert resp.status_code == 200
        assert "delete_target.sv.srt" in resp.json()["deleted_files"]
        mock_jf.assert_called_once()
        mock_plex.assert_called_once_with(str(video_path))


# Test E — source-as-target published notifies
@pytest.mark.asyncio
async def test_source_as_target_published_notifies(pipeline_settings, tmp_path, monkeypatch):
    """When source language matches target language and is published directly, _notify_media_servers is called."""
    video_path = tmp_path / "source_as_target.mkv"
    video_path.touch()
    
    # Create an extracted source file that resolved to Swedish
    src_srt = tmp_path / "source_as_target.temp_src.srt"
    with open(src_srt, "w", encoding="utf-8") as f:
        f.write(make_valid_srt(lang="sv", count=10))

    pipeline = SubtitlePipeline()

    # Mock source resolution returning resolved Swedish source
    content_sv = make_valid_srt(lang="sv", count=10)
    resolved_source = SubtitleSource(
        language="sv",
        path=str(src_srt),
        origin=SourceOrigin.EMBEDDED,
        content=content_sv,
        cues=list(srt.parse(content_sv))
    )

    with patch("app.services.pipeline.SourceResolver.resolve", new_callable=AsyncMock, return_value=resolved_source):
        notify_mock = AsyncMock()
        monkeypatch.setattr(pipeline, "_notify_media_servers", notify_mock)

        res = await pipeline.process_video_file(str(video_path), event_source="SONARR")

        assert res["status"] == "skipped"
        # Target file movie.sv.srt was published from source
        target_sv = tmp_path / "source_as_target.sv.srt"
        assert target_sv.exists()
        assert notify_mock.call_count == 1


# Test F — already exists (no files published) does NOT notify
@pytest.mark.asyncio
async def test_already_exists_no_publish_does_not_notify(pipeline_settings, tmp_path, monkeypatch):
    """When target subtitle already exists on disk and no new file is published, _notify_media_servers is NOT called."""
    video_path = tmp_path / "existing_movie.mkv"
    video_path.touch()
    sv_srt = tmp_path / "existing_movie.sv.srt"
    with open(sv_srt, "w", encoding="utf-8") as f:
        f.write(make_valid_srt(lang="sv", count=10))

    pipeline = SubtitlePipeline()

    notify_mock = AsyncMock()
    monkeypatch.setattr(pipeline, "_notify_media_servers", notify_mock)

    res = await pipeline.process_video_file(str(video_path), event_source="SONARR")

    assert res["status"] == "skipped"
    assert notify_mock.call_count == 0
