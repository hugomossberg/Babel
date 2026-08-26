import os
import pytest
import srt
from datetime import timedelta
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.pipeline import SubtitlePipeline, _publish_subtitle_atomic
from app.core.cleaner import subs_to_srt_string
from app.core import db
from app.services.source_resolver import SubtitleSource, SourceOrigin


HEALTHY_SWEDISH = [
    "Detta är en svensk mening.",
    "Här kommer den andra meningen.",
    "Och den tredje meningen också.",
    "Fjärde meningen är här nu.",
    "Femte meningen är helt komplett.",
    "Sjätte meningen ser mycket bra ut.",
    "Sjunde meningen för att säkra texten.",
    "Åttonde meningen gör allt godkänt."
]


def make_subs(texts):
    return [
        srt.Subtitle(index=i+1, start=timedelta(seconds=i*2), end=timedelta(seconds=i*2+1), content=t)
        for i, t in enumerate(texts)
    ]


@pytest.fixture(autouse=True)
def setup_tmp_db(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test_pipe_pub.db")
    monkeypatch.setattr("app.core.db.DB_PATH", test_db)
    db.init_db()
    yield test_db


@pytest.mark.asyncio
async def test_embedded_target_publishes_via_atomic_publisher(tmp_path):
    """Verify embedded target extraction uses _publish_subtitle_atomic and publishes cleanly when GREEN."""
    pipeline = SubtitlePipeline()
    video = tmp_path / "Movie.mkv"
    video.touch()
    target_srt = tmp_path / "Movie.sv.srt"

    valid_content = subs_to_srt_string(make_subs(HEALTHY_SWEDISH))

    def fake_get_setting(key, default):
        if key == "languages": return '[{"name": "Swedish", "code": "sv", "enabled": true}]'
        if key == "extract_target_embedded": return "true"
        if key == "enable_bazarr_check": return "false"
        if key == "extract_source_embedded": return "false"
        return default

    def fake_extract(vid, out, preferred_lang, **kwargs):
        with open(out, "w", encoding="utf-8") as f:
            f.write(valid_content)
        return True

    def selective_health(path, target_lang_code=None):
        if os.path.exists(path):
            return {"status": "GREEN", "reason": "Good"}
        return {"status": "RED", "reason": "Missing"}

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting), \
         patch("app.services.pipeline.extract_embedded_srt", side_effect=fake_extract), \
         patch("app.services.pipeline.evaluate_subtitle_health", side_effect=selective_health), \
         patch("app.services.pipeline.create_job", return_value=1), \
         patch("app.services.pipeline.update_job"), \
         patch("app.services.pipeline.append_job_log"):

        res = await pipeline._run_pipeline_logic(1, str(video), wait_seconds=0)

        # File must exist on disk and be parseable
        assert os.path.exists(str(target_srt))
        with open(str(target_srt), "r", encoding="utf-8") as f:
            parsed = list(srt.parse(f.read()))
            assert len(parsed) == len(HEALTHY_SWEDISH)
            assert parsed[0].content == HEALTHY_SWEDISH[0]


@pytest.mark.asyncio
async def test_embedded_target_preserves_concurrent_healthy_file(tmp_path):
    """Race test: verify that if an external healthy file appears while embedded extraction runs, it is preserved without overwrite."""
    pipeline = SubtitlePipeline()
    video = tmp_path / "Series.mkv"
    video.touch()
    target_srt = tmp_path / "Series.sv.srt"

    # Pre-existing external healthy file
    with open(str(target_srt), "w", encoding="utf-8") as f:
        f.write(subs_to_srt_string(make_subs(HEALTHY_SWEDISH)))

    def fake_get_setting(key, default):
        if key == "languages": return '[{"name": "Swedish", "code": "sv", "enabled": true}]'
        if key == "extract_target_embedded": return "true"
        if key == "enable_bazarr_check": return "false"
        if key == "extract_source_embedded": return "false"
        return default

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting), \
         patch("app.services.pipeline.create_job", return_value=1), \
         patch("app.services.pipeline.update_job"), \
         patch("app.services.pipeline.append_job_log"):

        res = await pipeline._run_pipeline_logic(1, str(video), wait_seconds=0)
        assert res["status"] in ("skipped", "already_exists")

        # Original external content was untouched
        with open(str(target_srt), "r", encoding="utf-8") as f:
            parsed = list(srt.parse(f.read()))
            assert parsed[0].content == HEALTHY_SWEDISH[0]


@pytest.mark.asyncio
async def test_source_equals_target_green_publishes_and_skips_ai(tmp_path):
    """Obligatory test: source language == target language AND health == GREEN -> canonical atomic publisher used, AI skipped."""
    pipeline = SubtitlePipeline()
    video = tmp_path / "Swedish_Movie.mkv"
    video.touch()
    target_srt = tmp_path / "Swedish_Movie.sv.srt"

    extract_dir = tmp_path / "extracted_sources"
    extract_dir.mkdir()
    source_srt = extract_dir / "source.extracted.sv.srt"
    valid_content = subs_to_srt_string(make_subs(HEALTHY_SWEDISH))
    with open(str(source_srt), "w", encoding="utf-8") as f:
        f.write(valid_content)

    def fake_get_setting(key, default):
        if key == "languages": return '[{"name": "Swedish", "code": "sv", "enabled": true}]'
        if key == "enable_bazarr_check": return "false"
        if key == "extract_target_embedded": return "false"
        if key == "extract_source_embedded": return "false"
        return default

    mock_source = SubtitleSource(
        path=str(source_srt),
        language="sv",
        origin=SourceOrigin.EMBEDDED,
        cues=make_subs(HEALTHY_SWEDISH),
        content=valid_content,
    )

    mock_translate = AsyncMock()

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting), \
         patch("app.services.pipeline.evaluate_subtitle_health", return_value={"status": "GREEN"}), \
         patch.object(pipeline.translator, "translate_srt_content", mock_translate), \
         patch.object(pipeline, "trigger_bazarr_search"), \
         patch("app.services.source_resolver.SourceResolver.resolve", return_value=mock_source), \
         patch("app.services.pipeline.create_job", return_value=1), \
         patch("app.services.pipeline.update_job"), \
         patch("app.services.pipeline.append_job_log"):

        res = await pipeline._run_pipeline_logic(1, str(video), wait_seconds=0)

        # Published via atomic publisher
        assert os.path.exists(str(target_srt))
        with open(str(target_srt), "r", encoding="utf-8") as f:
            parsed = list(srt.parse(f.read()))
            assert len(parsed) == len(HEALTHY_SWEDISH)

        # AI translation was never called
        mock_translate.assert_not_called()


@pytest.mark.asyncio
async def test_source_equals_target_yellow_does_not_publish_shortcut(tmp_path):
    """Obligatory test: source language == target language AND health == YELLOW -> canonical target NOT published via shortcut."""
    pipeline = SubtitlePipeline()
    video = tmp_path / "Swedish_Movie_Yellow.mkv"
    video.touch()
    target_srt = tmp_path / "Swedish_Movie_Yellow.sv.srt"

    extract_dir = tmp_path / "extracted_sources"
    extract_dir.mkdir()
    source_srt = extract_dir / "source.yellow.sv.srt"
    valid_content = subs_to_srt_string(make_subs(HEALTHY_SWEDISH))
    with open(str(source_srt), "w", encoding="utf-8") as f:
        f.write(valid_content)

    def fake_get_setting(key, default):
        if key == "languages": return '[{"name": "Swedish", "code": "sv", "enabled": true}]'
        if key == "enable_bazarr_check": return "false"
        if key == "extract_target_embedded": return "false"
        if key == "extract_source_embedded": return "false"
        return default

    mock_source = SubtitleSource(
        path=str(source_srt),
        language="sv",
        origin=SourceOrigin.EMBEDDED,
        cues=make_subs(HEALTHY_SWEDISH),
        content=valid_content,
    )

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting), \
         patch("app.services.pipeline.evaluate_subtitle_health", return_value={"status": "YELLOW", "reason": "Sync skew"}), \
         patch.object(pipeline, "trigger_bazarr_search"), \
         patch("app.services.source_resolver.SourceResolver.resolve", return_value=mock_source), \
         patch("app.services.pipeline.create_job", return_value=1), \
         patch("app.services.pipeline.update_job"), \
         patch("app.services.pipeline.append_job_log"):

        res = await pipeline._run_pipeline_logic(1, str(video), wait_seconds=0)

        # File was NOT published by the shortcut
        assert not os.path.exists(str(target_srt))


@pytest.mark.asyncio
async def test_source_equals_target_red_does_not_publish_shortcut(tmp_path):
    """Obligatory test: source language == target language AND health == RED -> canonical target NOT published via shortcut."""
    pipeline = SubtitlePipeline()
    video = tmp_path / "Swedish_Movie_Red.mkv"
    video.touch()
    target_srt = tmp_path / "Swedish_Movie_Red.sv.srt"

    extract_dir = tmp_path / "extracted_sources"
    extract_dir.mkdir()
    source_srt = extract_dir / "source.red.sv.srt"
    valid_content = subs_to_srt_string(make_subs(HEALTHY_SWEDISH))
    with open(str(source_srt), "w", encoding="utf-8") as f:
        f.write(valid_content)

    def fake_get_setting(key, default):
        if key == "languages": return '[{"name": "Swedish", "code": "sv", "enabled": true}]'
        if key == "enable_bazarr_check": return "false"
        if key == "extract_target_embedded": return "false"
        if key == "extract_source_embedded": return "false"
        return default

    mock_source = SubtitleSource(
        path=str(source_srt),
        language="sv",
        origin=SourceOrigin.EMBEDDED,
        cues=make_subs(HEALTHY_SWEDISH),
        content=valid_content,
    )

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting), \
         patch("app.services.pipeline.evaluate_subtitle_health", return_value={"status": "RED", "reason": "Corrupt timestamps"}), \
         patch.object(pipeline, "trigger_bazarr_search"), \
         patch("app.services.source_resolver.SourceResolver.resolve", return_value=mock_source), \
         patch("app.services.pipeline.create_job", return_value=1), \
         patch("app.services.pipeline.update_job"), \
         patch("app.services.pipeline.append_job_log"):

        res = await pipeline._run_pipeline_logic(1, str(video), wait_seconds=0)

        # File was NOT published by the shortcut
        assert not os.path.exists(str(target_srt))
