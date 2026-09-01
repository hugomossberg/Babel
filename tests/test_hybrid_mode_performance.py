import pytest
import os
import srt
import time
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from app.services.pipeline import SubtitlePipeline
from app.services.source_resolver import BazarrResult, BazarrResultCode
from app.core.db import get_job_by_id

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
def hybrid_db_settings(monkeypatch):
    def mock_get_setting(key, default=""):
        settings = {
            "ai_provider": "gemini",
            "gemini_api_key": "dummy_api_key",
            "gemini_model": "gemini-3.5-flash-lite",
            "batch_size": "50",
            "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
            "enable_bazarr_check": "true",
            "clean_sdh": "true",
            "extract_source_embedded": "true",
            "extract_target_embedded": "true",
            "auto_repair_unhealthy": "true",
            "wait_time_seconds": "15",
            "notify_jellyfin": "false"
        }
        return settings.get(key, default)
    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)
    monkeypatch.setattr("app.services.translator.get_setting", mock_get_setting)


@pytest.mark.asyncio
async def test_1_hybrid_target_already_exists_ai_never_called(hybrid_db_settings, tmp_path, monkeypatch):
    """1. Hybrid + target already exists -> AI never called, job marked ALREADY EXISTS."""
    video_path = tmp_path / "movie.mkv"
    video_path.touch()
    en_srt = tmp_path / "movie.en.srt"
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(make_valid_srt(lang="en", count=10))
    sv_srt = tmp_path / "movie.sv.srt"
    with open(sv_srt, "w", encoding="utf-8") as f:
        f.write(make_valid_srt(lang="sv", count=10))

    pipeline = SubtitlePipeline()
    translate_mock = AsyncMock()
    monkeypatch.setattr(pipeline.translator, "translate_srt_content", translate_mock)

    res = await pipeline.process_video_file(str(video_path), event_source="SONARR")

    assert res["status"] == "skipped"
    assert res["reason"] == "already_exists"
    assert translate_mock.call_count == 0
    job = get_job_by_id(res["job_id"])
    assert job["status"] == "ALREADY EXISTS"


@pytest.mark.asyncio
async def test_2_hybrid_target_appears_while_preparation_occurs(hybrid_db_settings, tmp_path, monkeypatch):
    """2 & 3. Hybrid + target appears while preparation occurs -> final check catches it -> AI never called."""
    video_path = tmp_path / "show.mkv"
    video_path.touch()
    en_srt = tmp_path / "show.en.srt"
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(make_valid_srt(lang="en", count=10))

    pipeline = SubtitlePipeline()
    translate_mock = AsyncMock()
    monkeypatch.setattr(pipeline.translator, "translate_srt_content", translate_mock)
    bazarr_trigger_mock = AsyncMock(return_value=BazarrResult(code=BazarrResultCode.TRIGGERED, language="sv", detail="Accepted"))
    monkeypatch.setattr(pipeline, "trigger_bazarr_search", bazarr_trigger_mock)

    sv_srt_path = str(tmp_path / "show.sv.srt")
    call_counts = {"sv": 0, "en": 0}

    def mock_find(vp, lang):
        if lang == "sv":
            call_counts["sv"] += 1
            if call_counts["sv"] == 1:
                # Initial check: no target exists yet
                return None
            else:
                # During prep, Bazarr writes the target file!
                with open(sv_srt_path, "w", encoding="utf-8") as f:
                    f.write(make_valid_srt(lang="sv", count=10))
                return sv_srt_path
        if lang == "en":
            call_counts["en"] += 1
            return str(en_srt)
        return None

    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find):
        res = await pipeline.process_video_file(str(video_path), event_source="SONARR")

    assert res["status"] == "skipped"
    assert res["reason"] == "bazarr_downloaded"
    assert translate_mock.call_count == 0
    bazarr_trigger_mock.assert_called_once_with(str(video_path), language="sv")
    job = get_job_by_id(res["job_id"])
    assert job["status"] == "BAZARR MATCH"


@pytest.mark.asyncio
async def test_3_hybrid_bazarr_miss_starts_ai_immediately_no_grace_sleep(hybrid_db_settings, tmp_path, monkeypatch):
    """4 & 5. Hybrid + Bazarr miss -> AI starts after preparation, NO fixed grace sleep."""
    video_path = tmp_path / "episode.mkv"
    video_path.touch()
    en_srt = tmp_path / "episode.en.srt"
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(make_valid_srt(lang="en", count=10))

    pipeline = SubtitlePipeline()
    translate_called = False

    async def mock_translate(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        nonlocal translate_called
        translate_called = True
        return [
            srt.Subtitle(index=i+1, start=sub.start, end=sub.end, content=f"Hej världen {i+1} detta är en svensk text")
            for i, sub in enumerate(subs)
        ]

    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)
    bazarr_trigger_mock = AsyncMock(return_value=BazarrResult(code=BazarrResultCode.MEDIA_NOT_FOUND, language="sv", detail="Not found"))
    monkeypatch.setattr(pipeline, "trigger_bazarr_search", bazarr_trigger_mock)

    # Track any asyncio.sleep calls.
    # Goal: verify that the OLD 15-second grace sleep is gone.
    # The concurrent target-presence poller uses a 2s sleep — that is intentional
    # and allowed. We only assert that no *legacy* grace sleeps >= 10s occurred.
    sleep_calls = []
    real_sleep = asyncio.sleep

    async def tracking_sleep(delay, *args, **kwargs):
        sleep_calls.append(delay)
        return await real_sleep(0)  # non-blocking for test

    monkeypatch.setattr("asyncio.sleep", tracking_sleep)

    res = await pipeline.process_video_file(str(video_path), event_source="SONARR")

    assert res["status"] == "translated"
    assert translate_called is True
    bazarr_trigger_mock.assert_called_once_with(str(video_path), language="sv")
    # Assert no 15-second (or other long) legacy grace-period sleep was called.
    # The 2s concurrent poller sleep is allowed — it is not a grace sleep.
    assert not any(d >= 10 for d in sleep_calls), (
        f"Legacy grace sleep detected: {sleep_calls}. "
        "No blocking grace period should exist in the hybrid pipeline."
    )
    job = get_job_by_id(res["job_id"])
    assert job["status"] == "TRANSLATED"
    logs = "".join(job["logs"])
    # v2.3.43: "Hybrid preparation completed" replaced by SourceResolver log messages
    assert "Source Resolver" in logs
    assert "Source selected" in logs or "source selected" in logs.lower()
    # No fixed grace sleep — confirmed by sleep_calls check above



@pytest.mark.asyncio
async def test_4_legacy_wait_time_seconds_in_db_ignored(hybrid_db_settings, tmp_path, monkeypatch):
    """14. Legacy wait_time_seconds value in DB (e.g. 999) must NOT reintroduce any delay."""
    video_path = tmp_path / "legacy_test.mkv"
    video_path.touch()
    en_srt = tmp_path / "legacy_test.en.srt"
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(make_valid_srt(lang="en", count=10))

    def mock_get_setting_with_999(key, default=""):
        if key == "wait_time_seconds":
            return "999"
        if key == "enable_bazarr_check":
            return "true"
        if key == "languages":
            return '[{"code": "sv", "name": "Swedish", "enabled": true}]'
        return default

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting_with_999)
    monkeypatch.setattr("app.services.translator.get_setting", mock_get_setting_with_999)

    pipeline = SubtitlePipeline()
    translate_called = False

    async def mock_translate(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        nonlocal translate_called
        translate_called = True
        return [
            srt.Subtitle(index=i+1, start=sub.start, end=sub.end, content=f"Hej världen {i+1} detta är en svensk text")
            for i, sub in enumerate(subs)
        ]

    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)
    monkeypatch.setattr(pipeline, "trigger_bazarr_search", AsyncMock(return_value=BazarrResult(code=BazarrResultCode.MEDIA_NOT_FOUND, language="sv", detail="Not found")))

    sleep_calls = []
    async def tracking_sleep(delay, *args, **kwargs):
        sleep_calls.append(delay)

    monkeypatch.setattr("asyncio.sleep", tracking_sleep)

    t0 = time.monotonic()
    res = await pipeline.process_video_file(str(video_path), wait_seconds=999, event_source="MANUAL")
    elapsed = time.monotonic() - t0

    assert res["status"] == "translated"
    assert translate_called is True
    assert 999 not in sleep_calls
    assert elapsed < 2.0  # Finished virtually instantaneously without 999s sleep!


@pytest.mark.asyncio
async def test_5_bazarr_unavailable_or_error_continues_to_ai(hybrid_db_settings, tmp_path, monkeypatch):
    """6. Bazarr unavailable/error -> AI preparation/fallback continues, no crash or delay."""
    video_path = tmp_path / "bazarr_down.mkv"
    video_path.touch()
    en_srt = tmp_path / "bazarr_down.en.srt"
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(make_valid_srt(lang="en", count=10))

    pipeline = SubtitlePipeline()
    translate_called = False

    async def mock_translate(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        nonlocal translate_called
        translate_called = True
        return [
            srt.Subtitle(index=i+1, start=sub.start, end=sub.end, content=f"Hej världen {i+1} detta är en svensk text")
            for i, sub in enumerate(subs)
        ]

    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)

    # Simulate Bazarr raising connection/HTTP error
    async def failing_bazarr_search(video_path, language="sv"):
        raise ConnectionRefusedError("Bazarr service unavailable on port 6767")

    monkeypatch.setattr(pipeline, "trigger_bazarr_search", failing_bazarr_search)

    res = await pipeline.process_video_file(str(video_path), event_source="SONARR")

    assert res["status"] == "translated"
    assert translate_called is True
    job = get_job_by_id(res["job_id"])
    assert job["status"] == "TRANSLATED"


@pytest.mark.asyncio
async def test_6_pure_ai_mode_skips_bazarr_completely(tmp_path, monkeypatch):
    """7. Pure AI Mode -> No Bazarr trigger or check, behavior completely fast and unchanged."""
    video_path = tmp_path / "pure_ai.mkv"
    video_path.touch()
    en_srt = tmp_path / "pure_ai.en.srt"
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(make_valid_srt(lang="en", count=10))

    def mock_pure_ai_settings(key, default=""):
        settings = {
            "ai_provider": "gemini",
            "gemini_api_key": "dummy_api_key",
            "gemini_model": "gemini-3.5-flash-lite",
            "batch_size": "50",
            "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
            "enable_bazarr_check": "false",  # PURE AI MODE
            "clean_sdh": "true",
            "extract_source_embedded": "true",
            "extract_target_embedded": "true",
            "auto_repair_unhealthy": "true",
            "notify_jellyfin": "false"
        }
        return settings.get(key, default)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_pure_ai_settings)
    monkeypatch.setattr("app.services.translator.get_setting", mock_pure_ai_settings)

    pipeline = SubtitlePipeline()
    translate_called = False

    async def mock_translate(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        nonlocal translate_called
        translate_called = True
        return [
            srt.Subtitle(index=i+1, start=sub.start, end=sub.end, content=f"Hej världen {i+1} detta är en svensk text")
            for i, sub in enumerate(subs)
        ]

    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)
    bazarr_trigger_mock = AsyncMock()
    monkeypatch.setattr(pipeline, "trigger_bazarr_search", bazarr_trigger_mock)

    res = await pipeline.process_video_file(str(video_path), event_source="SONARR")

    assert res["status"] == "translated"
    assert translate_called is True
    # In Pure AI mode, Bazarr search is never triggered
    assert bazarr_trigger_mock.call_count == 0


@pytest.mark.asyncio
async def test_7_preparation_failure_safe_status_ai_never_invoked(hybrid_db_settings, tmp_path, monkeypatch):
    """8. Preparation failure (no English source found) -> AI not invoked, safe status."""
    video_path = tmp_path / "no_source.mkv"
    video_path.touch()

    pipeline = SubtitlePipeline()
    translate_mock = AsyncMock()
    monkeypatch.setattr(pipeline.translator, "translate_srt_content", translate_mock)
    monkeypatch.setattr(pipeline, "trigger_bazarr_search", AsyncMock())

    # Mock extract_embedded_srt returning False and find_external_subtitle returning None
    monkeypatch.setattr("app.services.pipeline.extract_embedded_srt", lambda *args, **kwargs: False)
    monkeypatch.setattr("app.services.pipeline.find_external_subtitle", lambda *args, **kwargs: None)

    res = await pipeline.process_video_file(str(video_path), event_source="SONARR")

    assert res["status"] in ["waiting_source", "failed"]
    assert translate_mock.call_count == 0
    # Ensure no broken sv.srt was created
    assert not (tmp_path / "no_source.sv.srt").exists()


@pytest.mark.asyncio
async def test_8_human_subtitle_wins_cleans_temporary_extracted_artifacts(hybrid_db_settings, tmp_path, monkeypatch):
    """9. Human subtitle wins -> prepared temporary resources cleaned up."""
    video_path = tmp_path / "temp_clean.mkv"
    video_path.touch()

    pipeline = SubtitlePipeline()
    translate_mock = AsyncMock()
    monkeypatch.setattr(pipeline.translator, "translate_srt_content", translate_mock)
    monkeypatch.setattr(pipeline, "trigger_bazarr_search", AsyncMock(return_value=BazarrResult(code=BazarrResultCode.TRIGGERED, language="sv", detail="Accepted")))

    temp_extracted_file = tmp_path / "temp_clean.temp_extracted.en.srt"

    # Simulate embedded track extracted to temp file during preparation
    def mock_extract(vpath, outpath, preferred_lang="eng"):
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(make_valid_srt(lang="en", count=10))
        return True

    monkeypatch.setattr("app.services.pipeline.extract_embedded_srt", mock_extract)

    sv_target_file = str(tmp_path / "temp_clean.sv.srt")
    check_count = 0
    def mock_find(vpath, lang):
        nonlocal check_count
        if lang == "sv":
            check_count += 1
            if check_count == 1:
                return None
            else:
                # Found by Bazarr during final check
                with open(sv_target_file, "w", encoding="utf-8") as f:
                    f.write(make_valid_srt(lang="sv", count=10))
                return sv_target_file
        return None

    with patch("app.services.pipeline.find_external_subtitle", side_effect=mock_find):
        res = await pipeline.process_video_file(str(video_path), event_source="SONARR")

    assert res["status"] == "skipped"
    assert res["reason"] == "bazarr_downloaded"
    assert translate_mock.call_count == 0
    # Verify temp file was cleaned up
    assert not temp_extracted_file.exists()


@pytest.mark.asyncio
async def test_9_no_duplicate_jellyfin_notification(hybrid_db_settings, tmp_path, monkeypatch):
    """11. Verify single Jellyfin notification on completion."""
    video_path = tmp_path / "jellyfin_test.mkv"
    video_path.touch()
    en_srt = tmp_path / "jellyfin_test.en.srt"
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(make_valid_srt(lang="en", count=10))

    def mock_settings_jf(key, default=""):
        if key == "notify_jellyfin":
            return "true"
        if key == "enable_bazarr_check":
            return "true"
        if key == "languages":
            return '[{"code": "sv", "name": "Swedish", "enabled": true}]'
        return default

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_settings_jf)
    monkeypatch.setattr("app.services.translator.get_setting", mock_settings_jf)

    pipeline = SubtitlePipeline()
    monkeypatch.setattr(pipeline, "trigger_bazarr_search", AsyncMock())

    async def mock_translate(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        return [
            srt.Subtitle(index=i+1, start=sub.start, end=sub.end, content=f"Hej världen {i+1} detta är en svensk text")
            for i, sub in enumerate(subs)
        ]

    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)

    jf_notify_mock = AsyncMock()
    monkeypatch.setattr("app.services.pipeline.notify_jellyfin_library_refresh", jf_notify_mock)

    res = await pipeline.process_video_file(str(video_path), event_source="SONARR")

    assert res["status"] == "translated"
    # Exactly one refresh notification triggered
    assert jf_notify_mock.call_count == 1


@pytest.mark.asyncio
async def test_10_atomic_single_writer_no_clobber(hybrid_db_settings, tmp_path, monkeypatch):
    """10. Single-writer / Race safety: external subtitle appearing during publish is not overwritten."""
    video_path = tmp_path / "single_writer.mkv"
    video_path.touch()
    en_srt = tmp_path / "single_writer.en.srt"
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(make_valid_srt(lang="en", count=10))

    pipeline = SubtitlePipeline()
    translate_called = False

    async def mock_translate(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        nonlocal translate_called
        translate_called = True
        return [
            srt.Subtitle(index=i+1, start=sub.start, end=sub.end, content=f"AI översatt rad {i+1}")
            for i, sub in enumerate(subs)
        ]

    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)
    monkeypatch.setattr(pipeline, "trigger_bazarr_search", AsyncMock())

    sv_srt_path = str(tmp_path / "single_writer.sv.srt")
    original_link = os.link

    def mock_link(src, dst):
        if dst == sv_srt_path:
            # Simulate Bazarr atomic write precisely as publish occurs
            with open(sv_srt_path, "w", encoding="utf-8") as f:
                f.write(make_valid_srt(lang="sv", count=10))
            raise FileExistsError(f"File exists: {dst}")
        return original_link(src, dst)

    with patch("os.link", side_effect=mock_link):
        res = await pipeline.process_video_file(str(video_path), event_source="SONARR")

    assert res["status"] != "failed"
    assert translate_called is True

    # Check that external human sub is preserved (not overwritten with AI translation)
    with open(sv_srt_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Detta är en mycket bra svensk undertext" in content
    assert "AI översatt rad" not in content


@pytest.mark.asyncio
async def test_11_performance_benchmark_time_to_first_ai_request(hybrid_db_settings, tmp_path, monkeypatch):
    """15 & 16. Benchmark: measure time to first AI request under Hybrid miss vs legacy fixed delay."""
    video_path = tmp_path / "benchmark.mkv"
    video_path.touch()
    en_srt = tmp_path / "benchmark.en.srt"
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(make_valid_srt(lang="en", count=10))

    pipeline = SubtitlePipeline()
    time_at_ai_call = None
    start_time = None

    async def mock_translate(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        nonlocal time_at_ai_call
        time_at_ai_call = time.monotonic()
        return [
            srt.Subtitle(index=i+1, start=sub.start, end=sub.end, content=f"Hej {i+1} detta är en svensk text")
            for i, sub in enumerate(subs)
        ]

    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)
    monkeypatch.setattr(pipeline, "trigger_bazarr_search", AsyncMock(return_value=BazarrResult(code=BazarrResultCode.MEDIA_NOT_FOUND, language="sv", detail="Not found")))

    start_time = time.monotonic()
    res = await pipeline.process_video_file(str(video_path), event_source="SONARR")
    time_to_first_ai = time_at_ai_call - start_time

    assert res["status"] == "translated"
    assert time_to_first_ai is not None
    # In new architecture, time to first AI request is sub-second (no 15s grace wait!)
    assert time_to_first_ai < 1.0


@pytest.mark.asyncio
async def test_12_slow_bazarr_trigger_does_not_delay_source_prep_start(hybrid_db_settings, tmp_path, monkeypatch):
    """1. Slow Bazarr trigger (e.g. 2.0s HTTP response) does NOT delay source preparation start."""
    video_path = tmp_path / "slow_bazarr.mkv"
    video_path.touch()
    en_srt = tmp_path / "slow_bazarr.en.srt"
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(make_valid_srt(lang="en", count=10))

    pipeline = SubtitlePipeline()
    prep_start_timestamp = None
    job_start_timestamp = None
    ai_call_timestamp = None

    async def slow_bazarr_search(vpath, language="sv"):
        # Simulate 2.0s slow Bazarr network response
        await asyncio.sleep(2.0)

    monkeypatch.setattr(pipeline, "trigger_bazarr_search", slow_bazarr_search)

    # Instrument extract_embedded_srt to record exact start time
    original_sanitize = pipeline.translator.translate_srt_content

    async def tracking_translate(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        nonlocal ai_call_timestamp
        ai_call_timestamp = time.monotonic()
        return [
            srt.Subtitle(index=i+1, start=sub.start, end=sub.end, content=f"Hej {i+1} svensk text")
            for i, sub in enumerate(subs)
        ]

    monkeypatch.setattr(pipeline.translator, "translate_srt_content", tracking_translate)

    job_start_timestamp = time.monotonic()
    res = await pipeline.process_video_file(str(video_path), event_source="SONARR")
    total_elapsed = time.monotonic() - job_start_timestamp

    assert res["status"] == "translated"
    # The whole job completed in sub-second time without waiting for the 2.0s Bazarr request
    assert total_elapsed < 0.8
    assert ai_call_timestamp is not None
    assert (ai_call_timestamp - job_start_timestamp) < 0.8


@pytest.mark.asyncio
async def test_13_explicit_concurrency_ordering(hybrid_db_settings, tmp_path, monkeypatch):
    """6. Explicit concurrency ordering: verify source prep starts WHILE Bazarr search is actively in-flight."""
    video_path = tmp_path / "ordering.mkv"
    video_path.touch()
    en_srt = tmp_path / "ordering.en.srt"
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(make_valid_srt(lang="en", count=10))

    pipeline = SubtitlePipeline()
    bazarr_search_started = asyncio.Event()
    bazarr_search_finished = asyncio.Event()
    source_prep_ran_while_bazarr_inflight = False

    async def controlled_bazarr_search(vpath, language="sv"):
        bazarr_search_started.set()
        await asyncio.sleep(0.5)
        bazarr_search_finished.set()

    monkeypatch.setattr(pipeline, "trigger_bazarr_search", controlled_bazarr_search)

    # Intercept sanitize_srt_content_with_provenance to assert Bazarr search is in-flight
    from app.core import cleaner
    real_sanitize = cleaner.sanitize_srt_content_with_provenance

    async def tracking_sanitize(raw_text, **kwargs):
        nonlocal source_prep_ran_while_bazarr_inflight
        if bazarr_search_started.is_set() and not bazarr_search_finished.is_set():
            source_prep_ran_while_bazarr_inflight = True
        return await real_sanitize(raw_text, **kwargs)

    monkeypatch.setattr("app.services.pipeline.sanitize_srt_content_with_provenance", tracking_sanitize)

    async def mock_translate(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        return [
            srt.Subtitle(index=i+1, start=sub.start, end=sub.end, content=f"Hej {i+1} svensk text")
            for i, sub in enumerate(subs)
        ]

    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)

    res = await pipeline.process_video_file(str(video_path), event_source="SONARR")

    assert res["status"] == "translated"
    assert source_prep_ran_while_bazarr_inflight is True


@pytest.mark.asyncio
async def test_14_hung_or_slow_bazarr_trigger_cancelled_without_orphan_tasks(hybrid_db_settings, tmp_path, monkeypatch):
    """4. Hung Bazarr trigger cancelled safely upon prep completion -> no orphan tasks, no unhandled exceptions."""
    video_path = tmp_path / "hung_bazarr.mkv"
    video_path.touch()
    en_srt = tmp_path / "hung_bazarr.en.srt"
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(make_valid_srt(lang="en", count=10))

    pipeline = SubtitlePipeline()
    hung_event = asyncio.Event()

    async def hung_bazarr_search(vpath, language="sv"):
        # Hung HTTP request waiting forever until cancelled
        await hung_event.wait()

    monkeypatch.setattr(pipeline, "trigger_bazarr_search", hung_bazarr_search)

    async def mock_translate(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        return [
            srt.Subtitle(index=i+1, start=sub.start, end=sub.end, content=f"Hej {i+1} svensk text")
            for i, sub in enumerate(subs)
        ]

    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)

    res = await pipeline.process_video_file(str(video_path), event_source="SONARR")

    assert res["status"] == "translated"

    # Assert no background bazarr_search tasks remain running/orphaned
    current_tasks = [t for t in asyncio.all_tasks() if t.get_name().startswith("bazarr_search_")]
    assert len(current_tasks) == 0
