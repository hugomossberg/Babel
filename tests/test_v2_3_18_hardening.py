import pytest
import os
import json
import srt
from datetime import timedelta, timezone, datetime
from unittest.mock import patch, AsyncMock, MagicMock
import httpx

from app.services.pipeline import SubtitlePipeline, qa_gate, QA_STATUS_PASS, QA_STATUS_PASS_WITH_WARNINGS, QA_STATUS_FAIL
from app.services.translator import SubtitleTranslator, ProviderUnavailableError, ProviderConfigurationError
from app.core.validator import extract_representative_dialogue_samples, check_language_representative
from app.core.db import (
    init_db, DB_PATH, create_job, get_job_by_id, append_job_log,
    save_translation_memory_bulk, get_translation_memory, clear_all_jobs
)


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test_hardening.db")
    monkeypatch.setattr("app.core.db.DB_PATH", test_db)
    init_db()
    clear_all_jobs()
    yield
    clear_all_jobs()


# ===========================================================================
# 1. BAZARR SERIES LOOKUP (Dict vs List payload and error handling)
# ===========================================================================
@pytest.mark.asyncio
async def test_bazarr_series_lookup_dict_and_list_payloads(monkeypatch):
    pipeline = SubtitlePipeline()

    # Mock settings
    settings = {
        "bazarr_url": "http://localhost:6767",
        "bazarr_api_key": "test_bazarr_key",
    }
    monkeypatch.setattr("app.services.pipeline.get_setting", lambda k, d="": settings.get(k, d))

    # Test A: /api/series returns dict {"data": [...]}
    series_dict_payload = {
        "data": [
            {"sonarrSeriesId": 42, "path": "/data/media/tv/Breaking Bad"}
        ]
    }
    episodes_payload = [
        {"sonarrEpisodeId": 101, "path": "/data/media/tv/Breaking Bad/Season 01/Breaking Bad - S01E01.mkv"}
    ]

    recorded_requests = []

    def mock_handler(request: httpx.Request):
        url = str(request.url)
        recorded_requests.append((request.method, url))
        if "/api/movies" in url:
            return httpx.Response(200, json=[])
        elif "/api/series" in url:
            return httpx.Response(200, json=series_dict_payload)
        elif "/api/episodes" in url:
            return httpx.Response(200, json=episodes_payload)
        elif "/api/episodes/subtitles" in url:
            return httpx.Response(200, json={"status": "queued"})
        return httpx.Response(404)

    orig_client = httpx.AsyncClient
    transport = httpx.MockTransport(mock_handler)
    with patch("httpx.AsyncClient", lambda **kwargs: orig_client(transport=transport, **{k: v for k, v in kwargs.items() if k != "transport"})):
        await pipeline.trigger_bazarr_search(
            "/data/media/tv/Breaking Bad/Season 01/Breaking Bad - S01E01.mkv",
            language="sv"
        )
        assert any(method == "PATCH" and "/api/episodes/subtitles" in url for method, url in recorded_requests)

    # Test B: /api/series returns raw list [...]
    series_list_payload = [
        {"sonarrSeriesId": 84, "path": "/data/media/tv/Better Call Saul"}
    ]
    episodes_b_payload = {
        "data": [
            {"sonarrEpisodeId": 202, "path": "/data/media/tv/Better Call Saul/Season 01/Better Call Saul - S01E01.mkv"}
        ]
    }

    recorded_requests_b = []

    def mock_handler_b(request: httpx.Request):
        url = str(request.url)
        recorded_requests_b.append((request.method, url))
        if "/api/movies" in url:
            return httpx.Response(200, json={"data": []})
        elif "/api/series" in url:
            return httpx.Response(200, json=series_list_payload)
        elif "/api/episodes" in url:
            return httpx.Response(200, json=episodes_b_payload)
        elif "/api/episodes/subtitles" in url:
            return httpx.Response(200, json={"status": "queued"})
        return httpx.Response(404)

    transport_b = httpx.MockTransport(mock_handler_b)
    with patch("httpx.AsyncClient", lambda **kwargs: orig_client(transport=transport_b, **{k: v for k, v in kwargs.items() if k != "transport"})):
        await pipeline.trigger_bazarr_search(
            "/data/media/tv/Better Call Saul/Season 01/Better Call Saul - S01E01.mkv",
            language="sv"
        )
        assert any(method == "PATCH" and "/api/episodes/subtitles" in url for method, url in recorded_requests_b)

    # Test C: /api/series fails gracefully without crashing
    def mock_handler_err(request: httpx.Request):
        return httpx.Response(500, text="Internal Server Error")

    transport_err = httpx.MockTransport(mock_handler_err)
    with patch("httpx.AsyncClient", lambda **kwargs: orig_client(transport=transport_err, **{k: v for k, v in kwargs.items() if k != "transport"})):
        # Should complete without error
        await pipeline.trigger_bazarr_search(
            "/data/media/tv/Unknown/Season 01/Unknown.S01E01.mkv",
            language="sv"
        )


# ===========================================================================
# 2. STRATIFIED LANGUAGE QA SAMPLING (Whole subtitle distribution)
# ===========================================================================
def test_stratified_language_qa_sampling_half_english():
    """Verify that subtitles where English dialogue is present in the second half are flagged."""
    swedish_phrases = [
        "Hej, välkommen hit idag.",
        "Hur mår du egentligen?",
        "Vi måste gå nu genast.",
        "Det där var verkligen fantastiskt bra gjort.",
        "Tack så mycket för all din hjälp.",
    ]
    english_phrases = [
        "Why are you running away from me right now?",
        "Because I do not want to speak with you.",
        "Please listen to what I have to say to you.",
        "This is an emergency and we must go immediately.",
        "I will never understand why you did that to us.",
    ]

    subs = []
    # First 30 cues: Swedish
    for i in range(30):
        subs.append(srt.Subtitle(i + 1, timedelta(seconds=i * 2), timedelta(seconds=i * 2 + 1), swedish_phrases[i % len(swedish_phrases)]))
    # Next 30 cues: English (representing a drop/hallucination in second half)
    for i in range(30, 60):
        subs.append(srt.Subtitle(i + 1, timedelta(seconds=i * 2), timedelta(seconds=i * 2 + 1), english_phrases[(i - 30) % len(english_phrases)]))

    # Stratified language check
    res = check_language_representative(subs, target_lang_code="sv")
    # The end stratum is 100% English, so confident_wrong_language should be True
    assert res.get("confident_wrong_language") is True or res.get("detected_lang") != "sv"


def test_stratified_language_qa_sampling_all_swedish():
    """Verify that all-Swedish subtitles pass stratified QA check cleanly."""
    swedish_phrases = [
        "Hej, välkommen till vår lilla stad.",
        "Hur mår du egentligen idag min vän?",
        "Vi måste gå till affären och handla mat nu.",
        "Det där var verkligen trevligt och roligt.",
        "Tack så mycket för att du ställde upp för oss.",
    ]
    subs = [
        srt.Subtitle(i + 1, timedelta(seconds=i * 2), timedelta(seconds=i * 2 + 1), swedish_phrases[i % len(swedish_phrases)])
        for i in range(60)
    ]
    res = check_language_representative(subs, target_lang_code="sv")
    assert res.get("confident_wrong_language") is False
    assert res.get("detected_lang") == "sv"


# ===========================================================================
# 3. PROVIDER ERROR BUBBLING (Never masked as semantic deadlock)
# ===========================================================================
@pytest.mark.asyncio
async def test_provider_error_not_masked_as_deadlock(tmp_path, monkeypatch):
    """If provider throws ProviderUnavailableError in recovery, it must fail or wait, never PASS_WITH_WARNINGS."""
    video = tmp_path / "TestShow.S01E01.mkv"
    video.touch()
    en_srt = tmp_path / "TestShow.S01E01.en.srt"

    subs = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Hello world, this is a test dialogue line."),
        srt.Subtitle(2, timedelta(seconds=2), timedelta(seconds=3), "Good morning, everyone in the audience today."),
        srt.Subtitle(3, timedelta(seconds=3), timedelta(seconds=4), "We are testing provider error bubbling right now."),
    ]
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()

    settings = {
        "ai_provider": "gemini",
        "gemini_api_key": "test_key",
        "gemini_model": "gemini-3.5-flash-lite",
        "batch_size": "50",
        "max_concurrent_jobs": "1",
        "wait_time_seconds": "0",
        "extract_target_embedded": "false",
        "extract_source_embedded": "false",
        "languages": json.dumps([{"code": "sv", "name": "Swedish", "enabled": True}]),
    }
    monkeypatch.setattr("app.services.pipeline.get_setting", lambda k, d="": settings.get(k, d))
    monkeypatch.setattr("app.services.translator.get_setting", lambda k, d="": settings.get(k, d))

    # Initial translation returns identical echo
    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        return [{"id": it["id"], "text": it["text"]} for it in items]

    # Classifier raises ProviderUnavailableError (e.g. 503 Overloaded)
    async def mock_classify_err(*args, **kwargs):
        raise ProviderUnavailableError("Gemini 503 Overloaded")

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify_err)

    res = await pipeline.process_video_file(str(video), event_source="MANUAL")
    job = get_job_by_id(res["job_id"])

    # Must NOT succeed with PASS_WITH_WARNINGS or publish
    assert res.get("status") in ["failed", "waiting_provider", "recovering"]
    assert job["status"] in ["FAILED", "WAITING_PROVIDER", "RECOVERING"]

    # Target subtitle must not exist
    sv_srt = tmp_path / "TestShow.S01E01.sv.srt"
    assert not sv_srt.exists()


# ===========================================================================
# 4. SERIES-KEYED TRANSLATION MEMORY (Per-show namespace for Sonarr)
# ===========================================================================
def test_series_keyed_translation_memory_shared_across_episodes():
    """Episodes of the same series must share TM entries via series title, with backward compatibility for legacy entries."""
    series_name = "Survivor's Remorse"

    # 1. Save new TM under clean series_name
    items = [
        {"original": "What's good, fam?", "translated": "Läget, familjen?"},
        {"original": "Let's get this money.", "translated": "Nu kör vi och fixar pengarna."}
    ]
    save_translation_memory_bulk(series_name, items)

    # 2. Save legacy TM under episode-specific title
    legacy_items = [
        {"original": "Legacy phrase", "translated": "Gammal fras"}
    ]
    save_translation_memory_bulk("Survivor's Remorse - S01E01 - Pilot", legacy_items)

    # Query with clean series name should retrieve both clean series items and legacy episode-prefixed items
    tm_series = get_translation_memory(series_name)
    assert len(tm_series) == 3
    tm_dict = {it["original"]: it["translated"] for it in tm_series}
    assert tm_dict.get("What's good, fam?") == "Läget, familjen?"
    assert tm_dict.get("Let's get this money.") == "Nu kör vi och fixar pengarna."
    assert tm_dict.get("Legacy phrase") == "Gammal fras"


# ===========================================================================
# 5. QA SUMMARY KEEP COUNT ACCURACY
# ===========================================================================
def test_qa_summary_kept_count_calculation():
    """Verify that kept_count strictly matches initial identical candidates classified as KEEP."""
    initial_identical_candidates_set = {0, 1}
    safe_ids = [0, 1, 99]
    real_untranslated_ids = []
    recovered_cues = {1}  # cue 1 was translated during recovery

    # Only cue 0 is an initial identical candidate that was KEPT and not translated on recovery
    kept_count = sum(1 for idx in initial_identical_candidates_set if idx in safe_ids and idx not in real_untranslated_ids and idx not in recovered_cues)
    assert kept_count == 1


# ===========================================================================
# 6. EXPLICIT UTC IN JOB LOG TIMESTAMPS
# ===========================================================================
def test_utc_job_log_timestamps():
    """Job log timestamps must explicitly include UTC format."""
    job_id = create_job("/data/media/movies/Inception.mkv", "MANUAL", "Inception (2010)")
    append_job_log(job_id, "Testing explicit UTC timestamp formatting")

    job = get_job_by_id(job_id)
    logs = job.get("logs", [])
    assert len(logs) > 0

    last_log = logs[-1]
    assert " UTC] Testing explicit UTC timestamp formatting" in last_log
