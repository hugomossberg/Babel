import pytest
import os
import json
import srt
from datetime import timedelta
from unittest.mock import patch, AsyncMock, MagicMock

from app.services.pipeline import SubtitlePipeline
from app.services.translator import SubtitleTranslator, is_meaningful_translation, is_usable_translation
from app.core.db import get_job_by_id, DB_PATH, init_db

@pytest.fixture
def mock_db_settings(monkeypatch, tmp_path):
    test_db = str(tmp_path / "test_fast_rescue.db")
    monkeypatch.setattr("app.core.db.DB_PATH", test_db)
    init_db()

    def mock_get_setting(key, default=""):
        settings = {
            "ai_provider": "gemini",
            "gemini_api_key": "dummy_api_key",
            "gemini_model": "gemini-3.5-flash-lite",
            "escalate_to_pro": "false",
            "escalation_provider": "none",
            "escalation_model": "",
            "batch_size": "50",
            "max_concurrent_jobs": "1",
            "wait_time_seconds": "0",
            "extract_target_embedded": "false",
            "extract_source_embedded": "false",
            "languages": json.dumps([{"code": "sv", "name": "Swedish", "enabled": True}]),
        }
        return settings.get(key, default)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)
    monkeypatch.setattr("app.services.translator.get_setting", mock_get_setting)

    async def mock_escalate_none(*args, **kwargs):
        return None
    monkeypatch.setattr("app.services.translator.SubtitleTranslator.escalate_single_line", mock_escalate_none)
    monkeypatch.setattr("app.services.translator.SubtitleTranslator.first_pass_micro_repair_batch", AsyncMock(return_value=[]))


@pytest.mark.asyncio
async def test_1_batch_rescue_success(mock_db_settings, tmp_path, monkeypatch):
    """Test 1: 3 unresolved dialogue cues are rescued in exactly 1 batch call."""
    video_path = tmp_path / "Episode.S01E01.mkv"
    video_path.touch()
    en_srt = tmp_path / "Episode.S01E01.en.srt"

    cues_text = [
        "Hello?",
        "What are you doing?",
        "Come on, man."
    ]
    subs = [
        srt.Subtitle(index=i+1, start=timedelta(seconds=i*2), end=timedelta(seconds=i*2+1), content=cues_text[i])
        for i in range(len(cues_text))
    ]
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()

    # Initial batch translation returns untranslated English (simulating stubborn cues)
    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        return [{"id": item["id"], "text": item["text"]} for item in items]

    # Classifier classifies them as translate
    async def mock_classify(items, lang, title):
        return [{"id": item["id"], "action": "translate", "reason": "none", "text": ""} for item in items]

    rescue_call_counts = 0
    rescue_batches = []

    async def mock_rescue_batch(items, target_language, show_title="", attempt=1, job_id=None):
        nonlocal rescue_call_counts
        rescue_call_counts += 1
        rescue_batches.append((attempt, [it["id"] for it in items]))
        # Returns valid Swedish translations for all 3
        translations = {
            0: "Hej?",
            1: "Vad gör du?",
            2: "Kom igen, kompis."
        }
        return [{"id": it["id"], "text": translations[it["id"]]} for it in items]

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(pipeline.translator, "fast_final_rescue_batch", mock_rescue_batch)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
    assert res["status"] != "failed"

    job = get_job_by_id(res["job_id"])
    assert job["status"] == "TRANSLATED"
    assert rescue_call_counts == 1
    assert rescue_batches == [(1, [0, 1, 2])]

    # Verify target subtitle file was written and has Swedish content
    sv_srt = tmp_path / "Episode.S01E01.sv.srt"
    assert os.path.exists(sv_srt)
    with open(sv_srt, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Hej?" in content
    assert "Vad gör du?" in content
    assert "Kom igen, kompis." in content


@pytest.mark.asyncio
async def test_2_normalized_echoes_rejected(mock_db_settings, tmp_path, monkeypatch):
    """Test 2: Normalized echoes (Hello! for Hello?, <i>Come on.</i> for Come on!) are rejected."""
    assert not is_meaningful_translation("Hello?", "Hello!")
    assert not is_meaningful_translation("Come on!", "<i>Come on.</i>")
    assert not is_meaningful_translation("What did you say?", "what did you say?")

    video_path = tmp_path / "Episode.S01E02.mkv"
    video_path.touch()
    en_srt = tmp_path / "Episode.S01E02.en.srt"

    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="Hello, how are you?"),
        srt.Subtitle(index=2, start=timedelta(seconds=3), end=timedelta(seconds=4), content="Come on, let's go!"),
        srt.Subtitle(index=3, start=timedelta(seconds=5), end=timedelta(seconds=6), content="What did you say?")
    ]
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()
    async def mock_translate_batch(items, *args, **kwargs):
        return [{"id": i["id"], "text": i["text"]} for i in items]

    async def mock_classify(items, *args, **kwargs):
        return [{"id": i["id"], "action": "translate", "reason": "none", "text": ""} for i in items]

    async def mock_rescue_batch(items, target_language, show_title="", attempt=1, job_id=None):
        # Always return normalized echoes
        echoes = {0: "Hello, how are you!", 1: "<i>Come on, let's go!</i>", 2: "what did you say?"}
        return [{"id": it["id"], "text": echoes.get(it["id"], it["text"])} for it in items]

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(pipeline.translator, "fast_final_rescue_batch", mock_rescue_batch)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
    job = get_job_by_id(res["job_id"])
    # Must fail QA because normalized echoes are rejected
    assert job["status"] in ["RECOVERING", "FAILED"]
    sv_srt = tmp_path / "Episode.S01E02.sv.srt"
    assert not os.path.exists(sv_srt)


@pytest.mark.asyncio
async def test_3_second_batch_only_contains_failures(mock_db_settings, tmp_path, monkeypatch):
    """Test 3: Attempt 2 contains ONLY the failed cues from Attempt 1."""
    video_path = tmp_path / "Episode.S01E03.mkv"
    video_path.touch()
    en_srt = tmp_path / "Episode.S01E03.en.srt"

    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="Hello?"),
        srt.Subtitle(index=2, start=timedelta(seconds=3), end=timedelta(seconds=4), content="What are you doing?"),
        srt.Subtitle(index=3, start=timedelta(seconds=5), end=timedelta(seconds=6), content="Good morning.")
    ]
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()
    async def mock_translate_batch(items, *args, **kwargs):
        return [{"id": i["id"], "text": i["text"]} for i in items]

    async def mock_classify(items, *args, **kwargs):
        return [{"id": i["id"], "action": "translate", "reason": "none", "text": ""} for i in items]

    rescue_batches = []

    async def mock_rescue_batch(items, target_language, show_title="", attempt=1, job_id=None):
        item_ids = [it["id"] for it in items]
        rescue_batches.append((attempt, item_ids))
        if attempt == 1:
            # 0 and 2 translated, 1 returns normalized echo
            return [
                {"id": 0, "text": "Hej?"},
                {"id": 1, "text": "What are you doing!"}, # echo
                {"id": 2, "text": "God morgon."}
            ]
        else:
            # Attempt 2 returns valid Swedish for id 1
            return [{"id": 1, "text": "Vad gör du?"}]

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(pipeline.translator, "fast_final_rescue_batch", mock_rescue_batch)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
    job = get_job_by_id(res["job_id"])
    assert job["status"] == "TRANSLATED"

    # Verify batch sizes and IDs
    assert rescue_batches[0] == (1, [0, 1, 2])
    assert rescue_batches[1] == (2, [1])

    sv_srt = tmp_path / "Episode.S01E03.sv.srt"
    assert os.path.exists(sv_srt)
    with open(sv_srt, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Hej?" in content
    assert "Vad gör du?" in content
    assert "God morgon." in content


@pytest.mark.asyncio
async def test_4_max_two_rescue_calls(mock_db_settings, tmp_path, monkeypatch):
    """Test 4: Exactly max 2 rescue calls per loop, stopping and fail-closed when exhausted."""
    video_path = tmp_path / "Episode.S01E04.mkv"
    video_path.touch()
    en_srt = tmp_path / "Episode.S01E04.en.srt"

    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="Hello, how are you today?"),
        srt.Subtitle(index=2, start=timedelta(seconds=3), end=timedelta(seconds=4), content="I am doing quite well, thank you.")
    ]
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()
    async def mock_translate_batch(items, *args, **kwargs):
        return [{"id": i["id"], "text": i["text"]} for i in items]

    async def mock_classify(items, *args, **kwargs):
        return [{"id": i["id"], "action": "translate", "reason": "none", "text": ""} for i in items]

    rescue_calls_per_loop = []

    async def mock_rescue_batch(items, target_language, show_title="", attempt=1, job_id=None):
        rescue_calls_per_loop.append(attempt)
        # Always echo
        return [{"id": it["id"], "text": it["text"]} for it in items]

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(pipeline.translator, "fast_final_rescue_batch", mock_rescue_batch)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
    job = get_job_by_id(res["job_id"])
    assert job["status"] in ["RECOVERING", "FAILED"]

    # In each QA loop (max 3 loops), Fast Final Rescue runs max 2 attempts (attempt 1 and attempt 2)
    # Loop 1: [1, 2], Loop 2: [1, 2], Loop 3: [1, 2]
    assert rescue_calls_per_loop[:2] == [1, 2]
    assert all(a in (1, 2) for a in rescue_calls_per_loop)


@pytest.mark.asyncio
async def test_5_final_qa_still_blocks(mock_db_settings, tmp_path, monkeypatch):
    """Test 5: If any cue remains English after Fast Final Rescue, publishing is blocked."""
    video_path = tmp_path / "Episode.S01E05.mkv"
    video_path.touch()
    en_srt = tmp_path / "Episode.S01E05.en.srt"

    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="Hello, how are you today?"),
        srt.Subtitle(index=2, start=timedelta(seconds=3), end=timedelta(seconds=4), content="Goodbye, my dear friend.")
    ]
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()
    async def mock_translate_batch(items, *args, **kwargs):
        return [{"id": i["id"], "text": i["text"]} for i in items]

    async def mock_classify(items, *args, **kwargs):
        return [{"id": i["id"], "action": "translate", "reason": "none", "text": ""} for i in items]

    async def mock_rescue_batch(items, target_language, show_title="", attempt=1, job_id=None):
        # Rescues id 0 ("Hej, hur mår du idag?"), but fails id 1 ("Goodbye, my dear friend.")
        return [
            {"id": 0, "text": "Hej, hur mår du idag?"},
            {"id": 1, "text": "Goodbye, my dear friend."} # untranslated
        ]

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(pipeline.translator, "fast_final_rescue_batch", mock_rescue_batch)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
    job = get_job_by_id(res["job_id"])
    assert job["status"] in ["RECOVERING", "FAILED"]

    sv_srt = tmp_path / "Episode.S01E05.sv.srt"
    assert not os.path.exists(sv_srt)


@pytest.mark.asyncio
async def test_6_safe_keep_unaffected(mock_db_settings, tmp_path, monkeypatch):
    """Test 6: Safe KEEP lines (NASA, 911, ?!) are not dispatched to Fast Final Rescue."""
    video_path = tmp_path / "Episode.S01E06.mkv"
    video_path.touch()
    en_srt = tmp_path / "Episode.S01E06.en.srt"

    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="NASA"),
        srt.Subtitle(index=2, start=timedelta(seconds=3), end=timedelta(seconds=4), content="911"),
        srt.Subtitle(index=3, start=timedelta(seconds=5), end=timedelta(seconds=6), content="?!"),
        srt.Subtitle(index=4, start=timedelta(seconds=7), end=timedelta(seconds=8), content="Hello?")
    ]
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()
    async def mock_translate_batch(items, *args, **kwargs):
        return [{"id": i["id"], "text": i["text"]} for i in items]

    async def mock_classify(items, lang, title):
        results = []
        for it in items:
            t = it["text"]
            if t == "NASA":
                results.append({"id": it["id"], "action": "keep", "reason": "acronym", "text": t})
            elif t == "911":
                results.append({"id": it["id"], "action": "keep", "reason": "number", "text": t})
            elif t == "?!":
                results.append({"id": it["id"], "action": "keep", "reason": "symbol", "text": t})
            else:
                results.append({"id": it["id"], "action": "translate", "reason": "none", "text": ""} )
        return results

    rescued_ids = []

    async def mock_rescue_batch(items, target_language, show_title="", attempt=1, job_id=None):
        for it in items:
            rescued_ids.append(it["id"])
        return [{"id": 3, "text": "Hej?"}]

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(pipeline.translator, "fast_final_rescue_batch", mock_rescue_batch)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
    job = get_job_by_id(res["job_id"])
    assert job["status"] == "TRANSLATED"

    # Only line index 3 ("Hello?") must have been sent to rescue. 0, 1, 2 were safe KEEP!
    assert rescued_ids == [3]


@pytest.mark.asyncio
async def test_7_timestamps_untouched(mock_db_settings, tmp_path, monkeypatch):
    """Test 7: Fast Final Rescue never modifies subtitle timestamps."""
    video_path = tmp_path / "Episode.S01E07.mkv"
    video_path.touch()
    en_srt = tmp_path / "Episode.S01E07.en.srt"

    orig_start = timedelta(seconds=12, milliseconds=345)
    orig_end = timedelta(seconds=15, milliseconds=678)
    subs = [
        srt.Subtitle(index=1, start=orig_start, end=orig_end, content="Hello, are you there with me?"),
        srt.Subtitle(index=2, start=timedelta(seconds=16), end=timedelta(seconds=18), content="Yes, I am right here with you.")
    ]
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()
    async def mock_translate_batch(items, *args, **kwargs):
        return [{"id": i["id"], "text": i["text"]} for i in items]

    async def mock_classify(items, *args, **kwargs):
        return [{"id": i["id"], "action": "translate", "reason": "none", "text": ""} for i in items]

    async def mock_rescue_batch(items, *args, **kwargs):
        return [
            {"id": 0, "text": "Hej, är du där med mig?"},
            {"id": 1, "text": "Ja, jag är här med dig."}
        ]

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(pipeline.translator, "fast_final_rescue_batch", mock_rescue_batch)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
    job = get_job_by_id(res["job_id"])
    assert job["status"] == "TRANSLATED"

    sv_srt = tmp_path / "Episode.S01E07.sv.srt"
    with open(sv_srt, "r", encoding="utf-8") as f:
        published_subs = list(srt.parse(f.read()))

    assert len(published_subs) == 2
    assert published_subs[0].start == orig_start
    assert published_subs[0].end == orig_end
    assert published_subs[0].content == "Hej, är du där med mig?"


@pytest.mark.asyncio
async def test_8_malformed_missing_results(mock_db_settings, tmp_path, monkeypatch):
    """Test 8: Malformed, missing, and duplicate results are handled fail-closed."""
    video_path = tmp_path / "Episode.S01E08.mkv"
    video_path.touch()
    en_srt = tmp_path / "Episode.S01E08.en.srt"

    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="Hello, where are you going today?"),
        srt.Subtitle(index=2, start=timedelta(seconds=3), end=timedelta(seconds=4), content="What are you doing over there?")
    ]
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()
    async def mock_translate_batch(items, *args, **kwargs):
        return [{"id": i["id"], "text": i["text"]} for i in items]

    async def mock_classify(items, *args, **kwargs):
        return [{"id": i["id"], "action": "translate", "reason": "none", "text": ""} for i in items]

    async def mock_rescue_batch(items, target_language, show_title="", attempt=1, job_id=None):
        # Returns invalid results: non-existent id 999, corrupt entry, missing id 1
        return [
            {"id": 999, "text": "Hittepå"},
            "corrupt_entry",
            {"id": None, "text": "No id"},
            {"id": 0, "text": "Hej, vart är du på väg idag?"}
        ]

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(pipeline.translator, "fast_final_rescue_batch", mock_rescue_batch)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
    job = get_job_by_id(res["job_id"])
    # Cue 1 was missing from rescue results, so QA gate must fail-closed
    assert job["status"] in ["RECOVERING", "FAILED"]


@pytest.mark.asyncio
async def test_9_batch_performance_behavior(mock_db_settings, tmp_path, monkeypatch):
    """Test 9: 5 unresolved cues are dispatched in ONE single batch request rather than 5 separate requests."""
    video_path = tmp_path / "Episode.S01E09.mkv"
    video_path.touch()
    en_srt = tmp_path / "Episode.S01E09.en.srt"

    subs = [
        srt.Subtitle(index=i+1, start=timedelta(seconds=i), end=timedelta(seconds=i+1), content=f"Stubborn dialogue {i}")
        for i in range(5)
    ]
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()
    async def mock_translate_batch(items, *args, **kwargs):
        return [{"id": i["id"], "text": i["text"]} for i in items]

    async def mock_classify(items, *args, **kwargs):
        return [{"id": i["id"], "action": "translate", "reason": "none", "text": ""} for i in items]

    api_batch_call_count = 0
    batch_sizes = []

    async def mock_rescue_batch(items, target_language, show_title="", attempt=1, job_id=None):
        nonlocal api_batch_call_count
        api_batch_call_count += 1
        batch_sizes.append(len(items))
        return [{"id": it["id"], "text": f"Svensk replik {it['id']}"} for it in items]

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(pipeline.translator, "fast_final_rescue_batch", mock_rescue_batch)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
    job = get_job_by_id(res["job_id"])
    assert job["status"] == "TRANSLATED"

    # Exactly 1 batch API call with all 5 cues together
    assert api_batch_call_count == 1
    assert batch_sizes == [5]
