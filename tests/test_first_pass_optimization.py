import pytest
import os
import json
import srt
from datetime import timedelta
from unittest.mock import AsyncMock, patch

from app.services.translator import (
    SubtitleTranslator,
    is_safe_keep_prefilter,
    is_meaningful_translation,
    is_usable_translation
)
from app.services.pipeline import SubtitlePipeline, qa_gate
import app.core.db


@pytest.fixture
def mock_db_settings(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    os.makedirs(db_path.parent, exist_ok=True)
    monkeypatch.setattr("app.core.db.DB_PATH", str(db_path))
    app.core.db.init_db()

    def mock_get_setting(key, default=""):
        settings = {
            "ai_provider": "gemini",
            "gemini_api_key": "test-key-123",
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
    monkeypatch.setattr("app.core.db.get_setting", mock_get_setting)
    return db_path


# Requirement 11 & 12: Safe deterministic KEEP cues not sent to model
def test_1_safe_deterministic_keep_not_sent_to_model():
    """Pure numbers, symbols, non-verbal, and strict acronyms are prefiltered."""
    safe_samples = [
        "123",
        "12:30:00",
        "$500.00",
        "...",
        "---",
        "♪",
        "<i></i>",
        "   ",
        "FBI",
        "NASA",
        "CIA",
        "DNA",
        "<i>FBI</i>",
    ]
    for s in safe_samples:
        assert is_safe_keep_prefilter(s) is True, f"Expected '{s}' to be safe KEEP"


# Requirement 5, 6, 7, 8, 9, 10: Real dialogue sent to model (Conservative pre-filter)
def test_2_real_dialogue_never_prefiltered():
    """Real dialogue, titles, names, formatting-wrapped text, numbers+words must NOT be prefiltered."""
    dialogue_samples = [
        "Hello",
        "Hello!",
        "Right",
        "Okay",
        "Yes",
        "No",
        "Hey",
        "What?",
        "Why?",
        "Come on",
        "Let's go",
        "Stop",
        "Help",
        "John?",
        "Michael!",
        "Jesus!",
        "God!",
        "Dad",
        "Mom",
        "Sir",
        "Doctor",
        "Captain",
        "Room 101",
        "Line 0",
        "Level 5",
        "Plan B",
        "Take 5",
        "Channel 4",
        "May",
        "Will",
        "Hope",
        "Rose",
        "Summer",
        "April",
        "March",
        "John Smith",
        "<i>Hello</i>",
        "<i>Hello!</i>",
        "<i>What?</i>",
        "<i>Come on.</i>",
        "HELLO",
        "WHAT",
    ]
    for d in dialogue_samples:
        assert is_safe_keep_prefilter(d) is False, f"Expected dialogue '{d}' NOT to be filtered"


# Test 3: Zero micro-repair calls when all main pass translations are good
@pytest.mark.asyncio
async def test_3_zero_micro_repair_calls_on_good_batch(mock_db_settings, monkeypatch):
    """When all main pass translations are valid, 0 micro-repair calls are made."""
    translator = SubtitleTranslator()
    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=1), content="Hello"),
        srt.Subtitle(index=2, start=timedelta(seconds=1), end=timedelta(seconds=2), content="Good morning"),
    ]

    micro_repair_calls = 0

    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        return [
            {"id": 0, "text": "Hej"},
            {"id": 1, "text": "God morgon"},
        ]

    async def mock_first_pass_micro_repair_batch(items, target_language, show_title="", **kwargs):
        nonlocal micro_repair_calls
        micro_repair_calls += 1
        return []

    monkeypatch.setattr(translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(translator, "first_pass_micro_repair_batch", mock_first_pass_micro_repair_batch)

    res = await translator.translate_srt_content(subs, target_language="Swedish", batch_size=50)

    assert len(res) == 2
    assert res[0].content == "Hej"
    assert res[1].content == "God morgon"
    assert micro_repair_calls == 0


# Test 4: Exactly one micro-repair batch call when multiple cues fail
@pytest.mark.asyncio
async def test_4_exactly_one_micro_repair_batch_on_failures(mock_db_settings, monkeypatch):
    """When multiple cues fail in a batch, exactly 1 batch micro-repair call is made."""
    translator = SubtitleTranslator()
    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=1), content="Hello"),
        srt.Subtitle(index=2, start=timedelta(seconds=1), end=timedelta(seconds=2), content="Good morning"),
        srt.Subtitle(index=3, start=timedelta(seconds=2), end=timedelta(seconds=3), content="How are you?"),
    ]

    micro_repair_calls = []

    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        # Return identical echo for items 0 and 2
        return [
            {"id": 0, "text": "Hello"},
            {"id": 1, "text": "God morgon"},
            {"id": 2, "text": "How are you?"},
        ]

    async def mock_first_pass_micro_repair_batch(items, target_language, show_title="", **kwargs):
        micro_repair_calls.append(items)
        return [
            {"id": 0, "text": "Hej"},
            {"id": 2, "text": "Hur mår du?"},
        ]

    monkeypatch.setattr(translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(translator, "first_pass_micro_repair_batch", mock_first_pass_micro_repair_batch)

    res = await translator.translate_srt_content(subs, target_language="Swedish", batch_size=50)

    assert len(micro_repair_calls) == 1
    assert len(micro_repair_calls[0]) == 2
    assert res[0].content == "Hej"
    assert res[1].content == "God morgon"
    assert res[2].content == "Hur mår du?"


# Test 5: Micro-repair payload contains only failed dialogue with local context
@pytest.mark.asyncio
async def test_5_micro_repair_payload_contains_only_failed_dialogue_with_context(mock_db_settings, monkeypatch):
    """Micro-repair items include target, id, and local prev/next context."""
    translator = SubtitleTranslator()
    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=1), content="First line"),
        srt.Subtitle(index=2, start=timedelta(seconds=1), end=timedelta(seconds=2), content="Second line"),
        srt.Subtitle(index=3, start=timedelta(seconds=2), end=timedelta(seconds=3), content="Third line"),
    ]

    captured_payload = None

    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        # Only line 1 (Second line) fails
        return [
            {"id": 0, "text": "Första raden"},
            {"id": 1, "text": "Second line"}, # Echo
            {"id": 2, "text": "Tredje raden"},
        ]

    async def mock_first_pass_micro_repair_batch(items, target_language, show_title="", **kwargs):
        nonlocal captured_payload
        captured_payload = items
        return [{"id": 1, "text": "Andra raden"}]

    monkeypatch.setattr(translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(translator, "first_pass_micro_repair_batch", mock_first_pass_micro_repair_batch)

    res = await translator.translate_srt_content(subs, target_language="Swedish", batch_size=50)

    assert captured_payload is not None
    assert len(captured_payload) == 1
    item = captured_payload[0]
    assert item["id"] == 1
    assert item["target"] == "Second line"
    assert "First line" in item["context_before"]
    assert "Third line" in item["context_after"]
    assert res[1].content == "Andra raden"


# Requirement 1, 2, 3: Normalized echo rejection in main translation
def test_6_normalized_echo_identified_immediately():
    """Normalized echoes are flagged as invalid translations immediately."""
    assert is_meaningful_translation("Hello?", "Hello!") is False
    assert is_meaningful_translation("Hello", "<i>Hello.</i>") is False
    assert is_meaningful_translation("HELLO", "hello") is False
    assert is_meaningful_translation("Come on.", "Come on!") is False
    assert is_meaningful_translation("What?", "<i>What?</i>") is False
    assert is_meaningful_translation("Hello", "Hej") is True
    assert is_meaningful_translation("Come on", "Kom igen") is True


# Requirement 4 & 14: Normalized echo from micro-repair rejected
@pytest.mark.asyncio
async def test_7_normalized_echo_from_micro_repair_rejected(mock_db_settings, monkeypatch):
    """Normalized echo from micro-repair (e.g. Come on. -> Come on!) is rejected and original remains."""
    translator = SubtitleTranslator()
    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=1), content="Come on."),
    ]

    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        return [{"id": 0, "text": "Come on."}]

    async def mock_first_pass_micro_repair_batch(items, target_language, show_title="", **kwargs):
        # Micro repair returns normalized echo with different punctuation
        return [{"id": 0, "text": "Come on!"}]

    monkeypatch.setattr(translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(translator, "first_pass_micro_repair_batch", mock_first_pass_micro_repair_batch)

    res = await translator.translate_srt_content(subs, target_language="Swedish", batch_size=50)

    # Remains unresolved (original Come on.) so downstream recovery handles it
    assert res[0].content == "Come on."


# Test 8: Successful micro-repair updates text properly
@pytest.mark.asyncio
async def test_8_successful_micro_repair_updates_text(mock_db_settings, monkeypatch):
    """Successful micro-repair replaces cue text with Swedish translation."""
    translator = SubtitleTranslator()
    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=1), content="Sure"),
    ]

    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        return [{"id": 0, "text": "Sure"}]

    async def mock_first_pass_micro_repair_batch(items, target_language, show_title="", **kwargs):
        return [{"id": 0, "text": "Visst"}]

    monkeypatch.setattr(translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(translator, "first_pass_micro_repair_batch", mock_first_pass_micro_repair_batch)

    res = await translator.translate_srt_content(subs, target_language="Swedish", batch_size=50)
    assert res[0].content == "Visst"


# Test 9: Timestamps untouched (0ms drift)
@pytest.mark.asyncio
async def test_9_timestamps_untouched(mock_db_settings, monkeypatch):
    """Source start/end timestamps are strictly preserved."""
    translator = SubtitleTranslator()
    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=12, milliseconds=345), end=timedelta(seconds=14, milliseconds=678), content="Hello"),
        srt.Subtitle(index=2, start=timedelta(seconds=15, milliseconds=100), end=timedelta(seconds=18, milliseconds=200), content="123"),
    ]

    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        return [{"id": 0, "text": "Hej"}]

    monkeypatch.setattr(translator, "translate_batch", mock_translate_batch)

    res = await translator.translate_srt_content(subs, target_language="Swedish", batch_size=50)
    assert res[0].start == timedelta(seconds=12, milliseconds=345)
    assert res[0].end == timedelta(seconds=14, milliseconds=678)
    assert res[1].start == timedelta(seconds=15, milliseconds=100)
    assert res[1].end == timedelta(seconds=18, milliseconds=200)


# Test 10: Cue order/index/count unchanged
@pytest.mark.asyncio
async def test_10_cue_order_index_count_unchanged(mock_db_settings, monkeypatch):
    """Output contains exact same number of subtitles in exact order."""
    translator = SubtitleTranslator()
    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="One"),
        srt.Subtitle(index=2, start=timedelta(seconds=2), end=timedelta(seconds=3), content="100"),
        srt.Subtitle(index=3, start=timedelta(seconds=3), end=timedelta(seconds=4), content="Three"),
    ]

    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        return [{"id": 0, "text": "Ett"}, {"id": 2, "text": "Tre"}]

    monkeypatch.setattr(translator, "translate_batch", mock_translate_batch)

    res = await translator.translate_srt_content(subs, target_language="Swedish", batch_size=50)
    assert len(res) == 3
    assert res[0].index == 1 and res[0].content == "Ett"
    assert res[1].index == 2 and res[1].content == "100"
    assert res[2].index == 3 and res[2].content == "Tre"


# Requirement 11 & 12: Safe KEEP cues unaffected
@pytest.mark.asyncio
async def test_11_safe_keep_cues_unaffected(mock_db_settings, monkeypatch):
    """Deterministic safe KEEP cues bypass translation and remain intact."""
    translator = SubtitleTranslator()
    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="FBI"),
        srt.Subtitle(index=2, start=timedelta(seconds=2), end=timedelta(seconds=3), content="♪"),
        srt.Subtitle(index=3, start=timedelta(seconds=3), end=timedelta(seconds=4), content="123"),
    ]

    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        # Should not be called because all cues are safe KEEP
        return []

    monkeypatch.setattr(translator, "translate_batch", mock_translate_batch)

    res = await translator.translate_srt_content(subs, target_language="Swedish", batch_size=50)
    assert res[0].content == "FBI"
    assert res[1].content == "♪"
    assert res[2].content == "123"


# Test 12: Missing or malformed micro-repair output fails closed
@pytest.mark.asyncio
async def test_12_malformed_missing_micro_repair_fails_closed(mock_db_settings, monkeypatch):
    """Malformed or empty micro-repair output leaves cues untouched for downstream recovery."""
    translator = SubtitleTranslator()
    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=1), content="Hello"),
    ]

    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        return [{"id": 0, "text": "Hello"}]

    async def mock_first_pass_micro_repair_batch(items, target_language, show_title="", **kwargs):
        # Returns invalid response
        return "Not a valid list"

    monkeypatch.setattr(translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(translator, "first_pass_micro_repair_batch", mock_first_pass_micro_repair_batch)

    res = await translator.translate_srt_content(subs, target_language="Swedish", batch_size=50)
    assert res[0].content == "Hello"


# Test 13: Unknown repair IDs do not modify valid cues
@pytest.mark.asyncio
async def test_13_unknown_repair_ids_ignored(mock_db_settings, monkeypatch):
    """Out of range or mismatched IDs from micro repair are ignored."""
    translator = SubtitleTranslator()
    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=1), content="Hello"),
    ]

    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        return [{"id": 0, "text": "Hello"}]

    async def mock_first_pass_micro_repair_batch(items, target_language, show_title="", **kwargs):
        return [{"id": 999, "text": "Spurious data"}]

    monkeypatch.setattr(translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(translator, "first_pass_micro_repair_batch", mock_first_pass_micro_repair_batch)

    res = await translator.translate_srt_content(subs, target_language="Swedish", batch_size=50)
    assert res[0].content == "Hello"


# Test 14: No second micro-repair attempt is made
@pytest.mark.asyncio
async def test_14_no_second_micro_repair_attempt(mock_db_settings, monkeypatch):
    """Micro-repair is strictly called at most once per batch."""
    translator = SubtitleTranslator()
    subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=1), content="Stubborn"),
    ]

    repair_call_count = 0

    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        return [{"id": 0, "text": "Stubborn"}]

    async def mock_first_pass_micro_repair_batch(items, target_language, show_title="", **kwargs):
        nonlocal repair_call_count
        repair_call_count += 1
        return [{"id": 0, "text": "Stubborn"}]

    monkeypatch.setattr(translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(translator, "first_pass_micro_repair_batch", mock_first_pass_micro_repair_batch)

    res = await translator.translate_srt_content(subs, target_language="Swedish", batch_size=50)
    assert repair_call_count == 1
    assert res[0].content == "Stubborn"


# Test 15: Same primary provider and model used
@pytest.mark.asyncio
async def test_15_same_primary_provider_model_used(mock_db_settings, monkeypatch):
    """first_pass_micro_repair_batch uses the primary provider settings."""
    translator = SubtitleTranslator()
    repair_items = [{"id": 0, "target": "Hello", "prev_context": "", "next_context": ""}]

    captured_model = None

    class MockModels:
        def generate_content(self, model, contents, config=None):
            nonlocal captured_model
            captured_model = model
            return type('obj', (object,), {'text': '{"translations": [{"id": 0, "text": "Hej"}]}'})

    class MockClient:
        def __init__(self, api_key=None):
            self.models = MockModels()

    monkeypatch.setattr("app.services.translator.genai.Client", MockClient)

    res = await translator.first_pass_micro_repair_batch(repair_items, target_language="Swedish")
    assert captured_model == "gemini-3.5-flash-lite"
    assert res == [{"id": 0, "text": "Hej"}]


# Test 16: Downstream Fast Final Rescue still works
@pytest.mark.asyncio
async def test_16_downstream_fast_final_rescue_still_works(mock_db_settings, tmp_path, monkeypatch):
    """When first-pass micro repair does not resolve a cue, Fast Final Rescue recovers it."""
    video_path = tmp_path / "Show.S01E01.mkv"
    video_path.touch()
    en_srt = tmp_path / "Show.S01E01.en.srt"

    cues_text = [
        "Welcome to the show.",
        "What are you doing here today?",
        "This is a stubborn line.",
        "Everything will be fine.",
        "See you soon."
    ]
    subs = [
        srt.Subtitle(index=i+1, start=timedelta(seconds=i*2), end=timedelta(seconds=i*2+1), content=cues_text[i])
        for i in range(len(cues_text))
    ]
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    pipeline = SubtitlePipeline()

    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        # Initial pass translates cue 0, 1, 3, 4, but echoes cue 2
        results = []
        for it in items:
            if it["id"] == 2:
                results.append({"id": it["id"], "text": it["text"]}) # stubborn echo
            else:
                results.append({"id": it["id"], "text": f"Svenska {it['id']}"})
        return results

    async def mock_first_pass_micro_repair(items, target_language, show_title="", **kwargs):
        # Micro repair also echoes
        return [{"id": it["id"], "text": it["target"]} for it in items]

    async def mock_classify(items, lang, title):
        return [{"id": item["id"], "action": "translate", "reason": "none", "text": ""} for item in items]

    async def mock_fast_final_rescue(items, target_language, show_title="", **kwargs):
        # Fast Final Rescue downstream saves it
        return [{"id": it["id"], "text": "Räddad rad"} for it in items]

    monkeypatch.setattr(pipeline.translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(pipeline.translator, "first_pass_micro_repair_batch", mock_first_pass_micro_repair)
    monkeypatch.setattr(pipeline.translator, "classify_and_recover_identical", mock_classify)
    monkeypatch.setattr(pipeline.translator, "fast_final_rescue_batch", mock_fast_final_rescue)

    res = await pipeline.process_video_file(str(video_path), event_source="MANUAL")
    assert res["status"] != "failed"
    job = app.core.db.get_job_by_id(res["job_id"])
    assert job["status"] == "TRANSLATED"


# Test 17: Strict final QA still blocks unresolved English
def test_17_strict_final_qa_still_blocks():
    """QA gate fails if unresolved English dialogue remains."""
    source_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=1), content="Hello there"),
    ]
    translated_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=1), content="Hello there"),
    ]
    qa_res = qa_gate(source_subs, translated_subs, target_lang_code="sv")
    assert qa_res["passed"] is False
    assert 0 in qa_res["real_untranslated_ids"]


# Test 18: Dropped cues blocked
def test_18_dropped_cues_blocked():
    """QA gate fails if cue count or content is missing/dropped."""
    source_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=1), content="Hello there"),
        srt.Subtitle(index=2, start=timedelta(seconds=1), end=timedelta(seconds=2), content="How are you?"),
    ]
    translated_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=1), content="Hej där"),
    ]
    qa_res = qa_gate(source_subs, translated_subs, target_lang_code="sv")
    assert qa_res["passed"] is False
    assert qa_res["dropped_count"] > 0


# Test 19: Performance behavior - multiple failed cues in 1 batch
@pytest.mark.asyncio
async def test_19_batch_performance_behavior(mock_db_settings, monkeypatch):
    """5 failed cues in a batch are sent in 1 micro-repair call, not 5 separate calls."""
    translator = SubtitleTranslator()
    subs = []
    for i in range(10):
        subs.append(srt.Subtitle(index=i+1, start=timedelta(seconds=i), end=timedelta(seconds=i+1), content=f"Dialogue line {i}"))

    repair_call_count = 0

    async def mock_translate_batch(items, target_language, show_title="", **kwargs):
        # 5 cues fail (indices 0..4), 5 pass (indices 5..9)
        results = []
        for item in items:
            idx = item["id"]
            if idx < 5:
                results.append({"id": idx, "text": item["text"]}) # identical fail
            else:
                results.append({"id": idx, "text": f"Svensk rad {idx}"})
        return results

    async def mock_first_pass_micro_repair_batch(items, target_language, show_title="", **kwargs):
        nonlocal repair_call_count
        repair_call_count += 1
        assert len(items) == 5
        return [{"id": item["id"], "text": f"Lagad rad {item['id']}"} for item in items]

    monkeypatch.setattr(translator, "translate_batch", mock_translate_batch)
    monkeypatch.setattr(translator, "first_pass_micro_repair_batch", mock_first_pass_micro_repair_batch)

    res = await translator.translate_srt_content(subs, target_language="Swedish", batch_size=50)
    assert repair_call_count == 1
    for i in range(5):
        assert res[i].content == f"Lagad rad {i}"
    for i in range(5, 10):
        assert res[i].content == f"Svensk rad {i}"


# Requirement 13 & 14: Direct comparison verification for main & micro-repair
def test_20_meaningful_translation_semantics():
    """Verify is_meaningful_translation handles all source vs candidate semantics properly."""
    # Blank/whitespace
    assert is_meaningful_translation("Hello", "") is False
    assert is_meaningful_translation("Hello", "   ") is False
    assert is_meaningful_translation("Hello", "<i></i>") is False
    assert is_meaningful_translation("Hello", None) is False

    # Normalized equivalents (Punctuation, casing, tags)
    assert is_meaningful_translation("Hello?", "Hello!") is False
    assert is_meaningful_translation("Hello", "<i>Hello.</i>") is False
    assert is_meaningful_translation("HELLO", "hello") is False
    assert is_meaningful_translation("Come on.", "Come on!") is False
    assert is_meaningful_translation("What?", "<i>What?</i>") is False

    # Valid translations
    assert is_meaningful_translation("Hello", "Hej") is True
    assert is_meaningful_translation("Come on", "Kom igen") is True
    assert is_meaningful_translation("What?", "Vad?") is True
    assert is_meaningful_translation("Room 101", "Rum 101") is True
