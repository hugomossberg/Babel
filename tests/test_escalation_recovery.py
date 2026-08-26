import pytest
import json
import srt
from datetime import timedelta
from unittest.mock import patch, AsyncMock
from app.services.translator import SubtitleTranslator
from app.services.pipeline import SubtitlePipeline

@pytest.mark.asyncio
async def test_escalation_fail_closed_returns_none():
    translator = SubtitleTranslator()

    with patch("app.services.translator.get_setting") as mock_setting:
        mock_setting.side_effect = lambda k, default=None: "gemini" if k == "ai_provider" else default

        with patch("google.genai.Client") as mock_client:
            # A & B: Blank or whitespace returns None
            mock_client.return_value.models.generate_content.return_value.text = '{"translation": "   "}'
            res = await translator.escalate_single_line(1, "Hello", "", "", "Swedish", "Show")
            assert res is None

            # C: Unverified identical source candidate returns None (fails closed)
            mock_client.return_value.models.generate_content.return_value.text = '{"translation": "Hello"}'
            res = await translator.escalate_single_line(1, "Hello", "", "", "Swedish", "Show")
            assert res is None

            # D: Valid target translation returns translated text
            mock_client.return_value.models.generate_content.return_value.text = '{"translation": "Hej"}'
            res = await translator.escalate_single_line(1, "Hello", "", "", "Swedish", "Show")
            assert res == "Hej"

@pytest.mark.asyncio
async def test_escalation_hard_translate_prompt():
    translator = SubtitleTranslator()

    with patch("app.services.translator.get_setting") as mock_setting:
        mock_setting.side_effect = lambda k, default=None: "gemini" if k == "ai_provider" else default

        with patch("google.genai.Client") as mock_client:
            mock_client.return_value.models.generate_content.return_value.text = '{"translation": "Hej"}'

            # E: real_untranslated recovery
            res = await translator.escalate_single_line(1, "Come on, man.", "", "", "Swedish", "Show", is_real_untranslated=True)

            call_kwargs = mock_client.return_value.models.generate_content.call_args.kwargs
            config = call_kwargs.get("config")

            assert "TARGET is known to still be untranslated source-language dialogue" in config.system_instruction
            assert "Do NOT return the original source-language text" in config.system_instruction
            assert res == "Hej"

@pytest.mark.asyncio
async def test_safe_ids_reuse_in_recovery():
    """Verify full loop: classifier -> deterministic validation -> safe_ids populating -> subsequent recovery iteration -> final QA pass."""
    from app.services.translator import is_deterministically_safe_keep, validate_classifier_output

    # 1. Deterministic checks
    assert is_deterministically_safe_keep("NASA", "acronym") is True
    assert is_deterministically_safe_keep("Hello", "proper_noun") is False
    assert is_deterministically_safe_keep("Come here", "non_verbal") is False
    assert is_deterministically_safe_keep("[SIGHING]", "non_verbal") is False
    assert is_deterministically_safe_keep("♪ ♪", "non_verbal") is True

    pipeline = SubtitlePipeline()
    source_subs = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "911"),
        srt.Subtitle(2, timedelta(seconds=2), timedelta(seconds=3), "Hello"),
    ]
    # Initially, both lines are still unchanged source English
    translated_subs = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "911"),
        srt.Subtitle(2, timedelta(seconds=2), timedelta(seconds=3), "Hello"),
    ]

    # 2. Pipeline classifier step: simulate AI classifier classifying both candidate lines
    classifier_raw_response = json.dumps({
        "results": [
            {"id": 1, "action": "keep", "reason": "number"},
            {"id": 2, "action": "keep", "reason": "proper_noun"}  # Invalid keep: "Hello" is an English common word!
        ]
    })
    classifier_items = [
        {"id": 1, "text": "911"},
        {"id": 2, "text": "Hello"},
    ]
    validated_results = validate_classifier_output(classifier_raw_response, classifier_items)

    # Verify validation: id 1 kept, id 2 downgraded to translate
    safe_ids = set()
    for res in validated_results:
        # map 1-based cue id back to 0-based index
        idx = res["id"] - 1
        if res["action"] == "keep":
            safe_ids.add(idx)
        else:
            assert res["action"] == "translate"
            assert res["text"] == ""  # blanked out to force translation

    assert safe_ids == {0}  # Only 911 (idx 0) was deterministically validated as safe

    # 3. QA gate with safe_ids populated: 911 is accepted, Hello is identified as real untranslated
    qa_res = pipeline.qa_gate(source_subs, translated_subs, "sv", safe_ids=safe_ids)
    assert 0 not in qa_res["real_untranslated_ids"]
    assert 1 in qa_res["real_untranslated_ids"]
    assert qa_res["passed"] is False

    # 4. Next recovery iteration: only line 1 ("Hello") is dispatched to escalation translator
    with patch("app.services.translator.get_setting") as mock_setting:
        mock_setting.side_effect = lambda k, default=None: "gemini" if k == "ai_provider" else default
        with patch("google.genai.Client") as mock_client:
            mock_client.return_value.models.generate_content.return_value.text = '{"translation": "Hej"}'
            recovered_text = await pipeline.translator.escalate_single_line(
                2, source_subs[1].content, translated_subs[1].content, "", "Swedish", "Show", is_real_untranslated=True
            )
            assert recovered_text == "Hej"
            translated_subs[1].content = recovered_text

    # 5. Final QA Gate pass
    final_qa = pipeline.qa_gate(source_subs, translated_subs, "sv", safe_ids=safe_ids)
    assert final_qa["passed"] is True
    assert final_qa["dropped_count"] == 0
    assert len(final_qa["real_untranslated_ids"]) == 0
    assert final_qa["sync_diff_ms"] == 0

@pytest.mark.asyncio
async def test_pipeline_tm_publish_vs_qa_fail(tmp_path, monkeypatch):
    """Verify TM is saved only on successful publish with QA pass, and not saved when QA fails."""
    from app.core.db import DB_PATH, init_db, get_translation_memory, save_translation_memory_bulk
    test_db = str(tmp_path / "test_tm_pipeline.db")
    monkeypatch.setattr("app.core.db.DB_PATH", test_db)
    init_db()

    show_title = "Breaking Bad"

    # Scenario A: QA fails (e.g. dropped line) -> TM must not be saved
    pipeline = SubtitlePipeline()
    source_subs = [srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Good morning")]
    bad_trans = [srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "<i></i>")]
    qa_res = pipeline.qa_gate(source_subs, bad_trans, "sv")
    assert qa_res["passed"] is False

    # Simulate pipeline logic when QA fails
    if qa_res["passed"]:
        save_translation_memory_bulk(show_title, [{"original": "Good morning", "translated": "<i></i>"}])

    tm_before = get_translation_memory(show_title)
    assert len(tm_before) == 0

    # Scenario B: QA passes and published -> TM is saved
    good_trans = [srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "God morgon")]
    qa_good = pipeline.qa_gate(source_subs, good_trans, "sv")
    assert qa_good["passed"] is True

    # Simulate pipeline logic when QA passes and publish succeeds
    tm_items = []
    for idx in range(len(source_subs)):
        orig_t = source_subs[idx].content.strip()
        trans_t = good_trans[idx].content.strip()
        if orig_t and trans_t and orig_t != "<i></i>" and trans_t != "<i></i>":
            tm_items.append({"original": orig_t, "translated": trans_t})
    save_translation_memory_bulk(show_title, tm_items)

    tm_after = get_translation_memory(show_title)
    assert len(tm_after) == 1
    assert tm_after[0]["original"] == "Good morning"
    assert tm_after[0]["translated"] == "God morgon"

