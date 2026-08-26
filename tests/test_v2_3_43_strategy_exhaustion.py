import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.translator import SubtitleTranslator, ProviderUnavailableError


def _mock_settings(key, default=""):
    settings = {
        "ai_provider": "gemini",
        "gemini_model": "gemini-3.5-flash-lite",
        "escalate_to_pro": "false",
        "escalation_provider": "none",
        "escalation_model": "",
    }
    return settings.get(key, default)


@pytest.mark.asyncio
async def test_1_semantic_identical_source_failure_marks_strategy_exhausted(monkeypatch):
    """1. Semantic identical-source failure -> exact strategy key is marked exhausted."""
    translator = SubtitleTranslator()
    exhausted = set()

    monkeypatch.setattr("app.services.translator.get_setting", _mock_settings)

    async def mock_exec(**kwargs):
        # AI returns source text verbatim
        return json.dumps({"translation": "Hello world"})

    monkeypatch.setattr(translator, "_execute_single_escalation_call", mock_exec)

    res = await translator.escalate_single_line(
        target_idx=10,
        target_text="Hello world",
        prev_text="Previous line",
        next_text="Next line",
        target_language="Swedish",
        show_title="Test Show",
        is_real_untranslated=True,
        exhausted_strategies=exhausted,
        source_language="English",
    )

    # Returns None when unverified and untranslated (fail-closed)
    assert res is None
    # All 3 attempted strategies (contextual, strict, isolated) are now exhausted
    assert len(exhausted) == 3
    assert any(":contextual:" in k for k in exhausted)
    assert any(":strict:" in k for k in exhausted)
    assert any(":isolated:" in k for k in exhausted)


@pytest.mark.asyncio
async def test_2_same_strategy_skipped_in_next_qa_loop(monkeypatch):
    """2. Same strategy in next QA-loop -> 0 new API calls made for exhausted strategies."""
    translator = SubtitleTranslator()
    exhausted = set()

    monkeypatch.setattr("app.services.translator.get_setting", _mock_settings)

    call_count = 0
    async def mock_exec(**kwargs):
        nonlocal call_count
        call_count += 1
        return json.dumps({"translation": "Hello world"})

    monkeypatch.setattr(translator, "_execute_single_escalation_call", mock_exec)

    # Loop 1: All 3 strategies run and fail (3 API calls)
    res1 = await translator.escalate_single_line(
        target_idx=10,
        target_text="Hello world",
        prev_text="Previous line",
        next_text="Next line",
        target_language="Swedish",
        show_title="Test Show",
        is_real_untranslated=True,
        exhausted_strategies=exhausted,
        source_language="English",
    )
    assert res1 is None
    assert call_count == 3
    assert len(exhausted) == 3

    # Loop 2: Same cue, same context -> All strategies already exhausted -> 0 new API calls
    res2 = await translator.escalate_single_line(
        target_idx=10,
        target_text="Hello world",
        prev_text="Previous line",
        next_text="Next line",
        target_language="Swedish",
        show_title="Test Show",
        is_real_untranslated=True,
        exhausted_strategies=exhausted,
        source_language="English",
    )
    assert res2 is None
    # Call count remains 3 (0 new calls)
    assert call_count == 3


@pytest.mark.asyncio
async def test_3_contextual_exhausted_still_allows_strict_to_run(monkeypatch):
    """3. Pre-exhausted contextual strategy -> strict and isolated still execute."""
    translator = SubtitleTranslator()
    exhausted = set()

    monkeypatch.setattr("app.services.translator.get_setting", _mock_settings)

    # Pre-populate exhausted with the contextual strategy key
    context_fingerprint = hash(("Previous line", "Hello world", "Next line", True))
    strategy_key_contextual = f"10:gemini:gemini-3.5-flash-lite:contextual:{context_fingerprint}"
    exhausted.add(strategy_key_contextual)

    executed_attempts = []
    async def mock_exec(system_prompt="", prompt="", **kwargs):
        if "strict translation engine" in system_prompt and "TARGET:" in prompt:
            executed_attempts.append("strict")
            return json.dumps({"translation": "Hej världen"})
        elif "strict translation engine" in system_prompt and "SOURCE:" in prompt:
            executed_attempts.append("isolated")
            return json.dumps({"translation": "Hej världen"})
        else:
            executed_attempts.append("contextual")
            return json.dumps({"translation": "Hello world"})

    monkeypatch.setattr(translator, "_execute_single_escalation_call", mock_exec)

    res = await translator.escalate_single_line(
        target_idx=10,
        target_text="Hello world",
        prev_text="Previous line",
        next_text="Next line",
        target_language="Swedish",
        show_title="Test Show",
        is_real_untranslated=True,
        exhausted_strategies=exhausted,
        source_language="English",
    )

    # Contextual was skipped, strict was executed and succeeded!
    assert "contextual" not in executed_attempts
    assert "strict" in executed_attempts
    assert res == "Hej världen"


@pytest.mark.asyncio
async def test_4_strict_exhausted_still_allows_isolated_to_run(monkeypatch):
    """4. Pre-exhausted contextual + strict strategies -> isolated still executes."""
    translator = SubtitleTranslator()
    exhausted = set()

    monkeypatch.setattr("app.services.translator.get_setting", _mock_settings)

    context_fingerprint = hash(("Previous line", "Hello world", "Next line", True))
    exhausted.add(f"10:gemini:gemini-3.5-flash-lite:contextual:{context_fingerprint}")
    exhausted.add(f"10:gemini:gemini-3.5-flash-lite:strict:{context_fingerprint}")

    executed_attempts = []
    async def mock_exec(system_prompt="", prompt="", **kwargs):
        if "strict translation engine" in system_prompt and "SOURCE:" in prompt:
            executed_attempts.append("isolated")
            return json.dumps({"translation": "Hej världen"})
        elif "strict translation engine" in system_prompt and "TARGET:" in prompt:
            executed_attempts.append("strict")
            return json.dumps({"translation": "Hello world"})
        else:
            executed_attempts.append("contextual")
            return json.dumps({"translation": "Hello world"})

    monkeypatch.setattr(translator, "_execute_single_escalation_call", mock_exec)

    res = await translator.escalate_single_line(
        target_idx=10,
        target_text="Hello world",
        prev_text="Previous line",
        next_text="Next line",
        target_language="Swedish",
        show_title="Test Show",
        is_real_untranslated=True,
        exhausted_strategies=exhausted,
        source_language="English",
    )

    assert "contextual" not in executed_attempts
    assert "strict" not in executed_attempts
    assert "isolated" in executed_attempts
    assert res == "Hej världen"


@pytest.mark.asyncio
async def test_5_transient_provider_exception_does_not_mark_strategy_exhausted(monkeypatch):
    """5. Transient provider exception (e.g. connection error) -> NOT added to exhausted_strategies."""
    translator = SubtitleTranslator()
    exhausted = set()

    monkeypatch.setattr("app.services.translator.get_setting", _mock_settings)

    async def mock_exec(**kwargs):
        raise ConnectionResetError("Connection reset by peer (transient network glitch)")

    monkeypatch.setattr(translator, "_execute_single_escalation_call", mock_exec)

    with pytest.raises(ProviderUnavailableError):
        await translator.escalate_single_line(
            target_idx=10,
            target_text="Hello world",
            prev_text="Previous line",
            next_text="Next line",
            target_language="Swedish",
            show_title="Test Show",
            is_real_untranslated=True,
            exhausted_strategies=exhausted,
            source_language="English",
        )

    # Strategy was NOT exhausted because failure was a transient network error, not a semantic rejection
    assert len(exhausted) == 0


@pytest.mark.asyncio
async def test_6_shared_exhausted_set_persists_across_multiple_cues(monkeypatch):
    """6. Shared exhausted set correctly records failures across concurrent cue escalations."""
    translator = SubtitleTranslator()
    exhausted = set()

    monkeypatch.setattr("app.services.translator.get_setting", _mock_settings)

    async def mock_exec(target_text="", **kwargs):
        if target_text == "Stubborn 1":
            return json.dumps({"translation": "Stubborn 1"})  # Semantic rejection
        elif target_text == "Stubborn 2":
            return json.dumps({"translation": "Stubborn 2"})  # Semantic rejection
        return json.dumps({"translation": "Översatt"})

    monkeypatch.setattr(translator, "_execute_single_escalation_call", mock_exec)

    tasks = [
        translator.escalate_single_line(
            target_idx=1, target_text="Stubborn 1", prev_text="", next_text="",
            target_language="Swedish", show_title="Test", is_real_untranslated=True,
            exhausted_strategies=exhausted
        ),
        translator.escalate_single_line(
            target_idx=2, target_text="Stubborn 2", prev_text="", next_text="",
            target_language="Swedish", show_title="Test", is_real_untranslated=True,
            exhausted_strategies=exhausted
        ),
        translator.escalate_single_line(
            target_idx=3, target_text="Normal dialogue", prev_text="", next_text="",
            target_language="Swedish", show_title="Test", is_real_untranslated=True,
            exhausted_strategies=exhausted
        ),
    ]

    results = await asyncio.gather(*tasks)

    assert results[0] is None
    assert results[1] is None
    assert results[2] == "Översatt"

    # Cues 1 and 2 each exhausted 3 strategies (total 6)
    # Cue 3 succeeded on attempt 1 so none of its strategies are in exhausted
    assert len(exhausted) == 6
    assert any(k.startswith("1:") for k in exhausted)
    assert any(k.startswith("2:") for k in exhausted)
    assert not any(k.startswith("3:") for k in exhausted)
