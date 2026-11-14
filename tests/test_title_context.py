from datetime import timedelta

import pytest
import srt

import app.core.db
import asyncio

from app.services.translator import SubtitleTranslator, get_system_instruction, merge_title_context, new_recurring_names


def test_title_context_follows_glossary_in_system_prompt():
    prompt = get_system_instruction(
        "French", glossary="Ziegler = Ziegler", title_context="Ziegler - male - Chief Constable",
    )

    glossary = prompt.index("GLOSSARY")
    context = prompt.index("TITLE CONTEXT")
    assert glossary < context < prompt.index("Ziegler - male")
    assert "The GLOSSARY wins any conflict" in prompt
    assert "TITLE CONTEXT" not in get_system_instruction("French")


@pytest.mark.asyncio
async def test_title_context_is_generated_once_and_replaces_memory(tmp_path, monkeypatch):
    monkeypatch.setattr("app.core.db.DB_PATH", str(tmp_path / "test.db"))
    app.core.db.init_db()
    app.core.db.set_setting("ai_provider", "openai")
    app.core.db.set_setting("openai_api_key", "test-key")
    app.core.db.save_translation_memory_bulk(
        "Ludwig", [{"original": "Hi", "translated": "Salut"}],
        source_language="en", target_language="fr",
    )
    subs = [
        srt.Subtitle(index=1, start=timedelta(0), end=timedelta(seconds=1), content="{\\an8}<i>Chief Constable Ziegler.</i>"),
    ]
    prompts = []
    contexts = []

    async def fake_dispatch(**kwargs):
        prompts.append(kwargs["user_prompt"])
        return "TERMS: Chief Constable = directeur de la police"

    async def fake_translate_batch(items, target_language, context_lines=None, **kwargs):
        contexts.append(context_lines or [])
        return [{"id": it["id"], "text": "Directeur Ziegler."} for it in items]

    translator = SubtitleTranslator()
    monkeypatch.setattr(translator, "_dispatch_llm_completion", fake_dispatch)
    monkeypatch.setattr(translator, "translate_batch", fake_translate_batch)
    for _ in range(2):
        await translator.translate_srt_content(
            subs, target_language="French", source_language="English", show_title="Ludwig",
        )

    assert prompts == ["DIALOGUE:\nChief Constable Ziegler."]
    assert app.core.db.get_title_context("Ludwig", "en", "fr") == "TERMS: Chief Constable = directeur de la police"
    assert all(c.get("role") != "memory" for ctx in contexts for c in ctx)


def _subs(*lines):
    return [srt.Subtitle(index=i + 1, start=timedelta(seconds=i), end=timedelta(seconds=i + 1), content=t) for i, t in enumerate(lines)]


def test_new_recurring_names_skips_known_and_sentence_start_words():
    subs = _subs(*["Ask Jacob's class.", "Tell Gregory.", "See you on Monday.", "Jacob! Courtney, hi.", "Our Club rules.", "a club"] * 3)

    assert new_recurring_names(subs, "Gregory - male - assistant principal") == ["Jacob"]


def test_merge_appends_under_headings_and_keeps_existing_lines():
    existing = "CHARACTERS:\nGregory - male - teacher\n\nADDRESS:\nColleagues use tu.\n\nTERMS:\nprincipal = directrice\n\nNOTED (no entry needed): Club"
    merged = merge_title_context(existing, "CHARACTERS:\nJacob - male - teacher\nGregory - male - teacher\nTERMS:\nSvalbard = Svalbard")

    assert merged == existing.replace("Gregory - male - teacher", "Gregory - male - teacher\nJacob - male - teacher").replace(
        "directrice", "directrice\nSvalbard = Svalbard")
    assert merge_title_context(existing, "NONE") == existing
    assert merge_title_context(existing, "CHARACTERS:\nSidwell - unknown - Jacob's friend", names=["Jacob"]) == existing


@pytest.mark.asyncio
async def test_parallel_episodes_extend_the_title_context_once(tmp_path, monkeypatch):
    monkeypatch.setattr("app.core.db.DB_PATH", str(tmp_path / "test.db"))
    app.core.db.init_db()
    app.core.db.set_setting("ai_provider", "openai")
    app.core.db.set_setting("openai_api_key", "test-key")
    app.core.db.set_title_context("Abbott", "en", "fr", "CHARACTERS:\nGregory - male - teacher")
    subs = _subs(*["Ask Jacob.", "Fly to Svalbard."] * 3)
    calls = []

    async def fake_dispatch(**kwargs):
        calls.append(kwargs["system_prompt"])
        await asyncio.sleep(0)
        return "CHARACTERS:\nJacob - male - teacher"

    translator = SubtitleTranslator()
    monkeypatch.setattr(translator, "_dispatch_llm_completion", fake_dispatch)
    await asyncio.gather(*(translator.ensure_title_context(subs, "Abbott", "en", "fr") for _ in range(2)))
    await translator.ensure_title_context(subs, "Abbott", "en", "fr")

    assert len(calls) == 1 and "these names from the new dialogue: Jacob, Svalbard." in calls[0]
    assert app.core.db.get_title_context("Abbott", "en", "fr") == (
        "CHARACTERS:\nGregory - male - teacher\nJacob - male - teacher\n\nNOTED (no entry needed): Svalbard")
