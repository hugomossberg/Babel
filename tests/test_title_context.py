from datetime import timedelta

import pytest
import srt

import app.core.db
from app.services.translator import SubtitleTranslator, get_system_instruction


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
