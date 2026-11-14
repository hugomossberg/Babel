from datetime import timedelta

import pytest
import srt

import app.core.db
from app.core.cleaner import analyze_subtitle_cue
from app.services.translator import SubtitleTranslator, format_context_section


def test_context_sections_are_labelled_by_role():
    section = format_context_section([
        {"original": "Hi", "translated": "Salut", "role": "memory"},
        {"original": "Before", "translated": "Avant"},
        {"original": "After", "role": "after"},
    ])

    memory = section.index("SERIES TRANSLATION MEMORY")
    before = section.index("PRECEDING DIALOGUE")
    after = section.index("FOLLOWING DIALOGUE")
    assert memory < section.index('"Hi"') < before
    assert before < section.index('"Before" → Translated: "Avant"') < after
    assert after < section.index('"After"')
    assert format_context_section(None) == ""


@pytest.mark.asyncio
async def test_batches_get_memory_preceding_and_following_context(tmp_path, monkeypatch):
    monkeypatch.setattr("app.core.db.DB_PATH", str(tmp_path / "test.db"))
    app.core.db.init_db()
    app.core.db.save_translation_memory_bulk(
        "Show", [{"original": "Hi", "translated": "Salut"}],
        source_language="en", target_language="fr",
    )
    subs = [
        srt.Subtitle(index=i + 1, start=timedelta(seconds=i), end=timedelta(seconds=i + 1), content=text)
        for i, text in enumerate(["One", "Two", "Three", "Four"])
    ]
    contexts = {}

    async def fake_translate_batch(items, target_language, context_lines=None, **kwargs):
        contexts[items[0]["id"]] = context_lines or []
        return [{"id": it["id"], "text": f"fr {it['text']}"} for it in items]

    translator = SubtitleTranslator()
    monkeypatch.setattr(translator, "translate_batch", fake_translate_batch)
    await translator.translate_srt_content(
        subs, target_language="French", source_language="English", batch_size=2, show_title="Show",
    )

    def by_role(ctx, role):
        return [c["original"] for c in ctx if c.get("role") == role]

    assert by_role(contexts[0], "memory") == ["Hi"]
    assert by_role(contexts[0], "before") == []
    assert by_role(contexts[0], "after") == ["Three", "Four"]
    assert by_role(contexts[2], "before") == ["One", "Two"]
    assert by_role(contexts[2], "after") == []


@pytest.mark.asyncio
async def test_removed_speaker_labels_reach_the_model_as_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr("app.core.db.DB_PATH", str(tmp_path / "test.db"))
    app.core.db.init_db()
    raw = ["- Lucy: Hello.\n- Tom: Hi.", "RUSSELL: Go.", "Plain line."]
    provenance = {i: analyze_subtitle_cue(text, index=i + 1) for i, text in enumerate(raw)}
    subs = [
        srt.Subtitle(index=i + 1, start=timedelta(seconds=i), end=timedelta(seconds=i + 1),
                     content=provenance[i].cleaned_content)
        for i in range(len(raw))
    ]
    sent = []

    async def fake_translate_batch(items, target_language, context_lines=None, **kwargs):
        sent.extend(items)
        return [{"id": it["id"], "text": f"fr {it['text']}"} for it in items]

    translator = SubtitleTranslator()
    monkeypatch.setattr(translator, "translate_batch", fake_translate_batch)
    await translator.translate_srt_content(
        subs, target_language="French", source_language="English", provenance_map=provenance,
    )

    assert [it.get("speaker") for it in sent] == ["Lucy / Tom", "RUSSELL", None]
    assert "Lucy" not in sent[0]["text"]
