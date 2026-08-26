import pytest
from datetime import timedelta
import srt
from app.core.cleaner import (
    clean_subtitle_text,
    sanitize_srt_content,
    sanitize_srt_content_with_provenance,
    analyze_subtitle_cue,
    parse_structural_segments,
    classify_structural_segment_deterministic,
    extract_file_speaker_labels,
    subs_to_srt_string,
    EMPTY_PLACEHOLDER,
    StructuralType,
    SegmentClassification,
    ClassificationProvenance,
)
from app.services.translator import SubtitleTranslator


# ===========================================================================
# 1. STRUCTURAL SEGMENTATION TESTS
# ===========================================================================

def test_01_parse_bracketed_and_plain_text():
    raw = "[door closes] Where are you?"
    segs = parse_structural_segments(raw)
    assert len(segs) == 2
    assert segs[0].structural_type == StructuralType.BRACKETED
    assert segs[0].inner == "door closes"
    assert segs[1].structural_type == StructuralType.PLAIN_TEXT
    assert segs[1].inner == "Where are you?"


def test_02_parse_parenthesized_and_plain_text():
    raw = "(sighs) I cannot believe this."
    segs = parse_structural_segments(raw)
    assert len(segs) == 2
    assert segs[0].structural_type == StructuralType.PARENTHESIZED
    assert segs[0].inner == "sighs"
    assert segs[1].structural_type == StructuralType.PLAIN_TEXT


def test_03_parse_music_notes_tokens():
    raw = "♪ Never gonna give you up ♪"
    segs = parse_structural_segments(raw)
    assert any(s.structural_type == StructuralType.MUSIC_NOTES for s in segs)
    assert any(s.structural_type == StructuralType.PLAIN_TEXT and "Never gonna give you up" in s.inner for s in segs)


def test_04_parse_unicode_fullwidth_brackets():
    raw = "（ため息）どこにいるの？"
    segs = parse_structural_segments(raw)
    assert segs[0].structural_type == StructuralType.PARENTHESIZED
    assert segs[0].inner == "ため息"
    assert segs[1].structural_type == StructuralType.PLAIN_TEXT


def test_05_parse_speaker_prefix():
    raw = "OFFICER: Step out of the vehicle."
    segs = parse_structural_segments(raw)
    assert segs[0].structural_type == StructuralType.SPEAKER_PREFIX
    assert segs[0].inner == "OFFICER"
    assert segs[1].structural_type == StructuralType.PLAIN_TEXT


# ===========================================================================
# 2. DETERMINISTIC HEURISTICS & PRESERVATION TESTS
# ===========================================================================

def test_06_pure_music_notes_becomes_placeholder():
    assert clean_subtitle_text("♪♪♪") == EMPTY_PLACEHOLDER
    assert clean_subtitle_text("♬ ♫ ♪") == EMPTY_PLACEHOLDER


def test_07_bracketed_dialogue_sentence_preserved():
    assert clean_subtitle_text("[I know what you did.]") == "[I know what you did.]"
    assert clean_subtitle_text("[Take cover!]") == "[Take cover!]"
    assert clean_subtitle_text('[“Who is there?”]') == '[“Who is there?”]'


def test_08_parenthesized_dialogue_preserved():
    assert clean_subtitle_text("(Come here)") == "(Come here)"
    assert clean_subtitle_text("(Please wait)") == "(Please wait)"
    assert clean_subtitle_text("(I mean it.)") == "(I mean it.)"
    assert clean_subtitle_text("(Don't do that.)") == "(Don't do that.)"


def test_09_inline_parenthetical_in_sentence_preserved():
    assert clean_subtitle_text("This is (very) nice.") == "This is (very) nice."
    assert clean_subtitle_text("I told him (and I mean this) to stop.") == "I told him (and I mean this) to stop."


def test_10_file_level_speaker_labels_detected():
    srt_text = """1
00:00:01,000 --> 00:00:02,000
JOHN: Hello there.

2
00:00:03,000 --> 00:00:04,000
MARY: Good morning!

3
00:00:05,000 --> 00:00:06,000
JOHN: Where are we going?
"""
    subs = list(srt.parse(srt_text))
    labels = extract_file_speaker_labels(subs)
    assert "JOHN" in labels
    assert "MARY" not in labels  # only 1 occurrence


# ===========================================================================
# 3. MULTILINGUAL NON-LATIN & LANGUAGE-AGNOSTIC CUE CLEANING
# ===========================================================================

def test_11_mixed_cue_english():
    raw = "[door closes] Where are you?"
    cleaned = clean_subtitle_text(raw)
    assert cleaned == "Where are you?"


def test_12_mixed_cue_german():
    raw = "[Tür schließt] Wo bist du?"
    cleaned = clean_subtitle_text(raw)
    assert cleaned == "Wo bist du?"


def test_13_mixed_cue_french():
    raw = "[applaudissements] Merci à tous."
    cleaned = clean_subtitle_text(raw)
    assert cleaned == "Merci à tous."


def test_14_mixed_cue_spanish():
    raw = "[se cierra la puerta] ¿Dónde estás?"
    cleaned = clean_subtitle_text(raw)
    assert cleaned == "¿Dónde estás?"


def test_15_mixed_cue_polish():
    raw = "[drzwi zamykają się] Gdzie jesteś?"
    cleaned = clean_subtitle_text(raw)
    assert cleaned == "Gdzie jesteś?"


def test_16_mixed_cue_arabic():
    raw = "[يغلق الباب] أين أنت؟"
    cleaned = clean_subtitle_text(raw)
    assert cleaned == "أين أنت؟"


def test_17_mixed_cue_japanese():
    raw = "［ため息］どこにいるの？"
    cleaned = clean_subtitle_text(raw)
    assert cleaned == "どこにいるの？"


def test_18_mixed_cue_russian():
    raw = "[дверь закрывается] Где ты?"
    cleaned = clean_subtitle_text(raw)
    assert cleaned == "Где ты?"


def test_19_mixed_cue_korean():
    raw = "[문이 닫힌다] 어디 있어?"
    cleaned = clean_subtitle_text(raw)
    assert cleaned == "어디 있어?"


def test_20_pure_sdh_multilingual_becomes_placeholder():
    assert clean_subtitle_text("[Tür schließt]") == EMPTY_PLACEHOLDER
    assert clean_subtitle_text("[applaudissements]") == EMPTY_PLACEHOLDER
    assert clean_subtitle_text("[se cierra la puerta]") == EMPTY_PLACEHOLDER
    assert clean_subtitle_text("[drzwi zamykają się]") == EMPTY_PLACEHOLDER
    assert clean_subtitle_text("[يغلق الباب]") == EMPTY_PLACEHOLDER
    assert clean_subtitle_text("［ため息］") == EMPTY_PLACEHOLDER


# ===========================================================================
# 4. ASYNC SANITIZATION & SEMANTIC CLASSIFICATION WITH PROVENANCE
# ===========================================================================

@pytest.mark.asyncio
async def test_21_async_sanitize_with_mock_classifier():
    srt_text = """1
00:00:01,000 --> 00:00:03,000
[chuckles softly] Hey there.

2
00:00:04,000 --> 00:00:06,000
[I know what you did.]

3
00:00:07,000 --> 00:00:09,000
[dramatic music]
"""
    async def mock_classifier(items, source_language="English", job_id=None):
        results = []
        for item in items:
            t = item["text"]
            if "chuckles" in t or "music" in t:
                results.append({"id": item["id"], "classification": "NON_DIALOGUE"})
            else:
                results.append({"id": item["id"], "classification": "DIALOGUE"})
        return results

    subs, prov_map, cleaned_count = await sanitize_srt_content_with_provenance(
        srt_text,
        source_language="English",
        classifier_fn=mock_classifier
    )

    assert len(subs) == 3
    assert subs[0].content == "Hey there."
    assert prov_map[0].has_bracketed_dialogue is False
    assert "[chuckles softly]" in prov_map[0].removed_sdh_segments

    assert subs[1].content == "[I know what you did.]"
    assert prov_map[1].has_bracketed_dialogue is True

    assert subs[2].content == EMPTY_PLACEHOLDER
    assert prov_map[2].is_sdh_only is True


@pytest.mark.asyncio
async def test_22_classifier_fail_safe_preserves_dialogue():
    srt_text = """1
00:00:01,000 --> 00:00:03,000
(whispering) Don't look back.
"""
    async def broken_classifier(items, source_language="English", job_id=None):
        raise RuntimeError("AI Provider Network Timeout")

    subs, prov_map, cleaned_count = await sanitize_srt_content_with_provenance(
        srt_text,
        source_language="English",
        classifier_fn=broken_classifier
    )

    assert len(subs) == 1
    # Dialogue must NEVER be dropped on classifier failure
    assert "Don't look back." in subs[0].content


@pytest.mark.asyncio
async def test_23_classifier_uncertain_preserves_dialogue():
    srt_text = """1
00:00:01,000 --> 00:00:03,000
Doctor (quietly): Please hurry.
"""
    async def uncertain_classifier(items, source_language="English", job_id=None):
        return [{"id": items[0]["id"], "classification": "UNCERTAIN"}]

    subs, prov_map, cleaned_count = await sanitize_srt_content_with_provenance(
        srt_text,
        source_language="English",
        classifier_fn=uncertain_classifier
    )

    assert len(subs) == 1
    assert subs[0].content == "Doctor (quietly): Please hurry."
    assert prov_map[0].segments[0].provenance == ClassificationProvenance.FAIL_SAFE_PRESERVED


# ===========================================================================
# 5. PROVENANCE-DRIVEN TARGET GUARD IN TRANSLATOR
# ===========================================================================

@pytest.mark.asyncio
async def test_24_target_guard_strips_hallucinated_brackets_when_source_had_no_brackets(monkeypatch):
    from app.core.cleaner import CueProvenance
    translator = SubtitleTranslator()

    source_subs = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=3), "Where are you?")
    ]
    prov_map = {
        0: CueProvenance(
            index=1,
            raw_content="[door closes] Where are you?",
            cleaned_content="Where are you?",
            has_bracketed_dialogue=False,
            has_parenthesized_dialogue=False,
            removed_sdh_segments=["[door closes]"],
            is_sdh_only=False
        )
    }

    async def mock_translate_batch(items, target_language="Swedish", source_language="English", **kwargs):
        # AI returned a hallucinated bracketed SDH tag in Swedish
        return [{"id": 0, "text": "[skrattar till] Var är du?"}]

    monkeypatch.setattr(translator, "translate_batch", mock_translate_batch)

    translated = await translator.translate_srt_content(
        source_subs,
        target_language="Swedish",
        source_language="English",
        batch_size=50,
        provenance_map=prov_map
    )

    assert len(translated) == 1
    # Hallucinated bracket stripped because source had no bracketed dialogue!
    assert translated[0].content == "Var är du?"


@pytest.mark.asyncio
async def test_25_target_guard_preserves_legitimate_bracketed_dialogue(monkeypatch):
    from app.core.cleaner import CueProvenance
    translator = SubtitleTranslator()

    source_subs = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=3), "[I know what you did.]")
    ]
    prov_map = {
        0: CueProvenance(
            index=1,
            raw_content="[I know what you did.]",
            cleaned_content="[I know what you did.]",
            has_bracketed_dialogue=True,
            has_parenthesized_dialogue=False,
            removed_sdh_segments=[],
            is_sdh_only=False
        )
    }

    async def mock_translate_batch(items, target_language="Swedish", source_language="English", **kwargs):
        return [{"id": 0, "text": "[Jag vet vad du gjorde.]"}]

    monkeypatch.setattr(translator, "translate_batch", mock_translate_batch)

    translated = await translator.translate_srt_content(
        source_subs,
        target_language="Swedish",
        source_language="English",
        batch_size=50,
        provenance_map=prov_map
    )

    assert len(translated) == 1
    # Real bracketed dialogue kept because source had bracketed dialogue!
    assert translated[0].content == "[Jag vet vad du gjorde.]"


# ===========================================================================
# 6. SRT INTEGRITY & BATCH BOUNDS TESTS
# ===========================================================================

def test_26_subs_to_srt_string_sequential_index_integrity():
    subs = [
        srt.Subtitle(10, timedelta(seconds=1), timedelta(seconds=2), "First"),
        srt.Subtitle(25, timedelta(seconds=3), timedelta(seconds=4), EMPTY_PLACEHOLDER),
        srt.Subtitle(99, timedelta(seconds=5), timedelta(seconds=6), "Third"),
    ]
    srt_str = subs_to_srt_string(subs)
    parsed = list(srt.parse(srt_str))
    assert len(parsed) == 3
    assert [s.index for s in parsed] == [1, 2, 3]
    assert parsed[1].content == EMPTY_PLACEHOLDER


def test_27_no_language_wordlists_in_cleaner():
    import app.core.cleaner as cleaner_mod
    # Verify no static sound lists exist in cleaner module
    forbidden_words = ["applåder", "skratt", "viskning", "applaudissements", "lachen", "aplausos", "dörr"]
    mod_source = open(cleaner_mod.__file__, "r", encoding="utf-8").read()
    for word in forbidden_words:
        assert word not in mod_source, f"Forbidden static word '{word}' found in cleaner.py"


@pytest.mark.asyncio
async def test_28_bounded_batch_size_under_50():
    # Verify that classifier batches never exceed 50 items
    batch_sizes = []

    async def mock_classifier(chunk, source_language="English", job_id=None):
        batch_sizes.append(len(chunk))
        return [{"id": it["id"], "classification": "NON_DIALOGUE"} for it in chunk]

    srt_lines = []
    for i in range(125):
        srt_lines.append(f"{i+1}\n00:{i:02d}:00,000 --> 00:{i:02d}:01,000\nDoctor (quietly {i}): Please hurry.")
    srt_data = "\n\n".join(srt_lines)

    subs, prov_map, _ = await sanitize_srt_content_with_provenance(
        srt_data,
        source_language="English",
        classifier_fn=mock_classifier
    )

    assert all(bs <= 50 for bs in batch_sizes)
    assert batch_sizes == [50, 50, 25]


def test_29_analyze_subtitle_cue_provenance_structure():
    cue = analyze_subtitle_cue("MAN: [screams] Get down!", index=5)
    assert cue.index == 5
    assert len(cue.segments) >= 2
    assert cue.has_bracketed_dialogue is False
    assert "MAN:" in [s.raw for s in cue.segments]
    assert "Get down!" in cue.cleaned_content


def test_30_full_synchronous_sanitize_srt_content_regression():
    srt_data = """1
00:00:01,000 --> 00:00:03,000
[door slams]

2
00:00:04,000 --> 00:00:06,000
Hey, how are you? [laughs]

3
00:00:07,000 --> 00:00:09,000
♪♪♪

4
00:00:10,000 --> 00:00:12,000
(I mean it.)
"""
    subs, count = sanitize_srt_content(srt_data)
    assert len(subs) == 4
    assert subs[0].content == EMPTY_PLACEHOLDER
    assert subs[1].content == "Hey, how are you?"
    assert subs[2].content == EMPTY_PLACEHOLDER
    assert subs[3].content == "(I mean it.)"


# ===========================================================================
# 7. SHORT DIALOGUE & LANGUAGE-AGNOSTIC PRESERVATION REGRESSION TESTS
# ===========================================================================

def test_31_short_parenthetical_dialogue_preserved_english():
    # English short whisper/offscreen dialogue must NEVER be silently stripped
    assert clean_subtitle_text("(yes)") == "(yes)"
    assert clean_subtitle_text("(no)") == "(no)"
    assert clean_subtitle_text("(help)") == "(help)"
    assert clean_subtitle_text("(wait)") == "(wait)"
    assert clean_subtitle_text("(come here)") == "(come here)"
    assert clean_subtitle_text("(don't move)") == "(don't move)"


def test_32_short_parenthetical_dialogue_preserved_german():
    # German short whisper/offscreen dialogue must NEVER be silently stripped
    assert clean_subtitle_text("(ja)") == "(ja)"
    assert clean_subtitle_text("(nein)") == "(nein)"
    assert clean_subtitle_text("(hilfe)") == "(hilfe)"
    assert clean_subtitle_text("(warte)") == "(warte)"
    assert clean_subtitle_text("(komm her)") == "(komm her)"


def test_33_short_parenthetical_dialogue_preserved_french():
    # French short whisper/offscreen dialogue must NEVER be silently stripped
    assert clean_subtitle_text("(oui)") == "(oui)"
    assert clean_subtitle_text("(non)") == "(non)"
    assert clean_subtitle_text("(aidez-moi)") == "(aidez-moi)"
    assert clean_subtitle_text("(attends)") == "(attends)"
    assert clean_subtitle_text("(viens ici)") == "(viens ici)"


def test_34_no_music_word_decision_tokens_in_cleaner():
    # Verify 'music' is not used as a hardcoded decision word/string in cleaner
    import app.core.cleaner as cleaner_mod
    mod_source = open(cleaner_mod.__file__, "r", encoding="utf-8").read()
    assert '== "music"' not in mod_source, "Found '== \"music\"' in cleaner.py"
    assert "=='music'" not in mod_source, "Found '=='music'' in cleaner.py"
    assert '=="music"' not in mod_source, "Found '==\"music\"' in cleaner.py"


def test_35_no_monkeypatch_or_test_awareness_in_pipeline():
    # Verify production pipeline has no mock/monkeypatch detection
    import app.services.pipeline as pipeline_mod
    pipeline_source = open(pipeline_mod.__file__, "r", encoding="utf-8").read()
    assert "sanitize_srt_content !=" not in pipeline_source, "Found monkeypatch detection in pipeline.py"
    assert "_cleaner_mod" not in pipeline_source, "Found _cleaner_mod in pipeline.py"


@pytest.mark.asyncio
async def test_36_target_guard_strips_excess_hallucinated_brackets_in_same_cue(monkeypatch):
    # When source had 1 legitimate bracketed dialogue, but target hallucinates an EXTRA bracketed SDH tag
    from app.core.cleaner import CueProvenance
    translator = SubtitleTranslator()

    source_subs = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=3), "[Secret Agent]")
    ]
    prov_map = {
        0: CueProvenance(
            index=1,
            raw_content="[Secret Agent]",
            cleaned_content="[Secret Agent]",
            has_bracketed_dialogue=True,
            has_parenthesized_dialogue=False,
            bracketed_dialogue_inners=["Secret Agent"],
            parenthesized_dialogue_inners=[],
            removed_sdh_segments=[],
            is_sdh_only=False
        )
    }

    async def mock_translate_batch(items, target_language="Swedish", source_language="English", **kwargs):
        # Target produced the valid translation PLUS an extra hallucinated SDH sound effect tag in brackets
        return [{"id": 0, "text": "[Hemlig agent] [skrattar till]"}]

    monkeypatch.setattr(translator, "translate_batch", mock_translate_batch)

    translated = await translator.translate_srt_content(
        source_subs,
        target_language="Swedish",
        source_language="English",
        batch_size=50,
        provenance_map=prov_map
    )

    assert len(translated) == 1
    # Excess hallucinated bracket stripped, legitimate bracket preserved
    assert translated[0].content == "[Hemlig agent]"

