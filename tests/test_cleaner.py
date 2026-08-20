import pytest
from app.core.cleaner import clean_subtitle_text, sanitize_srt_content, EMPTY_PLACEHOLDER

def test_clean_sdh_brackets():
    raw = "[door opens] Hello John! (sighs)"
    cleaned = clean_subtitle_text(raw)
    assert cleaned == "Hello John!"

def test_clean_music_notes():
    raw = "♪ Never gonna give you up ♪"
    cleaned = clean_subtitle_text(raw)
    assert cleaned == "Never gonna give you up"

def test_clean_pure_sdh_becomes_placeholder():
    raw = "[Dramatic music playing]"
    cleaned = clean_subtitle_text(raw)
    assert cleaned == EMPTY_PLACEHOLDER

def test_clean_pure_notes_becomes_placeholder():
    raw = "♪♪♪"
    cleaned = clean_subtitle_text(raw)
    assert cleaned == EMPTY_PLACEHOLDER

def test_sanitize_srt_preserves_structure():
    srt_data = """1
00:00:01,000 --> 00:00:03,000
[door slams]

2
00:00:04,000 --> 00:00:06,000
Hey, how are you? (laughs)
"""
    subs, count = sanitize_srt_content(srt_data)
    assert len(subs) == 2
    assert subs[0].content == EMPTY_PLACEHOLDER
    assert subs[1].content == "Hey, how are you?"
    assert count == 2

def test_cleaner_keeps_real_dialogue_parentheses():
    from app.core.cleaner import clean_subtitle_text
    assert clean_subtitle_text("(I mean it.)") == "(I mean it.)"
    assert clean_subtitle_text("(Don't do that.)") == "(Don't do that.)"
    assert clean_subtitle_text("This is (very) nice.") == "This is (very) nice."
    
def test_cleaner_removes_known_sdh():
    from app.core.cleaner import clean_subtitle_text
    assert clean_subtitle_text("(laughing)") == "<i></i>"
    assert clean_subtitle_text("(door closes)") == "<i></i>"
    assert clean_subtitle_text("(phone ringing)") == "<i></i>"
    assert clean_subtitle_text("[door closes]") == "<i></i>"
