import pytest
from app.services.pipeline import qa_gate
import srt
from datetime import timedelta

def test_dropped_lines_in_qa_gate():
    source_subs = [
        srt.Subtitle(1, timedelta(seconds=0), timedelta(seconds=1), "Line 1"),
        srt.Subtitle(2, timedelta(seconds=1), timedelta(seconds=2), "Line 2"),
        srt.Subtitle(3, timedelta(seconds=2), timedelta(seconds=3), "Line 3")
    ]
    translated_subs = [
        srt.Subtitle(1, timedelta(seconds=0), timedelta(seconds=1), "Linje 1"),
        srt.Subtitle(2, timedelta(seconds=1), timedelta(seconds=2), ""),
        srt.Subtitle(3, timedelta(seconds=2), timedelta(seconds=3), "Linje 3")
    ]
    
    result = qa_gate(source_subs, translated_subs, "sv")
    assert result["dropped_count"] == 1
    assert result["passed"] is False

def test_wrong_language_in_qa_gate():
    source_subs = [
        srt.Subtitle(1, timedelta(seconds=0), timedelta(seconds=1), "Line 1"),
        srt.Subtitle(2, timedelta(seconds=1), timedelta(seconds=2), "Line 2")
    ]
    translated_subs = [
        srt.Subtitle(1, timedelta(seconds=0), timedelta(seconds=1), "This is English"),
        srt.Subtitle(2, timedelta(seconds=1), timedelta(seconds=2), "Also English")
    ]
    
    result = qa_gate(source_subs, translated_subs, "sv")
    assert result["passed"] is False
    assert any("language" in i.lower() for i in result["issues"])
