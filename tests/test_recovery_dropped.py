import pytest
from app.services.pipeline import qa_gate
from app.core.validator import check_dropped_lines
import srt
from datetime import timedelta

def test_dropped_lines_returned_in_qa_gate():
    source_subs = [
        srt.Subtitle(1, timedelta(seconds=0), timedelta(seconds=1), "Line 1"),
        srt.Subtitle(2, timedelta(seconds=1), timedelta(seconds=2), "Line 2")
    ]
    translated_subs = [
        srt.Subtitle(1, timedelta(seconds=0), timedelta(seconds=1), "Linje 1"),
        srt.Subtitle(2, timedelta(seconds=1), timedelta(seconds=2), "")
    ]
    
    result = qa_gate(source_subs, translated_subs, "sv")
    assert result["dropped_count"] == 1
    assert "dropped_details" in result
    assert result["dropped_details"][0]["index"] == 2
