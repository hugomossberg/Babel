import pytest
import datetime
import srt
from app.services.pipeline import qa_gate

def make_sub(idx, text):
    return srt.Subtitle(
        idx,
        start=datetime.timedelta(seconds=idx),
        end=datetime.timedelta(seconds=idx+1),
        content=text
    )

def test_qa_gate_passes():
    source = [make_sub(1, "Hello world"), make_sub(2, "How are you?")]
    trans = [make_sub(1, "Hej världen"), make_sub(2, "Hur mår du?")]
    
    res = qa_gate(source, trans, target_lang_code="sv")
    assert res["passed"] is True
    assert res["score"] == 100
    assert len(res["untranslated_ids"]) == 0
    assert res["dropped_count"] == 0

def test_qa_gate_fails_on_dropped():
    source = [make_sub(1, "Hello world"), make_sub(2, "How are you?"), make_sub(3, "Yes"), make_sub(4, "No")]
    trans = [make_sub(1, "Hej världen"), make_sub(2, ""), make_sub(3, ""), make_sub(4, "")]
    
    res = qa_gate(source, trans, target_lang_code="sv")
    assert res["passed"] is False
    assert res["dropped_count"] == 3

def test_qa_gate_detects_untranslated():
    source = [make_sub(1, "Hello world"), make_sub(2, "This should be translated")]
    trans = [make_sub(1, "Hej världen"), make_sub(2, "This should be translated")]
    
    res = qa_gate(source, trans, target_lang_code="sv")
    assert len(res["untranslated_ids"]) == 1
    assert res["untranslated_ids"][0] == 1
