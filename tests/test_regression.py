from app.core.validator import evaluate_subtitle_health
import os
def test_evaluate_subtitle_health_wrong_language(tmp_path):
    p = tmp_path / "wrong.srt"
    blocks = []
    for i in range(1, 21):
        blocks.append(f"{i}\n00:00:0{i},000 --> 00:00:0{i},500\nThis is definitely an english sentence that will be detected as english.")
    content = "\n\n".join(blocks)
    p.write_text(content)
    health = evaluate_subtitle_health(str(p), "sv")
    assert health["status"] == "RED"
    assert health["detected_language"] == "en"
import pytest
import srt
from app.services.pipeline import qa_gate
from app.core.cleaner import sanitize_srt_content
from app.services.bazarr_checker import find_external_subtitle
from datetime import timedelta

def test_qa_invalid_srt():
    source = [srt.Subtitle(1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="Hello"), srt.Subtitle(2, start=timedelta(seconds=2), end=timedelta(seconds=3), content="World")]
    trans = [srt.Subtitle(1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="Hej")]
    res = qa_gate(source, trans, target_lang_code="sv")
    assert res["passed"] == False
    assert any("line count mismatch" in issue.lower() for issue in res["issues"])

def test_qa_wrong_language():
    source = [srt.Subtitle(1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="Hello this is a long text to trigger detection")]
    trans = [srt.Subtitle(1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="Hello this is a long text to trigger detection")]
    res = qa_gate(source, trans, target_lang_code="sv")
    assert res["passed"] == False
    assert any("appears to be en" in issue.lower() for issue in res["issues"])

def test_qa_sync_drift_hard_fail():
    source = [srt.Subtitle(1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="Hello")]
    trans = [srt.Subtitle(1, start=timedelta(seconds=1), end=timedelta(seconds=2, milliseconds=1), content="Hej")]
    res = qa_gate(source, trans, target_lang_code="sv")
    assert res["passed"] == False
    assert any("drift" in issue.lower() for issue in res["issues"])

def test_cleaner_preserves_short_cues():
    text = "1\n00:00:01,000 --> 00:00:01,050\nHello"
    subs, _ = sanitize_srt_content(text)
    assert len(subs) == 1
    
def test_cleaner_conservative_sdh():
    text = "1\n00:00:01,000 --> 00:00:02,000\nI told him (and I mean this) to leave."
    subs, _ = sanitize_srt_content(text)
    assert subs[0].content == "I told him (and I mean this) to leave."

    text2 = "1\n00:00:01,000 --> 00:00:02,000\n(laughing)"
    subs2, _ = sanitize_srt_content(text2)
    assert subs2[0].content == "<i></i>"

from app.core.validator import evaluate_subtitle_health
import os

def test_evaluate_subtitle_health_empty_file(tmp_path):
    p = tmp_path / 'empty.srt'
    p.write_text('')
    health = evaluate_subtitle_health(str(p), 'sv')
    assert health['status'] == 'RED'
    assert 'empty or corrupted' in health['reason'].lower()

def test_evaluate_subtitle_health_wrong_language(tmp_path):
    p = tmp_path / "wrong.srt"
    blocks = []
    for i in range(1, 21):
        blocks.append(f"{i}\n00:00:0{i},000 --> 00:00:0{i},500\nThis is definitely an english sentence that will be detected as english.")
    content = "\n\n".join(blocks)
    p.write_text(content)
    health = evaluate_subtitle_health(str(p), "sv")
    assert health["status"] == "RED"
    assert health["detected_language"] == "en"
