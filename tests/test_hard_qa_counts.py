import pytest
from app.services.pipeline import qa_gate
import srt
from datetime import timedelta

def test_hard_qa_drop_counts():
    source = [srt.Subtitle(index=i, start=timedelta(seconds=i), end=timedelta(seconds=i+1), content=f"Source {i}") for i in range(1, 10)]
    
    # 0 dropped
    target_0 = [srt.Subtitle(index=i, start=timedelta(seconds=i), end=timedelta(seconds=i+1), content=f"This is a long English target text to pass the confidence threshold {i}") for i in range(1, 10)]
    assert qa_gate(source, target_0, target_lang_code="en")["passed"] is True
    
    # 1 dropped
    target_1 = target_0.copy()
    target_1[2] = srt.Subtitle(index=3, start=timedelta(seconds=3), end=timedelta(seconds=4), content="")
    res1 = qa_gate(source, target_1, target_lang_code="en")
    assert res1["passed"] is False
    assert res1["dropped_count"] == 1
    
    # 2 dropped
    target_2 = target_1.copy()
    target_2[3] = srt.Subtitle(index=4, start=timedelta(seconds=4), end=timedelta(seconds=5), content="")
    res2 = qa_gate(source, target_2, target_lang_code="en")
    assert res2["passed"] is False
    assert res2["dropped_count"] == 2
    
    # 3 dropped
    target_3 = target_2.copy()
    target_3[4] = srt.Subtitle(index=5, start=timedelta(seconds=5), end=timedelta(seconds=6), content="")
    res3 = qa_gate(source, target_3, target_lang_code="en")
    assert res3["passed"] is False
    assert res3["dropped_count"] == 3
