import pytest
import datetime
import srt
from app.core.validator import verify_sync, check_dropped_lines

def test_verify_sync_perfect():
    sub1 = [
        srt.Subtitle(1, start=datetime.timedelta(seconds=1), end=datetime.timedelta(seconds=3), content="Hello"),
        srt.Subtitle(2, start=datetime.timedelta(seconds=5), end=datetime.timedelta(seconds=8), content="World")
    ]
    sub2 = [
        srt.Subtitle(1, start=datetime.timedelta(seconds=1), end=datetime.timedelta(seconds=3), content="Hej"),
        srt.Subtitle(2, start=datetime.timedelta(seconds=5), end=datetime.timedelta(seconds=8), content="Värld")
    ]
    res = verify_sync(sub1, sub2)
    assert res["valid"] is True
    assert res["start_diff_ms"] == 0
    assert res["end_diff_ms"] == 0

def test_check_dropped_lines():
    orig = [
        srt.Subtitle(1, start=datetime.timedelta(seconds=1), end=datetime.timedelta(seconds=3), content="<i></i>"),
        srt.Subtitle(2, start=datetime.timedelta(seconds=4), end=datetime.timedelta(seconds=6), content="Important line")
    ]
    trans = [
        srt.Subtitle(1, start=datetime.timedelta(seconds=1), end=datetime.timedelta(seconds=3), content="<i></i>"),
        srt.Subtitle(2, start=datetime.timedelta(seconds=4), end=datetime.timedelta(seconds=6), content="")
    ]
    dropped_count, dropped = check_dropped_lines(orig, trans)
    assert dropped_count == 1
    assert dropped[0]["original"] == "Important line"
