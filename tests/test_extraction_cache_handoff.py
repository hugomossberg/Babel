"""Regression coverage for a cache fill between initial lookup and I/O admission."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import pytest
import srt

from app.core import db, extractor


CONTENT = '1\n00:00:01,000 --> 00:00:03,000\n<i>The witness is waiting.</i>\n'


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path / 'cache.db'))
    db.init_db()
    assert not extractor._in_flight_extractions
    yield
    assert not extractor._in_flight_extractions


def tracks(codec='SubRip/SRT', second=False):
    values = [{'id': 1, 'language': 'eng', 'codec': codec}]
    if second:
        values.append({'id': 2, 'language': 'eng', 'codec': 'SubRip/SRT'})
    return {'duration': 30, 'subtitles': values}


def test_late_cache_miss_does_not_repeat_completed_extraction(tmp_path, monkeypatch):
    video = tmp_path / 'shared.mkv'
    video.write_bytes(b'owned media identity')
    started, missed, finished = (threading.Event() for _ in range(3))
    original = extractor.get_cached_embedded_srt
    calls = []
    late_first = True

    def lookup(*args):
        nonlocal late_first
        value = original(*args)
        if threading.current_thread().name.startswith('late') and late_first:
            late_first = False
            assert value is None
            missed.set()
            assert finished.wait(3)
        return value

    def command(cmd, **kwargs):
        calls.append(cmd[0])
        started.set()
        assert missed.wait(3)
        Path(cmd[-1].split(':', 1)[1]).write_text(CONTENT)
        return 0

    def lead():
        try:
            return extractor.extract_embedded_srt(str(video), str(tmp_path / 'lead.srt'), tracks_info=tracks())
        finally:
            finished.set()

    monkeypatch.setattr(extractor, 'get_cached_embedded_srt', lookup)
    monkeypatch.setattr(extractor, '_run_cancellable_cmd', command)
    with ThreadPoolExecutor(1, thread_name_prefix='lead') as first, ThreadPoolExecutor(1, thread_name_prefix='late') as second:
        a = first.submit(lead)
        assert started.wait(3)
        b = second.submit(extractor.extract_embedded_srt, str(video), str(tmp_path / 'late.srt'), tracks_info=tracks())
        assert a.result(timeout=5) and b.result(timeout=5)
    assert calls == ['mkvextract']
    assert (tmp_path / 'lead.srt').read_text() == (tmp_path / 'late.srt').read_text() == CONTENT


class Admission:
    def __init__(self, fill):
        self.fill, self.releases = fill, 0

    def acquire(self, timeout):
        if self.fill:
            self.fill()
            self.fill = None
        return True

    def release(self):
        self.releases += 1


@pytest.mark.parametrize('codec,expected', [
    ('SubRip/SRT', '<font size="65" color="red"><i>Hello.</i></font>'),
    ('S_TEXT/ASS', '<font color="red"><i>Hello.</i></font>'),
])
def test_cache_filled_while_queued_preserves_codec_formatting(tmp_path, monkeypatch, codec, expected):
    video, output = tmp_path / 'queued.mkv', tmp_path / 'out.srt'
    video.write_bytes(b'owned media')
    styled = CONTENT.replace('<i>The witness is waiting.</i>', '<font size="65" color="red"><i>Hello.</i></font>')
    semaphore = Admission(lambda: extractor.save_cached_embedded_srt(str(video), 1, 'eng', styled))
    monkeypatch.setattr(extractor, '_extraction_semaphore', semaphore)
    calls = []
    monkeypatch.setattr(extractor, '_run_cancellable_cmd', lambda *a, **k: calls.append(a) or 1)
    assert extractor.extract_embedded_srt(str(video), str(output), tracks_info=tracks(codec))
    assert list(srt.parse(output.read_text()))[0].content == expected
    assert not calls
    assert semaphore.releases == 1


def test_rejected_queued_cache_still_tries_next_eligible_track(tmp_path, monkeypatch):
    video, output = tmp_path / 'tracks.mkv', tmp_path / 'out.srt'
    video.write_bytes(b'owned media')
    semaphore = Admission(lambda: extractor.save_cached_embedded_srt(str(video), 1, 'eng', CONTENT.replace('witness', 'WRONG')))
    monkeypatch.setattr(extractor, '_extraction_semaphore', semaphore)
    calls = []

    def command(cmd, **kwargs):
        calls.append(cmd[-1].split(':', 1)[0])
        output.write_text(CONTENT)
        return 0

    monkeypatch.setattr(extractor, '_run_cancellable_cmd', command)
    assert extractor.extract_embedded_srt(str(video), str(output), tracks_info=tracks(second=True),
                                        candidate_validator=lambda path: 'WRONG' not in Path(path).read_text())
    assert calls == ['2']
    assert output.read_text() == CONTENT
    assert semaphore.releases == 2


def test_cancellation_during_admission_cannot_publish_cached_output(tmp_path, monkeypatch):
    video, output = tmp_path / 'cancel.mkv', tmp_path / 'out.srt'
    video.write_bytes(b'owned media')
    cancel = threading.Event()

    def fill():
        extractor.save_cached_embedded_srt(str(video), 1, 'eng', CONTENT)
        cancel.set()

    semaphore = Admission(fill)
    monkeypatch.setattr(extractor, '_extraction_semaphore', semaphore)
    assert not extractor.extract_embedded_srt(str(video), str(output), tracks_info=tracks(), cancel_event=cancel)
    assert not output.exists()
    assert semaphore.releases == 1
