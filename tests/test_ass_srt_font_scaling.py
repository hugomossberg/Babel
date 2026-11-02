"""ASS canvas font sizes must not become absolute SRT player font sizes (#15)."""
from pathlib import Path
import shutil
import subprocess
import threading
from unittest.mock import patch

import pytest
import srt

from app.core import db, extractor


STYLED = ('1\n00:00:01,230 --> 00:00:03,450\n'
          '<font face="Montara Gothic" size="55" color="#e7eae2"><b>Hello!</b></font>\n\n'
          '2\n00:00:04,000 --> 00:00:06,000\n'
          '<font size="65"><i>{\\an8}Italic top\nSecond line</i></font>\n\n')
PLAYER_SIZED = STYLED.replace(' size="55"', '').replace(' size="65"', '')
ASS_CODECS = ['S_TEXT/ASS', 'S_TEXT/SSA', 'SubStationAlpha']


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path / 'cache.db'))
    db.init_db()


def tracks(codec):
    return {'subtitles': [{'id': 2, 'language': 'eng', 'codec': codec,
                           'title': 'English Full', 'forced': False}], 'duration': 10}


def assert_content_and_sync(output, expected):
    assert output == expected
    original, result = list(srt.parse(STYLED)), list(srt.parse(output))
    assert len(result) == len(original) == 2
    assert [(c.index, c.start, c.end) for c in result] == [
        (c.index, c.start, c.end) for c in original]
    assert '<b>Hello!</b>' in output
    assert '<i>{\\an8}Italic top\nSecond line</i>' in output


@pytest.mark.parametrize('codec', ASS_CODECS + ['SubRip/SRT', 'S_TEXT/WEBVTT'])
@pytest.mark.parametrize('backend', ['mkvextract', 'ffmpeg'])
def test_fresh_output_and_persistent_cache_use_player_size_only_for_ass(tmp_path, codec, backend):
    video = tmp_path / ('episode.mkv' if backend == 'mkvextract' else 'episode.mp4')
    video.write_bytes(b'media')
    output = tmp_path / 'out.srt'

    def run(cmd, **kwargs):
        destination = cmd[-1].split(':', 1)[1] if cmd[0] == 'mkvextract' else cmd[-1]
        Path(destination).write_text(STYLED)
        return 0

    with patch.object(extractor, '_run_cancellable_cmd', side_effect=run):
        assert extractor.extract_embedded_srt(str(video), str(output), 'en', tracks(codec))
    expected = PLAYER_SIZED if codec in ASS_CODECS else STYLED
    assert_content_and_sync(output.read_text(), expected)
    assert extractor.get_cached_embedded_srt(str(video), 2, 'en') == expected


@pytest.mark.parametrize('codec', ASS_CODECS + ['SubRip/SRT'])
def test_legacy_cached_extractions_are_normalized_without_reextracting(tmp_path, codec):
    video = tmp_path / 'episode.mkv'
    video.write_bytes(b'media')
    output = tmp_path / 'out.srt'
    assert extractor.save_cached_embedded_srt(str(video), 2, 'en', STYLED)
    with patch.object(extractor, '_run_cancellable_cmd') as run:
        assert extractor.extract_embedded_srt(str(video), str(output), 'en', tracks(codec))
    run.assert_not_called()
    assert_content_and_sync(output.read_text(), PLAYER_SIZED if codec in ASS_CODECS else STYLED)


def test_concurrent_extraction_cache_reuse_also_normalizes_ass(tmp_path, monkeypatch):
    video = tmp_path / 'episode.mkv'
    video.write_bytes(b'media')
    output = tmp_path / 'out.srt'
    ready = threading.Event()
    ready.set()
    monkeypatch.setattr(extractor, '_in_flight_extractions', {(str(video), 2, 'en'): ready})
    with patch.object(extractor, 'get_cached_embedded_srt', side_effect=[None, STYLED]), \
         patch.object(extractor, '_run_cancellable_cmd') as run:
        assert extractor.extract_embedded_srt(str(video), str(output), 'en', tracks('S_TEXT/ASS'))
    run.assert_not_called()
    assert_content_and_sync(output.read_text(), PLAYER_SIZED)


@pytest.mark.parametrize('font, expected', [
    ('<FONT SIZE = \'55\' FACE="Arial">', '<FONT FACE="Arial">'),
    ('<font color="#fff" size=65>', '<font color="#fff">'),
    ('<font data-size="55" size="50">', '<font data-size="55">'),
    ('<font face="A size=55 Font" size="50">', '<font face="A size=55 Font">'),
    ('<font face="A > Font" size="50">', '<font face="A > Font">'),
])
def test_font_attribute_variants_preserve_other_attributes(tmp_path, font, expected):
    video = tmp_path / 'episode.mkv'
    video.write_bytes(b'media')
    source = f'1\n00:00:01,000 --> 00:00:02,000\n{font}<u>Hello</u></font>\n'
    assert extractor.save_cached_embedded_srt(str(video), 2, 'en', source)
    output = tmp_path / 'out.srt'
    assert extractor.extract_embedded_srt(str(video), str(output), 'en', tracks('S_TEXT/ASS'))
    assert output.read_text() == source.replace(font, expected)


@pytest.mark.parametrize('fallback', [False, True])
def test_real_ass_to_srt_conversion_preserves_formatting_and_timestamps(tmp_path, fallback):
    """Use real container/extraction/conversion tools when installed (as in Docker)."""
    required = ['ffmpeg', 'mkvmerge', 'mkvextract']
    if not all(shutil.which(tool) for tool in required):
        pytest.skip('requires ffmpeg and mkvtoolnix')
    ass = tmp_path / 'source.ass'
    ass.write_text('''[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,55,&H00E2EAE7,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.23,0:00:03.45,Default,,0,0,0,,Hello, world!
Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,{\\an8\\i1}Italic top{\\i0}\\NSecond line
''')
    video, output = tmp_path / 'episode.mkv', tmp_path / 'out.srt'
    subprocess.run(['ffmpeg', '-y', '-i', str(ass), '-c:s', 'copy',
                    '-metadata:s:s:0', 'language=eng', str(video)],
                   capture_output=True, check=True, timeout=20)
    real_run = extractor._run_cancellable_cmd
    commands = []

    def run(cmd, **kwargs):
        commands.append(cmd[0])
        if fallback and cmd[0] == 'mkvextract':
            return 1
        return real_run(cmd, **kwargs)

    with patch.object(extractor, '_run_cancellable_cmd', side_effect=run):
        assert extractor.extract_embedded_srt(str(video), str(output), 'en', timeout=20)
    text = output.read_text()
    assert 'size=' not in text.lower()
    assert '<b>' in text and '<i>' in text and '{\\an8}' in text
    assert 'color="#e7eae2"' in text
    assert 'Italic top</i>\nSecond line' in text
    cues = list(srt.parse(text))
    assert len(cues) == 2
    assert [(c.start.total_seconds(), c.end.total_seconds()) for c in cues] == [(1.23, 3.45), (4.0, 6.0)]
    assert commands == ['mkvextract', 'ffmpeg']
