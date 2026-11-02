"""Source discovery must agree with extraction without guessing a track's language."""
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

import pytest
import srt

from app.core.extractor import extract_embedded_srt, inspect_mkv_tracks
from app.services.source_resolver import SourceResolver, SourceOrigin


def subtitles(text="We have to get out of here before they come back.", count=30, step=3):
    return srt.compose([srt.Subtitle(i + 1, timedelta(seconds=i * step),
                                   timedelta(seconds=i * step + 2), text)
                        for i in range(count)])


def resolver(video, tracks=None, targets=None):
    return SourceResolver(str(video), tracks, "en", targets or ["sv"], "", "", False, True, 0)


def track(language="und", title="English (SDH)", **kwargs):
    return dict(id=2, language=language, title=title, codec="SubRip/SRT", **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("language,title", [("und", "English (SDH)"), ("", "English"), (None, "English")])
async def test_title_language_is_discovered_and_extracted(tmp_path, language, title):
    video = tmp_path / "Episode.mkv"
    video.write_bytes(b"video")
    tracks = {"subtitles": [track(language, title)], "duration": 90, "audio": [{"language": "eng"}]}

    def extract(cmd, **kwargs):
        Path(cmd[-1].split(":", 1)[1]).write_text(subtitles())
        return 0

    with patch("app.core.extractor._run_cancellable_cmd", side_effect=extract), \
         patch("app.services.source_resolver.detect_language_heuristics", return_value={"lang": "en", "confidence": .99}):
        source = await resolver(video, tracks).resolve()
    assert source is not None
    assert source.language == "en"
    assert source.origin == SourceOrigin.EMBEDDED
    assert len(source.cues) == 30


def test_mkvmerge_ietf_language_is_used(tmp_path):
    video = tmp_path / "Episode.mkv"
    video.write_bytes(b"video")
    metadata = {"tracks": [{"type": "subtitles", "id": 2, "codec": "SubRip/SRT",
                            "properties": {"language": "und", "language_ietf": "en-US"}}]}
    with patch("app.core.extractor.subprocess.run", return_value=MagicMock(stdout=json.dumps(metadata))):
        tracks = inspect_mkv_tracks(str(video))
    assert tracks["subtitles"][0]["language"] == "en-US"


@pytest.mark.parametrize("title,forced", [("English Commentary", False), ("English Signs & Songs", False),
                                        ("English Forced", False), ("English", True)])
def test_unsafe_title_only_tracks_are_not_sources(tmp_path, title, forced):
    with patch("app.core.extractor._run_cancellable_cmd") as run:
        assert not extract_embedded_srt("video.mkv", str(tmp_path / "out.srt"), "en",
                                        {"subtitles": [track(title=title, forced=forced)]})
    run.assert_not_called()


def test_explicit_language_wins_over_conflicting_title(tmp_path):
    with patch("app.core.extractor._run_cancellable_cmd") as run:
        assert not extract_embedded_srt("video.mkv", str(tmp_path / "out.srt"), "en",
                                        {"subtitles": [track("fra", "English")]})
    run.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [False, True])
async def test_invalid_first_track_does_not_hide_next_full_track(tmp_path, cached):
    video = tmp_path / "Episode.mkv"
    video.write_bytes(b"video")
    tracks = {"subtitles": [track("eng", "English Full", default=True),
                            track("eng", "English SDH") | {"id": 4}], "duration": 1800}
    partial = subtitles(count=150, step=.5)
    full = subtitles(count=150, step=12)
    commands = []

    def extract(cmd, **kwargs):
        commands.append(cmd)
        track_id, output = cmd[-1].split(":", 1)
        Path(output).write_text(partial if track_id == "2" else full)
        return 0

    def cache(video_path, track_id, lang):
        return partial if cached and track_id == 2 else None

    with patch("app.core.extractor._run_cancellable_cmd", side_effect=extract), \
         patch("app.core.extractor.get_cached_embedded_srt", side_effect=cache), \
         patch("app.services.source_resolver.detect_language_heuristics", return_value={"lang": "en", "confidence": .99}):
        source = await resolver(video, tracks).resolve()
    assert source is not None
    assert len(source.cues) == 150
    assert commands[-1][-1].startswith("4:")


@pytest.mark.asyncio
@pytest.mark.parametrize("detected,confidence,usable", [("en", .99, True), ("sv", .99, False),
                                                        ("unknown", 0, False), ("en", .4, False)])
async def test_bare_sidecar_requires_confident_non_target_language(tmp_path, detected, confidence, usable):
    video = tmp_path / "Episode.mkv"
    video.write_bytes(b"video")
    video.with_suffix(".srt").write_text(subtitles())
    with patch("app.services.source_resolver.detect_language_heuristics",
               return_value={"lang": detected, "confidence": confidence}):
        source = await resolver(video, {"subtitles": [], "audio": [{"language": "eng"}], "duration": 90}).resolve()
    assert (source is not None) == usable
    if source:
        assert source.language == "en"
        assert source.origin == SourceOrigin.EXTERNAL


@pytest.mark.asyncio
async def test_mislabeled_target_content_is_not_a_translation_source(tmp_path):
    video = tmp_path / "Episode.mkv"
    video.write_bytes(b"video")
    (tmp_path / "Episode.en.srt").write_text(subtitles())
    with patch("app.services.source_resolver.detect_language_heuristics", return_value={"lang": "sv", "confidence": .99}):
        assert await resolver(video, {"subtitles": [], "audio": [{"language": "eng"}], "duration": 90}).resolve() is None


@pytest.mark.asyncio
async def test_non_mkv_fallback_preserves_original_subtitle_stream_index(tmp_path):
    video = tmp_path / "Episode.mp4"
    video.write_bytes(b"video")
    tracks = {"subtitles": [track("eng", "English Full", default=True) | {"id": 6},
                            track("eng", "English SDH") | {"id": 9}], "duration": 1800}
    calls = []
    full = subtitles(step=60)

    def extract(cmd, **kwargs):
        mapping = cmd[cmd.index("-map") + 1]
        calls.append(mapping)
        Path(cmd[-1]).write_text(subtitles() if mapping == "0:s:0" else full)
        return 0

    with patch("app.core.extractor._run_cancellable_cmd", side_effect=extract), \
         patch("app.services.source_resolver.detect_language_heuristics", return_value={"lang": "en", "confidence": .99}):
        source = await resolver(video, tracks).resolve()
    assert calls == ["0:s:0", "0:s:1"]
    assert source.content == full
    assert source.cues == list(srt.parse(full))


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["Episode2.srt", "Episode.forced.srt", "Episode.en.forced.srt"])
async def test_unrelated_or_forced_sidecars_are_not_selected(tmp_path, name):
    video = tmp_path / "Episode.mkv"
    video.write_bytes(b"video")
    (tmp_path / name).write_text(subtitles())
    with patch("app.services.source_resolver.detect_language_heuristics", return_value={"lang": "en", "confidence": .99}):
        assert await resolver(video, {"subtitles": [], "audio": [{"language": "eng"}], "duration": 90}).resolve() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["broken SRT", subtitles(step=3)])
async def test_bare_sidecars_still_require_structure_and_coverage(tmp_path, content):
    video = tmp_path / "Episode.mkv"
    video.write_bytes(b"video")
    video.with_suffix(".srt").write_text(content)
    with patch("app.services.source_resolver.detect_language_heuristics", return_value={"lang": "en", "confidence": .99}):
        assert await resolver(video, {"subtitles": [], "audio": [{"language": "eng"}], "duration": 1800}).resolve() is None


def test_unspecified_ietf_tag_does_not_hide_known_legacy_language(tmp_path):
    video = tmp_path / "Episode.mkv"
    video.write_bytes(b"video")
    metadata = {"tracks": [{"type": "subtitles", "id": 2, "codec": "SubRip/SRT",
                            "properties": {"language": "eng", "language_ietf": "und"}}]}
    with patch("app.core.extractor.subprocess.run", return_value=MagicMock(stdout=json.dumps(metadata))):
        tracks = inspect_mkv_tracks(str(video))
    assert tracks["subtitles"][0]["language"] == "eng"
