import asyncio
import os
import shutil
import subprocess
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import srt

import app.core.db as db_mod
from app.core.db import DB_PATH, create_job, get_job_by_id, init_db
from app.core.extractor import extract_embedded_srt, inspect_mkv_tracks
from app.core.trust_engine import (
    CandidateOrigin,
    SubtitleTrustEngine,
    TrustDecision,
    TrustResult,
    VerificationMode,
)
from app.services.pipeline import SubtitlePipeline

ASS_EN_CONTENT = """[Script Info]
Title: English ASS
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,English line one.
Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,English line two.
Dialogue: 0,0:00:07.00,0:00:09.00,Default,,0,0,0,,English line three.
Dialogue: 0,0:00:10.00,0:00:12.00,Default,,0,0,0,,English line four.
Dialogue: 0,0:00:13.00,0:00:15.00,Default,,0,0,0,,English line five.
"""

ASS_SV_CONTENT = """[Script Info]
Title: Swedish ASS
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Svensk rad ett.
Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,Svensk rad två.
Dialogue: 0,0:00:07.00,0:00:09.00,Default,,0,0,0,,Svensk rad tre.
Dialogue: 0,0:00:10.00,0:00:12.00,Default,,0,0,0,,Svensk rad fyra.
Dialogue: 0,0:00:13.00,0:00:15.00,Default,,0,0,0,,Svensk rad fem.
"""

SRT_SV_CONTENT = """1
00:00:01,000 --> 00:00:03,000
Svensk rad ett.

2
00:00:04,000 --> 00:00:06,000
Svensk rad två.

3
00:00:07,000 --> 00:00:09,000
Svensk rad tre.

4
00:00:10,000 --> 00:00:12,000
Svensk rad fyra.

5
00:00:13,000 --> 00:00:15,000
Svensk rad fem.
"""

VTT_SV_CONTENT = """WEBVTT

00:00:01.000 --> 00:00:03.000
Svensk rad ett.

00:00:04.000 --> 00:00:06.000
Svensk rad två.

00:00:07.000 --> 00:00:09.000
Svensk rad tre.

00:00:10.000 --> 00:00:12.000
Svensk rad fyra.

00:00:13.000 --> 00:00:15.000
Svensk rad fem.
"""


def _has_mkv_tooling() -> bool:
    return shutil.which("mkvmerge") is not None and shutil.which("mkvextract") is not None and shutil.which("ffmpeg") is not None


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """Each test gets its own isolated DB."""
    original = db_mod.DB_PATH
    db_mod.DB_PATH = str(tmp_path / "test.db")
    db_mod.init_db()
    yield tmp_path
    db_mod.DB_PATH = original


def create_mkv_fixture(out_path: str, tracks: list[tuple[str, str, str, bool]]) -> None:
    """
    Creates a real MKV container file using mkvmerge.
    tracks: list of (content, ext, lang_code, is_forced)
    """
    tmp_dir = os.path.dirname(out_path)
    cmd = ["mkvmerge", "-o", out_path]
    temp_files = []
    try:
        for idx, (content, ext, lang, forced) in enumerate(tracks):
            t_path = os.path.join(tmp_dir, f"track_{idx}.{ext}")
            with open(t_path, "w", encoding="utf-8") as f:
                f.write(content)
            temp_files.append(t_path)
            cmd.extend([
                "--language", f"0:{lang}",
                "--forced-display-flag", f"0:{'yes' if forced else 'no'}",
                t_path
            ])
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"mkvmerge failed: {res.stderr}")
    finally:
        for tf in temp_files:
            if os.path.exists(tf):
                os.remove(tf)


@pytest.mark.skipif(not _has_mkv_tooling(), reason="Real MKV tooling (mkvmerge, mkvextract, ffmpeg) required")
@pytest.mark.asyncio
async def test_1_embedded_swedish_ass_authoritative_acquisition(tmp_path, monkeypatch):
    """
    1. Proves authoritative embedded Swedish ASS acquisition:
       - MKV fixture contains English ASS and Swedish ASS.
       - Discovers and converts embedded Swedish ASS to valid SRT.
       - Publishes external .sv.srt.
       - Bazarr search is NOT called.
       - AI translation is NOT called.
       - Job logs reflect exact embedded selection path.
    """
    mkv_file = str(tmp_path / "The.Fast.and.the.Furious.2001.mkv")
    create_mkv_fixture(mkv_file, [
        (ASS_EN_CONTENT, "ass", "eng", False),
        (ASS_SV_CONTENT, "ass", "swe", False),
    ])

    pipeline = SubtitlePipeline()
    bazarr_mock = AsyncMock()
    translate_mock = AsyncMock()
    monkeypatch.setattr(pipeline, "trigger_bazarr_search", bazarr_mock)
    monkeypatch.setattr(pipeline.translator, "translate_srt_content", translate_mock)

    def mock_get_setting(key, default=""):
        settings = {
            "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
            "materialize_embedded_target": "true",
            "extract_target_embedded": "true",
            "extract_source_embedded": "true",
            "enable_bazarr": "true",
            "auto_repair_unhealthy": "false",
        }
        return settings.get(key, default)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)

    res = await pipeline.process_video_file(mkv_file, event_source="SONARR")

    assert res["status"] == "skipped"
    assert res["reason"] == "already_exists"

    # Verify published file
    published_sv = str(tmp_path / "The.Fast.and.the.Furious.2001.sv.srt")
    assert os.path.exists(published_sv)
    with open(published_sv, "r", encoding="utf-8-sig") as f:
        cues = list(srt.parse(f.read()))
    assert len(cues) == 5
    assert "Svensk rad ett." in cues[0].content

    # Invariant: 0 Bazarr triggers, 0 AI calls
    bazarr_mock.assert_not_called()
    translate_mock.assert_not_called()

    # Verify detailed logging
    job = get_job_by_id(res["job_id"])
    log_text = "\n".join(job.get("logs") or [])
    assert "Embedded target scan: Swedish SubStationAlpha candidate found" in log_text
    assert "Embedded target extraction:" in log_text
    assert "Embedded target validation: PASS" in log_text
    assert "Embedded Swedish target selected" in log_text
    assert "Bazarr skipped — embedded target satisfied language" in log_text
    assert "AI skipped" in log_text
    assert "AI calls: 0" in log_text


@pytest.mark.skipif(not _has_mkv_tooling(), reason="Real MKV tooling required")
@pytest.mark.asyncio
async def test_2_embedded_target_with_unrelated_existing_external_source(tmp_path, monkeypatch):
    """
    2. Proves existing external source (.en.srt) does NOT prevent Babel from discovering
       and selecting the embedded Swedish target track first.
    """
    mkv_file = str(tmp_path / "Movie.2024.mkv")
    create_mkv_fixture(mkv_file, [
        (ASS_EN_CONTENT, "ass", "eng", False),
        (ASS_SV_CONTENT, "ass", "swe", False),
    ])
    # Create external English SRT
    ext_en = str(tmp_path / "Movie.2024.en.srt")
    with open(ext_en, "w", encoding="utf-8") as f:
        f.write("1\n00:00:01,000 --> 00:00:03,000\nExternal English line.\n")

    pipeline = SubtitlePipeline()
    bazarr_mock = AsyncMock()
    translate_mock = AsyncMock()
    monkeypatch.setattr(pipeline, "trigger_bazarr_search", bazarr_mock)
    monkeypatch.setattr(pipeline.translator, "translate_srt_content", translate_mock)

    def mock_get_setting(key, default=""):
        settings = {
            "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
            "materialize_embedded_target": "true",
            "extract_target_embedded": "true",
            "extract_source_embedded": "true",
            "enable_bazarr": "true",
        }
        return settings.get(key, default)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)

    res = await pipeline.process_video_file(mkv_file, event_source="SONARR")

    assert res["status"] == "skipped"
    assert res["reason"] == "already_exists"

    published_sv = str(tmp_path / "Movie.2024.sv.srt")
    assert os.path.exists(published_sv)
    with open(published_sv, "r", encoding="utf-8-sig") as f:
        cues = list(srt.parse(f.read()))
    assert len(cues) == 5

    bazarr_mock.assert_not_called()
    translate_mock.assert_not_called()


@pytest.mark.skipif(not _has_mkv_tooling(), reason="Real MKV tooling required")
@pytest.mark.asyncio
async def test_2b_embedded_target_beats_unhealthy_external(tmp_path, monkeypatch):
    """
    2b. Proves embedded target replaces unhealthy/stale external subtitle.
    """
    mkv_file = str(tmp_path / "CorruptExternal.mkv")
    create_mkv_fixture(mkv_file, [
        (ASS_EN_CONTENT, "ass", "eng", False),
        (ASS_SV_CONTENT, "ass", "swe", False),
    ])
    # Create empty / corrupted external Swedish SRT
    ext_sv = str(tmp_path / "CorruptExternal.sv.srt")
    with open(ext_sv, "w", encoding="utf-8") as f:
        f.write("Broken corrupted subtitle text without cues")

    pipeline = SubtitlePipeline()
    bazarr_mock = AsyncMock()
    translate_mock = AsyncMock()
    monkeypatch.setattr(pipeline, "trigger_bazarr_search", bazarr_mock)
    monkeypatch.setattr(pipeline.translator, "translate_srt_content", translate_mock)

    def mock_get_setting(key, default=""):
        settings = {
            "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
            "materialize_embedded_target": "true",
            "extract_target_embedded": "true",
            "extract_source_embedded": "true",
            "enable_bazarr": "true",
        }
        return settings.get(key, default)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)

    res = await pipeline.process_video_file(mkv_file, event_source="SONARR")

    assert res["status"] == "skipped"
    assert res["reason"] == "already_exists"

    # Verify replaced with healthy embedded subtitle
    with open(ext_sv, "r", encoding="utf-8-sig") as f:
        cues = list(srt.parse(f.read()))
    assert len(cues) == 5
    assert "Svensk rad ett." in cues[0].content


@pytest.mark.skipif(not _has_mkv_tooling(), reason="Real MKV tooling required")
@pytest.mark.asyncio
async def test_3_forced_embedded_track_excluded_from_full_target(tmp_path, monkeypatch):
    """
    3. Proves embedded tracks marked forced (flagged forced=True) are NOT accepted as full targets.
    """
    mkv_file = str(tmp_path / "ActionMovie.mkv")
    # Only a forced Swedish track
    create_mkv_fixture(mkv_file, [
        (ASS_EN_CONTENT, "ass", "eng", False),
        (ASS_SV_CONTENT, "ass", "swe", True),  # is_forced = True
    ])

    pipeline = SubtitlePipeline()
    bazarr_called_langs = []

    async def mock_bazarr(video_path, language="sv", **kwargs):
        bazarr_called_langs.append(language)

    async def mock_translate(subs, target_language="Swedish", **kwargs):
        return [
            srt.Subtitle(
                index=s.index,
                start=s.start,
                end=s.end,
                content=f"Översatt {s.content}"
            )
            for s in subs
        ]

    monkeypatch.setattr(pipeline, "trigger_bazarr_search", mock_bazarr)
    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)
    monkeypatch.setattr("app.services.pipeline.qa_gate", lambda *args, **kwargs: {"passed": True, "score": 100})

    def mock_get_setting(key, default=""):
        settings = {
            "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
            "materialize_embedded_target": "true",
            "extract_target_embedded": "true",
            "extract_source_embedded": "true",
            "enable_bazarr": "true",
        }
        return settings.get(key, default)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)

    res = await pipeline.process_video_file(mkv_file, event_source="SONARR")

    # Because forced track was rejected, language fell through to translation
    assert "sv" in bazarr_called_langs
    assert res["status"] in ("translated", "success")
    published_sv = str(tmp_path / "ActionMovie.sv.srt")
    assert os.path.exists(published_sv)


@pytest.mark.skipif(not _has_mkv_tooling(), reason="Real MKV tooling required")
@pytest.mark.parametrize("fmt,content,ext", [
    ("srt", SRT_SV_CONTENT, "srt"),
    ("ass", ASS_SV_CONTENT, "ass"),
    ("vtt", VTT_SV_CONTENT, "vtt"),
])
@pytest.mark.asyncio
async def test_4_all_embedded_target_formats(tmp_path, monkeypatch, fmt, content, ext):
    """
    4. Proves embedded SRT, ASS, and VTT formats all extract and convert deterministically to valid SRT.
    """
    mkv_file = str(tmp_path / f"test_format_{fmt}.mkv")
    create_mkv_fixture(mkv_file, [
        (ASS_EN_CONTENT, "ass", "eng", False),
        (content, ext, "swe", False),
    ])

    pipeline = SubtitlePipeline()
    monkeypatch.setattr(pipeline, "trigger_bazarr_search", AsyncMock())
    monkeypatch.setattr(pipeline.translator, "translate_srt_content", AsyncMock())

    def mock_get_setting(key, default=""):
        settings = {
            "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
            "materialize_embedded_target": "true",
            "extract_target_embedded": "true",
            "extract_source_embedded": "true",
            "enable_bazarr": "false",
        }
        return settings.get(key, default)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)

    res = await pipeline.process_video_file(mkv_file, event_source="SONARR")

    assert res["status"] == "skipped"
    published_sv = str(tmp_path / f"test_format_{fmt}.sv.srt")
    assert os.path.exists(published_sv)
    with open(published_sv, "r", encoding="utf-8-sig") as f:
        parsed = list(srt.parse(f.read()))
    assert len(parsed) == 5
    assert "Svensk rad" in parsed[0].content


@pytest.mark.skipif(not _has_mkv_tooling(), reason="Real MKV tooling required")
@pytest.mark.asyncio
async def test_5_multilingual_partial_embedded_satisfaction(tmp_path, monkeypatch):
    """
    5. Proves multi-target configuration [sv, de]:
       - MKV contains embedded Swedish ASS, but NO German track.
       - Swedish is satisfied and published via embedded path.
       - German is queued and translated via AI.
       - Job does NOT terminate prematurely.
    """
    mkv_file = str(tmp_path / "Bilingual.2024.mkv")
    create_mkv_fixture(mkv_file, [
        (ASS_EN_CONTENT, "ass", "eng", False),
        (ASS_SV_CONTENT, "ass", "swe", False),
    ])

    pipeline = SubtitlePipeline()
    bazarr_searched_langs = []
    ai_translated_langs = []

    async def mock_bazarr(video_path, language="sv", **kwargs):
        bazarr_searched_langs.append(language)

    async def mock_translate(subs, target_language="German", **kwargs):
        ai_translated_langs.append(target_language)
        return [
            srt.Subtitle(index=s.index, start=s.start, end=s.end, content=f"DE: {s.content}")
            for s in subs
        ]

    monkeypatch.setattr(pipeline, "trigger_bazarr_search", mock_bazarr)
    monkeypatch.setattr(pipeline.translator, "translate_srt_content", mock_translate)
    monkeypatch.setattr("app.services.pipeline.qa_gate", lambda *args, **kwargs: {"passed": True, "score": 100})

    def mock_get_setting(key, default=""):
        settings = {
            "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}, {"code": "de", "name": "German", "enabled": true}]',
            "materialize_embedded_target": "true",
            "extract_target_embedded": "true",
            "extract_source_embedded": "true",
            "enable_bazarr": "true",
        }
        return settings.get(key, default)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)

    res = await pipeline.process_video_file(mkv_file, event_source="SONARR")

    assert res["status"] in ("translated", "success")

    # Both Swedish and German files exist on disk
    sv_file = str(tmp_path / "Bilingual.2024.sv.srt")
    de_file = str(tmp_path / "Bilingual.2024.de.srt")
    assert os.path.exists(sv_file)
    assert os.path.exists(de_file)

    # Swedish was NOT searched in Bazarr and NOT translated by AI
    assert "sv" not in bazarr_searched_langs
    assert "Swedish" not in ai_translated_langs

    # German WAS searched and translated
    assert "de" in bazarr_searched_langs
    assert "German" in ai_translated_langs


@pytest.mark.skipif(not _has_mkv_tooling(), reason="Real MKV tooling required")
@pytest.mark.asyncio
async def test_regression_embedded_target_satisfaction_when_bazarr_publication_blocked(tmp_path, monkeypatch):
    """
    Real-world P0 Regression Test:
    1. MKV has embedded Swedish target track.
    2. Embedded Swedish extraction succeeds.
    3. Trust validation returns PASS (score=100/100).
    4. Bazarr lifecycle/publication ownership reports ACTIVE (e.g. actively writing).
    5. Publication is deferred/blocked to prevent overwriting active Bazarr worker.
    6. Embedded Swedish remains authoritative and target-satisfied (status=skipped, already_exists).
    7. Bazarr target search triggers for sv == 0.
    8. AI translation calls for sv == 0.
    9. Source fallback for sv is never entered.
    10. Existing external file is NOT overwritten while ACTIVE.
    """
    from app.services.bazarr_coordinator import (
        BazarrJobInfo,
        BazarrJobPollStatus,
        BazarrJobsPollResult,
        bazarr_coordinator,
    )

    mkv_file = str(tmp_path / "How I Met Your Mother - S01E11 - The Limo.mkv")
    create_mkv_fixture(mkv_file, [
        (ASS_EN_CONTENT, "ass", "eng", False),
        (SRT_SV_CONTENT, "srt", "swe", False),
    ])

    # Pre-existing external file being written by Bazarr worker
    external_sv = tmp_path / "How I Met Your Mother - S01E11 - The Limo.sv.srt"
    initial_bazarr_content = "1\n00:00:00,100 --> 00:00:00,900\nBazarr partial..."
    external_sv.write_text(initial_bazarr_content, encoding="utf-8")

    pipeline = SubtitlePipeline()
    bazarr_mock = AsyncMock()
    translate_mock = AsyncMock()
    monkeypatch.setattr(pipeline, "trigger_bazarr_search", bazarr_mock)
    monkeypatch.setattr(pipeline.translator, "translate_srt_content", translate_mock)

    def mock_get_setting(key, default=""):
        settings = {
            "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
            "materialize_embedded_target": "true",
            "extract_target_embedded": "true",
            "extract_source_embedded": "true",
            "enable_bazarr_check": "true",
            "bazarr_url": "http://bazarr:6767",
            "bazarr_api_key": "test_key",
        }
        return settings.get(key, default)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)
    monkeypatch.setattr("app.core.db.get_setting", mock_get_setting)
    monkeypatch.setattr("app.services.bazarr_coordinator.get_setting", mock_get_setting)

    # Bazarr active sync job running on this exact file
    sync_job = BazarrJobInfo(
        job_id="sync_limo",
        job_name="Syncing Subtitles with Video",
        status="running",
        job_type="sync",
        progress_message="How I Met Your Mother - S01E11 - The Limo.sv.srt",
    )

    with patch.object(bazarr_coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_poll:
        mock_poll.return_value = BazarrJobsPollResult(
            status=BazarrJobPollStatus.ACTIVE,
            jobs=[sync_job],
        )

        res = await pipeline.process_video_file(mkv_file, event_source="SONARR")

    assert res["status"] == "skipped"
    assert res["reason"] == "already_exists"

    # Invariant: Bazarr search was NOT triggered, AI was NOT called
    bazarr_mock.assert_not_called()
    translate_mock.assert_not_called()

    # Bazarr file was not clobbered while active
    assert external_sv.read_text(encoding="utf-8") == initial_bazarr_content

    # Check job logs
    job = get_job_by_id(res["job_id"])
    log_text = "\n".join(job.get("logs") or [])
    assert "Embedded target validation: PASS" in log_text
    assert "External publication deferred (bazarr_actively_writing). Target language satisfied." in log_text
    assert "Bazarr skipped — embedded target satisfied language" in log_text
    assert "AI skipped" in log_text
    assert "AI calls: 0" in log_text


@pytest.mark.skipif(not _has_mkv_tooling(), reason="Real MKV tooling required")
@pytest.mark.asyncio
async def test_regression_embedded_target_control_bazarr_idle(tmp_path, monkeypatch):
    """
    Control Case:
    Embedded Swedish Trust PASS + Bazarr idle
    => Embedded target wins
    => External .sv.srt published
    => Bazarr target search calls = 0
    => AI translation calls = 0
    """
    from app.services.bazarr_coordinator import (
        BazarrJobPollStatus,
        BazarrJobsPollResult,
        bazarr_coordinator,
    )

    mkv_file = str(tmp_path / "HIMYM.S01E11.mkv")
    create_mkv_fixture(mkv_file, [
        (ASS_EN_CONTENT, "ass", "eng", False),
        (SRT_SV_CONTENT, "srt", "swe", False),
    ])

    pipeline = SubtitlePipeline()
    bazarr_mock = AsyncMock()
    translate_mock = AsyncMock()
    monkeypatch.setattr(pipeline, "trigger_bazarr_search", bazarr_mock)
    monkeypatch.setattr(pipeline.translator, "translate_srt_content", translate_mock)

    def mock_get_setting(key, default=""):
        settings = {
            "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
            "materialize_embedded_target": "true",
            "extract_target_embedded": "true",
            "extract_source_embedded": "true",
            "enable_bazarr_check": "true",
            "bazarr_url": "http://bazarr:6767",
            "bazarr_api_key": "test_key",
        }
        return settings.get(key, default)

    monkeypatch.setattr("app.services.pipeline.get_setting", mock_get_setting)
    monkeypatch.setattr("app.core.db.get_setting", mock_get_setting)
    monkeypatch.setattr("app.services.bazarr_coordinator.get_setting", mock_get_setting)

    with patch.object(bazarr_coordinator, "poll_system_jobs", new_callable=AsyncMock) as mock_poll:
        mock_poll.return_value = BazarrJobsPollResult(
            status=BazarrJobPollStatus.KNOWN_IDLE,
            jobs=[],
        )

        res = await pipeline.process_video_file(mkv_file, event_source="SONARR")

    assert res["status"] == "skipped"
    assert res["reason"] == "already_exists"

    # External file was published from embedded
    sv_published = tmp_path / "HIMYM.S01E11.sv.srt"
    assert sv_published.exists()
    cues = list(srt.parse(sv_published.read_text(encoding="utf-8-sig")))
    assert len(cues) == 5

    # Invariants
    bazarr_mock.assert_not_called()
    translate_mock.assert_not_called()
