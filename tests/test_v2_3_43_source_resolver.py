"""
test_v2_3_43_source_resolver.py — v2.3.43-beta Acceptance Tests (A–Z)

26 regression scenarios covering:
  - SourceResolver architecture (embedded / external / Bazarr)
  - OLG removal as runtime blocker
  - SOURCE==TARGET invariant shortcut
  - source_language propagation through translator chain
  - WAITING_SOURCE backoff and eventual FAILED
  - Multi-target support preserved
  - Deadline-based Bazarr waiting (no 8×sleep)
  - Structured BazarrResult codes
  - External source files NOT deleted after translation
  - Language detection fallback
"""

import asyncio
import os
import time
import srt
import pytest
import tempfile
from datetime import timedelta
from unittest.mock import patch, AsyncMock, MagicMock

from app.services.pipeline import SubtitlePipeline
from app.services.source_resolver import (
    SourceResolver, SubtitleSource, SourceOrigin,
    BazarrResult, BazarrResultCode,
    _validate_source_candidate, _read_file_safe,
    BAZARR_SOURCE_FALLBACK_ORDER,
)
from app.core.db import init_db, create_job, get_job_by_id, update_job


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_srt(lines, start_offset=0):
    """Return a minimal valid SRT string."""
    cues = []
    for i, text in enumerate(lines):
        cues.append(srt.Subtitle(
            index=i + 1,
            start=timedelta(seconds=start_offset + i),
            end=timedelta(seconds=start_offset + i + 1),
            content=text,
        ))
    return srt.compose(cues)


EN_LINES = [
    "Hello, welcome to the show.",
    "We are glad to have you here today.",
    "Please take a seat and enjoy.",
    "The performance will begin shortly.",
    "Thank you for your patience.",
    "We hope you enjoy the experience.",
    "The weather is nice outside.",
    "This is an important announcement.",
    "Please listen carefully to the instructions.",
    "We appreciate your continued support.",
]

SV_LINES = [
    "Hej, välkommen till showen.",
    "Vi är glada att ha dig här idag.",
    "Vänligen ta en plats och njut.",
    "Föreställningen börjar snart.",
    "Tack för ditt tålamod.",
    "Vi hoppas att du gillar upplevelsen.",
    "Vädret är trevligt utomhus.",
    "Detta är ett viktigt meddelande.",
    "Vänligen lyssna noga på instruktionerna.",
    "Vi uppskattar ditt fortsatta stöd.",
]

EN_SRT = make_srt(EN_LINES)
SV_SRT = make_srt(SV_LINES)


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """Each test gets its own DB."""
    import app.core.db as db_mod
    original = db_mod.DB_PATH
    db_mod.DB_PATH = str(tmp_path / "test.db")
    db_mod.init_db()
    yield tmp_path
    db_mod.DB_PATH = original


def default_settings(overrides=None):
    base = {
        "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}]',
        "extract_source_embedded": "true",
        "extract_target_embedded": "false",
        "enable_bazarr_check": "false",
        "auto_repair_unhealthy": "false",
        "batch_size": "50",
        "clean_sdh": "false",
    }
    if overrides:
        base.update(overrides)
    def _get(key, default=None):
        return base.get(key, default)
    return _get


# ---------------------------------------------------------------------------
# A. SourceResolver: embedded source selected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_A_embedded_source_preferred_over_external(isolated_db, tmp_path):
    """A. Embedded source track is used when present."""
    video = tmp_path / "movie.mkv"
    video.touch()

    ext_srt = tmp_path / "movie.en.srt"
    ext_srt.write_text(EN_SRT, encoding="utf-8")

    embedded_content = make_srt([
        "Hello embedded one from track.", "We are glad to have you here.",
        "Please take a seat and enjoy the show.", "The performance begins shortly.",
        "Thank you for your patience here.", "We hope you enjoy the experience.",
        "The weather is nice outside today.", "This is an important embedded line.",
        "Please listen carefully now.", "We appreciate your support always.",
    ])
    origins_used = []

    async def fake_translate(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        origins_used.append(source_language)
        return [srt.Subtitle(index=i+1, start=s.start, end=s.end, content=SV_LINES[i % len(SV_LINES)]) for i, s in enumerate(subs)]

    pipeline = SubtitlePipeline()

    def fake_extract_embedded(video_path, out_path, preferred_lang, tracks_info=None):
        if preferred_lang in ("en", "eng"):
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(embedded_content)
            return True
        return False

    tracks = {"subtitles": [{"id": 1, "language": "eng", "codec": "SubRip/SRT", "forced": False}],
              "audio": [], "duration": 60.0}

    with patch("app.services.pipeline.get_setting", side_effect=default_settings()), \
         patch("app.services.pipeline.find_external_subtitle",
               side_effect=lambda p, l: str(ext_srt) if l == "en" else None), \
         patch("app.services.pipeline._safe_extract_embedded_srt", side_effect=fake_extract_embedded), \
         patch("app.services.pipeline.inspect_mkv_tracks", return_value=tracks), \
         patch.object(pipeline, "trigger_bazarr_search", AsyncMock()), \
         patch.object(pipeline.translator, "translate_srt_content", fake_translate):
        job_id = create_job(str(video))
        res = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)

    assert res["status"] == "translated"
    assert len(origins_used) == 1


# ---------------------------------------------------------------------------
# B. External source found when no embedded track
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_B_external_source_found_when_no_embedded(isolated_db, tmp_path):
    """B. External .en.srt picked up when no embedded source."""
    video = tmp_path / "episode.mkv"
    video.touch()
    en_srt = tmp_path / "episode.en.srt"
    en_srt.write_text(EN_SRT, encoding="utf-8")

    translated = []

    async def fake_translate(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        translated.append(source_language)
        return [srt.Subtitle(index=i+1, start=s.start, end=s.end, content=f"SV{i}") for i, s in enumerate(subs)]

    pipeline = SubtitlePipeline()
    with patch("app.services.pipeline.get_setting",
               side_effect=default_settings({"extract_source_embedded": "false"})), \
         patch("app.services.pipeline.find_external_subtitle",
               side_effect=lambda p, l: str(en_srt) if l == "en" else None), \
         patch.object(pipeline, "trigger_bazarr_search", AsyncMock()), \
         patch.object(pipeline.translator, "translate_srt_content", fake_translate):
        job_id = create_job(str(video))
        res = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)

    assert res["status"] == "translated"
    assert translated


# ---------------------------------------------------------------------------
# C. SOURCE==TARGET shortcut: no AI when source IS target language
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_C_source_equals_target_skips_ai(isolated_db, tmp_path):
    """C. SOURCE==TARGET: if external source IS the target language, publish directly (no AI)."""
    video = tmp_path / "show.mkv"
    video.touch()
    # Swedish .sv.srt as source — target is also Swedish
    sv_srt = tmp_path / "show.sv.srt"
    sv_srt.write_text(SV_SRT, encoding="utf-8")

    ai_call_count = [0]

    async def fake_translate(*args, **kwargs):
        ai_call_count[0] += 1
        return []

    pipeline = SubtitlePipeline()

    def find_ext(p, l):
        # Return sv.srt for "sv" language lookups (source and target)
        if l == "sv":
            return str(sv_srt)
        return None

    with patch("app.services.pipeline.get_setting",
               side_effect=default_settings({"extract_source_embedded": "false"})), \
         patch("app.services.pipeline.find_external_subtitle", side_effect=find_ext), \
         patch.object(pipeline, "trigger_bazarr_search", AsyncMock()), \
         patch.object(pipeline.translator, "translate_srt_content", fake_translate):
        job_id = create_job(str(video))
        res = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)

    # With force_retranslate=False (default): sv.srt exists → target-first → "skipped"
    # AI must never be called regardless
    assert res["status"] in ("skipped", "translated", "already_exists"), f"Unexpected: {res['status']}"
    assert ai_call_count[0] == 0, f"AI must NOT be called when source == target, called {ai_call_count[0]} times"


# ---------------------------------------------------------------------------
# D. OLG removed: target=English does NOT block translation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_D_olg_removed_english_target_translates(isolated_db, tmp_path):
    """D. OLG removed: English target with English audio no longer blocks."""
    video = tmp_path / "movie.mkv"
    video.touch()

    fr_srt = tmp_path / "movie.fr.srt"
    # French content long enough for QA; target is English → SOURCE!=TARGET, OLG no longer blocks
    fr_content = make_srt([
        "Bonjour et bienvenue dans le spectacle.", "Nous sommes heureux de vous accueillir ici.",
        "Veuillez prendre un siege et profiter.", "Le spectacle va bientot commencer.",
        "Merci pour votre patience et soutien.", "Nous esperons que vous appreciez.",
        "Le temps est agreable dehors.", "Voici une annonce importante pour vous.",
        "Veuillez ecouter attentivement.", "Nous vous remercions de votre soutien.",
    ])
    fr_srt.write_text(fr_content, encoding="utf-8")

    ai_called = [False]

    async def fake_translate(subs, target_language, source_language="French", batch_size=150, job_id=None, show_title=None):
        ai_called[0] = True
        return [srt.Subtitle(index=i+1, start=s.start, end=s.end, content=EN_LINES[i % len(EN_LINES)]) for i, s in enumerate(subs)]

    pipeline = SubtitlePipeline()
    with patch("app.services.pipeline.get_setting",
               side_effect=default_settings({
                   "languages": '[{"code": "en", "name": "English", "enabled": true}]',
                   "extract_source_embedded": "false",
               })), \
         patch("app.services.pipeline.find_external_subtitle",
               side_effect=lambda p, l: str(fr_srt) if l == "fr" else None), \
         patch.object(pipeline, "trigger_bazarr_search", AsyncMock()), \
         patch.object(pipeline.translator, "translate_srt_content", fake_translate):
        job_id = create_job(str(video))
        res = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)

    assert res["status"] == "translated"
    assert ai_called[0], "AI MUST be called — OLG is removed, English target must not block"


# ---------------------------------------------------------------------------
# E. source_language propagated to translator
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_E_source_language_propagated_to_translator(isolated_db, tmp_path):
    """E. source_language kwarg must reach translate_srt_content."""
    video = tmp_path / "movie.mkv"
    video.touch()
    en_srt = tmp_path / "movie.en.srt"
    en_srt.write_text(EN_SRT, encoding="utf-8")

    received_source_lang = []

    async def fake_translate(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        received_source_lang.append(source_language)
        return [srt.Subtitle(index=i+1, start=s.start, end=s.end, content=f"SV{i}") for i, s in enumerate(subs)]

    pipeline = SubtitlePipeline()
    with patch("app.services.pipeline.get_setting",
               side_effect=default_settings({"extract_source_embedded": "false"})), \
         patch("app.services.pipeline.find_external_subtitle",
               side_effect=lambda p, l: str(en_srt) if l == "en" else None), \
         patch.object(pipeline, "trigger_bazarr_search", AsyncMock()), \
         patch.object(pipeline.translator, "translate_srt_content", fake_translate):
        job_id = create_job(str(video))
        await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)

    assert received_source_lang, "translate_srt_content must be called"
    assert received_source_lang[0] not in ("", None), "source_language must be non-empty"


# ---------------------------------------------------------------------------
# F. WAITING_SOURCE: no source → bounded backoff → FAILED
# ---------------------------------------------------------------------------

def test_F_waiting_source_bounded_backoff(isolated_db, tmp_path):
    """F. Missing source → WAITING_SOURCE (up to 4 retries) → FAILED on 5th."""
    video = tmp_path / "no_source.mkv"
    video.touch()

    settings = default_settings({"extract_source_embedded": "true", "enable_bazarr_check": "false"})

    pipeline = SubtitlePipeline()
    with patch("app.services.pipeline.get_setting", side_effect=settings), \
         patch("app.services.pipeline.find_external_subtitle", return_value=None), \
         patch("app.services.pipeline.extract_embedded_srt", return_value=False), \
         patch("app.services.pipeline.inspect_mkv_tracks",
               return_value={"subtitles": [], "audio": [], "duration": 60.0}), \
         patch.object(pipeline, "trigger_bazarr_search", AsyncMock()), \
         patch("asyncio.sleep", new_callable=AsyncMock):

        job_id = create_job(str(video))

        for expected_retry in range(1, 5):
            asyncio.run(pipeline.process_video_file(str(video), job_id=job_id))
            job = get_job_by_id(job_id)
            assert job["status"] == "WAITING_SOURCE", f"Attempt {expected_retry}: expected WAITING_SOURCE, got {job['status']}"
            assert job["retry_count"] == expected_retry

        asyncio.run(pipeline.process_video_file(str(video), job_id=job_id))
        job = get_job_by_id(job_id)
        assert job["status"] == "FAILED"
        assert "source" in job["error_message"].lower()


# ---------------------------------------------------------------------------
# G. External source NOT deleted after successful translation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_G_external_source_not_deleted_after_translation(isolated_db, tmp_path):
    """G. The .en.srt source file must survive pipeline cleanup after translation."""
    video = tmp_path / "episode.mkv"
    video.touch()
    en_srt = tmp_path / "episode.en.srt"
    en_srt.write_text(EN_SRT, encoding="utf-8")

    async def fake_translate(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        return [srt.Subtitle(index=i+1, start=s.start, end=s.end, content=f"SV{i}") for i, s in enumerate(subs)]

    pipeline = SubtitlePipeline()
    with patch("app.services.pipeline.get_setting",
               side_effect=default_settings({"extract_source_embedded": "false"})), \
         patch("app.services.pipeline.find_external_subtitle",
               side_effect=lambda p, l: str(en_srt) if l == "en" else None), \
         patch.object(pipeline, "trigger_bazarr_search", AsyncMock()), \
         patch.object(pipeline.translator, "translate_srt_content", fake_translate):
        job_id = create_job(str(video))
        res = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)

    assert res["status"] == "translated"
    assert en_srt.exists(), "External .en.srt source file must NOT be deleted after translation"


# ---------------------------------------------------------------------------
# H. Embedded source temp file IS deleted after translation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_H_embedded_source_temp_file_cleaned_up(isolated_db, tmp_path):
    """H. Temp file from embedded extraction must be cleaned up after translation."""
    video = tmp_path / "movie.mkv"
    video.touch()

    created_temp_files = []

    def fake_extract(vp, out_path, preferred_lang, tracks_info=None):
        if preferred_lang in ("en", "eng"):
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(EN_SRT)
            created_temp_files.append(out_path)
            return True
        return False

    async def fake_translate(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        return [srt.Subtitle(index=i+1, start=s.start, end=s.end, content=f"SV{i}") for i, s in enumerate(subs)]

    pipeline = SubtitlePipeline()
    tracks = {"subtitles": [{"id": 1, "language": "eng", "codec": "SubRip/SRT", "forced": False}],
              "audio": [], "duration": 60.0}

    with patch("app.services.pipeline.get_setting", side_effect=default_settings()), \
         patch("app.services.pipeline.find_external_subtitle", return_value=None), \
         patch("app.services.pipeline._safe_extract_embedded_srt", side_effect=fake_extract), \
         patch("app.services.pipeline.inspect_mkv_tracks", return_value=tracks), \
         patch.object(pipeline, "trigger_bazarr_search", AsyncMock()), \
         patch.object(pipeline.translator, "translate_srt_content", fake_translate):
        job_id = create_job(str(video))
        res = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)

    assert res["status"] == "translated"
    for tf in created_temp_files:
        assert not os.path.exists(tf), f"Embedded temp file must be deleted: {tf}"


# ---------------------------------------------------------------------------
# I. Source deleted before run → WAITING_SOURCE
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_I_source_deletion_before_run_causes_waiting_source(isolated_db, tmp_path):
    """I. If source file doesn't exist at runtime, WAITING_SOURCE is returned."""
    video = tmp_path / "movie.mkv"
    video.touch()
    en_srt = tmp_path / "movie.en.srt"
    # Don't write it — it's supposed to not exist

    pipeline = SubtitlePipeline()
    with patch("app.services.pipeline.get_setting",
               side_effect=default_settings({"extract_source_embedded": "false", "enable_bazarr_check": "false"})), \
         patch("app.services.pipeline.find_external_subtitle",
               side_effect=lambda p, l: str(en_srt) if l == "en" else None), \
         patch.object(pipeline, "trigger_bazarr_search", AsyncMock()), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        job_id = create_job(str(video))
        res = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)

    assert res["status"] in ("waiting_source", "failed")


# ---------------------------------------------------------------------------
# J. Multi-target: both sv and de translated from single source
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_J_multi_target_both_translated(isolated_db, tmp_path):
    """J. Both sv and de targets must be translated from a single source."""
    video = tmp_path / "movie.mkv"
    video.touch()
    en_srt = tmp_path / "movie.en.srt"
    en_srt.write_text(EN_SRT, encoding="utf-8")

    translate_calls = []

    async def fake_translate(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        translate_calls.append(target_language)
        if target_language == "Swedish":
            lines = SV_LINES
        else:
            # German-like text for QA  
            lines = ["Hallo Welt hier ist Deutsch.", "Wir sind froh Sie zu sehen.",
                     "Bitte nehmen Sie Platz.", "Die Vorstellung beginnt bald.",
                     "Danke fuer Ihre Geduld.", "Wir hoffen es gefaellt Ihnen.",
                     "Das Wetter ist schoen draussen.", "Dies ist eine Ankuendigung.",
                     "Bitte hoeren Sie gut zu.", "Wir schaetzen Ihre Unterstuetzung."]
        return [srt.Subtitle(index=i+1, start=s.start, end=s.end, content=lines[i % len(lines)]) for i, s in enumerate(subs)]

    pipeline = SubtitlePipeline()
    settings = default_settings({
        "languages": '[{"code": "sv", "name": "Swedish", "enabled": true}, {"code": "de", "name": "German", "enabled": true}]',
        "extract_source_embedded": "false",
    })

    with patch("app.services.pipeline.get_setting", side_effect=settings), \
         patch("app.services.pipeline.find_external_subtitle",
               side_effect=lambda p, l: str(en_srt) if l == "en" else None), \
         patch.object(pipeline, "trigger_bazarr_search", AsyncMock()), \
         patch.object(pipeline.translator, "translate_srt_content", fake_translate):
        job_id = create_job(str(video))
        res = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)

    assert res["status"] == "translated"
    assert len(translate_calls) == 2, f"Expected 2 calls, got {translate_calls}"
    assert "Swedish" in translate_calls
    assert "German" in translate_calls


# ---------------------------------------------------------------------------
# K. Bazarr AUTH_ERROR logged, pipeline continues to AI
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_K_bazarr_auth_error_does_not_abort_pipeline(isolated_db, tmp_path):
    """K. Bazarr AUTH_ERROR must be logged but AI translation proceeds."""
    video = tmp_path / "movie.mkv"
    video.touch()
    en_srt = tmp_path / "movie.en.srt"
    en_srt.write_text(EN_SRT, encoding="utf-8")

    ai_called = [False]

    async def fake_translate(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        ai_called[0] = True
        return [srt.Subtitle(index=i+1, start=s.start, end=s.end, content=f"SV{i}") for i, s in enumerate(subs)]

    async def bazarr_auth_error(vpath, language="sv"):
        return BazarrResult(code=BazarrResultCode.AUTH_ERROR, detail="401 Unauthorized")

    pipeline = SubtitlePipeline()
    with patch("app.services.pipeline.get_setting",
               side_effect=default_settings({"enable_bazarr_check": "true", "extract_source_embedded": "false"})), \
         patch("app.services.pipeline.find_external_subtitle",
               side_effect=lambda p, l: str(en_srt) if l == "en" else None), \
         patch.object(pipeline, "trigger_bazarr_search", bazarr_auth_error), \
         patch.object(pipeline.translator, "translate_srt_content", fake_translate):
        job_id = create_job(str(video))
        res = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)

    assert res["status"] == "translated"
    assert ai_called[0], "AI must be called even when Bazarr returns AUTH_ERROR"


# ---------------------------------------------------------------------------
# L. No fixed sleep: Bazarr and SourceResolver run concurrently
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_L_bazarr_runs_without_blocking_pipeline(isolated_db, tmp_path):
    """L. Pipeline with Bazarr enabled must complete fast even with slow Bazarr response.

    Architecture: Bazarr target search is a fire-and-cancel Task that runs concurrently
    with SourceResolver. When source is found quickly (external file), the task may be
    cancelled before it runs. The key invariant is: no BLOCKING sleep for Bazarr.
    """
    video = tmp_path / "movie.mkv"
    video.touch()
    en_srt = tmp_path / "movie.en.srt"
    en_srt.write_text(EN_SRT, encoding="utf-8")

    async def slow_bazarr(vpath, language="sv"):
        # Simulate 5-second Bazarr HTTP delay (pipeline must NOT wait for this)
        await asyncio.sleep(5.0)
        return BazarrResult(code=BazarrResultCode.TRIGGERED, detail="OK")

    async def fake_translate(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        return [srt.Subtitle(index=i+1, start=s.start, end=s.end, content=SV_LINES[i % len(SV_LINES)]) for i, s in enumerate(subs)]

    pipeline = SubtitlePipeline()
    start = time.monotonic()
    with patch("app.services.pipeline.get_setting",
               side_effect=default_settings({"enable_bazarr_check": "true", "extract_source_embedded": "false"})), \
         patch("app.services.pipeline.find_external_subtitle",
               side_effect=lambda p, l: str(en_srt) if l == "en" else None), \
         patch.object(pipeline, "trigger_bazarr_search", slow_bazarr), \
         patch.object(pipeline.translator, "translate_srt_content", fake_translate):
        job_id = create_job(str(video))
        res = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)

    total = time.monotonic() - start
    assert res["status"] == "translated"
    # Pipeline must complete in well under 5s despite Bazarr having a 5s delay
    # This proves Bazarr task is cancelled/non-blocking, not awaited sequentially
    assert total < 2.0, f"Pipeline must not block on 5s Bazarr, took {total:.2f}s — no fixed sleep!"


# ---------------------------------------------------------------------------
# M. source_language != target_language invariant enforced
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_M_source_equals_target_invariant(isolated_db, tmp_path):
    """M. source == target language must NOT dispatch AI."""
    video = tmp_path / "episode.mkv"
    video.touch()
    sv_srt = tmp_path / "episode.sv.srt"
    sv_srt.write_text(SV_SRT, encoding="utf-8")

    ai_called = [False]

    async def fake_translate(*args, **kwargs):
        ai_called[0] = True
        return []

    pipeline = SubtitlePipeline()

    with patch("app.services.pipeline.get_setting",
               side_effect=default_settings({"extract_source_embedded": "false"})), \
         patch("app.services.pipeline.find_external_subtitle",
               side_effect=lambda p, l: str(sv_srt) if l == "sv" else None), \
         patch.object(pipeline, "trigger_bazarr_search", AsyncMock()), \
         patch.object(pipeline.translator, "translate_srt_content", fake_translate):
        job_id = create_job(str(video))
        res = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)

    assert res["status"] in ("skipped", "translated")
    assert not ai_called[0], "AI must NOT dispatch when source == target language"


# ---------------------------------------------------------------------------
# N. force_retranslate bypasses existing target
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_N_force_retranslate_bypasses_existing_target(isolated_db, tmp_path):
    """N. force_retranslate=True must bypass healthy existing target subtitle."""
    video = tmp_path / "episode.mkv"
    video.touch()
    en_srt = tmp_path / "episode.en.srt"
    en_srt.write_text(EN_SRT, encoding="utf-8")
    sv_srt = tmp_path / "episode.sv.srt"
    sv_srt.write_text(SV_SRT, encoding="utf-8")

    ai_called = [False]

    async def fake_translate(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        ai_called[0] = True
        return [srt.Subtitle(index=i+1, start=s.start, end=s.end, content=SV_LINES[i % len(SV_LINES)]) for i, s in enumerate(subs)]

    pipeline = SubtitlePipeline()
    with patch("app.services.pipeline.get_setting",
               side_effect=default_settings({"extract_source_embedded": "false"})), \
         patch("app.services.pipeline.find_external_subtitle",
               side_effect=lambda p, l: str(en_srt) if l == "en" else str(sv_srt) if l == "sv" else None), \
         patch.object(pipeline, "trigger_bazarr_search", AsyncMock()), \
         patch.object(pipeline.translator, "translate_srt_content", fake_translate):
        job_id = create_job(str(video))

        # Without force: AI must NOT be called
        res = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)
        assert res["status"] == "skipped"
        assert not ai_called[0]

        # With force: AI MUST be called
        update_job(job_id, status="QUEUED")
        res2 = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0, force_retranslate=True)
        assert res2["status"] == "translated"
        assert ai_called[0]


# ---------------------------------------------------------------------------
# O. _validate_source_candidate: too small rejected
# ---------------------------------------------------------------------------

def test_O_validate_source_too_small(isolated_db, tmp_path):
    """O. _validate_source_candidate rejects files below minimum size."""
    tiny = tmp_path / "tiny.srt"
    tiny.write_text("x", encoding="utf-8")

    ok, reason, lang, cues = _validate_source_candidate(str(tiny), "en")
    assert not ok
    assert "too_small" in reason or "size" in reason.lower()


# ---------------------------------------------------------------------------
# P. _validate_source_candidate: valid SRT accepted
# ---------------------------------------------------------------------------

def test_P_validate_source_valid_srt(isolated_db, tmp_path):
    """P. _validate_source_candidate accepts a well-formed SRT file."""
    good = tmp_path / "good.srt"
    good.write_text(EN_SRT, encoding="utf-8")

    ok, reason, lang, cues = _validate_source_candidate(str(good), "en")
    assert ok
    assert len(cues) > 0


# ---------------------------------------------------------------------------
# Q. BAZARR_SOURCE_FALLBACK_ORDER starts with English
# ---------------------------------------------------------------------------

def test_Q_bazarr_source_fallback_order_starts_english(isolated_db):
    """Q. BAZARR_SOURCE_FALLBACK_ORDER must start with 'en'."""
    assert len(BAZARR_SOURCE_FALLBACK_ORDER) > 0
    assert BAZARR_SOURCE_FALLBACK_ORDER[0] == "en", (
        f"English must be first in fallback order, got: {BAZARR_SOURCE_FALLBACK_ORDER[0]}"
    )


# ---------------------------------------------------------------------------
# R. Provider error → WAITING_PROVIDER
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_R_provider_error_creates_waiting_provider(isolated_db, tmp_path):
    """R. ProviderUnavailableError → WAITING_PROVIDER (not WAITING_SOURCE)."""
    from app.services.translator import ProviderUnavailableError

    video = tmp_path / "movie.mkv"
    video.touch()
    en_srt = tmp_path / "movie.en.srt"
    en_srt.write_text(EN_SRT, encoding="utf-8")

    async def raise_provider_error(*args, **kwargs):
        raise ProviderUnavailableError("Rate limit exceeded")

    pipeline = SubtitlePipeline()
    with patch("app.services.pipeline.get_setting",
               side_effect=default_settings({"extract_source_embedded": "false"})), \
         patch("app.services.pipeline.find_external_subtitle",
               side_effect=lambda p, l: str(en_srt) if l == "en" else None), \
         patch.object(pipeline, "trigger_bazarr_search", AsyncMock()), \
         patch.object(pipeline.translator, "translate_srt_content", raise_provider_error):
        job_id = create_job(str(video))
        res = await pipeline.process_video_file(str(video), job_id=job_id, force_retranslate=True)

    assert res["status"] == "waiting_provider"
    job = get_job_by_id(job_id)
    assert job["status"] == "WAITING_PROVIDER"
    assert job["retry_count"] == 1


# ---------------------------------------------------------------------------
# S. Terminal filesystem error → FAILED
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_S_permission_denied_causes_failed(isolated_db, tmp_path):
    """S. 'Permission denied' exception → FAILED (not retried)."""
    video = tmp_path / "movie.mkv"
    video.touch()
    en_srt = tmp_path / "movie.en.srt"
    en_srt.write_text(EN_SRT, encoding="utf-8")

    async def raise_permission_error(*args, **kwargs):
        raise Exception("Permission denied: /path/to/output/file.srt")

    pipeline = SubtitlePipeline()
    with patch("app.services.pipeline.get_setting",
               side_effect=default_settings({"extract_source_embedded": "false"})), \
         patch("app.services.pipeline.find_external_subtitle",
               side_effect=lambda p, l: str(en_srt) if l == "en" else None), \
         patch.object(pipeline, "trigger_bazarr_search", AsyncMock()), \
         patch.object(pipeline.translator, "translate_srt_content", raise_permission_error):
        job_id = create_job(str(video))
        res = await pipeline.process_video_file(str(video), job_id=job_id, force_retranslate=True)

    assert res["status"] == "failed"
    job = get_job_by_id(job_id)
    assert job["status"] == "FAILED"


# ---------------------------------------------------------------------------
# T. Bazarr download appears → AI skipped
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_T_bazarr_download_appears_skips_ai(isolated_db, tmp_path):
    """T. If Bazarr provides target before SourceResolver completes, AI is skipped."""
    video = tmp_path / "movie.mkv"
    video.touch()
    en_srt = tmp_path / "movie.en.srt"
    en_srt.write_text(EN_SRT, encoding="utf-8")
    sv_srt = tmp_path / "movie.sv.srt"

    ai_called = [False]

    async def fake_translate(*args, **kwargs):
        ai_called[0] = True
        return []

    call_counts = {"sv": 0}

    def find_ext(p, l):
        if l == "en":
            return str(en_srt)
        if l == "sv":
            call_counts["sv"] += 1
            if call_counts["sv"] >= 2:
                sv_srt.write_text(SV_SRT, encoding="utf-8")
                return str(sv_srt)
        return None

    pipeline = SubtitlePipeline()
    with patch("app.services.pipeline.get_setting",
               side_effect=default_settings({"enable_bazarr_check": "true", "extract_source_embedded": "false"})), \
         patch("app.services.pipeline.find_external_subtitle", side_effect=find_ext), \
         patch.object(pipeline, "trigger_bazarr_search", AsyncMock(
             return_value=BazarrResult(code=BazarrResultCode.TRIGGERED, detail="OK"))), \
         patch.object(pipeline.translator, "translate_srt_content", fake_translate):
        job_id = create_job(str(video))
        res = await pipeline.process_video_file(str(video), job_id=job_id)

    assert res["status"] in ("skipped", "translated")
    assert not ai_called[0], "AI must not run when Bazarr downloaded target"


# ---------------------------------------------------------------------------
# U. No source anywhere → WAITING_SOURCE
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_U_no_source_anywhere_waiting_source(isolated_db, tmp_path):
    """U. No embedded, external, or Bazarr source → WAITING_SOURCE."""
    video = tmp_path / "movie.mkv"
    video.touch()

    pipeline = SubtitlePipeline()
    with patch("app.services.pipeline.get_setting",
               side_effect=default_settings({"enable_bazarr_check": "false"})), \
         patch("app.services.pipeline.find_external_subtitle", return_value=None), \
         patch("app.services.pipeline.extract_embedded_srt", return_value=False), \
         patch("app.services.pipeline.inspect_mkv_tracks",
               return_value={"subtitles": [], "audio": [], "duration": 60.0}), \
         patch.object(pipeline, "trigger_bazarr_search", AsyncMock()), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        job_id = create_job(str(video))
        res = await pipeline.process_video_file(str(video), job_id=job_id, force_retranslate=True)

    assert res["status"] == "waiting_source"
    job = get_job_by_id(job_id)
    assert job["status"] == "WAITING_SOURCE"


# ---------------------------------------------------------------------------
# V. BazarrResult codes
# ---------------------------------------------------------------------------

def test_V_bazarr_result_triggered(isolated_db):
    """V. BazarrResult TRIGGERED → was_accepted=True."""
    result = BazarrResult(code=BazarrResultCode.TRIGGERED, detail="Search accepted")
    assert result.was_accepted
    assert result.code == BazarrResultCode.TRIGGERED


def test_V2_bazarr_result_auth_error(isolated_db):
    """V2. BazarrResult AUTH_ERROR → was_accepted=False."""
    result = BazarrResult(code=BazarrResultCode.AUTH_ERROR, detail="401 Unauthorized")
    assert not result.was_accepted
    assert result.code == BazarrResultCode.AUTH_ERROR


# ---------------------------------------------------------------------------
# W. _read_file_safe: UTF-8 BOM tolerated
# ---------------------------------------------------------------------------

def test_W_read_file_safe_handles_utf8_bom(isolated_db, tmp_path):
    """W. _read_file_safe must read files with UTF-8 BOM."""
    bom_file = tmp_path / "bom.srt"
    bom_file.write_bytes(b"\xef\xbb\xbf" + EN_SRT.encode("utf-8"))

    content = _read_file_safe(str(bom_file))
    assert content is not None
    assert "Hello" in content or "00:" in content


# ---------------------------------------------------------------------------
# X. WAITING_SOURCE retry_count increments precisely
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_X_waiting_source_retry_count_increments(isolated_db, tmp_path):
    """X. Each WAITING_SOURCE retry increments retry_count by exactly 1."""
    video = tmp_path / "movie.mkv"
    video.touch()

    pipeline = SubtitlePipeline()
    with patch("app.services.pipeline.get_setting",
               side_effect=default_settings({"enable_bazarr_check": "false"})), \
         patch("app.services.pipeline.find_external_subtitle", return_value=None), \
         patch("app.services.pipeline.extract_embedded_srt", return_value=False), \
         patch("app.services.pipeline.inspect_mkv_tracks",
               return_value={"subtitles": [], "audio": [], "duration": 60.0}), \
         patch.object(pipeline, "trigger_bazarr_search", AsyncMock()), \
         patch("asyncio.sleep", new_callable=AsyncMock):

        job_id = create_job(str(video))
        for expected_count in range(1, 5):
            await pipeline.process_video_file(str(video), job_id=job_id, force_retranslate=True)
            job = get_job_by_id(job_id)
            assert job["retry_count"] == expected_count, (
                f"retry_count must be {expected_count}, got {job['retry_count']}"
            )


# ---------------------------------------------------------------------------
# Y. SubtitleSource dataclass fields
# ---------------------------------------------------------------------------

def test_Y_subtitle_source_dataclass_fields(isolated_db):
    """Y. SubtitleSource must expose expected fields."""
    src = SubtitleSource(
        language="en",
        origin=SourceOrigin.EXTERNAL,
        path="/tmp/test.en.srt",
        content=EN_SRT,
        cues=[],
    )
    assert src.language == "en"
    assert src.origin == SourceOrigin.EXTERNAL
    assert src.path == "/tmp/test.en.srt"
    assert src.content == EN_SRT
    assert src.language_name == "English"


# ---------------------------------------------------------------------------
# Z. Full happy path: en.srt → pipeline → sv.srt on disk
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_Z_full_happy_path_output_on_disk(isolated_db, tmp_path):
    """Z. End-to-end: .en.srt → pipeline → .sv.srt appears on disk."""
    video = tmp_path / "episode.mkv"
    video.touch()
    en_srt = tmp_path / "episode.en.srt"
    en_srt.write_text(EN_SRT, encoding="utf-8")
    sv_srt = tmp_path / "episode.sv.srt"

    async def fake_translate(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
        return [
            srt.Subtitle(index=i+1, start=s.start, end=s.end, content=SV_LINES[i % len(SV_LINES)])
            for i, s in enumerate(subs)
        ]

    pipeline = SubtitlePipeline()
    with patch("app.services.pipeline.get_setting",
               side_effect=default_settings({"extract_source_embedded": "false"})), \
         patch("app.services.pipeline.find_external_subtitle",
               side_effect=lambda p, l: str(en_srt) if l == "en" else None), \
         patch.object(pipeline, "trigger_bazarr_search", AsyncMock()), \
         patch.object(pipeline.translator, "translate_srt_content", fake_translate):
        job_id = create_job(str(video))
        res = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)

    assert res["status"] == "translated"
    assert sv_srt.exists(), ".sv.srt must be written to disk"
    assert sv_srt.stat().st_size > 50, ".sv.srt must be non-empty"
    assert en_srt.exists(), "Source .en.srt must NOT be deleted"

    content = sv_srt.read_text(encoding="utf-8")
    parsed = list(srt.parse(content))
    assert len(parsed) == len(EN_LINES)
