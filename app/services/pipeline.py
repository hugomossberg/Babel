import asyncio
import os
import time
import json
import logging
from typing import Dict, Any, Optional, List, Set
import srt
import httpx

from app.core.cleaner import sanitize_srt_content, subs_to_srt_string
from app.core.extractor import extract_embedded_srt, inspect_mkv_tracks
from app.core.validator import verify_sync, check_dropped_lines, evaluate_subtitle_health, detect_language_heuristics
from app.core.db import create_job, update_job, append_job_log, get_setting
from app.services.bazarr_checker import check_existing_swedish_subtitle, check_existing_english_subtitle, find_external_subtitle
from app.services.translator import SubtitleTranslator, is_usable_translation, ProviderUnavailableError, get_provider_capabilities
from app.services.jellyfin_notifier import notify_jellyfin_library_refresh

logger = logging.getLogger("babel.pipeline")


# ---------------------------------------------------------------------------
# BABEL QA GATE — The most important function in the entire project.
# A translated file is NEVER published unless it passes every check.
# ---------------------------------------------------------------------------
def qa_gate(
    source_subs: list,
    translated_subs: list,
    target_lang_code: str,
    job_id: Optional[int] = None,
    safe_ids: Optional[list] = None
) -> Dict[str, Any]:
    """
    Final Quality Assurance gate. Returns a dict with:
      - passed: bool
      - score: int (0-100)
      - issues: list of strings describing problems
      - untranslated_ids: list of subtitle indices where original text was kept
    """
    issues = []
    untranslated_ids = []
    safe_ids = safe_ids or []
    score = 100

    # 1. Line count must match exactly
    if len(translated_subs) != len(source_subs):
        issues.append(f"Line count mismatch: source={len(source_subs)}, translated={len(translated_subs)}")
        score -= 50

    # 2. Check for untranslated lines (original English still present)
    min_len = min(len(source_subs), len(translated_subs))
    for i in range(min_len):
        orig = source_subs[i].content.strip()
        trans = translated_subs[i].content.strip()

        # Skip empty placeholders
        if not orig or orig == "<i></i>":
            continue

        # If translated text is identical to original AND original looks like English
        if trans == orig:
            untranslated_ids.append(i)

    def is_safe_identical_line(text: str) -> bool:
        stripped = text.strip()
        # siffror / symboler
        if not any(c.isalpha() for c in stripped):
            return True
        return False

    real_untranslated_ids = [
        i for i in untranslated_ids
        if i not in safe_ids and not is_safe_identical_line(source_subs[i].content)
    ]

    if real_untranslated_ids:
        pct = round(len(real_untranslated_ids) / min_len * 100, 1)
        issues.append(f"{len(real_untranslated_ids)} lines ({pct}%) still contain original English text")
        # Small number is warning, large number is failure
        if pct > 5.0:
            score -= 40
        elif pct > 1.0:
            score -= 20
        else:
            score -= 5

    # 3. Check for completely empty translations (dropped lines)
    dropped_count, dropped_details = check_dropped_lines(source_subs, translated_subs)
    if dropped_count > 0:
        pct = round(dropped_count / len(source_subs) * 100, 1)
        issues.append(f"{dropped_count} lines ({pct}%) were dropped (empty in translation)")
        if pct > 2.0:
            score -= 30
        else:
            score -= 10

    # 4. Verify sync (every cue must match)
    sync_report = verify_sync(source_subs, translated_subs)
    max_drift = max(sync_report.get("start_diff_ms", 0), sync_report.get("end_diff_ms", 0))
    if max_drift > 0:
        issues.append(f"Timestamp drift detected: {max_drift}ms")
        if max_drift > 500:
            score -= 30
        elif max_drift > 50:
            score -= 10

    # 5. Målspråkskontroll (Grov)
    wrong_language = False
    if translated_subs:
        sample_text = " ".join([s.content for s in translated_subs[:80] if s.content.strip() and s.content.strip() != "<i></i>"])
        if len(sample_text) >= 20:
            lang_info = detect_language_heuristics(sample_text)
            detected = lang_info["lang"]
            if detected != "unknown" and detected != target_lang_code[:2].lower() and lang_info["confidence"] > 0.8:
                wrong_language = True
                issues.append(f"Language mismatch: expected {target_lang_code}, detected {detected} ({lang_info['confidence']*100:.0f}% confidence)")
                score -= 30

    # 6. Valid SRT structure
    structure_valid = True
    try:
        srt_text = subs_to_srt_string(translated_subs)
        reparsed = list(srt.parse(srt_text))
        if len(reparsed) != len(translated_subs):
            issues.append(f"SRT re-parse mismatch: wrote {len(translated_subs)}, re-parsed {len(reparsed)}")
            score -= 50
            structure_valid = False
    except Exception as e:
        issues.append(f"Invalid SRT structure: {e}")
        score -= 50
        structure_valid = False

    score = max(0, score)

    sync_valid = max_drift == 0
    passed = (
        score >= 60
        and dropped_count == 0
        and len(real_untranslated_ids) == 0
        and sync_valid
        and not wrong_language
        and structure_valid
        and len(translated_subs) == len(source_subs)
    )

    return {
        "passed": passed,
        "score": score,
        "issues": issues,
        "untranslated_ids": untranslated_ids,  # Keep full list for recovery attempts
        "real_untranslated_ids": real_untranslated_ids,
        "dropped_count": dropped_count,
        "dropped_details": dropped_details,
        "sync_diff_ms": max_drift,
    }


class SubtitlePipeline:
    def __init__(self):
        self.translator = SubtitleTranslator()
        self._video_semaphore = None
        self._current_max_jobs = 1
        self._active_tasks: Dict[int, asyncio.Task] = {}
        # Bug #17: Per-video locking to prevent duplicate processing
        self._active_video_paths: Set[str] = set()
        self._video_lock = asyncio.Lock()

    def cancel_job(self, job_id: int):
        if job_id in self._active_tasks:
            task = self._active_tasks[job_id]
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled active task for job_id={job_id}")

        # Clean up partial progress file
        import os
        import glob
        from app.core.db import DB_PATH
        data_dir = os.path.dirname(DB_PATH)
        for partial_file in glob.glob(os.path.join(data_dir, f"job_{job_id}_*_partial.json")):
            try:
                os.remove(partial_file)
            except Exception:
                pass

    def _get_semaphore(self) -> asyncio.Semaphore:
        max_jobs = int(get_setting("max_concurrent_jobs", "1"))
        if self._video_semaphore is None or self._current_max_jobs != max_jobs:
            self._current_max_jobs = max_jobs
            self._video_semaphore = asyncio.Semaphore(max_jobs)
        return self._video_semaphore

    def get_configured_languages(self) -> List[Dict[str, Any]]:
        raw = get_setting("languages", "[]")
        try:
            langs = json.loads(raw)
            return [l for l in langs if l.get("enabled", True)]
        except Exception:
            return [{"name": "Swedish", "code": "sv", "enabled": True}]

    # Bug #3: Accept language parameter instead of hardcoded "sv"
    async def trigger_bazarr_search(self, video_path: str, language: str = "sv"):
        bazarr_url = get_setting("bazarr_url", "http://bazarr:6767").rstrip("/")
        bazarr_api_key = get_setting("bazarr_api_key", "")
        if not bazarr_api_key:
            return

        headers = {"X-API-KEY": bazarr_api_key}
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                # Bug #4: Use Bazarr API properly — search movies first (simpler),
                # then series. For series, we match by video path in the movies/episodes list.

                # 1. Try Movies
                m_res = await client.get(f"{bazarr_url}/api/movies", headers=headers)
                if m_res.status_code == 200:
                    movies = m_res.json().get("data", [])
                    norm_target = os.path.normpath(video_path)
                    for m in movies:
                        m_path = os.path.normpath(m.get("path", ""))
                        if m_path == norm_target or os.path.basename(m_path) == os.path.basename(norm_target):
                            r_id = m.get("radarrId")
                            if r_id:
                                logger.info(f"Triggering Bazarr movie subtitle search for radarrId {r_id}, lang={language}")
                                await client.patch(
                                    f"{bazarr_url}/api/movies/subtitles",
                                    headers=headers,
                                    params={
                                        "radarrid": r_id,
                                        "language": language,
                                        "forced": "False",
                                        "hi": "False"
                                    }
                                )
                                return

                # 2. Try Series — get all series first, then find episodes for matching series
                s_res = await client.get(f"{bazarr_url}/api/series", headers=headers)
                if s_res.status_code == 200:
                    all_series = s_res.json().get("data", [])
                    norm_target = os.path.normpath(video_path)

                    for series in all_series:
                        series_path = os.path.normpath(series.get("path", ""))
                        # Check if the video path starts with this series' path
                        if norm_target.startswith(series_path):
                            s_id = series.get("sonarrSeriesId")
                            if not s_id:
                                continue
                            # Get episodes for this specific series
                            ep_res = await client.get(
                                f"{bazarr_url}/api/episodes",
                                headers=headers,
                                params={"seriesid[]": s_id}
                            )
                            if ep_res.status_code == 200:
                                episodes = ep_res.json().get("data", [])
                                for ep in episodes:
                                    ep_path = os.path.normpath(ep.get("path", ""))
                                    if ep_path == norm_target or os.path.basename(ep_path) == os.path.basename(norm_target):
                                        e_id = ep.get("sonarrEpisodeId")
                                        if e_id:
                                            logger.info(f"Triggering Bazarr episode subtitle search for series {s_id}, episode {e_id}, lang={language}")
                                            await client.patch(
                                                f"{bazarr_url}/api/episodes/subtitles",
                                                headers=headers,
                                                params={
                                                    "seriesid": s_id,
                                                    "episodeid": e_id,
                                                    "language": language,
                                                    "forced": "False",
                                                    "hi": "False"
                                                }
                                            )
                                            return
                            break  # Found the right series, no need to continue
            except Exception as e:
                logger.warning(f"Failed to trigger Bazarr search: {e}")

    async def process_video_file(
        self,
        video_path: str,
        wait_seconds: Optional[int] = None,
        event_source: str = "MANUAL",
        title: Optional[str] = None,
        force_retranslate: bool = False,
        job_id: Optional[int] = None
    ) -> Dict[str, Any]:
        # Bug #17: Prevent duplicate processing of the same video
        async with self._video_lock:
            norm_path = os.path.normpath(video_path)
            if norm_path in self._active_video_paths:
                logger.warning(f"Skipping duplicate request for {norm_path} — already being processed")
                if job_id:
                    # Job was already claimed but we can't process it. Revert to RECOVERING to try again later.
                    from app.core.db import update_job
                    from datetime import datetime, timezone, timedelta
                    next_retry = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
                    update_job(job_id, status="RECOVERING", next_retry_at=next_retry)
                return {"status": "skipped", "reason": "already_processing", "video_path": norm_path}
            self._active_video_paths.add(norm_path)

        if job_id is None:
            job_id = create_job(video_path=video_path, event_source=event_source, title=title)

        current_task = asyncio.current_task()
        if current_task:
            self._active_tasks[job_id] = current_task

        try:
            return await self._execute_process_video(
                job_id=job_id,
                video_path=video_path,
                wait_seconds=wait_seconds,
                event_source=event_source,
                force_retranslate=force_retranslate,
                title=title
            )
        except asyncio.CancelledError:
            logger.info(f"Job {job_id} was cancelled by user.")
            base_path, _ = os.path.splitext(video_path)
            temp_extracted_srt = f"{base_path}.temp_extracted.en.srt"
            if os.path.exists(temp_extracted_srt):
                try: os.remove(temp_extracted_srt)
                except: pass
            update_job(job_id, status="CANCELLED", error_message="Cancelled by user")
            append_job_log(job_id, "Job was cancelled by user. Stopped AI translation.")
            raise
        finally:
            self._active_tasks.pop(job_id, None)
            async with self._video_lock:
                self._active_video_paths.discard(os.path.normpath(video_path))

    async def _execute_process_video(
        self,
        job_id: int,
        video_path: str,
        wait_seconds: Optional[int] = None,
        event_source: str = "MANUAL",
        force_retranslate: bool = False,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        sem = self._get_semaphore()
        async with sem:
            return await self._run_pipeline_logic(
                job_id=job_id,
                video_path=video_path,
                wait_seconds=wait_seconds,
                event_source=event_source,
                force_retranslate=force_retranslate,
                title=title
            )

    async def _maybe_notify_jellyfin(self):
        """Bug #21: Only notify Jellyfin if the setting is enabled."""
        if get_setting("notify_jellyfin", "true").lower() == "true":
            await notify_jellyfin_library_refresh()

    async def _run_pipeline_logic(
        self,
        job_id: int,
        video_path: str,
        wait_seconds: Optional[int] = None,
        event_source: str = "MANUAL",
        force_retranslate: bool = False,
        title: Optional[str] = None
    ) -> Dict[str, Any]:

        enable_bazarr = get_setting("enable_bazarr_check", "true").lower() == "true"
        extract_target_embedded = get_setting("extract_target_embedded", "true").lower() == "true"
        extract_source_embedded = get_setting("extract_source_embedded", "true").lower() == "true"
        auto_repair_unhealthy = get_setting("auto_repair_unhealthy", "true").lower() == "true"
        strict_sync_lock = get_setting("strict_sync_lock", "true").lower() == "true"

        start_time = time.time()
        update_job(job_id, status="TRANSLATING")
        append_job_log(job_id, f"Processing file: {video_path}")

        if not os.path.exists(video_path):
            err = f"File not found: {video_path}"
            append_job_log(job_id, f"ERROR: {err}")
            update_job(job_id, status="FAILED", error_message=err, duration_seconds=round(time.time() - start_time, 2))
            return {"status": "error", "message": err, "job_id": job_id}

        target_languages = self.get_configured_languages()
        if not target_languages:
            append_job_log(job_id, "No target language configured.")
            update_job(job_id, status="ACTION_REQUIRED")
            return {"status": "action_required", "reason": "no_target_languages"}

        base_path, _ = os.path.splitext(video_path)

        # -------------------------------------------------------------
        # Bug #11: Reordered pipeline — check LOCAL targets FIRST
        #
        # Order:
        #   1. External target on disk
        #   2. Embedded target in video
        #   3. Bazarr target search
        #   4. (If no target found) Resolve source → AI translate
        # -------------------------------------------------------------

        # Track which languages still need translation
        langs_needing_translation = []

        if not force_retranslate:
            for lang_info in target_languages:
                lang_code = lang_info["code"]
                lang_name = lang_info["name"]
                target_output_path = f"{base_path}.{lang_code}.srt"

                # Bug #14: Use generalized find_external_subtitle
                existing_target = find_external_subtitle(video_path, lang_code)
                if existing_target:
                    if auto_repair_unhealthy:
                        health = evaluate_subtitle_health(existing_target, target_lang_code=lang_code)
                        if health.get("status") == "RED":
                            append_job_log(job_id, f"Existing {lang_name} subtitle found but unhealthy ({health['reason']}). Will re-translate.")
                            langs_needing_translation.append(lang_info)
                            continue
                    append_job_log(job_id, f"Healthy {lang_name} subtitle already exists: {os.path.basename(existing_target)}. Skipping.")
                    continue

                # Bug #13: Don't return after first embedded target — continue loop
                if extract_target_embedded:
                    if not os.path.exists(target_output_path):
                        temp_target_path = f"{target_output_path}.tmp_embed"
                        extracted_target = extract_embedded_srt(video_path, temp_target_path, preferred_lang=lang_code)
                        if extracted_target and os.path.exists(temp_target_path):
                            # Always validate extracted embedded target, regardless of auto_repair setting
                            health = evaluate_subtitle_health(temp_target_path, target_lang_code=lang_code)
                            status = health.get("status", "UNKNOWN")

                            if status == "GREEN":
                                os.replace(temp_target_path, target_output_path)
                                append_job_log(job_id, f"Extracted healthy embedded {lang_name} track to {os.path.basename(target_output_path)}.")
                                continue
                            elif status == "YELLOW":
                                append_job_log(job_id, f"Extracted embedded {lang_name} track is YELLOW ({health.get('reason')}). Queuing for deeper QA validation.")
                                try:
                                    os.remove(temp_target_path)
                                except Exception:
                                    pass
                                # Fall through to append to langs_needing_translation
                            else:
                                append_job_log(job_id, f"Extracted embedded {lang_name} track rejected: {status} ({health.get('reason', 'Unknown error')}).")
                                try:
                                    os.remove(temp_target_path)
                                except Exception:
                                    pass
                                # Fall through to append to langs_needing_translation

                # This language needs translation
                langs_needing_translation.append(lang_info)
        else:
            langs_needing_translation = list(target_languages)

        # If all languages are covered, we're done
        if not langs_needing_translation:
            duration = round(time.time() - start_time, 2)
            update_job(job_id, status="ALREADY EXISTS", reason="All target subtitles already exist", duration_seconds=duration)
            return {"status": "skipped", "reason": "already_exists", "job_id": job_id}

        # -------------------------------------------------------------
        # HYBRID MODE: Check Bazarr for human target subtitles
        # -------------------------------------------------------------
        if enable_bazarr and not force_retranslate:
            bazarr_wait = wait_seconds if wait_seconds is not None else int(get_setting("wait_time_seconds", "15"))
            if bazarr_wait > 0:
                # Trigger Bazarr search for each missing target language
                for lang_info in langs_needing_translation:
                    append_job_log(job_id, f"Hybrid Mode: Triggering Bazarr search for {lang_info['name']} ({lang_info['code']})...")
                    await self.trigger_bazarr_search(video_path, language=lang_info["code"])

                # Wait and poll for results
                elapsed = 0
                while elapsed < bazarr_wait:
                    await asyncio.sleep(2)
                    elapsed += 2

                    still_missing = []
                    for lang_info in langs_needing_translation:
                        existing = find_external_subtitle(video_path, lang_info["code"])
                        if existing:
                            if auto_repair_unhealthy:
                                health = evaluate_subtitle_health(existing, target_lang_code=lang_info["code"])
                                if health.get("status") == "RED":
                                    still_missing.append(lang_info)
                                    continue
                            append_job_log(job_id, f"Bazarr found {lang_info['name']} subtitle: {os.path.basename(existing)}")
                        else:
                            still_missing.append(lang_info)

                    langs_needing_translation = still_missing
                    if not langs_needing_translation:
                        break

                if not langs_needing_translation:
                    duration = round(time.time() - start_time, 2)
                    update_job(job_id, status="BAZARR MATCH", reason="Bazarr found all target subtitles", duration_seconds=duration)
                    await self._maybe_notify_jellyfin()
                    return {"status": "skipped", "reason": "bazarr_downloaded", "job_id": job_id}

                append_job_log(job_id, f"Bazarr grace period ended ({bazarr_wait}s). {len(langs_needing_translation)} language(s) still need AI translation.")

        # -------------------------------------------------------------
        # LOCATE SOURCE SUBTITLE for AI translation
        # Priority: 1. Embedded English, 2. External English, 3. Bazarr English fallback
        # -------------------------------------------------------------
        raw_srt_text = ""
        temp_extracted_srt = f"{base_path}.temp_extracted.en.srt"

        # PRIORITY 1: Embedded English inside Video (Best sync guarantee)
        if extract_source_embedded:
            append_job_log(job_id, "Checking video container for embedded English track (Priority 1: Best Sync)...")
            extracted = extract_embedded_srt(video_path, temp_extracted_srt, preferred_lang="eng")
            if extracted and os.path.exists(temp_extracted_srt):
                append_job_log(job_id, "Successfully extracted embedded English track from video.")
                try:
                    with open(temp_extracted_srt, "r", encoding="utf-8-sig") as f:
                        raw_srt_text = f.read()
                except UnicodeDecodeError:
                    with open(temp_extracted_srt, "r", encoding="windows-1252") as f:
                        raw_srt_text = f.read()

        # PRIORITY 2: External English subtitle on disk
        if not raw_srt_text:
            en_srt_source = find_external_subtitle(video_path, "en")
            if en_srt_source and os.path.exists(en_srt_source):
                append_job_log(job_id, f"Found external source subtitle: {os.path.basename(en_srt_source)}")
                try:
                    with open(en_srt_source, "r", encoding="utf-8-sig") as f:
                        raw_srt_text = f.read()
                except UnicodeDecodeError:
                    with open(en_srt_source, "r", encoding="windows-1252") as f:
                        raw_srt_text = f.read()

        # PRIORITY 3: Bazarr Safety Net — ask Bazarr for English subtitle
        if not raw_srt_text and enable_bazarr:
            append_job_log(job_id, "No embedded or external English sub found. Querying Bazarr for English subtitle...")
            await self.trigger_bazarr_search(video_path, language="en")

            for _ in range(8):
                await asyncio.sleep(2)
                en_srt_source = find_external_subtitle(video_path, "en")
                if en_srt_source:
                    append_job_log(job_id, f"Bazarr retrieved English subtitle: {os.path.basename(en_srt_source)}")
                    try:
                        with open(en_srt_source, "r", encoding="utf-8-sig") as f:
                            raw_srt_text = f.read()
                    except UnicodeDecodeError:
                        with open(en_srt_source, "r", encoding="windows-1252") as f:
                            raw_srt_text = f.read()
                    break

        if not raw_srt_text:
            err = "No English subtitle source found (neither embedded, external nor via Bazarr safety net)"
            append_job_log(job_id, f"ERROR: {err}")
            duration = round(time.time() - start_time, 2)
            update_job(job_id, status="FAILED", error_message=err, duration_seconds=duration)
            if os.path.exists(temp_extracted_srt):
                try: os.remove(temp_extracted_srt)
                except: pass
            return {"status": "failed", "reason": err, "job_id": job_id}

        try:
            # -------------------------------------------------------------
            # PRE-PROCESSING & SDH CLEANER
            # -------------------------------------------------------------
            clean_sdh_enabled = get_setting("clean_sdh", "true").lower() == "true"
            if clean_sdh_enabled:
                subs, cleaned_count = sanitize_srt_content(raw_srt_text)
                append_job_log(job_id, f"Sanitizer processed {len(subs)} blocks. Cleaned noise on {cleaned_count} blocks.")
            else:
                subs = list(srt.parse(raw_srt_text))
                cleaned_count = 0
                append_job_log(job_id, f"SDH cleaner disabled. Parsed {len(subs)} blocks directly.")

            total_source_lines = len(subs)
            batch_size = int(get_setting("batch_size", "50"))
            ai_provider = get_setting("ai_provider", "gemini").lower()
            if ai_provider == "openai":
                active_engine_name = f"OpenAI ({get_setting('openai_model', 'gpt-4o-mini')})"
            elif ai_provider == "deepl":
                active_engine_name = "DeepL Translate"
            elif ai_provider in ["ollama", "localai"]:
                active_engine_name = f"Ollama ({get_setting('ollama_model', 'llama3')})"
            else:
                active_engine_name = f"Gemini ({get_setting('gemini_model', 'gemini-3.5-flash-lite')})"

            update_job(job_id, total_lines=total_source_lines, processed_lines=0, current_batch="Starting...")
            output_files = []
            successful_langs = []
            skipped_langs = []
            total_dropped = 0
            max_sync_diff = 0

            # -------------------------------------------------------------
            # TRANSLATION & QUALITY ASSURANCE
            # -------------------------------------------------------------

            # Bug #20: Respect Original Language Guard toggle
            original_language_guard = get_setting("original_language_guard", "true").lower() == "true"

            primary_audio_lang = "und"
            if original_language_guard:
                mkv_info = inspect_mkv_tracks(video_path)
                # Bug #32: Better audio track detection — prefer default, skip forced/commentary
                for audio in mkv_info.get("audio", []):
                    audio_title = (audio.get("title") or "").lower()
                    if any(bad in audio_title for bad in ["commentary", "director", "description"]):
                        continue
                    if audio.get("default") and not audio.get("forced"):
                        primary_audio_lang = audio.get("language", "und").lower()
                        break
                if primary_audio_lang == "und" and mkv_info.get("audio"):
                    # Fallback: first non-forced, non-commentary audio
                    for audio in mkv_info.get("audio", []):
                        audio_title = (audio.get("title") or "").lower()
                        if not audio.get("forced") and not any(bad in audio_title for bad in ["commentary", "director"]):
                            primary_audio_lang = audio.get("language", "und").lower()
                            break

            for lang_info in langs_needing_translation:
                lang_name = lang_info["name"]
                lang_code = lang_info["code"]
                target_output_path = f"{base_path}.{lang_code}.srt"
                safe_ids = []

                # Check Original Language Guard
                if original_language_guard:
                    from app.core.languages import get_language
                    lang_obj = get_language(lang_code)
                    protected_langs = lang_obj.aliases if lang_obj else [lang_code]

                    if primary_audio_lang in protected_langs:
                        append_job_log(job_id, f"Original Language Guard: Primary audio is '{primary_audio_lang}'. Skipping translation to {lang_name}.")
                        # Bug #31: Don't count guard-skipped as "successful translation"
                        skipped_langs.append(lang_name)
                        continue

                # Mid-job target race protection (Before Translation)
                existing_before_trans = find_external_subtitle(video_path, lang_code)
                if existing_before_trans:
                    health = evaluate_subtitle_health(existing_before_trans, target_lang_code=lang_code)
                    if health.get("status") == "GREEN":
                        append_job_log(job_id, "Target appeared while job was running. Skipping AI translation.")
                        successful_langs.append(lang_code)
                        continue

                append_job_log(job_id, f"Translating to {lang_name} ({lang_code}) using {active_engine_name} ({total_source_lines} lines)...")

                t_main_start = time.time()
                translated_subs = await self.translator.translate_srt_content(
                    subs=subs,
                    target_language=lang_name,
                    batch_size=batch_size,
                    job_id=job_id,
                    show_title=title
                )
                t_main_end = time.time()
                append_job_log(job_id, f"Timing: Main translation phase completed in {round(t_main_end - t_main_start, 1)}s")

                # --- NEVER GIVE UP RECOVERY LOOP ---
                max_qa_loops = 3
                qa_loop_count = 0
                known_untranslated_ids = set()
                exhausted_strategies = set()
                recovered_cues = set()

                while qa_loop_count < max_qa_loops:
                    qa_loop_count += 1

                    if qa_loop_count > 1:
                        append_job_log(job_id, f"--- QA RECOVERY LOOP {qa_loop_count}/{max_qa_loops} ---")


                    # Strict Sync Lock: Match source timestamps exactly to guarantee 0ms drift
                    if strict_sync_lock and len(translated_subs) == len(subs):
                        for idx in range(len(subs)):
                            translated_subs[idx].start = subs[idx].start
                            translated_subs[idx].end = subs[idx].end

                    # -------------------------------------------------------
                    # Bug #1, #5: FINAL QA GATE — never publish a broken file
                    # -------------------------------------------------------
                    qa_result = qa_gate(subs, translated_subs, target_lang_code=lang_code, job_id=job_id, safe_ids=safe_ids)

                    if qa_loop_count == 1:
                        initial_candidates_count = len(qa_result.get("untranslated_ids", []))

                    if qa_result["passed"]:
                        break # Success! Exit the recovery loop.

                    # Attempt recovery for any untranslated or dropped lines
                    if qa_result["untranslated_ids"] or qa_result.get("dropped_count", 0) > 0:
                        try:
                            # 1. Primary Recovery for Identical lines (English still present)
                            if qa_result["untranslated_ids"]:
                                identical_ids = qa_result["untranslated_ids"]
                                append_job_log(job_id, f"QA: PRIMARY_RECOVERY ({len(identical_ids)} identical candidates need review)")
                                recovery_payload = [
                                    {"id": idx, "text": subs[idx].content}
                                    for idx in identical_ids
                                    if subs[idx].content.strip() and subs[idx].content.strip() != "<i></i>" and idx not in safe_ids and idx not in known_untranslated_ids
                                ]
                                if recovery_payload:
                                    provider = get_setting("ai_provider", "gemini").lower()
                                    caps = get_provider_capabilities(provider)
                                    if caps["supports_identical_classification"]:
                                        recovery_results = []
                                        chunk_size = 20
                                        for i in range(0, len(recovery_payload), chunk_size):
                                            chunk = recovery_payload[i:i + chunk_size]
                                            try:
                                                chunk_res = await self.translator.classify_and_recover_identical(
                                                    chunk, lang_name, title or ""
                                                )
                                                recovery_results.extend(chunk_res)
                                            except Exception as e:
                                                append_job_log(job_id, f"QA Primary Recovery chunk failed: {e}")
                                                continue

                                        for r in recovery_results:
                                            idx = r.get("id")
                                            if idx is None: continue
                                            action = r.get("action")
                                            if action == "keep":
                                                safe_ids.append(idx)
                                                raw_reason = r.get("reason", "none").lower()
                                                reason_map = {
                                                    "proper_noun": "Name / Proper Noun",
                                                    "brand": "Brand Name",
                                                    "acronym": "Acronym / Abbreviation",
                                                    "number": "Number",
                                                    "symbol": "Symbol / Punctuation",
                                                    "non_verbal": "Non-verbal Sound",
                                                    "none": "Unspecified Reason"
                                                }
                                                reason_str = reason_map.get(raw_reason, raw_reason.replace("_", " ").title())
                                                append_job_log(job_id, f"QA Recovery: Model kept line {idx} ({reason_str})")
                                            elif action == "translate" and "text" in r:
                                                if not is_usable_translation(r["text"]):
                                                    append_job_log(job_id, f"QA Recovery: Rejected blank/invalid translation for line {idx}")
                                                elif r["text"] != subs[idx].content:
                                                    translated_subs[idx].content = r["text"]
                                                    append_job_log(job_id, f"QA Recovery: Translated line {idx}")
                                                    recovered_cues.add(idx)
                                    else:
                                        # Deterministic fallback for DeepL
                                        recovery_results = []
                                        chunk_size = 20
                                        for i in range(0, len(recovery_payload), chunk_size):
                                            chunk = recovery_payload[i:i + chunk_size]
                                            try:
                                                chunk_res = await self.translator.translate_batch(
                                                    chunk, target_language=lang_name, show_title=title or ""
                                                )
                                                recovery_results.extend(chunk_res)
                                            except Exception as e:
                                                append_job_log(job_id, f"QA Recovery chunk failed (DeepL): {e}")
                                                continue

                                        for r in recovery_results:
                                            idx = r.get("id")
                                            if idx is None: continue
                                            text = r.get("text", "")
                                            if is_usable_translation(text) and text != subs[idx].content and text != translated_subs[idx].content:
                                                translated_subs[idx].content = text
                                                append_job_log(job_id, f"QA Recovery: Translated line {idx}")
                                                recovered_cues.add(idx)

                            # Re-run QA after primary recovery with safe_ids context
                            qa_result = qa_gate(subs, translated_subs, target_lang_code=lang_code, job_id=job_id, safe_ids=safe_ids)
                            if qa_result["passed"]:
                                break

                            # 2. Escalation Stage: Contextual Single-Line Recovery (Dropped + Unresolved Identical)
                            real_unresolved = qa_result.get("real_untranslated_ids", [])
                            dropped_unresolved = [d["index"] - 1 for d in qa_result.get("dropped_details", [])]
                            all_unresolved = list(set(real_unresolved + dropped_unresolved))
                            all_unresolved.sort()


                            if all_unresolved:
                                # Add to known untranslated so they don't get classified again
                                for uid in all_unresolved:
                                    known_untranslated_ids.add(uid)

                                append_job_log(job_id, f"Targeted Recovery: translating {len(all_unresolved)} unresolved dialogue cues")
                                batch_payload = [{"id": idx, "text": subs[idx].content} for idx in all_unresolved]

                                try:
                                    targeted_results = await self.translator.translate_batch(batch_payload, target_language=lang_name, show_title=title or "")
                                    targeted_success = 0
                                    for r in targeted_results:
                                        idx = r.get("id")
                                        if idx is None: continue
                                        text = r.get("text", "")
                                        if is_usable_translation(text) and text != subs[idx].content:
                                            translated_subs[idx].content = text
                                            targeted_success += 1
                                    append_job_log(job_id, f"Targeted Recovery: translated {targeted_success}/{len(all_unresolved)}")
                                except Exception as e:
                                    append_job_log(job_id, f"Targeted Recovery failed: {e}")

                                # Re-run QA again to get the remaining stubborn cues
                                qa_result = qa_gate(subs, translated_subs, target_lang_code=lang_code, job_id=job_id, safe_ids=safe_ids)
                                real_unresolved = qa_result.get("real_untranslated_ids", [])
                                dropped_unresolved = [d["index"] - 1 for d in qa_result.get("dropped_details", [])]
                                all_unresolved = list(set(real_unresolved + dropped_unresolved))
                                all_unresolved.sort()

                            if all_unresolved:
                                esc_enabled = get_setting("escalate_to_pro", "false").lower() == "true"

                                esc_prov = get_setting("escalation_provider", "none")
                                esc_mod = get_setting("escalation_model", "")
                                if esc_enabled and esc_prov != "none" and esc_mod:
                                    esc_info = f"{esc_prov} / {esc_mod}"
                                else:
                                    esc_info = "Primary Model (Contextual Mode)"
                                append_job_log(job_id, f"Escalation Stage: {len(all_unresolved)} lines still unresolved. Escalating using {esc_info}...")
                                esc_sem = asyncio.Semaphore(3)
                                async def escalate_one(idx):
                                    prev_idx = max(0, idx - 1)
                                    next_idx = min(len(subs) - 1, idx + 1)
                                    prev_text = translated_subs[prev_idx].content if prev_idx != idx else ""
                                    next_text = subs[next_idx].content if next_idx != idx else ""
                                    target_text = subs[idx].content
                                    async with esc_sem:
                                        try:
                                            esc_text = await self.translator.escalate_single_line(
                                                idx, target_text, prev_text, next_text, lang_name, title or "",
                                                is_real_untranslated=(idx in real_unresolved),
                                                job_id=job_id,
                                                exhausted_strategies=exhausted_strategies
                                            )
                                            is_dropped = idx in dropped_unresolved
                                            if esc_text:
                                                esc_clean = esc_text.strip()
                                                orig_clean = target_text.strip()
                                                is_orig_real = orig_clean and orig_clean != "<i></i>"
                                                is_esc_empty = not esc_clean or esc_clean == "<i></i>"
                                                if not esc_clean:
                                                    append_job_log(job_id, f"Escalation: Rejected empty text for line {idx}")
                                                elif is_orig_real and is_esc_empty:
                                                    append_job_log(job_id, f"Escalation: Rejected fake empty/tag for real dialogue at line {idx}")
                                                elif is_dropped and esc_clean == orig_clean:
                                                    append_job_log(job_id, f"Escalation: Rejected identical fallback for dropped line {idx}")
                                                elif esc_text != translated_subs[idx].content:
                                                    translated_subs[idx].content = esc_text
                                                    append_job_log(job_id, f"Escalation: Translated line {idx} using dialogue context")
                                                    recovered_cues.add(idx)
                                        except Exception as e:
                                            append_job_log(job_id, f"Escalation failed for line {idx}: {e}")

                                esc_tasks = [escalate_one(idx) for idx in all_unresolved]
                                if esc_tasks:
                                    t_esc_start = time.time()
                                    await asyncio.gather(*esc_tasks)
                                    t_esc_end = time.time()
                                    append_job_log(job_id, f"Timing: Escalation phase ({len(esc_tasks)} cues) completed in {round(t_esc_end - t_esc_start, 1)}s")

                                # Final QA rerun after escalation
                                qa_result = qa_gate(subs, translated_subs, target_lang_code=lang_code, job_id=job_id, safe_ids=safe_ids)
                                if qa_result["passed"]:
                                    break
                                else:
                                    # Clear partial state so next loop does a clean re-translate of failed batches
                                    from app.core.db import DB_PATH
                                    data_dir = os.path.dirname(DB_PATH)
                                    partial_file = os.path.join(data_dir, f"job_{job_id}_{lang_code}_partial.json") if job_id else None
                                    if partial_file and os.path.exists(partial_file):
                                        try:
                                            with open(partial_file, "r", encoding="utf-8") as f:
                                                pdata = json.load(f)
                                            plines = pdata.get("lines", {})
                                            for uid in all_unresolved:
                                                if str(uid) in plines:
                                                    del plines[str(uid)]
                                            tmp_p = partial_file + ".tmp"
                                            with open(tmp_p, "w", encoding="utf-8") as f:
                                                json.dump(pdata, f, ensure_ascii=False)
                                            os.replace(tmp_p, partial_file)
                                        except Exception as e:
                                            append_job_log(job_id, f"Failed to clean partial state: {e}")

                                    # To prevent getting completely stuck in a loop, throw an error if this was the last loop
                                    if qa_loop_count == max_qa_loops:
                                        append_job_log(job_id, f"QA loop exhausted ({max_qa_loops} attempts). Stopping recovery loop for {lang_name}.")

                        except Exception as e:
                            append_job_log(job_id, f"QA Recovery/Escalation failed: {e}")
                # Final QA Log
                score = qa_result["score"]
                if qa_result["passed"]:
                    append_job_log(job_id, f"QA Gate PASSED (Score: {score}/100)")
                else:
                    issues_str = "; ".join(qa_result.get("issues", []))
                    append_job_log(job_id, f"QA Gate FAILED (Score: {score}/100) — Issues: {issues_str}")

                # Bug #7: Track both start and end diff
                sync_report = verify_sync(subs, translated_subs)
                dropped_count, _ = check_dropped_lines(subs, translated_subs)
                total_dropped += dropped_count
                this_max_diff = max(sync_report.get("start_diff_ms", 0), sync_report.get("end_diff_ms", 0))
                if this_max_diff > max_sync_diff:
                    max_sync_diff = this_max_diff

                if qa_result["passed"]:
                    # Target Race Protection (Before Publish)
                    existing = find_external_subtitle(video_path, lang_code)
                    skip_publish = False
                    if existing and not force_retranslate:
                        health = evaluate_subtitle_health(existing, target_lang_code=lang_code)
                        if health.get("status") == "GREEN":
                            append_job_log(job_id, "External target appeared before publish. Babel output not published.")
                            successful_langs.append(lang_code)
                            skip_publish = True
                        else:
                            backup_path = existing + ".babel-replaced"
                            try:
                                os.rename(existing, backup_path)
                                append_job_log(job_id, f"Renamed existing subtitle to {os.path.basename(backup_path)}")
                            except Exception:
                                pass

                    if not skip_publish:
                        temp_output = target_output_path + ".tmp"
                        with open(temp_output, "w", encoding="utf-8") as f:
                            f.write(subs_to_srt_string(translated_subs))
                        try:
                            os.chmod(temp_output, 0o666)
                        except Exception as e:
                            logger.warning(f"Could not set permissions for {temp_output}: {e}")

                        # Atomic replace to prevent partial reads by media server
                        os.replace(temp_output, target_output_path)
                        output_files.append(target_output_path)
                        successful_langs.append(lang_code)

                    # Clean up partial progress file since we successfully finished
                    from app.core.db import DB_PATH
                    data_dir = os.path.dirname(DB_PATH)
                    partial_file = os.path.join(data_dir, f"job_{job_id}_{lang_code}_partial.json") if job_id else None
                    if partial_file and os.path.exists(partial_file):
                        try:
                            os.remove(partial_file)
                        except Exception:
                            pass

                    if not skip_publish:
                        # Save Translation Memory only after QA PASS and if we actually published
                        if title:
                            try:
                                from app.core.db import save_translation_memory_bulk
                                tm_items = []
                                for idx in range(len(subs)):
                                    orig_t = subs[idx].content.strip()
                                    trans_t = translated_subs[idx].content.strip()
                                    if orig_t and trans_t and orig_t != "<i></i>" and trans_t != "<i></i>":
                                        tm_items.append({"original": orig_t, "translated": trans_t})
                                save_translation_memory_bulk(title, tm_items)
                            except Exception as e:
                                logger.error(f"Failed to save translation memory: {e}")

                # Build a detailed QA summary for the user
                unresolved_count = len(qa_result.get("real_untranslated_ids", []))
                identical_candidates = initial_candidates_count if 'initial_candidates_count' in locals() else 0
                kept_count = len(safe_ids) if 'safe_ids' in locals() else 0
                recovered_count = len(recovered_cues) if 'recovered_cues' in locals() else 0

                if qa_result["passed"]:
                    summary_status = "PASS (Tolerated)" if unresolved_count > 0 else "PASS"
                else:
                    summary_status = "FAIL"
                summary_lines = [
                    "--- QA Summary ---",
                    f"{dropped_count} dropped lines",
                    f"{this_max_diff} ms sync drift",
                    f"{identical_candidates} identical candidates",
                    f"{kept_count} classified as KEEP (safe)",
                    f"{recovered_count} translated on recovery",
                    f"{unresolved_count} unresolved English lines",
                    f"Result: {summary_status} (Score: {qa_result['score']}/100)",
                    "------------------"
                ]
                for line in summary_lines:
                    append_job_log(job_id, line)

                if qa_result["passed"]:
                    append_job_log(job_id, f"Published {os.path.basename(target_output_path)}")
                else:
                    append_job_log(job_id, f"BLOCKED: {lang_name} translation failed QA. File NOT published.")

            # Clean up temp file
            if os.path.exists(temp_extracted_srt):
                try: os.remove(temp_extracted_srt)
                except: pass

            total_duration = round(time.time() - start_time, 2)

            # Bug #31: Determine correct final status
            # If we didn't translate all requested languages because of a QA failure,
            # this is a recoverable state! We should go to RECOVERING, not FAILED.
            # Bug #31: Determine correct final status
            if len(successful_langs) + len(skipped_langs) == len(langs_needing_translation):
                if successful_langs:
                    final_status = "TRANSLATED"
                else:
                    final_status = "SKIPPED"
            elif successful_langs:
                final_status = "PARTIAL"
            else:
                final_status = "RECOVERING"

            update_args = {
                "status": final_status,
                "target_languages": ",".join(successful_langs),
                "total_lines": total_source_lines,
                "cleaned_sdh_lines": cleaned_count,
                "dropped_lines": total_dropped,
                "sync_diff_ms": max_sync_diff,
                "output_files": json.dumps(output_files),
                "duration_seconds": total_duration
            }
            if final_status in ["RECOVERING", "PARTIAL"]:
                from datetime import datetime, timezone, timedelta
                from app.core.db import get_job_by_id
                job_data = get_job_by_id(job_id)
                current_retries = job_data.get("retry_count", 0) if job_data else 0

                backoff_mins = 1
                if current_retries == 1: backoff_mins = 5
                elif current_retries == 2: backoff_mins = 15
                elif current_retries == 3: backoff_mins = 30
                elif current_retries >= 4: backoff_mins = 60

                update_args["next_retry_at"] = (datetime.now(timezone.utc) + timedelta(minutes=backoff_mins)).isoformat()
                update_args["retry_count"] = current_retries + 1
                append_job_log(job_id, f"Job needs recovery. Will resume in {backoff_mins} min (Worker attempt {current_retries + 1}).")

            update_job(job_id, **update_args)

            await self._maybe_notify_jellyfin()

            return {
                "status": final_status.lower(),
                "job_id": job_id,
                "duration": total_duration,
                "output_files": output_files
            }

        except ProviderUnavailableError as e:
            if os.path.exists(temp_extracted_srt):
                try: os.remove(temp_extracted_srt)
                except: pass
            total_duration = round(time.time() - start_time, 2)

            from app.core.db import get_job_by_id
            from datetime import datetime, timezone, timedelta

            job_data = get_job_by_id(job_id)
            current_retries = job_data.get("retry_count", 0) if job_data else 0

            # Simple backoff: 2 minutes, then 5, then 15, max 30.
            backoff_mins = 2
            if current_retries == 1: backoff_mins = 5
            elif current_retries == 2: backoff_mins = 15
            elif current_retries > 2: backoff_mins = 30

            now = datetime.now(timezone.utc)
            next_retry_at = (now + timedelta(minutes=backoff_mins)).isoformat()

            append_job_log(job_id, f"PROVIDER ERROR: {str(e)}. Will retry at {next_retry_at} (Retry {current_retries + 1}).")
            update_job(
                job_id,
                status="WAITING_PROVIDER",
                error_message=str(e),
                duration_seconds=total_duration,
                retry_count=current_retries + 1,
                next_retry_at=next_retry_at,
                last_error=str(e)
            )
            return {"status": "waiting_provider", "error": str(e), "job_id": job_id}
        except Exception as e:
            if os.path.exists(temp_extracted_srt):
                try: os.remove(temp_extracted_srt)
                except: pass
            total_duration = round(time.time() - start_time, 2)

            err_str = str(e).lower()
            if any(t in err_str for t in ["timeout", "429", "500", "502", "503", "504", "connection"]):
                from datetime import datetime, timezone, timedelta
                next_retry_at = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
                append_job_log(job_id, f"NETWORK/PROVIDER ERROR: {str(e)}. Retrying later.")
                update_job(job_id, status="WAITING_PROVIDER", error_message=str(e), duration_seconds=total_duration, next_retry_at=next_retry_at)
                return {"status": "waiting_provider", "error": str(e), "job_id": job_id}

            elif any(t in err_str for t in ["permission denied", "no such file", "not found", "read-only"]):
                append_job_log(job_id, f"TERMINAL FILESYSTEM ERROR: {str(e)}")
                update_job(job_id, status="FAILED", error_message=str(e), duration_seconds=total_duration)
                return {"status": "failed", "error": str(e), "job_id": job_id}

            else:
                from datetime import datetime, timezone, timedelta
                next_retry_at = (datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat()
                append_job_log(job_id, f"UNEXPECTED ERROR: {str(e)}. Treating as recoverable.")
                update_job(job_id, status="RETRY_PENDING", error_message=str(e), duration_seconds=total_duration, next_retry_at=next_retry_at)
                return {"status": "retry_pending", "error": str(e), "job_id": job_id}

pipeline = SubtitlePipeline()
