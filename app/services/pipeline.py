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
from app.services.translator import SubtitleTranslator, get_provider_capabilities
from app.services.jellyfin_notifier import notify_jellyfin_library_refresh

logger = logging.getLogger("babel.pipeline")


# ---------------------------------------------------------------------------
# BABEL QA GATE — The most important function in the entire project.
# A translated file is NEVER published unless it passes every check.
# ---------------------------------------------------------------------------
def qa_gate(
    source_subs: list,
    translated_subs: list,
    target_lang_code: str = "sv",
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

    # 5. Language detection on translated output
    wrong_language = False
    if translated_subs:
        sample_text = " ".join([s.content for s in translated_subs[:80] if s.content.strip() and s.content.strip() != "<i></i>"])
        if sample_text:
            detected = detect_language_heuristics(sample_text)
            if detected == "en" and target_lang_code != "en":
                wrong_language = True
                issues.append(f"Target language detection failed: output appears to be English, expected {target_lang_code}")
                score -= 30

    # 6. Valid SRT structure
    try:
        srt_text = subs_to_srt_string(translated_subs)
        reparsed = list(srt.parse(srt_text))
        if len(reparsed) != len(translated_subs):
            issues.append(f"SRT re-parse mismatch: wrote {len(translated_subs)}, re-parsed {len(reparsed)}")
            score -= 20
    except Exception as e:
        issues.append(f"Invalid SRT structure: {e}")
        score -= 30

    score = max(0, score)
    # Feature: Tolerate up to 3 unresolved English lines to avoid failing 99% perfect jobs
    # Bug fix: Enforce hard fail for sync drift and wrong language
    sync_valid = max_drift == 0
    passed = (
        score >= 60 
        and dropped_count == 0 
        and len(real_untranslated_ids) == 0
        and sync_valid
        and not wrong_language
    )
    
    return {
        "passed": passed,
        "score": score,
        "issues": issues,
        "untranslated_ids": untranslated_ids,  # Keep full list for recovery attempts
        "real_untranslated_ids": real_untranslated_ids,
        "dropped_count": dropped_count,
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
        force_retranslate: bool = False
    ) -> Dict[str, Any]:
        # Bug #17: Prevent duplicate processing of the same video
        async with self._video_lock:
            norm_path = os.path.normpath(video_path)
            if norm_path in self._active_video_paths:
                logger.warning(f"Skipping duplicate request for {norm_path} — already being processed")
                return {"status": "skipped", "reason": "already_processing", "video_path": norm_path}
            self._active_video_paths.add(norm_path)

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
            target_languages = [{"name": "Swedish", "code": "sv", "enabled": True}]

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
                        extracted_target = extract_embedded_srt(video_path, target_output_path, preferred_lang=lang_code)
                        if extracted_target and os.path.exists(target_output_path):
                            append_job_log(job_id, f"Extracted embedded {lang_name} track to {os.path.basename(target_output_path)}.")
                            continue

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
                
            AUDIO_LANG_MAP = {
                "sv": ["swe", "sve", "sv"],
                "da": ["dan", "da"],
                "no": ["nor", "nob", "nno", "no"],
                "fi": ["fin", "fi"],
                "de": ["ger", "deu", "de"],
                "es": ["spa", "es"],
                "fr": ["fre", "fra", "fr"],
                "en": ["eng", "en"]
            }

            for lang_info in langs_needing_translation:
                lang_name = lang_info["name"]
                lang_code = lang_info["code"]
                target_output_path = f"{base_path}.{lang_code}.srt"
                safe_ids = []
                
                # Check Original Language Guard
                if original_language_guard:
                    protected_langs = AUDIO_LANG_MAP.get(lang_code, [lang_code])
                    if primary_audio_lang in protected_langs:
                        append_job_log(job_id, f"Original Language Guard: Primary audio is '{primary_audio_lang}'. Skipping translation to {lang_name}.")
                        # Bug #31: Don't count guard-skipped as "successful translation"
                        skipped_langs.append(lang_name)
                        continue

                append_job_log(job_id, f"Translating to {lang_name} ({lang_code}) using {active_engine_name} ({total_source_lines} lines)...")

                translated_subs = await self.translator.translate_srt_content(
                    subs=subs,
                    target_language=lang_name,
                    batch_size=batch_size,
                    job_id=job_id,
                    show_title=title
                )

                # Strict Sync Lock: Match source timestamps exactly to guarantee 0ms drift
                if strict_sync_lock and len(translated_subs) == len(subs):
                    for idx in range(len(subs)):
                        translated_subs[idx].start = subs[idx].start
                        translated_subs[idx].end = subs[idx].end

                # -------------------------------------------------------
                # Bug #1, #5: FINAL QA GATE — never publish a broken file
                # -------------------------------------------------------
                qa_result = qa_gate(subs, translated_subs, target_lang_code=lang_code, job_id=job_id)
                initial_candidates_count = len(qa_result.get("untranslated_ids", []))

                # Attempt recovery for any untranslated lines
                if qa_result["untranslated_ids"]:
                    append_job_log(job_id, f"Initial QA: RECOVERY_REQUIRED ({initial_candidates_count} candidates need review)")
                    recovery_payload = [
                        {"id": idx, "text": subs[idx].content}
                        for idx in qa_result["untranslated_ids"]
                        if subs[idx].content.strip() and subs[idx].content.strip() != "<i></i>"
                    ]
                    if recovery_payload:
                        try:
                            provider = get_setting("ai_provider", "gemini").lower()
                            caps = get_provider_capabilities(provider)
                            if caps["supports_identical_classification"]:
                                recovery_results = []
                                chunk_size = 20
                                for i in range(0, len(recovery_payload), chunk_size):
                                    chunk = recovery_payload[i:i + chunk_size]
                                    chunk_res = await self.translator.classify_and_recover_identical(
                                        chunk, lang_name, title or ""
                                    )
                                    recovery_results.extend(chunk_res)
                                
                                for r in recovery_results:
                                    idx = r.get("id")
                                    if idx is None: continue
                                    action = r.get("action")
                                    if action == "keep":
                                        safe_ids.append(idx)
                                        append_job_log(job_id, f"QA Recovery: Model kept line {idx} ({r.get('reason', 'no reason')})")
                                    elif action == "translate" and "text" in r:
                                        if r["text"] != subs[idx].content:
                                            translated_subs[idx].content = r["text"]
                                            append_job_log(job_id, f"QA Recovery: Translated line {idx}")
                            else:
                                # Deterministic fallback for DeepL
                                recovery_results = []
                                chunk_size = 20
                                for i in range(0, len(recovery_payload), chunk_size):
                                    chunk = recovery_payload[i:i + chunk_size]
                                    chunk_res = await self.translator.translate_batch(
                                        chunk, target_language=lang_name, show_title=title or ""
                                    )
                                    recovery_results.extend(chunk_res)
                                
                                res_dict = {r["id"]: r["text"] for r in recovery_results if "id" in r and "text" in r}
                                for idx, text in res_dict.items():
                                    if text != subs[idx].content:  # actually changed
                                        translated_subs[idx].content = text
                                        append_job_log(job_id, f"QA Recovery: Translated line {idx} (DeepL Fallback)")
                            
                            # Re-run QA after recovery with safe_ids context
                            qa_result = qa_gate(subs, translated_subs, target_lang_code=lang_code, job_id=job_id, safe_ids=safe_ids)
                        except Exception as e:
                            append_job_log(job_id, f"QA Recovery failed: {e}")
                
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
                    # Bug #16: Rename conflicting old subs instead of leaving duplicates
                    existing = find_external_subtitle(video_path, lang_code)
                    if existing and not force_retranslate:
                        backup_path = existing + ".babel-replaced"
                        try:
                            os.rename(existing, backup_path)
                            append_job_log(job_id, f"Renamed existing subtitle to {os.path.basename(backup_path)}")
                        except Exception:
                            pass

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

                # Build a detailed QA summary for the user
                unresolved_count = len(qa_result.get("real_untranslated_ids", []))
                identical_candidates = initial_candidates_count if 'initial_candidates_count' in locals() else 0
                kept_count = len(safe_ids) if 'safe_ids' in locals() else 0
                recovered_count = identical_candidates - kept_count - unresolved_count
                if recovered_count < 0: recovered_count = 0

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
            if len(successful_langs) + len(skipped_langs) == len(langs_needing_translation):
                if successful_langs:
                    final_status = "TRANSLATED"
                else:
                    final_status = "SKIPPED"
            elif successful_langs:
                final_status = "PARTIAL"
            else:
                final_status = "FAILED"

            update_job(
                job_id,
                status=final_status,
                target_languages=",".join(successful_langs),
                total_lines=total_source_lines,
                cleaned_sdh_lines=cleaned_count,
                dropped_lines=total_dropped,
                sync_diff_ms=max_sync_diff,
                output_files=json.dumps(output_files),
                duration_seconds=total_duration
            )

            await self._maybe_notify_jellyfin()

            return {
                "status": "success" if successful_langs else "failed",
                "job_id": job_id,
                "duration": total_duration,
                "output_files": output_files
            }

        except Exception as e:
            if os.path.exists(temp_extracted_srt):
                try: os.remove(temp_extracted_srt)
                except: pass
            total_duration = round(time.time() - start_time, 2)
            append_job_log(job_id, f"FATAL ERROR: {str(e)}")
            update_job(job_id, status="FAILED", error_message=str(e), duration_seconds=total_duration)
            return {"status": "failed", "error": str(e), "job_id": job_id}

pipeline = SubtitlePipeline()
