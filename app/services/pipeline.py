import asyncio
import os
import time
import json
import logging
from typing import Dict, Any, Optional, List
import srt
import httpx

from app.core.cleaner import sanitize_srt_content, subs_to_srt_string
from app.core.extractor import extract_embedded_srt, inspect_mkv_tracks
from app.core.validator import verify_sync, check_dropped_lines, evaluate_subtitle_health
from app.core.db import create_job, update_job, append_job_log, get_setting
from app.services.bazarr_checker import check_existing_swedish_subtitle, check_existing_english_subtitle
from app.services.translator import SubtitleTranslator
from app.services.jellyfin_notifier import notify_jellyfin_library_refresh

logger = logging.getLogger("babel.pipeline")

class SubtitlePipeline:
    def __init__(self):
        self.translator = SubtitleTranslator()
        self._video_semaphore = None
        self._current_max_jobs = 1
        self._active_tasks: Dict[int, asyncio.Task] = {}

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

    async def trigger_bazarr_search(self, video_path: str):
        bazarr_url = get_setting("bazarr_url", "http://dev-bazarr:6767").rstrip("/")
        bazarr_api_key = get_setting("bazarr_api_key", "")
        if not bazarr_api_key:
            return
        
        headers = {"X-API-KEY": bazarr_api_key}
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                # 1. Check if it matches an Episode in Bazarr
                res = await client.get(f"{bazarr_url}/api/episodes", headers=headers)
                if res.status_code == 200:
                    episodes = res.json().get("data", [])
                    norm_target = os.path.normpath(video_path)
                    for ep in episodes:
                        ep_path = os.path.normpath(ep.get("path", ""))
                        if ep_path == norm_target or os.path.basename(ep_path) == os.path.basename(norm_target):
                            s_id = ep.get("sonarrSeriesId")
                            e_id = ep.get("sonarrEpisodeId")
                            if s_id and e_id:
                                logger.info(f"Triggering Bazarr episode subtitle search for series {s_id}, episode {e_id}")
                                await client.patch(
                                    f"{bazarr_url}/api/episodes/subtitles",
                                    headers=headers,
                                    params={
                                        "seriesid": s_id,
                                        "episodeid": e_id,
                                        "language": "sv",
                                        "forced": "False",
                                        "hi": "False"
                                    }
                                )
                                return

                # 2. Check if it matches a Movie in Bazarr
                m_res = await client.get(f"{bazarr_url}/api/movies", headers=headers)
                if m_res.status_code == 200:
                    movies = m_res.json().get("data", [])
                    norm_target = os.path.normpath(video_path)
                    for m in movies:
                        m_path = os.path.normpath(m.get("path", ""))
                        if m_path == norm_target or os.path.basename(m_path) == os.path.basename(norm_target):
                            r_id = m.get("radarrId")
                            if r_id:
                                logger.info(f"Triggering Bazarr movie subtitle search for radarrId {r_id}")
                                await client.patch(
                                    f"{bazarr_url}/api/movies/subtitles",
                                    headers=headers,
                                    params={
                                        "radarrid": r_id,
                                        "language": "sv",
                                        "forced": "False",
                                        "hi": "False"
                                    }
                                )
                                return
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

    async def _run_pipeline_logic(
        self,
        job_id: int,
        video_path: str,
        wait_seconds: Optional[int] = None,
        event_source: str = "MANUAL",
        force_retranslate: bool = False,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        # Orphaned subtitle cleanup (Upgrades/Renames)
        if event_source in ["SONARR", "RADARR"] and os.path.exists(os.path.dirname(video_path)):
            dir_path = os.path.dirname(video_path)
            base_name, _ = os.path.splitext(os.path.basename(video_path))
            import re
            
            # Find the episode identifier (e.g. S01E01) or use the movie title prefix
            ep_match = re.search(r'(S\d+E\d+)', base_name, re.IGNORECASE)
            search_token = ep_match.group(1).lower() if ep_match else ""
            
            # If it's a movie, the search token is the first few words of the base_name to prevent wiping flat directories
            if not search_token:
                search_token = " ".join(base_name.split(".")[:2]).split("(")[0].strip().lower()

            try:
                for fname in os.listdir(dir_path):
                    if fname.endswith(".srt"):
                        f_base, _ = os.path.splitext(fname)
                        if f_base != base_name and search_token and search_token in fname.lower():
                            os.remove(os.path.join(dir_path, fname))
                            append_job_log(job_id, f"Cleaned up orphaned subtitle from previous version: {fname}")
            except Exception as e:
                pass
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
        # 0. HYBRID MODE: CHECK BAZARR FOR HUMAN SUBTITLES (Grace Period)
        # -------------------------------------------------------------
        if enable_bazarr and not force_retranslate:
            # Check if Swedish sub already exists first
            existing_swe = check_existing_swedish_subtitle(video_path)
            if not existing_swe:
                bazarr_wait = wait_seconds if wait_seconds is not None else int(get_setting("wait_time_seconds", "15"))
                if bazarr_wait > 0:
                    append_job_log(job_id, f"Hybrid Mode: Triggering Bazarr search and waiting {bazarr_wait}s for human subtitles...")
                    await self.trigger_bazarr_search(video_path)
                    
                    # Poll every 2 seconds during the grace period
                    elapsed = 0
                    while elapsed < bazarr_wait:
                        await asyncio.sleep(2)
                        elapsed += 2
                        existing_swe = check_existing_swedish_subtitle(video_path)
                        if existing_swe:
                            if auto_repair_unhealthy:
                                health = evaluate_subtitle_health(existing_swe)
                                if not health.get("status") == "RED":
                                    append_job_log(job_id, f"Bazarr found healthy human subtitle ({os.path.basename(existing_swe)}). Subtitle downloaded successfully.")
                                    duration = round(time.time() - start_time, 2)
                                    update_job(job_id, status="BAZARR MATCH", reason="Bazarr found human subtitle", duration_seconds=duration)
                                    await notify_jellyfin_library_refresh()
                                    return {"status": "skipped", "reason": "bazarr_downloaded", "file": existing_swe, "job_id": job_id}
                                else:
                                    append_job_log(job_id, f"Bazarr subtitle is corrupted or poor quality ({health.get('reason')}). Proceeding with Babel AI engine...")
                                    break
                            else:
                                append_job_log(job_id, f"Bazarr found human subtitle ({os.path.basename(existing_swe)}). Subtitle downloaded successfully.")
                                duration = round(time.time() - start_time, 2)
                                update_job(job_id, status="BAZARR MATCH", reason="Bazarr found human subtitle", duration_seconds=duration)
                                await notify_jellyfin_library_refresh()
                                return {"status": "skipped", "reason": "bazarr_downloaded", "file": existing_swe, "job_id": job_id}

                    append_job_log(job_id, f"Bazarr grace period ended ({bazarr_wait}s). No human subtitle found. Proceeding with Babel AI engine.")

        # -------------------------------------------------------------
        # 1. CHECK IF TARGET SUBTITLE CAN BE EXTRACTED DIRECTLY (Priority 0)
        # -------------------------------------------------------------
        if extract_target_embedded and not force_retranslate:
            for lang_info in target_languages:
                lang_code = lang_info["code"]
                target_output_path = f"{base_path}.{lang_code}.srt"
                
                if not os.path.exists(target_output_path):
                    extracted_target = extract_embedded_srt(video_path, target_output_path, preferred_lang=lang_code)
                    if extracted_target and os.path.exists(target_output_path):
                        append_job_log(job_id, f"Found and extracted embedded target track ({lang_info['name']}) directly to {os.path.basename(target_output_path)}.")
                        duration = round(time.time() - start_time, 2)
                        update_job(
                            job_id,
                            status="TRANSLATED",
                            reason=f"Extracted embedded {lang_info['name']}",
                            duration_seconds=duration,
                            output_files=json.dumps([target_output_path])
                        )
                        await notify_jellyfin_library_refresh()
                        return {"status": "success", "extracted_embedded": True, "job_id": job_id}

        # -------------------------------------------------------------
        # 2. CHECK IF HEALTHY SUBTITLE ALREADY EXISTS ON DISK
        # -------------------------------------------------------------
        if not force_retranslate:
            existing_swe = check_existing_swedish_subtitle(video_path)
            if existing_swe:
                if auto_repair_unhealthy:
                    health = evaluate_subtitle_health(existing_swe)
                    if health.get("status") == "RED":
                        append_job_log(job_id, f"Existing subtitle found but unhealthy ({health['reason']}). Triggering AI re-translation...")
                    else:
                        append_job_log(job_id, f"Healthy subtitle already exists: {os.path.basename(existing_swe)}. Skipping.")
                        update_job(job_id, status="ALREADY EXISTS", reason="Healthy subtitle exists", duration_seconds=round(time.time() - start_time, 2))
                        return {"status": "skipped", "reason": "already_exists", "file": existing_swe}
                else:
                    append_job_log(job_id, f"Subtitle already exists: {os.path.basename(existing_swe)}. Skipping.")
                    update_job(job_id, status="ALREADY EXISTS", reason="Subtitle already exists", duration_seconds=round(time.time() - start_time, 2))
                    return {"status": "skipped", "reason": "already_exists", "file": existing_swe}

        # -------------------------------------------------------------
        # 3. LOCATE SOURCE SUBTITLE (1: Embedded English, 2: External English, 3: Bazarr Fallback)
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
            en_srt_source = check_existing_english_subtitle(video_path)
            if en_srt_source and os.path.exists(en_srt_source):
                append_job_log(job_id, f"Found external source subtitle: {os.path.basename(en_srt_source)}")
                try:
                    with open(en_srt_source, "r", encoding="utf-8-sig") as f:
                        raw_srt_text = f.read()
                except UnicodeDecodeError:
                    with open(en_srt_source, "r", encoding="windows-1252") as f:
                        raw_srt_text = f.read()

        # PRIORITY 3: Fetch from Bazarr (via API wrapper)
        if not raw_srt_text and enable_bazarr:
            append_job_log(job_id, "Attempting to fetch missing subtitle via Bazarr API...")
            en_srt_source = download_subtitle_from_bazarr(video_path, job_id=job_id)
            if en_srt_source and os.path.exists(en_srt_source):
                try:
                    with open(en_srt_source, "r", encoding="utf-8-sig") as f:
                        raw_srt_text = f.read()
                except UnicodeDecodeError:
                    with open(en_srt_source, "r", encoding="windows-1252") as f:
                        raw_srt_text = f.read()

        # PRIORITY 3: Bazarr Safety Net Fallback (Always active if video is naked and has no subs)
        if not raw_srt_text:
            append_job_log(job_id, "No embedded or external English sub found. Querying Bazarr safety net for English subtitle...")
            await self.trigger_bazarr_search(video_path)
            
            # Wait up to 15s in 2-second intervals for Bazarr to drop an English sub
            for _ in range(8):
                await asyncio.sleep(2)
                en_srt_source = check_existing_english_subtitle(video_path)
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
            # 4. PRE-PROCESSING & SDH CLEANER
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
            repaired_langs = []
            total_dropped = 0
            max_sync_diff = 0

            # -------------------------------------------------------------
            # 5. TRANSLATION & STRICT SYNC ENFORCEMENT
            # -------------------------------------------------------------
            
            # Original Language Guard
            from app.core.extractor import inspect_mkv_tracks
            mkv_info = inspect_mkv_tracks(video_path)
            primary_audio_lang = "und"
            for audio in mkv_info.get("audio", []):
                if audio.get("default") or audio.get("forced"):
                    primary_audio_lang = audio.get("language", "und").lower()
                    break
            if primary_audio_lang == "und" and mkv_info.get("audio"):
                primary_audio_lang = mkv_info["audio"][0].get("language", "und").lower()
                
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

            for lang_info in target_languages:
                lang_name = lang_info["name"]
                lang_code = lang_info["code"]
                target_output_path = f"{base_path}.{lang_code}.srt"
                
                # Check Original Language Guard
                protected_langs = AUDIO_LANG_MAP.get(lang_code, [lang_code])
                if primary_audio_lang in protected_langs:
                    append_job_log(job_id, f"Original Language Guard: The primary audio is already '{primary_audio_lang}'. Skipping AI translation to {lang_name} to prevent bad round-trip translation.")
                    successful_langs.append(lang_name) # Count as success so the job completes
                    continue

                append_job_log(job_id, f"Translating to {lang_name} ({lang_code}) using {active_engine_name} (0 to {total_source_lines} lines)...")

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

                sync_report = verify_sync(subs, translated_subs)
                dropped_count, _ = check_dropped_lines(subs, translated_subs)

                total_dropped += dropped_count
                if sync_report.get("start_diff_ms", 0) > max_sync_diff:
                    max_sync_diff = sync_report.get("start_diff_ms", 0)

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
                append_job_log(job_id, f"Successfully saved {os.path.basename(target_output_path)} (Sync Diff: {sync_report.get('start_diff_ms', 0)}ms, Dropped: {dropped_count})")

            # Clean up temp file
            if os.path.exists(temp_extracted_srt):
                try: os.remove(temp_extracted_srt)
                except: pass

            total_duration = round(time.time() - start_time, 2)
            final_status = "TRANSLATED"

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

            await notify_jellyfin_library_refresh()

            return {
                "status": "success",
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
