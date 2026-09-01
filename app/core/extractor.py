import subprocess
import json
import os
import time
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

def inspect_mkv_tracks(video_path: str) -> Dict[str, List[Dict]]:
    """
    Uses mkvmerge / ffprobe to quickly extract information about embedded audio and subtitle tracks.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if os.path.getsize(video_path) == 0:
        return {"subtitles": [], "audio": [], "duration": 0.0}

    tracks = {"subtitles": [], "audio": []}
    try:
        cmd = ["mkvmerge", "-J", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
        data = json.loads(result.stdout)

        duration = 0.0
        container = data.get("container", {}).get("properties", {})
        if "duration" in container:
            duration = float(container["duration"]) / 1e9 # mkvmerge returns nanoseconds

        tracks["duration"] = duration
        for track in data.get("tracks", []):
            track_type = track.get("type")
            properties = track.get("properties", {})
            lang = properties.get("language", "und")
            track_id = track.get("id")
            codec = track.get("codec")
            forced = properties.get("forced_track", False)
            default = properties.get("default_track", False)
            title = properties.get("track_name", "")

            track_info = {
                "id": track_id,
                "language": lang,
                "codec": codec,
                "forced": forced,
                "default": default,
                "title": title
            }

            if track_type == "subtitles":
                tracks["subtitles"].append(track_info)
            elif track_type == "audio":
                tracks["audio"].append(track_info)

        return tracks
    except Exception:
        # Fallback to ffprobe
        try:
            cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
            data = json.loads(result.stdout)

            duration = 0.0
            if "format" in data and "duration" in data["format"]:
                duration = float(data["format"]["duration"])
            tracks["duration"] = duration

            for stream in data.get("streams", []):
                codec_type = stream.get("codec_type")
                tags = stream.get("tags", {})
                lang = tags.get("language", "und")
                track_id = stream.get("index")
                codec = stream.get("codec_name", "")

                disposition = stream.get("disposition", {})
                forced = bool(disposition.get("forced", 0))
                default = bool(disposition.get("default", 0))
                title = tags.get("title", "")

                track_info = {
                    "id": track_id,
                    "language": lang,
                    "codec": codec,
                    "forced": forced,
                    "default": default,
                    "title": title
                }

                if codec_type == "subtitle":
                    tracks["subtitles"].append(track_info)
                elif codec_type == "audio":
                    tracks["audio"].append(track_info)

            return tracks
        except Exception as e2:
            return {"subtitles": [], "audio": [], "error": str(e2)}

def get_cached_embedded_srt(video_path: str, track_id: Optional[int], lang: str) -> Optional[str]:
    """
    Retrieve cached extracted SRT content if media file matches exact size and mtime_ns.
    Returns parsed/validated SRT string or None if cache miss / invalid.
    """
    norm_path = os.path.normpath(video_path)
    try:
        stat = os.stat(video_path)
        file_size = stat.st_size
        mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))
    except Exception:
        return None

    try:
        from app.core.languages import normalize_language_code
        norm_lang = normalize_language_code(lang, default=lang.lower()).lower()
        import app.core.db
        import sqlite3
        with sqlite3.connect(app.core.db.DB_PATH, timeout=5.0) as conn:
            cursor = conn.cursor()
            if track_id is None:
                row = cursor.execute(
                    """SELECT content FROM embedded_extraction_cache
                       WHERE video_path = ? AND file_size = ? AND mtime_ns = ? AND track_id IS NULL AND track_language = ?""",
                    (norm_path, file_size, mtime_ns, norm_lang)
                ).fetchone()
            else:
                row = cursor.execute(
                    """SELECT content FROM embedded_extraction_cache
                       WHERE video_path = ? AND file_size = ? AND mtime_ns = ? AND track_id = ? AND track_language = ?""",
                    (norm_path, file_size, mtime_ns, track_id, norm_lang)
                ).fetchone()
            if not row or not row[0]:
                return None
            content = row[0]
            import srt
            parsed = list(srt.parse(content))
            if not parsed:
                return None
            return content
    except Exception as e:
        logger.debug(f"Cache lookup failed for {norm_path}: {e}")
        return None


def save_cached_embedded_srt(video_path: str, track_id: Optional[int], lang: str, content: str) -> bool:
    """Save extracted SRT content to persistent cache."""
    norm_path = os.path.normpath(video_path)
    try:
        stat = os.stat(video_path)
        file_size = stat.st_size
        mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))
    except Exception:
        return False

    try:
        from app.core.languages import normalize_language_code
        norm_lang = normalize_language_code(lang, default=lang.lower()).lower()
        import app.core.db
        from datetime import datetime, timezone
        import sqlite3
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(app.core.db.DB_PATH, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO embedded_extraction_cache
                   (video_path, file_size, mtime_ns, track_id, track_language, content, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (norm_path, file_size, mtime_ns, track_id, norm_lang, content, now)
            )
            conn.commit()
            return True
    except Exception as e:
        logger.debug(f"Failed to save extraction cache for {norm_path}: {e}")
        return False


def invalidate_cached_embedded_srt(video_path: str) -> bool:
    """Invalidate all cached extractions for a video path."""
    norm_path = os.path.normpath(video_path)
    try:
        from app.core.db import DB_PATH
        import sqlite3
        with sqlite3.connect(DB_PATH, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM embedded_extraction_cache WHERE video_path = ?", (norm_path,))
            conn.commit()
            return True
    except Exception as e:
        logger.debug(f"Failed to invalidate extraction cache for {norm_path}: {e}")
        return False

import threading
import signal
import tempfile
from typing import Tuple

DEFAULT_EXTRACTION_TIMEOUT = 300.0  # 5 minutes bounded timeout for large MKVs
CONVERSION_TIMEOUT = 60.0

# Concurrency limits for disk-I/O-heavy embedded subtitle extraction
DEFAULT_MAX_CONCURRENT_EXTRACTIONS = 2
_extraction_semaphore = threading.Semaphore(DEFAULT_MAX_CONCURRENT_EXTRACTIONS)
_in_flight_lock = threading.Lock()
_in_flight_extractions: Dict[Tuple[str, Optional[int], str], threading.Event] = {}


def _kill_proc_group(proc: subprocess.Popen) -> None:
    """Safely terminate a process group without leaving orphaned subprocesses."""
    try:
        if hasattr(os, "killpg") and hasattr(os, "getpgid"):
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
        else:
            proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=2.0)
    except Exception:
        pass


def _run_cancellable_cmd(
    cmd: List[str],
    timeout: float = DEFAULT_EXTRACTION_TIMEOUT,
    cancel_event: Optional[Any] = None
) -> int:
    """
    Run an external command with timeout and cancel_event process termination.
    Eliminates pipe deadlock by redirecting stdout to DEVNULL and stderr to a temp file.
    """
    if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
        return -1

    if cancel_event is None:
        popen_kwargs: Dict[str, Any] = {}
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        try:
            res = subprocess.run(cmd, capture_output=True, check=True, timeout=timeout, **popen_kwargs)
            return getattr(res, "returncode", 0)
        except subprocess.CalledProcessError as cpe:
            return cpe.returncode

    popen_kwargs: Dict[str, Any] = {
        "stdout": subprocess.DEVNULL,
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True

    stderr_f = tempfile.TemporaryFile()
    try:
        proc = subprocess.Popen(cmd, stderr=stderr_f, **popen_kwargs)
    except Exception:
        stderr_f.close()
        raise

    t_start = time.monotonic()
    try:
        while True:
            if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                _kill_proc_group(proc)
                return -1

            ret = proc.poll()
            if ret is not None:
                return ret

            if (time.monotonic() - t_start) > timeout:
                _kill_proc_group(proc)
                raise subprocess.TimeoutExpired(cmd, timeout)

            time.sleep(0.02)
    except subprocess.TimeoutExpired:
        _kill_proc_group(proc)
        raise
    except Exception:
        _kill_proc_group(proc)
        raise
    finally:
        try:
            if proc.poll() is None:
                _kill_proc_group(proc)
        except Exception:
            pass
        try:
            stderr_f.close()
        except Exception:
            pass


def extract_embedded_srt(
    video_path: str,
    output_srt_path: str,
    preferred_lang: str = "eng",
    tracks_info: Optional[Dict[str, Any]] = None,
    cancel_event: Optional[Any] = None,
    timeout: float = DEFAULT_EXTRACTION_TIMEOUT,
    **kwargs
) -> bool:
    """
    Extracts the best matching embedded subtitle track to an SRT file using fast mkvextract for MKVs (with ffmpeg fallback)
    and ffmpeg for non-MKV containers.
    Prefers non-forced SRT / SubRip / text subtitles.
    Checks and populates persistent extraction cache. Supports cancel_event for process-level termination.
    """
    if "timeout" in kwargs and (timeout == DEFAULT_EXTRACTION_TIMEOUT or timeout is None):
        timeout = kwargs["timeout"]
    if "cancel_event" in kwargs and cancel_event is None:
        cancel_event = kwargs["cancel_event"]

    if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
        return False

    t_total_start = time.perf_counter()

    if tracks_info is None:
        t_probe_start = time.perf_counter()
        tracks_info = inspect_mkv_tracks(video_path)
        t_probe_elapsed = time.perf_counter() - t_probe_start
    else:
        t_probe_elapsed = 0.0

    sub_tracks = tracks_info.get("subtitles", [])
    if not sub_tracks:
        return False

    selected_track_id = None
    selected_sub_index = None
    from app.core.languages import get_language, normalize_language_code
    lang_obj = get_language(preferred_lang)
    lang_prefixes = lang_obj.aliases if lang_obj else [preferred_lang.lower()]
    norm_lang = normalize_language_code(preferred_lang, default=preferred_lang.lower()).lower()
    TEXT_CODECS = {"SubRip/SRT", "S_TEXT/UTF8", "S_TEXT/ASS", "S_TEXT/SSA", "S_TEXT/WEBVTT", "SubStationAlpha", "WebVTT"}

    candidates = []
    for i, track in enumerate(sub_tracks):
        lang = track.get("language", "").lower()
        forced = track.get("forced", False)
        codec = track.get("codec", "")
        title = track.get("title", "").lower() if track.get("title") else ""

        # Skip outright bad tracks
        if any(bad in title for bad in ["commentary", "director", "description", "audio description"]):
            continue

        is_text_codec = any(tc.lower() in codec.lower() for tc in TEXT_CODECS) or "srt" in codec.lower() or "text" in codec.lower() or "ass" in codec.lower() or "ssa" in codec.lower() or "vtt" in codec.lower() or "utf" in codec.lower()

        track_norm = normalize_language_code(lang, default=lang).lower() if lang else None
        title_lang_obj = get_language(title) if title else None
        title_norm = title_lang_obj.code.lower() if title_lang_obj else None

        lang_matches = (
            (track_norm is not None and track_norm == norm_lang)
            or (title_norm is not None and title_norm == norm_lang)
            or (not track_norm and not title_norm and any(lp == lang for lp in lang_prefixes))
        )

        if lang_matches and is_text_codec:
            score = 100
            if forced or any(kw in title for kw in ["forced", "signs", "songs", "foreign", "parts", "descriptive"]):
                continue  # Skip forced tracks entirely for full translation

            if any(kw in title for kw in ["full", "sdh", "normal", "dialogue"]):
                score += 20

            if track.get("default"):
                score += 10

            candidates.append({"score": score, "id": track.get("id"), "index": i, "codec": codec})

    candidates.sort(key=lambda x: x["score"], reverse=True)
    if not candidates:
        return False

    duration = tracks_info.get("duration", 0.0)
    import srt

    # Determine file size for throughput telemetry
    try:
        media_file_size_bytes = os.path.getsize(video_path)
    except Exception:
        media_file_size_bytes = 0
    media_size_gb = round(media_file_size_bytes / (1024 ** 3), 2)

    is_mkv = video_path.lower().endswith(".mkv") or video_path.lower().endswith(".mka")
    container_name = "MKV" if is_mkv else ("MP4" if video_path.lower().endswith(".mp4") else "OTHER")

    for cand in candidates:
        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            return False

        selected_track_id = cand["id"]
        selected_sub_index = cand["index"]
        selected_codec = cand["codec"].lower()

        # Step 1: Check persistent extraction cache
        t_cache_lookup_start = time.perf_counter()
        cached_content = get_cached_embedded_srt(video_path, selected_track_id, preferred_lang)
        t_cache_lookup = time.perf_counter() - t_cache_lookup_start

        if cached_content:
            logger.info(f"Source cache: HIT for track {selected_track_id} ({preferred_lang}) in {t_cache_lookup:.4f}s")
            try:
                with open(output_srt_path, "w", encoding="utf-8") as f:
                    f.write(cached_content)
                if os.path.exists(output_srt_path) and os.path.getsize(output_srt_path) > 0:
                    logger.info(f"Extraction skipped: cached source (track={selected_track_id}, lang={preferred_lang})")
                    return True
            except Exception as write_err:
                logger.warning(f"Failed to write cached SRT to {output_srt_path}: {write_err}")

        logger.info(f"Source cache: MISS for track {selected_track_id} ({preferred_lang})")

        # Step 2: Single-flight synchronization for concurrent requests of the same track
        flight_key = (os.path.normpath(video_path), selected_track_id, norm_lang)
        is_lead_extractor = False
        flight_event: Optional[threading.Event] = None

        with _in_flight_lock:
            if flight_key in _in_flight_extractions:
                flight_event = _in_flight_extractions[flight_key]
            else:
                flight_event = threading.Event()
                _in_flight_extractions[flight_key] = flight_event
                is_lead_extractor = True

        if not is_lead_extractor and flight_event is not None:
            logger.info(f"Awaiting concurrent in-flight extraction for track {selected_track_id} ({norm_lang})...")
            # Wait for lead extractor to complete while honoring cancel_event
            while not flight_event.is_set():
                if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                    return False
                flight_event.wait(timeout=0.05)

            # Check cache again — lead extractor should have populated it
            cached_after = get_cached_embedded_srt(video_path, selected_track_id, preferred_lang)
            if cached_after:
                try:
                    with open(output_srt_path, "w", encoding="utf-8") as f:
                        f.write(cached_after)
                    if os.path.exists(output_srt_path) and os.path.getsize(output_srt_path) > 0:
                        logger.info(f"Extraction reused from concurrent in-flight lead extractor (track {selected_track_id})")
                        return True
                except Exception as write_err:
                    logger.warning(f"Failed to write reused cached SRT: {write_err}")

        # Step 3: Lead extractor acquires bounded extraction semaphore
        acquired_sem = False
        t_wait_sem_start = time.perf_counter()
        try:
            while not acquired_sem:
                if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                    return False
                acquired_sem = _extraction_semaphore.acquire(timeout=0.05)

            t_wait_sem = time.perf_counter() - t_wait_sem_start

            success = False
            mkv_timed_out = False
            t_cand_start = time.monotonic()
            used_backend = "mkvextract" if (is_mkv and selected_track_id is not None) else "ffmpeg"
            subproc_duration = 0.0

            if is_mkv and selected_track_id is not None:
                # Primary for Matroska: mkvextract
                try:
                    logger.info(
                        f"Extracting embedded {preferred_lang.upper()} track {selected_track_id} with mkvextract "
                        f"(size={media_size_gb}GB, timeout={timeout:.0f}s, sem_wait={t_wait_sem:.2f}s)..."
                    )
                    t_sp_start = time.perf_counter()
                    cmd = ["mkvextract", "tracks", video_path, f"{selected_track_id}:{output_srt_path}"]
                    ret = _run_cancellable_cmd(cmd, timeout=timeout, cancel_event=cancel_event)
                    subproc_duration = time.perf_counter() - t_sp_start

                    if ret == 0 and os.path.exists(output_srt_path) and os.path.getsize(output_srt_path) > 0:
                        if any(x in selected_codec for x in ["ass", "ssa", "vtt", "webvtt", "substation", "alpha"]):
                            if "ass" in selected_codec or "substation" in selected_codec or "alpha" in selected_codec:
                                intermediate_ext = ".ass"
                            elif "ssa" in selected_codec:
                                intermediate_ext = ".ssa"
                            elif "vtt" in selected_codec or "webvtt" in selected_codec:
                                intermediate_ext = ".vtt"
                            else:
                                intermediate_ext = ".tmp"
                            temp_file = output_srt_path + intermediate_ext
                            os.rename(output_srt_path, temp_file)
                            t_conv_start = time.perf_counter()
                            ffmpeg_cmd = ["ffmpeg", "-y", "-i", temp_file, "-c:s", "srt", "-f", "srt", output_srt_path]
                            ret_ff = _run_cancellable_cmd(ffmpeg_cmd, timeout=CONVERSION_TIMEOUT, cancel_event=cancel_event)
                            conv_duration = time.perf_counter() - t_conv_start
                            logger.info(f"Converted {intermediate_ext} to SRT via ffmpeg in {conv_duration:.2f}s (ret={ret_ff})")
                            try:
                                os.remove(temp_file)
                            except Exception:
                                pass
                            if ret_ff != 0 and os.path.exists(output_srt_path):
                                try:
                                    os.remove(output_srt_path)
                                except Exception:
                                    pass

                        if os.path.exists(output_srt_path) and os.path.getsize(output_srt_path) > 0:
                            try:
                                with open(output_srt_path, "r", encoding="utf-8-sig") as f:
                                    test_subs = list(srt.parse(f.read()))
                                if test_subs:
                                    success = True
                                else:
                                    if os.path.exists(output_srt_path):
                                        try:
                                            os.remove(output_srt_path)
                                        except Exception:
                                            pass
                            except Exception:
                                if os.path.exists(output_srt_path):
                                    try:
                                        os.remove(output_srt_path)
                                    except Exception:
                                        pass
                    elif ret == -1:
                        logger.info(f"mkvextract cancelled for track {selected_track_id}.")
                        if os.path.exists(output_srt_path):
                            try:
                                os.remove(output_srt_path)
                            except Exception:
                                pass
                        return False
                except subprocess.TimeoutExpired:
                    mkv_timed_out = True
                    logger.warning(f"mkvextract timed out after {timeout:.0f}s for track {selected_track_id}.")
                    if os.path.exists(output_srt_path):
                        try:
                            os.remove(output_srt_path)
                        except Exception:
                            pass
                except Exception as mkv_err:
                    logger.warning(f"mkvextract failed for track {selected_track_id}: {mkv_err}")
                    if os.path.exists(output_srt_path):
                        try:
                            os.remove(output_srt_path)
                        except Exception:
                            pass

            if not success:
                if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                    return False

                # If mkvextract timed out on MKV, skip ffmpeg fallback
                if is_mkv and mkv_timed_out:
                    logger.warning(f"Skipping ffmpeg fallback for track {selected_track_id} because mkvextract timed out.")
                else:
                    elapsed_so_far = time.monotonic() - t_cand_start
                    rem_timeout = max(10.0, timeout - elapsed_so_far)
                    backend_desc = "fallback" if is_mkv else "primary"
                    used_backend = "ffmpeg"
                    try:
                        logger.info(f"Extracting embedded {preferred_lang.upper()} track {selected_sub_index} with ffmpeg ({backend_desc}, timeout {rem_timeout:.0f}s)...")
                        t_sp_start = time.perf_counter()
                        cmd = [
                            "ffmpeg", "-y", "-i", video_path,
                            "-map", f"0:s:{selected_sub_index}",
                            "-c:s", "srt",
                            "-f", "srt",
                            output_srt_path
                        ]
                        ret = _run_cancellable_cmd(cmd, timeout=rem_timeout, cancel_event=cancel_event)
                        subproc_duration = time.perf_counter() - t_sp_start

                        if ret == 0 and os.path.exists(output_srt_path) and os.path.getsize(output_srt_path) > 0:
                            try:
                                with open(output_srt_path, "r", encoding="utf-8-sig") as f:
                                    test_subs = list(srt.parse(f.read()))
                                if test_subs:
                                    success = True
                                else:
                                    if os.path.exists(output_srt_path):
                                        try:
                                            os.remove(output_srt_path)
                                        except Exception:
                                            pass
                            except Exception:
                                if os.path.exists(output_srt_path):
                                    try:
                                        os.remove(output_srt_path)
                                    except Exception:
                                        pass
                        elif ret == -1:
                            logger.info(f"ffmpeg extraction cancelled for track {selected_sub_index}.")
                            if os.path.exists(output_srt_path):
                                try:
                                    os.remove(output_srt_path)
                                except Exception:
                                    pass
                            return False
                    except subprocess.TimeoutExpired:
                        logger.warning(f"ffmpeg extraction timed out after {rem_timeout:.0f}s for track {selected_sub_index}.")
                        if os.path.exists(output_srt_path):
                            try:
                                os.remove(output_srt_path)
                            except Exception:
                                pass
                    except Exception as ff_err:
                        logger.warning(f"ffmpeg extraction failed for track {selected_sub_index}: {ff_err}")
                        if os.path.exists(output_srt_path):
                            try:
                                os.remove(output_srt_path)
                            except Exception:
                                pass

            if success:
                try:
                    t_parse_start = time.perf_counter()
                    with open(output_srt_path, "r", encoding="utf-8-sig") as f:
                        content = f.read()

                    subs = list(srt.parse(content))
                    t_parse = time.perf_counter() - t_parse_start
                    if not subs:
                        logger.warning(f"Extracted subtitle {output_srt_path} is empty, removing and trying next candidate.")
                        if os.path.exists(output_srt_path):
                            try:
                                os.remove(output_srt_path)
                            except Exception:
                                pass
                        continue

                    last_end = subs[-1].end.total_seconds()

                    # Conservative logic to detect obvious partial sources
                    if duration > 1200:  # 20 mins
                        if len(subs) < 100 and last_end < (duration * 0.25):
                            logger.warning(f"Extracted subtitle {output_srt_path} rejected as partial source ({len(subs)} cues, ends at {last_end:.1f}s of {duration:.1f}s).")
                            if os.path.exists(output_srt_path):
                                try:
                                    os.remove(output_srt_path)
                                except Exception:
                                    pass
                            continue

                    t_total = time.perf_counter() - t_total_start
                    throughput_mb_s = (media_file_size_bytes / (1024 * 1024)) / subproc_duration if subproc_duration > 0 else 0.0

                    logger.info(
                        f"Embedded extraction successful:\n"
                        f"  container = {container_name}\n"
                        f"  selected track id = {selected_track_id}\n"
                        f"  codec = {cand.get('codec')}\n"
                        f"  method = {used_backend}\n"
                        f"  probe = {t_probe_elapsed:.2f}s\n"
                        f"  extract_wait = {t_wait_sem:.2f}s\n"
                        f"  extraction subprocess = {subproc_duration:.2f}s\n"
                        f"  parse = {t_parse:.3f}s\n"
                        f"  total = {t_total:.2f}s\n"
                        f"  file size = {media_size_gb} GB (throughput = {throughput_mb_s:.1f} MB/s)\n"
                        f"  cache = MISS"
                    )

                    save_cached_embedded_srt(video_path, selected_track_id, preferred_lang, content)
                    return True

                except Exception as parse_err:
                    logger.warning(f"Failed to parse extracted subtitle {output_srt_path}: {parse_err}")
                    if os.path.exists(output_srt_path):
                        try:
                            os.remove(output_srt_path)
                        except Exception:
                            pass
                    continue
        finally:
            if acquired_sem:
                _extraction_semaphore.release()
            if is_lead_extractor and flight_event is not None:
                with _in_flight_lock:
                    _in_flight_extractions.pop(flight_key, None)
                flight_event.set()

    return False
