import os
import re
import logging
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from app.core.db import (
    get_setting,
    get_cached_embedded_subtitle_tracks,
    set_cached_embedded_subtitle_tracks,
    bulk_get_cached_embedded_subtitle_tracks,
)
from app.core.languages import get_language, normalize_language_code

logger = logging.getLogger("babel.scanner")

VIDEO_EXTS = (".mkv", ".mp4", ".m4v", ".avi")

_SUB_LINE_CACHE: Dict[str, tuple[float, int, int]] = {}
_EMBEDDED_TRACKS_CACHE: Dict[str, tuple[int, int, Dict[str, Any]]] = {}

FAILED_PROBE_COOLDOWN_SEC: float = 300.0  # 5-minute cooldown for transient container probe failures

TEXT_CODECS = {
    "subrip/srt", "subrip", "srt", "s_text/utf8", "s_text/ass", "s_text/ssa",
    "s_text/webvtt", "substationalpha", "webvtt", "ass", "ssa", "text", "utf-8"
}


def _is_failed_probe_expired(cache_entry: Dict[str, Any], cooldown_sec: float = FAILED_PROBE_COOLDOWN_SEC) -> bool:
    """Check if a cached probe failure is older than the cooldown period."""
    if not isinstance(cache_entry, dict) or cache_entry.get("status") != "failed":
        return False
    updated_at_str = cache_entry.get("updated_at")
    if not updated_at_str:
        return True  # No timestamp -> treat as expired
    try:
        updated_dt = datetime.fromisoformat(updated_at_str)
        if updated_dt.tzinfo is None:
            updated_dt = updated_dt.replace(tzinfo=timezone.utc)
        now_dt = datetime.now(timezone.utc)
        age = (now_dt - updated_dt).total_seconds()
        return age >= cooldown_sec
    except Exception:
        return True


class EmbeddedSubtitleProber:
    """
    Background worker pool with bounded concurrency (max 4 workers)
    for non-blocking probing and persistent caching of embedded media container subtitle tracks.
    """
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="BabelEmbeddedProbe")
        self._lock = threading.Lock()
        self._queued_paths: set[str] = set()
        self._in_flight: set[str] = set()

    def enqueue(self, items: List[tuple[str, int, int]]) -> None:
        """
        Enqueues uncached video items (video_path, file_size, mtime_ns) for background probing.
        Avoids duplicate queuing.
        """
        with self._lock:
            to_submit = []
            for path, size, mtime_ns in items:
                norm_path = os.path.normpath(path)
                if norm_path not in self._queued_paths and norm_path not in self._in_flight:
                    self._queued_paths.add(norm_path)
                    to_submit.append((norm_path, size, mtime_ns))

            for path, size, mtime_ns in to_submit:
                self._executor.submit(self._worker_probe, path, size, mtime_ns)

    def _worker_probe(self, video_path: str, file_size: int, mtime_ns: int) -> None:
        with self._lock:
            self._queued_paths.discard(video_path)
            self._in_flight.add(video_path)

        probe_status = "ok"
        sub_tracks: List[Dict[str, Any]] = []
        try:
            from app.core.extractor import inspect_mkv_tracks
            tracks_info = inspect_mkv_tracks(video_path)
            if isinstance(tracks_info, dict) and "error" in tracks_info:
                probe_status = "failed"
                sub_tracks = []
            elif isinstance(tracks_info, dict):
                probe_status = "ok"
                sub_tracks = tracks_info.get("subtitles", []) or []
            else:
                probe_status = "ok"
                sub_tracks = []
        except Exception as e:
            logger.debug(f"Failed probing container tracks for {video_path}: {e}")
            probe_status = "failed"
            sub_tracks = []
        finally:
            now_iso = datetime.now(timezone.utc).isoformat()
            payload = {"status": probe_status, "tracks": sub_tracks, "updated_at": now_iso}
            try:
                set_cached_embedded_subtitle_tracks(video_path, file_size, mtime_ns, payload)
                _EMBEDDED_TRACKS_CACHE[video_path] = (file_size, mtime_ns, payload)
            except Exception as e:
                logger.error(f"Failed to persist embedded tracks cache for {video_path}: {e}")
            with self._lock:
                self._in_flight.discard(video_path)

    def is_active(self) -> bool:
        with self._lock:
            return len(self._queued_paths) > 0 or len(self._in_flight) > 0

    def get_pending_count(self) -> int:
        with self._lock:
            return len(self._queued_paths) + len(self._in_flight)

    def wait_completion(self, timeout: Optional[float] = None) -> None:
        """Helper to wait for pending background probes (useful in tests/sync)."""
        import time
        start = time.time()
        while self.is_active():
            if timeout and (time.time() - start) > timeout:
                break
            time.sleep(0.02)


embedded_prober = EmbeddedSubtitleProber(max_workers=4)


def is_embedded_probing_active() -> bool:
    """Check if background embedded subtitle probing is currently active."""
    return embedded_prober.is_active()


def _fast_count_subtitle_lines(path: str) -> int:
    """Cheaply estimate subtitle line count with mtime/size caching."""
    try:
        st = os.stat(path)
        cached = _SUB_LINE_CACHE.get(path)
        if cached and cached[0] == st.st_mtime and cached[1] == st.st_size:
            return cached[2]

        count = 0
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                count += chunk.count(b"-->")

        _SUB_LINE_CACHE[path] = (st.st_mtime, st.st_size, count)
        return count
    except Exception:
        return 0


def _get_target_lang_codes() -> List[str]:
    """Get configured target language codes."""
    raw = get_setting("languages", '[]')
    try:
        langs = json.loads(raw)
        return [l["code"] for l in langs if l.get("enabled", True)]
    except Exception:
        return ["sv"]


def _get_target_lang_aliases() -> List[str]:
    """Get configured target language codes and all their aliases."""
    codes = _get_target_lang_codes()
    aliases = set()
    for code in codes:
        if code:
            aliases.add(code.lower())
            lang_obj = get_language(code)
            if lang_obj:
                for a in lang_obj.aliases:
                    aliases.add(a.lower())
    return list(aliases) if aliases else ["sv", "swe", "swedish"]


def _get_all_known_lang_tokens() -> set:
    from app.core.languages import LANGUAGES
    tokens = set()
    for lang in LANGUAGES:
        tokens.add(lang.code.lower())
        for a in lang.aliases:
            tokens.add(a.lower())
    return tokens


ALL_KNOWN_LANG_TOKENS = _get_all_known_lang_tokens()
KNOWN_MODIFIERS = {"forced", "hi", "sdh", "cc", "signs", "songs", "default"}


def is_subtitle_for_video(video_basename: str, sub_filename: str) -> bool:
    """Check if sub_filename belongs to video_basename and is a valid external SRT (not temp/backup)."""
    sub_lower = sub_filename.lower()
    if not sub_lower.endswith(".srt"):
        return False
    if ".temp" in sub_lower or ".tmp" in sub_lower or ".babel-replaced" in sub_lower:
        return False
    base_lower = video_basename.lower()
    if not sub_lower.startswith(base_lower):
        return False
    remainder = sub_lower[len(base_lower):]
    if remainder == ".srt":
        return True
    if remainder.startswith(".") and remainder.endswith(".srt"):
        middle = remainder[1:-4]
        tokens = [t for t in re.split(r'[._-]+', middle) if t]
        if 1 <= len(tokens) <= 6:
            first_part = tokens[0]
            if first_part in ALL_KNOWN_LANG_TOKENS or first_part in KNOWN_MODIFIERS:
                return True
    return False


def is_target_language_subtitle(sub_filename: str, target_aliases: List[str]) -> bool:
    """Check if subtitle file matches any target language alias and is not forced/signs/songs."""
    fname_lower = sub_filename.lower()
    if fname_lower.endswith(".srt"):
        fname_lower = fname_lower[:-4]
    tokens = [t for t in re.split(r'[._-]+', fname_lower) if t]
    if any(tag in ["forced", "signs", "songs"] for tag in tokens):
        return False
    target_norm_codes = {normalize_language_code(a, default=a).lower() for a in target_aliases if a}

    # 1. Check dot-separated segments directly (e.g. 'Show.S01E01.pt-br' -> 'pt-br')
    for seg in fname_lower.split("."):
        lang_obj = get_language(seg)
        if lang_obj and lang_obj.code.lower() in target_norm_codes:
            return True

    # 2. Check compound and single tokens longest-first
    for i in range(len(tokens)):
        for j in range(len(tokens), i, -1):
            sub = "-".join(tokens[i:j])
            lang_obj = get_language(sub)
            if lang_obj:
                if lang_obj.code.lower() in target_norm_codes:
                    return True
                break  # Longest language match for this span found; do not check shorter sub-tokens
    return False


def is_qualifying_embedded_subtitle_track(track: Dict[str, Any], target_aliases: List[str]) -> bool:
    """
    Check if an embedded subtitle track matches the configured target language(s)
    and is a usable text subtitle track (not forced, signs, commentary, etc.).
    """
    lang = (track.get("language") or "").strip().lower()
    codec = (track.get("codec") or "").strip().lower()
    title = (track.get("title") or "").strip().lower()
    forced = bool(track.get("forced", False))

    # Check language
    target_set = {a.lower() for a in target_aliases if a}
    if not lang or lang == "und":
        title_tokens = [t for t in re.split(r'[._\s-]+', title) if t]
        if not any(t in target_set for t in title_tokens):
            return False
    else:
        norm_lang = normalize_language_code(lang, default=lang).lower()
        target_norm_codes = {normalize_language_code(a, default=a).lower() for a in target_aliases if a}
        lang_matches = (
            norm_lang in target_norm_codes
            or lang in target_set
        )
        if not lang_matches:
            return False

    # Exclude commentary / director / audio description
    if any(bad in title for bad in ["commentary", "director", "description", "audio description"]):
        return False

    # Exclude forced / signs / songs
    if forced or any(kw in title for kw in ["forced", "signs", "songs", "foreign", "parts", "descriptive"]):
        return False

    # Check text codec
    is_text = any(tc in codec for tc in TEXT_CODECS) or any(k in codec for k in ["srt", "text", "ass", "utf", "vtt", "subrip", "ssa"])
    if not is_text:
        return False

    return True


def scan_library_folders(root_path: str, category: str = "series") -> List[Dict[str, Any]]:
    """
    Scans the media library path and returns a list of media items
    along with their existing external and embedded target subtitles.

    NON-BLOCKING & SMART PROBING:
    - If qualifying external target subtitle exists -> Complete immediately, no embedded probe needed.
    - If external target missing + valid cache exists -> evaluates cached tracks immediately.
    - If external target missing + cache miss -> returns immediately with embedded_status_known=False
      and enqueues background probing.
    """
    if not root_path or not os.path.exists(root_path):
        return []

    target_aliases = _get_target_lang_aliases()
    results = []
    uncached_to_probe: List[tuple[str, int, int]] = []

    if category == "series":
        try:
            # 1. First collect all candidate files from disk
            discovered_shows: List[tuple[str, List[tuple[str, str, str, List[str]]]]] = []
            all_video_tuples: List[tuple[str, int, int]] = []

            for show_name in sorted(os.listdir(root_path)):
                show_dir = os.path.normpath(os.path.join(root_path, show_name))
                if not os.path.isdir(show_dir):
                    continue

                show_entries = []
                for root, _, files in os.walk(show_dir):
                    v_files = [f for f in sorted(files) if f.lower().endswith(VIDEO_EXTS)]
                    if not v_files:
                        continue
                    s_files = [f for f in files if f.lower().endswith(".srt")]
                    rel_season = os.path.basename(root)
                    season_name = rel_season if rel_season != show_name else "Root"

                    for file in v_files:
                        video_full_path = os.path.normpath(os.path.join(root, file))
                        size_bytes = 0
                        mtime_ns = 0
                        try:
                            st = os.stat(video_full_path)
                            size_bytes = int(st.st_size)
                            mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
                        except Exception:
                            pass
                        all_video_tuples.append((video_full_path, size_bytes, mtime_ns))
                        show_entries.append((season_name, file, video_full_path, s_files))

                if show_entries:
                    discovered_shows.append((show_name, show_entries))

            # 2. Bulk query persistent SQLite cache for all video files in library
            cached_tracks_map = bulk_get_cached_embedded_subtitle_tracks(all_video_tuples)

            # 3. Assemble series & episode records
            for show_name, show_entries in discovered_shows:
                show_episodes = []
                for season_name, file, video_full_path, s_files in show_entries:
                    base_name, _ = os.path.splitext(file)

                    subs = []
                    for f in s_files:
                        if is_subtitle_for_video(base_name, f):
                            sub_path = os.path.normpath(os.path.join(os.path.dirname(video_full_path), f))
                            subs.append({
                                "filename": f,
                                "path": sub_path,
                                "lines": _fast_count_subtitle_lines(sub_path)
                            })

                    size_mb = 0
                    mtime = 0.0
                    size_bytes = 0
                    mtime_ns = 0
                    try:
                        st = os.stat(video_full_path)
                        size_bytes = int(st.st_size)
                        size_mb = round(st.st_size / (1024 * 1024), 1)
                        mtime = float(st.st_mtime)
                        mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
                    except Exception:
                        pass

                    has_external_target = any(is_target_language_subtitle(sub["filename"], target_aliases) for sub in subs)

                    # Check persistent cache / L1 cache
                    cache_entry = cached_tracks_map.get(video_full_path)
                    if cache_entry is None:
                        l1 = _EMBEDDED_TRACKS_CACHE.get(video_full_path)
                        if l1 and l1[0] == size_bytes and l1[1] == mtime_ns:
                            cache_entry = l1[2]

                    # Check if this is an expired failure
                    if cache_entry is not None and isinstance(cache_entry, dict) and cache_entry.get("status") == "failed":
                        if _is_failed_probe_expired(cache_entry):
                            cache_entry = None  # Treat as cache miss for re-probing
                            _EMBEDDED_TRACKS_CACHE.pop(video_full_path, None)

                    if cache_entry is not None:
                        # CACHE HIT (either "ok" or recent "failed" within cooldown)
                        if isinstance(cache_entry, list):
                            probe_status = "ok"
                            sub_tracks = cache_entry
                        elif isinstance(cache_entry, dict):
                            probe_status = cache_entry.get("status", "ok")
                            sub_tracks = cache_entry.get("tracks", []) or []
                        else:
                            probe_status = "ok"
                            sub_tracks = []

                        if probe_status == "ok":
                            embedded_status_known = True
                            has_embedded_target = any(is_qualifying_embedded_subtitle_track(t, target_aliases) for t in sub_tracks)
                            has_target_sub = has_external_target or has_embedded_target
                            has_any_sub = len(subs) > 0 or len(sub_tracks) > 0
                            if has_external_target and has_embedded_target:
                                target_sub_source = "both"
                            elif has_external_target:
                                target_sub_source = "external"
                            elif has_embedded_target:
                                target_sub_source = "embedded"
                            else:
                                target_sub_source = None
                        else:
                            # Recent cached probe failure (within cooldown): do NOT enqueue retry
                            has_embedded_target = False
                            if has_external_target:
                                embedded_status_known = True
                                has_target_sub = True
                                target_sub_source = "external"
                                has_any_sub = True
                            else:
                                embedded_status_known = False
                                has_target_sub = False
                                target_sub_source = None
                                has_any_sub = len(subs) > 0
                    else:
                        # CACHE MISS or EXPIRED FAILURE
                        if has_external_target:
                            # External target exists: Complete immediately, NEVER probe or queue!
                            embedded_status_known = True
                            has_embedded_target = False
                            has_target_sub = True
                            target_sub_source = "external"
                            has_any_sub = True
                        else:
                            # External target missing & (no cache or expired failure) -> queue for background probing
                            uncached_to_probe.append((video_full_path, size_bytes, mtime_ns))
                            embedded_status_known = False
                            has_embedded_target = False
                            has_target_sub = False
                            target_sub_source = None
                            has_any_sub = len(subs) > 0

                    show_episodes.append({
                        "filename": file,
                        "path": video_full_path,
                        "season": season_name,
                        "size_mb": size_mb,
                        "mtime": mtime,
                        "subtitles": subs,
                        "has_any_sub": has_any_sub,
                        "has_target_sub": has_target_sub,
                        "has_embedded_target": has_embedded_target,
                        "embedded_status_known": embedded_status_known,
                        "target_sub_source": target_sub_source
                    })

                if show_episodes:
                    max_mtime = max((ep.get("mtime", 0.0) for ep in show_episodes), default=0.0)
                    results.append({
                        "title": show_name,
                        "mtime": max_mtime,
                        "episodes": show_episodes
                    })
        except Exception as e:
            logger.error(f"Error scanning series path {root_path}: {e}")

    else:
        # Movies scan
        try:
            discovered_movies: List[tuple[str, str, List[str]]] = []
            all_video_tuples = []

            for root, _, files in os.walk(root_path):
                v_files = [f for f in sorted(files) if f.lower().endswith(VIDEO_EXTS)]
                if not v_files:
                    continue
                s_files = [f for f in files if f.lower().endswith(".srt")]

                for file in v_files:
                    video_full_path = os.path.normpath(os.path.join(root, file))
                    size_bytes = 0
                    mtime_ns = 0
                    try:
                        st = os.stat(video_full_path)
                        size_bytes = int(st.st_size)
                        mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
                    except Exception:
                        pass
                    all_video_tuples.append((video_full_path, size_bytes, mtime_ns))
                    discovered_movies.append((file, video_full_path, s_files))

            # Bulk query persistent SQLite cache
            cached_tracks_map = bulk_get_cached_embedded_subtitle_tracks(all_video_tuples)

            for file, video_full_path, s_files in discovered_movies:
                base_name, _ = os.path.splitext(file)

                subs = []
                for f in s_files:
                    if is_subtitle_for_video(base_name, f):
                        sub_path = os.path.normpath(os.path.join(os.path.dirname(video_full_path), f))
                        subs.append({
                            "filename": f,
                            "path": sub_path,
                            "lines": _fast_count_subtitle_lines(sub_path)
                        })

                size_mb = 0
                mtime = 0.0
                size_bytes = 0
                mtime_ns = 0
                try:
                    st = os.stat(video_full_path)
                    size_bytes = int(st.st_size)
                    size_mb = round(st.st_size / (1024 * 1024), 1)
                    mtime = float(st.st_mtime)
                    mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
                except Exception:
                    pass

                has_external_target = any(is_target_language_subtitle(sub["filename"], target_aliases) for sub in subs)

                cache_entry = cached_tracks_map.get(video_full_path)
                if cache_entry is None:
                    l1 = _EMBEDDED_TRACKS_CACHE.get(video_full_path)
                    if l1 and l1[0] == size_bytes and l1[1] == mtime_ns:
                        cache_entry = l1[2]

                # Check if this is an expired failure
                if cache_entry is not None and isinstance(cache_entry, dict) and cache_entry.get("status") == "failed":
                    if _is_failed_probe_expired(cache_entry):
                        cache_entry = None  # Treat as cache miss for re-probing
                        _EMBEDDED_TRACKS_CACHE.pop(video_full_path, None)

                if cache_entry is not None:
                    # CACHE HIT (either "ok" or recent "failed" within cooldown)
                    if isinstance(cache_entry, list):
                        probe_status = "ok"
                        sub_tracks = cache_entry
                    elif isinstance(cache_entry, dict):
                        probe_status = cache_entry.get("status", "ok")
                        sub_tracks = cache_entry.get("tracks", []) or []
                    else:
                        probe_status = "ok"
                        sub_tracks = []

                    if probe_status == "ok":
                        embedded_status_known = True
                        has_embedded_target = any(is_qualifying_embedded_subtitle_track(t, target_aliases) for t in sub_tracks)
                        has_target_sub = has_external_target or has_embedded_target
                        has_any_sub = len(subs) > 0 or len(sub_tracks) > 0
                        if has_external_target and has_embedded_target:
                            target_sub_source = "both"
                        elif has_external_target:
                            target_sub_source = "external"
                        elif has_embedded_target:
                            target_sub_source = "embedded"
                        else:
                            target_sub_source = None
                    else:
                        # Recent cached probe failure (within cooldown): do NOT enqueue retry
                        has_embedded_target = False
                        if has_external_target:
                            embedded_status_known = True
                            has_target_sub = True
                            target_sub_source = "external"
                            has_any_sub = True
                        else:
                            embedded_status_known = False
                            has_target_sub = False
                            target_sub_source = None
                            has_any_sub = len(subs) > 0
                else:
                    # CACHE MISS or EXPIRED FAILURE
                    if has_external_target:
                        # External target exists: Complete immediately, NEVER probe or queue!
                        embedded_status_known = True
                        has_embedded_target = False
                        has_target_sub = True
                        target_sub_source = "external"
                        has_any_sub = True
                    else:
                        # External target missing & (no cache or expired failure) -> queue for background probing
                        uncached_to_probe.append((video_full_path, size_bytes, mtime_ns))
                        embedded_status_known = False
                        has_embedded_target = False
                        has_target_sub = False
                        target_sub_source = None
                        has_any_sub = len(subs) > 0

                results.append({
                    "filename": file,
                    "path": video_full_path,
                    "size_mb": size_mb,
                    "mtime": mtime,
                    "subtitles": subs,
                    "has_any_sub": has_any_sub,
                    "has_target_sub": has_target_sub,
                    "has_embedded_target": has_embedded_target,
                    "embedded_status_known": embedded_status_known,
                    "target_sub_source": target_sub_source
                })
        except Exception as e:
            logger.error(f"Error scanning movies path {root_path}: {e}")

    # Enqueue only media that is actually missing external target subs & uncached
    if uncached_to_probe:
        embedded_prober.enqueue(uncached_to_probe)

    return results
