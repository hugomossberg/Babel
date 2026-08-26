import os
import re
import logging
import json
import srt
from typing import List, Dict, Any
from app.core.db import get_setting

from app.core.languages import get_language

logger = logging.getLogger("babel.scanner")

VIDEO_EXTS = (".mkv", ".mp4", ".m4v", ".avi")

_SUB_LINE_CACHE: Dict[str, tuple[float, int, int]] = {}

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
    target_set = {a.lower() for a in target_aliases}
    return any(t in target_set for t in tokens)

def scan_library_folders(root_path: str, category: str = "series") -> List[Dict[str, Any]]:
    """
    Scans the media library path and returns a list of media items
    along with their existing external subtitles.
    """
    if not root_path or not os.path.exists(root_path):
        return []

    target_aliases = _get_target_lang_aliases()
    results = []

    if category == "series":
        try:
            for show_name in sorted(os.listdir(root_path)):
                show_dir = os.path.normpath(os.path.join(root_path, show_name))
                if not os.path.isdir(show_dir):
                    continue

                show_episodes = []
                for root, _, files in os.walk(show_dir):
                    v_files = [f for f in sorted(files) if f.lower().endswith(VIDEO_EXTS)]
                    if not v_files:
                        continue
                    s_files = [f for f in files if f.lower().endswith(".srt")]
                    rel_season = os.path.basename(root)
                    season_name = rel_season if rel_season != show_name else "Root"

                    for file in v_files:
                        video_full_path = os.path.normpath(os.path.join(root, file))
                        base_name, _ = os.path.splitext(file)

                        subs = []
                        for f in s_files:
                            if is_subtitle_for_video(base_name, f):
                                sub_path = os.path.normpath(os.path.join(root, f))
                                subs.append({
                                    "filename": f,
                                    "path": sub_path,
                                    "lines": _fast_count_subtitle_lines(sub_path)
                                })

                        size_mb = 0
                        try:
                            size_mb = round(os.path.getsize(video_full_path) / (1024 * 1024), 1)
                        except Exception:
                            pass

                        has_target_sub = any(is_target_language_subtitle(sub["filename"], target_aliases) for sub in subs)

                        show_episodes.append({
                            "filename": file,
                            "path": video_full_path,
                            "season": season_name,
                            "size_mb": size_mb,
                            "subtitles": subs,
                            "has_any_sub": len(subs) > 0,
                            "has_target_sub": has_target_sub
                        })

                if show_episodes:
                    results.append({
                        "title": show_name,
                        "episodes": show_episodes
                    })
        except Exception as e:
            logger.error(f"Error scanning series path {root_path}: {e}")

    else:
        # Movies scan
        try:
            for root, _, files in os.walk(root_path):
                v_files = [f for f in sorted(files) if f.lower().endswith(VIDEO_EXTS)]
                if not v_files:
                    continue
                s_files = [f for f in files if f.lower().endswith(".srt")]

                for file in v_files:
                    video_full_path = os.path.normpath(os.path.join(root, file))
                    base_name, _ = os.path.splitext(file)

                    subs = []
                    for f in s_files:
                        if is_subtitle_for_video(base_name, f):
                            sub_path = os.path.normpath(os.path.join(root, f))
                            subs.append({
                                "filename": f,
                                "path": sub_path,
                                "lines": _fast_count_subtitle_lines(sub_path)
                            })

                    size_mb = 0
                    try:
                        size_mb = round(os.path.getsize(video_full_path) / (1024 * 1024), 1)
                    except Exception:
                        pass

                    has_target_sub = any(is_target_language_subtitle(sub["filename"], target_aliases) for sub in subs)

                    results.append({
                        "filename": file,
                        "path": video_full_path,
                        "size_mb": size_mb,
                        "subtitles": subs,
                        "has_any_sub": len(subs) > 0,
                        "has_target_sub": has_target_sub
                    })
        except Exception as e:
            logger.error(f"Error scanning movies path {root_path}: {e}")

    return results
