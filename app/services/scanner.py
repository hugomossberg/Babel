import os
import logging
import json
import srt
from typing import List, Dict, Any
from app.core.db import get_setting

logger = logging.getLogger("babel.scanner")

VIDEO_EXTS = (".mkv", ".mp4", ".m4v", ".avi")

def _fast_count_subtitle_lines(path: str) -> int:
    """Cheaply estimate subtitle line count by counting '-->' without parsing."""
    try:
        count = 0
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                count += chunk.count(b"-->")
        return count
    except Exception:
        return 0

def _get_target_lang_codes():
    """Get configured target language codes."""
    raw = get_setting("languages", '[]')
    try:
        langs = json.loads(raw)
        return [l["code"] for l in langs if l.get("enabled", True)]
    except Exception:
        return ["sv"]

def scan_library_folders(root_path: str, category: str = "series") -> List[Dict[str, Any]]:
    """
    Scans the media library path and returns a list of media items
    along with their existing external subtitles.
    """
    if not root_path or not os.path.exists(root_path):
        return []

    target_langs = _get_target_lang_codes()
    results = []

    if category == "series":
        # Group by show directory
        try:
            for show_name in sorted(os.listdir(root_path)):
                show_dir = os.path.normpath(os.path.join(root_path, show_name))
                if not os.path.isdir(show_dir):
                    continue

                show_episodes = []
                for root, _, files in os.walk(show_dir):
                    for file in sorted(files):
                        if file.lower().endswith(VIDEO_EXTS):
                            video_full_path = os.path.normpath(os.path.join(root, file))
                            base_name, _ = os.path.splitext(file)

                            # Check for external subtitles in the same folder (ignoring temp files)
                            subs = []
                            import re
                            pattern = re.compile(re.escape(base_name) + r'(\.[a-z]{2,3}(-[A-Z]{2})?)?(\.(forced|hi|sdh|cc|signs|songs))?\.srt$', re.IGNORECASE)
                            try:
                                for f in os.listdir(root):
                                    if pattern.match(f) and ".temp" not in f:
                                        sub_path = os.path.normpath(os.path.join(root, f))
                                        subs.append({
                                            "filename": f,
                                            "path": sub_path,
                                            "lines": _fast_count_subtitle_lines(sub_path)
                                        })
                            except Exception:
                                pass

                            size_mb = 0
                            try:
                                size_mb = round(os.path.getsize(video_full_path) / (1024 * 1024), 1)
                            except Exception:
                                pass

                            rel_season = os.path.basename(root)

                            has_target_sub = False
                            for sub in subs:
                                fname_lower = sub["filename"].lower()
                                if "forced" in fname_lower or "signs" in fname_lower or "songs" in fname_lower:
                                    continue
                                if any(f".{lang}." in fname_lower for lang in target_langs):
                                    has_target_sub = True
                                    break

                            show_episodes.append({
                                "filename": file,
                                "path": video_full_path,
                                "season": rel_season if rel_season != show_name else "Root",
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
                for file in sorted(files):
                    if file.lower().endswith(VIDEO_EXTS):
                        video_full_path = os.path.normpath(os.path.join(root, file))
                        base_name, _ = os.path.splitext(file)

                        subs = []
                        import re
                        pattern = re.compile(re.escape(base_name) + r'(\.[a-z]{2,3}(-[A-Z]{2})?)?(\.(forced|hi|sdh|cc|signs|songs))?\.srt$', re.IGNORECASE)
                        try:
                            for f in os.listdir(root):
                                if pattern.match(f) and ".temp" not in f:
                                    sub_path = os.path.normpath(os.path.join(root, f))
                                    subs.append({
                                        "filename": f,
                                        "path": sub_path,
                                        "lines": _fast_count_subtitle_lines(sub_path)
                                    })
                        except Exception:
                            pass

                        size_mb = 0
                        try:
                            size_mb = round(os.path.getsize(video_full_path) / (1024 * 1024), 1)
                        except Exception:
                            pass

                        has_target_sub = False
                        for sub in subs:
                            fname_lower = sub["filename"].lower()
                            if "forced" in fname_lower or "signs" in fname_lower or "songs" in fname_lower:
                                continue
                            if any(f".{lang}." in fname_lower for lang in target_langs):
                                has_target_sub = True
                                break

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
