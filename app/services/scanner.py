import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger("babel.scanner")

VIDEO_EXTS = (".mkv", ".mp4", ".avi")

def scan_library_folders(root_path: str, category: str = "series") -> List[Dict[str, Any]]:
    """
    Scans the media library path and returns a list of media items
    along with their existing external subtitles.
    """
    if not root_path or not os.path.exists(root_path):
        return []

    results = []

    if category == "series":
        # Group by show directory
        try:
            for show_name in sorted(os.listdir(root_path)):
                show_dir = os.path.join(root_path, show_name)
                if not os.path.isdir(show_dir):
                    continue

                show_episodes = []
                for root, _, files in os.walk(show_dir):
                    for file in sorted(files):
                        if file.lower().endswith(VIDEO_EXTS):
                            video_full_path = os.path.join(root, file)
                            base_name, _ = os.path.splitext(file)
                            
                            # Check for external subtitles in the same folder (ignoring temp files)
                            subs = []
                            try:
                                for f in os.listdir(root):
                                    if f.startswith(base_name) and f.endswith(".srt") and not ".temp" in f:
                                        sub_path = os.path.join(root, f)
                                        line_count = 0
                                        try:
                                            with open(sub_path, "r", encoding="utf-8", errors="ignore") as sf:
                                                line_count = sum(1 for line in sf if line.strip().isdigit())
                                        except Exception:
                                            pass
                                        subs.append({
                                            "filename": f,
                                            "path": sub_path,
                                            "lines": line_count
                                        })
                            except Exception:
                                pass

                            size_mb = 0
                            try:
                                size_mb = round(os.path.getsize(video_full_path) / (1024 * 1024), 1)
                            except Exception:
                                pass

                            rel_season = os.path.basename(root)
                            show_episodes.append({
                                "filename": file,
                                "path": video_full_path,
                                "season": rel_season if rel_season != show_name else "Root",
                                "size_mb": size_mb,
                                "subtitles": subs,
                                "has_any_sub": len(subs) > 0
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
                        video_full_path = os.path.join(root, file)
                        base_name, _ = os.path.splitext(file)
                        
                        subs = []
                        try:
                            for f in os.listdir(root):
                                if f.startswith(base_name) and f.endswith(".srt") and not ".temp" in f:
                                    sub_path = os.path.join(root, f)
                                    line_count = 0
                                    try:
                                        with open(sub_path, "r", encoding="utf-8", errors="ignore") as sf:
                                            line_count = sum(1 for line in sf if line.strip().isdigit())
                                    except Exception:
                                        pass
                                    subs.append({
                                        "filename": f,
                                        "path": sub_path,
                                        "lines": line_count
                                    })
                        except Exception:
                            pass

                        size_mb = 0
                        try:
                            size_mb = round(os.path.getsize(video_full_path) / (1024 * 1024), 1)
                        except Exception:
                            pass

                        results.append({
                            "filename": file,
                            "path": video_full_path,
                            "size_mb": size_mb,
                            "subtitles": subs,
                            "has_any_sub": len(subs) > 0
                        })
        except Exception as e:
            logger.error(f"Error scanning movies path {root_path}: {e}")

    return results
