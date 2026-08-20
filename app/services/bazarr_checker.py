import os
import glob
from typing import Optional

def check_existing_swedish_subtitle(video_path: str) -> Optional[str]:
    """
    Checks if a Swedish subtitle already exists for the given video file.
    Checks common patterns:
      - <video_name>.sv.srt
      - <video_name>.swe.srt
      - <video_name>.swedish.srt
      - <video_name>.sv.forced.srt
    """
    base_path, _ = os.path.splitext(video_path)
    directory = os.path.dirname(video_path)
    base_name = os.path.basename(base_path)

    patterns = [
        f"{base_path}.sv.srt",
        f"{base_path}.swe.srt",
        f"{base_path}.swedish.srt",
        f"{base_path}.sv.default.srt",
        f"{base_path}.srt",  # In cases where local is Swedish default
    ]

    for p in patterns:
        if os.path.exists(p) and os.path.getsize(p) > 100:
            return p

    # Fuzzy glob search in same directory
    if os.path.exists(directory):
        for f in glob.glob(os.path.join(directory, f"{glob.escape(base_name)}*.srt")):
            fname = os.path.basename(f).lower()
            if (".sv." in fname or ".swe." in fname or ".swedish." in fname) and os.path.getsize(f) > 100:
                return f

    return None

def check_existing_english_subtitle(video_path: str) -> Optional[str]:
    """
    Checks if an external English subtitle already exists (e.g. downloaded by Bazarr).
    """
    base_path, _ = os.path.splitext(video_path)
    directory = os.path.dirname(video_path)
    base_name = os.path.basename(base_path)

    patterns = [
        f"{base_path}.en.srt",
        f"{base_path}.eng.srt",
        f"{base_path}.english.srt",
        f"{base_path}.en.default.srt",
    ]

    for p in patterns:
        if os.path.exists(p) and os.path.getsize(p) > 100:
            return p

    if os.path.exists(directory):
        for f in glob.glob(os.path.join(directory, f"{glob.escape(base_name)}*.srt")):
            fname = os.path.basename(f).lower()
            if (".en." in fname or ".eng." in fname or ".english." in fname) and os.path.getsize(f) > 100:
                return f

    return None
