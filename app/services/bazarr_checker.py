import os
import glob
from typing import Optional

def find_external_subtitle(video_path: str, lang_code: str) -> Optional[str]:
    """
    Checks if an external subtitle already exists for the given video file and language code.
    Checks common patterns based on the language code.
    """
    base_path, _ = os.path.splitext(video_path)
    directory = os.path.dirname(video_path)
    base_name = os.path.basename(base_path)

    lang_map = {
        "sv": [".sv", ".swe", ".swedish", ".sv.default", ".sv.forced"],
        "en": [".en", ".eng", ".english", ".en.default", ".en.forced"],
        "de": [".de", ".ger", ".german", ".de.default", ".de.forced"],
        "fr": [".fr", ".fre", ".french", ".fr.default", ".fr.forced"]
    }

    suffixes = lang_map.get(lang_code.lower(), [f".{lang_code.lower()}"])

    # 1. Exact matches
    for suffix in suffixes:
        p = f"{base_path}{suffix}.srt"
        if os.path.exists(p) and os.path.getsize(p) > 100:
            return p

    # 2. Fuzzy glob search
    if os.path.exists(directory):
        for f in glob.glob(os.path.join(directory, f"{glob.escape(base_name)}*.srt")):
            fname = os.path.basename(f).lower()
            # Do NOT include bare .srt match. It must have the lang suffix.
            for suffix in suffixes:
                if f"{suffix}." in fname and os.path.getsize(f) > 100:
                    return f

    return None

def check_existing_swedish_subtitle(video_path: str) -> Optional[str]:
    """
    Checks if a Swedish subtitle already exists for the given video file.
    Checks common patterns:
      - <video_name>.sv.srt
      - <video_name>.swe.srt
      - <video_name>.swedish.srt
      - <video_name>.sv.forced.srt
    """
    return find_external_subtitle(video_path, "sv")

def check_existing_english_subtitle(video_path: str) -> Optional[str]:
    """
    Checks if an external English subtitle already exists (e.g. downloaded by Bazarr).
    """
    return find_external_subtitle(video_path, "en")
