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

    from app.core.languages import get_language
    lang_obj = get_language(lang_code)
    
    if lang_obj:
        suffixes = [f".{a}" for a in lang_obj.aliases]
        suffixes.append(f".{lang_code.lower()}.default")
    else:
        suffixes = [f".{lang_code.lower()}", f".{lang_code.lower()}.default"]

    # 1. Exact matches
    for suffix in suffixes:
        p = f"{base_path}{suffix}.srt"
        if os.path.exists(p) and os.path.getsize(p) > 100:
            if "forced" not in p.lower() and "signs" not in p.lower() and "songs" not in p.lower():
                return p

    # 2. Fuzzy glob search
    if os.path.exists(directory):
        for f in glob.glob(os.path.join(directory, f"{glob.escape(base_name)}*.srt")):
            fname = os.path.basename(f).lower()
            # Do NOT include bare .srt match. It must have the lang suffix.
            for suffix in suffixes:
                if f"{suffix}." in fname and os.path.getsize(f) > 100:
                    if "forced" not in fname.lower() and "signs" not in fname.lower() and "songs" not in fname.lower():
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
