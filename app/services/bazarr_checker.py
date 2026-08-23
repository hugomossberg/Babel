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

    # 2. Strict boundary directory search
    if os.path.exists(directory):
        base_name_lower = base_name.lower()
        if lang_obj:
            target_aliases = [a.lower() for a in lang_obj.aliases]
        else:
            target_aliases = [lang_code.lower()]

        try:
            for fname in os.listdir(directory):
                fname_lower = fname.lower()
                if not fname_lower.endswith(".srt"):
                    continue
                if ".temp" in fname_lower or ".tmp" in fname_lower or ".babel-replaced" in fname_lower:
                    continue
                if not fname_lower.startswith(base_name_lower):
                    continue
                remainder = fname_lower[len(base_name_lower):]
                # Remainder must start with '.' (strict boundary) and end with '.srt'
                if not remainder.startswith(".") or not remainder.endswith(".srt"):
                    continue
                middle = remainder[1:-4]
                if not middle:
                    continue
                parts = middle.split(".")
                if any(tag in ["forced", "signs", "songs"] for tag in parts):
                    continue

                matched = False
                for lang in target_aliases:
                    if lang in parts:
                        matched = True
                        break
                    if any(p.startswith(f"{lang}-") or p.startswith(f"{lang}_") for p in parts):
                        matched = True
                        break

                if matched:
                    full_p = os.path.join(directory, fname)
                    if os.path.getsize(full_p) > 100:
                        return full_p
        except Exception:
            pass

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
