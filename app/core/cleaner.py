import re
from typing import List, Tuple
import srt

# Regex for SDH descriptions: [door closes], (chuckles), [SIGHING], etc.
SDH_BRACKET_REGEX = re.compile(r'\[.*?\]|\(.*?\)', re.DOTALL)

# Regex for music notes and signs: ♪, ♫, ♬, ♩
MUSIC_NOTES_REGEX = re.compile(r'[♪♫♬♩]+')

# Placeholder that preserves the block structure for parsers and locks timestamps
EMPTY_PLACEHOLDER = "<i></i>"

def clean_subtitle_text(text: str) -> str:
    """
    Cleans SDH noise and music notes from a subtitle line.
    If the line becomes completely empty or only contains noise,
    returns the EMPTY_PLACEHOLDER '<i></i>' to preserve sync lock.
    """
    if not text:
        return EMPTY_PLACEHOLDER

    # Remove SDH brackets [ ... ] and ( ... )
    cleaned = SDH_BRACKET_REGEX.sub('', text)

    # Remove music notes
    cleaned = MUSIC_NOTES_REGEX.sub('', cleaned)

    # Clean whitespace and strip empty lines
    lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
    cleaned_text = '\n'.join(lines).strip()

    if not cleaned_text or cleaned_text == "":
        return EMPTY_PLACEHOLDER

    return cleaned_text

def sanitize_srt_content(srt_content: str) -> Tuple[List[srt.Subtitle], int]:
    """
    Parses an SRT file content, cleans SDH/music noise from all subtitle blocks,
    and returns a list of sanitized Subtitle objects along with cleaned count.
    """
    subs = list(srt.parse(srt_content))
    cleaned_count = 0

    valid_subs = []
    for sub in subs:
        duration_ms = (sub.end - sub.start).total_seconds() * 1000
        if duration_ms < 100:
            continue
            
        original = sub.content
        cleaned = clean_subtitle_text(original)
        if original != cleaned:
            cleaned_count += 1
            
        sub.content = cleaned
        valid_subs.append(sub)

    return valid_subs, cleaned_count

def subs_to_srt_string(subs: List[srt.Subtitle]) -> str:
    """Formats subtitle objects back to valid SRT text, ensuring sequential numbering."""
    for i, sub in enumerate(subs):
        sub.index = i + 1
    return srt.compose(subs)
