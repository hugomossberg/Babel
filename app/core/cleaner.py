import re
from typing import List, Tuple
import srt

# Regex for SDH descriptions: [door closes], (chuckles), [SIGHING], etc.
SDH_BRACKET_REGEX = re.compile(r'\[.*?\]', re.DOTALL)
# Only remove parentheticals if they are all caps and stand alone on a line.
SDH_PAREN_REGEX = re.compile(r'^\([^a-z0-9]*[A-Z\s]+[^a-z0-9]*\)$', re.DOTALL)
# Also remove inline parentheticals if they contain known SDH words
SDH_KEYWORDS_REGEX = re.compile(r'\((laughing|laughs|sighs|sighing|gasps|groans|grunts|chuckles|clears throat|music playing|speaking|whispers|shouts|crying|cries|sobs|panting|pants|cheering|cheers|applauding|applause|screaming|screams|yells|yelling|exhales|inhales|grunts).*?\)', re.IGNORECASE)

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

    # Remove standard SDH brackets [ ... ]
    cleaned = SDH_BRACKET_REGEX.sub('', text)
    # Remove conservative parentheticals
    cleaned = SDH_KEYWORDS_REGEX.sub('', cleaned)
    
    # Process line by line for full-line parentheticals
    processed_lines = []
    for line in cleaned.split('\n'):
        line_stripped = line.strip()
        if SDH_PAREN_REGEX.match(line_stripped):
            continue
        processed_lines.append(line)
    cleaned = '\n'.join(processed_lines)

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

    for sub in subs:
        original = sub.content
        cleaned = clean_subtitle_text(original)
        if original != cleaned:
            cleaned_count += 1
        sub.content = cleaned

    return subs, cleaned_count

def subs_to_srt_string(subs: List[srt.Subtitle]) -> str:
    """Formats subtitle objects back to valid SRT text, ensuring sequential numbering."""
    for i, sub in enumerate(subs):
        sub.index = i + 1
    return srt.compose(subs)
