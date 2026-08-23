import re
from typing import List, Tuple
import srt

# Strong evidence SDH patterns (sound effects, reactions, music indicators, language/audio directions)
SDH_PATTERNS = [
    # Vocalizations & non-verbal reactions
    re.compile(r"^(?:softly\s+|loudly\s+|both\s+|all\s+|crowd\s+|audience\s+|distant\s+|heavy\s+|people\s+|reporters?\s+|men\s+|women\s+)?(?:laughing|laughs|laughter|chuckle|chuckles|sigh|sighs|sighing|gasp|gasps|gasping|groan|groans|groaning|grunt|grunts|grunting|clears throat|snicker|snickers|snort|snorts|sniffles?|crying|cries|sobs|sobbing|panting|pants|cheering|cheers|applause|applauding|screaming|screams|yells|yelling|shouts|shouting|exhales|inhales|cough|coughs|coughing|sneezes?|sneezing|whispers|whispering|shushing|humming|whistling|breathing|clamoring|clamor|murmuring|murmurs|chattering|commotion)$", re.IGNORECASE),
    # Music descriptions
    re.compile(r"^(?:(?:dramatic|soft|upbeat|electronic|orchestral|rock|classical|theme|suspenseful|eerie|instrumental|jazz|pop|techno|ambient|piano|guitar|synthesizer|hip-hop|indie|folk|country|somber|sombre|tense|triumphant|ominous|playful|fast|slow|loud|faint|distant)?\s*music(?:\s+(?:playing|starts|fades|swells|stops|resumes|plays|ends|continues|in background))?|(?:music|song|tune|melody)\s+(?:playing|starts|fades|swells|stops|resumes|plays|ends|continues|in background)|theme song(?: plays| playing)?)$", re.IGNORECASE),
    # Environmental sounds & Foley
    re.compile(r"^(?:distant\s+|loud\s+|faint\s+|sudden\s+)?(?:door\s+(?:opens?|closes?|slams?|creaks?|knocks?|knocking|unlocks?|locks?|clicks?|shut|jiggles?)|phone\s+(?:rings?|ringing|buzzes?|buzzing|vibrates?|vibrating|chimes?|beeps?)|engine\s+(?:starts?|revs?|roars?|idles?|hums?|stops?|cuts out)|car\s+(?:starts?|horns?|honking|honks?|accelerates?|drives off|passes?|approaches?|speeds off|skids?|skidding|crashes?|crashing|engine)|tires?\s+(?:screech(?:es|ing)?|squeals?|squealing)|glass\s+(?:shatters?|shattering|breaks?|breaking|clinks?|cracks?)|footsteps(?:\s+(?:approaching|receding|running|walking|echoing|fading|pounding))?|birds?\s+(?:chirping|singing|squawking|twittering)|dogs?\s+(?:barking|barks?|growling|growls?|whining|whines?|yelping|yelps?)|cats?\s+(?:meowing|meows?|purring|purrs?|hissing|hisses?)|clock\s+(?:ticking|ticks?|chimes?)|bell\s+(?:rings?|ringing|chimes?|tolling|tolls?)|alarm\s+(?:beeps?|beeping|blares?|blaring|rings?|ringing|sounds?|sounding)|siren\s+(?:wails?|wailing|blares?|blaring|sounds?|sounding)|thunder(?:\s+(?:rumbles?|rumbling|cracks?|crashes?))?|lightning|wind\s+(?:howls?|howling|blowing|blows?|whooshes?|whooshing|gusts?)|rain\s+(?:pouring|falls?|falling|patters?|pattering)|water\s+(?:dripping|drips?|splashing|splashes?|rushing|running)|explosion|explosions|gunshots?|gunfire)$", re.IGNORECASE),
    # Language / audio indicators
    re.compile(r"^(?:(?:speaking|speaks|singing|sings)\s+(?:in\s+)?(?:foreign language|spanish|french|german|russian|japanese|arabic|chinese|italian|mandarin|cantonese|korean|portuguese|hindi|latin|sign language|broken english)|in\s+(?:foreign language|spanish|french|german|russian|japanese|arabic|chinese|italian|mandarin|cantonese|korean|portuguese|latin|sign language)|inaudible|indistinct\s+(?:chatter|talking|voices|speech|whispering|shouting|clamor)|muffled\s+(?:speech|speaking|voices|voice|screams|groans|cries)|overlapping\s+chatter|silence)$", re.IGNORECASE),
]

# Regex for music notes and signs: ♪, ♫, ♬, ♩
MUSIC_NOTES_REGEX = re.compile(r'[♪♫♬♩]+')

# Placeholder that preserves the block structure for parsers and locks timestamps
EMPTY_PLACEHOLDER = "<i></i>"

def is_sdh_description(inner: str) -> bool:
    clean = inner.strip().strip(".,!?:;\"\'")
    if not clean:
        return True
    return any(p.match(clean) for p in SDH_PATTERNS)

def clean_subtitle_text(text: str) -> str:
    """
    Cleans SDH noise and music notes from a subtitle line.
    If the line becomes completely empty or only contains noise,
    returns the EMPTY_PLACEHOLDER '<i></i>' to preserve sync lock.
    """
    if not text:
        return EMPTY_PLACEHOLDER

    def replace_bracket(m):
        inner = m.group(1)
        return "" if is_sdh_description(inner) else m.group(0)

    # Remove SDH descriptions in brackets [ ... ]
    cleaned = re.sub(r'\[(.*?)\]', replace_bracket, text)
    # Remove SDH descriptions in parentheses ( ... )
    cleaned = re.sub(r'\((.*?)\)', replace_bracket, cleaned)

    # Remove music notes
    cleaned = MUSIC_NOTES_REGEX.sub('', cleaned)

    # Clean whitespace and strip empty/SDH-only lines
    lines = []
    for line in cleaned.split('\n'):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        # Check if line without brackets was a pure SDH indicator
        if is_sdh_description(line_stripped):
            continue
        lines.append(line_stripped)

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
