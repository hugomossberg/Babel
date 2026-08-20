import re
import srt
from typing import List, Dict, Any, Tuple, Optional
from app.core.languages import get_language, LANGUAGES

def detect_language_heuristics(text: str) -> str:
    """
    Generaliserad språkdetektering.
    Skannar mot inbyggda lexikon från app.core.languages
    och returnerar språkkoden (tex 'sv', 'en', 'de').
    """
    words = set(re.findall(r'\b[a-zåäöA-ZÅÄÖ]+\b', text.lower()))
    if not words:
        return "unknown"
    
    # Simple hardcoded dictionaries for demonstration since we lack pycld2
    DICTIONARIES = {
        "en": {"the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me"},
        "sv": {"och", "det", "att", "i", "en", "jag", "hon", "som", "han", "på", "den", "med", "var", "sig", "för", "så", "till", "är", "men", "ett", "om", "hade", "de", "av", "icke", "mig", "du", "henne", "då", "sin", "nu", "har", "inte", "hans", "honom", "skulle", "hennes", "där", "min", "man", "ej", "vid", "kunde", "något", "från", "ut", "när", "efter", "upp", "vi"},
        "de": {"der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich", "des", "auf", "für", "ist", "im", "dem", "nicht", "ein", "die", "eine", "als", "auch", "es", "an", "werden", "aus", "er", "hat", "dass", "sie", "nach", "wird", "bei", "einer", "der", "um", "am", "sind", "noch", "wie", "einem", "über", "einen", "das", "so", "sie", "zum", "war", "haben", "nur"},
        "fr": {"le", "la", "les", "et", "de", "un", "une", "des", "en", "à", "que", "qui", "dans", "pour", "il", "elle", "est", "sont", "pas", "ne", "ce", "se", "sur", "avec", "par", "je", "tu", "nous", "vous", "ils", "elles", "au", "aux", "ou", "où", "mon", "ton", "son", "ma", "ta", "sa", "mes", "tes", "ses", "notre", "votre", "leur", "nos", "vos", "leurs", "être", "avoir"},
        "es": {"el", "la", "los", "las", "y", "de", "un", "una", "unos", "unas", "en", "a", "que", "quien", "dentro", "para", "él", "ella", "es", "son", "no", "ni", "este", "esta", "se", "sobre", "con", "por", "yo", "tú", "nosotros", "vosotros", "ellos", "ellas", "al", "o", "mi", "tu", "su", "mis", "tus", "sus", "nuestro", "vuestro", "ser", "estar", "tener", "hacer", "decir"},
    }

    scores = {}
    for lang_code, dic in DICTIONARIES.items():
        score = len(words.intersection(dic))
        if score > 0:
            scores[lang_code] = score
            
    if not scores:
        return "unknown"
        
    best_lang = max(scores, key=scores.get)
    if scores[best_lang] >= 3:
        return best_lang
    return "unknown"
