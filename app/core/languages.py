import re
from typing import Optional, List

class Language:
    def __init__(self, code: str, aliases: List[str], display_name: str, deepl_code: Optional[str] = None, deepl_source_code: Optional[str] = None, bazarr_code: Optional[str] = None):
        self.code = code
        self.aliases = list(dict.fromkeys([a.lower() for a in aliases] + [code.lower()]))
        self.display_name = display_name
        self.deepl_code = deepl_code or code.upper()
        self.deepl_source_code = deepl_source_code or (self.deepl_code.split("-")[0] if self.deepl_code else self.code.upper())
        self.bazarr_code = bazarr_code or code

LANGUAGES = [
    Language("sv", ["swe", "swedish", "sve", "svenska"], "Swedish", "SV"),
    Language("en", ["eng", "english"], "English", "EN-US"),
    Language("de", ["deu", "ger", "german", "deutsch"], "German", "DE"),
    Language("fr", ["fra", "fre", "french", "francais", "français"], "French", "FR"),
    Language("es", ["spa", "spanish", "espanol", "español"], "Spanish", "ES"),
    Language("it", ["ita", "italian", "italiano"], "Italian", "IT"),
    Language("nl", ["nld", "dut", "dutch", "nederlands"], "Dutch", "NL"),
    Language("pl", ["pol", "polish", "polski"], "Polish", "PL"),
    Language("pt", ["por", "portuguese", "português", "portugues", "pt-pt", "pt_pt", "por-pt", "por_pt", "european portuguese", "portuguese (portugal)", "português (portugal)", "portugues (portugal)"], "Portuguese", "PT-PT", "PT", "pt"),
    Language("pt-BR", ["pt-br", "pt_br", "por-br", "por_br", "pob", "pb", "brazilian portuguese", "brazilian", "português brasileiro", "portugues brasileiro", "portugues-brasileiro", "português (brasil)", "portugues (brasil)", "portuguese (brazil)", "portuguese (brazilian)", "pt-brazil", "pt (br)"], "Brazilian Portuguese", "PT-BR", "PT", "pt-BR"),
    Language("ru", ["rus", "russian", "русский"], "Russian", "RU"),
    Language("ja", ["jpn", "japanese", "日本語"], "Japanese", "JA"),
    Language("zh", ["zho", "chi", "chinese", "zh-cn", "zh-tw", "中文"], "Chinese", "ZH"),
    Language("ko", ["kor", "korean", "한국어"], "Korean", "KO"),
    Language("fi", ["fin", "finnish", "suomi"], "Finnish", "FI"),
    Language("da", ["dan", "danish", "dansk"], "Danish", "DA"),
    Language("no", ["nor", "nob", "nno", "norwegian", "norsk"], "Norwegian", "NB"),
    Language("bg", ["bul", "bulgarian", "български"], "Bulgarian", "BG"),
    Language("cs", ["ces", "cze", "czech", "čeština", "cestina"], "Czech", "CS"),
    Language("ro", ["ron", "rum", "romanian", "română", "romana"], "Romanian", "RO"),
    Language("hu", ["hun", "hungarian", "magyar"], "Hungarian", "HU"),
    Language("tr", ["tur", "turkish", "türkçe", "turkce"], "Turkish", "TR"),
    Language("el", ["ell", "gre", "greek", "ελληνικά", "ellinika"], "Greek", "EL"),
    Language("sr", ["srp", "scc", "serbian", "српски", "srpski"], "Serbian", "SR"),
    Language("hr", ["hrv", "scr", "croatian", "hrvatski"], "Croatian", "HR"),
    Language("bs", ["bos", "bosnian", "bosanski"], "Bosnian", "BS")
]

def get_language(query: str) -> Optional[Language]:
    if not query: return None
    q = query.lower().strip()
    # 1. Exact alias match
    for lang in LANGUAGES:
        if q in lang.aliases:
            return lang
    # 2. Exact display name match
    for lang in LANGUAGES:
        if q == lang.display_name.lower():
            return lang
    # 3. Display name prefix match (e.g. "Brazilian Portuguese (Subtitles)")
    for lang in LANGUAGES:
        if q.startswith(lang.display_name.lower()):
            return lang
    # 4. Prefix match for compound queries split by hyphen/underscore/dot (e.g. "pt-br.default" -> "pt-br")
    parts = re.split(r'[-_.]', q)
    for i in range(len(parts) - 1, 0, -1):
        sub_hyphen = "-".join(parts[:i])
        sub_under = "_".join(parts[:i])
        for lang in LANGUAGES:
            if sub_hyphen in lang.aliases or sub_under in lang.aliases or sub_hyphen == lang.code.lower():
                return lang
    # 5. Base fallback for dialects (e.g. "de-AT" -> "de", "es-MX" -> "es")
    base = parts[0]
    if base and base != q:
        for lang in LANGUAGES:
            if base in lang.aliases or base == lang.code.lower():
                return lang
    return None

def normalize_language_code(query: str, default: Optional[str] = None) -> str:
    """
    Central language normalization returning canonical code (e.g. 'sv', 'en', 'pt', 'pt-BR').
    Fallback to default if provided, or sanitized code / raw query.
    """
    if not query:
        return default or "unknown"
    lang_obj = get_language(query)
    if lang_obj:
        return lang_obj.code
    q = query.strip()
    return default if default is not None else q

def get_display_language_name(query: str, default: Optional[str] = None) -> str:
    """
    Central lookup returning human-readable display name (e.g. 'Brazilian Portuguese', 'Portuguese', 'Swedish').
    Fallback to default if provided, or the sanitized query.
    """
    if not query:
        return default or "unknown"
    lang_obj = get_language(query)
    if lang_obj:
        return lang_obj.display_name
    return default if default is not None else query.strip()

def get_bazarr_language_code(query: str, default: Optional[str] = None) -> str:
    """
    Canonical lookup returning valid Bazarr language code.
    """
    if not query:
        return default or "unknown"
    lang = get_language(query)
    if lang:
        return getattr(lang, "bazarr_code", lang.code)
    return normalize_language_code(query, default=default)

def get_deepl_source_code(query: str, default: str = "EN") -> str:
    """
    Canonical lookup returning valid DeepL source language code (e.g. 'DE', 'ES', 'NL', 'SV', 'PT', 'FR', 'EN').
    """
    if not query:
        return default
    lang = get_language(query)
    if lang:
        return lang.deepl_source_code
    q = query.strip().upper()
    if len(q) == 2:
        return q
    return default

def get_deepl_target_code(query: str, default: str = "EN-US") -> str:
    """
    Canonical lookup returning valid DeepL target language code (e.g. 'DE', 'ES', 'NL', 'SV', 'PT-PT', 'PT-BR', 'EN-GB', 'FR', 'EN-US').
    """
    if not query:
        return default
    q = query.strip()
    q_norm = q.upper().replace("_", "-")
    if q_norm in ("PT-BR", "PT-PT", "EN-GB", "EN-US", "ZH-HANS", "ZH-HANT"):
        return q_norm
    lang = get_language(query)
    if lang:
        return lang.deepl_code
    return q.upper() if q else default
