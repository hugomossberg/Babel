from typing import Optional, List

class Language:
    def __init__(self, code: str, aliases: List[str], display_name: str, deepl_code: Optional[str] = None, deepl_source_code: Optional[str] = None):
        self.code = code
        self.aliases = list(dict.fromkeys([a.lower() for a in aliases] + [code.lower()]))
        self.display_name = display_name
        self.deepl_code = deepl_code or code.upper()
        self.deepl_source_code = deepl_source_code or (self.deepl_code.split("-")[0] if self.deepl_code else self.code.upper())

LANGUAGES = [
    Language("sv", ["swe", "swedish", "sve", "svenska"], "Swedish", "SV"),
    Language("en", ["eng", "english"], "English", "EN-US"),
    Language("de", ["deu", "ger", "german", "deutsch"], "German", "DE"),
    Language("fr", ["fra", "fre", "french", "francais", "français"], "French", "FR"),
    Language("es", ["spa", "spanish", "espanol", "español"], "Spanish", "ES"),
    Language("it", ["ita", "italian", "italiano"], "Italian", "IT"),
    Language("nl", ["nld", "dut", "dutch", "nederlands"], "Dutch", "NL"),
    Language("pl", ["pol", "polish", "polski"], "Polish", "PL"),
    Language("pt", ["por", "portuguese", "português"], "Portuguese", "PT-PT"),
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
    for lang in LANGUAGES:
        if q in lang.aliases:
            return lang
        if q == lang.display_name.lower() or q.startswith(lang.display_name.lower()):
            return lang
    return None

def normalize_language_code(query: str, default: Optional[str] = None) -> str:
    """
    Central language normalization returning canonical ISO 639-1 code.
    Fallback to default if provided, or sanitized 2-char code / raw query.
    """
    if not query:
        return default or "unknown"
    lang_obj = get_language(query)
    if lang_obj:
        return lang_obj.code
    q = query.lower().strip()
    return default if default is not None else q[:2]

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
    Canonical lookup returning valid DeepL target language code (e.g. 'DE', 'ES', 'NL', 'SV', 'PT-PT', 'FR', 'EN-US').
    """
    if not query:
        return default
    lang = get_language(query)
    if lang:
        return lang.deepl_code
    q = query.strip().upper()
    return q if q else default
