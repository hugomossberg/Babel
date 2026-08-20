from typing import Optional, List

class Language:
    def __init__(self, code: str, aliases: List[str], display_name: str, deepl_code: Optional[str] = None):
        self.code = code
        self.aliases = [a.lower() for a in aliases] + [code.lower()]
        self.display_name = display_name
        self.deepl_code = deepl_code or code.upper()

LANGUAGES = [
    Language("sv", ["swe", "swedish", "sve"], "Swedish", "SV"),
    Language("en", ["eng", "english"], "English", "EN-US"),
    Language("de", ["deu", "ger", "german", "tyska"], "German", "DE"),
    Language("fr", ["fra", "fre", "french", "franska"], "French", "FR"),
    Language("es", ["spa", "spanish", "spanska"], "Spanish", "ES"),
    Language("it", ["ita", "italian", "italienska"], "Italian", "IT"),
    Language("nl", ["nld", "dut", "dutch", "holländska"], "Dutch", "NL"),
    Language("pl", ["pol", "polish", "polska"], "Polish", "PL"),
    Language("pt", ["por", "portuguese", "portugisiska"], "Portuguese", "PT-PT"),
    Language("ru", ["rus", "russian", "ryska"], "Russian", "RU"),
    Language("ja", ["jpn", "japanese", "japanska"], "Japanese", "JA"),
    Language("zh", ["zho", "chi", "chinese", "kinesiska"], "Chinese", "ZH"),
    Language("ko", ["kor", "korean", "koreanska"], "Korean", "KO"),
    Language("fi", ["fin", "finnish", "finska"], "Finnish", "FI"),
    Language("da", ["dan", "danish", "danska"], "Danish", "DA"),
    Language("no", ["nor", "nob", "nno", "norwegian", "norska"], "Norwegian", "NB")
]

def get_language(query: str) -> Optional[Language]:
    if not query: return None
    q = query.lower().strip()
    for lang in LANGUAGES:
        if q in lang.aliases:
            return lang
        if q.startswith(lang.display_name.lower()):
            return lang
    return None
