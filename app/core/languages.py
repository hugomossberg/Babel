from typing import Optional, List

class Language:
    def __init__(self, code: str, aliases: List[str], display_name: str, deepl_code: Optional[str] = None):
        self.code = code
        self.aliases = [a.lower() for a in aliases] + [code.lower()]
        self.display_name = display_name
        self.deepl_code = deepl_code or code.upper()

LANGUAGES = [
    Language("sv", ["swe", "swedish", "sve", "svenska"], "Swedish", "SV"),
    Language("en", ["eng", "english"], "English", "EN-US"),
    Language("de", ["deu", "ger", "german"], "German", "DE"),
    Language("fr", ["fra", "fre", "french"], "French", "FR"),
    Language("es", ["spa", "spanish"], "Spanish", "ES"),
    Language("it", ["ita", "italian"], "Italian", "IT"),
    Language("nl", ["nld", "dut", "dutch"], "Dutch", "NL"),
    Language("pl", ["pol", "polish"], "Polish", "PL"),
    Language("pt", ["por", "portuguese"], "Portuguese", "PT-PT"),
    Language("ru", ["rus", "russian"], "Russian", "RU"),
    Language("ja", ["jpn", "japanese"], "Japanese", "JA"),
    Language("zh", ["zho", "chi", "chinese"], "Chinese", "ZH"),
    Language("ko", ["kor", "korean"], "Korean", "KO"),
    Language("fi", ["fin", "finnish"], "Finnish", "FI"),
    Language("da", ["dan", "danish"], "Danish", "DA"),
    Language("no", ["nor", "nob", "nno", "norwegian"], "Norwegian", "NB")
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
