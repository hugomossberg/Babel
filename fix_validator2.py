import langdetect
from langdetect import DetectorFactory
from app.core.languages import get_language

DetectorFactory.seed = 0

def detect_language_heuristics(text: str) -> dict:
    if not text or not text.strip():
        return {"lang": "unknown", "confidence": 0.0}
    
    try:
        langs = langdetect.detect_langs(text)
        if not langs:
            return {"lang": "unknown", "confidence": 0.0}
            
        best_match = langs[0]
        detected_code = best_match.lang.lower()
        confidence = best_match.prob
        
        # Normalize via registry
        registry_lang = get_language(detected_code)
        normalized_code = registry_lang.code if registry_lang else detected_code
        
        return {"lang": normalized_code, "confidence": confidence}
    except Exception:
        return {"lang": "unknown", "confidence": 0.0}

print(detect_language_heuristics("the and a have for"))
print(detect_language_heuristics("och det att i en"))
print(detect_language_heuristics("der die und in den"))
print(detect_language_heuristics("这是一些中文测试"))
