import re
import os
import srt
from typing import Dict, Any, List, Tuple, Optional

SWEDISH_COMMON_WORDS = {
    "och", "att", "det", "som", "på", "är", "av", "för", "med", "till", "den", 
    "har", "de", "inte", "om", "ett", "var", "jag", "ska", "här", "vi", "du", 
    "han", "hon", "vad", "kan", "från", "nu", "så", "hur", "när", "mig", "dig",
    "alla", "bara", "där", "blir", "blev", "vill", "kommer", "efter", "något", "mycket",
    "också", "skulle", "kunde", "måste", "henne", "honom", "deras", "våra", "inget"
}

ENGLISH_COMMON_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it", "for", "not", "on", "with", "he",
    "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about",
    "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than",
    "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two",
    "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any", "these", "give",
    "day", "most", "us", "did", "is", "are", "was", "were", "has", "had", "been", "why", "where"
}

import langdetect
from langdetect import DetectorFactory
from app.core.languages import get_language

# Seed for deterministic tests/results
DetectorFactory.seed = 0

def detect_language_heuristics(text: str, expected_language: Optional[str] = None) -> dict:
    """
    Robust language detection for all languages.
    Accepts optional expected_language (code, alias, or name) for target-aware disambiguation.
    Returns a dict with 'lang' (normalized ISO code) and 'confidence'.
    """
    if not text or len(text.strip()) < 10:
        return {"lang": "unknown", "confidence": 0.0}
    
    try:
        # Pre-cleaning for language detection:
        # Remove HTML/formatting tags, ASS tags, speaker labels (e.g. ">> Jimmy:"), bracketed cues ("[ CHEERS ]")
        cleaned = re.sub(r'<[^>]+>', ' ', text)
        cleaned = re.sub(r'\{[^}]+\}', ' ', cleaned)
        cleaned = re.sub(r'^\s*>>\s*[^:\n]+:\s*', ' ', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\[[^\]]+\]', ' ', cleaned)
        cleaned = re.sub(r'\([^)]+\)', ' ', cleaned)
        cleaned = re.sub(r'[♪♬♩♫#]+', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        if len(cleaned) < 10:
            return {"lang": "unknown", "confidence": 0.0}
        
        # Lowercase is mandatory for langdetect to prevent ALL-CAPS uppercase bias
        # (langdetect's character n-grams heavily bias uppercase text toward German 'de')
        langs = langdetect.detect_langs(cleaned.lower())
        if not langs:
            return {"lang": "unknown", "confidence": 0.0}
            
        best_match = langs[0]
        detected_code = best_match.lang.lower()
        confidence = best_match.prob

        expected_norm = None
        if expected_language:
            exp_lang_obj = get_language(expected_language)
            expected_norm = exp_lang_obj.code if exp_lang_obj else expected_language.lower().strip()[:2]
        
        words = set(re.findall(r"\b\w+\b", cleaned.lower()))
        swedish_word_matches = words & SWEDISH_COMMON_WORDS
        english_word_matches = words & ENGLISH_COMMON_WORDS

        # Swedish heuristic assistance:
        # Only assist Swedish if expected_norm is 'sv' OR expected_norm is None
        if expected_norm == "sv" or expected_norm is None:
            if len(swedish_word_matches) >= 2 and detected_code in {"no", "da", "unknown"}:
                detected_code = "sv"
                confidence = max(confidence, 0.95)
            elif len(swedish_word_matches) >= 1 and detected_code in {"no", "da", "unknown"} and expected_norm == "sv":
                detected_code = "sv"
                confidence = max(confidence, 0.90)

        # English heuristic assistance (prevents langdetect false positive collisions on short phrases, e.g., 'What did you do?' -> cy/so):
        if len(english_word_matches) >= 2 and len(swedish_word_matches) == 0:
            if detected_code not in {"en", "sv", "no", "da", "de", "fr", "es", "it", "bg", "cs", "ro", "hu", "tr", "el", "pl", "pt", "ru", "ja", "zh", "ko", "fi", "nl"}:
                detected_code = "en"
                confidence = max(confidence, 0.95)
            elif detected_code == "cy":
                detected_code = "en"
                confidence = max(confidence, 0.95)

        # Normalize via central language registry
        registry_lang = get_language(detected_code)
        normalized_code = registry_lang.code if registry_lang else detected_code
        
        return {"lang": normalized_code, "confidence": confidence}
    except Exception:
        return {"lang": "unknown", "confidence": 0.0}


def extract_representative_dialogue_samples(sub_blocks: List[srt.Subtitle], max_blocks: int = 90) -> Dict[str, Any]:
    """
    Extracts stratified representative dialogue text and indices from subtitle blocks across
    beginning, middle, and end of the subtitle file.
    Returns a dict with 'beginning', 'middle', 'end', 'all', and corresponding '*_indices'.
    """
    valid_items = [
        (idx, s.content.strip()) for idx, s in enumerate(sub_blocks)
        if hasattr(s, "content") and s.content and s.content.strip() and s.content.strip() != "<i></i>"
    ]
    n = len(valid_items)
    if n == 0:
        return {
            "beginning": [], "middle": [], "end": [], "all": [],
            "beginning_indices": [], "middle_indices": [], "end_indices": [], "all_indices": []
        }

    if n <= max_blocks:
        chunk_size = max(1, n // 3)
        beg_items = valid_items[:chunk_size]
        mid_items = valid_items[chunk_size:chunk_size * 2]
        end_items = valid_items[chunk_size * 2:]
        return {
            "beginning": [t for _, t in beg_items],
            "middle": [t for _, t in mid_items],
            "end": [t for _, t in end_items],
            "all": [t for _, t in valid_items],
            "beginning_indices": [idx for idx, _ in beg_items],
            "middle_indices": [idx for idx, _ in mid_items],
            "end_indices": [idx for idx, _ in end_items],
            "all_indices": [idx for idx, _ in valid_items]
        }

    per_stratum = max_blocks // 3  # 30

    # 1. Beginning
    beg_items = valid_items[:per_stratum]

    # 2. Middle
    mid_start = max(0, (n // 2) - (per_stratum // 2))
    mid_items = valid_items[mid_start:mid_start + per_stratum]

    # 3. End
    end_items = valid_items[-per_stratum:]

    all_items = beg_items + mid_items + end_items

    return {
        "beginning": [t for _, t in beg_items],
        "middle": [t for _, t in mid_items],
        "end": [t for _, t in end_items],
        "all": [t for _, t in all_items],
        "beginning_indices": [idx for idx, _ in beg_items],
        "middle_indices": [idx for idx, _ in mid_items],
        "end_indices": [idx for idx, _ in end_items],
        "all_indices": [idx for idx, _ in all_items]
    }

FOREIGN_LANGUAGE_INDICATORS = {
    "de": {"der", "die", "das", "und", "ist", "in", "den", "von", "zu", "mit", "sich", "des", "auf", "für", "nicht", "eine", "einer", "einem", "einen", "ja", "nein", "herr", "frau", "bitte", "danke", "guten", "tag", "soldaten", "angriff", "befehl", "wir", "sie", "ihr", "mein", "dein"},
    "fr": {"le", "la", "les", "un", "une", "des", "et", "est", "que", "qui", "dans", "en", "pour", "avec", "sur", "pas", "plus", "oui", "non", "merci", "bonjour", "monsieur", "madame", "nous", "vous", "ils", "elles", "mon", "ton", "son"},
    "es": {"el", "la", "los", "las", "un", "una", "unos", "unas", "y", "en", "de", "que", "es", "por", "con", "para", "no", "si", "gracias", "hola", "señor", "señora", "amigo", "amigos", "nosotros", "ellos", "mi", "tu", "su"},
    "it": {"il", "la", "lo", "i", "gli", "le", "un", "una", "uno", "e", "ed", "di", "in", "che", "per", "con", "su", "non", "si", "grazie", "ciao", "signore", "signora", "amico", "amici", "noi", "loro", "mio", "tuo", "suo"}
}

def is_verified_foreign_text(text: str, detected_lang: str) -> bool:
    """
    Verifies if text contains authentic indicators of a non-English foreign language
    to avoid false positives on short English phrases (e.g. 'Stubborn dialogue').
    """
    if not text or not detected_lang:
        return False
    if detected_lang in {"ru", "uk", "bg", "ja", "zh", "ko", "ar", "he", "el", "hi", "th"}:
        return True
    
    indicators = FOREIGN_LANGUAGE_INDICATORS.get(detected_lang)
    if indicators:
        words = set(re.findall(r"\b\w+\b", text.lower()))
        return bool(words & indicators)
    
    words = set(re.findall(r"\b\w+\b", text.lower()))
    if len(text.strip()) >= 30 and not (words & ENGLISH_COMMON_WORDS):
        return True
    return False

def classify_cue_language_mismatch(
    target_text: str,
    source_text: str,
    target_lang_code: str = "sv",
    source_lang_code: str = "en"
) -> Dict[str, Any]:
    """
    Classifies cue-level language status with source-awareness.
    Returns: {
        "status": "SAFE_INVARIANT" | "CORRECT_TARGET" | "LEGIT_FOREIGN_PRESERVED" | "WRONG_TARGET_LANGUAGE" | "UNCERTAIN",
        "target_lang": str,
        "source_lang": str,
        "details": str
    }
    """
    t_clean = re.sub(r'<[^>]+>', ' ', target_text or '')
    t_clean = re.sub(r'\{[^}]+\}', ' ', t_clean)
    t_clean = re.sub(r'^\s*>>\s*[^:\n]+:\s*', ' ', t_clean, flags=re.MULTILINE)
    t_clean = re.sub(r'\[[^\]]+\]', ' ', t_clean)
    t_clean = re.sub(r'\([^)]+\)', ' ', t_clean)
    t_clean = re.sub(r'[♪♬♩♫#]+', ' ', t_clean).strip()
    
    # 1. Safe invariant / empty check
    if not t_clean or t_clean == "<i></i>" or not any(c.isalpha() for c in t_clean):
        return {"status": "SAFE_INVARIANT", "target_lang": "unknown", "source_lang": "unknown", "details": "Non-verbal/empty/symbols"}

    target_norm = target_lang_code[:2].lower()
    source_norm = source_lang_code[:2].lower()

    t_info = detect_language_heuristics(target_text, expected_language=target_norm)
    t_lang = t_info["lang"]
    t_conf = t_info["confidence"]

    s_info = detect_language_heuristics(source_text, expected_language=source_norm) if source_text else {"lang": "unknown", "confidence": 0.0}
    s_lang = s_info["lang"]
    s_conf = s_info["confidence"]

    # 2. Correct target language or uncertain/short
    if t_lang == target_norm or t_lang == "unknown" or t_conf < 0.75:
        return {"status": "CORRECT_TARGET" if t_lang == target_norm else "UNCERTAIN", "target_lang": t_lang, "source_lang": s_lang, "details": "Target language matched or uncertain"}

    # 3. Target is detected as a foreign language (not target_norm, e.g. 'de', 'fr', 'es', 'it')
    # Check if source also contains this foreign dialogue:
    if s_lang == t_lang and s_lang not in {source_norm, "unknown"} and s_conf >= 0.75:
        if is_verified_foreign_text(source_text, s_lang):
            return {"status": "LEGIT_FOREIGN_PRESERVED", "target_lang": t_lang, "source_lang": s_lang, "details": f"Preserved foreign dialogue ({t_lang})"}

    norm_t = re.sub(r'[^\w]', '', (target_text or '').lower())
    norm_s = re.sub(r'[^\w]', '', (source_text or '').lower())
    if norm_t and norm_t == norm_s and s_lang not in {source_norm, "unknown"} and is_verified_foreign_text(source_text, s_lang):
        return {"status": "LEGIT_FOREIGN_PRESERVED", "target_lang": t_lang, "source_lang": s_lang, "details": "Identical non-English source dialogue"}

    # Source is English dialogue but target became a foreign non-target language:
    if s_lang == source_norm or (norm_t != norm_s and t_conf >= 0.8):
        return {"status": "WRONG_TARGET_LANGUAGE", "target_lang": t_lang, "source_lang": s_lang, "details": f"Target translated to {t_lang} instead of {target_norm}"}

    return {"status": "UNCERTAIN", "target_lang": t_lang, "source_lang": s_lang, "details": "Uncertain mismatch"}

def check_language_representative(
    sub_blocks: List[srt.Subtitle],
    target_lang_code: str,
    source_sub_blocks: Optional[List[srt.Subtitle]] = None
) -> Dict[str, Any]:
    """
    Evaluates language across stratified samples of the file.
    Source-aware: Distinguishes between legitimate foreign dialogue preserved from source
    and erroneous AI translations into the wrong language.
    """
    samples = extract_representative_dialogue_samples(sub_blocks)
    target_norm = target_lang_code[:2].lower()

    if not samples["all"]:
        return {
            "confident_wrong_language": False,
            "detected_lang": "unknown",
            "confidence": 0.0,
            "section": "overall",
            "wrong_language_cue_ids": [],
            "legit_foreign_cue_ids": [],
            "details": "No dialogue text found"
        }

    def evaluate_mismatch_cues(cue_indices: List[int], detected_lang: str) -> Tuple[bool, List[int], List[int]]:
        if not source_sub_blocks:
            return True, [], []
        
        wrong_ids = []
        legit_ids = []
        for idx in cue_indices:
            if idx >= len(sub_blocks):
                continue
            t_content = sub_blocks[idx].content
            s_content = source_sub_blocks[idx].content if idx < len(source_sub_blocks) else ""
            classification = classify_cue_language_mismatch(t_content, s_content, target_lang_code=target_lang_code)
            status = classification["status"]
            if status == "WRONG_TARGET_LANGUAGE":
                wrong_ids.append(idx)
            elif status == "LEGIT_FOREIGN_PRESERVED":
                legit_ids.append(idx)

        if legit_ids and not wrong_ids:
            return False, [], legit_ids

        return True, (wrong_ids if wrong_ids else cue_indices), legit_ids

    accumulated_legit_ids = []

    # 1. Stratified section checks (beginning, middle, end)
    for sec in ["beginning", "middle", "end"]:
        sec_texts = samples[sec]
        sec_text = " ".join(sec_texts)
        if len(sec_text) >= 50 and len(sec_texts) >= 5:
            lang_info = detect_language_heuristics(sec_text, expected_language=target_norm)
            det = lang_info["lang"]
            conf = lang_info["confidence"]

            if det != "unknown" and det != target_norm and conf > 0.85:
                sec_indices = samples.get(f"{sec}_indices", [])
                is_wrong, wrong_ids, legit_ids = evaluate_mismatch_cues(sec_indices, det)
                if is_wrong:
                    return {
                        "confident_wrong_language": True,
                        "detected_lang": det,
                        "confidence": conf,
                        "section": sec,
                        "wrong_language_cue_ids": wrong_ids,
                        "legit_foreign_cue_ids": legit_ids,
                        "details": f"{sec.capitalize()} section detected as {det} ({conf*100:.0f}% conf)"
                    }
                else:
                    accumulated_legit_ids.extend(legit_ids)

    # 2. Overall check
    full_sample_text = " ".join(samples["all"])
    if len(full_sample_text) >= 20:
        lang_info = detect_language_heuristics(full_sample_text, expected_language=target_norm)
        det = lang_info["lang"]
        conf = lang_info["confidence"]

        if det != "unknown" and det != target_norm and conf > 0.8:
            all_indices = samples.get("all_indices", [])
            is_wrong, wrong_ids, legit_ids = evaluate_mismatch_cues(all_indices, det)
            if is_wrong:
                return {
                    "confident_wrong_language": True,
                    "detected_lang": det,
                    "confidence": conf,
                    "section": "overall",
                    "wrong_language_cue_ids": wrong_ids,
                    "legit_foreign_cue_ids": legit_ids,
                    "details": f"Overall sample detected as {det} ({conf*100:.0f}% conf)"
                }
            else:
                accumulated_legit_ids.extend(legit_ids)

    # Return overall detected language info
    lang_info = detect_language_heuristics(" ".join(samples["all"]), expected_language=target_norm)
    det = lang_info["lang"]
    conf = lang_info["confidence"]

    return {
        "confident_wrong_language": False,
        "detected_lang": det,
        "confidence": conf,
        "section": "overall",
        "wrong_language_cue_ids": [],
        "legit_foreign_cue_ids": list(set(accumulated_legit_ids)),
        "details": "Language check passed"
    }

def parse_srt_safe(srt_text: str) -> List[srt.Subtitle]:
    try:
        return list(srt.parse(srt_text))
    except Exception:
        blocks = []
        raw_blocks = re.split(r'\n\s*\n', srt_text.strip())
        for b in raw_blocks:
            lines = [l.strip() for l in b.split('\n') if l.strip()]
            if len(lines) >= 3 and '-->' in lines[1]:
                try:
                    time_parts = lines[1].split('-->')
                    start = srt.srt_timestamp_to_timedelta(time_parts[0].strip())
                    end = srt.srt_timestamp_to_timedelta(time_parts[1].strip())
                    content = "\n".join(lines[2:])
                    blocks.append(srt.Subtitle(index=len(blocks)+1, start=start, end=end, content=content))
                except Exception:
                    continue
        return blocks

def verify_sync(original_subs: List[srt.Subtitle], translated_subs: List[srt.Subtitle]) -> Dict[str, Any]:
    if not original_subs or not translated_subs:
        return {
            "valid": False,
            "error": "Empty subtitle list",
            "start_diff_ms": -1,
            "end_diff_ms": -1
        }

    max_start_diff = 0
    max_end_diff = 0
    
    min_len = min(len(original_subs), len(translated_subs))
    for i in range(min_len):
        orig_start_ms = int(original_subs[i].start.total_seconds() * 1000)
        trans_start_ms = int(translated_subs[i].start.total_seconds() * 1000)
        start_diff = abs(orig_start_ms - trans_start_ms)
        if start_diff > max_start_diff:
            max_start_diff = start_diff
            
        orig_end_ms = int(original_subs[i].end.total_seconds() * 1000)
        trans_end_ms = int(translated_subs[i].end.total_seconds() * 1000)
        end_diff = abs(orig_end_ms - trans_end_ms)
        if end_diff > max_end_diff:
            max_end_diff = end_diff

    len_orig = len(original_subs)
    len_trans = len(translated_subs)
    count_diff = abs(len_orig - len_trans)

    is_valid = (max_start_diff == 0) and (max_end_diff == 0) and (count_diff == 0)

    return {
        "valid": is_valid,
        "original_count": len_orig,
        "translated_count": len_trans,
        "count_diff": count_diff,
        "start_diff_ms": max_start_diff,
        "end_diff_ms": max_end_diff
    }

def check_dropped_lines(original_subs: List[srt.Subtitle], translated_subs: List[srt.Subtitle]) -> Tuple[int, List[Dict[str, Any]]]:
    dropped = []
    min_len = min(len(original_subs), len(translated_subs))

    for i in range(min_len):
        orig = original_subs[i].content.strip()
        trans = translated_subs[i].content.strip()

        is_orig_real = orig and orig != "<i></i>"
        is_trans_empty = not trans or trans == "<i></i>"

        if is_orig_real and is_trans_empty:
            dropped.append({
                "index": i + 1,
                "timestamp": str(original_subs[i].start),
                "original": orig
            })

    if len(original_subs) > len(translated_subs):
        for i in range(len(translated_subs), len(original_subs)):
            orig = original_subs[i].content.strip()
            if orig and orig != "<i></i>":
                dropped.append({
                    "index": i + 1,
                    "timestamp": str(original_subs[i].start),
                    "original": orig
                })

    return len(dropped), dropped

def evaluate_subtitle_health(
    sub_file_path: str,
    target_lang_code: str = "sv",
    reference_sub_blocks: Optional[List[srt.Subtitle]] = None,
    video_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Kvalitetskontroll:
    1. Finns filen och är den > 200 bytes?
    2. Kan den parsas som giltig SRT?
    3. Matchar språket det förväntade målspråket (eller är den engelsk/fel)?
    4. Är för många rader tomma?
    5. Om referens finns: Har den sync drift (>500ms) eller saknas repliker?
    """
    if not os.path.exists(sub_file_path) or os.path.getsize(sub_file_path) < 200:
        return {
            "status": "RED",
            "health_score": 0,
            "reason": "File is empty or corrupted (< 200 bytes)",
            "lines": 0,
            "detected_language": "none"
        }

    try:
        with open(sub_file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        return {"status": "RED", "health_score": 0, "reason": f"Unreadable file: {e}", "lines": 0}

    sub_blocks = parse_srt_safe(content)
    if not sub_blocks or len(sub_blocks) < 5:
        return {
            "status": "RED",
            "health_score": 0,
            "reason": f"Corrupted structure: only {len(sub_blocks)} subtitle lines parsed",
            "lines": len(sub_blocks),
            "detected_language": "none"
        }

    # 1. Språkdetektering
    lang_check = check_language_representative(sub_blocks, target_lang_code, source_sub_blocks=reference_sub_blocks)
    detected_lang = lang_check["detected_lang"]
    confidence = lang_check["confidence"]

    if lang_check["confident_wrong_language"]:
        return {
            "status": "RED",
            "health_score": 10,
            "reason": f"Wrong language detected in {lang_check['section']}: Found {detected_lang} (conf: {confidence:.2f}), expected {target_lang_code}",
            "lines": len(sub_blocks),
            "detected_language": detected_lang
        }

    target_norm = target_lang_code[:2].lower()
    if detected_lang != "unknown" and detected_lang != target_norm and confidence < 0.8:
        return {
            "status": "YELLOW",
            "health_score": 50,
            "reason": f"Low confidence language mismatch: Found {detected_lang} (conf: {confidence:.2f}), expected {target_lang_code}",
            "lines": len(sub_blocks),
            "detected_language": detected_lang
        }

    # 2. Tomma rader
    empty_lines = sum(1 for s in sub_blocks if not s.content.strip() or s.content.strip() == "...")
    if empty_lines / len(sub_blocks) > 0.25:
        return {
            "status": "RED",
            "health_score": 25,
            "reason": f"Too many blank lines: {empty_lines}/{len(sub_blocks)} are empty",
            "lines": len(sub_blocks),
            "detected_language": detected_lang
        }

    # 3. Referenssynk
    if reference_sub_blocks:
        sync_info = verify_sync(reference_sub_blocks, sub_blocks)
        dropped_count, _ = check_dropped_lines(reference_sub_blocks, sub_blocks)
        total_ref = len(reference_sub_blocks)
        
        max_time_diff = max(sync_info.get("start_diff_ms", 0), sync_info.get("end_diff_ms", 0))
        dropped_ratio = dropped_count / total_ref if total_ref > 0 else 0

        if max_time_diff > 500 or dropped_ratio > 0.10:
            return {
                "status": "RED",
                "health_score": 20,
                "reason": f"Severe sync drift ({max_time_diff}ms) or {dropped_count} dropped lines",
                "lines": len(sub_blocks),
                "sync_diff_ms": max_time_diff,
                "dropped_lines": dropped_count,
                "detected_language": detected_lang
            }
        elif max_time_diff > 50 or dropped_ratio > 0.02:
            return {
                "status": "YELLOW",
                "health_score": 75,
                "reason": f"Minor drift/missing: {max_time_diff}ms drift, {dropped_count} dropped lines",
                "lines": len(sub_blocks),
                "sync_diff_ms": max_time_diff,
                "dropped_lines": dropped_count,
                "detected_language": detected_lang
            }

    if detected_lang == "unknown":
        return {
            "status": "YELLOW",
            "health_score": 75,
            "reason": f"Healthy structure ({len(sub_blocks)} lines), language detection uncertain (unknown)",
            "lines": len(sub_blocks),
            "detected_language": "unknown"
        }

    return {
        "status": "GREEN",
        "health_score": 100,
        "reason": f"Verified healthy {detected_lang.upper()} ({len(sub_blocks)} lines)",
        "lines": len(sub_blocks),
        "detected_language": detected_lang
    }
