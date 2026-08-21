import re
import os
import srt
from typing import Dict, Any, List, Tuple, Optional

SWEDISH_COMMON_WORDS = {
    "och", "att", "det", "som", "på", "är", "en", "av", "för", "med", "till", "den", 
    "har", "de", "inte", "om", "ett", "men", "var", "jag", "ska", "här", "vi", "du", 
    "han", "hon", "vad", "kan", "man", "från", "nu", "så", "hur", "när", "mig", "dig",
    "alla", "bara", "där", "blir", "blev", "vill", "kommer", "efter", "något", "mycket"
}

import langdetect
from langdetect import DetectorFactory
from app.core.languages import get_language

# Seed for deterministic tests/results
DetectorFactory.seed = 0

def detect_language_heuristics(text: str) -> dict:
    """
    Robust language detection for all languages.
    Returns a dict with 'lang' (normalized ISO code) and 'confidence'.
    """
    if not text or len(text.strip()) < 10:
        return {"lang": "unknown", "confidence": 0.0}
    
    try:
        langs = langdetect.detect_langs(text)
        if not langs:
            return {"lang": "unknown", "confidence": 0.0}
            
        best_match = langs[0]
        detected_code = best_match.lang.lower()
        confidence = best_match.prob
        
        # Normalize via central language registry
        registry_lang = get_language(detected_code)
        normalized_code = registry_lang.code if registry_lang else detected_code
        
        return {"lang": normalized_code, "confidence": confidence}
    except Exception:
        return {"lang": "unknown", "confidence": 0.0}

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
    full_sample_text = " ".join([s.content for s in sub_blocks[:80]])
    lang_info = detect_language_heuristics(full_sample_text)
    detected_lang = lang_info["lang"]
    confidence = lang_info["confidence"]

    if detected_lang != "unknown" and detected_lang != target_lang_code[:2].lower():
        if confidence < 0.8:
            return {
                "status": "YELLOW",
                "health_score": 50,
                "reason": f"Low confidence language mismatch: Found {detected_lang} (conf: {confidence:.2f}), expected {target_lang_code}",
                "lines": len(sub_blocks),
                "detected_language": detected_lang
            }
        else:
            return {
                "status": "RED",
                "health_score": 10,
                "reason": f"Wrong language detected: Found {detected_lang} (conf: {confidence:.2f}), expected {target_lang_code}",
                "lines": len(sub_blocks),
                "detected_language": detected_lang
            }
    elif detected_lang == "unknown":
        return {
            "status": "RED",
            "health_score": 0,
            "reason": "Language detection failed (unknown)",
            "lines": len(sub_blocks),
            "detected_language": "unknown"
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

    return {
        "status": "GREEN",
        "health_score": 100,
        "reason": f"Verified healthy {detected_lang.upper()} ({len(sub_blocks)} lines)",
        "lines": len(sub_blocks),
        "detected_language": detected_lang
    }
