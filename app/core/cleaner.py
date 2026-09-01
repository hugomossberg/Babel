import re
import unicodedata
import asyncio
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any
import srt

# Placeholder that preserves the block structure for parsers and locks timestamps
EMPTY_PLACEHOLDER = "<i></i>"


class SanitizerResult(tuple):
    subs: List[srt.Subtitle]
    provenance_map: Dict[int, Any]
    cleaned_count: int
    telemetry: Dict[str, Any]

    def __new__(
        cls,
        subs: List[srt.Subtitle],
        provenance_map: Dict[int, Any],
        cleaned_count: int,
        telemetry: Optional[Dict[str, Any]] = None,
    ):
        inst = super().__new__(cls, (subs, provenance_map, cleaned_count))
        inst.subs = subs
        inst.provenance_map = provenance_map
        inst.cleaned_count = cleaned_count
        inst.telemetry = telemetry or {}
        return inst


# Regex for music notes and signs: ♪, ♫, ♬, ♩
MUSIC_NOTES_REGEX = re.compile(r'[♪♫♬♩]+')

# Structural speaker prefix pattern: e.g. "MAN:", "WOMAN:", "JOHN:", "OFFICER 1:", "HOMME:"
# Matches uppercase/titlecase tokens at line start followed by colon, avoiding URLs and timestamps
SPEAKER_PREFIX_REGEX = re.compile(r'^([^\n:]{1,35}):(?!\/|\d|\w+:\/\/)\s*(.*)$')

# Structural brackets / parentheses tokenizer (including Unicode full-width brackets （ ） ［ ］)
STRUCTURAL_TOKEN_REGEX = re.compile(r'(\[[^\]]*\]|［[^］]*］|\([^\)]*\)|（[^）]*）|[♪♫♬♩]+)')


class StructuralType(str, Enum):
    BRACKETED = "bracketed"          # [...] or ［...］
    PARENTHESIZED = "parenthesized"  # (...) or （...）
    MUSIC_NOTES = "music_notes"      # ♪, ♫, ♬, ♩
    SPEAKER_PREFIX = "speaker_prefix"# LABEL:
    PLAIN_TEXT = "plain_text"        # regular dialogue text


class SegmentClassification(str, Enum):
    DIALOGUE = "DIALOGUE"
    NON_DIALOGUE = "NON_DIALOGUE"
    UNCERTAIN = "UNCERTAIN"


class ClassificationProvenance(str, Enum):
    STRUCTURAL_SYNTAX = "structural_syntax"         # Pure syntax: music notes, empty wrapper, numbers
    DETERMINISTIC_HEURISTIC = "deterministic_heuristic" # Punctuation-based sentence or file-level repeated label
    SEMANTIC_VERIFIED = "semantic_verified"         # Bounded bulk semantic classifier AI call
    FAIL_SAFE_PRESERVED = "fail_safe_preserved"     # Provider failure / UNCERTAIN / timeout -> keep dialogue


@dataclass
class SegmentAnalysis:
    raw: str
    inner: str
    structural_type: StructuralType
    classification: SegmentClassification = SegmentClassification.DIALOGUE
    provenance: ClassificationProvenance = ClassificationProvenance.STRUCTURAL_SYNTAX
    reason: str = ""
    line_index: int = 0


@dataclass
class CueProvenance:
    index: int
    raw_content: str
    cleaned_content: str
    segments: List[SegmentAnalysis] = field(default_factory=list)
    has_bracketed_dialogue: bool = False
    has_parenthesized_dialogue: bool = False
    bracketed_dialogue_inners: List[str] = field(default_factory=list)
    parenthesized_dialogue_inners: List[str] = field(default_factory=list)
    removed_sdh_segments: List[str] = field(default_factory=list)
    is_sdh_only: bool = False


def extract_file_speaker_labels(subs: List[srt.Subtitle]) -> Set[str]:
    """
    Language-agnostic file-level structural indexer:
    Scans all cues in an SRT to find repeated speaker labels (e.g. 'JOHN:', 'HOMME:', 'MARIA:').
    A label appearing across 2 or more cues is strong structural evidence of a speaker prefix.
    """
    label_counts: Dict[str, int] = {}
    for sub in subs:
        if not sub.content:
            continue
        for line in sub.content.split('\n'):
            line_str = line.strip()
            m = SPEAKER_PREFIX_REGEX.match(line_str)
            if m:
                label = m.group(1).strip()
                # Check if label looks like a speaker name (mostly uppercase or titlecase, no punctuation except digits/spaces/dashes)
                if len(label) <= 30 and not any(c in label for c in ".,?!;"):
                    norm_label = label.upper()
                    label_counts[norm_label] = label_counts.get(norm_label, 0) + 1

    return {label for label, count in label_counts.items() if count >= 2}


def parse_structural_segments(text: str) -> List[SegmentAnalysis]:
    """
    Language-agnostic structural segmenter:
    Splits a subtitle block into structural candidate segments without making semantic assumptions.
    """
    if not text:
        return []

    segments: List[SegmentAnalysis] = []
    lines = text.split('\n')

    for line_idx, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue

        # Fast check: pure music symbols only (e.g. ♪ ♪, ♫, ♬, ♩ without alphanumeric text)
        if MUSIC_NOTES_REGEX.search(line_clean):
            cleaned_m = MUSIC_NOTES_REGEX.sub('', line_clean).strip()
            if not any(c.isalnum() for c in cleaned_m):
                segments.append(SegmentAnalysis(
                    raw=line_clean,
                    inner="",
                    structural_type=StructuralType.MUSIC_NOTES,
                    classification=SegmentClassification.NON_DIALOGUE,
                    provenance=ClassificationProvenance.STRUCTURAL_SYNTAX,
                    reason="pure_music_symbols",
                    line_index=line_idx
                ))
                continue

        # Check for structural speaker prefix at line start
        prefix_match = SPEAKER_PREFIX_REGEX.match(line_clean)
        remaining_line = line_clean
        if prefix_match:
            prefix_label = prefix_match.group(1).strip()
            # Only treat as prefix if it looks like an identifier/name
            if len(prefix_label) <= 30 and not any(c in prefix_label for c in ".,?!;"):
                segments.append(SegmentAnalysis(
                    raw=f"{prefix_label}:",
                    inner=prefix_label,
                    structural_type=StructuralType.SPEAKER_PREFIX,
                    classification=SegmentClassification.UNCERTAIN,
                    provenance=ClassificationProvenance.STRUCTURAL_SYNTAX,
                    reason="line_initial_speaker_prefix",
                    line_index=line_idx
                ))
                remaining_line = prefix_match.group(2).strip()

        if not remaining_line:
            continue

        # Tokenize bracketed, parenthesized, music notes, and plain text
        raw_tokens = [t for t in STRUCTURAL_TOKEN_REGEX.split(remaining_line) if t and t.strip()]
        line_segs: List[SegmentAnalysis] = []

        for token in raw_tokens:
            token_stripped = token.strip()
            if (token_stripped.startswith('[') and token_stripped.endswith(']')) or \
               (token_stripped.startswith('［') and token_stripped.endswith('］')):
                inner = token_stripped[1:-1].strip()
                line_segs.append(SegmentAnalysis(
                    raw=token_stripped,
                    inner=inner,
                    structural_type=StructuralType.BRACKETED,
                    classification=SegmentClassification.UNCERTAIN,
                    provenance=ClassificationProvenance.STRUCTURAL_SYNTAX,
                    reason="bracketed_token",
                    line_index=line_idx
                ))
            elif (token_stripped.startswith('(') and token_stripped.endswith(')')) or \
                 (token_stripped.startswith('（') and token_stripped.endswith('）')):
                inner = token_stripped[1:-1].strip()
                line_segs.append(SegmentAnalysis(
                    raw=token_stripped,
                    inner=inner,
                    structural_type=StructuralType.PARENTHESIZED,
                    classification=SegmentClassification.UNCERTAIN,
                    provenance=ClassificationProvenance.STRUCTURAL_SYNTAX,
                    reason="parenthesized_token",
                    line_index=line_idx
                ))
            elif MUSIC_NOTES_REGEX.search(token_stripped):
                cleaned_note = MUSIC_NOTES_REGEX.sub('', token_stripped).strip()
                if not any(c.isalnum() for c in cleaned_note):
                    line_segs.append(SegmentAnalysis(
                        raw=token_stripped,
                        inner="",
                        structural_type=StructuralType.MUSIC_NOTES,
                        classification=SegmentClassification.NON_DIALOGUE,
                        provenance=ClassificationProvenance.STRUCTURAL_SYNTAX,
                        reason="pure_music_symbols",
                        line_index=line_idx
                    ))
                else:
                    # Token contains text attached to music symbols (e.g. ♪ singing softly ♪ or ♪ never gonna give you up ♪)
                    # Structure only creates an UNCERTAIN candidate for classification
                    line_segs.append(SegmentAnalysis(
                        raw=token_stripped,
                        inner=cleaned_note,
                        structural_type=StructuralType.MUSIC_NOTES,
                        classification=SegmentClassification.UNCERTAIN,
                        provenance=ClassificationProvenance.STRUCTURAL_SYNTAX,
                        reason="music_note_with_text",
                        line_index=line_idx
                    ))
            else:
                line_segs.append(SegmentAnalysis(
                    raw=token_stripped,
                    inner=token_stripped,
                    structural_type=StructuralType.PLAIN_TEXT,
                    classification=SegmentClassification.DIALOGUE,
                    provenance=ClassificationProvenance.STRUCTURAL_SYNTAX,
                    reason="plain_dialogue_text",
                    line_index=line_idx
                ))

        # Check for inline parentheticals surrounded by plain text on the same line
        for s_idx, seg in enumerate(line_segs):
            if seg.structural_type == StructuralType.PARENTHESIZED:
                if 0 < s_idx < len(line_segs) - 1:
                    prev_is_plain = (line_segs[s_idx - 1].structural_type == StructuralType.PLAIN_TEXT)
                    next_is_plain = (line_segs[s_idx + 1].structural_type == StructuralType.PLAIN_TEXT)
                    if prev_is_plain and next_is_plain:
                        seg.classification = SegmentClassification.DIALOGUE
                        seg.provenance = ClassificationProvenance.DETERMINISTIC_HEURISTIC
                        seg.reason = "inline_sentence_parenthetical"

        segments.extend(line_segs)

    return segments


def classify_structural_segment_deterministic(
    segment: SegmentAnalysis,
    file_speaker_labels: Optional[Set[str]] = None
) -> Optional[SegmentClassification]:
    """
    Deterministic language-agnostic pre-classifier based purely on typography & syntax.
    Returns:
    - SegmentClassification.NON_DIALOGUE if proven non-dialogue syntactically.
    - SegmentClassification.DIALOGUE if proven dialogue syntactically (e.g. dialogue quotes).
    - None if ambiguous and requires semantic classification / fail-safe handling.
    """
    if segment.structural_type == StructuralType.MUSIC_NOTES:
        inner = segment.inner.strip()
        if not inner or not any(c.isalnum() for c in inner):
            segment.provenance = ClassificationProvenance.STRUCTURAL_SYNTAX
            return SegmentClassification.NON_DIALOGUE
        return None

    if segment.structural_type == StructuralType.PLAIN_TEXT:
        segment.provenance = ClassificationProvenance.STRUCTURAL_SYNTAX
        return SegmentClassification.DIALOGUE

    if segment.structural_type == StructuralType.SPEAKER_PREFIX:
        norm = segment.inner.strip().upper()
        if file_speaker_labels and norm in file_speaker_labels:
            segment.provenance = ClassificationProvenance.DETERMINISTIC_HEURISTIC
            segment.reason = "repeated_file_speaker_label"
            return SegmentClassification.NON_DIALOGUE
        # If uppercase single-word label like 'MAN', 'NARRATOR', 'HOMME', treat structurally as speaker prefix
        if norm.isupper() and len(norm.split()) <= 2 and all(c.isalnum() or c in " _-'" for c in norm):
            segment.provenance = ClassificationProvenance.DETERMINISTIC_HEURISTIC
            segment.reason = "uppercase_speaker_label"
            return SegmentClassification.NON_DIALOGUE
        return None

    # For BRACKETED and PARENTHESIZED segments
    inner = segment.inner.strip()
    if not inner:
        segment.provenance = ClassificationProvenance.STRUCTURAL_SYNTAX
        return SegmentClassification.NON_DIALOGUE

    # Pure music notes inside brackets, e.g. [♪♪] or (♪) without text
    if MUSIC_NOTES_REGEX.search(inner) and not any(c.isalnum() for c in inner):
        segment.provenance = ClassificationProvenance.STRUCTURAL_SYNTAX
        return SegmentClassification.NON_DIALOGUE

    # Pure numbers, timecodes, or standard subtitling metadata flags e.g. [12:30], (1), [CC]
    if inner.isdigit() or re.match(r'^\d+:\d+(?::\d+)?$', inner) or inner.upper() in {"CC", "SDH", "SUBTITLES", "AUDIO"}:
        segment.provenance = ClassificationProvenance.STRUCTURAL_SYNTAX
        return SegmentClassification.NON_DIALOGUE

    # Dialogue quotes: e.g. ["Yes"], [“Wait”], [«Oui»] -> DIALOGUE
    has_dialogue_quotes = (inner.startswith(('"', '“', '«')) and inner.endswith(('"', '”', '»')))
    has_terminal_punct = inner.endswith(('.', '?', '!', '…', '."', '?"', '!"'))
    word_count = len(inner.split())

    if segment.structural_type == StructuralType.BRACKETED:
        # Standard subtitling standard: square brackets [...] are sound/audio metadata unless containing dialogue quotes or sentence punctuation
        if has_dialogue_quotes or (has_terminal_punct and word_count >= 2):
            segment.provenance = ClassificationProvenance.DETERMINISTIC_HEURISTIC
            segment.reason = "bracketed_dialogue_syntax"
            return SegmentClassification.DIALOGUE
        else:
            segment.provenance = ClassificationProvenance.STRUCTURAL_SYNTAX
            segment.reason = "bracketed_sdh_syntax"
            return SegmentClassification.NON_DIALOGUE

    if segment.structural_type == StructuralType.PARENTHESIZED:
        # Parenthesized dialogue check:
        # Full sentences with terminal punctuation or quotes -> DIALOGUE
        if has_dialogue_quotes or has_terminal_punct:
            segment.provenance = ClassificationProvenance.DETERMINISTIC_HEURISTIC
            segment.reason = "parenthesized_sentence_dialogue"
            return SegmentClassification.DIALOGUE

        # Ambiguous parentheticals (such as (yes), (no), (help), (come here), (sighs), (laughs), (applause)):
        # MUST NOT be classified deterministically by word count or casing.
        # They will be resolved by semantic classification or fail-safe preserved as dialogue.
        return None


def analyze_subtitle_cue(
    text: str,
    file_speaker_labels: Optional[Set[str]] = None,
    index: int = 1
) -> CueProvenance:
    """
    Analyzes a subtitle cue text into segments and performs deterministic classification.
    Ambiguous segments remain UNCERTAIN.
    """
    segments = parse_structural_segments(text)
    for seg in segments:
        if seg.classification == SegmentClassification.UNCERTAIN:
            det = classify_structural_segment_deterministic(seg, file_speaker_labels)
            if det is not None:
                seg.classification = det

    # Build cleaned content from DIALOGUE and UNCERTAIN segments
    line_parts: Dict[int, List[str]] = {}
    removed_sdh = []
    has_bracketed_dialogue = False
    has_parenthesized_dialogue = False
    bracketed_inners = []
    parenthesized_inners = []

    for seg in segments:
        if seg.classification == SegmentClassification.NON_DIALOGUE:
            removed_sdh.append(seg.raw)
        else:
            if seg.structural_type == StructuralType.BRACKETED:
                has_bracketed_dialogue = True
                bracketed_inners.append(seg.inner.strip())
            elif seg.structural_type == StructuralType.PARENTHESIZED:
                has_parenthesized_dialogue = True
                parenthesized_inners.append(seg.inner.strip())
            line_parts.setdefault(seg.line_index, []).append(seg.raw)

    cleaned_lines = []
    for l_idx in sorted(line_parts.keys()):
        l_str = " ".join(line_parts[l_idx]).strip()
        if l_str:
            cleaned_lines.append(l_str)

    cleaned = "\n".join(cleaned_lines).strip()
    is_sdh_only = False
    if not cleaned or cleaned == "":
        cleaned = EMPTY_PLACEHOLDER
        is_sdh_only = True

    return CueProvenance(
        index=index,
        raw_content=text,
        cleaned_content=cleaned,
        segments=segments,
        has_bracketed_dialogue=has_bracketed_dialogue,
        has_parenthesized_dialogue=has_parenthesized_dialogue,
        bracketed_dialogue_inners=bracketed_inners,
        parenthesized_dialogue_inners=parenthesized_inners,
        removed_sdh_segments=removed_sdh,
        is_sdh_only=is_sdh_only
    )


def clean_subtitle_text(text: str) -> str:
    """
    Synchronous backward-compatible wrapper for cleaning SDH noise from a subtitle string.
    Uses language-agnostic structural segmentation, deterministic heuristics, and fail-safe dialogue preservation.
    """
    if not text:
        return EMPTY_PLACEHOLDER

    # Fast check: pure music symbols without any alphanumeric text
    if MUSIC_NOTES_REGEX.search(text):
        cleaned_m = MUSIC_NOTES_REGEX.sub('', text).strip()
        if not any(c.isalnum() for c in cleaned_m):
            return EMPTY_PLACEHOLDER

    prov = analyze_subtitle_cue(text)
    # In synchronous mode without semantic classifier, fail safe:
    # Any ambiguous segment that wasn't proven NON_DIALOGUE is preserved as dialogue
    return prov.cleaned_content


def is_sdh_description(inner: str) -> bool:
    """
    Backward-compatible helper: checks if a string is deterministically non-dialogue.
    Does not use any static language wordlists.
    """
    if not inner or not inner.strip():
        return True
    seg = SegmentAnalysis(
        raw=f"[{inner}]",
        inner=inner,
        structural_type=StructuralType.BRACKETED,
        classification=SegmentClassification.UNCERTAIN
    )
    det = classify_structural_segment_deterministic(seg)
    return det == SegmentClassification.NON_DIALOGUE


def strip_speaker_prefix(line: str) -> str:
    """Strips structural speaker label prefixes such as 'MAN: Hello' -> 'Hello'."""
    m = SPEAKER_PREFIX_REGEX.match(line.strip())
    if m:
        label = m.group(1).strip()
        if len(label) <= 30 and not any(c in label for c in ".,?!;"):
            return m.group(2).strip()
    return line


def sanitize_srt_content(srt_content: str) -> Tuple[List[srt.Subtitle], int]:
    """
    Synchronous backward-compatible SRT sanitizer.
    """
    subs = list(srt.parse(srt_content))
    file_speaker_labels = extract_file_speaker_labels(subs)
    cleaned_count = 0

    for idx, sub in enumerate(subs):
        original = sub.content
        prov = analyze_subtitle_cue(original, file_speaker_labels=file_speaker_labels, index=idx + 1)
        if original != prov.cleaned_content:
            cleaned_count += 1
        sub.content = prov.cleaned_content

    return subs, cleaned_count


async def sanitize_srt_content_with_provenance(
    srt_content: str,
    source_language: str = "English",
    classifier_fn: Optional[Any] = None,
    job_id: Optional[int] = None,
    concurrency: Optional[int] = None,
) -> SanitizerResult:
    """
    Asynchronous, fully dynamic, language-agnostic SRT sanitizer with complete provenance tracking.
    1. Extracts file-level structural evidence (e.g. repeated speaker prefixes).
    2. Performs structural segmentation into structural candidates.
    3. Runs deterministic syntax/typography pre-classification.
    4. Globally collects ambiguous candidates across the file and performs bounded concurrent semantic classification.
    5. Fails safe: if classifier fails, times out, or returns UNCERTAIN, dialogue is preserved.
    6. Returns sanitized subs, per-cue CueProvenance mapping, count of cleaned cues, and diagnostic telemetry.
    """
    t_total_start = time.perf_counter()

    # Phase 1: Local Analysis (SRT parsing, structural segmentation, deterministic heuristics)
    t_analysis_start = time.perf_counter()
    subs = list(srt.parse(srt_content))
    file_speaker_labels = extract_file_speaker_labels(subs)
    provenance_map: Dict[int, CueProvenance] = {}
    cleaned_count = 0

    ambiguous_items: List[Dict[str, Any]] = []
    # Map from normalized text to list of (cue_idx, seg_idx) for global deduplication
    text_to_segments: Dict[str, List[Tuple[int, int]]] = {}

    for cue_idx, sub in enumerate(subs):
        prov = analyze_subtitle_cue(sub.content, file_speaker_labels=file_speaker_labels, index=cue_idx + 1)
        provenance_map[cue_idx] = prov

        for seg_idx, seg in enumerate(prov.segments):
            if seg.classification == SegmentClassification.UNCERTAIN:
                norm_text = seg.inner.strip()
                if norm_text:
                    if norm_text not in text_to_segments:
                        text_to_segments[norm_text] = []
                        ambiguous_items.append({
                            "id": len(ambiguous_items),
                            "text": norm_text
                        })
                    text_to_segments[norm_text].append((cue_idx, seg_idx))

    t_local_analysis = time.perf_counter() - t_analysis_start

    # Phase 2: Bounded bulk semantic classification (if ambiguous items exist and classifier_fn available)
    t_classifier_start = time.perf_counter()
    batch_size = 50  # Strict bounded batch size invariant
    effective_concurrency = 1
    chunks: List[List[Dict[str, Any]]] = []

    if ambiguous_items and classifier_fn is not None:
        if concurrency is None:
            try:
                from app.core.db import get_positive_int_setting
                concurrency_setting = get_positive_int_setting("batch_concurrency", 2)
            except Exception:
                concurrency_setting = 2
        else:
            concurrency_setting = concurrency

        effective_concurrency = max(1, min(int(concurrency_setting), 4))
        chunks = [ambiguous_items[i:i + batch_size] for i in range(0, len(ambiguous_items), batch_size)]
        id_to_classification: Dict[int, SegmentClassification] = {}
        sem = asyncio.Semaphore(effective_concurrency)

        async def _classify_chunk_task(chk: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Any, Optional[BaseException]]:
            async with sem:
                try:
                    res = await classifier_fn(
                        chk,
                        source_language=source_language,
                        job_id=job_id
                    )
                    return chk, res, None
                except asyncio.CancelledError:
                    raise
                except Exception as ex:
                    return chk, None, ex

        # Run chunks concurrently with bounded semaphore
        tasks = [asyncio.create_task(_classify_chunk_task(c)) for c in chunks]
        try:
            chunk_results = await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        for chk, results, exc in chunk_results:
            if exc is not None or results is None:
                # Fail-safe per chunk: provider error -> treat all items in chunk as UNCERTAIN (preserved dialogue)
                for item in chk:
                    id_to_classification[item["id"]] = SegmentClassification.UNCERTAIN
            else:
                if isinstance(results, list):
                    for r in results:
                        if isinstance(r, dict) and "id" in r and "classification" in r:
                            c_val = str(r["classification"]).upper().strip()
                            if c_val == "NON_DIALOGUE":
                                id_to_classification[r["id"]] = SegmentClassification.NON_DIALOGUE
                            elif c_val == "DIALOGUE":
                                id_to_classification[r["id"]] = SegmentClassification.DIALOGUE
                            else:
                                id_to_classification[r["id"]] = SegmentClassification.UNCERTAIN
                # Any item in chunk not explicitly mapped defaults to UNCERTAIN
                for item in chk:
                    if item["id"] not in id_to_classification:
                        id_to_classification[item["id"]] = SegmentClassification.UNCERTAIN

        # Apply classification results back to segments deterministically
        for item in ambiguous_items:
            item_id = item["id"]
            item_text = item["text"]
            cls_res = id_to_classification.get(item_id, SegmentClassification.UNCERTAIN)

            for cue_idx, seg_idx in text_to_segments.get(item_text, []):
                seg = provenance_map[cue_idx].segments[seg_idx]
                if cls_res == SegmentClassification.NON_DIALOGUE:
                    seg.classification = SegmentClassification.NON_DIALOGUE
                    seg.provenance = ClassificationProvenance.SEMANTIC_VERIFIED
                    seg.reason = "semantic_non_dialogue"
                elif cls_res == SegmentClassification.DIALOGUE:
                    seg.classification = SegmentClassification.DIALOGUE
                    seg.provenance = ClassificationProvenance.SEMANTIC_VERIFIED
                    seg.reason = "semantic_dialogue"
                else:
                    # UNCERTAIN -> fail-safe preserve
                    seg.classification = SegmentClassification.DIALOGUE
                    seg.provenance = ClassificationProvenance.FAIL_SAFE_PRESERVED
                    seg.reason = "fail_safe_uncertain_preserved"
    else:
        # No classifier provided or no ambiguous items -> fail-safe: all ambiguous items become preserved dialogue
        for norm_text, seg_refs in text_to_segments.items():
            for cue_idx, seg_idx in seg_refs:
                seg = provenance_map[cue_idx].segments[seg_idx]
                seg.classification = SegmentClassification.DIALOGUE
                seg.provenance = ClassificationProvenance.FAIL_SAFE_PRESERVED
                seg.reason = "fail_safe_no_classifier"

    t_classifier_wait = time.perf_counter() - t_classifier_start

    # Phase 3: Final content reconstruction and provenance assembly
    t_recon_start = time.perf_counter()
    for cue_idx, sub in enumerate(subs):
        prov = provenance_map[cue_idx]
        line_parts: Dict[int, List[str]] = {}
        removed_sdh = []
        has_bracketed_dialogue = False
        has_parenthesized_dialogue = False
        bracketed_inners = []
        parenthesized_inners = []

        for seg in prov.segments:
            if seg.classification == SegmentClassification.NON_DIALOGUE:
                removed_sdh.append(seg.raw)
            else:
                if seg.structural_type == StructuralType.BRACKETED:
                    has_bracketed_dialogue = True
                    bracketed_inners.append(seg.inner.strip())
                elif seg.structural_type == StructuralType.PARENTHESIZED:
                    has_parenthesized_dialogue = True
                    parenthesized_inners.append(seg.inner.strip())
                line_parts.setdefault(seg.line_index, []).append(seg.raw)

        cleaned_lines = []
        for l_idx in sorted(line_parts.keys()):
            l_str = " ".join(line_parts[l_idx]).strip()
            if l_str:
                cleaned_lines.append(l_str)

        cleaned_text = "\n".join(cleaned_lines).strip()
        is_sdh_only = False
        if not cleaned_text or cleaned_text == "":
            cleaned_text = EMPTY_PLACEHOLDER
            is_sdh_only = True

        prov.cleaned_content = cleaned_text
        prov.has_bracketed_dialogue = has_bracketed_dialogue
        prov.has_parenthesized_dialogue = has_parenthesized_dialogue
        prov.bracketed_dialogue_inners = bracketed_inners
        prov.parenthesized_dialogue_inners = parenthesized_inners
        prov.removed_sdh_segments = removed_sdh
        prov.is_sdh_only = is_sdh_only

        if sub.content != cleaned_text:
            cleaned_count += 1
        sub.content = cleaned_text

    t_reconstruction = time.perf_counter() - t_recon_start
    t_total = time.perf_counter() - t_total_start

    telemetry = {
        "total_s": round(t_total, 2),
        "local_analysis_s": round(t_local_analysis, 2),
        "classifier_wait_s": round(t_classifier_wait, 2),
        "reconstruction_s": round(t_reconstruction, 2),
        "ambiguous_unique": len(ambiguous_items),
        "classifier_batches": len(chunks),
        "classifier_concurrency": effective_concurrency if chunks else 1,
    }

    return SanitizerResult(subs, provenance_map, cleaned_count, telemetry=telemetry)



def subs_to_srt_string(subs: List[srt.Subtitle]) -> str:
    """Formats subtitle objects back to valid SRT text, ensuring sequential numbering.

    INVARIANT: Every input cue must appear in the output.
    srt.compose(reindex=True) silently omits cues where start >= end (zero duration)
    or content is empty.
    We ensure sequential numbering and non-empty content on fresh Subtitle instances
    (without mutating the input list) and compose with reindex=False so that no cues
    are dropped.
    """
    formatted_subs = []
    for i, sub in enumerate(subs):
        content = sub.content
        if not content or not content.strip():
            content = EMPTY_PLACEHOLDER
        formatted_subs.append(
            srt.Subtitle(
                index=i + 1,
                start=sub.start,
                end=sub.end,
                content=content,
                proprietary=getattr(sub, "proprietary", None) or "",
            )
        )
    return srt.compose(formatted_subs, reindex=False)
