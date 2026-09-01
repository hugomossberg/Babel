import re
import os
import srt
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

class BatchSemanticState:
    UNVERIFIED = "UNVERIFIED"
    ALIGNED = "ALIGNED"
    SUSPECT = "SUSPECT"
    CONFIRMED_CORRUPT = "CONFIRMED_CORRUPT"
    REPAIRING = "REPAIRING"
    REPAIRED = "REPAIRED"
    FAILED_REPAIR = "FAILED_REPAIR"

class IncidentState:
    DISCOVERED = "DISCOVERED"
    CONFIRMED = "CONFIRMED"
    REPAIRING = "REPAIRING"
    REPAIRED = "REPAIRED"
    FAILED_REPAIR = "FAILED_REPAIR"
    VERIFIED = "VERIFIED"

@dataclass
class PrimaryBatchInfo:
    batch_idx: int          # 0-indexed batch number (0, 1, 2, ...)
    start_idx: int          # 0-indexed inclusive start cue index in the file
    end_idx: int            # 0-indexed inclusive end cue index in the file
    state: str = "UNVERIFIED"
    repair_attempts: int = 0
    verdict: str = "ALIGNED" # "ALIGNED" | "SUSPECT" | "CORRUPT" | "UNCERTAIN"
    confidence: str = "HIGH" # "HIGH" | "MEDIUM" | "LOW"
    details: str = ""

    @property
    def cue_count(self) -> int:
        return self.end_idx - self.start_idx + 1

    @property
    def start_id(self) -> int:
        return self.start_idx + 1

    @property
    def end_id(self) -> int:
        return self.end_idx + 1

    @property
    def is_terminal(self) -> bool:
        return self.state in {BatchSemanticState.REPAIRED, BatchSemanticState.FAILED_REPAIR, BatchSemanticState.ALIGNED}

    @property
    def is_repairable(self) -> bool:
        if self.state in {BatchSemanticState.FAILED_REPAIR, BatchSemanticState.REPAIRED, BatchSemanticState.ALIGNED, BatchSemanticState.UNVERIFIED}:
            return False
        return self.repair_attempts < 2 and self.state in {BatchSemanticState.SUSPECT, BatchSemanticState.CONFIRMED_CORRUPT}

@dataclass
class AlignmentRegion:
    start_idx: int  # 0-indexed inclusive start cue index
    end_idx: int    # 0-indexed inclusive end cue index
    verdict: str    # "SHIFT_PLUS_1" | "SHIFT_MINUS_1" | "MERGED" | "UNCERTAIN"
    confidence: str # "HIGH" | "MEDIUM" | "LOW"
    details: str = ""

@dataclass
class AlignmentIncident:
    start_idx: int  # 0-indexed inclusive start cue index
    end_idx: int    # 0-indexed inclusive end cue index
    verdict: str    # "SHIFT_PLUS_1" | "SHIFT_MINUS_1" | "MERGED" | "CONFLICTING_SHIFT" | "COMPLEX_SHIFT" | "UNCERTAIN"
    confidence: str # "HIGH" | "MEDIUM" | "LOW"
    supporting_findings: List[Dict[str, Any]] = None
    details: str = ""
    confirmation_required: bool = False
    incident_id: str = ""
    state: str = "DISCOVERED"
    repair_attempts: int = 0

    def __post_init__(self):
        if not self.incident_id:
            self.incident_id = f"inc_{self.start_idx}_{self.end_idx}"
        if self.supporting_findings is None:
            self.supporting_findings = []

class SemanticIncidentTracker:
    """
    Maintains persistent canonical alignment incident and primary batch state across the entire lifetime of a translation job.
    Enforces that:
    1. Primary batches are the canonical provenance unit for semantic integrity.
    2. A batch or incident gets at most 2 total repair attempts across all QA loops.
    3. Terminal states (ALIGNED, REPAIRED, FAILED_REPAIR) are never re-created, re-audited, or re-repaired.
    4. Findings within the same primary batch coalesce into a single batch-level semantic entity.
    5. Global recovery budget strictly bounds total recovery attempts across the entire job.
    """
    def __init__(self, total_cues: int = 0, batch_size: int = 50):
        self._incidents: Dict[str, AlignmentIncident] = {}
        self._batches: Dict[int, PrimaryBatchInfo] = {}
        self.total_cues = total_cues
        self.batch_size = max(1, batch_size) if batch_size > 0 else 50
        self.total_recovery_dispatches = 0
        if total_cues > 0:
            self.init_batches(total_cues, self.batch_size)

    def init_batches(self, total_cues: int, batch_size: int = 50) -> List[PrimaryBatchInfo]:
        self.total_cues = total_cues
        self.batch_size = max(1, batch_size) if batch_size > 0 else 50
        batches = []
        b_idx = 0
        for s in range(0, total_cues, self.batch_size):
            e = min(total_cues - 1, s + self.batch_size - 1)
            if b_idx not in self._batches:
                b_info = PrimaryBatchInfo(batch_idx=b_idx, start_idx=s, end_idx=e)
                self._batches[b_idx] = b_info
            batches.append(self._batches[b_idx])
            b_idx += 1
        return batches

    def get_batches(self) -> List[PrimaryBatchInfo]:
        return [self._batches[k] for k in sorted(self._batches.keys())]

    def find_batch_for_cue(self, cue_idx: int) -> Optional[PrimaryBatchInfo]:
        for b in self._batches.values():
            if b.start_idx <= cue_idx <= b.end_idx:
                return b
        return None

    def find_batch_by_index(self, batch_idx: int) -> Optional[PrimaryBatchInfo]:
        return self._batches.get(batch_idx)

    def get_repairable_batches(self) -> List[PrimaryBatchInfo]:
        return [b for b in self.get_batches() if b.is_repairable]

    def get_active_corrupt_batches(self) -> List[PrimaryBatchInfo]:
        return [b for b in self.get_batches() if b.state in {BatchSemanticState.CONFIRMED_CORRUPT, BatchSemanticState.FAILED_REPAIR, BatchSemanticState.REPAIRING}]

    def find_overlapping_incident(self, start_idx: int, end_idx: int, tolerance: int = 4) -> Optional[AlignmentIncident]:
        """Finds any tracked incident that overlaps or is adjacent (within tolerance) to the given range."""
        for inc in self._incidents.values():
            if not (end_idx < inc.start_idx - tolerance or start_idx > inc.end_idx + tolerance):
                return inc
        return None

    def register_or_merge(self, clustered_incidents: List[AlignmentIncident]) -> List[AlignmentIncident]:
        """
        Registers new clustered findings against existing persistent incidents.
        Also synchronizes with primary batch provenance.
        """
        canonical_list: List[AlignmentIncident] = []
        for new_inc in clustered_incidents:
            existing = self.find_overlapping_incident(new_inc.start_idx, new_inc.end_idx)
            if existing:
                existing.start_idx = min(existing.start_idx, new_inc.start_idx)
                existing.end_idx = max(existing.end_idx, new_inc.end_idx)
                if new_inc.details and new_inc.details not in existing.details:
                    existing.details = f"{existing.details}; {new_inc.details}".strip("; ")
                if existing.state == IncidentState.VERIFIED:
                    existing.state = IncidentState.DISCOVERED
                canonical_list.append(existing)
            else:
                self._incidents[new_inc.incident_id] = new_inc
                canonical_list.append(new_inc)

            # Map incident to underlying primary batches
            for b in self._batches.values():
                if not (new_inc.end_idx < b.start_idx or new_inc.start_idx > b.end_idx):
                    if b.state not in {BatchSemanticState.REPAIRED, BatchSemanticState.FAILED_REPAIR}:
                        b.state = BatchSemanticState.SUSPECT
                        b.verdict = new_inc.verdict
                        b.confidence = new_inc.confidence
                        b.details = new_inc.details
        return canonical_list

    def get_repairable_incidents(self, incidents: List[AlignmentIncident]) -> List[AlignmentIncident]:
        """Returns only incidents eligible for repair (repair_attempts < 2 and not terminal)."""
        repairable = []
        for inc in incidents:
            if inc.state in {IncidentState.FAILED_REPAIR, IncidentState.REPAIRED, IncidentState.VERIFIED}:
                continue
            if inc.repair_attempts >= 2:
                inc.state = IncidentState.FAILED_REPAIR
                continue
            repairable.append(inc)
        return repairable

    def resolve_incidents_for_batch(self, b: PrimaryBatchInfo) -> None:
        """
        Resolves active incidents overlapping with a confirmed ALIGNED/REPAIRED batch.
        Only resolves an incident if it no longer overlaps with any OTHER active corrupt batch.
        """
        for inc in self._incidents.values():
            if inc.state in {IncidentState.FAILED_REPAIR, IncidentState.VERIFIED, IncidentState.REPAIRED}:
                continue

            # Check if this incident overlaps with the newly resolved batch 'b'
            if not (inc.end_idx < b.start_idx or inc.start_idx > b.end_idx):
                # Incident overlaps with 'b'. Does it overlap with ANY OTHER active batch?
                overlaps_other_active_batch = False
                for other_b in self._batches.values():
                    if other_b.batch_idx == b.batch_idx:
                        continue
                    if not (inc.end_idx < other_b.start_idx or inc.start_idx > other_b.end_idx):
                        if other_b.state in {BatchSemanticState.SUSPECT, BatchSemanticState.CONFIRMED_CORRUPT, BatchSemanticState.FAILED_REPAIR, BatchSemanticState.REPAIRING}:
                            overlaps_other_active_batch = True
                            break

                if not overlaps_other_active_batch:
                    inc.state = IncidentState.VERIFIED

    def get_all_active_issues(self) -> List[str]:
        """Returns issues for any currently unresolved / failed batches or incidents."""
        issues = []
        for b in sorted(self._batches.values(), key=lambda x: x.start_idx):
            if b.state in {BatchSemanticState.FAILED_REPAIR, BatchSemanticState.CONFIRMED_CORRUPT, BatchSemanticState.REPAIRING}:
                issues.append(f"{b.verdict} at cues {b.start_id}-{b.end_id}: {b.details or 'Semantic alignment anomaly'}")
        for inc in sorted(self._incidents.values(), key=lambda x: x.start_idx):
            if inc.state in {IncidentState.FAILED_REPAIR, IncidentState.CONFIRMED, IncidentState.REPAIRING}:
                inc_desc = f"{inc.verdict} at cues {inc.start_idx + 1}-{inc.end_idx + 1}: {inc.details}"
                if not any(f"cues {inc.start_idx + 1}" in iss for iss in issues):
                    issues.append(inc_desc)
        return issues

def extract_batch_alignment_samples(
    source_subs: List[srt.Subtitle],
    translated_subs: List[srt.Subtitle],
    start_idx: int,
    end_idx: int,
    max_pairs: int = 8,
    anomaly_indices: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Extracts stratified representative source and target cue pairs for a specific primary batch.

    Coverage strategy (all in a single consolidated request):
    1. Batch start (first 1–2 cues) — catches early shift.
    2. Batch end (last 1–2 cues) — catches trailing shift.
    3. Batch boundary cues (start_idx and end_idx exactly) — inter-batch boundary integrity.
    4. Stratified interior positions (beginning-quarter, mid, end-quarter).
    5. Split-sentence hotspots: cues whose source text ends without sentence-terminal
       punctuation (.?!), indicating a continuation line whose partner cue must also align.
    6. Deterministic suspect spans from anomaly_indices if provided.

    All chosen positions are de-duplicated and sorted before being returned.
    The total number of samples stays at or below max_pairs.
    """
    min_len = min(len(source_subs), len(translated_subs))
    s_idx = max(0, start_idx)
    e_idx = min(min_len - 1, end_idx)
    span_len = e_idx - s_idx + 1
    if span_len <= 0:
        return []

    def _is_valid(i: int) -> bool:
        if not hasattr(source_subs[i], "content"):
            return False
        c = source_subs[i].content.strip()
        return bool(c) and c != "<i></i>"

    valid_indices = [i for i in range(s_idx, e_idx + 1) if _is_valid(i)]
    if not valid_indices:
        valid_indices = list(range(s_idx, e_idx + 1))

    if len(valid_indices) <= max_pairs:
        chosen = set(valid_indices)
    else:
        chosen: set = set()

        # --- 1. Batch boundary (first and last valid cue in batch) ---
        chosen.add(valid_indices[0])
        chosen.add(valid_indices[-1])

        # --- 2. Stratified interior: quarters ---
        n = len(valid_indices)
        for frac in (0.25, 0.5, 0.75):
            chosen.add(valid_indices[int(frac * (n - 1))])

        # --- 3. Split-sentence hotspots ---
        # A cue without terminal punctuation is likely a split sentence;
        # the *following* cue is the most critical alignment check point.
        _TERMINAL = {".", "!", "?", "…", '"', "'", "»", "›"}
        for i in valid_indices:
            if len(chosen) >= max_pairs:
                break
            content = source_subs[i].content.strip().rstrip()
            # Strip HTML-like tags for terminal check
            bare = re.sub(r"<[^>]+>", "", content).strip()
            if bare and bare[-1] not in _TERMINAL:
                # Include next cue as the continuation partner
                if i + 1 <= e_idx and _is_valid(i + 1):
                    chosen.add(i + 1)
                else:
                    chosen.add(i)

        # --- 4. Deterministic suspect spans from anomaly_indices ---
        if anomaly_indices:
            for ai in anomaly_indices:
                if s_idx <= ai <= e_idx and _is_valid(ai):
                    chosen.add(ai)
                    if len(chosen) >= max_pairs:
                        break

        # --- 5. Fill remaining budget with evenly-spaced interior points ---
        step = max(1, n // (max_pairs + 1))
        for j in range(0, n, step):
            if len(chosen) >= max_pairs:
                break
            chosen.add(valid_indices[j])

    chosen_indices = sorted(list(chosen))[:max_pairs]

    samples = []
    for idx in chosen_indices:
        s_text = source_subs[idx].content.replace('\n', ' ').strip()
        t_text = translated_subs[idx].content.replace('\n', ' ').strip() if idx < len(translated_subs) else ""
        samples.append({
            "id": idx + 1,
            "source": s_text,
            "target": t_text
        })
    return samples


def cluster_alignment_findings(
    raw_findings: List[Dict[str, Any]],
    total_cues: int = 0,
    gap_tolerance: int = 6
) -> List[AlignmentIncident]:
    """
    Deterministically clusters raw semantic alignment findings into canonical AlignmentIncident objects.

    Rules:
    1. Sort raw findings by (start_idx, end_idx).
    2. Merge findings if they overlap or are within gap_tolerance cues.
    3. Conflicting verdicts (SHIFT_PLUS_1 vs SHIFT_MINUS_1) within overlapping area become CONFLICTING_SHIFT.
    4. Mixed MERGED and SHIFT verdicts become COMPLEX_SHIFT.
    5. Sets confirmation_required flag based on evidence strength, consistency, and cluster size.
    """
    if not raw_findings:
        return []

    # Filter to actionable alignment verdicts
    actionable = [
        dict(f) for f in raw_findings
        if f.get("verdict") in {"SHIFT_PLUS_1", "SHIFT_MINUS_1", "MERGED"}
        and f.get("confidence") in {"HIGH", "MEDIUM", "LOW"}
    ]
    if not actionable:
        return []

    actionable.sort(key=lambda x: (x["start_idx"], x["end_idx"]))

    clusters: List[List[Dict[str, Any]]] = []
    for f in actionable:
        if not clusters:
            clusters.append([f])
        else:
            prev_cluster = clusters[-1]
            cluster_end = max(item["end_idx"] for item in prev_cluster)
            if f["start_idx"] <= cluster_end + gap_tolerance:
                prev_cluster.append(f)
            else:
                clusters.append([f])

    incidents: List[AlignmentIncident] = []
    for cluster in clusters:
        start_idx = min(f["start_idx"] for f in cluster)
        end_idx = max(f["end_idx"] for f in cluster)
        if total_cues > 0:
            start_idx = max(0, min(total_cues - 1, start_idx))
            end_idx = max(0, min(total_cues - 1, end_idx))
        else:
            start_idx = max(0, start_idx)
            end_idx = max(0, end_idx)

        verdicts = {f["verdict"] for f in cluster}
        confidences = {f["confidence"] for f in cluster}

        # Resolve combined verdict
        if len(verdicts) == 1:
            verdict = list(verdicts)[0]
        elif "SHIFT_PLUS_1" in verdicts and "SHIFT_MINUS_1" in verdicts:
            verdict = "CONFLICTING_SHIFT"
        elif "MERGED" in verdicts:
            verdict = "COMPLEX_SHIFT"
        else:
            verdict = cluster[0]["verdict"]

        # Resolve confidence
        if "HIGH" in confidences:
            confidence = "HIGH"
        elif "MEDIUM" in confidences:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # Determine confirmation gate
        span_len = end_idx - start_idx + 1
        if verdict in {"CONFLICTING_SHIFT", "COMPLEX_SHIFT"}:
            confirmation_required = True
        elif len(cluster) == 1:
            if confidence == "HIGH" and span_len <= 8:
                confirmation_required = False
            else:
                confirmation_required = True
        elif confidence == "LOW":
            confirmation_required = True
        elif span_len > 15 and len(cluster) < 3:
            confirmation_required = True
        else:
            confirmation_required = False

        # Aggregate details
        details_list = []
        for f in cluster:
            det = f.get("details", "").strip()
            if det and det not in details_list:
                details_list.append(det)
        details = "; ".join(details_list)

        incidents.append(AlignmentIncident(
            start_idx=start_idx,
            end_idx=end_idx,
            verdict=verdict,
            confidence=confidence,
            supporting_findings=cluster,
            details=details,
            confirmation_required=confirmation_required
        ))

    return incidents

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
from app.core.languages import get_language, normalize_language_code

# Seed for deterministic tests/results
DetectorFactory.seed = 0

# Related / ambiguous language groups where general language detectors
# cannot reliably distinguish closely-related languages (e.g., BCS family).
SIMILAR_LANGUAGE_GROUPS = [
    {"sr", "hr", "bs"}
]

def are_languages_compatible(lang_a: Optional[str], lang_b: Optional[str]) -> bool:
    """Checks if two language codes are identical or belong to the same closely-related family."""
    if not lang_a or not lang_b:
        return False
    from app.core.languages import normalize_language_code
    norm_a = normalize_language_code(lang_a).lower()
    norm_b = normalize_language_code(lang_b).lower()
    if norm_a == norm_b:
        return True
    for group in SIMILAR_LANGUAGE_GROUPS:
        if norm_a in group and norm_b in group:
            return True
    return False

CJK_PUNCT_PATTERN = re.compile(r'[\u3001\u3002\u3008-\u3011\u3014-\u301F\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u30FB]')
CJK_LETTERS_PATTERN = re.compile(r'[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u309F\u30A0-\u30FA\uAC00-\uD7AF]')

ARABIC_PUNCT_PATTERN = re.compile(r'[\u060C\u061B\u061F\u066A-\u066D]')
ARABIC_LETTERS_PATTERN = re.compile(r'[\u0620-\u064A\u0671-\u06D3\u06FA-\u06FC]')

ENTITY_INTRO_PATTERN = re.compile(
    r'\b(?:name\s+(?:is|was)|named|called|heter|heißt|heisst|nomm[eé]|appel[eé]|llamado|titulad[oa]|titre|título|title|band(?:et)?|film(?:en)?|movie|bok(?:en)?|book|song|låt(?:en)?|lied|chanson|canción|album(?:et)?)\b',
    re.IGNORECASE
)

def is_legitimate_entity_or_quoted_context(token: str, full_text: str, source_text: Optional[str] = None) -> bool:
    """
    Checks if an isolated foreign-script token is in a plausible entity, title, proper name, or quoted context.
    """
    if source_text and token in source_text:
        return True

    escaped_tok = re.escape(token)

    # 1. Quoted or bracketed token: e.g. "東京", “東京”, «東京», '東京', [東京], (東京)
    if re.search(r'[\"“«\'\[\(]\s*' + escaped_tok + r'\s*[\"”»\'\]\)]', full_text):
        return True

    # 2. Entire cue is essentially the entity itself: e.g. '李明', '東京事変'
    clean_no_tok = re.sub(escaped_tok, '', full_text).strip(' \t\n\r.,!?:;-—\"\'')
    if not clean_no_tok or len(clean_no_tok.split()) <= 1:
        return True

    # 3. Adjacent to a Capitalized / Titlecase word: e.g. '東京 Story', 'Story 東京', 'Project 東京'
    if re.search(r'\b[A-ZÀ-ÖØ-Þ]\w+\s+' + escaped_tok, full_text) or re.search(escaped_tok + r'\s+[A-ZÀ-ÖØ-Þ]\w+\b', full_text):
        return True

    # 4. Preceded by an entity introducer: e.g. 'name is 李明', 'Bandet heter 東京事変', 'book called 北京日記'
    if re.search(r'(' + ENTITY_INTRO_PATTERN.pattern + r')\s*(?:[^\w\n]+\s*)?' + escaped_tok, full_text):
        return True

    return False

import unicodedata

KNOWN_MEASUREMENT_UNITS = {
    'km', 'm', 'cm', 'mm', 'kg', 'g', 'mg', 'l', 'ml', 'cl', 'dl',
    'hz', 'khz', 'mhz', 'ghz', 'kb', 'mb', 'gb', 'tb', 'mph', 'kph',
    'fps', 'dpi', 'v', 'w', 'kw', 'mw', 'a', 'ma', 'pa', 'kpa', 'bar',
    'psi', 'rpm', 'min', 'sec', 'sek', 's', 'h', 'tim', 'db', 'cal',
    'kcal', 'oz', 'lb', 'ft', 'yd', 'in', 'pt', 'px', 'em', 'rem', 'µg', 'µm'
}
KNOWN_ORDINAL_SUFFIXES = {'st', 'nd', 'rd', 'th', 'er', 're', 'e', 'eme', 'ème', 'a', 'an', 'en', 'te', 'de'}

def extract_digit_leading_letter_tokens(text: str) -> List[Tuple[str, str, str]]:
    """
    Script-agnostic Unicode token extractor for digit-letter fused sequences.
    Identifies tokens starting with digits (0-9) immediately followed by Unicode letters (Category L*: Lu, Ll, Lt, Lm, Lo)
    and any associated combining marks (Mn, Mc, Me).
    """
    tokens = []
    n = len(text)
    i = 0
    while i < n:
        if text[i].isdigit():
            # Check left boundary: must not be preceded by a letter or combining mark
            if i > 0:
                prev_cat = unicodedata.category(text[i - 1])
                if prev_cat.startswith('L') or prev_cat in {'Mn', 'Mc', 'Me'}:
                    i += 1
                    continue

            start_idx = i
            while i < n and text[i].isdigit():
                i += 1

            digit_end = i
            digits = text[start_idx:digit_end]

            # Check if immediately followed by a Unicode letter
            if i < n and unicodedata.category(text[i]).startswith('L'):
                letter_start = i
                while i < n:
                    cat = unicodedata.category(text[i])
                    if cat.startswith('L') or cat in {'Mn', 'Mc', 'Me'}:
                        i += 1
                    else:
                        break

                letter_end = i
                letters = text[letter_start:letter_end]
                full_tok = text[start_idx:letter_end]
                tokens.append((full_tok, digits, letters))
                continue
        i += 1
    return tokens

def detect_cross_script_contamination(
    text: str,
    target_lang_code: Optional[str] = None,
    source_text: Optional[str] = None
) -> List[str]:
    """
    Language- and script-aware detector for foreign script text, punctuation, and alphanumeric contamination artifacts.
    Identifies:
    1. Cross-script punctuation hallucinations (e.g. CJK commas/periods injected into Latin text).
    2. Foreign script text token contamination (e.g. isolated Han/CJK words injected into Latin/Cyrillic sentences).
    3. Suspicious fused alphanumeric token contamination (e.g. '2den', '4hello', '7bonjour' fused LLM tokens across any script).
    Strictly preserves legitimate proper names, titles, technical acronyms (F1, 4K, 2D, MP3, H2O, S13E20), units, and ordinals.
    """
    if not text or not text.strip():
        return []

    t_code = (target_lang_code or "").lower().strip()[:2]
    issues = []

    # Check 1: CJK text or punctuation contamination in non-CJK target context
    if t_code not in {"ja", "zh", "ko"}:
        cjk_punct = CJK_PUNCT_PATTERN.findall(text)
        if cjk_punct:
            if not (source_text and any(p in source_text for p in cjk_punct)):
                issues.append(f"Unrelated CJK punctuation in non-CJK context: {''.join(sorted(set(cjk_punct)))}")

        cjk_tokens = [m.group(0) for m in re.finditer(CJK_LETTERS_PATTERN.pattern + r'+', text)]
        for tok in cjk_tokens:
            if not is_legitimate_entity_or_quoted_context(tok, text, source_text):
                issues.append(f"Foreign CJK text token contamination in non-CJK context: {tok}")

    # Check 2: Arabic text or punctuation contamination in non-Arabic target context
    if t_code not in {"ar", "fa", "ur", "ps", "ug", "he"}:
        ar_punct = ARABIC_PUNCT_PATTERN.findall(text)
        if ar_punct:
            if not (source_text and any(p in source_text for p in ar_punct)):
                issues.append(f"Unrelated Arabic punctuation in non-Arabic context: {''.join(sorted(set(ar_punct)))}")

        ar_tokens = [m.group(0) for m in re.finditer(ARABIC_LETTERS_PATTERN.pattern + r'+', text)]
        for tok in ar_tokens:
            if not is_legitimate_entity_or_quoted_context(tok, text, source_text):
                issues.append(f"Foreign Arabic text token contamination in non-Arabic context: {tok}")

    # Check 3: Suspicious fused alphanumeric token contamination across all scripts (Unicode Category L*)
    for full_tok, digits, letters in extract_digit_leading_letter_tokens(text):
        if source_text and full_tok in source_text:
            continue
        if letters.lower() in KNOWN_MEASUREMENT_UNITS:
            continue
        if letters.lower() in KNOWN_ORDINAL_SUFFIXES:
            continue
        if letters.isupper() and len(letters) <= 3:
            continue
        issues.append(f"Suspicious fused alphanumeric token contamination: {full_tok}")

    return issues

# Backward-compatible alias
detect_cross_script_punctuation_contamination = detect_cross_script_contamination

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
            expected_norm = exp_lang_obj.code if exp_lang_obj else normalize_language_code(expected_language)

        words_list = re.findall(r"\b\w+\b", cleaned.lower())
        words = set(words_list)
        swedish_word_matches = words & SWEDISH_COMMON_WORDS
        english_word_matches = words & ENGLISH_COMMON_WORDS
        # Expected target canonicalization for regional dialects:
        # Generic language detectors classify regional dialects under their base language (e.g. pt-BR -> pt).
        # When expected is a canonical regional dialect and detector identifies its base language,
        # preserve the requested canonical target code for downstream validation without altering statistical confidence.
        if expected_norm and detected_code:
            expected_obj = get_language(expected_norm)
            detected_obj = get_language(detected_code)
            if expected_obj and detected_obj and expected_obj.code != detected_obj.code:
                if expected_obj.code.split("-")[0].lower() == detected_obj.code.split("-")[0].lower():
                    detected_code = expected_obj.code

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
        is_registered = get_language(detected_code) is not None
        if len(english_word_matches) >= 2 and len(swedish_word_matches) == 0:
            token_count = len(words_list)
            english_ratio = len(english_word_matches) / token_count if token_count > 0 else 0.0
            if not is_registered:
                # Detector returned an unregistered / spurious language code (e.g. cy, so, af, tl)
                detected_code = "en"
                confidence = max(confidence, 0.95)
            elif detected_code != "en" and confidence < 0.80 and english_ratio >= 0.50:
                # Detector returned a registered language with low confidence, but text is overwhelmingly English
                detected_code = "en"
                confidence = max(confidence, 0.95)

        # Serbian Cyrillic assistance:
        # langdetect lacks an 'sr' profile and often classifies Serbian Cyrillic as 'mk'.
        # If expected is 'sr' and text contains Cyrillic letters:
        # Only assist to 'sr' when POSITIVE Serbian diagnostic evidence is present (diagnostic letters ђ, ћ
        # or distinctive grammatical words је, није, ово, шта, данас, пре, сви, овај, dobrodošli/хвала/молимо)
        # AND positive Macedonian diagnostic evidence is ABSENT.
        # Note: the Macedonian marker list must contain only words that are NOT also
        # standard Serbian. Words like ова / овој / нема / треба / во occur naturally in
        # Serbian and would otherwise veto the assist on genuine Serbian Cyrillic text.
        # Neutral or ambiguous Cyrillic without positive Serbian evidence stays 'mk' / unchanged.
        if expected_norm == "sr" and detected_code == "mk":
            has_cyrillic = bool(re.search(r'[\u0400-\u04FF]', cleaned))
            if has_cyrillic:
                has_sr_evidence = bool(re.search(r'[\u0402\u0452\u040B\u045B]|\b(је|није|ово|шта|данас|пре|сви|овај|овог|овом|добродошли|хвала|молимо)\b', cleaned, re.IGNORECASE))
                has_mk_evidence = bool(re.search(r'[\u0403\u0453\u0405\u0455\u040C\u045C]|\b(оваа|овие|денес|зошто|сака|веќе|тука|многу|додека)\b', cleaned, re.IGNORECASE))
                if has_sr_evidence and not has_mk_evidence:
                    detected_code = "sr"

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

    target_norm = normalize_language_code(target_lang_code)
    source_norm = normalize_language_code(source_lang_code)

    t_info = detect_language_heuristics(target_text, expected_language=target_norm)
    t_lang = t_info["lang"]
    t_conf = t_info["confidence"]

    s_info = detect_language_heuristics(source_text, expected_language=source_norm) if source_text else {"lang": "unknown", "confidence": 0.0}
    s_lang = s_info["lang"]
    s_conf = s_info["confidence"]

    # 2. Correct target language, compatible related language, or uncertain/short
    if are_languages_compatible(t_lang, target_norm) or t_lang == "unknown" or t_conf < 0.75:
        return {"status": "CORRECT_TARGET" if are_languages_compatible(t_lang, target_norm) else "UNCERTAIN", "target_lang": t_lang, "source_lang": s_lang, "details": "Target language matched or uncertain"}

    # 3. Target is detected as a foreign language (not target_norm and not compatible, e.g. 'de', 'fr', 'es', 'it')
    # Check if source also contains this foreign dialogue:
    if s_lang == t_lang and not are_languages_compatible(s_lang, source_norm) and s_lang != "unknown" and s_conf >= 0.75:
        if is_verified_foreign_text(source_text, s_lang):
            return {"status": "LEGIT_FOREIGN_PRESERVED", "target_lang": t_lang, "source_lang": s_lang, "details": f"Preserved foreign dialogue ({t_lang})"}

    norm_t = re.sub(r'[^\w]', '', (target_text or '').lower())
    norm_s = re.sub(r'[^\w]', '', (source_text or '').lower())
    if norm_t and norm_t == norm_s and not are_languages_compatible(s_lang, source_norm) and s_lang != "unknown" and is_verified_foreign_text(source_text, s_lang):
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
    target_norm = normalize_language_code(target_lang_code)

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

            if det != "unknown" and not are_languages_compatible(det, target_norm) and conf > 0.85:
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

        if det != "unknown" and not are_languages_compatible(det, target_norm) and conf > 0.8:
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

    target_norm = normalize_language_code(target_lang_code)
    if detected_lang != "unknown" and not are_languages_compatible(detected_lang, target_norm) and confidence < 0.8:
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


def verify_timing_integrity(
    source_cues: List[srt.Subtitle],
    translated_cues: List[srt.Subtitle],
    max_allowed_drift_ms: int = 0
) -> Dict[str, Any]:
    """
    Verifies that subtitle translation preserved the exact timing of original source cues.
    Returns detailed diagnostics if any cue has timing drift.
    """
    if not source_cues or not translated_cues:
        return {
            "valid": False,
            "error": "Empty cue list",
            "mismatch_count": 0,
            "max_start_delta_ms": -1,
            "max_end_delta_ms": -1
        }

    mismatches = []
    max_start_delta = 0
    max_end_delta = 0
    first_mismatch = None

    min_len = min(len(source_cues), len(translated_cues))
    for i in range(min_len):
        src_start = int(source_cues[i].start.total_seconds() * 1000)
        tra_start = int(translated_cues[i].start.total_seconds() * 1000)
        d_start = abs(src_start - tra_start)

        src_end = int(source_cues[i].end.total_seconds() * 1000)
        tra_end = int(translated_cues[i].end.total_seconds() * 1000)
        d_end = abs(src_end - tra_end)

        if d_start > max_start_delta:
            max_start_delta = d_start
        if d_end > max_end_delta:
            max_end_delta = d_end

        if d_start > max_allowed_drift_ms or d_end > max_allowed_drift_ms:
            mismatches.append(i + 1)
            if first_mismatch is None:
                first_mismatch = {
                    "cue_index": i + 1,
                    "source_start": str(source_cues[i].start),
                    "source_end": str(source_cues[i].end),
                    "translated_start": str(translated_cues[i].start),
                    "translated_end": str(translated_cues[i].end),
                    "delta_start_ms": d_start,
                    "delta_end_ms": d_end
                }

    count_mismatch = abs(len(source_cues) - len(translated_cues))
    is_valid = len(mismatches) == 0 and count_mismatch == 0

    return {
        "valid": is_valid,
        "source_count": len(source_cues),
        "translated_count": len(translated_cues),
        "mismatch_count": len(mismatches) + count_mismatch,
        "max_start_delta_ms": max_start_delta,
        "max_end_delta_ms": max_end_delta,
        "first_mismatch": first_mismatch
    }
