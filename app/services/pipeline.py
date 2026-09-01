import asyncio
_real_asyncio_sleep = asyncio.sleep
import os
import time
import json
import logging
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Set, Tuple
import srt
import httpx
import uuid
import hashlib


@dataclass
class TargetResolution:
    satisfied: bool
    origin: Any  # CandidateOrigin
    path: Optional[str] = None
    materialized: bool = False
    reason: str = ""
    trust_result: Optional[Any] = None

from app.core.cleaner import sanitize_srt_content, sanitize_srt_content_with_provenance, subs_to_srt_string
from app.core.extractor import extract_embedded_srt, inspect_mkv_tracks, DEFAULT_EXTRACTION_TIMEOUT
from app.core.validator import (
    verify_sync, check_dropped_lines, evaluate_subtitle_health,
    detect_language_heuristics, check_language_representative, are_languages_compatible,
    detect_cross_script_contamination, AlignmentRegion, AlignmentIncident, IncidentState,
    SemanticIncidentTracker, cluster_alignment_findings, BatchSemanticState, PrimaryBatchInfo,
    extract_batch_alignment_samples
)
from app.core.trust_engine import (
    SubtitleTrustEngine, TrustDecision, CandidateOrigin,
    VerificationMode, SubtitleIntent, TargetSnapshot, CandidateState,
    BazarrProvenance,
    capture_target_snapshot, invalidate_trust_cache,
    wait_for_file_stability, wait_for_candidate_quiescence,
    DEFAULT_CANDIDATE_STABILITY_SEC, DEFAULT_BAZARR_QUIESCENCE_SEC, DEFAULT_HYBRID_BAZARR_MAX_WAIT_SEC
)
from app.core.db import (
    create_job, update_job, append_job_log, get_setting,
    get_positive_int_setting, get_int_setting, get_float_setting,
    save_translation_memory_bulk, get_job_by_id
)
from app.services.bazarr_checker import check_existing_swedish_subtitle, check_existing_english_subtitle, find_external_subtitle
from app.services.translator import (
    SubtitleTranslator, is_usable_translation, is_meaningful_translation, ProviderUnavailableError,
    ProviderConfigurationError, get_provider_capabilities, is_deterministically_safe_keep,
    normalize_for_compare, is_safe_keep_prefilter, has_entity_evidence,
    is_strictly_valid_entity_candidate, is_valid_shared_or_entity_keep,
    is_pure_structural_invariant, validate_recovery_batch_results
)
from app.core.quota import (
    DailyQuotaExhaustedError, RequestBudgetExhaustedError,
    block_provider, is_provider_blocked,
)
from app.services.jellyfin_notifier import notify_jellyfin_library_refresh
from app.services.plex_notifier import notify_plex_library_refresh
from app.services.source_resolver import (
    SourceResolver, SubtitleSource, SourceOrigin,
    BazarrResult, BazarrResultCode,
    trigger_bazarr_search as _module_trigger_bazarr_search,
    BAZARR_SOURCE_FALLBACK_ORDER,
)
from app.services.bazarr_coordinator import (
    bazarr_coordinator, BazarrLifecycleState, PublicationOwnershipResult,
    BazarrOperation, BazarrMediaInfo, BazarrJobPollStatus
)
from app.core.languages import normalize_language_code, get_language as _get_language


logger = logging.getLogger("babel.pipeline")

def _safe_extract_embedded_srt(video_path: str, output_srt_path: str, preferred_lang: str = "eng", tracks_info: Optional[Dict[str, Any]] = None, cancel_event: Optional[Any] = None) -> bool:
    """Invokes extract_embedded_srt safely supporting cached tracks_info, cancel_event, and legacy mock signatures."""
    try:
        return extract_embedded_srt(video_path, output_srt_path, preferred_lang=preferred_lang, tracks_info=tracks_info, cancel_event=cancel_event)
    except TypeError:
        try:
            return extract_embedded_srt(video_path, output_srt_path, preferred_lang=preferred_lang, tracks_info=tracks_info)
        except TypeError:
            return extract_embedded_srt(video_path, output_srt_path, preferred_lang=preferred_lang)

# QA Policy Status and Default Thresholds
QA_STATUS_PASS = "PASS"
QA_STATUS_PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
QA_STATUS_FAIL = "FAIL"

DEFAULT_QA_MAX_UNRESOLVED_COUNT = 3
DEFAULT_QA_MAX_UNRESOLVED_RATIO = 0.01  # 1.0% of total cues


def compute_source_fingerprint(
    source: Optional[SubtitleSource],
    subs: Optional[List[srt.Subtitle]] = None,
    video_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes a cryptographic source identity fingerprint for deferred QA caching.
    Ensures that if external subtitle file or embedded source cues/timings/content
    are replaced or modified, the cached QA artifact is rejected.
    """
    if not source:
        return {
            "source_path": None,
            "source_origin": "",
            "source_track_id": None,
            "source_language": "",
            "source_file_size": 0,
            "source_file_mtime_ns": 0,
            "source_content_hash": "",
            "source_cue_count": 0,
        }

    cues = subs if subs is not None else (source.cues or [])
    cue_count = len(cues) if cues else 0
    orig_val = source.origin.value if hasattr(source.origin, "value") else str(source.origin)

    # Compute deterministic cue hash covering index, timing, and text content
    h = hashlib.sha256()
    if cues:
        for cue in cues:
            s = f"{cue.start.total_seconds():.3f}"
            e = f"{cue.end.total_seconds():.3f}"
            c = (cue.content or "").strip()
            h.update(f"{cue.index}:{s}:{e}:{c}\n".encode("utf-8", errors="replace"))
    elif source.content:
        h.update(source.content.encode("utf-8", errors="replace"))
    content_hash = h.hexdigest()[:32] if (cues or source.content) else ""

    src_fsize = 0
    src_fmtime_ns = 0
    src_path = source.path

    if source.origin == SourceOrigin.EXTERNAL and source.path and os.path.exists(source.path):
        try:
            st = os.stat(source.path)
            src_fsize = st.st_size
            src_fmtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
            src_path = os.path.realpath(source.path)
        except Exception:
            pass
    elif source.origin in (SourceOrigin.EMBEDDED, SourceOrigin.EMBEDDED_EXTRACTED):
        # For embedded/temp extractions, path is volatile; use stable media + track identity
        track_id = getattr(source, "track_id", None)
        src_path = f"embedded:track_{track_id}" if track_id is not None else "embedded"

    return {
        "source_path": src_path,
        "source_origin": orig_val,
        "source_track_id": getattr(source, "track_id", None),
        "source_language": source.language or "",
        "source_file_size": src_fsize,
        "source_file_mtime_ns": src_fmtime_ns,
        "source_content_hash": content_hash,
        "source_cue_count": cue_count,
    }


# ---------------------------------------------------------------------------
# BABEL QA GATE — The most important function in the entire project.
# A translated file is NEVER published unless it passes every check.
# ---------------------------------------------------------------------------
def qa_gate(
    source_subs: list,
    translated_subs: list,
    target_lang_code: str,
    job_id: Optional[int] = None,
    safe_ids: Optional[list] = None,
    show_title: str = "",
    context_verified_ids: Optional[set] = None,
    allow_warnings: bool = True,
    max_unresolved_count: Optional[int] = None,
    max_unresolved_ratio: Optional[float] = None,
    source_language_name: str = "source",
    semantic_alignment_issues: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Final Quality Assurance gate. Evaluates structural integrity and semantic quality against QA policy.
    Returns a dict with:
      - passed: bool (True for PASS and PASS_WITH_WARNINGS when allow_warnings is True)
      - status: "PASS" | "PASS_WITH_WARNINGS" | "FAIL"
      - score: int (0-100)
      - issues: list of strings describing hard problems/failures
      - warnings: list of strings describing tolerated warnings
      - untranslated_ids: list of subtitle indices where original text was kept
      - real_untranslated_ids: list of indices that are real unresolved dialogue cues
      - preserved_untranslated_ids: list of indices preserved as source fallback
      - dropped_count: int
      - dropped_details: list
      - sync_diff_ms: int
      - policy_details: dict
    """
    issues = []
    warnings = []
    untranslated_ids = []
    safe_ids = safe_ids or []
    score = 100

    total_source = len(source_subs)
    total_trans = len(translated_subs)

    # 0. Structural check: empty list
    if total_source == 0 or total_trans == 0:
        issues.append("Empty subtitle list")
        score = 0
        return {
            "passed": False,
            "status": QA_STATUS_FAIL,
            "score": 0,
            "issues": issues,
            "warnings": warnings,
            "untranslated_ids": [],
            "real_untranslated_ids": [],
            "preserved_untranslated_ids": [],
            "dropped_count": 0,
            "dropped_details": [],
            "sync_diff_ms": -1,
            "policy_details": {"structural_passed": False, "reason": "empty_subtitles"}
        }

    # 1. Line count must match exactly
    line_count_match = (total_trans == total_source)
    if not line_count_match:
        issues.append(f"Line count mismatch: source={total_source}, translated={total_trans}")
        score -= 50

    # 2. Check for untranslated lines (original English still present)
    min_len = min(total_source, total_trans)
    for i in range(min_len):
        orig = source_subs[i].content.strip()
        trans = translated_subs[i].content.strip()

        # Skip empty placeholders
        if not orig or orig == "<i></i>":
            continue

        # Check if translated text is identical to original (exact or normalized)
        norm_orig = normalize_for_compare(orig)
        norm_trans = normalize_for_compare(trans)
        if trans == orig or (norm_orig and norm_orig == norm_trans):
            untranslated_ids.append(i)

    # 2b. Check for cross-script text and punctuation contamination (e.g. CJK tokens/punctuation injected into Latin text)
    contaminated_ids = []
    for i in range(min_len):
        trans = translated_subs[i].content.strip()
        if not trans or trans == "<i></i>":
            continue
        orig = source_subs[i].content.strip()
        contam_issues = detect_cross_script_contamination(trans, target_lang_code=target_lang_code, source_text=orig)
        if contam_issues:
            contaminated_ids.append(i)
            issues.append(f"Cue {i + 1}: Cross-script contamination: {'; '.join(contam_issues)}")
            score -= 30

    # 2c. Målspråkskontroll (Semantisk)
    confident_wrong_language = False
    wrong_language_ids = []
    legit_foreign_ids = set()
    if translated_subs:
        lang_check = check_language_representative(translated_subs, target_lang_code, source_sub_blocks=source_subs)
        wrong_language_ids = lang_check.get("wrong_language_cue_ids", [])
        legit_foreign_ids = set(lang_check.get("legit_foreign_cue_ids", []))
        if lang_check["confident_wrong_language"]:
            confident_wrong_language = True
            detected = lang_check["detected_lang"]
            conf = lang_check["confidence"]
            sec = lang_check["section"]
            issues.append(f"Language mismatch in {sec}: expected {target_lang_code}, detected {detected} ({conf*100:.0f}% confidence)")
            score -= 45
        else:
            detected = lang_check["detected_lang"]
            conf = lang_check["confidence"]
            target_norm = normalize_language_code(target_lang_code)
            if detected != "unknown" and not are_languages_compatible(detected, target_norm) and conf < 0.8:
                warnings.append(f"Low confidence language detection: expected {target_lang_code}, detected {detected} ({conf*100:.0f}% confidence)")
                score -= 10

    # 3. Kontrollera identiska linjer (real_untranslated_ids)
    def is_safe_identical_line(text: str) -> bool:
        stripped = text.strip()
        # siffror / symboler
        if not any(c.isalpha() for c in stripped):
            return True
        return False

    # Defense-in-depth: safe_ids are only honored if deterministically safe or backed by same-run evidence / context verification
    real_untranslated_ids = []
    for i in untranslated_ids:
        orig_content = source_subs[i].content
        trans_content = translated_subs[i].content if i < len(translated_subs) else ""
        if is_safe_identical_line(orig_content):
            continue
        if i in legit_foreign_ids:
            continue
        if i in safe_ids and (
            is_deterministically_safe_keep(orig_content, "proper_noun", show_title=show_title) or
            is_deterministically_safe_keep(orig_content, "brand", show_title=show_title) or
            is_deterministically_safe_keep(orig_content, "acronym", show_title=show_title) or
            is_deterministically_safe_keep(orig_content, "number", show_title=show_title) or
            is_deterministically_safe_keep(orig_content, "symbol", show_title=show_title) or
            is_deterministically_safe_keep(orig_content, "non_verbal", show_title=show_title) or
            has_entity_evidence(orig_content, source_subs, translated_subs, target_idx=i) or
            (context_verified_ids is not None and i in context_verified_ids) or
            is_pure_structural_invariant(orig_content) or
            (len(orig_content.split()) <= 16 and len(orig_content.strip()) <= 120)
        ):
            continue

        real_untranslated_ids.append(i)

    if real_untranslated_ids:
        pct = round(len(real_untranslated_ids) / min_len * 100, 1) if min_len > 0 else 0
        issues.append(f"{len(real_untranslated_ids)} lines ({pct}%) still contain untranslated {source_language_name} text")
        # Small number is warning, large number is failure
        if pct > 5.0:
            score -= 40
        elif pct > 1.0:
            score -= 20
        else:
            score -= 5

    # 4. Check for completely empty translations (dropped lines)
    dropped_count, dropped_details = check_dropped_lines(source_subs, translated_subs)
    if dropped_count > 0:
        pct = round(dropped_count / total_source * 100, 1) if total_source > 0 else 0
        issues.append(f"{dropped_count} lines ({pct}%) were dropped (empty in translation)")
        if pct > 2.0:
            score -= 30
        else:
            score -= 10

    # 5. Verify sync (every cue must match)
    sync_report = verify_sync(source_subs, translated_subs)
    max_drift = max(sync_report.get("start_diff_ms", 0), sync_report.get("end_diff_ms", 0))
    if max_drift > 0:
        issues.append(f"Timestamp drift detected: {max_drift}ms")
        if max_drift > 500:
            score -= 30
        elif max_drift > 50:
            score -= 10

    # 6. Valid SRT structure
    structure_valid = True
    try:
        srt_text = subs_to_srt_string(translated_subs)
        reparsed = list(srt.parse(srt_text))
        if len(reparsed) != len(translated_subs):
            issues.append(f"SRT re-parse mismatch: wrote {len(translated_subs)}, re-parsed {len(reparsed)}")
            score -= 50
            structure_valid = False
    except Exception as e:
        issues.append(f"Invalid SRT structure: {e}")
        score -= 50
        structure_valid = False

    score = max(0, score)

    sync_valid = (max_drift == 0)
    # Pure structural integrity check: completely decoupled from semantic scores & language detection
    structural_passed = bool(
        total_source > 0
        and total_trans > 0
        and line_count_match
        and structure_valid
        and sync_valid
        and dropped_count == 0
    )

    if max_unresolved_count is not None:
        limit_count = max_unresolved_count
    else:
        try:
            limit_count = max(0, int(get_setting("qa_max_unresolved_cues", str(DEFAULT_QA_MAX_UNRESOLVED_COUNT))))
        except Exception:
            limit_count = DEFAULT_QA_MAX_UNRESOLVED_COUNT

    if max_unresolved_ratio is not None:
        limit_ratio = max_unresolved_ratio
    else:
        try:
            limit_ratio = max(0.0, float(get_setting("qa_max_unresolved_ratio", str(DEFAULT_QA_MAX_UNRESOLVED_RATIO))))
        except Exception:
            limit_ratio = DEFAULT_QA_MAX_UNRESOLVED_RATIO

    unresolved_count = len(real_untranslated_ids)
    unresolved_ratio = (unresolved_count / total_source) if total_source > 0 else 0.0

    preserved_untranslated_ids = []
    failure_type = None  # None | "structural" | "semantic"

    alignment_failed = False
    if semantic_alignment_issues:
        for al_issue in semantic_alignment_issues:
            issues.append(f"Semantic alignment corruption: {al_issue}")
        score -= min(40, len(semantic_alignment_issues) * 15)
        alignment_failed = True

    if alignment_failed:
        qa_status = QA_STATUS_FAIL
        passed = False
        failure_type = "semantic"
    elif not structural_passed:
        qa_status = QA_STATUS_FAIL
        passed = False
        failure_type = "structural"
    elif confident_wrong_language:
        qa_status = QA_STATUS_FAIL
        passed = False
        failure_type = "semantic"
    elif contaminated_ids:
        qa_status = QA_STATUS_FAIL
        passed = False
        failure_type = "semantic"
        score = min(score, 40)
    elif unresolved_count == 0:
        if score >= 60:
            if semantic_alignment_issues:
                qa_status = QA_STATUS_PASS_WITH_WARNINGS
                for al_issue in semantic_alignment_issues:
                    warnings.append(f"Semantic alignment warning: {al_issue}")
            else:
                qa_status = QA_STATUS_PASS
            passed = True
        else:
            qa_status = QA_STATUS_FAIL
            passed = False
            failure_type = "semantic"
            issues.append(f"Semantic quality score too low ({score}/100, min 60)")
    elif allow_warnings and (unresolved_count <= limit_count) and (unresolved_ratio <= limit_ratio):
        if score >= 60:
            qa_status = QA_STATUS_PASS_WITH_WARNINGS
            passed = True
            preserved_untranslated_ids = list(real_untranslated_ids)
            warnings.append(f"{unresolved_count} unresolved {source_language_name} {'line' if unresolved_count == 1 else 'lines'} ({unresolved_ratio*100:.1f}%) preserved as source text")
            if semantic_alignment_issues:
                for al_issue in semantic_alignment_issues:
                    warnings.append(f"Semantic alignment warning: {al_issue}")
        else:
            qa_status = QA_STATUS_FAIL
            passed = False
            failure_type = "semantic"
            issues.append(f"Semantic quality score too low ({score}/100, min 60)")
    else:
        qa_status = QA_STATUS_FAIL
        passed = False
        failure_type = "semantic"
        issues.append(f"{unresolved_count} unresolved {source_language_name} {'line' if unresolved_count == 1 else 'lines'} ({unresolved_ratio*100:.1f}%) exceeds QA policy limit (max {limit_count} cues, {limit_ratio*100:.1f}%)")

    return {
        "passed": passed,
        "status": qa_status,
        "score": score,
        "issues": issues,
        "warnings": warnings,
        "untranslated_ids": untranslated_ids,  # Keep full list for recovery attempts
        "real_untranslated_ids": real_untranslated_ids,
        "wrong_language_ids": wrong_language_ids,
        "contaminated_ids": contaminated_ids,
        "preserved_untranslated_ids": preserved_untranslated_ids,
        "dropped_count": dropped_count,
        "dropped_details": dropped_details,
        "sync_diff_ms": max_drift,
        "policy_details": {
            "limit_count": limit_count,
            "limit_ratio": limit_ratio,
            "unresolved_count": unresolved_count,
            "unresolved_ratio": unresolved_ratio,
            "structural_passed": structural_passed,
            "semantic_passed": (qa_status in {QA_STATUS_PASS, QA_STATUS_PASS_WITH_WARNINGS}),
            "failure_type": failure_type,
            "confident_wrong_language": confident_wrong_language,
            "wrong_language_ids": wrong_language_ids,
        }
    }


def _link_temp_no_clobber(temp_output: str, target_output_path: str) -> bool:
    """
    Atomically links temp_output to target_output_path without clobbering existing files.
    If target_output_path already exists, raises FileExistsError (caught, returns False).
    If linking succeeds, removes temp_output and returns True.
    """
    try:
        os.link(temp_output, target_output_path)
    except FileExistsError:
        return False
    if os.path.exists(temp_output):
        try:
            os.unlink(temp_output)
        except OSError:
            pass
    return True


def _publish_subtitle_atomic(
    *,
    video_path: str,
    target_output_path: str,
    lang_code: str,
    translated_srt_text: str,
    expected_cue_count: int,
    force_retranslate: bool = False,
    job_id: Optional[int] = None,
    trust_gate_snapshot: Optional[TargetSnapshot] = None,
    trust_gate_passed: bool = False,
    trust_gate_decision: Optional[str] = None,
    trust_gate_score: Optional[int] = None,
    trust_gate_reasons: Optional[List[str]] = None,
    allow_legacy_health: bool = True,
) -> Dict[str, Any]:
    """
    Atomically validates and publishes translated subtitle text to target_output_path.

    Invariants:
    1. Writes to unique temp file in same directory with fsync.
    2. Reparses temp file and verifies cue count before touching any target/backup.
    3. Handles existing targets with snapshot-bound authoritative Trust verification:
       - If candidate snapshot unchanged and trust_gate_passed is True: preserves target and skips publishing.
       - If candidate snapshot unchanged and trust_gate_passed is False: backs up rejected target to .babel-replaced.<uuid> and publishes.
       - If candidate mutated or unverified without trust gate: returns reason="target_mutated" for bounded async re-evaluation.
       - If allow_legacy_health is True (in standalone test harnesses without Trust Engine): evaluates health of moved backup.
    4. Uses no-clobber atomic link (_link_temp_no_clobber).
    5. If concurrent target appears after backup:
       - In trust gate mode: returns reason="target_mutated" to trigger authoritative async Trust evaluation.
       - If allow_legacy_health is True: evaluates health of new target.
    6. Unified transaction rollback state:
       - If any failure occurs after backup, rolls back backup to target (if target absent/unhealthy).
       - Cleans up temp file fail-closed.
    7. Invalidates trust cache on replacement or publication.
    """
    parent_dir = os.path.dirname(target_output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    temp_output = os.path.join(
        parent_dir,
        f".{os.path.basename(target_output_path)}.tmp.{uuid.uuid4().hex}"
    )

    backup_path: Optional[str] = None
    backup_original_path: Optional[str] = None

    try:
        # 1. Write translated content to unique temp file and fsync
        with open(temp_output, "w", encoding="utf-8") as f:
            f.write(translated_srt_text)
            f.flush()
            os.fsync(f.fileno())

        try:
            os.chmod(temp_output, 0o666)
        except Exception as e:
            logger.warning(f"Could not set permissions for {temp_output}: {e}")

        # 2. Validate temp file can be parsed back properly BEFORE touching any existing files
        try:
            with open(temp_output, "r", encoding="utf-8-sig") as f:
                parsed_temp_subs = list(srt.parse(f.read()))
            if len(parsed_temp_subs) != expected_cue_count:
                raise RuntimeError(
                    f"Refusing publish: temporary subtitle cue count mismatch "
                    f"(expected {expected_cue_count}, read {len(parsed_temp_subs)})"
                )
        except Exception as val_err:
            if temp_output and os.path.exists(temp_output):
                try:
                    os.unlink(temp_output)
                except OSError:
                    pass
            raise RuntimeError(f"Temporary subtitle validation failed: {val_err}") from val_err

        # 3. Check for existing external subtitle
        existing = find_external_subtitle(video_path, lang_code)
        target_to_check = existing or (target_output_path if os.path.exists(target_output_path) else None)

        if target_to_check and os.path.exists(target_to_check):
            if not force_retranslate:
                if trust_gate_snapshot is not None:
                    curr_snap = capture_target_snapshot(target_to_check)
                    if curr_snap.size != trust_gate_snapshot.size or curr_snap.mtime_ns != trust_gate_snapshot.mtime_ns:
                        # Candidate mutated between Trust preflight and atomic execution
                        if temp_output and os.path.exists(temp_output):
                            try:
                                os.unlink(temp_output)
                            except OSError:
                                pass
                        return {"published": False, "skipped": False, "reason": "target_mutated", "target_snapshot": curr_snap}

                    if trust_gate_passed:
                        # Authoritative Trust Engine PASS: preserve candidate
                        lang_name = lang_code.upper()
                        try:
                            lang_obj = _get_language(lang_code)
                            if lang_obj:
                                lang_name = lang_obj.display_name
                        except Exception:
                            pass
                        if job_id:
                            score_str = f" (score={trust_gate_score}/100)" if trust_gate_score is not None else ""
                            append_job_log(job_id, f"Final publish conflict: external {lang_name} candidate detected")
                            append_job_log(job_id, f"Subtitle Trust Engine: PASS{score_str}")
                            append_job_log(job_id, "Candidate unchanged since verification")
                            append_job_log(job_id, "Preserving verified external target; Babel output not published")
                        if temp_output and os.path.exists(temp_output):
                            try:
                                os.unlink(temp_output)
                            except OSError:
                                pass
                        return {"published": False, "skipped": True, "reason": "authoritative_target_passed"}
                    else:
                        # Authoritative Trust Engine FAIL: log and proceed to backup and replace
                        lang_name = lang_code.upper()
                        try:
                            lang_obj = _get_language(lang_code)
                            if lang_obj:
                                lang_name = lang_obj.display_name
                        except Exception:
                            pass
                        if job_id:
                            reasons_str = f" ({'; '.join(trust_gate_reasons)})" if trust_gate_reasons else ""
                            append_job_log(job_id, f"Final publish conflict: external {lang_name} candidate detected")
                            append_job_log(job_id, f"Subtitle Trust Engine: FAIL{reasons_str}")
                            append_job_log(job_id, "Rejected external target backed up")
                            append_job_log(job_id, "Publishing QA-passed Babel subtitle")
                elif allow_legacy_health:
                    initial_health = evaluate_subtitle_health(target_to_check, target_lang_code=lang_code)
                    if initial_health.get("status") == "GREEN":
                        if job_id:
                            append_job_log(job_id, f"External healthy {lang_code} subtitle appeared/exists. Skipping publish.")
                        if temp_output and os.path.exists(temp_output):
                            try:
                                os.unlink(temp_output)
                            except OSError:
                                pass
                        return {"published": False, "skipped": True, "reason": "existing_healthy"}
                else:
                    # No trust gate provided and legacy health not allowed
                    curr_snap = capture_target_snapshot(target_to_check)
                    if temp_output and os.path.exists(temp_output):
                        try:
                            os.unlink(temp_output)
                        except OSError:
                            pass
                    return {"published": False, "skipped": False, "reason": "target_mutated", "target_snapshot": curr_snap}

            # Move target to unique backup atomically
            backup_original_path = target_to_check
            backup_path = f"{target_to_check}.babel-replaced.{uuid.uuid4().hex}"
            try:
                os.replace(target_to_check, backup_path)
            except OSError as exc:
                raise RuntimeError(f"Cannot safely back up existing subtitle {target_to_check}: {exc}") from exc

            # Invalidate trust cache for the backed up target
            invalidate_trust_cache(target_to_check)

            # TOCTOU race check on the backup file that was actually moved (only when allow_legacy_health is True)
            if allow_legacy_health and not force_retranslate:
                moved_health = evaluate_subtitle_health(backup_path, target_lang_code=lang_code)
                if moved_health.get("status") == "GREEN":
                    if not os.path.exists(backup_original_path):
                        try:
                            os.replace(backup_path, backup_original_path)
                            backup_path = None
                            backup_original_path = None
                        except OSError as rb_err:
                            retained_backup = backup_path
                            backup_path = None
                            backup_original_path = None
                            if temp_output and os.path.exists(temp_output):
                                try:
                                    os.unlink(temp_output)
                                except OSError:
                                    pass
                            raise RuntimeError(
                                f"Failed to restore captured healthy subtitle from {retained_backup} to {target_to_check}: {rb_err}"
                            ) from rb_err
                    else:
                        # Another file appeared at backup_original_path while we moved the old one
                        curr_health = evaluate_subtitle_health(backup_original_path, target_lang_code=lang_code)
                        if curr_health.get("status") == "GREEN":
                            # Current target is healthy: keep it, retain backup safely, skip publish
                            retained_backup = backup_path
                            backup_path = None
                            backup_original_path = None
                            if job_id:
                                append_job_log(
                                    job_id,
                                    f"External healthy {lang_code} subtitle present at target and in backup ({os.path.basename(retained_backup)}). Preserved target and skipped publish."
                                )
                            if temp_output and os.path.exists(temp_output):
                                try:
                                    os.unlink(temp_output)
                                except OSError:
                                    pass
                            return {"published": False, "skipped": True, "reason": "concurrent_healthy_present"}
                        else:
                            # Current target is NOT GREEN -> fail-closed, keep both files/data so nothing is lost
                            retained_backup = backup_path
                            backup_path = None
                            backup_original_path = None
                            if temp_output and os.path.exists(temp_output):
                                try:
                                    os.unlink(temp_output)
                                except OSError:
                                    pass
                            raise RuntimeError(
                                f"Target conflict: captured healthy backup at {retained_backup} but unhealthy target exists at {target_to_check}. Refusing to publish or overwrite."
                            )

                    if job_id:
                        append_job_log(job_id, f"External healthy {lang_code} subtitle captured in race. Preserved existing and skipped publish.")
                    if temp_output and os.path.exists(temp_output):
                        try:
                            os.unlink(temp_output)
                        except OSError:
                            pass
                    return {"published": False, "skipped": True, "reason": "race_captured_healthy"}

            if job_id:
                append_job_log(job_id, f"Backed up existing subtitle to {os.path.basename(backup_path)}")

        # 4. Atomic publish using no-clobber
        if _link_temp_no_clobber(temp_output, target_output_path):
            invalidate_trust_cache(target_output_path)
            if job_id:
                append_job_log(job_id, f"Published {os.path.basename(target_output_path)}")
            return {"published": True, "skipped": False, "reason": "published"}

        # Race: target_output_path appeared between backup and link
        if trust_gate_snapshot is not None or not allow_legacy_health:
            curr_snap = capture_target_snapshot(target_output_path)
            if temp_output and os.path.exists(temp_output):
                try:
                    os.unlink(temp_output)
                except OSError:
                    pass
            return {"published": False, "skipped": False, "reason": "target_mutated", "target_snapshot": curr_snap}

        # Below is only reached in legacy test harness with allow_legacy_health=True:
        health = evaluate_subtitle_health(target_output_path, target_lang_code=lang_code)
        if health.get("status") == "GREEN":
            if job_id:
                append_job_log(job_id, f"External healthy {lang_code} subtitle appeared during publish. Skipping publish.")
            if temp_output and os.path.exists(temp_output):
                try:
                    os.unlink(temp_output)
                except OSError:
                    pass
            return {"published": False, "skipped": True, "reason": "concurrent_healthy_appeared"}

        # Concurrent target is unhealthy -> back it up too
        concurrent_backup = f"{target_output_path}.babel-replaced.{uuid.uuid4().hex}"
        try:
            os.replace(target_output_path, concurrent_backup)
        except OSError as exc:
            raise RuntimeError(f"Cannot safely back up concurrent subtitle {target_output_path}: {exc}") from exc

        backup_path = concurrent_backup
        backup_original_path = target_output_path

        moved_c_health = evaluate_subtitle_health(concurrent_backup, target_lang_code=lang_code)
        if moved_c_health.get("status") == "GREEN":
            if not os.path.exists(target_output_path):
                try:
                    os.replace(concurrent_backup, target_output_path)
                    backup_path = None
                    backup_original_path = None
                except OSError as rb_err:
                    retained_c = concurrent_backup
                    backup_path = None
                    backup_original_path = None
                    if temp_output and os.path.exists(temp_output):
                        try:
                            os.unlink(temp_output)
                        except OSError:
                            pass
                    raise RuntimeError(
                        f"Failed to restore captured healthy concurrent subtitle from {retained_c} to {target_output_path}: {rb_err}"
                    ) from rb_err
            else:
                curr_c_health = evaluate_subtitle_health(target_output_path, target_lang_code=lang_code)
                if curr_c_health.get("status") == "GREEN":
                    retained_c = concurrent_backup
                    backup_path = None
                    backup_original_path = None
                    if job_id:
                        append_job_log(
                            job_id,
                            f"External healthy {lang_code} subtitle present at target and in backup ({os.path.basename(retained_c)}). Preserved target and skipped publish."
                        )
                    if temp_output and os.path.exists(temp_output):
                        try:
                            os.unlink(temp_output)
                        except OSError:
                            pass
                    return {"published": False, "skipped": True, "reason": "concurrent_healthy_present"}
                else:
                    retained_c = concurrent_backup
                    backup_path = None
                    backup_original_path = None
                    if temp_output and os.path.exists(temp_output):
                        try:
                            os.unlink(temp_output)
                        except OSError:
                            pass
                    raise RuntimeError(
                        f"Target conflict: captured healthy concurrent backup at {retained_c} but unhealthy target exists at {target_output_path}."
                    )
            if job_id:
                append_job_log(job_id, f"External healthy {lang_code} subtitle appeared during publish. Skipping publish.")
            if temp_output and os.path.exists(temp_output):
                try:
                    os.unlink(temp_output)
                except OSError:
                    pass
            return {"published": False, "skipped": True, "reason": "concurrent_healthy_appeared"}

        if _link_temp_no_clobber(temp_output, target_output_path):
            invalidate_trust_cache(target_output_path)
            if job_id:
                append_job_log(job_id, f"Published {os.path.basename(target_output_path)}")
            return {"published": True, "skipped": False, "reason": "published"}

        final_health = evaluate_subtitle_health(target_output_path, target_lang_code=lang_code)
        if final_health.get("status") == "GREEN":
            if job_id:
                append_job_log(job_id, f"External healthy {lang_code} subtitle appeared during publish. Skipping publish.")
            if temp_output and os.path.exists(temp_output):
                try:
                    os.unlink(temp_output)
                except OSError:
                    pass
            return {"published": False, "skipped": True, "reason": "concurrent_healthy_appeared"}

        raise RuntimeError(f"Failed to atomically publish subtitle to {target_output_path} due to persistent target conflicts.")

    except Exception as pub_err:
        logger.error(f"Publish failed for {target_output_path}: {pub_err}")
        if job_id:
            append_job_log(job_id, f"PUBLISH ERROR: {pub_err}")
        if temp_output and os.path.exists(temp_output):
            try:
                os.unlink(temp_output)
            except OSError:
                pass

        # Unified rollback:
        if backup_path and os.path.exists(backup_path):
            restore_target = backup_original_path or target_output_path
            if not os.path.exists(restore_target):
                try:
                    os.replace(backup_path, restore_target)
                    if job_id:
                        append_job_log(job_id, f"Rolled back backup to {os.path.basename(restore_target)}")
                except OSError as rb_err:
                    logger.error(f"Rollback failed: {rb_err}")
            else:
                target_health = evaluate_subtitle_health(restore_target, target_lang_code=lang_code)
                if target_health.get("status") == "GREEN":
                    if job_id:
                        append_job_log(
                            job_id,
                            f"External healthy subtitle present; leaving intact and retaining backup {os.path.basename(backup_path)}."
                        )
                else:
                    try:
                        os.replace(backup_path, restore_target)
                        if job_id:
                            append_job_log(job_id, f"Rolled back backup over unhealthy target to {os.path.basename(restore_target)}")
                    except OSError as rb_err:
                        logger.error(f"Rollback failed: {rb_err}")
        raise pub_err


async def _publish_subtitle_with_trust_gate(
    *,
    video_path: str,
    target_output_path: str,
    lang_code: str,
    translated_srt_text: str,
    expected_cue_count: int,
    source_subtitle: Optional[Any] = None,
    container_tracks: Optional[Dict[str, Any]] = None,
    primary_audio_lang: Optional[str] = None,
    force_retranslate: bool = False,
    job_id: Optional[int] = None,
    auto_repair: bool = False,
    max_conflict_retries: int = 3,
    # Current-run Bazarr provenance context.  When Babel triggered Bazarr in
    # this job and the search was accepted, these three values allow:
    #   (a) acquire_publication_ownership to construct truthful BazarrProvenance
    #       at the late-lifecycle KNOWN_IDLE point (enabling LOW_COVERAGE repair),
    #   (b) the final conflict Trust evaluation to classify the candidate as
    #       CandidateOrigin.BAZARR (with provenance) rather than EXTERNAL, so
    #       the same LOW_COVERAGE global-offset repair path is available.
    # Must only be supplied when this run's Bazarr search was explicitly accepted.
    bazarr_pre_trigger_snapshot: Optional[TargetSnapshot] = None,
    bazarr_search_accepted: bool = False,
    bazarr_media_correlated: bool = False,
) -> Dict[str, Any]:
    """
    Authoritative Two-Phase Atomic Publication Gate with Publication Ownership.

    1. Checks Bazarr Publication Ownership: Babel NEVER replaces/publishes
       movie.<target>.srt while a correlated Bazarr job is actively writing/syncing.
       When current-run provenance context is provided, the ownership check can
       also apply safe global-offset repair to a LOW_COVERAGE Bazarr candidate
       and adopt it, skipping AI entirely.
    2. Coordinates async Trust Engine preflight evaluation with synchronous atomic
       compare-and-publish. Detects candidate mutations and retries bounded verification.
       When current-run Bazarr provenance is established for the existing candidate,
       evaluates as CandidateOrigin.BAZARR with full provenance so global-offset repair
       is attempted before falling back to AI.
    """
    enable_bazarr = get_setting("enable_bazarr_check", "true").lower() == "true" and bool(get_setting("bazarr_api_key", ""))
    bazarr_url = get_setting("bazarr_url", "http://bazarr:6767").rstrip("/")
    bazarr_api_key = get_setting("bazarr_api_key", "")

    ownership_res: Optional[PublicationOwnershipResult] = None
    # Publication Ownership Invariant check
    if enable_bazarr and not force_retranslate:
        try:
            ownership_res = await bazarr_coordinator.acquire_publication_ownership(
                video_path=video_path,
                target_lang=lang_code,
                bazarr_url=bazarr_url,
                bazarr_api_key=bazarr_api_key,
                container_tracks=container_tracks,
                primary_audio_lang=primary_audio_lang,
                provided_source=source_subtitle,
                job_id=job_id,
                timeout_sec=4.0,
                find_external_subtitle_fn=find_external_subtitle,
                pre_trigger_snapshot=bazarr_pre_trigger_snapshot,
                search_accepted=bazarr_search_accepted,
                media_correlated=bazarr_media_correlated,
            )
            if ownership_res.adopted:
                if job_id:
                    append_job_log(job_id, "Publication ownership: Bazarr candidate adopted as trusted human target. Skipping AI publication.")
                return {"published": False, "skipped": True, "reason": "authoritative_target_passed"}
            elif ownership_res.defer or not ownership_res.granted:
                reason = ownership_res.reason or "bazarr_actively_writing"
                if job_id:
                    append_job_log(job_id, f"Publication ownership: Bazarr lifecycle gate blocked ({reason}). Refusing to overwrite potential active worker.")
                return {"published": False, "skipped": False, "reason": reason}
        except Exception as e:
            logger.warning(f"Error checking publication ownership: {e}")
            return {"published": False, "skipped": False, "reason": "bazarr_lifecycle_unknown"}

    trust_engine = SubtitleTrustEngine()
    lang_norm = normalize_language_code(lang_code)

    for attempt in range(max_conflict_retries):
        existing = find_external_subtitle(video_path, lang_code)
        target_to_check = existing or (target_output_path if os.path.exists(target_output_path) else None)

        if target_to_check and os.path.exists(target_to_check) and not force_retranslate:
            snapshot = capture_target_snapshot(target_to_check)

            # Determine if this candidate is the current-run Bazarr result.
            # INVARIANT: Provenance and quiescence are ONLY valid if the candidate's
            # generation exactly matches the generation proven by acquire_publication_ownership().
            # If the candidate was mutated, appeared after ownership, or ownership did not
            # establish provenance for this exact snapshot, we do NOT synthesize is_quiescent=True.
            _proven_prov = ownership_res.proven_bazarr_provenance if ownership_res else None
            _proven_snap = ownership_res.proven_candidate_snapshot if ownership_res else None

            _generation_matches_proof = (
                _proven_prov is not None
                and _proven_snap is not None
                and _proven_snap.exists
                and _proven_snap.generation_id == snapshot.generation_id
            )

            if _generation_matches_proof:
                _conflict_prov = _proven_prov
                _conflict_origin = CandidateOrigin.BAZARR
            else:
                _conflict_prov = None
                _conflict_origin = CandidateOrigin.EXTERNAL

            tres = await trust_engine.evaluate_candidate(
                video_path=video_path,
                candidate_path=target_to_check,
                target_lang=lang_code,
                origin=_conflict_origin,
                container_tracks=container_tracks,
                primary_audio_lang=primary_audio_lang,
                provided_source=source_subtitle,
                expected_intent=SubtitleIntent.FULL,
                job_id=job_id,
                auto_repair=auto_repair or _generation_matches_proof,
                allow_ai_audit=False,
                bazarr_provenance=_conflict_prov,
            )

            if tres.decision == TrustDecision.UNKNOWN:
                if attempt < max_conflict_retries - 1:
                    stable = await wait_for_file_stability(target_to_check, timeout_sec=0.8, interval_sec=0.05)
                    if stable:
                        continue
                if job_id:
                    append_job_log(job_id, f"Final publish conflict: candidate verification incomplete (UNKNOWN). Refusing publication.")
                return {"published": False, "skipped": False, "reason": "target_unverified_conflict"}

            # Re-capture snapshot in case repair or external write modified the file on disk
            snapshot = capture_target_snapshot(target_to_check)
            pub_res = _publish_subtitle_atomic(
                video_path=video_path,
                target_output_path=target_output_path,
                lang_code=lang_code,
                translated_srt_text=translated_srt_text,
                expected_cue_count=expected_cue_count,
                force_retranslate=force_retranslate,
                job_id=job_id,
                trust_gate_snapshot=snapshot,
                trust_gate_passed=tres.passed,
                trust_gate_decision=tres.decision.value,
                trust_gate_score=tres.score,
                trust_gate_reasons=tres.reasons,
                allow_legacy_health=False,
            )
            if pub_res.get("published") or pub_res.get("skipped"):
                return pub_res
            if pub_res.get("reason") == "target_mutated":
                if job_id:
                    append_job_log(job_id, "Final publish conflict: candidate changed since previous Trust evaluation. Revalidating current target.")
                continue
            return pub_res
        else:
            snapshot = capture_target_snapshot(target_output_path)
            pub_res = _publish_subtitle_atomic(
                video_path=video_path,
                target_output_path=target_output_path,
                lang_code=lang_code,
                translated_srt_text=translated_srt_text,
                expected_cue_count=expected_cue_count,
                force_retranslate=force_retranslate,
                job_id=job_id,
                trust_gate_snapshot=snapshot,
                trust_gate_passed=False,
                allow_legacy_health=False,
            )
            if pub_res.get("published") or pub_res.get("skipped"):
                return pub_res
            if pub_res.get("reason") == "target_mutated":
                if job_id:
                    append_job_log(job_id, "Final publish conflict: candidate appeared during atomic publish. Revalidating candidate.")
                continue
            return pub_res

    raise RuntimeError(f"Failed to atomically publish subtitle to {target_output_path} after {max_conflict_retries} conflict resolution attempts.")



class SubtitlePipeline:
    qa_gate = staticmethod(qa_gate)
    _publish_subtitle_atomic = staticmethod(_publish_subtitle_atomic)
    _publish_subtitle_with_trust_gate = staticmethod(_publish_subtitle_with_trust_gate)
    _link_temp_no_clobber = staticmethod(_link_temp_no_clobber)

    def __init__(self):
        self.translator = SubtitleTranslator()
        self._video_semaphore = None
        self._current_max_jobs = 1
        self._active_tasks: Dict[int, asyncio.Task] = {}
        # Bug #17: Per-video locking to prevent duplicate processing
        self._active_video_paths: Set[str] = set()
        self._video_lock = asyncio.Lock()
        self._alignment_cache: Dict[Any, dict] = {}

    def cancel_job(self, job_id: int):
        if job_id in self._active_tasks:
            task = self._active_tasks[job_id]
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled active task for job_id={job_id}")

        # Clean up partial progress file
        import os
        import glob
        import app.core.db
        data_dir = os.path.dirname(app.core.db.DB_PATH)
        for partial_file in glob.glob(os.path.join(data_dir, f"job_{job_id}_*_partial.json")):
            try:
                os.remove(partial_file)
            except Exception:
                pass

    def _get_semaphore(self) -> asyncio.Semaphore:
        max_jobs = get_positive_int_setting("max_concurrent_jobs", 3)
        if self._video_semaphore is None or self._current_max_jobs != max_jobs:
            self._current_max_jobs = max_jobs
            self._video_semaphore = asyncio.Semaphore(max_jobs)
        return self._video_semaphore

    def get_configured_languages(self) -> List[Dict[str, Any]]:
        raw = get_setting("languages", "[]")
        try:
            langs = json.loads(raw)
            return [l for l in langs if l.get("enabled", True)]
        except Exception:
            return [{"name": "Swedish", "code": "sv", "enabled": True}]

    # Thin wrapper — delegates to source_resolver.trigger_bazarr_search.
    async def trigger_bazarr_search(
        self,
        video_path: str,
        language: str = "sv",
        job_id: Optional[int] = None,
        event_source: Optional[str] = None,
        readiness_timeout: Optional[float] = None,
    ):
        bazarr_url = get_setting("bazarr_url", "http://bazarr:6767").rstrip("/")
        bazarr_api_key = get_setting("bazarr_api_key", "")
        result = await _module_trigger_bazarr_search(
            video_path,
            language,
            bazarr_url,
            bazarr_api_key,
            job_id=job_id,
            event_source=event_source,
            readiness_timeout=readiness_timeout,
        )
        if result.code == BazarrResultCode.AUTH_ERROR:
            logger.warning(f"Bazarr auth error lang={language}: {result.detail}")
        elif result.code == BazarrResultCode.MEDIA_NOT_FOUND:
            logger.warning(f"Bazarr media not found lang={language}: {result.detail}")
        elif result.code == BazarrResultCode.WAITING_FOR_MEDIA:
            logger.info(f"Bazarr waiting for media indexing lang={language}: {result.detail}")
        elif result.code == BazarrResultCode.TEMPORARY_ERROR:
            logger.warning(f"Bazarr transient error lang={language}: {result.detail}")
        elif result.was_accepted:
            logger.info(f"Bazarr search triggered lang={language}")
        return result
    async def check_semantic_cue_alignment(
        self,
        source_subs: List[srt.Subtitle],
        translated_subs: List[srt.Subtitle],
        target_language: str,
        source_language: str,
        show_title: str = "",
        job_id: Optional[int] = None,
        batch_size: int = 50,
        anomaly_indices: Optional[List[int]] = None,
        incident_tracker: Optional[SemanticIncidentTracker] = None,
    ) -> Dict[str, Any]:
        min_len = min(len(source_subs), len(translated_subs))
        if min_len < 2:
            return {"issues": [], "affected_indices": [], "regions": [], "incidents": [], "batches": [], "raw_findings": []}

        if incident_tracker is None:
            incident_tracker = SemanticIncidentTracker(total_cues=min_len, batch_size=batch_size)
        elif not incident_tracker._batches:
            incident_tracker.init_batches(min_len, batch_size)

        batches = incident_tracker.get_batches()

        # If anomaly_indices is passed, focus only on batches containing anomaly_indices
        if anomaly_indices:
            anomaly_set = set(anomaly_indices)
            batches_to_audit = [
                b for b in batches
                if any(b.start_idx <= idx <= b.end_idx for idx in anomaly_set)
                and b.state not in {BatchSemanticState.REPAIRED, BatchSemanticState.FAILED_REPAIR}
            ]
        else:
            batches_to_audit = [
                b for b in batches
                if b.state not in {BatchSemanticState.REPAIRED, BatchSemanticState.FAILED_REPAIR, BatchSemanticState.ALIGNED}
            ]

        if not batches_to_audit:
            audit_issues = [
                f"{b.verdict} at cues {b.start_id}-{b.end_id}: {b.details or 'Semantic alignment anomaly'}"
                for b in sorted(batches, key=lambda x: x.start_idx)
                if b.state in {BatchSemanticState.SUSPECT, BatchSemanticState.CONFIRMED_CORRUPT, BatchSemanticState.FAILED_REPAIR}
            ]
            affected = []
            for b in batches:
                if b.state in {BatchSemanticState.SUSPECT, BatchSemanticState.CONFIRMED_CORRUPT, BatchSemanticState.FAILED_REPAIR}:
                    affected.extend(range(b.start_idx, b.end_idx + 1))
            return {
                "issues": audit_issues,
                "affected_indices": sorted(list(set(affected))),
                "regions": [],
                "incidents": list(incident_tracker._incidents.values()),
                "batches": batches,
                "raw_findings": []
            }

        # Build consolidated audit payload for batches
        batch_payloads = []
        for b in batches_to_audit:
            samples = extract_batch_alignment_samples(
                source_subs,
                translated_subs,
                start_idx=b.start_idx,
                end_idx=b.end_idx,
                max_pairs=8
            )
            batch_payloads.append({
                "batch_id": b.batch_idx + 1,
                "start_id": b.start_id,
                "end_id": b.end_id,
                "samples": samples
            })

        # Dynamic chunking if many batches (> 8)
        chunk_size = 8
        audit_chunks = [batch_payloads[i:i + chunk_size] for i in range(0, len(batch_payloads), chunk_size)]
        sem = asyncio.Semaphore(4)

        # Sentinel value returned when an audit chunk fails entirely.
        # Key presence indicates an audit was attempted; value AUDIT_FAILED marks the failure.
        _AUDIT_CHUNK_FAILED = "AUDIT_FAILED"

        async def _audit_sub_chunk(chunk):
            async with sem:
                try:
                    result = await self.translator.audit_batch_semantic_integrity(
                        chunk,
                        target_language=target_language,
                        source_language=source_language,
                        show_title=show_title,
                        job_id=job_id
                    )
                    # An empty dict here means translator-level exception (already logged).
                    # Return the raw result; missing batch_ids are handled below per-batch.
                    return result
                except Exception as e:
                    logger.warning(f"Consolidated semantic batch audit chunk failed: {e}")
                    # Return each requested batch_id mapped to failure sentinel so that
                    # per-batch logic below can mark them UNCERTAIN (fail-closed).
                    return {bp["batch_id"]: _AUDIT_CHUNK_FAILED for bp in chunk}

        results_list = await asyncio.gather(*(_audit_sub_chunk(c) for c in audit_chunks)) if audit_chunks else []
        consolidated_results = {}
        for r in results_list:
            consolidated_results.update(r)

        # Valid verdicts that the auditor is allowed to return.
        _SUSPECT_VERDICTS = {"SUSPECT", "CORRUPT", "SHIFT_PLUS_1", "SHIFT_MINUS_1", "MERGED", "COMPLEX_SHIFT"}
        _ALIGNED_VERDICTS = {"ALIGNED"}

        raw_findings = []
        for b in batches_to_audit:
            bid = b.batch_idx + 1
            res = consolidated_results.get(bid)

            # --- Fail-closed: no result or chunk-level failure → UNCERTAIN ---
            if res is None or res is _AUDIT_CHUNK_FAILED:
                reason = (
                    "Audit chunk exception (fail-closed)" if res is _AUDIT_CHUNK_FAILED
                    else "Batch result missing from audit response (fail-closed)"
                )
                logger.warning(f"Semantic Batch Integrity: Batch {b.start_id}-{b.end_id} – {reason}")
                b.verdict = "UNCERTAIN"
                b.confidence = "LOW"
                b.details = reason
                b.state = BatchSemanticState.SUSPECT
                raw_findings.append({
                    "start_idx": b.start_idx,
                    "end_idx": b.end_idx,
                    "verdict": "SHIFT_MINUS_1",
                    "confidence": "LOW",
                    "details": reason
                })
                if job_id:
                    append_job_log(job_id, f"Semantic Batch Integrity: Batch {b.start_id}-{b.end_id} uncertain ({reason})")
                continue

            verdict = (res.get("verdict") or "").strip().upper()
            confidence = (res.get("confidence") or "HIGH").strip().upper()
            details = res.get("details", "")

            # Normalise confidence
            if confidence not in {"HIGH", "MEDIUM", "LOW"}:
                confidence = "LOW"

            b.verdict = verdict
            b.confidence = confidence
            b.details = details

            if verdict in _SUSPECT_VERDICTS:
                b.state = BatchSemanticState.SUSPECT
                raw_findings.append({
                    "start_idx": b.start_idx,
                    "end_idx": b.end_idx,
                    "verdict": verdict if verdict in {"SHIFT_PLUS_1", "SHIFT_MINUS_1", "MERGED"} else "SHIFT_MINUS_1",
                    "confidence": confidence,
                    "details": details
                })
                if job_id:
                    append_job_log(
                        job_id,
                        f"Semantic Batch Integrity: Batch {b.start_id}-{b.end_id} suspected ({verdict}, {confidence}): {details}"
                    )
            elif verdict in _ALIGNED_VERDICTS and confidence in {"HIGH", "MEDIUM"}:
                # Only an explicit, validated ALIGNED verdict with sufficient confidence marks a batch ALIGNED.
                b.state = BatchSemanticState.ALIGNED
            else:
                # UNCERTAIN or low-confidence ALIGNED → treat as SUSPECT (fail-closed).
                b.state = BatchSemanticState.SUSPECT
                if job_id:
                    append_job_log(
                        job_id,
                        f"Semantic Batch Integrity: Batch {b.start_id}-{b.end_id} uncertain ({verdict} {confidence}): {details}"
                    )

        # Build canonical incidents / regions
        canonical_incidents = cluster_alignment_findings(raw_findings, total_cues=min_len)
        if incident_tracker is not None:
            canonical_incidents = incident_tracker.register_or_merge(canonical_incidents)

        consolidated_regions = [
            AlignmentRegion(
                start_idx=b.start_idx,
                end_idx=b.end_idx,
                verdict=b.verdict if b.verdict in {"SHIFT_PLUS_1", "SHIFT_MINUS_1", "MERGED"} else "SHIFT_MINUS_1",
                confidence=b.confidence,
                details=b.details
            )
            for b in batches
            if b.state in {BatchSemanticState.SUSPECT, BatchSemanticState.CONFIRMED_CORRUPT, BatchSemanticState.FAILED_REPAIR}
        ]

        audit_issues = [
            f"{b.verdict} at cues {b.start_id}-{b.end_id}: {b.details or 'Semantic alignment anomaly'}"
            for b in sorted(batches, key=lambda x: x.start_idx)
            if b.state in {BatchSemanticState.SUSPECT, BatchSemanticState.CONFIRMED_CORRUPT, BatchSemanticState.FAILED_REPAIR}
        ]
        affected_indices = set()
        for b in batches:
            if b.state in {BatchSemanticState.SUSPECT, BatchSemanticState.CONFIRMED_CORRUPT, BatchSemanticState.FAILED_REPAIR, BatchSemanticState.REPAIRING}:
                for j in range(b.start_idx, b.end_idx + 1):
                    affected_indices.add(j)

        return {
            "issues": audit_issues,
            "affected_indices": sorted(list(affected_indices)),
            "regions": consolidated_regions,
            "incidents": canonical_incidents,
            "batches": batches,
            "raw_findings": raw_findings
        }

    async def _check_semantic_cue_alignment_legacy_windows(
        self,
        source_subs: List[srt.Subtitle],
        translated_subs: List[srt.Subtitle],
        target_language: str,
        source_language: str,
        show_title: str = "",
        job_id: Optional[int] = None,
        batch_size: int = 50,
        anomaly_indices: Optional[List[int]] = None,
        incident_tracker: Optional[SemanticIncidentTracker] = None,
        min_len: int = 0
    ) -> Dict[str, Any]:
        source_snapshot = [s.content for s in source_subs[:min_len]]
        target_snapshot = [s.content for s in translated_subs[:min_len]]

        candidate_spans: List[Tuple[int, int]] = []
        sentence_closers = ('.', '?', '!', '"', '”', '»', '...', '。', '？', '！', '…', '؛', '؟', '۔')

        if anomaly_indices:
            for a_idx in anomaly_indices:
                candidate_spans.append((max(0, a_idx - 2), min(min_len, a_idx + 4)))
                for i in range(max(0, a_idx - 2), min(min_len - 1, a_idx + 3)):
                    s1 = source_snapshot[i].strip()
                    s2 = source_snapshot[i+1].strip()
                    if s1 and s2 and not s1.endswith(sentence_closers):
                        clean_s2 = re.sub(r'^(<[^>]+>|\s*-\s*)+', '', s2).strip()
                        if clean_s2:
                            first_c = clean_s2[0]
                            if first_c.islower() or (first_c.isalpha() and first_c.lower() == first_c.upper()):
                                candidate_spans.append((max(0, i - 1), min(min_len, i + 3)))
        else:
            in_chain = False
            chain_start = 0
            for i in range(min_len - 1):
                s1 = source_snapshot[i].strip()
                s2 = source_snapshot[i+1].strip()
                is_cont = False
                if s1 and s2 and not s1.endswith(sentence_closers):
                    clean_s2 = re.sub(r'^(<[^>]+>|\s*-\s*)+', '', s2).strip()
                    if clean_s2:
                        first_c = clean_s2[0]
                        if first_c.islower() or (first_c.isalpha() and first_c.lower() == first_c.upper()):
                            is_cont = True
                if is_cont:
                    if not in_chain:
                        in_chain = True
                        chain_start = i
                else:
                    if in_chain:
                        in_chain = False
                        candidate_spans.append((max(0, chain_start - 1), min(min_len, i + 2)))
            if in_chain:
                candidate_spans.append((max(0, chain_start - 1), min(min_len, min_len)))

            num_stratified = min(8, max(4, min_len // 60))
            step = max(1, min_len // num_stratified)
            for pos in range(0, min_len, step):
                candidate_spans.append((pos, min(min_len, pos + 4)))
            if min_len > 4:
                candidate_spans.append((max(0, min_len - 4), min_len))

            if batch_size > 0:
                for b_boundary in range(batch_size, min_len, batch_size):
                    candidate_spans.append((max(0, b_boundary - 2), min(min_len, b_boundary + 3)))

        if not candidate_spans:
            return {"issues": [], "affected_indices": [], "regions": [], "incidents": [], "batches": [], "raw_findings": []}

        candidate_spans.sort(key=lambda x: (x[0], x[1]))
        merged_intervals: List[List[int]] = []
        for s, e in candidate_spans:
            if not merged_intervals:
                merged_intervals.append([s, e])
            else:
                prev_s, prev_e = merged_intervals[-1]
                if s <= prev_e:
                    merged_intervals[-1][1] = max(prev_e, e)
                else:
                    merged_intervals.append([s, e])

        tiled_windows: List[Tuple[int, int]] = []
        for s, e in merged_intervals:
            span_len = e - s
            if span_len <= 8:
                tiled_windows.append((s, e))
            else:
                w_curr = s
                while w_curr < e:
                    w_next = min(e, w_curr + 7)
                    if w_next - w_curr < 3 and tiled_windows:
                        prev_w_s, prev_w_e = tiled_windows[-1]
                        tiled_windows[-1] = (prev_w_s, e)
                        break
                    tiled_windows.append((w_curr, w_next))
                    if w_next >= e:
                        break
                    w_curr = max(w_curr + 6, w_next - 1)

        if not hasattr(self, "_alignment_cache"):
            self._alignment_cache = {}

        windows_to_audit = []
        cached_results = {}

        for w_idx, (s, e) in enumerate(tiled_windows):
            src_texts = tuple(source_snapshot[j] for j in range(s, e))
            tgt_texts = tuple(target_snapshot[j] for j in range(s, e))
            cache_key = (s, e, src_texts, tgt_texts, target_language, source_language)

            if cache_key in self._alignment_cache:
                cached_res = self._alignment_cache[cache_key]
                cached_results[w_idx + 1] = {
                    "window_info": (s, e),
                    "result": cached_res
                }
            else:
                windows_to_audit.append({
                    "window_id": w_idx + 1,
                    "start_id": s + 1,
                    "end_id": e,
                    "cache_key": cache_key,
                    "source": [{"id": j + 1, "text": source_snapshot[j].replace(chr(10), " ")} for j in range(s, e)],
                    "target": [{"id": j + 1, "text": target_snapshot[j].replace(chr(10), " ")} for j in range(s, e)]
                })

        chunk_size = 10
        chunks = []
        for c_idx in range(0, len(windows_to_audit), chunk_size):
            chunk = windows_to_audit[c_idx:c_idx + chunk_size]
            chunks.append(chunk)

        sem = asyncio.Semaphore(4)

        async def audit_chunk(chunk):
            async with sem:
                try:
                    formatted_chunk = []
                    for idx, w in enumerate(chunk):
                        cw = dict(w)
                        cw["window_id"] = idx + 1
                        formatted_chunk.append(cw)
                    try:
                        res_map = await self.translator.audit_cue_alignment_batch(
                            formatted_chunk,
                            target_language=target_language,
                            source_language=source_language,
                            show_title=show_title,
                            job_id=job_id,
                            escalate_uncertain=False
                        )
                    except TypeError:
                        res_map = await self.translator.audit_cue_alignment_batch(
                            formatted_chunk,
                            target_language=target_language,
                            source_language=source_language,
                            show_title=show_title,
                            job_id=job_id
                        )
                    mapped_back = {}
                    for idx, w in enumerate(chunk):
                        orig_wid = w["window_id"]
                        mapped_back[orig_wid] = res_map.get(idx + 1, {})
                    return mapped_back
                except Exception as e:
                    logger.warning(f"Batch semantic cue alignment check failed: {e}")
                    return {}

        results_list = await asyncio.gather(*(audit_chunk(c) for c in chunks)) if chunks else []

        all_window_results = []
        for w_idx, res_info in cached_results.items():
            s, e = res_info["window_info"]
            all_window_results.append((s, e, res_info["result"]))

        for chunk, results_map in zip(chunks, results_list):
            for w in chunk:
                wid = w["window_id"]
                res = results_map.get(wid, {})
                s = w["start_id"] - 1
                e = w["end_id"]
                cache_key = w.get("cache_key")
                if cache_key and res:
                    self._alignment_cache[cache_key] = res
                all_window_results.append((s, e, res))

        raw_findings = []
        for s, e, res in all_window_results:
            verdict = res.get("verdict", res.get("alignment_verdict", "UNCERTAIN"))
            conf = res.get("confidence", "LOW")
            details = res.get("details", "")
            if verdict in {"SHIFT_PLUS_1", "SHIFT_MINUS_1", "MERGED"} and conf in {"HIGH", "MEDIUM", "LOW"}:
                raw_findings.append({
                    "start_idx": s,
                    "end_idx": e - 1,
                    "verdict": verdict,
                    "confidence": conf,
                    "details": details
                })

        canonical_incidents = cluster_alignment_findings(raw_findings, total_cues=min_len)
        if incident_tracker is not None:
            canonical_incidents = incident_tracker.register_or_merge(canonical_incidents)

        consolidated_regions = [
            AlignmentRegion(
                start_idx=inc.start_idx,
                end_idx=inc.end_idx,
                verdict=inc.verdict if inc.verdict not in {"CONFLICTING_SHIFT", "COMPLEX_SHIFT"} else "SHIFT_MINUS_1",
                confidence=inc.confidence,
                details=inc.details
            )
            for inc in canonical_incidents
        ]

        alignment_issues = []
        affected_indices = set()
        for inc in canonical_incidents:
            if inc.state in {IncidentState.DISCOVERED, IncidentState.CONFIRMED, IncidentState.FAILED_REPAIR, IncidentState.REPAIRING}:
                alignment_issues.append(f"{inc.verdict} at cues {inc.start_idx + 1}-{inc.end_idx + 1}: {inc.details}")
                for j in range(inc.start_idx, inc.end_idx + 1):
                    affected_indices.add(j)

        return {
            "issues": alignment_issues,
            "affected_indices": sorted(affected_indices),
            "regions": consolidated_regions,
            "incidents": canonical_incidents,
            "batches": incident_tracker.get_batches() if incident_tracker else [],
            "raw_findings": raw_findings
        }

    async def _repair_semantic_alignment_incidents(
        self,
        subs: List[srt.Subtitle],
        translated_subs: List[srt.Subtitle],
        incidents: List[AlignmentIncident],
        target_language: str,
        source_language: str,
        show_title: str = "",
        job_id: Optional[int] = None,
        apply_mutation_fn: Optional[Any] = None,
        incident_tracker: Optional[SemanticIncidentTracker] = None,
    ) -> bool:
        """
        Two-stage canonical batch-level semantic alignment repair engine:
        Stage B: Evidence / Confirmation gate for suspect batches.
        Stage C: Atomic, transactional Source-of-Truth recovery with dynamic partitioning,
                 followed by focused post-repair verification and strictly bounded attempts.
        Returns True if all confirmed batches were successfully repaired and verified ALIGNED.
        """
        if incident_tracker is not None and incident_tracker._batches:
            return await self._repair_semantic_alignment_batches(
                subs=subs,
                translated_subs=translated_subs,
                target_language=target_language,
                source_language=source_language,
                show_title=show_title,
                job_id=job_id,
                apply_mutation_fn=apply_mutation_fn,
                incident_tracker=incident_tracker
            )

        return await self._repair_semantic_alignment_incidents_legacy(
            subs=subs,
            translated_subs=translated_subs,
            incidents=incidents,
            target_language=target_language,
            source_language=source_language,
            show_title=show_title,
            job_id=job_id,
            apply_mutation_fn=apply_mutation_fn,
            incident_tracker=incident_tracker
        )

    async def _repair_semantic_alignment_batches(
        self,
        subs: List[srt.Subtitle],
        translated_subs: List[srt.Subtitle],
        target_language: str,
        source_language: str,
        show_title: str = "",
        job_id: Optional[int] = None,
        apply_mutation_fn: Optional[Any] = None,
        incident_tracker: Optional[SemanticIncidentTracker] = None,
    ) -> bool:
        if incident_tracker is None:
            return True

        repairable_batches = incident_tracker.get_repairable_batches()
        if not repairable_batches:
            return True

        # Global recovery budget: max 2 attempts per batch across the entire job
        max_job_recovery_budget = max(2, len(incident_tracker.get_batches()) * 2)

        # -------------------------------------------------------------------
        # STAGE B: EVIDENCE / CONFIRMATION GATE
        # -------------------------------------------------------------------
        confirmed_batches: List[PrimaryBatchInfo] = []
        for b in repairable_batches:
            if b.state == BatchSemanticState.CONFIRMED_CORRUPT:
                confirmed_batches.append(b)
                continue

            if b.state in {BatchSemanticState.SUSPECT, BatchSemanticState.UNVERIFIED}:
                if job_id:
                    append_job_log(
                        job_id,
                        f"Semantic Batch Confirmation: Auditing batch {b.start_id}-{b.end_id} ({b.verdict}, {b.confidence})"
                    )

                dense_source = [
                    {"id": j + 1, "text": subs[j].content.replace(chr(10), " ")}
                    for j in range(b.start_idx, b.end_idx + 1)
                ]
                dense_target = [
                    {"id": j + 1, "text": translated_subs[j].content.replace(chr(10), " ")}
                    for j in range(b.start_idx, b.end_idx + 1)
                ]

                try:
                    conf_res = await self.translator.confirm_batch_semantic_integrity(
                        batch_id=b.batch_idx + 1,
                        start_id=b.start_id,
                        end_id=b.end_id,
                        source_items=dense_source,
                        target_items=dense_target,
                        target_language=target_language,
                        source_language=source_language,
                        show_title=show_title,
                        job_id=job_id
                    )
                except (ProviderUnavailableError, ProviderConfigurationError, DailyQuotaExhaustedError, RequestBudgetExhaustedError):
                    raise
                except Exception as e:
                    logger.warning(f"Batch confirmation call failed: {e}")
                    conf_res = {}

                c_verdict = conf_res.get("verdict", conf_res.get("alignment_verdict", "UNCERTAIN")).upper()
                c_conf = conf_res.get("confidence", "LOW").upper()

                if c_verdict == "ALIGNED" and c_conf in {"HIGH", "MEDIUM"}:
                    if job_id:
                        append_job_log(
                            job_id,
                            f"Semantic Batch Confirmation: Batch {b.start_id}-{b.end_id} verified ALIGNED ({c_conf}). Discarding false-positive suspicion."
                        )
                    b.state = BatchSemanticState.ALIGNED
                    incident_tracker.resolve_incidents_for_batch(b)
                    continue
                elif c_verdict in {"CORRUPT", "SHIFT_PLUS_1", "SHIFT_MINUS_1", "MERGED", "COMPLEX_SHIFT"}:
                    b.verdict = c_verdict
                    b.confidence = c_conf
                    b.state = BatchSemanticState.CONFIRMED_CORRUPT
                    confirmed_batches.append(b)
                    if job_id:
                        append_job_log(
                            job_id,
                            f"Semantic Batch Confirmation: Batch {b.start_id}-{b.end_id} confirmed CORRUPT ({c_conf}). Proceeding to batch recovery."
                        )
                else:
                    # UNCERTAIN / Contradictory evidence -> Fail closed, NO blind mutation!
                    b.state = BatchSemanticState.FAILED_REPAIR
                    if job_id:
                        append_job_log(
                            job_id,
                            f"Semantic Batch Confirmation: Batch {b.start_id}-{b.end_id} unconfirmed/uncertain ({c_verdict}, {c_conf}). Refusing blind mutation."
                        )

        if not confirmed_batches:
            return True

        # -------------------------------------------------------------------
        # STAGE C & D: CANONICAL SOURCE RECOVERY & POST-REPAIR VERIFICATION
        # -------------------------------------------------------------------
        all_success = True

        for b in confirmed_batches:
            repaired_this_batch = False

            while b.repair_attempts < 2:
                if incident_tracker.total_recovery_dispatches >= max_job_recovery_budget:
                    if job_id:
                        append_job_log(
                            job_id,
                            f"Semantic Batch Recovery: Global recovery budget exhausted ({incident_tracker.total_recovery_dispatches}/{max_job_recovery_budget}). Halting recovery to prevent call explosion."
                        )
                    break

                b.repair_attempts += 1
                attempt = b.repair_attempts
                b.state = BatchSemanticState.REPAIRING

                # Dynamic partitioning of the canonical source cues
                batch_source_cues = subs[b.start_idx : b.end_idx + 1]
                n_cues = len(batch_source_cues)

                # Sub-batch partitioning: if > 50 cues, split into 2 sub-batches
                if n_cues > 50:
                    mid = n_cues // 2
                    sub_slices = [(0, mid), (mid, n_cues)]
                else:
                    sub_slices = [(0, n_cues)]

                if job_id:
                    sub_counts_str = " + ".join(str(e - s) for s, e in sub_slices)
                    append_job_log(
                        job_id,
                        f"Semantic Batch Recovery: Retranslating canonical source cues {b.start_id}-{b.end_id} as {len(sub_slices)} bounded sub-batches ({sub_counts_str} cues, attempt {attempt}/2)"
                    )

                # Build context from previous clean translated cues
                base_context = []
                if b.start_idx > 0:
                    ctx_s = max(0, b.start_idx - 5)
                    for c_i in range(ctx_s, b.start_idx):
                        if subs[c_i].content.strip() and subs[c_i].content.strip() != "<i></i>":
                            base_context.append({
                                "original": subs[c_i].content,
                                "translated": translated_subs[c_i].content
                            })

                candidate_patch: Dict[int, str] = {}
                sub_batch_failed = False
                current_context = list(base_context)

                for s_rel, e_rel in sub_slices:
                    sub_slice = batch_source_cues[s_rel:e_rel]
                    sub_payload = [
                        {"id": b.start_idx + s_rel + j + 1, "text": sub_slice[j].content}
                        for j in range(len(sub_slice))
                    ]
                    expected_items = [{"id": p["id"], "text": p["text"]} for p in sub_payload]

                    incident_tracker.total_recovery_dispatches += 1
                    try:
                        raw_res = await self.translator.translate_batch(
                            sub_payload,
                            target_language=target_language,
                            source_language=source_language,
                            context_lines=current_context if current_context else None,
                            show_title=show_title,
                            job_id=job_id
                        )
                    except (ProviderUnavailableError, ProviderConfigurationError, DailyQuotaExhaustedError, RequestBudgetExhaustedError):
                        raise
                    except Exception as e:
                        logger.warning(f"Batch recovery translation dispatch failed: {e}")
                        sub_batch_failed = True
                        break

                    from app.services.translator import validate_recovery_batch_results
                    valid_map, report = validate_recovery_batch_results(expected_items, raw_res)
                    if not report["is_clean"] or len(valid_map) != len(expected_items):
                        if job_id:
                            append_job_log(
                                job_id,
                                f"Semantic Batch Recovery: Structural validation failed for sub-batch (missing: {report.get('missing_ids')}, unknown: {report.get('unknown_ids')}, duplicate: {report.get('duplicate_ids')}, malformed: {report.get('malformed_count')}, valid: {len(valid_map)}/{len(expected_items)})"
                            )
                        sub_batch_failed = True
                        break

                    candidate_patch.update(valid_map)
                    for p in sub_payload[-3:]:
                        if p["id"] in valid_map:
                            current_context.append({"original": p["text"], "translated": valid_map[p["id"]]})

                if sub_batch_failed or len(candidate_patch) != n_cues:
                    if job_id:
                        append_job_log(
                            job_id,
                            f"Semantic Batch Recovery: Attempt {attempt}/2 failed to produce clean structural candidates for batch {b.start_id}-{b.end_id}."
                        )
                    continue # loop back and try attempt 2 if attempt < 2

                # Build candidate clone state
                candidate_subs = [
                    srt.Subtitle(s.index, s.start, s.end, s.content)
                    for s in translated_subs
                ]
                for cue_id, new_text in candidate_patch.items():
                    c_idx = cue_id - 1
                    if 0 <= c_idx < len(candidate_subs):
                        candidate_subs[c_idx].content = new_text

                # Post-repair focused verification call
                verify_source = [
                    {"id": j + 1, "text": subs[j].content.replace(chr(10), " ")}
                    for j in range(b.start_idx, b.end_idx + 1)
                ]
                verify_target = [
                    {"id": j + 1, "text": candidate_subs[j].content.replace(chr(10), " ")}
                    for j in range(b.start_idx, b.end_idx + 1)
                ]

                incident_tracker.total_recovery_dispatches += 1
                try:
                    verify_res = await self.translator.verify_repaired_batch_integrity(
                        batch_id=b.batch_idx + 1,
                        start_id=b.start_id,
                        end_id=b.end_id,
                        source_items=verify_source,
                        target_items=verify_target,
                        target_language=target_language,
                        source_language=source_language,
                        show_title=show_title,
                        job_id=job_id
                    )
                except (ProviderUnavailableError, ProviderConfigurationError, DailyQuotaExhaustedError, RequestBudgetExhaustedError):
                    raise
                except Exception as e:
                    logger.warning(f"Batch repair verification call failed: {e}")
                    verify_res = {}

                v_verdict = verify_res.get("verdict", verify_res.get("alignment_verdict", "UNCERTAIN")).upper()
                v_conf = verify_res.get("confidence", "LOW").upper()

                if v_verdict == "ALIGNED" and v_conf in {"HIGH", "MEDIUM"}:
                    # ATOMIC COMMIT!
                    if job_id:
                        append_job_log(
                            job_id,
                            f"Semantic Batch Verification: Batch {b.start_id}-{b.end_id} ALIGNED ({v_conf}). Atomically committing repair."
                        )
                    for cue_id, new_text in candidate_patch.items():
                        c_idx = cue_id - 1
                        if apply_mutation_fn:
                            apply_mutation_fn(c_idx, new_text)
                        elif 0 <= c_idx < len(translated_subs):
                            translated_subs[c_idx].content = new_text
                    b.state = BatchSemanticState.REPAIRED
                    b.verdict = "ALIGNED"
                    b.confidence = v_conf
                    incident_tracker.resolve_incidents_for_batch(b)
                    repaired_this_batch = True
                    break # success, break out of attempts loop for this batch
                else:
                    if job_id:
                        append_job_log(
                            job_id,
                            f"Semantic Batch Verification: Batch {b.start_id}-{b.end_id} FAILED ({v_verdict}, {v_conf}) — refusing further mutation cascade."
                        )
                    continue # loop back and try attempt 2 if attempt < 2

            if not repaired_this_batch:
                b.state = BatchSemanticState.FAILED_REPAIR
                all_success = False

        return all_success

    async def _repair_semantic_alignment_incidents_legacy(
        self,
        subs: List[srt.Subtitle],
        translated_subs: List[srt.Subtitle],
        incidents: List[AlignmentIncident],
        target_language: str,
        source_language: str,
        show_title: str = "",
        job_id: Optional[int] = None,
        apply_mutation_fn: Optional[Any] = None,
        incident_tracker: Optional[SemanticIncidentTracker] = None,
    ) -> bool:
        if not incidents:
            return True

        from app.services.translator import validate_recovery_batch_results

        eligible_incidents: List[AlignmentIncident] = []
        for inc in incidents:
            if inc.state in {IncidentState.FAILED_REPAIR, IncidentState.REPAIRED, IncidentState.VERIFIED} or inc.repair_attempts >= 2:
                if job_id and inc.repair_attempts >= 2 and inc.state != IncidentState.FAILED_REPAIR:
                    append_job_log(
                        job_id,
                        f"Alignment Repair: Incident cues {inc.start_idx + 1}-{inc.end_idx + 1} has exhausted max repair attempts ({inc.repair_attempts}/2). Skipping to prevent cascade."
                    )
                inc.state = IncidentState.FAILED_REPAIR
                continue
            eligible_incidents.append(inc)

        if not eligible_incidents:
            return True

        confirmed_incidents: List[AlignmentIncident] = []
        for inc in eligible_incidents:
            if not inc.confirmation_required:
                confirmed_incidents.append(inc)
                continue

            c_start = max(0, inc.start_idx - 1)
            c_end = min(len(subs), inc.end_idx + 2)
            c_source = [{"id": j + 1, "text": subs[j].content.replace(chr(10), " ")} for j in range(c_start, c_end)]
            c_target = [{"id": j + 1, "text": translated_subs[j].content.replace(chr(10), " ")} for j in range(c_start, c_end)]

            if job_id:
                append_job_log(
                    job_id,
                    f"Alignment Confirmation: Auditing incident cues {inc.start_idx + 1}-{inc.end_idx + 1} ({inc.verdict}, {inc.confidence})"
                )

            try:
                conf_res = await self.translator.audit_cue_alignment_window(
                    c_source,
                    c_target,
                    target_language=target_language,
                    source_language=source_language,
                    show_title=show_title,
                    job_id=job_id
                )
            except (ProviderUnavailableError, ProviderConfigurationError, DailyQuotaExhaustedError, RequestBudgetExhaustedError):
                raise
            except Exception as e:
                logger.warning(f"Alignment confirmation call failed: {e}")
                conf_res = {}

            c_verdict = conf_res.get("alignment_verdict", "UNCERTAIN")
            c_conf = conf_res.get("confidence", "LOW")

            if c_verdict == "ALIGNED" and c_conf in {"HIGH", "MEDIUM"}:
                if job_id:
                    append_job_log(
                        job_id,
                        f"Alignment Confirmation: Incident cues {inc.start_idx + 1}-{inc.end_idx + 1} verified ALIGNED ({c_conf}). Discarding false-positive finding."
                    )
                inc.state = IncidentState.VERIFIED
                continue
            elif c_verdict in {"SHIFT_PLUS_1", "SHIFT_MINUS_1", "MERGED", "COMPLEX_SHIFT"}:
                inc.verdict = c_verdict
                inc.confidence = c_conf
                inc.confirmation_required = False
                inc.state = IncidentState.CONFIRMED
                confirmed_incidents.append(inc)
                if job_id:
                    append_job_log(
                        job_id,
                        f"Alignment Confirmation: Incident cues {inc.start_idx + 1}-{inc.end_idx + 1} confirmed {c_verdict} ({c_conf}). Proceeding to repair."
                    )
            else:
                if inc.confidence == "HIGH" and inc.verdict in {"SHIFT_PLUS_1", "SHIFT_MINUS_1", "MERGED"}:
                    inc.state = IncidentState.CONFIRMED
                    confirmed_incidents.append(inc)
                else:
                    inc.state = IncidentState.VERIFIED
                    if job_id:
                        append_job_log(
                            job_id,
                            f"Alignment Confirmation: Incident cues {inc.start_idx + 1}-{inc.end_idx + 1} unconfirmed ({c_verdict}, {c_conf}). Skipping mutation to prevent false positive."
                        )

        if not confirmed_incidents:
            return True

        confirmed_incidents.sort(key=lambda x: (x.end_idx - x.start_idx), reverse=True)
        active_repair_targets = confirmed_incidents

        all_success = True

        for inc in active_repair_targets:
            repaired_this_incident = False
            inc.state = IncidentState.REPAIRING

            while inc.repair_attempts < 2:
                inc.repair_attempts += 1
                attempt = inc.repair_attempts

                if attempt == 1:
                    r_start = inc.start_idx
                    r_end = inc.end_idx
                else:
                    r_start = max(0, inc.start_idx - 2)
                    r_end = min(len(subs) - 1, inc.end_idx + 2)

                segment_size = 25
                repair_segments = []
                curr_s = r_start
                while curr_s <= r_end:
                    curr_e = min(r_end, curr_s + segment_size - 1)
                    repair_segments.append((curr_s, curr_e))
                    curr_s = curr_e + 1

                candidate_patch: Dict[int, str] = {}
                segment_failed = False

                for seg_start, seg_end in repair_segments:
                    seg_ids = [j + 1 for j in range(seg_start, seg_end + 1)]
                    ctx_start = max(0, seg_start - 2)
                    ctx_end = min(len(subs), seg_end + 3)

                    source_ctx = [{"id": j + 1, "text": subs[j].content.replace(chr(10), " ")} for j in range(ctx_start, ctx_end)]
                    target_ctx = [{"id": j + 1, "text": translated_subs[j].content.replace(chr(10), " ")} for j in range(ctx_start, ctx_end)]
                    expected_items = [{"id": j + 1, "text": subs[j].content} for j in range(seg_start, seg_end + 1)]

                    if job_id:
                        append_job_log(
                            job_id,
                            f"Alignment Repair: Attempt {attempt}/2 for cues {seg_start + 1}-{seg_end + 1} ({inc.verdict})"
                        )

                    try:
                        repair_results = await self.translator.repair_alignment_region(
                            repair_cue_ids=seg_ids,
                            source_context_items=source_ctx,
                            target_context_items=target_ctx,
                            target_language=target_language,
                            source_language=source_language,
                            show_title=show_title,
                            verdict=inc.verdict,
                            details=inc.details,
                            job_id=job_id
                        )
                    except Exception as e:
                        logger.warning(f"Alignment repair call attempt {attempt} failed: {e}")
                        segment_failed = True
                        break

                    valid_map, report = validate_recovery_batch_results(expected_items, repair_results)
                    if not report.get("is_clean") or len(valid_map) != len(expected_items):
                        if job_id:
                            append_job_log(
                                job_id,
                                f"Alignment Repair: Validation failed for cues {seg_start + 1}-{seg_end + 1} (missing: {report.get('missing_ids')}, unknown: {report.get('unknown_ids')})"
                            )
                        segment_failed = True
                        break

                    candidate_patch.update(valid_map)

                if segment_failed:
                    continue

                candidate_subs = [
                    srt.Subtitle(s.index, s.start, s.end, s.content)
                    for s in translated_subs
                ]
                for cue_id, new_text in candidate_patch.items():
                    c_idx = cue_id - 1
                    if 0 <= c_idx < len(candidate_subs):
                        candidate_subs[c_idx].content = new_text

                v_start = max(0, r_start - 1)
                v_end = min(len(subs), r_end + 2)

                verify_source = [{"id": j + 1, "text": subs[j].content.replace(chr(10), " ")} for j in range(v_start, v_end)]
                verify_target = [{"id": j + 1, "text": candidate_subs[j].content.replace(chr(10), " ")} for j in range(v_start, v_end)]

                try:
                    verify_res = await self.translator.audit_cue_alignment_window(
                        verify_source,
                        verify_target,
                        target_language=target_language,
                        source_language=source_language,
                        show_title=show_title,
                        job_id=job_id
                    )
                except (ProviderUnavailableError, ProviderConfigurationError, DailyQuotaExhaustedError, RequestBudgetExhaustedError):
                    raise
                except Exception as e:
                    logger.warning(f"Alignment repair local verify call failed: {e}")
                    verify_res = {}

                verdict = verify_res.get("alignment_verdict", "UNCERTAIN")
                confidence = verify_res.get("confidence", "LOW")

                if verdict != "ALIGNED" and (v_start < r_start or v_end > r_end + 1):
                    core_source = [{"id": j + 1, "text": subs[j].content.replace(chr(10), " ")} for j in range(r_start, r_end + 1)]
                    core_target = [{"id": j + 1, "text": candidate_subs[j].content.replace(chr(10), " ")} for j in range(r_start, r_end + 1)]
                    try:
                        core_verify = await self.translator.audit_cue_alignment_window(
                            core_source,
                            core_target,
                            target_language=target_language,
                            source_language=source_language,
                            show_title=show_title,
                            job_id=job_id
                        )
                        c_verdict = core_verify.get("alignment_verdict", "UNCERTAIN")
                        c_conf = core_verify.get("confidence", "LOW")
                        if c_verdict == "ALIGNED" and c_conf in {"HIGH", "MEDIUM"}:
                            left_safe = (v_start >= r_start)
                            if not left_safe:
                                if incident_tracker is not None and any(other is not inc and other.start_idx <= v_start <= other.end_idx for other in incident_tracker._incidents.values()):
                                    left_safe = True
                                else:
                                    left_src = [{"id": j + 1, "text": subs[j].content.replace(chr(10), " ")} for j in range(v_start, r_end + 1)]
                                    left_tgt = [{"id": j + 1, "text": candidate_subs[j].content.replace(chr(10), " ")} for j in range(v_start, r_end + 1)]
                                    left_v = await self.translator.audit_cue_alignment_window(left_src, left_tgt, target_language=target_language, source_language=source_language, show_title=show_title, job_id=job_id)
                                    if left_v.get("alignment_verdict") == "ALIGNED" and left_v.get("confidence") in {"HIGH", "MEDIUM"}:
                                        left_safe = True

                            right_safe = (v_end - 1 <= r_end)
                            if not right_safe:
                                if incident_tracker is not None and any(other is not inc and other.start_idx <= (v_end - 1) <= other.end_idx for other in incident_tracker._incidents.values()):
                                    right_safe = True
                                else:
                                    right_src = [{"id": j + 1, "text": subs[j].content.replace(chr(10), " ")} for j in range(r_start, v_end)]
                                    right_tgt = [{"id": j + 1, "text": candidate_subs[j].content.replace(chr(10), " ")} for j in range(r_start, v_end)]
                                    right_v = await self.translator.audit_cue_alignment_window(right_src, right_tgt, target_language=target_language, source_language=source_language, show_title=show_title, job_id=job_id)
                                    if right_v.get("alignment_verdict") == "ALIGNED" and right_v.get("confidence") in {"HIGH", "MEDIUM"}:
                                        right_safe = True

                            if left_safe and right_safe:
                                verdict = c_verdict
                                confidence = c_conf
                                if job_id:
                                    append_job_log(
                                        job_id,
                                        f"Alignment Repair: Core region cues {r_start + 1}-{r_end + 1} and boundaries verified safe ({c_conf})."
                                    )
                    except (ProviderUnavailableError, ProviderConfigurationError, DailyQuotaExhaustedError, RequestBudgetExhaustedError):
                        raise
                    except Exception as e:
                        logger.warning(f"Core alignment verify call failed: {e}")

                if verdict == "ALIGNED" and confidence in {"HIGH", "MEDIUM"}:
                    if job_id:
                        append_job_log(
                            job_id,
                            f"Alignment Repair: Local verify PASSED (ALIGNED, {confidence}). Atomically committing repair for cues {r_start + 1}-{r_end + 1}."
                        )
                    for cue_id, new_text in candidate_patch.items():
                        c_idx = cue_id - 1
                        if apply_mutation_fn:
                            apply_mutation_fn(c_idx, new_text)
                        elif 0 <= c_idx < len(translated_subs):
                            translated_subs[c_idx].content = new_text

                    repaired_this_incident = True
                    inc.state = IncidentState.REPAIRED
                    break
                else:
                    if job_id:
                        append_job_log(
                            job_id,
                            f"Alignment Repair: Local verify returned {verdict} ({confidence}) for cues {r_start + 1}-{r_end + 1}. Discarding candidate clone."
                        )

            if not repaired_this_incident:
                all_success = False
                inc.state = IncidentState.FAILED_REPAIR
                if job_id:
                    append_job_log(
                        job_id,
                        f"Alignment Repair: Bounded attempts exhausted for cues {inc.start_idx + 1}-{inc.end_idx + 1}. Marked FAILED_REPAIR."
                    )

        return all_success

    async def _repair_semantic_alignment_regions(
        self,
        subs: List[srt.Subtitle],
        translated_subs: List[srt.Subtitle],
        regions: List[AlignmentRegion],
        target_language: str,
        source_language: str,
        show_title: str = "",
        job_id: Optional[int] = None,
        apply_mutation_fn: Optional[Any] = None,
        incident_tracker: Optional[SemanticIncidentTracker] = None,
    ) -> bool:
        """
        Backward-compatible wrapper converting AlignmentRegion to AlignmentIncident.
        """
        if not regions:
            return True
        incidents = [
            AlignmentIncident(
                start_idx=r.start_idx,
                end_idx=r.end_idx,
                verdict=r.verdict,
                confidence=r.confidence,
                supporting_findings=[{"start_idx": r.start_idx, "end_idx": r.end_idx, "verdict": r.verdict, "confidence": r.confidence, "details": r.details}],
                details=r.details,
                confirmation_required=False
            )
            for r in regions
        ]
        return await self._repair_semantic_alignment_incidents(
            subs=subs,
            translated_subs=translated_subs,
            incidents=incidents,
            target_language=target_language,
            source_language=source_language,
            show_title=show_title,
            job_id=job_id,
            apply_mutation_fn=apply_mutation_fn,
            incident_tracker=incident_tracker
        )

    async def process_video_file(
        self,
        video_path: str,
        wait_seconds: Optional[int] = None,
        event_source: str = "MANUAL",
        title: Optional[str] = None,
        force_retranslate: bool = False,
        job_id: Optional[int] = None,
        series_title: Optional[str] = None
    ) -> Dict[str, Any]:
        # Bug #17: Prevent duplicate processing of the same video
        async with self._video_lock:
            norm_path = os.path.normpath(video_path)
            if norm_path in self._active_video_paths:
                logger.warning(f"Skipping duplicate request for {norm_path} — already being processed")
                if job_id:
                    # Job was already claimed but we can't process it. Revert to RECOVERING to try again later.
                    from datetime import datetime, timezone, timedelta
                    next_retry = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
                    update_job(job_id, status="RECOVERING", next_retry_at=next_retry)
                return {"status": "skipped", "reason": "already_processing", "video_path": norm_path}
            self._active_video_paths.add(norm_path)

        if job_id is None:
            job_id = create_job(video_path=video_path, event_source=event_source, title=title)

        current_task = asyncio.current_task()
        if current_task:
            self._active_tasks[job_id] = current_task

        try:
            return await self._execute_process_video(
                job_id=job_id,
                video_path=video_path,
                wait_seconds=wait_seconds,
                event_source=event_source,
                force_retranslate=force_retranslate,
                title=title,
                series_title=series_title
            )
        except asyncio.CancelledError:
            logger.info(f"Job {job_id} was cancelled by user.")
            base_path, _ = os.path.splitext(video_path)
            temp_extracted_srt = f"{base_path}.temp_extracted.en.srt"
            if os.path.exists(temp_extracted_srt):
                try: os.remove(temp_extracted_srt)
                except: pass
            update_job(job_id, status="CANCELLED", error_message="Cancelled by user")
            append_job_log(job_id, "Job was cancelled by user. Stopped AI translation.")
            raise
        except Exception as exc:
            logger.exception(f"Unhandled exception in pipeline task for job {job_id}: {exc}")
            # Outer safety boundary: ensure job does not remain in an unowned active status without a scheduled resume
            try:
                from app.core.db import get_job_by_id, ACTIVE_JOB_STATUSES
                job_data = get_job_by_id(job_id)
                if job_data:
                    curr_status = job_data.get("status")
                    if curr_status in ACTIVE_JOB_STATUSES and not job_data.get("next_retry_at"):
                        append_job_log(job_id, f"CRITICAL: Pipeline task crashed unexpectedly: {exc}")
                        update_job(
                            job_id,
                            status="FAILED",
                            error_message=f"Unhandled pipeline crash: {exc}",
                            last_error=str(exc)
                        )
            except Exception as db_err:
                logger.error(f"Failed to update crashed job {job_id} status in outer boundary: {db_err}")
            raise
        finally:
            self._active_tasks.pop(job_id, None)
            async with self._video_lock:
                self._active_video_paths.discard(os.path.normpath(video_path))

    async def _execute_process_video(
        self,
        job_id: int,
        video_path: str,
        wait_seconds: Optional[int] = None,
        event_source: str = "MANUAL",
        force_retranslate: bool = False,
        title: Optional[str] = None,
        series_title: Optional[str] = None
    ) -> Dict[str, Any]:
        sem = self._get_semaphore()
        async with sem:
            return await self._run_pipeline_logic(
                job_id=job_id,
                video_path=video_path,
                wait_seconds=wait_seconds,
                event_source=event_source,
                force_retranslate=force_retranslate,
                title=title,
                series_title=series_title
            )

    async def _maybe_notify_jellyfin(self):
        """Bug #21: Only notify Jellyfin if the setting is enabled."""
        if get_setting("notify_jellyfin", "true").lower() == "true":
            await notify_jellyfin_library_refresh()

    async def _maybe_notify_plex(self, published_path: Optional[str] = None):
        """Only notify Plex if the setting is enabled."""
        if get_setting("notify_plex", "false").lower() == "true":
            await notify_plex_library_refresh(published_path)

    async def _notify_media_servers(self, published_path: Optional[str] = None):
        """Notifies every configured media server that new subtitles are available."""
        await self._maybe_notify_jellyfin()
        await self._maybe_notify_plex(published_path)

    async def _run_pipeline_logic(
        self,
        job_id: int,
        video_path: str,
        wait_seconds: Optional[int] = None,
        event_source: str = "MANUAL",
        force_retranslate: bool = False,
        title: Optional[str] = None,
        series_title: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs core pipeline execution logic. Handles ALREADY EXISTS / already_exists check and failure safety."""
        try:
            return await self._run_pipeline_logic_impl(
                job_id=job_id,
                video_path=video_path,
                wait_seconds=wait_seconds,
                event_source=event_source,
                force_retranslate=force_retranslate,
                title=title,
                series_title=series_title
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(f"Unhandled exception in pipeline logic for job {job_id}: {exc}")
            try:
                from app.core.db import get_job_by_id, ACTIVE_JOB_STATUSES
                job_data = get_job_by_id(job_id)
                if job_data:
                    curr_status = job_data.get("status")
                    if curr_status in ACTIVE_JOB_STATUSES and not job_data.get("next_retry_at"):
                        append_job_log(job_id, f"CRITICAL: Pipeline logic failed unexpectedly: {exc}")
                        update_job(
                            job_id,
                            status="FAILED",
                            error_message=f"Pipeline error: {exc}",
                            last_error=str(exc)
                        )
            except Exception as db_err:
                logger.error(f"Failed to update crashed job {job_id} in pipeline wrapper: {db_err}")
            return {"status": "error", "error": str(exc), "job_id": job_id}

    async def _run_pipeline_logic_impl(
        self,
        job_id: int,
        video_path: str,
        wait_seconds: Optional[int] = None,
        event_source: str = "MANUAL",
        force_retranslate: bool = False,
        title: Optional[str] = None,
        series_title: Optional[str] = None
    ) -> Dict[str, Any]:
        from datetime import datetime, timezone, timedelta

        enable_bazarr = get_setting("enable_bazarr_check", "true").lower() == "true"
        extract_target_embedded = get_setting("extract_target_embedded", "true").lower() == "true"
        extract_source_embedded = get_setting("extract_source_embedded", "true").lower() == "true"
        # original_language_guard is read from DB for backward compat but NOT used
        # as a runtime blocker in v2.3.43+. Audio lang is a source-prioritisation signal only.
        auto_repair_unhealthy = get_setting("auto_repair_unhealthy", "true").lower() == "true"
        strict_sync_lock = get_setting("strict_sync_lock", "true").lower() == "true"
        effective_tm_key = series_title or (title.split(" - S")[0] if title and " - S" in title else title)
        bazarr_url = get_setting("bazarr_url", "http://bazarr:6767").rstrip("/")
        bazarr_api_key = get_setting("bazarr_api_key", "")
        source_search_deadline = float(get_setting("source_search_deadline_seconds", "45"))
        source_poll_interval   = float(get_setting("source_poll_interval_seconds", "3"))

        start_time = time.time()
        job_data = get_job_by_id(job_id) if job_id else None
        prev_dur = float(job_data.get("duration_seconds") or 0.0) if job_data else 0.0
        # Sentinel vars so exception handlers don't NameError if raised before assignment
        source_subtitle = None
        temp_extracted_srt = f"{os.path.splitext(video_path)[0]}.temp_src.srt"
        update_job(job_id, status="TRANSLATING")
        append_job_log(job_id, f"Processing file: {video_path}")

        if not os.path.exists(video_path):
            err = f"File not found: {video_path}"
            append_job_log(job_id, f"ERROR: {err}")
            update_job(job_id, status="FAILED", error_message=err, duration_seconds=round(prev_dur + (time.time() - start_time), 2))
            return {"status": "error", "message": err, "job_id": job_id}

        target_languages = self.get_configured_languages()
        if not target_languages:
            append_job_log(job_id, "No target language configured.")
            update_job(job_id, status="ACTION_REQUIRED")
            return {"status": "action_required", "reason": "no_target_languages"}

        base_path, _ = os.path.splitext(video_path)

        # -------------------------------------------------------------
        # Phase 1: Authoritative Target Acquisition Priority
        #
        # Order:
        #   1. Embedded TARGET in video (First priority)
        #   2. External trusted target on disk (Second priority)
        #   3. Bazarr target search (Third priority)
        #   4. AI translation (Fourth priority)
        # -------------------------------------------------------------

        # Track which languages still need translation
        langs_needing_translation = []

        # Single efficient container probe per job (cached for target, source, and audio inspection)
        container_tracks: Optional[Dict[str, Any]] = None
        t_probe_ms = 0.0
        if extract_target_embedded or extract_source_embedded:
            t_probe_start = time.perf_counter()
            try:
                container_tracks = await asyncio.to_thread(inspect_mkv_tracks, video_path)
            except Exception as e:
                logger.warning(f"Failed to probe container tracks for {video_path}: {e}")
                container_tracks = {"subtitles": [], "audio": []}
            t_probe_ms = round((time.perf_counter() - t_probe_start) * 1000, 1)

        # Audio detection — SOURCE PRIORITISATION SIGNAL only (never blocks translation).
        primary_audio_lang = "und"
        _atracks = (container_tracks or {}).get("audio", [])
        for _at in _atracks:
            _att = (_at.get("title") or "").lower()
            if any(x in _att for x in ["commentary", "director", "description"]):
                continue
            if _at.get("default") and not _at.get("forced"):
                primary_audio_lang = _at.get("language", "und").lower()
                break
        if primary_audio_lang == "und" and _atracks:
            for _at in _atracks:
                _att = (_at.get("title") or "").lower()
                if not _at.get("forced") and not any(x in _att for x in ["commentary", "director"]):
                    primary_audio_lang = _at.get("language", "und").lower()
                    break
        append_job_log(job_id,
            f"Audio signal: {normalize_language_code(primary_audio_lang, default=primary_audio_lang).upper()}"
            f" — source prioritisation only, never blocks translation")

        published_embedded_target = False
        embedded_target_satisfied = False
        trust_engine = SubtitleTrustEngine()
        target_resolutions: Dict[str, TargetResolution] = {}
        langs_needing_translation = []

        # Capture pre-trigger baseline snapshots of external targets to guarantee truthful provenance
        initial_target_snapshots: Dict[str, TargetSnapshot] = {}
        initial_target_paths: Dict[str, Optional[str]] = {}
        for _t_info in target_languages:
            _t_code = normalize_language_code(_t_info["code"])
            _t_ex = find_external_subtitle(video_path, _t_code)
            initial_target_paths[_t_code] = _t_ex
            if _t_ex:
                initial_target_snapshots[_t_code] = capture_target_snapshot(_t_ex)
            else:
                initial_target_snapshots[_t_code] = TargetSnapshot(path="", exists=False, size=0, mtime_ns=0)

        if not force_retranslate:
            for lang_info in target_languages:
                lang_code = lang_info["code"]
                lang_name = lang_info["name"]
                target_output_path = f"{base_path}.{lang_code}.srt"

                # 1. PRIORITY 1: Existing trusted external subtitle on disk
                existing_target = initial_target_paths.get(normalize_language_code(lang_code))
                if existing_target:
                    _init_tres = await trust_engine.evaluate_candidate(
                        video_path=video_path,
                        candidate_path=existing_target,
                        target_lang=lang_code,
                        origin=CandidateOrigin.EXTERNAL,
                        container_tracks=container_tracks,
                        primary_audio_lang=primary_audio_lang,
                        job_id=job_id,
                        auto_repair=auto_repair_unhealthy,
                        allow_ai_audit=False,
                    )
                    if _init_tres.passed:
                        append_job_log(job_id, f"Verified healthy {lang_name} subtitle already exists: {os.path.basename(existing_target)} (score={_init_tres.score}/100). Skipping.")
                        target_resolutions[lang_code] = TargetResolution(
                            satisfied=True,
                            origin=CandidateOrigin.EXTERNAL,
                            path=existing_target,
                            materialized=True,
                            reason=f"Verified healthy {lang_name} subtitle already exists: {os.path.basename(existing_target)}",
                            trust_result=_init_tres,
                        )
                        continue
                    elif _init_tres.decision == TrustDecision.UNKNOWN:
                        append_job_log(job_id, f"{lang_name} target detected ({os.path.basename(existing_target)}); awaiting Trust verification. Resolving source...")
                    else:
                        append_job_log(job_id, f"Existing {lang_name} subtitle found but rejected by Trust Engine ({_init_tres.decision.value}: {'; '.join(_init_tres.reasons)}). Continuing with Babel fallback.")

                # 2. PRIORITY 2: Embedded TARGET in video container
                if extract_target_embedded:
                    sub_tracks = (container_tracks or {}).get("subtitles", [])
                    lang_obj = _get_language(lang_code)
                    lang_prefixes = lang_obj.aliases if lang_obj else [lang_code.lower()]
                    matching_embedded_tracks = []
                    for tr in sub_tracks:
                        tl = (tr.get("language") or "").lower()
                        tc = (tr.get("codec") or "")
                        tt = (tr.get("title") or "").lower()
                        tf = tr.get("forced", False)
                        is_text = any(
                            tc.lower() in x.lower() or x.lower() in tc.lower()
                            for x in [
                                "SubRip/SRT", "S_TEXT/UTF8", "S_TEXT/ASS", "S_TEXT/SSA",
                                "S_TEXT/WEBVTT", "SubStationAlpha", "WebVTT", "srt", "text",
                                "ass", "ssa", "vtt", "utf", "substation"
                            ]
                        )
                        if any(lp == tl or tl.startswith(lp) for lp in lang_prefixes) and is_text:
                            if not tf and not any(kw in tt for kw in ["forced", "signs", "songs", "foreign", "parts", "descriptive", "commentary", "director", "description"]):
                                matching_embedded_tracks.append(tr)

                    is_empty_probe = (not sub_tracks and not (container_tracks or {}).get("audio") and (container_tracks or {}).get("duration", 0.0) == 0.0)
                    if matching_embedded_tracks or (container_tracks is None and not existing_target) or (is_empty_probe and not existing_target):
                        first_cand = matching_embedded_tracks[0] if matching_embedded_tracks else {}
                        codec_display = first_cand.get("codec", "unknown")
                        track_desc = f"track {first_cand.get('id')}" if first_cand.get('id') is not None else "track"

                        if matching_embedded_tracks:
                            append_job_log(job_id, f"Embedded target scan: {lang_name} {codec_display} candidate found ({track_desc})")

                        materialize_embedded_target = get_setting("materialize_embedded_target", "false").lower() == "true"
                        if not materialize_embedded_target and matching_embedded_tracks:
                            # Part 2: Satisfy WITHOUT materialization / WITHOUT mkvextract!
                            append_job_log(job_id, f"Embedded {lang_name} target satisfies language via container metadata ({codec_display}, {track_desc}). Materialization skipped.")
                            target_resolutions[lang_code] = TargetResolution(
                                satisfied=True,
                                origin=CandidateOrigin.EMBEDDED,
                                path=None,
                                materialized=False,
                                reason=f"Embedded {lang_name} target satisfies language via container metadata ({codec_display}, {track_desc})",
                            )
                            embedded_target_satisfied = True
                            continue
                        else:
                            parent_dir = os.path.dirname(target_output_path) or "."
                            temp_target_path = os.path.join(
                                parent_dir,
                                f".{os.path.basename(target_output_path)}.tmp_embed.{uuid.uuid4().hex}.srt"
                            )
                            published = False
                            t_emb_start = time.perf_counter()
                            try:
                                extracted = await asyncio.to_thread(
                                    _safe_extract_embedded_srt,
                                    video_path,
                                    temp_target_path,
                                    preferred_lang=lang_code,
                                    tracks_info=container_tracks
                                )
                                t_emb_duration = time.perf_counter() - t_emb_start
                                if extracted and os.path.exists(temp_target_path) and os.path.getsize(temp_target_path) > 0:
                                    append_job_log(job_id, f"Embedded target extraction: {track_desc} → SRT ({t_emb_duration:.2f}s)")
                                    # Always validate extracted embedded target via Trust Engine
                                    _emb_tres = await trust_engine.evaluate_candidate(
                                        video_path=video_path,
                                        candidate_path=temp_target_path,
                                        target_lang=lang_code,
                                        origin=CandidateOrigin.EMBEDDED,
                                        container_tracks=container_tracks,
                                        primary_audio_lang=primary_audio_lang,
                                        expected_intent=SubtitleIntent.FULL,
                                        job_id=job_id,
                                        auto_repair=auto_repair_unhealthy,
                                        allow_ai_audit=False,
                                    )
                                    if _emb_tres.passed:
                                        append_job_log(job_id, f"Embedded target validation: PASS (score={_emb_tres.score}/100)")
                                        with open(temp_target_path, "r", encoding="utf-8-sig", errors="ignore") as _ef:
                                            _emb_text = _ef.read()
                                        try:
                                            _emb_cues = list(srt.parse(_emb_text))
                                        except Exception:
                                            _emb_cues = []
                                        pub_res = await _publish_subtitle_with_trust_gate(
                                            video_path=video_path,
                                            target_output_path=target_output_path,
                                            lang_code=lang_code,
                                            translated_srt_text=_emb_text,
                                            expected_cue_count=len(_emb_cues),
                                            source_subtitle=None,
                                            container_tracks=container_tracks,
                                            primary_audio_lang=primary_audio_lang,
                                            force_retranslate=force_retranslate,
                                            job_id=job_id,
                                            auto_repair=auto_repair_unhealthy,
                                        )
                                        if pub_res.get("published"):
                                            published = True
                                            published_embedded_target = True
                                            embedded_target_satisfied = True
                                            target_resolutions[lang_code] = TargetResolution(
                                                satisfied=True,
                                                origin=CandidateOrigin.EMBEDDED,
                                                path=target_output_path,
                                                materialized=True,
                                                reason=f"Extracted healthy embedded {lang_name} track to {os.path.basename(target_output_path)} (score={_emb_tres.score}/100).",
                                                trust_result=_emb_tres,
                                            )
                                            append_job_log(job_id, f"Embedded {lang_name} target selected")
                                            append_job_log(job_id, f"Extracted healthy embedded {lang_name} track to {os.path.basename(target_output_path)} (score={_emb_tres.score}/100).")
                                            continue
                                        elif pub_res.get("skipped"):
                                            published = True
                                            embedded_target_satisfied = True
                                            target_resolutions[lang_code] = TargetResolution(
                                                satisfied=True,
                                                origin=CandidateOrigin.EXTERNAL,
                                                path=target_output_path,
                                                materialized=True,
                                                reason=f"External healthy {lang_name} subtitle appeared during embedded extraction.",
                                            )
                                            append_job_log(job_id, f"External healthy {lang_name} subtitle appeared during embedded extraction. Preserving external subtitle.")
                                            continue
                                        else:
                                            embedded_target_satisfied = True
                                            target_resolutions[lang_code] = TargetResolution(
                                                satisfied=True,
                                                origin=CandidateOrigin.EMBEDDED,
                                                path=None,
                                                materialized=False,
                                                reason=f"Embedded {lang_name} target validated (PASS, score={_emb_tres.score}/100). External publication deferred ({pub_res.get('reason')}). Target language satisfied.",
                                                trust_result=_emb_tres,
                                            )
                                            append_job_log(job_id, f"Embedded {lang_name} target validated (PASS, score={_emb_tres.score}/100). External publication deferred ({pub_res.get('reason')}). Target language satisfied.")
                                            continue
                                    else:
                                        append_job_log(job_id, f"Embedded target validation: FAIL ({_emb_tres.decision.value}: {'; '.join(_emb_tres.reasons)}).")
                                elif matching_embedded_tracks:
                                    append_job_log(job_id, f"Embedded target extraction failed: {codec_display}")
                            finally:
                                if not published and os.path.exists(temp_target_path):
                                    try:
                                        os.remove(temp_target_path)
                                    except Exception:
                                        pass

                # 3. If neither external nor embedded satisfied the language, it needs Bazarr/AI
                langs_needing_translation.append(lang_info)
        else:
            langs_needing_translation = list(target_languages)

        # If all languages are covered, we're done
        if not langs_needing_translation:
            duration = round(time.time() - start_time, 2)
            has_embedded = any(r.origin == CandidateOrigin.EMBEDDED for r in target_resolutions.values())
            has_external = any(r.origin == CandidateOrigin.EXTERNAL for r in target_resolutions.values())
            has_materialized_embedded = any(r.origin == CandidateOrigin.EMBEDDED and r.materialized for r in target_resolutions.values())

            if has_embedded:
                append_job_log(job_id, "Bazarr skipped — embedded target satisfied language")
                append_job_log(job_id, "AI skipped")
                append_job_log(job_id, "AI calls: 0")

            if has_embedded and not has_external:
                if has_materialized_embedded:
                    reason_str = "Embedded target extracted and published"
                else:
                    first_emb_lang = next((l["name"] for l in target_languages if l["code"] in target_resolutions and target_resolutions[l["code"]].origin == CandidateOrigin.EMBEDDED), "")
                    reason_str = f"Embedded target satisfies language ({first_emb_lang})" if len(target_resolutions) == 1 else "Embedded targets satisfy all languages"
            elif has_embedded and has_external:
                reason_str = "All target subtitles already exist (embedded + external)"
            else:
                reason_str = "All target subtitles already exist"

            update_job(job_id, status="ALREADY EXISTS", reason=reason_str, duration_seconds=duration)
            if has_materialized_embedded:
                await self._notify_media_servers(video_path)
            return {"status": "skipped", "reason": "already_exists", "job_id": job_id}
        # Design:
        #   1. Trigger Bazarr TARGET searches immediately (fire-and-forget HTTP PATCH).
        #   2. Run source resolution concurrently with a lightweight target-presence
        #      poller.  If a healthy target appears on disk while source extraction is
        #      still running (e.g. embedded MKV extraction taking 20-100s), the poller
        #      task fires, source resolution is cancelled, and we skip straight to the
        #      BAZARR MATCH / ALREADY EXISTS path.
        #   3. The final authoritative filesystem check (below) is always performed.
        #
        # This eliminates the La Haine / Godland pattern where Bazarr had already
        # provided the target subtitle but the job still waited 101s for embedded
        # English source extraction to complete before checking.
        prep_start_time = time.time()

        _bazarr_accepted_by_lang: Dict[str, bool] = {}
        _bazarr_correlated_by_lang: Dict[str, bool] = {}

        def is_bazarr_accepted_for_lang(lcode: str) -> bool:
            return _bazarr_accepted_by_lang.get(normalize_language_code(lcode), False)

        def is_bazarr_correlated_for_lang(lcode: str) -> bool:
            return _bazarr_correlated_by_lang.get(normalize_language_code(lcode), False)

        def get_correlated_bazarr_origin(cand_path: Optional[str], lcode: str) -> CandidateOrigin:
            """
            Infers candidate origin based on per-language Bazarr trigger state and pre-Bazarr snapshots.
            If authoritative Bazarr correlation is absent (Bazarr disabled or search not accepted),
            or if candidate generation matches the pre-Bazarr state, it is labeled EXTERNAL.
            If a new/changed generation appeared after Bazarr search was accepted for this specific language, it is labeled BAZARR.
            """
            norm_code = normalize_language_code(lcode)
            if not cand_path or not is_bazarr_accepted_for_lang(norm_code):
                return CandidateOrigin.EXTERNAL
            cand_snap = capture_target_snapshot(cand_path)
            if not cand_snap.exists:
                return CandidateOrigin.EXTERNAL
            init_snap = initial_target_snapshots.get(norm_code)
            if init_snap and init_snap.exists and init_snap.generation_id == cand_snap.generation_id:
                return CandidateOrigin.EXTERNAL
            return CandidateOrigin.BAZARR

        get_authoritative_candidate_origin = get_correlated_bazarr_origin

        def get_bazarr_provenance_for_lang(
            lcode: str,
            is_finalized: bool = False,
            is_quiescent: bool = False,
            poll_state: Optional[Any] = None,
            media_correlated: bool = False,
            candidate_snapshot: Optional[TargetSnapshot] = None,
        ) -> BazarrProvenance:
            norm_code = normalize_language_code(lcode)
            accepted = is_bazarr_accepted_for_lang(norm_code)
            init_snap = initial_target_snapshots.get(norm_code)
            return BazarrProvenance(
                video_path=video_path,
                target_lang=norm_code,
                search_accepted=accepted,
                pre_trigger_snapshot=init_snap,
                is_finalized=is_finalized,
                is_quiescent=is_quiescent,
                media_correlated=media_correlated,
                poll_state=poll_state,
                candidate_snapshot=candidate_snapshot,
            )

        # Step 1: Trigger Bazarr TARGET searches (non-blocking HTTP task)
        bazarr_tasks: List[asyncio.Task] = []
        if enable_bazarr and not force_retranslate:
            for _blinfo in langs_needing_translation:
                _blc = _blinfo["code"]
                _bln = _blinfo["name"]
                append_job_log(job_id, f"Hybrid Mode: Triggering Bazarr search for {_bln} ({_blc})...")
                async def _do_btarget(_lc=_blc, _ln=_bln):
                    try:
                        import inspect
                        sig = inspect.signature(self.trigger_bazarr_search)
                        call_kwargs = {"language": _lc}
                        if "job_id" in sig.parameters:
                            call_kwargs["job_id"] = job_id
                        if "event_source" in sig.parameters:
                            call_kwargs["event_source"] = event_source
                        _r = await self.trigger_bazarr_search(video_path, **call_kwargs)
                        if isinstance(_r, BazarrResult):
                            if _r.code == BazarrResultCode.AUTH_ERROR:
                                append_job_log(job_id, f"Bazarr auth error for {_ln}: {_r.detail}")
                            elif _r.code == BazarrResultCode.MEDIA_NOT_FOUND:
                                append_job_log(job_id, f"Bazarr: {_ln} not found in library — {_r.detail}")
                            elif _r.code == BazarrResultCode.WAITING_FOR_MEDIA:
                                append_job_log(job_id, f"Bazarr correlation: {_ln} WAITING_FOR_MEDIA (indexing in progress)")
                            elif _r.code == BazarrResultCode.TEMPORARY_ERROR:
                                append_job_log(job_id, f"Bazarr transient error for {_ln}: {_r.detail}")
                            elif _r.was_accepted:
                                append_job_log(job_id, f"Bazarr target search for {_ln} accepted.")
                                _bazarr_accepted_by_lang[normalize_language_code(_lc)] = True
                                if getattr(_r, "media_correlated", False):
                                    _bazarr_correlated_by_lang[normalize_language_code(_lc)] = True
                        elif _r is True:
                            append_job_log(job_id, f"Bazarr target search for {_ln} accepted.")
                            _bazarr_accepted_by_lang[normalize_language_code(_lc)] = True
                            _bazarr_correlated_by_lang[normalize_language_code(_lc)] = True
                    except asyncio.CancelledError:
                        raise
                    except Exception as _e:
                        logger.warning(f"Bazarr target search {_lc}: {_e}")
                _bt = asyncio.create_task(_do_btarget(), name=f"bazarr_tgt_{job_id}_{_blc}")
                bazarr_tasks.append(_bt)

        # Step 2: Source Resolution — runs concurrently with target polling.
        append_job_log(job_id, "Source Resolver: scanning for usable source subtitle...")
        t_ext_start = time.perf_counter()
        _tlcodes = [normalize_language_code(l["code"]) for l in langs_needing_translation]
        _resolver = SourceResolver(
            video_path=video_path,
            container_tracks=container_tracks,
            primary_audio_lang=primary_audio_lang,
            target_languages=_tlcodes,
            bazarr_url=bazarr_url,
            bazarr_api_key=bazarr_api_key,
            enable_bazarr=enable_bazarr,
            extract_source_embedded=extract_source_embedded,
            source_search_deadline=source_search_deadline,
            source_poll_interval=source_poll_interval,
            job_id=job_id,
            event_source=event_source,
            # Pass the function from THIS module's namespace so test mocks
            # on app.services.pipeline.find_external_subtitle are respected.
            find_external_subtitle_fn=find_external_subtitle,
        )
        _source_task = asyncio.create_task(_resolver.resolve(), name=f"source_resolve_{job_id}")

        # Step 3: Concurrent target-presence poller.
        # Polls the filesystem every 0.5s while source is resolving.
        # Exits immediately if Bazarr is disabled or force_retranslate is set.
        # Stays active as long as source extraction is running (up to extraction timeout + margin).
        _target_found_early = asyncio.Event()
        _target_poll_interval = 0.5  # Responsive 500ms polling while extraction is running
        _max_race_timeout = DEFAULT_EXTRACTION_TIMEOUT + 30.0
        _poll_deadline = time.monotonic() + _max_race_timeout

        async def _poll_target_arrival():
            """Lightweight poller: check if ALL langs now have a healthy target on disk.
            Exits immediately if Bazarr is disabled; otherwise polls every _target_poll_interval
            seconds until a target is found or source resolution finishes.
            """
            if not enable_bazarr or force_retranslate or not langs_needing_translation:
                return
            while not _target_found_early.is_set():
                if _source_task.done():
                    return
                if time.monotonic() >= _poll_deadline:
                    return

                # Fast disk presence check: verify if candidates exist for all target languages
                all_found_on_disk = True
                for _pli in langs_needing_translation:
                    if not find_external_subtitle(video_path, _pli["code"]):
                        all_found_on_disk = False
                        break

                if not all_found_on_disk:
                    try:
                        await _real_asyncio_sleep(_target_poll_interval)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        return
                    continue

                # Files exist on disk — check Bazarr system jobs if reachable
                poll_res = None
                if bazarr_api_key:
                    try:
                        from app.services.bazarr_coordinator import _normalize_poll_result, BazarrJobPollStatus
                        raw_poll = await bazarr_coordinator.poll_system_jobs(bazarr_url, bazarr_api_key)
                        poll_res = _normalize_poll_result(raw_poll)
                    except Exception as _e:
                        logger.warning(f"Early target poller poll_system_jobs error: {_e}")
                        poll_res = None

                all_covered = True
                for _pli in langs_needing_translation:
                    _pex = find_external_subtitle(video_path, _pli["code"])
                    if not _pex:
                        all_covered = False
                        break

                    if poll_res is not None and poll_res.status not in (BazarrJobPollStatus.UNKNOWN, None):
                        _search_j, _sync_j = bazarr_coordinator.classify_jobs_for_target(poll_res, video_path, _pli["code"])
                        if _search_j or _sync_j:
                            # Job is still active/syncing — candidate is PROVISIONAL, do not declare early win
                            all_covered = False
                            break

                    _p_stable = await wait_for_file_stability(_pex, min_stability_sec=0.12, timeout_sec=0.4, interval_sec=0.025)
                    if not _p_stable:
                        all_covered = False
                        break
                    _p_snap = capture_target_snapshot(_pex)
                    _p_poll_state = poll_res.status if (poll_res is not None and poll_res.status != BazarrJobPollStatus.UNKNOWN) else None
                    _p_prov = get_bazarr_provenance_for_lang(
                        _pli["code"],
                        is_finalized=True,
                        is_quiescent=True,
                        poll_state=_p_poll_state,
                        media_correlated=is_bazarr_correlated_for_lang(_pli["code"]),
                        candidate_snapshot=_p_snap,
                    )
                    _ptres = await trust_engine.evaluate_candidate(
                        video_path=video_path,
                        candidate_path=_pex,
                        target_lang=_pli["code"],
                        origin=get_authoritative_candidate_origin(_pex, _pli["code"]),
                        container_tracks=container_tracks,
                        primary_audio_lang=primary_audio_lang,
                        job_id=job_id,
                        auto_repair=auto_repair_unhealthy,
                        allow_ai_audit=False,
                        bazarr_provenance=_p_prov,
                    )
                    if not _ptres.passed:
                        all_covered = False
                        break
                if all_covered:
                    _target_found_early.set()
                    return

                try:
                    await _real_asyncio_sleep(_target_poll_interval)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    return

        _poll_task = asyncio.create_task(_poll_target_arrival(), name=f"target_poll_{job_id}")

        # Wait for EITHER source resolution OR early target detection to complete.
        source_subtitle = None
        _early_target_win = False
        try:
            _done, _pending = await asyncio.wait(
                {_source_task, _poll_task},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=_max_race_timeout,
            )

            # CANCELLATION INVARIANT:
            # Cancelling source resolution (_resolver.cancel() / _source_task.cancel()) is strictly
            # permitted ONLY AFTER an authoritative SubtitleTrustEngine PASS has verified the target
            # against a valid reference or same-container provenance. A reference-less or UNKNOWN
            # candidate must NEVER cancel in-flight source extraction.
            if _target_found_early.is_set() or (_poll_task in _done and _target_found_early.is_set() and not _source_task.done()):
                _resolver.cancel()
                _source_task.cancel()
                try:
                    await _source_task
                except (asyncio.CancelledError, Exception):
                    pass
                _early_target_win = True
                _target_win_elapsed = round(time.time() - prep_start_time, 2)
                append_job_log(job_id, "Target/Source race winner: Bazarr")
                append_job_log(job_id, f"Target subtitle found after {_target_win_elapsed}s")
                append_job_log(job_id, f"Embedded extraction cancelled: target subtitle won race after {_target_win_elapsed}s")
                append_job_log(job_id, "Source extraction cancelled")
                append_job_log(job_id, "AI skipped")
                append_job_log(job_id, "AI calls: 0")
                append_job_log(job_id, "Estimated AI cost: $0.00")
                append_job_log(job_id, f"Total source preparation: {_target_win_elapsed}s")
            else:
                # Source finished first (or both finished simultaneously).
                _poll_task.cancel()
                try:
                    await _poll_task
                except (asyncio.CancelledError, Exception):
                    pass
                if not _source_task.done():
                    source_subtitle = await _source_task
                elif _source_task.cancelled():
                    source_subtitle = None
                else:
                    exc = _source_task.exception()
                    if exc:
                        raise exc
                    source_subtitle = _source_task.result()

        except asyncio.CancelledError:
            # Pipeline job itself was cancelled — clean up both tasks and terminate processes
            _resolver.cancel()
            for _t in (_source_task, _poll_task):
                if not _t.done():
                    _t.cancel()
            await asyncio.gather(_source_task, _poll_task, return_exceptions=True)
            raise

        t_extract_ms = round((time.perf_counter() - t_ext_start) * 1000, 1)

        # Cancel background Bazarr target tasks — filesystem check below is authoritative
        for _btt in bazarr_tasks:
            if not _btt.done():
                _btt.cancel()
        if bazarr_tasks:
            await asyncio.gather(*bazarr_tasks, return_exceptions=True)

        # ── Final Bazarr / External target check ──────────────────────────────
        # Authoritative: check filesystem regardless of how we got here.
        # This also handles the early-target-win path from the concurrent poller.
        if enable_bazarr and not force_retranslate:
            _still_miss = []
            _resolved_origins = {}
            for _blinfo in langs_needing_translation:
                _bex = find_external_subtitle(video_path, _blinfo["code"])
                if _bex:
                    _origin = get_authoritative_candidate_origin(_bex, _blinfo["code"])
                    _b_snap = capture_target_snapshot(_bex)
                    _post_poll_status = None
                    if _early_target_win:
                        _post_poll_status = BazarrJobPollStatus.KNOWN_IDLE
                    elif bazarr_url and bazarr_api_key:
                        try:
                            _pjobs = await bazarr_coordinator.poll_system_jobs(bazarr_url, bazarr_api_key, timeout=3.0)
                            if _pjobs is not None and _pjobs.status == BazarrJobPollStatus.KNOWN_IDLE:
                                _search_j, _sync_j = bazarr_coordinator.classify_jobs_for_target(_pjobs, video_path, _blinfo["code"])
                                if not _search_j and not _sync_j:
                                    _post_poll_status = BazarrJobPollStatus.KNOWN_IDLE
                        except Exception:
                            _post_poll_status = None

                    _b_prov = get_bazarr_provenance_for_lang(
                        _blinfo["code"],
                        is_finalized=_early_target_win or (_post_poll_status == BazarrJobPollStatus.KNOWN_IDLE),
                        is_quiescent=True,
                        poll_state=_post_poll_status,
                        media_correlated=is_bazarr_correlated_for_lang(_blinfo["code"]),
                        candidate_snapshot=_b_snap,
                    )
                    _btres = await trust_engine.evaluate_candidate(
                        video_path=video_path,
                        candidate_path=_bex,
                        target_lang=_blinfo["code"],
                        origin=_origin,
                        container_tracks=container_tracks,
                        primary_audio_lang=primary_audio_lang,
                        provided_source=source_subtitle,
                        job_id=job_id,
                        auto_repair=auto_repair_unhealthy,
                        allow_ai_audit=True,
                        bazarr_provenance=_b_prov,
                    )
                    if not _btres.passed:
                        append_job_log(job_id, f"Observed {_blinfo['name']} subtitle on disk rejected by Trust Engine ({_btres.decision.value}: {'; '.join(_btres.reasons)})")
                        append_job_log(job_id, "No trusted target available — continuing with Babel fallback")
                        _still_miss.append(_blinfo)
                        continue
                    _resolved_origins[_blinfo["code"]] = _origin
                    if _btres.repair and _btres.repair.get("applied_shift_sec") is not None:
                        append_job_log(job_id, f"Safe timing repair applied: shifted timestamps by {_btres.repair['applied_shift_sec']:+.2f}s")
                        append_job_log(job_id, f"Revalidation: PASS (score={_btres.score}/100)")
                        append_job_log(job_id, "Using repaired human subtitle")
                    else:
                        _ref_log = f", ref={_btres.reference.get('language', 'none')}" if _btres.reference else ""
                        append_job_log(job_id, f"Subtitle Trust Engine: PASS (score={_btres.score}/100{_ref_log})")
                        append_job_log(job_id, "Using verified external target")
                else:
                    _still_miss.append(_blinfo)
            langs_needing_translation = _still_miss
            if not langs_needing_translation:
                _bzdur = round(time.time() - start_time, 2)
                _all_bazarr = bool(_resolved_origins) and all(orig == CandidateOrigin.BAZARR for orig in _resolved_origins.values())
                _all_external = bool(_resolved_origins) and all(orig == CandidateOrigin.EXTERNAL for orig in _resolved_origins.values())

                if _all_bazarr:
                    append_job_log(job_id, "Target/Source race winner: Bazarr")
                    append_job_log(job_id, f"Bazarr target observed: {_bzdur}s")
                    _status = "BAZARR MATCH"
                    _reason = "Bazarr found all targets"
                    _res_reason = "bazarr_downloaded"
                elif _all_external:
                    append_job_log(job_id, "Target/Source race winner: Pre-existing external target")
                    append_job_log(job_id, f"Pre-existing external target verified after source resolution: {_bzdur}s")
                    _status = "ALREADY EXISTS"
                    _reason = "Pre-existing target verified after source resolution"
                    _res_reason = "already_exists"
                elif bool(_resolved_origins) and any(orig == CandidateOrigin.BAZARR for orig in _resolved_origins.values()):
                    append_job_log(job_id, "Target/Source race winner: Mixed (Bazarr + existing)")
                    append_job_log(job_id, f"All targets resolved without AI: {_bzdur}s")
                    _status = "BAZARR MATCH"
                    _reason = "Targets resolved without AI (Bazarr + existing)"
                    _res_reason = "bazarr_downloaded"
                else:
                    append_job_log(job_id, "Target/Source race winner: Pre-existing external target")
                    append_job_log(job_id, f"Pre-existing external target verified after source resolution: {_bzdur}s")
                    _status = "ALREADY EXISTS"
                    _reason = "Pre-existing target verified after source resolution"
                    _res_reason = "already_exists"

                append_job_log(job_id, "AI skipped")
                append_job_log(job_id, "AI calls: 0")
                append_job_log(job_id, "Estimated AI cost: $0.00")
                append_job_log(job_id, f"Total source preparation: {round(time.time() - prep_start_time, 2)}s")
                update_job(job_id, status=_status, reason=_reason, duration_seconds=_bzdur, sync_diff_ms=-1, dropped_lines=0)
                if source_subtitle and source_subtitle.origin == SourceOrigin.EMBEDDED:
                    try: os.remove(source_subtitle.path)
                    except Exception: pass
                await self._notify_media_servers(video_path)
                return {"status": "skipped", "reason": _res_reason, "job_id": job_id}

        # ── SOURCE == TARGET shortcut ─────────────────────────────────────────
        # Invariant: source_language != target_language for every AI dispatch.
        # If the resolved source IS a target language, publish it directly.
        published_source_as_target = False
        if source_subtitle:
            for _stlinfo in list(langs_needing_translation):
                if source_subtitle.language == normalize_language_code(_stlinfo["code"]):
                    _stout = f"{base_path}.{_stlinfo['code']}.srt"
                    _storigin = CandidateOrigin.EMBEDDED if source_subtitle.origin == SourceOrigin.EMBEDDED else CandidateOrigin.EXTERNAL
                    _st_tres = await trust_engine.evaluate_candidate(
                        video_path=video_path,
                        candidate_path=source_subtitle.path,
                        target_lang=_stlinfo["code"],
                        origin=_storigin,
                        container_tracks=container_tracks,
                        primary_audio_lang=primary_audio_lang,
                        provided_source=source_subtitle,
                        expected_intent=SubtitleIntent.FULL,
                        job_id=job_id,
                        auto_repair=auto_repair_unhealthy,
                        allow_ai_audit=False,
                    )
                    if _st_tres.passed:
                        try:
                            with open(source_subtitle.path, "r", encoding="utf-8-sig", errors="ignore") as _sf:
                                _st_content = _sf.read()
                            try:
                                _st_cues = list(srt.parse(_st_content))
                            except Exception:
                                _st_cues = getattr(source_subtitle, "cues", None) or []
                            if _st_cues:
                                pub_res = await _publish_subtitle_with_trust_gate(
                                    video_path=video_path,
                                    target_output_path=_stout,
                                    lang_code=_stlinfo["code"],
                                    translated_srt_text=_st_content,
                                    expected_cue_count=len(_st_cues),
                                    source_subtitle=source_subtitle,
                                    container_tracks=container_tracks,
                                    primary_audio_lang=primary_audio_lang,
                                    force_retranslate=force_retranslate,
                                    job_id=job_id,
                                    auto_repair=auto_repair_unhealthy,
                                )
                                if pub_res.get("published"):
                                    published_source_as_target = True
                                    append_job_log(job_id,
                                        f"Source IS {_stlinfo['name']} (target language). "
                                        f"Published directly — no AI needed (score={_st_tres.score}/100).")
                                    langs_needing_translation.remove(_stlinfo)
                                elif pub_res.get("skipped"):
                                    append_job_log(job_id,
                                        f"External healthy {_stlinfo['name']} subtitle exists/appeared. "
                                        f"Preserved external file.")
                                    langs_needing_translation.remove(_stlinfo)
                        except Exception as _ste:
                            append_job_log(job_id, f"Failed to publish source-as-target {_stlinfo['name']}: {_ste}")
                    else:
                        append_job_log(job_id, f"Source in target language {_stlinfo['name']} rejected by Trust Engine ({_st_tres.decision.value}: {'; '.join(_st_tres.reasons)}).")

        if not langs_needing_translation:
            _stdur = round(time.time() - start_time, 2)
            append_job_log(job_id, "AI skipped")
            append_job_log(job_id, "AI calls: 0")
            append_job_log(job_id, "Estimated AI cost: $0.00")
            update_job(job_id, status="ALREADY EXISTS",
                       reason="All targets resolved without AI", duration_seconds=_stdur, sync_diff_ms=-1, dropped_lines=0)
            if published_source_as_target:
                await self._notify_media_servers(video_path)
            return {"status": "skipped", "reason": "already_exists", "job_id": job_id}

        # ── No source found → WAITING_SOURCE ─────────────────────────────────
        temp_extracted_srt = source_subtitle.path if source_subtitle else f"{base_path}.temp_src.srt"
        if not source_subtitle:
            _is_wfm = getattr(_resolver, "is_waiting_for_media", False)
            if _is_wfm:
                err = "Bazarr media indexing in progress (WAITING_FOR_MEDIA) — queued for retry"
            else:
                err = "No usable source subtitle found (no embedded, external or Bazarr source)"
            _nsdur = round(prev_dur + (time.time() - start_time), 2)
            _jd = get_job_by_id(job_id)
            _retries = _jd.get("retry_count", 0) if _jd else 0
            if _retries < 4:
                _bof = [1, 5, 15, 30][_retries]
                from datetime import datetime as _dt2, timezone as _tz2, timedelta as _td2
                _nra = (_dt2.now(_tz2.utc) + _td2(minutes=_bof)).isoformat()
                update_job(job_id, status="WAITING_SOURCE", error_message=err,
                           duration_seconds=_nsdur, retry_count=_retries + 1, next_retry_at=_nra, sync_diff_ms=-1, dropped_lines=0)
                return {"status": "waiting_source", "reason": err, "job_id": job_id}
            else:
                update_job(job_id, status="FAILED", error_message=err, duration_seconds=_nsdur, sync_diff_ms=-1, dropped_lines=0)
                return {"status": "failed", "reason": err, "job_id": job_id}

        # Log source race winner
        _src_winner_label = "Embedded source" if source_subtitle.origin == SourceOrigin.EMBEDDED else ("External source" if source_subtitle.origin == SourceOrigin.EXTERNAL else "Bazarr source")
        append_job_log(job_id, f"Target/Source race winner: {_src_winner_label}")
        append_job_log(job_id, f"Source language: {source_subtitle.language_name} ({source_subtitle.language})")
        append_job_log(job_id,
            f"Source selected: {source_subtitle.language_name} ({source_subtitle.language}) "
            f"via {source_subtitle.origin.value} [{len(source_subtitle.cues)} cues]. "
            f"Targets remaining: {[l['code'] for l in langs_needing_translation]}")

        # Lifecycle timing settings
        candidate_stability_sec = float(get_setting("bazarr_candidate_stability_seconds", get_setting("candidate_stability_sec", str(DEFAULT_CANDIDATE_STABILITY_SEC))))
        bazarr_quiescence_sec = float(get_setting("bazarr_quiescence_seconds", get_setting("bazarr_quiescence_sec", str(DEFAULT_BAZARR_QUIESCENCE_SEC))))

        # Bazarr Adaptive Coordination & Quiescence Lifecycle Check
        if enable_bazarr and not force_retranslate:
            hybrid_max_wait = float(get_setting("hybrid_bazarr_max_wait_sec", get_setting("bazarr_grace_seconds", str(DEFAULT_HYBRID_BAZARR_MAX_WAIT_SEC))))

            if hybrid_max_wait > 0 and langs_needing_translation:
                t_coord_start = time.monotonic()
                resolved_langs = []
                _c_origins = {}
                last_decision_summary = ""

                for _g_lang in list(langs_needing_translation):
                    _g_code = _g_lang["code"]
                    _c_prov_snap = initial_target_snapshots.get(normalize_language_code(_g_code))
                    _c_accepted = is_bazarr_accepted_for_lang(_g_code)
                    _c_state, _c_cand, _c_tres = await bazarr_coordinator.coordinate_target(
                        video_path=video_path,
                        target_lang=_g_code,
                        bazarr_url=bazarr_url,
                        bazarr_api_key=bazarr_api_key,
                        max_wait_seconds=hybrid_max_wait,
                        candidate_stability_sec=candidate_stability_sec,
                        quiescence_sec=bazarr_quiescence_sec,
                        container_tracks=container_tracks,
                        primary_audio_lang=primary_audio_lang,
                        provided_source=source_subtitle,
                        job_id=job_id,
                        auto_repair=auto_repair_unhealthy,
                        find_external_subtitle_fn=find_external_subtitle,
                        pre_trigger_snapshot=_c_prov_snap,
                        search_accepted=_c_accepted,
                        media_correlated=is_bazarr_correlated_for_lang(_g_code),
                    )
                    if _c_state == BazarrLifecycleState.FINALIZED_WITH_TARGET and _c_tres and _c_tres.passed:
                        resolved_langs.append(_g_lang)
                        _c_origins[_g_code] = get_authoritative_candidate_origin(_c_cand, _g_code)
                    elif _c_tres:
                        last_decision_summary = f"{_c_tres.decision.value}: {'; '.join(_c_tres.reasons)}"

                for _rl in resolved_langs:
                    langs_needing_translation.remove(_rl)

                if not langs_needing_translation:
                    _coord_elapsed = round(time.monotonic() - t_coord_start, 2)
                    _all_bazarr = bool(_c_origins) and all(orig == CandidateOrigin.BAZARR for orig in _c_origins.values())
                    _all_external = bool(_c_origins) and all(orig == CandidateOrigin.EXTERNAL for orig in _c_origins.values())

                    if _all_bazarr:
                        append_job_log(job_id, f"Bazarr coordination: {_coord_elapsed}s (quiescent and verified)")
                        append_job_log(job_id, "Target subtitle verified and finalized")
                        append_job_log(job_id, "Target/Source race winner: Bazarr")
                        _status = "BAZARR MATCH"
                        _reason = "Bazarr found all targets"
                        _res_reason = "bazarr_downloaded"
                    elif _all_external:
                        append_job_log(job_id, f"Bazarr coordination: {_coord_elapsed}s (pre-existing target verified)")
                        append_job_log(job_id, "Pre-existing target verified and finalized")
                        append_job_log(job_id, "Target/Source race winner: Pre-existing external target")
                        _status = "ALREADY EXISTS"
                        _reason = "Pre-existing target verified after source resolution"
                        _res_reason = "already_exists"
                    elif bool(_c_origins) and any(orig == CandidateOrigin.BAZARR for orig in _c_origins.values()):
                        append_job_log(job_id, f"Bazarr coordination: {_coord_elapsed}s (mixed targets verified)")
                        append_job_log(job_id, "Target/Source race winner: Mixed (Bazarr + existing)")
                        _status = "BAZARR MATCH"
                        _reason = "Targets resolved without AI (Bazarr + existing)"
                        _res_reason = "bazarr_downloaded"
                    else:
                        append_job_log(job_id, f"Bazarr coordination: {_coord_elapsed}s (pre-existing target verified)")
                        append_job_log(job_id, "Pre-existing target verified and finalized")
                        append_job_log(job_id, "Target/Source race winner: Pre-existing external target")
                        _status = "ALREADY EXISTS"
                        _reason = "Pre-existing target verified after source resolution"
                        _res_reason = "already_exists"

                    append_job_log(job_id, "AI skipped")
                    append_job_log(job_id, "AI calls: 0")
                    append_job_log(job_id, "Estimated AI cost: $0.00")
                    if source_subtitle and source_subtitle.origin == SourceOrigin.EMBEDDED:
                        try: os.remove(source_subtitle.path)
                        except Exception: pass
                    _bzdur2 = round(prev_dur + (time.time() - start_time), 2)
                    update_job(job_id, status=_status, reason=_reason, duration_seconds=_bzdur2, sync_diff_ms=-1, dropped_lines=0)
                    await self._notify_media_servers(video_path)
                    return {"status": "skipped", "reason": _res_reason, "job_id": job_id}
                else:
                    _any_target_on_disk = any(find_external_subtitle(video_path, _gl["code"]) for _gl in langs_needing_translation)
                    if _any_target_on_disk and last_decision_summary:
                        append_job_log(job_id, f"Bazarr coordination: target detected but rejected by Trust Engine ({last_decision_summary}); continuing with Babel fallback")
                    elif _any_target_on_disk:
                        append_job_log(job_id, "Bazarr coordination: target detected but rejected by Trust Engine; continuing with Babel fallback")
                    else:
                        append_job_log(job_id, "Bazarr coordination: no target subtitle found")


        try:
            from app.core.db import DeferStage
            current_pipeline_stage = DeferStage.PRIMARY
            # ── PIN JOB PROVIDER & CONTEXT ────────────────────────────────────
            # Pin the job to its configured AI provider and model before ANY AI
            # call (including SDH classifier / pre-translation cleanup).
            from app.core.ai_providers import context_from_settings, resolve_job_provider_context
            from app.core.db import pin_job_provider

            configured_primary = context_from_settings()

            esc_enabled  = get_setting("escalate_to_pro", "false").lower() == "true"
            configured_escalation = context_from_settings(escalation=True) if esc_enabled else None

            pin_job_provider(
                job_id,
                primary_provider=configured_primary.provider,
                primary_model=configured_primary.model,
                escalation_enabled=bool(configured_escalation),
                escalation_provider=configured_escalation.provider if configured_escalation else None,
                escalation_model=configured_escalation.model if configured_escalation else None,
            )

            primary_ctx = resolve_job_provider_context(job_id)
            active_engine_name = primary_ctx.engine_label
            active_model_name = primary_ctx.model
            ai_provider = primary_ctx.provider

            # ── PRE-PROCESSING & SDH CLEANER (on resolved source subtitle) ──────
            raw_srt_text = source_subtitle.content
            t_clean_start = time.perf_counter()
            clean_sdh_enabled = get_setting("clean_sdh", "true").lower() == "true"
            provenance_map = {}
            if clean_sdh_enabled:
                classifier_fn = getattr(self.translator, "classify_sdh_segments", None)
                source_lang_name = source_subtitle.language_name if (source_subtitle and source_subtitle.language_name) else "unknown"
                res = await sanitize_srt_content_with_provenance(
                    raw_srt_text,
                    source_language=source_lang_name,
                    classifier_fn=classifier_fn,
                    job_id=job_id
                )
                subs, provenance_map, cleaned_count = res
                telem = getattr(res, "telemetry", {})
                append_job_log(job_id, f"Sanitizer processed {len(subs)} blocks. Cleaned noise on {cleaned_count} blocks.")
                if telem:
                    append_job_log(
                        job_id,
                        f"SDH cleaner: total={telem.get('total_s', 0.0)}s "
                        f"(local_analysis={telem.get('local_analysis_s', 0.0)}s, "
                        f"classifier_wait={telem.get('classifier_wait_s', 0.0)}s, "
                        f"reconstruction={telem.get('reconstruction_s', 0.0)}s, "
                        f"ambiguous_unique={telem.get('ambiguous_unique', 0)}, "
                        f"classifier_batches={telem.get('classifier_batches', 0)}, "
                        f"classifier_concurrency={telem.get('classifier_concurrency', 1)})"
                    )
            else:
                subs = list(srt.parse(raw_srt_text))
                cleaned_count = 0
                append_job_log(job_id, f"SDH cleaner disabled. Parsed {len(subs)} blocks directly.")
            t_clean_ms = round((time.perf_counter() - t_clean_start) * 1000, 1)

            prep_duration_ms = round((time.time() - prep_start_time) * 1000, 1)
            src_origin = getattr(source_subtitle, "origin", None)
            src_origin_val = src_origin.value if hasattr(src_origin, "value") else str(src_origin or "unknown")
            perf_breakdown = (
                f"(probe={round(t_probe_ms / 1000, 2)}s, "
                f"extract={round(t_extract_ms / 1000, 2)}s, "
                f"clean={round(t_clean_ms / 1000, 2)}s, "
                f"method={src_origin_val})"
            )

            append_job_log(job_id, f"Source preparation: {round(prep_duration_ms / 1000, 2)}s {perf_breakdown}")

            total_source_lines = len(subs)
            batch_size = get_positive_int_setting("batch_size", 150)

            # ------------------------------------------------------------------
            # Phase 1: Minimum Budget Admission (new primary jobs only)
            # Only run for fresh/new jobs — do NOT gate resumed/partial jobs.
            # ------------------------------------------------------------------
            from app.core.quota import check_minimum_budget_admission
            from app.core.db import update_deferred_metadata, DeferReason, DeferStage

            job_snapshot = None
            try:
                job_snapshot = get_job_by_id(job_id)
            except Exception:
                pass

            is_resumed_job = (
                event_source == "RETRY"
                or (job_snapshot and job_snapshot.get("retry_count", 0) > 0)
                or (job_snapshot and job_snapshot.get("processed_lines", 0) > 0)
            )

            if not is_resumed_job:
                admission = check_minimum_budget_admission(
                    provider   = ai_provider,
                    num_cues   = total_source_lines,
                    batch_size = batch_size,
                )
                if not admission["admitted"]:
                    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
                    now_utc     = _dt.now(_tz.utc)
                    next_retry  = (now_utc + _td(days=1)).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    ).isoformat()
                    admit_msg = (
                        f"DEFERRED: Insufficient local RPD budget for '{ai_provider}' "
                        f"to start translation. "
                        f"Estimated minimum requests: {admission['estimated_minimum']} "
                        f"(ceil({total_source_lines}/{batch_size})). "
                        f"Available: {admission['available']}. "
                        f"Increase daily budget or wait for reset."
                    )
                    append_job_log(job_id, admit_msg)
                    update_job(
                        job_id,
                        status        = "DEFERRED",
                        error_message = admit_msg,
                        next_retry_at = next_retry,
                        last_error    = f"INSUFFICIENT_LOCAL_BUDGET: need {admission['estimated_minimum']}, have {admission['available']}",
                    )
                    update_deferred_metadata(
                        job_id,
                        defer_reason     = DeferReason.INSUFFICIENT_LOCAL_BUDGET,
                        waiting_provider = ai_provider,
                        waiting_model    = active_model_name,
                        defer_stage      = DeferStage.PRIMARY,
                    )
                    if source_subtitle is not None and source_subtitle.origin == SourceOrigin.EMBEDDED:
                        if os.path.exists(temp_extracted_srt):
                            try: os.remove(temp_extracted_srt)
                            except Exception: pass
                    return {
                        "status": "deferred",
                        "error": "INSUFFICIENT_LOCAL_BUDGET",
                        "job_id": job_id,
                        "estimated_minimum": admission["estimated_minimum"],
                        "available": admission["available"],
                    }

            update_job(job_id, total_lines=total_source_lines, processed_lines=0, current_batch="Starting...")

            output_files = []
            successful_langs = []
            failed_langs_details = []
            rescued_langs = []
            external_resolved_langs = []
            ai_translated_langs = []
            skipped_langs = []
            deferred_pub_langs = {}
            total_dropped = 0
            max_sync_diff = 0
            is_semantic_deadlock = False

            # -------------------------------------------------------------
            # TRANSLATION & QUALITY ASSURANCE
            # -------------------------------------------------------------

            # Audio lang was detected earlier as source prioritisation signal.
            # OLG blocking removed in v2.3.43. source!=target invariant enforced above.

            for lang_info in langs_needing_translation:
                lang_name = lang_info["name"]
                lang_code = lang_info["code"]
                target_output_path = f"{base_path}.{lang_code}.srt"
                safe_ids = [idx for idx, sub in enumerate(subs) if is_safe_keep_prefilter(sub.content)]
                context_verified_ids = set()

                # Pass 2D: Atomic Pre-AI Target Check
                # Skip when force_retranslate=True — user explicitly requested re-translation
                if not force_retranslate:
                    existing_before_trans = find_external_subtitle(video_path, lang_code)
                    if existing_before_trans:
                        _pre_stable = await wait_for_file_stability(
                            existing_before_trans,
                            min_stability_sec=candidate_stability_sec,
                            timeout_sec=0.8,
                            interval_sec=0.025
                        )
                        if _pre_stable:
                            _pre_snap = capture_target_snapshot(existing_before_trans)
                            _pre_origin = get_correlated_bazarr_origin(existing_before_trans, lang_code)
                            _pre_tres = await trust_engine.evaluate_candidate(
                                video_path=video_path,
                                candidate_path=existing_before_trans,
                                target_lang=lang_code,
                                origin=_pre_origin,
                                container_tracks=container_tracks,
                                primary_audio_lang=primary_audio_lang,
                                provided_source=source_subtitle,
                                job_id=job_id,
                                auto_repair=auto_repair_unhealthy,
                                allow_ai_audit=True,
                            )
                            if _pre_tres.passed:
                                _pre_final_snap = capture_target_snapshot(existing_before_trans)
                                if _pre_final_snap.generation_id == _pre_snap.generation_id:
                                    append_job_log(job_id, "Target appeared before AI start")
                                    append_job_log(job_id, "Using verified external target")
                                    append_job_log(job_id, "AI skipped")
                                    append_job_log(job_id, "AI calls: 0")
                                    append_job_log(job_id, "Estimated AI cost: $0.00")
                                    successful_langs.append(lang_code)
                                    if _pre_origin == CandidateOrigin.BAZARR:
                                        rescued_langs.append(lang_code)
                                    else:
                                        external_resolved_langs.append(lang_code)
                                    continue
                            else:
                                append_job_log(job_id, f"Final target check: existing target rejected by Trust Engine ({_pre_tres.decision.value}: {'; '.join(_pre_tres.reasons)})")
                                append_job_log(job_id, "AI fallback required")
                        else:
                            append_job_log(job_id, "Final target check: existing target unstable; AI fallback required")
                    else:
                        append_job_log(job_id, "Final target check: no target subtitle found")
                else:
                    append_job_log(job_id, "Force re-translate enabled — skipping external target check")

                # Check if a previously QA-passed translation exists for this job and language
                # (e.g. publication was deferred because Bazarr was actively writing).
                import app.core.db
                _data_dir = os.path.dirname(app.core.db.DB_PATH)
                _qapassed_file = os.path.join(_data_dir, f"job_{job_id}_{lang_code}_qapassed.json") if job_id else None
                if _qapassed_file and os.path.exists(_qapassed_file) and not force_retranslate:
                    try:
                        with open(_qapassed_file, "r", encoding="utf-8") as _qf:
                            _qdata = json.load(_qf)

                        # Identity verification to prevent stale artifact application to modified video or source
                        _cur_can_path = os.path.realpath(video_path)
                        _cur_stat = os.stat(video_path)
                        _cur_fsize = _cur_stat.st_size
                        _cur_fmtime_ns = getattr(_cur_stat, "st_mtime_ns", int(_cur_stat.st_mtime * 1e9))
                        _cur_src_ident = compute_source_fingerprint(source_subtitle, subs=subs, video_path=video_path)

                        media_match = (
                            _qdata.get("canonical_video_path") == _cur_can_path
                            and _qdata.get("media_file_size") == _cur_fsize
                            and _qdata.get("media_file_mtime_ns") == _cur_fmtime_ns
                            and _qdata.get("lang_code") == lang_code
                            and bool(_qdata.get("translated_srt_text"))
                        )

                        source_match = (
                            _qdata.get("source_content_hash") == _cur_src_ident["source_content_hash"]
                            and _qdata.get("source_cue_count") == _cur_src_ident["source_cue_count"]
                            and _qdata.get("source_origin") == _cur_src_ident["source_origin"]
                            and _qdata.get("source_language", _cur_src_ident["source_language"]) == _cur_src_ident["source_language"]
                        )

                        # If external source on disk, also verify file path, size, and mtime
                        if _cur_src_ident["source_origin"] == "EXTERNAL" and _cur_src_ident["source_path"]:
                            source_match = (
                                source_match
                                and _qdata.get("source_path") == _cur_src_ident["source_path"]
                                and _qdata.get("source_file_size") == _cur_src_ident["source_file_size"]
                                and _qdata.get("source_file_mtime_ns") == _cur_src_ident["source_file_mtime_ns"]
                            )

                        identity_match = media_match and source_match

                        if not identity_match:
                            append_job_log(job_id, f"Cached QA-passed translation artifact identity mismatch (media or source file modified). Invalidating artifact and rerunning translation for {lang_name}.")
                            try: os.remove(_qapassed_file)
                            except Exception: pass
                        else:
                            _t_srt_text = _qdata.get("translated_srt_text")
                            _exp_cues = _qdata.get("expected_cue_count", 0)
                            append_job_log(job_id, f"Resuming publication for {lang_name}: AI translation previously passed QA (identity verified). Attempting publication gate...")
                            _saved_pre_snap_dict = _qdata.get("bazarr_pre_trigger_snapshot")
                            _saved_pre_snap = (
                                TargetSnapshot(**_saved_pre_snap_dict)
                                if _saved_pre_snap_dict
                                else initial_target_snapshots.get(normalize_language_code(lang_code))
                            )
                            _saved_accepted = _qdata.get("bazarr_search_accepted", is_bazarr_accepted_for_lang(lang_code))
                            _saved_correlated = _qdata.get("bazarr_media_correlated", is_bazarr_correlated_for_lang(lang_code))

                            _res_pub = await _publish_subtitle_with_trust_gate(
                                video_path=video_path,
                                target_output_path=target_output_path,
                                lang_code=lang_code,
                                translated_srt_text=_t_srt_text,
                                expected_cue_count=_exp_cues,
                                source_subtitle=source_subtitle,
                                container_tracks=container_tracks,
                                primary_audio_lang=primary_audio_lang,
                                force_retranslate=force_retranslate,
                                job_id=job_id,
                                auto_repair=auto_repair_unhealthy,
                                bazarr_pre_trigger_snapshot=_saved_pre_snap,
                                bazarr_search_accepted=_saved_accepted,
                                bazarr_media_correlated=_saved_correlated,
                            )
                            if _res_pub.get("published"):
                                output_files.append(target_output_path)
                                successful_langs.append(lang_code)
                                ai_translated_langs.append(lang_code)
                                try: os.remove(_qapassed_file)
                                except Exception: pass
                                append_job_log(job_id, f"Deferred publication SUCCESS for {lang_name}. AI re-translation bypassed.")
                                continue
                            elif _res_pub.get("skipped"):
                                successful_langs.append(lang_code)
                                _ad_cand = find_external_subtitle(video_path, lang_code)
                                if _saved_accepted or _saved_correlated or _res_pub.get("reason") == "authoritative_target_passed" or (_ad_cand and get_correlated_bazarr_origin(_ad_cand, lang_code) == CandidateOrigin.BAZARR):
                                    rescued_langs.append(lang_code)
                                else:
                                    external_resolved_langs.append(lang_code)
                                try: os.remove(_qapassed_file)
                                except Exception: pass
                                append_job_log(job_id, f"Bazarr target adopted during deferred publication for {lang_name}. AI re-translation bypassed.")
                                continue
                            else:
                                deferred_pub_langs[lang_code] = _res_pub.get("reason") or "bazarr_actively_writing"
                                append_job_log(job_id, f"Deferred publication still blocked ({_res_pub.get('reason')}). Keeping QA result for next retry.")
                                continue
                    except Exception as _q_err:
                        logger.warning(f"Failed to resume deferred publication: {_q_err}")

                append_job_log(job_id, f"AI starting: {active_engine_name}")
                append_job_log(job_id, f"Translating to {lang_name} ({lang_code}) using {active_engine_name} ({total_source_lines} lines)...")

                mid_translation_rescued = False
                mid_seen_gen: Optional[str] = None
                mid_gen_change_time: Optional[float] = None

                async def _check_mid_translation_candidate() -> bool:
                    nonlocal mid_translation_rescued, mid_seen_gen, mid_gen_change_time
                    if force_retranslate:
                        return False
                    cand = find_external_subtitle(video_path, lang_code)
                    if not cand:
                        return False
                    snap = capture_target_snapshot(cand)
                    if not snap.exists:
                        return False
                    now = time.monotonic()
                    if mid_seen_gen != snap.generation_id:
                        mid_seen_gen = snap.generation_id
                        mid_gen_change_time = now

                    _stable = await wait_for_file_stability(
                        cand,
                        min_stability_sec=candidate_stability_sec,
                        timeout_sec=0.3,
                        interval_sec=0.025
                    )
                    if not _stable:
                        return False

                    # Check quiescence before declaring mid-translation early stop
                    time_since_change = now - (mid_gen_change_time or now)
                    if time_since_change < bazarr_quiescence_sec:
                        return False

                    _mid_orig = get_correlated_bazarr_origin(cand, lang_code)
                    _tres = await trust_engine.evaluate_candidate(
                        video_path=video_path,
                        candidate_path=cand,
                        target_lang=lang_code,
                        origin=_mid_orig,
                        container_tracks=container_tracks,
                        primary_audio_lang=primary_audio_lang,
                        provided_source=source_subtitle,
                        job_id=job_id,
                        auto_repair=auto_repair_unhealthy,
                        allow_ai_audit=True,
                    )
                    if _tres.passed:
                        _post_snap = capture_target_snapshot(cand)
                        if _post_snap.generation_id == snap.generation_id:
                            _orig_name = "Bazarr" if _mid_orig == CandidateOrigin.BAZARR else "External"
                            append_job_log(job_id, f"{_orig_name} candidate arrived mid-translation")
                            append_job_log(job_id, "Late candidate verified; stopping remaining AI work")
                            append_job_log(job_id, "Using verified external target")
                            mid_translation_rescued = True
                            return True
                    return False

                t_main_start = time.time()
                try:
                    translated_subs = await self.translator.translate_srt_content(
                        subs=subs,
                        target_language=lang_name,
                        source_language=source_subtitle.language_name,
                        batch_size=batch_size,
                        job_id=job_id,
                        show_title=effective_tm_key or title,
                        provenance_map=provenance_map,
                        early_stop_check=_check_mid_translation_candidate
                    )
                except TypeError as _te:
                    translated_subs = await self.translator.translate_srt_content(
                        subs=subs,
                        target_language=lang_name,
                        source_language=source_subtitle.language_name,
                        batch_size=batch_size,
                        job_id=job_id,
                        show_title=effective_tm_key or title
                    )
                t_main_end = time.time()
                append_job_log(job_id, f"Timing: Main translation phase completed in {round(t_main_end - t_main_start, 1)}s")

                if mid_translation_rescued:
                    successful_langs.append(lang_code)
                    _cand_p = find_external_subtitle(video_path, lang_code)
                    if _cand_p and get_correlated_bazarr_origin(_cand_p, lang_code) == CandidateOrigin.BAZARR:
                        rescued_langs.append(lang_code)
                    else:
                        external_resolved_langs.append(lang_code)
                    continue

                # Check if Bazarr candidate arrived while AI was translating
                if not force_retranslate:
                    _post_ai_target = find_external_subtitle(video_path, lang_code)
                    if _post_ai_target:
                        _post_stable = await wait_for_file_stability(_post_ai_target, min_stability_sec=candidate_stability_sec, timeout_sec=0.6, interval_sec=0.025)
                        if _post_stable:
                            _post_snap = capture_target_snapshot(_post_ai_target)
                            _post_origin = get_correlated_bazarr_origin(_post_ai_target, lang_code)
                            _post_tres = await trust_engine.evaluate_candidate(
                                video_path=video_path,
                                candidate_path=_post_ai_target,
                                target_lang=lang_code,
                                origin=_post_origin,
                                container_tracks=container_tracks,
                                primary_audio_lang=primary_audio_lang,
                                provided_source=source_subtitle,
                                job_id=job_id,
                                auto_repair=auto_repair_unhealthy,
                                allow_ai_audit=True,
                            )
                            if _post_tres.passed:
                                _post_final_snap = capture_target_snapshot(_post_ai_target)
                                if _post_final_snap.generation_id == _post_snap.generation_id:
                                    _orig_lbl = "Bazarr" if _post_origin == CandidateOrigin.BAZARR else "External"
                                    append_job_log(job_id, f"{_orig_lbl} candidate arrived after AI start")
                                    append_job_log(job_id, "Late candidate verified; stopping remaining AI work")
                                    append_job_log(job_id, "Using verified external target")
                                    successful_langs.append(lang_code)
                                    if _post_origin == CandidateOrigin.BAZARR:
                                        rescued_langs.append(lang_code)
                                    else:
                                        external_resolved_langs.append(lang_code)
                                    continue

                # --- NEVER GIVE UP RECOVERY LOOP ---
                max_qa_loops = 3
                qa_loop_count = 0
                known_untranslated_ids = set()
                exhausted_strategies = set()
                recovered_cues = set()
                previous_unresolved_set: Optional[Set[int]] = None
                recovered_at_loop_start = 0
                source_preserved_cues = set()
                initial_identical_candidates_set: Set[int] = set()
                alignment_dirty = True
                incident_tracker = SemanticIncidentTracker(total_cues=total_source_lines, batch_size=batch_size)
                latest_alignment_issues = []
                latest_affected_indices = set()
                mutated_cue_indices: Set[int] = set()

                def _apply_mutation(idx: int, new_text: str):
                    nonlocal alignment_dirty
                    if 0 <= idx < len(translated_subs) and translated_subs[idx].content != new_text:
                        translated_subs[idx].content = new_text
                        mutated_cue_indices.add(idx)
                        alignment_dirty = True

                while qa_loop_count < max_qa_loops:
                    qa_loop_count += 1

                    if qa_loop_count > 1:
                        append_job_log(job_id, f"--- QA RECOVERY LOOP {qa_loop_count}/{max_qa_loops} ---")

                    # Strict Sync Lock: Match source timestamps exactly to guarantee 0ms drift
                    if strict_sync_lock and len(translated_subs) == len(subs):
                        for idx in range(len(subs)):
                            translated_subs[idx].start = subs[idx].start
                            translated_subs[idx].end = subs[idx].end

                    if alignment_dirty:
                        recheck_anomalies = list(mutated_cue_indices) if qa_loop_count > 1 else None
                        alignment_rep = await self.check_semantic_cue_alignment(
                            subs, translated_subs,
                            target_language=lang_name,
                            source_language=source_subtitle.language_name if source_subtitle else "English",
                            show_title=title or "",
                            job_id=job_id,
                            batch_size=batch_size,
                            anomaly_indices=recheck_anomalies,
                            incident_tracker=incident_tracker
                        )
                        found_incidents = alignment_rep.get("incidents", []) if isinstance(alignment_rep, dict) else []
                        repairable_incidents = incident_tracker.get_repairable_incidents(found_incidents) if incident_tracker else []
                        has_repairable_batches = bool(incident_tracker and incident_tracker.get_repairable_batches())
                        if has_repairable_batches or repairable_incidents:
                            await self._repair_semantic_alignment_incidents(
                                subs, translated_subs,
                                incidents=repairable_incidents,
                                target_language=lang_name,
                                source_language=source_subtitle.language_name if source_subtitle else "English",
                                show_title=title or "",
                                job_id=job_id,
                                apply_mutation_fn=_apply_mutation,
                                incident_tracker=incident_tracker
                            )
                        latest_alignment_issues = incident_tracker.get_all_active_issues() if incident_tracker else []
                        latest_affected_indices = set(alignment_rep.get("affected_indices", [])) if isinstance(alignment_rep, dict) else set()
                        mutated_cue_indices.clear()
                        alignment_dirty = False

                    # -------------------------------------------------------
                    # Bug #1, #5: FINAL QA GATE — never publish a broken file
                    # Inside recovery loop: require clean pass without warnings to attempt recovery
                    # -------------------------------------------------------
                    qa_result = qa_gate(
                        subs, translated_subs,
                        target_lang_code=lang_code,
                        job_id=job_id,
                        safe_ids=safe_ids,
                        show_title=title or "",
                        context_verified_ids=context_verified_ids,
                        allow_warnings=False,
                        source_language_name=source_subtitle.language_name if source_subtitle else "source",
                        semantic_alignment_issues=latest_alignment_issues if latest_alignment_issues else None,
                    )

                    if qa_loop_count == 1:
                        initial_identical_candidates_set = set(qa_result.get("untranslated_ids", []))

                    if qa_result["passed"]:
                        break  # Clean PASS! Exit the recovery loop.

                    current_unresolved_set = set(
                        qa_result.get("real_untranslated_ids", []) +
                        qa_result.get("wrong_language_ids", []) +
                        qa_result.get("contaminated_ids", []) +
                        [d.get("index", d.get("id", 1)) - 1 for d in qa_result.get("dropped_details", [])]
                    )

                    # Stagnation Guard: If unresolved set is unchanged and 0 new cues were recovered, break immediately
                    if previous_unresolved_set is not None and current_unresolved_set == previous_unresolved_set and len(recovered_cues) == recovered_at_loop_start:
                        append_job_log(job_id, f"QA Recovery: Stagnation detected ({len(current_unresolved_set)} unresolved cues unchanged with 0 progress). Breaking QA recovery loop.")
                        is_semantic_deadlock = True
                        break

                    previous_unresolved_set = set(current_unresolved_set)
                    recovered_at_loop_start = len(recovered_cues)

                    # Attempt recovery for any untranslated, wrong-language, contaminated, or dropped lines
                    if qa_result["untranslated_ids"] or qa_result.get("dropped_count", 0) > 0 or qa_result.get("wrong_language_ids") or qa_result.get("contaminated_ids"):
                        try:
                            # 1. Primary Recovery for Identical lines (English still present)
                            if qa_result["untranslated_ids"]:
                                identical_ids = qa_result["untranslated_ids"]
                                append_job_log(job_id, f"QA: PRIMARY_RECOVERY ({len(identical_ids)} identical candidates need review)")
                                recovery_payload = [
                                    {"id": idx, "text": subs[idx].content}
                                    for idx in identical_ids
                                    if subs[idx].content.strip() and subs[idx].content.strip() != "<i></i>" and idx not in safe_ids and idx not in known_untranslated_ids
                                ]
                                if recovery_payload:
                                    provider = primary_ctx.provider
                                    from app.core.ai_providers import get_model_capabilities
                                    caps = get_model_capabilities(primary_ctx.provider, primary_ctx.model)
                                    if caps.semantic_audit:
                                        recovery_results = []
                                        chunk_size = 20
                                        for i in range(0, len(recovery_payload), chunk_size):
                                            chunk = recovery_payload[i:i + chunk_size]
                                            try:
                                                try:
                                                    chunk_res = await self.translator.classify_and_recover_identical(
                                                        chunk, lang_name, effective_tm_key or title or "",
                                                        source_subs=subs,
                                                        translated_subs=translated_subs,
                                                        job_id=job_id,
                                                        source_language=source_subtitle.language_name if source_subtitle else "source",
                                                    )
                                                except TypeError:
                                                    chunk_res = await self.translator.classify_and_recover_identical(
                                                        chunk, lang_name, effective_tm_key or title or "",
                                                        job_id=job_id,
                                                        source_language=source_subtitle.language_name if source_subtitle else "source",
                                                    )
                                                recovery_results.extend(chunk_res)
                                            except (ProviderUnavailableError, ProviderConfigurationError, DailyQuotaExhaustedError, RequestBudgetExhaustedError):
                                                raise
                                            except Exception as e:
                                                append_job_log(job_id, f"QA Primary Recovery chunk failed: {e}")
                                                continue

                                        for r in recovery_results:
                                            idx = r.get("id")
                                            if idx is None: continue
                                            action = r.get("action")
                                            if action == "keep":
                                                raw_reason = r.get("reason", "none").lower()
                                                is_struct_safe = is_pure_structural_invariant(subs[idx].content)
                                                is_det_safe = is_deterministically_safe_keep(subs[idx].content, raw_reason, show_title=title or "")
                                                is_ev_safe = has_entity_evidence(subs[idx].content, subs, translated_subs, target_idx=idx)
                                                is_sem_verified = bool(r.get("semantic_verified", False))
                                                is_ctx_legacy = ("context_verified" in raw_reason) and is_strictly_valid_entity_candidate(subs[idx].content)

                                                if is_struct_safe or is_det_safe or is_ev_safe or is_sem_verified or is_ctx_legacy:
                                                    if idx not in safe_ids:
                                                        safe_ids.append(idx)
                                                    if is_sem_verified or is_ctx_legacy:
                                                        context_verified_ids.add(idx)
                                                        append_job_log(job_id, f"QA Recovery: Model kept cue {idx + 1} (Semantic-verified Invariant: '{subs[idx].content.strip()}')")
                                                    elif is_ev_safe and not is_det_safe:
                                                        append_job_log(job_id, f"QA Recovery: Model kept cue {idx + 1} (Evidence-based Entity: '{subs[idx].content.strip()}')")
                                                    else:
                                                        reason_map = {
                                                            "proper_noun": "Name / Proper Noun",
                                                            "brand": "Brand Name",
                                                            "acronym": "Acronym / Abbreviation",
                                                            "number": "Number",
                                                            "symbol": "Symbol / Punctuation",
                                                            "non_verbal": "Non-verbal Sound",
                                                            "shared_word": "Shared Word",
                                                            "none": "Unspecified Reason"
                                                        }
                                                        reason_str = reason_map.get(raw_reason, raw_reason.replace("_", " ").title())
                                                        append_job_log(job_id, f"QA Recovery: Model kept cue {idx + 1} ({reason_str})")
                                                else:
                                                    append_job_log(job_id, f"QA Recovery: Rejected unsafe KEEP for cue {idx + 1} ('{subs[idx].content}'). Forcing translation.")
                                                    known_untranslated_ids.add(idx)
                                            elif action == "translate" and "text" in r:
                                                if not is_usable_translation(r["text"]):
                                                    append_job_log(job_id, f"QA Recovery: Rejected blank/invalid translation for cue {idx + 1}")
                                                elif is_meaningful_translation(subs[idx].content, r["text"]):
                                                    _apply_mutation(idx, r["text"])
                                                    append_job_log(job_id, f"QA Recovery: Translated cue {idx + 1}")
                                                    recovered_cues.add(idx)
                                                elif is_valid_shared_or_entity_keep(subs[idx].content, r["text"], lang_code):
                                                    _apply_mutation(idx, r["text"])
                                                    if idx not in safe_ids:
                                                        safe_ids.append(idx)
                                                    append_job_log(job_id, f"QA Recovery: Valid shared translation for cue {idx + 1}")
                                                    recovered_cues.add(idx)
                                    else:
                                        # Deterministic fallback for DeepL
                                        recovery_results = []
                                        chunk_size = 20
                                        for i in range(0, len(recovery_payload), chunk_size):
                                            chunk = recovery_payload[i:i + chunk_size]
                                            try:
                                                chunk_res = await self.translator.translate_batch(
                                                    chunk, target_language=lang_name, show_title=effective_tm_key or title or "", job_id=job_id
                                                )
                                                recovery_results.extend(chunk_res)
                                            except (ProviderUnavailableError, ProviderConfigurationError):
                                                raise
                                            except Exception as e:
                                                append_job_log(job_id, f"QA Recovery chunk failed (DeepL): {e}")
                                                continue

                                        for r in recovery_results:
                                            idx = r.get("id")
                                            if idx is None: continue
                                            text = r.get("text", "")
                                            if is_meaningful_translation(subs[idx].content, text):
                                                _apply_mutation(idx, text)
                                                append_job_log(job_id, f"QA Recovery: Translated cue {idx + 1}")
                                                recovered_cues.add(idx)

                            # Re-run QA after primary recovery with safe_ids context
                            qa_result = qa_gate(subs, translated_subs, target_lang_code=lang_code, job_id=job_id, safe_ids=safe_ids, show_title=title or "", context_verified_ids=context_verified_ids, allow_warnings=False, source_language_name=source_subtitle.language_name if source_subtitle else "source")
                            if qa_result["passed"]:
                                break

                            # Helper function to construct rich contextual items for bulk recovery
                            def _build_contextual_items(target_ids):
                                items = []
                                for idx in target_ids:
                                    ctx_before_parts = []
                                    for b_idx in range(max(0, idx - 3), idx):
                                        bc = subs[b_idx].content.strip()
                                        if bc and bc != "<i></i>":
                                            btc = translated_subs[b_idx].content.strip() if b_idx < len(translated_subs) else ""
                                            if btc and btc != "<i></i>" and is_meaningful_translation(bc, btc):
                                                ctx_before_parts.append(f"{bc} [{lang_code.upper()}: {btc}]")
                                            else:
                                                ctx_before_parts.append(bc)

                                    ctx_after_parts = []
                                    for a_idx in range(idx + 1, min(len(subs), idx + 4)):
                                        ac = subs[a_idx].content.strip()
                                        if ac and ac != "<i></i>":
                                            ctx_after_parts.append(ac)

                                    items.append({
                                        "id": idx,
                                        "target": subs[idx].content,
                                        "context_before": " | ".join(ctx_before_parts) if ctx_before_parts else "(none)",
                                        "context_after": " | ".join(ctx_after_parts) if ctx_after_parts else "(none)"
                                    })
                                return items

                            async def _verify_identical_candidates_batch(target_ids, phase_label="QA Recovery"):
                                to_verify = [
                                    rid for rid in sorted(set(target_ids))
                                    if rid not in safe_ids
                                    and rid not in context_verified_ids
                                    and 0 <= rid < len(subs)
                                    and subs[rid].content.strip()
                                    and subs[rid].content.strip() != "<i></i>"
                                ]
                                if not to_verify:
                                    return 0

                                items = _build_contextual_items(to_verify)
                                for c in items:
                                    c["proposed_reason"] = "recovery_identical"

                                logger.info(f"{phase_label}: batch verifying {len(items)} identical recovery candidates via semantic invariant auditor")
                                verified_ids = set()
                                try:
                                    verified_ids = await self.translator.verify_alphabetic_invariants_batch(
                                        items,
                                        target_language=lang_name,
                                        source_language=source_subtitle.language_name if source_subtitle else "source",
                                        show_title=effective_tm_key or title or "",
                                        job_id=job_id
                                    )
                                except (ProviderUnavailableError, ProviderConfigurationError, DailyQuotaExhaustedError, RequestBudgetExhaustedError) as e:
                                    logger.warning(f"{phase_label}: semantic invariant verification provider error: {e}")
                                except Exception as e:
                                    logger.error(f"{phase_label}: batch semantic invariant verification failed: {e}")

                                verified_count = 0
                                for rid in to_verify:
                                    if rid in verified_ids:
                                        if rid not in safe_ids:
                                            safe_ids.append(rid)
                                        context_verified_ids.add(rid)
                                        _apply_mutation(rid, subs[rid].content)
                                        recovered_cues.add(rid)
                                        verified_count += 1
                                        append_job_log(job_id, f"{phase_label}: Model kept cue {rid + 1} (Semantic-verified Invariant: '{subs[rid].content.strip()}')")
                                    else:
                                        if phase_label == "Escalation":
                                            append_job_log(job_id, f"Escalation: Rejected identical fallback for cue {rid + 1}")
                                        logger.info(f"{phase_label}: candidate cue {rid + 1} ('{subs[rid].content.strip()}') rejected by semantic invariant auditor")

                                return verified_count

                            # Gather all unresolved cues (untranslated, wrong language, contaminated, dropped)
                            real_unresolved = qa_result.get("real_untranslated_ids", [])
                            wrong_lang_unresolved = qa_result.get("wrong_language_ids", [])
                            contaminated_unresolved = qa_result.get("contaminated_ids", [])
                            dropped_unresolved = [d.get("index", d.get("id", 1)) - 1 for d in qa_result.get("dropped_details", [])]
                            all_unresolved = [
                                idx for idx in sorted(set(real_unresolved + wrong_lang_unresolved + contaminated_unresolved + dropped_unresolved))
                                if idx not in safe_ids and subs[idx].content.strip() and subs[idx].content.strip() != "<i></i>"
                                and f"escalation:{idx}" not in exhausted_strategies
                            ]

                            # =========================================================================
                            # STAGE 1: Bulk Contextual Recovery (Single structured bulk call)
                            # =========================================================================
                            if all_unresolved:
                                for uid in all_unresolved:
                                    known_untranslated_ids.add(uid)

                                strategy_key_a = f"bulk_contextual:{tuple(all_unresolved)}"
                                if strategy_key_a not in exhausted_strategies:
                                    append_job_log(job_id, f"Bulk Contextual Recovery: translating {len(all_unresolved)} unresolved dialogue cues")
                                    t_bcr_start = time.perf_counter()
                                    bcr_items = _build_contextual_items(all_unresolved)
                                    try:
                                        bcr_results = await self.translator.fast_final_rescue_batch(
                                            bcr_items,
                                            target_language=lang_name,
                                            source_language=source_subtitle.language_name if source_subtitle else "source",
                                            show_title=effective_tm_key or title or "",
                                            attempt=1,
                                            job_id=job_id
                                        )
                                        valid_bcr_map, bcr_report = validate_recovery_batch_results(bcr_items, bcr_results)
                                        bcr_success = 0
                                        bcr_identical_candidates = []
                                        for rid, text in valid_bcr_map.items():
                                            if is_usable_translation(text):
                                                if is_meaningful_translation(subs[rid].content, text):
                                                    _apply_mutation(rid, text)
                                                    recovered_cues.add(rid)
                                                    bcr_success += 1
                                                elif is_valid_shared_or_entity_keep(subs[rid].content, text, lang_code):
                                                    _apply_mutation(rid, text)
                                                    if rid not in safe_ids: safe_ids.append(rid)
                                                    recovered_cues.add(rid)
                                                    bcr_success += 1
                                                else:
                                                    bcr_identical_candidates.append(rid)

                                        if bcr_identical_candidates:
                                            verified_bcr = await _verify_identical_candidates_batch(bcr_identical_candidates, phase_label="QA Recovery")
                                            bcr_success += verified_bcr

                                        t_bcr_dur = round(time.perf_counter() - t_bcr_start, 1)
                                        append_job_log(job_id, f"Bulk Contextual Recovery: recovered {bcr_success}/{len(all_unresolved)} cues in {t_bcr_dur}s")
                                        append_job_log(job_id, f"Targeted Recovery: translated {bcr_success}/{len(all_unresolved)}")
                                        if bcr_success > 0:
                                            recovered_at_loop_start = -1
                                        else:
                                            exhausted_strategies.add(strategy_key_a)
                                    except (ProviderUnavailableError, ProviderConfigurationError, DailyQuotaExhaustedError, RequestBudgetExhaustedError):
                                        raise
                                    except Exception as e:
                                        append_job_log(job_id, f"Bulk Contextual Recovery failed: {e}")
                                        exhausted_strategies.add(strategy_key_a)

                                still_unresolved = [
                                    idx for idx in all_unresolved
                                    if idx not in safe_ids and not is_meaningful_translation(subs[idx].content, translated_subs[idx].content) and not is_valid_shared_or_entity_keep(subs[idx].content, translated_subs[idx].content, lang_code)
                                ]

                                # =========================================================================
                                # STAGE 2: Bulk Strict Recovery (Semantically distinct alternative bulk call)
                                # =========================================================================
                                if still_unresolved:
                                    strategy_key_b = f"bulk_strict:{tuple(still_unresolved)}"
                                    if strategy_key_b not in exhausted_strategies:
                                        append_job_log(job_id, f"Bulk Strict Recovery: translating {len(still_unresolved)} remaining cues with direct focus")
                                        t_bsr_start = time.perf_counter()
                                        bsr_items = _build_contextual_items(still_unresolved)
                                        try:
                                            bsr_results = await self.translator.fast_final_rescue_batch(
                                                bsr_items,
                                                target_language=lang_name,
                                                source_language=source_subtitle.language_name if source_subtitle else "source",
                                                show_title=effective_tm_key or title or "",
                                                attempt=2,
                                                job_id=job_id
                                            )
                                            valid_bsr_map, bsr_report = validate_recovery_batch_results(bsr_items, bsr_results)
                                            bsr_success = 0
                                            bsr_identical_candidates = []
                                            for rid, text in valid_bsr_map.items():
                                                if is_usable_translation(text):
                                                    if is_meaningful_translation(subs[rid].content, text):
                                                        _apply_mutation(rid, text)
                                                        recovered_cues.add(rid)
                                                        bsr_success += 1
                                                    elif is_valid_shared_or_entity_keep(subs[rid].content, text, lang_code):
                                                        _apply_mutation(rid, text)
                                                        if rid not in safe_ids: safe_ids.append(rid)
                                                        recovered_cues.add(rid)
                                                        bsr_success += 1
                                                    else:
                                                        bsr_identical_candidates.append(rid)

                                            if bsr_identical_candidates:
                                                verified_bsr = await _verify_identical_candidates_batch(bsr_identical_candidates, phase_label="QA Recovery")
                                                bsr_success += verified_bsr

                                            t_bsr_dur = round(time.perf_counter() - t_bsr_start, 1)
                                            append_job_log(job_id, f"Bulk Strict Recovery: recovered {bsr_success}/{len(still_unresolved)} cues in {t_bsr_dur}s")
                                            if bsr_success > 0:
                                                recovered_at_loop_start = -1
                                            else:
                                                exhausted_strategies.add(strategy_key_b)
                                        except (ProviderUnavailableError, ProviderConfigurationError, DailyQuotaExhaustedError, RequestBudgetExhaustedError):
                                            raise
                                        except Exception as e:
                                            append_job_log(job_id, f"Bulk Strict Recovery failed: {e}")
                                            exhausted_strategies.add(strategy_key_b)

                                # Re-run QA after Stage 1 & 2
                                qa_result = qa_gate(subs, translated_subs, target_lang_code=lang_code, job_id=job_id, safe_ids=safe_ids, show_title=title or "", context_verified_ids=context_verified_ids, allow_warnings=False, source_language_name=source_subtitle.language_name if source_subtitle else "source")
                                if qa_result["passed"]:
                                    break

                            # =========================================================================
                            # STAGE 3: Per-Cue Escalation (Last resort only for stubborn cues)
                            # =========================================================================
                            real_unresolved = qa_result.get("real_untranslated_ids", [])
                            wrong_lang_unresolved = qa_result.get("wrong_language_ids", [])
                            dropped_unresolved = [d.get("index", d.get("id", 1)) - 1 for d in qa_result.get("dropped_details", [])]
                            final_unresolved = [
                                idx for idx in sorted(set(real_unresolved + wrong_lang_unresolved + dropped_unresolved))
                                if idx not in safe_ids and subs[idx].content.strip() and subs[idx].content.strip() != "<i></i>"
                            ]

                            if final_unresolved:
                                _esc_ctx = resolve_job_provider_context(job_id, escalation=True)
                                if _esc_ctx.provider != primary_ctx.provider or _esc_ctx.model != primary_ctx.model:
                                    esc_info = f"{_esc_ctx.provider} / {_esc_ctx.model}"
                                else:
                                    esc_info = "Primary Model (Contextual Mode)"

                                append_job_log(job_id, f"Escalation Stage: {len(final_unresolved)} lines still unresolved. Escalating using {esc_info}...")
                                esc_sem = asyncio.Semaphore(3)
                                esc_identical_candidates = []

                                async def escalate_one(idx):
                                    prev_idx = max(0, idx - 1)
                                    next_idx = min(len(subs) - 1, idx + 1)
                                    prev_text = translated_subs[prev_idx].content if prev_idx != idx else ""
                                    next_text = subs[next_idx].content if next_idx != idx else ""
                                    target_text = subs[idx].content
                                    async with esc_sem:
                                        try:
                                            esc_text = await self.translator.escalate_single_line(
                                                idx, target_text, prev_text, next_text, lang_name, effective_tm_key or title or "",
                                                is_real_untranslated=(idx in real_unresolved or idx in wrong_lang_unresolved),
                                                job_id=job_id,
                                                exhausted_strategies=exhausted_strategies,
                                                source_language=source_subtitle.language_name if source_subtitle else "source",
                                                context_verified_ids=context_verified_ids,
                                            )
                                            if esc_text:
                                                esc_clean = esc_text.strip()
                                                orig_clean = target_text.strip()
                                                is_orig_real = orig_clean and orig_clean != "<i></i>"
                                                is_esc_empty = not esc_clean or esc_clean == "<i></i>"
                                                if not esc_clean:
                                                    append_job_log(job_id, f"Escalation: Rejected empty text for cue {idx + 1}")
                                                elif is_orig_real and is_esc_empty:
                                                    append_job_log(job_id, f"Escalation: Rejected fake empty/tag for real dialogue at cue {idx + 1}")
                                                elif is_meaningful_translation(target_text, esc_text):
                                                    _apply_mutation(idx, esc_text)
                                                    append_job_log(job_id, f"Escalation: Translated cue {idx + 1} using dialogue context")
                                                    recovered_cues.add(idx)
                                                elif is_valid_shared_or_entity_keep(target_text, esc_text, lang_code):
                                                    _apply_mutation(idx, esc_text)
                                                    if idx not in safe_ids: safe_ids.append(idx)
                                                    append_job_log(job_id, f"Escalation: Valid shared translation for cue {idx + 1}")
                                                    recovered_cues.add(idx)
                                                elif idx in context_verified_ids:
                                                    # Early semantic verification succeeded in escalation
                                                    _apply_mutation(idx, esc_text)
                                                    if idx not in safe_ids: safe_ids.append(idx)
                                                    append_job_log(job_id, f"Escalation: Model kept cue {idx + 1} (Semantic-verified Invariant: '{subs[idx].content.strip()}')")
                                                    recovered_cues.add(idx)
                                                else:
                                                    append_job_log(job_id, f"Escalation: Rejected identical fallback for cue {idx + 1}")
                                        except (ProviderUnavailableError, ProviderConfigurationError, DailyQuotaExhaustedError, RequestBudgetExhaustedError):
                                            raise
                                        except Exception as e:
                                            append_job_log(job_id, f"Escalation failed for cue {idx + 1}: {e}")
                                            exhausted_strategies.add(f"escalation:{idx}")

                                esc_tasks = [escalate_one(idx) for idx in final_unresolved]
                                if esc_tasks:
                                    t_esc_start = time.perf_counter()
                                    current_pipeline_stage = DeferStage.ESCALATION
                                    await asyncio.gather(*esc_tasks)
                                    current_pipeline_stage = DeferStage.PRIMARY
                                    t_esc_end = time.perf_counter()
                                    append_job_log(job_id, f"Timing: Escalation phase ({len(esc_tasks)} cues) completed in {round(t_esc_end - t_esc_start, 1)}s")

                                # Final QA rerun after escalation
                                qa_result = qa_gate(subs, translated_subs, target_lang_code=lang_code, job_id=job_id, safe_ids=safe_ids, show_title=title or "", context_verified_ids=context_verified_ids, allow_warnings=False, source_language_name=source_subtitle.language_name if source_subtitle else "source")
                                if qa_result["passed"]:
                                    break

                            # Early Stagnation / Exhaustion check at end of loop:
                            # If all unresolved cues were attempted across all recovery stages without progress, stop redundant calls now!
                            current_unresolved_after_recovery = set(
                                qa_result.get("real_untranslated_ids", [])
                                + qa_result.get("wrong_language_ids", [])
                                + [d.get("index", d.get("id", 1)) - 1 for d in qa_result.get("dropped_details", [])]
                            )
                            unresolved_exhausted = [
                                idx for idx in current_unresolved_after_recovery
                                if f"escalation:{idx}" in exhausted_strategies
                            ]
                            if current_unresolved_after_recovery and len(unresolved_exhausted) == len(current_unresolved_after_recovery):
                                append_job_log(job_id, f"QA Recovery: All {len(current_unresolved_after_recovery)} remaining unresolved cues exhausted all recovery strategies. Proceeding to QA fallback.")
                                is_semantic_deadlock = True
                                break

                            if current_unresolved_after_recovery and len(recovered_cues) == recovered_at_loop_start:
                                append_job_log(job_id, f"QA Recovery: Stagnation detected ({len(current_unresolved_after_recovery)} unresolved cues unchanged with 0 progress). Breaking QA recovery loop.")
                                is_semantic_deadlock = True
                                break

                            # To prevent getting completely stuck in a loop, mark deadlock if this was the last loop
                            if qa_loop_count == max_qa_loops:
                                append_job_log(job_id, f"QA loop exhausted ({max_qa_loops} attempts). Stopping recovery loop for {lang_name}.")
                                is_semantic_deadlock = True

                        except (ProviderUnavailableError, ProviderConfigurationError, DailyQuotaExhaustedError, RequestBudgetExhaustedError):
                            raise
                        except Exception as e:
                            append_job_log(job_id, f"QA Recovery/Escalation failed: {e}")

                # ---------------------------------------------------------------------------
                # POST-RECOVERY EVALUATION & QA FALLBACK
                # If bounded recovery is exhausted and some unresolved source-language cues remain:
                # Check if they meet semantic deadlock criteria and apply source preservation fallback.
                # ---------------------------------------------------------------------------
                real_unresolved_ids = qa_result.get("real_untranslated_ids", [])
                dropped_details = qa_result.get("dropped_details", [])
                dropped_unresolved = [
                    d.get("index", d.get("id", 1)) - 1
                    for d in dropped_details
                    if isinstance(d, dict)
                ]
                final_fallback_cues = [
                    idx for idx in sorted(set(real_unresolved_ids + dropped_unresolved))
                    if 0 <= idx < len(subs)
                    and subs[idx].content.strip()
                    and subs[idx].content.strip() != "<i></i>"
                    and re.sub(r'<[^>]+>', '', subs[idx].content).strip()
                ]
                if final_fallback_cues:
                    is_semantic_deadlock = True
                    for cue_idx in final_fallback_cues:
                        cue_id = subs[cue_idx].index if hasattr(subs[cue_idx], 'index') and subs[cue_idx].index else cue_idx + 1
                        append_job_log(job_id, f"Semantic deadlock detected for cue {cue_id}")
                        append_job_log(job_id, "QA fallback: preserving original source text")
                        _apply_mutation(cue_idx, subs[cue_idx].content)
                        source_preserved_cues.add(cue_idx)

                # Strict Sync Lock: guarantee 0ms drift
                if strict_sync_lock and len(translated_subs) == len(subs):
                    for idx in range(len(subs)):
                        translated_subs[idx].start = subs[idx].start
                        translated_subs[idx].end = subs[idx].end

                if alignment_dirty:
                    recheck_anomalies = list(mutated_cue_indices)
                    alignment_rep = await self.check_semantic_cue_alignment(
                        subs, translated_subs,
                        target_language=lang_name,
                        source_language=source_subtitle.language_name if source_subtitle else "English",
                        show_title=title or "",
                        job_id=job_id,
                        batch_size=batch_size,
                        anomaly_indices=recheck_anomalies if recheck_anomalies else None,
                        incident_tracker=incident_tracker
                    )
                    found_incidents = alignment_rep.get("incidents", []) if isinstance(alignment_rep, dict) else []
                    repairable_incidents = incident_tracker.get_repairable_incidents(found_incidents) if incident_tracker else []
                    has_repairable_batches = bool(incident_tracker and incident_tracker.get_repairable_batches())
                    if has_repairable_batches or repairable_incidents:
                        await self._repair_semantic_alignment_incidents(
                            subs, translated_subs,
                            incidents=repairable_incidents,
                            target_language=lang_name,
                            source_language=source_subtitle.language_name if source_subtitle else "English",
                            show_title=title or "",
                            job_id=job_id,
                            apply_mutation_fn=_apply_mutation,
                            incident_tracker=incident_tracker
                        )
                    latest_alignment_issues = incident_tracker.get_all_active_issues() if incident_tracker else []
                    latest_affected_indices = set(alignment_rep.get("affected_indices", [])) if isinstance(alignment_rep, dict) else set()
                    mutated_cue_indices.clear()
                    alignment_dirty = False

                # Final QA Evaluation against central QA policy (allow_warnings=True)
                qa_result = qa_gate(
                    subs,
                    translated_subs,
                    target_lang_code=lang_code,
                    job_id=job_id,
                    safe_ids=safe_ids,
                    show_title=title or "",
                    context_verified_ids=context_verified_ids,
                    allow_warnings=True,
                    source_language_name=source_subtitle.language_name if source_subtitle else "source",
                    semantic_alignment_issues=latest_alignment_issues if latest_alignment_issues else None,
                )

                # Final QA Log
                score = qa_result["score"]
                if qa_result.get("status") == QA_STATUS_PASS_WITH_WARNINGS:
                    append_job_log(job_id, f"QA Gate PASSED_WITH_WARNINGS (Score: {score}/100)")
                elif qa_result["passed"]:
                    append_job_log(job_id, f"QA Gate PASSED (Score: {score}/100)")
                else:
                    issues_str = "; ".join(qa_result.get("issues", []))
                    append_job_log(job_id, f"QA Gate FAILED (Score: {score}/100) — Issues: {issues_str}")

                # Bug #7: Track both start and end diff
                sync_report = verify_sync(subs, translated_subs)
                dropped_count, _ = check_dropped_lines(subs, translated_subs)
                total_dropped += dropped_count
                this_max_diff = max(sync_report.get("start_diff_ms", 0), sync_report.get("end_diff_ms", 0))
                if this_max_diff > max_sync_diff:
                    max_sync_diff = this_max_diff

                if qa_result["passed"]:
                    translated_srt_text = subs_to_srt_string(translated_subs)
                    pub_res = await _publish_subtitle_with_trust_gate(
                        video_path=video_path,
                        target_output_path=target_output_path,
                        lang_code=lang_code,
                        translated_srt_text=translated_srt_text,
                        expected_cue_count=len(translated_subs),
                        source_subtitle=source_subtitle,
                        container_tracks=container_tracks,
                        primary_audio_lang=primary_audio_lang,
                        force_retranslate=force_retranslate,
                        job_id=job_id,
                        auto_repair=auto_repair_unhealthy,
                        bazarr_pre_trigger_snapshot=initial_target_snapshots.get(normalize_language_code(lang_code)),
                        bazarr_search_accepted=is_bazarr_accepted_for_lang(lang_code),
                        bazarr_media_correlated=is_bazarr_correlated_for_lang(lang_code),
                    )

                    import app.core.db
                    data_dir = os.path.dirname(app.core.db.DB_PATH)
                    qapassed_file = os.path.join(data_dir, f"job_{job_id}_{lang_code}_qapassed.json") if job_id else None

                    if pub_res.get("published"):
                        output_files.append(target_output_path)
                        successful_langs.append(lang_code)
                        ai_translated_langs.append(lang_code)
                        if qapassed_file and os.path.exists(qapassed_file):
                            try: os.remove(qapassed_file)
                            except Exception: pass
                    elif pub_res.get("skipped"):
                        successful_langs.append(lang_code)
                        _pub_cand = find_external_subtitle(video_path, lang_code)
                        if _pub_cand and get_correlated_bazarr_origin(_pub_cand, lang_code) == CandidateOrigin.BAZARR:
                            rescued_langs.append(lang_code)
                        else:
                            external_resolved_langs.append(lang_code)
                        if qapassed_file and os.path.exists(qapassed_file):
                            try: os.remove(qapassed_file)
                            except Exception: pass
                    elif job_id:
                        # QA passed, but publication was deferred (e.g. Bazarr actively writing or ownership lock)
                        deferred_pub_langs[lang_code] = pub_res.get("reason") or "bazarr_actively_writing"
                        # Save the QA-passed translation artifact with full identity verification & atomic write
                        try:
                            can_vid_path = os.path.realpath(video_path)
                            _stat = os.stat(video_path)
                            _fsize = _stat.st_size
                            _fmtime_ns = getattr(_stat, "st_mtime_ns", int(_stat.st_mtime * 1e9))
                            _src_ident = compute_source_fingerprint(source_subtitle, subs=subs, video_path=video_path)

                            payload = {
                                "job_id": job_id,
                                "canonical_video_path": can_vid_path,
                                "media_file_size": _fsize,
                                "media_file_mtime_ns": _fmtime_ns,
                                "target_output_path": os.path.realpath(target_output_path),
                                "lang_code": lang_code,
                                **_src_ident,
                                "translated_srt_text": translated_srt_text,
                                "expected_cue_count": len(translated_subs),
                                "qa_score": score,
                                "qa_issues": qa_result.get("issues", []),
                                "created_at": datetime.now(timezone.utc).isoformat(),
                                "bazarr_pre_trigger_snapshot": (
                                    {
                                        "path": initial_target_snapshots[normalize_language_code(lang_code)].path,
                                        "exists": initial_target_snapshots[normalize_language_code(lang_code)].exists,
                                        "size": initial_target_snapshots[normalize_language_code(lang_code)].size,
                                        "mtime_ns": initial_target_snapshots[normalize_language_code(lang_code)].mtime_ns,
                                        "content_hash": initial_target_snapshots[normalize_language_code(lang_code)].content_hash,
                                    }
                                    if initial_target_snapshots.get(normalize_language_code(lang_code))
                                    else None
                                ),
                                "bazarr_search_accepted": is_bazarr_accepted_for_lang(lang_code),
                                "bazarr_media_correlated": is_bazarr_correlated_for_lang(lang_code),
                            }
                            tmp_qapassed = f"{qapassed_file}.tmp.{uuid.uuid4().hex}"
                            with open(tmp_qapassed, "w", encoding="utf-8") as _qf:
                                json.dump(payload, _qf, indent=2)
                                _qf.flush()
                                os.fsync(_qf.fileno())
                            os.replace(tmp_qapassed, qapassed_file)
                            append_job_log(job_id, f"QA passed but publication deferred ({pub_res.get('reason')}). Cached translation artifact for instant retry without AI re-translation.")
                        except Exception as _qe:
                            logger.warning(f"Failed to cache QA passed translation: {_qe}")

                    # Clean up partial progress file since we successfully finished
                    partial_file = os.path.join(data_dir, f"job_{job_id}_{lang_code}_partial.json") if job_id else None
                    if partial_file and os.path.exists(partial_file):
                        try:
                            os.remove(partial_file)
                        except Exception:
                            pass

                    if pub_res.get("published"):
                        # Save Translation Memory only after QA PASS and if we actually published
                        # Exclude fallback source-preserved cues
                        if effective_tm_key:
                            try:
                                from app.core.db import save_translation_memory_bulk
                                tm_items = []
                                for idx in range(len(subs)):
                                    if idx in source_preserved_cues:
                                        continue
                                    orig_t = subs[idx].content.strip()
                                    trans_t = translated_subs[idx].content.strip()
                                    if orig_t and trans_t and orig_t != "<i></i>" and trans_t != "<i></i>" and orig_t != trans_t:
                                        tm_items.append({"original": orig_t, "translated": trans_t})
                                if tm_items:
                                    _src = source_subtitle.language if source_subtitle else "en"
                                    save_translation_memory_bulk(
                                        effective_tm_key,
                                        tm_items,
                                        source_language=_src,
                                        target_language=lang_code
                                    )
                            except Exception as e:
                                logger.error(f"Failed to save translation memory: {e}")

                    # Build a detailed QA summary for the user
                    unresolved_count = len(qa_result.get("real_untranslated_ids", []))
                    initial_candidates_count = len(initial_identical_candidates_set)
                    kept_count = sum(1 for idx in initial_identical_candidates_set if idx in safe_ids and idx not in qa_result.get("real_untranslated_ids", []) and idx not in recovered_cues)
                    recovered_count = len(recovered_cues) if 'recovered_cues' in locals() else 0
                    source_preserved_count = len(source_preserved_cues)

                    summary_status = qa_result.get("status", "PASS" if qa_result["passed"] else "FAIL")
                    unresolved_label = f"{unresolved_count} unresolved {source_subtitle.language_name if source_subtitle else 'source'} {'line' if unresolved_count == 1 else 'lines'}"
                    fallback_label = f"{source_preserved_count} source-preserved {'fallback' if source_preserved_count == 1 else 'fallbacks'}"

                    summary_lines = [
                        "--- QA Summary ---",
                        f"{dropped_count} dropped lines",
                        f"{this_max_diff} ms sync drift",
                        f"{initial_candidates_count} identical candidates",
                        f"{kept_count} classified as KEEP (safe)",
                        f"{recovered_count} translated on recovery",
                        unresolved_label,
                        fallback_label,
                        f"Result: {summary_status} (Score: {qa_result['score']}/100)",
                        "------------------"
                    ]
                    for line in summary_lines:
                        append_job_log(job_id, line)
                else:
                    # Translation or QA failed completely for this language
                    failed_lang_info = {"code": lang_code, "name": lang_name, "issues": qa_result["issues"]}
                    failed_langs_details.append(failed_lang_info)
                    append_job_log(job_id, f"Translation of {lang_name} FAILED QA Gate. Searching for late external candidate...")

                    # Late rescue: check if human subtitle arrived during the failed AI run
                    if not force_retranslate:
                        rescue_p = find_external_subtitle(video_path, lang_code)
                        if rescue_p:
                            r_stable = await wait_for_file_stability(
                                rescue_p,
                                min_stability_sec=candidate_stability_sec,
                                timeout_sec=0.6,
                                interval_sec=0.025
                            )
                            if r_stable:
                                r_snap = capture_target_snapshot(rescue_p)
                                r_origin = get_correlated_bazarr_origin(rescue_p, lang_code)
                                r_tres = await trust_engine.evaluate_candidate(
                                    video_path=video_path,
                                    candidate_path=rescue_p,
                                    target_lang=lang_code,
                                    origin=r_origin,
                                    container_tracks=container_tracks,
                                    primary_audio_lang=primary_audio_lang,
                                    provided_source=source_subtitle,
                                    job_id=job_id,
                                    auto_repair=auto_repair_unhealthy,
                                    allow_ai_audit=True,
                                )
                                if r_tres.passed:
                                    r_final_snap = capture_target_snapshot(rescue_p)
                                    if r_final_snap.generation_id == r_snap.generation_id:
                                        append_job_log(job_id, "Late external target verified; using human subtitle")
                                        append_job_log(job_id, f"Rescue verification passed (score={r_tres.score}/100)")
                                        successful_langs.append(lang_code)
                                        if r_origin == CandidateOrigin.BAZARR:
                                            rescued_langs.append(lang_code)
                                        else:
                                            external_resolved_langs.append(lang_code)
                                else:
                                    append_job_log(job_id, f"Late external target unusable for rescue ({r_tres.decision.value}: {'; '.join(r_tres.reasons)})")
                            else:
                                append_job_log(job_id, "Late external target unusable for rescue (unstable/being written)")
                        else:
                            append_job_log(job_id, "No external rescue candidate found; proceeding with failure")

            # Clean up temp file (only if it was an EMBEDDED extraction, not an external subtitle)
            if source_subtitle is not None and source_subtitle.origin == SourceOrigin.EMBEDDED:
                if source_subtitle.path and os.path.exists(source_subtitle.path):
                    try: os.remove(source_subtitle.path)
                    except Exception: pass
                if os.path.exists(temp_extracted_srt):
                    try: os.remove(temp_extracted_srt)
                    except Exception: pass

            job_data = get_job_by_id(job_id) if job_id else None
            prev_dur = float(job_data.get("duration_seconds") or 0.0) if job_data else 0.0
            total_duration = round(prev_dur + (time.time() - start_time), 2)

            # Determine correct final status & reason
            reason_str = None
            if len(successful_langs) + len(skipped_langs) == len(langs_needing_translation):
                if ai_translated_langs:
                    final_status = "TRANSLATED"
                elif rescued_langs and len(rescued_langs) == len(langs_needing_translation):
                    final_status = "BAZARR MATCH"
                    reason_str = "Bazarr found all targets"
                    max_sync_diff = -1
                elif external_resolved_langs and len(external_resolved_langs) == len(langs_needing_translation):
                    final_status = "ALREADY EXISTS"
                    reason_str = "External target appeared during processing"
                    max_sync_diff = -1
                elif (rescued_langs or external_resolved_langs) and (len(rescued_langs) + len(external_resolved_langs) == len(langs_needing_translation)):
                    final_status = "BAZARR MATCH"
                    reason_str = "Targets resolved without AI (Bazarr + existing)"
                    max_sync_diff = -1
                elif successful_langs:
                    final_status = "TRANSLATED"
                else:
                    final_status = "SKIPPED"
            elif successful_langs:
                final_status = "PARTIAL"
            elif deferred_pub_langs:
                # QA Gate passed, but publication was deferred (e.g. Bazarr actively writing or ownership lock)
                if any("bazarr" in str(r).lower() for r in deferred_pub_langs.values()):
                    final_status = "WAITING_FOR_BAZARR"
                else:
                    final_status = "WAITING_FOR_PUBLICATION"
            elif not qa_result["passed"]:
                final_status = "FAILED"
            else:
                final_status = "RECOVERING"

            update_args = {
                "status": final_status,
                "target_languages": ",".join(successful_langs),
                "total_lines": total_source_lines,
                "cleaned_sdh_lines": cleaned_count,
                "dropped_lines": total_dropped,
                "sync_diff_ms": max_sync_diff,
                "output_files": json.dumps(output_files),
                "duration_seconds": total_duration
            }
            if reason_str:
                update_args["reason"] = reason_str
            if final_status == "FAILED":
                unresolved_count = len(qa_result.get("real_untranslated_ids", []))
                policy_failure_type = qa_result.get("policy_details", {}).get("failure_type")
                if policy_failure_type == "structural":
                    issues_str = "; ".join(qa_result.get("issues", []))
                    update_args["error_message"] = f"Structural integrity failure: {issues_str}. File NOT published."
                    append_job_log(job_id, f"PERMANENT FAILURE: Structural integrity failure ({issues_str}). File NOT published.")
                elif is_semantic_deadlock:
                    update_args["error_message"] = f"Semantic deadlock: {unresolved_count} unresolved cues failed QA. Bounded recovery exhausted."
                    append_job_log(job_id, f"PERMANENT FAILURE: Semantic deadlock detected. Provider succeeded but bounded recovery exhausted with 0 progress. File NOT published.")
                else:
                    issues_str = "; ".join(qa_result.get("issues", []))
                    update_args["error_message"] = f"QA Gate failed: {unresolved_count} unresolved cues ({issues_str}). File NOT published."
                    append_job_log(job_id, f"PERMANENT FAILURE: QA Gate failed with {unresolved_count} unresolved cues. File NOT published.")
                update_args["next_retry_at"] = None
            elif final_status in ["WAITING_FOR_BAZARR", "WAITING_FOR_PUBLICATION"]:
                from datetime import datetime, timezone, timedelta
                job_data = get_job_by_id(job_id)
                current_retries = job_data.get("retry_count", 0) if job_data else 0

                MAX_PUB_WAIT_ATTEMPTS = 5
                if current_retries >= MAX_PUB_WAIT_ATTEMPTS:
                    final_status = "FAILED"
                    update_args["status"] = "FAILED"
                    update_args["error_message"] = f"Publication wait attempts exhausted ({MAX_PUB_WAIT_ATTEMPTS}/{MAX_PUB_WAIT_ATTEMPTS}). Target remained blocked."
                    append_job_log(job_id, f"PERMANENT FAILURE: Publication wait attempts exhausted ({MAX_PUB_WAIT_ATTEMPTS}/{MAX_PUB_WAIT_ATTEMPTS}).")
                else:
                    backoff_mins = 1
                    if current_retries == 1: backoff_mins = 5
                    elif current_retries == 2: backoff_mins = 15
                    elif current_retries == 3: backoff_mins = 30
                    elif current_retries >= 4: backoff_mins = 60

                    update_args["next_retry_at"] = (datetime.now(timezone.utc) + timedelta(minutes=backoff_mins)).isoformat()
                    update_args["retry_count"] = current_retries + 1
                    wait_target = "Bazarr to finalize" if final_status == "WAITING_FOR_BAZARR" else "publication ownership"
                    append_job_log(job_id, f"Publication waiting for {wait_target}. Will resume publication arbitration in {backoff_mins} min (Worker attempt {current_retries + 1}/{MAX_PUB_WAIT_ATTEMPTS}).")
            elif final_status in ["RECOVERING", "PARTIAL"]:
                from datetime import datetime, timezone, timedelta
                job_data = get_job_by_id(job_id)
                current_retries = job_data.get("retry_count", 0) if job_data else 0

                MAX_RECOVERY_ATTEMPTS = 5
                if current_retries >= MAX_RECOVERY_ATTEMPTS:
                    final_status = "FAILED"
                    update_args["status"] = "FAILED"
                    update_args["error_message"] = f"Recovery attempts exhausted ({MAX_RECOVERY_ATTEMPTS}/{MAX_RECOVERY_ATTEMPTS}). Subtitle failed QA."
                    append_job_log(job_id, f"PERMANENT FAILURE: Recovery attempts exhausted ({MAX_RECOVERY_ATTEMPTS}/{MAX_RECOVERY_ATTEMPTS}).")
                else:
                    backoff_mins = 1
                    if current_retries == 1: backoff_mins = 5
                    elif current_retries == 2: backoff_mins = 15
                    elif current_retries == 3: backoff_mins = 30
                    elif current_retries >= 4: backoff_mins = 60

                    update_args["next_retry_at"] = (datetime.now(timezone.utc) + timedelta(minutes=backoff_mins)).isoformat()
                    update_args["retry_count"] = current_retries + 1
                    append_job_log(job_id, f"Job needs recovery. Will resume in {backoff_mins} min (Worker attempt {current_retries + 1}/{MAX_RECOVERY_ATTEMPTS}).")

            update_job(job_id, **update_args)

            if output_files:
                await self._notify_media_servers(video_path)

            return {
                "status": final_status.lower(),
                "job_id": job_id,
                "duration": total_duration,
                "output_files": output_files
            }

        except DailyQuotaExhaustedError as e:
            if source_subtitle is not None and source_subtitle.origin == SourceOrigin.EMBEDDED:
                if os.path.exists(temp_extracted_srt):
                    try: os.remove(temp_extracted_srt)
                    except: pass
            total_duration = round(time.time() - start_time, 2)

            from datetime import datetime, timezone, timedelta
            from app.core.quota import get_provider_block_info, block_provider

            # Ensure provider block is persisted if not already done by translator
            block_info = get_provider_block_info(e.provider)
            if not block_info.get("blocked"):
                block_provider(
                    e.provider,
                    reason=f"Daily quota exhausted: {e.raw_message[:200]}",
                    retry_after_seconds=e.retry_after_seconds,
                )
                block_info = get_provider_block_info(e.provider)

            blocked_until = block_info.get("blocked_until")
            reset_type = block_info.get("reset_type", "estimated")
            probe_attempt = block_info.get("probe_attempt", 0)

            if reset_type == "exact":
                resume_msg = f"exact reset scheduled at {blocked_until}"
            else:
                resume_msg = f"next probe scheduled at {blocked_until} (attempt {probe_attempt})"

            log_msg = (
                f"DEFERRED: Daily provider quota reached for '{e.provider}'. "
                f"Job deferred — {resume_msg}. "
                f"No provider requests will be made until quota resets."
            )
            append_job_log(job_id, log_msg)

            next_retry_at = blocked_until  # scheduler will pick it up after reset/probe time
            update_job(
                job_id,
                status="DEFERRED",
                error_message=f"Daily provider quota reached for '{e.provider}'. {resume_msg}.",
                duration_seconds=total_duration,
                next_retry_at=next_retry_at,
                last_error=str(e),
            )
            # Phase 1: Persist structured deferred metadata for FIFO scheduling
            try:
                from app.core.db import update_deferred_metadata, DeferReason
                update_deferred_metadata(
                    job_id,
                    defer_reason     = DeferReason.PROVIDER_QUOTA,
                    waiting_provider = e.provider.lower(),
                    waiting_model    = None,
                    defer_stage      = current_pipeline_stage,
                )
            except Exception:
                pass
            return {"status": "deferred", "error": str(e), "job_id": job_id}

        except RequestBudgetExhaustedError as e:
            if source_subtitle is not None and source_subtitle.origin == SourceOrigin.EMBEDDED:
                if os.path.exists(temp_extracted_srt):
                    try: os.remove(temp_extracted_srt)
                    except: pass
            total_duration = round(time.time() - start_time, 2)

            from datetime import datetime, timezone, timedelta

            # Defer until next UTC midnight (daily budget resets at 00:00 UTC)
            now = datetime.now(timezone.utc)
            tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            next_retry_at = tomorrow.isoformat()

            log_msg = (
                f"DEFERRED: Daily request budget reached for '{e.provider}' "
                f"({e.used}/{e.budget} requests used today). "
                f"Job deferred until {next_retry_at} (UTC midnight reset)."
            )
            append_job_log(job_id, log_msg)

            update_job(
                job_id,
                status="DEFERRED",
                error_message=f"Daily request budget reached ({e.used}/{e.budget}). Deferred until {next_retry_at}.",
                duration_seconds=total_duration,
                next_retry_at=next_retry_at,
                last_error=str(e),
            )
            # Phase 1: Persist structured deferred metadata for FIFO scheduling
            try:
                from app.core.db import update_deferred_metadata, DeferReason
                _snap2 = get_job_by_id(job_id)

                _waiting_prov = (e.provider.lower() if e.provider else "") or (
                    _snap2.get("primary_provider", "") if _snap2 else ""
                )

                update_deferred_metadata(
                    job_id,
                    defer_reason     = DeferReason.LOCAL_RPD,
                    waiting_provider = _waiting_prov,
                    waiting_model    = None,
                    defer_stage      = current_pipeline_stage,
                )
            except Exception:
                pass
            return {"status": "deferred", "error": str(e), "job_id": job_id}

        except ProviderConfigurationError as e:
            if source_subtitle is not None and source_subtitle.origin == SourceOrigin.EMBEDDED:
                if os.path.exists(temp_extracted_srt):
                    try: os.remove(temp_extracted_srt)
                    except: pass
            total_duration = round(prev_dur + (time.time() - start_time), 2)
            append_job_log(job_id, f"CRITICAL CONFIGURATION ERROR: {str(e)}")
            update_job(
                job_id,
                status="FAILED",
                error_message=str(e),
                duration_seconds=total_duration,
                last_error=str(e)
            )
            return {"status": "failed", "error": str(e), "job_id": job_id}


        except ProviderUnavailableError as e:
            # Only delete temp file if it was an embedded extraction (not an external subtitle)
            _src_is_temp = source_subtitle is not None and source_subtitle.origin == SourceOrigin.EMBEDDED
            if _src_is_temp and os.path.exists(temp_extracted_srt):
                try: os.remove(temp_extracted_srt)
                except: pass
            total_duration = round(prev_dur + (time.time() - start_time), 2)

            from datetime import datetime, timezone, timedelta

            job_data = get_job_by_id(job_id)
            current_retries = job_data.get("retry_count", 0) if job_data else 0

            # Simple backoff: 2 minutes, then 5, then 15, max 30.
            backoff_mins = 2
            if current_retries == 1: backoff_mins = 5
            elif current_retries == 2: backoff_mins = 15
            elif current_retries > 2: backoff_mins = 30

            now = datetime.now(timezone.utc)
            next_retry_at = (now + timedelta(minutes=backoff_mins)).isoformat()

            append_job_log(job_id, f"PROVIDER ERROR: {str(e)}. Will retry at {next_retry_at} (Retry {current_retries + 1}).")
            update_job(
                job_id,
                status="WAITING_PROVIDER",
                error_message=str(e),
                duration_seconds=total_duration,
                retry_count=current_retries + 1,
                next_retry_at=next_retry_at,
                last_error=str(e)
            )
            return {"status": "waiting_provider", "error": str(e), "job_id": job_id}
        except Exception as e:
            # Only delete temp file if it was an embedded extraction (not an external subtitle)
            _src_is_temp = source_subtitle is not None and source_subtitle.origin == SourceOrigin.EMBEDDED
            if _src_is_temp and os.path.exists(temp_extracted_srt):
                try: os.remove(temp_extracted_srt)
                except: pass
            total_duration = round(prev_dur + (time.time() - start_time), 2)

            err_str = str(e).lower()
            if any(t in err_str for t in ["timeout", "429", "500", "502", "503", "504", "connection"]):
                from datetime import datetime, timezone, timedelta
                next_retry_at = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
                append_job_log(job_id, f"NETWORK/PROVIDER ERROR: {str(e)}. Retrying later.")
                update_job(job_id, status="WAITING_PROVIDER", error_message=str(e), duration_seconds=total_duration, next_retry_at=next_retry_at)
                return {"status": "waiting_provider", "error": str(e), "job_id": job_id}

            elif any(t in err_str for t in ["permission denied", "no such file", "not found", "read-only"]):
                append_job_log(job_id, f"TERMINAL FILESYSTEM ERROR: {str(e)}")
                update_job(job_id, status="FAILED", error_message=str(e), duration_seconds=total_duration)
                return {"status": "failed", "error": str(e), "job_id": job_id}

            else:
                from datetime import datetime, timezone, timedelta
                next_retry_at = (datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat()
                append_job_log(job_id, f"UNEXPECTED ERROR: {str(e)}. Treating as recoverable.")
                update_job(job_id, status="RETRY_PENDING", error_message=str(e), duration_seconds=total_duration, next_retry_at=next_retry_at)
                return {"status": "retry_pending", "error": str(e), "job_id": job_id}

pipeline = SubtitlePipeline()
