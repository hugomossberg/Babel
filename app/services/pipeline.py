import asyncio
import os
import time
import json
import logging
from typing import Dict, Any, Optional, List, Set
import srt
import httpx
import uuid

from app.core.cleaner import sanitize_srt_content, subs_to_srt_string
from app.core.extractor import extract_embedded_srt, inspect_mkv_tracks
from app.core.validator import verify_sync, check_dropped_lines, evaluate_subtitle_health, detect_language_heuristics, check_language_representative, are_languages_compatible
from app.core.db import (
    create_job, update_job, append_job_log, get_setting,
    get_positive_int_setting, get_int_setting, get_float_setting,
    save_translation_memory_bulk
)
from app.services.bazarr_checker import check_existing_swedish_subtitle, check_existing_english_subtitle, find_external_subtitle
from app.services.translator import (
    SubtitleTranslator, is_usable_translation, is_meaningful_translation, ProviderUnavailableError,
    ProviderConfigurationError, get_provider_capabilities, is_deterministically_safe_keep,
    normalize_for_compare, is_safe_keep_prefilter, has_entity_evidence,
    is_strictly_valid_entity_candidate
)
from app.services.jellyfin_notifier import notify_jellyfin_library_refresh

logger = logging.getLogger("babel.pipeline")

def _safe_extract_embedded_srt(video_path: str, output_srt_path: str, preferred_lang: str = "eng", tracks_info: Optional[Dict[str, Any]] = None) -> bool:
    """Invokes extract_embedded_srt safely supporting both cached tracks_info and legacy mock signatures."""
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

    # 2. Målspråkskontroll (Semantisk)
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
            target_norm = target_lang_code[:2].lower()
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
            (context_verified_ids is not None and i in context_verified_ids and is_strictly_valid_entity_candidate(orig_content))
        ):
            continue

        real_untranslated_ids.append(i)

    if real_untranslated_ids:
        pct = round(len(real_untranslated_ids) / min_len * 100, 1) if min_len > 0 else 0
        issues.append(f"{len(real_untranslated_ids)} lines ({pct}%) still contain original English text")
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

    if not structural_passed:
        qa_status = QA_STATUS_FAIL
        passed = False
        failure_type = "structural"
    elif confident_wrong_language:
        qa_status = QA_STATUS_FAIL
        passed = False
        failure_type = "semantic"
    elif unresolved_count == 0:
        if score >= 60:
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
            warnings.append(f"{unresolved_count} unresolved English {'line' if unresolved_count == 1 else 'lines'} ({unresolved_ratio*100:.1f}%) preserved as source text")
        else:
            qa_status = QA_STATUS_FAIL
            passed = False
            failure_type = "semantic"
            issues.append(f"Semantic quality score too low ({score}/100, min 60)")
    else:
        qa_status = QA_STATUS_FAIL
        passed = False
        failure_type = "semantic"
        issues.append(f"{unresolved_count} unresolved English {'line' if unresolved_count == 1 else 'lines'} ({unresolved_ratio*100:.1f}%) exceeds QA policy limit (max {limit_count} cues, {limit_ratio*100:.1f}%)")

    return {
        "passed": passed,
        "status": qa_status,
        "score": score,
        "issues": issues,
        "warnings": warnings,
        "untranslated_ids": untranslated_ids,  # Keep full list for recovery attempts
        "real_untranslated_ids": real_untranslated_ids,
        "wrong_language_ids": wrong_language_ids,
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
) -> Dict[str, Any]:
    """
    Atomically validates and publishes translated subtitle text to target_output_path.

    Invariants:
    1. Writes to unique temp file in same directory with fsync.
    2. Reparses temp file and verifies cue count before touching any target/backup.
    3. Handles existing targets with atomic backup and race-check:
       - Evaluates health of moved backup file.
       - If backup is GREEN, restores it and skips publishing (unless force_retranslate).
    4. Uses no-clobber atomic link (_link_temp_no_clobber).
    5. If concurrent target appears after backup:
       - Evaluates health of new target.
       - If GREEN, preserves new target and skips publishing.
       - If unhealthy, backs it up and links temp.
    6. Unified transaction rollback state:
       - If any failure occurs after backup, rolls back backup to target (if target absent/unhealthy).
       - Cleans up temp file fail-closed.
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

            # Move unhealthy target to unique backup atomically
            backup_original_path = target_to_check
            backup_path = f"{target_to_check}.babel-replaced.{uuid.uuid4().hex}"
            try:
                os.replace(target_to_check, backup_path)
            except OSError as exc:
                raise RuntimeError(f"Cannot safely back up existing subtitle {target_to_check}: {exc}") from exc

            # TOCTOU race check on the backup file that was actually moved
            if not force_retranslate:
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
            if job_id:
                append_job_log(job_id, f"Published {os.path.basename(target_output_path)}")
            return {"published": True, "skipped": False, "reason": "published"}

        # Race: target_output_path appeared between backup and link
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


class SubtitlePipeline:
    qa_gate = staticmethod(qa_gate)
    _publish_subtitle_atomic = staticmethod(_publish_subtitle_atomic)
    _link_temp_no_clobber = staticmethod(_link_temp_no_clobber)

    def __init__(self):
        self.translator = SubtitleTranslator()
        self._video_semaphore = None
        self._current_max_jobs = 1
        self._active_tasks: Dict[int, asyncio.Task] = {}
        # Bug #17: Per-video locking to prevent duplicate processing
        self._active_video_paths: Set[str] = set()
        self._video_lock = asyncio.Lock()

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
        max_jobs = get_positive_int_setting("max_concurrent_jobs", 1)
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

    # Bug #3: Accept language parameter instead of hardcoded "sv"
    async def trigger_bazarr_search(self, video_path: str, language: str = "sv"):
        bazarr_url = get_setting("bazarr_url", "http://bazarr:6767").rstrip("/")
        bazarr_api_key = get_setting("bazarr_api_key", "")
        if not bazarr_api_key:
            return

        headers = {"X-API-KEY": bazarr_api_key}
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                # Bug #4: Use Bazarr API properly — search movies first (simpler),
                # then series. For series, we match by video path in the movies/episodes list.

                # 1. Try Movies
                m_res = await client.get(f"{bazarr_url}/api/movies", headers=headers)
                if m_res.status_code == 200:
                    try:
                        m_json = m_res.json()
                        movies = m_json.get("data", []) if isinstance(m_json, dict) else (m_json if isinstance(m_json, list) else [])
                    except Exception:
                        movies = []
                    norm_target = os.path.normpath(video_path)
                    for m in movies:
                        if not isinstance(m, dict):
                            continue
                        m_path = os.path.normpath(m.get("path", ""))
                        if m_path == norm_target or os.path.basename(m_path) == os.path.basename(norm_target):
                            r_id = m.get("radarrId")
                            if r_id:
                                logger.info(f"Triggering Bazarr movie subtitle search for radarrId {r_id}, lang={language}")
                                await client.patch(
                                    f"{bazarr_url}/api/movies/subtitles",
                                    headers=headers,
                                    params={
                                        "radarrid": r_id,
                                        "language": language,
                                        "forced": "False",
                                        "hi": "False"
                                    }
                                )
                                return

                # 2. Try Series — get all series first, then find episodes for matching series
                s_res = await client.get(f"{bazarr_url}/api/series", headers=headers)
                if s_res.status_code == 200:
                    try:
                        s_json = s_res.json()
                        all_series = s_json.get("data", []) if isinstance(s_json, dict) else (s_json if isinstance(s_json, list) else [])
                    except Exception:
                        all_series = []
                    from pathlib import Path
                    target_p = Path(video_path).resolve()
                    for series in all_series:
                        if not isinstance(series, dict):
                            continue
                        raw_series_path = series.get("path", "")
                        if not raw_series_path:
                            continue
                        series_p = Path(raw_series_path).resolve()
                        is_match = False
                        try:
                            is_match = target_p.is_relative_to(series_p)
                        except Exception:
                            is_match = False

                        if is_match or series_p.name == target_p.parent.name or series_p.name == target_p.parent.parent.name:
                            s_id = series.get("sonarrSeriesId")
                            if not s_id:
                                continue
                            # Get episodes for this specific series
                            ep_res = await client.get(
                                f"{bazarr_url}/api/episodes",
                                headers=headers,
                                params={"seriesid[]": s_id}
                            )
                            if ep_res.status_code == 200:
                                try:
                                    ep_json = ep_res.json()
                                    episodes = ep_json.get("data", []) if isinstance(ep_json, dict) else (ep_json if isinstance(ep_json, list) else [])
                                except Exception:
                                    episodes = []
                                for ep in episodes:
                                    if not isinstance(ep, dict):
                                        continue
                                    ep_raw_path = ep.get("path", "")
                                    if not ep_raw_path:
                                        continue
                                    ep_p = Path(ep_raw_path).resolve()
                                    if ep_p == target_p or ep_p.name == target_p.name:
                                        e_id = ep.get("sonarrEpisodeId")
                                        if e_id:
                                            logger.info(f"Triggering Bazarr episode subtitle search for series {s_id}, episode {e_id}, lang={language}")
                                            await client.patch(
                                                f"{bazarr_url}/api/episodes/subtitles",
                                                headers=headers,
                                                params={
                                                    "seriesid": s_id,
                                                    "episodeid": e_id,
                                                    "language": language,
                                                    "forced": "False",
                                                    "hi": "False"
                                                }
                                            )
                                            return
                            break  # Found the right series, no need to continue
            except Exception as e:
                logger.warning(f"Failed to trigger Bazarr search: {e}")

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
                    from app.core.db import update_job
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

        enable_bazarr = get_setting("enable_bazarr_check", "true").lower() == "true"
        extract_target_embedded = get_setting("extract_target_embedded", "true").lower() == "true"
        extract_source_embedded = get_setting("extract_source_embedded", "true").lower() == "true"
        original_language_guard = get_setting("original_language_guard", "true").lower() == "true"
        auto_repair_unhealthy = get_setting("auto_repair_unhealthy", "true").lower() == "true"
        strict_sync_lock = get_setting("strict_sync_lock", "true").lower() == "true"
        effective_tm_key = series_title or (title.split(" - S")[0] if title and " - S" in title else title)

        start_time = time.time()
        update_job(job_id, status="TRANSLATING")
        append_job_log(job_id, f"Processing file: {video_path}")

        if not os.path.exists(video_path):
            err = f"File not found: {video_path}"
            append_job_log(job_id, f"ERROR: {err}")
            update_job(job_id, status="FAILED", error_message=err, duration_seconds=round(time.time() - start_time, 2))
            return {"status": "error", "message": err, "job_id": job_id}

        target_languages = self.get_configured_languages()
        if not target_languages:
            append_job_log(job_id, "No target language configured.")
            update_job(job_id, status="ACTION_REQUIRED")
            return {"status": "action_required", "reason": "no_target_languages"}

        base_path, _ = os.path.splitext(video_path)

        # -------------------------------------------------------------
        # Bug #11: Reordered pipeline — check LOCAL targets FIRST
        #
        # Order:
        #   1. External target on disk
        #   2. Embedded target in video
        #   3. Bazarr target search
        #   4. (If no target found) Resolve source → AI translate
        # -------------------------------------------------------------

        # Track which languages still need translation
        langs_needing_translation = []

        # Single efficient container probe per job (cached for target, source, and audio inspection)
        container_tracks: Optional[Dict[str, Any]] = None
        t_probe_ms = 0.0
        if extract_target_embedded or extract_source_embedded or original_language_guard:
            t_probe_start = time.perf_counter()
            try:
                container_tracks = await asyncio.to_thread(inspect_mkv_tracks, video_path)
            except Exception as e:
                logger.warning(f"Failed to probe container tracks for {video_path}: {e}")
                container_tracks = {"subtitles": [], "audio": []}
            t_probe_ms = round((time.perf_counter() - t_probe_start) * 1000, 1)

        if not force_retranslate:
            for lang_info in target_languages:
                lang_code = lang_info["code"]
                lang_name = lang_info["name"]
                target_output_path = f"{base_path}.{lang_code}.srt"

                # Bug #14: Use generalized find_external_subtitle
                existing_target = find_external_subtitle(video_path, lang_code)
                if existing_target:
                    if auto_repair_unhealthy:
                        health = evaluate_subtitle_health(existing_target, target_lang_code=lang_code)
                        if health.get("status") == "RED":
                            append_job_log(job_id, f"Existing {lang_name} subtitle found but unhealthy ({health['reason']}). Will re-translate.")
                            langs_needing_translation.append(lang_info)
                            continue
                    append_job_log(job_id, f"Healthy {lang_name} subtitle already exists: {os.path.basename(existing_target)}. Skipping.")
                    continue

                # Bug #13: Don't return after first embedded target — continue loop
                if extract_target_embedded:
                    if not os.path.exists(target_output_path):
                        temp_target_path = f"{target_output_path}.tmp_embed.{uuid.uuid4().hex}"
                        published = False
                        try:
                            extracted = await asyncio.to_thread(
                                _safe_extract_embedded_srt,
                                video_path,
                                temp_target_path,
                                preferred_lang=lang_code,
                                tracks_info=container_tracks
                            )
                            if extracted and os.path.exists(temp_target_path):
                                # Always validate extracted embedded target, regardless of auto_repair setting
                                health = evaluate_subtitle_health(temp_target_path, target_lang_code=lang_code)
                                status = health.get("status", "UNKNOWN")

                                if status == "GREEN":
                                    # Check if external target appeared while extraction was running
                                    concurrent_existing = find_external_subtitle(video_path, lang_code)
                                    target_to_check = concurrent_existing or (target_output_path if os.path.exists(target_output_path) else None)
                                    if target_to_check and os.path.exists(target_to_check):
                                        curr_health = evaluate_subtitle_health(target_to_check, target_lang_code=lang_code)
                                        if curr_health.get("status") == "GREEN":
                                            append_job_log(job_id, f"External healthy {lang_name} subtitle appeared during embedded extraction. Preserving external subtitle.")
                                            continue
                                    os.replace(temp_target_path, target_output_path)
                                    published = True
                                    append_job_log(job_id, f"Extracted healthy embedded {lang_name} track to {os.path.basename(target_output_path)}.")
                                    continue
                                elif status == "YELLOW":
                                    append_job_log(job_id, f"Extracted embedded {lang_name} track is YELLOW ({health.get('reason')}). Queuing for deeper QA validation.")
                                else:
                                    append_job_log(job_id, f"Extracted embedded {lang_name} track rejected: {status} ({health.get('reason', 'Unknown error')}).")
                        finally:
                            if not published and os.path.exists(temp_target_path):
                                try:
                                    os.remove(temp_target_path)
                                except Exception:
                                    pass

                # This language needs translation
                langs_needing_translation.append(lang_info)
        else:
            langs_needing_translation = list(target_languages)

        # If all languages are covered, we're done
        if not langs_needing_translation:
            duration = round(time.time() - start_time, 2)
            update_job(job_id, status="ALREADY EXISTS", reason="All target subtitles already exist", duration_seconds=duration)
            return {"status": "skipped", "reason": "already_exists", "job_id": job_id}

        # -------------------------------------------------------------
        # HYBRID MODE: Trigger Bazarr search in background (True Concurrency)
        # -------------------------------------------------------------
        prep_start_time = time.time()
        bazarr_tasks: List[asyncio.Task] = []
        if enable_bazarr and not force_retranslate:
            for lang_info in langs_needing_translation:
                append_job_log(job_id, f"Hybrid Mode: Triggering Bazarr search for {lang_info['name']} ({lang_info['code']})...")
                async def _do_search(lcode=lang_info["code"]):
                    try:
                        await self.trigger_bazarr_search(video_path, language=lcode)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.warning(f"Failed to trigger Bazarr search for {lcode}: {e}")

                t = asyncio.create_task(_do_search(), name=f"bazarr_search_{job_id}_{lang_info['code']}")
                bazarr_tasks.append(t)

        # -------------------------------------------------------------
        # LOCATE SOURCE SUBTITLE for AI translation (Fallback Prep)
        # Priority: 1. Embedded English, 2. External English, 3. Bazarr English fallback
        # -------------------------------------------------------------
        raw_srt_text = ""
        temp_extracted_srt = f"{base_path}.temp_extracted.en.srt"

        # PRIORITY 1: Embedded English inside Video (Best sync guarantee)
        t_extract_ms = 0.0
        if extract_source_embedded:
            append_job_log(job_id, "Checking video container for embedded English track (Priority 1: Best Sync)...")
            t_ext_start = time.perf_counter()
            extracted = await asyncio.to_thread(
                _safe_extract_embedded_srt,
                video_path,
                temp_extracted_srt,
                preferred_lang="eng",
                tracks_info=container_tracks
            )
            t_extract_ms = round((time.perf_counter() - t_ext_start) * 1000, 1)
            if extracted and os.path.exists(temp_extracted_srt):
                append_job_log(job_id, "Successfully extracted embedded English track from video.")
                try:
                    with open(temp_extracted_srt, "r", encoding="utf-8-sig") as f:
                        raw_srt_text = f.read()
                except UnicodeDecodeError:
                    with open(temp_extracted_srt, "r", encoding="windows-1252") as f:
                        raw_srt_text = f.read()

        # PRIORITY 2: External English subtitle on disk
        if not raw_srt_text:
            en_srt_source = find_external_subtitle(video_path, "en")
            if en_srt_source and os.path.exists(en_srt_source):
                append_job_log(job_id, f"Found external source subtitle: {os.path.basename(en_srt_source)}")
                try:
                    with open(en_srt_source, "r", encoding="utf-8-sig") as f:
                        raw_srt_text = f.read()
                except UnicodeDecodeError:
                    with open(en_srt_source, "r", encoding="windows-1252") as f:
                        raw_srt_text = f.read()

        # PRIORITY 3: Bazarr Safety Net — ask Bazarr for English subtitle
        if not raw_srt_text and enable_bazarr:
            append_job_log(job_id, "No embedded or external English sub found. Querying Bazarr for English subtitle...")
            try:
                await self.trigger_bazarr_search(video_path, language="en")
            except Exception as e:
                logger.warning(f"Failed to trigger Bazarr safety net search: {e}")

            for _ in range(8):
                await asyncio.sleep(2)
                en_srt_source = find_external_subtitle(video_path, "en")
                if en_srt_source:
                    append_job_log(job_id, f"Bazarr retrieved English subtitle: {os.path.basename(en_srt_source)}")
                    try:
                        with open(en_srt_source, "r", encoding="utf-8-sig") as f:
                            raw_srt_text = f.read()
                    except UnicodeDecodeError:
                        with open(en_srt_source, "r", encoding="windows-1252") as f:
                            raw_srt_text = f.read()
                    break

        if not raw_srt_text:
            for t in bazarr_tasks:
                if not t.done():
                    t.cancel()
            if bazarr_tasks:
                await asyncio.gather(*bazarr_tasks, return_exceptions=True)

            err = "No English subtitle source found (neither embedded, external nor via Bazarr safety net)"
            append_job_log(job_id, f"ERROR: {err}")
            duration = round(time.time() - start_time, 2)

            from app.core.db import get_job_by_id
            job_data = get_job_by_id(job_id)
            current_retries = job_data.get("retry_count", 0) if job_data else 0

            if current_retries < 4:
                backoff_mins = [1, 5, 15, 30][current_retries]
                from datetime import datetime, timezone, timedelta
                next_retry_at = (datetime.now(timezone.utc) + timedelta(minutes=backoff_mins)).isoformat()

                update_job(job_id,
                           status="WAITING_SOURCE",
                           error_message=err,
                           duration_seconds=duration,
                           retry_count=current_retries + 1,
                           next_retry_at=next_retry_at)
                if os.path.exists(temp_extracted_srt):
                    try: os.remove(temp_extracted_srt)
                    except: pass
                return {"status": "waiting_source", "reason": err, "job_id": job_id}
            else:
                update_job(job_id, status="FAILED", error_message=err, duration_seconds=duration)
                if os.path.exists(temp_extracted_srt):
                    try: os.remove(temp_extracted_srt)
                    except: pass
                return {"status": "failed", "reason": err, "job_id": job_id}

        try:
            # -------------------------------------------------------------
            # PRE-PROCESSING & SDH CLEANER
            # -------------------------------------------------------------
            t_clean_start = time.perf_counter()
            clean_sdh_enabled = get_setting("clean_sdh", "true").lower() == "true"
            if clean_sdh_enabled:
                subs, cleaned_count = sanitize_srt_content(raw_srt_text)
                append_job_log(job_id, f"Sanitizer processed {len(subs)} blocks. Cleaned noise on {cleaned_count} blocks.")
            else:
                subs = list(srt.parse(raw_srt_text))
                cleaned_count = 0
                append_job_log(job_id, f"SDH cleaner disabled. Parsed {len(subs)} blocks directly.")
            t_clean_ms = round((time.perf_counter() - t_clean_start) * 1000, 1)

            prep_duration_ms = round((time.time() - prep_start_time) * 1000, 1)
            perf_breakdown = (
                f"(probe={round(t_probe_ms / 1000, 2)}s, "
                f"extract={round(t_extract_ms / 1000, 2)}s, "
                f"clean={round(t_clean_ms / 1000, 2)}s, "
                f"audio=0.00s)"
            )

            # Ensure all background Bazarr trigger tasks are cleanly resolved/cancelled before final check
            for t in bazarr_tasks:
                if not t.done():
                    t.cancel()
            if bazarr_tasks:
                await asyncio.gather(*bazarr_tasks, return_exceptions=True)

            # -------------------------------------------------------------
            # FINAL BAZARR CHECK (Target check before initiating AI translation)
            # -------------------------------------------------------------
            if enable_bazarr and not force_retranslate:
                still_missing = []
                for lang_info in langs_needing_translation:
                    existing = find_external_subtitle(video_path, lang_info["code"])
                    if existing:
                        if auto_repair_unhealthy:
                            health = evaluate_subtitle_health(existing, target_lang_code=lang_info["code"])
                            if health.get("status") == "RED":
                                still_missing.append(lang_info)
                                continue
                        append_job_log(job_id, f"Bazarr found {lang_info['name']} subtitle: {os.path.basename(existing)}")
                    else:
                        still_missing.append(lang_info)

                langs_needing_translation = still_missing
                if not langs_needing_translation:
                    duration = round(time.time() - start_time, 2)
                    update_job(job_id, status="BAZARR MATCH", reason="Bazarr found all target subtitles", duration_seconds=duration)
                    if os.path.exists(temp_extracted_srt):
                        try: os.remove(temp_extracted_srt)
                        except Exception: pass
                    await self._maybe_notify_jellyfin()
                    return {"status": "skipped", "reason": "bazarr_downloaded", "job_id": job_id}
                else:
                    append_job_log(job_id, f"Hybrid preparation completed in {prep_duration_ms}ms {perf_breakdown}. Bazarr result: miss. Starting AI immediately (fixed grace delay avoided).")
            else:
                append_job_log(job_id, f"Source preparation completed in {round(prep_duration_ms / 1000, 2)}s {perf_breakdown}.")

            total_source_lines = len(subs)
            batch_size = get_positive_int_setting("batch_size", 50)
            ai_provider = get_setting("ai_provider", "gemini").lower()
            if ai_provider == "openai":
                active_engine_name = f"OpenAI ({get_setting('openai_model', 'gpt-4o-mini')})"
            elif ai_provider == "deepl":
                active_engine_name = "DeepL Translate"
            elif ai_provider in ["ollama", "localai"]:
                active_engine_name = f"Ollama ({get_setting('ollama_model', 'llama3')})"
            else:
                active_engine_name = f"Gemini ({get_setting('gemini_model', 'gemini-3.5-flash-lite')})"

            update_job(job_id, total_lines=total_source_lines, processed_lines=0, current_batch="Starting...")
            output_files = []
            successful_langs = []
            skipped_langs = []
            total_dropped = 0
            max_sync_diff = 0
            is_semantic_deadlock = False

            # -------------------------------------------------------------
            # TRANSLATION & QUALITY ASSURANCE
            # -------------------------------------------------------------

            # Bug #20: Respect Original Language Guard toggle
            original_language_guard = get_setting("original_language_guard", "true").lower() == "true"

            primary_audio_lang = "und"
            if original_language_guard:
                audio_tracks = container_tracks.get("audio", []) if container_tracks else []
                if not audio_tracks:
                    try:
                        mkv_info = await asyncio.to_thread(inspect_mkv_tracks, video_path)
                        audio_tracks = mkv_info.get("audio", [])
                    except Exception:
                        audio_tracks = []

                # Bug #32: Better audio track detection — prefer default, skip forced/commentary
                for audio in audio_tracks:
                    audio_title = (audio.get("title") or "").lower()
                    if any(bad in audio_title for bad in ["commentary", "director", "description"]):
                        continue
                    if audio.get("default") and not audio.get("forced"):
                        primary_audio_lang = audio.get("language", "und").lower()
                        break
                if primary_audio_lang == "und" and audio_tracks:
                    # Fallback: first non-forced, non-commentary audio
                    for audio in audio_tracks:
                        audio_title = (audio.get("title") or "").lower()
                        if not audio.get("forced") and not any(bad in audio_title for bad in ["commentary", "director"]):
                            primary_audio_lang = audio.get("language", "und").lower()
                            break

            for lang_info in langs_needing_translation:
                lang_name = lang_info["name"]
                lang_code = lang_info["code"]
                target_output_path = f"{base_path}.{lang_code}.srt"
                safe_ids = [idx for idx, sub in enumerate(subs) if is_safe_keep_prefilter(sub.content)]
                context_verified_ids = set()

                # Check Original Language Guard
                if original_language_guard:
                    from app.core.languages import get_language
                    lang_obj = get_language(lang_code)
                    protected_langs = lang_obj.aliases if lang_obj else [lang_code]

                    if primary_audio_lang in protected_langs:
                        append_job_log(job_id, f"Original Language Guard: Primary audio is '{primary_audio_lang}'. Skipping translation to {lang_name}.")
                        # Bug #31: Don't count guard-skipped as "successful translation"
                        skipped_langs.append(lang_name)
                        continue

                # Mid-job target race protection (Before Translation)
                existing_before_trans = find_external_subtitle(video_path, lang_code)
                if existing_before_trans:
                    health = evaluate_subtitle_health(existing_before_trans, target_lang_code=lang_code)
                    if health.get("status") == "GREEN":
                        append_job_log(job_id, "Target appeared while job was running. Skipping AI translation.")
                        successful_langs.append(lang_code)
                        continue

                append_job_log(job_id, f"Translating to {lang_name} ({lang_code}) using {active_engine_name} ({total_source_lines} lines)...")

                t_main_start = time.time()
                translated_subs = await self.translator.translate_srt_content(
                    subs=subs,
                    target_language=lang_name,
                    batch_size=batch_size,
                    job_id=job_id,
                    show_title=effective_tm_key or title
                )
                t_main_end = time.time()
                append_job_log(job_id, f"Timing: Main translation phase completed in {round(t_main_end - t_main_start, 1)}s")

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

                while qa_loop_count < max_qa_loops:
                    qa_loop_count += 1

                    if qa_loop_count > 1:
                        append_job_log(job_id, f"--- QA RECOVERY LOOP {qa_loop_count}/{max_qa_loops} ---")

                    # Strict Sync Lock: Match source timestamps exactly to guarantee 0ms drift
                    if strict_sync_lock and len(translated_subs) == len(subs):
                        for idx in range(len(subs)):
                            translated_subs[idx].start = subs[idx].start
                            translated_subs[idx].end = subs[idx].end

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
                        allow_warnings=False
                    )

                    if qa_loop_count == 1:
                        initial_identical_candidates_set = set(qa_result.get("untranslated_ids", []))

                    if qa_result["passed"]:
                        break  # Clean PASS! Exit the recovery loop.

                    current_unresolved_set = set(
                        qa_result.get("real_untranslated_ids", []) +
                        qa_result.get("wrong_language_ids", []) +
                        [d.get("index", d.get("id", 1)) - 1 for d in qa_result.get("dropped_details", [])]
                    )

                    # Stagnation Guard: If unresolved set is unchanged and 0 new cues were recovered, break immediately
                    if previous_unresolved_set is not None and current_unresolved_set == previous_unresolved_set and len(recovered_cues) == recovered_at_loop_start:
                        append_job_log(job_id, f"QA Recovery: Stagnation detected ({len(current_unresolved_set)} unresolved cues unchanged with 0 progress). Breaking QA recovery loop.")
                        is_semantic_deadlock = True
                        break

                    previous_unresolved_set = set(current_unresolved_set)
                    recovered_at_loop_start = len(recovered_cues)

                    # Attempt recovery for any untranslated, wrong-language or dropped lines
                    if qa_result["untranslated_ids"] or qa_result.get("dropped_count", 0) > 0 or qa_result.get("wrong_language_ids"):
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
                                    provider = get_setting("ai_provider", "gemini").lower()
                                    caps = get_provider_capabilities(provider)
                                    if caps["supports_identical_classification"]:
                                        recovery_results = []
                                        chunk_size = 20
                                        for i in range(0, len(recovery_payload), chunk_size):
                                            chunk = recovery_payload[i:i + chunk_size]
                                            try:
                                                try:
                                                    chunk_res = await self.translator.classify_and_recover_identical(
                                                        chunk, lang_name, effective_tm_key or title or "",
                                                        source_subs=subs,
                                                        translated_subs=translated_subs
                                                    )
                                                except TypeError:
                                                    chunk_res = await self.translator.classify_and_recover_identical(
                                                        chunk, lang_name, effective_tm_key or title or ""
                                                    )
                                                recovery_results.extend(chunk_res)
                                            except (ProviderUnavailableError, ProviderConfigurationError):
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
                                                is_det_safe = is_deterministically_safe_keep(subs[idx].content, raw_reason, show_title=title or "")
                                                is_ev_safe = has_entity_evidence(subs[idx].content, subs, translated_subs, target_idx=idx)
                                                is_ctx_safe = ("context_verified" in raw_reason) and is_strictly_valid_entity_candidate(subs[idx].content)
                                                if is_det_safe or is_ev_safe or is_ctx_safe:
                                                    if idx not in safe_ids:
                                                        safe_ids.append(idx)
                                                    if is_ctx_safe:
                                                        context_verified_ids.add(idx)
                                                        append_job_log(job_id, f"QA Recovery: Model kept cue {idx + 1} (Context-verified Entity: '{subs[idx].content.strip()}')")
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
                                                    translated_subs[idx].content = r["text"]
                                                    append_job_log(job_id, f"QA Recovery: Translated cue {idx + 1}")
                                                    recovered_cues.add(idx)
                                    else:
                                        # Deterministic fallback for DeepL
                                        recovery_results = []
                                        chunk_size = 20
                                        for i in range(0, len(recovery_payload), chunk_size):
                                            chunk = recovery_payload[i:i + chunk_size]
                                            try:
                                                chunk_res = await self.translator.translate_batch(
                                                    chunk, target_language=lang_name, show_title=effective_tm_key or title or ""
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
                                                translated_subs[idx].content = text
                                                append_job_log(job_id, f"QA Recovery: Translated cue {idx + 1}")
                                                recovered_cues.add(idx)

                            # Re-run QA after primary recovery with safe_ids context
                            qa_result = qa_gate(subs, translated_subs, target_lang_code=lang_code, job_id=job_id, safe_ids=safe_ids, show_title=title or "", context_verified_ids=context_verified_ids, allow_warnings=False)
                            if qa_result["passed"]:
                                break

                            # 2. Escalation Stage: Contextual Single-Line Recovery (Dropped + Unresolved Identical + Wrong Language)
                            real_unresolved = qa_result.get("real_untranslated_ids", [])
                            wrong_lang_unresolved = qa_result.get("wrong_language_ids", [])
                            dropped_unresolved = [d.get("index", d.get("id", 1)) - 1 for d in qa_result.get("dropped_details", [])]
                            all_unresolved = list(set(real_unresolved + wrong_lang_unresolved + dropped_unresolved))
                            all_unresolved.sort()

                            if all_unresolved:
                                for uid in all_unresolved:
                                    known_untranslated_ids.add(uid)

                                append_job_log(job_id, f"Targeted Recovery: translating {len(all_unresolved)} unresolved dialogue cues")
                                batch_payload = [{"id": idx, "text": subs[idx].content} for idx in all_unresolved]

                                try:
                                    targeted_results = await self.translator.translate_batch(batch_payload, target_language=lang_name, show_title=effective_tm_key or title or "")
                                    targeted_success = 0
                                    for r in targeted_results:
                                        idx = r.get("id")
                                        if idx is None: continue
                                        text = r.get("text", "")
                                        if is_meaningful_translation(subs[idx].content, text):
                                            translated_subs[idx].content = text
                                            recovered_cues.add(idx)
                                            targeted_success += 1
                                    append_job_log(job_id, f"Targeted Recovery: translated {targeted_success}/{len(all_unresolved)}")
                                    if targeted_success > 0:
                                        recovered_at_loop_start = -1  # Mark progress
                                    else:
                                        append_job_log(job_id, f"Targeted Recovery: 0/{len(all_unresolved)} cues translated.")
                                except (ProviderUnavailableError, ProviderConfigurationError):
                                    raise
                                except Exception as e:
                                    append_job_log(job_id, f"Targeted Recovery failed: {e}")

                                # Re-run QA again to get the remaining stubborn cues
                                qa_result = qa_gate(subs, translated_subs, target_lang_code=lang_code, job_id=job_id, safe_ids=safe_ids, show_title=title or "", context_verified_ids=context_verified_ids, allow_warnings=False)
                                real_unresolved = qa_result.get("real_untranslated_ids", [])
                                wrong_lang_unresolved = qa_result.get("wrong_language_ids", [])
                                dropped_unresolved = [d.get("index", d.get("id", 1)) - 1 for d in qa_result.get("dropped_details", [])]
                                all_unresolved = list(set(real_unresolved + wrong_lang_unresolved + dropped_unresolved))
                                all_unresolved.sort()

                            if all_unresolved:
                                esc_enabled = get_setting("escalate_to_pro", "false").lower() == "true"

                                esc_prov = get_setting("escalation_provider", "none")
                                esc_mod = get_setting("escalation_model", "")
                                if esc_enabled and esc_prov != "none" and esc_mod:
                                    esc_info = f"{esc_prov} / {esc_mod}"
                                else:
                                    esc_info = "Primary Model (Contextual Mode)"
                                append_job_log(job_id, f"Escalation Stage: {len(all_unresolved)} lines still unresolved. Escalating using {esc_info}...")
                                esc_sem = asyncio.Semaphore(3)
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
                                                exhausted_strategies=exhausted_strategies
                                            )
                                            is_dropped = idx in dropped_unresolved
                                            if esc_text:
                                                esc_clean = esc_text.strip()
                                                orig_clean = target_text.strip()
                                                is_orig_real = orig_clean and orig_clean != "<i></i>"
                                                is_esc_empty = not esc_clean or esc_clean == "<i></i>"
                                                if not esc_clean:
                                                    append_job_log(job_id, f"Escalation: Rejected empty text for cue {idx + 1}")
                                                elif is_orig_real and is_esc_empty:
                                                    append_job_log(job_id, f"Escalation: Rejected fake empty/tag for real dialogue at cue {idx + 1}")
                                                elif not is_meaningful_translation(target_text, esc_text):
                                                    append_job_log(job_id, f"Escalation: Rejected identical fallback for cue {idx + 1}")
                                                else:
                                                    translated_subs[idx].content = esc_text
                                                    append_job_log(job_id, f"Escalation: Translated cue {idx + 1} using dialogue context")
                                                    recovered_cues.add(idx)
                                        except (ProviderUnavailableError, ProviderConfigurationError):
                                            raise
                                        except Exception as e:
                                            append_job_log(job_id, f"Escalation failed for cue {idx + 1}: {e}")

                                esc_tasks = [escalate_one(idx) for idx in all_unresolved]
                                if esc_tasks:
                                    t_esc_start = time.time()
                                    await asyncio.gather(*esc_tasks)
                                    t_esc_end = time.time()
                                    append_job_log(job_id, f"Timing: Escalation phase ({len(esc_tasks)} cues) completed in {round(t_esc_end - t_esc_start, 1)}s")

                                # Final QA rerun after escalation
                                qa_result = qa_gate(subs, translated_subs, target_lang_code=lang_code, job_id=job_id, safe_ids=safe_ids, show_title=title or "", context_verified_ids=context_verified_ids, allow_warnings=False)
                                if qa_result["passed"]:
                                    break

                                # 3. FAST FINAL RESCUE (Batch recovery for stubborn unresolved dialogue cues)
                                real_unresolved = qa_result.get("real_untranslated_ids", [])
                                wrong_lang_unresolved = qa_result.get("wrong_language_ids", [])
                                dropped_unresolved = [d.get("index", d.get("id", 1)) - 1 for d in qa_result.get("dropped_details", [])]
                                rescue_candidate_ids = [
                                    idx for idx in sorted(set(real_unresolved + wrong_lang_unresolved + dropped_unresolved))
                                    if idx not in safe_ids and subs[idx].content.strip() and subs[idx].content.strip() != "<i></i>"
                                ]

                                if rescue_candidate_ids:
                                    total_to_rescue = len(rescue_candidate_ids)
                                    append_job_log(job_id, f"Fast Final Rescue: attempting {total_to_rescue} unresolved dialogue cues in one batch")
                                    t_rescue_total_start = time.time()

                                    def _make_rescue_items(target_ids):
                                        items = []
                                        for idx in target_ids:
                                            ctx_before_parts = []
                                            for b_idx in range(max(0, idx - 3), idx):
                                                bc = subs[b_idx].content.strip()
                                                if bc and bc != "<i></i>":
                                                    btc = translated_subs[b_idx].content.strip()
                                                    if btc and btc != "<i></i>" and is_meaningful_translation(bc, btc):
                                                        ctx_before_parts.append(f"{bc} (SV: {btc})")
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

                                    # Attempt 1
                                    rescue_items_1 = _make_rescue_items(rescue_candidate_ids)
                                    t_att1_start = time.time()
                                    try:
                                        results_1 = await self.translator.fast_final_rescue_batch(
                                            rescue_items_1,
                                            target_language=lang_name,
                                            show_title=effective_tm_key or title or "",
                                            attempt=1,
                                            job_id=job_id
                                        )
                                    except (ProviderUnavailableError, ProviderConfigurationError):
                                        raise
                                    except Exception as e:
                                        append_job_log(job_id, f"Fast Final Rescue attempt 1 failed: {e}")
                                        results_1 = []
                                    t_att1_end = time.time()

                                    rec_att1_count = 0
                                    seen_ids_1 = set()
                                    for r in results_1:
                                        if not isinstance(r, dict):
                                            continue
                                        rid = r.get("id")
                                        if rid is None or rid not in rescue_candidate_ids:
                                            continue
                                        if rid in seen_ids_1:
                                            continue
                                        seen_ids_1.add(rid)
                                        text = r.get("text", "")
                                        if is_usable_translation(text) and is_meaningful_translation(subs[rid].content, text):
                                            translated_subs[rid].content = text
                                            recovered_cues.add(rid)
                                            rec_att1_count += 1

                                    append_job_log(job_id, f"Fast Final Rescue attempt 1: recovered {rec_att1_count}/{total_to_rescue} in {round(t_att1_end - t_att1_start, 1)}s")

                                    still_unresolved_ids = [
                                        idx for idx in rescue_candidate_ids
                                        if not is_meaningful_translation(subs[idx].content, translated_subs[idx].content)
                                    ]

                                    # Attempt 2 (MAX ONE second attempt, only for remaining unresolved cues)
                                    if still_unresolved_ids:
                                        rescue_items_2 = _make_rescue_items(still_unresolved_ids)
                                        t_att2_start = time.time()
                                        try:
                                            results_2 = await self.translator.fast_final_rescue_batch(
                                                rescue_items_2,
                                                target_language=lang_name,
                                                show_title=effective_tm_key or title or "",
                                                attempt=2,
                                                job_id=job_id
                                            )
                                        except (ProviderUnavailableError, ProviderConfigurationError):
                                            raise
                                        except Exception as e:
                                            append_job_log(job_id, f"Fast Final Rescue attempt 2 failed: {e}")
                                            results_2 = []
                                        t_att2_end = time.time()

                                        rec_att2_count = 0
                                        seen_ids_2 = set()
                                        for r in results_2:
                                            if not isinstance(r, dict):
                                                continue
                                            rid = r.get("id")
                                            if rid is None or rid not in still_unresolved_ids:
                                                continue
                                            if rid in seen_ids_2:
                                                continue
                                            seen_ids_2.add(rid)
                                            text = r.get("text", "")
                                            if is_usable_translation(text) and is_meaningful_translation(subs[rid].content, text):
                                                translated_subs[rid].content = text
                                                recovered_cues.add(rid)
                                                rec_att2_count += 1

                                        append_job_log(job_id, f"Fast Final Rescue attempt 2: recovered {rec_att2_count}/{len(still_unresolved_ids)} in {round(t_att2_end - t_att2_start, 1)}s")

                                    t_rescue_total_end = time.time()
                                    final_unresolved_rescue = [
                                        idx for idx in rescue_candidate_ids
                                        if not is_meaningful_translation(subs[idx].content, translated_subs[idx].content)
                                    ]
                                    total_recovered_in_rescue = total_to_rescue - len(final_unresolved_rescue)

                                    if not final_unresolved_rescue:
                                        append_job_log(job_id, f"Fast Final Rescue completed: recovered {total_recovered_in_rescue}/{total_to_rescue} in {round(t_rescue_total_end - t_rescue_total_start, 1)}s")
                                    else:
                                        append_job_log(job_id, f"Fast Final Rescue completed: {len(final_unresolved_rescue)} cues remain unresolved")

                                # Final QA rerun after Fast Final Rescue
                                qa_result = qa_gate(subs, translated_subs, target_lang_code=lang_code, job_id=job_id, safe_ids=safe_ids, show_title=title or "", context_verified_ids=context_verified_ids, allow_warnings=False)
                                if qa_result["passed"]:
                                    break
                                else:
                                    # Clear partial state so next loop does a clean re-translate of failed batches
                                    import app.core.db
                                    data_dir = os.path.dirname(app.core.db.DB_PATH)
                                    partial_file = os.path.join(data_dir, f"job_{job_id}_{lang_code}_partial.json") if job_id else None
                                    if partial_file and os.path.exists(partial_file):
                                        try:
                                            with open(partial_file, "r", encoding="utf-8") as f:
                                                pdata = json.load(f)
                                            plines = pdata.get("lines", {})
                                            all_failed = list(set(qa_result.get("real_untranslated_ids", []) + qa_result.get("wrong_language_ids", []) + [d["index"] - 1 for d in qa_result.get("dropped_details", [])]))
                                            for uid in all_failed:
                                                if str(uid) in plines:
                                                    del plines[str(uid)]
                                            tmp_p = partial_file + ".tmp"
                                            with open(tmp_p, "w", encoding="utf-8") as f:
                                                json.dump(pdata, f, ensure_ascii=False)
                                            os.replace(tmp_p, partial_file)
                                        except Exception as e:
                                            append_job_log(job_id, f"Failed to clean partial state: {e}")

                                    # Early Stagnation / Deadlock check at end of loop:
                                    # If all unresolved cues were attempted across all recovery stages without progress, stop redundant calls now!
                                    current_unresolved_after_rescue = set(qa_result.get("real_untranslated_ids", []) + qa_result.get("wrong_language_ids", []) + [d.get("index", d.get("id", 1)) - 1 for d in qa_result.get("dropped_details", [])])
                                    if current_unresolved_after_rescue and len(recovered_cues) == recovered_at_loop_start:
                                        append_job_log(job_id, f"QA Recovery: Stagnation detected ({len(current_unresolved_after_rescue)} unresolved cues unchanged with 0 progress). Breaking QA recovery loop.")
                                        is_semantic_deadlock = True
                                        break

                                    # To prevent getting completely stuck in a loop, mark deadlock if this was the last loop
                                    if qa_loop_count == max_qa_loops:
                                        append_job_log(job_id, f"QA loop exhausted ({max_qa_loops} attempts). Stopping recovery loop for {lang_name}.")
                                        is_semantic_deadlock = True

                        except (ProviderUnavailableError, ProviderConfigurationError):
                            raise
                        except Exception as e:
                            append_job_log(job_id, f"QA Recovery/Escalation failed: {e}")

                # ---------------------------------------------------------------------------
                # POST-RECOVERY EVALUATION & QA FALLBACK
                # If bounded recovery is exhausted and some unresolved English cues remain:
                # Check if they meet semantic deadlock criteria and apply source preservation fallback.
                # ---------------------------------------------------------------------------
                real_unresolved_remaining = qa_result.get("real_untranslated_ids", [])
                if real_unresolved_remaining:
                    is_semantic_deadlock = True
                    for cue_idx in real_unresolved_remaining:
                        cue_id = subs[cue_idx].index if hasattr(subs[cue_idx], 'index') and subs[cue_idx].index else cue_idx + 1
                        append_job_log(job_id, f"Semantic deadlock detected for cue {cue_id}")
                        append_job_log(job_id, "QA fallback: preserving original source text")
                        translated_subs[cue_idx].content = subs[cue_idx].content
                        source_preserved_cues.add(cue_idx)

                # Strict Sync Lock: guarantee 0ms drift
                if strict_sync_lock and len(translated_subs) == len(subs):
                    for idx in range(len(subs)):
                        translated_subs[idx].start = subs[idx].start
                        translated_subs[idx].end = subs[idx].end

                # Final QA Evaluation against central QA policy (allow_warnings=True)
                qa_result = qa_gate(
                    subs,
                    translated_subs,
                    target_lang_code=lang_code,
                    job_id=job_id,
                    safe_ids=safe_ids,
                    show_title=title or "",
                    context_verified_ids=context_verified_ids,
                    allow_warnings=True
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
                    pub_res = _publish_subtitle_atomic(
                        video_path=video_path,
                        target_output_path=target_output_path,
                        lang_code=lang_code,
                        translated_srt_text=translated_srt_text,
                        expected_cue_count=len(translated_subs),
                        force_retranslate=force_retranslate,
                        job_id=job_id,
                    )

                    if pub_res.get("published"):
                        output_files.append(target_output_path)
                        successful_langs.append(lang_code)
                    elif pub_res.get("skipped"):
                        successful_langs.append(lang_code)

                    # Clean up partial progress file since we successfully finished
                    import app.core.db
                    data_dir = os.path.dirname(app.core.db.DB_PATH)
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
                                    save_translation_memory_bulk(effective_tm_key, tm_items)
                            except Exception as e:
                                logger.error(f"Failed to save translation memory: {e}")

                # Build a detailed QA summary for the user
                unresolved_count = len(qa_result.get("real_untranslated_ids", []))
                initial_candidates_count = len(initial_identical_candidates_set)
                kept_count = sum(1 for idx in initial_identical_candidates_set if idx in safe_ids and idx not in qa_result.get("real_untranslated_ids", []) and idx not in recovered_cues)
                recovered_count = len(recovered_cues) if 'recovered_cues' in locals() else 0
                source_preserved_count = len(source_preserved_cues)

                summary_status = qa_result.get("status", "PASS" if qa_result["passed"] else "FAIL")
                unresolved_label = f"{unresolved_count} unresolved English {'line' if unresolved_count == 1 else 'lines'}"
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

                if not qa_result["passed"]:
                    append_job_log(job_id, f"BLOCKED: {lang_name} translation failed QA. File NOT published.")

            # Clean up temp file
            if os.path.exists(temp_extracted_srt):
                try: os.remove(temp_extracted_srt)
                except: pass

            total_duration = round(time.time() - start_time, 2)

            # Determine correct final status
            if len(successful_langs) + len(skipped_langs) == len(langs_needing_translation):
                if successful_langs:
                    final_status = "TRANSLATED"
                else:
                    final_status = "SKIPPED"
            elif successful_langs:
                final_status = "PARTIAL"
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
            if final_status == "FAILED":
                unresolved_count = len(qa_result.get("real_untranslated_ids", []))
                if is_semantic_deadlock:
                    update_args["error_message"] = f"Semantic deadlock: {unresolved_count} unresolved cues failed QA. Bounded recovery exhausted."
                    append_job_log(job_id, f"PERMANENT FAILURE: Semantic deadlock detected. Provider succeeded but bounded recovery exhausted with 0 progress. File NOT published.")
                else:
                    update_args["error_message"] = f"QA Gate failed: {unresolved_count} unresolved cues. File NOT published."
                    append_job_log(job_id, f"PERMANENT FAILURE: QA Gate failed with {unresolved_count} unresolved cues. File NOT published.")
                update_args["next_retry_at"] = None
            elif final_status in ["RECOVERING", "PARTIAL"]:
                from datetime import datetime, timezone, timedelta
                from app.core.db import get_job_by_id
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

            await self._maybe_notify_jellyfin()

            return {
                "status": final_status.lower(),
                "job_id": job_id,
                "duration": total_duration,
                "output_files": output_files
            }

        except ProviderConfigurationError as e:
            if os.path.exists(temp_extracted_srt):
                try: os.remove(temp_extracted_srt)
                except: pass
            total_duration = round(time.time() - start_time, 2)
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
            if os.path.exists(temp_extracted_srt):
                try: os.remove(temp_extracted_srt)
                except: pass
            total_duration = round(time.time() - start_time, 2)

            from app.core.db import get_job_by_id
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
            if os.path.exists(temp_extracted_srt):
                try: os.remove(temp_extracted_srt)
                except: pass
            total_duration = round(time.time() - start_time, 2)

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
