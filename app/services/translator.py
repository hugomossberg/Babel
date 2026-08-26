from app.core.ai_providers import ProviderContext, resolve_job_provider_context
import asyncio
import json
import logging
import functools
import unicodedata
import re
import contextvars
from typing import List, Optional, Dict, Any, Tuple, Set

import threading

logger = logging.getLogger("babel.translator")

# ---------------------------------------------------------------------------
# Phase 2: Token metadata propagation from threadpool workers to asyncio tasks.
#
# Problem: translate_batch_gemini uses loop.run_in_executor → threadpool.
# asyncio ContextVars are task-local; mutations in thread copies do NOT
# propagate back to the asyncio task that spawned them.
#
# Solution: Two-tier approach:
#   1. _usage_token_ctx (ContextVar[str|None]): holds the current request_uid.
#      with_retry sets this in asyncio-task context. Threads see a snapshot of
#      this value (via run_in_executor's context copy) — but READS ONLY.
#   2. _USAGE_TOKEN_STORE (dict): a module-level thread-safe store mapping
#      request_uid → token metadata dict. Threads WRITE here after SDK call.
#      with_retry READS here after await func() completes.
#
# This ensures:
#   - Each retry attempt starts with a fresh request_uid → no stale data.
#   - Concurrent asyncio tasks use different uids → no cross-task contamination.
#   - Thread writes are visible to the asyncio task (shared mutable dict).
#   - Entries are removed immediately after read to avoid unbounded growth.
# ---------------------------------------------------------------------------

_usage_token_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_usage_token_ctx", default=None
)
_USAGE_TOKEN_STORE: Dict[str, Dict[str, Optional[int]]] = {}
_USAGE_TOKEN_STORE_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# SDK dispatch-started signal.
# Threads set _mark_sdk_started(uid) IMMEDIATELY BEFORE the provider SDK call.
# with_retry checks _pop_sdk_started(uid) to determine whether the exception
# happened before or after the actual provider network attempt:
#   - _pop_sdk_started returns True  → actual SDK invocation → create FAILED row
#   - _pop_sdk_started returns False → pre-SDK failure (config/setup) → zero rows
# ---------------------------------------------------------------------------
_SDK_DISPATCH_STARTED: Dict[str, bool] = {}
_SDK_DISPATCH_STARTED_LOCK = threading.Lock()


def _mark_sdk_started(request_uid: Optional[str]) -> None:
    """Thread-safe signal: actual provider SDK call is about to begin."""
    if not request_uid:
        return
    with _SDK_DISPATCH_STARTED_LOCK:
        _SDK_DISPATCH_STARTED[request_uid] = True


def _pop_sdk_started(request_uid: Optional[str]) -> bool:
    """Thread-safe read + delete. Returns True if SDK was invoked, False otherwise."""
    if not request_uid:
        return False
    with _SDK_DISPATCH_STARTED_LOCK:
        return _SDK_DISPATCH_STARTED.pop(request_uid, False)


def _store_token_meta(request_uid: Optional[str], meta: Dict[str, Optional[int]]) -> None:
    """Thread-safe write of token metadata keyed by request_uid."""
    if not request_uid:
        return
    with _USAGE_TOKEN_STORE_LOCK:
        _USAGE_TOKEN_STORE[request_uid] = meta


def _pop_token_meta(request_uid: Optional[str]) -> Optional[Dict[str, Optional[int]]]:
    """Thread-safe read + delete of token metadata. Returns None if not found."""
    if not request_uid:
        return None
    with _USAGE_TOKEN_STORE_LOCK:
        return _USAGE_TOKEN_STORE.pop(request_uid, None)

from app.core.quota import QuotaError, DailyQuotaExhaustedError, RequestBudgetExhaustedError

class ProviderUnavailableError(Exception):
    """Transient error: rate limit, network timeout, 5xx server error."""
    pass

class ProviderConfigurationError(Exception):
    """Permanent error: 401 Unauthorized, 403 Forbidden, invalid API key, invalid model name."""
    pass

def is_usable_translation(text) -> bool:
    if text is None:
        return False
    val = str(text).strip()
    if val == "":
        return False
    if val == "<i></i>":
        return False
    return True

def with_retry(func_or_provider=None, **decorator_kwargs):
    """
    Decorator for AI provider API calls. Responsibilities:
    1. Single-Flight Quota & Circuit Breaker Gate:
       - Checks provider / model scope state in SQLite.
       - If ACTIVE: consumes 1 local request budget slot.
       - If BLOCKED (before next_probe_at): raises DailyQuotaExhaustedError (0 API calls).
       - If HALF_OPEN (after next_probe_at): atomically claims the single-flight probe lease.
         Any competing callers receive DailyQuotaExhaustedError (0 API calls).
    2. Atomic request budget gate (consumes exactly 1 slot per actual request and per retry).
    3. Transient RPM / network error backoff and retries.
    4. Immediate propagation and recording of daily quota failures and permanent configuration errors.
    5. Automatic reset of circuit breaker to ACTIVE upon successful probe / response.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            from app.core.quota import (
                acquire_dispatch_slot, record_provider_success, record_provider_quota_exhausted,
                get_daily_budget, get_daily_requests_used,
                DailyQuotaExhaustedError, RequestBudgetExhaustedError,
                classify_provider_error, extract_retry_after_from_exception,
            )

            # Determine provider
            explicit_provider = decorator_kwargs.get("provider")
            if not explicit_provider and isinstance(func_or_provider, str):
                explicit_provider = func_or_provider

            if explicit_provider:
                provider = explicit_provider.lower()
            elif "provider" in kwargs and kwargs["provider"]:
                provider = str(kwargs["provider"]).lower()
            elif func.__name__.endswith("_gemini"):
                provider = "gemini"
            elif func.__name__.endswith("_openai"):
                provider = "openai"
            elif func.__name__.endswith("_anthropic"):
                provider = "anthropic"
            elif func.__name__.endswith("_openrouter"):
                provider = "openrouter"
            elif func.__name__.endswith("_deepseek"):
                provider = "deepseek"
            elif func.__name__.endswith("_custom"):
                provider = "custom"
            elif func.__name__.endswith("_deepl"):
                provider = "deepl"
            elif func.__name__.endswith("_ollama"):
                provider = "ollama"
            else:
                try:
                    from app.core.ai_providers import context_from_settings, normalize_provider
                    # Check if provider_ctx or job_id are in kwargs for pinned resolution
                    _pctx = kwargs.get("provider_ctx")
                    _jid = kwargs.get("job_id")
                    if _pctx is not None:
                        provider = normalize_provider(_pctx.provider)
                    elif _jid is not None:
                        from app.core.ai_providers import resolve_job_provider_context
                        _ctx = resolve_job_provider_context(_jid)
                        provider = _ctx.provider
                    else:
                        _ctx = context_from_settings()
                        provider = _ctx.provider
                except Exception:
                    provider = "gemini"

            model_name = kwargs.get("model_name") or kwargs.get("model") or decorator_kwargs.get("model")
            # Fallback: if model_name not in kwargs (e.g. passed as positional arg),
            # resolve from pinned job or settings using central registry.
            if not model_name:
                try:
                    from app.core.ai_providers import context_from_settings, resolve_job_provider_context
                    _pctx = kwargs.get("provider_ctx")
                    _jid = kwargs.get("job_id")
                    if _pctx is not None:
                        model_name = _pctx.model
                    elif _jid is not None:
                        _ctx = resolve_job_provider_context(_jid)
                        model_name = _ctx.model
                    else:
                        _ctx = context_from_settings(provider)
                        model_name = _ctx.model
                except Exception:
                    pass
            job_id = kwargs.get("job_id")

            retries = 3
            backoffs = [5, 15, 30]

            for attempt in range(retries + 1):
                # 1. Single-Flight Quota & Circuit Breaker Gate
                # BLOCKED → zero usage rows (Phase 2 accounting invariant)
                allowed, dispatch_info = acquire_dispatch_slot(
                    provider=provider,
                    model=model_name,
                    job_id=job_id,
                )
                if not allowed:
                    if dispatch_info.get("reason") == "REQUEST_BUDGET_EXHAUSTED":
                        budget = get_daily_budget(provider)
                        used = get_daily_requests_used(provider)
                        raise RequestBudgetExhaustedError(provider=provider, used=used, budget=budget or 0)
                    else:
                        blocked_until = dispatch_info.get("blocked_until")
                        reset_type = dispatch_info.get("reset_type", "estimated")
                        reason = dispatch_info.get("reason", "Daily quota exhausted")
                        raise DailyQuotaExhaustedError(
                            provider=provider,
                            retry_after_seconds=None,
                            raw_message=f"Provider '{provider}' circuit breaker {dispatch_info.get('state', 'BLOCKED')} (blocked until {blocked_until}, reset_type={reset_type}). Reason: {reason}",
                        )

                # ---------------------------------------------------------------
                # PHASE 2 USAGE ACCOUNTING — authoritative dispatch boundary.
                # acquire_dispatch_slot returned True → the SDK call WILL be made.
                # Insert PENDING row BEFORE the call so failed attempts are also recorded.
                # Invariants:
                #   - _request_uid is set ONLY when a fresh PENDING row was inserted.
                #   - If record_dispatch returns False (duplicate uid or DB error), we
                #     clear _request_uid so complete_dispatch is never called on a stale row.
                #   - acquire_dispatch_slot does NOT consume quota if it returns False.
                #   - The usage ledger does NOT affect quota counting.
                # ---------------------------------------------------------------
                _request_uid = None
                _model_for_usage = model_name or ""
                _stage_for_usage = _infer_usage_stage(func.__name__, kwargs)
                try:
                    from app.core.usage import (
                        generate_request_uid, record_dispatch, complete_dispatch,
                        cancel_dispatch, calculate_estimated_cost, UsageStatus,
                    )
                    _candidate_uid = generate_request_uid()
                    _inserted = record_dispatch(
                        request_uid=_candidate_uid,
                        provider=provider,
                        model=_model_for_usage,
                        stage=_stage_for_usage,
                        job_id=job_id,
                    )
                    if _inserted:
                        # Row created — bind _request_uid only on confirmed insert
                        _request_uid = _candidate_uid
                    else:
                        # INSERT OR IGNORE rejected (duplicate uid is theoretically impossible
                        # with uuid4, but we handle it defensively). Do not set _request_uid,
                        # which means complete_dispatch will NOT be called below.
                        logger.warning(
                            "Usage ledger: record_dispatch returned False for uid=%s "
                            "(idempotency guard or DB error). Accounting row skipped for this attempt.",
                            _candidate_uid,
                        )
                except Exception as _acc_err:
                    logger.warning("Usage ledger: pre-dispatch record failed (non-fatal): %s", _acc_err)
                    # _request_uid remains None → complete_dispatch never called.

                try:
                    # Publish current request_uid to ContextVar so threads can look up
                    # the store key. Also clears any stale entries from a previous attempt.
                    _usage_token_ctx.set(_request_uid)
                    if _request_uid:
                        _pop_token_meta(_request_uid)  # Discard any stale entry for this uid
                        _pop_sdk_started(_request_uid)  # Discard any stale sdk-started flag

                    result = await func(*args, **kwargs)
                    # 2. SUCCESS: Reset circuit breaker to ACTIVE
                    record_provider_success(provider=provider, model=model_name)

                    # Clean up sdk-started flag (set by _mark_sdk_started inside provider thread)
                    _pop_sdk_started(_request_uid)

                    # Usage accounting: read token metadata from shared store
                    # (written by _capture_gemini_tokens / _capture_openai_tokens in the thread)
                    if _request_uid:
                        try:
                            token_meta = _pop_token_meta(_request_uid) or {
                                "input_tokens": None,
                                "cached_input_tokens": None,
                                "output_tokens": None,
                                "thinking_tokens": None,
                            }
                            est_cost = calculate_estimated_cost(
                                provider=provider,
                                model=_model_for_usage,
                                input_tokens=token_meta.get("input_tokens"),
                                cached_input_tokens=token_meta.get("cached_input_tokens"),
                                output_tokens=token_meta.get("output_tokens"),
                                thinking_tokens=token_meta.get("thinking_tokens"),
                            )
                            complete_dispatch(
                                request_uid=_request_uid,
                                status=UsageStatus.SUCCESS,
                                input_tokens=token_meta.get("input_tokens"),
                                cached_input_tokens=token_meta.get("cached_input_tokens"),
                                output_tokens=token_meta.get("output_tokens"),
                                thinking_tokens=token_meta.get("thinking_tokens"),
                                estimated_cost_usd=est_cost,
                            )
                        except Exception as _acc_err:
                            logger.warning("Usage ledger: post-dispatch accounting failed: %s", _acc_err)

                    return result
                except Exception as e:
                    # Determine if the actual SDK call was attempted.
                    # _mark_sdk_started() is called inside provider threads immediately
                    # BEFORE client.models.generate_content / client.chat.completions.create.
                    # If SDK was NOT reached (pre-SDK exception: missing API key, client init,
                    # prompt build failure), cancel the PENDING row → zero usage rows.
                    _sdk_was_called = _pop_sdk_started(_request_uid)

                    def _cancel_or_fail(err_type: str) -> None:
                        """Cancel PENDING row if pre-SDK, else mark FAILED."""
                        if not _request_uid:
                            return
                        try:
                            if _sdk_was_called:
                                complete_dispatch(
                                    request_uid=_request_uid,
                                    status=UsageStatus.FAILED,
                                    error_type=err_type,
                                )
                            else:
                                cancel_dispatch(_request_uid)
                        except Exception as _acc_err:
                            logger.warning(
                                "Usage ledger: accounting error (non-fatal): %s", _acc_err
                            )

                    # Re-raise quota/budget errors immediately — never retry
                    if isinstance(e, (DailyQuotaExhaustedError, RequestBudgetExhaustedError)):
                        _cancel_or_fail(type(e).__name__)
                        raise

                    # Re-raise already-classified errors immediately
                    if isinstance(e, (ProviderConfigurationError, ProviderUnavailableError)):
                        _cancel_or_fail(type(e).__name__)
                        raise

                    retry_after = extract_retry_after_from_exception(e)
                    signal = classify_provider_error(e, provider, model=model_name, retry_after_seconds=retry_after)

                    if signal == "AUTH_ERROR":
                        _cancel_or_fail("AUTH_ERROR")
                        raise ProviderConfigurationError(
                            f"Permanent provider configuration error in {func.__name__}: {str(e)}"
                        )

                    if signal == "DAILY_QUOTA_EXHAUSTED":
                        _cancel_or_fail("DAILY_QUOTA_EXHAUSTED")
                        # Record failure and transition circuit breaker to BLOCKED
                        record_provider_quota_exhausted(
                            provider=provider,
                            reason=str(e),
                            retry_after_seconds=signal.retry_after_seconds or retry_after,
                            model=model_name,
                            scope_type=signal.scope_type,
                            scope_id=signal.scope_id,
                        )
                        raise DailyQuotaExhaustedError(
                            provider=provider,
                            retry_after_seconds=signal.retry_after_seconds or retry_after,
                            raw_message=str(e),
                        )

                    if signal == "PERMANENT_REQUEST_ERROR":
                        _cancel_or_fail("PERMANENT_REQUEST_ERROR")
                        raise ProviderConfigurationError(
                            f"Permanent request error in {func.__name__}: {str(e)}"
                        )

                    # TRANSIENT_RPM or PROVIDER_UNAVAILABLE: short backoff + retry
                    recoverable = signal in ("TRANSIENT_RPM", "PROVIDER_UNAVAILABLE")
                    if not recoverable or attempt == retries:
                        _cancel_or_fail(str(signal) if signal else type(e).__name__)
                        if recoverable:
                            raise ProviderUnavailableError(
                                f"Provider unavailable after {retries} retries: {str(e)}"
                            )
                        raise e

                    # Transient failure on this attempt — cancel/fail and let loop create a NEW row on retry
                    _cancel_or_fail(str(signal) if signal else type(e).__name__)
                    _request_uid = None  # Reset: next iteration generates a new request_uid

                    wait_time = backoffs[attempt]
                    logger.warning(
                        "Transient provider error in %s (Attempt %d/%d, provider=%s, class=%s): %s. Waiting %ds...",
                        func.__name__, attempt + 1, retries, provider, signal, e, wait_time,
                    )
                    await asyncio.sleep(wait_time)

        return wrapper

    if callable(func_or_provider):
        return decorator(func_or_provider)
    return decorator


from google import genai
from google.genai import types
import openai
import httpx
import srt

from app.core.db import DB_PATH, get_setting, update_job, append_job_log, save_translation_memory, get_translation_memory, get_positive_int_setting

def _infer_usage_stage(func_name: str, kwargs: dict) -> str:
    """
    Infer the UsageStage for a with_retry-decorated dispatch based on the function name.
    Stage describes WHY the request was made, not which provider handled it.

    Explicit override: pass _usage_stage in kwargs to override (not used in current code
    but available for future callers without requiring function signature changes).

    Mapping:
      translate_batch_gemini / _openai / _deepl / _ollama → PRIMARY
      first_pass_micro_repair_batch                       → MICRO_REPAIR
      fast_final_rescue_batch                             → RECOVERY
      classify_and_recover_identical                      → RECOVERY
      _execute_single_escalation_call                     → ESCALATION
      verify_single_occurrence_entities                   → ENTITY_VERIFY
      (any other)                                         → PRIMARY (safe default)
    """
    from app.core.usage import UsageStage

    # Allow explicit kwarg override
    explicit = kwargs.get("_usage_stage")
    if explicit:
        return str(explicit)

    fn = func_name.lower()
    if "repair_alignment" in fn or "alignment_repair" in fn or "repair_batch" in fn:
        return UsageStage.RECOVERY
    if "audit" in fn or "alignment" in fn or "confirm_batch" in fn or "verify_repaired_batch" in fn:
        return UsageStage.SEMANTIC_AUDIT
    if "translate_batch" in fn:
        return UsageStage.PRIMARY
    if "micro_repair" in fn:
        return UsageStage.MICRO_REPAIR
    if "rescue" in fn:
        return UsageStage.RECOVERY
    if "recover" in fn:
        return UsageStage.RECOVERY
    if "escalation" in fn or "escalate" in fn:
        return UsageStage.ESCALATION
    if "entity" in fn or "verify" in fn:
        return UsageStage.ENTITY_VERIFY
    if "classif" in fn:
        return UsageStage.CLASSIFIER
    # Default: treat as primary translation attempt
    return UsageStage.PRIMARY


def _capture_tokens(provider: str, response: Any) -> Any:
    """
    Extract token usage from provider response (dict or SDK object) and write to the
    module-level token store keyed by current request_uid.
    """
    try:
        from app.core.usage import extract_usage_from_response
        request_uid = _usage_token_ctx.get(None)
        if request_uid:
            token_meta = extract_usage_from_response(provider, response)
            _store_token_meta(request_uid, token_meta)
    except Exception:
        pass
    return response


def _capture_gemini_tokens(response: Any) -> Any:
    return _capture_tokens("gemini", response)


def _capture_openai_tokens(response: Any) -> Any:
    return _capture_tokens("openai", response)


def is_safe_keep_prefilter(text: str) -> bool:
    """
    Fail-closed deterministic pre-filter for cues that are provably safe to KEEP unchanged
    before sending to AI translation.
    Matches pure numbers/timestamps, symbols, empty/formatting tags, and strict acronyms
    according to existing policy.
    Real dialogue is NEVER filtered.
    """
    if text is None:
        return True
    clean_text = str(text).strip()
    clean_text = re.sub(r'<[^>]+>', '', clean_text)
    clean_text = re.sub(r'\{[^}]+\}', '', clean_text).strip()
    if not clean_text or clean_text == "<i></i>":
        return True

    has_letters = any(c.isalpha() for c in clean_text)
    has_digits = any(c.isdigit() for c in clean_text)

    # 1. Pure symbols / non-verbal cues (no letters, no digits)
    if not has_letters and not has_digits:
        return True

    # 2. Pure numbers / timestamps / measurements (has digits, but NO letters)
    if has_digits and not has_letters:
        return True

    # 3. Strict Acronyms (e.g. "FBI", "NASA", "DNA")
    if is_deterministically_safe_keep(clean_text, "acronym"):
        return True

    return False

def get_system_instruction(target_language: str, glossary: str = "", show_title: str = "", source_language: str = "English") -> str:
    glossary_section = ""
    if glossary and glossary.strip():
        glossary_section = "\n\nGLOSSARY - Always use these exact translations:\n" + glossary.strip() + "\n"

    show_context = ""
    if show_title:
        show_context = f"\nYou are translating subtitles for: \"{show_title}\". Adapt tone and terminology accordingly.\n"

    return f"""You are a professional film/TV subtitle translator translating from {source_language} to {target_language}.{show_context}
Translate the numbered subtitle blocks to natural, idiomatic {target_language}.
{glossary_section}
STRICT RULES:
1. Translate accurately and idiomatically into natural {target_language}. Translate every real dialogue line into {target_language}; do not copy {source_language} dialogue unchanged.
2. Do not classify or explain. Return translations only.
3. Preserve character names and proper nouns where appropriate, but translate all surrounding dialogue naturally.
4. Preserve all subtitle formatting tags exactly as they appear (e.g. <i>, </i>, and ASS tags like {{\\an8}}). Do not encode them into HTML entities.
5. If a block is empty or contains only '<i></i>', keep it exactly as '<i></i>'.
6. If a line starts with a speaker name or label in capital letters (e.g. ALICE:, OFFICER:), keep character names intact and only translate descriptive titles if appropriate, maintaining the colon separator.
7. You MUST return a JSON object with a key "translations" containing the array of objects with integer "id" and string "text".
8. Keep translations concise and natural. Split lines naturally using "\\n" if a line exceeds 42 characters, but NEVER exceed 2 lines per subtitle block. Combine or condense text if necessary.
9. STRICT 1-TO-1 CUE SYNCHRONIZATION & ID CONTRACT:
Every input item has an integer "id" tied to exact video timestamps. You MUST return EXACTLY ONE translation object for EVERY input item in the array.
You MUST preserve each input item's exact "id" in ascending numerical order.
NEVER skip an ID, never invent IDs, and never shift subsequent lines into earlier or later IDs.
NEVER merge two or more cues into a single ID.
Even if an item is a single word, exclamation, or '<i></i>', you MUST emit an object for its exact ID.
If a sentence or thought is split across multiple cues, translate ONLY the corresponding fragment into each respective cue ID.
Example:
Input: [{{"id": 10, "text": "And I got a really bad rash"}}, {{"id": 11, "text": "from the pony."}}]
Correct Output: {{"translations": [{{"id": 10, "text": "Och jag fick ett rejält utslag"}}, {{"id": 11, "text": "av ponnyn."}}]}}
"""

def extract_json_safely(raw_text: str) -> List[dict]:
    """Robust JSON parser that extracts translations even if formatting has minor hiccups."""
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        lines = raw_text.split('\n')
        if lines[0].startswith("```"): lines = lines[1:]
        if lines and lines[-1].startswith("```"): lines = lines[:-1]
        raw_text = "\n".join(lines).strip()

    # 1. Direct parse
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            if "translations" in data and isinstance(data["translations"], list):
                return data["translations"]
            if "results" in data and isinstance(data["results"], list):
                return data["results"]
            if "items" in data and isinstance(data["items"], list):
                return data["items"]
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # Repair common JSON issues without stripping \t, \n, \r
    try:
        repaired_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]+', '', raw_text)
        repaired_text = re.sub(r',\s*([\]}])', r'\1', repaired_text)
        data = json.loads(repaired_text)
        if isinstance(data, dict):
            if "translations" in data and isinstance(data["translations"], list):
                return data["translations"]
            if "results" in data and isinstance(data["results"], list):
                return data["results"]
            if "items" in data and isinstance(data["items"], list):
                return data["items"]
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # 2. Regex match for "translations": [...] or "results": [...] or "items": [...]
    match = re.search(r'"(?:translations|results|items)"\s*:\s*(\[[\s\S]*?\])\s*\}?', raw_text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # 3. Fallback: individual regex search
    items = []
    pattern = re.compile(r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"text"\s*:\s*"(.*?)"\s*\}', re.DOTALL)
    for m in pattern.finditer(raw_text):
        try:
            item_id = int(m.group(1))
            item_text = m.group(2)
            items.append({"id": item_id, "text": item_text})
        except Exception:
            pass

    if items:
        return items

    raise ValueError(f"Could not extract JSON translation array from response: {raw_text[:100]}...")


def build_translation_prompt(items: List[dict], target_language: str, context_section: str = "") -> str:
    """
    Builds the authoritative 1-to-1 contiguous-ID translation prompt contract.
    """
    first_id = items[0]["id"] if items else 0
    last_id = items[-1]["id"] if items else 0
    return (
        f"Translate the following {len(items)} subtitle lines into {target_language}.\n"
        f"INPUT CONTRACT: Input items have IDs from {first_id} to {last_id} (total {len(items)} items).\n"
        f"You MUST return exactly {len(items)} translation objects in \"translations\", one for each input item, "
        f"strictly preserving the exact matching integer \"id\" for each item in ascending numerical order.\n"
        f"CRITICAL REQUIREMENT: Do NOT combine adjacent lines, do NOT omit short utterances or <i></i> lines. "
        f"Every single input ID from {first_id} to {last_id} must have exactly one corresponding translation object in numerical order."
        f"{context_section}\n\n"
        + json.dumps(items, ensure_ascii=False)
    )


def build_translation_output_schema(items: List[dict], target_language: str = "Swedish") -> dict:
    """
    Builds the canonical structured output schema for translation batches.
    """
    first_id = items[0]["id"] if items else 0
    last_id = items[-1]["id"] if items else 0
    return {
        "type": "OBJECT",
        "properties": {
            "translations": {
                "type": "ARRAY",
                "description": f"Array of exactly {len(items)} translated subtitle items matching each input item ID from {first_id} to {last_id}",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "id": {"type": "INTEGER", "description": "The exact integer ID from the input item"},
                        "text": {"type": "STRING", "description": f"The translated {target_language} subtitle text, preserving tags or exact '<i></i>'"}
                    },
                    "required": ["id", "text"]
                }
            }
        },
        "required": ["translations"]
    }


def validate_batch_translation_results(
    expected_items: List[dict],
    raw_results: Any
) -> Tuple[Dict[int, str], Dict[str, Any]]:
    """
    Deterministically validates provider translation output against expected items.

    IMPORTANT ARCHITECTURAL DISTINCTION:
    This function verifies the STRUCTURAL EXACT-ID CONTRACT only (syntax validity, presence
    of all expected IDs, no unknown/duplicate/malformed IDs).
    A report with is_clean=True (or is_structurally_clean=True) guarantees ONLY that the JSON
    schema and integer ID set are intact; it does NOT imply that the semantic source->target
    mapping is verified or trustworthy. Semantic alignment is independently audited.

    Rules enforced:
    1. Expected IDs must be respected. Unknown IDs are rejected.
    2. Duplicate IDs are rejected/flagged.
    3. Missing IDs are identified for downstream targeted recovery (never shifted).
    4. Mapping is strictly by ID (array order is never authoritative).
    5. Structural validity: each entry must be a dict with integer 'id' and string 'text'.

    Returns:
        (valid_translations_dict, validation_report)
    """
    expected_ids = {it["id"] for it in expected_items if isinstance(it, dict) and "id" in it}
    expected_map = {it["id"]: it["text"] for it in expected_items if isinstance(it, dict) and "id" in it}

    valid_map: Dict[int, str] = {}
    seen_ids: Set[int] = set()
    duplicate_ids: Set[int] = set()
    unknown_ids: Set[int] = set()
    malformed_count = 0

    if isinstance(raw_results, list):
        for r in raw_results:
            if not isinstance(r, dict) or "id" not in r or "text" not in r:
                malformed_count += 1
                continue
            rid = r["id"]
            if not isinstance(rid, int):
                try:
                    rid = int(rid)
                except (ValueError, TypeError):
                    malformed_count += 1
                    continue
            if rid not in expected_ids:
                unknown_ids.add(rid)
                continue
            if rid in seen_ids:
                duplicate_ids.add(rid)
                continue
            seen_ids.add(rid)
            cand = r["text"]
            if is_meaningful_translation(expected_map[rid], cand):
                valid_map[rid] = cand

    missing_ids = expected_ids - seen_ids
    is_clean = len(missing_ids) == 0 and len(unknown_ids) == 0 and len(duplicate_ids) == 0 and malformed_count == 0
    report = {
        "expected_count": len(expected_ids),
        "received_count": len(seen_ids),
        "valid_count": len(valid_map),
        "missing_ids": sorted(list(missing_ids)),
        "unknown_ids": sorted(list(unknown_ids)),
        "duplicate_ids": sorted(list(duplicate_ids)),
        "malformed_count": malformed_count,
        "is_clean": is_clean,
        "is_structurally_clean": is_clean
    }
    return valid_map, report

def validate_recovery_batch_results(
    expected_items: List[dict],
    raw_results: Any
) -> Tuple[Dict[int, str], Dict[str, Any]]:
    """
    Deterministically validates provider recovery output against expected items.

    Rules enforced:
    1. Expected IDs must be respected. Unknown IDs are rejected.
    2. Duplicate IDs are rejected/flagged.
    3. Missing IDs are identified (never shifted).
    4. Mapping is strictly by ID (array order is never authoritative).
    5. Numeric string IDs are normalized to integer ("336" -> 336).
    6. No index-base conversion or +/- 1 guessing is allowed.
    """
    expected_ids = {it["id"] for it in expected_items if isinstance(it, dict) and "id" in it}

    valid_map: Dict[int, str] = {}
    seen_ids: Set[int] = set()
    duplicate_ids: Set[int] = set()
    unknown_ids: Set[int] = set()
    malformed_count = 0

    if isinstance(raw_results, list):
        for r in raw_results:
            if not isinstance(r, dict) or "id" not in r or "text" not in r:
                malformed_count += 1
                continue
            rid = r["id"]
            if not isinstance(rid, int):
                try:
                    rid = int(rid)
                except (ValueError, TypeError):
                    malformed_count += 1
                    continue
            if rid not in expected_ids:
                unknown_ids.add(rid)
                continue
            if rid in seen_ids:
                duplicate_ids.add(rid)
                continue
            seen_ids.add(rid)
            text_val = r["text"]
            if isinstance(text_val, str):
                valid_map[rid] = text_val
            else:
                malformed_count += 1
    else:
        malformed_count += 1

    missing_ids = expected_ids - seen_ids
    is_clean = len(missing_ids) == 0 and len(unknown_ids) == 0 and len(duplicate_ids) == 0 and malformed_count == 0 and len(valid_map) == len(expected_ids)
    report = {
        "expected_count": len(expected_ids),
        "received_count": len(seen_ids),
        "valid_count": len(valid_map),
        "missing_ids": sorted(list(missing_ids)),
        "unknown_ids": sorted(list(unknown_ids)),
        "duplicate_ids": sorted(list(duplicate_ids)),
        "malformed_count": malformed_count,
        "is_clean": is_clean,
        "is_structurally_clean": is_clean
    }
    return valid_map, report

def validate_audit_batch_results(
    expected_windows: List[dict],
    raw_results: Any
) -> Tuple[Dict[int, dict], Dict[str, Any]]:
    """
    Deterministically validates semantic alignment audit responses against expected windows.

    Rules enforced:
    1. Every expected window_id MUST be accounted for.
    2. Unknown window_ids are rejected.
    3. Duplicate window_ids are rejected/flagged.
    4. Structural validity: each entry must have integer 'window_id', valid 'verdict', and 'confidence'.
    5. Missing results are flagged for split/focused re-audit (never silently marked ALIGNED).

    Returns:
        (valid_results_map, validation_report)
    """
    expected_ids = {w["window_id"] for w in expected_windows if isinstance(w, dict) and "window_id" in w}
    valid_verdicts = {"ALIGNED", "SHIFT_PLUS_1", "SHIFT_MINUS_1", "MERGED", "UNCERTAIN"}
    valid_confidences = {"HIGH", "MEDIUM", "LOW"}

    valid_map: Dict[int, dict] = {}
    seen_ids: Set[int] = set()
    duplicate_ids: Set[int] = set()
    unknown_ids: Set[int] = set()
    malformed_count = 0

    results_list = []
    if isinstance(raw_results, dict):
        results_list = raw_results.get("results", [])
    elif isinstance(raw_results, list):
        results_list = raw_results

    for r in results_list:
        if not isinstance(r, dict) or "window_id" not in r:
            malformed_count += 1
            continue
        wid = r["window_id"]
        if not isinstance(wid, int):
            try:
                wid = int(wid)
            except (ValueError, TypeError):
                malformed_count += 1
                continue
        if wid not in expected_ids:
            unknown_ids.add(wid)
            continue
        if wid in seen_ids:
            duplicate_ids.add(wid)
            continue
        seen_ids.add(wid)

        verdict = str(r.get("verdict", "UNCERTAIN")).upper().strip()
        if verdict not in valid_verdicts:
            verdict = "UNCERTAIN"
        confidence = str(r.get("confidence", "LOW")).upper().strip()
        if confidence not in valid_confidences:
            confidence = "LOW"
        details = str(r.get("details", "")).strip()

        valid_map[wid] = {
            "window_id": wid,
            "verdict": verdict,
            "confidence": confidence,
            "details": details
        }

    missing_ids = expected_ids - seen_ids
    report = {
        "expected_count": len(expected_ids),
        "received_count": len(seen_ids),
        "valid_count": len(valid_map),
        "missing_ids": sorted(list(missing_ids)),
        "unknown_ids": sorted(list(unknown_ids)),
        "duplicate_ids": sorted(list(duplicate_ids)),
        "malformed_count": malformed_count,
        "is_complete": len(missing_ids) == 0 and len(unknown_ids) == 0 and len(duplicate_ids) == 0 and malformed_count == 0
    }
    return valid_map, report

def get_provider_capabilities(provider: str) -> dict:
    if provider == "deepl":
        return {
            "supports_structured_output": False,
            "supports_identical_classification": False,
            "supports_context": False
        }
    return {
        "supports_structured_output": True,
        "supports_identical_classification": True,
        "supports_context": True
    }

ENGLISH_COMMON_WORDS = {
    # Pronouns & determiners
    "i", "me", "my", "myself", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "we", "us", "our", "ours", "ourselves", "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "mostly", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "really",
    "just", "even", "ever", "never", "always", "almost", "still", "already", "well", "also",
    "a", "an", "the",
    # Prepositions & conjunctions
    "and", "but", "if", "or", "because", "as", "until", "while", "of", "at",
    "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
    # Common verbs & contractions
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "will", "wills", "shall", "must", "can", "could", "may", "might",
    "would", "should", "ought", "i'm", "you're", "he's", "she's",
    "it's", "we're", "they're", "i've", "you've", "we've", "they've",
    "i'd", "you'd", "he'd", "she'd", "we'd", "they'd", "i'll", "you'll",
    "he'll", "she'll", "we'll", "they'll", "isn't", "aren't", "wasn't",
    "weren't", "hasn't", "haven't", "hadn't", "doesn't", "don't", "didn't",
    "won't", "wouldn't", "shan't", "shouldn't", "can't", "cannot", "couldn't",
    "mustn't", "let's", "that's", "who's", "what's", "here's", "there's",
    "when's", "where's", "why's", "how's",
    # Common conversational words & verbs
    "hello", "hi", "hey", "goodbye", "bye", "please", "thanks", "thank",
    "welcome", "yes", "yeah", "yep", "sure", "ok", "okay", "alright",
    "nope", "maybe", "sorry", "excuse", "pardon",
    "come", "came", "go", "goes", "went", "gone", "going",
    "get", "gets", "got", "gotten", "getting",
    "make", "makes", "made", "making",
    "know", "knows", "knew", "known", "knowing",
    "think", "thinks", "thought", "thinking",
    "take", "takes", "took", "taken", "taking",
    "see", "sees", "saw", "seen", "seeing",
    "look", "looks", "looked", "looking",
    "give", "gives", "gave", "given", "giving",
    "tell", "tells", "told", "telling",
    "ask", "asks", "asked", "asking",
    "work", "works", "worked", "working",
    "seem", "seems", "seemed", "seeming",
    "feel", "feels", "felt", "feeling",
    "try", "tries", "tried", "trying",
    "leave", "leaves", "left", "leaving",
    "call", "calls", "called", "calling",
    "need", "needs", "needed", "needing",
    "want", "wants", "wanted", "wanting",
    "help", "helps", "helped", "helping",
    "talk", "talks", "talked", "talking",
    "turn", "turns", "turned", "turning",
    "start", "starts", "started", "starting",
    "show", "shows", "showed", "shown", "showing",
    "hear", "hears", "heard", "hearing",
    "play", "plays", "played", "playing",
    "run", "runs", "ran", "running",
    "move", "moves", "moved", "moving",
    "like", "likes", "liked", "liking",
    "live", "lives", "lived", "living",
    "believe", "believes", "believed", "believing",
    "hold", "holds", "held", "holding",
    "bring", "brings", "brought", "bringing",
    "happen", "happens", "happened", "happening",
    "write", "writes", "wrote", "written", "writing",
    "sit", "sits", "sat", "sitting",
    "stand", "stands", "stood", "standing",
    "lose", "loses", "lost", "losing",
    "pay", "pays", "paid", "paying",
    "meet", "meets", "met", "meeting",
    "stop", "stops", "stopped", "stopping",
    "speak", "speaks", "spoke", "spoken", "speaking",
    "read", "reads", "reading",
    "allow", "allows", "allowed", "allowing",
    "open", "opens", "opened", "opening",
    "walk", "walks", "walked", "walking",
    "win", "wins", "won", "winning",
    "remember", "remembers", "remembered", "remembering",
    "love", "loves", "loved", "loving",
    "wait", "waits", "waited", "waiting",
    "die", "dies", "died", "dying",
    "send", "sends", "sent", "sending",
    "stay", "stays", "stayed", "staying",
    "fall", "falls", "fell", "fallen", "falling",
    "kill", "kills", "killed", "killing",
    "shoot", "fire", "jump", "hide", "hope",
    "man", "men", "woman", "women", "boy", "boys", "girl", "girls",
    "guy", "guys", "kid", "kids", "child", "children", "baby", "babies",
    "friend", "friends", "family",
    "dad", "mom", "papa", "mama", "father", "mother", "sister", "brother",
    "son", "daughter", "uncle", "aunt", "cousin", "grandma", "grandpa",
    "day", "days", "night", "nights", "time", "times",
    "thing", "things", "way", "ways", "life", "world", "house", "home",
    "room", "door", "car", "head", "hand", "hands", "eye", "eyes",
    "floor", "step", "station", "office", "point", "target", "mission", "operation",
    "plan", "level", "channel", "route", "section", "unit", "sector", "zone",
    "alert", "warning", "danger",
    "right", "wrong", "good", "bad", "great", "fine",
    "now", "today", "tonight", "tomorrow", "yesterday",
    "sir", "ma'am", "mr", "mrs", "ms", "miss", "dr", "doctor", "officer",
    "captain", "lieutenant", "sergeant", "colonel", "general", "major", "chief", "boss",
    "king", "queen", "prince", "princess", "lady",
    "god", "lord", "jesus", "christ", "damn", "hell", "shit", "fuck",
    "dialogue", "source", "line", "lines", "scene", "episode", "season", "part", "track",
    # SDH / descriptive audio sound terms
    "sigh", "sighs", "sighing", "gasp", "gasps", "gasping", "screams", "screaming",
    "cries", "crying", "music", "playing", "door", "closes", "closing", "opens", "opening",
    "whispering", "chuckles", "snickers", "applause", "cheering", "groans", "groaning",
    "grunt", "grunts", "grunting", "laughter", "laughing", "snorts", "snorting",
    "cough", "coughs", "coughing", "sneeze", "sneezes", "sneezing",
    # Months & seasons & numbers as words
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "spring", "summer", "autumn", "fall", "winter", "rose", "roses", "flower", "flowers",
    "bear", "bears", "born", "bore", "borne", "bearing",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "twenty", "thirty", "fifty", "hundred", "thousand", "million",
    "first", "second", "third", "last", "next",
    "red", "green", "blue", "black", "white", "yellow", "gold", "silver",
    # Additional conversational & dialogue vocabulary
    "party", "parties", "cool", "nice", "wow", "bad", "nah", "congratulations",
    "here", "there", "everywhere", "nowhere", "somewhere", "everyone", "everybody",
    "someone", "somebody", "anyone", "anybody", "nobody", "everything", "something",
    "anything", "nothing", "soon", "later", "morning", "afternoon", "evening",
    "hurry", "listen", "watch", "quick", "quiet", "silence", "shutup", "ready",
    "enough", "exactly", "absolutely", "definitely", "probably", "perhaps", "certainly",
}

KNOWN_NON_VERBAL_SOUNDS = {
    "ah", "aah", "aaah", "eh", "ehh", "er", "ha", "haha", "hahaha", "heh", "hehe", "hoho",
    "hm", "hmm", "hmmm", "ho", "huh", "mm", "mmm", "mmmm", "oh", "ooh", "oooh", "ouch",
    "ow", "pfft", "psst", "shh", "shhh", "tsk", "tsk-tsk", "uh", "uh-huh", "uh-oh",
    "um", "umm", "ummm", "ugh", "ughh", "whew", "whoa", "woah", "yay", "yuck", "grr", "argh"
}

KNOWN_TECH_TERMS_AND_BRANDS = {
    "wifi", "wi-fi", "bluetooth", "youtube", "google", "tiktok", "instagram", "twitter",
    "facebook", "netflix", "spotify", "xbox", "playstation", "iphone", "ipad", "android",
    "gps", "usb", "vip", "tv", "dj", "pc", "sim", "pin", "led", "lcd", "ai", "vr", "hd",
    "4k", "uhd", "dvd", "vcr", "cd", "fbi", "cia", "nasa", "dna", "nato", "unicef", "interpol",
    "bmw"
}

def _looks_like_strict_proper_noun(text: str) -> bool:
    """Proper nouns cannot be deterministically proven safe to leave untranslated."""
    return False

def is_deterministically_safe_keep(text: str, reason: str, show_title: str = "") -> bool:
    """
    Fail-closed deterministic check to verify if a line is truly safe to KEEP unchanged.
    Returns True ONLY if deterministically defensible. Otherwise False.
    """
    if not text:
        return False

    clean_text = text.strip()
    # Strip HTML/ASS formatting
    clean_text = re.sub(r'<[^>]+>', '', clean_text)
    clean_text = re.sub(r'\{[^}]+\}', '', clean_text).strip()
    if not clean_text or clean_text == "<i></i>":
        return True

    reason = (reason or "").strip().lower()
    allowed_reasons = {"proper_noun", "brand", "acronym", "number", "symbol", "non_verbal"}
    if reason not in allowed_reasons:
        return False

    has_letters = any(c.isalpha() for c in clean_text)
    has_digits = any(c.isdigit() for c in clean_text)
    words = [w for w in re.split(r"[^\w']+", clean_text) if w and any(c.isalnum() for c in w)]

    if reason == "symbol":
        return not has_letters and not has_digits

    elif reason == "number":
        if not has_digits:
            return False
        for w in words:
            w_clean = re.sub(r"[^\w]", "", w).lower()
            if w_clean in ENGLISH_COMMON_WORDS:
                return False
        return True

    elif reason == "non_verbal":
        if not has_letters:
            return True
        for w in words:
            w_clean = re.sub(r"[^\w]", "", w).lower()
            if not w_clean:
                continue
            if w_clean in ENGLISH_COMMON_WORDS:
                return False
            is_known = w_clean in KNOWN_NON_VERBAL_SOUNDS
            is_onomatopoeia = bool(re.match(r'^(m+|h+a+|a+h+|o+h+|h+e+h+|h+m+|u+g+h+|o+o+h+|s+h+|b+e+e+p+|b+i+p+|d+i+n+g+|p+i+n+g+)$', w_clean))
            if not (is_known or is_onomatopoeia):
                return False
        return True

    elif reason in {"acronym", "brand"}:
        if not words or not has_letters:
            return False
        full_clean = re.sub(r"[^\w-]", "", clean_text).lower()
        if full_clean in KNOWN_TECH_TERMS_AND_BRANDS:
            return True
        for w in words:
            w_clean = re.sub(r"[^\w-]", "", w).lower()
            if not w_clean or w_clean not in KNOWN_TECH_TERMS_AND_BRANDS:
                return False
        return True

    elif reason == "proper_noun":
        # Fail-closed: Proper nouns cannot be deterministically proven safe to leave untranslated.
        # Defaults to TRANSLATE so the AI model safely evaluates context and localization.
        return False

    return False

def normalize_for_compare(text: str) -> str:
    """Helper to normalize text for comparison by ignoring punctuation/symbols/tags/casing."""
    if not text:
        return ""
    t = re.sub(r'<[^>]+>', '', text)
    t = re.sub(r'\{[^}]+\}', '', t)
    t = unicodedata.normalize('NFKC', t)
    t = re.sub(r'[^\w]', '', t)
    return t.casefold()

def is_meaningful_translation(source_text: str, candidate_text: str) -> bool:
    """
    Fail-closed check to verify if candidate_text is a usable translation
    that is not identical or normalized-equivalent to source_text (consistent with FINAL QA).
    """
    if not is_usable_translation(candidate_text):
        return False
    return normalize_for_compare(candidate_text) != normalize_for_compare(source_text)

def has_entity_evidence(
    target_text: str,
    source_subs: list,
    translated_subs: list,
    target_idx: Optional[int] = None,
    min_evidence: int = 1
) -> bool:
    """
    Strict evidence-based entity resolver (same subtitle run).
    Verifies if every entity token in target_text is proven to be an unchanged
    proper noun / entity name across English source and translated target text
    within genuinely translated cues from the exact same subtitle run.

    Invariants:
    1. Returns False if target_text is empty or contains any word in ENGLISH_COMMON_WORDS.
    2. Candidate must contain only non-common-word alphabetic tokens.
    3. Every entity token must appear with exact token boundaries (\\b) in BOTH source and
       translated text of at least min_evidence distinct, genuinely translated cues (where
       is_meaningful_translation(source, translated) is True).
    4. Never derives evidence from source-copy, identical, or KEEP cues.
    5. No substring matching, no fuzzy matching, no loose TitleCase heuristics.
    """
    if not target_text or not source_subs or not translated_subs:
        return False

    clean_text = target_text.strip()
    clean_text = re.sub(r'<[^>]+>', '', clean_text)
    clean_text = re.sub(r'\{[^}]+\}', '', clean_text).strip()
    if not clean_text or clean_text == "<i></i>":
        return False

    # Extract alphabetic words/tokens
    words = [w for w in re.split(r"[^\w\x27-]+", clean_text) if w and any(c.isalpha() for c in w)]
    if not words:
        return False

    # Reject if ANY word is in ENGLISH_COMMON_WORDS or is too short
    for w in words:
        w_clean = re.sub(r"[^\w]", "", w).lower()
        if not w_clean or w_clean in ENGLISH_COMMON_WORDS or len(w_clean) < 2:
            return False

    min_len = min(len(source_subs), len(translated_subs))
    for w in words:
        w_pattern = re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE)
        evidence_count = 0
        for k in range(min_len):
            if target_idx is not None and k == target_idx:
                continue
            src = source_subs[k].content.strip()
            trans = translated_subs[k].content.strip()
            if not src or src == "<i></i>" or not trans or trans == "<i></i>":
                continue
            # CRITICAL: Evidence MUST come from genuinely translated lines, NOT copies/KEEPs
            if not is_meaningful_translation(src, trans):
                continue
            if w_pattern.search(src) and w_pattern.search(trans):
                evidence_count += 1
                if evidence_count >= min_evidence:
                    break

        if evidence_count < min_evidence:
            return False

    return True

NAME_PARTICLES = {
    "von", "van", "de", "der", "del", "da", "di", "el", "al", "san", "santa",
    "st", "ste", "mc", "mac", "o", "d", "und", "and", "och", "et",
    "dr", "mr", "mrs", "ms", "prof", "herr", "frau", "sir", "lord", "lady",
    "madame", "monsieur", "don", "dona", "doña", "signor", "signora",
    "senor", "senora", "señor", "señora", "fru", "froken", "fröken"
}

SHARED_CROSS_LINGUAL_WORDS = {
    "ja", "nej", "mamma", "mama", "papa", "pappa", "hej", "hallo", "hallå",
    "ok", "okej", "okay", "bravo", "amen", "stop", "stopp", "taxi", "hotel", "hotell",
    "bar", "restaurant", "restaurang", "film", "radio", "telefon", "tv", "sms",
    "monster", "dacia"
}

def is_strictly_valid_entity_candidate(text: str) -> bool:
    """
    Fail-closed deterministic check to verify if candidate text has the lexical form
    of a valid named entity token (single name or name list) before allowing contextual entity verification.

    Invariants:
    1. Must not be empty or placeholder (<i></i>).
    2. Must contain 1 to 16 alphabetic tokens.
    3. Every token must start with an uppercase letter or be a recognized name particle.
    4. No token may match ANY word in ENGLISH_COMMON_WORDS (unless a recognized name particle).
    5. No token may match ANY word in KNOWN_NON_VERBAL_SOUNDS.
    6. No token may match ANY word in KNOWN_TECH_TERMS_AND_BRANDS (handled separately).
    7. Every token must be at least 2 characters long (or single letter initial).
    """
    if not text:
        return False
    clean_text = text.strip()
    clean_text = re.sub(r'<[^>]+>', '', clean_text)
    clean_text = re.sub(r'\{[^}]+\}', '', clean_text).strip()
    if not clean_text or clean_text == "<i></i>":
        return False

    tokens = [t for t in re.split(r"[^\w\x27-]+", clean_text) if t and any(c.isalpha() for c in t)]
    if not tokens or len(tokens) > 16:
        return False

    for t in tokens:
        clean_t = re.sub(r"[^\w]", "", t).lower()
        if not clean_t:
            return False
        if clean_t in NAME_PARTICLES:
            continue
        if len(clean_t) < 2 and not t[0].isupper():
            return False
        if clean_t in ENGLISH_COMMON_WORDS:
            return False
        if clean_t in KNOWN_NON_VERBAL_SOUNDS:
            return False
        if clean_t in KNOWN_TECH_TERMS_AND_BRANDS:
            return False
        # Must start with uppercase
        first_alpha = next((c for c in t if c.isalpha()), None)
        if not first_alpha or not first_alpha.isupper():
            return False

    return True

def is_pure_structural_invariant(text: str) -> bool:
    """
    Deterministic language-agnostic check for purely structural (non-lexical) invariants:
    1. Empty / placeholder cues ('<i></i>', '{}', '').
    2. Pure numbers / timestamps / timecodes / symbols / percentages ('0153...', '4', '12:30', '100%').
    3. Pure punctuation / dialogue dashes / ellipses ('...', '---', '???').
    """
    if not text:
        return True
    clean_text = text.strip()
    clean_text = re.sub(r'<[^>]+>', '', clean_text)
    clean_text = re.sub(r'\{[^}]+\}', '', clean_text).strip()
    if not clean_text or clean_text == "<i></i>":
        return True

    # Multi-line handling
    lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
    if len(lines) > 1:
        return all(is_pure_structural_invariant(line) for line in lines)

    # Strip leading dialogue dashes
    line = re.sub(r'^[-\u2013\u2014\u2212\s]+', '', clean_text).strip()
    if not line:
        return True

    # Pure numbers, symbols, timecodes, punctuation (no alphabetic characters)
    if not any(c.isalpha() for c in line):
        return True

    return False

def is_valid_shared_or_entity_keep(orig_text: str, trans_text: str, target_lang: str = "") -> bool:
    """
    Legacy helper / structural validator:
    Accepts pure structural invariants, multi-line structural cues, or recognized cross-lingual shared words.
    Rejects normal dialogue words (e.g. 'Hello', 'Where are you?') and long sentences.
    """
    if not orig_text or not trans_text:
        return False

    # Check structural / character similarity
    if normalize_for_compare(orig_text) != normalize_for_compare(trans_text):
        return False

    clean_orig = orig_text.strip()
    clean_orig = re.sub(r'<[^>]+>', '', clean_orig)
    clean_orig = re.sub(r'\{[^}]+\}', '', clean_orig).strip()
    if not clean_orig or clean_orig == "<i></i>":
        return True

    # Pure structural invariant check
    if is_pure_structural_invariant(clean_orig):
        return True

    # Multi-line dialogue handling
    lines = [line.strip() for line in clean_orig.split('\n') if line.strip()]
    if len(lines) > 1:
        return all(is_valid_shared_or_entity_keep(line, line, target_lang) for line in lines)

    # Strip leading dialogue dashes
    line = re.sub(r'^[-\u2013\u2014\u2212\s]+', '', clean_orig).strip()
    if not line:
        return True

    norm = normalize_for_compare(line)
    if norm in SHARED_CROSS_LINGUAL_WORDS:
        return True

    # Check for digit/symbol compound with short words (e.g. '... 4. Ja.', '12:30. OK.')
    if any(c.isdigit() for c in line):
        tokens = [t for t in re.split(r"[^\w\x27-]+", line) if t and any(c.isalpha() for c in t)]
        if len(tokens) <= 3 and all(normalize_for_compare(t) in SHARED_CROSS_LINGUAL_WORDS or any(c.isupper() for c in t) for t in tokens):
            return True

    return False

def validate_entity_verification_output(
    raw_text: str,
    candidates: list,
    show_title: str = ""
) -> set:
    """
    Validates the AI entity verification output fail-closed.
    Returns a set of verified cue IDs that are proven to be proper named entities in context.
    """
    if not raw_text or not candidates:
        return set()

    candidate_map = {c["id"]: c.get("target", "") for c in candidates}
    verified_ids = set()

    clean_text = raw_text.strip()
    if clean_text.startswith("```"):
        lines = clean_text.split('\n')
        if lines[0].startswith("```"): lines = lines[1:]
        if lines and lines[-1].startswith("```"): lines = lines[:-1]
        clean_text = "\n".join(lines).strip()

    try:
        data = json.loads(clean_text)
        results = data.get("results", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        for r in results:
            if not isinstance(r, dict):
                continue
            rid = r.get("id")
            if rid not in candidate_map:
                continue
            target_text = candidate_map[rid]
            if not is_strictly_valid_entity_candidate(target_text):
                continue

            verdict = str(r.get("verdict", "")).upper()
            entity_type = str(r.get("entity_type", "")).upper()
            confidence = str(r.get("confidence", "")).upper()

            if (
                verdict == "NAMED_ENTITY"
                and entity_type in {"PERSON_NAME", "PLACE_NAME", "ORGANIZATION"}
                and confidence == "HIGH"
            ):
                verified_ids.add(rid)
    except Exception as e:
        logger.error(f"Entity Verification: JSON parse failed: {e}")

    return verified_ids

def validate_semantic_invariant_verification_output(
    raw_text: str,
    candidates: list,
    show_title: str = ""
) -> set:
    """
    Validates the AI batch semantic invariant verification output fail-closed.
    Returns a set of verified cue IDs that are proven to be legitimate invariants in target language.

    Invariants:
    1. Candidate must be within lexical sanity bounds (<= 16 tokens, <= 120 chars).
    2. invariant_in_target MUST be True.
    3. explanation MUST be non-empty.
    4. Any ambiguity, false verdict, missing ID, or schema violation -> rejected (not in returned set).
    """
    if not raw_text or not candidates:
        return set()

    candidate_map = {c["id"]: c.get("target", c.get("text", "")) for c in candidates}
    verified_ids = set()

    clean_text = raw_text.strip()
    if clean_text.startswith("```"):
        lines = clean_text.split('\n')
        if lines[0].startswith("```"): lines = lines[1:]
        if lines and lines[-1].startswith("```"): lines = lines[:-1]
        clean_text = "\n".join(lines).strip()

    try:
        data = json.loads(clean_text)
        results = data.get("results", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])

        for r in results:
            if not isinstance(r, dict):
                continue
            rid = r.get("id")
            if rid not in candidate_map:
                continue
            target_text = candidate_map[rid]
            clean_target = re.sub(r'<[^>]+>', '', target_text).strip()
            tokens = [t for t in re.split(r"[^\w\x27-]+", clean_target) if t and any(c.isalnum() for c in t)]
            if len(tokens) > 16 or len(clean_target) > 120:
                logger.warning(f"Semantic Invariant Verification: ID {rid} ('{target_text}') rejected: exceeds lexical bounds")
                continue

            inv = r.get("invariant_in_target") is True
            expl = str(r.get("explanation", "")).strip()

            if inv and expl:
                logger.info(f"Semantic Invariant Verification: ID {rid} ('{target_text}') verified as invariant ({expl})")
                verified_ids.add(rid)
            else:
                logger.info(f"Semantic Invariant Verification: ID {rid} ('{target_text}') rejected (invariant={inv}, explanation='{expl}')")
    except Exception as e:
        logger.error(f"Semantic Invariant Verification: JSON parse failed: {e}")

    return verified_ids

def validate_classifier_output(
    raw_text: str,
    original_items: list,
    show_title: str = "",
    source_subs: Optional[list] = None,
    translated_subs: Optional[list] = None
) -> list:
    """
    Language-agnostic validation of classify_and_recover_identical output.
    Ensures every returned item matches an expected input ID, actions are valid,
    and KEEP actions are verified with positive semantic evidence, fail-closed
    against adversarial dialogue classifications across all languages and scripts.
    """
    if not raw_text:
        return []

    expected_ids = {item["id"]: item.get("text", "") for item in original_items}
    clean_text = raw_text.strip()
    if clean_text.startswith("```"):
        lines = clean_text.split('\n')
        if lines[0].startswith("```"): lines = lines[1:]
        if lines and lines[-1].startswith("```"): lines = lines[:-1]
        clean_text = "\n".join(lines).strip()

    valid_results = []
    returned_ids = set()
    kept = 0
    translated = 0
    rejected = 0

    try:
        data = json.loads(clean_text)
        if isinstance(data, dict):
            results = data.get("results", [])
        elif isinstance(data, list):
            results = data
        else:
            results = []

        for r in results:
            if not isinstance(r, dict):
                continue
            rid = r.get("id")
            if rid is None:
                continue

            act = str(r.get("action", "")).lower()
            if rid not in expected_ids:
                logger.warning(f"Classifier validation: rejected result for unknown ID {rid}")
                rejected += 1
                continue

            original_text = expected_ids[rid]
            reason = str(r.get("reason", "")).lower()

            if act == "keep":
                # Deterministic check:
                # 1. Pure structural invariants (numbers, timestamps, %, pure punctuation, empty)
                # 2. Structural symbols and numbers without alphabetic text
                # 3. Same-run entity evidence
                is_det_safe = False
                if is_pure_structural_invariant(original_text):
                    is_det_safe = True
                elif reason in {"symbol", "number"} and is_deterministically_safe_keep(original_text, reason, show_title=show_title):
                    is_det_safe = True

                is_ev_safe = False
                if not is_det_safe and source_subs is not None and translated_subs is not None:
                    is_ev_safe = has_entity_evidence(original_text, source_subs, translated_subs, target_idx=rid)

                is_shared_safe = False
                clean_orig = re.sub(r'<[^>]+>', '', original_text).strip()
                tokens = [t for t in re.split(r"[^\w\x27-]+", clean_orig) if t and any(c.isalnum() for c in t)]

                if reason == "number":
                    if any(c.isdigit() for c in original_text) or not any(c.isalpha() for c in original_text):
                        is_shared_safe = True
                elif reason == "symbol":
                    if any(c in "$%#@&*+-/\\=<>~" for c in original_text) or not any(c.isalnum() for c in original_text):
                        is_shared_safe = True
                elif not any(c.isalpha() for c in clean_orig):
                    # Pure non-alphabetic tokens (e.g. ♪ ♪, ♫, ...) are language-neutral
                    is_shared_safe = True

                if is_ev_safe:
                    logger.info(f"Classifier validation: ID {rid} allowed KEEP via same-run entity evidence ('{original_text}')")
                    reason = f"evidence_{reason}"
                elif is_det_safe:
                    pass
                elif is_shared_safe:
                    logger.info(f"Classifier validation: ID {rid} allowed KEEP as verified language-pair invariant ('{original_text}')")
                    reason = f"invariant_{reason}"
                else:
                    logger.info(f"Classifier validation: ID {rid} flagged for batch semantic verification ('{original_text}', reason={reason})")
                    act = "translate"
                    proposed_reason = reason
                    reason = "needs_semantic_verification"
                    r["proposed_reason"] = proposed_reason
                    r["text"] = ""

            if act == "keep":
                kept += 1
                valid_results.append({"id": rid, "action": "keep", "reason": reason, "text": original_text})
                returned_ids.add(rid)
            elif act == "translate":
                translated += 1
                provided_text = str(r.get("text", "")).strip()
                # If provided translation is empty, unusable, or echoes source without meaningful translation, clear to force recovery
                if reason != "needs_semantic_verification" and not is_meaningful_translation(original_text, provided_text):
                    logger.info(f"Classifier validation: ID {rid} TRANSLATE has empty/echo text, clearing to force recovery")
                    provided_text = ""
                res_obj = {"id": rid, "action": "translate", "reason": reason or "translate", "text": provided_text}
                if "proposed_reason" in r:
                    res_obj["proposed_reason"] = r["proposed_reason"]
                valid_results.append(res_obj)
                returned_ids.add(rid)
            else:
                logger.warning(f"Classifier validation: rejected result for ID {rid} due to invalid action {act}")
                rejected += 1
                continue

        logger.info(f"Classifier validation: Validated {kept} KEEP, {translated} TRANSLATE, {rejected} REJECTED")
    except Exception as e:
        logger.error(f"Classifier validation: JSON parse failed: {e}")

    for item in original_items:
        if item["id"] not in returned_ids:
            logger.info(f"Classifier validation: Failsafe triggered for ID {item['id']}, forcing TRANSLATE")
            valid_results.append({"id": item["id"], "action": "translate", "reason": "malformed_fallback", "text": ""})

    return valid_results

class SubtitleTranslator:
    def _get_active_model_for_provider(self, provider: str) -> str:
        """Resolve active model for provider from settings. Uses central registry defaults.

        Raises ValueError for unknown/unsupported providers — never silently falls back to Gemini.
        """
        from app.core.ai_providers import context_from_settings, normalize_provider, get_provider_spec
        p = normalize_provider(provider or "gemini")
        get_provider_spec(p)  # raises ValueError("Unsupported AI provider: ...") for unknown providers
        return context_from_settings(p).model

    def _convert_to_openai_json_schema(self, schema: dict) -> dict:
        if not isinstance(schema, dict):
            return schema
        res = {}
        for k, v in schema.items():
            if k == "type" and isinstance(v, str):
                res["type"] = v.lower()
            elif k == "properties" and isinstance(v, dict):
                res["properties"] = {pk: self._convert_to_openai_json_schema(pv) for pk, pv in v.items()}
            elif k == "items" and isinstance(v, dict):
                res["items"] = self._convert_to_openai_json_schema(v)
            else:
                res[k] = v
        if res.get("type") == "object":
            if "additionalProperties" not in res:
                res["additionalProperties"] = False
            if "properties" in res:
                res["required"] = list(res.get("required", [])) + [
                    pk for pk in res["properties"].keys() if pk not in res.get("required", [])
                ]
        return res

    async def _dispatch_llm_completion(
        self,
        provider: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[dict] = None,
        temperature: float = 0.1,
        job_id: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Optional[str]:
        from app.core.ai_providers import normalize_provider, get_model_capabilities
        p = normalize_provider(provider or "gemini")
        if p == "gemini":
            from google import genai
            from google.genai import types
            api_key = get_setting("gemini_api_key", "")
            client = genai.Client(api_key=api_key or "dummy")
            m_name = model_name or get_setting("gemini_model", "gemini-3.5-flash-lite")
            caps = get_model_capabilities("gemini", m_name)
            config_kwargs = {
                "system_instruction": system_prompt,
                "max_output_tokens": 8192,
            }
            if temperature is not None and caps.temperature:
                config_kwargs["temperature"] = temperature
            if schema and caps.structured_output:
                config_kwargs["response_mime_type"] = "application/json"
                config_kwargs["response_schema"] = schema
            elif schema and caps.json_object:
                config_kwargs["response_mime_type"] = "application/json"
            config = types.GenerateContentConfig(**config_kwargs)

            def do_gemini():
                _mark_sdk_started(_usage_token_ctx.get(None))
                return client.models.generate_content(
                    model=m_name,
                    contents=[user_prompt],
                    config=config,
                )

            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, contextvars.copy_context().run, do_gemini)
            _capture_tokens("gemini", resp)
            return resp.text

        elif p in ("openai", "openrouter", "deepseek", "custom"):
            import openai
            if p == "openai":
                api_key = get_setting("openai_api_key", "")
                client = openai.OpenAI(api_key=api_key or "dummy")
                m_name = model_name or get_setting("openai_model", "gpt-4o-mini")
            elif p == "openrouter":
                api_key = get_setting("openrouter_api_key", "")
                client = openai.OpenAI(
                    api_key=api_key or "dummy",
                    base_url="https://openrouter.ai/api/v1",
                    default_headers={"HTTP-Referer": "https://github.com/hugomossberg/Babel", "X-Title": "Babel"},
                )
                m_name = model_name or get_setting("openrouter_model", "")
            elif p == "deepseek":
                api_key = get_setting("deepseek_api_key", "")
                client = openai.OpenAI(
                    api_key=api_key or "dummy",
                    base_url="https://api.deepseek.com",
                )
                m_name = model_name or get_setting("deepseek_model", "deepseek-v4-flash")
            else:  # custom
                api_key = get_setting("custom_openai_api_key", "") or "dummy"
                base_url = get_setting("custom_openai_url", "http://localhost:8000/v1")
                client = openai.OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )
                m_name = model_name or get_setting("custom_openai_model", "default")

            def do_openai_compat():
                _mark_sdk_started(_usage_token_ctx.get(None))
                caps = get_model_capabilities(p, m_name)
                kwargs = {
                    "model": m_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                }
                if reasoning_effort is not None:
                    kwargs["reasoning_effort"] = reasoning_effort
                effective_reasoning_effort = (
                    kwargs.get("reasoning_effort")
                    or (isinstance(kwargs.get("reasoning"), dict) and kwargs["reasoning"].get("effort"))
                    or caps.default_reasoning_effort
                )
                effective_reasoning_none = (effective_reasoning_effort == "none")
                can_send_temperature = caps.temperature or (caps.temperature_requires_no_reasoning and effective_reasoning_none)
                if temperature is not None and can_send_temperature:
                    kwargs["temperature"] = temperature
                if schema and caps.structured_output:
                    converted_schema = self._convert_to_openai_json_schema(schema)
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_output",
                            "strict": True,
                            "schema": converted_schema,
                        },
                    }
                elif schema and caps.json_object:
                    kwargs["response_format"] = {"type": "json_object"}
                # Provider-specific thinking control:
                if p == "deepseek" and caps.thinking_control:
                    kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                return client.chat.completions.create(**kwargs)

            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, contextvars.copy_context().run, do_openai_compat)
            _capture_tokens(p, resp)
            try:
                return resp.choices[0].message.content
            except Exception:
                return ""

        elif p == "anthropic":
            api_key = get_setting("anthropic_api_key", "") or "dummy"
            m_name = model_name or get_setting("anthropic_model", "claude-sonnet-5")
            caps = get_model_capabilities("anthropic", m_name)
            _mark_sdk_started(_usage_token_ctx.get(None))
            # Estimate output budget based on prompt size, capped at model max
            _estimated_output = min(caps.max_output_tokens or 8192, max(2048, len(user_prompt) // 2))

            # Determine whether to use native structured output (output_config.format)
            # Supported: claude-sonnet-4.5+, claude-opus-4.1+, all 5.x models
            # Fallback for older models: inject JSON schema into system prompt
            _use_native_output_config = bool(schema and caps.native_output_config)

            _system_prompt = system_prompt
            if schema and not _use_native_output_config:
                # Strict JSON fallback: describe schema in system prompt
                import json as _json
                _system_prompt = (
                    f"{system_prompt}\n\n"
                    "IMPORTANT: You MUST respond with ONLY valid JSON matching this schema:\n"
                    f"{_json.dumps(schema, indent=2)}\n"
                    "No markdown, no explanation, no preamble. Raw JSON only."
                )

            payload = {
                "model": m_name,
                "max_tokens": _estimated_output,
                "system": _system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }
            # Only send temperature if model capabilities allow it
            if temperature is not None and caps.temperature:
                payload["temperature"] = temperature
            # Native structured output via output_config.format
            if _use_native_output_config:
                payload["output_config"] = {
                    "format": {
                        "type": "json_schema",
                        "schema": schema,
                    }
                }
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json=payload,

                )
                resp.raise_for_status()
                data = resp.json()
                # Detect truncated response — max_tokens means output was cut off
                stop_reason = data.get("stop_reason", "")
                if stop_reason == "max_tokens":
                    import logging as _log
                    _log.getLogger(__name__).warning(
                        "Anthropic response TRUNCATED (stop_reason=max_tokens) for model %s — treating as incomplete", m_name
                    )
                    return None  # Upstream retry/recovery will handle this
                _capture_tokens("anthropic", data)
                texts = [block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"]
                return "".join(texts)

        elif p == "ollama":
            ollama_url = get_setting("ollama_url", "http://localhost:11434").rstrip("/")
            m_name = model_name or get_setting("ollama_model", "llama3")
            caps = get_model_capabilities(provider or "ollama", m_name)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            _mark_sdk_started(_usage_token_ctx.get(None))
            req_payload = {
                "model": m_name,
                "prompt": full_prompt,
                "stream": False,
            }
            if caps.json_object or caps.structured_output:
                req_payload["format"] = "json"
            if temperature is not None and caps.temperature:
                req_payload["options"] = {"temperature": temperature}
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{ollama_url}/api/generate",
                    json=req_payload,
                )
                resp.raise_for_status()
                data = resp.json()
                _capture_tokens("ollama", data)  # prompt_eval_count → input_tokens, eval_count → output_tokens
                return data.get("response", "")

        return None

    @with_retry
    async def classify_and_recover_identical(
        self,
        items: list,
        target_language: str,
        show_title: str,
        source_subs: Optional[list] = None,
        translated_subs: Optional[list] = None,
        job_id: Optional[int] = None,
        source_language: str = "English",
        provider_ctx=None,
    ) -> list:
        from app.core.ai_providers import resolve_job_provider_context, context_from_settings
        _ctx = provider_ctx or (resolve_job_provider_context(job_id) if job_id else context_from_settings())
        provider = _ctx.provider
        if provider == "deepl":
            return [] # fallback to translation
        if provider in ("gemini", "openai", "openrouter", "deepseek", "anthropic") and not get_setting(f"{provider}_api_key", ""):
            return []

        system_prompt = f"""You are a subtitle localization quality assurance AI for {target_language}.
The following lines were identical between {source_language} and candidate {target_language} output.
Evaluate whether each line is legitimately INVARIANT in {target_language} (KEEP) or is translatable dialogue/SDH that MUST be translated into natural {target_language} (TRANSLATE).

KEEP is ONLY valid when:
- 'proper_noun': character name, place name, or named entity conventionally unchanged in {target_language}.
- 'brand' / 'acronym': technical term, organization, or brand.
- 'number' / 'symbol': numerical values, timestamps, or symbols.
- 'non_verbal': untranslatable acoustic sound effect or onomatopoeia identical in {target_language}.
- 'shared_word' / 'cognate': universal loanword or interjection with the exact same spelling and meaning in BOTH {source_language} and {target_language}.

CRITICAL RULES:
- Ordinary conversational dialogue, phrases, commands, and sentences in {source_language} MUST NEVER be classified as KEEP.
- Descriptive non-verbal / SDH cues that describe an action or event (e.g. "[SIGHING]", "(door closes)", "[music playing]") MUST ALWAYS be TRANSLATE with the actual translated text in {target_language}.
- Invariant non-verbal cues are ONLY genuine acoustic sound effects / onomatopoeia that have the exact same written representation in {target_language}.

Return ONLY a JSON object with a single key 'results' containing an array of objects.
"""
        prompt = f"Context: {show_title}\n\nLines:\n" + json.dumps(items, ensure_ascii=False)

        schema = {
            "type": "OBJECT",
            "properties": {
                "results": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "id": {"type": "INTEGER"},
                            "action": {"type": "STRING", "enum": ["keep", "translate"]},
                            "reason": {"type": "STRING", "enum": ["proper_noun", "brand", "acronym", "number", "symbol", "non_verbal", "shared_word", "cognate", "none"]},
                            "text": {"type": "STRING"}
                        },
                        "required": ["id", "action", "text"]
                    }
                }
            },
            "required": ["results"]
        }

        model_name = _ctx.model
        raw_resp = await self._dispatch_llm_completion(
            provider=provider,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=prompt,
            schema=schema,
            temperature=0.1,
            job_id=job_id,
        )

        if not raw_resp:
            return []

        validated_results = validate_classifier_output(raw_resp, items, show_title=show_title, source_subs=source_subs, translated_subs=translated_subs)

        # Gather any ambiguous lexical KEEP candidates for batch semantic verification
        items_map = {item["id"]: item["text"] for item in items}
        semantic_candidates = []
        for r in validated_results:
            if r.get("reason") in {"needs_semantic_verification", "needs_context_verification"} and source_subs is not None:
                rid = r["id"]
                target_text = items_map.get(rid, "")

                # Build local dialogue context: up to 3 cues before and 3 cues after
                ctx_b_parts = []
                for b_idx in range(max(0, rid - 3), rid):
                    bc = source_subs[b_idx].content.strip()
                    if bc and bc != "<i></i>":
                        btc = translated_subs[b_idx].content.strip() if translated_subs and b_idx < len(translated_subs) else ""
                        if btc and btc != "<i></i>" and is_meaningful_translation(bc, btc):
                            ctx_b_parts.append(f"{bc} [SV: {btc}]")
                        else:
                            ctx_b_parts.append(bc)

                ctx_a_parts = []
                for a_idx in range(rid + 1, min(len(source_subs), rid + 4)):
                    ac = source_subs[a_idx].content.strip()
                    if ac and ac != "<i></i>":
                        ctx_a_parts.append(ac)

                semantic_candidates.append({
                    "id": rid,
                    "target": target_text,
                    "proposed_reason": r.get("proposed_reason", "invariant"),
                    "context_before": " | ".join(ctx_b_parts) if ctx_b_parts else "(none)",
                    "context_after": " | ".join(ctx_a_parts) if ctx_a_parts else "(none)"
                })

        if semantic_candidates:
            logger.info(f"Semantic Invariant Verification: evaluating {len(semantic_candidates)} ambiguous alphabetic candidates in 1 batch call")
            try:
                verified_ids = await self.verify_single_occurrence_entities(
                    semantic_candidates,
                    target_language,
                    show_title=show_title,
                    job_id=job_id,
                    source_language=source_language,
                )
                for r in validated_results:
                    rid = r["id"]
                    if r.get("reason") in {"needs_semantic_verification", "needs_context_verification"}:
                        if rid in verified_ids:
                            r["action"] = "keep"
                            r["reason"] = f"verified_{r.get('proposed_reason', 'invariant')}"
                            r["semantic_verified"] = True
                            r["text"] = items_map.get(rid, "")
                            logger.info(f"Semantic Invariant Verification: ID {rid} ('{r['text']}') ACCEPTED as verified invariant KEEP")
                        else:
                            r["action"] = "translate"
                            r["reason"] = "unverified_invariant"
                            r["semantic_verified"] = False
                            r["text"] = ""
                            logger.info(f"Semantic Invariant Verification: ID {rid} ('{items_map.get(rid, '')}') REJECTED -> remains TRANSLATE")
            except (ProviderUnavailableError, ProviderConfigurationError, DailyQuotaExhaustedError, RequestBudgetExhaustedError):
                raise
            except Exception as e:
                err_str = str(e).lower()
                permanent = any(x in err_str for x in [
                    "401", "403", "unauthorized", "forbidden", "api key not valid",
                    "invalid api key", "not configured", "model_not_found", "permission_denied"
                ])
                if permanent:
                    raise ProviderConfigurationError(f"Permanent provider configuration error in semantic invariant verification: {str(e)}")
                logger.error(f"Semantic Invariant Verification call failed: {e}")

        return validated_results

    @with_retry
    async def verify_alphabetic_invariants_batch(
        self,
        candidates: list,
        target_language: str,
        show_title: str = "",
        job_id: Optional[int] = None,
        source_language: str = "English",
        provider_ctx=None,
    ) -> set:
        """
        Bounded, structured single batch AI semantic invariant verifier with local cue context.
        Audits proposed identical/unchanged subtitle cues against target_language localization standards.
        Returns a set of validated invariant cue IDs.
        """
        if not candidates:
            return set()

        from app.core.ai_providers import resolve_job_provider_context, context_from_settings
        _ctx = provider_ctx or (resolve_job_provider_context(job_id) if job_id else context_from_settings())
        provider = _ctx.provider
        if provider == "deepl":
            return set()
        if provider in ("gemini", "openai", "openrouter", "deepseek", "anthropic") and not get_setting(f"{provider}_api_key", ""):
            return set()

        system_prompt = f"""You are an independent localization quality auditor for {source_language} -> {target_language}.
You are verifying subtitle cues where the proposed {target_language} translation is identically spelled to the {source_language} source text.

Your task is to determine whether the proposed text is a valid, faithful, and natural {target_language} rendering in this dialogue context, OR if it is an error (untranslated {source_language} text left in the subtitle).

DECISION CRITERIA:
Set "invariant_in_target": true IF:
1. The text is an entity, proper name, brand, acronym, number, symbol, or non-verbal sound effect that is naturally kept identical in {target_language}, OR
2. The text is a valid and natural translation/localization in {target_language} for this dialogue context, even if identically spelled to the source text (e.g. shared cognates, identical loanwords, affirmations, greetings, short calls, proper names with shared particles, or short commands/verbs that are identical in both languages in this context).

Set "invariant_in_target": false IF:
1. The text is descriptive SDH (e.g. '[SIGHING]', '(door closes)', '[music playing]'), OR
2. The text is UNTRANSLATED {source_language} conversational dialogue that requires a different, non-identical translation in {target_language} (e.g. sentences, questions, or vocabulary where {target_language} uses different words).

Output strict structured JSON with key 'results' containing an array of verification objects:
- id: integer
- invariant_in_target: boolean (true if valid in {target_language}, false if untranslated/invalid)
- explanation: non-empty string explaining the linguistic reason
"""

        items_formatted = []
        for it in candidates:
            item_str = (
                f"Item ID {it['id']}:\n"
                f"PROPOSED REASON: {it.get('proposed_reason', 'unknown')}\n"
                f"CONTEXT BEFORE: {it.get('context_before', '(none)')}\n"
                f"SOURCE / PROPOSED TEXT: {it.get('target', it.get('text', ''))}\n"
                f"CONTEXT AFTER: {it.get('context_after', '(none)')}"
            )
            items_formatted.append(item_str)

        prompt = (
            f"Context / Show Title: {show_title}\n\n"
            + "\n\n".join(items_formatted)
            + f"\n\nAudit each of the {len(candidates)} items strictly as JSON."
        )

        schema = {
            "type": "OBJECT",
            "properties": {
                "results": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "id": {"type": "INTEGER"},
                            "invariant_in_target": {"type": "BOOLEAN"},
                            "explanation": {"type": "STRING"}
                        },
                        "required": ["id", "invariant_in_target", "explanation"]
                    }
                }
            },
            "required": ["results"]
        }

        model_name = _ctx.model
        raw_resp = await self._dispatch_llm_completion(
            provider=provider,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=prompt,
            schema=schema,
            temperature=0.0,
            job_id=job_id,
        )
        return validate_semantic_invariant_verification_output(raw_resp or "", candidates, show_title=show_title)

    @with_retry
    async def verify_single_occurrence_entities(
        self,
        candidates: list,
        target_language: str,
        show_title: str = "",
        job_id: Optional[int] = None,
        source_language: str = "English",
    ) -> set:
        """
        Backward-compatible wrapper for single-occurrence entity verification,
        delegating to verify_alphabetic_invariants_batch.
        """
        return await self.verify_alphabetic_invariants_batch(
            candidates,
            target_language,
            show_title=show_title,
            job_id=job_id,
            source_language=source_language
        )

    @with_retry
    async def classify_sdh_segments(
        self,
        items: list,
        source_language: str = "unknown",
        job_id: Optional[int] = None,
        provider_ctx=None,
    ) -> list:
        """
        Language-agnostic semantic classifier for ambiguous subtitle structural segments.
        Classifies each candidate snippet as:
        - "NON_DIALOGUE": audio description, sound effect, Foley, reaction, or speaker label.
        - "DIALOGUE": genuine spoken character dialogue or narrative speech.
        - "UNCERTAIN": ambiguous phrasing.
        Fail-safe, provider-agnostic, and strictly bounded.
        """
        if not items:
            return []

        from app.core.ai_providers import resolve_job_provider_context, context_from_settings
        _ctx = provider_ctx or (resolve_job_provider_context(job_id) if job_id else context_from_settings())
        provider = _ctx.provider
        if provider == "deepl":
            return []  # DeepL does not do classification; fail-safe fallback to dialogue
        if provider in ("gemini", "openai", "openrouter", "deepseek", "anthropic") and not get_setting(f"{provider}_api_key", ""):
            return []

        lang_label = source_language if (source_language and source_language.lower() != "unknown") else "the source language"
        system_prompt = f"""You are a subtitle sound effect and dialogue classifier.
Analyze each subtitle segment in {lang_label} and determine whether it represents:
- "NON_DIALOGUE": An audio description, sound effect, ambient noise, music indicator, or speaker label (e.g. footsteps, door closing, music playing, sighs, laughing, screams, narrator tags).
- "DIALOGUE": Real spoken dialogue, lines spoken by a character, or speech that should be translated and heard.
- "UNCERTAIN": Ambiguous phrasing that could be spoken dialogue.

CRITICAL RULES:
- If a segment could be genuine spoken dialogue or a spoken sentence, classify as "DIALOGUE".
- Descriptions of sounds, actions, background noises, or audio cues must be classified as "NON_DIALOGUE".
- Return ONLY strict JSON with key 'results' containing array of objects with 'id' and 'classification'.
"""
        prompt = f"Source language: {lang_label}\n\nSegments to classify:\n" + json.dumps(items, ensure_ascii=False)
        schema = {
            "type": "OBJECT",
            "properties": {
                "results": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "id": {"type": "INTEGER"},
                            "classification": {"type": "STRING", "enum": ["NON_DIALOGUE", "DIALOGUE", "UNCERTAIN"]}
                        },
                        "required": ["id", "classification"]
                    }
                }
            },
            "required": ["results"]
        }

        model_name = _ctx.model
        raw_resp = await self._dispatch_llm_completion(
            provider=provider,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=prompt,
            schema=schema,
            temperature=0.0,
            job_id=job_id,
        )

        if not raw_resp:
            return []

        try:
            import json as _json
            data = _json.loads(raw_resp)
            if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
                return data["results"]
            elif isinstance(data, list):
                return data
            return []
        except Exception as e:
            logger.warning(f"Failed to parse sdh classification response: {e}")
            return []

    def __init__(self):
        self._cached_gemini_key = None
        self._cached_gemini_client = None
        self._cached_openai_key = None
        self._cached_openai_client = None

    def get_gemini_client(self):
        api_key = get_setting("gemini_api_key", "")
        if not api_key:
            raise ValueError("Gemini API Key is not configured in settings.")
        if self._cached_gemini_client and self._cached_gemini_key == api_key:
            return self._cached_gemini_client
        self._cached_gemini_key = api_key
        from google import genai
        self._cached_gemini_client = genai.Client(api_key=api_key)
        return self._cached_gemini_client

    def get_openai_client(self):
        api_key = get_setting("openai_api_key", "")
        if not api_key:
            raise ValueError("OpenAI API Key is not configured in settings.")
        if self._cached_openai_client and self._cached_openai_key == api_key:
            return self._cached_openai_client
        self._cached_openai_key = api_key
        import openai
        self._cached_openai_client = openai.OpenAI(api_key=api_key)
        return self._cached_openai_client

    @with_retry
    async def _execute_single_escalation_call(
        self,
        provider: str,
        model_name: str,
        system_prompt: str,
        prompt: str,
        schema: dict,
        target_language: str,
        target_text: str,
        source_language: str = "English",
        job_id: Optional[int] = None,
    ) -> Optional[str]:
        p = (provider or "").lower().strip()
        if p == "deepl":
            import httpx
            api_key = get_setting("deepl_api_key", "")
            from app.core.languages import get_deepl_source_code, get_deepl_target_code
            target_lang_code = get_deepl_target_code(target_language)
            source_lang_code = get_deepl_source_code(source_language)
            url = "https://api-free.deepl.com/v2/translate" if api_key.endswith(":fx") else "https://api.deepl.com/v2/translate"
            _mark_sdk_started(_usage_token_ctx.get(None))
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                resp = await http_client.post(
                    url,
                    headers={"Authorization": f"DeepL-Auth-Key {api_key}"},
                    json={"text": [target_text], "target_lang": target_lang_code, "source_lang": source_lang_code},
                )
                resp.raise_for_status()
                data = resp.json()
                raw_res = data["translations"][0]["text"]
                return json.dumps({"translation": raw_res})

        return await self._dispatch_llm_completion(
            provider=p,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=prompt,
            schema=schema,
            temperature=0.1,
            job_id=job_id,
        )

    async def escalate_single_line(self, target_idx: int, target_text: str, prev_text: str, next_text: str, target_language: str, show_title: str, is_real_untranslated: bool = False, job_id: Optional[int] = None, exhausted_strategies: set = None, source_language: str = "source", context_verified_ids: set = None) -> Optional[str]:
        import logging
        import unicodedata
        import re
        logger = logging.getLogger(__name__)

        from app.core.ai_providers import resolve_job_provider_context, context_from_settings, normalize_provider
        if job_id:
            _primary_ctx = resolve_job_provider_context(job_id)
            _esc_ctx = resolve_job_provider_context(job_id, escalation=True)
        else:
            _primary_ctx = context_from_settings()
            _esc_ctx = context_from_settings(escalation=True)
        primary_provider = _primary_ctx.provider

        attempts = [
            {"provider": _esc_ctx.provider, "model": _esc_ctx.model, "type": "contextual"},
            {"provider": _esc_ctx.provider, "model": _esc_ctx.model, "type": "strict"},
            {"provider": _esc_ctx.provider, "model": _esc_ctx.model, "type": "isolated"},
        ]

        audited_rejected_texts = set()

        for i, attempt in enumerate(attempts):
            provider = attempt["provider"]
            attempt_type = attempt["type"]
            model_name = attempt["model"]

            context_fingerprint = hash((prev_text, target_text, next_text, is_real_untranslated))
            strategy_key = f"{target_idx}:{provider}:{model_name}:{attempt_type}:{context_fingerprint}"
            if exhausted_strategies is not None and strategy_key in exhausted_strategies:
                continue

            if attempt_type == "contextual":
                if is_real_untranslated:
                    _src_label = source_language if source_language != "source" else "source-language"
                    system_prompt = f"You MUST translate TARGET into {target_language}.\nTARGET is known to still be untranslated {_src_label} dialogue.\nDo NOT return the original {_src_label} text.\nDo NOT return an empty string.\nPrevious/Next are context only.\nReturn a JSON object with a single key 'translation' containing only the translated TARGET."
                else:
                    system_prompt = f"You are a subtitle translator. Translate the TARGET line to {target_language}. The Previous and Next lines are for context only. Return a JSON object with a single key 'translation' containing the translated string."
                prompt = f"Context: {show_title}\n\nPrevious: {prev_text}\nTARGET: {target_text}\nNext: {next_text}\n\nTranslate TARGET:"
            elif attempt_type == "strict":
                system_prompt = f"You are a strict translation engine."
                prompt = f"This cue has already failed QA because source-language dialogue remains.\n\nTranslate TARGET into {target_language}.\n\nTARGET:\n\"{target_text}\"\n\nPrevious/Next may only help disambiguation.\n\nYou MUST NOT:\n- classify TARGET\n- decide KEEP\n- return TARGET unchanged\n- return blank text\n\nReturn the actual translated TARGET.\n\nContext:\nPrevious: {prev_text}\nNext: {next_text}"
            else:
                system_prompt = f"You are a strict translation engine."
                prompt = f"Translate this subtitle dialogue into {target_language}.\n\nSOURCE:\n\"{target_text}\"\n\nReturn only the translated dialogue.\nDo not return the source text.\nDo not explain or classify."

            schema = {
                "type": "OBJECT",
                "properties": {
                    "translation": {"type": "STRING"}
                },
                "required": ["translation"]
            }

            def _safe_parse(raw_resp: str) -> Optional[str]:
                if not raw_resp:
                    return None
                clean_text = raw_resp.strip()
                if clean_text.startswith("```"):
                    lines = clean_text.split('\n')
                    if lines[0].startswith("```"): lines = lines[1:]
                    if lines and lines[-1].startswith("```"): lines = lines[:-1]
                    clean_text = "\n".join(lines).strip()
                try:
                    data = json.loads(clean_text)
                    res = data.get("translation", "")
                    if not is_usable_translation(res):
                        logger.info(f"Escalation line {target_idx} attempt {i+1}/3: rejected blank")
                        if job_id:
                            from app.core.db import append_job_log
                            append_job_log(job_id, f"Escalation cue {target_idx + 1} attempt {i+1}/3: rejected blank")
                        return None
                    return res
                except Exception as e:
                    logger.error(f"Escalation line {target_idx} attempt {i+1}/3 JSON parse failed: {e}. Raw: {raw_resp[:50]}")
                    if job_id:
                        from app.core.db import append_job_log
                        append_job_log(job_id, f"Escalation cue {target_idx + 1} attempt {i+1}/3: invalid semantic response")
                    return None

            try:
                raw_out = await self._execute_single_escalation_call(
                    provider=provider,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    schema=schema,
                    target_language=target_language,
                    target_text=target_text,
                    source_language=source_language,
                    job_id=job_id,
                )
                res = _safe_parse(raw_out)
                if res:
                    if is_meaningful_translation(target_text, res):
                        logger.info(f"Escalation line {target_idx} attempt {i+1}/3: translated successfully")
                        if job_id:
                            from app.core.db import append_job_log
                            append_job_log(job_id, f"Escalation cue {target_idx + 1} attempt {i+1}/3: translated successfully")
                        return res
                    else:
                        # Identical candidate: perform early semantic verification if not already audited
                        if res in audited_rejected_texts:
                            logger.info(f"Escalation line {target_idx} attempt {i+1}/3: identical candidate already rejected by semantic verifier, skipping duplicate verification call")
                            if exhausted_strategies is not None:
                                exhausted_strategies.add(strategy_key)
                        else:
                            candidate_items = [{
                                "id": target_idx,
                                "target": res,
                                "text": target_text,
                                "context_before": prev_text,
                                "context_after": next_text,
                                "proposed_reason": f"escalation_{attempt_type}"
                            }]
                            verified_ids = set()
                            try:
                                verified_ids = await self.verify_alphabetic_invariants_batch(
                                    candidate_items,
                                    target_language=target_language,
                                    source_language=source_language,
                                    show_title=show_title,
                                    job_id=job_id
                                )
                            except (DailyQuotaExhaustedError, RequestBudgetExhaustedError, ProviderConfigurationError):
                                raise
                            except Exception as e:
                                logger.error(f"Escalation line {target_idx} semantic invariant audit failed: {e}")

                            if target_idx in verified_ids:
                                logger.info(f"Escalation line {target_idx} attempt {i+1}/3: early verified as semantic invariant")
                                if context_verified_ids is not None:
                                    context_verified_ids.add(target_idx)
                                if job_id:
                                    from app.core.db import append_job_log
                                    append_job_log(job_id, f"Escalation: Cue {target_idx + 1} early-verified as Semantic Invariant on attempt {i+1}")
                                return res
                            else:
                                audited_rejected_texts.add(res)
                                if exhausted_strategies is not None:
                                    exhausted_strategies.add(strategy_key)
                                logger.info(f"Escalation line {target_idx} attempt {i+1}/3: candidate identical to source rejected by semantic verifier")
                                if job_id:
                                    from app.core.db import append_job_log
                                    append_job_log(job_id, f"Escalation cue {target_idx + 1} attempt {i+1}/3: candidate rejected by semantic verifier")
                elif exhausted_strategies is not None:
                    exhausted_strategies.add(strategy_key)

            except (DailyQuotaExhaustedError, RequestBudgetExhaustedError, ProviderConfigurationError):
                raise
            except Exception as e:
                logger.error(f"Escalation line {target_idx} API call failed: {e}")
                raise ProviderUnavailableError(f"Escalation failed: {e}") from e

        if job_id:
            from app.core.db import append_job_log
            append_job_log(job_id, f"Escalation cue {target_idx + 1} exhausted 3 semantic attempts")
        return None

    @with_retry
    async def translate_batch_gemini(self, items: List[dict], target_language: str, model_name: str, source_language: str = "English", context_lines: List[dict] = None, show_title: str = "", job_id: Optional[int] = None) -> List[dict]:
        client = self.get_gemini_client()

        context_section = ""
        if context_lines:
            context_section = "\n\nCONTEXT from previous batch (DO NOT translate these, use them only for tone and consistency):\n"
            for ctx in context_lines:
                if ctx.get("translated"):
                    context_section += f'  Original: "{ctx["original"]}" → Translated: "{ctx["translated"]}"\n'
                else:
                    context_section += f'  Original: "{ctx["original"]}"\n'

        prompt = build_translation_prompt(items, target_language, context_section)
        schema = build_translation_output_schema(items, target_language)

        glossary = get_setting("glossary", "")
        from app.core.ai_providers import get_model_capabilities
        caps = get_model_capabilities("gemini", model_name)
        config_kwargs = {
            "system_instruction": get_system_instruction(target_language, glossary=glossary, show_title=show_title, source_language=source_language),
            "response_mime_type": "application/json",
            "response_schema": schema,
            "max_output_tokens": 8192,
        }
        if caps.temperature:
            config_kwargs["temperature"] = 0.1
        config = types.GenerateContentConfig(**config_kwargs)

        def call_gemini(model_to_use):
            _mark_sdk_started(_usage_token_ctx.get(None))
            return client.models.generate_content(
                model=model_to_use,
                contents=prompt,
                config=config
            )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, contextvars.copy_context().run, lambda: call_gemini(model_name))

        return extract_json_safely(_capture_gemini_tokens(response).text)

    @with_retry
    async def translate_batch_openai(self, items: List[dict], target_language: str, model_name: str, source_language: str = "English", context_lines: List[dict] = None, show_title: str = "", job_id: Optional[int] = None) -> List[dict]:
        client = self.get_openai_client()

        context_section = ""
        if context_lines:
            context_section = "\n\nCONTEXT from previous batch (DO NOT translate these, use them only for tone and consistency):\n"
            for ctx in context_lines:
                if ctx.get("translated"):
                    context_section += f'  Original: "{ctx["original"]}" → Translated: "{ctx["translated"]}"\n'
                else:
                    context_section += f'  Original: "{ctx["original"]}"\n'

        prompt = build_translation_prompt(items, target_language, context_section)
        schema = build_translation_output_schema(items, target_language)

        glossary = get_setting("glossary", "")
        from app.core.ai_providers import get_model_capabilities
        caps = get_model_capabilities("openai", model_name)

        def call_openai(model_to_use):
            _mark_sdk_started(_usage_token_ctx.get(None))
            kwargs = {
                "model": model_to_use,
                "messages": [
                    {"role": "system", "content": get_system_instruction(target_language, glossary=glossary, show_title=show_title, source_language=source_language)},
                    {"role": "user", "content": prompt}
                ],
            }
            if schema and caps.structured_output:
                converted_schema = self._convert_to_openai_json_schema(schema)
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "translation_response",
                        "strict": True,
                        "schema": converted_schema,
                    },
                }
            elif caps.json_object:
                kwargs["response_format"] = {"type": "json_object"}
            effective_reasoning_effort = (
                kwargs.get("reasoning_effort")
                or (isinstance(kwargs.get("reasoning"), dict) and kwargs["reasoning"].get("effort"))
                or caps.default_reasoning_effort
            )
            effective_reasoning_none = (effective_reasoning_effort == "none")
            can_send_temperature = caps.temperature or (caps.temperature_requires_no_reasoning and effective_reasoning_none)
            if can_send_temperature:
                kwargs["temperature"] = 0.1
            return client.chat.completions.create(**kwargs)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, contextvars.copy_context().run, lambda: call_openai(model_name))

        return extract_json_safely(_capture_openai_tokens(response).choices[0].message.content)

    @with_retry
    async def translate_batch_deepl(self, items: List[dict], target_language: str, source_language: str = "English", context_lines: List[dict] = None, job_id: Optional[int] = None) -> List[dict]:
        api_key = get_setting("deepl_api_key", "")
        if not api_key:
            raise ValueError("DeepL API Key is not configured.")

        from app.core.languages import get_deepl_source_code, get_deepl_target_code
        target_lang_code = get_deepl_target_code(target_language)
        source_lang_code = get_deepl_source_code(source_language)
        url = "https://api-free.deepl.com/v2/translate" if api_key.endswith(":fx") else "https://api.deepl.com/v2/translate"

        texts = [it["text"] for it in items]
        model_type = get_setting("deepl_model_type", "prefer_quality_optimized")
        _mark_sdk_started(_usage_token_ctx.get(None))
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url,
                headers={"Authorization": f"DeepL-Auth-Key {api_key}"},
                json={"text": texts, "target_lang": target_lang_code, "source_lang": source_lang_code, "model_type": model_type}
            )
            resp.raise_for_status()
            data = resp.json()
            translations = data.get("translations", [])
            if len(translations) != len(items):
                raise ValueError(f"DeepL returned {len(translations)} translations, but expected {len(items)}.")
            return [{"id": items[i]["id"], "text": translations[i]["text"]} for i in range(len(items))]

    @with_retry
    async def translate_batch_ollama(self, items: List[dict], target_language: str, model_name: str, source_language: str = "English", context_lines: List[dict] = None, show_title: str = "", job_id: Optional[int] = None, provider: str = "ollama") -> List[dict]:
        ollama_url = get_setting("ollama_url", "http://localhost:11434").rstrip("/")

        context_section = ""
        if context_lines:
            context_section = "\n\nCONTEXT from previous batch (DO NOT translate these, use them only for tone and consistency):\n"
            for ctx in context_lines:
                if ctx.get("translated"):
                    context_section += f'  Original: "{ctx["original"]}" → Translated: "{ctx["translated"]}"\n'
                else:
                    context_section += f'  Original: "{ctx["original"]}"\n'

        glossary = get_setting("glossary", "")
        prompt = f"{get_system_instruction(target_language, glossary=glossary, show_title=show_title, source_language=source_language)}\n\n{build_translation_prompt(items, target_language, context_section)}"

        from app.core.ai_providers import get_model_capabilities
        caps = get_model_capabilities(provider or "ollama", model_name)
        req_payload = {
            "model": model_name or "llama3",
            "prompt": prompt,
            "stream": False,
        }
        if caps.json_object or caps.structured_output:
            req_payload["format"] = "json"
        if caps.temperature:
            req_payload["options"] = {"temperature": 0.1}

        _mark_sdk_started(_usage_token_ctx.get(None))
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{ollama_url}/api/generate",
                json=req_payload
            )
            resp.raise_for_status()
            data = resp.json()
            return extract_json_safely(data.get("response", "{}"))

    @with_retry
    async def audit_cue_alignment_window(
        self,
        source_items: List[dict],
        translated_items: List[dict],
        target_language: str = "Swedish",
        source_language: str = "English",
        show_title: str = "",
        job_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Language-agnostic semantic alignment auditor.
        Analyzes pairs of source and translated subtitle cues within a window
        and determines if 1-to-1 alignment is preserved, or if a semantic shift/merge occurred.
        """
        if not source_items or not translated_items:
            return {"alignment_verdict": "ALIGNED", "confidence": "HIGH", "details": "Empty input"}

        from app.core.ai_providers import resolve_job_provider_context, context_from_settings
        _audit_ctx = resolve_job_provider_context(job_id) if job_id else context_from_settings()
        show_context = f' for "{show_title}"' if show_title else ""
        prompt = (
            f"You are a strict subtitle synchronization and alignment auditor{show_context}.\n"
            f"Analyze the following {len(source_items)} pairs of SOURCE and TRANSLATED subtitle cues (paired by integer ID).\n"
            "Determine if each translated cue strictly corresponds 1-to-1 to its matching source cue ID, "
            "or if there is a semantic alignment shift (e.g. translated cue N actually translates source cue N+1 or N-1), "
            "or if multiple source cues were merged into one target cue causing subsequent cues to shift.\n\n"
            "CRITICAL RULES FOR AUDIT:\n"
            "1. If a sentence naturally spans across two consecutive cues (split sentence) and both parts are translated accurately in their corresponding cues, the verdict MUST be ALIGNED.\n"
            "2. Only mark MERGED if one target cue absorbs the entire content of multiple source cues, leaving subsequent target cues empty, duplicate, or shifted.\n"
            "3. Only mark SHIFT_PLUS_1 or SHIFT_MINUS_1 if target cues are sequentially misaligned against their source cue IDs.\n\n"
            "SOURCE CUES:\n" + json.dumps(source_items, ensure_ascii=False) + "\n\n"
            "TRANSLATED CUES:\n" + json.dumps(translated_items, ensure_ascii=False) + "\n\n"
            "Output a JSON object with keys:\n"
            "  \"alignment_verdict\": \"ALIGNED\" | \"SHIFT_PLUS_1\" | \"SHIFT_MINUS_1\" | \"MERGED\" | \"UNCERTAIN\"\n"
            "  \"confidence\": \"HIGH\" | \"MEDIUM\" | \"LOW\"\n"
            "  \"details\": \"concise explanation\"\n"
        )

        schema = {
            "type": "OBJECT",
            "properties": {
                "alignment_verdict": {"type": "STRING"},
                "confidence": {"type": "STRING"},
                "details": {"type": "STRING"}
            },
            "required": ["alignment_verdict", "confidence"]
        }

        provider = _audit_ctx.provider
        model_name = _audit_ctx.model
        system_prompt = f"You are a strict subtitle synchronization and alignment auditor{show_context}."

        try:
            raw_text = await self._dispatch_llm_completion(
                provider=provider,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=prompt,
                schema=schema,
                temperature=0.0,
                job_id=job_id,
            )
            clean = (raw_text or "").strip()
            if clean.startswith("```"):
                lines = clean.split('\n')
                if lines[0].startswith("```"): lines = lines[1:]
                if lines and lines[-1].startswith("```"): lines = lines[:-1]
                clean = "\n".join(lines).strip()
            data = json.loads(clean)
            if isinstance(data, dict) and "alignment_verdict" in data:
                return data
        except Exception as e:
            logger.warning(f"Alignment audit exception: {e}")

        return {"alignment_verdict": "UNCERTAIN", "confidence": "LOW", "details": "Could not determine"}

    @with_retry
    async def audit_cue_alignment_batch(
        self,
        windows: List[dict],
        target_language: str = "Swedish",
        source_language: str = "English",
        show_title: str = "",
        job_id: Optional[int] = None,
        escalate_uncertain: bool = True
    ) -> Dict[int, dict]:
        """
        Language-agnostic batch semantic alignment auditor with strict 1-to-1 window contract,
        deterministic response validation, and automatic focused single-window escalation.
        """
        if not windows:
            return {}

        formatted_windows = []
        for idx, w in enumerate(windows):
            fw = dict(w)
            if "window_id" not in fw:
                fw["window_id"] = idx + 1
            formatted_windows.append(fw)

        from app.core.ai_providers import resolve_job_provider_context, context_from_settings
        _batch_ctx = resolve_job_provider_context(job_id) if job_id else context_from_settings()
        show_context = f' for "{show_title}"' if show_title else ""
        prompt = (
            f"You are a strict subtitle synchronization and alignment auditor{show_context}.\n"
            f"Analyze the following {len(formatted_windows)} suspect windows of paired SOURCE ({source_language}) and TRANSLATED ({target_language}) subtitle cues.\n"
            "For EACH window in the input array, determine if translated cues strictly correspond 1-to-1 to their matching source cue ID, "
            "or if there is a semantic shift (e.g. target cue N translates source cue N+1, N+2, or N-1), or a merge.\n\n"
            "CRITICAL RULES FOR AUDIT:\n"
            "1. If a sentence naturally spans across two consecutive cues (split sentence) and both parts are translated accurately in their corresponding cues, the verdict MUST be ALIGNED.\n"
            "2. Only mark MERGED if one target cue absorbs the entire content of multiple source cues, leaving subsequent target cues empty, duplicate, or shifted.\n"
            "3. Only mark SHIFT_PLUS_1 or SHIFT_MINUS_1 if target cues are sequentially misaligned against their source cue IDs.\n\n"
            "You MUST return EXACTLY ONE result object for EVERY input window_id.\n\n"
            "WINDOWS TO AUDIT:\n" + json.dumps(formatted_windows, ensure_ascii=False) + "\n\n"
            "Output JSON format:\n"
            "{\n"
            "  \"results\": [\n"
            "    {\"window_id\": int, \"verdict\": \"ALIGNED\" | \"SHIFT_PLUS_1\" | \"SHIFT_MINUS_1\" | \"MERGED\" | \"UNCERTAIN\", \"confidence\": \"HIGH\" | \"MEDIUM\" | \"LOW\", \"details\": str}\n"
            "  ]\n"
            "}"
        )

        schema = {
            "type": "OBJECT",
            "properties": {
                "results": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "window_id": {"type": "INTEGER"},
                            "verdict": {"type": "STRING"},
                            "confidence": {"type": "STRING"},
                            "details": {"type": "STRING"}
                        },
                        "required": ["window_id", "verdict", "confidence", "details"]
                    }
                }
            },
            "required": ["results"]
        }

        provider = _batch_ctx.provider
        model_name = _batch_ctx.model
        system_prompt = f"You are a strict subtitle synchronization and alignment auditor{show_context}."
        raw_text = await self._dispatch_llm_completion(
            provider=provider,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=prompt,
            schema=schema,
            temperature=0.0,
            job_id=job_id,
        )

        parsed_data = {}
        try:
            clean = (raw_text or "").strip()
            if clean.startswith("```"):
                lines = clean.split('\n')
                if lines[0].startswith("```"): lines = lines[1:]
                if lines and lines[-1].startswith("```"): lines = lines[:-1]
                clean = "\n".join(lines).strip()
            parsed_data = json.loads(clean)
        except Exception as e:
            logger.warning(f"Batch alignment audit JSON parse failed: {e}")

        valid_map, val_report = validate_audit_batch_results(formatted_windows, parsed_data)

        # Focused escalation for any missing windows (and UNCERTAIN windows if escalate_uncertain is enabled)
        missing_ids = val_report.get("missing_ids", [])
        uncertain_ids = [wid for wid, res in valid_map.items() if res.get("verdict") == "UNCERTAIN"] if escalate_uncertain else []
        escalate_ids = set(missing_ids + uncertain_ids)

        if escalate_ids:
            window_by_id = {w["window_id"]: w for w in formatted_windows}
            esc_sem = asyncio.Semaphore(3)

            async def _escalate_single(wid: int):
                if wid not in window_by_id:
                    return
                target_w = window_by_id[wid]
                async with esc_sem:
                    try:
                        focused_res = await self.audit_cue_alignment_window(
                            target_w.get("source", []),
                            target_w.get("target", []),
                            target_language=target_language,
                            source_language=source_language,
                            show_title=show_title,
                            job_id=job_id
                        )
                        valid_map[wid] = {
                            "window_id": wid,
                            "verdict": focused_res.get("alignment_verdict", "UNCERTAIN"),
                            "confidence": focused_res.get("confidence", "LOW"),
                            "details": focused_res.get("details", "Focused single-window escalation")
                        }
                    except Exception as e:
                        logger.warning(f"Focused escalation for window {wid} failed: {e}")
                        if wid not in valid_map:
                            valid_map[wid] = {
                                "window_id": wid,
                                "verdict": "UNCERTAIN",
                                "confidence": "LOW",
                                "details": f"Escalation failed: {e}"
                            }

            await asyncio.gather(*(_escalate_single(wid) for wid in escalate_ids))

        return valid_map

    # Backward-compatible alias
    audit_cue_alignment_bulk_windows = audit_cue_alignment_batch

    @with_retry
    async def audit_batch_semantic_integrity(
        self,
        batch_payloads: List[Dict[str, Any]],
        target_language: str = "Swedish",
        source_language: str = "English",
        show_title: str = "",
        job_id: Optional[int] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Consolidated multi-batch semantic alignment auditor.
        Evaluates stratified representative cue samples across multiple primary batches in a single AI call.
        Returns a dict mapping batch_id -> {
            "batch_id": int,
            "verdict": "ALIGNED" | "SUSPECT" | "CORRUPT" | "UNCERTAIN",
            "confidence": "HIGH" | "MEDIUM" | "LOW",
            "details": str
        }
        """
        if not batch_payloads:
            return {}

        from app.core.ai_providers import resolve_job_provider_context, context_from_settings
        _ctx = resolve_job_provider_context(job_id) if job_id else context_from_settings()
        show_context = f' for "{show_title}"' if show_title else ""

        prompt = (
            f"You are a strict subtitle synchronization and semantic alignment auditor{show_context}.\n"
            f"Analyze the following {len(batch_payloads)} primary translation batches.\n"
            f"For EACH batch, representative pairs of SOURCE ({source_language}) and TRANSLATED ({target_language}) subtitle cues are provided (paired strictly by integer ID).\n\n"
            "For EACH batch, determine if the translated cues strictly correspond 1-to-1 to their matching source cue ID, "
            "or if there is a semantic alignment shift (e.g. target cue N actually translates source cue N+1, N+2, or N-1, or cues were merged/skipped causing sequential misalignment).\n\n"
            "CRITICAL AUDIT RULES:\n"
            "1. 'ALIGNED': Each target cue accurately translates the source cue with the same integer ID. Natural split sentences spanning across consecutive cues that are properly translated in their respective cues MUST be judged ALIGNED.\n"
            "2. 'SUSPECT' / 'CORRUPT': Target cues are sequentially misaligned against their source cue IDs (e.g. target text belongs to a different source cue number), or translation has collapsed/shifted.\n"
            "3. 'UNCERTAIN': Ambiguous, insufficient non-verbal cues, or conflicting evidence.\n\n"
            "You MUST return EXACTLY ONE result object for EVERY batch_id.\n\n"
            "BATCHES TO AUDIT:\n" + json.dumps(batch_payloads, ensure_ascii=False) + "\n\n"
            "Output JSON format:\n"
            "{\n"
            "  \"batches\": [\n"
            "    {\"batch_id\": int, \"verdict\": \"ALIGNED\" | \"SUSPECT\" | \"CORRUPT\" | \"UNCERTAIN\", \"confidence\": \"HIGH\" | \"MEDIUM\" | \"LOW\", \"details\": str}\n"
            "  ]\n"
            "}"
        )

        schema = {
            "type": "OBJECT",
            "properties": {
                "batches": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "batch_id": {"type": "INTEGER"},
                            "verdict": {"type": "STRING"},
                            "confidence": {"type": "STRING"},
                            "details": {"type": "STRING"}
                        },
                        "required": ["batch_id", "verdict", "confidence", "details"]
                    }
                }
            },
            "required": ["batches"]
        }

        provider = _ctx.provider
        model_name = _ctx.model
        system_prompt = f"You are a strict subtitle synchronization and semantic alignment auditor{show_context}."

        try:
            raw_text = await self._dispatch_llm_completion(
                provider=provider,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=prompt,
                schema=schema,
                temperature=0.0,
                job_id=job_id,
            )
            clean = (raw_text or "").strip()
            if clean.startswith("```"):
                lines = clean.split('\n')
                if lines[0].startswith("```"): lines = lines[1:]
                if lines and lines[-1].startswith("```"): lines = lines[:-1]
                clean = "\n".join(lines).strip()
            data = json.loads(clean)
            items = []
            if isinstance(data, dict):
                items = data.get("batches", data.get("results", []))
            # ----------------------------------------------------------------
            # CONTRACT VALIDATION — performed on the RAW item list before
            # collapsing to a Dict, so that duplicates are detectable.
            # ----------------------------------------------------------------
            _VALID_VERDICTS = {"ALIGNED", "SUSPECT", "CORRUPT", "UNCERTAIN"}
            requested_ids = {bp["batch_id"] for bp in batch_payloads}

            # Pass 1 – scan raw items for duplicates and unknown IDs.
            # Any contract violation immediately taints this entire response:
            # if the AI hallucinated IDs or returned duplicates we cannot
            # trust which other entries may also be wrong.
            seen_ids: set = set()
            duplicate_ids: set = set()
            unknown_ids: set = set()

            for item in items:
                if not isinstance(item, dict) or "batch_id" not in item:
                    continue
                bid = item["batch_id"]
                if bid not in requested_ids:
                    unknown_ids.add(bid)
                elif bid in seen_ids:
                    duplicate_ids.add(bid)
                else:
                    seen_ids.add(bid)

            contract_violated = bool(duplicate_ids or unknown_ids)

            if unknown_ids:
                logger.warning(
                    f"audit_batch_semantic_integrity: AI returned unknown batch_id(s) "
                    f"{sorted(unknown_ids)} not in requested={sorted(requested_ids)} "
                    f"– audit contract violation, all requested batches → UNCERTAIN"
                )
            if duplicate_ids:
                logger.warning(
                    f"audit_batch_semantic_integrity: AI returned duplicate batch_id(s) "
                    f"{sorted(duplicate_ids)} – audit contract violation, "
                    f"affected batches → UNCERTAIN"
                )

            # Pass 2 – if contract is violated, all requested batches become UNCERTAIN.
            # A tainted AI response cannot be trusted for any batch within it.
            if contract_violated:
                return {
                    bp["batch_id"]: {
                        "batch_id": bp["batch_id"],
                        "verdict": "UNCERTAIN",
                        "confidence": "LOW",
                        "details": (
                            f"Audit contract violation: "
                            f"duplicate_ids={sorted(duplicate_ids)} "
                            f"unknown_ids={sorted(unknown_ids)} (fail-closed)"
                        )
                    }
                    for bp in batch_payloads
                }

            # Pass 3 – collapse clean raw items into results dict.
            results = {}
            for item in items:
                if not isinstance(item, dict) or "batch_id" not in item:
                    continue
                bid = item["batch_id"]
                if bid not in requested_ids:
                    continue  # already handled above
                results[bid] = {
                    "batch_id": bid,
                    "verdict": item.get("verdict", "UNCERTAIN"),
                    "confidence": item.get("confidence", "LOW"),
                    "details": item.get("details", "")
                }

            # Pass 4 – clamp invalid verdict values to UNCERTAIN (fail-closed).
            for bid, v in results.items():
                raw_verdict = (v.get("verdict") or "").strip().upper()
                if raw_verdict not in _VALID_VERDICTS:
                    logger.warning(
                        f"audit_batch_semantic_integrity: batch {bid} returned invalid verdict "
                        f"'{raw_verdict}' – clamping to UNCERTAIN"
                    )
                    v["verdict"] = "UNCERTAIN"
                    v["confidence"] = "LOW"

            # Pass 5 – fill missing batch_ids with UNCERTAIN (no evidence ≠ aligned).
            for bp in batch_payloads:
                bid = bp["batch_id"]
                if bid not in results:
                    logger.warning(
                        f"audit_batch_semantic_integrity: batch {bid} missing from AI response "
                        f"– marking UNCERTAIN (fail-closed)"
                    )
                    results[bid] = {
                        "batch_id": bid,
                        "verdict": "UNCERTAIN",
                        "confidence": "LOW",
                        "details": "Batch result absent from AI response (fail-closed)"
                    }
            return results
        except Exception as e:
            logger.warning(f"Consolidated batch semantic audit exception: {e}")
            return {}

    @with_retry
    async def confirm_batch_semantic_integrity(
        self,
        batch_id: int,
        start_id: int,
        end_id: int,
        source_items: List[dict],
        target_items: List[dict],
        target_language: str = "Swedish",
        source_language: str = "English",
        show_title: str = "",
        job_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Focused single-batch confirmation call.
        Verifies if target cues in suspect batch strictly correspond 1-to-1 to their source cue ID.
        """
        if not source_items or not target_items:
            return {"verdict": "ALIGNED", "confidence": "HIGH", "details": "Empty input"}

        from app.core.ai_providers import resolve_job_provider_context, context_from_settings
        _ctx = resolve_job_provider_context(job_id) if job_id else context_from_settings()
        show_context = f' for "{show_title}"' if show_title else ""

        prompt = (
            f"You are a strict subtitle synchronization and semantic alignment auditor{show_context}.\n"
            f"Examine primary batch {batch_id} (cues {start_id}-{end_id}) containing {len(source_items)} pairs of SOURCE ({source_language}) and TRANSLATED ({target_language}) cues.\n"
            "Determine if each translated cue semantically corresponds strictly 1-to-1 to the source cue with the same ID.\n\n"
            "VERDICTS:\n"
            "- 'ALIGNED': Target cues correctly translate matching source cue IDs. Split sentences across cues are ALIGNED.\n"
            "- 'CORRUPT': Target cues are misaligned, shifted, delayed, advanced, or merged across cue IDs.\n"
            "- 'UNCERTAIN': Contradictory evidence or cannot determine with high confidence.\n\n"
            "SOURCE CUES:\n" + json.dumps(source_items, ensure_ascii=False) + "\n\n"
            "TRANSLATED CUES:\n" + json.dumps(target_items, ensure_ascii=False) + "\n\n"
            "Output a JSON object with keys:\n"
            "  \"verdict\": \"ALIGNED\" | \"CORRUPT\" | \"UNCERTAIN\"\n"
            "  \"confidence\": \"HIGH\" | \"MEDIUM\" | \"LOW\"\n"
            "  \"details\": \"concise explanation\"\n"
        )

        schema = {
            "type": "OBJECT",
            "properties": {
                "verdict": {"type": "STRING"},
                "confidence": {"type": "STRING"},
                "details": {"type": "STRING"}
            },
            "required": ["verdict", "confidence"]
        }

        provider = _ctx.provider
        model_name = _ctx.model
        system_prompt = f"You are a strict subtitle synchronization and alignment auditor{show_context}."

        try:
            raw_text = await self._dispatch_llm_completion(
                provider=provider,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=prompt,
                schema=schema,
                temperature=0.0,
                job_id=job_id,
            )
            clean = (raw_text or "").strip()
            if clean.startswith("```"):
                lines = clean.split('\n')
                if lines[0].startswith("```"): lines = lines[1:]
                if lines and lines[-1].startswith("```"): lines = lines[:-1]
                clean = "\n".join(lines).strip()
            data = json.loads(clean)
            if isinstance(data, dict) and "verdict" in data:
                return data
            elif isinstance(data, dict) and "alignment_verdict" in data:
                v = data["alignment_verdict"]
                return {
                    "verdict": "ALIGNED" if v == "ALIGNED" else ("CORRUPT" if v in {"SHIFT_PLUS_1", "SHIFT_MINUS_1", "MERGED", "COMPLEX_SHIFT"} else "UNCERTAIN"),
                    "confidence": data.get("confidence", "LOW"),
                    "details": data.get("details", "")
                }
        except Exception as e:
            logger.warning(f"Batch confirmation exception: {e}")

        return {"verdict": "UNCERTAIN", "confidence": "LOW", "details": "Confirmation failed"}

    @with_retry
    async def verify_repaired_batch_integrity(
        self,
        batch_id: int,
        start_id: int,
        end_id: int,
        source_items: List[dict],
        target_items: List[dict],
        target_language: str = "Swedish",
        source_language: str = "English",
        show_title: str = "",
        job_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Focused post-repair semantic verification call for a newly repaired primary batch.
        """
        return await self.confirm_batch_semantic_integrity(
            batch_id=batch_id,
            start_id=start_id,
            end_id=end_id,
            source_items=source_items,
            target_items=target_items,
            target_language=target_language,
            source_language=source_language,
            show_title=show_title,
            job_id=job_id
        )

    @with_retry
    async def translate_batch_anthropic(self, items: List[dict], target_language: str, model_name: str, source_language: str = "English", context_lines: List[dict] = None, show_title: str = "", job_id: Optional[int] = None) -> List[dict]:
        context_section = ""
        if context_lines:
            context_section = "\n\nCONTEXT from previous batch (DO NOT translate these, use them only for tone and consistency):\n"
            for ctx in context_lines:
                if ctx.get("translated"):
                    context_section += f'  Original: "{ctx["original"]}" → Translated: "{ctx["translated"]}"\n'
                else:
                    context_section += f'  Original: "{ctx["original"]}"\n'

        prompt = build_translation_prompt(items, target_language, context_section)
        schema = build_translation_output_schema(items, target_language)
        glossary = get_setting("glossary", "")
        system_prompt = get_system_instruction(target_language, glossary=glossary, show_title=show_title, source_language=source_language)

        raw = await self._dispatch_llm_completion(
            provider="anthropic",
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=prompt,
            schema=schema,
            temperature=0.1,
            job_id=job_id,
        )
        return extract_json_safely(raw or "{}")

    @with_retry
    async def translate_batch_openrouter(self, items: List[dict], target_language: str, model_name: str, source_language: str = "English", context_lines: List[dict] = None, show_title: str = "", job_id: Optional[int] = None) -> List[dict]:
        context_section = ""
        if context_lines:
            context_section = "\n\nCONTEXT from previous batch (DO NOT translate these, use them only for tone and consistency):\n"
            for ctx in context_lines:
                if ctx.get("translated"):
                    context_section += f'  Original: "{ctx["original"]}" → Translated: "{ctx["translated"]}"\n'
                else:
                    context_section += f'  Original: "{ctx["original"]}"\n'

        prompt = build_translation_prompt(items, target_language, context_section)
        schema = build_translation_output_schema(items, target_language)
        glossary = get_setting("glossary", "")
        system_prompt = get_system_instruction(target_language, glossary=glossary, show_title=show_title, source_language=source_language)

        raw = await self._dispatch_llm_completion(
            provider="openrouter",
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=prompt,
            schema=schema,
            temperature=0.1,
            job_id=job_id,
        )
        return extract_json_safely(raw or "{}")

    @with_retry
    async def translate_batch_deepseek(self, items: List[dict], target_language: str, model_name: str, source_language: str = "English", context_lines: List[dict] = None, show_title: str = "", job_id: Optional[int] = None) -> List[dict]:
        context_section = ""
        if context_lines:
            context_section = "\n\nCONTEXT from previous batch (DO NOT translate these, use them only for tone and consistency):\n"
            for ctx in context_lines:
                if ctx.get("translated"):
                    context_section += f'  Original: "{ctx["original"]}" → Translated: "{ctx["translated"]}"\n'
                else:
                    context_section += f'  Original: "{ctx["original"]}"\n'

        prompt = build_translation_prompt(items, target_language, context_section)
        schema = build_translation_output_schema(items, target_language)
        glossary = get_setting("glossary", "")
        system_prompt = get_system_instruction(target_language, glossary=glossary, show_title=show_title, source_language=source_language)

        raw = await self._dispatch_llm_completion(
            provider="deepseek",
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=prompt,
            schema=schema,
            temperature=0.1,
            job_id=job_id,
        )
        return extract_json_safely(raw or "{}")

    @with_retry
    async def translate_batch_custom(self, items: List[dict], target_language: str, model_name: str, source_language: str = "English", context_lines: List[dict] = None, show_title: str = "", job_id: Optional[int] = None) -> List[dict]:
        context_section = ""
        if context_lines:
            context_section = "\n\nCONTEXT from previous batch (DO NOT translate these, use them only for tone and consistency):\n"
            for ctx in context_lines:
                if ctx.get("translated"):
                    context_section += f'  Original: "{ctx["original"]}" → Translated: "{ctx["translated"]}"\n'
                else:
                    context_section += f'  Original: "{ctx["original"]}"\n'

        prompt = build_translation_prompt(items, target_language, context_section)
        schema = build_translation_output_schema(items, target_language)
        glossary = get_setting("glossary", "")
        system_prompt = get_system_instruction(target_language, glossary=glossary, show_title=show_title, source_language=source_language)

        raw = await self._dispatch_llm_completion(
            provider="custom",
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=prompt,
            schema=schema,
            temperature=0.1,
            job_id=job_id,
        )
        return extract_json_safely(raw or "{}")

    async def translate_batch(self, items: List[dict], target_language: str = "English", source_language: str = "English", context_lines: List[dict] = None, show_title: str = "", job_id: Optional[int] = None, provider_ctx=None) -> List[dict]:
        from app.core.ai_providers import resolve_job_provider_context, context_from_settings, normalize_provider
        _ctx = provider_ctx or (resolve_job_provider_context(job_id) if job_id else context_from_settings())
        provider = _ctx.provider
        model = _ctx.model

        if provider == "openai":
            return await self.translate_batch_openai(items, target_language, model, source_language=source_language, context_lines=context_lines, show_title=show_title, job_id=job_id)
        elif provider == "anthropic":
            return await self.translate_batch_anthropic(items, target_language, model, source_language=source_language, context_lines=context_lines, show_title=show_title, job_id=job_id)
        elif provider == "openrouter":
            return await self.translate_batch_openrouter(items, target_language, model, source_language=source_language, context_lines=context_lines, show_title=show_title, job_id=job_id)
        elif provider == "deepseek":
            return await self.translate_batch_deepseek(items, target_language, model, source_language=source_language, context_lines=context_lines, show_title=show_title, job_id=job_id)
        elif provider in ["custom", "custom_openai"]:
            return await self.translate_batch_custom(items, target_language, model, source_language=source_language, context_lines=context_lines, show_title=show_title, job_id=job_id)
        elif provider == "deepl":
            return await self.translate_batch_deepl(items, target_language, source_language=source_language, context_lines=context_lines, job_id=job_id)
        elif provider in ["ollama", "localai"]:
            return await self.translate_batch_ollama(items, target_language, model, source_language=source_language, context_lines=context_lines, show_title=show_title, job_id=job_id, provider=provider)
        elif provider == "gemini":
            return await self.translate_batch_gemini(items, target_language, model, source_language=source_language, context_lines=context_lines, show_title=show_title, job_id=job_id)
        else:
            from app.core.ai_providers import get_provider_spec
            raise ValueError(f"Unsupported AI provider for translation: {provider}")


    async def translate_srt_content(
        self,
        subs: List[srt.Subtitle],
        target_language: str = "English",
        source_language: str = "English",
        batch_size: int = 150,
        job_id: Optional[int] = None,
        show_title: Optional[str] = None,
        provenance_map: Optional[dict] = None
    ) -> List[srt.Subtitle]:
        total_lines = len(subs)
        batches = []
        for i in range(0, total_lines, batch_size):
            chunk = subs[i:i + batch_size]
            batch_payload = [{"id": j, "text": sub.content} for j, sub in enumerate(chunk, start=i)]
            batches.append((i, chunk, batch_payload))

        translated_subs = [
            srt.Subtitle(index=sub.index, start=sub.start, end=sub.end, content=sub.content)
            for sub in subs
        ]

        import os
        import app.core.db
        from app.core.languages import get_language
        data_dir = os.path.dirname(app.core.db.DB_PATH)
        lang_obj = get_language(target_language)
        lang_code = lang_obj.code if lang_obj else target_language.lower()[:2]
        partial_file = os.path.join(data_dir, f"job_{job_id}_{lang_code}_partial.json") if job_id else None

        import hashlib
        fingerprint = hashlib.md5("".join(s.content for s in subs).encode("utf-8")).hexdigest() + "_" + lang_code

        partial_dict = {}
        if partial_file and os.path.exists(partial_file):
            try:
                with open(partial_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if data.get("fingerprint") == fingerprint:
                    lines_data = data.get("lines", {})
                    partial_dict = {int(k): v for k, v in lines_data.items()}
                    for k, v in partial_dict.items():
                        if k < len(translated_subs):
                            translated_subs[k].content = v
                    logger.info(f"Loaded partial progress for job {job_id} ({len(partial_dict)} lines)")
                else:
                    logger.warning(f"Partial progress fingerprint mismatch for job {job_id}. Discarding.")
            except Exception as e:
                logger.error(f"Failed to load partial progress for job {job_id}: {e}")

        processed_count = 0
        context_window_size = 5
        global_tm_context = []
        if show_title:
            try:
                tm_context = get_translation_memory(
                    show_title,
                    limit=10,
                    source_language=source_language,
                    target_language=target_language
                )
                if tm_context:
                    global_tm_context.extend(tm_context)
            except Exception:
                pass

        state_lock = asyncio.Lock()
        successfully_dispatched_cues = set()
        concurrency = get_positive_int_setting("batch_concurrency", 2)
        sem = asyncio.Semaphore(concurrency)

        async def process_batch(batch_idx, start_idx, chunk, payload):
            nonlocal processed_count
            end_idx = min(start_idx + len(payload), total_lines)

            if payload and all(p["id"] in partial_dict for p in payload):
                async with state_lock:
                    processed_count += len(payload)
                    if job_id:
                        update_job(job_id, processed_lines=processed_count, current_batch=f"Skipping cached lines {start_idx + 1}-{end_idx} / {total_lines}")
                return

            all_safe_keep = all(is_safe_keep_prefilter(p["text"]) for p in payload)
            if all_safe_keep:
                async with state_lock:
                    for p in payload:
                        idx = p["id"]
                        translated_subs[idx].content = p["text"]
                        partial_dict[idx] = p["text"]
                    processed_count += len(payload)
                    if job_id:
                        update_job(job_id, processed_lines=processed_count, current_batch=f"Lines {start_idx + 1}-{end_idx} / {total_lines}")
                return

            if job_id:
                async with state_lock:
                    update_job(job_id, current_batch=f"Translating lines {start_idx + 1}-{end_idx} of {total_lines}")

            try:
                batch_context = list(global_tm_context)
                if start_idx > 0:
                    ctx_start = max(0, start_idx - context_window_size)
                    for idx in range(ctx_start, start_idx):
                        if subs[idx].content.strip() and subs[idx].content.strip() != "<i></i>":
                            batch_context.append({
                                "original": subs[idx].content,
                                "translated": partial_dict.get(idx, "")
                            })

                # Build missing payload preserving contiguous item sequence
                missing_payload = []
                orig_map = {}
                for p in payload:
                    idx = p["id"]
                    if idx not in partial_dict:
                        missing_payload.append(p)
                        orig_map[idx] = p["text"]

                res_dict = {}
                if missing_payload:
                    async def _translate_sub_payload_atomic(sub_payload: List[dict], sub_context=None, depth: int = 0) -> Dict[int, str]:
                        if not sub_payload:
                            return {}

                        # Attempt 1: Translate the sub_payload
                        try:
                            results = await self.translate_batch(
                                sub_payload,
                                target_language=target_language,
                                source_language=source_language,
                                context_lines=sub_context if sub_context else None,
                                show_title=show_title or "",
                                job_id=job_id
                            )
                        except TypeError:
                            results = await self.translate_batch(
                                sub_payload,
                                target_language=target_language,
                                context_lines=sub_context if sub_context else None,
                                show_title=show_title or "",
                            )

                        valid_map, report = validate_batch_translation_results(sub_payload, results)
                        if report["is_clean"]:
                            # Clean ID contract: all expected IDs present, no missing/unknown/duplicate/malformed IDs.
                            # Commit valid translations; content-invalid/identical items remain unresolved for downstream recovery.
                            return valid_map

                        # TRUE Structural anomaly detected (missing/unknown/duplicate/malformed IDs). Discard corrupted batch output.
                        sub_start_id = min(p["id"] for p in sub_payload) + 1
                        sub_end_id = max(p["id"] for p in sub_payload) + 1
                        missing_cnt = len(report["missing_ids"])
                        unk_dup_cnt = len(report["unknown_ids"]) + len(report["duplicate_ids"]) + report.get("malformed_count", 0)
                        logger.warning(
                            f"Primary batch lines {sub_start_id}-{sub_end_id} structural anomaly: "
                            f"missing={missing_cnt}, unknown/dup/malformed={unk_dup_cnt}. Discarding candidate output."
                        )
                        if job_id:
                            append_job_log(
                                job_id,
                                f"Primary batch lines {sub_start_id}-{sub_end_id}: True structural anomaly detected "
                                f"(missing: {missing_cnt}, unknown/dup/malformed: {unk_dup_cnt}). "
                                f"Discarding batch and retrying atomically from source-of-truth."
                            )

                        # Retry 1: Same sub_payload from original source of truth
                        if depth == 0 and len(sub_payload) > 1:
                            try:
                                retry_results = await self.translate_batch(
                                    sub_payload,
                                    target_language=target_language,
                                    source_language=source_language,
                                    context_lines=sub_context if sub_context else None,
                                    show_title=show_title or "",
                                    job_id=job_id
                                )
                                r_valid_map, r_report = validate_batch_translation_results(sub_payload, retry_results)
                                if r_report["is_clean"]:
                                    if job_id:
                                        append_job_log(job_id, f"Primary batch lines {sub_start_id}-{sub_end_id}: Retry succeeded (clean ID contract verified).")
                                    return r_valid_map
                            except Exception as e:
                                logger.warning(f"Primary batch retry failed: {e}")

                        # If retry was still anomalous and payload can be partitioned (bounded to depth < 2):
                        if depth < 2 and len(sub_payload) > 1:
                            mid = len(sub_payload) // 2
                            left_payload = sub_payload[:mid]
                            right_payload = sub_payload[mid:]
                            if job_id:
                                append_job_log(
                                    job_id,
                                    f"Primary batch lines {sub_start_id}-{sub_end_id}: Partitioning into sub-batches "
                                    f"({len(left_payload)} + {len(right_payload)} cues) to isolate structural anomaly."
                                )
                            left_map = await _translate_sub_payload_atomic(left_payload, sub_context=sub_context, depth=depth + 1)
                            right_ctx = list(sub_context) if sub_context else []
                            for p in left_payload[-3:]:
                                if p["id"] in left_map:
                                    right_ctx.append({"original": p["text"], "translated": left_map[p["id"]]})
                            right_map = await _translate_sub_payload_atomic(right_payload, sub_context=right_ctx, depth=depth + 1)
                            combined = dict(left_map)
                            combined.update(right_map)
                            return combined
                        else:
                            return valid_map

                    atomic_results = await _translate_sub_payload_atomic(missing_payload, sub_context=batch_context)
                    res_dict.update(atomic_results)

                for p in payload:
                    if p["id"] in partial_dict:
                        res_dict[p["id"]] = partial_dict[p["id"]]

                async with state_lock:
                    for p in payload:
                        successfully_dispatched_cues.add(p["id"])
                        idx = p["id"]
                        if idx in res_dict:
                            translated_subs[idx].content = res_dict[idx]
                            partial_dict[idx] = translated_subs[idx].content
                        elif is_safe_keep_prefilter(p["text"]):
                            translated_subs[idx].content = p["text"]
                            partial_dict[idx] = p["text"]

                    if partial_file:
                        try:
                            wrapper = {"fingerprint": fingerprint, "lines": partial_dict}
                            tmp_file = partial_file + f".tmp.{start_idx}"
                            with open(tmp_file, "w", encoding="utf-8") as f:
                                json.dump(wrapper, f, ensure_ascii=False)
                            os.replace(tmp_file, partial_file)
                        except Exception as e:
                            logger.error(f"Failed to save partial progress for job {job_id}: {e}")

                    processed_count += len(payload)
                    if job_id:
                        update_job(job_id, processed_lines=processed_count)

            except (ProviderUnavailableError, ProviderConfigurationError) as e:
                async with state_lock:
                    if job_id:
                        update_job(job_id, processed_lines=processed_count)
                raise e
            except Exception as e:
                # Re-raise quota errors so they propagate to the pipeline exception handler
                # which will set the job to DEFERRED (not FAILED)
                if isinstance(e, (DailyQuotaExhaustedError, RequestBudgetExhaustedError)):
                    async with state_lock:
                        if job_id:
                            update_job(job_id, processed_lines=processed_count)
                    raise

                err_str = str(e).lower()
                permanent = any(x in err_str for x in [
                    "401", "403", "unauthorized", "forbidden", "api key not valid",
                    "invalid api key", "not configured", "model_not_found", "permission_denied"
                ])
                if permanent:
                    async with state_lock:
                        if job_id:
                            update_job(job_id, processed_lines=processed_count)
                    raise ProviderConfigurationError(f"Permanent provider configuration error: {str(e)}")

                logger.error(f"Batch {start_idx} failed: {e}")
                async with state_lock:
                    processed_count += len(payload)
                    if job_id:
                        append_job_log(job_id, f"Warning: Lines {start_idx + 1}-{end_idx} could not be translated: {e}. Keeping original text.")
                        update_job(job_id, processed_lines=processed_count)


        async def run_batch_with_sem(batch_idx, start_idx, chunk, payload):
            async with sem:
                await process_batch(batch_idx, start_idx, chunk, payload)

        tasks = []
        for batch_idx, (start_idx, chunk, payload) in enumerate(batches):
            tasks.append(run_batch_with_sem(batch_idx, start_idx, chunk, payload))

        if tasks:
            await asyncio.gather(*tasks)

        # ------------------------------------------------------------------
        # Consolidated Global First-Pass Micro Repair
        # Instead of firing a separate micro-repair call per batch, collect all
        # unresolved/failed dialogue cues globally across the entire file and
        # repair them in consolidated bulk batches (reducing AI roundtrips from N down to 1).
        # ------------------------------------------------------------------
        unresolved_indices = [
            i for i, sub in enumerate(subs)
            if i in successfully_dispatched_cues
            and sub.content.strip()
            and sub.content.strip() != "<i></i>"
            and not is_safe_keep_prefilter(sub.content)
            and not is_meaningful_translation(sub.content, translated_subs[i].content)
        ]

        if unresolved_indices:
            global_repair_items = []
            for idx in unresolved_indices:
                ctx_before_parts = []
                for b_idx in range(max(0, idx - 2), idx):
                    bc = subs[b_idx].content.strip()
                    if bc and bc != "<i></i>":
                        btc = translated_subs[b_idx].content.strip()
                        if btc and btc != "<i></i>" and is_meaningful_translation(bc, btc):
                            ctx_before_parts.append(f"{bc} ({btc})")
                        else:
                            ctx_before_parts.append(bc)

                ctx_after_parts = []
                for a_idx in range(idx + 1, min(len(subs), idx + 3)):
                    ac = subs[a_idx].content.strip()
                    if ac and ac != "<i></i>":
                        ctx_after_parts.append(ac)

                global_repair_items.append({
                    "id": idx,
                    "target": subs[idx].content,
                    "context_before": " | ".join(ctx_before_parts) if ctx_before_parts else "(none)",
                    "context_after": " | ".join(ctx_after_parts) if ctx_after_parts else "(none)"
                })

            repair_chunk_size = 50
            total_recovered_micro = 0
            for r_start in range(0, len(global_repair_items), repair_chunk_size):
                r_chunk = global_repair_items[r_start:r_start + repair_chunk_size]
                try:
                    repair_results = await self.first_pass_micro_repair_batch(
                        r_chunk,
                        target_language=target_language,
                        source_language=source_language,
                        show_title=show_title or "",
                        job_id=job_id
                    )
                    if isinstance(repair_results, list):
                        for r in repair_results:
                            if isinstance(r, dict) and "id" in r and "text" in r:
                                rid = r["id"]
                                cand = r["text"]
                                if 0 <= rid < len(subs) and is_meaningful_translation(subs[rid].content, cand):
                                    translated_subs[rid].content = cand
                                    total_recovered_micro += 1
                except (ProviderUnavailableError, ProviderConfigurationError, DailyQuotaExhaustedError, RequestBudgetExhaustedError):
                    raise
                except Exception as e:
                    logger.warning(f"Global first-pass micro repair exception for chunk {r_start}: {e}")
                    if job_id:
                        append_job_log(job_id, f"Global First-Pass Micro Repair failed for chunk {r_start}: {e}")

            if job_id:
                append_job_log(job_id, f"Global First-Pass Micro Repair: evaluated {len(global_repair_items)} missing/identical cues -> recovered {total_recovered_micro}/{len(global_repair_items)}")

        # Post-process sanity clean: provenance-driven SDH and hallucination guard
        for idx, sub in enumerate(translated_subs):
            if sub.content and sub.content.strip() and sub.content.strip() != "<i></i>":
                prov = provenance_map.get(idx) if provenance_map else None
                source_bracketed_inners = getattr(prov, "bracketed_dialogue_inners", []) if prov else []
                source_paren_inners = getattr(prov, "parenthesized_dialogue_inners", []) if prov else []
                has_bracketed_dialogue = getattr(prov, "has_bracketed_dialogue", False) if prov else bool(source_bracketed_inners)
                has_parenthesized_dialogue = getattr(prov, "has_parenthesized_dialogue", False) if prov else bool(source_paren_inners)
                num_source_bracketed = max(len(source_bracketed_inners), 1 if has_bracketed_dialogue else 0)

                # Bracketed segments guard:
                bracket_matches = list(re.finditer(r'\[(.*?)\]|［(.*?)］', sub.content))
                if bracket_matches:
                    if num_source_bracketed == 0:
                        # Source had ZERO bracketed dialogue -> strip all hallucinated brackets
                        cleaned_bracket = re.sub(r'\[(.*?)\]|［(.*?)］', '', sub.content).strip()
                        sub.content = cleaned_bracket if cleaned_bracket else "<i></i>"
                    elif len(bracket_matches) > num_source_bracketed:
                        # Source had N bracketed segments, but target produced > N bracketed segments.
                        # Strip excess hallucinated bracket segments.
                        excess = len(bracket_matches) - num_source_bracketed
                        new_content = sub.content
                        for m in reversed(bracket_matches):
                            if excess <= 0:
                                break
                            start, end = m.span()
                            new_content = (new_content[:start] + new_content[end:]).strip()
                            excess -= 1
                        sub.content = new_content if new_content else "<i></i>"

                # Parenthesized segments guard:
                if not has_parenthesized_dialogue and len(source_paren_inners) == 0 and re.search(r'^\((.*?)\)$|^（(.*?)）$', sub.content.strip()):
                    sub.content = "<i></i>"

        return translated_subs

    @with_retry
    async def first_pass_micro_repair_batch(
        self,
        repair_items: List[dict],
        target_language: str,
        source_language: str = "English",
        show_title: str = "",
        job_id: Optional[int] = None,
        provider_ctx=None,
    ) -> List[dict]:
        if not repair_items:
            return []

        from app.core.ai_providers import resolve_job_provider_context, context_from_settings
        _ctx = provider_ctx or (resolve_job_provider_context(job_id) if job_id else context_from_settings())
        provider = _ctx.provider

        system_prompt = f"""You are a professional subtitle translator translating {source_language} dialogue to {target_language}.

The following dialogue TARGET lines failed the initial translation pass because they were copied unchanged or returned with incomplete translation.

Translate every TARGET line into natural, idiomatic {target_language} now.

STRICT RULES:
- Translate every TARGET into natural {target_language}.
- Do NOT copy the {source_language} TARGET unchanged.
- Do NOT classify.
- Do NOT return KEEP.
- Do NOT explain.
- Preserve character names/proper nouns where appropriate, but translate all surrounding dialogue.
- Keep subtitle wording concise and natural.
- Preserve meaning and tone.
- Return exactly one result for every requested id.
- Never invent ids.
- Output strict structured JSON only with a key "results" containing an array of objects with integer "id" and string "text"."""

        items_formatted = []
        for it in repair_items:
            item_str = (
                f"Item ID {it['id']}:\n"
                f"CONTEXT BEFORE: {it.get('context_before', '(none)')}\n"
                f"TARGET: {it['target']}\n"
                f"CONTEXT AFTER: {it.get('context_after', '(none)')}"
            )
            items_formatted.append(item_str)

        prompt = (
            f"Show Context: {show_title}\n\n"
            + "\n\n".join(items_formatted)
            + f"\n\nReturn a JSON object with key 'results' containing an array of objects with integer 'id' and string 'text' for all {len(repair_items)} items."
        )

        schema = {
            "type": "OBJECT",
            "properties": {
                "results": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "id": {"type": "INTEGER"},
                            "text": {"type": "STRING"}
                        },
                        "required": ["id", "text"]
                    }
                }
            },
            "required": ["results"]
        }

        if provider == "deepl":
            import httpx
            api_key = get_setting("deepl_api_key", "")
            if not api_key:
                raise ValueError("DeepL API Key is not configured.")
            from app.core.languages import get_deepl_source_code, get_deepl_target_code
            target_lang_code = get_deepl_target_code(target_language)
            source_lang_code = get_deepl_source_code(source_language)
            url = "https://api-free.deepl.com/v2/translate" if api_key.endswith(":fx") else "https://api.deepl.com/v2/translate"
            texts = [it["target"] for it in repair_items]
            _mark_sdk_started(_usage_token_ctx.get(None))
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    url,
                    headers={"Authorization": f"DeepL-Auth-Key {api_key}"},
                    json={"text": texts, "target_lang": target_lang_code, "source_lang": source_lang_code}
                )
                resp.raise_for_status()
                data = resp.json()
                translations = data.get("translations", [])
                return [{"id": repair_items[i]["id"], "text": translations[i]["text"]} for i in range(min(len(repair_items), len(translations)))]

        model_name = _ctx.model
        raw_resp = await self._dispatch_llm_completion(
            provider=provider,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=prompt,
            schema=schema,
            temperature=0.1,
            job_id=job_id,
        )
        return extract_json_safely(raw_resp or "{}")

    @with_retry
    async def bulk_contextual_recovery(
        self,
        recovery_items: List[dict],
        target_language: str,
        source_language: str = "English",
        show_title: str = "",
        job_id: Optional[int] = None,
        provider_ctx=None,
    ) -> List[dict]:
        if not recovery_items:
            return []

        from app.core.ai_providers import resolve_job_provider_context, context_from_settings
        _ctx = provider_ctx or (resolve_job_provider_context(job_id) if job_id else context_from_settings())
        provider = _ctx.provider
        show_context = f"\nShow / Context: \"{show_title}\"" if show_title else ""

        system_prompt = f"""You are a professional film/TV subtitle translator translating dialogue from {source_language} into natural, idiomatic {target_language}.{show_context}

The following dialogue TARGET lines need recovery translation. Use the surrounding dialogue context (before/after) to understand the scene, speaker tone, and conversational flow.

STRICT RULES:
1. Translate every TARGET line accurately and idiomatically into natural {target_language}.
2. Use the CONTEXT BEFORE and CONTEXT AFTER for tone, character gender, and scene coherence.
3. If a TARGET line contains character names, proper nouns, or titles, preserve them naturally while translating surrounding dialogue.
4. If a word or phrase has the exact same spelling and meaning in both {source_language} and {target_language} (for example identical interjections or invariant names), return the legitimate {target_language} text.
5. Keep subtitle wording concise and natural (max 2 lines per subtitle block).
6. Do NOT return empty text or placeholder '<i></i>' for real dialogue lines.
7. Do NOT classify or explain.
8. Return exactly one result for every requested integer id. Never invent ids.
9. Use only standard punctuation appropriate for the target language (do not mix unrelated foreign script punctuation such as CJK symbols into Latin text).
10. Output strict structured JSON with a key 'results' containing an array of objects with integer 'id' and string 'text'."""

        items_formatted = []
        for it in recovery_items:
            item_str = (
                f"Item ID {it['id']}:\n"
                f"CONTEXT BEFORE: {it.get('context_before', '(none)')}\n"
                f"TARGET: {it['target']}\n"
                f"CONTEXT AFTER: {it.get('context_after', '(none)')}"
            )
            items_formatted.append(item_str)

        prompt = (
            f"Context: {show_title}\n\n"
            + "\n\n".join(items_formatted)
            + f"\n\nReturn a JSON object with key 'results' containing an array of objects with integer 'id' and string 'text' for all {len(recovery_items)} items."
        )

        schema = {
            "type": "OBJECT",
            "properties": {
                "results": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "id": {"type": "INTEGER"},
                            "text": {"type": "STRING"}
                        },
                        "required": ["id", "text"]
                    }
                }
            },
            "required": ["results"]
        }

        if provider == "deepl":
            import httpx
            api_key = get_setting("deepl_api_key", "")
            if not api_key:
                raise ValueError("DeepL API Key is not configured.")
            from app.core.languages import get_deepl_source_code, get_deepl_target_code
            target_lang_code = get_deepl_target_code(target_language)
            source_lang_code = get_deepl_source_code(source_language)
            url = "https://api-free.deepl.com/v2/translate" if api_key.endswith(":fx") else "https://api.deepl.com/v2/translate"
            texts = [it["target"] for it in recovery_items]
            _mark_sdk_started(_usage_token_ctx.get(None))
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    url,
                    headers={"Authorization": f"DeepL-Auth-Key {api_key}"},
                    json={"text": texts, "target_lang": target_lang_code, "source_lang": source_lang_code}
                )
                resp.raise_for_status()
                data = resp.json()
                translations = data.get("translations", [])
                return [{"id": recovery_items[i]["id"], "text": translations[i]["text"]} for i in range(min(len(recovery_items), len(translations)))]

        model_name = _ctx.model
        raw_resp = await self._dispatch_llm_completion(
            provider=provider,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=prompt,
            schema=schema,
            temperature=0.1,
            job_id=job_id,
        )
        return extract_json_safely(raw_resp or "{}")

    @with_retry
    async def bulk_strict_recovery(
        self,
        recovery_items: List[dict],
        target_language: str,
        source_language: str = "English",
        show_title: str = "",
        job_id: Optional[int] = None,
        provider_ctx=None,
    ) -> List[dict]:
        if not recovery_items:
            return []

        from app.core.ai_providers import resolve_job_provider_context, context_from_settings
        _ctx = provider_ctx or (resolve_job_provider_context(job_id) if job_id else context_from_settings())
        provider = _ctx.provider
        show_context = f"\nShow / Context: \"{show_title}\"" if show_title else ""

        system_prompt = f"""You are a strict subtitle translation engine translating from {source_language} to {target_language}.{show_context}

The following dialogue lines must be translated into {target_language}.

STRICT RULES:
1. Provide a direct, idiomatic translation into {target_language} for every requested line.
2. Do not copy untranslated {source_language} text unchanged unless it is an invariant proper name or identical word in {target_language}.
3. Do not classify or explain.
4. Return exactly one result for every requested integer id. Never invent ids.
5. Use only standard punctuation appropriate for the target language (do not mix unrelated foreign script punctuation such as CJK symbols into Latin text).
6. Return a JSON object with key 'results' containing an array of objects with integer 'id' and string 'text'."""

        items_formatted = []
        for it in recovery_items:
            item_str = (
                f"Item ID {it['id']}:\n"
                f"CONTEXT BEFORE: {it.get('context_before', '(none)')}\n"
                f"TARGET: {it['target']}\n"
                f"CONTEXT AFTER: {it.get('context_after', '(none)')}"
            )
            items_formatted.append(item_str)

        prompt = (
            f"Context: {show_title}\n\n"
            + "\n\n".join(items_formatted)
            + f"\n\nReturn a JSON object with key 'results' containing an array of objects with integer 'id' and string 'text' for all {len(recovery_items)} items."
        )

        schema = {
            "type": "OBJECT",
            "properties": {
                "results": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "id": {"type": "INTEGER"},
                            "text": {"type": "STRING"}
                        },
                        "required": ["id", "text"]
                    }
                }
            },
            "required": ["results"]
        }

        if provider == "deepl":
            import httpx
            api_key = get_setting("deepl_api_key", "")
            if not api_key:
                raise ValueError("DeepL API Key is not configured.")
            from app.core.languages import get_deepl_source_code, get_deepl_target_code
            target_lang_code = get_deepl_target_code(target_language)
            source_lang_code = get_deepl_source_code(source_language)
            url = "https://api-free.deepl.com/v2/translate" if api_key.endswith(":fx") else "https://api.deepl.com/v2/translate"
            texts = [it["target"] for it in recovery_items]
            _mark_sdk_started(_usage_token_ctx.get(None))
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    url,
                    headers={"Authorization": f"DeepL-Auth-Key {api_key}"},
                    json={"text": texts, "target_lang": target_lang_code, "source_lang": source_lang_code}
                )
                resp.raise_for_status()
                data = resp.json()
                translations = data.get("translations", [])
                return [{"id": recovery_items[i]["id"], "text": translations[i]["text"]} for i in range(min(len(recovery_items), len(translations)))]

        model_name = _ctx.model
        raw_resp = await self._dispatch_llm_completion(
            provider=provider,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=prompt,
            schema=schema,
            temperature=0.1,
            job_id=job_id,
        )
        return extract_json_safely(raw_resp or "{}")

    async def fast_final_rescue_batch(
        self,
        rescue_items: List[dict],
        target_language: str,
        source_language: str = "English",
        show_title: str = "",
        attempt: int = 1,
        job_id: Optional[int] = None
    ) -> List[dict]:
        if attempt == 2:
            return await self.bulk_strict_recovery(
                rescue_items,
                target_language=target_language,
                source_language=source_language,
                show_title=show_title,
                job_id=job_id
            )
        return await self.bulk_contextual_recovery(
            rescue_items,
            target_language=target_language,
            source_language=source_language,
            show_title=show_title,
            job_id=job_id
        )

    @with_retry
    async def repair_alignment_region(
        self,
        repair_cue_ids: List[int],
        source_context_items: List[dict],
        target_context_items: List[dict],
        target_language: str = "Swedish",
        source_language: str = "English",
        show_title: str = "",
        verdict: str = "SHIFT",
        details: str = "",
        job_id: Optional[int] = None
    ) -> List[dict]:
        """
        Atomic alignment repair for a specific contiguous region of subtitle cues.
        Receives source & target context and strictly outputs translations ONLY for requested repair_cue_ids.
        """
        if not repair_cue_ids:
            return []

        from app.core.ai_providers import resolve_job_provider_context, context_from_settings
        _ctx = resolve_job_provider_context(job_id) if job_id else context_from_settings()
        provider = _ctx.provider
        show_context = f'\nShow / Context: "{show_title}"' if show_title else ""

        repair_set = set(repair_cue_ids)
        min_id = min(repair_cue_ids)
        max_id = max(repair_cue_ids)

        repair_source_items = [
            {"id": it["id"], "text": it["text"]}
            for it in source_context_items
            if it.get("id") in repair_set
        ]
        if not repair_source_items:
            repair_source_items = [it for it in source_context_items if it.get("id") in repair_set] or source_context_items

        left_guard = [
            {"id": it["id"], "text": it["text"]}
            for it in target_context_items
            if it.get("id") is not None and it["id"] < min_id
        ]
        right_guard = [
            {"id": it["id"], "text": it["text"]}
            for it in target_context_items
            if it.get("id") is not None and it["id"] > max_id
        ]

        system_prompt = f"""You are an expert subtitle translation and synchronization repair specialist translating from {source_language} to {target_language}.{show_context}

A semantic alignment anomaly was detected in the target subtitles.
Your task is to perform a HARD SOURCE REMAP: translate each canonical SOURCE cue directly and independently to its matching ID.

STRICT CONTRACT:
1. HARD SOURCE REMAP: Provide the direct, idiomatic 1-to-1 {target_language} translation for EVERY requested cue ID in {repair_cue_ids}.
2. Each translated cue MUST correspond strictly and exclusively to its matching single source cue ID (no delay, no advance, no omission, no merging across IDs).
3. DO NOT copy, shift, or infer from any old corrupted target sequence.
4. Output EXACTLY the requested cue IDs {repair_cue_ids}. Do NOT include IDs outside this range.
5. Return a JSON object with key 'translations' (or 'results') containing an array of objects with integer 'id' and string 'text'."""

        prompt_parts = [
            f"SOURCE OF TRUTH TO TRANSLATE (Direct 1-to-1 mapping into {target_language}):",
            json.dumps(repair_source_items, ensure_ascii=False)
        ]
        if left_guard:
            prompt_parts.extend([
                f"\nLEFT CONTEXT (REFERENCE ONLY / DO NOT COPY / DO NOT REMAP):",
                json.dumps(left_guard, ensure_ascii=False)
            ])
        if right_guard:
            prompt_parts.extend([
                f"\nRIGHT CONTEXT (REFERENCE ONLY / DO NOT COPY / DO NOT REMAP):",
                json.dumps(right_guard, ensure_ascii=False)
            ])
        prompt_parts.extend([
            f"\nREQUESTED REPAIR CUE IDS:\n{repair_cue_ids}",
            f"\nTranslate each listed source cue directly into {target_language} for its exact matching ID."
        ])
        prompt = "\n".join(prompt_parts)

        schema = {
            "type": "OBJECT",
            "properties": {
                "translations": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "id": {"type": "INTEGER"},
                            "text": {"type": "STRING"}
                        },
                        "required": ["id", "text"]
                    }
                }
            },
            "required": ["translations"]
        }

        if provider == "deepl":
            api_key = get_setting("deepl_api_key", "")
            if not api_key:
                raise ValueError("DeepL API Key is not configured.")
            from app.core.languages import get_deepl_source_code, get_deepl_target_code
            target_lang_code = get_deepl_target_code(target_language)
            source_lang_code = get_deepl_source_code(source_language)
            url = "https://api-free.deepl.com/v2/translate" if api_key.endswith(":fx") else "https://api.deepl.com/v2/translate"
            repair_set = set(repair_cue_ids)
            filtered_sources = [it for it in source_context_items if it["id"] in repair_set]
            texts = [it["text"] for it in filtered_sources]
            _mark_sdk_started(_usage_token_ctx.get(None))
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    url,
                    headers={"Authorization": f"DeepL-Auth-Key {api_key}"},
                    json={"text": texts, "target_lang": target_lang_code, "source_lang": source_lang_code}
                )
                resp.raise_for_status()
                data = resp.json()
                translations = data.get("translations", [])
                return [{"id": filtered_sources[i]["id"], "text": translations[i]["text"]} for i in range(min(len(filtered_sources), len(translations)))]

        model_name = _ctx.model
        raw_resp = await self._dispatch_llm_completion(
            provider=provider,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=prompt,
            schema=schema,
            temperature=0.1,
            job_id=job_id,
        )
        return extract_json_safely(raw_resp or "{}")
