import asyncio
import json
import logging
import functools
import unicodedata
import re
import contextvars
from typing import List, Optional, Dict, Any

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
            elif func.__name__.endswith("_deepl"):
                provider = "deepl"
            elif func.__name__.endswith("_ollama"):
                provider = "ollama"
            else:
                try:
                    from app.core.db import get_setting as _gs
                    provider = _gs("ai_provider", "gemini").lower()
                except Exception:
                    provider = "gemini"

            model_name = kwargs.get("model_name") or kwargs.get("model") or decorator_kwargs.get("model")
            # Fallback: if model_name not in kwargs (e.g. passed as positional arg),
            # read from settings based on inferred provider for usage accounting purposes.
            if not model_name:
                try:
                    from app.core.db import get_setting as _gs
                    if provider == "gemini":
                        model_name = _gs("gemini_model", "gemini-3.5-flash-lite")
                    elif provider == "openai":
                        model_name = _gs("openai_model", "gpt-4o-mini")
                    elif provider == "ollama":
                        model_name = _gs("ollama_model", "llama3")
                    elif provider == "deepl":
                        model_name = "deepl"
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
                            }
                            est_cost = calculate_estimated_cost(
                                provider=provider,
                                model=_model_for_usage,
                                input_tokens=token_meta.get("input_tokens"),
                                cached_input_tokens=token_meta.get("cached_input_tokens"),
                                output_tokens=token_meta.get("output_tokens"),
                            )
                            complete_dispatch(
                                request_uid=_request_uid,
                                status=UsageStatus.SUCCESS,
                                input_tokens=token_meta.get("input_tokens"),
                                cached_input_tokens=token_meta.get("cached_input_tokens"),
                                output_tokens=token_meta.get("output_tokens"),
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


def _capture_gemini_tokens(response: Any) -> Any:
    """
    Extract token usage from a Gemini GenerateContentResponse and write to the
    module-level token store keyed by the current request_uid (from ContextVar).
    Called inside provider functions (which run in threadpool) before returning.
    Returns the response unchanged.
    """
    try:
        from app.core.usage import extract_gemini_usage
        request_uid = _usage_token_ctx.get(None)  # Thread sees asyncio-task snapshot
        if request_uid:
            token_meta = extract_gemini_usage(response)
            _store_token_meta(request_uid, token_meta)
    except Exception:
        pass
    return response


def _capture_openai_tokens(response: Any) -> Any:
    """
    Extract token usage from an OpenAI ChatCompletion response and write to the
    module-level token store keyed by the current request_uid (from ContextVar).
    Called inside provider functions (which run in threadpool) before returning.
    Returns the response unchanged.
    """
    try:
        from app.core.usage import extract_openai_usage
        request_uid = _usage_token_ctx.get(None)  # Thread sees asyncio-task snapshot
        if request_uid:
            token_meta = extract_openai_usage(response)
            _store_token_meta(request_uid, token_meta)
    except Exception:
        pass
    return response


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

def get_system_instruction(target_language: str, glossary: str = "", show_title: str = "") -> str:
    glossary_section = ""
    if glossary and glossary.strip():
        glossary_section = "\n\nGLOSSARY - Always use these exact translations:\n" + glossary.strip() + "\n"

    show_context = ""
    if show_title:
        show_context = f"\nYou are translating subtitles for: \"{show_title}\". Adapt tone and terminology accordingly.\n"

    return f"""You are a professional film/TV subtitle translator translating from English to {target_language}.{show_context}
Translate the numbered subtitle blocks to natural, idiomatic {target_language}.
{glossary_section}
STRICT RULES:
1. Translate accurately and idiomatically into natural {target_language}. Translate every real dialogue line into {target_language}; do not copy English dialogue unchanged.
2. Do not classify or explain. Return translations only.
3. Preserve character names and proper nouns where appropriate, but translate all surrounding dialogue naturally.
4. Preserve all subtitle formatting tags exactly as they appear (e.g. <i>, </i>, and ASS tags like {{\\an8}}). Do not encode them into HTML entities.
5. If a block is empty or contains only '<i></i>', keep it exactly as '<i></i>'.
6. If a line starts with a speaker name or label in capital letters (e.g. ALICE:, OFFICER:), keep character names intact and only translate descriptive titles if appropriate, maintaining the colon separator.
7. You MUST return a JSON object with a key "translations" containing the array of objects with integer "id" and string "text".
8. Keep translations concise and natural. Split lines naturally using "\\n" if a line exceeds 42 characters, but NEVER exceed 2 lines per subtitle block. Combine or condense text if necessary.
Example:
{{"translations": [{{"id": 1, "text": "Translated text"}}, {{"id": 2, "text": "Translated text"}}]}}
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
            is_onomatopoeia = bool(re.match(r'^(m+|h+a+|a+h+|o+h+|h+e+h+|h+m+|u+g+h+|o+o+h+|s+h+)$', w_clean))
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

def is_strictly_valid_entity_candidate(text: str) -> bool:
    """
    Fail-closed deterministic check to verify if candidate text has the lexical form
    of a valid named entity token before allowing contextual entity verification.

    Invariants:
    1. Must not be empty or placeholder (<i></i>).
    2. Must contain 1 to 3 alphabetic tokens.
    3. Every token must start with an uppercase letter.
    4. No token may match ANY word in ENGLISH_COMMON_WORDS.
    5. No token may match ANY word in KNOWN_NON_VERBAL_SOUNDS.
    6. No token may match ANY word in KNOWN_TECH_TERMS_AND_BRANDS (handled separately).
    7. Every token must be at least 2 characters long.
    """
    if not text:
        return False
    clean_text = text.strip()
    clean_text = re.sub(r'<[^>]+>', '', clean_text)
    clean_text = re.sub(r'\{[^}]+\}', '', clean_text).strip()
    if not clean_text or clean_text == "<i></i>":
        return False

    tokens = [t for t in re.split(r"[^\w\x27-]+", clean_text) if t and any(c.isalpha() for c in t)]
    if not tokens or len(tokens) > 3:
        return False

    for t in tokens:
        clean_t = re.sub(r"[^\w]", "", t).lower()
        if not clean_t or len(clean_t) < 2:
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

def validate_entity_verification_output(
    raw_text: str,
    candidates: list,
    show_title: str = ""
) -> set:
    """
    Validates the AI entity verification output fail-closed.
    Returns a set of verified cue IDs that are proven to be proper named entities in context.

    Invariants:
    1. Candidate must strictly pass is_strictly_valid_entity_candidate(target).
    2. Verdict must be NAMED_ENTITY.
    3. Entity type must be in {'PERSON_NAME', 'PLACE_NAME', 'ORGANIZATION'}.
    4. Confidence MUST be HIGH.
    5. Any ambiguity, medium/low confidence, or schema violation -> rejected (not in returned set).
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
            if rid not in candidate_map:
                continue
            target_text = candidate_map[rid]
            if not is_strictly_valid_entity_candidate(target_text):
                logger.warning(f"Entity Verification: ID {rid} ('{target_text}') rejected: failed deterministic candidate check")
                continue

            verdict = str(r.get("verdict", "")).upper()
            entity_type = str(r.get("entity_type", "")).upper()
            confidence = str(r.get("confidence", "")).upper()

            if (
                verdict == "NAMED_ENTITY"
                and entity_type in {"PERSON_NAME", "PLACE_NAME", "ORGANIZATION"}
                and confidence == "HIGH"
            ):
                logger.info(f"Entity Verification: ID {rid} ('{target_text}') verified as {entity_type} with HIGH confidence ({r.get('explanation', '')})")
                verified_ids.add(rid)
            else:
                logger.info(f"Entity Verification: ID {rid} ('{target_text}') rejected (verdict={verdict}, confidence={confidence})")
    except Exception as e:
        logger.error(f"Entity Verification: JSON parse failed: {e}")

    return verified_ids

def validate_classifier_output(
    raw_text: str,
    items: list,
    show_title: str = "",
    source_subs: Optional[list] = None,
    translated_subs: Optional[list] = None
) -> list:
    logger.info(f"Classifier validation: received {len(items)} candidates. Raw type: {type(raw_text)}")

    valid_results = []
    returned_ids = set()

    clean_text = raw_text.strip()
    if clean_text.startswith("```"):
        lines = clean_text.split('\n')
        if lines[0].startswith("```"): lines = lines[1:]
        if lines and lines[-1].startswith("```"): lines = lines[:-1]
        clean_text = "\n".join(lines).strip()

    try:
        data = json.loads(clean_text)

        # Handle dict or list root
        if isinstance(data, dict):
            results = data.get("results", [])
        elif isinstance(data, list):
            results = data
        else:
            results = []
            logger.warning(f"Classifier validation: Unexpected JSON root type {type(data)}")

        logger.info(f"Classifier validation: parsed {len(results)} results from JSON")

        expected_ids = {item["id"]: item["text"] for item in items}

        kept = 0
        translated = 0
        rejected = 0

        for r in results:
            if not isinstance(r, dict):
                logger.warning(f"Classifier validation: rejected result item of type {type(r)}")
                rejected += 1
                continue

            rid = r.get("id")
            act = str(r.get("action", "")).lower()
            if rid not in expected_ids:
                logger.warning(f"Classifier validation: rejected result for unknown ID {rid}")
                rejected += 1
                continue

            original_text = expected_ids[rid]
            reason = str(r.get("reason", "")).lower()

            if act == "keep":
                is_det_safe = is_deterministically_safe_keep(original_text, reason, show_title=show_title)
                is_ev_safe = False
                if not is_det_safe and source_subs is not None and translated_subs is not None:
                    is_ev_safe = has_entity_evidence(original_text, source_subs, translated_subs, target_idx=rid)

                if is_ev_safe:
                    logger.info(f"Classifier validation: ID {rid} allowed KEEP via same-run entity evidence ('{original_text}')")
                    reason = f"evidence_{reason}"
                elif not is_det_safe:
                    if is_strictly_valid_entity_candidate(original_text):
                        logger.info(f"Classifier validation: ID {rid} flagged for context entity verification ('{original_text}')")
                        act = "translate"
                        reason = "needs_context_verification"
                        r["text"] = ""
                    else:
                        logger.info(f"Classifier validation: ID {rid} downgraded KEEP->TRANSLATE (not deterministically safe: reason={reason}, text='{original_text}')")
                        act = "translate"
                        r["text"] = ""

            if act == "keep":
                kept += 1
                valid_results.append({"id": rid, "action": "keep", "reason": reason, "text": original_text})
                returned_ids.add(rid)
            elif act == "translate":
                translated += 1
                provided_text = str(r.get("text", "")).strip()
                # If provided translation is empty, unusable, or echoes source, clear text to force recovery
                if not is_meaningful_translation(original_text, provided_text):
                    logger.info(f"Classifier validation: ID {rid} TRANSLATE has empty/echo text, clearing to force recovery")
                    provided_text = ""
                valid_results.append({"id": rid, "action": "translate", "reason": reason or "translate", "text": provided_text})
                returned_ids.add(rid)
            else:
                logger.warning(f"Classifier validation: rejected result for ID {rid} due to invalid action {act}")
                rejected += 1
                continue

        logger.info(f"Classifier validation: Validated {kept} KEEP, {translated} TRANSLATE, {rejected} REJECTED")
    except Exception as e:
        logger.error(f"Classifier validation: JSON parse failed: {e}")

    # Failsafe: Any missing items must be translated with empty text to force recovery
    for item in items:
        if item["id"] not in returned_ids:
            logger.info(f"Classifier validation: Failsafe triggered for ID {item['id']}, forcing TRANSLATE")
            valid_results.append({"id": item["id"], "action": "translate", "reason": "malformed_fallback", "text": ""})

    return valid_results

class SubtitleTranslator:
    @with_retry
    async def classify_and_recover_identical(
        self,
        items: list,
        target_language: str,
        show_title: str,
        source_subs: Optional[list] = None,
        translated_subs: Optional[list] = None,
        job_id: Optional[int] = None
    ) -> list:
        provider = get_setting("ai_provider", "gemini").lower()
        if provider == "deepl":
            return [] # fallback to translation

        system_prompt = f"""You are a subtitle quality assurance AI for {target_language}.
The following lines were identical in English and {target_language}.
Decide for each line whether it should be KEPT identical (e.g. proper nouns, brands, numbers, untranslatable sounds) or TRANSLATED.
If a line is a song lyric, classify it as TRANSLATE. Do not keep song lyrics in English unless it is an untranslatable proper noun.

KEEP -> text may remain identical
TRANSLATE -> text MUST contain the actual translated target-language text, never merely the source.

Return ONLY a JSON object with a single key 'results' containing an array of objects.
Valid reasons for KEEP: proper_noun, brand, acronym, number, symbol, non_verbal.
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
                            "reason": {"type": "STRING", "enum": ["proper_noun", "brand", "acronym", "number", "symbol", "non_verbal", "none"]},
                            "text": {"type": "STRING"}
                        },
                        "required": ["id", "action", "text"]
                    }
                }
            },
            "required": ["results"]
        }

        raw_resp = ""
        # We will reuse the client calls directly here to keep it self-contained
        if provider == "gemini":
            from google import genai
            from google.genai import types
            api_key = get_setting("gemini_api_key", "")
            model_name = get_setting("gemini_model", "gemini-3.5-flash-lite")
            client = genai.Client(api_key=api_key)

            def do_gemini():
                _mark_sdk_started(_usage_token_ctx.get(None))
                return client.models.generate_content(
                    model=model_name,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.1
                    )
                )
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, contextvars.copy_context().run, do_gemini)
            _capture_gemini_tokens(resp)
            raw_resp = resp.text

        elif provider == "openai":
            import openai
            api_key = get_setting("openai_api_key", "")
            model_name = get_setting("openai_model", "gpt-4o-mini")
            client = openai.Client(api_key=api_key)

            def do_openai():
                _mark_sdk_started(_usage_token_ctx.get(None))
                return client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, contextvars.copy_context().run, do_openai)
            try:
                _capture_openai_tokens(resp)
                raw_resp = resp.choices[0].message.content
            except Exception:
                raw_resp = ""

        elif provider == "ollama":
            import httpx
            ollama_url = get_setting("ollama_url", "http://localhost:11434").rstrip("/")
            model_name = get_setting("ollama_model", "llama3")
            full_prompt = f"{system_prompt}\n\n{prompt}"
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{ollama_url}/api/generate",
                    json={"model": model_name, "prompt": full_prompt, "format": "json", "stream": False}
                )
                try:
                    raw_resp = resp.json()["response"]
                except Exception:
                    raw_resp = ""

        if not raw_resp:
            return []

        validated_results = validate_classifier_output(raw_resp, items, show_title=show_title, source_subs=source_subs, translated_subs=translated_subs)

        # Check if any single-occurrence entity candidate needs bounded contextual verification
        items_map = {item["id"]: item["text"] for item in items}
        context_candidates = []
        for r in validated_results:
            if r.get("reason") == "needs_context_verification" and source_subs is not None:
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

                context_candidates.append({
                    "id": rid,
                    "target": target_text,
                    "context_before": " | ".join(ctx_b_parts) if ctx_b_parts else "(none)",
                    "context_after": " | ".join(ctx_a_parts) if ctx_a_parts else "(none)"
                })

        if context_candidates:
            logger.info(f"Context Entity Verification: evaluating {len(context_candidates)} single-occurrence candidates")
            try:
                verified_ids = await self.verify_single_occurrence_entities(
                    context_candidates,
                    target_language,
                    show_title=show_title,
                    job_id=job_id
                )
                for r in validated_results:
                    rid = r["id"]
                    if r.get("reason") == "needs_context_verification":
                        if rid in verified_ids:
                            r["action"] = "keep"
                            r["reason"] = "context_verified_proper_noun"
                            r["text"] = items_map.get(rid, "")
                            logger.info(f"Context Entity Verification: ID {rid} ('{r['text']}') ACCEPTED as context-verified KEEP")
                        else:
                            r["reason"] = "unverified_entity"
                            logger.info(f"Context Entity Verification: ID {rid} ('{items_map.get(rid, '')}') REJECTED -> remains TRANSLATE")
            except (ProviderUnavailableError, ProviderConfigurationError, DailyQuotaExhaustedError, RequestBudgetExhaustedError):
                raise
            except Exception as e:
                err_str = str(e).lower()
                permanent = any(x in err_str for x in [
                    "401", "403", "unauthorized", "forbidden", "api key not valid",
                    "invalid api key", "not configured", "model_not_found", "permission_denied"
                ])
                if permanent:
                    raise ProviderConfigurationError(f"Permanent provider configuration error in entity verification: {str(e)}")
                logger.error(f"Context Entity Verification call failed: {e}")

        return validated_results

    @with_retry
    async def verify_single_occurrence_entities(
        self,
        candidates: list,
        target_language: str,
        show_title: str = "",
        job_id: Optional[int] = None
    ) -> set:
        """
        Bounded, structured single-call AI entity classifier with local cue context.
        Only classifies entity vs translatable text. Never writes translations.
        Only HIGH confidence named entities may be accepted.
        """
        if not candidates:
            return set()

        provider = get_setting("ai_provider", "gemini").lower()
        if provider == "deepl":
            return set()

        system_prompt = f"""You are a linguistic named-entity classifier for subtitle localization ({target_language}).
Your sole task is to analyze whether the target subtitle text in each dialogue excerpt is strictly an invariant proper named entity (such as a person's name or place name that should remain unchanged in {target_language} subtitles) or translatable conversational English.

CRITICAL INSTRUCTIONS:
1. NEVER provide translations. Only classify the target text into the specified schema.
2. If the target text is a common English word, phrase, greeting, exclamation, action, verb, or translatable expression, classify as TRANSLATABLE_TEXT.
3. If the context is unclear, ambiguous, or the target text could be a normal word/phrase rather than a proper noun, classify as AMBIGUOUS with LOW or MEDIUM confidence.
4. Only classify as NAMED_ENTITY with HIGH confidence if the local dialogue context clearly and unambiguously demonstrates that the target is a proper name (e.g., a person being directly addressed, called upon, or referred to as a person).
5. Output strict structured JSON with key 'results' containing an array of classification objects."""

        items_formatted = []
        for it in candidates:
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
            + f"\n\nClassify each of the {len(candidates)} items strictly as JSON."
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
                            "verdict": {
                                "type": "STRING",
                                "enum": ["NAMED_ENTITY", "TRANSLATABLE_TEXT", "AMBIGUOUS"]
                            },
                            "entity_type": {
                                "type": "STRING",
                                "enum": ["PERSON_NAME", "PLACE_NAME", "ORGANIZATION", "NOT_AN_ENTITY"]
                            },
                            "confidence": {
                                "type": "STRING",
                                "enum": ["HIGH", "MEDIUM", "LOW"]
                            },
                            "explanation": {"type": "STRING"}
                        },
                        "required": ["id", "verdict", "entity_type", "confidence"]
                    }
                }
            },
            "required": ["results"]
        }

        if provider == "gemini":
            from google import genai
            from google.genai import types
            api_key = get_setting("gemini_api_key", "")
            model_name = get_setting("gemini_model", "gemini-3.5-flash-lite")
            client = genai.Client(api_key=api_key)

            def do_gemini():
                _mark_sdk_started(_usage_token_ctx.get(None))
                return client.models.generate_content(
                    model=model_name,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.0
                    )
                )
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, contextvars.copy_context().run, do_gemini)
            _capture_gemini_tokens(resp)
            return validate_entity_verification_output(resp.text, candidates, show_title=show_title)

        elif provider == "openai":
            import openai
            api_key = get_setting("openai_api_key", "")
            model_name = get_setting("openai_model", "gpt-4o-mini")
            client = openai.Client(api_key=api_key)

            def do_openai():
                _mark_sdk_started(_usage_token_ctx.get(None))
                return client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, contextvars.copy_context().run, do_openai)
            try:
                _capture_openai_tokens(resp)
                content = resp.choices[0].message.content
                return validate_entity_verification_output(content, candidates, show_title=show_title)
            except Exception:
                return set()

        elif provider == "ollama":
            import httpx
            ollama_url = get_setting("ollama_url", "http://localhost:11434").rstrip("/")
            model_name = get_setting("ollama_model", "llama3")
            full_prompt = f"{system_prompt}\n\n{prompt}"
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{ollama_url}/api/generate",
                    json={"model": model_name, "prompt": full_prompt, "format": "json", "stream": False}
                )
                try:
                    return validate_entity_verification_output(resp.json()["response"], candidates, show_title=show_title)
                except Exception:
                    return set()

        return set()

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
        self._cached_gemini_client = genai.Client(api_key=api_key)
        return self._cached_gemini_client

    def get_openai_client(self):
        api_key = get_setting("openai_api_key", "")
        if not api_key:
            raise ValueError("OpenAI API Key is not configured in settings.")
        if self._cached_openai_client and self._cached_openai_key == api_key:
            return self._cached_openai_client
        self._cached_openai_key = api_key
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
        job_id: Optional[int] = None,
    ) -> Optional[str]:
        if provider == "gemini":
            from google import genai
            from google.genai import types
            api_key = get_setting("gemini_api_key", "")
            client = genai.Client(api_key=api_key)
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=schema,
            )
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                contextvars.copy_context().run,
                lambda: (_mark_sdk_started(_usage_token_ctx.get(None)), client.models.generate_content(
                    model=model_name, contents=prompt, config=config
                ))[-1],
            )
            _capture_gemini_tokens(resp)
            return resp.text

        elif provider == "openai":
            import openai
            api_key = get_setting("openai_api_key", "")
            client = openai.OpenAI(api_key=api_key)
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                contextvars.copy_context().run,
                lambda: (_mark_sdk_started(_usage_token_ctx.get(None)), client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "esc", "schema": schema, "strict": True},
                    },
                ))[-1],
            )
            _capture_openai_tokens(resp)
            return resp.choices[0].message.content

        elif provider == "deepl":
            import httpx
            api_key = get_setting("deepl_api_key", "")
            from app.core.languages import get_language
            lang_obj = get_language(target_language)
            target_lang_code = lang_obj.deepl_code if lang_obj else target_language.upper()[:2]
            url = "https://api-free.deepl.com/v2/translate" if api_key.endswith(":fx") else "https://api.deepl.com/v2/translate"
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                resp = await http_client.post(
                    url,
                    headers={"Authorization": f"DeepL-Auth-Key {api_key}"},
                    json={"text": [target_text], "target_lang": target_lang_code, "source_lang": "EN"},
                )
                resp.raise_for_status()
                data = resp.json()
                raw_res = data["translations"][0]["text"]
                return json.dumps({"translation": raw_res})

        return None

    async def escalate_single_line(self, target_idx: int, target_text: str, prev_text: str, next_text: str, target_language: str, show_title: str, is_real_untranslated: bool = False, job_id: Optional[int] = None, exhausted_strategies: set = None) -> Optional[str]:
        import logging
        import unicodedata
        import re
        logger = logging.getLogger(__name__)

        primary_provider = get_setting("ai_provider", "gemini").lower()
        escalate_enabled = get_setting("escalate_to_pro", "false").lower() == "true"
        esc_provider = get_setting("escalation_provider", "none").lower()

        configured_esc = esc_provider if escalate_enabled and esc_provider != "none" else primary_provider

        attempts = [
            {"provider": configured_esc, "type": "contextual"},
            {"provider": configured_esc, "type": "strict"},
            {"provider": configured_esc, "type": "isolated"},
        ]

        for i, attempt in enumerate(attempts):
            provider = attempt["provider"]
            attempt_type = attempt["type"]

            model_name = ""
            if provider == "gemini":
                model_name = get_setting("gemini_model", "gemini-3.5-flash-lite")
                if escalate_enabled and provider == esc_provider:
                    esc_model = get_setting("escalation_model", "")
                    if esc_model: model_name = esc_model
            elif provider == "openai":
                model_name = get_setting("openai_model", "gpt-4o-mini")
                if escalate_enabled and provider == esc_provider:
                    esc_model = get_setting("escalation_model", "")
                    if esc_model: model_name = esc_model
            elif provider == "deepl":
                model_name = "deepl"

            context_fingerprint = hash((prev_text, target_text, next_text, is_real_untranslated))
            strategy_key = f"{target_idx}:{provider}:{model_name}:{attempt_type}:{context_fingerprint}"
            if exhausted_strategies is not None and strategy_key in exhausted_strategies:
                continue

            if attempt_type == "contextual":
                if is_real_untranslated:
                    system_prompt = f"You MUST translate TARGET into {target_language}.\nTARGET is known to still be untranslated English dialogue.\nDo NOT return the English source.\nDo NOT return an empty string.\nPrevious/Next are context only.\nReturn a JSON object with a single key 'translation' containing only the translated TARGET."
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
                    if not is_meaningful_translation(target_text, res):
                        logger.info(f"Escalation line {target_idx} attempt {i+1}/3: rejected identical source")
                        if job_id:
                            from app.core.db import append_job_log
                            append_job_log(job_id, f"Escalation cue {target_idx + 1} attempt {i+1}/3: rejected identical source")
                        return None

                    logger.info(f"Escalation line {target_idx} attempt {i+1}/3: translated successfully")
                    if job_id:
                        from app.core.db import append_job_log
                        append_job_log(job_id, f"Escalation cue {target_idx + 1} attempt {i+1}/3: translated successfully")
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
                    job_id=job_id,
                )
                res = _safe_parse(raw_out)
                if res:
                    return res

                if exhausted_strategies is not None:
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
    async def translate_batch_gemini(self, items: List[dict], target_language: str, model_name: str, context_lines: List[dict] = None, show_title: str = "", job_id: Optional[int] = None) -> List[dict]:
        client = self.get_gemini_client()

        context_section = ""
        if context_lines:
            context_section = "\n\nCONTEXT from previous batch (DO NOT translate these, use them only for tone and consistency):\n"
            for ctx in context_lines:
                if ctx.get("translated"):
                    context_section += f'  Original: "{ctx["original"]}" → Translated: "{ctx["translated"]}"\n'
                else:
                    context_section += f'  Original: "{ctx["original"]}"\n'

        prompt = f"Translate the following {len(items)} subtitle lines into {target_language}:{context_section}\n\n" + json.dumps(items, ensure_ascii=False)

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

        glossary = get_setting("glossary", "")
        config = types.GenerateContentConfig(
            system_instruction=get_system_instruction(target_language, glossary=glossary, show_title=show_title),
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.1,
        )

        def call_gemini(model_to_use):
            _mark_sdk_started(_usage_token_ctx.get(None))
            return client.models.generate_content(
                model=model_to_use,
                contents=prompt,
                config=config
            )

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(None, contextvars.copy_context().run, lambda: call_gemini(model_name))
        except Exception as e:
            # Automatic Fallback if Google deprecated or changed model name
            if "404" in str(e) or "not found" in str(e).lower():
                fallback_models = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-flash-latest"]
                for fb in fallback_models:
                    if fb != model_name:
                        try:
                            logger.warning(f"Model {model_name} failed with 404, falling back to {fb}")
                            response = await loop.run_in_executor(None, contextvars.copy_context().run, lambda: call_gemini(fb))
                            break
                        except Exception:
                            continue
                else:
                    raise e
            else:
                raise e

        return extract_json_safely(_capture_gemini_tokens(response).text)

    @with_retry
    async def translate_batch_openai(self, items: List[dict], target_language: str, model_name: str, context_lines: List[dict] = None, show_title: str = "", job_id: Optional[int] = None) -> List[dict]:
        client = self.get_openai_client()

        context_section = ""
        if context_lines:
            context_section = "\n\nCONTEXT from previous batch (DO NOT translate these, use them only for tone and consistency):\n"
            for ctx in context_lines:
                if ctx.get("translated"):
                    context_section += f'  Original: "{ctx["original"]}" → Translated: "{ctx["translated"]}"\n'
                else:
                    context_section += f'  Original: "{ctx["original"]}"\n'

        prompt = f"Translate the following {len(items)} subtitle lines into {target_language}:{context_section}\n\n" + json.dumps(items, ensure_ascii=False)

        glossary = get_setting("glossary", "")

        def call_openai(model_to_use):
            _mark_sdk_started(_usage_token_ctx.get(None))
            return client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": get_system_instruction(target_language, glossary=glossary, show_title=show_title)},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(None, contextvars.copy_context().run, lambda: call_openai(model_name))
        except Exception as e:
            if "404" in str(e) or "model_not_found" in str(e).lower():
                logger.warning(f"Model {model_name} failed with 404, falling back to gpt-4o-mini")
                response = await loop.run_in_executor(None, contextvars.copy_context().run, lambda: call_openai("gpt-4o-mini"))
            else:
                raise e

        return extract_json_safely(_capture_openai_tokens(response).choices[0].message.content)

    @with_retry
    async def translate_batch_deepl(self, items: List[dict], target_language: str, context_lines: List[dict] = None, job_id: Optional[int] = None) -> List[dict]:
        api_key = get_setting("deepl_api_key", "")
        if not api_key:
            raise ValueError("DeepL API Key is not configured.")

        from app.core.languages import get_language
        lang_obj = get_language(target_language)
        target_lang_code = lang_obj.deepl_code if lang_obj else target_language.upper()[:2]
        url = "https://api-free.deepl.com/v2/translate" if api_key.endswith(":fx") else "https://api.deepl.com/v2/translate"

        texts = [it["text"] for it in items]
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url,
                headers={"Authorization": f"DeepL-Auth-Key {api_key}"},
                json={"text": texts, "target_lang": target_lang_code, "source_lang": "EN"}
            )
            resp.raise_for_status()
            data = resp.json()
            translations = data.get("translations", [])
            if len(translations) != len(items):
                raise ValueError(f"DeepL returned {len(translations)} translations, but expected {len(items)}.")
            return [{"id": items[i]["id"], "text": translations[i]["text"]} for i in range(len(items))]

    @with_retry
    async def translate_batch_ollama(self, items: List[dict], target_language: str, model_name: str, context_lines: List[dict] = None, show_title: str = "", job_id: Optional[int] = None) -> List[dict]:
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

        prompt = f"{get_system_instruction(target_language, glossary=glossary, show_title=show_title)}\n\nTranslate the following JSON list into {target_language}:{context_section}\n{json.dumps(items, ensure_ascii=False)}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{ollama_url}/api/generate",
                json={"model": model_name or "llama3", "prompt": prompt, "format": "json", "stream": False}
            )
            resp.raise_for_status()
            data = resp.json()
            return extract_json_safely(data.get("response", "{}"))

    async def translate_batch(self, items: List[dict], target_language: str = "English", context_lines: List[dict] = None, show_title: str = "", job_id: Optional[int] = None) -> List[dict]:
        provider = get_setting("ai_provider", "gemini").lower()

        if provider == "openai":
            model = get_setting("openai_model", "gpt-4o-mini")
            return await self.translate_batch_openai(items, target_language, model, context_lines=context_lines, show_title=show_title, job_id=job_id)
        elif provider == "deepl":
            return await self.translate_batch_deepl(items, target_language, context_lines=context_lines, job_id=job_id)
        elif provider in ["ollama", "localai"]:
            model = get_setting("ollama_model", "llama3")
            return await self.translate_batch_ollama(items, target_language, model, context_lines=context_lines, show_title=show_title, job_id=job_id)
        else:
            model = get_setting("gemini_model", "gemini-3.5-flash-lite")
            return await self.translate_batch_gemini(items, target_language, model, context_lines=context_lines, show_title=show_title, job_id=job_id)


    async def translate_srt_content(
        self,
        subs: List[srt.Subtitle],
        target_language: str = "English",
        batch_size: int = 50,
        job_id: Optional[int] = None,
        show_title: Optional[str] = None
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
                tm_context = get_translation_memory(show_title, limit=10)
                if tm_context:
                    global_tm_context.extend(tm_context)
            except Exception:
                pass

        state_lock = asyncio.Lock()
        concurrency = get_positive_int_setting("batch_concurrency", 3)
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

                # Safe keep pre-filter: exclude safe KEEP items from missing_payload
                missing_payload = []
                orig_map = {}
                for p in payload:
                    idx = p["id"]
                    if idx not in partial_dict:
                        if is_safe_keep_prefilter(p["text"]):
                            # Deterministically safe to keep: assign immediately
                            translated_subs[idx].content = p["text"]
                            partial_dict[idx] = p["text"]
                        else:
                            missing_payload.append(p)
                            orig_map[idx] = p["text"]

                res_dict = {}
                if missing_payload:
                    # 1. Main translation pass
                    results = await self.translate_batch(
                        missing_payload,
                        target_language=target_language,
                        context_lines=batch_context if batch_context else None,
                        show_title=show_title or "",
                        job_id=job_id
                    )

                    # 2. Immediate deterministic validation (Fail-closed against echoes, empty, unmeaningful)
                    if isinstance(results, list):
                        for r in results:
                            if isinstance(r, dict) and "id" in r and "text" in r:
                                rid = r["id"]
                                if rid in orig_map:
                                    cand = r["text"]
                                    if is_meaningful_translation(orig_map[rid], cand):
                                        res_dict[rid] = cand

                    # 3. First-Pass Micro Repair (Max 1 batch call for failed real dialogue cues)
                    failed_cues = [p for p in missing_payload if p["id"] not in res_dict]
                    if failed_cues:
                        repair_items = []
                        for p in failed_cues:
                            idx = p["id"]
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

                            repair_items.append({
                                "id": idx,
                                "target": p["text"],
                                "context_before": " | ".join(ctx_before_parts) if ctx_before_parts else "(none)",
                                "context_after": " | ".join(ctx_after_parts) if ctx_after_parts else "(none)"
                            })

                        try:
                            repair_results = await self.first_pass_micro_repair_batch(
                                repair_items,
                                target_language=target_language,
                                show_title=show_title or "",
                                job_id=job_id
                            )
                            recovered_micro_count = 0
                            if isinstance(repair_results, list):
                                for r in repair_results:
                                    if isinstance(r, dict) and "id" in r and "text" in r:
                                        rid = r["id"]
                                        if rid in orig_map:
                                            cand = r["text"]
                                            if is_meaningful_translation(orig_map[rid], cand):
                                                res_dict[rid] = cand
                                                recovered_micro_count += 1
                            if job_id:
                                append_job_log(job_id, f"First-Pass Micro Repair: evaluated {len(failed_cues)} missing/identical cues (batch {start_idx + 1}-{end_idx}) -> recovered {recovered_micro_count}/{len(failed_cues)}")
                        except (ProviderUnavailableError, ProviderConfigurationError, DailyQuotaExhaustedError, RequestBudgetExhaustedError):
                            raise
                        except Exception as e:
                            err_str = str(e).lower()
                            permanent = any(x in err_str for x in [
                                "401", "403", "unauthorized", "forbidden", "api key not valid",
                                "invalid api key", "not configured", "model_not_found", "permission_denied"
                            ])
                            if permanent:
                                raise ProviderConfigurationError(f"Permanent provider configuration error in micro repair: {str(e)}")
                            logger.warning(f"First-pass micro repair exception for batch {start_idx + 1}-{end_idx}: {e}")
                            if job_id:
                                append_job_log(job_id, f"First-Pass Micro Repair failed for batch {start_idx + 1}-{end_idx}: {e}")

                for p in payload:
                    if p["id"] in partial_dict:
                        res_dict[p["id"]] = partial_dict[p["id"]]

                async with state_lock:
                    for p in payload:
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

        return translated_subs

    @with_retry
    async def first_pass_micro_repair_batch(
        self,
        repair_items: List[dict],
        target_language: str,
        show_title: str = "",
        job_id: Optional[int] = None
    ) -> List[dict]:
        if not repair_items:
            return []

        provider = get_setting("ai_provider", "gemini").lower()

        system_prompt = f"""You are a professional subtitle translator translating English dialogue to {target_language}.

The following dialogue TARGET lines failed the initial translation pass because they were copied unchanged or returned with incomplete translation.

Translate every TARGET line into natural, idiomatic {target_language} now.

STRICT RULES:
- Translate every TARGET into natural {target_language}.
- Do NOT copy the English TARGET unchanged.
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

        if provider == "gemini":
            from google import genai
            from google.genai import types
            api_key = get_setting("gemini_api_key", "")
            model_name = get_setting("gemini_model", "gemini-3.5-flash-lite")
            client = genai.Client(api_key=api_key)

            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.1
            )

            def call_gemini(model_to_use):
                _mark_sdk_started(_usage_token_ctx.get(None))
                return client.models.generate_content(
                    model=model_to_use,
                    contents=prompt,
                    config=config
                )

            loop = asyncio.get_event_loop()
            try:
                response = await loop.run_in_executor(None, contextvars.copy_context().run, lambda: call_gemini(model_name))
            except Exception as e:
                if "404" in str(e) or "not found" in str(e).lower():
                    fallback_models = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-flash-latest"]
                    for fb in fallback_models:
                        if fb != model_name:
                            try:
                                logger.warning(f"Model {model_name} failed with 404 in First-Pass Micro Repair, falling back to {fb}")
                                response = await loop.run_in_executor(None, contextvars.copy_context().run, lambda: call_gemini(fb))
                                break
                            except Exception:
                                continue
                    else:
                        raise e
                else:
                    raise e

            return extract_json_safely(_capture_gemini_tokens(response).text)

        elif provider == "openai":
            import openai
            api_key = get_setting("openai_api_key", "")
            model_name = get_setting("openai_model", "gpt-4o-mini")
            client = openai.OpenAI(api_key=api_key)

            def call_openai(model_to_use):
                _mark_sdk_started(_usage_token_ctx.get(None))
                return client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )

            loop = asyncio.get_event_loop()
            try:
                response = await loop.run_in_executor(None, contextvars.copy_context().run, lambda: call_openai(model_name))
            except Exception as e:
                if "404" in str(e) or "model_not_found" in str(e).lower():
                    response = await loop.run_in_executor(None, contextvars.copy_context().run, lambda: call_openai("gpt-4o-mini"))
                else:
                    raise e

            return extract_json_safely(_capture_openai_tokens(response).choices[0].message.content)

        elif provider in ["ollama", "localai"]:
            import httpx
            ollama_url = get_setting("ollama_url", "http://localhost:11434").rstrip("/")
            model_name = get_setting("ollama_model", "llama3")
            full_prompt = f"{system_prompt}\n\n{prompt}"
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{ollama_url}/api/generate",
                    json={"model": model_name or "llama3", "prompt": full_prompt, "format": "json", "stream": False}
                )
                resp.raise_for_status()
                data = resp.json()
                return extract_json_safely(data.get("response", "{}"))

        elif provider == "deepl":
            import httpx
            api_key = get_setting("deepl_api_key", "")
            if not api_key:
                raise ValueError("DeepL API Key is not configured.")
            from app.core.languages import get_language
            lang_obj = get_language(target_language)
            target_lang_code = lang_obj.deepl_code if lang_obj else target_language.upper()[:2]
            url = "https://api-free.deepl.com/v2/translate" if api_key.endswith(":fx") else "https://api.deepl.com/v2/translate"
            texts = [it["target"] for it in repair_items]
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    url,
                    headers={"Authorization": f"DeepL-Auth-Key {api_key}"},
                    json={"text": texts, "target_lang": target_lang_code, "source_lang": "EN"}
                )
                resp.raise_for_status()
                data = resp.json()
                translations = data.get("translations", [])
                return [{"id": repair_items[i]["id"], "text": translations[i]["text"]} for i in range(min(len(repair_items), len(translations)))]

        return []

    @with_retry
    async def fast_final_rescue_batch(
        self,
        rescue_items: List[dict],
        target_language: str,
        show_title: str = "",
        attempt: int = 1,
        job_id: Optional[int] = None
    ) -> List[dict]:
        if not rescue_items:
            return []

        provider = get_setting("ai_provider", "gemini").lower()

        if attempt == 2:
            system_prompt = f"""You are repairing failed subtitle translations.

Your previous response failed validation because some TARGET lines were copied from the English source.

Translate the remaining TARGET lines into natural {target_language} now.

IMPORTANT:
- Do not copy the English TARGET.
- Do not classify.
- Do not return KEEP.
- Do not explain.
- Preserve names/proper nouns when appropriate, but translate all surrounding dialogue.
- Keep subtitle wording concise and natural.
- Preserve meaning and tone.
- Return exactly one result for every supplied id.
- Never invent ids.
- Output strict structured JSON only."""
        else:
            system_prompt = f"""You are repairing failed subtitle translations.

Translate every TARGET from English into natural {target_language}.

IMPORTANT:
- Every TARGET in this request has already failed QA because it was copied from the English source.
- Do NOT return the English source unchanged.
- Do NOT classify.
- Do NOT return KEEP.
- Do NOT explain.
- Preserve names/proper nouns when appropriate, but translate all surrounding dialogue.
- Keep subtitle wording concise and natural.
- Preserve meaning and tone.
- Return exactly one result for every supplied id.
- Never invent ids.
- Output strict structured JSON only."""

        items_formatted = []
        for it in rescue_items:
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
            + f"\n\nReturn a JSON object with key 'results' containing an array of objects with integer 'id' and string 'text' for all {len(rescue_items)} items."
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

        if provider == "gemini":
            from google import genai
            from google.genai import types
            api_key = get_setting("gemini_api_key", "")
            model_name = get_setting("gemini_model", "gemini-3.5-flash-lite")
            client = genai.Client(api_key=api_key)

            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.1
            )

            def call_gemini(model_to_use):
                _mark_sdk_started(_usage_token_ctx.get(None))
                return client.models.generate_content(
                    model=model_to_use,
                    contents=prompt,
                    config=config
                )

            loop = asyncio.get_event_loop()
            try:
                response = await loop.run_in_executor(None, contextvars.copy_context().run, lambda: call_gemini(model_name))
            except Exception as e:
                if "404" in str(e) or "not found" in str(e).lower():
                    fallback_models = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-flash-latest"]
                    for fb in fallback_models:
                        if fb != model_name:
                            try:
                                logger.warning(f"Model {model_name} failed with 404 in Fast Final Rescue, falling back to {fb}")
                                response = await loop.run_in_executor(None, contextvars.copy_context().run, lambda: call_gemini(fb))
                                break
                            except Exception:
                                continue
                    else:
                        raise e
                else:
                    raise e

            return extract_json_safely(_capture_gemini_tokens(response).text)

        elif provider == "openai":
            import openai
            api_key = get_setting("openai_api_key", "")
            model_name = get_setting("openai_model", "gpt-4o-mini")
            client = openai.OpenAI(api_key=api_key)

            def call_openai(model_to_use):
                _mark_sdk_started(_usage_token_ctx.get(None))
                return client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )

            loop = asyncio.get_event_loop()
            try:
                response = await loop.run_in_executor(None, contextvars.copy_context().run, lambda: call_openai(model_name))
            except Exception as e:
                if "404" in str(e) or "model_not_found" in str(e).lower():
                    response = await loop.run_in_executor(None, contextvars.copy_context().run, lambda: call_openai("gpt-4o-mini"))
                else:
                    raise e

            return extract_json_safely(_capture_openai_tokens(response).choices[0].message.content)

        elif provider in ["ollama", "localai"]:
            import httpx
            ollama_url = get_setting("ollama_url", "http://localhost:11434").rstrip("/")
            model_name = get_setting("ollama_model", "llama3")
            full_prompt = f"{system_prompt}\n\n{prompt}"
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{ollama_url}/api/generate",
                    json={"model": model_name or "llama3", "prompt": full_prompt, "format": "json", "stream": False}
                )
                resp.raise_for_status()
                data = resp.json()
                return extract_json_safely(data.get("response", "{}"))

        elif provider == "deepl":
            import httpx
            api_key = get_setting("deepl_api_key", "")
            if not api_key:
                raise ValueError("DeepL API Key is not configured.")
            from app.core.languages import get_language
            lang_obj = get_language(target_language)
            target_lang_code = lang_obj.deepl_code if lang_obj else target_language.upper()[:2]
            url = "https://api-free.deepl.com/v2/translate" if api_key.endswith(":fx") else "https://api.deepl.com/v2/translate"
            texts = [it["target"] for it in rescue_items]
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    url,
                    headers={"Authorization": f"DeepL-Auth-Key {api_key}"},
                    json={"text": texts, "target_lang": target_lang_code, "source_lang": "EN"}
                )
                resp.raise_for_status()
                data = resp.json()
                translations = data.get("translations", [])
                return [{"id": rescue_items[i]["id"], "text": translations[i]["text"]} for i in range(min(len(rescue_items), len(translations)))]

        return []
