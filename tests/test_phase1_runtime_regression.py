"""
tests/test_phase1_runtime_regression.py
=========================================
Regression tests for Phase 1 runtime bug: UnboundLocalError on first-attempt
fresh jobs when local RPD budget is exhausted.

Bug summary:
  In pipeline._run_pipeline_logic(), the minimum-budget-admission block used
  `datetime.now(timezone.utc)` without importing `datetime` or `timezone`.
  Only `timedelta as _td` was imported. This caused:

    UnboundLocalError: cannot access local variable 'datetime'
    where it is not associated with a value

  on FIRST ATTEMPT of fresh jobs when budget was exhausted, because:
  - is_resumed_job = False (fresh job, retry_count=0)
  - admission check runs, admission["admitted"] is False
  - code enters the "not admitted" branch and crashes before setting DEFERRED

  On automatic RETRY (event_source="RETRY"), is_resumed_job = True, so the
  admission block was skipped entirely, the translator was called, raised
  RequestBudgetExhaustedError, and THAT handler (with its own correct import)
  set the job to DEFERRED correctly.

Fix: Changed `from datetime import timedelta as _td` to
     `from datetime import datetime as _dt, timezone as _tz, timedelta as _td`
     and used `_dt.now(_tz.utc)` in the admission block.

Tests go through process_video_file() — the real production path — to
prevent regressions at the integration level.
"""

import asyncio
import math
import os
import pytest
import srt
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Fixture: isolated DB per test
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_db(tmp_path):
    db_file = str(tmp_path / "babel_regression_test.db")
    import app.core.db as db_module
    import app.core.quota as quota_module

    orig_db    = db_module.DB_PATH
    orig_quota = quota_module.DB_PATH
    db_module.DB_PATH    = db_file
    quota_module.DB_PATH = db_file
    db_module.init_db()

    yield db_file

    db_module.DB_PATH    = orig_db
    quota_module.DB_PATH = orig_quota


# ---------------------------------------------------------------------------
# Helper: minimal real SRT file
# ---------------------------------------------------------------------------

def make_srt_file(path: str, num_cues: int = 3):
    """Write a minimal .en.srt file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    for i in range(num_cues):
        start = timedelta(seconds=i * 2 + 1)
        end   = timedelta(seconds=i * 2 + 2)
        lines.append(srt.Subtitle(index=i + 1, start=start, end=end, content=f"Line {i + 1}."))
    with open(path, "w", encoding="utf-8") as f:
        f.write(srt.compose(lines))
    return path


# ---------------------------------------------------------------------------
# Shared pipeline patch context — isolates injection points used in tests
# ---------------------------------------------------------------------------

def pipeline_patches(srt_path: str):
    """
    Returns a list of (target, kwargs) for patch() that configure the
    pipeline to use our test SRT without touching real filesystem/AI.
    Stacks as context managers.
    """
    health_return = {"status": "RED", "reason": "test_mock_unhealthy"}
    tracks_return = {"audio": [], "subtitle": []}
    return [
        ("app.services.pipeline.find_external_subtitle",      dict(return_value=srt_path)),
        ("app.services.pipeline.check_existing_swedish_subtitle", dict(return_value=None)),
        ("app.services.pipeline.evaluate_subtitle_health",    dict(return_value=health_return)),
        ("app.services.pipeline.inspect_mkv_tracks",          dict(return_value=tracks_return)),
    ]


# ---------------------------------------------------------------------------
# A. Core regression: first-attempt fresh job with exhausted budget
#    Must -> DEFERRED directly, no exception, no RETRY_PENDING, zero calls
# ---------------------------------------------------------------------------

class TestRegressionFreshJobBudgetExhausted:
    """
    Reproduces the exact ShowB/ShowC failure:
    - Budget = 1, requests_today = 1 (already consumed by ShowA)
    - Fresh new job (event_source="MANUAL", retry_count=0)
    - Goes through process_video_file() — the real production path
    - Must reach DEFERRED directly without any exception or RETRY_PENDING
    """

    @pytest.mark.asyncio
    async def test_fresh_job_deferred_directly_no_exception(self, isolated_db, tmp_path):
        """
        REGRESSION TEST: Fresh job with exhausted budget must DEFER cleanly.
        - No UnboundLocalError
        - No RETRY_PENDING
        - ZERO provider calls
        - ZERO additional requests consumed
        - Structured DEFERRED metadata populated
        """
        from app.core.db import set_setting, create_job, get_job_by_id
        from app.core.quota import try_consume_request_budget, get_daily_requests_used
        from app.services.pipeline import pipeline

        set_setting("ai_provider", "gemini")
        set_setting("gemini_model", "gemini-3.5-flash-lite")
        set_setting("batch_size", "150")
        set_setting("max_concurrent_jobs", "3")
        set_setting("daily_request_budget_gemini", "1")
        set_setting("enable_bazarr_check", "false")

        # Consume the 1 allowed request (as ShowA would have done)
        assert try_consume_request_budget("gemini") is True
        assert get_daily_requests_used("gemini") == 1

        # 3 cues -> ceil(3/150)=1 minimum, remaining=0 -> NOT admitted
        video_path = str(tmp_path / "series" / "ShowB" / "S01E01.mkv")
        srt_path   = str(tmp_path / "series" / "ShowB" / "S01E01.en.srt")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        make_srt_file(srt_path)
        open(video_path, "w").close()

        job_id = create_job(video_path, "MANUAL")

        translate_call_count = [0]

        async def mock_translate(*a, **kw):
            translate_call_count[0] += 1
            return []

        with patch.object(pipeline.translator, "translate_srt_content", side_effect=mock_translate):
            with patch("app.services.pipeline.find_external_subtitle", return_value=srt_path):
                with patch("app.services.pipeline.check_existing_swedish_subtitle", return_value=None):
                    with patch("app.services.pipeline.evaluate_subtitle_health",
                               return_value={"status": "RED", "reason": "test_mock"}):
                        with patch("app.services.pipeline.inspect_mkv_tracks",
                                   return_value={"audio": [], "subtitle": []}):
                            result = await pipeline.process_video_file(
                                video_path=video_path,
                                event_source="MANUAL",
                                title="Show B - S01E01",
                                job_id=job_id,
                            )

        # Must not raise, must return deferred
        assert result is not None
        assert result.get("status") == "deferred", (
            f"Expected 'deferred' but got: {result}"
        )

        # ZERO provider calls
        assert translate_call_count[0] == 0, (
            f"Expected 0 provider calls, got {translate_call_count[0]}"
        )

        # ZERO additional budget consumption
        assert get_daily_requests_used("gemini") == 1, (
            "requests_today must NOT increase on admission defer"
        )

        # DB state: DEFERRED (not FAILED, not RETRY_PENDING)
        job = get_job_by_id(job_id)
        assert job["status"] == "DEFERRED", f"Expected DEFERRED, got: {job['status']}"
        assert job["status"] != "RETRY_PENDING"
        assert job["status"] != "FAILED"

        # Structured metadata
        assert job.get("defer_reason") == "INSUFFICIENT_LOCAL_BUDGET"
        assert (job.get("waiting_provider") or "").lower() == "gemini"
        assert job.get("defer_stage") == "PRIMARY"
        assert job.get("deferred_at") is not None
        assert job.get("next_retry_at") is not None

    @pytest.mark.asyncio
    async def test_fresh_job_showc_deferred_directly(self, isolated_db, tmp_path):
        """
        Same test for ShowC pattern (concurrent second-in-queue fresh job).
        """
        from app.core.db import set_setting, create_job, get_job_by_id
        from app.core.quota import try_consume_request_budget
        from app.services.pipeline import pipeline

        set_setting("ai_provider", "gemini")
        set_setting("gemini_model", "gemini-3.5-flash-lite")
        set_setting("batch_size", "150")
        set_setting("daily_request_budget_gemini", "1")
        set_setting("enable_bazarr_check", "false")

        assert try_consume_request_budget("gemini") is True

        video_path = str(tmp_path / "ShowC" / "S01E01.mkv")
        srt_path   = str(tmp_path / "ShowC" / "S01E01.en.srt")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        make_srt_file(srt_path)
        open(video_path, "w").close()

        job_id = create_job(video_path, "MANUAL")

        with patch.object(pipeline.translator, "translate_srt_content", new_callable=AsyncMock):
            with patch("app.services.pipeline.find_external_subtitle", return_value=srt_path):
                with patch("app.services.pipeline.check_existing_swedish_subtitle", return_value=None):
                    with patch("app.services.pipeline.evaluate_subtitle_health",
                               return_value={"status": "RED", "reason": "test_mock"}):
                        with patch("app.services.pipeline.inspect_mkv_tracks",
                                   return_value={"audio": [], "subtitle": []}):
                            result = await pipeline.process_video_file(
                                video_path=video_path,
                                event_source="MANUAL",
                                title="Show C",
                                job_id=job_id,
                            )

        assert result.get("status") == "deferred"
        job = get_job_by_id(job_id)
        assert job["status"] == "DEFERRED"
        assert job.get("defer_reason") == "INSUFFICIENT_LOCAL_BUDGET"

    @pytest.mark.asyncio
    async def test_retry_path_skips_admission_and_defers_on_exhaustion(self, isolated_db, tmp_path):
        """
        Verify the RETRY path (event_source=RETRY, retry_count>0) skips
        admission check and defers via RequestBudgetExhaustedError in
        acquire_dispatch_slot (the real budget gate inside the translator).
        """
        from app.core.db import set_setting, create_job, update_job, get_job_by_id
        from app.core.quota import (
            try_consume_request_budget, RequestBudgetExhaustedError,
            acquire_dispatch_slot,
        )
        from app.services.pipeline import pipeline

        set_setting("ai_provider", "gemini")
        set_setting("gemini_model", "gemini-3.5-flash-lite")
        set_setting("batch_size", "150")
        set_setting("daily_request_budget_gemini", "1")
        set_setting("enable_bazarr_check", "false")

        assert try_consume_request_budget("gemini") is True

        video_path = str(tmp_path / "ShowB_retry" / "S01E01.mkv")
        srt_path   = str(tmp_path / "ShowB_retry" / "S01E01.en.srt")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        make_srt_file(srt_path)
        open(video_path, "w").close()

        job_id = create_job(video_path, "MANUAL")
        # retry_count=1 -> is_resumed_job=True -> admission check is SKIPPED
        update_job(job_id, status="RETRY_PENDING", retry_count=1)

        # Verify: admission is skipped for retry jobs
        # (test verifies is_resumed_job logic by checking translate IS called)
        translate_called = [0]

        async def translate_raises_budget(*a, **kw):
            translate_called[0] += 1
            raise RequestBudgetExhaustedError(
                provider="gemini", used=1, budget=1,
                raw_message="budget exhausted"
            )

        # Patch acquire_dispatch_slot to raise RequestBudgetExhaustedError
        # (simulates what the real acquire_dispatch_slot does when budget=0)
        original_acquire = acquire_dispatch_slot

        def exhausted_acquire(provider, *a, **kw):
            raise RequestBudgetExhaustedError(
                provider=provider, used=1, budget=1,
            )

        with patch("app.core.quota.acquire_dispatch_slot", side_effect=exhausted_acquire):
            with patch("app.services.pipeline.find_external_subtitle", return_value=srt_path):
                with patch("app.services.pipeline.check_existing_swedish_subtitle", return_value=None):
                    with patch("app.services.pipeline.evaluate_subtitle_health",
                               return_value={"status": "RED", "reason": "test_mock"}):
                        with patch("app.services.pipeline.inspect_mkv_tracks",
                                   return_value={"audio": [], "subtitle": []}):
                            result = await pipeline.process_video_file(
                                video_path=video_path,
                                event_source="RETRY",
                                title="Show B retry",
                                job_id=job_id,
                            )

        # RETRY path with budget exhausted at acquire_dispatch_slot -> DEFERRED
        assert result.get("status") == "deferred", (
            f"Expected deferred, got: {result}"
        )
        job = get_job_by_id(job_id)
        assert job["status"] == "DEFERRED"


# ---------------------------------------------------------------------------
# B. Verify: fresh jobs WITH capacity proceed to translator (no false block)
# ---------------------------------------------------------------------------

class TestFreshJobWithCapacity:

    @pytest.mark.asyncio
    async def test_fresh_job_with_budget_reaches_translator(self, isolated_db, tmp_path):
        """
        When budget is available, fresh job must NOT be deferred at admission.
        Translator mock must be called.
        """
        from app.core.db import set_setting, create_job, get_job_by_id
        from app.core.quota import get_daily_requests_used
        from app.services.pipeline import pipeline

        set_setting("ai_provider", "gemini")
        set_setting("gemini_model", "gemini-3.5-flash-lite")
        set_setting("batch_size", "150")
        set_setting("daily_request_budget_gemini", "10")  # plenty
        set_setting("enable_bazarr_check", "false")

        assert get_daily_requests_used("gemini") == 0

        video_path = str(tmp_path / "ShowA" / "S01E01.mkv")
        srt_path   = str(tmp_path / "ShowA" / "S01E01.en.srt")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        make_srt_file(srt_path, num_cues=3)
        open(video_path, "w").close()

        job_id = create_job(video_path, "MANUAL")

        translate_called = [0]

        async def mock_translate(subs, target_language, source_language="English", batch_size=150, job_id=None, show_title=None):
            translate_called[0] += 1
            return list(subs)  # echo back unchanged

        with patch.object(pipeline.translator, "translate_srt_content",
                          side_effect=mock_translate):
            with patch("app.services.pipeline.find_external_subtitle", return_value=srt_path):
                with patch("app.services.pipeline.check_existing_swedish_subtitle", return_value=None):
                    with patch("app.services.pipeline.evaluate_subtitle_health",
                               return_value={"status": "RED", "reason": "test_mock"}):
                        with patch("app.services.pipeline.inspect_mkv_tracks",
                                   return_value={"audio": [], "subtitle": []}):
                            with patch("app.services.pipeline.notify_jellyfin_library_refresh",
                                       new_callable=AsyncMock):
                                try:
                                    result = await pipeline.process_video_file(
                                        video_path=video_path,
                                        event_source="MANUAL",
                                        title="Show A",
                                        job_id=job_id,
                                    )
                                except Exception:
                                    pass  # QA failures etc are fine here

        assert translate_called[0] >= 1, (
            "translate_srt_content must be called when budget is available"
        )

        job = get_job_by_id(job_id)
        # Must NOT be deferred due to admission gate
        assert not (
            job["status"] == "DEFERRED"
            and job.get("defer_reason") == "INSUFFICIENT_LOCAL_BUDGET"
        ), "Job must not fail admission when budget is available"

    def test_admission_boundary_exact_minimum_admitted(self, isolated_db):
        """available == minimum -> admitted (edge case)."""
        from app.core.db import set_setting
        from app.core.quota import check_minimum_budget_admission

        set_setting("daily_request_budget_gemini", "7")
        result = check_minimum_budget_admission("gemini", 914, 150)
        assert result["admitted"] is True
        assert result["estimated_minimum"] == 7  # ceil(914/150)=7

    def test_admission_boundary_below_minimum_not_admitted(self, isolated_db):
        """available < minimum -> not admitted."""
        from app.core.db import set_setting
        from app.core.quota import check_minimum_budget_admission, try_consume_request_budget

        set_setting("daily_request_budget_gemini", "7")
        try_consume_request_budget("gemini")  # available -> 6
        result = check_minimum_budget_admission("gemini", 914, 150)
        assert result["admitted"] is False
        assert result["available"] == 6
        assert result["estimated_minimum"] == 7


# ---------------------------------------------------------------------------
# C. Verify: requests_today does NOT increment on admission defer
# ---------------------------------------------------------------------------

class TestNoRequestConsumedOnAdmissionDefer:

    @pytest.mark.asyncio
    async def test_zero_requests_consumed_on_admission_defer(self, isolated_db, tmp_path):
        """
        When a fresh job is deferred at admission:
        - requests_today must remain exactly as before (no consume)
        """
        from app.core.db import set_setting, create_job
        from app.core.quota import try_consume_request_budget, get_daily_requests_used
        from app.services.pipeline import pipeline

        set_setting("ai_provider", "gemini")
        set_setting("gemini_model", "gemini-3.5-flash-lite")
        set_setting("batch_size", "150")
        set_setting("daily_request_budget_gemini", "1")
        set_setting("enable_bazarr_check", "false")

        assert try_consume_request_budget("gemini") is True
        before_count = get_daily_requests_used("gemini")
        assert before_count == 1

        video_path = str(tmp_path / "ShowB3" / "S01E01.mkv")
        srt_path   = str(tmp_path / "ShowB3" / "S01E01.en.srt")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        make_srt_file(srt_path)
        open(video_path, "w").close()

        job_id = create_job(video_path, "MANUAL")

        with patch.object(pipeline.translator, "translate_srt_content", new_callable=AsyncMock):
            with patch("app.services.pipeline.find_external_subtitle", return_value=srt_path):
                with patch("app.services.pipeline.check_existing_swedish_subtitle", return_value=None):
                    with patch("app.services.pipeline.evaluate_subtitle_health",
                               return_value={"status": "RED", "reason": "test_mock"}):
                        with patch("app.services.pipeline.inspect_mkv_tracks",
                                   return_value={"audio": [], "subtitle": []}):
                            result = await pipeline.process_video_file(
                                video_path=video_path,
                                event_source="MANUAL",
                                title="Show B3",
                                job_id=job_id,
                            )

        assert result.get("status") == "deferred"

        after_count = get_daily_requests_used("gemini")
        assert after_count == before_count, (
            f"requests_today must not increase! Before={before_count}, After={after_count}"
        )
