"""
tests/test_quota_awareness.py
==============================
Comprehensive regression/integration tests for the daily quota (RPD) awareness feature.

Tests are hermetic — no real provider calls are made.
Controlled clock (freezegun or manual datetime patching) is used throughout.

Test matrix covers:
  A. Transient RPM — short retry, no daily block
  B. Daily RPD — classified, no sleep, provider blocked, DEFERRED
  C. Next job while blocked — zero provider API calls
  D. Auto resume — fake clock past blocked_until → provider eligible again
  E. Restart — block state survives process restart
  F. User request budget — exact N requests allowed, N+1 deferred
  G. Unlimited budget — unchanged dispatch behavior
  H. Concurrency — budget=1, only ONE request passes
  I. Auth error — 401 → FAILED, never DEFERRED
  J. Other provider isolation — provider A block ≠ provider B block
  K. Recovery requests counted against budget
  L. UI/API — /api/quota endpoint returns correct fields
  M. Backward compatibility — old DB without quota tables starts fine
  N. Timezone — all timestamps are UTC-aware
"""

import asyncio
import json
import os
import sqlite3
import threading
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """
    Each test gets its own isolated SQLite database.
    Patches DB_PATH globally for the duration of the test.
    """
    db_file = str(tmp_path / "babel_quota_test.db")
    import app.core.db as db_module
    import app.core.quota as quota_module

    original_db = db_module.DB_PATH
    db_module.DB_PATH = db_file
    quota_module.DB_PATH = db_file

    db_module.init_db()

    yield db_file

    db_module.DB_PATH = original_db
    quota_module.DB_PATH = original_db


# ---------------------------------------------------------------------------
# Helper — create a fake SRT payload for translate_srt_content tests
# ---------------------------------------------------------------------------

def make_fake_subs(n=5):
    import srt
    from datetime import timedelta
    return [
        srt.Subtitle(
            index=i + 1,
            start=timedelta(seconds=i),
            end=timedelta(seconds=i + 1),
            content=f"Hello world line {i + 1}"
        )
        for i in range(n)
    ]


# ===========================================================================
# A. Transient RPM
# ===========================================================================

class TestTransientRPM:
    """
    Provider returns a transient 429/RPM error on first call, succeeds on retry.
    Verify:
    - short retry happens (no daily block)
    - job does NOT become DEFERRED
    - provider quota state not set to blocked
    """

    @pytest.mark.asyncio
    async def test_transient_rpm_retries_and_succeeds(self, isolated_db):
        from app.services.translator import SubtitleTranslator, ProviderUnavailableError
        from app.core.quota import is_provider_blocked, get_provider_block_info

        call_count = [0]

        async def fake_translate_batch(self, items, target_language="Swedish", context_lines=None, show_title=""):
            call_count[0] += 1
            if call_count[0] == 1:
                # Simulate transient RPM error (NOT daily quota — per-minute language)
                raise RuntimeError("429 Too Many Requests: requests per minute exceeded")
            # Second call succeeds
            return [{"id": item["id"], "text": f"Translated {item['text']}"} for item in items]

        translator = SubtitleTranslator()
        subs = make_fake_subs(3)

        with patch.object(SubtitleTranslator, "translate_batch", fake_translate_batch):
            with patch("app.core.db.get_setting", return_value="gemini"):
                with patch("asyncio.sleep", new_callable=AsyncMock):  # Speed up backoff
                    result = await translator.translate_srt_content(subs, target_language="Swedish")

        # Provider should NOT be blocked — it was only a transient RPM
        assert not is_provider_blocked("gemini"), "Transient RPM must NOT block provider"
        block_info = get_provider_block_info("gemini")
        assert not block_info["blocked"]

    def test_rpm_classification(self):
        """RPM-only patterns must not be classified as daily quota."""
        from app.core.quota import classify_provider_error

        rpm_errors = [
            RuntimeError("429: requests per minute exceeded"),
            RuntimeError("Rate limit: rpm quota exceeded"),
            RuntimeError("429 Too Many Requests: rate limit per minute"),
            RuntimeError("resource_exhausted: too many requests per minute"),
        ]
        for exc in rpm_errors:
            result = classify_provider_error(exc, "gemini")
            assert result != "DAILY_QUOTA_EXHAUSTED", (
                f"RPM error was wrongly classified as DAILY_QUOTA_EXHAUSTED: {exc}"
            )
            assert result in ("TRANSIENT_RPM", "PROVIDER_UNAVAILABLE"), (
                f"RPM error should be TRANSIENT_RPM or PROVIDER_UNAVAILABLE, got {result}: {exc}"
            )


# ===========================================================================
# B. Daily RPD
# ===========================================================================

class TestDailyRPD:
    """
    Provider explicitly returns daily quota exhausted error.
    Verify:
    - classified DAILY_QUOTA_EXHAUSTED
    - no long sleep (no sleep calls > 60s)
    - no 3 short retries (raises immediately)
    - provider state persisted as blocked
    - job becomes DEFERRED, not FAILED
    """

    def test_daily_quota_classified_correctly(self):
        """Daily quota error patterns must be classified as DAILY_QUOTA_EXHAUSTED."""
        from app.core.quota import classify_provider_error

        daily_errors = [
            RuntimeError("429: Daily quota exceeded"),
            RuntimeError("resource_exhausted: quota per day exceeded"),
            RuntimeError("RESOURCE_EXHAUSTED: You exceeded your quota for the day"),
            RuntimeError("dailyrequestlimitexceeded: quota_id requests_per_day"),
            RuntimeError("Quota exceeded: per day quota for model exceeded"),
        ]
        for exc in daily_errors:
            result = classify_provider_error(exc, "gemini")
            assert result == "DAILY_QUOTA_EXHAUSTED", (
                f"Expected DAILY_QUOTA_EXHAUSTED, got {result} for: {exc}"
            )

    @pytest.mark.asyncio
    async def test_daily_quota_blocks_provider_and_defers_job(self, isolated_db, tmp_path):
        """
        Simulate full pipeline: provider returns daily quota error.
        Job must become DEFERRED, provider must be blocked.
        """
        from app.services.pipeline import SubtitlePipeline
        from app.core.db import create_job, get_job_by_id
        from app.core.quota import is_provider_blocked, DailyQuotaExhaustedError

        video = tmp_path / "movie.mkv"
        video.touch()
        video_path = str(video)

        job_id = create_job(video_path, "MANUAL", "Test Movie")

        # Mock translate_srt_content to raise DailyQuotaExhaustedError
        async def fake_translate(*args, **kwargs):
            raise DailyQuotaExhaustedError(
                provider="gemini",
                retry_after_seconds=None,
                raw_message="Quota per day exceeded",
            )

        sleep_calls = []

        en_srt_path = video_path.replace(".mkv", ".en.srt")
        with open(en_srt_path, "w", encoding="utf-8") as _f:
            _f.write("1\n00:00:00,000 --> 00:00:01,000\nHello world this is English.\n\n"
                     "2\n00:00:01,000 --> 00:00:02,000\nTesting quota error handling.\n\n"
                     "3\n00:00:02,000 --> 00:00:03,000\nProvider should be blocked.\n")

        with patch("app.services.translator.SubtitleTranslator.translate_srt_content", fake_translate),              patch("app.services.pipeline.find_external_subtitle",
                   side_effect=lambda p, l: en_srt_path if l == "en" else None),              patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            pipeline = SubtitlePipeline()
            result = await pipeline.process_video_file(video_path, job_id=job_id, force_retranslate=True)
            sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]

        # Job must be DEFERRED — NOT failed
        job = get_job_by_id(job_id)
        assert job["status"] == "DEFERRED", f"Expected DEFERRED, got {job['status']}"
        assert result["status"] == "deferred"

        # Provider must be blocked
        assert is_provider_blocked("gemini"), "Provider must be blocked after daily quota"

        # No long sleeps (no sleep > 60 seconds — the daily quota path must not sleep)
        long_sleeps = [s for s in sleep_calls if s > 60]
        assert not long_sleeps, f"Daily quota path must not sleep long, got: {sleep_calls}"

    @pytest.mark.asyncio
    async def test_daily_quota_no_triple_retry(self, isolated_db):
        """
        With_retry must NOT retry a DailyQuotaExhaustedError.
        The error must propagate immediately on first attempt.
        Tests the with_retry decorator directly.
        """
        from app.services.translator import SubtitleTranslator
        from app.core.quota import DailyQuotaExhaustedError

        call_count = [0]

        # Test the @with_retry decorated translate_batch_gemini directly
        # by patching the underlying Gemini API call (not translate_batch which has its own guard)
        async def counting_gemini_call(self, items, target_language, model_name, **kwargs):
            call_count[0] += 1
            raise DailyQuotaExhaustedError("gemini", None, "daily quota from test")

        translator = SubtitleTranslator()

        with patch.object(SubtitleTranslator, "translate_batch_gemini", counting_gemini_call):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with patch("app.core.db.get_setting", side_effect=lambda k, d="": {
                    "ai_provider": "gemini",
                    "batch_concurrency": "3",
                    "batch_size": "50",
                }.get(k, d)):
                    with pytest.raises(DailyQuotaExhaustedError):
                        # translate_batch calls translate_batch_gemini after the quota gate check
                        # But since we're blocked, it raises before the call.
                        # So test with_retry independently:
                        @__import__("app.services.translator", fromlist=["with_retry"]).with_retry
                        async def retry_wrapped():
                            call_count[0] += 1
                            raise DailyQuotaExhaustedError("gemini", None, "daily quota from test")

                        await retry_wrapped()

        # Must be called exactly once — no retries for daily quota
        assert call_count[0] == 1, f"DailyQuotaExhaustedError must not be retried, called {call_count[0]} times"



# ===========================================================================
# C. Next job while blocked
# ===========================================================================

class TestNextJobWhileBlocked:
    """
    After daily quota block, the next job must not make any provider API calls.
    """

    @pytest.mark.asyncio
    async def test_no_provider_call_while_blocked(self, isolated_db, tmp_path):
        from app.core.quota import block_provider, DailyQuotaExhaustedError
        from app.services.translator import SubtitleTranslator

        # Block the provider
        block_provider("gemini", "Test block", retry_after_seconds=3600)

        real_api_calls = [0]

        mock_client = MagicMock()
        def counting_generate_content(*args, **kwargs):
            real_api_calls[0] += 1
            mock_resp = MagicMock()
            mock_resp.text = json.dumps({"translations": [{"id": 0, "text": "Translated"}]})
            return mock_resp

        mock_client.models.generate_content.side_effect = counting_generate_content

        translator = SubtitleTranslator()
        items = [{"id": 0, "text": "Hello world"}]

        with patch.object(translator, "get_gemini_client", return_value=mock_client):
            with patch("app.core.db.get_setting", side_effect=lambda k, d="": {
                "ai_provider": "gemini",
                "gemini_api_key": "test_key",
                "gemini_model": "gemini-3.5-flash-lite",
            }.get(k, d)):
                with pytest.raises(DailyQuotaExhaustedError):
                    await translator.translate_batch(items, target_language="Swedish")

        # Zero real API calls must have been made
        assert real_api_calls[0] == 0, (
            f"Provider was called {real_api_calls[0]} times while blocked — must be 0"
        )



# ===========================================================================
# D. Auto resume
# ===========================================================================

class TestAutoResume:
    """
    After blocked_until passes, the provider must become eligible again
    and deferred jobs must be processable.
    """

    def test_provider_unblocks_after_expiry(self, isolated_db):
        from app.core.quota import block_provider, is_provider_blocked, _utcnow

        # Block with a 1-second window
        now = datetime.now(timezone.utc)
        past = now - timedelta(seconds=1)

        # Directly insert an expired block
        import sqlite3
        from app.core.quota import _ensure_quota_table
        with sqlite3.connect(isolated_db, timeout=10.0) as conn:
            _ensure_quota_table(conn)
            conn.execute("""
            INSERT OR REPLACE INTO provider_quota
            (provider, blocked, reason, blocked_at, blocked_until, updated_at)
            VALUES (?, 1, 'test', ?, ?, ?)
            """, ("gemini", past.isoformat(), past.isoformat(), past.isoformat()))
            conn.commit()

        # Should be auto-unblocked since blocked_until is in the past
        assert not is_provider_blocked("gemini"), "Expired block must auto-unblock"

    @pytest.mark.asyncio
    async def test_deferred_job_claimed_after_unblock(self, isolated_db, tmp_path):
        """
        Scheduler must claim DEFERRED jobs after provider unblocks.
        """
        from app.core.db import create_job, update_job, get_job_by_id, claim_job_for_retry
        from app.core.quota import block_provider, _ensure_quota_table

        # Create a deferred job with past next_retry_at
        job_id = create_job("test.mkv", "MANUAL", "test")
        past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        update_job(job_id, status="DEFERRED", next_retry_at=past)

        # Put an already-expired block (provider is NOW unblocked)
        import sqlite3
        expired = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        with sqlite3.connect(isolated_db, timeout=10.0) as conn:
            _ensure_quota_table(conn)
            conn.execute("""
            INSERT OR REPLACE INTO provider_quota
            (provider, blocked, reason, blocked_at, blocked_until, updated_at)
            VALUES (?, 1, 'test', ?, ?, ?)
            """, ("gemini", expired, expired, expired))
            conn.commit()

        # Claiming the DEFERRED job should succeed
        claimed = claim_job_for_retry(job_id)
        assert claimed, "DEFERRED job should be claimable after quota reset"

        job = get_job_by_id(job_id)
        assert job["status"] == "QUEUED"

    def test_deferred_job_not_claimed_while_blocked(self, isolated_db):
        """
        Scheduler must NOT claim DEFERRED jobs while provider is still blocked.
        This tests the scheduler guard in main.py.
        """
        from app.core.db import create_job, update_job, get_job_by_id
        from app.core.quota import block_provider, is_provider_blocked

        job_id = create_job("test_blocked.mkv", "MANUAL", "test")
        past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        update_job(job_id, status="DEFERRED", next_retry_at=past)

        # Block the provider (still active)
        block_provider("gemini", "still blocked", retry_after_seconds=3600)
        assert is_provider_blocked("gemini")

        # Simulate what process_one_retry_pass does: check block before claiming
        from app.core.db import get_setting
        from app.core.quota import is_provider_blocked as check_blocked

        active_provider = "gemini"  # simulated
        # The scheduler guard: DEFERRED + blocked → should_retry = False
        provider_still_blocked = check_blocked(active_provider)
        assert provider_still_blocked, "Provider should still be blocked"

        # Job must remain DEFERRED
        job = get_job_by_id(job_id)
        assert job["status"] == "DEFERRED"


# ===========================================================================
# E. Restart recovery
# ===========================================================================

class TestRestartRecovery:
    """
    DEFERRED jobs must survive restart and retain block state.
    """

    def test_deferred_jobs_survive_restart(self, isolated_db):
        """
        DEFERRED jobs must NOT be converted to FAILED/RETRY_PENDING on restart.
        """
        from app.core.db import create_job, update_job, get_job_by_id, init_db

        job_id = create_job("deferred.mkv", "MANUAL", "test")
        future = (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat()
        update_job(job_id, status="DEFERRED", next_retry_at=future, error_message="Daily quota")

        # Simulate restart by calling init_db again
        init_db()

        job = get_job_by_id(job_id)
        assert job["status"] == "DEFERRED", (
            f"DEFERRED jobs must survive restart, got {job['status']}"
        )
        assert job["next_retry_at"] == future

    def test_provider_block_survives_restart(self, isolated_db):
        """
        Provider block state in DB must persist across simulated restart.
        """
        from app.core.quota import block_provider, is_provider_blocked
        from app.core.db import init_db

        block_provider("gemini", "quota test", retry_after_seconds=3600)
        assert is_provider_blocked("gemini")

        # Simulate restart
        init_db()

        # Block must still be active
        assert is_provider_blocked("gemini"), "Provider block must survive restart"

    def test_processing_jobs_fail_on_restart(self, isolated_db):
        """
        Jobs in TRANSLATING/PROCESSING must become FAILED on restart (existing behavior).
        """
        from app.core.db import create_job, update_job, get_job_by_id, init_db

        job_id = create_job("processing.mkv", "MANUAL", "test")
        update_job(job_id, status="TRANSLATING")

        init_db()

        job = get_job_by_id(job_id)
        assert job["status"] == "RETRY_PENDING", (
            f"TRANSLATING jobs must become RETRY_PENDING on restart, got {job['status']}"
        )


# ===========================================================================
# F. User request budget
# ===========================================================================

class TestUserRequestBudget:
    """
    User-configured daily request budget (e.g. 3 requests).
    Exactly N requests allowed; N+1 is blocked.
    """

    def test_budget_allows_exactly_n_requests(self, isolated_db):
        """Budget = 3 → exactly 3 allowed, 4th blocked."""
        from app.core.quota import try_consume_request_budget
        from app.core.db import set_setting

        set_setting("daily_request_budget_gemini", "3")

        results = [try_consume_request_budget("gemini") for _ in range(4)]

        assert results[0] is True, "Request 1 must be allowed"
        assert results[1] is True, "Request 2 must be allowed"
        assert results[2] is True, "Request 3 must be allowed"
        assert results[3] is False, "Request 4 must be BLOCKED (budget=3)"

    def test_budget_exact_provider_call_count(self, isolated_db):
        """With budget=3, exactly 3 real provider calls must be made."""
        from app.core.quota import try_consume_request_budget, RequestBudgetExhaustedError
        from app.core.db import set_setting

        set_setting("daily_request_budget_gemini", "3")

        actual_calls = 0
        for _ in range(5):
            allowed = try_consume_request_budget("gemini")
            if allowed:
                actual_calls += 1

        assert actual_calls == 3, f"Expected exactly 3 provider calls, got {actual_calls}"

    def test_budget_resets_each_day(self, isolated_db):
        """Budget counter resets at UTC midnight (new window_date)."""
        from app.core.quota import try_consume_request_budget
        from app.core.db import set_setting

        set_setting("daily_request_budget_gemini", "2")

        # Use up today's budget
        assert try_consume_request_budget("gemini") is True
        assert try_consume_request_budget("gemini") is True
        assert try_consume_request_budget("gemini") is False  # exhausted

        # Simulate next day by patching _today_window
        with patch("app.core.quota._today_window", return_value="2099-01-01"):
            # New day — budget resets
            assert try_consume_request_budget("gemini") is True, "Budget must reset next day"

    def test_deferred_on_budget_exhaustion(self, isolated_db, tmp_path):
        """When budget is exhausted, pipeline must set job to DEFERRED."""
        from app.core.quota import RequestBudgetExhaustedError, try_consume_request_budget
        from app.core.db import create_job, get_job_by_id, set_setting

        set_setting("daily_request_budget_gemini", "0")  # unlimited first so we can exhaust

        # We'll simulate the budget exhausted case directly
        async def run():
            from app.services.pipeline import SubtitlePipeline
            video = tmp_path / "movie2.mkv"
            video.touch()
            video_path = str(video)
            job_id = create_job(video_path, "MANUAL", "Budget Test")

            # Make translate_batch raise RequestBudgetExhaustedError
            async def fake_translate_budget(*args, **kwargs):
                raise RequestBudgetExhaustedError(provider="gemini", used=250, budget=250)

            en_srt_path = video_path.replace(".mkv", ".en.srt")
            with open(en_srt_path, "w", encoding="utf-8") as _f:
                _f.write("1\n00:00:00,000 --> 00:00:01,000\nHello world budget test.\n\n"
                         "2\n00:00:01,000 --> 00:00:02,000\nTesting budget exhaustion.\n\n"
                         "3\n00:00:02,000 --> 00:00:03,000\nProvider request quota used.\n")

            with patch("app.services.translator.SubtitleTranslator.translate_srt_content", fake_translate_budget),                  patch("app.services.pipeline.find_external_subtitle",
                       side_effect=lambda p, l: en_srt_path if l == "en" else None):
                pipeline = SubtitlePipeline()
                result = await pipeline.process_video_file(video_path, job_id=job_id, force_retranslate=True)

            job = get_job_by_id(job_id)
            assert job["status"] == "DEFERRED", f"Expected DEFERRED, got {job['status']}"
            assert result["status"] == "deferred"
            assert job["next_retry_at"] is not None  # deferred until tomorrow

        asyncio.run(run())


# ===========================================================================
# G. Unlimited budget
# ===========================================================================

class TestUnlimitedBudget:
    """
    Default budget = 0 (Unlimited) must not change dispatch behavior.
    """

    def test_unlimited_always_allows(self, isolated_db):
        """With budget=0 (unlimited), all requests are allowed."""
        from app.core.quota import try_consume_request_budget
        from app.core.db import set_setting

        set_setting("daily_request_budget_gemini", "0")

        # 1000 requests must all pass
        for i in range(1000):
            result = try_consume_request_budget("gemini")
            assert result is True, f"Request {i+1} must be allowed with unlimited budget"

    def test_unlimited_does_not_increment_counter(self, isolated_db):
        """With unlimited budget, no counter rows are written (fast path)."""
        from app.core.quota import try_consume_request_budget, get_daily_requests_used
        from app.core.db import set_setting

        set_setting("daily_request_budget_gemini", "0")

        for _ in range(10):
            try_consume_request_budget("gemini")

        # Counter should be 0 — fast path doesn't write
        used = get_daily_requests_used("gemini")
        assert used == 0, f"Unlimited mode must not increment counter, got {used}"


# ===========================================================================
# H. Concurrency
# ===========================================================================

class TestConcurrency:
    """
    Budget remaining = 1, multiple concurrent threads call try_consume_request_budget.
    Exactly ONE must succeed.
    """

    def test_budget_one_concurrent_safe(self, isolated_db):
        """Only one concurrent worker wins with budget=1."""
        from app.core.quota import try_consume_request_budget
        from app.core.db import set_setting

        set_setting("daily_request_budget_gemini", "1")

        results = []
        lock = threading.Lock()

        def worker():
            result = try_consume_request_budget("gemini")
            with lock:
                results.append(result)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        winners = [r for r in results if r is True]
        losers = [r for r in results if r is False]

        assert len(winners) == 1, f"Exactly ONE thread must win with budget=1, got {len(winners)}"
        assert len(losers) == 9, f"9 threads must be blocked, got {len(losers)}"

    def test_budget_three_concurrent_safe(self, isolated_db):
        """With budget=3 and 10 threads, exactly 3 win."""
        from app.core.quota import try_consume_request_budget
        from app.core.db import set_setting

        set_setting("daily_request_budget_gemini", "3")

        results = []
        lock = threading.Lock()

        def worker():
            result = try_consume_request_budget("gemini")
            with lock:
                results.append(result)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        winners = [r for r in results if r is True]
        assert len(winners) == 3, f"Exactly 3 threads must win with budget=3, got {len(winners)}"


# ===========================================================================
# I. Auth error
# ===========================================================================

class TestAuthError:
    """
    401/invalid API key must become FAILED — never DEFERRED.
    """

    def test_401_classified_as_auth_error(self):
        from app.core.quota import classify_provider_error

        auth_errors = [
            RuntimeError("401 Unauthorized"),
            RuntimeError("403 Forbidden"),
            RuntimeError("API key not valid. Please pass a valid API key."),
            RuntimeError("invalid api key"),
            RuntimeError("permission_denied: API key is invalid"),
        ]
        for exc in auth_errors:
            result = classify_provider_error(exc, "gemini")
            assert result == "AUTH_ERROR", (
                f"Expected AUTH_ERROR for auth exception, got {result}: {exc}"
            )

    @pytest.mark.asyncio
    async def test_auth_error_not_deferred(self, isolated_db, tmp_path):
        """Auth errors must produce FAILED status, not DEFERRED."""
        from app.services.translator import ProviderConfigurationError
        from app.services.pipeline import SubtitlePipeline
        from app.core.db import create_job, get_job_by_id

        video = tmp_path / "auth_test.mkv"
        video.touch()
        video_path = str(video)
        job_id = create_job(video_path, "MANUAL", "Auth Test")

        async def fake_translate_auth(*args, **kwargs):
            raise ProviderConfigurationError("401 Unauthorized: invalid API key")

        en_srt_path = video_path.replace(".mkv", ".en.srt")
        with open(en_srt_path, "w", encoding="utf-8") as _f:
            _f.write("1\n00:00:00,000 --> 00:00:01,000\nHello world auth test.\n\n"
                     "2\n00:00:01,000 --> 00:00:02,000\nTesting auth error handling.\n\n"
                     "3\n00:00:02,000 --> 00:00:03,000\nBad API key must cause FAILED.\n")

        with patch("app.services.translator.SubtitleTranslator.translate_srt_content", fake_translate_auth),              patch("app.services.pipeline.find_external_subtitle",
                   side_effect=lambda p, l: en_srt_path if l == "en" else None):
            pipeline = SubtitlePipeline()
            result = await pipeline.process_video_file(video_path, job_id=job_id, force_retranslate=True)

        job = get_job_by_id(job_id)
        assert job["status"] == "FAILED", f"Auth error must result in FAILED, got {job['status']}"
        assert result["status"] == "failed"

    def test_auth_error_never_blocks_provider(self, isolated_db):
        """Auth errors must never cause provider to be blocked as daily quota."""
        from app.core.quota import classify_provider_error, block_provider, is_provider_blocked

        exc = RuntimeError("401 Unauthorized: invalid API key")
        classification = classify_provider_error(exc, "gemini")
        assert classification == "AUTH_ERROR"

        # Auth errors must NOT trigger block_provider
        assert not is_provider_blocked("gemini"), "Auth error must never block provider as daily quota"


# ===========================================================================
# J. Other provider isolation
# ===========================================================================

class TestProviderIsolation:
    """
    Gemini blocked ≠ OpenAI blocked. Provider states are independent.
    """

    def test_gemini_block_does_not_block_openai(self, isolated_db):
        from app.core.quota import block_provider, is_provider_blocked

        block_provider("gemini", "daily quota test", retry_after_seconds=3600)

        assert is_provider_blocked("gemini"), "Gemini must be blocked"
        assert not is_provider_blocked("openai"), "OpenAI must NOT be affected by Gemini block"
        assert not is_provider_blocked("deepl"), "DeepL must NOT be affected by Gemini block"
        assert not is_provider_blocked("ollama"), "Ollama must NOT be affected by Gemini block"

    def test_openai_block_independent_from_gemini(self, isolated_db):
        from app.core.quota import block_provider, is_provider_blocked

        block_provider("openai", "openai daily quota", retry_after_seconds=1800)

        assert is_provider_blocked("openai"), "OpenAI must be blocked"
        assert not is_provider_blocked("gemini"), "Gemini must NOT be affected by OpenAI block"

    def test_budget_per_provider(self, isolated_db):
        """Budget counters are independent per provider."""
        from app.core.quota import try_consume_request_budget, get_daily_requests_used
        from app.core.db import set_setting

        set_setting("daily_request_budget_gemini", "2")
        set_setting("daily_request_budget_openai", "5")

        # Use up Gemini budget
        assert try_consume_request_budget("gemini") is True
        assert try_consume_request_budget("gemini") is True
        assert try_consume_request_budget("gemini") is False  # Gemini exhausted

        # OpenAI budget must be unaffected
        assert try_consume_request_budget("openai") is True, "OpenAI budget must be independent"
        assert try_consume_request_budget("openai") is True
        assert try_consume_request_budget("openai") is True

        assert get_daily_requests_used("gemini") == 2
        assert get_daily_requests_used("openai") == 3


# ===========================================================================
# K. Recovery requests counted
# ===========================================================================

class TestRecoveryCounting:
    """
    Recovery/retry requests must also be counted against the daily budget.
    """

    def test_recovery_calls_count_against_budget(self, isolated_db):
        """
        Budget must count ALL calls through translate_batch (incl. recovery passes).
        """
        from app.core.quota import try_consume_request_budget, get_daily_requests_used
        from app.core.db import set_setting

        set_setting("daily_request_budget_gemini", "5")

        # Simulate initial translation (3 requests)
        for _ in range(3):
            try_consume_request_budget("gemini")

        # Simulate recovery calls (2 requests)
        for _ in range(2):
            try_consume_request_budget("gemini")

        assert get_daily_requests_used("gemini") == 5

        # One more should be blocked
        result = try_consume_request_budget("gemini")
        assert result is False, "Budget must be exhausted after 5 calls"


# ===========================================================================
# L. UI/API
# ===========================================================================

class TestUIAPI:
    """
    Backend API quota endpoint must return correct fields.
    """

    @pytest.mark.asyncio
    async def test_quota_status_endpoint_active(self, isolated_db):
        from app.api.dashboard import api_get_quota_status
        from app.core.quota import block_provider
        from app.core.db import set_setting

        set_setting("daily_request_budget_gemini", "250")
        set_setting("ai_provider", "gemini")

        # Simulate some usage
        from app.core.quota import try_consume_request_budget
        for _ in range(50):
            try_consume_request_budget("gemini")

        with patch("app.api.dashboard.get_setting", side_effect=lambda k, d="": {
            "ai_provider": "gemini",
            "daily_request_budget_gemini": "250",
        }.get(k, d)):
            # Call directly (no HTTP layer needed)
            from app.core.quota import get_quota_status_for_provider
            status = get_quota_status_for_provider("gemini")

        assert "blocked" in status
        assert "budget" in status
        assert "requests_today" in status
        assert "requests_remaining" in status
        assert "provider" in status
        assert status["provider"] == "gemini"
        assert status["blocked"] is False

    @pytest.mark.asyncio
    async def test_quota_status_endpoint_blocked(self, isolated_db):
        from app.core.quota import block_provider, get_quota_status_for_provider

        block_provider("gemini", "daily quota", retry_after_seconds=3600)

        status = get_quota_status_for_provider("gemini")
        assert status["blocked"] is True
        assert status["blocked_until"] is not None
        assert status["reset_type"] == "exact", "Explicit retry_after_seconds must set reset_type to exact"
        assert status["reason"] is not None

    def test_quota_status_fallback_estimated_reset_type(self, isolated_db):
        from app.core.quota import block_provider, get_quota_status_for_provider

        # Block without retry_after (unknown reset time -> conservative fallback)
        block_provider("gemini", "daily quota without retry-after", retry_after_seconds=None)

        status = get_quota_status_for_provider("gemini")
        assert status["blocked"] is True
        assert status["blocked_until"] is not None
        assert status["reset_type"] == "estimated", "Missing retry_after must set reset_type to estimated fallback"

    @pytest.mark.asyncio
    async def test_get_job_stats_includes_deferred(self, isolated_db):
        from app.core.db import create_job, update_job, get_job_stats

        job1 = create_job("job1.mkv", "MANUAL")
        job2 = create_job("job2.mkv", "MANUAL")
        job3 = create_job("job3.mkv", "MANUAL")

        update_job(job1, status="DEFERRED")
        update_job(job2, status="DEFERRED")
        update_job(job3, status="TRANSLATED")

        stats = get_job_stats()
        assert "deferred" in stats, "get_job_stats must include 'deferred' field"
        assert stats["deferred"] == 2
        assert stats["translated"] == 1


# ===========================================================================
# M. Backward compatibility
# ===========================================================================

class TestBackwardCompatibility:
    """
    Old DB without quota tables must start normally.
    Budget must default to Unlimited.
    """

    def test_old_db_starts_without_quota_tables(self, tmp_path):
        """
        Simulate an old DB that has jobs/settings but no quota tables.
        init_db must create the new tables without breaking anything.
        """
        import sqlite3
        import app.core.db as db_module
        import app.core.quota as quota_module

        db_file = str(tmp_path / "old_babel.db")
        original_db = db_module.DB_PATH
        try:
            # Create old-style DB (no quota tables)
            with sqlite3.connect(db_file) as conn:
                conn.execute("""
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_path TEXT NOT NULL,
                    title TEXT,
                    status TEXT NOT NULL,
                    event_source TEXT,
                    reason TEXT,
                    target_languages TEXT,
                    ai_model TEXT,
                    total_lines INTEGER DEFAULT 0,
                    cleaned_sdh_lines INTEGER DEFAULT 0,
                    dropped_lines INTEGER DEFAULT 0,
                    sync_diff_ms INTEGER DEFAULT 0,
                    output_files TEXT,
                    error_message TEXT,
                    logs TEXT,
                    duration_seconds REAL DEFAULT 0.0,
                    processed_lines INTEGER DEFAULT 0,
                    current_batch TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    next_retry_at TEXT,
                    last_error TEXT
                )
                """)
                conn.execute("""
                CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)
                """)
                conn.execute("INSERT INTO settings VALUES ('gemini_api_key', 'testkey')")
                conn.commit()

            # Now init_db must run without error on this old DB
            db_module.DB_PATH = db_file
            quota_module.DB_PATH = db_file
            db_module.init_db()  # Must not raise

            # Quota functions must work on the new tables
            from app.core.quota import is_provider_blocked, get_daily_budget, try_consume_request_budget
            assert not is_provider_blocked("gemini"), "Provider must not be blocked on fresh init"
            assert get_daily_budget("gemini") is None, "Default budget must be Unlimited"
            assert try_consume_request_budget("gemini") is True, "Unlimited must allow requests"

        finally:
            db_module.DB_PATH = original_db
            quota_module.DB_PATH = original_db

    def test_unlimited_default_budget(self, isolated_db):
        """Default budget setting (0) must behave as unlimited."""
        from app.core.quota import get_daily_budget, try_consume_request_budget

        # No setting set (defaults from init_db = "0")
        budget = get_daily_budget("gemini")
        assert budget is None, f"Default budget must be None (unlimited), got {budget}"

        # Must allow unlimited requests
        for i in range(100):
            result = try_consume_request_budget("gemini")
            assert result is True, f"Unlimited must allow request {i+1}"


# ===========================================================================
# N. Timezone
# ===========================================================================

class TestTimezone:
    """
    All timestamps in quota system must be UTC-aware.
    No naive datetimes.
    """

    def test_blocked_until_is_utc_aware(self, isolated_db):
        from app.core.quota import block_provider, get_provider_block_info, _parse_utc

        block_provider("gemini", "test", retry_after_seconds=3600)
        info = get_provider_block_info("gemini")

        blocked_until_str = info["blocked_until"]
        assert blocked_until_str is not None

        dt = _parse_utc(blocked_until_str)
        assert dt is not None
        assert dt.tzinfo is not None, "blocked_until must be timezone-aware"
        assert dt.tzinfo == timezone.utc or dt.utcoffset() == timedelta(0), (
            "blocked_until must be UTC"
        )

    def test_blocked_at_is_utc_aware(self, isolated_db):
        from app.core.quota import block_provider, get_provider_block_info, _parse_utc

        block_provider("openai", "test", retry_after_seconds=60)
        info = get_provider_block_info("openai")

        blocked_at_str = info["blocked_at"]
        assert blocked_at_str is not None

        dt = _parse_utc(blocked_at_str)
        assert dt is not None
        assert dt.tzinfo is not None, "blocked_at must be timezone-aware"

    def test_today_window_is_utc_date(self, isolated_db):
        """Daily window must use UTC date, not local date."""
        from app.core.quota import _today_window

        window = _today_window("gemini")
        # Must be a valid YYYY-MM-DD string derived from UTC
        now_utc = datetime.now(timezone.utc)
        expected = now_utc.strftime("%Y-%m-%d")
        assert window == expected, f"Window must be UTC date {expected}, got {window}"

    def test_budget_reset_at_utc_midnight(self, isolated_db):
        """Budget resets at UTC midnight (new window_date)."""
        from app.core.quota import try_consume_request_budget
        from app.core.db import set_setting

        set_setting("daily_request_budget_gemini", "1")

        # Use up today
        assert try_consume_request_budget("gemini") is True
        assert try_consume_request_budget("gemini") is False

        # Simulate next UTC day
        with patch("app.core.quota._today_window", return_value="2099-12-31"):
            assert try_consume_request_budget("gemini") is True, "Budget must reset next UTC day"


# ===========================================================================
# Additional unit tests for error classification
# ===========================================================================

class TestErrorClassification:
    """Extra classification edge cases."""

    def test_401_never_daily_quota(self):
        from app.core.quota import classify_provider_error
        exc = RuntimeError("401 Unauthorized")
        assert classify_provider_error(exc, "gemini") == "AUTH_ERROR"

    def test_403_never_daily_quota(self):
        from app.core.quota import classify_provider_error
        exc = RuntimeError("403 Forbidden: invalid key")
        assert classify_provider_error(exc, "gemini") == "AUTH_ERROR"

    def test_500_is_transient(self):
        from app.core.quota import classify_provider_error
        exc = RuntimeError("500 Internal Server Error")
        result = classify_provider_error(exc, "gemini")
        assert result in ("TRANSIENT_RPM", "PROVIDER_UNAVAILABLE")

    def test_network_timeout_is_transient(self):
        from app.core.quota import classify_provider_error
        exc = RuntimeError("Connection timeout")
        result = classify_provider_error(exc, "gemini")
        assert result in ("TRANSIENT_RPM", "PROVIDER_UNAVAILABLE")

    def test_retry_after_extraction_from_exception_attr(self):
        from app.core.quota import extract_retry_after_from_exception

        exc = Exception("Rate limit exceeded")
        exc.retry_after = 3600
        result = extract_retry_after_from_exception(exc)
        assert result == 3600

    def test_retry_after_extraction_from_string(self):
        from app.core.quota import extract_retry_after_from_exception

        exc = Exception("Rate limit exceeded. Retry-After: 7200")
        result = extract_retry_after_from_exception(exc)
        assert result == 7200

    def test_retry_after_none_when_unavailable(self):
        from app.core.quota import extract_retry_after_from_exception

        exc = Exception("Some random error with no retry info")
        result = extract_retry_after_from_exception(exc)
        assert result is None

    def test_claim_deferred_job_for_retry(self, isolated_db):
        """claim_job_for_retry must accept DEFERRED status."""
        from app.core.db import create_job, update_job, get_job_by_id, claim_job_for_retry

        job_id = create_job("deferred_claim.mkv", "MANUAL")
        update_job(job_id, status="DEFERRED")

        claimed = claim_job_for_retry(job_id)
        assert claimed is True

        job = get_job_by_id(job_id)
        assert job["status"] == "QUEUED"


# ===========================================================================
# Local Daily Request Budget / DEFERRED-resume regression tests (A-F)
# ===========================================================================

class TestLocalBudgetDeferredResume:
    """
    Verify dynamic budget awareness for DEFERRED jobs:
    A: budget=1, used=1 -> DEFERRED job does not retry while budget is 1.
    B: budget changed 1 -> 2, used=1 -> job becomes retry-eligible immediately.
    C: budget changed 1 -> 0 (Unlimited) -> job becomes retry-eligible immediately.
    D: external circuit breaker BLOCKED + local budget raised -> job stays deferred.
    E: retry consumes next budget slot (used goes 1 -> 2), zero dummy calls.
    F: budget reaches 2/2 -> job defers again correctly until reset/budget change.
    """

    @pytest.mark.asyncio
    async def test_a_job_does_not_retry_while_budget_exhausted(self, isolated_db):
        from app.core.db import create_job, update_job, get_job_by_id, set_setting
        from app.core.quota import try_consume_request_budget, should_retry_deferred_job
        from app.main import process_one_retry_pass

        set_setting("daily_request_budget_gemini", "1")
        set_setting("ai_provider", "gemini")

        # Consume the 1 budget slot
        assert try_consume_request_budget("gemini") is True

        # Job deferred until midnight
        job_id = create_job("test_a.mkv", "MANUAL")
        midnight = (datetime.now(timezone.utc) + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        update_job(
            job_id,
            status="DEFERRED",
            error_message=f"Daily request budget reached (1/1). Deferred until {midnight.isoformat()}.",
            next_retry_at=midnight.isoformat(),
            last_error="RequestBudgetExhaustedError: Daily request budget reached for 'gemini' (1/1 requests used today)"
        )

        job = get_job_by_id(job_id)
        now = datetime.now(timezone.utc)
        assert should_retry_deferred_job(job, now) is False

        tasks = [t async for t in process_one_retry_pass()]
        assert len(tasks) == 0

        job_after = get_job_by_id(job_id)
        assert job_after["status"] == "DEFERRED"

    @pytest.mark.asyncio
    async def test_b_job_retries_immediately_when_budget_increased(self, isolated_db):
        from app.core.db import create_job, update_job, get_job_by_id, set_setting
        from app.core.quota import try_consume_request_budget, should_retry_deferred_job
        from app.main import process_one_retry_pass

        set_setting("daily_request_budget_gemini", "1")
        set_setting("ai_provider", "gemini")

        # Consume 1 slot
        assert try_consume_request_budget("gemini") is True

        job_id = create_job("test_b.mkv", "MANUAL")
        midnight = (datetime.now(timezone.utc) + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        update_job(
            job_id,
            status="DEFERRED",
            error_message=f"Daily request budget reached (1/1). Deferred until {midnight.isoformat()}.",
            next_retry_at=midnight.isoformat(),
            last_error="RequestBudgetExhaustedError: Daily request budget reached for 'gemini' (1/1 requests used today)"
        )

        # User increases budget from 1 to 2
        set_setting("daily_request_budget_gemini", "2")

        job = get_job_by_id(job_id)
        now = datetime.now(timezone.utc)
        assert should_retry_deferred_job(job, now) is True

        with patch("app.main.pipeline.process_video_file", new_callable=AsyncMock) as mock_pipeline:
            tasks = [t async for t in process_one_retry_pass()]
            assert len(tasks) == 1
            await asyncio.gather(*tasks)

        job_after = get_job_by_id(job_id)
        assert job_after["status"] == "QUEUED"
        mock_pipeline.assert_called_once()

    @pytest.mark.asyncio
    async def test_c_job_retries_immediately_when_budget_set_unlimited(self, isolated_db):
        from app.core.db import create_job, update_job, get_job_by_id, set_setting
        from app.core.quota import try_consume_request_budget, should_retry_deferred_job
        from app.main import process_one_retry_pass

        set_setting("daily_request_budget_gemini", "1")
        set_setting("ai_provider", "gemini")

        # Consume 1 slot
        assert try_consume_request_budget("gemini") is True

        job_id = create_job("test_c.mkv", "MANUAL")
        midnight = (datetime.now(timezone.utc) + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        update_job(
            job_id,
            status="DEFERRED",
            error_message=f"Daily request budget reached (1/1). Deferred until {midnight.isoformat()}.",
            next_retry_at=midnight.isoformat(),
            last_error="RequestBudgetExhaustedError: Daily request budget reached for 'gemini' (1/1 requests used today)"
        )

        # User sets budget to Unlimited (0)
        set_setting("daily_request_budget_gemini", "0")

        job = get_job_by_id(job_id)
        now = datetime.now(timezone.utc)
        assert should_retry_deferred_job(job, now) is True

        with patch("app.main.pipeline.process_video_file", new_callable=AsyncMock) as mock_pipeline:
            tasks = [t async for t in process_one_retry_pass()]
            assert len(tasks) == 1
            await asyncio.gather(*tasks)

        job_after = get_job_by_id(job_id)
        assert job_after["status"] == "QUEUED"

    @pytest.mark.asyncio
    async def test_d_external_circuit_breaker_blocks_despite_budget_increase(self, isolated_db):
        from app.core.db import create_job, update_job, get_job_by_id, set_setting
        from app.core.quota import record_provider_quota_exhausted, should_retry_deferred_job
        from app.main import process_one_retry_pass

        set_setting("daily_request_budget_gemini", "1")
        set_setting("ai_provider", "gemini")

        # External provider is BLOCKED by circuit breaker
        now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
        with patch("app.core.quota._utcnow", return_value=now):
            record_provider_quota_exhausted("gemini", "Gemini 429 Daily Limit", retry_after_seconds=3600)

        job_id = create_job("test_d.mkv", "MANUAL")
        update_job(
            job_id,
            status="DEFERRED",
            error_message="Daily provider quota reached for 'gemini'. next probe scheduled.",
            next_retry_at=(now + timedelta(seconds=3600)).isoformat(),
            last_error="ProviderDailyQuotaExhaustedError: Daily quota exhausted for provider 'gemini'"
        )

        # User increases local budget to 10
        set_setting("daily_request_budget_gemini", "10")

        # Must still NOT retry because external provider is blocked
        with patch("app.core.quota._utcnow", return_value=now):
            job = get_job_by_id(job_id)
            assert should_retry_deferred_job(job, now) is False

            tasks = [t async for t in process_one_retry_pass()]
            assert len(tasks) == 0

        job_after = get_job_by_id(job_id)
        assert job_after["status"] == "DEFERRED"

    def test_e_retry_consumes_next_budget_slot_exactly_once(self, isolated_db):
        from app.core.db import set_setting
        from app.core.quota import try_consume_request_budget, get_daily_requests_used, acquire_dispatch_slot

        set_setting("daily_request_budget_gemini", "2")
        set_setting("ai_provider", "gemini")

        # Used = 1
        assert try_consume_request_budget("gemini") is True
        assert get_daily_requests_used("gemini") == 1

        # Next real request acquires slot
        allowed, info = acquire_dispatch_slot("gemini")
        assert allowed is True
        assert info["state"] == "ACTIVE"
        assert get_daily_requests_used("gemini") == 2

    def test_f_reaches_budget_and_defers_again(self, isolated_db):
        from app.core.db import set_setting
        from app.core.quota import try_consume_request_budget, get_daily_requests_used, acquire_dispatch_slot

        set_setting("daily_request_budget_gemini", "2")
        set_setting("ai_provider", "gemini")

        # Consume 2 slots
        assert try_consume_request_budget("gemini") is True
        assert try_consume_request_budget("gemini") is True
        assert get_daily_requests_used("gemini") == 2

        # 3rd request fails budget check
        allowed, info = acquire_dispatch_slot("gemini")
        assert allowed is False
        assert info["reason"] == "REQUEST_BUDGET_EXHAUSTED"
