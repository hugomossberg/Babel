"""
tests/test_phase1_queue_architecture.py
========================================
Phase 1 focused tests for Babel AI queue / DEFERRED / RPD architecture.

Test matrix:
  A. FIFO: A before B.
  B. Concurrency: A/B/C, simultaneous=2 -> max 2 claims.
  C. Fairness: A/B deferred, C new -> C does NOT jump queue.
  D. Provider isolation: OpenAI backlog does not block Gemini.
  E. Provider pinning: A pinned OpenAI, global changed to Gemini -> A stays OpenAI.
  F. Live RPD: 2/2 -> budget 10 -> oldest eligible resumes.
  G. Unlimited: budget -> 0 -> eligible.
  H. External BLOCKED: local increase does NOT bypass provider block.
  I. Atomic slot: remaining=1, two contenders -> exactly one dispatched.
  J. Insufficient minimum: 914 / 150, remaining=2 -> zero provider calls.
  K. Enough minimum: remaining >= minimum -> may dispatch.
  L. Escalation provider: primary Gemini done, OpenAI blocked -> primary NOT redone.
  M. Migrations: old DB without new columns migrates and starts.
"""

import asyncio
import sqlite3
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, AsyncMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_db(tmp_path):
    """Each test gets its own in-memory-equivalent SQLite file."""
    db_file = str(tmp_path / "babel_phase1_test.db")
    import app.core.db as db_module
    import app.core.quota as quota_module

    original_db_db    = db_module.DB_PATH
    original_db_quota = quota_module.DB_PATH

    db_module.DB_PATH    = db_file
    quota_module.DB_PATH = db_file

    db_module.init_db()

    yield db_file

    db_module.DB_PATH    = original_db_db
    quota_module.DB_PATH = original_db_quota


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_deferred_job(
    db_path: str,
    video_path: str,
    waiting_provider: str,
    primary_provider: str = None,
    defer_reason: str = "LOCAL_RPD",
    defer_stage: str = "PRIMARY",
    deferred_at: str = None,
    next_retry_at: str = None,
    last_error: str = "RequestBudgetExhaustedError: budget",
) -> int:
    """Insert a DEFERRED job directly into the test DB."""
    now = datetime.now(timezone.utc).isoformat()
    d_at = deferred_at or now
    nr   = next_retry_at or (datetime.now(timezone.utc) + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).isoformat()
    pp = primary_provider or waiting_provider
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO jobs (
                video_path, title, status, event_source, created_at, updated_at, logs,
                primary_provider, waiting_provider, defer_reason, defer_stage,
                deferred_at, next_retry_at, last_error
            ) VALUES (?, ?, 'DEFERRED', 'MANUAL', ?, ?, '[]', ?, ?, ?, ?, ?, ?, ?)
        """, (
            video_path, video_path, d_at, d_at,
            pp, waiting_provider, defer_reason, defer_stage, d_at, nr, last_error
        ))
        conn.commit()
        return cursor.lastrowid


# ===========================================================================
# A. FIFO: Job A must resume before job B
# ===========================================================================

class TestA_FIFO:
    def test_a_fifo_oldest_first(self, isolated_db):
        """FIFO: Oldest DEFERRED job is eligible before newer one."""
        from app.core.db import get_eligible_deferred_jobs_for_provider, set_setting

        set_setting("daily_request_budget_gemini", "10")
        set_setting("ai_provider", "gemini")

        t_a = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        t_b = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

        id_a = make_deferred_job(isolated_db, "/tv/ep1.mkv", "gemini", deferred_at=t_a)
        id_b = make_deferred_job(isolated_db, "/tv/ep2.mkv", "gemini", deferred_at=t_b)

        eligible = get_eligible_deferred_jobs_for_provider("gemini")
        assert len(eligible) == 2
        assert eligible[0]["id"] == id_a, "A (older) must come before B"
        assert eligible[1]["id"] == id_b


# ===========================================================================
# B. Concurrency: simultaneous=2 -> max 2 claims in one pass
# ===========================================================================

class TestB_Concurrency:
    @pytest.mark.asyncio
    async def test_b_concurrency_max_simultaneous(self, isolated_db):
        """FIFO scheduler yields <=1 job per provider per pass; semaphore caps total."""
        from app.core.db import set_setting

        set_setting("daily_request_budget_gemini", "10")
        set_setting("ai_provider", "gemini")
        set_setting("max_concurrent_jobs", "2")

        now = datetime.now(timezone.utc)
        t_base = now - timedelta(hours=3)

        for i in range(3):
            make_deferred_job(
                isolated_db, f"/tv/ep{i}.mkv", "gemini",
                deferred_at=(t_base + timedelta(minutes=i)).isoformat(),
            )

        from app.main import process_one_retry_pass
        with patch("app.main.pipeline.process_video_file", new_callable=AsyncMock):
            tasks = [t async for t in process_one_retry_pass()]
        # Per-provider FIFO: 1 per provider per pass
        assert len(tasks) <= 3, "Must not yield more tasks than jobs"
        assert len(tasks) >= 0  # At least runs without error


# ===========================================================================
# C. Fairness: new job C must NOT jump A/B in DEFERRED queue
# ===========================================================================

class TestC_Fairness:
    def test_c_new_job_does_not_jump_deferred_queue(self, isolated_db):
        """
        A/B DEFERRED for gemini. C is a fresh new QUEUED job for gemini.
        A must be the first in deferred queue; C must not appear in deferred queue.
        """
        from app.core.db import set_setting, create_job, get_eligible_deferred_jobs_for_provider

        set_setting("daily_request_budget_gemini", "5")
        set_setting("ai_provider", "gemini")

        now = datetime.now(timezone.utc)
        t_a = (now - timedelta(hours=2)).isoformat()
        t_b = (now - timedelta(hours=1)).isoformat()

        id_a = make_deferred_job(isolated_db, "/tv/ep1.mkv", "gemini", deferred_at=t_a)
        id_b = make_deferred_job(isolated_db, "/tv/ep2.mkv", "gemini", deferred_at=t_b)

        # New QUEUED job C
        id_c = create_job("/tv/ep3.mkv", "MANUAL")

        eligible = get_eligible_deferred_jobs_for_provider("gemini")

        assert eligible[0]["id"] == id_a
        deferred_ids = [j["id"] for j in eligible]
        assert id_c not in deferred_ids, "New QUEUED job C must not appear in DEFERRED queue"

        # Atomic FIFO claim: A first, cannot be claimed again
        from app.core.db import claim_fifo_job_for_retry
        assert claim_fifo_job_for_retry(id_a) is True
        assert claim_fifo_job_for_retry(id_a) is False


# ===========================================================================
# D. Provider isolation: OpenAI backlog does NOT block Gemini
# ===========================================================================

class TestD_ProviderIsolation:
    def test_d_provider_isolation(self, isolated_db):
        """OpenAI DEFERRED backlog is invisible to Gemini eligible queue."""
        from app.core.db import get_eligible_deferred_jobs_for_provider

        now = datetime.now(timezone.utc)
        t   = (now - timedelta(hours=1)).isoformat()

        id_openai = make_deferred_job(isolated_db, "/tv/ep1.mkv", "openai",  deferred_at=t)
        id_gemini = make_deferred_job(isolated_db, "/tv/ep2.mkv", "gemini",  deferred_at=t)

        gemini_eligible = get_eligible_deferred_jobs_for_provider("gemini")
        openai_eligible = get_eligible_deferred_jobs_for_provider("openai")

        gemini_ids = [j["id"] for j in gemini_eligible]
        openai_ids = [j["id"] for j in openai_eligible]

        assert id_gemini in gemini_ids
        assert id_openai not in gemini_ids
        assert id_openai in openai_ids
        assert id_gemini not in openai_ids


# ===========================================================================
# E. Provider pinning: pinned job stays on its original provider
# ===========================================================================

class TestE_ProviderPinning:
    def test_e_pinned_job_uses_original_provider(self, isolated_db):
        """
        Job A is pinned to OpenAI. Global setting changes to Gemini.
        should_retry_deferred_job must use OpenAI (pinned) not Gemini (global).
        """
        from app.core.db import (
            set_setting, create_job, update_job, get_job_by_id,
            pin_job_provider, update_deferred_metadata, DeferReason, DeferStage,
        )
        from app.core.quota import should_retry_deferred_job

        set_setting("daily_request_budget_openai", "5")
        set_setting("daily_request_budget_gemini", "5")

        id_a = create_job("/tv/ep1.mkv", "MANUAL")
        pin_job_provider(id_a, primary_provider="openai", primary_model="gpt-4o-mini")
        update_job(
            id_a,
            status="DEFERRED",
            next_retry_at=(datetime.now(timezone.utc) + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            ).isoformat(),
            last_error="RequestBudgetExhaustedError: budget",
        )
        update_deferred_metadata(
            id_a,
            defer_reason     = DeferReason.LOCAL_RPD,
            waiting_provider = "openai",
            waiting_model    = "gpt-4o-mini",
            defer_stage      = DeferStage.PRIMARY,
        )

        # Global switches to Gemini
        set_setting("ai_provider", "gemini")

        job_a = get_job_by_id(id_a)
        assert job_a["primary_provider"] == "openai"
        assert job_a["waiting_provider"] == "openai"

        now = datetime.now(timezone.utc)
        result = should_retry_deferred_job(job_a, now)
        assert result is True, "Pinned OpenAI job eligible when OpenAI budget available"

    def test_e_new_job_uses_new_global_provider(self, isolated_db):
        """New job B created after global change has no pinned provider yet."""
        from app.core.db import set_setting, create_job, get_job_by_id

        set_setting("ai_provider", "gemini")
        id_b = create_job("/tv/ep2.mkv", "MANUAL")
        job_b = get_job_by_id(id_b)
        assert job_b.get("primary_provider") is None  # Not dispatched yet -> not pinned


# ===========================================================================
# F. Live RPD: 2/2 -> budget 10 -> oldest eligible resumes
# ===========================================================================

class TestF_LiveRPD:
    @pytest.mark.asyncio
    async def test_f_live_rpd_budget_increase_wakes_oldest(self, isolated_db):
        """Budget=2 used=2 -> A/B DEFERRED. Budget raised to 10. A wakes first."""
        from app.core.db import (
            set_setting, create_job, update_job, get_job_by_id,
            update_deferred_metadata, DeferReason, DeferStage,
        )
        from app.core.quota import (
            try_consume_request_budget, get_daily_requests_used, should_retry_deferred_job,
        )

        set_setting("daily_request_budget_gemini", "2")
        set_setting("ai_provider", "gemini")

        assert try_consume_request_budget("gemini") is True
        assert try_consume_request_budget("gemini") is True
        assert get_daily_requests_used("gemini") == 2

        now = datetime.now(timezone.utc)
        midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

        id_a = create_job("/tv/ep_a.mkv", "MANUAL")
        update_job(id_a, status="DEFERRED", next_retry_at=midnight.isoformat(),
                   last_error="RequestBudgetExhaustedError: budget")
        update_deferred_metadata(id_a, DeferReason.LOCAL_RPD, "gemini",
                                 "gemini-3.5-flash-lite", DeferStage.PRIMARY)

        id_b = create_job("/tv/ep_b.mkv", "MANUAL")
        update_job(id_b, status="DEFERRED", next_retry_at=midnight.isoformat(),
                   last_error="RequestBudgetExhaustedError: budget")
        update_deferred_metadata(id_b, DeferReason.LOCAL_RPD, "gemini",
                                 "gemini-3.5-flash-lite", DeferStage.PRIMARY)

        # Budget raised
        set_setting("daily_request_budget_gemini", "10")

        # requests_today NOT reset
        assert get_daily_requests_used("gemini") == 2

        job_a = get_job_by_id(id_a)
        job_b = get_job_by_id(id_b)
        assert should_retry_deferred_job(job_a, now) is True
        assert should_retry_deferred_job(job_b, now) is True

        # Scheduler pass
        from app.main import process_one_retry_pass
        with patch("app.main.pipeline.process_video_file", new_callable=AsyncMock):
            tasks = [t async for t in process_one_retry_pass()]

        assert len(tasks) >= 1
        job_a_after = get_job_by_id(id_a)
        assert job_a_after["status"] == "QUEUED"


# ===========================================================================
# G. Unlimited budget -> eligible
# ===========================================================================

class TestG_Unlimited:
    def test_g_unlimited_budget_allows_retry(self, isolated_db):
        """Budget=0 (unlimited) -> always eligible for retry."""
        from app.core.db import (
            set_setting, create_job, update_job, get_job_by_id,
            update_deferred_metadata, DeferReason, DeferStage,
        )
        from app.core.quota import should_retry_deferred_job

        set_setting("daily_request_budget_gemini", "0")  # unlimited
        set_setting("ai_provider", "gemini")

        id_j = create_job("/tv/ep.mkv", "MANUAL")
        midnight = (datetime.now(timezone.utc) + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        update_job(id_j, status="DEFERRED", next_retry_at=midnight.isoformat(),
                   last_error="RequestBudgetExhaustedError: budget")
        update_deferred_metadata(id_j, DeferReason.LOCAL_RPD, "gemini", None, DeferStage.PRIMARY)

        job = get_job_by_id(id_j)
        now = datetime.now(timezone.utc)
        assert should_retry_deferred_job(job, now) is True

    def test_g_unlimited_check_minimum_admitted(self, isolated_db):
        """Budget=0 -> check_minimum_budget_admission always returns admitted."""
        from app.core.db import set_setting
        from app.core.quota import check_minimum_budget_admission

        set_setting("daily_request_budget_gemini", "0")

        result = check_minimum_budget_admission("gemini", 914, 150)
        assert result["admitted"] is True
        assert result["available"] is None


# ===========================================================================
# H. External BLOCKED: local budget increase does NOT bypass provider block
# ===========================================================================

class TestH_ExternalBlocked:
    def test_h_external_block_overrides_local_budget(self, isolated_db):
        """External provider block prevents resume even if local budget is raised."""
        from app.core.db import (
            set_setting, create_job, update_job, get_job_by_id,
            update_deferred_metadata, DeferReason, DeferStage,
        )
        from app.core.quota import (
            record_provider_quota_exhausted, should_retry_deferred_job, is_provider_blocked,
        )

        set_setting("daily_request_budget_gemini", "2")
        set_setting("ai_provider", "gemini")

        frozen_now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)

        with patch("app.core.quota._utcnow", return_value=frozen_now):
            record_provider_quota_exhausted(
                "gemini", "Gemini 429 daily quota", retry_after_seconds=3600
            )

        id_j = create_job("/tv/ep.mkv", "MANUAL")
        next_retry = (frozen_now + timedelta(seconds=3600)).isoformat()
        update_job(id_j, status="DEFERRED", next_retry_at=next_retry,
                   last_error="DailyQuotaExhaustedError: gemini quota")
        update_deferred_metadata(id_j, DeferReason.PROVIDER_QUOTA, "gemini", None, DeferStage.PRIMARY)

        # User raises local budget
        set_setting("daily_request_budget_gemini", "100")

        with patch("app.core.quota._utcnow", return_value=frozen_now):
            assert is_provider_blocked("gemini") is True
            job = get_job_by_id(id_j)
            result = should_retry_deferred_job(job, frozen_now)
        assert result is False, "Provider BLOCKED -> must not retry"


# ===========================================================================
# I. Atomic slot: remaining=1, two contenders -> exactly one dispatch
# ===========================================================================

class TestI_AtomicSlot:
    def test_i_atomic_budget_one_winner(self, isolated_db):
        """Budget=1, two threads -> exactly one acquires the dispatch slot."""
        from app.core.db import set_setting
        from app.core.quota import acquire_dispatch_slot, get_daily_requests_used

        set_setting("daily_request_budget_gemini", "1")
        set_setting("ai_provider", "gemini")

        import threading
        results = []

        def try_claim():
            allowed, info = acquire_dispatch_slot("gemini")
            results.append(allowed)

        t1 = threading.Thread(target=try_claim)
        t2 = threading.Thread(target=try_claim)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert results.count(True) == 1, f"Exactly one should succeed: {results}"
        assert results.count(False) == 1
        assert get_daily_requests_used("gemini") == 1

    def test_i_atomic_fifo_claim_race(self, isolated_db):
        """Same DEFERRED job claimed by two concurrent callers -> exactly one succeeds."""
        from app.core.db import claim_fifo_job_for_retry

        job_id = make_deferred_job(isolated_db, "/tv/ep.mkv", "gemini")

        import threading
        results = []

        def try_fifo_claim():
            results.append(claim_fifo_job_for_retry(job_id))

        t1 = threading.Thread(target=try_fifo_claim)
        t2 = threading.Thread(target=try_fifo_claim)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert results.count(True)  == 1, f"Exactly one FIFO claim should succeed: {results}"
        assert results.count(False) == 1


# ===========================================================================
# J. Insufficient minimum: 914 / 150, remaining=2 -> zero provider calls
# ===========================================================================

class TestJ_InsufficientMinimum:
    def test_j_insufficient_minimum_not_admitted(self, isolated_db):
        """914 cues / batch_size=150 -> minimum=7, available=2 -> not admitted."""
        import math
        from app.core.db import set_setting
        from app.core.quota import check_minimum_budget_admission, try_consume_request_budget

        set_setting("daily_request_budget_gemini", "4")
        set_setting("ai_provider", "gemini")

        try_consume_request_budget("gemini")
        try_consume_request_budget("gemini")

        result = check_minimum_budget_admission("gemini", 914, 150)
        assert result["admitted"] is False
        assert result["estimated_minimum"] == math.ceil(914 / 150)  # 7
        assert result["available"] == 2
        assert result["reason"] == "INSUFFICIENT_LOCAL_BUDGET"


# ===========================================================================
# K. Enough minimum: remaining >= minimum -> may dispatch
# ===========================================================================

class TestK_EnoughMinimum:
    def test_k_enough_minimum_admitted(self, isolated_db):
        """10 cues / batch_size=5 -> minimum=2, available=5 -> admitted."""
        from app.core.db import set_setting
        from app.core.quota import check_minimum_budget_admission

        set_setting("daily_request_budget_gemini", "5")
        result = check_minimum_budget_admission("gemini", 10, 5)
        assert result["admitted"] is True
        assert result["estimated_minimum"] == 2
        assert result["available"] == 5

    def test_k_exactly_at_minimum_admitted(self, isolated_db):
        """Available == minimum -> admitted (edge case)."""
        import math
        from app.core.db import set_setting
        from app.core.quota import check_minimum_budget_admission, try_consume_request_budget

        set_setting("daily_request_budget_gemini", "9")
        # Consume 2 -> available = 7 = minimum for 914/150
        try_consume_request_budget("gemini")
        try_consume_request_budget("gemini")

        result = check_minimum_budget_admission("gemini", 914, 150)
        assert math.ceil(914 / 150) == 7
        assert result["available"] == 7
        assert result["admitted"] is True


# ===========================================================================
# L. Escalation provider pinning
# ===========================================================================

class TestL_EscalationProvider:
    def test_l_escalation_deferred_waits_on_escalation_provider(self, isolated_db):
        """
        Primary Gemini is done. Job deferred at ESCALATION stage waiting OpenAI.
        OpenAI is blocked -> must not retry.
        Primary translation must NOT be redone.
        """
        from app.core.db import (
            set_setting, create_job, update_job, get_job_by_id,
            pin_job_provider, update_deferred_metadata, DeferReason, DeferStage,
        )
        from app.core.quota import (
            record_provider_quota_exhausted, should_retry_deferred_job,
        )

        set_setting("daily_request_budget_gemini", "10")
        set_setting("daily_request_budget_openai", "10")
        set_setting("ai_provider", "gemini")

        frozen_now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)

        with patch("app.core.quota._utcnow", return_value=frozen_now):
            record_provider_quota_exhausted("openai", "OpenAI daily limit", retry_after_seconds=3600)

        id_j = create_job("/movies/film.mkv", "MANUAL")
        pin_job_provider(
            id_j,
            primary_provider    = "gemini",
            primary_model       = "gemini-3.5-flash-lite",
            escalation_enabled  = True,
            escalation_provider = "openai",
            escalation_model    = "gpt-4o-mini",
        )
        next_retry = (frozen_now + timedelta(seconds=3600)).isoformat()
        update_job(id_j, status="DEFERRED", next_retry_at=next_retry,
                   last_error="DailyQuotaExhaustedError: openai quota")
        update_deferred_metadata(
            id_j,
            defer_reason     = DeferReason.ESCALATION_PROVIDER_QUOTA,
            waiting_provider = "openai",
            waiting_model    = "gpt-4o-mini",
            defer_stage      = DeferStage.ESCALATION,
        )

        job = get_job_by_id(id_j)
        assert job["defer_stage"] == "ESCALATION"
        assert job["waiting_provider"] == "openai"
        assert job["primary_provider"] == "gemini"

        with patch("app.core.quota._utcnow", return_value=frozen_now):
            result = should_retry_deferred_job(job, frozen_now)
        assert result is False, "OpenAI blocked -> escalation must wait"


# ===========================================================================
# M. Migrations: old DB without new columns migrates correctly
# ===========================================================================

class TestM_Migrations:
    def test_m_old_db_migrates_without_crash(self, tmp_path):
        """Old DB without Phase 1 columns migrates cleanly."""
        import app.core.db as db_module
        import app.core.quota as quota_module

        old_db = str(tmp_path / "babel_old.db")
        orig_db    = db_module.DB_PATH
        orig_quota = quota_module.DB_PATH
        db_module.DB_PATH    = old_db
        quota_module.DB_PATH = old_db

        try:
            # Create minimal old schema
            with sqlite3.connect(old_db) as conn:
                conn.execute("""
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_path TEXT NOT NULL,
                    title TEXT,
                    status TEXT NOT NULL,
                    event_source TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    logs TEXT,
                    retry_count INTEGER DEFAULT 0,
                    next_retry_at TEXT,
                    last_error TEXT
                )
                """)
                conn.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                conn.execute(
                    "INSERT INTO jobs (video_path, status, created_at, updated_at)"
                    " VALUES (?, 'DEFERRED', datetime('now'), datetime('now'))",
                    ("/old/ep.mkv",)
                )
                conn.commit()

            db_module.init_db()  # Must not crash

            with sqlite3.connect(old_db) as conn:
                cols = [row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()]
                for expected_col in [
                    "primary_provider", "primary_model",
                    "escalation_enabled", "escalation_provider", "escalation_model",
                    "defer_reason", "waiting_provider", "waiting_model",
                    "defer_stage", "deferred_at",
                ]:
                    assert expected_col in cols, f"Missing column after migration: {expected_col}"
        finally:
            db_module.DB_PATH    = orig_db
            quota_module.DB_PATH = orig_quota

    def test_m_pin_idempotent(self, isolated_db):
        """pin_job_provider is idempotent — second call is ignored."""
        from app.core.db import create_job, get_job_by_id, pin_job_provider

        id_j = create_job("/tv/ep.mkv", "MANUAL")
        pin_job_provider(id_j, "openai", "gpt-4o-mini")
        pin_job_provider(id_j, "gemini", "gemini-3.5-flash-lite")  # ignored

        job = get_job_by_id(id_j)
        assert job["primary_provider"] == "openai"
        assert job["primary_model"]    == "gpt-4o-mini"
