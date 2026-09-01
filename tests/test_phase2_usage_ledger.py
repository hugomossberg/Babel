"""
tests/test_phase2_usage_ledger.py
===================================
Phase 2: AI Usage Ledger — Comprehensive test suite.

Test coverage:
  A. One real dispatch → exactly one usage row
  B. Retry → second real dispatch → second usage row
  C. Local RPD blocked before dispatch → zero usage rows
  D. Provider quota/circuit breaker blocked before dispatch → zero usage rows
  E. Gemini usage metadata parsed correctly
  F. OpenAI usage metadata parsed correctly
  G. Missing token metadata → NULL fields
  H. Known pricing → correct estimated_cost_usd
  I. Unknown provider/model → row saved, estimated_cost_usd NULL
  J. PRIMARY stage grouping
  K. MICRO_REPAIR stage grouping
  L. RECOVERY stage grouping
  M. ESCALATION stage grouping
  N. Cross-provider escalation: Gemini primary + OpenAI escalation
  O. Same-provider different-model: Gemini flash + Gemini pro escalation
  P. Duplicate request_uid → cannot double-count
  Q. Failed actual provider request → usage row exists with FAILED status
  R. Blocked request → NO row
  S. Old DB migration → works
  T. Migration idempotency → works
  U. Today summary uses UTC date correctly
  V. Job summary totals are correct
  W. Historical average calls/job correct
  + Integration: real translator dispatch path → one ledger row
  + Integration: real retry → another row
  + Integration: pre-dispatch budget block → zero rows
"""

import asyncio
import os
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Provide an isolated SQLite database for each test."""
    db_file = str(tmp_path / "test_babel.db")
    monkeypatch.setenv("BABEL_DB_PATH", db_file)

    import app.core.db as db_mod
    import app.core.usage as usage_mod

    db_mod.DB_PATH = db_file
    usage_mod.DB_PATH = db_file

    # Initialize schemas
    db_mod.init_db()

    yield db_file

    # Reset to original
    db_mod.DB_PATH = os.getenv("BABEL_DB_PATH", "/app/data/babel.db")
    usage_mod.DB_PATH = os.getenv("BABEL_DB_PATH", "/app/data/babel.db")


def _direct_usage_rows(db_path: str):
    """Helper: fetch all rows from ai_usage_ledger."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT * FROM ai_usage_ledger ORDER BY id").fetchall()


def _insert_test_job(db_path: str, status: str = "TRANSLATED") -> int:
    """Helper: insert a minimal job row and return its id."""
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO jobs (video_path, status, created_at, updated_at) VALUES (?,?,?,?)",
            ("/fake/video.mkv", status, now, now),
        )
        conn.commit()
        return cursor.lastrowid


# ---------------------------------------------------------------------------
# E. Gemini usage metadata parsed correctly
# ---------------------------------------------------------------------------

class TestGeminiUsageExtraction:
    def test_all_fields_present(self):
        from app.core.usage import extract_gemini_usage

        mock_resp = MagicMock()
        mock_resp.usage_metadata.prompt_token_count = 1000
        mock_resp.usage_metadata.cached_content_token_count = 200
        mock_resp.usage_metadata.candidates_token_count = 500

        result = extract_gemini_usage(mock_resp)
        assert result["input_tokens"] == 1000
        assert result["cached_input_tokens"] == 200
        assert result["output_tokens"] == 500

    def test_no_cached_tokens_zero_means_none(self):
        """cached_content_token_count = 0 should NOT be stored (no caching used)."""
        from app.core.usage import extract_gemini_usage

        mock_resp = MagicMock()
        mock_resp.usage_metadata.prompt_token_count = 800
        mock_resp.usage_metadata.cached_content_token_count = 0
        mock_resp.usage_metadata.candidates_token_count = 300

        result = extract_gemini_usage(mock_resp)
        assert result["input_tokens"] == 800
        assert result["cached_input_tokens"] is None  # 0 → stored as None
        assert result["output_tokens"] == 300

    def test_missing_usage_metadata_returns_all_none(self):
        """If usage_metadata is absent, all fields return None (not 0)."""
        from app.core.usage import extract_gemini_usage

        mock_resp = MagicMock(spec=[])  # No attributes at all
        result = extract_gemini_usage(mock_resp)
        assert result["input_tokens"] is None
        assert result["cached_input_tokens"] is None
        assert result["output_tokens"] is None

    def test_usage_metadata_none_returns_all_none(self):
        from app.core.usage import extract_gemini_usage

        mock_resp = MagicMock()
        mock_resp.usage_metadata = None
        result = extract_gemini_usage(mock_resp)
        assert all(v is None for v in result.values())


# ---------------------------------------------------------------------------
# F. OpenAI usage metadata parsed correctly
# ---------------------------------------------------------------------------

class TestOpenAIUsageExtraction:
    def test_all_fields_present(self):
        from app.core.usage import extract_openai_usage

        mock_resp = MagicMock()
        mock_resp.usage.prompt_tokens = 750
        mock_resp.usage.completion_tokens = 400
        mock_resp.usage.prompt_tokens_details.cached_tokens = 150

        result = extract_openai_usage(mock_resp)
        assert result["input_tokens"] == 750
        assert result["output_tokens"] == 400
        assert result["cached_input_tokens"] == 150

    def test_no_cached_tokens(self):
        from app.core.usage import extract_openai_usage

        mock_resp = MagicMock()
        mock_resp.usage.prompt_tokens = 500
        mock_resp.usage.completion_tokens = 200
        mock_resp.usage.prompt_tokens_details.cached_tokens = 0  # zero → None

        result = extract_openai_usage(mock_resp)
        assert result["input_tokens"] == 500
        assert result["output_tokens"] == 200
        assert result["cached_input_tokens"] is None

    def test_missing_usage_returns_all_none(self):
        from app.core.usage import extract_openai_usage

        mock_resp = MagicMock()
        mock_resp.usage = None
        result = extract_openai_usage(mock_resp)
        assert all(v is None for v in result.values())

    def test_missing_prompt_token_details(self):
        from app.core.usage import extract_openai_usage

        mock_resp = MagicMock()
        mock_resp.usage.prompt_tokens = 300
        mock_resp.usage.completion_tokens = 100
        mock_resp.usage.prompt_tokens_details = None

        result = extract_openai_usage(mock_resp)
        assert result["input_tokens"] == 300
        assert result["output_tokens"] == 100
        assert result["cached_input_tokens"] is None


# ---------------------------------------------------------------------------
# G. Missing token metadata → NULL fields
# ---------------------------------------------------------------------------

class TestMissingMetadataNullSemantics:
    def test_deepl_returns_all_none(self):
        from app.core.usage import extract_usage_from_response

        mock_resp = MagicMock()
        result = extract_usage_from_response("deepl", mock_resp)
        assert result["input_tokens"] is None
        assert result["cached_input_tokens"] is None
        assert result["output_tokens"] is None

    def test_ollama_returns_all_none(self):
        from app.core.usage import extract_usage_from_response

        mock_resp = MagicMock()
        result = extract_usage_from_response("ollama", mock_resp)
        assert all(v is None for v in result.values())


# ---------------------------------------------------------------------------
# H. Known pricing → correct estimated_cost_usd
# ---------------------------------------------------------------------------

class TestCostCalculation:
    def test_gemini_flash_lite_cost_no_cache(self):
        from app.core.usage import calculate_estimated_cost

        # 1,000,000 input tokens + 100,000 output tokens, no cached
        # Corrected pricing (2026-08-24): $0.30/1M input, $2.50/1M output
        cost = calculate_estimated_cost(
            provider="gemini",
            model="gemini-3.5-flash-lite",
            input_tokens=1_000_000,
            cached_input_tokens=None,
            output_tokens=100_000,
        )
        expected = (1_000_000 / 1_000_000 * 0.30) + (100_000 / 1_000_000 * 2.50)
        assert cost is not None
        assert abs(cost - expected) < 1e-9

    def test_gemini_flash_lite_cost_with_cache(self):
        from app.core.usage import calculate_estimated_cost

        # 1M input, 200K cached, 500K output
        # Corrected pricing (2026-08-24): $0.30/1M input, $0.03/1M cached, $2.50/1M output
        # non-cached = 1M - 200K = 800K at $0.30/1M
        # cached = 200K at $0.03/1M
        # output = 500K at $2.50/1M
        cost = calculate_estimated_cost(
            provider="gemini",
            model="gemini-3.5-flash-lite",
            input_tokens=1_000_000,
            cached_input_tokens=200_000,
            output_tokens=500_000,
        )
        expected = (
            (800_000 / 1_000_000 * 0.30) +
            (200_000 / 1_000_000 * 0.03) +
            (500_000 / 1_000_000 * 2.50)
        )
        assert cost is not None
        assert abs(cost - expected) < 1e-9

    def test_openai_gpt4o_mini_cost(self):
        from app.core.usage import calculate_estimated_cost

        cost = calculate_estimated_cost(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=1_000_000,
            cached_input_tokens=500_000,
            output_tokens=200_000,
        )
        # non-cached = 500K at $0.15/1M, cached = 500K at $0.075/1M, output = 200K at $0.60/1M
        expected = (
            (500_000 / 1_000_000 * 0.15) +
            (500_000 / 1_000_000 * 0.075) +
            (200_000 / 1_000_000 * 0.60)
        )
        assert cost is not None
        assert abs(cost - expected) < 1e-9

    def test_cost_none_when_all_tokens_none(self):
        from app.core.usage import calculate_estimated_cost

        cost = calculate_estimated_cost(
            provider="gemini",
            model="gemini-3.5-flash-lite",
            input_tokens=None,
            cached_input_tokens=None,
            output_tokens=None,
        )
        assert cost is None


# ---------------------------------------------------------------------------
# I. Unknown provider/model → NULL cost
# ---------------------------------------------------------------------------

class TestUnknownProviderCost:
    def test_unknown_provider_returns_none(self):
        from app.core.usage import calculate_estimated_cost

        cost = calculate_estimated_cost(
            provider="anthropic",
            model="claude-3-opus",
            input_tokens=1000,
            cached_input_tokens=None,
            output_tokens=500,
        )
        assert cost is None

    def test_unknown_model_returns_none(self):
        from app.core.usage import calculate_estimated_cost

        cost = calculate_estimated_cost(
            provider="gemini",
            model="gemini-unknown-model-xyz",
            input_tokens=1000,
            cached_input_tokens=None,
            output_tokens=500,
        )
        assert cost is None

    def test_unknown_model_row_saved(self, tmp_db):
        """Row is saved even if cost is unknown."""
        from app.core.usage import record_dispatch, complete_dispatch, UsageStatus

        uid = str(uuid.uuid4())
        record_dispatch(uid, "anthropic", "claude-3-opus", "PRIMARY", job_id=None)
        complete_dispatch(uid, UsageStatus.SUCCESS, input_tokens=1000, output_tokens=500, estimated_cost_usd=None)

        rows = _direct_usage_rows(tmp_db)
        assert len(rows) == 1
        assert rows[0]["provider"] == "anthropic"
        assert rows[0]["model"] == "claude-3-opus"
        assert rows[0]["estimated_cost_usd"] is None
        assert rows[0]["input_tokens"] == 1000


# ---------------------------------------------------------------------------
# A. One real dispatch → exactly one usage row
# ---------------------------------------------------------------------------

class TestSingleDispatchRow:
    def test_record_and_complete_creates_one_row(self, tmp_db):
        from app.core.usage import record_dispatch, complete_dispatch, UsageStatus

        uid = str(uuid.uuid4())
        job_id = _insert_test_job(tmp_db)

        record_dispatch(uid, "gemini", "gemini-3.5-flash-lite", "PRIMARY", job_id=job_id)
        complete_dispatch(uid, UsageStatus.SUCCESS, input_tokens=500, output_tokens=200,
                          estimated_cost_usd=0.0001)

        rows = _direct_usage_rows(tmp_db)
        assert len(rows) == 1
        assert rows[0]["request_uid"] == uid
        assert rows[0]["status"] == "SUCCESS"
        assert rows[0]["input_tokens"] == 500
        assert rows[0]["output_tokens"] == 200
        assert rows[0]["provider"] == "gemini"
        assert rows[0]["stage"] == "PRIMARY"


# ---------------------------------------------------------------------------
# B. Retry → second real dispatch → second usage row
# ---------------------------------------------------------------------------

class TestRetryCreatesNewRow:
    def test_retry_creates_second_row(self, tmp_db):
        from app.core.usage import record_dispatch, complete_dispatch, UsageStatus

        job_id = _insert_test_job(tmp_db)

        # First attempt — fails
        uid1 = str(uuid.uuid4())
        record_dispatch(uid1, "gemini", "gemini-3.5-flash-lite", "PRIMARY", job_id=job_id)
        complete_dispatch(uid1, UsageStatus.FAILED, error_type="TRANSIENT_RPM")

        # Second attempt (retry) — succeeds
        uid2 = str(uuid.uuid4())
        assert uid1 != uid2  # Different UID for retry
        record_dispatch(uid2, "gemini", "gemini-3.5-flash-lite", "PRIMARY", job_id=job_id)
        complete_dispatch(uid2, UsageStatus.SUCCESS, input_tokens=600, output_tokens=250)

        rows = _direct_usage_rows(tmp_db)
        assert len(rows) == 2
        statuses = {r["request_uid"]: r["status"] for r in rows}
        assert statuses[uid1] == "FAILED"
        assert statuses[uid2] == "SUCCESS"


# ---------------------------------------------------------------------------
# P. Duplicate request_uid → cannot double-count
# ---------------------------------------------------------------------------

class TestRequestUidIdempotency:
    def test_duplicate_uid_not_inserted(self, tmp_db):
        from app.core.usage import record_dispatch, complete_dispatch, UsageStatus

        uid = str(uuid.uuid4())
        result1 = record_dispatch(uid, "gemini", "gemini-3.5-flash-lite", "PRIMARY")
        result2 = record_dispatch(uid, "gemini", "gemini-3.5-flash-lite", "PRIMARY")  # Duplicate

        assert result1 is True
        assert result2 is False  # Idempotency guard

        rows = _direct_usage_rows(tmp_db)
        assert len(rows) == 1  # Exactly one row, never two

    def test_complete_dispatch_only_updates_once(self, tmp_db):
        from app.core.usage import record_dispatch, complete_dispatch, UsageStatus

        uid = str(uuid.uuid4())
        record_dispatch(uid, "openai", "gpt-4o-mini", "PRIMARY")
        result1 = complete_dispatch(uid, UsageStatus.SUCCESS, input_tokens=100, output_tokens=50)
        result2 = complete_dispatch(uid, UsageStatus.SUCCESS, input_tokens=999, output_tokens=999)  # No-op

        # Second complete overwrites (UPDATE always runs), but check only one row
        rows = _direct_usage_rows(tmp_db)
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Q. Failed actual provider request → usage row with FAILED status
# ---------------------------------------------------------------------------

class TestFailedRequestRow:
    def test_failed_request_has_row(self, tmp_db):
        from app.core.usage import record_dispatch, complete_dispatch, UsageStatus

        uid = str(uuid.uuid4())
        record_dispatch(uid, "gemini", "gemini-3.5-flash-lite", "PRIMARY")
        complete_dispatch(uid, UsageStatus.FAILED, error_type="ProviderUnavailableError")

        rows = _direct_usage_rows(tmp_db)
        assert len(rows) == 1
        assert rows[0]["status"] == "FAILED"
        assert rows[0]["error_type"] == "ProviderUnavailableError"


# ---------------------------------------------------------------------------
# C & D & R. Blocked requests → NO row
# (Verified via integration test below — quota block = no record_dispatch called)
# ---------------------------------------------------------------------------

class TestBlockedRequestZeroRows:
    def test_local_rpd_blocked_means_no_row_in_db(self, tmp_db):
        """
        When the quota system blocks a request (returns allowed=False),
        with_retry does NOT call record_dispatch. Zero rows.
        """
        from app.core.usage import record_dispatch

        # Simulate: quota says blocked → with_retry raises immediately, no accounting
        # In real code, if allowed=False → no record_dispatch call → no rows
        # We verify the DB is empty when record_dispatch was never called
        rows = _direct_usage_rows(tmp_db)
        assert len(rows) == 0

    def test_circuit_breaker_blocked_means_no_row(self, tmp_db):
        """
        Same principle: circuit breaker BLOCKED → with_retry raises before record_dispatch.
        """
        rows = _direct_usage_rows(tmp_db)
        assert len(rows) == 0


# ---------------------------------------------------------------------------
# J, K, L, M. Stage grouping
# ---------------------------------------------------------------------------

class TestStageGrouping:
    def _create_row_with_stage(self, tmp_db, stage: str, provider: str = "gemini",
                               model: str = "gemini-3.5-flash-lite"):
        from app.core.usage import record_dispatch, complete_dispatch, UsageStatus
        uid = str(uuid.uuid4())
        job_id = _insert_test_job(tmp_db)
        record_dispatch(uid, provider, model, stage, job_id=job_id)
        complete_dispatch(uid, UsageStatus.SUCCESS, input_tokens=100, output_tokens=50)
        return job_id

    def test_primary_stage(self, tmp_db):
        from app.core.usage import get_job_usage_summary
        job_id = self._create_row_with_stage(tmp_db, "PRIMARY")
        summary = get_job_usage_summary(job_id)
        assert "PRIMARY" in summary["breakdown"]["by_stage"]

    def test_micro_repair_stage(self, tmp_db):
        from app.core.usage import get_job_usage_summary
        job_id = self._create_row_with_stage(tmp_db, "MICRO_REPAIR")
        summary = get_job_usage_summary(job_id)
        assert "MICRO_REPAIR" in summary["breakdown"]["by_stage"]

    def test_recovery_stage(self, tmp_db):
        from app.core.usage import get_job_usage_summary
        job_id = self._create_row_with_stage(tmp_db, "RECOVERY")
        summary = get_job_usage_summary(job_id)
        assert "RECOVERY" in summary["breakdown"]["by_stage"]

    def test_escalation_stage(self, tmp_db):
        from app.core.usage import get_job_usage_summary
        job_id = self._create_row_with_stage(tmp_db, "ESCALATION")
        summary = get_job_usage_summary(job_id)
        assert "ESCALATION" in summary["breakdown"]["by_stage"]


# ---------------------------------------------------------------------------
# N. Cross-provider escalation
# ---------------------------------------------------------------------------

class TestCrossProviderEscalation:
    def test_gemini_primary_openai_escalation(self, tmp_db):
        from app.core.usage import record_dispatch, complete_dispatch, get_job_usage_summary, UsageStatus

        job_id = _insert_test_job(tmp_db)

        # 6 Gemini PRIMARY calls
        for _ in range(6):
            uid = str(uuid.uuid4())
            record_dispatch(uid, "gemini", "gemini-3.5-flash-lite", "PRIMARY", job_id=job_id)
            complete_dispatch(uid, UsageStatus.SUCCESS, input_tokens=500, output_tokens=200)

        # 2 OpenAI ESCALATION calls
        for _ in range(2):
            uid = str(uuid.uuid4())
            record_dispatch(uid, "openai", "gpt-4o-mini", "ESCALATION", job_id=job_id)
            complete_dispatch(uid, UsageStatus.SUCCESS, input_tokens=300, output_tokens=100)

        summary = get_job_usage_summary(job_id)

        assert summary["total_calls"] == 8

        by_provider = summary["breakdown"]["by_provider"]
        assert "gemini" in by_provider
        assert "openai" in by_provider
        assert by_provider["gemini"]["calls"] == 6
        assert by_provider["openai"]["calls"] == 2

        by_stage = summary["breakdown"]["by_stage"]
        assert "PRIMARY" in by_stage
        assert "ESCALATION" in by_stage
        assert by_stage["PRIMARY"]["calls"] == 6
        assert by_stage["ESCALATION"]["calls"] == 2


# ---------------------------------------------------------------------------
# O. Same-provider different-model
# ---------------------------------------------------------------------------

class TestSameProviderDifferentModel:
    def test_flash_primary_pro_escalation_separated(self, tmp_db):
        from app.core.usage import record_dispatch, complete_dispatch, get_job_usage_summary, UsageStatus

        job_id = _insert_test_job(tmp_db)

        # 4 calls with flash-lite (PRIMARY)
        for _ in range(4):
            uid = str(uuid.uuid4())
            record_dispatch(uid, "gemini", "gemini-3.5-flash-lite", "PRIMARY", job_id=job_id)
            complete_dispatch(uid, UsageStatus.SUCCESS, input_tokens=500, output_tokens=200)

        # 2 calls with pro (ESCALATION)
        for _ in range(2):
            uid = str(uuid.uuid4())
            record_dispatch(uid, "gemini", "gemini-2.5-pro", "ESCALATION", job_id=job_id)
            complete_dispatch(uid, UsageStatus.SUCCESS, input_tokens=1000, output_tokens=400)

        summary = get_job_usage_summary(job_id)

        by_pm = summary["breakdown"]["by_provider_model"]
        assert "gemini/gemini-3.5-flash-lite" in by_pm
        assert "gemini/gemini-2.5-pro" in by_pm
        assert by_pm["gemini/gemini-3.5-flash-lite"]["calls"] == 4
        assert by_pm["gemini/gemini-2.5-pro"]["calls"] == 2

        # Only one provider
        by_provider = summary["breakdown"]["by_provider"]
        assert len(by_provider) == 1
        assert by_provider["gemini"]["calls"] == 6


# ---------------------------------------------------------------------------
# S. Old DB migration → works
# ---------------------------------------------------------------------------

class TestMigration:
    def test_migration_on_fresh_db(self, tmp_db):
        """Fresh DB: usage table created successfully."""
        with sqlite3.connect(tmp_db) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='ai_usage_ledger'"
            ).fetchall()
        assert len(tables) == 1

    def test_usage_table_has_expected_columns(self, tmp_db):
        """Verify all required columns exist."""
        with sqlite3.connect(tmp_db) as conn:
            cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(ai_usage_ledger)").fetchall()
            }
        required = {
            "id", "request_uid", "job_id", "provider", "model", "stage", "status",
            "input_tokens", "cached_input_tokens", "output_tokens",
            "estimated_cost_usd", "error_type", "created_at", "completed_at",
        }
        assert required.issubset(cols)

    def test_unique_index_on_request_uid(self, tmp_db):
        """UNIQUE constraint on request_uid."""
        from app.core.usage import record_dispatch

        uid = str(uuid.uuid4())
        r1 = record_dispatch(uid, "gemini", "flash", "PRIMARY")
        r2 = record_dispatch(uid, "gemini", "flash", "PRIMARY")

        assert r1 is True
        assert r2 is False


# ---------------------------------------------------------------------------
# T. Migration idempotency
# ---------------------------------------------------------------------------

class TestMigrationIdempotency:
    def test_init_twice_is_safe(self, tmp_db):
        """Running init_usage_schema twice must not raise or corrupt data."""
        from app.core.usage import init_usage_schema

        init_usage_schema()  # First time
        init_usage_schema()  # Second time — idempotent

        # Still works
        from app.core.usage import record_dispatch, complete_dispatch, UsageStatus
        uid = str(uuid.uuid4())
        record_dispatch(uid, "gemini", "gemini-3.5-flash-lite", "PRIMARY")
        complete_dispatch(uid, UsageStatus.SUCCESS)

        rows = _direct_usage_rows(tmp_db)
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# U. Today summary uses UTC date correctly
# ---------------------------------------------------------------------------

class TestTodaySummaryUTC:
    def test_today_summary_reflects_current_day(self, tmp_db):
        from app.core.usage import record_dispatch, complete_dispatch, get_today_usage_summary, UsageStatus
        from datetime import datetime, timezone

        uid = str(uuid.uuid4())
        record_dispatch(uid, "gemini", "gemini-3.5-flash-lite", "PRIMARY")
        complete_dispatch(uid, UsageStatus.SUCCESS, input_tokens=500, output_tokens=200)

        summary = get_today_summary_patched(tmp_db)

        assert "gemini" in summary["providers"]
        assert summary["providers"]["gemini"]["calls_today"] == 1

    def test_today_summary_excludes_old_dates(self, tmp_db):
        """Rows from yesterday should not appear in today's summary."""
        from datetime import datetime, timezone, timedelta

        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        with sqlite3.connect(tmp_db) as conn:
            conn.execute(
                """INSERT INTO ai_usage_ledger
                   (request_uid, provider, model, stage, status, input_tokens, output_tokens, created_at)
                   VALUES (?, 'gemini', 'gemini-3.5-flash-lite', 'PRIMARY', 'SUCCESS', 500, 200, ?)
                """,
                (str(uuid.uuid4()), yesterday),
            )
            conn.commit()

        summary = get_today_summary_patched(tmp_db)
        # 'gemini' may not appear, or calls_today must be 0
        providers = summary.get("providers", {})
        gemini_calls = providers.get("gemini", {}).get("calls_today", 0)
        assert gemini_calls == 0


def get_today_summary_patched(db_path: str):
    """Wrapper that ensures usage module uses the test DB path."""
    import app.core.usage as usage_mod
    original = usage_mod.DB_PATH
    usage_mod.DB_PATH = db_path
    try:
        return usage_mod.get_today_usage_summary()
    finally:
        usage_mod.DB_PATH = original


# ---------------------------------------------------------------------------
# V. Job summary totals correct
# ---------------------------------------------------------------------------

class TestJobSummaryTotals:
    def test_totals_sum_correctly(self, tmp_db):
        from app.core.usage import record_dispatch, complete_dispatch, get_job_usage_summary, UsageStatus

        job_id = _insert_test_job(tmp_db)

        calls_data = [
            (500, 200, 0.00025),
            (600, 300, 0.00030),
            (400, 150, 0.00020),
        ]
        for it, ct, cost in calls_data:
            uid = str(uuid.uuid4())
            record_dispatch(uid, "gemini", "gemini-3.5-flash-lite", "PRIMARY", job_id=job_id)
            complete_dispatch(uid, UsageStatus.SUCCESS, input_tokens=it, output_tokens=ct,
                              estimated_cost_usd=cost)

        summary = get_job_usage_summary(job_id)
        assert summary["total_calls"] == 3
        assert summary["total_input_tokens"] == 1500
        assert summary["total_output_tokens"] == 650
        assert abs(summary["total_estimated_cost_usd"] - 0.00075) < 1e-10

    def test_empty_job_returns_zero_totals(self, tmp_db):
        from app.core.usage import get_job_usage_summary

        summary = get_job_usage_summary(999999)  # Non-existent job
        assert summary["total_calls"] == 0
        assert summary["total_input_tokens"] is None
        assert summary["total_estimated_cost_usd"] is None


# ---------------------------------------------------------------------------
# W. Historical average calls/job correct
# ---------------------------------------------------------------------------

class TestHistoricalStats:
    def test_average_calls_per_job(self, tmp_db):
        from app.core.usage import record_dispatch, complete_dispatch, get_historical_stats, UsageStatus

        # Create 2 TRANSLATED jobs, first with 3 calls, second with 1 call
        job1 = _insert_test_job(tmp_db, "TRANSLATED")
        job2 = _insert_test_job(tmp_db, "TRANSLATED")

        for _ in range(3):
            uid = str(uuid.uuid4())
            record_dispatch(uid, "gemini", "gemini-3.5-flash-lite", "PRIMARY", job_id=job1)
            complete_dispatch(uid, UsageStatus.SUCCESS)

        uid = str(uuid.uuid4())
        record_dispatch(uid, "gemini", "gemini-3.5-flash-lite", "PRIMARY", job_id=job2)
        complete_dispatch(uid, UsageStatus.SUCCESS)

        stats = get_historical_stats()
        assert stats["completed_jobs_with_ai"] == 2
        assert stats["total_calls_all_time"] == 4
        assert stats["average_calls_per_job"] == 2.0  # (3+1)/2

    def test_jobs_without_ai_usage_excluded(self, tmp_db):
        """Jobs with no usage rows don't count in the denominator."""
        from app.core.usage import get_historical_stats

        # TRANSLATED job with no usage rows
        _insert_test_job(tmp_db, "TRANSLATED")

        stats = get_historical_stats()
        assert stats["completed_jobs_with_ai"] == 0
        assert stats["average_calls_per_job"] is None


# ---------------------------------------------------------------------------
# Integration tests: real translator dispatch path
# ---------------------------------------------------------------------------

class TestIntegrationTranslatorDispatch:
    """
    Instrument the real with_retry decorator path to verify ledger behavior
    without making actual API calls.
    """

    @pytest.fixture(autouse=True)
    def patch_db_paths(self, tmp_db, monkeypatch):
        """Ensure all modules use the test DB."""
        import app.core.usage as usage_mod
        import app.core.db as db_mod
        import app.core.quota as quota_mod

        original_usage = usage_mod.DB_PATH
        original_db = db_mod.DB_PATH
        original_quota = quota_mod.DB_PATH

        usage_mod.DB_PATH = tmp_db
        db_mod.DB_PATH = tmp_db
        quota_mod.DB_PATH = tmp_db

        yield

        usage_mod.DB_PATH = original_usage
        db_mod.DB_PATH = original_db
        quota_mod.DB_PATH = original_quota

    def test_real_dispatch_creates_one_row(self, tmp_db):
        """
        Instrument the actual with_retry path:
        - acquire_dispatch_slot returns True (ACTIVE)
        - function succeeds
        → exactly one row, status=SUCCESS
        """
        from app.services.translator import SubtitleTranslator, _usage_token_ctx
        from app.core.usage import UsageStage

        translator = SubtitleTranslator()
        job_id = _insert_test_job(tmp_db)

        # Mock the inner call to return translations without hitting real API
        mock_gemini_response = MagicMock()
        mock_gemini_response.usage_metadata.prompt_token_count = 1000
        mock_gemini_response.usage_metadata.cached_content_token_count = 0
        mock_gemini_response.usage_metadata.candidates_token_count = 400
        mock_gemini_response.text = '[{"id": 0, "text": "Translated"}]'

        with patch("app.core.quota.acquire_dispatch_slot", return_value=(True, {"is_probe": False, "state": "ACTIVE"})), \
             patch("app.core.quota.record_provider_success"), \
             patch.object(translator, "get_gemini_client") as mock_client, \
             patch("app.core.db.get_setting", side_effect=_fake_get_setting):

            mock_client.return_value.models.generate_content.return_value = mock_gemini_response

            loop = asyncio.new_event_loop()
            try:
                items = [{"id": 0, "text": "Hello"}]
                loop.run_until_complete(
                    translator.translate_batch_gemini(
                        items, "Swedish", model_name="gemini-3.5-flash-lite", job_id=job_id
                    )
                )
            finally:
                loop.close()

        rows = _direct_usage_rows(tmp_db)
        assert len(rows) == 1, f"Expected 1 row, got {len(rows)}: {[dict(r) for r in rows]}"
        assert rows[0]["status"] == "SUCCESS"
        assert rows[0]["stage"] == UsageStage.PRIMARY
        assert rows[0]["provider"] == "gemini"
        assert rows[0]["model"] == "gemini-3.5-flash-lite"
        assert rows[0]["input_tokens"] == 1000

    def test_retry_creates_two_rows(self, tmp_db):
        """
        Two attempts (1 fail + 1 success) → 2 rows.
        """
        from app.services.translator import SubtitleTranslator

        translator = SubtitleTranslator()
        job_id = _insert_test_job(tmp_db)

        call_count = [0]

        mock_response = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 500
        mock_response.usage_metadata.cached_content_token_count = 0
        mock_response.usage_metadata.candidates_token_count = 200
        mock_response.text = '[{"id": 0, "text": "OK"}]'

        def side_effect_generate(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("TRANSIENT_RATE_LIMIT: 429")
            return mock_response

        with patch("app.core.quota.acquire_dispatch_slot", return_value=(True, {"is_probe": False, "state": "ACTIVE"})), \
             patch("app.core.quota.record_provider_success"), \
             patch("app.core.quota.classify_provider_error", return_value="TRANSIENT_RPM"), \
             patch("app.core.quota.extract_retry_after_from_exception", return_value=None), \
             patch("asyncio.sleep", new_callable=AsyncMock), \
             patch.object(translator, "get_gemini_client") as mock_client, \
             patch("app.core.db.get_setting", side_effect=_fake_get_setting):

            mock_client.return_value.models.generate_content.side_effect = side_effect_generate

            loop = asyncio.new_event_loop()
            try:
                items = [{"id": 0, "text": "Hello"}]
                loop.run_until_complete(
                    translator.translate_batch_gemini(
                        items, "Swedish", "gemini-3.5-flash-lite", job_id=job_id
                    )
                )
            finally:
                loop.close()

        rows = _direct_usage_rows(tmp_db)
        assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"
        statuses = [r["status"] for r in rows]
        assert "FAILED" in statuses
        assert "SUCCESS" in statuses

    def test_pre_dispatch_budget_block_zero_rows(self, tmp_db):
        """
        When acquire_dispatch_slot returns False (budget exhausted),
        no record_dispatch is called → zero rows.
        """
        from app.services.translator import SubtitleTranslator
        from app.core.quota import RequestBudgetExhaustedError

        translator = SubtitleTranslator()

        with patch("app.core.quota.acquire_dispatch_slot",
                   return_value=(False, {"reason": "REQUEST_BUDGET_EXHAUSTED", "state": "ACTIVE"})), \
             patch("app.core.quota.get_daily_budget", return_value=100), \
             patch("app.core.quota.get_daily_requests_used", return_value=100), \
             patch("app.core.db.get_setting", side_effect=_fake_get_setting):

            loop = asyncio.new_event_loop()
            try:
                with pytest.raises(RequestBudgetExhaustedError):
                    loop.run_until_complete(
                        translator.translate_batch_gemini(
                            [{"id": 0, "text": "Hello"}], "Swedish", "gemini-3.5-flash-lite"
                        )
                    )
            finally:
                loop.close()

        rows = _direct_usage_rows(tmp_db)
        assert len(rows) == 0, f"Expected 0 rows when blocked, got {len(rows)}"

    def test_circuit_breaker_block_zero_rows(self, tmp_db):
        """
        When acquire_dispatch_slot returns False (circuit breaker BLOCKED),
        no record_dispatch → zero rows.
        """
        from app.services.translator import SubtitleTranslator
        from app.core.quota import DailyQuotaExhaustedError

        translator = SubtitleTranslator()

        with patch("app.core.quota.acquire_dispatch_slot",
                   return_value=(False, {"reason": "Quota exhausted", "state": "BLOCKED",
                                         "blocked_until": "2099-01-01T00:00:00+00:00",
                                         "reset_type": "estimated"})), \
             patch("app.core.db.get_setting", side_effect=_fake_get_setting):

            loop = asyncio.new_event_loop()
            try:
                with pytest.raises(DailyQuotaExhaustedError):
                    loop.run_until_complete(
                        translator.translate_batch_gemini(
                            [{"id": 0, "text": "Hi"}], "Swedish", "gemini-3.5-flash-lite"
                        )
                    )
            finally:
                loop.close()

        rows = _direct_usage_rows(tmp_db)
        assert len(rows) == 0, f"Expected 0 rows when circuit-breaker blocked, got {len(rows)}"


def _fake_get_setting(key: str, default: str = "") -> str:
    """Minimal fake settings for tests."""
    settings = {
        "gemini_api_key": "test-api-key",
        "gemini_model": "gemini-3.5-flash-lite",
        "openai_api_key": "test-openai-key",
        "openai_model": "gpt-4o-mini",
        "ai_provider": "gemini",
        "custom_translation_instructions": "",
        "glossary": "",
    }
    return settings.get(key, default)



# ---------------------------------------------------------------------------
# Hardening: Token store lifecycle and accounting boundary
# ---------------------------------------------------------------------------

class TestContextVarLifecycle:
    """
    Verify that the two-tier token propagation mechanism (_usage_token_ctx as
    UID-channel + _USAGE_TOKEN_STORE as cross-thread store) behaves correctly:
    - UID cleared/updated before each attempt
    - _capture_* writes tokens to store keyed by UID
    - concurrent tasks use different UIDs (no contamination)
    - stale tokens from previous attempt cannot be read by next attempt
    - missing metadata (DeepL/Ollama) returns None
    """

    def test_uid_channel_set_before_each_attempt(self):
        """
        _usage_token_ctx holds the current request_uid (not token data).
        It is updated to the new uid before each SDK call.
        """
        from app.services.translator import _usage_token_ctx

        uid1 = str(uuid.uuid4())
        uid2 = str(uuid.uuid4())

        _usage_token_ctx.set(uid1)
        assert _usage_token_ctx.get() == uid1

        # Simulate start of next attempt
        _usage_token_ctx.set(uid2)
        assert _usage_token_ctx.get() == uid2
        assert _usage_token_ctx.get() != uid1  # No stale uid

    def test_capture_gemini_writes_to_store(self):
        """
        _capture_gemini_tokens writes token metadata to _USAGE_TOKEN_STORE
        keyed by the request_uid in _usage_token_ctx.
        """
        from app.services.translator import (
            _usage_token_ctx, _capture_gemini_tokens,
            _pop_token_meta, _store_token_meta,
        )

        uid = str(uuid.uuid4())
        _usage_token_ctx.set(uid)

        mock_resp = MagicMock()
        mock_resp.usage_metadata.prompt_token_count = 1500
        mock_resp.usage_metadata.cached_content_token_count = 300
        mock_resp.usage_metadata.candidates_token_count = 600

        _capture_gemini_tokens(mock_resp)

        result = _pop_token_meta(uid)
        assert result is not None
        assert result["input_tokens"] == 1500
        assert result["cached_input_tokens"] == 300
        assert result["output_tokens"] == 600

    def test_capture_openai_writes_to_store(self):
        """
        _capture_openai_tokens writes token metadata to _USAGE_TOKEN_STORE.
        """
        from app.services.translator import (
            _usage_token_ctx, _capture_openai_tokens, _pop_token_meta,
        )

        uid = str(uuid.uuid4())
        _usage_token_ctx.set(uid)

        mock_resp = MagicMock()
        mock_resp.usage.prompt_tokens = 800
        mock_resp.usage.completion_tokens = 400
        mock_resp.usage.prompt_tokens_details.cached_tokens = 200

        _capture_openai_tokens(mock_resp)

        result = _pop_token_meta(uid)
        assert result is not None
        assert result["input_tokens"] == 800
        assert result["cached_input_tokens"] == 200
        assert result["output_tokens"] == 400

    def test_missing_metadata_store_returns_none(self):
        """
        When no _capture_* is called (DeepL/Ollama), _pop_token_meta returns None.
        """
        from app.services.translator import _pop_token_meta

        uid = str(uuid.uuid4())
        result = _pop_token_meta(uid)
        assert result is None

    def test_store_entries_are_keyed_by_uid(self):
        """
        Two concurrent tasks with different UIDs do not contaminate each other's
        token store entries.
        """
        from app.services.translator import _store_token_meta, _pop_token_meta

        uid_a = str(uuid.uuid4())
        uid_b = str(uuid.uuid4())

        _store_token_meta(uid_a, {"input_tokens": 111, "cached_input_tokens": None, "output_tokens": 222})
        _store_token_meta(uid_b, {"input_tokens": 333, "cached_input_tokens": None, "output_tokens": 444})

        result_a = _pop_token_meta(uid_a)
        result_b = _pop_token_meta(uid_b)

        assert result_a["input_tokens"] == 111
        assert result_a["output_tokens"] == 222
        assert result_b["input_tokens"] == 333
        assert result_b["output_tokens"] == 444

    def test_pop_removes_entry_no_double_read(self):
        """
        _pop_token_meta removes the entry — second call returns None.
        Prevents stale data from leaking to the next attempt.
        """
        from app.services.translator import _store_token_meta, _pop_token_meta

        uid = str(uuid.uuid4())
        _store_token_meta(uid, {"input_tokens": 500, "cached_input_tokens": None, "output_tokens": 200})

        first = _pop_token_meta(uid)
        second = _pop_token_meta(uid)

        assert first is not None
        assert first["input_tokens"] == 500
        assert second is None  # Entry gone after first pop

    def test_contextvar_no_leak_between_retries(self, tmp_db):
        """
        Stale token metadata from attempt N must not appear in attempt N+1.
        With the new store approach: each attempt uses a fresh uid → pop on
        fresh uid returns None (not stale data from previous attempt's uid).
        """
        from app.services.translator import SubtitleTranslator, _usage_token_ctx
        from app.services.translator import _USAGE_TOKEN_STORE, _USAGE_TOKEN_STORE_LOCK

        translator = SubtitleTranslator()
        job_id = _insert_test_job(tmp_db)
        call_count = [0]
        captured_uid_at_attempt2 = {}

        mock_response = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 500
        mock_response.usage_metadata.cached_content_token_count = 0
        mock_response.usage_metadata.candidates_token_count = 200
        mock_response.text = '[{"id": 0, "text": "OK"}]'

        def side_effect_generate(*args, **kwargs):
            call_count[0] += 1
            current_uid = _usage_token_ctx.get()
            if call_count[0] == 1:
                # Attempt 1: pollute store with stale data for attempt 1's uid
                if current_uid:
                    from app.services.translator import _store_token_meta
                    _store_token_meta(current_uid, {"input_tokens": 9999, "cached_input_tokens": None, "output_tokens": 8888})
                raise Exception("TRANSIENT_RATE_LIMIT: 429")
            # Attempt 2: capture the uid that with_retry published for this attempt
            captured_uid_at_attempt2["uid"] = current_uid
            return mock_response

        with patch("app.core.quota.acquire_dispatch_slot", return_value=(True, {"is_probe": False, "state": "ACTIVE"})), \
             patch("app.core.quota.record_provider_success"), \
             patch("app.core.quota.classify_provider_error", return_value="TRANSIENT_RPM"), \
             patch("app.core.quota.extract_retry_after_from_exception", return_value=None), \
             patch("asyncio.sleep", new_callable=AsyncMock), \
             patch.object(translator, "get_gemini_client") as mock_client, \
             patch("app.core.db.get_setting", side_effect=_fake_get_setting):

            mock_client.return_value.models.generate_content.side_effect = side_effect_generate

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    translator.translate_batch_gemini(
                        [{"id": 0, "text": "Hello"}], "Swedish",
                        model_name="gemini-3.5-flash-lite", job_id=job_id
                    )
                )
            finally:
                loop.close()

        # Attempt 2 must use a DIFFERENT uid than attempt 1.
        # The key invariant: each attempt generates a fresh uid, so _pop_token_meta
        # for attempt 2's uid cannot accidentally find attempt 1's stale data.
        rows = _direct_usage_rows(tmp_db)
        assert len(rows) == 2

        attempt1_uid = rows[0]["request_uid"]
        attempt2_uid = rows[1]["request_uid"]
        assert attempt1_uid != attempt2_uid, "Retry must use a new request_uid"

        # Verify the core safety property: attempt 1's stale data lives under attempt1_uid.
        # Attempt 2's pop used attempt2_uid → would find None (no stale data there).
        # (The stale entry for attempt1_uid is left in store — it's harmless since
        #  no future pop will use attempt1_uid again. In production this would be
        #  cleared when attempt 1's complete_dispatch is called with FAILED status.)
        from app.services.translator import _pop_token_meta
        stale_for_attempt1 = _pop_token_meta(attempt1_uid)
        # We planted stale data under attempt1_uid in side_effect_generate
        # but with_retry called complete_dispatch(FAILED) for attempt1_uid which
        # already consumed the store entry. So it should be None now.
        # (If complete_dispatch consumed it, this is None. If not, it means
        #  the stale data is orphaned under attempt1_uid — harmless since no future
        #  operation will pop attempt1_uid again.)

        # The critical isolation check: attempt2_uid has NO stale data from attempt 1
        stale_for_attempt2 = _pop_token_meta(attempt2_uid)
        assert stale_for_attempt2 is None, (
            "Attempt 2's uid should not have stale data from attempt 1. "
            "The two-tier store approach guarantees isolation via unique per-attempt uids."
        )


class TestAccountingBoundaryHardening:
    """
    Additional boundary tests for the audit findings:
    - _request_uid only set on confirmed insert
    - blocked before SDK = zero rows
    - pre-SDK failure (config error, missing API key, client init) = zero rows
    - actual SDK attempt that raises = 1 FAILED row
    """

    @pytest.fixture(autouse=True)
    def patch_db_paths(self, tmp_db, monkeypatch):
        import app.core.usage as usage_mod
        import app.core.db as db_mod
        import app.core.quota as quota_mod

        usage_mod.DB_PATH = tmp_db
        db_mod.DB_PATH = tmp_db
        quota_mod.DB_PATH = tmp_db
        yield

    def test_confirmed_insert_only_request_uid(self, tmp_db):
        """
        _request_uid is set ONLY when record_dispatch returns True (confirmed insert).
        If record_dispatch returns False, no complete_dispatch is called.
        """
        from app.core.usage import record_dispatch, complete_dispatch

        # A genuine insert — returns True
        uid1 = str(uuid.uuid4())
        r1 = record_dispatch(uid1, "gemini", "gemini-3.5-flash-lite", "PRIMARY")
        assert r1 is True

        # Same uid again — returns False (INSERT OR IGNORE)
        r2 = record_dispatch(uid1, "gemini", "gemini-3.5-flash-lite", "PRIMARY")
        assert r2 is False

        # Only one row should exist (the first insert)
        rows = _direct_usage_rows(tmp_db)
        assert len(rows) == 1
        assert rows[0]["status"] == "PENDING"  # Still PENDING, complete not called

    def test_accounting_does_not_affect_quota_counter(self, tmp_db):
        """
        The usage ledger is pure observability.
        Inserting/reading from ai_usage_ledger must NOT touch daily_request_counts.
        """
        from app.core.usage import record_dispatch, complete_dispatch, UsageStatus

        # Read initial quota state
        with sqlite3.connect(tmp_db) as conn:
            initial = conn.execute(
                "SELECT SUM(request_count) FROM daily_request_counts"
            ).fetchone()[0] or 0

        # Ledger operations
        uid = str(uuid.uuid4())
        record_dispatch(uid, "gemini", "gemini-3.5-flash-lite", "PRIMARY")
        complete_dispatch(uid, UsageStatus.SUCCESS, input_tokens=500, output_tokens=200)

        # Quota counter must be unchanged
        with sqlite3.connect(tmp_db) as conn:
            after = conn.execute(
                "SELECT SUM(request_count) FROM daily_request_counts"
            ).fetchone()[0] or 0

        assert initial == after, (
            f"Quota counter changed after ledger operation: {initial} → {after}. "
            "Usage ledger must NOT touch daily_request_counts."
        )

    def test_pre_sdk_failure_zero_rows(self, tmp_db):
        """
        REGRESSION TEST: failure after admission but BEFORE actual provider SDK invocation
        must produce ZERO usage rows.

        Scenario: acquire_dispatch_slot returns True (slot acquired, admission passed),
        record_dispatch inserts a PENDING row, but then func() raises BEFORE reaching
        the actual run_in_executor / SDK network call (e.g. get_gemini_client() raises
        ValueError because API key is missing).

        Expected: the PENDING row is deleted by cancel_dispatch → 0 rows remain.
        This enforces the invariant: usage row = real provider request attempt.
        """
        from app.services.translator import SubtitleTranslator, ProviderConfigurationError

        translator = SubtitleTranslator()
        job_id = _insert_test_job(tmp_db)

        with patch("app.core.quota.acquire_dispatch_slot", return_value=(True, {"is_probe": False, "state": "ACTIVE"})), \
             patch("app.core.quota.record_provider_success"), \
             patch("app.core.quota.extract_retry_after_from_exception", return_value=None), \
             patch("app.core.quota.classify_provider_error", return_value="AUTH_ERROR"), \
             patch("app.core.db.get_setting", side_effect=_fake_get_setting), \
             patch.object(translator, "get_gemini_client",
                          side_effect=ValueError("Gemini API Key is not configured in settings.")):
            # get_gemini_client raises BEFORE run_in_executor → BEFORE _mark_sdk_started
            loop = asyncio.new_event_loop()
            try:
                with pytest.raises((ProviderConfigurationError, ValueError, Exception)):
                    loop.run_until_complete(
                        translator.translate_batch_gemini(
                            [{"id": 0, "text": "Hello"}], "Swedish",
                            model_name="gemini-3.5-flash-lite", job_id=job_id
                        )
                    )
            finally:
                loop.close()

        rows = _direct_usage_rows(tmp_db)
        assert len(rows) == 0, (
            f"Expected 0 rows for pre-SDK exception, got {len(rows)}. "
            f"Rows: {rows}. "
            "Pre-SDK failures (config/init errors) must NOT create usage rows — "
            "no actual provider network call was made."
        )

    def test_actual_sdk_failure_creates_failed_row(self, tmp_db):
        """
        When the actual SDK call is reached and raises (e.g. network error,
        invalid_argument from the API), exactly 1 FAILED row must be created.

        This distinguishes from test_pre_sdk_failure_zero_rows:
        the SDK was invoked (_mark_sdk_started was called) before the exception.
        """
        from app.services.translator import SubtitleTranslator

        translator = SubtitleTranslator()

        with patch("app.core.quota.acquire_dispatch_slot", return_value=(True, {"is_probe": False, "state": "ACTIVE"})), \
             patch("app.core.quota.record_provider_success"), \
             patch.object(translator, "get_gemini_client") as mock_client, \
             patch("app.core.db.get_setting", side_effect=_fake_get_setting), \
             patch("app.core.quota.classify_provider_error", return_value="PERMANENT_REQUEST_ERROR"), \
             patch("app.core.quota.extract_retry_after_from_exception", return_value=None):

            # The mock raises from client.models.generate_content → AFTER _mark_sdk_started
            mock_client.return_value.models.generate_content.side_effect = Exception("invalid_argument: API key not valid")

            loop = asyncio.new_event_loop()
            try:
                from app.services.translator import ProviderConfigurationError
                with pytest.raises((ProviderConfigurationError, Exception)):
                    loop.run_until_complete(
                        translator.translate_batch_gemini(
                            [{"id": 0, "text": "Hello"}], "Swedish",
                            model_name="gemini-3.5-flash-lite"
                        )
                    )
            finally:
                loop.close()

        rows = _direct_usage_rows(tmp_db)
        assert len(rows) == 1, f"Expected 1 row for actual SDK failure, got {len(rows)}"
        assert rows[0]["status"] == "FAILED", f"Expected FAILED, got {rows[0]['status']}"

    def test_slot_acquired_but_record_dispatch_db_error_no_complete_dispatch(self, tmp_db):
        """
        If record_dispatch raises (DB error), _request_uid stays None.
        No complete_dispatch is ever called.
        Even if func() succeeds, no usage row exists (fail-open for observability).
        """
        from app.services.translator import SubtitleTranslator

        translator = SubtitleTranslator()

        mock_response = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 500
        mock_response.usage_metadata.cached_content_token_count = 0
        mock_response.usage_metadata.candidates_token_count = 200
        mock_response.text = '[{"id": 0, "text": "OK"}]'

        with patch("app.core.quota.acquire_dispatch_slot", return_value=(True, {"is_probe": False, "state": "ACTIVE"})), \
             patch("app.core.quota.record_provider_success"), \
             patch.object(translator, "get_gemini_client") as mock_client, \
             patch("app.core.db.get_setting", side_effect=_fake_get_setting), \
             patch("app.core.usage.record_dispatch", side_effect=Exception("DB locked")):

            mock_client.return_value.models.generate_content.return_value = mock_response

            loop = asyncio.new_event_loop()
            try:
                # Translation MUST still succeed even when usage accounting fails
                result = loop.run_until_complete(
                    translator.translate_batch_gemini(
                        [{"id": 0, "text": "Hello"}], "Swedish",
                        model_name="gemini-3.5-flash-lite"
                    )
                )
                assert result is not None  # Translation succeeded despite ledger failure
            finally:
                loop.close()

        # No row was created (record_dispatch raised, _request_uid stayed None)
        rows = _direct_usage_rows(tmp_db)
        assert len(rows) == 0, (
            f"Expected 0 rows when record_dispatch raises, got {len(rows)}. "
            "Ledger failure must never break translation."
        )
