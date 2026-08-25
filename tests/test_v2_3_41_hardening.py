"""
tests/test_v2_3_41_hardening.py
================================
v2.3.41-beta Production Hardening Regression Tests

Tests A-H:
  A. Webhook secret: BABEL_WEBHOOK_SECRET env always wins over stale DB value
  B. Webhook secret: absent/empty env preserves existing DB value
  C. Thinking tokens: extract_gemini_usage returns thoughts_token_count separately
  D. Thinking tokens: calculate_estimated_cost includes thinking in billable output
  E. DeepL/Ollama dispatch markers present; no 404 in-dispatch fallback
  F. Budget fail-closed: try_consume_request_budget returns False on DB error when budget > 0
  G. All-time usage: get_historical_stats total_calls_all_time includes FAILED jobs
  H. clear_all_jobs: usage ledger rows deleted together with jobs
"""

import os
import pytest
import sqlite3
from unittest.mock import patch


# ===========================================================================
# A & B. Webhook secret canonical resolution
# ===========================================================================

class TestWebhookSecretResolution:

    def test_A_env_secret_overwrites_stale_db_value(self, tmp_path):
        """BABEL_WEBHOOK_SECRET env non-empty → INSERT OR REPLACE; stale DB value overwritten."""
        import app.core.db as db_mod

        db_path = str(tmp_path / "test.db")

        # Pre-seed a stale value before init_db
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("INSERT INTO settings (key, value) VALUES ('webhook_secret', 'stale_value')")
            conn.commit()

        with patch("app.core.db.DB_PATH", db_path), \
             patch.dict(os.environ, {"BABEL_WEBHOOK_SECRET": "newenv_secret"}):
            db_mod.DB_PATH = db_path
            db_mod.init_db()

        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT value FROM settings WHERE key='webhook_secret'").fetchone()
        assert row is not None
        assert row[0] == "newenv_secret", f"Expected 'newenv_secret', got '{row[0]}'"

    def test_B_empty_env_preserves_db_value(self, tmp_path):
        """BABEL_WEBHOOK_SECRET absent/empty → existing DB value preserved."""
        import app.core.db as db_mod

        db_path = str(tmp_path / "test.db")

        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("INSERT INTO settings (key, value) VALUES ('webhook_secret', 'keep_this')")
            conn.commit()

        env_without_secret = {k: v for k, v in os.environ.items() if k != "BABEL_WEBHOOK_SECRET"}
        with patch("app.core.db.DB_PATH", db_path), \
             patch.dict(os.environ, env_without_secret, clear=True):
            db_mod.DB_PATH = db_path
            db_mod.init_db()

        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT value FROM settings WHERE key='webhook_secret'").fetchone()
        assert row is not None
        assert row[0] == "keep_this", f"DB value should be preserved, got '{row[0]}'"


# ===========================================================================
# C. Thinking tokens: extract_gemini_usage
# ===========================================================================

class TestThinkingTokenExtraction:

    def test_C_thoughts_token_count_extracted_separately(self):
        """thoughts_token_count → thinking_tokens separate from candidates_token_count."""
        from app.core.usage import extract_gemini_usage

        class FakeUsage:
            prompt_token_count = 1000
            cached_content_token_count = 0
            candidates_token_count = 300
            thoughts_token_count = 200

        class FakeResponse:
            usage_metadata = FakeUsage()

        result = extract_gemini_usage(FakeResponse())
        assert result["input_tokens"] == 1000
        assert result["output_tokens"] == 300
        assert result["thinking_tokens"] == 200
        assert result["cached_input_tokens"] is None  # 0 → None

    def test_C_no_thinking_attribute_returns_none(self):
        """No thoughts_token_count attribute → thinking_tokens = None."""
        from app.core.usage import extract_gemini_usage

        class FakeUsage:
            prompt_token_count = 500
            cached_content_token_count = None
            candidates_token_count = 100

        class FakeResponse:
            usage_metadata = FakeUsage()

        result = extract_gemini_usage(FakeResponse())
        assert result["thinking_tokens"] is None

    def test_C_zero_thoughts_returns_none(self):
        """thoughts_token_count=0 → thinking_tokens=None (same as cached=0 convention)."""
        from app.core.usage import extract_gemini_usage

        class FakeUsage:
            prompt_token_count = 500
            cached_content_token_count = None
            candidates_token_count = 100
            thoughts_token_count = 0

        class FakeResponse:
            usage_metadata = FakeUsage()

        result = extract_gemini_usage(FakeResponse())
        assert result["thinking_tokens"] is None

    def test_C_openai_includes_thinking_key_as_none(self):
        """extract_openai_usage always includes thinking_tokens=None key."""
        from app.core.usage import extract_openai_usage

        class FakeDetails:
            cached_tokens = 0

        class FakeUsage:
            prompt_tokens = 1000
            completion_tokens = 500
            prompt_tokens_details = FakeDetails()

        class FakeResponse:
            usage = FakeUsage()

        result = extract_openai_usage(FakeResponse())
        assert "thinking_tokens" in result
        assert result["thinking_tokens"] is None

    def test_C_unknown_provider_includes_thinking_key(self):
        """extract_usage_from_response for deepl/ollama includes thinking_tokens=None key."""
        from app.core.usage import extract_usage_from_response
        result = extract_usage_from_response("deepl", object())
        assert "thinking_tokens" in result
        assert result["thinking_tokens"] is None


# ===========================================================================
# D. Billable output calculation includes thinking tokens
# ===========================================================================

class TestBillableOutput:

    def test_D_thinking_tokens_billed_at_output_rate(self):
        """
        billable_output = output_tokens + thinking_tokens; both at output_per_1m rate.
        gemini-3.5-flash-lite: input=$0.30/1M, output=$2.50/1M
        1M input + 300K output + 200K thinking → 500K billable output
        Cost = 1.0*0.30 + 0.5*2.50 = 0.30 + 1.25 = $1.55
        """
        from app.core.usage import calculate_estimated_cost
        cost = calculate_estimated_cost(
            "gemini", "gemini-3.5-flash-lite",
            input_tokens=1_000_000,
            cached_input_tokens=None,
            output_tokens=300_000,
            thinking_tokens=200_000,
        )
        expected = 1.0 * 0.30 + 0.5 * 2.50
        assert cost is not None
        assert abs(cost - expected) < 1e-9, f"Expected {expected}, got {cost}"

    def test_D_no_thinking_no_regression(self):
        """thinking_tokens=None → same result as before this fix."""
        from app.core.usage import calculate_estimated_cost
        cost = calculate_estimated_cost(
            "gemini", "gemini-3.5-flash-lite",
            input_tokens=1_000_000,
            cached_input_tokens=None,
            output_tokens=500_000,
        )
        expected = 1.0 * 0.30 + 0.5 * 2.50
        assert cost is not None
        assert abs(cost - expected) < 1e-9

    def test_D_only_thinking_no_output(self):
        """output_tokens=None but thinking_tokens=500K → billed at output rate."""
        from app.core.usage import calculate_estimated_cost
        cost = calculate_estimated_cost(
            "gemini", "gemini-3.5-flash-lite",
            input_tokens=None,
            cached_input_tokens=None,
            output_tokens=None,
            thinking_tokens=500_000,
        )
        expected = 0.5 * 2.50
        assert cost is not None
        assert abs(cost - expected) < 1e-9

    def test_D_all_none_including_thinking_returns_none(self):
        """All tokens None (including thinking) → cost is None."""
        from app.core.usage import calculate_estimated_cost
        cost = calculate_estimated_cost(
            "gemini", "gemini-3.5-flash-lite",
            input_tokens=None,
            cached_input_tokens=None,
            output_tokens=None,
            thinking_tokens=None,
        )
        assert cost is None


# ===========================================================================
# E. Dispatch invariant: no 404 fallback; DeepL/Ollama have _mark_sdk_started
# ===========================================================================

class TestDispatchInvariant:

    def test_E_translate_batch_deepl_has_sdk_started_marker(self):
        """translate_batch_deepl must call _mark_sdk_started before HTTP dispatch."""
        import inspect
        from app.services import translator
        src = inspect.getsource(translator.SubtitleTranslator.translate_batch_deepl)
        assert "_mark_sdk_started" in src

    def test_E_translate_batch_ollama_has_sdk_started_marker(self):
        """translate_batch_ollama must call _mark_sdk_started before HTTP dispatch."""
        import inspect
        from app.services import translator
        src = inspect.getsource(translator.SubtitleTranslator.translate_batch_ollama)
        assert "_mark_sdk_started" in src

    def test_E_no_404_fallback_in_translate_batch_gemini(self):
        """translate_batch_gemini must not contain in-dispatch 404 fallback loop."""
        import inspect
        from app.services import translator
        src = inspect.getsource(translator.SubtitleTranslator.translate_batch_gemini)
        assert "fallback_models" not in src

    def test_E_no_404_fallback_in_translate_batch_openai(self):
        """translate_batch_openai must not contain in-dispatch 404 fallback."""
        import inspect
        from app.services import translator
        src = inspect.getsource(translator.SubtitleTranslator.translate_batch_openai)
        assert "fallback_models" not in src

    def test_E_no_404_fallback_in_micro_repair_gemini(self):
        """first_pass_micro_repair_batch must not contain 404 fallback loop."""
        import inspect
        from app.services import translator
        src = inspect.getsource(translator.SubtitleTranslator.first_pass_micro_repair_batch)
        assert "fallback_models" not in src

    def test_E_no_404_fallback_in_fast_final_rescue_gemini(self):
        """fast_final_rescue_batch must not contain 404 fallback loop."""
        import inspect
        from app.services import translator
        src = inspect.getsource(translator.SubtitleTranslator.fast_final_rescue_batch)
        assert "fallback_models" not in src


# ===========================================================================
# F. Budget fail-closed on DB error when budget > 0
# ===========================================================================

class TestBudgetFailClosed:

    def test_F_fail_closed_when_budget_active_and_db_error(self):
        """try_consume_request_budget returns False when budget > 0 and DB unavailable."""
        with patch("app.core.quota.get_daily_budget", return_value=100), \
             patch("app.core.quota.DB_PATH", "/nonexistent/impossible/path/db.sqlite"):
            from app.core.quota import try_consume_request_budget
            result = try_consume_request_budget("gemini")
        assert result is False, "Must fail CLOSED (False) when budget > 0 and DB errors"

    def test_F_fail_open_when_budget_unlimited(self):
        """try_consume_request_budget returns True when budget is None (unlimited)."""
        with patch("app.core.quota.get_daily_budget", return_value=None):
            from app.core.quota import try_consume_request_budget
            result = try_consume_request_budget("gemini")
        assert result is True, "Must return True (allow) when budget is Unlimited"


# ===========================================================================
# G. All-time usage includes FAILED job costs
# ===========================================================================

class TestAllTimeUsageStats:

    @pytest.fixture
    def usage_db_with_failed_job(self, tmp_path):
        """DB with 1 TRANSLATED job + 1 FAILED job, each with one usage row."""
        import app.core.db as db_mod
        from app.core.usage import init_usage_schema

        db_path = str(tmp_path / "test_usage.db")

        with patch("app.core.db.DB_PATH", db_path), \
             patch("app.core.usage.DB_PATH", db_path):
            db_mod.DB_PATH = db_path
            db_mod.init_db()
            init_usage_schema()

            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "INSERT INTO jobs (id, video_path, status, created_at, updated_at) "
                    "VALUES (1, '/tv/ok.mkv', 'TRANSLATED', '2026-08-25T10:00:00Z', '2026-08-25T10:00:00Z')"
                )
                conn.execute(
                    "INSERT INTO jobs (id, video_path, status, created_at, updated_at) "
                    "VALUES (2, '/tv/fail.mkv', 'FAILED', '2026-08-25T11:00:00Z', '2026-08-25T11:00:00Z')"
                )
                conn.execute(
                    "INSERT INTO ai_usage_ledger "
                    "(request_uid, job_id, provider, model, stage, status, estimated_cost_usd, created_at) "
                    "VALUES ('uid-ok', 1, 'gemini', 'gemini-3.5-flash-lite', 'PRIMARY', 'SUCCESS', 0.001, '2026-08-25T10:01:00Z')"
                )
                conn.execute(
                    "INSERT INTO ai_usage_ledger "
                    "(request_uid, job_id, provider, model, stage, status, estimated_cost_usd, created_at) "
                    "VALUES ('uid-fail', 2, 'gemini', 'gemini-3.5-flash-lite', 'PRIMARY', 'FAILED', 0.0005, '2026-08-25T11:01:00Z')"
                )
                conn.commit()

        return db_path

    def test_G_all_time_calls_include_failed_job(self, usage_db_with_failed_job):
        """total_calls_all_time counts rows from BOTH TRANSLATED and FAILED jobs."""
        with patch("app.core.usage.DB_PATH", usage_db_with_failed_job):
            from app.core.usage import get_historical_stats
            stats = get_historical_stats()

        assert stats["completed_jobs_with_ai"] == 1, "Only TRANSLATED jobs in average denominator"
        assert stats["total_calls_all_time"] == 2, \
            f"Expected 2 all-time calls (success+failed), got {stats['total_calls_all_time']}"

    def test_G_all_time_cost_includes_failed_job(self, usage_db_with_failed_job):
        """total_estimated_cost_all_time sums costs from ALL job statuses."""
        with patch("app.core.usage.DB_PATH", usage_db_with_failed_job):
            from app.core.usage import get_historical_stats
            stats = get_historical_stats()

        assert stats["total_estimated_cost_all_time"] is not None
        assert abs(stats["total_estimated_cost_all_time"] - 0.0015) < 1e-9, \
            f"Expected 0.0015, got {stats['total_estimated_cost_all_time']}"

    def test_G_average_cost_only_from_translated(self, usage_db_with_failed_job):
        """average_estimated_cost_per_job uses only TRANSLATED job rows."""
        with patch("app.core.usage.DB_PATH", usage_db_with_failed_job):
            from app.core.usage import get_historical_stats
            stats = get_historical_stats()

        # 1 translated job with cost 0.001 → avg = 0.001
        assert stats["average_estimated_cost_per_job"] is not None
        assert abs(stats["average_estimated_cost_per_job"] - 0.001) < 1e-9


# ===========================================================================
# H. clear_all_jobs cascades to ai_usage_ledger
# ===========================================================================

class TestClearAllJobsCascade:

    def test_H_clear_all_jobs_removes_ledger_rows(self, tmp_path):
        """clear_all_jobs() must also delete all ai_usage_ledger rows."""
        import app.core.db as db_mod
        from app.core.usage import init_usage_schema

        db_path = str(tmp_path / "test_clear.db")

        with patch("app.core.db.DB_PATH", db_path), \
             patch("app.core.usage.DB_PATH", db_path):
            db_mod.DB_PATH = db_path
            db_mod.init_db()
            init_usage_schema()

            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "INSERT INTO jobs (id, video_path, status, created_at, updated_at) "
                    "VALUES (1, '/tv/show.mkv', 'TRANSLATED', '2026-08-25T10:00:00Z', '2026-08-25T10:00:00Z')"
                )
                conn.execute(
                    "INSERT INTO ai_usage_ledger "
                    "(request_uid, job_id, provider, model, stage, status, created_at) "
                    "VALUES ('uid-clear', 1, 'gemini', 'gemini-3.5-flash-lite', 'PRIMARY', 'SUCCESS', '2026-08-25T10:01:00Z')"
                )
                conn.commit()

            # Confirm data is seeded
            with sqlite3.connect(db_path) as conn:
                assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
                assert conn.execute("SELECT COUNT(*) FROM ai_usage_ledger").fetchone()[0] == 1

            db_mod.clear_all_jobs()

            with sqlite3.connect(db_path) as conn:
                jobs_count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
                ledger_count = conn.execute("SELECT COUNT(*) FROM ai_usage_ledger").fetchone()[0]

        assert jobs_count == 0, "jobs table must be empty after clear_all_jobs"
        assert ledger_count == 0, "ai_usage_ledger must be empty after clear_all_jobs"

    def test_H_clear_all_jobs_works_without_ledger_table(self, tmp_path):
        """clear_all_jobs must not crash if ai_usage_ledger doesn't exist yet."""
        import app.core.db as db_mod

        db_path = str(tmp_path / "test_no_ledger.db")

        with patch("app.core.db.DB_PATH", db_path):
            db_mod.DB_PATH = db_path
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "CREATE TABLE jobs "
                    "(id INTEGER PRIMARY KEY, video_path TEXT, status TEXT, "
                    "created_at TEXT, updated_at TEXT)"
                )
                conn.execute(
                    "INSERT INTO jobs VALUES (1, '/tv/show.mkv', 'TRANSLATED', "
                    "'2026-08-25T10:00:00Z', '2026-08-25T10:00:00Z')"
                )
                conn.commit()

            db_mod.clear_all_jobs()  # Must not raise

            with sqlite3.connect(db_path) as conn:
                count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        assert count == 0
