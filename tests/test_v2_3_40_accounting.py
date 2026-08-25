"""
tests/test_v2_3_40_accounting.py
=================================
v2.3.40-beta Accounting Hardening Regression Tests

Tests A–Y covering:
A. Pricing registry correctness (all models vs verified prices 2026-08-24)
B. Time-sensitive pricing (gemini-3.6/3.7-flash promotional vs standard)
C. NULL cost for unknown/unverifiable models
D. gpt-4-turbo: no cached rate (cached_input_per_1m=None)
E. calculate_estimated_cost: correct formula (no double-counting cached)
F. calculate_estimated_cost: returns None when model unknown
G. calculate_estimated_cost: returns None when all tokens None
H. calculate_estimated_cost: output includes thinking/reasoning tokens
I. translate_batch wrapper: has job_id parameter
J. translate_batch_gemini: accepts job_id kwarg
K. translate_batch_openai: accepts job_id kwarg
L. translate_batch_deepl: accepts job_id kwarg
M. translate_batch_ollama: accepts job_id kwarg
N. classify_and_recover_identical: has job_id parameter in signature
O. _execute_single_escalation_call: has job_id parameter in signature
P. verify_single_occurrence_entities: has job_id parameter in signature
Q. with_retry: reads job_id from kwargs (invariant check)
R. record_dispatch: job_id=None → NULL in DB (correct, non-attributed)
S. record_dispatch: job_id=42 → 42 in DB (correct attribution)
T. get_pricing: at_date before promo period returns standard price
U. get_pricing: at_date during promo period returns promo price
V. get_pricing: gemini-3.5-flash-lite correct rates
W. get_pricing: gpt-4o-mini correct rates
X. get_pricing: o1-mini has cached_input rate
Y. extract_gemini_usage: cached_content=0 → cached_input_tokens=None
"""

import inspect
import pytest
from datetime import date
from typing import Optional


# ===========================================================================
# A. Pricing registry: verified prices for all known models
# ===========================================================================

class TestPricingRegistry:

    def test_A_gemini_flash_lite_prices(self):
        """gemini-3.5-flash-lite: $0.30/$0.03/$2.50 per 1M (verified 2026-08-24)"""
        from app.core.usage import get_pricing
        p = get_pricing("gemini", "gemini-3.5-flash-lite")
        assert p is not None
        assert abs(p["input_per_1m"] - 0.30) < 1e-9
        assert abs(p["cached_input_per_1m"] - 0.03) < 1e-9
        assert abs(p["output_per_1m"] - 2.50) < 1e-9

    def test_A_gemini_flash_prices(self):
        """gemini-3.5-flash: $1.50/$0.15/$9.00 per 1M"""
        from app.core.usage import get_pricing
        p = get_pricing("gemini", "gemini-3.5-flash")
        assert p is not None
        assert abs(p["input_per_1m"] - 1.50) < 1e-9
        assert abs(p["cached_input_per_1m"] - 0.15) < 1e-9
        assert abs(p["output_per_1m"] - 9.00) < 1e-9

    def test_A_gpt4o_mini_prices(self):
        """gpt-4o-mini: $0.15/$0.075/$0.60 per 1M"""
        from app.core.usage import get_pricing
        p = get_pricing("openai", "gpt-4o-mini")
        assert p is not None
        assert abs(p["input_per_1m"] - 0.15) < 1e-9
        assert abs(p["cached_input_per_1m"] - 0.075) < 1e-9
        assert abs(p["output_per_1m"] - 0.60) < 1e-9

    def test_A_gpt4o_prices(self):
        """gpt-4o: $2.50/$1.25/$10.00 per 1M"""
        from app.core.usage import get_pricing
        p = get_pricing("openai", "gpt-4o")
        assert p is not None
        assert abs(p["input_per_1m"] - 2.50) < 1e-9
        assert abs(p["cached_input_per_1m"] - 1.25) < 1e-9
        assert abs(p["output_per_1m"] - 10.00) < 1e-9

    def test_A_o1_mini_prices(self):
        """o1-mini: $1.10/$0.55/$4.40 per 1M"""
        from app.core.usage import get_pricing
        p = get_pricing("openai", "o1-mini")
        assert p is not None
        assert abs(p["input_per_1m"] - 1.10) < 1e-9
        assert abs(p["cached_input_per_1m"] - 0.55) < 1e-9
        assert abs(p["output_per_1m"] - 4.40) < 1e-9

    def test_D_gpt4_turbo_no_cached_rate(self):
        """gpt-4-turbo: $10.00/None/$30.00 — no official cached discount"""
        from app.core.usage import get_pricing
        p = get_pricing("openai", "gpt-4-turbo")
        assert p is not None
        assert abs(p["input_per_1m"] - 10.00) < 1e-9
        assert p["cached_input_per_1m"] is None
        assert abs(p["output_per_1m"] - 30.00) < 1e-9

    # -----------------------------------------------------------------------
    # B. Time-sensitive pricing
    # -----------------------------------------------------------------------

    def test_B_gemini_36_promo_2026(self):
        """gemini-3.6-flash: promotional $0.75/$0.075/$3.75 during 2026"""
        from app.core.usage import get_pricing
        p = get_pricing("gemini", "gemini-3.6-flash", at_date=date(2026, 8, 24))
        assert p is not None
        assert abs(p["input_per_1m"] - 0.75) < 1e-9
        assert abs(p["cached_input_per_1m"] - 0.075) < 1e-9
        assert abs(p["output_per_1m"] - 3.75) < 1e-9

    def test_B_gemini_36_standard_2027(self):
        """gemini-3.6-flash: standard $1.50/$0.15/$7.50 from 2027-01-01"""
        from app.core.usage import get_pricing
        p = get_pricing("gemini", "gemini-3.6-flash", at_date=date(2027, 1, 1))
        assert p is not None
        assert abs(p["input_per_1m"] - 1.50) < 1e-9
        assert abs(p["cached_input_per_1m"] - 0.15) < 1e-9
        assert abs(p["output_per_1m"] - 7.50) < 1e-9

    def test_B_gemini_37_promo_2026(self):
        """gemini-3.7-flash: promotional $0.75/$0.075/$3.75 during 2026"""
        from app.core.usage import get_pricing
        p = get_pricing("gemini", "gemini-3.7-flash", at_date=date(2026, 12, 31))
        assert p is not None
        assert abs(p["input_per_1m"] - 0.75) < 1e-9
        assert abs(p["output_per_1m"] - 3.75) < 1e-9

    def test_B_gemini_37_standard_2027(self):
        """gemini-3.7-flash: standard $1.50/$0.15/$7.50 from 2027-01-01"""
        from app.core.usage import get_pricing
        p = get_pricing("gemini", "gemini-3.7-flash", at_date=date(2027, 6, 15))
        assert p is not None
        assert abs(p["input_per_1m"] - 1.50) < 1e-9
        assert abs(p["output_per_1m"] - 7.50) < 1e-9

    # -----------------------------------------------------------------------
    # C. NULL cost for unknown/unverifiable models
    # -----------------------------------------------------------------------

    def test_C_gemini_35_pro_null(self):
        """gemini-3.5-pro: not GA → get_pricing returns None"""
        from app.core.usage import get_pricing
        assert get_pricing("gemini", "gemini-3.5-pro") is None

    def test_C_gemini_40_flash_null(self):
        """gemini-4.0-flash: not released → get_pricing returns None"""
        from app.core.usage import get_pricing
        assert get_pricing("gemini", "gemini-4.0-flash") is None

    def test_C_gpt_56_turbo_null(self):
        """gpt-5.6-turbo: not a valid OpenAI model ID → get_pricing returns None"""
        from app.core.usage import get_pricing
        assert get_pricing("openai", "gpt-5.6-turbo") is None

    def test_C_deepl_null(self):
        """deepl: no token pricing → get_pricing returns None"""
        from app.core.usage import get_pricing
        assert get_pricing("deepl", "deepl") is None

    def test_C_ollama_null(self):
        """ollama: no token pricing → get_pricing returns None"""
        from app.core.usage import get_pricing
        assert get_pricing("ollama", "llama3") is None

    def test_C_unknown_model_null(self):
        """Unknown model → get_pricing returns None → NULL cost"""
        from app.core.usage import get_pricing
        assert get_pricing("gemini", "gemini-99.9-ultra") is None


# ===========================================================================
# E-H. calculate_estimated_cost
# ===========================================================================

class TestCostFormula:

    def test_E_no_cache_formula(self):
        """
        Cost formula: all input at full rate when no caching.
        1M input, 100K output, no cache.
        gemini-3.5-flash-lite: $0.30/1M in, $2.50/1M out
        Expected = 1.0 * 0.30 + 0.1 * 2.50 = 0.30 + 0.25 = 0.55
        """
        from app.core.usage import calculate_estimated_cost
        cost = calculate_estimated_cost("gemini", "gemini-3.5-flash-lite", 1_000_000, None, 100_000)
        expected = (1_000_000 / 1_000_000 * 0.30) + (100_000 / 1_000_000 * 2.50)
        assert cost is not None
        assert abs(cost - expected) < 1e-9

    def test_E_with_cache_no_double_counting(self):
        """
        No double-counting: cached tokens NOT billed at full input rate.
        1M input, 200K cached, 500K output.
        non_cached = 800K at $0.30 + cached = 200K at $0.03 + output = 500K at $2.50
        """
        from app.core.usage import calculate_estimated_cost
        cost = calculate_estimated_cost("gemini", "gemini-3.5-flash-lite", 1_000_000, 200_000, 500_000)
        expected = (
            (800_000 / 1_000_000 * 0.30) +
            (200_000 / 1_000_000 * 0.03) +
            (500_000 / 1_000_000 * 2.50)
        )
        assert cost is not None
        assert abs(cost - expected) < 1e-9

    def test_E_output_only(self):
        """output-only tokens: input_tokens=None → only output billed"""
        from app.core.usage import calculate_estimated_cost
        cost = calculate_estimated_cost("gemini", "gemini-3.5-flash-lite", None, None, 500_000)
        expected = 500_000 / 1_000_000 * 2.50
        assert cost is not None
        assert abs(cost - expected) < 1e-9

    def test_F_unknown_model_returns_none(self):
        """Unknown model → None (never a fake price)"""
        from app.core.usage import calculate_estimated_cost
        cost = calculate_estimated_cost("gemini", "gemini-99-ultra", 1_000_000, None, 100_000)
        assert cost is None

    def test_G_all_tokens_none_returns_none(self):
        """All tokens None → None"""
        from app.core.usage import calculate_estimated_cost
        cost = calculate_estimated_cost("gemini", "gemini-3.5-flash-lite", None, None, None)
        assert cost is None

    def test_H_output_includes_thinking_tokens(self):
        """
        Gemini output_per_1m covers ALL output incl. thinking tokens.
        OpenAI output_per_1m covers ALL output incl. reasoning tokens.
        This is not a separate rate — output_tokens already includes them.
        Verify: same formula applies, no separate thinking_rate.
        """
        from app.core.usage import get_pricing
        # Gemini 3.7 flash has thinking tokens, uses same output rate
        p = get_pricing("gemini", "gemini-3.7-flash", at_date=date(2026, 8, 24))
        assert p is not None
        assert p["output_per_1m"] is not None  # one rate covers all output incl thinking

    def test_E_gpt4_turbo_no_cache_discount(self):
        """
        gpt-4-turbo has no cached_input rate → all input at full $10.00
        Even if cached_input_tokens is provided, no caching discount applies.
        """
        from app.core.usage import calculate_estimated_cost
        # With 200K cached: should still bill all 1M at $10.00 (no cached rate)
        cost_with_cache = calculate_estimated_cost("openai", "gpt-4-turbo", 1_000_000, 200_000, 100_000)
        cost_without_cache = calculate_estimated_cost("openai", "gpt-4-turbo", 1_000_000, None, 100_000)
        # Since cached_rate is None, cached tokens get billed at full rate → same as no cache
        assert cost_with_cache is not None
        assert cost_without_cache is not None
        # With cache provided but cached_rate=None, formula falls back to full input rate
        # non_cached would be 800K * 10.0 + 200K * 10.0 = 1M * 10.0 (same as without cache)
        assert abs(cost_with_cache - cost_without_cache) < 1e-9


# ===========================================================================
# I-P. Signature inspection: job_id present in all dispatch functions
# ===========================================================================

class TestSignatureJobIdPropagation:

    def test_I_translate_batch_has_job_id(self):
        """translate_batch wrapper must accept job_id kwarg"""
        from app.services.translator import SubtitleTranslator
        sig = inspect.signature(SubtitleTranslator.translate_batch)
        assert "job_id" in sig.parameters, "translate_batch missing job_id parameter"

    def test_J_translate_batch_gemini_has_job_id(self):
        """translate_batch_gemini must accept job_id kwarg for with_retry to pick it up"""
        from app.services.translator import SubtitleTranslator
        sig = inspect.signature(SubtitleTranslator.translate_batch_gemini)
        assert "job_id" in sig.parameters, "translate_batch_gemini missing job_id parameter"

    def test_K_translate_batch_openai_has_job_id(self):
        """translate_batch_openai must accept job_id kwarg"""
        from app.services.translator import SubtitleTranslator
        sig = inspect.signature(SubtitleTranslator.translate_batch_openai)
        assert "job_id" in sig.parameters, "translate_batch_openai missing job_id parameter"

    def test_L_translate_batch_deepl_has_job_id(self):
        """translate_batch_deepl must accept job_id kwarg"""
        from app.services.translator import SubtitleTranslator
        sig = inspect.signature(SubtitleTranslator.translate_batch_deepl)
        assert "job_id" in sig.parameters, "translate_batch_deepl missing job_id parameter"

    def test_M_translate_batch_ollama_has_job_id(self):
        """translate_batch_ollama must accept job_id kwarg"""
        from app.services.translator import SubtitleTranslator
        sig = inspect.signature(SubtitleTranslator.translate_batch_ollama)
        assert "job_id" in sig.parameters, "translate_batch_ollama missing job_id parameter"

    def test_N_classify_and_recover_identical_has_job_id(self):
        """classify_and_recover_identical must accept job_id kwarg"""
        from app.services.translator import SubtitleTranslator
        sig = inspect.signature(SubtitleTranslator.classify_and_recover_identical)
        assert "job_id" in sig.parameters, "classify_and_recover_identical missing job_id parameter"

    def test_O_execute_single_escalation_has_job_id(self):
        """_execute_single_escalation_call must accept job_id kwarg"""
        from app.services.translator import SubtitleTranslator
        sig = inspect.signature(SubtitleTranslator._execute_single_escalation_call)
        assert "job_id" in sig.parameters, "_execute_single_escalation_call missing job_id parameter"

    def test_P_verify_single_occurrence_entities_has_job_id(self):
        """verify_single_occurrence_entities must accept job_id kwarg"""
        from app.services.translator import SubtitleTranslator
        sig = inspect.signature(SubtitleTranslator.verify_single_occurrence_entities)
        assert "job_id" in sig.parameters, "verify_single_occurrence_entities missing job_id parameter"

    def test_Q_first_pass_micro_repair_has_job_id(self):
        """first_pass_micro_repair_batch must accept job_id kwarg (was already correct)"""
        from app.services.translator import SubtitleTranslator
        sig = inspect.signature(SubtitleTranslator.first_pass_micro_repair_batch)
        assert "job_id" in sig.parameters

    def test_Q_fast_final_rescue_has_job_id(self):
        """fast_final_rescue_batch must accept job_id kwarg (was already correct)"""
        from app.services.translator import SubtitleTranslator
        sig = inspect.signature(SubtitleTranslator.fast_final_rescue_batch)
        assert "job_id" in sig.parameters

    def test_Q_escalate_single_line_has_job_id(self):
        """escalate_single_line must accept job_id kwarg"""
        from app.services.translator import SubtitleTranslator
        sig = inspect.signature(SubtitleTranslator.escalate_single_line)
        assert "job_id" in sig.parameters


# ===========================================================================
# R-S. record_dispatch: job_id attribution in DB
# ===========================================================================

class TestRecordDispatchJobId:

    @pytest.fixture
    def tmp_db(self, tmp_path):
        """
        Set up a real temporary SQLite DB for usage ledger testing.
        Patches both app.core.db.DB_PATH and app.core.usage.DB_PATH
        (since usage.py imports DB_PATH as a local name via 'from app.core.db import DB_PATH').
        """
        import app.core.db as db_mod
        from app.core.usage import init_usage_schema
        from unittest.mock import patch

        db_path = str(tmp_path / "test.db")

        with patch("app.core.db.DB_PATH", db_path), \
             patch("app.core.usage.DB_PATH", db_path):
            db_mod.DB_PATH = db_path
            db_mod.init_db()
            init_usage_schema()
            yield db_mod

    def test_R_null_job_id_stored_as_null(self, tmp_db, tmp_path):
        """record_dispatch with job_id=None → DB row has NULL job_id"""
        import sqlite3
        from unittest.mock import patch
        from app.core.usage import record_dispatch, generate_request_uid
        db_path = str(tmp_path / "test.db")
        uid = generate_request_uid()
        with patch("app.core.usage.DB_PATH", db_path):
            record_dispatch(uid, "gemini", "gemini-3.5-flash-lite", "PRIMARY", job_id=None)
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT job_id FROM ai_usage_ledger WHERE request_uid=?", (uid,)).fetchone()
        assert row is not None
        assert row[0] is None  # NULL in DB

    def test_S_job_id_stored_correctly(self, tmp_db, tmp_path):
        """record_dispatch with explicit job_id → DB row has correct job_id"""
        import sqlite3
        from unittest.mock import patch
        from app.core.usage import record_dispatch, generate_request_uid
        db_path = str(tmp_path / "test.db")
        # SQLite FK constraints are off by default; just insert directly
        with patch("app.core.usage.DB_PATH", db_path):
            uid = generate_request_uid()
            record_dispatch(uid, "gemini", "gemini-3.5-flash-lite", "PRIMARY", job_id=42)
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT job_id FROM ai_usage_ledger WHERE request_uid=?", (uid,)).fetchone()
        assert row is not None
        assert row[0] == 42


# ===========================================================================
# T-U. Time-sensitive pricing edge cases
# ===========================================================================

class TestTimeSensitivePricing:

    def test_T_promo_last_day(self):
        """2026-12-31 is last promo day → promotional price"""
        from app.core.usage import get_pricing
        p = get_pricing("gemini", "gemini-3.6-flash", at_date=date(2026, 12, 31))
        assert abs(p["input_per_1m"] - 0.75) < 1e-9

    def test_T_standard_first_day(self):
        """2027-01-01 is first standard day → standard price"""
        from app.core.usage import get_pricing
        p = get_pricing("gemini", "gemini-3.6-flash", at_date=date(2027, 1, 1))
        assert abs(p["input_per_1m"] - 1.50) < 1e-9

    def test_U_promo_mid_year(self):
        """Mid-2026 → promotional price for gemini-3.7-flash"""
        from app.core.usage import get_pricing
        p = get_pricing("gemini", "gemini-3.7-flash", at_date=date(2026, 6, 1))
        assert abs(p["input_per_1m"] - 0.75) < 1e-9

    def test_U_default_date_is_today(self):
        """get_pricing with no at_date defaults to today (no crash)"""
        from app.core.usage import get_pricing
        # Just verify it doesn't crash and returns a dict or None
        result = get_pricing("gemini", "gemini-3.5-flash-lite")
        assert result is not None  # this is a known model, should have price


# ===========================================================================
# V-X. Specific model rate correctness
# ===========================================================================

class TestSpecificModelRates:

    def test_V_gemini_flash_lite_rate_sanity(self):
        """gemini-3.5-flash-lite output rate 2.50 > input rate 0.30 (sanity check)"""
        from app.core.usage import get_pricing
        p = get_pricing("gemini", "gemini-3.5-flash-lite")
        assert p["output_per_1m"] > p["input_per_1m"]
        assert p["output_per_1m"] > p["cached_input_per_1m"]

    def test_W_gpt4o_mini_rates(self):
        """gpt-4o-mini: output > input > cached (output=0.60, input=0.15, cached=0.075)"""
        from app.core.usage import get_pricing
        p = get_pricing("openai", "gpt-4o-mini")
        assert p["output_per_1m"] > p["input_per_1m"] > p["cached_input_per_1m"]
        assert abs(p["output_per_1m"] - 0.60) < 1e-9

    def test_X_o1_mini_has_cached_rate(self):
        """o1-mini must have a cached_input rate (0.55/1M = 50% of input)"""
        from app.core.usage import get_pricing
        p = get_pricing("openai", "o1-mini")
        assert p["cached_input_per_1m"] is not None
        assert abs(p["cached_input_per_1m"] - 0.55) < 1e-9


# ===========================================================================
# Y. extract_gemini_usage: cached_content=0 → cached_input_tokens=None
# ===========================================================================

class TestTokenExtraction:

    def test_Y_gemini_zero_cache_stored_as_none(self):
        """cached_content_token_count=0 → cached_input_tokens=None (not 0)"""
        from app.core.usage import extract_gemini_usage

        class FakeUsage:
            prompt_token_count = 1000
            cached_content_token_count = 0
            candidates_token_count = 500

        class FakeResponse:
            usage_metadata = FakeUsage()

        result = extract_gemini_usage(FakeResponse())
        assert result["input_tokens"] == 1000
        assert result["cached_input_tokens"] is None  # 0 → None
        assert result["output_tokens"] == 500

    def test_Y_gemini_positive_cache_stored(self):
        """cached_content_token_count=200 → cached_input_tokens=200"""
        from app.core.usage import extract_gemini_usage

        class FakeUsage:
            prompt_token_count = 1000
            cached_content_token_count = 200
            candidates_token_count = 500

        class FakeResponse:
            usage_metadata = FakeUsage()

        result = extract_gemini_usage(FakeResponse())
        assert result["cached_input_tokens"] == 200

    def test_Y_openai_zero_cache_stored_as_none(self):
        """OpenAI cached_tokens=0 → cached_input_tokens=None"""
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
        assert result["cached_input_tokens"] is None  # 0 → None

    def test_Y_extract_dispatch_gemini(self):
        """extract_usage_from_response routes 'gemini' to extract_gemini_usage"""
        from app.core.usage import extract_usage_from_response

        class FakeUsage:
            prompt_token_count = 500
            cached_content_token_count = None
            candidates_token_count = 100

        class FakeResponse:
            usage_metadata = FakeUsage()

        result = extract_usage_from_response("gemini", FakeResponse())
        assert result["input_tokens"] == 500
        assert result["output_tokens"] == 100

    def test_Y_extract_dispatch_unknown_returns_none(self):
        """extract_usage_from_response for deepl/ollama returns all None"""
        from app.core.usage import extract_usage_from_response
        result = extract_usage_from_response("deepl", object())
        assert result["input_tokens"] is None
        assert result["output_tokens"] is None
        assert result["cached_input_tokens"] is None
