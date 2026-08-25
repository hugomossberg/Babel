"""
tests/test_model_catalog.py
===========================
v2.3.40-beta Model Catalog & Pricing Regression Tests

Coverage (§8 in hardening spec):
A. Unverified alias → no pricing fallback → None
B. Known model → exact official pricing
C. Invalid model ID → not exposed in current model choices
D. Valid model with unknown pricing → still selectable (UI), cost None
E. Primary/escalation dropdowns use same valid model catalog semantics
F. Time-sensitive Gemini pricing continues to work
G. No pricing fallback to sibling/nearest/cheapest model

Additional:
H. gemini-flash-latest alias removed (was conservative/fallback guess)
I. Circuit breaker quota invariant: HALF_OPEN winner consumes exactly 1 slot
J. Lease loser consumes 0 slots
"""

import pytest
from datetime import date, datetime, timezone, timedelta
from unittest.mock import patch


# ---------------------------------------------------------------------------
# A. Unverified alias → no pricing fallback → None
# ---------------------------------------------------------------------------

class TestUnverifiedAliasNoFallback:

    def test_A_gemini_flash_latest_no_pricing(self):
        """
        gemini-flash-latest is an unverified alias.
        It MUST NOT get conservative/fallback pricing from a sibling model.
        get_pricing must return None — caller will use NULL cost.
        """
        from app.core.usage import get_pricing
        result = get_pricing("gemini", "gemini-flash-latest")
        assert result is None, (
            "gemini-flash-latest is an unverified alias without documented pricing. "
            "Must return None, not fallback to 3.5-flash-lite rates."
        )

    def test_A_no_family_fallback_to_sibling(self):
        """
        Unknown model must NOT fall back to nearest/cheapest sibling.
        e.g. 'gemini-3.5-ultra' must not return gemini-3.5-flash-lite rates.
        """
        from app.core.usage import get_pricing
        assert get_pricing("gemini", "gemini-3.5-ultra") is None
        assert get_pricing("gemini", "gemini-3.5-flash-turbo") is None
        assert get_pricing("openai", "gpt-4o-max") is None


# ---------------------------------------------------------------------------
# B. Known model → exact official pricing
# ---------------------------------------------------------------------------

class TestKnownModelExactPricing:

    def test_B_gemini_flash_lite_exact(self):
        """gemini-3.5-flash-lite: verified $0.30/$0.03/$2.50 per 1M"""
        from app.core.usage import get_pricing
        p = get_pricing("gemini", "gemini-3.5-flash-lite")
        assert p is not None
        assert abs(p["input_per_1m"] - 0.30) < 1e-9
        assert abs(p["cached_input_per_1m"] - 0.03) < 1e-9
        assert abs(p["output_per_1m"] - 2.50) < 1e-9

    def test_B_gemini_35_flash_exact(self):
        """gemini-3.5-flash: verified $1.50/$0.15/$9.00 per 1M"""
        from app.core.usage import get_pricing
        p = get_pricing("gemini", "gemini-3.5-flash")
        assert p is not None
        assert abs(p["input_per_1m"] - 1.50) < 1e-9
        assert abs(p["cached_input_per_1m"] - 0.15) < 1e-9
        assert abs(p["output_per_1m"] - 9.00) < 1e-9

    def test_B_gpt4o_mini_exact(self):
        """gpt-4o-mini: verified $0.15/$0.075/$0.60 per 1M"""
        from app.core.usage import get_pricing
        p = get_pricing("openai", "gpt-4o-mini")
        assert p is not None
        assert abs(p["input_per_1m"] - 0.15) < 1e-9
        assert abs(p["cached_input_per_1m"] - 0.075) < 1e-9
        assert abs(p["output_per_1m"] - 0.60) < 1e-9

    def test_B_gpt4o_exact(self):
        """gpt-4o: verified $2.50/$1.25/$10.00 per 1M"""
        from app.core.usage import get_pricing
        p = get_pricing("openai", "gpt-4o")
        assert p is not None
        assert abs(p["input_per_1m"] - 2.50) < 1e-9
        assert abs(p["cached_input_per_1m"] - 1.25) < 1e-9
        assert abs(p["output_per_1m"] - 10.00) < 1e-9

    def test_B_o1_mini_exact(self):
        """o1-mini: verified $1.10/$0.55/$4.40 per 1M"""
        from app.core.usage import get_pricing
        p = get_pricing("openai", "o1-mini")
        assert p is not None
        assert abs(p["input_per_1m"] - 1.10) < 1e-9
        assert abs(p["cached_input_per_1m"] - 0.55) < 1e-9
        assert abs(p["output_per_1m"] - 4.40) < 1e-9

    def test_B_gpt4_turbo_exact(self):
        """gpt-4-turbo: verified $10.00/None/$30.00 per 1M (no cached discount)"""
        from app.core.usage import get_pricing
        p = get_pricing("openai", "gpt-4-turbo")
        assert p is not None
        assert abs(p["input_per_1m"] - 10.00) < 1e-9
        assert p["cached_input_per_1m"] is None
        assert abs(p["output_per_1m"] - 30.00) < 1e-9


# ---------------------------------------------------------------------------
# C. Invalid model ID → not exposed in HTML model choices
#    (verified against official Google/OpenAI docs 2026-08-24)
# ---------------------------------------------------------------------------

class TestInvalidModelNotExposed:
    """
    These model IDs were verified INVALID against official documentation.
    They must NOT appear in the active UI dropdowns.
    """

    INVALID_IDS = [
        "gemini-3.5-pro",    # Does not exist in Gemini API (verified ai.google.dev)
        "gemini-4.0-flash",  # Does not exist — Gemini 4 not released (verified ai.google.dev)
        "gpt-5.6-turbo",     # Fabricated ID — does not exist in OpenAI API (verified platform.openai.com)
    ]

    def _get_html_content(self):
        import os
        html_path = os.path.join(os.path.dirname(__file__), "..", "app", "templates", "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_C_gemini_35_pro_not_in_dropdown(self):
        """gemini-3.5-pro (INVALID) must not appear as a selectable option"""
        html = self._get_html_content()
        assert 'value="gemini-3.5-pro"' not in html, (
            "gemini-3.5-pro is not a valid Gemini model ID and must not be in the UI dropdown."
        )

    def test_C_gemini_40_flash_not_in_dropdown(self):
        """gemini-4.0-flash (INVALID) must not appear as a selectable option"""
        html = self._get_html_content()
        assert 'value="gemini-4.0-flash"' not in html, (
            "gemini-4.0-flash does not exist (Gemini 4 not released). Must not be in the UI dropdown."
        )

    def test_C_gpt_56_turbo_not_in_dropdown(self):
        """gpt-5.6-turbo (INVALID) must not appear as a selectable option"""
        html = self._get_html_content()
        assert 'value="gpt-5.6-turbo"' not in html, (
            "gpt-5.6-turbo is a fabricated model ID. Must not be in the UI dropdown."
        )

    def test_C_invalid_models_no_pricing(self):
        """All invalid model IDs must return None from get_pricing"""
        from app.core.usage import get_pricing
        assert get_pricing("gemini", "gemini-3.5-pro") is None
        assert get_pricing("gemini", "gemini-4.0-flash") is None
        assert get_pricing("openai", "gpt-5.6-turbo") is None


# ---------------------------------------------------------------------------
# D. Valid model with unknown pricing → still selectable, cost None
# ---------------------------------------------------------------------------

class TestValidModelUnknownPricing:

    def test_D_gemini_36_flash_has_pricing(self):
        """gemini-3.6-flash: valid model with verified pricing"""
        from app.core.usage import get_pricing
        p = get_pricing("gemini", "gemini-3.6-flash", at_date=date(2026, 8, 24))
        assert p is not None, "gemini-3.6-flash is valid and has verified pricing"

    def test_D_gemini_37_flash_has_pricing(self):
        """gemini-3.7-flash: valid model with verified pricing"""
        from app.core.usage import get_pricing
        p = get_pricing("gemini", "gemini-3.7-flash", at_date=date(2026, 8, 24))
        assert p is not None, "gemini-3.7-flash is valid and has verified pricing"

    def test_D_unknown_pricing_returns_none_not_zero(self):
        """Unknown pricing must return None, not 0.0 — UI shows dash, not $0.00"""
        from app.core.usage import calculate_estimated_cost
        result = calculate_estimated_cost("gemini", "gemini-99-unknown", 100_000, None, 50_000)
        assert result is None, "Unknown model pricing must return None, not 0.0"


# ---------------------------------------------------------------------------
# E. Primary/escalation dropdowns use same valid model catalog semantics
# ---------------------------------------------------------------------------

class TestDropdownModelCatalogSemantics:
    """
    Both primary and escalation dropdowns must only contain verified model IDs.
    """

    def _get_html_options(self):
        import os, re
        html_path = os.path.join(os.path.dirname(__file__), "..", "app", "templates", "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        options = re.findall(r'<option value="([^"]+)"', html)
        return options

    KNOWN_INVALID = {"gemini-3.5-pro", "gemini-4.0-flash", "gpt-5.6-turbo"}
    KNOWN_VALID_GEMINI = {"gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.7-flash"}
    KNOWN_VALID_OPENAI = {"gpt-4o-mini", "gpt-4o", "o1-mini", "gpt-4-turbo"}

    def test_E_no_invalid_models_in_any_dropdown(self):
        """No invalid model IDs must appear in any dropdown in the HTML"""
        options = self._get_html_options()
        for invalid_id in self.KNOWN_INVALID:
            assert invalid_id not in options, (
                f"Invalid model '{invalid_id}' found in UI dropdowns — must be removed."
            )

    def test_E_valid_gemini_models_present(self):
        """All known valid Gemini models should be in the dropdown"""
        options = self._get_html_options()
        for model_id in self.KNOWN_VALID_GEMINI:
            assert model_id in options, (
                f"Valid Gemini model '{model_id}' missing from UI dropdowns."
            )

    def test_E_valid_openai_models_present(self):
        """All known valid OpenAI models should be in the dropdown"""
        options = self._get_html_options()
        for model_id in self.KNOWN_VALID_OPENAI:
            assert model_id in options, (
                f"Valid OpenAI model '{model_id}' missing from UI dropdowns."
            )


# ---------------------------------------------------------------------------
# F. Time-sensitive Gemini pricing continues to work
# ---------------------------------------------------------------------------

class TestTimeSensitivePricingContinues:

    def test_F_gemini_36_promo_active_2026(self):
        """Promotional pricing for gemini-3.6-flash is active during 2026"""
        from app.core.usage import get_pricing
        p = get_pricing("gemini", "gemini-3.6-flash", at_date=date(2026, 8, 24))
        assert p is not None
        assert abs(p["input_per_1m"] - 0.75) < 1e-9, "Promo rate expected: $0.75/1M in"
        assert abs(p["output_per_1m"] - 3.75) < 1e-9

    def test_F_gemini_36_standard_from_2027(self):
        """Standard pricing for gemini-3.6-flash kicks in from 2027-01-01"""
        from app.core.usage import get_pricing
        p = get_pricing("gemini", "gemini-3.6-flash", at_date=date(2027, 1, 1))
        assert p is not None
        assert abs(p["input_per_1m"] - 1.50) < 1e-9, "Standard rate expected: $1.50/1M in"

    def test_F_gemini_37_promo_last_day(self):
        """Promotional pricing for gemini-3.7-flash is active on last promo day 2026-12-31"""
        from app.core.usage import get_pricing
        p = get_pricing("gemini", "gemini-3.7-flash", at_date=date(2026, 12, 31))
        assert p is not None
        assert abs(p["input_per_1m"] - 0.75) < 1e-9

    def test_F_gemini_37_standard_from_2027(self):
        """Standard pricing for gemini-3.7-flash from 2027-01-01"""
        from app.core.usage import get_pricing
        p = get_pricing("gemini", "gemini-3.7-flash", at_date=date(2027, 1, 1))
        assert p is not None
        assert abs(p["input_per_1m"] - 1.50) < 1e-9

    def test_F_no_runtime_web_fetch(self):
        """Pricing lookup must use static _PRICING_TABLE only — no network calls"""
        import socket
        original_getaddrinfo = socket.getaddrinfo

        def block_network(*args, **kwargs):
            raise OSError("Network access blocked in test")

        from app.core.usage import get_pricing
        socket.getaddrinfo = block_network
        try:
            p = get_pricing("gemini", "gemini-3.5-flash-lite")
            assert p is not None, "Must work without network"
        finally:
            socket.getaddrinfo = original_getaddrinfo


# ---------------------------------------------------------------------------
# G. No pricing fallback to sibling/nearest/cheapest model
# ---------------------------------------------------------------------------

class TestNoPricingFallback:

    def test_G_no_fallback_for_new_gemini_versions(self):
        """New/unknown Gemini model versions must not fall back to known models"""
        from app.core.usage import get_pricing
        assert get_pricing("gemini", "gemini-3.8-flash") is None
        assert get_pricing("gemini", "gemini-3.9-pro") is None
        assert get_pricing("gemini", "gemini-4.1-flash") is None

    def test_G_no_fallback_for_new_openai_versions(self):
        """New/unknown OpenAI model versions must not fall back to known models"""
        from app.core.usage import get_pricing
        assert get_pricing("openai", "gpt-5") is None
        assert get_pricing("openai", "gpt-4o-ultra") is None


# ---------------------------------------------------------------------------
# H. gemini-flash-latest alias removed (was conservative pricing guess)
# ---------------------------------------------------------------------------

class TestGeminiFlashLatestAlias:

    def test_H_gemini_flash_latest_no_pricing(self):
        """
        gemini-flash-latest was previously given conservative 3.5-flash-lite rates.
        After the fix, get_pricing must return None for this alias.
        """
        from app.core.usage import get_pricing
        result = get_pricing("gemini", "gemini-flash-latest")
        assert result is None, (
            "gemini-flash-latest must not have alias/fallback pricing. "
            "Cost must be NULL until officially documented by Google."
        )

    def test_H_alias_not_in_pricing_table(self):
        """gemini-flash-latest must not appear in _PRICING_TABLE"""
        from app.core.usage import _PRICING_TABLE
        models_in_table = {entry["model"] for entry in _PRICING_TABLE if entry["provider"] == "gemini"}
        assert "gemini-flash-latest" not in models_in_table, (
            "gemini-flash-latest must be removed from _PRICING_TABLE."
        )


# ---------------------------------------------------------------------------
# I/J. Circuit breaker quota invariant: HALF_OPEN winner/loser slot counts
# ---------------------------------------------------------------------------

class TestCircuitBreakerQuotaInvariant:
    """
    Quota invariant:
    - HALF_OPEN probe winner: exactly +1 daily slot
    - Lease loser (probe in flight): 0 daily slots consumed
    - Blocked provider: 0 slots consumed
    - Normal ACTIVE request: exactly +1 slot
    """

    @pytest.fixture(autouse=True)
    def isolated_quota_db(self, tmp_path):
        db_file = str(tmp_path / "quota_test.db")
        import app.core.db as db_mod
        import app.core.quota as q_mod
        orig_db = db_mod.DB_PATH
        db_mod.DB_PATH = db_file
        q_mod.DB_PATH = db_file
        db_mod.init_db()
        q_mod.set_jitter_override(0.0)
        yield db_file
        q_mod.set_jitter_override(None)
        db_mod.DB_PATH = orig_db
        q_mod.DB_PATH = orig_db

    def test_I_half_open_winner_consumes_exactly_1_slot(self):
        """HALF_OPEN probe winner acquires exactly 1 daily_request_counts slot"""
        from app.core.quota import (
            record_provider_quota_exhausted, acquire_dispatch_slot,
            get_daily_requests_used
        )
        from app.core.db import set_setting
        set_setting("daily_request_budget_gemini", "10")

        now = datetime(2026, 8, 24, 10, 0, 0, tzinfo=timezone.utc)
        with patch("app.core.quota._utcnow", return_value=now):
            record_provider_quota_exhausted("gemini", "Quota", retry_after_seconds=900)

        future = now + timedelta(seconds=910)
        with patch("app.core.quota._utcnow", return_value=future):
            allowed, info = acquire_dispatch_slot("gemini", job_id="probe_test")
            assert allowed is True
            assert info["is_probe"] is True
            assert info["state"] == "HALF_OPEN"
            assert get_daily_requests_used("gemini") == 1, (
                "HALF_OPEN probe winner must consume exactly 1 daily slot"
            )

    def test_J_lease_loser_consumes_0_slots(self):
        """Lease loser (probe in flight) consumes 0 daily slots"""
        from app.core.quota import (
            record_provider_quota_exhausted, acquire_dispatch_slot,
            get_daily_requests_used
        )
        from app.core.db import set_setting
        set_setting("daily_request_budget_gemini", "10")

        now = datetime(2026, 8, 24, 10, 0, 0, tzinfo=timezone.utc)
        with patch("app.core.quota._utcnow", return_value=now):
            record_provider_quota_exhausted("gemini", "Quota", retry_after_seconds=900)

        future = now + timedelta(seconds=910)
        with patch("app.core.quota._utcnow", return_value=future):
            allowed1, info1 = acquire_dispatch_slot("gemini", job_id="winner")
            assert allowed1 is True
            allowed2, info2 = acquire_dispatch_slot("gemini", job_id="loser")
            assert allowed2 is False
            assert info2["reason"] == "Probe request currently in flight"
            # Winner = 1 slot, loser = 0 additional
            assert get_daily_requests_used("gemini") == 1

    def test_I_active_provider_normal_request_consumes_1_slot(self):
        """Normal ACTIVE request (no block) consumes exactly 1 slot"""
        from app.core.quota import acquire_dispatch_slot, get_daily_requests_used
        from app.core.db import set_setting
        set_setting("daily_request_budget_gemini", "10")

        now = datetime(2026, 8, 24, 10, 0, 0, tzinfo=timezone.utc)
        with patch("app.core.quota._utcnow", return_value=now):
            allowed, info = acquire_dispatch_slot("gemini", job_id="normal_job")
            assert allowed is True
            assert info["is_probe"] is False
            assert info["state"] == "ACTIVE"
            assert get_daily_requests_used("gemini") == 1

    def test_J_blocked_provider_request_consumes_0_slots(self):
        """Blocked provider dispatch (still within blocked_until) consumes 0 slots"""
        from app.core.quota import (
            record_provider_quota_exhausted, acquire_dispatch_slot,
            get_daily_requests_used
        )
        from app.core.db import set_setting
        set_setting("daily_request_budget_gemini", "10")

        now = datetime(2026, 8, 24, 10, 0, 0, tzinfo=timezone.utc)
        with patch("app.core.quota._utcnow", return_value=now):
            record_provider_quota_exhausted("gemini", "Blocked hard", retry_after_seconds=3600)
            allowed, info = acquire_dispatch_slot("gemini", job_id="blocked_job")
            assert allowed is False
            assert get_daily_requests_used("gemini") == 0, (
                "Blocked provider dispatch must consume 0 daily slots"
            )
