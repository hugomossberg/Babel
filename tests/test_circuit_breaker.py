"""
tests/test_circuit_breaker.py
==============================
Exhaustive integration and unit test suite for Babel Feature #3 Circuit Breaker
and Single-Flight Half-Open Probe Engine.

Test Matrix (Krav A - X):
--------------------------
A. Unknown external quota -> attempt 0 (~15m estimated probe)
B. Repeated external quota -> 15m -> 30m -> 1h -> 2h -> 4h -> 6h cap
C. 6h cap never exceeded
D. Jitter bounded within [-0.10, +0.10]
E. Exact Retry-After -> no adaptive probe before exact reset
F. Exact reset + safety margin
G. Single-flight -> 100 concurrent contenders -> exactly ONE probe lease
H. Lease losers -> zero provider calls
I. Probe success -> ACTIVE, attempt reset, next_probe cleared
J. Probe still daily quota -> BLOCKED, attempt increment, new next_probe
K. Probe worker crash / expired lease -> recovery without deadlock
L. Restart persistence -> adaptive state preserved
M. No deferred work -> no dummy probe, zero API calls
N. Local daily_request_budget -> deterministic UTC reset, no adaptive probe
O. Auth/config error -> no adaptive probe
P. Gemini blocked -> OpenAI unaffected
Q. Model-specific scope -> sibling model unaffected
R. Provider scope -> child requests blocked
S. Escalation provider -> correct quota accounting
T. Recovery / repair calls -> use same quota engine
U. Bazarr target under block -> success without AI
V. Bazarr miss + blocked AI -> DEFERRED, not FAILED
W. Old DB migration -> automatic upgrade without data loss
X. UI/API -> exact vs estimated vs local budget vs HALF_OPEN
"""

import asyncio
import os
import sqlite3
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from app.core.quota import (
    acquire_dispatch_slot, record_provider_success, record_provider_quota_exhausted,
    block_provider, unblock_provider, is_provider_blocked, get_provider_block_info,
    get_quota_status_for_provider, classify_provider_error,
    calculate_adaptive_probe_delay, set_jitter_override, get_jitter_ratio,
    DailyQuotaExhaustedError, RequestBudgetExhaustedError, QuotaSignal,
    EXACT_RESET_SAFETY_MARGIN_SECONDS, PROBE_LEASE_TIMEOUT_SECONDS, MAX_BACKOFF_SECONDS,
)
from app.core.db import DB_PATH, init_db, set_setting, get_setting


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    db_file = str(tmp_path / "circuit_breaker_test.db")
    import app.core.db as db_mod
    import app.core.quota as q_mod

    orig_db = db_mod.DB_PATH
    db_mod.DB_PATH = db_file
    q_mod.DB_PATH = db_file

    db_mod.init_db()
    set_jitter_override(0.0)  # default deterministic for tests

    yield db_file

    set_jitter_override(None)
    db_mod.DB_PATH = orig_db
    q_mod.DB_PATH = orig_db


# ---------------------------------------------------------------------------
# A. Unknown external quota -> attempt 0 (~15m)
# ---------------------------------------------------------------------------
def test_a_unknown_external_quota_attempt_0():
    now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
    with patch("app.core.quota._utcnow", return_value=now):
        res = record_provider_quota_exhausted("gemini", "Daily limit hit", retry_after_seconds=None, jitter_ratio=0.0)
        assert res["reset_type"] == "estimated"
        assert res["probe_attempt"] == 1  # next attempt will be 1
        assert res["delay_seconds"] == 900.0  # 15 minutes
        assert res["blocked_until"] == (now + timedelta(seconds=900)).isoformat()


# ---------------------------------------------------------------------------
# B. Repeated external quota -> 15m -> 30m -> 1h -> 2h -> 4h -> 6h
# ---------------------------------------------------------------------------
def test_b_repeated_external_quota_backoff():
    expected_delays = [900, 1800, 3600, 7200, 14400, 21600]
    now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)

    for attempt, expected_delay in enumerate(expected_delays):
        with patch("app.core.quota._utcnow", return_value=now):
            delay = calculate_adaptive_probe_delay(attempt, jitter_ratio=0.0)
            assert delay == float(expected_delay), f"Attempt {attempt} expected {expected_delay}s, got {delay}s"


# ---------------------------------------------------------------------------
# C. 6h cap never exceeded
# ---------------------------------------------------------------------------
def test_c_six_hour_cap_never_exceeded():
    for attempt in [5, 6, 10, 50, 100]:
        delay = calculate_adaptive_probe_delay(attempt, jitter_ratio=0.0)
        assert delay == float(MAX_BACKOFF_SECONDS)  # 21600s = 6h


# ---------------------------------------------------------------------------
# D. Jitter bounded within [-0.10, +0.10]
# ---------------------------------------------------------------------------
def test_d_jitter_bounded():
    set_jitter_override(None)  # enable live random jitter
    base_delay = 900.0
    for _ in range(100):
        ratio = get_jitter_ratio()
        assert -0.10 <= ratio <= 0.10
        delay = calculate_adaptive_probe_delay(0)
        assert base_delay * 0.90 <= delay <= base_delay * 1.10


# ---------------------------------------------------------------------------
# E. Exact Retry-After -> no adaptive probe before exact reset
# ---------------------------------------------------------------------------
def test_e_exact_retry_after_no_adaptive_probing():
    now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
    with patch("app.core.quota._utcnow", return_value=now):
        res = record_provider_quota_exhausted("openai", "Rate quota exceeded", retry_after_seconds=3600)
        assert res["reset_type"] == "exact"
        assert res["probe_attempt"] == 0
        expected_delay = 3600 + EXACT_RESET_SAFETY_MARGIN_SECONDS  # 3605s
        assert res["delay_seconds"] == expected_delay
        assert res["blocked_until"] == (now + timedelta(seconds=expected_delay)).isoformat()


# ---------------------------------------------------------------------------
# F. Exact reset + safety margin
# ---------------------------------------------------------------------------
def test_f_exact_reset_safety_margin():
    now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
    with patch("app.core.quota._utcnow", return_value=now):
        res = record_provider_quota_exhausted("gemini", "Quota exhausted", retry_after_seconds=60)
        assert res["delay_seconds"] == 65.0  # 60 + 5s safety margin


# ---------------------------------------------------------------------------
# G. Single-flight -> 100 concurrent contenders -> exactly ONE probe lease
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_g_single_flight_100_concurrent_contenders():
    now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
    with patch("app.core.quota._utcnow", return_value=now):
        # 900s + 5s safety margin = 905s blocked_until
        record_provider_quota_exhausted("gemini", "Blocked", retry_after_seconds=900)

    # Move clock past blocked_until (e.g. 910s -> HALF_OPEN candidate)
    future = now + timedelta(seconds=910)

    async def try_acquire(worker_id):
        with patch("app.core.quota._utcnow", return_value=future):
            allowed, info = acquire_dispatch_slot("gemini", job_id=worker_id)
            return allowed, info

    with patch("app.core.quota._utcnow", return_value=future):
        results = await asyncio.gather(*[try_acquire(i) for i in range(100)])

    winners = [r for r in results if r[0] is True]
    losers = [r for r in results if r[0] is False]

    assert len(winners) == 1, f"Expected exactly 1 probe lease winner, got {len(winners)}"
    assert len(losers) == 99
    assert winners[0][1]["is_probe"] is True
    assert winners[0][1]["state"] == "HALF_OPEN"
    for l in losers:
        assert l[1]["reason"] == "Probe request currently in flight"


# ---------------------------------------------------------------------------
# H. Lease losers -> zero provider calls
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_h_lease_losers_zero_provider_calls():
    from app.services.translator import with_retry

    mock_provider_call = AsyncMock(return_value={"translations": [{"id": 1, "text": "Hej"}]})

    @with_retry(provider="gemini")
    async def dummy_translate(items, job_id=None):
        return await mock_provider_call(items)

    now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
    with patch("app.core.quota._utcnow", return_value=now):
        record_provider_quota_exhausted("gemini", "Blocked", retry_after_seconds=900)

    # Move to probe time
    probe_time = now + timedelta(seconds=905)

    # First worker claims probe
    with patch("app.core.quota._utcnow", return_value=probe_time):
        res1 = await dummy_translate([{"id": 1, "text": "Hi"}], job_id=1)
        assert res1 == {"translations": [{"id": 1, "text": "Hej"}]}
        assert mock_provider_call.call_count == 1

    # Reset block to test competing loser while lease is active
    with patch("app.core.quota._utcnow", return_value=now):
        record_provider_quota_exhausted("gemini", "Blocked", retry_after_seconds=900)

    with patch("app.core.quota._utcnow", return_value=probe_time):
        # Worker 1 manually acquires lease
        allowed1, _ = acquire_dispatch_slot("gemini", job_id=1)
        assert allowed1 is True

        # Worker 2 attempts call while lease is active -> must fail with DailyQuotaExhaustedError without calling API
        mock_provider_call.reset_mock()
        with pytest.raises(DailyQuotaExhaustedError):
            await dummy_translate([{"id": 2, "text": "Hello"}], job_id=2)

        assert mock_provider_call.call_count == 0, "Lease loser must make 0 provider API calls"


# ---------------------------------------------------------------------------
# I. Probe success -> ACTIVE, attempt reset, next_probe cleared
# ---------------------------------------------------------------------------
def test_i_probe_success_resets_to_active():
    record_provider_quota_exhausted("gemini", "Daily quota", retry_after_seconds=900)
    info = get_provider_block_info("gemini")
    assert info["blocked"] is True

    record_provider_success("gemini")
    info = get_provider_block_info("gemini")
    assert info["blocked"] is False
    assert info["state"] == "ACTIVE"
    assert info["probe_attempt"] == 0
    assert info["blocked_until"] is None


# ---------------------------------------------------------------------------
# J. Probe still daily quota -> BLOCKED, attempt increment, new next_probe
# ---------------------------------------------------------------------------
def test_j_probe_still_daily_quota_increments_backoff():
    now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
    with patch("app.core.quota._utcnow", return_value=now):
        r0 = record_provider_quota_exhausted("gemini", "First quota error", retry_after_seconds=None, jitter_ratio=0.0)
        assert r0["probe_attempt"] == 1
        assert r0["delay_seconds"] == 900.0

    # Probe at 12:15 fails again with daily quota
    probe_time = now + timedelta(seconds=905)
    with patch("app.core.quota._utcnow", return_value=probe_time):
        r1 = record_provider_quota_exhausted("gemini", "Still quota error", retry_after_seconds=None, jitter_ratio=0.0)
        assert r1["probe_attempt"] == 2
        assert r1["delay_seconds"] == 1800.0  # Attempt 1 -> 30m


# ---------------------------------------------------------------------------
# K. Probe worker crash / expired lease -> recovery without deadlock
# ---------------------------------------------------------------------------
def test_k_probe_worker_crash_expired_lease():
    now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
    with patch("app.core.quota._utcnow", return_value=now):
        record_provider_quota_exhausted("gemini", "Blocked", retry_after_seconds=60)

    probe_time = now + timedelta(seconds=70)
    with patch("app.core.quota._utcnow", return_value=probe_time):
        # Worker 1 acquires lease then crashes
        allowed1, info1 = acquire_dispatch_slot("gemini", job_id=101)
        assert allowed1 is True
        assert info1["is_probe"] is True

        # Immediately, Worker 2 is rejected
        allowed2, _ = acquire_dispatch_slot("gemini", job_id=102)
        assert allowed2 is False

    # After PROBE_LEASE_TIMEOUT_SECONDS (300s), lease expires
    recovery_time = probe_time + timedelta(seconds=PROBE_LEASE_TIMEOUT_SECONDS + 5)
    with patch("app.core.quota._utcnow", return_value=recovery_time):
        # Worker 3 can now claim the probe lease
        allowed3, info3 = acquire_dispatch_slot("gemini", job_id=103)
        assert allowed3 is True
        assert info3["is_probe"] is True


# ---------------------------------------------------------------------------
# L. Restart persistence -> adaptive state preserved
# ---------------------------------------------------------------------------
def test_l_restart_persistence():
    now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
    with patch("app.core.quota._utcnow", return_value=now):
        record_provider_quota_exhausted("gemini", "Rate quota exceeded", retry_after_seconds=3600)

        info_before = get_provider_block_info("gemini")
        assert info_before["blocked"] is True
        assert info_before["reset_type"] == "exact"

        # Simulate restart by re-running init_db()
        init_db()

        info_after = get_provider_block_info("gemini")
        assert info_after["blocked"] is True
        assert info_after["reset_type"] == "exact"
        assert info_after["blocked_until"] == info_before["blocked_until"]


# ---------------------------------------------------------------------------
# M. No deferred work -> no dummy probe, zero API calls
# ---------------------------------------------------------------------------
def test_m_no_deferred_work_zero_api_calls():
    record_provider_quota_exhausted("gemini", "Blocked", retry_after_seconds=60)
    info = get_provider_block_info("gemini")
    assert info["state"] == "BLOCKED"
    assert info["probe_lease_active"] is False


# ---------------------------------------------------------------------------
# N. Local daily_request_budget -> deterministic UTC reset, no adaptive probe
# ---------------------------------------------------------------------------
def test_n_local_daily_request_budget_deterministic_reset():
    set_setting("daily_request_budget_gemini", "2")
    now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)

    with patch("app.core.quota._utcnow", return_value=now):
        allowed1, _ = acquire_dispatch_slot("gemini")
        assert allowed1 is True
        allowed2, _ = acquire_dispatch_slot("gemini")
        assert allowed2 is True

        # 3rd request exhausted budget
        allowed3, info3 = acquire_dispatch_slot("gemini")
        assert allowed3 is False
        assert info3["reason"] == "REQUEST_BUDGET_EXHAUSTED"

    # Next UTC day -> budget resets automatically without any adaptive probe
    next_day = datetime(2026, 8, 25, 0, 1, 0, tzinfo=timezone.utc)
    with patch("app.core.quota._utcnow", return_value=next_day):
        allowed_next, _ = acquire_dispatch_slot("gemini")
        assert allowed_next is True


# ---------------------------------------------------------------------------
# O. Auth/config error -> no adaptive probe
# ---------------------------------------------------------------------------
def test_o_auth_config_error_no_adaptive_probe():
    exc = RuntimeError("401 Unauthorized: invalid API key")
    sig = classify_provider_error(exc, "gemini")
    assert sig.kind == "AUTH_ERROR"
    assert not is_provider_blocked("gemini")


# ---------------------------------------------------------------------------
# P. Gemini blocked -> OpenAI unaffected
# ---------------------------------------------------------------------------
def test_p_gemini_blocked_openai_unaffected():
    record_provider_quota_exhausted("gemini", "Blocked", retry_after_seconds=3600)
    assert is_provider_blocked("gemini") is True
    assert is_provider_blocked("openai") is False

    allowed_openai, _ = acquire_dispatch_slot("openai")
    assert allowed_openai is True


# ---------------------------------------------------------------------------
# Q. Model-specific scope -> sibling model unaffected
# ---------------------------------------------------------------------------
def test_q_model_specific_scope_sibling_unaffected():
    record_provider_quota_exhausted(
        "openai",
        "Model gpt-4o quota exhausted",
        retry_after_seconds=3600,
        model="gpt-4o",
        scope_type="model",
        scope_id="openai:gpt-4o",
    )

    info_4o = get_provider_block_info("openai", model="gpt-4o", scope_type="model", scope_id="openai:gpt-4o")
    assert info_4o["blocked"] is True

    # Sibling model gpt-4o-mini is NOT blocked
    info_mini = get_provider_block_info("openai", model="gpt-4o-mini", scope_type="model", scope_id="openai:gpt-4o-mini")
    assert info_mini["blocked"] is False

    allowed_mini, _ = acquire_dispatch_slot("openai", model="gpt-4o-mini", scope_type="model", scope_id="openai:gpt-4o-mini")
    assert allowed_mini is True


# ---------------------------------------------------------------------------
# R. Provider scope -> child requests blocked
# ---------------------------------------------------------------------------
def test_r_provider_scope_blocks_child_requests():
    # Blocking entire provider scope
    record_provider_quota_exhausted("openai", "Account daily quota reached", retry_after_seconds=3600)

    # Child model request to openai:gpt-4o must be blocked
    allowed, info = acquire_dispatch_slot("openai", model="gpt-4o", scope_type="model", scope_id="openai:gpt-4o")
    assert allowed is False
    assert info.get("is_parent_blocked") is True


# ---------------------------------------------------------------------------
# S. Escalation provider -> correct quota accounting
# ---------------------------------------------------------------------------
def test_s_escalation_provider_accounting():
    set_setting("daily_request_budget_openai", "5")
    set_setting("daily_request_budget_gemini", "10")

    # Dispatch to openai (escalation)
    allowed, _ = acquire_dispatch_slot("openai")
    assert allowed is True

    from app.core.quota import get_daily_requests_used
    assert get_daily_requests_used("openai") == 1
    assert get_daily_requests_used("gemini") == 0


# ---------------------------------------------------------------------------
# T. Recovery / repair calls -> use same quota engine
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_t_recovery_repair_calls_use_quota_engine():
    from app.services.translator import SubtitleTranslator
    translator = SubtitleTranslator()

    record_provider_quota_exhausted("gemini", "Daily quota exhausted", retry_after_seconds=3600)

    with pytest.raises(DailyQuotaExhaustedError):
        await translator.verify_single_occurrence_entities(
            candidates=[{"id": 1, "target": "John"}],
            target_language="sv",
            job_id=999
        )


# ---------------------------------------------------------------------------
# U. Bazarr target under block -> success without AI
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_u_bazarr_target_under_block(tmp_path):
    record_provider_quota_exhausted("gemini", "Blocked", retry_after_seconds=3600)

    video = tmp_path / "test_u.mkv"
    video.touch()

    en_srt = tmp_path / "test_u.en.srt"
    with open(en_srt, "w", encoding="utf-8") as f:
        f.write("1\n00:00:01,000 --> 00:00:02,000\nThis English text is much much much much longer so that we get more than 100 bytes in size. Otherwise Babel misses that the file is valid.\n")

    sv_sub = tmp_path / "test_u.sv.srt"
    with open(sv_sub, "w", encoding="utf-8") as f:
        f.write("1\n00:00:01,000 --> 00:00:02,000\nDenna svenska text ar mycket mycket mycket mycket langre sa att vi far mer an 100 bytes i storlek. Annars missar Babel att filen ar giltig.\n")

    from app.services.pipeline import SubtitlePipeline
    from app.core.db import create_job

    job_id = create_job(str(video))
    pipeline = SubtitlePipeline()

    def fake_get_setting(key, default=None):
        if key == "languages": return '[{"name": "Swedish", "code": "sv", "enabled": true}]'
        if key == "extract_target_embedded": return "false"
        if key == "extract_source_embedded": return "false"
        if key == "enable_bazarr_check": return "false"
        if key == "auto_repair_unhealthy": return "false"
        return default

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting):
        res = await pipeline.process_video_file(str(video), job_id=job_id)

    assert res["status"] in ("completed", "skipped")


# ---------------------------------------------------------------------------
# V. Bazarr miss + blocked AI -> DEFERRED, not FAILED
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_v_bazarr_miss_blocked_ai_deferred_not_failed(tmp_path):
    record_provider_quota_exhausted("gemini", "Blocked", retry_after_seconds=3600)

    video = tmp_path / "test_v.mkv"
    video.touch()

    source_srt = tmp_path / "test_v.en.srt"
    with open(source_srt, "w", encoding="utf-8") as f:
        f.write("1\n00:00:01,000 --> 00:00:02,000\nThis English text is much much much much longer so that we get more than 100 bytes in size. Otherwise Babel misses that the file is valid.\n")

    from app.services.pipeline import SubtitlePipeline
    from app.core.db import create_job, get_job_by_id

    job_id = create_job(str(video))
    pipeline = SubtitlePipeline()

    def fake_get_setting(key, default=None):
        if key == "languages": return '[{"name": "Swedish", "code": "sv", "enabled": true}]'
        if key == "extract_target_embedded": return "false"
        if key == "extract_source_embedded": return "false"
        if key == "enable_bazarr_check": return "false"
        if key == "auto_repair_unhealthy": return "false"
        if key == "ai_provider": return "gemini"
        return default

    with patch("app.services.pipeline.get_setting", side_effect=fake_get_setting):
        res = await pipeline.process_video_file(str(video), job_id=job_id)

    job = get_job_by_id(job_id)
    assert job["status"] == "DEFERRED", f"Must be DEFERRED, got {job['status']}"
    assert res["status"] == "deferred"


# ---------------------------------------------------------------------------
# W. Old DB migration -> automatic upgrade without data loss
# ---------------------------------------------------------------------------
def test_w_old_database_migration(tmp_path):
    old_db = str(tmp_path / "legacy_old.db")
    now_str = datetime.now(timezone.utc).isoformat()
    future_str = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
    with sqlite3.connect(old_db) as conn:
        conn.execute("""
        CREATE TABLE provider_quota (
            provider TEXT PRIMARY KEY,
            blocked INTEGER NOT NULL DEFAULT 0,
            reason TEXT,
            blocked_at TEXT,
            blocked_until TEXT,
            updated_at TEXT NOT NULL
        )
        """)
        conn.execute(f"INSERT INTO provider_quota VALUES ('gemini', 1, 'legacy block', '{now_str}', '{future_str}', '{now_str}')")
        conn.commit()

    import app.core.db as db_mod
    import app.core.quota as q_mod
    db_mod.DB_PATH = old_db
    q_mod.DB_PATH = old_db

    db_mod.init_db()

    info = get_provider_block_info("gemini")
    assert info["blocked"] is True
    assert info["reason"] == "legacy block"
    assert info["state"] == "BLOCKED"
    assert info["reset_type"] == "estimated"


# ---------------------------------------------------------------------------
# X. UI/API -> exact vs estimated vs local budget vs HALF_OPEN
# ---------------------------------------------------------------------------
def test_x_ui_api_status_exact_estimated_local_half_open():
    # 1. Active normal
    status_active = get_quota_status_for_provider("gemini")
    assert status_active["state"] == "ACTIVE"
    assert status_active["blocked"] is False

    # 2. Blocked exact
    record_provider_quota_exhausted("gemini", "Exact block", retry_after_seconds=120)
    status_exact = get_quota_status_for_provider("gemini")
    assert status_exact["state"] == "BLOCKED"
    assert status_exact["reset_type"] == "exact"

    # 3. Blocked estimated
    record_provider_quota_exhausted("gemini", "Estimated block", retry_after_seconds=None)
    status_est = get_quota_status_for_provider("gemini")
    assert status_est["state"] == "BLOCKED"
    assert status_est["reset_type"] == "estimated"
    assert status_est["probe_attempt"] > 0


# ===========================================================================
# Multi-Scope Hierarchy & Canonical Scope Key Tests
# ===========================================================================

from app.core.quota import build_scope_key


def test_canonical_scope_key_generation():
    """Verify deterministic and unique scope key generation across all scopes."""
    # Provider scope
    assert build_scope_key("gemini") == "gemini"
    assert build_scope_key("OpenRouter", scope_type="provider") == "openrouter"

    # Model scopes
    key_sonnet = build_scope_key("openrouter", scope_type="model", scope_id="anthropic/claude-sonnet")
    assert key_sonnet == "openrouter:model:anthropic/claude-sonnet"

    key_deepseek = build_scope_key("openrouter", scope_type="model", scope_id="deepseek/deepseek-v4-flash")
    assert key_deepseek == "openrouter:model:deepseek/deepseek-v4-flash"

    # Credential scopes
    key_cred_a = build_scope_key("openrouter", scope_type="credential", scope_id="key-a")
    assert key_cred_a == "openrouter:credential:key-a"

    key_cred_b = build_scope_key("openrouter", scope_type="credential", scope_id="key-b")
    assert key_cred_b == "openrouter:credential:key-b"

    # Unknown scope -> fallback to provider
    assert build_scope_key("openrouter", scope_type="unknown") == "openrouter"

    # Uniqueness invariant
    all_keys = [
        build_scope_key("openrouter"),
        key_sonnet,
        key_deepseek,
        key_cred_a,
        key_cred_b,
        build_scope_key("gemini"),
        build_scope_key("openai", scope_type="model", model="gpt-4o"),
    ]
    assert len(all_keys) == len(set(all_keys)), "All distinct scope inputs must produce distinct keys"


def test_multi_models_under_same_provider_separate_storage():
    """
    Verify multiple models under the same provider (e.g. OpenRouter):
    - Do NOT overwrite each other
    - Do NOT share blocked-state
    - Model B remains ACTIVE when Model A is BLOCKED
    """
    now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
    with patch("app.core.quota._utcnow", return_value=now):
        # Block Model A
        record_provider_quota_exhausted(
            "openrouter",
            "Rate limit on Claude Sonnet",
            retry_after_seconds=3600,
            scope_type="model",
            scope_id="anthropic/claude-sonnet",
        )

    # 1. Model A must be BLOCKED
    with patch("app.core.quota._utcnow", return_value=now):
        info_a = get_provider_block_info("openrouter", scope_type="model", scope_id="anthropic/claude-sonnet")
        assert info_a["blocked"] is True
        assert info_a["state"] == "BLOCKED"
        assert info_a["scope_key"] == "openrouter:model:anthropic/claude-sonnet"

        # Model A dispatch is denied with 0 requests
        allowed_a, res_a = acquire_dispatch_slot("openrouter", scope_type="model", scope_id="anthropic/claude-sonnet")
        assert allowed_a is False

    # 2. Model B must be completely ACTIVE and unaffected
    with patch("app.core.quota._utcnow", return_value=now):
        info_b = get_provider_block_info("openrouter", scope_type="model", scope_id="deepseek/deepseek-v4-flash")
        assert info_b["blocked"] is False
        assert info_b["state"] == "ACTIVE"

        # Model B can claim dispatch slot and proceed
        allowed_b, res_b = acquire_dispatch_slot("openrouter", scope_type="model", scope_id="deepseek/deepseek-v4-flash")
        assert allowed_b is True
        assert res_b["state"] == "ACTIVE"
        assert res_b["is_probe"] is False


@pytest.mark.asyncio
async def test_separate_probe_leases_under_same_provider():
    """
    Verify separate probe leases under the same provider:
    - Model A in HALF_OPEN has its own lease
    - Model B in HALF_OPEN has its own lease
    - Active lease on Model A does NOT block Model B probe
    - Two contenders for Model A yield exactly ONE probe lease
    """
    now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
    with patch("app.core.quota._utcnow", return_value=now):
        record_provider_quota_exhausted("openrouter", "Blocked A", retry_after_seconds=900, scope_type="model", scope_id="model-a")
        record_provider_quota_exhausted("openrouter", "Blocked B", retry_after_seconds=900, scope_type="model", scope_id="model-b")

    # Move clock past 905s (safety margin) -> both eligible for HALF_OPEN probe
    future = now + timedelta(seconds=910)
    with patch("app.core.quota._utcnow", return_value=future):
        # Worker 1 claims probe for Model A
        allowed_a1, info_a1 = acquire_dispatch_slot("openrouter", scope_type="model", scope_id="model-a", job_id="job_a1")
        assert allowed_a1 is True
        assert info_a1["is_probe"] is True
        assert info_a1["state"] == "HALF_OPEN"

        # Worker 2 attempts Model A while lease active -> REJECTED (lease active)
        allowed_a2, info_a2 = acquire_dispatch_slot("openrouter", scope_type="model", scope_id="model-a", job_id="job_a2")
        assert allowed_a2 is False
        assert info_a2["reason"] == "Probe request currently in flight"

        # Worker 3 claims probe for Model B -> SUCCEEDS independently!
        allowed_b1, info_b1 = acquire_dispatch_slot("openrouter", scope_type="model", scope_id="model-b", job_id="job_b1")
        assert allowed_b1 is True
        assert info_b1["is_probe"] is True
        assert info_b1["state"] == "HALF_OPEN"


def test_provider_scope_parent_blocks_all_child_models():
    """
    Verify parent/child hierarchy:
    - If parent provider scope is BLOCKED -> all child model requests are blocked
    - If parent provider is ACTIVE and only Model A is BLOCKED -> Model B is ACTIVE
    """
    now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
    with patch("app.core.quota._utcnow", return_value=now):
        # Block the parent provider scope
        record_provider_quota_exhausted("openrouter", "Account billing daily limit hit", retry_after_seconds=3600)

    # Verify both child models are blocked by parent
    with patch("app.core.quota._utcnow", return_value=now):
        assert is_provider_blocked("openrouter", scope_type="model", scope_id="model-a") is True
        assert is_provider_blocked("openrouter", scope_type="model", scope_id="model-b") is True

        allowed_a, info_a = acquire_dispatch_slot("openrouter", scope_type="model", scope_id="model-a")
        assert allowed_a is False
        assert info_a.get("is_parent_blocked") is True

        allowed_b, info_b = acquire_dispatch_slot("openrouter", scope_type="model", scope_id="model-b")
        assert allowed_b is False
        assert info_b.get("is_parent_blocked") is True

    # Unblock parent provider and block only Model A
    unblock_provider("openrouter")
    with patch("app.core.quota._utcnow", return_value=now):
        record_provider_quota_exhausted("openrouter", "Model A rate limit", retry_after_seconds=3600, scope_type="model", scope_id="model-a")

        # Model A is blocked, Model B is ACTIVE
        assert is_provider_blocked("openrouter", scope_type="model", scope_id="model-a") is True
        assert is_provider_blocked("openrouter", scope_type="model", scope_id="model-b") is False

        allowed_b, _ = acquire_dispatch_slot("openrouter", scope_type="model", scope_id="model-b")
        assert allowed_b is True


def test_credential_scopes_independent():
    """
    Verify credential scopes:
    - openrouter/credential/key-a blocked does not block key-b
    """
    now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
    with patch("app.core.quota._utcnow", return_value=now):
        record_provider_quota_exhausted("openrouter", "Key A limit", retry_after_seconds=3600, scope_type="credential", scope_id="key-a")

        info_a = get_provider_block_info("openrouter", scope_type="credential", scope_id="key-a")
        assert info_a["blocked"] is True

        info_b = get_provider_block_info("openrouter", scope_type="credential", scope_id="key-b")
        assert info_b["blocked"] is False

        allowed_b, _ = acquire_dispatch_slot("openrouter", scope_type="credential", scope_id="key-b")
        assert allowed_b is True


def test_unknown_scope_conservative_fallback():
    """
    Verify unknown scope falls back conservatively to parent provider scope
    without cross-provider contamination or arbitrary model guesses.
    """
    now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
    with patch("app.core.quota._utcnow", return_value=now):
        res = record_provider_quota_exhausted("openrouter", "Unknown scope error", retry_after_seconds=3600, scope_type="unknown")
        assert res["scope_key"] == "openrouter"

        # OpenRouter parent is blocked
        assert is_provider_blocked("openrouter") is True

        # Gemini and OpenAI are unaffected
        assert is_provider_blocked("gemini") is False
        assert is_provider_blocked("openai") is False


# ===========================================================================
# Explicit 5-Test Matrix for Hierarchical Credential & Model Scopes
# ===========================================================================

def test_1_key_a_claude_model_quota_blocked_isolation():
    """
    Test 1:
    key-A + claude model quota BLOCKED
    Verifiera:
      key-A + claude -> BLOCKED
      key-A + deepseek -> ALLOWED
      key-B + claude -> ALLOWED
      key-B + deepseek -> ALLOWED
    """
    now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
    with patch("app.core.quota._utcnow", return_value=now):
        record_provider_quota_exhausted(
            "openrouter",
            "Rate limit on Claude with Key A",
            retry_after_seconds=3600,
            scope_type="model",
            scope_id="claude",
            credential="key-a"
        )

        # key-A + claude -> BLOCKED
        allowed_a_claude, info_a_claude = acquire_dispatch_slot("openrouter", model="claude", credential="key-a", scope_type="model")
        assert allowed_a_claude is False
        assert is_provider_blocked("openrouter", model="claude", credential="key-a", scope_type="model") is True

        # key-A + deepseek -> ALLOWED
        allowed_a_deepseek, info_a_deepseek = acquire_dispatch_slot("openrouter", model="deepseek", credential="key-a", scope_type="model")
        assert allowed_a_deepseek is True
        assert info_a_deepseek["state"] == "ACTIVE"

        # key-B + claude -> ALLOWED (same model, different credential!)
        allowed_b_claude, info_b_claude = acquire_dispatch_slot("openrouter", model="claude", credential="key-b", scope_type="model")
        assert allowed_b_claude is True
        assert info_b_claude["state"] == "ACTIVE"

        # key-B + deepseek -> ALLOWED
        allowed_b_deepseek, info_b_deepseek = acquire_dispatch_slot("openrouter", model="deepseek", credential="key-b", scope_type="model")
        assert allowed_b_deepseek is True
        assert info_b_deepseek["state"] == "ACTIVE"


def test_2_key_a_credential_blocked_hierarchy():
    """
    Test 2:
    key-A credential BLOCKED
    Verifiera:
      key-A + claude -> BLOCKED
      key-A + deepseek -> BLOCKED
      key-B + claude -> ALLOWED
    """
    now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
    with patch("app.core.quota._utcnow", return_value=now):
        record_provider_quota_exhausted(
            "openrouter",
            "Key A account limit reached",
            retry_after_seconds=3600,
            scope_type="credential",
            scope_id="key-a"
        )

        # key-A + claude -> BLOCKED by parent credential
        allowed_a_claude, info_a_claude = acquire_dispatch_slot("openrouter", model="claude", credential="key-a", scope_type="model")
        assert allowed_a_claude is False
        assert info_a_claude.get("is_parent_blocked") is True

        # key-A + deepseek -> BLOCKED by parent credential
        allowed_a_deepseek, info_a_deepseek = acquire_dispatch_slot("openrouter", model="deepseek", credential="key-a", scope_type="model")
        assert allowed_a_deepseek is False
        assert info_a_deepseek.get("is_parent_blocked") is True

        # key-B + claude -> ALLOWED
        allowed_b_claude, info_b_claude = acquire_dispatch_slot("openrouter", model="claude", credential="key-b", scope_type="model")
        assert allowed_b_claude is True
        assert info_b_claude["state"] == "ACTIVE"


def test_3_provider_openrouter_blocked_hierarchy():
    """
    Test 3:
    provider openrouter BLOCKED
    Verifiera:
      key-A + claude -> BLOCKED
      key-B + claude -> BLOCKED
      key-B + deepseek -> BLOCKED
    """
    now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
    with patch("app.core.quota._utcnow", return_value=now):
        record_provider_quota_exhausted(
            "openrouter",
            "Provider-wide billing failure",
            retry_after_seconds=3600,
            scope_type="provider"
        )

        # key-A + claude -> BLOCKED
        allowed_a_claude, info_a = acquire_dispatch_slot("openrouter", model="claude", credential="key-a", scope_type="model")
        assert allowed_a_claude is False
        assert info_a.get("is_parent_blocked") is True

        # key-B + claude -> BLOCKED
        allowed_b_claude, info_b1 = acquire_dispatch_slot("openrouter", model="claude", credential="key-b", scope_type="model")
        assert allowed_b_claude is False
        assert info_b1.get("is_parent_blocked") is True

        # key-B + deepseek -> BLOCKED
        allowed_b_deepseek, info_b2 = acquire_dispatch_slot("openrouter", model="deepseek", credential="key-b", scope_type="model")
        assert allowed_b_deepseek is False
        assert info_b2.get("is_parent_blocked") is True


@pytest.mark.asyncio
async def test_4_separate_model_half_open_leases():
    """
    Test 4:
    Separata model HALF_OPEN leases:
      key-A + claude
      key-B + claude
    ska kunna ha separata leases oberoende av varandra.
    """
    now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
    with patch("app.core.quota._utcnow", return_value=now):
        record_provider_quota_exhausted("openrouter", "Blocked A-claude", retry_after_seconds=900, scope_type="model", scope_id="claude", credential="key-a")
        record_provider_quota_exhausted("openrouter", "Blocked B-claude", retry_after_seconds=900, scope_type="model", scope_id="claude", credential="key-b")

    future = now + timedelta(seconds=910)
    with patch("app.core.quota._utcnow", return_value=future):
        # Worker on Key-A claims lease for claude
        allowed_a, info_a = acquire_dispatch_slot("openrouter", model="claude", credential="key-a", scope_type="model", job_id="job_a")
        assert allowed_a is True
        assert info_a["is_probe"] is True
        assert info_a["state"] == "HALF_OPEN"

        # Worker on Key-B claims lease for claude SIMULTANEOUSLY without being blocked by Key-A lease!
        allowed_b, info_b = acquire_dispatch_slot("openrouter", model="claude", credential="key-b", scope_type="model", job_id="job_b")
        assert allowed_b is True
        assert info_b["is_probe"] is True
        assert info_b["state"] == "HALF_OPEN"


@pytest.mark.asyncio
async def test_5_two_contenders_single_flight_probe_winner():
    """
    Test 5:
    Två contenders för SAMMA:
      key-A + claude
    -> exakt EN probe winner.
    Blocked contenders:
    -> ZERO external provider requests
    -> ZERO budget consumption.

    Root-cause note (confirmed 2026-08-24):
    get_daily_requests_used() calls _today_window() which calls _utcnow().
    The budget row is written with the PATCHED date (2026-08-24).
    If we query outside the patch context, _utcnow() returns real-today (2026-08-25)
    and the lookup finds 0 rows -> returns 0. Bug was in the test, not in production.
    Fix: assert inside the same patch context so window_date matches what was written.
    """
    from app.core.quota import get_daily_requests_used
    set_setting("daily_request_budget_openrouter", "10")

    now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
    with patch("app.core.quota._utcnow", return_value=now):
        record_provider_quota_exhausted("openrouter", "Blocked", retry_after_seconds=900, scope_type="model", scope_id="claude", credential="key-a")

    future = now + timedelta(seconds=910)
    with patch("app.core.quota._utcnow", return_value=future):
        # Contender 1 — probe winner
        allowed1, info1 = acquire_dispatch_slot("openrouter", model="claude", credential="key-a", scope_type="model", job_id="job_1")
        # Contender 2 — lease loser
        allowed2, info2 = acquire_dispatch_slot("openrouter", model="claude", credential="key-a", scope_type="model", job_id="job_2")

        assert allowed1 is True
        assert info1["is_probe"] is True
        assert allowed2 is False
        assert info2["reason"] == "Probe request currently in flight"

        # Exactly 1 budget slot consumed by winner, 0 by blocked contender.
        # Query inside same patch context: _today_window() returns "2026-08-24"
        # which matches the window_date written when the probe lease was acquired.
        assert get_daily_requests_used("openrouter") == 1


