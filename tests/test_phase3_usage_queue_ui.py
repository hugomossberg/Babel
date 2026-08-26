"""
tests/test_phase3_usage_queue_ui.py
===================================
Phase 3: Comprehensive tests for AI Usage & Queue Top-Level UI, Deferred Queue API,
Active vs Unsaved Quota semantics, API Key visibility, and Job Details AI Usage.
"""

import pytest
import sqlite3
import re
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from app.main import app
import app.core.db as db_module
import app.core.quota as quota_module
import app.core.usage as usage_module
from app.core.usage import generate_request_uid, record_dispatch, complete_dispatch, UsageStage, UsageStatus


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    db_file = str(tmp_path / "babel_phase3_test.db")
    orig_db_core = db_module.DB_PATH
    orig_db_quota = quota_module.DB_PATH
    orig_db_usage = usage_module.DB_PATH

    db_module.DB_PATH = db_file
    quota_module.DB_PATH = db_file
    usage_module.DB_PATH = db_file

    db_module.init_db()

    yield db_file

    db_module.DB_PATH = orig_db_core
    quota_module.DB_PATH = orig_db_quota
    usage_module.DB_PATH = orig_db_usage


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. API Endpoints: Queue / Deferred
# ---------------------------------------------------------------------------

def test_deferred_queue_empty(client):
    """When no DEFERRED jobs exist, /api/queue/deferred returns empty list."""
    res = client.get("/api/queue/deferred")
    assert res.status_code == 200
    assert res.json() == []

    # Alias /api/queue also returns empty list
    res_alias = client.get("/api/queue")
    assert res_alias.status_code == 200
    assert res_alias.json() == []


def test_deferred_queue_fifo_order_and_metadata(client):
    """Deferred queue returns jobs in FIFO order by deferred_at / created_at."""
    t0 = datetime(2026, 8, 24, 10, 0, 0, tzinfo=timezone.utc).isoformat()
    t1 = datetime(2026, 8, 24, 10, 5, 0, tzinfo=timezone.utc).isoformat()
    t2 = datetime(2026, 8, 24, 10, 10, 0, tzinfo=timezone.utc).isoformat()

    with sqlite3.connect(db_module.DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO jobs (id, title, video_path, status, waiting_provider, waiting_model, defer_stage, defer_reason, deferred_at, created_at, updated_at)
            VALUES (10, 'Job Middle', '/tv/Show/s01e02.mkv', 'DEFERRED', 'gemini', 'gemini-3.5-flash-lite', 'PRIMARY', 'INSUFFICIENT_LOCAL_BUDGET', ?, ?, ?)
            """,
            (t1, t1, t1)
        )
        conn.execute(
            """
            INSERT INTO jobs (id, title, video_path, status, waiting_provider, waiting_model, defer_stage, defer_reason, deferred_at, created_at, updated_at)
            VALUES (20, 'Job Oldest', '/tv/Show/s01e01.mkv', 'DEFERRED', 'openai', 'gpt-4o-mini', 'MICRO_REPAIR', 'LOCAL_RPD', ?, ?, ?)
            """,
            (t0, t0, t0)
        )
        conn.execute(
            """
            INSERT INTO jobs (id, title, video_path, status, waiting_provider, waiting_model, defer_stage, defer_reason, deferred_at, created_at, updated_at)
            VALUES (30, 'Job Newest', '/tv/Show/s01e03.mkv', 'DEFERRED', 'gemini', 'gemini-3.5-flash-lite', 'PRIMARY', 'PROVIDER_QUOTA', ?, ?, ?)
            """,
            (t2, t2, t2)
        )
        conn.execute(
            """
            INSERT INTO jobs (id, title, video_path, status, created_at, updated_at)
            VALUES (40, 'Job Translated', '/tv/Show/s01e04.mkv', 'TRANSLATED', ?, ?)
            """,
            (t0, t0)
        )
        conn.commit()

    res = client.get("/api/queue/deferred")
    assert res.status_code == 200
    queue = res.json()

    assert len(queue) == 3
    # FIFO order: Oldest (id 20) -> Middle (id 10) -> Newest (id 30)
    assert queue[0]["id"] == 20
    assert queue[0]["title"] == "Job Oldest"
    assert queue[0]["waiting_provider"] == "openai"
    assert queue[0]["defer_reason"] == "LOCAL_RPD"
    assert queue[0]["defer_stage"] == "MICRO_REPAIR"

    assert queue[1]["id"] == 10
    assert queue[1]["title"] == "Job Middle"
    assert queue[1]["waiting_provider"] == "gemini"
    assert queue[1]["defer_reason"] == "INSUFFICIENT_LOCAL_BUDGET"

    assert queue[2]["id"] == 30
    assert queue[2]["title"] == "Job Newest"
    assert queue[2]["defer_reason"] == "PROVIDER_QUOTA"


# ---------------------------------------------------------------------------
# 2. Historical Stats & Sample Size Threshold
# ---------------------------------------------------------------------------

def test_historical_stats_sample_threshold(client):
    """Historical stats indicates has_sufficient_history based on MIN_SAMPLE_THRESHOLD = 3."""
    res = client.get("/api/usage/stats")
    assert res.status_code == 200
    data = res.json()
    assert data["completed_jobs_with_ai"] == 0
    assert data["has_sufficient_history"] is False
    assert data["min_sample_threshold"] == 3

    with sqlite3.connect(db_module.DB_PATH) as conn:
        conn.execute("INSERT INTO jobs (id, title, video_path, status, created_at, updated_at) VALUES (1, 'Ep1', '/tv/Show/s01e01.mkv', 'TRANSLATED', '2026-08-24T10:00:00', '2026-08-24T10:00:00')")
        conn.execute("INSERT INTO jobs (id, title, video_path, status, created_at, updated_at) VALUES (2, 'Ep2', '/tv/Show/s01e02.mkv', 'TRANSLATED', '2026-08-24T10:00:00', '2026-08-24T10:00:00')")
        conn.commit()

    uid1 = generate_request_uid()
    record_dispatch(uid1, "gemini", "gemini-3.5-flash-lite", UsageStage.PRIMARY, job_id=1)
    complete_dispatch(uid1, UsageStatus.SUCCESS, input_tokens=1000, output_tokens=200, estimated_cost_usd=0.001)

    uid2 = generate_request_uid()
    record_dispatch(uid2, "gemini", "gemini-3.5-flash-lite", UsageStage.PRIMARY, job_id=2)
    complete_dispatch(uid2, UsageStatus.SUCCESS, input_tokens=1000, output_tokens=200, estimated_cost_usd=0.001)

    res = client.get("/api/usage/stats")
    data = res.json()
    assert data["completed_jobs_with_ai"] == 2
    assert data["has_sufficient_history"] is False

    with sqlite3.connect(db_module.DB_PATH) as conn:
        conn.execute("INSERT INTO jobs (id, title, video_path, status, created_at, updated_at) VALUES (3, 'Ep3', '/tv/Show/s01e03.mkv', 'TRANSLATED', '2026-08-24T10:00:00', '2026-08-24T10:00:00')")
        conn.commit()

    uid3 = generate_request_uid()
    record_dispatch(uid3, "gemini", "gemini-3.5-flash-lite", UsageStage.PRIMARY, job_id=3)
    complete_dispatch(uid3, UsageStatus.SUCCESS, input_tokens=1000, output_tokens=200, estimated_cost_usd=0.001)

    res = client.get("/api/usage/stats")
    data = res.json()
    assert data["completed_jobs_with_ai"] == 3
    assert data["has_sufficient_history"] is True
    assert data["average_calls_per_job"] == 1.0


# ---------------------------------------------------------------------------
# 3. Settings: Daily Request Limit Save & Quota State
# ---------------------------------------------------------------------------

def test_daily_request_limit_saved_and_quota_status(client):
    """Daily request budget configures correctly and is returned by /api/quota."""
    res = client.post("/api/settings/ai", json={"daily_request_budget_gemini": 10, "ai_provider": "gemini"})
    assert res.status_code == 200

    q_res = client.get("/api/quota")
    assert q_res.status_code == 200
    q_data = q_res.json()
    assert q_data["providers"]["gemini"]["budget"] == 10
    assert q_data["providers"]["gemini"]["requests_today"] == 0
    assert q_data["providers"]["gemini"]["requests_remaining"] == 10

    res = client.post("/api/settings/ai", json={"daily_request_budget_gemini": 0, "ai_provider": "gemini"})
    assert res.status_code == 200

    q_res = client.get("/api/quota")
    q_data = q_res.json()
    assert q_data["providers"]["gemini"]["budget"] is None
    assert q_data["providers"]["gemini"]["requests_remaining"] is None


# ---------------------------------------------------------------------------
# 4. Top-Level UI Template Navigation & Layout Verification
# ---------------------------------------------------------------------------

def test_ui_top_level_navigation_structure():
    """Verify Usage & Queue is in top-level nav between Library and Settings, and NOT in Settings sidebar."""
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    # Requirement A: Usage & Queue in top-level nav
    assert "currentTab = 'usage'" in html
    assert "Usage & Queue" in html

    # Requirement B: Nav order: Activity -> Library -> Usage & Queue -> Settings
    nav_match = re.search(r'<nav[^>]*>(.*?)</nav>', html, re.DOTALL)
    assert nav_match is not None, "Top nav element not found"
    nav_content = nav_match.group(1)

    idx_dashboard = nav_content.find("currentTab = 'dashboard'")
    idx_media = nav_content.find("currentTab = 'media'")
    idx_usage = nav_content.find("currentTab = 'usage'")
    idx_settings = nav_content.find("currentTab = 'settings'")

    assert idx_dashboard != -1, "Activity nav button missing"
    assert idx_media != -1, "Library nav button missing"
    assert idx_usage != -1, "Usage & Queue nav button missing"
    assert idx_settings != -1, "Settings nav button missing"
    assert idx_dashboard < idx_media < idx_usage < idx_settings, "Top nav buttons must be: Activity -> Library -> Usage & Queue -> Settings"

    # Requirement C: NOT in Settings sidebar
    assert "@click=\"settingsTab = 'usage'\"" not in html
    assert "x-show=\"settingsTab === 'usage'\"" not in html

    # Top-level section for usage exists
    assert "x-show=\"currentTab === 'usage'\"" in html


def test_ui_top_summary_cards_on_top_level_page():
    """Top-level Usage & Queue page contains all 4 summary cards."""
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    assert "Requests Today" in html
    assert "Daily Request Limit" in html
    assert "Remaining" in html
    assert "Estimated Cost Today" in html
    assert "getActiveProviderQuota()" in html
    assert "formatCost(todayUsage.total ? todayUsage.total.estimated_cost_today : null)" in html


def test_ui_daily_limit_configuration_on_top_level_page():
    """Top-level Usage & Queue page contains Daily Request Limit configuration using daily_request_budget_."""
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    assert "Daily Request Limit" in html
    assert "daily_request_budget_" in html
    assert "Save Limit" in html


def test_ui_today_breakdown_and_smart_capacity():
    """index.html contains today's provider breakdown and smart capacity estimate."""
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    assert "Today's Usage by Provider" in html
    assert "Capacity Estimate" in html
    assert "usageStats.has_sufficient_history" in html
    assert "Not enough history" in html
    assert "Avg AI calls" in html
    assert "Estimated episodes remaining today" in html


def test_ui_deferred_queue_section():
    """index.html contains deferred queue list and empty state."""
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    assert "Deferred Jobs Queue" in html
    assert "Queue is empty" in html
    assert "deferredQueue" in html
    assert "getDeferReasonText(job.defer_reason)" in html


def test_ui_job_details_modal_ai_usage():
    """index.html Job Details modal contains AI Usage section and token summary."""
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    assert "AI Usage" in html
    assert "No AI usage recorded" in html
    assert "loadingJobUsage" in html
    assert "jobUsage.total_calls" in html
    assert "getStageLabel(item.stage)" in html
    assert "formatTokens(jobUsage.total_input_tokens)" in html
    assert "formatTokens(jobUsage.total_cached_input_tokens)" in html
    assert "formatTokens(jobUsage.total_output_tokens)" in html


def test_ui_api_key_visibility_logic():
    """index.html API key visibility respects primary provider and escalate_to_pro."""
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    # Gemini key condition
    gemini_pattern = r'settingsData\.ai\.ai_provider === [\'"]gemini[\'"]\s*\|\|\s*\(settingsData\.ai\.escalate_to_pro\s*&&\s*settingsData\.ai\.escalation_provider === [\'"]gemini[\'"]\)'
    assert re.search(gemini_pattern, html), "Gemini API key visibility must check escalate_to_pro"

    # OpenAI key condition
    openai_pattern = r'settingsData\.ai\.ai_provider === [\'"]openai[\'"]\s*\|\|\s*\(settingsData\.ai\.escalate_to_pro\s*&&\s*settingsData\.ai\.escalation_provider === [\'"]openai[\'"]\)'
    assert re.search(openai_pattern, html), "OpenAI API key visibility must check escalate_to_pro"


def test_ui_js_methods_exist():
    """index.html babelApp() contains all Phase 3 required methods and formatters."""
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    assert "formatCost(val)" in html
    assert "formatTokens(n)" in html
    assert "getStageLabel(stage)" in html
    assert "getDeferReasonText(reason)" in html
    assert "formatRelativeTime(isoStr)" in html
    assert "getActiveProviderQuota()" in html
    assert "loadTodayUsage()" in html
    assert "loadUsageStats()" in html
    assert "loadDeferredQueue()" in html
    assert "loadJobUsage(jobId)" in html
    assert "loadUsageData(" in html
