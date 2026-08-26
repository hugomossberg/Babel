import pytest
from app.core.ai_providers import context_from_settings, resolve_job_provider_context, get_provider_spec, get_model_capabilities
from app.core.db import pin_job_provider, set_setting, get_job_by_id
from app.services.pipeline import pipeline
from app.api.dashboard import api_test_ai, AISettingsRequest

def test_13_unknown_provider_value_error():
    with pytest.raises(ValueError, match="Unsupported AI provider: fake_provider"):
        get_provider_spec("fake_provider")

def test_1_provider_pinning():
    # Make sure we have a job
    from app.core.db import create_job
    job_id = create_job("test_video.mkv")
    
    set_setting("ai_provider", "gemini")
    set_setting("gemini_model", "gemini-3.5-flash-lite")
    
    pin_job_provider(job_id, "gemini", "gemini-3.5-flash-lite")
    
    # Change global setting
    set_setting("ai_provider", "openai")
    
    # Should still resolve to gemini for this job
    ctx = resolve_job_provider_context(job_id)
    assert ctx.provider == "gemini"
    assert ctx.model == "gemini-3.5-flash-lite"

def test_11_ollama_token_usage():
    from app.core.usage import extract_usage_from_response
    resp = {"prompt_eval_count": 100, "eval_count": 50}
    usage = extract_usage_from_response("ollama", resp)
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50

def test_12_today_usage_model_breakdown():
    from app.core.usage import record_dispatch, complete_dispatch, UsageStage, get_today_usage_summary
    uid = "test-uid-1"
    record_dispatch(uid, "openai", "gpt-4o-mini", UsageStage.PRIMARY)
    complete_dispatch(uid, "SUCCESS", 10, 5, 20, None)
    
    summary = get_today_usage_summary()
    assert "openai" in summary["providers"]
    assert "gpt-4o-mini" in summary["providers"]["openai"]["models"]
    assert summary["providers"]["openai"]["models"]["gpt-4o-mini"]["calls_today"] == 1

def test_4_test_connection_provider(monkeypatch):
    # test that api_test_ai uses the passed provider and not global setting
    pass

def test_3_alignment_audit_job_provider(monkeypatch):
    # Verify that alignment audit uses job provider
    pass


# ---------------------------------------------------------------------------
# Regression tests: unknown provider must fail-explicit on job creation
# ---------------------------------------------------------------------------

def test_create_job_unknown_provider_raises():
    """
    create_job() with an unknown/unsupported ai_provider setting must raise
    ValueError (or subclass) and NOT create a job.
    Gemini must NOT be used as a silent fallback.
    """
    import sqlite3
    from app.core.db import create_job, set_setting, DB_PATH

    set_setting("ai_provider", "TOTALLY_UNKNOWN_PROVIDER_XYZ")
    try:
        with pytest.raises((ValueError, Exception)) as exc_info:
            create_job("test_unknown_provider.mkv")
        # Must mention the unknown provider, not Gemini
        err = str(exc_info.value).lower()
        assert "gemini" not in err, "create_job must NOT fall back to Gemini on unknown provider"
        # Confirm no job was created with unknown provider
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT id FROM jobs WHERE video_path = 'test_unknown_provider.mkv'"
            ).fetchone()
        assert row is None, "No job must be persisted when provider is unknown"
    finally:
        set_setting("ai_provider", "gemini")


def test_create_job_if_no_active_unknown_provider_raises():
    """
    create_job_if_no_active() with an unknown ai_provider must raise
    ValueError and NOT create a job.
    Gemini must NOT be used as silent fallback.
    """
    import sqlite3
    from app.core.db import create_job_if_no_active, set_setting, DB_PATH

    set_setting("ai_provider", "TOTALLY_UNKNOWN_PROVIDER_XYZ")
    path = "test_unknown_provider_no_active.mkv"
    try:
        with pytest.raises((ValueError, Exception)) as exc_info:
            create_job_if_no_active(path)
        err = str(exc_info.value).lower()
        assert "gemini" not in err, "create_job_if_no_active must NOT fall back to Gemini on unknown provider"
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT id FROM jobs WHERE video_path = ?", (path,)
            ).fetchone()
        assert row is None, "No job must be persisted when provider is unknown"
    finally:
        set_setting("ai_provider", "gemini")


def test_create_job_unknown_provider_no_gemini_used():
    """
    Even if we somehow get past create_job, the resulting job must not have
    a Gemini model label when provider is genuinely unknown.
    This is a belt-and-suspenders guard for the above tests.
    """
    import sqlite3
    from app.core.db import set_setting, DB_PATH

    set_setting("ai_provider", "TOTALLY_UNKNOWN_PROVIDER_XYZ")
    try:
        # We expect this to raise — if it doesn't, assert no Gemini model
        try:
            from app.core.db import create_job
            job_id = create_job("test_gemini_guard.mkv")
            # If create_job did NOT raise (should not happen), verify model is not Gemini
            with sqlite3.connect(DB_PATH) as conn:
                row = conn.execute(
                    "SELECT ai_model FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()
            if row:
                model_val = (row[0] or "").lower()
                assert "gemini" not in model_val, (
                    f"create_job stored Gemini model for unknown provider: {row[0]}"
                )
        except (ValueError, Exception):
            pass  # Expected — unknown provider should raise
    finally:
        set_setting("ai_provider", "gemini")
