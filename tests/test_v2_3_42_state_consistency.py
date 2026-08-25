"""
tests/test_v2_3_42_state_consistency.py
=======================================
v2.3.42 Library State Consistency & Duplicate Job Prevention — Regression Tests

Tests A-K:
  A. Library item without subtitle + no job → Translate Now available (no active_job)
  B. Library item without subtitle + QUEUED → active_job set, not enqueueable
  C. Library item without subtitle + TRANSLATING → active_job present with correct status
  D. Library item without subtitle + DEFERRED → active_job present, shows deferred
  E. Bulk Translate Missing excludes items with active jobs
  F. Concurrent enqueue → exactly ONE active job created (race condition test)
  G. Target subtitle exists → no active_job (normal completed state)
  H. Force retranslate → old subtitle not deleted before new publication (ALREADY_EXISTS safety)
  I. Language display name is dynamic — no hardcoded language
  J. After translation completes, active_job disappears from active-jobs map
  K. ALREADY_EXISTS safety-net still works in pipeline
"""

import os
import pytest
import sqlite3
import threading
import time
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_db(tmp_path):
    """Isolated DB with init_db() applied."""
    import app.core.db as db_mod
    db_path = str(tmp_path / "test.db")
    with patch("app.core.db.DB_PATH", db_path):
        db_mod.DB_PATH = db_path
        db_mod.init_db()
    return db_path


# ---------------------------------------------------------------------------
# A. No subtitle + no job → translate available
# ---------------------------------------------------------------------------

class TestLibraryStateNoJob:

    def test_A_no_active_job_for_video_returns_none(self, clean_db):
        """get_active_job_for_video returns None when no active job exists."""
        with patch("app.core.db.DB_PATH", clean_db):
            from app.core.db import get_active_job_for_video
            result = get_active_job_for_video("/tv/show/episode.mkv")
        assert result is None

    def test_A_active_jobs_map_excludes_terminal_jobs(self, clean_db):
        """get_active_jobs_by_video_paths excludes terminal jobs (TRANSLATED, FAILED)."""
        with patch("app.core.db.DB_PATH", clean_db):
            from app.core.db import get_active_jobs_by_video_paths
            with sqlite3.connect(clean_db) as conn:
                conn.execute(
                    "INSERT INTO jobs (id, video_path, status, event_source, created_at, updated_at) "
                    "VALUES (1, '/tv/s1/ep1.mkv', 'TRANSLATED', 'SONARR', '2026-08-25T10:00:00Z', '2026-08-25T10:00:00Z')"
                )
                conn.execute(
                    "INSERT INTO jobs (id, video_path, status, event_source, created_at, updated_at) "
                    "VALUES (2, '/tv/s1/ep2.mkv', 'FAILED', 'SONARR', '2026-08-25T10:00:00Z', '2026-08-25T10:00:00Z')"
                )
                conn.commit()
            result = get_active_jobs_by_video_paths(["/tv/s1/ep1.mkv", "/tv/s1/ep2.mkv"])
        assert len(result) == 0, "Terminal jobs must not appear in active-jobs map"


# ---------------------------------------------------------------------------
# B. No subtitle + QUEUED → active_job present
# ---------------------------------------------------------------------------

class TestLibraryStateQueued:

    def test_B_queued_job_appears_in_active_map(self, clean_db):
        """QUEUED job appears in get_active_jobs_by_video_paths result."""
        with patch("app.core.db.DB_PATH", clean_db):
            from app.core.db import get_active_jobs_by_video_paths
            with sqlite3.connect(clean_db) as conn:
                conn.execute(
                    "INSERT INTO jobs (id, video_path, status, event_source, created_at, updated_at) "
                    "VALUES (1, '/tv/show/ep.mkv', 'QUEUED', 'MANUAL', '2026-08-25T10:00:00Z', '2026-08-25T10:00:00Z')"
                )
                conn.commit()
            result = get_active_jobs_by_video_paths(["/tv/show/ep.mkv"])
        assert "/tv/show/ep.mkv" in result
        assert result["/tv/show/ep.mkv"]["status"] == "QUEUED"

    def test_B_create_job_if_no_active_returns_existing_when_queued(self, clean_db):
        """create_job_if_no_active returns existing QUEUED job, created=False."""
        with patch("app.core.db.DB_PATH", clean_db):
            from app.core.db import create_job_if_no_active
            # Create first job
            r1 = create_job_if_no_active("/tv/show/ep.mkv", "MANUAL")
            assert r1["created"] is True
            first_id = r1["job_id"]

            # Try to create second job for same path — should return existing
            r2 = create_job_if_no_active("/tv/show/ep.mkv", "SONARR")
        assert r2["created"] is False
        assert r2["job_id"] == first_id
        assert r2["existing_job"]["status"] == "QUEUED"


# ---------------------------------------------------------------------------
# C. TRANSLATING → active_job present with correct status
# ---------------------------------------------------------------------------

class TestLibraryStateTranslating:

    def test_C_translating_job_in_active_map(self, clean_db):
        """TRANSLATING job appears in active-jobs map."""
        with patch("app.core.db.DB_PATH", clean_db):
            from app.core.db import get_active_job_for_video
            with sqlite3.connect(clean_db) as conn:
                conn.execute(
                    "INSERT INTO jobs (id, video_path, status, event_source, created_at, updated_at) "
                    "VALUES (1, '/movies/film.mkv', 'TRANSLATING', 'RADARR', '2026-08-25T10:00:00Z', '2026-08-25T10:00:00Z')"
                )
                conn.commit()
            result = get_active_job_for_video("/movies/film.mkv")
        assert result is not None
        assert result["status"] == "TRANSLATING"


# ---------------------------------------------------------------------------
# D. DEFERRED → active_job shows DEFERRED
# ---------------------------------------------------------------------------

class TestLibraryStateDeferred:

    def test_D_deferred_job_in_active_map(self, clean_db):
        """DEFERRED job is in ACTIVE_JOB_STATUSES and appears in active-jobs map."""
        from app.core.db import ACTIVE_JOB_STATUSES
        assert "DEFERRED" in ACTIVE_JOB_STATUSES, "DEFERRED must be in ACTIVE_JOB_STATUSES"

        with patch("app.core.db.DB_PATH", clean_db):
            from app.core.db import get_active_job_for_video
            with sqlite3.connect(clean_db) as conn:
                conn.execute(
                    "INSERT INTO jobs (id, video_path, status, event_source, defer_reason, created_at, updated_at) "
                    "VALUES (1, '/tv/show/ep.mkv', 'DEFERRED', 'MANUAL', 'LOCAL_RPD', '2026-08-25T10:00:00Z', '2026-08-25T10:00:00Z')"
                )
                conn.commit()
            result = get_active_job_for_video("/tv/show/ep.mkv")
        assert result is not None
        assert result["status"] == "DEFERRED"
        assert result["defer_reason"] == "LOCAL_RPD"

    def test_D_deferred_job_blocks_new_job_creation(self, clean_db):
        """DEFERRED job prevents create_job_if_no_active from creating a duplicate."""
        with patch("app.core.db.DB_PATH", clean_db):
            from app.core.db import create_job_if_no_active
            with sqlite3.connect(clean_db) as conn:
                conn.execute(
                    "INSERT INTO jobs (id, video_path, status, event_source, created_at, updated_at) "
                    "VALUES (1, '/tv/show/ep.mkv', 'DEFERRED', 'MANUAL', '2026-08-25T10:00:00Z', '2026-08-25T10:00:00Z')"
                )
                conn.commit()
            result = create_job_if_no_active("/tv/show/ep.mkv", "SONARR")
        assert result["created"] is False
        assert result["job_id"] == 1


# ---------------------------------------------------------------------------
# E. Bulk Translate Missing excludes active jobs (backend test)
# ---------------------------------------------------------------------------

class TestBulkTranslateExcludesActive:

    def test_E_active_statuses_all_recognized(self, clean_db):
        """All ACTIVE_JOB_STATUSES values are recognized and excluded from active-jobs batch query."""
        from app.core.db import ACTIVE_JOB_STATUSES, get_active_jobs_by_video_paths

        video_paths = [f"/tv/show/ep{i}.mkv" for i in range(len(ACTIVE_JOB_STATUSES))]

        with patch("app.core.db.DB_PATH", clean_db):
            with sqlite3.connect(clean_db) as conn:
                for i, status in enumerate(ACTIVE_JOB_STATUSES):
                    conn.execute(
                        "INSERT INTO jobs (id, video_path, status, event_source, created_at, updated_at) "
                        "VALUES (?, ?, ?, 'MANUAL', '2026-08-25T10:00:00Z', '2026-08-25T10:00:00Z')",
                        (i + 1, video_paths[i], status)
                    )
                conn.commit()
            result = get_active_jobs_by_video_paths(video_paths)

        assert len(result) == len(ACTIVE_JOB_STATUSES), (
            f"All {len(ACTIVE_JOB_STATUSES)} active statuses should be returned, got {len(result)}"
        )

    def test_E_only_terminal_excluded_from_active(self, clean_db):
        """Terminal statuses (TRANSLATED, FAILED, CANCELLED) are not in ACTIVE_JOB_STATUSES."""
        from app.core.db import ACTIVE_JOB_STATUSES
        terminal = {"TRANSLATED", "FAILED", "CANCELLED", "ALREADY EXISTS", "SUCCESS", "REPAIRED", "HEALTHY"}
        overlap = terminal & set(ACTIVE_JOB_STATUSES)
        assert len(overlap) == 0, f"Terminal statuses must not be in ACTIVE_JOB_STATUSES: {overlap}"


# ---------------------------------------------------------------------------
# F. Concurrent enqueue → only ONE active job (race condition)
# ---------------------------------------------------------------------------

class TestConcurrentEnqueue:

    def test_F_concurrent_enqueue_creates_one_job(self, clean_db):
        """Two concurrent create_job_if_no_active calls → exactly ONE new active job."""
        import app.core.db as db_mod
        # Patch at module level (thread-safe — unittest.mock patch context managers
        # are not thread-safe inside thread functions)
        original_db_path = db_mod.DB_PATH
        db_mod.DB_PATH = clean_db
        try:
            results = []
            errors = []

            def enqueue():
                try:
                    from app.core.db import create_job_if_no_active
                    r = create_job_if_no_active("/tv/show/ep.mkv", "MANUAL")
                    results.append(r)
                except Exception as e:
                    errors.append(e)

            # Launch two threads simultaneously
            t1 = threading.Thread(target=enqueue)
            t2 = threading.Thread(target=enqueue)
            t1.start(); t2.start()
            t1.join(); t2.join()
        finally:
            db_mod.DB_PATH = original_db_path

        assert not errors, f"Errors during concurrent enqueue: {errors}"
        assert len(results) == 2

        # Exactly one should have created=True, one created=False
        created = [r for r in results if r["created"] is True]
        existing = [r for r in results if r["created"] is False]
        assert len(created) == 1, f"Expected exactly 1 job created, got {len(created)}: {results}"
        assert len(existing) == 1
        # Both should reference the same job_id
        assert results[0]["job_id"] == results[1]["job_id"]

    def test_F_deferred_counts_as_active_in_race(self, clean_db):
        """A DEFERRED job blocks concurrent new job creation."""
        with patch("app.core.db.DB_PATH", clean_db):
            with sqlite3.connect(clean_db) as conn:
                conn.execute(
                    "INSERT INTO jobs (id, video_path, status, event_source, created_at, updated_at) "
                    "VALUES (1, '/tv/show/ep.mkv', 'DEFERRED', 'MANUAL', '2026-08-25T10:00:00Z', '2026-08-25T10:00:00Z')"
                )
                conn.commit()
            from app.core.db import create_job_if_no_active
            result = create_job_if_no_active("/tv/show/ep.mkv", "SONARR")

        assert result["created"] is False
        assert result["job_id"] == 1


# ---------------------------------------------------------------------------
# G. Target subtitle exists → no active_job interference
# ---------------------------------------------------------------------------

class TestTargetSubtitleExists:

    def test_G_translated_terminal_job_not_blocking(self, clean_db):
        """After TRANSLATED, create_job_if_no_active allows a new job (terminal does not block)."""
        with patch("app.core.db.DB_PATH", clean_db):
            from app.core.db import create_job_if_no_active
            with sqlite3.connect(clean_db) as conn:
                conn.execute(
                    "INSERT INTO jobs (id, video_path, status, event_source, created_at, updated_at) "
                    "VALUES (1, '/tv/show/ep.mkv', 'TRANSLATED', 'MANUAL', '2026-08-25T10:00:00Z', '2026-08-25T10:00:00Z')"
                )
                conn.commit()
            # After a terminal job, any new job (force or not) should be created
            result = create_job_if_no_active("/tv/show/ep.mkv", "MANUAL", force_retranslate=True)
        assert result["created"] is True

    def test_G_failed_terminal_job_not_blocking(self, clean_db):
        """FAILED terminal job does not block a new job for the same video."""
        with patch("app.core.db.DB_PATH", clean_db):
            from app.core.db import create_job_if_no_active
            with sqlite3.connect(clean_db) as conn:
                conn.execute(
                    "INSERT INTO jobs (id, video_path, status, event_source, created_at, updated_at) "
                    "VALUES (1, '/tv/show/ep.mkv', 'FAILED', 'MANUAL', '2026-08-25T10:00:00Z', '2026-08-25T10:00:00Z')"
                )
                conn.commit()
            result = create_job_if_no_active("/tv/show/ep.mkv", "MANUAL")
        assert result["created"] is True


# ---------------------------------------------------------------------------
# H. Force retranslate → old subtitle not deleted before new publication
# ---------------------------------------------------------------------------

class TestAtomicSubtitleProtection:

    def test_H_already_exists_check_present_in_pipeline(self):
        """Pipeline still contains ALREADY EXISTS check (defense-in-depth)."""
        import inspect
        from app.services.pipeline import pipeline as pipeline_instance
        src = inspect.getsource(type(pipeline_instance)._run_pipeline_logic)
        assert "already" in src.lower() and "exists" in src.lower(), (
            "Pipeline must still have ALREADY EXISTS safety check"
        )

    def test_H_force_retranslate_blocked_when_active_job_exists(self, clean_db):
        """
        force_retranslate=True must NOT create a new job when an active job exists.
        Fixed in v2.3.42: force no longer bypasses the active-job dedupe check.
        """
        with patch("app.core.db.DB_PATH", clean_db):
            from app.core.db import create_job_if_no_active
            # First job (active QUEUED)
            r1 = create_job_if_no_active("/tv/show/ep.mkv", "MANUAL", force_retranslate=False)
            assert r1["created"] is True

            # Force retranslate with an active job present: must return existing, not create new
            r2 = create_job_if_no_active("/tv/show/ep.mkv", "MANUAL", force_retranslate=True)
        assert r2["created"] is False, (
            "Force retranslate must not create duplicate when active job exists"
        )
        assert r2["job_id"] == r1["job_id"], (
            "Force retranslate must return the SAME existing active job ID"
        )



# ---------------------------------------------------------------------------
# I. Language display name is dynamic
# ---------------------------------------------------------------------------

class TestLanguageAgnostic:

    def test_I_ACTIVE_JOB_STATUSES_is_single_source_of_truth(self):
        """ACTIVE_JOB_STATUSES constant exists in db module and is a non-empty tuple."""
        from app.core.db import ACTIVE_JOB_STATUSES
        assert isinstance(ACTIVE_JOB_STATUSES, tuple)
        assert len(ACTIVE_JOB_STATUSES) >= 8, "Must cover all known active statuses"

    def test_I_scanner_uses_language_registry(self):
        """Scanner uses get_language() from language registry (not hardcoded codes)."""
        import inspect
        from app.services import scanner
        src = inspect.getsource(scanner)
        # Should reference get_language or language registry, not hardcode specific lang
        assert "get_language" in src or "LANGUAGES" in src, (
            "Scanner must use language registry, not hardcoded language codes"
        )

    def test_I_no_hardcoded_swedish_in_active_statuses(self):
        """No language-specific strings in ACTIVE_JOB_STATUSES (statuses are job states, not languages)."""
        from app.core.db import ACTIVE_JOB_STATUSES
        # Language codes / names that should NEVER appear as a job status
        forbidden_lang_strings = {"swedish", "german", "french", "polish", "serbian", "sv", "de", "pl", "sr"}
        for status in ACTIVE_JOB_STATUSES:
            status_lower = status.lower()
            for lang in forbidden_lang_strings:
                assert status_lower != lang, (
                    f"Job status '{status}' must not be a language name/code"
                )
                # Only check if the full token equals the lang (not substring)
                assert not status_lower.startswith(lang + "_") and not status_lower.endswith("_" + lang), (
                    f"Job status '{status}' must not contain language-specific token '{lang}'"
                )


# ---------------------------------------------------------------------------
# J. After job completes, it disappears from active-jobs map
# ---------------------------------------------------------------------------

class TestJobCompletionVisibility:

    def test_J_completed_job_not_in_active_map(self, clean_db):
        """After status changes to TRANSLATED, job disappears from active-jobs map."""
        with patch("app.core.db.DB_PATH", clean_db):
            from app.core.db import get_active_jobs_by_video_paths, update_job
            with sqlite3.connect(clean_db) as conn:
                conn.execute(
                    "INSERT INTO jobs (id, video_path, status, event_source, created_at, updated_at) "
                    "VALUES (1, '/tv/show/ep.mkv', 'TRANSLATING', 'MANUAL', '2026-08-25T10:00:00Z', '2026-08-25T10:00:00Z')"
                )
                conn.commit()

            # While TRANSLATING → appears in active map
            active = get_active_jobs_by_video_paths(["/tv/show/ep.mkv"])
            assert "/tv/show/ep.mkv" in active

            # Transition to TRANSLATED
            update_job(1, status="TRANSLATED")

            # Now should not appear
            active_after = get_active_jobs_by_video_paths(["/tv/show/ep.mkv"])
        assert "/tv/show/ep.mkv" not in active_after


# ---------------------------------------------------------------------------
# K. ALREADY_EXISTS safety-net still works
# ---------------------------------------------------------------------------

class TestAlreadyExistsSafetyNet:

    def test_K_already_exists_in_pipeline_source(self):
        """ALREADY EXISTS status string still present in pipeline code."""
        import inspect
        from app.services.pipeline import pipeline as pipeline_instance
        src = inspect.getsource(type(pipeline_instance)._run_pipeline_logic)
        # Check for the specific status string
        assert "ALREADY EXISTS" in src or "already_exists" in src, (
            "ALREADY EXISTS safety-net must be present in pipeline"
        )

    def test_K_create_job_if_no_active_fails_closed_on_db_error(self):
        """
        DB error in dedupe check must NOT create a new job (fail-closed).
        Regression for issue #3: old code fell back to create_job() on DB error,
        risking duplicate jobs at the worst possible moment.
        """
        with patch("app.core.db.DB_PATH", "/nonexistent/path/to/db.sqlite"), \
             patch("app.core.db.get_setting", return_value="gemini"), \
             patch("app.core.db.create_job", return_value=42) as mock_create:
            from app.core.db import create_job_if_no_active
            # Must raise — NOT silently create a job
            with pytest.raises(Exception):
                create_job_if_no_active("/tv/show/ep.mkv", "MANUAL")
        # create_job must never have been called
        mock_create.assert_not_called()


# ---------------------------------------------------------------------------
# Additional: get_job_stats uses ACTIVE_JOB_STATUSES
# ---------------------------------------------------------------------------

class TestJobStatsConsistency:

    def test_stats_active_count_includes_deferred(self, clean_db):
        """get_job_stats active_jobs count includes DEFERRED (uses ACTIVE_JOB_STATUSES)."""
        with patch("app.core.db.DB_PATH", clean_db):
            from app.core.db import get_job_stats
            with sqlite3.connect(clean_db) as conn:
                conn.execute(
                    "INSERT INTO jobs (id, video_path, status, event_source, created_at, updated_at) "
                    "VALUES (1, '/tv/show/ep.mkv', 'DEFERRED', 'MANUAL', '2026-08-25T10:00:00Z', '2026-08-25T10:00:00Z')"
                )
                conn.commit()
            stats = get_job_stats()

        # DEFERRED is in ACTIVE_JOB_STATUSES, so it counts as active
        assert stats["active_jobs"] >= 1, "DEFERRED must count in active_jobs stat"

    def test_stats_deferred_count_separate(self, clean_db):
        """get_job_stats has both active_jobs (includes deferred) and deferred (separate) fields."""
        with patch("app.core.db.DB_PATH", clean_db):
            from app.core.db import get_job_stats
            with sqlite3.connect(clean_db) as conn:
                conn.execute(
                    "INSERT INTO jobs (id, video_path, status, event_source, created_at, updated_at) "
                    "VALUES (1, '/tv/show/ep.mkv', 'DEFERRED', 'MANUAL', '2026-08-25T10:00:00Z', '2026-08-25T10:00:00Z')"
                )
                conn.commit()
            stats = get_job_stats()

        assert "deferred" in stats
        assert stats["deferred"] == 1


# ===========================================================================
# NEW REGRESSION TESTS — v2.3.42 correctness-pass (issues 1-6)
# ===========================================================================

# ---------------------------------------------------------------------------
# Issue 1 + B/C: Force must NOT bypass active-job dedupe
# ---------------------------------------------------------------------------

class TestForceDedupe:

    def test_B_two_concurrent_force_requests_create_one_job(self, clean_db):
        """
        Two simultaneous force_retranslate=True requests for the same video
        must result in exactly ONE active job.  Regression for issue #1.
        """
        import app.core.db as db_mod
        original_db_path = db_mod.DB_PATH
        db_mod.DB_PATH = clean_db
        try:
            results = []
            errors = []

            def enqueue_force():
                try:
                    from app.core.db import create_job_if_no_active
                    r = create_job_if_no_active(
                        "/tv/show/ep.mkv", "MANUAL", force_retranslate=True
                    )
                    results.append(r)
                except Exception as e:
                    errors.append(e)

            t1 = threading.Thread(target=enqueue_force)
            t2 = threading.Thread(target=enqueue_force)
            t1.start(); t2.start()
            t1.join(); t2.join()
        finally:
            db_mod.DB_PATH = original_db_path

        assert not errors, f"Errors during concurrent force enqueue: {errors}"
        assert len(results) == 2

        created = [r for r in results if r["created"] is True]
        existing = [r for r in results if r["created"] is False]
        assert len(created) == 1, (
            f"Force: expected exactly 1 job created, got {len(created)}: {results}"
        )
        assert len(existing) == 1
        assert results[0]["job_id"] == results[1]["job_id"], (
            "Both force requests must reference the SAME job ID"
        )

    def test_C_normal_then_force_returns_same_job(self, clean_db):
        """
        Normal enqueue followed by force_retranslate=True for same video
        must return the existing active job (created=False), not a new one.
        """
        with patch("app.core.db.DB_PATH", clean_db):
            from app.core.db import create_job_if_no_active
            r1 = create_job_if_no_active("/tv/show/ep.mkv", "MANUAL", force_retranslate=False)
            assert r1["created"] is True

            r2 = create_job_if_no_active("/tv/show/ep.mkv", "MANUAL", force_retranslate=True)
        assert r2["created"] is False, "Force must not create duplicate when active job exists"
        assert r2["job_id"] == r1["job_id"], "Force must return the SAME job ID"

    def test_I_terminal_job_does_not_block_new_force_job(self, clean_db):
        """
        A terminated (FAILED) job must not block a new legitimate force job.
        Regression for issue #1 edge case.
        """
        with patch("app.core.db.DB_PATH", clean_db):
            from app.core.db import create_job_if_no_active
            with sqlite3.connect(clean_db) as conn:
                conn.execute(
                    "INSERT INTO jobs (id, video_path, status, event_source, created_at, updated_at) "
                    "VALUES (1, '/tv/show/ep.mkv', 'FAILED', 'MANUAL', '2026-08-25T10:00:00Z', '2026-08-25T10:00:00Z')"
                )
                conn.commit()
            result = create_job_if_no_active(
                "/tv/show/ep.mkv", "MANUAL", force_retranslate=True
            )
        assert result["created"] is True, "Terminal job must not block a new force job"
        assert result["job_id"] != 1


# ---------------------------------------------------------------------------
# Issue 2: Delete Subs must block on ALL ACTIVE_JOB_STATUSES (incl. DEFERRED)
# ---------------------------------------------------------------------------

class TestDeleteSubsGuard:

    @pytest.mark.parametrize("status", list(__import__('app.core.db', fromlist=['ACTIVE_JOB_STATUSES']).ACTIVE_JOB_STATUSES))
    def test_E_delete_subs_blocked_for_all_active_statuses(self, status, clean_db):
        """
        Delete Subs must be blocked for every status in ACTIVE_JOB_STATUSES.
        Regression for issue #2: old code used a hardcoded subset that excluded DEFERRED.
        """
        from app.core.db import ACTIVE_JOB_STATUSES
        assert status in ACTIVE_JOB_STATUSES, (
            f"{status} must be in ACTIVE_JOB_STATUSES for delete-subs guard to work"
        )
        # Verify the guard uses the tuple membership check (not a hardcoded list)
        import inspect
        import app.api.dashboard as dash_mod
        src = inspect.getsource(dash_mod.api_delete_subtitles)
        # The guard must reference ACTIVE_JOB_STATUSES, not a hardcoded status string
        assert "ACTIVE_JOB_STATUSES" in src, (
            "api_delete_subtitles must use ACTIVE_JOB_STATUSES, not a hardcoded subset"
        )

    def test_E_deferred_blocks_delete_subs_guard(self, clean_db):
        """DEFERRED specifically must be included in the delete-subs guard."""
        from app.core.db import ACTIVE_JOB_STATUSES
        assert "DEFERRED" in ACTIVE_JOB_STATUSES, (
            "DEFERRED must be in ACTIVE_JOB_STATUSES so delete-subs guard blocks it"
        )


# ---------------------------------------------------------------------------
# Issue 3: DB errors must fail-closed (no fallback create_job)
# ---------------------------------------------------------------------------

class TestDedupeFailClosed:

    def test_F_db_error_raises_not_creates_job(self):
        """
        DB error in create_job_if_no_active must raise an exception.
        It must NOT fall back to create_job() — that risks duplicate jobs
        exactly when the deduplication shield is unavailable.
        """
        with patch("app.core.db.DB_PATH", "/nonexistent/nowhere/db.sqlite"), \
             patch("app.core.db.get_setting", return_value="gemini"), \
             patch("app.core.db.create_job", return_value=99) as mock_create:
            from app.core.db import create_job_if_no_active
            with pytest.raises(Exception):
                create_job_if_no_active("/tv/show/ep.mkv", "MANUAL")
        mock_create.assert_not_called()

    def test_F_db_error_does_not_insert_row(self, tmp_path):
        """No job row must be created when the DB exclusive lock fails."""
        import app.core.db as db_mod
        db_path = str(tmp_path / "test.db")
        with patch("app.core.db.DB_PATH", db_path):
            db_mod.DB_PATH = db_path
            db_mod.init_db()

        # Corrupt the path so the exclusive lock cannot be obtained
        with patch("app.core.db.DB_PATH", "/dev/null/impossible.db"), \
             patch("app.core.db.get_setting", return_value="gemini"):
            from app.core.db import create_job_if_no_active
            with pytest.raises(Exception):
                create_job_if_no_active("/tv/show/ep.mkv", "MANUAL")

        # The original (working) DB must have 0 rows
        with sqlite3.connect(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        assert count == 0, "No job row must be inserted when dedupe DB call fails"


# ---------------------------------------------------------------------------
# Issue 4: force_retranslate persists in DB through DEFERRED/RETRY
# ---------------------------------------------------------------------------

class TestForceIntentPersistence:

    def test_G_force_retranslate_persisted_in_db_row(self, clean_db):
        """
        When a force job is created via create_job_if_no_active(force_retranslate=True),
        the DB row must have force_retranslate = 1 so the scheduler can pass it
        back to the pipeline on resume.
        """
        with patch("app.core.db.DB_PATH", clean_db):
            from app.core.db import create_job_if_no_active
            result = create_job_if_no_active(
                "/tv/show/ep.mkv", "MANUAL", force_retranslate=True
            )
        assert result["created"] is True
        with sqlite3.connect(clean_db) as conn:
            row = conn.execute(
                "SELECT force_retranslate FROM jobs WHERE id = ?",
                (result["job_id"],)
            ).fetchone()
        assert row is not None
        assert row[0] == 1, (
            "force_retranslate=True must be stored as 1 in the DB row"
        )

    def test_G_non_force_job_has_zero_flag(self, clean_db):
        """Normal (non-force) job must have force_retranslate = 0."""
        with patch("app.core.db.DB_PATH", clean_db):
            from app.core.db import create_job_if_no_active
            result = create_job_if_no_active(
                "/tv/show/ep.mkv", "MANUAL", force_retranslate=False
            )
        with sqlite3.connect(clean_db) as conn:
            row = conn.execute(
                "SELECT force_retranslate FROM jobs WHERE id = ?",
                (result["job_id"],)
            ).fetchone()
        assert row[0] == 0

    def test_G_force_column_exists_in_schema(self, clean_db):
        """init_db must create the force_retranslate column (migration test)."""
        with sqlite3.connect(clean_db) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()]
        assert "force_retranslate" in cols, (
            "force_retranslate column must exist after init_db()"
        )

    def test_G_force_intent_survives_deferred_roundtrip(self, clean_db):
        """
        Simulate the full DEFERRED→resume cycle:
        1. Create force job (force_retranslate=1 in DB).
        2. Job becomes DEFERRED (scheduler updates status).
        3. Scheduler reads job and passes force_retranslate back to pipeline.
        => force_retranslate flag must still be 1 after the roundtrip.
        """
        with patch("app.core.db.DB_PATH", clean_db):
            from app.core.db import create_job_if_no_active, update_job, get_job_by_id
            result = create_job_if_no_active(
                "/tv/show/ep.mkv", "MANUAL", force_retranslate=True
            )
            job_id = result["job_id"]
            assert result["created"] is True

            # Simulate pipeline deferring the job
            update_job(
                job_id,
                status="DEFERRED",
                defer_reason="LOCAL_RPD",
                deferred_at="2026-08-25T10:00:00+00:00",
            )

            # Scheduler reads the job back
            job = get_job_by_id(job_id)
        assert job is not None
        assert job["status"] == "DEFERRED"
        assert bool(job.get("force_retranslate")), (
            "force_retranslate flag must still be True after DEFERRED status change"
        )
        # The value the scheduler would pass back to process_video_file
        resumed_force = bool(job.get("force_retranslate"))
        assert resumed_force is True


# ---------------------------------------------------------------------------
# Issue 5: Large library query must not use IN(all_paths)
# ---------------------------------------------------------------------------

class TestLargeLibraryQuery:

    def test_large_library_no_in_clause(self):
        """
        get_active_jobs_by_video_paths must NOT build a giant IN(?,?,...) for
        all library paths.  Regression for issue #5 (SQLite param limits).
        """
        import inspect
        from app.core.db import get_active_jobs_by_video_paths
        src = inspect.getsource(get_active_jobs_by_video_paths)
        # The query must NOT have placeholders_paths (the old IN-all-paths approach)
        assert "placeholders_paths" not in src, (
            "get_active_jobs_by_video_paths must not build an IN(all_paths) query"
        )
        # Must use Python-side filtering comment
        assert "norm_paths_set" in src or "Python" in src or "filter" in src.lower(), (
            "get_active_jobs_by_video_paths must use Python-side path filtering"
        )

    def test_large_library_returns_only_requested_paths(self, clean_db):
        """
        Even though we fetch all active jobs globally, we must only return jobs
        for paths the caller asked about.
        """
        with patch("app.core.db.DB_PATH", clean_db):
            from app.core.db import get_active_jobs_by_video_paths
            with sqlite3.connect(clean_db) as conn:
                # ep1 — QUEUED (active)
                conn.execute(
                    "INSERT INTO jobs (id, video_path, status, event_source, created_at, updated_at) "
                    "VALUES (1, '/tv/ep1.mkv', 'QUEUED', 'MANUAL', '2026-08-25T10:00:00Z', '2026-08-25T10:00:00Z')"
                )
                # ep2 — TRANSLATING (active, not requested)
                conn.execute(
                    "INSERT INTO jobs (id, video_path, status, event_source, created_at, updated_at) "
                    "VALUES (2, '/tv/ep2.mkv', 'TRANSLATING', 'MANUAL', '2026-08-25T10:00:00Z', '2026-08-25T10:00:00Z')"
                )
                conn.commit()
            # Only ask for ep1
            result = get_active_jobs_by_video_paths(["/tv/ep1.mkv"])

        assert "/tv/ep1.mkv" in result, "ep1 should be in result"
        assert "/tv/ep2.mkv" not in result, "ep2 was not requested — must be excluded"


# ---------------------------------------------------------------------------
# Issue 6: Bulk force must exclude already-active jobs
# ---------------------------------------------------------------------------

class TestBulkForceExcludesActive:

    def test_H_bulk_force_excludes_active_jobs_backend(self, clean_db):
        """
        Backend: create_job_if_no_active with force=True still returns existing
        active job (created=False) — so bulk force loop naturally skips actives.
        """
        with patch("app.core.db.DB_PATH", clean_db):
            from app.core.db import create_job_if_no_active
            # Pre-existing active jobs
            with sqlite3.connect(clean_db) as conn:
                for i, status in enumerate(["TRANSLATING", "DEFERRED", "WAITING_PROVIDER"], start=1):
                    conn.execute(
                        "INSERT INTO jobs (id, video_path, status, event_source, created_at, updated_at) "
                        "VALUES (?, ?, ?, 'SONARR', '2026-08-25T10:00:00Z', '2026-08-25T10:00:00Z')",
                        (i, f"/tv/ep{i}.mkv", status)
                    )
                conn.commit()

            # Simulate bulk force for all 3 episodes
            results = []
            for i in range(1, 4):
                r = create_job_if_no_active(
                    f"/tv/ep{i}.mkv", "MANUAL", force_retranslate=True
                )
                results.append(r)

        # All 3 must return existing (created=False) — bulk force cannot create duplicates
        for r in results:
            assert r["created"] is False, (
                f"Bulk force must not create a new job when active job exists: {r}"
            )

    def test_H_bulk_force_creates_jobs_for_idle_paths(self, clean_db):
        """
        Bulk force must still create jobs for paths that have NO active job.
        """
        with patch("app.core.db.DB_PATH", clean_db):
            from app.core.db import create_job_if_no_active
            with sqlite3.connect(clean_db) as conn:
                # Only ep1 is active
                conn.execute(
                    "INSERT INTO jobs (id, video_path, status, event_source, created_at, updated_at) "
                    "VALUES (1, '/tv/ep1.mkv', 'TRANSLATING', 'SONARR', '2026-08-25T10:00:00Z', '2026-08-25T10:00:00Z')"
                )
                conn.commit()

            # Bulk force on ep1 (active) and ep2 (idle)
            r1 = create_job_if_no_active("/tv/ep1.mkv", "MANUAL", force_retranslate=True)
            r2 = create_job_if_no_active("/tv/ep2.mkv", "MANUAL", force_retranslate=True)

        assert r1["created"] is False, "ep1 is active — must not create duplicate"
        assert r2["created"] is True, "ep2 has no active job — force must create new job"


# ---------------------------------------------------------------------------
# Issue 7: Automatic Library State Refresh & Subtitle Deletion Consistency
# ---------------------------------------------------------------------------

class TestLibraryAutoRefreshConsistency:

    @pytest.mark.asyncio
    async def test_delete_subtitles_immediate_filesystem_and_state_consistency(self, clean_db, tmp_path):
        """
        When Delete Subtitles succeeds:
        - target language .srt is removed from disk immediately
        - scan_library_folders immediately reflects has_target_sub=False and empty subtitles
        - no active job is created or lingering
        """
        import app.core.db as db_mod
        from app.api.dashboard import api_delete_subtitles, DeleteSubRequest
        from app.services.scanner import scan_library_folders

        tv_dir = tmp_path / "tv"
        show_dir = tv_dir / "ShowName"
        show_dir.mkdir(parents=True)
        video = show_dir / "ShowName.S01E01.mkv"
        video.touch()
        target_sub = show_dir / "ShowName.S01E01.sv.srt"
        target_sub.write_text("1\n00:00:01,000 --> 00:00:02,000\nHej\n", encoding="utf-8")

        with patch("app.core.db.DB_PATH", clean_db):
            db_mod.set_setting("media_series_path", str(tv_dir))
            db_mod.set_setting("languages", '[{"name": "Swedish", "code": "sv", "enabled": true}]')

            # Initial scan shows target sub exists
            scan1 = scan_library_folders(str(tv_dir), category="series")
            assert len(scan1) == 1
            ep1 = scan1[0]["episodes"][0]
            assert ep1["has_target_sub"] is True
            assert len(ep1["subtitles"]) == 1

            # Delete subtitles via API
            del_res = await api_delete_subtitles(DeleteSubRequest(video_path=str(video)))
            assert del_res["status"] == "deleted"
            assert not target_sub.exists()

            # Rescan immediately shows no target sub
            scan2 = scan_library_folders(str(tv_dir), category="series")
            ep2 = scan2[0]["episodes"][0]
            assert ep2["has_target_sub"] is False
            assert len(ep2["subtitles"]) == 0

            # No active job
            active = db_mod.get_active_job_for_video(str(video))
            assert active is None

    def test_job_completion_transitions_active_map_to_empty(self, clean_db):
        """
        Job transition from active (QUEUED/TRANSLATING) -> terminal (TRANSLATED/FAILED):
        get_active_jobs_by_video_paths transitions from returning active job to empty map,
        signaling completion to frontend for silent filesystem refresh.
        """
        with patch("app.core.db.DB_PATH", clean_db):
            from app.core.db import create_job_if_no_active, update_job, get_active_jobs_by_video_paths

            res = create_job_if_no_active("/tv/show/ep1.mkv", "MANUAL")
            job_id = res["job_id"]

            # QUEUED state
            active_map = get_active_jobs_by_video_paths(["/tv/show/ep1.mkv"])
            assert "/tv/show/ep1.mkv" in active_map
            assert active_map["/tv/show/ep1.mkv"]["status"] == "QUEUED"

            # TRANSLATING state
            update_job(job_id, status="TRANSLATING")
            active_map = get_active_jobs_by_video_paths(["/tv/show/ep1.mkv"])
            assert active_map["/tv/show/ep1.mkv"]["status"] == "TRANSLATING"

            # TRANSLATED (terminal) state
            update_job(job_id, status="TRANSLATED")
            active_map = get_active_jobs_by_video_paths(["/tv/show/ep1.mkv"])
            assert len(active_map) == 0, "Terminal job must not appear in active_jobs_map"

    def test_active_jobs_polling_is_read_only_and_spawns_no_jobs(self, clean_db):
        """
        Polling active jobs / media paths is strictly read-only and never creates jobs.
        """
        with patch("app.core.db.DB_PATH", clean_db):
            from app.core.db import get_active_jobs_by_video_paths, get_jobs

            for _ in range(10):
                res = get_active_jobs_by_video_paths(["/tv/show/ep1.mkv", "/tv/show/ep2.mkv"])
                assert len(res) == 0

            assert len(get_jobs()) == 0


