import sqlite3
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core import db


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def clean_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_activity_feed.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path))
    db.init_db()
    return db_path


def insert_job(
    db_path,
    video_path="test.mkv",
    title="Test Show S01E01",
    status="TRANSLATED",
    event_source="SONARR",
    total_lines=100,
    duration_seconds=10.0,
    created_at="2026-09-01T12:00:00Z",
):
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO jobs (
                video_path, title, status, event_source, total_lines,
                duration_seconds, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                video_path,
                title,
                status,
                event_source,
                total_lines,
                duration_seconds,
                created_at,
                created_at,
            ),
        )
        job_id = cursor.lastrowid
        conn.commit()
    return job_id


# ===========================================================================
# A. Default behavior
# ===========================================================================
def test_default_jobs_behavior(client, clean_db):
    """GET /api/jobs returns newest jobs first up to limit 50."""
    for i in range(60):
        insert_job(
            clean_db,
            video_path=f"video_{i:02d}.mkv",
            title=f"Episode {i:02d}",
            status="TRANSLATED",
            created_at=f"2026-09-01T12:{i:02d}:00Z",
        )

    res = client.get("/api/jobs")
    assert res.status_code == 200
    jobs = res.json()
    assert len(jobs) == 50
    # Newest (highest ID / created_at) must come first
    assert jobs[0]["title"] == "Episode 59"
    assert jobs[49]["title"] == "Episode 10"


# ===========================================================================
# B. Status filtering happens BEFORE LIMIT
# ===========================================================================
def test_status_filtering_happens_before_limit(client, clean_db):
    """
    80 jobs total:
    - 20 oldest are FAILED
    - 60 newest are TRANSLATED
    /api/jobs?status=FAILED must find all 20 FAILED jobs even though
    they are outside the top 50 newest jobs in the table.
    """
    # 20 older FAILED jobs
    for i in range(20):
        insert_job(
            clean_db,
            video_path=f"failed_{i:02d}.mkv",
            title=f"Failed Ep {i:02d}",
            status="FAILED",
            created_at=f"2026-08-01T10:{i:02d}:00Z",
        )

    # 60 newer TRANSLATED jobs
    for i in range(60):
        insert_job(
            clean_db,
            video_path=f"translated_{i:02d}.mkv",
            title=f"Translated Ep {i:02d}",
            status="TRANSLATED",
            created_at=f"2026-09-01T10:{i:02d}:00Z",
        )

    # Without filter -> 50 newest (all TRANSLATED)
    res_all = client.get("/api/jobs")
    assert res_all.status_code == 200
    assert len(res_all.json()) == 50
    assert all(j["status"] == "TRANSLATED" for j in res_all.json())

    # With server-side status filter -> MUST return the 20 FAILED jobs
    res_failed = client.get("/api/jobs?status=FAILED")
    assert res_failed.status_code == 200
    failed_jobs = res_failed.json()
    assert len(failed_jobs) == 20
    assert all(j["status"] == "FAILED" for j in failed_jobs)
    assert failed_jobs[0]["title"] == "Failed Ep 19"
    assert failed_jobs[-1]["title"] == "Failed Ep 00"


def test_status_filtering_large_set_respects_limit(client, clean_db):
    """When >50 matching jobs exist, status filter applies before limit 50."""
    for i in range(70):
        insert_job(
            clean_db,
            video_path=f"failed_{i:02d}.mkv",
            title=f"Failed Ep {i:02d}",
            status="FAILED",
            created_at=f"2026-08-01T10:{i:02d}:00Z",
        )

    res = client.get("/api/jobs?status=FAILED")
    assert res.status_code == 200
    jobs = res.json()
    assert len(jobs) == 50
    assert all(j["status"] == "FAILED" for j in jobs)
    # Newest failed job first
    assert jobs[0]["title"] == "Failed Ep 69"


def test_status_filtering_all_supported_statuses(client, clean_db):
    """Verify various supported statuses filter cleanly."""
    statuses = [
        "TRANSLATED",
        "BAZARR MATCH",
        "FAILED",
        "ALREADY EXISTS",
        "DEFERRED",
        "QUEUED",
        "TRANSLATING",
        "WAITING_FOR_BAZARR",
        "WAITING_PROVIDER",
    ]
    for s in statuses:
        insert_job(clean_db, title=f"Job {s}", status=s)

    for s in statuses:
        res = client.get(f"/api/jobs?status={s}")
        assert res.status_code == 200
        jobs = res.json()
        assert len(jobs) == 1
        assert jobs[0]["status"] == s
        assert jobs[0]["title"] == f"Job {s}"


# ===========================================================================
# C. Combined filter + sort
# ===========================================================================
def test_combined_filter_and_sort(client, clean_db):
    """Test filtering by status combined with numeric sorting by duration."""
    durations = [10.5, 95.2, 5.0, 120.0, 42.1]
    for d in durations:
        insert_job(
            clean_db,
            title=f"Failed dur {d}",
            status="FAILED",
            duration_seconds=d,
        )
    # Insert TRANSLATED jobs with huge duration that shouldn't appear
    insert_job(
        clean_db,
        title="Translated dur 999",
        status="TRANSLATED",
        duration_seconds=999.0,
    )

    # Longest FAILED jobs first
    res = client.get("/api/jobs?status=FAILED&sort_by=duration&order=desc")
    assert res.status_code == 200
    jobs = res.json()
    assert len(jobs) == 5
    assert all(j["status"] == "FAILED" for j in jobs)
    returned_durations = [j["duration_seconds"] for j in jobs]
    assert returned_durations == [120.0, 95.2, 42.1, 10.5, 5.0]

    # Shortest FAILED jobs first
    res_asc = client.get("/api/jobs?status=FAILED&sort_by=duration&order=asc")
    assert res_asc.status_code == 200
    jobs_asc = res_asc.json()
    assert len(jobs_asc) == 5
    returned_durations_asc = [j["duration_seconds"] for j in jobs_asc]
    assert returned_durations_asc == [5.0, 10.5, 42.1, 95.2, 120.0]


# ===========================================================================
# D. Lines numerical sorting
# ===========================================================================
def test_lines_numerical_sorting(client, clean_db):
    """
    Ensure lines are sorted numerically (3 < 20 < 100),
    NOT lexicographically ('100' < '20' < '3').
    """
    line_counts = [100, 3, 20, 1000, 5, 250]
    for lc in line_counts:
        insert_job(clean_db, title=f"Lines {lc}", total_lines=lc)

    # Ascending
    res_asc = client.get("/api/jobs?sort_by=lines&order=asc")
    assert res_asc.status_code == 200
    returned_asc = [j["total_lines"] for j in res_asc.json()]
    assert returned_asc == [3, 5, 20, 100, 250, 1000]

    # Descending
    res_desc = client.get("/api/jobs?sort_by=lines&order=desc")
    assert res_desc.status_code == 200
    returned_desc = [j["total_lines"] for j in res_desc.json()]
    assert returned_desc == [1000, 250, 100, 20, 5, 3]


# ===========================================================================
# E. Source sorting
# ===========================================================================
def test_source_sorting(client, clean_db):
    """Verify source (event_source) sorting database-side."""
    sources = ["SONARR", "MANUAL", "RADARR", "API"]
    for s in sources:
        insert_job(clean_db, title=f"Job {s}", event_source=s)

    # Ascending
    res_asc = client.get("/api/jobs?sort_by=source&order=asc")
    assert res_asc.status_code == 200
    returned_asc = [j["event_source"] for j in res_asc.json()]
    assert returned_asc == ["API", "MANUAL", "RADARR", "SONARR"]

    # Descending
    res_desc = client.get("/api/jobs?sort_by=source&order=desc")
    assert res_desc.status_code == 200
    returned_desc = [j["event_source"] for j in res_desc.json()]
    assert returned_desc == ["SONARR", "RADARR", "MANUAL", "API"]


# ===========================================================================
# F. Time sorting
# ===========================================================================
def test_time_sorting(client, clean_db):
    """Verify chronological time sorting ASC and DESC."""
    timestamps = [
        "2026-01-01T10:00:00Z",
        "2026-03-15T12:00:00Z",
        "2026-02-10T08:00:00Z",
    ]
    for ts in timestamps:
        insert_job(clean_db, title=f"Time {ts}", created_at=ts)

    # Ascending
    res_asc = client.get("/api/jobs?sort_by=time&order=asc")
    assert res_asc.status_code == 200
    returned_asc = [j["created_at"] for j in res_asc.json()]
    assert returned_asc == [
        "2026-01-01T10:00:00Z",
        "2026-02-10T08:00:00Z",
        "2026-03-15T12:00:00Z",
    ]

    # Descending
    res_desc = client.get("/api/jobs?sort_by=time&order=desc")
    assert res_desc.status_code == 200
    returned_desc = [j["created_at"] for j in res_desc.json()]
    assert returned_desc == [
        "2026-03-15T12:00:00Z",
        "2026-02-10T08:00:00Z",
        "2026-01-01T10:00:00Z",
    ]


# ===========================================================================
# G. Status precedence sorting
# ===========================================================================
def test_status_precedence_sorting(client, clean_db):
    """
    Verify status precedence CASE expression:
    1. Active work (TRANSLATING, QUEUED)
    2. Waiting/Deferred (WAITING_PROVIDER, DEFERRED, WAITING_FOR_BAZARR)
    3. FAILED
    4. Finished (TRANSLATED, BAZARR MATCH, ALREADY EXISTS)
    5. Other/Unknown (CANCELLED, UNKNOWN)
    """
    statuses = [
        "TRANSLATING",           # Group 1 (Active)
        "QUEUED",                # Group 1 (Active)
        "WAITING_FOR_BAZARR",    # Group 2 (Waiting)
        "DEFERRED",              # Group 2 (Waiting)
        "WAITING_PROVIDER",      # Group 2 (Waiting)
        "FAILED",                # Group 3 (Failed)
        "TRANSLATED",            # Group 4 (Finished)
        "BAZARR MATCH",          # Group 4 (Finished)
        "ALREADY EXISTS",        # Group 4 (Finished)
        "CANCELLED",             # Group 5 (Other)
    ]
    for s in statuses:
        insert_job(clean_db, title=f"Job {s}", status=s)

    # Ascending sort
    res_asc = client.get("/api/jobs?sort_by=status&order=asc")
    assert res_asc.status_code == 200
    returned_statuses = [j["status"] for j in res_asc.json()]

    # Validate grouping order
    def get_group(status):
        if status in ("TRANSLATING", "QUEUED"):
            return 1
        elif status in ("WAITING_FOR_BAZARR", "DEFERRED", "WAITING_PROVIDER"):
            return 2
        elif status == "FAILED":
            return 3
        elif status in ("TRANSLATED", "BAZARR MATCH", "ALREADY EXISTS"):
            return 4
        return 5

    groups_asc = [get_group(s) for s in returned_statuses]
    # Groups must be monotonically non-decreasing
    assert groups_asc == sorted(groups_asc)
    assert groups_asc[0] == 1
    assert groups_asc[-1] == 5

    # Descending sort reverses grouping naturally
    res_desc = client.get("/api/jobs?sort_by=status&order=desc")
    assert res_desc.status_code == 200
    returned_statuses_desc = [j["status"] for j in res_desc.json()]
    groups_desc = [get_group(s) for s in returned_statuses_desc]
    assert groups_desc == sorted(groups_desc, reverse=True)
    assert groups_desc[0] == 5
    assert groups_desc[-1] == 1


# ===========================================================================
# H & I. SQL injection safety and invalid sort/order handling
# ===========================================================================
def test_invalid_sort_by_and_order_fail_safely(client, clean_db):
    """
    Test that malicious or invalid sort_by and order inputs
    never get interpolated into SQL and fall back safely to default.
    """
    insert_job(clean_db, title="Job 1", created_at="2026-01-01T10:00:00Z")
    insert_job(clean_db, title="Job 2", created_at="2026-01-02T10:00:00Z")

    # SQL Injection attempt in sort_by
    res = client.get("/api/jobs?sort_by=id;DROP+TABLE+jobs")
    assert res.status_code == 200
    jobs = res.json()
    assert len(jobs) == 2
    # Verify jobs table still intact
    assert jobs[0]["title"] == "Job 2"

    # Non-existent column
    res2 = client.get("/api/jobs?sort_by=non_existent_column")
    assert res2.status_code == 200
    assert len(res2.json()) == 2

    # Malicious order value
    res3 = client.get("/api/jobs?sort_by=duration&order=ASC;SELECT+1")
    assert res3.status_code == 200
    assert len(res3.json()) == 2


# ===========================================================================
# J. Status all / empty string handling
# ===========================================================================
def test_status_all_and_empty_fetches_all_jobs(client, clean_db):
    """status='' or status='ALL' or status='all' fetches all jobs without filter."""
    insert_job(clean_db, title="Job Translated", status="TRANSLATED")
    insert_job(clean_db, title="Job Failed", status="FAILED")

    res_empty = client.get("/api/jobs?status=")
    assert res_empty.status_code == 200
    assert len(res_empty.json()) == 2

    res_all_upper = client.get("/api/jobs?status=ALL")
    assert res_all_upper.status_code == 200
    assert len(res_all_upper.json()) == 2

    res_all_lower = client.get("/api/jobs?status=all")
    assert res_all_lower.status_code == 200
    assert len(res_all_lower.json()) == 2


# ===========================================================================
# K. Frontend UI verification
# ===========================================================================
def test_frontend_template_contains_filter_and_sort():
    """Verify index.html contains the necessary markup and Alpine functions."""
    with open("app/templates/index.html") as f:
        html = f.read()

    # Sortable header clicks
    assert "toggleActivitySort('source')" in html
    assert "toggleActivitySort('status')" in html
    assert "toggleActivitySort('lines')" in html
    assert "toggleActivitySort('duration')" in html
    assert "toggleActivitySort('time')" in html

    # Status filter chips
    assert "setActivityFilter('')" in html
    assert "setActivityFilter('TRANSLATED')" in html
    assert "setActivityFilter('BAZARR MATCH')" in html
    assert "setActivityFilter('FAILED')" in html
    assert "setActivityFilter('ALREADY EXISTS')" in html
    assert "setActivityFilter('DEFERRED')" in html

    # Alpine state and loadJobs query params
    assert "activityStatusFilter" in html
    assert "activitySortBy" in html
    assert "activitySortOrder" in html
    assert "params.set('status', this.activityStatusFilter)" in html
    assert "params.set('sort_by', this.activitySortBy)" in html
