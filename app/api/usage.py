"""
app/api/usage.py
================
Phase 2: AI Usage & Cost API endpoints.

Routes:
  GET /api/usage/job/{job_id}  — Per-job usage summary
  GET /api/usage/today         — Today's (UTC) provider summary
  GET /api/usage/stats         — Historical statistics

All responses are stable JSON. Frontend (Phase 3) consumes pre-aggregated data
and does NOT need to do any accounting reconstruction.
"""

import logging
from fastapi import APIRouter, HTTPException
from typing import Any, Dict

from app.core.usage import (
    get_job_usage_summary,
    get_today_usage_summary,
    get_historical_stats,
)

logger = logging.getLogger("babel.api.usage")

router = APIRouter()


@router.get("/usage/job/{job_id}", response_model=None)
async def get_job_usage(job_id: int) -> Dict[str, Any]:
    """
    Return AI usage summary for a specific job.

    Response fields:
      job_id                   - The requested job ID
      total_calls              - Total provider requests made for this job
      total_input_tokens       - Summed prompt tokens (null if all unknown)
      total_cached_input_tokens - Summed cached tokens (null if none/unknown)
      total_output_tokens      - Summed completion tokens (null if unknown)
      total_estimated_cost_usd - Summed estimated cost in USD (null if unknown)
      breakdown                - Nested breakdown by stage, provider, model
      raw_rows                 - Count of usage rows (includes FAILED)
    """
    try:
        summary = get_job_usage_summary(job_id)
        return summary
    except Exception as e:
        logger.error("GET /usage/job/%s error: %s", job_id, e)
        raise HTTPException(status_code=500, detail=f"Usage summary error: {str(e)}")


@router.get("/usage/today", response_model=None)
async def get_today_usage() -> Dict[str, Any]:
    """
    Return today's (UTC) AI usage aggregated by provider.

    Response fields:
      date_utc    - Current UTC date (YYYY-MM-DD)
      providers   - Per-provider summary dict
        {provider}: {
          calls_today
          input_tokens_today
          cached_input_tokens_today
          output_tokens_today
          estimated_cost_today
        }
      total       - Same fields summed across all providers
    """
    try:
        summary = get_today_usage_summary()
        return summary
    except Exception as e:
        logger.error("GET /usage/today error: %s", e)
        raise HTTPException(status_code=500, detail=f"Today usage error: {str(e)}")


@router.get("/usage/stats", response_model=None)
async def get_usage_stats() -> Dict[str, Any]:
    """
    Return historical AI usage statistics.

    Response fields:
      completed_jobs_with_ai        - TRANSLATED jobs that used AI
      average_calls_per_job         - Avg API calls per AI-processed job (null if no data)
      average_estimated_cost_per_job - Avg USD cost per job (null if unknown)
      total_calls_all_time          - All-time total provider calls
      total_estimated_cost_all_time  - All-time total estimated cost (null if unknown)
    """
    try:
        stats = get_historical_stats()
        return stats
    except Exception as e:
        logger.error("GET /usage/stats error: %s", e)
        raise HTTPException(status_code=500, detail=f"Usage stats error: {str(e)}")
