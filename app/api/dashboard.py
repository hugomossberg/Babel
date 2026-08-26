import os
import json
import time
import asyncio
import httpx
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from google import genai

from app.core.db import (
    get_jobs, get_job_by_id, get_job_stats, delete_job, clear_all_jobs,
    get_setting, set_setting, get_positive_int_setting, get_jobs_by_status
)
from app.core.security import mask_secret, is_masked_secret, resolve_secret_key
from app.services.docker_controller import docker_controller
from app.services.bazarr_controller import bazarr_controller
from app.services.scanner import scan_library_folders
from app.services.jellyfin_notifier import notify_jellyfin_library_refresh, check_jellyfin_connection
from app.services.plex_notifier import notify_plex_library_refresh, check_plex_connection
from app.core.ai_providers import get_default_model, get_provider_catalog, normalize_provider

router = APIRouter()

class AISettingsRequest(BaseModel):
    ai_provider: Optional[str] = None
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    anthropic_model: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openrouter_model: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    deepseek_model: Optional[str] = None
    custom_openai_url: Optional[str] = None
    custom_openai_api_key: Optional[str] = None
    custom_openai_model: Optional[str] = None
    deepl_api_key: Optional[str] = None
    deepl_model_type: Optional[str] = None
    ollama_url: Optional[str] = None
    ollama_model: Optional[str] = None
    escalation_provider: Optional[str] = None
    escalation_model: Optional[str] = None
    escalate_to_pro: Optional[bool] = None
    batch_size: Optional[int] = Field(default=None, ge=1)
    max_concurrent_jobs: Optional[int] = Field(default=None, ge=1)
    batch_concurrency: Optional[int] = Field(default=None, ge=1)
    glossary: Optional[str] = None
    # Daily request budget: 0 = Unlimited (default), positive integer = daily limit
    daily_request_budget_gemini: Optional[int] = Field(default=None, ge=0)
    daily_request_budget_openai: Optional[int] = Field(default=None, ge=0)
    daily_request_budget_anthropic: Optional[int] = Field(default=None, ge=0)
    daily_request_budget_openrouter: Optional[int] = Field(default=None, ge=0)
    daily_request_budget_deepseek: Optional[int] = Field(default=None, ge=0)
    daily_request_budget_custom: Optional[int] = Field(default=None, ge=0)
    daily_request_budget_deepl: Optional[int] = Field(default=None, ge=0)
    daily_request_budget_ollama: Optional[int] = Field(default=None, ge=0)


class TestAIRequest(BaseModel):
    provider: str
    api_key: Optional[str] = ""
    model: Optional[str] = ""
    url: Optional[str] = ""

class TestBazarrRequest(BaseModel):
    bazarr_url: str
    bazarr_api_key: str

class TestJellyfinRequest(BaseModel):
    jellyfin_url: str
    jellyfin_api_key: Optional[str] = ""

class TestPlexRequest(BaseModel):
    plex_url: str
    plex_token: Optional[str] = ""

class TargetLangItem(BaseModel):
    name: str
    code: str
    enabled: bool

class LanguagesSettingsRequest(BaseModel):
    languages: List[TargetLangItem]

class MediaFoldersSettingsRequest(BaseModel):
    media_series_path: str
    media_movies_path: str
    remote_path_prefix: Optional[str] = ""
    local_path_prefix: Optional[str] = ""


class IntegrationsSettingsRequest(BaseModel):
    enable_bazarr_check: bool
    bazarr_url: str
    bazarr_api_key: str
    bazarr_container_name: str
    wait_time_seconds: Optional[int] = None  # Deprecated legacy parameter, ignored by runtime
    notify_jellyfin: bool
    jellyfin_url: str
    jellyfin_api_key: str
    notify_plex: Optional[bool] = False
    plex_url: Optional[str] = ""
    plex_token: Optional[str] = ""
    plex_path_babel_prefix: Optional[str] = ""
    plex_path_plex_prefix: Optional[str] = ""

class ContainerActionRequest(BaseModel):
    container_name: str
    action: str

class DeleteSubRequest(BaseModel):
    video_path: str

@router.get("/stats")
async def api_stats() -> Dict[str, Any]:
    return get_job_stats()

@router.get("/active-jobs")
async def api_active_jobs() -> Dict[str, Any]:
    """
    Return all currently active (non-terminal) jobs as a dict keyed by video_path.
    Cheap endpoint — pure DB query, no filesystem scan.
    Used by Library auto-refresh to update per-item job state.
    """
    from app.core.db import ACTIVE_JOB_STATUSES, get_jobs_by_status
    active = get_jobs_by_status(list(ACTIVE_JOB_STATUSES))
    result = {}
    for job in active:
        vp = os.path.normpath(job.get("video_path", ""))
        result[vp] = {
            "id": job.get("id"),
            "status": job.get("status"),
            "processed_lines": job.get("processed_lines"),
            "total_lines": job.get("total_lines"),
            "defer_reason": job.get("defer_reason"),
            "error_message": job.get("error_message"),
            "waiting_provider": job.get("waiting_provider"),
            "waiting_model": job.get("waiting_model"),
            "primary_provider": job.get("primary_provider"),
            "primary_model": job.get("primary_model"),
        }
    return result

@router.get("/jobs")
async def api_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    # Bug #33: GET endpoint should NEVER delete/mutate data
    return get_jobs(limit=limit)

@router.get("/jobs/{job_id}")
async def api_job_detail(job_id: int) -> Dict[str, Any]:
    job = get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.get("/queue/deferred")
@router.get("/queue")
async def api_get_deferred_queue() -> List[Dict[str, Any]]:
    """Return all DEFERRED jobs in FIFO order."""
    deferred_jobs = get_jobs_by_status(["DEFERRED"])
    deferred_jobs.sort(
        key=lambda j: (j.get("deferred_at") or j.get("created_at") or "", j.get("id") or 0)
    )
    return deferred_jobs

@router.delete("/jobs/{job_id}")
async def api_delete_job(job_id: int) -> Dict[str, Any]:
    from app.services.pipeline import pipeline
    pipeline.cancel_job(job_id)
    delete_job(job_id)
    return {"status": "deleted", "id": job_id}

@router.delete("/jobs")
async def api_clear_jobs() -> Dict[str, Any]:
    from app.services.pipeline import pipeline
    # Cancel all active asyncio tasks BEFORE clearing DB.
    # If we cleared DB first, running tasks could attempt to publish results
    # to non-existent job rows (FK violation) or create orphaned usage rows.
    active_job_ids = list(pipeline._active_tasks.keys())
    for job_id in active_job_ids:
        pipeline.cancel_job(job_id)
    clear_all_jobs()
    return {"status": "cleared"}

@router.delete("/subtitles")
async def api_delete_subtitles(req: DeleteSubRequest):
    from app.core.security import validate_media_path
    video_path = validate_media_path(req.video_path)

    # Guard: refuse subtitle deletion while an active job is processing this video.
    # Deleting during TRANSLATING would leave the video with no subtitle if the job fails.
    # Uses the full ACTIVE_JOB_STATUSES set — the single source of truth — not a subset.
    from app.core.db import get_active_job_for_video, ACTIVE_JOB_STATUSES
    active_job = get_active_job_for_video(video_path)
    if active_job and active_job.get("status") in ACTIVE_JOB_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete subtitles: an active job ({active_job['status']}) is "
                   f"currently processing this video. Wait for it to complete first."
        )


    base_no_ext, _ = os.path.splitext(video_path)
    parent_dir = os.path.dirname(video_path)
    base_name = os.path.basename(base_no_ext)

    # Bug #37: Only delete target language SRTs, not source subtitles
    # Get configured target languages to know which files Babel created
    target_codes = set()
    try:
        langs = json.loads(get_setting("languages", "[]"))
        target_codes = {l["code"] for l in langs if l.get("enabled", True)}
    except Exception:
        target_codes = set()

    from app.core.languages import get_language
    delete_suffixes = set()
    for code in target_codes:
        lang_obj = get_language(code)
        variants = lang_obj.aliases if lang_obj else [code]
        for v in variants:
            delete_suffixes.add(f".{v}.srt")

    deleted_files = []
    if os.path.exists(parent_dir):
        for f in os.listdir(parent_dir):
            if not f.startswith(base_name + "."):
                continue
            fname_lower = f.lower()
            # Only delete if it matches a target language suffix
            if any(fname_lower.endswith(suffix) for suffix in delete_suffixes):
                full_sub_path = os.path.join(parent_dir, f)
                try:
                    os.remove(full_sub_path)
                    deleted_files.append(f)
                except Exception:
                    pass
            # Also clean up babel-replaced backups
            if f.endswith(".babel-replaced"):
                try:
                    os.remove(os.path.join(parent_dir, f))
                    deleted_files.append(f)
                except Exception:
                    pass

    # Clean up job history for this video
    from app.core.db import DB_PATH
    import sqlite3
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM jobs WHERE video_path = ?", (req.video_path,))
        conn.commit()

    await notify_jellyfin_library_refresh()
    if get_setting("notify_plex", "false").lower() == "true":
        await notify_plex_library_refresh(video_path)
    invalidate_media_cache()
    return {"status": "deleted", "deleted_files": deleted_files}

@router.get("/settings/all")
async def api_get_all_settings() -> Dict[str, Any]:
    raw_key = get_setting("gemini_api_key", "")

    langs_json = get_setting("languages", "")
    languages = []
    if langs_json:
        try:
            languages = json.loads(langs_json)
        except Exception:
            pass
    if not languages:
        languages = [{"name": "Swedish", "code": "sv", "enabled": True}]

    docker_info = await docker_controller.get_container_status(get_setting("bazarr_container_name", "bazarr"))

    from app.core.languages import LANGUAGES
    available_languages = [{"name": lang.display_name, "code": lang.code} for lang in LANGUAGES]

    return {
        "ai": {
            "ai_provider": get_setting("ai_provider", "gemini"),
            "gemini_api_key": mask_secret(raw_key),
            "has_api_key": bool(raw_key),
            "gemini_model": get_setting("gemini_model", get_default_model("gemini")),
            "openai_api_key": mask_secret(get_setting("openai_api_key", "")),
            "has_openai_key": bool(get_setting("openai_api_key", "")),
            "openai_model": get_setting("openai_model", get_default_model("openai")),
            "anthropic_api_key": mask_secret(get_setting("anthropic_api_key", "")),
            "has_anthropic_key": bool(get_setting("anthropic_api_key", "")),
            "anthropic_model": get_setting("anthropic_model", get_default_model("anthropic")),
            "openrouter_api_key": mask_secret(get_setting("openrouter_api_key", "")),
            "has_openrouter_key": bool(get_setting("openrouter_api_key", "")),
            "openrouter_model": get_setting("openrouter_model", get_default_model("openrouter")),
            "deepseek_api_key": mask_secret(get_setting("deepseek_api_key", "")),
            "has_deepseek_key": bool(get_setting("deepseek_api_key", "")),
            "deepseek_model": get_setting("deepseek_model", get_default_model("deepseek")),
            "custom_openai_url": get_setting("custom_openai_url", "http://localhost:8000/v1"),
            "custom_openai_api_key": mask_secret(get_setting("custom_openai_api_key", "")),
            "has_custom_openai_key": bool(get_setting("custom_openai_api_key", "")),
            "custom_openai_model": get_setting("custom_openai_model", get_default_model("custom")),
            "deepl_api_key": mask_secret(get_setting("deepl_api_key", "")),
            "deepl_model_type": get_setting("deepl_model_type", get_default_model("deepl")),
            "ollama_url": get_setting("ollama_url", "http://localhost:11434"),
            "ollama_model": get_setting("ollama_model", get_default_model("ollama")),
            "escalation_provider": get_setting("escalation_provider", "none"),
            "escalation_model": get_setting("escalation_model", ""),
            "escalate_to_pro": get_setting("escalate_to_pro", "false").lower() == "true",
            "batch_size": get_positive_int_setting("batch_size", 150),
            "max_concurrent_jobs": get_positive_int_setting("max_concurrent_jobs", 3),
            "batch_concurrency": get_positive_int_setting("batch_concurrency", 2),
            "glossary": get_setting("glossary", ""),
            # Daily request budgets (0 = Unlimited)
            "daily_request_budget_gemini": int(get_setting("daily_request_budget_gemini", "0") or "0"),
            "daily_request_budget_openai": int(get_setting("daily_request_budget_openai", "0") or "0"),
            "daily_request_budget_anthropic": int(get_setting("daily_request_budget_anthropic", "0") or "0"),
            "daily_request_budget_openrouter": int(get_setting("daily_request_budget_openrouter", "0") or "0"),
            "daily_request_budget_deepseek": int(get_setting("daily_request_budget_deepseek", "0") or "0"),
            "daily_request_budget_custom": int(get_setting("daily_request_budget_custom", "0") or "0"),
            "daily_request_budget_deepl": int(get_setting("daily_request_budget_deepl", "0") or "0"),
            "daily_request_budget_ollama": int(get_setting("daily_request_budget_ollama", "0") or "0"),
        },
        "modules": {
            "clean_sdh": get_setting("clean_sdh", "true").lower() == "true",
            "extract_target_embedded": get_setting("extract_target_embedded", "true").lower() == "true",
            "extract_source_embedded": get_setting("extract_source_embedded", "true").lower() == "true",
            "auto_repair_unhealthy": get_setting("auto_repair_unhealthy", "true").lower() == "true",
            "strict_sync_lock": get_setting("strict_sync_lock", "true").lower() == "true",
            # original_language_guard removed in v2.3.43 — kept in DB for backward compat but no longer active
        },
        "languages": languages,
        "available_languages": available_languages,
        "folders": {
            "media_series_path": get_setting("media_series_path", "/tv"),
            "media_movies_path": get_setting("media_movies_path", "/movies"),
            "remote_path_prefix": get_setting("remote_path_prefix", ""),
            "local_path_prefix": get_setting("local_path_prefix", "")
        },
        "integrations": {
            "enable_bazarr_check": get_setting("enable_bazarr_check", "true").lower() == "true",
            "bazarr_url": get_setting("bazarr_url", "http://bazarr:6767"),
            "bazarr_api_key": mask_secret(get_setting("bazarr_api_key", "")),
            "bazarr_container_name": get_setting("bazarr_container_name", "bazarr"),
            "bazarr_container_status": docker_info,
            "wait_time_seconds": int(get_setting("wait_time_seconds", "15")),
            "notify_jellyfin": get_setting("notify_jellyfin", "false").lower() == "true",
            "jellyfin_url": get_setting("jellyfin_url", "http://jellyfin:8096"),
            "jellyfin_api_key": mask_secret(get_setting("jellyfin_api_key", "")),
            "notify_plex": get_setting("notify_plex", "false").lower() == "true",
            "plex_url": get_setting("plex_url", ""),
            "plex_token": mask_secret(get_setting("plex_token", "")),
            "plex_path_babel_prefix": get_setting("plex_path_babel_prefix", ""),
            "plex_path_plex_prefix": get_setting("plex_path_plex_prefix", "")
        }
    }

@router.post("/integrations/container-action")
async def api_container_action(req: ContainerActionRequest):
    info = await docker_controller.get_container_status(req.container_name)
    target_action = req.action
    if target_action == "toggle":
        target_action = "stop" if info.get("running") else "start"
    res = await docker_controller.toggle_container(req.container_name, target_action)
    return res

@router.get("/quota")
async def api_get_quota_status():
    """Return per-provider quota / budget status for UI display."""
    from app.core.quota import get_quota_status_for_provider
    active_provider = get_setting("ai_provider", "gemini").lower()
    providers = ["gemini", "openai", "anthropic", "openrouter", "deepseek", "custom", "deepl", "ollama"]
    quota_data = {p: get_quota_status_for_provider(p) for p in providers}
    return {
        "active_provider": active_provider,
        "providers": quota_data,
    }

@router.post("/quota/{provider}/unblock")
async def api_unblock_provider(provider: str):
    """Manually unblock a provider (admin action)."""
    from app.core.quota import unblock_provider
    allowed = {"gemini", "openai", "anthropic", "openrouter", "deepseek", "custom", "deepl", "ollama"}
    if provider not in allowed:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{provider}'")
    unblock_provider(provider)
    return {"status": "unblocked", "provider": provider}

@router.post("/settings/ai")
async def api_save_ai_settings(req: AISettingsRequest):
    if req.ai_provider is not None and req.ai_provider.strip():
        set_setting("ai_provider", req.ai_provider)
    if req.gemini_api_key is not None and not is_masked_secret(req.gemini_api_key):
        set_setting("gemini_api_key", req.gemini_api_key.strip())
    if req.gemini_model is not None and req.gemini_model.strip():
        set_setting("gemini_model", req.gemini_model)
    if req.openai_api_key is not None and not is_masked_secret(req.openai_api_key):
        set_setting("openai_api_key", req.openai_api_key.strip())
    if req.openai_model is not None and req.openai_model.strip():
        set_setting("openai_model", req.openai_model)
    if req.anthropic_api_key is not None and not is_masked_secret(req.anthropic_api_key):
        set_setting("anthropic_api_key", req.anthropic_api_key.strip())
    if req.anthropic_model is not None and req.anthropic_model.strip():
        set_setting("anthropic_model", req.anthropic_model.strip())
    if req.openrouter_api_key is not None and not is_masked_secret(req.openrouter_api_key):
        set_setting("openrouter_api_key", req.openrouter_api_key.strip())
    if req.openrouter_model is not None and req.openrouter_model.strip():
        set_setting("openrouter_model", req.openrouter_model.strip())
    if req.deepseek_api_key is not None and not is_masked_secret(req.deepseek_api_key):
        set_setting("deepseek_api_key", req.deepseek_api_key.strip())
    if req.deepseek_model is not None and req.deepseek_model.strip():
        set_setting("deepseek_model", req.deepseek_model.strip())
    if req.custom_openai_url is not None and req.custom_openai_url.strip():
        set_setting("custom_openai_url", req.custom_openai_url.strip())
    if req.custom_openai_api_key is not None and not is_masked_secret(req.custom_openai_api_key):
        set_setting("custom_openai_api_key", req.custom_openai_api_key.strip())
    if req.custom_openai_model is not None and req.custom_openai_model.strip():
        set_setting("custom_openai_model", req.custom_openai_model.strip())
    if req.deepl_api_key is not None and not is_masked_secret(req.deepl_api_key):
        set_setting("deepl_api_key", req.deepl_api_key.strip())
    if req.deepl_model_type is not None and req.deepl_model_type.strip():
        set_setting("deepl_model_type", req.deepl_model_type.strip())
    if req.ollama_url is not None and req.ollama_url.strip():
        set_setting("ollama_url", req.ollama_url.strip())
    if req.ollama_model is not None and req.ollama_model.strip():
        set_setting("ollama_model", req.ollama_model.strip())
    if req.escalation_provider is not None and req.escalation_provider.strip():
        set_setting("escalation_provider", req.escalation_provider)
    if req.escalation_model is not None and req.escalation_model.strip():
        set_setting("escalation_model", req.escalation_model.strip())
    if req.escalate_to_pro is not None:
        set_setting("escalate_to_pro", "true" if req.escalate_to_pro else "false")
    if req.batch_size is not None:
        safe_batch_size = max(1, int(req.batch_size))
        set_setting("batch_size", str(safe_batch_size))
    if req.max_concurrent_jobs is not None:
        safe_max_jobs = max(1, int(req.max_concurrent_jobs))
        set_setting("max_concurrent_jobs", str(safe_max_jobs))
    if req.batch_concurrency is not None:
        safe_batch_concurrency = max(1, int(req.batch_concurrency))
        set_setting("batch_concurrency", str(safe_batch_concurrency))
    if req.glossary is not None:
        set_setting("glossary", req.glossary)
    # Daily request budgets: 0 = unlimited
    for provider_key in ["gemini", "openai", "anthropic", "openrouter", "deepseek", "custom", "deepl", "ollama"]:
        budget_field = f"daily_request_budget_{provider_key}"
        budget_val = getattr(req, budget_field, None)
        if budget_val is not None:
            set_setting(budget_field, str(max(0, int(budget_val))))
    return {"status": "saved"}


_MODELS_CACHE: Dict[str, tuple[float, List[Dict[str, str]]]] = {}


@router.get("/settings/providers")
async def api_get_providers():
    from app.core.ai_providers import PROVIDERS
    return {"providers": {k: {"id": v.id, "label": v.label, "general_llm": v.general_llm, "default_model": v.default_model} for k, v in PROVIDERS.items()}}

@router.get("/settings/models")
async def api_get_provider_models(provider: str = Query("gemini"), url: Optional[str] = None, refresh: bool = Query(False)):
    import logging
    _log = logging.getLogger(__name__)
    from app.core.ai_providers import (
        get_provider_catalog,
        get_default_model,
        normalize_provider,
        filter_gemini_models,
        filter_openai_models,
        filter_anthropic_models,
        filter_openrouter_models,
        filter_ollama_models,
        filter_custom_models,
    )
    prov = normalize_provider(provider or "gemini")
    cache_key = f"{prov}:{url or ''}"
    now = datetime.now().timestamp()

    if not refresh and cache_key in _MODELS_CACHE:
        cached_time, cached_data = _MODELS_CACHE[cache_key]
        ttl = 60.0 if prov in ["ollama", "custom"] else 3600.0
        if now - cached_time < ttl:
            return {"provider": prov, "default_model": get_default_model(prov), "models": cached_data, "cached": True}

    models: List[Dict[str, str]] = []

    if prov == "gemini":
        key = resolve_secret_key("", "gemini_api_key")
        if key:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    res = await client.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={key}")
                    if res.status_code == 200:
                        data = res.json()
                        discovered = filter_gemini_models(data.get("models", []))
                        if discovered:
                            models = discovered
            except Exception as e:
                _log.debug(f"Gemini live discovery failed: {e}")
        if not models:
            models = get_provider_catalog("gemini")

    elif prov == "openai":
        key = resolve_secret_key("", "openai_api_key")
        if key:
            try:
                headers = {"Authorization": f"Bearer {key}"}
                async with httpx.AsyncClient(timeout=5.0) as client:
                    res = await client.get("https://api.openai.com/v1/models", headers=headers)
                    if res.status_code == 200:
                        data = res.json()
                        discovered = filter_openai_models(data.get("data", []))
                        if discovered:
                            models = discovered
            except Exception as e:
                _log.debug(f"OpenAI live discovery failed: {e}")
        if not models:
            models = get_provider_catalog("openai")

    elif prov == "anthropic":
        key = resolve_secret_key("", "anthropic_api_key")
        if key:
            try:
                headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
                async with httpx.AsyncClient(timeout=5.0) as client:
                    res = await client.get("https://api.anthropic.com/v1/models", headers=headers)
                    if res.status_code == 200:
                        data = res.json()
                        discovered = filter_anthropic_models(data.get("data", []))
                        if discovered:
                            models = discovered
            except Exception as e:
                _log.debug(f"Anthropic live discovery failed: {e}")
        if not models:
            models = get_provider_catalog("anthropic")

    elif prov == "deepseek":
        models = get_provider_catalog("deepseek")

    elif prov == "openrouter":
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                res = await client.get("https://openrouter.ai/api/v1/models")
                if res.status_code == 200:
                    data = res.json()
                    discovered = filter_openrouter_models(data.get("data", []))
                    if discovered:
                        models = discovered
                else:
                    raise Exception(f"Status {res.status_code}")
        except Exception as e:
            _log.debug(f"OpenRouter live discovery failed: {e}")
            if cache_key in _MODELS_CACHE:
                return {"provider": prov, "default_model": get_default_model(prov), "models": _MODELS_CACHE[cache_key][1], "cached": True}
            models = get_provider_catalog("openrouter")
            return {"provider": prov, "default_model": get_default_model(prov), "models": models, "cached": False}

    elif prov in ["ollama", "localai"]:
        endpoint = (url or get_setting("ollama_url", "http://localhost:11434")).rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{endpoint}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    discovered = filter_ollama_models(data.get("models", []))
                    if discovered:
                        models = discovered
                else:
                    raise Exception("Ollama error")
        except Exception as e:
            _log.debug(f"Ollama live discovery failed: {e}")
            if cache_key in _MODELS_CACHE:
                return {"provider": prov, "default_model": get_default_model(prov), "models": _MODELS_CACHE[cache_key][1], "cached": True}
            models = get_provider_catalog("ollama")
            return {"provider": prov, "default_model": get_default_model(prov), "models": models, "cached": False}

    elif prov in ["custom", "custom_openai"]:
        endpoint = (url or get_setting("custom_openai_url", "http://localhost:8000/v1")).rstrip("/")
        try:
            key = resolve_secret_key("", "custom_openai_api_key")
            headers = {"Authorization": f"Bearer {key or 'dummy'}"}
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{endpoint}/models", headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    discovered = filter_custom_models(data.get("data", []))
                    if discovered:
                        models = discovered
                else:
                    raise Exception("Custom error")
        except Exception as e:
            _log.debug(f"Custom OpenAI live discovery failed: {e}")
            if cache_key in _MODELS_CACHE:
                return {"provider": prov, "default_model": get_default_model(prov), "models": _MODELS_CACHE[cache_key][1], "cached": True}
            models = get_provider_catalog("custom")
            return {"provider": prov, "default_model": get_default_model(prov), "models": models, "cached": False}

    elif prov == "deepl":
        models = get_provider_catalog("deepl")

    if not models:
        models = get_provider_catalog(prov)

    _MODELS_CACHE[cache_key] = (now, models)
    return {"provider": prov, "default_model": get_default_model(prov), "models": models, "cached": False}


@router.post("/settings/test-ai")
async def api_test_ai(req: TestAIRequest):
    provider = (req.provider or "gemini").lower()
    setting_map = {
        "deepl": "deepl_api_key",
        "openai": "openai_api_key",
        "gemini": "gemini_api_key",
        "anthropic": "anthropic_api_key",
        "openrouter": "openrouter_api_key",
        "deepseek": "deepseek_api_key",
        "custom": "custom_openai_api_key",
        "custom_openai": "custom_openai_api_key",
    }
    key = resolve_secret_key(req.api_key, setting_map.get(provider, "gemini_api_key")) if provider in setting_map else (req.api_key.strip() if req.api_key else "")

    if provider == "deepl":
        if not key:
            raise HTTPException(status_code=400, detail="No DeepL API Key provided or saved")
        url = "https://api-free.deepl.com/v2/usage" if key.endswith(":fx") else "https://api.deepl.com/v2/usage"
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                res = await client.get(url, headers={"Authorization": f"DeepL-Auth-Key {key}"})
                res.raise_for_status()
                return {"status": "ok", "response": "DeepL Connected"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
    elif provider in ["ollama", "localai"]:
        url = (req.url or get_setting("ollama_url", "http://localhost:11434")).rstrip("/")
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                res = await client.get(f"{url}/api/tags")
                res.raise_for_status()
                return {"status": "ok", "response": "Ollama Connected"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Cannot reach Ollama at {url}: {e}")
    elif provider == "openai":
        import openai
        if not key:
            raise HTTPException(status_code=400, detail="No OpenAI API Key provided or saved")
        try:
            client = openai.OpenAI(api_key=key)
            resp = client.chat.completions.create(
                model=req.model or "gpt-4o-mini",
                messages=[{"role": "user", "content": "Respond with 'Connected'"}],
                max_tokens=10
            )
            return {"status": "ok", "response": "Connected"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    elif provider == "anthropic":
        if not key:
            raise HTTPException(status_code=400, detail="No Anthropic API Key provided or saved")
        test_model = req.model or "claude-sonnet-5"
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                res = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": test_model,
                        "max_tokens": 10,
                        "messages": [{"role": "user", "content": "Respond with 'Connected'"}],
                    },
                )
                if res.status_code != 200:
                    try:
                        err_json = res.json()
                        err_msg = err_json.get("error", {}).get("message", res.text)
                    except Exception:
                        err_msg = res.text
                    raise HTTPException(status_code=400, detail=f"Anthropic error ({res.status_code}): {err_msg}")
                return {"status": "ok", "response": "Connected"}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
    elif provider == "openrouter":
        import openai
        if not key:
            raise HTTPException(status_code=400, detail="No OpenRouter API Key provided or saved")
        try:
            client = openai.OpenAI(
                api_key=key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://github.com/hugomossberg/Babel",
                    "X-Title": "Babel Subtitles",
                },
            )
            resp = client.chat.completions.create(
                model=req.model or "anthropic/claude-sonnet-5",
                messages=[{"role": "user", "content": "Respond with 'Connected'"}],
                max_tokens=10,
            )
            return {"status": "ok", "response": "Connected"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    elif provider == "deepseek":
        import openai
        if not key:
            raise HTTPException(status_code=400, detail="No DeepSeek API Key provided or saved")
        try:
            client = openai.OpenAI(
                api_key=key,
                base_url="https://api.deepseek.com",
            )
            resp = client.chat.completions.create(
                model=req.model or "deepseek-v4-flash",
                messages=[{"role": "user", "content": "Respond with 'Connected'"}],
                max_tokens=10,
            )
            return {"status": "ok", "response": "Connected"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    elif provider in ["custom", "custom_openai"]:
        import openai
        base_url = (req.url or get_setting("custom_openai_url", "http://localhost:8000/v1")).rstrip("/")
        if not base_url:
            raise HTTPException(status_code=400, detail="No Custom OpenAI Base URL provided or saved")
        try:
            client = openai.OpenAI(
                api_key=key or "dummy",
                base_url=base_url,
            )
            test_model = req.model or get_setting("custom_openai_model", "default")
            resp = client.chat.completions.create(
                model=test_model,
                messages=[{"role": "user", "content": "Respond with 'Connected'"}],
                max_tokens=10,
            )
            return {"status": "ok", "response": "Connected"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        if not key:
            raise HTTPException(status_code=400, detail="No Gemini API Key provided or saved")
        try:
            client = genai.Client(api_key=key)
            test_model = req.model or "gemini-3.5-flash-lite"
            loop = asyncio.get_event_loop()

            def do_gemini_test():
                # models.get validates the API key and model existence in ~0.1s without queuing in generation backend
                try:
                    client.models.get(model=test_model)
                    return True
                except Exception:
                    # Fallback to direct generate_content
                    client.models.generate_content(
                        model=test_model,
                        contents="Respond with 'Connected'"
                    )
                    return True

            await asyncio.wait_for(loop.run_in_executor(None, do_gemini_test), timeout=15.0)
            return {"status": "ok", "response": "Connected"}
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail=f"Connection timed out for model '{test_model}'. Model may be overloaded or unavailable.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail=f"Connection timed out for model '{test_model}'. Model may be overloaded or unavailable.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

@router.post("/settings/test-bazarr")
async def api_test_bazarr(req: TestBazarrRequest):
    key = resolve_secret_key(req.bazarr_api_key, "bazarr_api_key")
    if not key:
        raise HTTPException(status_code=400, detail="No Bazarr API Key provided or saved")

    res = await bazarr_controller.get_status(req.bazarr_url, key)
    if res.get("connected"):
        return {"status": "ok", "version": res.get("version")}
    else:
        raise HTTPException(status_code=400, detail=res.get("message", "Connection failed"))

@router.post("/settings/test-jellyfin")
async def api_test_jellyfin(req: TestJellyfinRequest):
    key = resolve_secret_key(req.jellyfin_api_key, "jellyfin_api_key")
    if not key:
        raise HTTPException(status_code=400, detail="No Jellyfin API Token provided or saved")
    if not req.jellyfin_url:
        raise HTTPException(status_code=400, detail="No Jellyfin Server URL provided")

    res = await check_jellyfin_connection(req.jellyfin_url, key)
    if res.get("connected"):
        return {"status": "ok", "version": res.get("version"), "server_name": res.get("server_name")}
    else:
        raise HTTPException(status_code=400, detail=res.get("message", "Connection failed"))

@router.post("/settings/test-plex")
async def api_test_plex(req: TestPlexRequest):
    token = resolve_secret_key(req.plex_token, "plex_token")
    if not token:
        raise HTTPException(status_code=400, detail="No Plex Token provided or saved")
    if not req.plex_url:
        raise HTTPException(status_code=400, detail="No Plex Server URL provided")

    res = await check_plex_connection(req.plex_url, token)
    if res.get("connected"):
        return {"status": "ok", "sections_count": res.get("sections_count")}
    else:
        raise HTTPException(status_code=400, detail=res.get("message", "Connection failed"))

class ModulesSettingsRequest(BaseModel):
    clean_sdh: bool
    extract_target_embedded: bool
    extract_source_embedded: bool
    auto_repair_unhealthy: bool
    strict_sync_lock: bool
    # original_language_guard removed in v2.3.43 (kept for backward compat with old clients)
    original_language_guard: bool = False

@router.post("/settings/modules")
async def api_save_modules(req: ModulesSettingsRequest):
    set_setting("clean_sdh", "true" if req.clean_sdh else "false")
    set_setting("extract_target_embedded", "true" if req.extract_target_embedded else "false")
    set_setting("extract_source_embedded", "true" if req.extract_source_embedded else "false")
    set_setting("auto_repair_unhealthy", "true" if req.auto_repair_unhealthy else "false")
    set_setting("strict_sync_lock", "true" if req.strict_sync_lock else "false")
    # original_language_guard is silently accepted (backward compat) but no longer used in runtime
    return {"status": "saved"}

@router.post("/settings/languages")
async def api_save_languages(req: LanguagesSettingsRequest):
    set_setting("languages", json.dumps([l.dict() for l in req.languages]))
    return {"status": "saved"}

@router.post("/settings/folders")
async def api_save_folders(req: MediaFoldersSettingsRequest):
    set_setting("media_series_path", req.media_series_path)
    set_setting("media_movies_path", req.media_movies_path)
    set_setting("remote_path_prefix", req.remote_path_prefix)
    set_setting("local_path_prefix", req.local_path_prefix)
    return {"status": "saved"}

@router.post("/settings/integrations")
async def api_save_integrations(req: IntegrationsSettingsRequest):
    set_setting("enable_bazarr_check", "true" if req.enable_bazarr_check else "false")
    set_setting("bazarr_url", req.bazarr_url)
    if not is_masked_secret(req.bazarr_api_key):
        set_setting("bazarr_api_key", req.bazarr_api_key.strip() if req.bazarr_api_key else "")
    set_setting("bazarr_container_name", req.bazarr_container_name)
    if req.wait_time_seconds is not None:
        set_setting("wait_time_seconds", str(req.wait_time_seconds))
    set_setting("notify_jellyfin", "true" if req.notify_jellyfin else "false")
    set_setting("jellyfin_url", req.jellyfin_url)
    if not is_masked_secret(req.jellyfin_api_key):
        set_setting("jellyfin_api_key", req.jellyfin_api_key.strip() if req.jellyfin_api_key else "")
    set_setting("notify_plex", "true" if req.notify_plex else "false")
    set_setting("plex_url", (req.plex_url or "").strip())
    if not is_masked_secret(req.plex_token):
        set_setting("plex_token", req.plex_token.strip() if req.plex_token else "")
    set_setting("plex_path_babel_prefix", (req.plex_path_babel_prefix or "").strip())
    set_setting("plex_path_plex_prefix", (req.plex_path_plex_prefix or "").strip())
    return {"status": "saved"}

_media_cache: Dict[str, Any] = {}
_media_cache_time: float = 0.0
_media_cache_paths: list = []
_scan_lock = asyncio.Lock()
MEDIA_CACHE_TTL = 30.0  # seconds

def invalidate_media_cache():
    global _media_cache, _media_cache_time, _media_cache_paths
    _media_cache = {}
    _media_cache_time = 0.0
    _media_cache_paths = []

async def _attach_active_jobs_to_media(series_data: list, movies_data: list, all_paths: list) -> dict:
    from app.core.db import get_active_jobs_by_video_paths
    active_jobs_map = await asyncio.to_thread(get_active_jobs_by_video_paths, all_paths)

    def _slim_job(job: dict) -> dict:
        return {
            "id": job.get("id"),
            "status": job.get("status"),
            "processed_lines": job.get("processed_lines"),
            "total_lines": job.get("total_lines"),
            "defer_reason": job.get("defer_reason"),
            "error_message": job.get("error_message"),
            "waiting_provider": job.get("waiting_provider"),
            "waiting_model": job.get("waiting_model"),
            "primary_provider": job.get("primary_provider"),
            "primary_model": job.get("primary_model"),
        }

    for show in series_data:
        for ep in show.get("episodes", []):
            job = active_jobs_map.get(ep["path"])
            ep["active_job"] = _slim_job(job) if job else None

    for movie in movies_data:
        job = active_jobs_map.get(movie["path"])
        movie["active_job"] = _slim_job(job) if job else None

    return {
        "series": series_data,
        "movies": movies_data,
    }

@router.get("/media-files")
async def api_get_media_files(force: bool = False):
    global _media_cache, _media_cache_time, _media_cache_paths

    now = time.time()
    has_valid_cache = bool(_media_cache and (now - _media_cache_time < MEDIA_CACHE_TTL))

    if not force and has_valid_cache:
        return await _attach_active_jobs_to_media(
            _media_cache.get("series", []),
            _media_cache.get("movies", []),
            _media_cache_paths,
        )

    # If another scan is already running in background:
    if _scan_lock.locked():
        # If we have existing cached data, return it immediately so the UI is never blocked or empty
        if _media_cache:
            return await _attach_active_jobs_to_media(
                _media_cache.get("series", []),
                _media_cache.get("movies", []),
                _media_cache_paths,
            )
        # If no cache exists at all, wait for the scan to complete rather than throwing 429
        async with _scan_lock:
            return await _attach_active_jobs_to_media(
                _media_cache.get("series", []),
                _media_cache.get("movies", []),
                _media_cache_paths,
            )

    async with _scan_lock:
        now = time.time()
        if not force and _media_cache and (now - _media_cache_time < MEDIA_CACHE_TTL):
            return await _attach_active_jobs_to_media(
                _media_cache.get("series", []),
                _media_cache.get("movies", []),
                _media_cache_paths,
            )

        series_path = get_setting("media_series_path", "/tv")
        movies_path = get_setting("media_movies_path", "/movies")

        series_data = await asyncio.to_thread(scan_library_folders, series_path, "series")
        movies_data = await asyncio.to_thread(scan_library_folders, movies_path, "movies")

        all_paths: list = []
        for show in series_data:
            for ep in show.get("episodes", []):
                all_paths.append(ep["path"])
        for movie in movies_data:
            all_paths.append(movie["path"])

        _media_cache = {
            "series": series_data,
            "movies": movies_data,
        }
        _media_cache_paths = all_paths
        _media_cache_time = time.time()

        return await _attach_active_jobs_to_media(series_data, movies_data, all_paths)


from app.services.updates_controller import updates_controller
from pydantic import BaseModel

class TriggerUpdateReq(BaseModel):
    target_version: str

@router.get("/updates")
async def api_get_updates(force: bool = False):
    return await updates_controller.get_update_info(force_refresh=force)

@router.post("/updates/trigger")
async def api_trigger_update(req: TriggerUpdateReq):
    res = await updates_controller.trigger_update(req.target_version)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
    return res
