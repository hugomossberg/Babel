import os
import json
import asyncio
import httpx
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from google import genai

from app.core.db import (
    get_jobs, get_job_by_id, get_job_stats, delete_job, clear_all_jobs,
    get_setting, set_setting
)
from app.services.docker_controller import docker_controller
from app.services.bazarr_controller import bazarr_controller
from app.services.scanner import scan_library_folders
from app.services.jellyfin_notifier import notify_jellyfin_library_refresh

router = APIRouter()

class AISettingsRequest(BaseModel):
    ai_provider: Optional[str] = "gemini"
    gemini_api_key: Optional[str] = ""
    gemini_model: Optional[str] = "gemini-3.5-flash-lite"
    openai_api_key: Optional[str] = ""
    openai_model: Optional[str] = "gpt-4o-mini"
    deepl_api_key: Optional[str] = ""
    ollama_url: Optional[str] = "http://localhost:11434"
    ollama_model: Optional[str] = "llama3"
    escalation_provider: Optional[str] = "none"
    escalation_model: Optional[str] = ""
    escalate_to_pro: bool = False
    batch_size: int
    max_concurrency: int
    max_concurrent_jobs: Optional[int] = 1
    glossary: Optional[str] = ""

class TestAIRequest(BaseModel):
    provider: str
    api_key: Optional[str] = ""
    model: Optional[str] = ""
    url: Optional[str] = ""

class TestBazarrRequest(BaseModel):
    bazarr_url: str
    bazarr_api_key: str

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
    wait_time_seconds: int
    notify_jellyfin: bool
    jellyfin_url: str
    jellyfin_api_key: str

class ContainerActionRequest(BaseModel):
    container_name: str
    action: str

class DeleteSubRequest(BaseModel):
    video_path: str

@router.get("/stats")
async def api_stats() -> Dict[str, Any]:
    return get_job_stats()

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

@router.delete("/jobs/{job_id}")
async def api_delete_job(job_id: int) -> Dict[str, Any]:
    from app.services.pipeline import pipeline
    pipeline.cancel_job(job_id)
    delete_job(job_id)
    return {"status": "deleted", "id": job_id}

@router.delete("/jobs")
async def api_clear_jobs() -> Dict[str, Any]:
    clear_all_jobs()
    return {"status": "cleared"}

@router.delete("/subtitles")
async def api_delete_subtitles(req: DeleteSubRequest):
    base_no_ext, _ = os.path.splitext(req.video_path)
    parent_dir = os.path.dirname(req.video_path)
    base_name = os.path.basename(base_no_ext)
    
    # Bug #37: Only delete target language SRTs, not source subtitles
    # Get configured target languages to know which files Babel created
    target_codes = set()
    try:
        langs = json.loads(get_setting("languages", "[]"))
        target_codes = {l["code"] for l in langs if l.get("enabled", True)}
    except Exception:
        target_codes = {"sv"}
    
    # Also include common variants
    LANG_VARIANTS = {
        "sv": ["sv", "swe", "swedish"],
        "da": ["da", "dan", "danish"],
        "no": ["no", "nor", "nob", "norwegian"],
        "de": ["de", "ger", "german"],
        "fr": ["fr", "fre", "french"],
        "es": ["es", "spa", "spanish"],
        "fi": ["fi", "fin", "finnish"],
    }
    delete_suffixes = set()
    for code in target_codes:
        variants = LANG_VARIANTS.get(code, [code])
        for v in variants:
            delete_suffixes.add(f".{v}.srt")
    
    deleted_files = []
    if os.path.exists(parent_dir):
        for f in os.listdir(parent_dir):
            if not f.startswith(base_name):
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
    return {"status": "deleted", "deleted_files": deleted_files}

@router.get("/settings/all")
async def api_get_all_settings() -> Dict[str, Any]:
    # Bug #41: Mask API keys so they aren't sent to the browser in plaintext
    def mask_key(key: str) -> str:
        if not key or len(key) < 8:
            return key
        return "••••••••" + key[-4:]

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

    return {
        "ai": {
            "ai_provider": get_setting("ai_provider", "gemini"),
            "gemini_api_key": mask_key(raw_key),
            "has_api_key": bool(raw_key),
            "gemini_model": get_setting("gemini_model", "gemini-3.5-flash-lite"),
            "openai_api_key": mask_key(get_setting("openai_api_key", "")),
            "has_openai_key": bool(get_setting("openai_api_key", "")),
            "openai_model": get_setting("openai_model", "gpt-4o-mini"),
            "deepl_api_key": mask_key(get_setting("deepl_api_key", "")),
            "ollama_url": get_setting("ollama_url", "http://localhost:11434"),
            "ollama_model": get_setting("ollama_model", "llama3"),
            "escalation_provider": get_setting("escalation_provider", "none"),
            "escalation_model": get_setting("escalation_model", ""),
            "escalate_to_pro": get_setting("escalate_to_pro", "false").lower() == "true",
            "batch_size": int(get_setting("batch_size", "50")),
            "max_concurrency": int(get_setting("max_concurrency", "1")),
            "max_concurrent_jobs": int(get_setting("max_concurrent_jobs", "1")),
            "glossary": get_setting("glossary", "")
        },
        "modules": {
            "clean_sdh": get_setting("clean_sdh", "true").lower() == "true",
            "extract_target_embedded": get_setting("extract_target_embedded", "true").lower() == "true",
            "extract_source_embedded": get_setting("extract_source_embedded", "true").lower() == "true",
            "auto_repair_unhealthy": get_setting("auto_repair_unhealthy", "true").lower() == "true",
            "strict_sync_lock": get_setting("strict_sync_lock", "true").lower() == "true",
            "original_language_guard": get_setting("original_language_guard", "true").lower() == "true",
        },
        "languages": languages,
        "folders": {
            "media_series_path": get_setting("media_series_path", "/tv"),
            "media_movies_path": get_setting("media_movies_path", "/movies"),
            "remote_path_prefix": get_setting("remote_path_prefix", ""),
            "local_path_prefix": get_setting("local_path_prefix", "")
        },
        "integrations": {
            "enable_bazarr_check": get_setting("enable_bazarr_check", "true").lower() == "true",
            "bazarr_url": get_setting("bazarr_url", "http://bazarr:6767"),
            "bazarr_api_key": mask_key(get_setting("bazarr_api_key", "")),
            "bazarr_container_name": get_setting("bazarr_container_name", "bazarr"),
            "bazarr_container_status": docker_info,
            "wait_time_seconds": int(get_setting("wait_time_seconds", "15")),
            "notify_jellyfin": get_setting("notify_jellyfin", "true").lower() == "true",
            "jellyfin_url": get_setting("jellyfin_url", "http://jellyfin:8096"),
            "jellyfin_api_key": mask_key(get_setting("jellyfin_api_key", ""))
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

@router.post("/settings/ai")
async def api_save_ai_settings(req: AISettingsRequest):
    if req.ai_provider:
        set_setting("ai_provider", req.ai_provider)
    if req.gemini_api_key is not None and not req.gemini_api_key.startswith("••••••••"):
        set_setting("gemini_api_key", req.gemini_api_key.strip())
    if req.gemini_model:
        set_setting("gemini_model", req.gemini_model)
    if req.openai_api_key is not None and not req.openai_api_key.startswith("••••••••"):
        set_setting("openai_api_key", req.openai_api_key.strip())
    if req.openai_model:
        set_setting("openai_model", req.openai_model)
    if req.deepl_api_key is not None and not req.deepl_api_key.startswith("••••••••"):
        set_setting("deepl_api_key", req.deepl_api_key.strip())
    if req.ollama_url:
        set_setting("ollama_url", req.ollama_url.strip())
    if req.ollama_model:
        set_setting("ollama_model", req.ollama_model.strip())
    if req.escalation_provider:
        set_setting("escalation_provider", req.escalation_provider)
    if req.escalation_model is not None:
        set_setting("escalation_model", req.escalation_model.strip())
    set_setting("escalate_to_pro", "true" if req.escalate_to_pro else "false")
    set_setting("batch_size", str(req.batch_size))
    set_setting("max_concurrency", str(req.max_concurrency))
    if req.max_concurrent_jobs is not None:
        set_setting("max_concurrent_jobs", str(max(1, req.max_concurrent_jobs)))
    if req.glossary is not None:
        set_setting("glossary", req.glossary)
    return {"status": "saved"}

@router.post("/settings/test-ai")
async def api_test_ai(req: TestAIRequest):
    provider = (req.provider or "gemini").lower()
    key = req.api_key.strip() if req.api_key else ""
    if key.startswith("••••••••"):
        if provider == "deepl":
            key = get_setting("deepl_api_key", "")
        elif provider == "openai":
            key = get_setting("openai_api_key", "")
        elif provider == "gemini":
            key = get_setting("gemini_api_key", "")
            
    if provider == "deepl":
        if not key:
            key = get_setting("deepl_api_key", "")
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
            key = get_setting("openai_api_key", "")
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
    else:
        if not key:
            key = get_setting("gemini_api_key", "")
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

@router.post("/settings/test-bazarr")
async def api_test_bazarr(req: TestBazarrRequest):
    res = await bazarr_controller.get_status(req.bazarr_url, req.bazarr_api_key)
    if res.get("connected"):
        return {"status": "ok", "version": res.get("version")}
    else:
        raise HTTPException(status_code=400, detail=res.get("message", "Connection failed"))

class ModulesSettingsRequest(BaseModel):
    clean_sdh: bool
    extract_target_embedded: bool
    extract_source_embedded: bool
    auto_repair_unhealthy: bool
    strict_sync_lock: bool
    original_language_guard: bool

@router.post("/settings/modules")
async def api_save_modules(req: ModulesSettingsRequest):
    set_setting("clean_sdh", "true" if req.clean_sdh else "false")
    set_setting("extract_target_embedded", "true" if req.extract_target_embedded else "false")
    set_setting("extract_source_embedded", "true" if req.extract_source_embedded else "false")
    set_setting("auto_repair_unhealthy", "true" if req.auto_repair_unhealthy else "false")
    set_setting("strict_sync_lock", "true" if req.strict_sync_lock else "false")
    set_setting("original_language_guard", "true" if req.original_language_guard else "false")
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
    if not req.bazarr_api_key.startswith("••••••••"):
        set_setting("bazarr_api_key", req.bazarr_api_key)
    set_setting("bazarr_container_name", req.bazarr_container_name)
    set_setting("wait_time_seconds", str(req.wait_time_seconds))
    set_setting("notify_jellyfin", "true" if req.notify_jellyfin else "false")
    set_setting("jellyfin_url", req.jellyfin_url)
    if not req.jellyfin_api_key.startswith("••••••••"):
        set_setting("jellyfin_api_key", req.jellyfin_api_key)
    return {"status": "saved"}

@router.get("/media-files")
async def api_get_media_files():
    series_path = get_setting("media_series_path", "/tv")
    movies_path = get_setting("media_movies_path", "/movies")
    
    series_data = scan_library_folders(series_path, "series")
    movies_data = scan_library_folders(movies_path, "movies")
    
    return {
        "series": series_data,
        "movies": movies_data
    }
