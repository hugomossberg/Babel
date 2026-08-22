import logging
import os

from fastapi import APIRouter, BackgroundTasks, Request, HTTPException
from pathlib import Path
from typing import Optional

def validate_webhook_token(request: Request):
    from app.core.db import get_setting
    import secrets
    expected = get_setting("webhook_secret", "").strip()
    if not expected:
        return # Auth not configured
    
    token = request.query_params.get("secret", "").strip()
    if not token:
        token = request.headers.get("X-Webhook-Secret", "").strip()
    if not token:
        auth_header = request.headers.get("Authorization", "").strip()
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()

    if not token or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

def validate_path(video_path: str) -> str:
    from app.core.security import validate_media_path
    return validate_media_path(video_path)

from pydantic import BaseModel

from app.services.pipeline import pipeline

logger = logging.getLogger("babel.webhooks")
router = APIRouter(prefix="/webhook", tags=["webhooks"])

class ManualProcessRequest(BaseModel):
    video_path: str
    wait_seconds: Optional[int] = None
    title: Optional[str] = None
    force_retranslate: Optional[bool] = False

from app.core.db import get_setting

def translate_path(remote_path: str) -> str:
    if not remote_path:
        return remote_path

    from app.core.db import get_setting
    remote_prefix = get_setting("remote_path_prefix", "").strip()
    local_prefix = get_setting("local_path_prefix", "").strip()
    
    if remote_prefix:
        r_prefs = [p.strip() for p in remote_prefix.split(',')]
        l_prefs = [p.strip() for p in local_prefix.split(',')]
        
        for i, r_pref in enumerate(r_prefs):
            l_pref = l_prefs[i] if i < len(l_prefs) else ""
            if r_pref and remote_path.startswith(r_pref):
                return remote_path.replace(r_pref, l_pref, 1)

    return remote_path

@router.post("/sonarr")
async def sonarr_webhook(request: Request, background_tasks: BackgroundTasks):
    validate_webhook_token(request)
    try:
        payload = await request.json()
    except Exception:
        return {"status": "ignored", "reason": "invalid_json"}

    event_type = payload.get("eventType")
    logger.info(f"Received Sonarr webhook event: {event_type}")

    if event_type == "Test":
        return {"status": "success", "message": "Babel webhook test successful"}

    if event_type in ["Download", "Upgrade", "Rename"]:
        episode_file = payload.get("episodeFile", {})
        series = payload.get("series", {})
        episodes = payload.get("episodes", [])
        
        video_path = None
        if series and episode_file:
            series_path = series.get("path", "")
            rel_path = episode_file.get("relativePath", "")
            if series_path and rel_path:
                video_path = os.path.normpath(os.path.join(series_path, rel_path.lstrip("/")))
        
        # Fallback to episode_file.path if relativePath wasn't available
        if not video_path:
            video_path = episode_file.get("path")

        title = None
        if series and episodes:
            ep_title = episodes[0].get("title", "")
            s_num = episodes[0].get("seasonNumber", 0)
            e_num = episodes[0].get("episodeNumber", 0)
            title = f"{series.get('title', '')} - S{s_num:02d}E{e_num:02d} - {ep_title}"

        series_title = series.get("title", "") if series else None

        if video_path:
            video_path = translate_path(video_path)
            try: video_path = validate_path(video_path)
            except HTTPException as e:
                logger.error(f"Webhook path validation failed: {e.detail}")
                return {"status": "error", "reason": e.detail}
            logger.info(f"Queueing Babel processing for Sonarr episode: {video_path}")
            background_tasks.add_task(
                pipeline.process_video_file,
                video_path=video_path,
                event_source="SONARR",
                title=title,
                series_title=series_title
            )
            return {"status": "queued", "event": event_type, "video_path": video_path, "title": title}

    return {"status": "ignored", "event": event_type}

@router.post("/radarr")
async def radarr_webhook(request: Request, background_tasks: BackgroundTasks):
    validate_webhook_token(request)
    try:
        payload = await request.json()
    except Exception:
        return {"status": "ignored", "reason": "invalid_json"}

    event_type = payload.get("eventType")
    logger.info(f"Received Radarr webhook event: {event_type}")

    if event_type == "Test":
        return {"status": "success", "message": "Babel webhook test successful"}

    if event_type in ["Download", "Upgrade", "Rename"]:
        movie_file = payload.get("movieFile", {})
        movie = payload.get("movie", {})
        video_path = None
        if movie and movie_file:
            folder_path = movie.get("folderPath") or movie.get("path", "")
            rel_path = movie_file.get("relativePath", "")
            if folder_path and rel_path:
                video_path = os.path.normpath(os.path.join(folder_path, rel_path.lstrip("/")))
                
        if not video_path:
            video_path = movie_file.get("path") or movie.get("path")
        
        title = None
        if movie:
            title = f"{movie.get('title', '')} ({movie.get('year', '')})"

        if video_path:
            video_path = translate_path(video_path)
            try: video_path = validate_path(video_path)
            except HTTPException as e:
                logger.error(f"Webhook path validation failed: {e.detail}")
                return {"status": "error", "reason": e.detail}
            logger.info(f"Queueing Babel processing for Radarr movie: {video_path}")
            background_tasks.add_task(
                pipeline.process_video_file,
                video_path=video_path,
                event_source="RADARR",
                title=title,
                series_title=title
            )
            return {"status": "queued", "event": event_type, "video_path": video_path, "title": title}

    return {"status": "ignored", "event": event_type}

@router.post("/process")
async def manual_process(req: ManualProcessRequest, background_tasks: BackgroundTasks):
    # Bug #43: Validate that video_path is under configured media directories
    import os, re
    series_path = get_setting("media_series_path", "/tv")
    movies_path = get_setting("media_movies_path", "/movies")
    norm_path = os.path.normpath(req.video_path)
    
    req.video_path = validate_path(req.video_path)

    title = req.title
    if not title:
        base = os.path.basename(req.video_path)
        base = os.path.splitext(base)[0]
        title = re.sub(r'(?i)(WEBDL|WEB-DL|WEB|HDTV|Bluray|720p|1080p|2160p|4K|x264|x265|HDR|AMZN).*', '', base).strip(' -._')

    background_tasks.add_task(
        pipeline.process_video_file,
        video_path=req.video_path,
        wait_seconds=req.wait_seconds,
        event_source="MANUAL",
        title=title,
        force_retranslate=req.force_retranslate
    )
    return {"status": "queued", "video_path": req.video_path, "wait_seconds": req.wait_seconds}
