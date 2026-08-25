import logging
import os
import secrets
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Request, HTTPException
from pydantic import BaseModel

from app.core import db
from app.services.pipeline import pipeline
from app.services.updates_controller import updates_controller

logger = logging.getLogger("babel.webhooks")
router = APIRouter(prefix="/webhook", tags=["webhooks"])

def validate_webhook_token(request: Request):
    from app.core.db import get_setting
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

class ManualProcessRequest(BaseModel):
    video_path: str
    wait_seconds: Optional[int] = None
    title: Optional[str] = None
    force_retranslate: Optional[bool] = False

def translate_path(remote_path: str) -> str:
    if not remote_path:
        return remote_path

    remote_prefix = db.get_setting("remote_path_prefix", "").strip()
    local_prefix = db.get_setting("local_path_prefix", "").strip()
    
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
    if updates_controller.is_locked_for_update():
        raise HTTPException(status_code=503, detail="System is in update maintenance mode, jobs are locked.")
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
            try:
                video_path = validate_path(video_path)
            except HTTPException as e:
                logger.error(f"Webhook path validation failed: {e.detail}")
                return {"status": "error", "reason": e.detail}

            try:
                result = db.create_job_if_no_active(
                    video_path=video_path,
                    event_source="SONARR",
                    title=title,
                    force_retranslate=False,
                )
            except Exception as e:
                logger.error(f"Sonarr webhook: dedupe DB error for {video_path}: {e}")
                raise HTTPException(
                    status_code=503,
                    detail="Dedupe check temporarily unavailable. Please retry."
                )
            job_id = result["job_id"]
            if not result["created"]:
                logger.info(
                    f"Sonarr webhook: active job {job_id} already exists for {video_path} "
                    f"(status={result['existing_job']['status']}) — not creating duplicate"
                )
                return {
                    "status": "already_active",
                    "event": event_type,
                    "video_path": video_path,
                    "job_id": job_id,
                    "existing_status": result["existing_job"]["status"],
                }
            logger.info(f"Queueing Babel processing for Sonarr episode: {video_path} (Job ID: {job_id})")
            background_tasks.add_task(
                pipeline.process_video_file,
                video_path=video_path,
                event_source="SONARR",
                title=title,
                series_title=series_title,
                job_id=job_id
            )
            return {"status": "queued", "event": event_type, "video_path": video_path, "title": title, "job_id": job_id}


    return {"status": "ignored", "event": event_type}

@router.post("/radarr")
async def radarr_webhook(request: Request, background_tasks: BackgroundTasks):
    validate_webhook_token(request)
    if updates_controller.is_locked_for_update():
        raise HTTPException(status_code=503, detail="System is in update maintenance mode, jobs are locked.")
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
            try:
                video_path = validate_path(video_path)
            except HTTPException as e:
                logger.error(f"Webhook path validation failed: {e.detail}")
                return {"status": "error", "reason": e.detail}

            try:
                result = db.create_job_if_no_active(
                    video_path=video_path,
                    event_source="RADARR",
                    title=title,
                    force_retranslate=False,
                )
            except Exception as e:
                logger.error(f"Radarr webhook: dedupe DB error for {video_path}: {e}")
                raise HTTPException(
                    status_code=503,
                    detail="Dedupe check temporarily unavailable. Please retry."
                )
            job_id = result["job_id"]
            if not result["created"]:
                logger.info(
                    f"Radarr webhook: active job {job_id} already exists for {video_path} "
                    f"(status={result['existing_job']['status']}) — not creating duplicate"
                )
                return {
                    "status": "already_active",
                    "event": event_type,
                    "video_path": video_path,
                    "job_id": job_id,
                    "existing_status": result["existing_job"]["status"],
                }
            logger.info(f"Queueing Babel processing for Radarr movie: {video_path} (Job ID: {job_id})")
            background_tasks.add_task(
                pipeline.process_video_file,
                video_path=video_path,
                event_source="RADARR",
                title=title,
                series_title=title,
                job_id=job_id
            )
            return {"status": "queued", "event": event_type, "video_path": video_path, "title": title, "job_id": job_id}


    return {"status": "ignored", "event": event_type}

@router.post("/process")
async def manual_process(req: ManualProcessRequest, background_tasks: BackgroundTasks):
    if updates_controller.is_locked_for_update():
        raise HTTPException(status_code=503, detail="System is in update maintenance mode, jobs are locked.")
    
    import re
    req.video_path = validate_path(req.video_path)

    title = req.title
    if not title:
        base = os.path.basename(req.video_path)
        base = os.path.splitext(base)[0]
        title = re.sub(r'(?i)(WEBDL|WEB-DL|WEB|HDTV|Bluray|720p|1080p|2160p|4K|x264|x265|HDR|AMZN).*', '', base).strip(' -._')

    try:
        result = db.create_job_if_no_active(
            video_path=req.video_path,
            event_source="MANUAL",
            title=title,
            force_retranslate=req.force_retranslate,
        )
    except Exception as e:
        logger.error(f"Manual process: dedupe DB error for {req.video_path}: {e}")
        raise HTTPException(
            status_code=503,
            detail="Dedupe check temporarily unavailable. Please retry."
        )
    job_id = result["job_id"]

    if not result["created"]:
        existing = result["existing_job"]
        logger.info(
            f"Manual process: active job {job_id} already exists for {req.video_path} "
            f"(status={existing['status']}) — returning existing job"
        )
        return {
            "status": "already_active",
            "job_id": job_id,
            "video_path": req.video_path,
            "existing_status": existing["status"],
            "defer_reason": existing.get("defer_reason"),
        }

    logger.info(f"Queueing manual Babel processing: {req.video_path} (Job ID: {job_id})")

    background_tasks.add_task(
        pipeline.process_video_file,
        video_path=req.video_path,
        wait_seconds=req.wait_seconds,
        event_source="MANUAL",
        title=title,
        force_retranslate=req.force_retranslate,
        job_id=job_id
    )
    return {"status": "queued", "job_id": job_id, "video_path": req.video_path, "wait_seconds": req.wait_seconds}
