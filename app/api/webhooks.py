import logging
from fastapi import APIRouter, BackgroundTasks, Request
from pydantic import BaseModel
from typing import Optional

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
    remote_prefix = get_setting("remote_path_prefix", "").strip()
    local_prefix = get_setting("local_path_prefix", "").strip()
    if remote_prefix and remote_path.startswith(remote_prefix):
        return remote_path.replace(remote_prefix, local_prefix, 1)
    return remote_path

@router.post("/sonarr")
async def sonarr_webhook(request: Request, background_tasks: BackgroundTasks):
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
        
        video_path = episode_file.get("path")
        if not video_path and series:
            series_path = series.get("path", "")
            rel_path = episode_file.get("relativePath", "")
            if series_path and rel_path:
                video_path = f"{series_path}/{rel_path}"

        title = None
        if series and episodes:
            ep_title = episodes[0].get("title", "")
            s_num = episodes[0].get("seasonNumber", 0)
            e_num = episodes[0].get("episodeNumber", 0)
            title = f"{series.get('title', '')} - S{s_num:02d}E{e_num:02d} - {ep_title}"

        if video_path:
            video_path = translate_path(video_path)
            logger.info(f"Queueing Babel processing for Sonarr episode: {video_path}")
            background_tasks.add_task(
                pipeline.process_video_file,
                video_path=video_path,
                event_source="SONARR",
                title=title
            )
            return {"status": "queued", "event": event_type, "video_path": video_path, "title": title}

    return {"status": "ignored", "event": event_type}

@router.post("/radarr")
async def radarr_webhook(request: Request, background_tasks: BackgroundTasks):
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
        video_path = movie_file.get("path") or movie.get("path")
        
        title = None
        if movie:
            title = f"{movie.get('title', '')} ({movie.get('year', '')})"

        if video_path:
            video_path = translate_path(video_path)
            logger.info(f"Queueing Babel processing for Radarr movie: {video_path}")
            background_tasks.add_task(
                pipeline.process_video_file,
                video_path=video_path,
                event_source="RADARR",
                title=title
            )
            return {"status": "queued", "event": event_type, "video_path": video_path, "title": title}

    return {"status": "ignored", "event": event_type}

@router.post("/process")
async def manual_process(req: ManualProcessRequest, background_tasks: BackgroundTasks):
    title = req.title
    if not title:
        import os, re
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
