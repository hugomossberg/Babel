import logging
import os
import secrets
import base64
import asyncio
from datetime import datetime, timezone
from app.core.db import get_jobs_by_status
from app.services.pipeline import pipeline
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import APP_NAME, VERSION, AUTH_USERNAME, AUTH_PASSWORD
from app.core.db import init_db
from app.api.webhooks import router as webhooks_router
from app.api.dashboard import router as dashboard_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

init_db()

app = FastAPI(
    title=APP_NAME,
    version=VERSION,
    description="Automated ultra-fast AI subtitle extractor, SDH-cleaner and translator for Servarr stack"
)

# Bug #39: Optional Basic Auth Middleware
class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not AUTH_USERNAME or not AUTH_PASSWORD:
            return await call_next(request)
        
        # Don't require auth for webhooks (Sonarr/Radarr need access)
        if request.url.path.startswith("/webhook/sonarr") or request.url.path.startswith("/webhook/radarr"):
            return await call_next(request)
            
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Basic "):
            return Response(status_code=401, headers={"WWW-Authenticate": "Basic"})
            
        try:
            encoded_credentials = auth_header.split(" ")[1]
            decoded_credentials = base64.b64decode(encoded_credentials).decode("utf-8")
            username, password = decoded_credentials.split(":", 1)
            
            if not (secrets.compare_digest(username, AUTH_USERNAME) and 
                    secrets.compare_digest(password, AUTH_PASSWORD)):
                return Response(status_code=401, headers={"WWW-Authenticate": "Basic"})
        except Exception:
            return Response(status_code=401, headers={"WWW-Authenticate": "Basic"})
            
        return await call_next(request)

app.add_middleware(BasicAuthMiddleware)

static_dir = "/app/app/static" if os.path.exists("/app/app/static") else "app/static"
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.include_router(webhooks_router)
app.include_router(dashboard_router, prefix="/api")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    template_path = "/app/app/templates/index.html" if os.path.exists("/app/app/templates/index.html") else "app/templates/index.html"
    with open(template_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(retry_waiting_jobs())

async def retry_waiting_jobs():
    while True:
        try:
            jobs = get_jobs_by_status(["WAITING_PROVIDER"])
            for job in jobs:
                try:
                    updated_at = datetime.fromisoformat(job["updated_at"])
                    now = datetime.now(timezone.utc)
                    if (now - updated_at).total_seconds() > 900: # 15 minutes
                        logging.info(f"Retrying WAITING_PROVIDER job {job['id']}")
                        asyncio.create_task(pipeline.process_video_file(
                            job["video_path"],
                            event_source="RETRY",
                            title=job["title"]
                        ))
                except Exception as e:
                    logging.error(f"Error checking retry for job {job['id']}: {e}")
        except Exception as e:
            logging.error(f"Error in retry loop: {e}")
        await asyncio.sleep(60)

@app.get("/health")
async def health():
    return {"status": "healthy"}
