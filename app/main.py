import logging
import os
import secrets
import base64
import asyncio
from contextlib import asynccontextmanager
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.core.db import recover_stale_queued_jobs
    recover_stale_queued_jobs()
    task = asyncio.create_task(retry_waiting_jobs())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    lifespan=lifespan,

    title=APP_NAME,
    version=VERSION,
    description="Automated ultra-fast AI subtitle extractor, SDH-cleaner and translator for Servarr stack"
)

# Bug #39: Optional Basic Auth Middleware
class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not AUTH_USERNAME or not AUTH_PASSWORD:
            return await call_next(request)
        
        # Don't require auth for specific webhooks or health check
        if request.url.path in ["/webhook/sonarr", "/webhook/radarr"] or request.url.path == "/health":
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
    from app.config import VERSION
    template_path = "/app/app/templates/index.html" if os.path.exists("/app/app/templates/index.html") else "app/templates/index.html"
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()
        html = html.replace("{{VERSION}}", VERSION)
        return HTMLResponse(content=html)



async def process_one_retry_pass():
    from app.core.db import claim_job_for_retry
    try:
        jobs = get_jobs_by_status(["WAITING_PROVIDER", "RETRY_PENDING", "RECOVERING", "PARTIAL"])
        for job in jobs:
            try:
                should_retry = False
                now = datetime.now(timezone.utc)
                if job["status"] in ["RETRY_PENDING", "WAITING_PROVIDER", "RECOVERING", "PARTIAL"]:
                    if job.get("next_retry_at"):
                        next_retry_at = datetime.fromisoformat(job["next_retry_at"])
                        if now >= next_retry_at:
                            should_retry = True
                    else:
                        updated_at = datetime.fromisoformat(job["updated_at"])
                        if (now - updated_at).total_seconds() > 300: # 5 min fallback
                            should_retry = True
                
                if should_retry:
                    if claim_job_for_retry(job["id"]):
                        logging.info(f"Retrying job {job['id']}")
                        # wait for the task to finish if it's running synchronously in a test
                        # but in production, we want to run them concurrently.
                        # For testability without sleep, maybe we can await them or just create task
                        # Let's keep create_task but return the tasks so tests can await them
                        yield asyncio.create_task(pipeline.process_video_file(
                            job["video_path"],
                            event_source="RETRY",
                            title=job["title"],
                            job_id=job["id"]
                        ))
            except Exception as e:
                logging.error(f"Error checking retry for job {job['id']}: {e}")
    except Exception as e:
        logging.error(f"Error in retry loop: {e}")

background_tasks = set()

async def retry_waiting_jobs():
    while True:
        async for task in process_one_retry_pass():
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)
        await asyncio.sleep(10)

@app.get("/health")
async def health():
    return {"status": "healthy"}
