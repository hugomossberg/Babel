import logging
import os
import secrets
import base64
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import app.core.db as db_core
from app.core.db import get_jobs_by_status
from app.core.quota import should_retry_deferred_job
from app.services.pipeline import pipeline
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import APP_NAME, VERSION, AUTH_USERNAME, AUTH_PASSWORD
from app.core.db import init_db
from app.api.webhooks import router as webhooks_router
from app.api.dashboard import router as dashboard_router
from app.api.usage import router as usage_router

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

        if request.url.path == "/health":
            return await call_next(request)

        if request.url.path in ["/webhook/sonarr", "/webhook/radarr"]:
            from app.core.db import get_setting
            webhook_secret = get_setting("webhook_secret", "").strip()
            if webhook_secret:
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
app.include_router(usage_router, prefix="/api")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    from app.config import VERSION
    template_path = "/app/app/templates/index.html" if os.path.exists("/app/app/templates/index.html") else "app/templates/index.html"
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()
        html = html.replace("{{VERSION}}", VERSION)
        return HTMLResponse(
            content=html,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            }
        )



async def process_one_retry_pass():
    from app.services.updates_controller import updates_controller
    if updates_controller.is_locked_for_update():
        return

    try:
        now = datetime.now(timezone.utc)

        # ------------------------------------------------------------------
        # Phase 1: FIFO DEFERRED resume per provider
        # ------------------------------------------------------------------
        # Collect all DEFERRED jobs, group by pinned provider, order FIFO.
        # Oldest eligible job per provider is processed first.
        # Provider backlogs are independent — OpenAI backlog ≠ Gemini backlog.
        # ------------------------------------------------------------------
        deferred_jobs = get_jobs_by_status(["DEFERRED"])

        # Group by resolved provider, maintain FIFO order
        from collections import defaultdict
        provider_queues: dict = defaultdict(list)
        for job in sorted(
            deferred_jobs,
            key=lambda j: (j.get("deferred_at") or j.get("created_at") or "", j.get("id") or 0)
        ):
            if should_retry_deferred_job(job, now):
                p = (
                    job.get("waiting_provider")
                    or job.get("primary_provider")
                    or "unknown"
                ).strip().lower()
                provider_queues[p].append(job)

        # Yield one eligible job per provider, oldest first
        for _provider, eligible in provider_queues.items():
            for job in eligible:
                if db_core.claim_fifo_job_for_retry(job["id"]):
                    logging.info(
                        f"FIFO resume: job {job['id']} (provider={_provider}, "
                        f"defer_reason={job.get('defer_reason')}, "
                        f"stage={job.get('defer_stage')}, was DEFERRED)"
                    )
                    yield asyncio.create_task(pipeline.process_video_file(
                        job.get("video_path", ""),
                        event_source="RETRY",
                        title=job.get("title", ""),
                        job_id=job["id"],
                        force_retranslate=bool(job.get("force_retranslate")),
                    ))
                    break  # One claim per provider per pass — respect FIFO

        # ------------------------------------------------------------------
        # Phase 2: Non-DEFERRED retry states (RETRY_PENDING, RECOVERING, etc.)
        # ------------------------------------------------------------------
        other_jobs = get_jobs_by_status([
            "WAITING_PROVIDER", "RETRY_PENDING", "RECOVERING", "PARTIAL", "WAITING_SOURCE"
        ])
        for job in other_jobs:
            try:
                should_retry = False
                if job.get("next_retry_at"):
                    next_retry_at = datetime.fromisoformat(job["next_retry_at"])
                    if next_retry_at.tzinfo is None:
                        next_retry_at = next_retry_at.replace(tzinfo=timezone.utc)
                    if now >= next_retry_at:
                        should_retry = True
                else:
                    updated_at = datetime.fromisoformat(job["updated_at"])
                    if updated_at.tzinfo is None:
                        updated_at = updated_at.replace(tzinfo=timezone.utc)
                    if (now - updated_at).total_seconds() > 300:  # 5 min fallback
                        should_retry = True

                if should_retry:
                    if db_core.claim_job_for_retry(job["id"]):
                        logging.info(f"Retrying job {job['id']} (was {job.get('status')})")
                        yield asyncio.create_task(pipeline.process_video_file(
                            job.get("video_path", ""),
                            event_source="RETRY",
                            title=job.get("title", ""),
                            job_id=job["id"],
                            force_retranslate=bool(job.get("force_retranslate")),
                        ))
            except Exception as e:
                logging.error(f"Error checking retry for job {job.get('id')}: {e}")
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
