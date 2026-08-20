import logging
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.core.db import init_db
from app.api.webhooks import router as webhooks_router
from app.api.dashboard import router as dashboard_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

init_db()

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Automated ultra-fast AI subtitle extractor, SDH-cleaner and translator for Servarr stack"
)

static_dir = "/app/app/static" if os.path.exists("/app/app/static") else "app/static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.include_router(webhooks_router)
app.include_router(dashboard_router, prefix="/api")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    template_path = "/app/app/templates/index.html" if os.path.exists("/app/app/templates/index.html") else "app/templates/index.html"
    with open(template_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health():
    return {"status": "healthy"}
