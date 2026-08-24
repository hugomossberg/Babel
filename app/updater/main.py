import os
import re
import asyncio
import secrets
import logging
import httpx
from fastapi import FastAPI, HTTPException, Request, Header, Depends
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Babel Updater")
logger = logging.getLogger("babel.updater")

DOCKER_SOCKET = "/var/run/docker.sock"
BABEL_CONTAINER_NAME = os.getenv("BABEL_CONTAINER_NAME", "babel")
ALLOWED_IMAGE = os.getenv("ALLOWED_IMAGE", "ghcr.io/hugomossberg/babel")
AUTH_FILE_PATH = "/app/auth/updater_secret"

def bootstrap_updater_secret() -> str:
    env_sec = os.getenv("BABEL_UPDATER_SECRET", os.getenv("UPDATER_SECRET", "")).strip()
    if env_sec:
        return env_sec
    try:
        os.makedirs(os.path.dirname(AUTH_FILE_PATH), exist_ok=True)
        if not os.path.exists(AUTH_FILE_PATH):
            new_sec = secrets.token_hex(32)
            import tempfile
            fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(AUTH_FILE_PATH))
            with os.fdopen(fd, 'w') as f:
                f.write(new_sec)
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, AUTH_FILE_PATH)
        with open(AUTH_FILE_PATH, "r") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to bootstrap updater secret: {e}")
        return ""

UPDATER_SECRET = bootstrap_updater_secret()
UPDATER_STATE = "idle"
UPDATE_TASK: Optional[asyncio.Task] = None

async def call_docker(method: str, path: str, json_data=None, timeout=30.0):
    transport = httpx.AsyncHTTPTransport(uds=DOCKER_SOCKET)
    async with httpx.AsyncClient(transport=transport, base_url="http://docker", timeout=timeout) as client:
        if method == "GET":
            return await client.get(path)
        elif method == "POST":
            return await client.post(path, json=json_data)
        elif method == "DELETE":
            return await client.delete(path)

class UpdateRequest(BaseModel):
    target_version: str

def verify_updater_token(request: Request, x_updater_token: Optional[str] = Header(None)):
    token = x_updater_token or ""
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()

    if not UPDATER_SECRET:
        logger.error("UPDATER_SECRET is not configured on updater service")
        raise HTTPException(status_code=401, detail="Updater authentication not configured")

    if not token or not secrets.compare_digest(token, UPDATER_SECRET):
        raise HTTPException(status_code=401, detail="Invalid or missing updater token")

async def rollback(old_container_name: str, new_container_name: str):
    global UPDATER_STATE
    UPDATER_STATE = "rolling_back"
    logger.warning(f"Initiating rollback: removing {new_container_name}, restoring {old_container_name}")
    try:
        await call_docker("DELETE", f"/containers/{new_container_name}?force=true")
        await call_docker("POST", f"/containers/{old_container_name}/rename?name={new_container_name}")
        await call_docker("POST", f"/containers/{new_container_name}/start")
        UPDATER_STATE = "rolled_back"
        logger.info(f"Rollback to {new_container_name} completed successfully")
    except Exception as e:
        UPDATER_STATE = "failed"
        logger.critical(f"CRITICAL: Rollback failed for {old_container_name} -> {new_container_name}: {e}")

@app.get("/health")
def health():
    return {"status": "updater_healthy"}

@app.get("/status")
def get_status(token_auth: None = Depends(verify_updater_token)):
    return {"status": UPDATER_STATE}

async def run_update(tag: str):
    global UPDATER_STATE
    old_name = f"{BABEL_CONTAINER_NAME}_rollback"
    replaced = False
    try:
        full_image = f"{ALLOWED_IMAGE}:{tag}"
        UPDATER_STATE = "inspecting"
        
        # 1. Inspect old container
        res = await call_docker("GET", f"/containers/{BABEL_CONTAINER_NAME}/json")
        if res.status_code != 200:
            logger.error(f"Failed to find container '{BABEL_CONTAINER_NAME}'")
            UPDATER_STATE = "failed"
            return
        
        old_config = res.json()

        # 2. Pull new image
        UPDATER_STATE = "pulling"
        pull_res = await call_docker("POST", f"/images/create?fromImage={ALLOWED_IMAGE}&tag={tag}", timeout=120.0)
        if pull_res.status_code != 200:
            logger.error(f"Failed to pull image: {pull_res.status_code} {pull_res.text}")
            UPDATER_STATE = "failed"
            return

        config = old_config.get("Config", {})
        host_config = old_config.get("HostConfig", {})
        network_settings = old_config.get("NetworkSettings", {})

        # 3. Retag new image to match the old container's image name
        # This ensures docker-compose continues to work seamlessly without seeing a modified image name.
        old_image_name = config.get("Image", "")
        if old_image_name:
            if ":" in old_image_name:
                repo, original_tag = old_image_name.rsplit(":", 1)
            else:
                repo, original_tag = old_image_name, "latest"

            retag_res = await call_docker("POST", f"/images/{full_image}/tag?repo={repo}&tag={original_tag}")
            if retag_res.status_code != 201:
                logger.warning(f"Failed to retag {full_image} to {old_image_name}, HTTP {retag_res.status_code}")
        else:
            config["Image"] = full_image

        config["HostConfig"] = host_config
        if "EndpointsConfig" in network_settings:
            config["NetworkingConfig"] = {"EndpointsConfig": network_settings["EndpointsConfig"]}

        old_name = f"{BABEL_CONTAINER_NAME}_rollback"

        # Defensive check: if leftover rollback container exists from previous interrupted run, remove it safely
        try:
            inspect_rb = await call_docker("GET", f"/containers/{old_name}/json")
            if inspect_rb.status_code == 200:
                logger.info(f"Removing leftover rollback container {old_name}")
                await call_docker("DELETE", f"/containers/{old_name}?force=true")
        except Exception as e:
            logger.warning(f"Check for leftover rollback container: {e}")

        UPDATER_STATE = "replacing"
        # Stop old
        await call_docker("POST", f"/containers/{BABEL_CONTAINER_NAME}/stop")
        await call_docker("POST", f"/containers/{BABEL_CONTAINER_NAME}/rename?name={old_name}")
        replaced = True

        # Create new
        create_res = await call_docker("POST", f"/containers/create?name={BABEL_CONTAINER_NAME}", json_data=config)
        if create_res.status_code != 201:
            logger.error(f"Failed to create new container: {create_res.text}")
            await rollback(old_name, BABEL_CONTAINER_NAME)
            return

        # Start new
        start_res = await call_docker("POST", f"/containers/{BABEL_CONTAINER_NAME}/start")
        if start_res.status_code not in (204, 200, 304):
            logger.error(f"Failed to start new container: {start_res.status_code}")
            await rollback(old_name, BABEL_CONTAINER_NAME)
            return

        UPDATER_STATE = "verifying"

        # Health watchdog loop
        for _ in range(12):
            await asyncio.sleep(5)
            health_res = await call_docker("GET", f"/containers/{BABEL_CONTAINER_NAME}/json")
            if health_res.status_code != 200:
                continue

            c_info = health_res.json()
            state_info = c_info.get("State", {})
            health_status = state_info.get("Health", {}).get("Status", "unknown")
            container_status = state_info.get("Status", "")

            if health_status == "healthy":
                logger.info("New container verified healthy. Removing rollback container.")
                await call_docker("DELETE", f"/containers/{old_name}?v=false&force=true")
                UPDATER_STATE = "success"
                return
            elif health_status == "unhealthy" or container_status in ["exited", "dead"]:
                logger.error(f"New container unhealthy (health: {health_status}, status: {container_status}). Rolling back.")
                await rollback(old_name, BABEL_CONTAINER_NAME)
                return

        # If timeout waiting for healthy
        logger.error("Health verification timed out waiting for healthy status. Rolling back.")
        await rollback(old_name, BABEL_CONTAINER_NAME)
    except Exception as e:
        logger.critical(f"Exception during update execution: {e}")
        if replaced:
            await rollback(old_name, BABEL_CONTAINER_NAME)
        else:
            UPDATER_STATE = "failed"

@app.post("/update")
async def perform_update(req: UpdateRequest, token_auth: None = Depends(verify_updater_token)):
    global UPDATER_STATE, UPDATE_TASK
    if (UPDATE_TASK is not None and not UPDATE_TASK.done()) or UPDATER_STATE in ["inspecting", "pulling", "replacing", "verifying", "rolling_back", "updating"]:
        raise HTTPException(status_code=409, detail=f"Update already in progress ({UPDATER_STATE})")

    tag = req.target_version.strip()
    if not re.match(r'^v?\d+\.\d+\.\d+(?:-beta)?$', tag):
        raise HTTPException(status_code=400, detail="Invalid target_version format")

    UPDATER_STATE = "inspecting"
    UPDATE_TASK = asyncio.create_task(run_update(tag))

    return {"status": "started", "message": "Update started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8767)
