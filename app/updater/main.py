import os
import asyncio
import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

app = FastAPI()

DOCKER_SOCKET = "/var/run/docker.sock"
BABEL_CONTAINER_NAME = os.getenv("BABEL_CONTAINER_NAME", "babel")
ALLOWED_IMAGE = os.getenv("ALLOWED_IMAGE", "ghcr.io/hugomossberg/babel")

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
    image: str

async def rollback(old_container_name: str, new_container_name: str):
    await call_docker("DELETE", f"/containers/{new_container_name}?force=true")
    await call_docker("POST", f"/containers/{old_container_name}/rename?name={new_container_name}")
    await call_docker("POST", f"/containers/{new_container_name}/start")

@app.post("/update")
async def perform_update(req: UpdateRequest):
    if not req.image.startswith(ALLOWED_IMAGE):
        raise HTTPException(status_code=403, detail="Image not allowed")
    
    # 1. Inspect old container
    res = await call_docker("GET", f"/containers/{BABEL_CONTAINER_NAME}/json")
    if res.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Failed to find container '{BABEL_CONTAINER_NAME}'")
    
    old_config = res.json()
    
    # 2. Pull new image
    parts = req.image.split(":")
    tag = parts[1] if len(parts) > 1 else "latest"
    # Provide a long timeout for the image pull. 
    pull_res = await call_docker("POST", f"/images/create?fromImage={parts[0]}&tag={tag}", timeout=120.0)
    if pull_res.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to pull image")
        
    config = old_config.get("Config", {})
    host_config = old_config.get("HostConfig", {})
    network_settings = old_config.get("NetworkSettings", {})
    
    # Keep identical mounts, envs, ports.
    config["Image"] = req.image
    config["HostConfig"] = host_config
    if "EndpointsConfig" in network_settings:
        config["NetworkingConfig"] = {"EndpointsConfig": network_settings["EndpointsConfig"]}
        
    old_name = f"{BABEL_CONTAINER_NAME}_rollback"

    # Stop old
    await call_docker("POST", f"/containers/{BABEL_CONTAINER_NAME}/stop")
    await call_docker("POST", f"/containers/{BABEL_CONTAINER_NAME}/rename?name={old_name}")
    
    # Create new
    create_res = await call_docker("POST", f"/containers/create?name={BABEL_CONTAINER_NAME}", json_data=config)
    if create_res.status_code != 201:
        await rollback(old_name, BABEL_CONTAINER_NAME)
        raise HTTPException(status_code=500, detail=f"Failed to create new container: {create_res.text}")
        
    # Start new
    start_res = await call_docker("POST", f"/containers/{BABEL_CONTAINER_NAME}/start")
    if start_res.status_code not in (204, 200, 304):
        await rollback(old_name, BABEL_CONTAINER_NAME)
        raise HTTPException(status_code=500, detail="Failed to start new container")
        
    # Because making the HTTP request block on Docker healthchecks can cause timeouts on the client,
    # we can start a background task or just wait here up to 20 seconds.
    # The healthcheck usually takes 10s based on the compose settings.
    async def verify_health():
        for _ in range(8):
            await asyncio.sleep(5)
            health_res = await call_docker("GET", f"/containers/{BABEL_CONTAINER_NAME}/json")
            if health_res.status_code != 200:
                continue
            
            c_info = health_res.json()
            health_status = c_info.get("State", {}).get("Health", {}).get("Status", "")
            
            if health_status == "healthy":
                await call_docker("DELETE", f"/containers/{old_name}?v=false&force=true")
                return
            elif health_status == "unhealthy" or c_info.get("State", {}).get("Status") == "exited":
                await rollback(old_name, BABEL_CONTAINER_NAME)
                return

    # Fire and forget health watchdog
    asyncio.create_task(verify_health())

    return {"status": "started", "message": "Update triggered. Awaiting health verification."}

@app.get("/health")
def health():
    return {"status": "updater_healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8767)
