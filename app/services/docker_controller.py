import os
import httpx
import logging
from typing import Dict, Any

logger = logging.getLogger("babel.docker")

class DockerController:
    """
    Interacts with the Docker socket (/var/run/docker.sock) via Unix domain socket HTTP.
    Allows managing containers (like Bazarr) directly and universally.
    """
    def __init__(self, socket_path: str = "/var/run/docker.sock"):
        self.socket_path = socket_path

    def is_available(self) -> bool:
        return os.path.exists(self.socket_path)

    async def get_container_status(self, container_name_or_keyword: str = "bazarr") -> Dict[str, Any]:
        if not self.is_available():
            return {"available": False, "status": "unknown", "message": "Docker socket not mounted"}

        try:
            transport = httpx.AsyncHTTPTransport(uds=self.socket_path)
            async with httpx.AsyncClient(transport=transport, base_url="http://docker") as client:
                res = await client.get("/containers/json?all=true")
                if res.status_code == 200:
                    containers = res.json()
                    for c in containers:
                        names = c.get("Names", [])
                        if any(container_name_or_keyword in n for n in names):
                            state = c.get("State", "unknown")
                            matched_name = names[0].lstrip("/")
                            cid = c.get("Id", "")[:12]
                            return {
                                "available": True,
                                "name": matched_name,
                                "id": cid,
                                "state": state,
                                "running": state == "running"
                            }
                    return {"available": True, "status": "not_found", "running": False, "message": f"Container matching '{container_name_or_keyword}' not found"}
        except Exception as e:
            logger.error(f"Docker socket error: {e}")
            return {"available": False, "running": False, "message": str(e)}

    async def toggle_container(self, container_name_or_keyword: str = "bazarr", action: str = "stop") -> Dict[str, Any]:
        if not self.is_available():
            return {"success": False, "message": "Docker socket not available"}

        info = await self.get_container_status(container_name_or_keyword)
        cid = info.get("id") or info.get("name")
        if not cid:
            return {"success": False, "message": f"Container '{container_name_or_keyword}' not found"}

        try:
            transport = httpx.AsyncHTTPTransport(uds=self.socket_path)
            async with httpx.AsyncClient(transport=transport, base_url="http://docker") as client:
                if action == "stop":
                    res = await client.post(f"/containers/{cid}/stop")
                elif action == "start":
                    res = await client.post(f"/containers/{cid}/start")
                elif action == "restart":
                    res = await client.post(f"/containers/{cid}/restart")
                else:
                    return {"success": False, "message": f"Invalid action: {action}"}

                if res.status_code in (204, 304, 200):
                    return {"success": True, "action": action, "container": info.get("name")}
                else:
                    return {"success": False, "message": f"Docker HTTP {res.status_code}: {res.text}"}
        except Exception as e:
            return {"success": False, "message": str(e)}

docker_controller = DockerController()
