import re
import httpx
import logging
from typing import Dict, Any
from datetime import datetime, timezone

from app.config import VERSION

logger = logging.getLogger("babel.updates")

class UpdatesController:
    def __init__(self):
        self.github_repo = "hugomossberg/Babel"
        self.cached_info = None
        self.cache_time = None
        self.update_status = "idle" # idle, updating
        
    def _parse_version(self, version_str: str) -> tuple:
        """Parses vX.Y.Z-beta into a comparable tuple: (X, Y, Z, is_beta, beta_num) or similar for basic semantic compare."""
        # Simple regex for vX.Y.Z or vX.Y.Z-beta
        match = re.match(r'^v?(\d+)\.(\d+)\.(\d+)(?:-beta)?$', version_str)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        return (0, 0, 0)
        
    async def is_updater_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                res = await client.get("http://babel-updater:8767/health")
                return res.status_code == 200 and res.json().get("status") == "updater_healthy"
        except Exception:
            return False

    async def get_update_info(self, force_refresh: bool = False) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).timestamp()
        
        # Use cache if within 2 hours
        if not force_refresh and self.cached_info and self.cache_time and (now - self.cache_time) < 7200:
            return self.cached_info

        is_beta_channel = "-beta" in VERSION
        
        # Default response
        info = {
            "current_version": VERSION,
            "latest_version": VERSION,
            "update_available": False,
            "release_url": "",
            "release_notes": "",
            "published_at": None,
            "one_click_update_available": await self.is_updater_available(),
            "updater_status": self.update_status,
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(
                    f"https://api.github.com/repos/{self.github_repo}/releases",
                    headers={"Accept": "application/vnd.github.v3+json", "User-Agent": f"Babel/{VERSION}"}
                )
                
                if res.status_code == 200:
                    releases = res.json()
                    
                    latest_release = None
                    for rel in releases:
                        tag_name = rel.get("tag_name", "")
                        is_rel_beta = "-beta" in tag_name
                        
                        # Match channel
                        if is_beta_channel and not is_rel_beta:
                            continue
                        if not is_beta_channel and is_rel_beta:
                            continue
                            
                        # Compare logic
                        current_tuple = self._parse_version(VERSION)
                        rel_tuple = self._parse_version(tag_name)
                        
                        if rel_tuple > current_tuple:
                            if latest_release is None or rel_tuple > self._parse_version(latest_release.get("tag_name", "")):
                                latest_release = rel
                                
                    if latest_release:
                        info["update_available"] = True
                        info["latest_version"] = latest_release.get("tag_name")
                        info["release_url"] = latest_release.get("html_url")
                        info["published_at"] = latest_release.get("published_at")
                        
                        body = latest_release.get("body", "")
                        if len(body) > 1000:
                            body = body[:1000] + "...\n\n[View full release on GitHub]"
                        info["release_notes"] = body
                        
        except Exception as e:
            logger.warning(f"Failed to check for updates: {e}")
            
        self.cached_info = info
        self.cache_time = now
        return info

    async def trigger_update(self, target_version: str):
        if self.update_status == "updating":
            return {"success": False, "message": "Update already in progress"}
            
        info = await self.get_update_info()
        if not info["update_available"] or target_version != info["latest_version"]:
            return {"success": False, "message": "Target version is not valid"}
            
        from app.core.db import get_jobs_by_status
        active_jobs = get_jobs_by_status(["EXTRACTING", "TRANSLATING", "QUEUED", "WAITING_PROVIDER", "RECOVERING"])
        if len(active_jobs) > 0:
            return {"success": False, "message": "Active jobs are running. Update blocked."}

        self.update_status = "updating"
        
        try:
            image_target = f"ghcr.io/{self.github_repo}:{target_version}"
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(
                    "http://babel-updater:8767/update",
                    json={"image": image_target}
                )
                if res.status_code == 200:
                    return {"success": True, "message": "Update initiated. System will restart shortly."}
                else:
                    self.update_status = "idle"
                    return {"success": False, "message": f"Updater error: {res.text}"}
        except httpx.ConnectError:
            self.update_status = "idle"
            return {"success": False, "message": "Updater service unreachable. Is babel-updater running?"}
        except httpx.TimeoutException:
            # We timeout because it takes time to pull maybe?
            # Wait, the updater returns immediately with async health watchdog
            self.update_status = "idle"
            return {"success": False, "message": "Updater timed out."}
        except Exception as e:
            self.update_status = "idle"
            return {"success": False, "message": str(e)}

updates_controller = UpdatesController()
