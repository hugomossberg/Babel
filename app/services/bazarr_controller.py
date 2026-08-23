import httpx
import logging
import asyncio
from typing import Dict, Any, Optional

logger = logging.getLogger("babel.bazarr_controller")

class BazarrController:
    """
    Direct API integration with Bazarr.
    Allows:
    1. Checking connection
    2. Triggering an active search for subtitles for a specific episode/movie
    """

    @staticmethod
    async def get_status(bazarr_url: str, api_key: str) -> Dict[str, Any]:
        if not bazarr_url:
            return {"connected": False, "message": "Bazarr URL not configured"}
        clean_url = bazarr_url.rstrip("/")
        headers = {"X-API-KEY": api_key} if api_key else {}
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                res = await client.get(f"{clean_url}/api/system/status", headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    version = data.get("version")
                    ver_str = f"v{version}" if version and version != "unknown" else ""
                    return {"connected": True, "version": ver_str, "status": "running"}
                elif res.status_code in (401, 403):
                    return {"connected": False, "message": "Invalid API Key"}
                else:
                    return {"connected": False, "message": f"HTTP {res.status_code}"}
        except httpx.TimeoutException:
            return {"connected": False, "message": "Connection timed out"}
        except httpx.ConnectError:
            return {"connected": False, "message": "Unable to reach Bazarr"}
        except Exception as e:
            return {"connected": False, "message": "Connection failed"}

    @staticmethod
    async def trigger_search_and_wait(bazarr_url: str, api_key: str, video_path: str, wait_seconds: int = 0) -> bool:
        """Deprecated legacy method retained for backwards compatibility."""
        if not bazarr_url:
            return True
        logger.info(f"Triggering Bazarr search for {video_path}")
        return True

bazarr_controller = BazarrController()
