from typing import Dict, Any
import httpx
import logging
from app.core.db import get_setting

logger = logging.getLogger("babel.jellyfin")

async def check_jellyfin_connection(url: str, api_key: str) -> Dict[str, Any]:
    """
    Tests connection to Jellyfin using a lightweight authenticated read-only API call (GET /System/Info).
    Verifies authentication and API reachability without exposing secrets.
    """
    if not url:
        return {"connected": False, "message": "Jellyfin Server URL not configured"}
    clean_url = url.rstrip("/")
    if not api_key:
        return {"connected": False, "message": "No Jellyfin API Token provided or saved"}

    endpoint = f"{clean_url}/System/Info"
    headers = {
        "X-Emby-Token": api_key,
        "Accept": "application/json"
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.get(endpoint, headers=headers)
            if res.status_code == 200:
                try:
                    data = res.json()
                    version = data.get("Version", "")
                    ver_str = f"v{version}" if version else ""
                    server_name = data.get("ServerName", "")
                    return {"connected": True, "version": ver_str, "server_name": server_name, "status": "ok"}
                except Exception:
                    return {"connected": True, "version": "", "status": "ok"}
            elif res.status_code in (401, 403):
                return {"connected": False, "message": "Invalid Jellyfin API Token (Unauthorized)"}
            elif res.status_code == 404:
                return {"connected": False, "message": "Jellyfin API endpoint not found (HTTP 404)"}
            else:
                return {"connected": False, "message": f"Jellyfin returned HTTP {res.status_code}"}
    except httpx.TimeoutException:
        return {"connected": False, "message": "Connection timed out"}
    except httpx.ConnectError:
        return {"connected": False, "message": "Unable to reach Jellyfin server"}
    except Exception as e:
        logger.warning(f"Jellyfin connection test error: {e}")
        return {"connected": False, "message": "Connection failed"}

async def notify_jellyfin_library_refresh():
    """
    Sends a library refresh command to Jellyfin so new subtitles appear immediately in the UI.
    """
    if get_setting("notify_jellyfin", "false").lower() != "true":
        return
    url = get_setting("jellyfin_url", "").rstrip("/")
    api_key = get_setting("jellyfin_api_key", "")
    if not url or not api_key:
        return
        
    endpoint = f"{url}/Library/Refresh"
    headers = {
        "X-Emby-Token": api_key,
        "Content-Type": "application/json"
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(endpoint, headers=headers)
            if res.status_code in [200, 204]:
                logger.info("Successfully notified Jellyfin of new subtitle (Library Refresh triggered)")
            else:
                logger.warning(f"Jellyfin refresh returned status code {res.status_code}")
    except Exception as e:
        logger.warning(f"Could not notify Jellyfin: {e}")
