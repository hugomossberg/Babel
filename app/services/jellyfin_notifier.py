import httpx
import logging
from app.core.db import get_setting

logger = logging.getLogger("babel.jellyfin")

async def notify_jellyfin_library_refresh():
    """
    Sends a library refresh command to Jellyfin so new subtitles appear immediately in the UI.
    """
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
