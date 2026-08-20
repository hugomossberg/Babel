import httpx
import logging
from app.config import settings

logger = logging.getLogger("babel.jellyfin")

async def notify_jellyfin_library_refresh():
    """
    Sends a library refresh command to Jellyfin so new subtitles appear immediately in the UI.
    """
    url = f"{settings.jellyfin_url}/Library/Refresh"
    headers = {
        "X-Emby-Token": settings.jellyfin_api_key,
        "Content-Type": "application/json"
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(url, headers=headers)
            if res.status_code in [200, 204]:
                logger.info("Successfully notified Jellyfin of new subtitle (Library Refresh triggered)")
            else:
                logger.warning(f"Jellyfin refresh returned status code {res.status_code}")
    except Exception as e:
        logger.warning(f"Could not notify Jellyfin: {e}")
