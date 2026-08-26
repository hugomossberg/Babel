import logging
import os
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple
from urllib.parse import quote

import httpx

from app.core.db import get_setting

logger = logging.getLogger("babel.plex")


def _map_to_plex_path(local_path: str) -> str:
    """
    Translates a path as Babel sees it into the path Plex sees.

    Plex frequently runs outside Babel's container (natively on the host, or in a
    container with different mounts), so the same file can have two different
    absolute paths. Configured as comma-separated prefix pairs, mirroring the
    Sonarr/Radarr remote path mapping already used for webhooks.
    """
    babel_prefix = get_setting("plex_path_babel_prefix", "").strip()
    plex_prefix = get_setting("plex_path_plex_prefix", "").strip()
    if not babel_prefix:
        return local_path

    b_prefs = [p.strip() for p in babel_prefix.split(",")]
    p_prefs = [p.strip() for p in plex_prefix.split(",")]
    for i, b_pref in enumerate(b_prefs):
        p_pref = p_prefs[i] if i < len(p_prefs) else ""
        if b_pref and local_path.startswith(b_pref):
            return local_path.replace(b_pref, p_pref, 1)
    return local_path


async def _get_sections(client: httpx.AsyncClient, url: str, token: str) -> List[Tuple[str, List[str]]]:
    """Returns [(section_key, [location_paths])] for every library section."""
    res = await client.get(f"{url}/library/sections", params={"X-Plex-Token": token})
    if res.status_code != 200:
        logger.warning(f"Plex returned status {res.status_code} listing library sections")
        return []
    try:
        root = ET.fromstring(res.text)
    except ET.ParseError as e:
        logger.warning(f"Could not parse Plex library sections response: {e}")
        return []

    sections = []
    for directory in root.findall("Directory"):
        key = directory.get("key")
        if not key:
            continue
        locations = [loc.get("path") for loc in directory.findall("Location") if loc.get("path")]
        sections.append((key, locations))
    return sections


def _match_section(sections: List[Tuple[str, List[str]]], plex_path: str) -> Optional[str]:
    """
    Picks the section whose location contains plex_path, preferring the longest
    (most specific) match so nested libraries resolve to the correct section.
    """
    best_key = None
    best_len = -1
    for key, locations in sections:
        for loc in locations:
            if plex_path.startswith(loc.rstrip("/") + os.sep) and len(loc) > best_len:
                best_key, best_len = key, len(loc)
    return best_key


async def notify_plex_library_refresh(published_path: Optional[str] = None):
    """
    Asks Plex to scan for newly published subtitles.

    Plex only picks up external subtitle files during a library scan, so a
    published subtitle stays invisible until one runs.

    When published_path is supplied, only the section containing that file is
    scanned. Plex scans incrementally, so this is far cheaper than refreshing
    every section on a large library. Falls back to refreshing all sections when
    the path cannot be matched.
    """
    url = get_setting("plex_url", "").rstrip("/")
    token = get_setting("plex_token", "")
    if not url or not token:
        return

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            sections = await _get_sections(client, url, token)
            if not sections:
                logger.warning("Plex reported no library sections; skipping refresh")
                return

            target_keys = [key for key, _ in sections]
            if published_path:
                plex_path = _map_to_plex_path(published_path)
                matched = _match_section(sections, plex_path)
                if matched:
                    target_keys = [matched]
                else:
                    logger.info(
                        f"No Plex section matched {plex_path}; refreshing all sections. "
                        f"If Plex sees this file under a different path, set the Plex path mapping."
                    )

            for key in target_keys:
                res = await client.get(f"{url}/library/sections/{key}/refresh",
                                       params={"X-Plex-Token": token})
                if res.status_code in (200, 201, 204):
                    logger.info(f"Triggered Plex scan for library section {key}")
                else:
                    logger.warning(f"Plex scan for section {key} returned status {res.status_code}")
    except Exception as e:
        logger.warning(f"Could not notify Plex: {e}")
