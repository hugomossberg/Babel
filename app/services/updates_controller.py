import re
import asyncio
import httpx
import logging
from typing import Dict, Any
from datetime import datetime, timezone

from app.config import VERSION, GITHUB_REPO, UPDATER_URL
import os

logger = logging.getLogger("babel.updates")

RELEASE_CACHE_TTL = 900  # 15 minutes cache for GitHub releases

class UpdatesController:
    def __init__(self):
        self.github_repo = GITHUB_REPO
        self.cached_release_metadata = None
        self.cache_time = None
        self.last_check_ok = True
        self.last_check_error = None
        self.next_allowed_github_check = 0.0
        self.rate_limited = False
        self.update_status = "idle"  # idle, updating, success, failed, rolled_back
        self.is_maintenance_locked = False
        self._check_lock = asyncio.Lock()
        self._update_lock = asyncio.Lock()
        self.AUTH_FILE_PATH = "/app/auth/updater_secret"

    def _get_updater_secret(self) -> str:
        env_sec = os.getenv("BABEL_UPDATER_SECRET", os.getenv("UPDATER_SECRET", "")).strip()
        if env_sec:
            return env_sec
        try:
            with open(self.AUTH_FILE_PATH, "r") as f:
                return f.read().strip()
        except Exception:
            return ""

    def is_locked_for_update(self) -> bool:
        """Returns True if the system is currently locked for an in-progress update or update preparation."""
        return self.is_maintenance_locked or self.update_status in ["updating", "inspecting", "pulling", "replacing", "verifying", "rolling_back"]

    def _parse_version(self, version_str: str) -> tuple:
        """Parses vX.Y.Z-beta or X.Y.Z into a comparable tuple: (X, Y, Z)."""
        if not version_str:
            return (0, 0, 0)
        match = re.match(r'^v?(\d+)\.(\d+)\.(\d+)(?:-.*)?$', version_str.strip())
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        return (0, 0, 0)

    async def get_real_updater_status(self) -> tuple[bool, str]:
        updater_real_status = "idle"
        updater_available = False
        headers = {}
        if secret := self._get_updater_secret():
            headers["X-Updater-Token"] = secret

        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(f"{UPDATER_URL}/health")
                if res.status_code == 200 and res.json().get("status") == "updater_healthy":
                    updater_available = True

                status_res = await client.get(f"{UPDATER_URL}/status", headers=headers)
                if status_res.status_code == 200:
                    updater_real_status = status_res.json().get("status", "idle")
                    if updater_real_status in ["inspecting", "pulling", "replacing", "verifying", "rolling_back", "updating"]:
                        self.update_status = "updating"
                        self.is_maintenance_locked = True
                    elif updater_real_status in ["success", "failed", "rolled_back", "idle"]:
                        self.update_status = updater_real_status
                        self.is_maintenance_locked = False
                else:
                    updater_available = False
        except Exception:
            pass

        return updater_available, updater_real_status

    async def is_updater_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                res = await client.get(f"{UPDATER_URL}/health")
                return res.status_code == 200 and res.json().get("status") == "updater_healthy"
        except Exception:
            return False

    async def get_runtime_status(self) -> Dict[str, Any]:
        avail, st = await self.get_real_updater_status()
        return {
            "current_version": VERSION,
            "updater_status": st,
            "one_click_update_available": avail,
        }

    async def get_update_info(self, force_refresh: bool = False) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).timestamp()

        # 1. ALWAYS query live runtime updater status (never served from stale cache)
        avail, st = await self.get_real_updater_status()

        # 2. Rate limit cooldown check: force_refresh MUST NOT bypass cooldown
        is_cooldown_active = now < self.next_allowed_github_check

        # Check if GitHub release metadata needs refresh (TTL: 900s / 15m)
        cache_valid = (
            not force_refresh
            and self.cached_release_metadata is not None
            and self.cache_time is not None
            and (now - self.cache_time) < RELEASE_CACHE_TTL
            and self.last_check_ok
        )

        if not cache_valid and not is_cooldown_active:
            async with self._check_lock:
                now = datetime.now(timezone.utc).timestamp()
                is_cooldown_active = now < self.next_allowed_github_check
                cache_valid = (
                    not force_refresh
                    and self.cached_release_metadata is not None
                    and self.cache_time is not None
                    and (now - self.cache_time) < RELEASE_CACHE_TTL
                    and self.last_check_ok
                )

                if not cache_valid and not is_cooldown_active:
                    is_beta_channel = "-beta" in VERSION

                    try:
                        async with httpx.AsyncClient(timeout=5.0) as client:
                            res = await client.get(
                                f"https://api.github.com/repos/{self.github_repo}/releases",
                                headers={"Accept": "application/vnd.github.v3+json", "User-Agent": f"Babel/{VERSION}"}
                            )

                            if res.status_code == 200:
                                releases = res.json()
                                latest_release = None
                                if isinstance(releases, list):
                                    for rel in releases:
                                        if not isinstance(rel, dict):
                                            continue
                                        tag_name = rel.get("tag_name", "")
                                        is_rel_beta = "-beta" in tag_name

                                        # Match channel
                                        if is_beta_channel and not is_rel_beta:
                                            continue
                                        if not is_beta_channel and is_rel_beta:
                                            continue

                                        # Compare versions
                                        current_tuple = self._parse_version(VERSION)
                                        rel_tuple = self._parse_version(tag_name)

                                        if rel_tuple > current_tuple:
                                            if latest_release is None or rel_tuple > self._parse_version(latest_release.get("tag_name", "")):
                                                latest_release = rel

                                    release_meta = {}
                                    if latest_release:
                                        release_meta["latest_version"] = latest_release.get("tag_name", "")
                                        release_meta["release_url"] = latest_release.get("html_url", "")
                                        release_meta["published_at"] = latest_release.get("published_at")

                                        body = latest_release.get("body", "") or ""
                                        if len(body) > 1000:
                                            body = body[:1000] + "...\n\n[View full release on GitHub]"
                                        release_meta["release_notes"] = body
                                    else:
                                        release_meta["latest_version"] = VERSION
                                        release_meta["release_url"] = ""
                                        release_meta["release_notes"] = ""
                                        release_meta["published_at"] = None

                                    self.cached_release_metadata = release_meta
                                    self.cache_time = now
                                    self.last_check_ok = True
                                    self.last_check_error = None
                                    self.next_allowed_github_check = 0.0
                                    self.rate_limited = False
                                else:
                                    logger.warning(f"Unexpected GitHub releases response format: {releases}")
                                    self.last_check_ok = False
                                    self.last_check_error = "Unexpected response format from GitHub"
                            elif res.status_code == 403:
                                reset_header = res.headers.get("x-ratelimit-reset") or res.headers.get("X-RateLimit-Reset")
                                fallback_cooldown = 300.0  # 5 minutes
                                if reset_header:
                                    try:
                                        reset_ts = float(reset_header)
                                        self.next_allowed_github_check = reset_ts if reset_ts > now else (now + fallback_cooldown)
                                    except ValueError:
                                        self.next_allowed_github_check = now + fallback_cooldown
                                else:
                                    self.next_allowed_github_check = now + fallback_cooldown

                                if not self.rate_limited:
                                    logger.warning(
                                        f"GitHub API rate limit exceeded (403). Cooldown active until "
                                        f"{datetime.fromtimestamp(self.next_allowed_github_check, timezone.utc).isoformat()}"
                                    )
                                    self.rate_limited = True

                                self.last_check_ok = False
                                self.last_check_error = "GitHub API rate limit exceeded (403)"
                            else:
                                logger.warning(f"GitHub release check returned status {res.status_code}")
                                self.last_check_ok = False
                                self.last_check_error = f"GitHub API returned HTTP {res.status_code}"
                    except Exception as e:
                        logger.warning(f"Failed to check for updates: {e}")
                        self.last_check_ok = False
                        self.last_check_error = str(e) or "Failed to contact GitHub"
        elif is_cooldown_active:
            self.last_check_ok = False
            self.last_check_error = "GitHub API rate limit cooldown active"

        # 3. Construct response
        if self.cached_release_metadata is not None:
            meta = self.cached_release_metadata
            latest_tag = meta.get("latest_version") or VERSION
            is_update_available = (self._parse_version(latest_tag) > self._parse_version(VERSION))
            metadata_stale = not self.last_check_ok

            return {
                "current_version": VERSION,
                "latest_version": latest_tag,
                "update_available": is_update_available,
                "release_url": meta.get("release_url", ""),
                "release_notes": meta.get("release_notes", ""),
                "published_at": meta.get("published_at"),
                "one_click_update_available": avail,
                "updater_status": st,
                "update_check_ok": self.last_check_ok,
                "update_check_error": self.last_check_error,
                "metadata_stale": metadata_stale,
            }
        else:
            return {
                "current_version": VERSION,
                "latest_version": None,
                "update_available": False,
                "release_url": "",
                "release_notes": "",
                "published_at": None,
                "one_click_update_available": avail,
                "updater_status": st,
                "update_check_ok": False,
                "update_check_error": self.last_check_error or "Update check failed",
                "metadata_stale": False,
            }

    async def trigger_update(self, target_version: str) -> Dict[str, Any]:
        if self.is_locked_for_update():
            return {"success": False, "message": "Update already in progress"}

        async with self._update_lock:
            if self.is_locked_for_update():
                return {"success": False, "message": "Update already in progress"}

            # 1. Acquire atomic maintenance lock in Babel before checking jobs
            self.is_maintenance_locked = True

            try:
                # 2. Check updater availability and status
                avail, current_st = await self.get_real_updater_status()
                if not avail:
                    self.is_maintenance_locked = False
                    return {"success": False, "message": "Updater service unreachable. Is babel-updater running?"}

                if current_st in ["inspecting", "pulling", "replacing", "verifying", "rolling_back", "updating"]:
                    self.update_status = "updating"
                    self.is_maintenance_locked = False
                    return {"success": False, "message": f"Update already in progress ({current_st})"}

                # 3. Check active jobs (with maintenance lock held, no new jobs can enter)
                from app.core.db import get_jobs_by_status
                active_jobs = get_jobs_by_status([
                    "EXTRACTING", "TRANSLATING", "QUEUED", "WAITING_PROVIDER", "RECOVERING",
                    "PARTIAL", "WAITING_SOURCE", "WAITING_FOR_BAZARR", "WAITING_FOR_PUBLICATION"
                ])
                if len(active_jobs) > 0:
                    self.is_maintenance_locked = False
                    return {"success": False, "message": "Active jobs are running. Update blocked."}

                # 4. Target version validation & no-downgrade enforcement
                target_v = (target_version or "").strip()
                if not target_v:
                    self.is_maintenance_locked = False
                    return {"success": False, "message": "Target version cannot be empty"}

                if not re.match(r'^v?\d+\.\d+\.\d+(?:-beta)?$', target_v):
                    self.is_maintenance_locked = False
                    return {"success": False, "message": "Invalid target_version format"}

                info = await self.get_update_info()
                if not info.get("update_available"):
                    self.is_maintenance_locked = False
                    return {"success": False, "message": "No update available"}

                latest_v = info.get("latest_version", "").strip()
                if target_v != latest_v:
                    self.is_maintenance_locked = False
                    return {"success": False, "message": f"Target version {target_v} does not match verified latest release {latest_v}"}

                current_v = VERSION.strip()
                current_tuple = self._parse_version(current_v)
                target_tuple = self._parse_version(target_v)

                if target_tuple <= current_tuple:
                    self.is_maintenance_locked = False
                    return {"success": False, "message": "Target version must be newer than current version (no downgrade/re-install)"}

                current_is_beta = "-beta" in current_v
                target_is_beta = "-beta" in target_v
                if current_is_beta != target_is_beta:
                    self.is_maintenance_locked = False
                    return {"success": False, "message": "Target version channel mismatch"}

                self.update_status = "updating"

                # 5. Call updater with internal auth header
                headers = {}
                if secret := self._get_updater_secret():
                    headers["X-Updater-Token"] = secret

                async with httpx.AsyncClient(timeout=10.0) as client:
                    res = await client.post(
                        f"{UPDATER_URL}/update",
                        json={"target_version": target_v},
                        headers=headers
                    )
                    if res.status_code == 200:
                        return {"success": True, "message": "Update initiated. System will restart shortly."}
                    elif res.status_code == 409:
                        avail, live_st = await self.get_real_updater_status()
                        self.is_maintenance_locked = False
                        if live_st in ["inspecting", "pulling", "replacing", "verifying", "rolling_back", "updating"]:
                            self.update_status = "updating"
                            return {"success": False, "message": f"Update already in progress ({live_st})"}
                        else:
                            self.update_status = "idle"
                            return {"success": False, "message": f"Updater conflict: {res.text}"}
                    else:
                        self.update_status = "idle"
                        self.is_maintenance_locked = False
                        return {"success": False, "message": f"Updater error: {res.text}"}

            except httpx.ConnectError:
                self.update_status = "idle"
                self.is_maintenance_locked = False
                return {"success": False, "message": "Updater service unreachable. Is babel-updater running?"}
            except Exception as e:
                try:
                    avail, live_st = await self.get_real_updater_status()
                    if live_st in ["inspecting", "pulling", "replacing", "verifying", "rolling_back", "updating"]:
                        self.update_status = "updating"
                        return {"success": True, "message": "Update already in progress"}
                except Exception:
                    pass
                self.update_status = "idle"
                self.is_maintenance_locked = False
                return {"success": False, "message": f"Updater request failed: {str(e)}"}

updates_controller = UpdatesController()
