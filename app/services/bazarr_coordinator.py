"""
Bazarr Coordinator for Babel v2.3.43-beta.

Authoritative Bazarr lifecycle coordination component providing:
- Media correlation with ARR identifiers (Radarr / Sonarr) and normalized path matching
- Operation deduplication & idempotent target search triggers
- Real Bazarr job state polling via GET /api/system/jobs (0.4-0.5s interval)
- Provisional vs Finalized candidate lifecycle state machine
- Authoritative Trust Engine finalization gate
- Fast hybrid speculative AI coordination with early-stop signaling
- Publication ownership invariant (no overwriting active Bazarr workers)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

from app.core.languages import normalize_language_code, get_language, LANGUAGES, get_bazarr_language_code
from app.core.trust_engine import (
    SubtitleTrustEngine,
    TrustDecision,
    TrustResult,
    CandidateOrigin,
    CandidateState,
    TargetSnapshot,
    BazarrProvenance,
    capture_target_snapshot,
    wait_for_file_stability,
    DEFAULT_CANDIDATE_STABILITY_SEC,
    DEFAULT_BAZARR_QUIESCENCE_SEC,
    DEFAULT_HYBRID_BAZARR_MAX_WAIT_SEC,
)
from app.core.db import get_setting
from app.services.bazarr_checker import find_external_subtitle
from app.services.source_resolver import (
    BazarrResult,
    BazarrResultCode,
    SubtitleSource,
)

logger = logging.getLogger("babel.bazarr_coordinator")


# ---------------------------------------------------------------------------
# Enums and Data Classes
# ---------------------------------------------------------------------------

class BazarrCorrelationStatus(str, Enum):
    INDEXED         = "INDEXED"
    NOT_INDEXED     = "NOT_INDEXED"
    AUTH_ERROR      = "AUTH_ERROR"
    TEMPORARY_ERROR = "TEMPORARY_ERROR"


class BazarrLifecycleState(str, Enum):
    NOT_STARTED           = "NOT_STARTED"
    WAITING_FOR_MEDIA     = "WAITING_FOR_MEDIA"
    SEARCH_QUEUED         = "SEARCH_QUEUED"
    SEARCHING             = "SEARCHING"
    TARGET_APPEARED       = "TARGET_APPEARED"
    SYNCING               = "SYNCING"
    FINALIZING            = "FINALIZING"
    FINALIZED_WITH_TARGET = "FINALIZED_WITH_TARGET"
    FINALIZED_NO_TARGET   = "FINALIZED_NO_TARGET"
    UNKNOWN               = "UNKNOWN"
    INDETERMINATE         = "INDETERMINATE"
    FAILED                = "FAILED"
    TIMED_OUT             = "TIMED_OUT"


class BazarrJobPollStatus(str, Enum):
    KNOWN_IDLE = "KNOWN_IDLE"
    ACTIVE     = "ACTIVE"
    UNKNOWN    = "UNKNOWN"


@dataclass
class BazarrMediaInfo:
    """Canonical representation of media correlated between Babel and Bazarr."""
    media_type: str = "movie"          # "movie" or "episode"
    radarr_id: Optional[int] = None
    sonarr_series_id: Optional[int] = None
    sonarr_episode_id: Optional[int] = None
    bazarr_id: Optional[int] = None
    title: str = ""
    year: Optional[int] = None
    video_path: str = ""
    bazarr_path: str = ""
    is_indexed: bool = False
    status: BazarrCorrelationStatus = BazarrCorrelationStatus.NOT_INDEXED
    error_message: Optional[str] = None
    http_status: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.is_indexed and self.status == BazarrCorrelationStatus.NOT_INDEXED:
            self.status = BazarrCorrelationStatus.INDEXED


@dataclass
class BazarrJobInfo:
    """Active or recently tracked Bazarr system job."""
    job_id: str
    job_name: str
    status: str                        # "running", "pending", "completed", "failed"
    last_run_time: Optional[str] = None
    is_progress: bool = False
    progress_value: int = 0
    progress_max: int = 100
    progress_message: str = ""
    job_type: str = "other"            # "search", "sync", "other"
    matched_language: Optional[str] = None
    matched_file: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return (
            self.status in ("running", "pending")
            or self.is_progress
            or (self.status not in ("completed", "failed") and bool(self.job_name))
        )


@dataclass
class BazarrJobsPollResult:
    """Explicit typed result of querying Bazarr system jobs."""
    status: BazarrJobPollStatus
    jobs: List[BazarrJobInfo] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def is_known(self) -> bool:
        return self.status != BazarrJobPollStatus.UNKNOWN

    @property
    def is_idle(self) -> bool:
        return self.status == BazarrJobPollStatus.KNOWN_IDLE

    def __iter__(self):
        return iter(self.jobs)

    def __len__(self):
        return len(self.jobs)


def _normalize_poll_result(raw: Any) -> BazarrJobsPollResult:
    """Normalize raw poll return value into BazarrJobsPollResult."""
    if isinstance(raw, BazarrJobsPollResult):
        return raw
    if isinstance(raw, list):
        has_active = any(getattr(j, "is_active", False) for j in raw)
        return BazarrJobsPollResult(
            status=BazarrJobPollStatus.ACTIVE if has_active else BazarrJobPollStatus.KNOWN_IDLE,
            jobs=raw,
        )
    return BazarrJobsPollResult(
        status=BazarrJobPollStatus.UNKNOWN,
        error="invalid_poll_result_type",
    )


@dataclass
class BazarrOperation:
    """State tracking for a single target acquisition operation."""
    op_key: str
    video_path: str
    target_lang: str
    subtitle_intent: str = "full"
    media_info: Optional[BazarrMediaInfo] = None
    state: BazarrLifecycleState = BazarrLifecycleState.NOT_STARTED
    is_search_triggered: bool = False
    trigger_time: float = 0.0
    last_poll_time: float = 0.0
    active_jobs: List[BazarrJobInfo] = field(default_factory=list)
    candidate_path: Optional[str] = None
    last_evaluated_generation: Optional[str] = None
    trust_result: Optional[TrustResult] = None
    is_provisional: bool = False
    error_message: Optional[str] = None


@dataclass
class PublicationOwnershipResult:
    """Result of requesting publication ownership before AI file publish."""
    granted: bool
    reason: str
    adopted: bool = False
    defer: bool = False
    trust_result: Optional[TrustResult] = None
    proven_bazarr_provenance: Optional["BazarrProvenance"] = None
    proven_candidate_snapshot: Optional[TargetSnapshot] = None


# ---------------------------------------------------------------------------
# Path and Language Matching Utilities
# ---------------------------------------------------------------------------

def _normalize_path_components(p: str) -> List[str]:
    norm = os.path.normpath(p).replace("\\", "/")
    return [part for part in norm.lstrip("/").split("/") if part]


def paths_correlate(path_a: str, path_b: str) -> bool:
    """
    Check if two paths refer to the same media file across Docker mount prefixes.
    e.g. '/media/Movies/Title (2003)/Title.mkv' == '/data/media/Movies/Title (2003)/Title.mkv'
    """
    if not path_a or not path_b:
        return False
    p_a = os.path.normpath(path_a)
    p_b = os.path.normpath(path_b)
    if p_a == p_b:
        return True

    # Basename check (must match exactly)
    base_a = os.path.basename(p_a)
    base_b = os.path.basename(p_b)
    if base_a != base_b:
        return False

    # Check suffix depth match (e.g. parent folder / movie title folder)
    parts_a = _normalize_path_components(p_a)
    parts_b = _normalize_path_components(p_b)
    depth = min(3, len(parts_a), len(parts_b))
    if depth >= 2:
        return parts_a[-depth:] == parts_b[-depth:]
    return True


def _extract_job_language_codes(text: str, ignore_title: str = "") -> Set[str]:
    """
    Extract canonical language codes explicitly mentioned as separate tokens in text,
    excluding the matched media title span to avoid false positives without erasing
    explicit language codes appearing outside the title.
    Disambiguates compound codes (e.g. pt-BR) from simple codes (e.g. pt).
    """
    if not text:
        return set()

    text_lower = text.lower()
    if ignore_title:
        ignore_lower = ignore_title.lower().strip()
        if ignore_lower:
            if ignore_lower in text_lower:
                text_lower = text_lower.replace(ignore_lower, " ", 1)
            else:
                title_parts = [re.escape(t) for t in re.split(r'[\s:._/\\()\[\]{}#,+-]+', ignore_lower) if t]
                if title_parts:
                    pattern = re.compile(r'[\s:._/\\()\[\]{}#,+-]+'.join(title_parts), re.IGNORECASE)
                    text_lower = pattern.sub(" ", text_lower, count=1)

    found_codes: Set[str] = set()

    # Collect all (pattern, canonical_code) pairs and test longest patterns first
    patterns: List[Tuple[str, str]] = []
    for lang in LANGUAGES:
        for alias in lang.aliases:
            patterns.append((alias.lower(), lang.code))
        patterns.append((lang.display_name.lower(), lang.code))
        patterns.append((lang.code.lower(), lang.code))

    # De-duplicate and sort by length descending
    unique_patterns = sorted(list(dict.fromkeys(patterns)), key=lambda x: len(x[0]), reverse=True)

    working_text = text_lower
    for pat, code in unique_patterns:
        # Check boundary with regex: not preceded or followed by alphanumeric
        escaped_pat = re.escape(pat)
        regex = re.compile(r'(?<![a-z0-9])' + escaped_pat + r'(?![a-z0-9])')
        if regex.search(working_text):
            found_codes.add(code)
            # Mask out matched pattern to avoid sub-alias collisions (e.g. 'pt' inside 'pt-br')
            working_text = regex.sub(" " * len(pat), working_text)

    return found_codes


def _language_matches_job_text(lang_code: str, text: str) -> bool:
    """Check if job description refers to the specified language code or name using exact token boundaries."""
    if not text:
        return False
    norm_target = normalize_language_code(lang_code, default=lang_code)
    found_codes = _extract_job_language_codes(text)
    if norm_target in found_codes:
        return True
    return False


# ---------------------------------------------------------------------------
# Bazarr Coordinator Service
# ---------------------------------------------------------------------------

class BazarrCoordinator:
    """
    Central coordinator managing all interactions with Bazarr's API,
    active job polling, media correlation, and publication ownership.
    """

    def __init__(self):
        self._operations: Dict[str, BazarrOperation] = {}
        self._media_cache: Dict[str, BazarrMediaInfo] = {}
        self._in_flight_triggers: Dict[str, asyncio.Future[BazarrResult]] = {}
        self._lock = asyncio.Lock()

    def _get_op_key(
        self,
        video_path: str,
        target_lang: str,
        subtitle_intent: str = "full",
        arr_id: Optional[int] = None,
        media_type: Optional[str] = None
    ) -> str:
        norm_lang = normalize_language_code(target_lang, default=target_lang).lower()
        if arr_id and media_type:
            return f"{media_type}:{arr_id}:{norm_lang}:{subtitle_intent}"
        canonical = os.path.realpath(os.path.abspath(video_path))
        path_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return f"path:{path_hash}:{norm_lang}:{subtitle_intent}"

    def reset(self) -> None:
        """Clear all active operations and media cache (primarily for testing)."""
        self._operations.clear()
        self._media_cache.clear()
        self._in_flight_triggers.clear()
        for attr in ["poll_system_jobs", "correlate_media", "coordinate_target", "acquire_publication_ownership", "trigger_search"]:
            self.__dict__.pop(attr, None)

    # ── Media Correlation ───────────────────────────────────────────────────

    async def correlate_media(
        self,
        video_path: str,
        bazarr_url: str,
        bazarr_api_key: str,
        radarr_id: Optional[int] = None,
        sonarr_series_id: Optional[int] = None,
        sonarr_episode_id: Optional[int] = None,
        media_type: Optional[str] = None,
        timeout: float = 4.0,
    ) -> BazarrMediaInfo:
        """
        Authoritatively correlate a Babel video path with Bazarr's database.
        Priority hierarchy:
          1. ARR IDs (radarr_id / sonarr_episode_id)
          2. Bazarr entity IDs / canonical metadata
          3. Normalized relative path / basename fallback
          4. Title / Year fallback
        """
        if not bazarr_url or not bazarr_api_key:
            return BazarrMediaInfo(
                media_type=media_type or "movie",
                radarr_id=radarr_id,
                sonarr_series_id=sonarr_series_id,
                sonarr_episode_id=sonarr_episode_id,
                video_path=video_path,
                is_indexed=False,
                status=BazarrCorrelationStatus.AUTH_ERROR if not bazarr_api_key else BazarrCorrelationStatus.TEMPORARY_ERROR,
                error_message="Missing Bazarr URL or API key",
            )

        clean_url = bazarr_url.rstrip("/")
        headers = {"X-API-KEY": bazarr_api_key}
        base_name = os.path.basename(video_path)
        norm_video_path = os.path.normpath(video_path)

        if norm_video_path in self._media_cache and self._media_cache[norm_video_path].is_indexed:
            return self._media_cache[norm_video_path]

        async with httpx.AsyncClient(timeout=timeout) as client:
            # --- 1. Radarr ID / Movie lookup ---
            if radarr_id or media_type == "movie" or not (sonarr_series_id or sonarr_episode_id):
                try:
                    params = {"radarrid[]": str(radarr_id)} if radarr_id else {}
                    res = await client.get(f"{clean_url}/api/movies", headers=headers, params=params)
                    if res.status_code in (401, 403):
                        return BazarrMediaInfo(
                            media_type="movie",
                            radarr_id=radarr_id,
                            video_path=video_path,
                            is_indexed=False,
                            status=BazarrCorrelationStatus.AUTH_ERROR,
                            http_status=res.status_code,
                            error_message=f"HTTP {res.status_code} Auth Error",
                        )
                    elif res.status_code >= 500:
                        return BazarrMediaInfo(
                            media_type="movie",
                            radarr_id=radarr_id,
                            video_path=video_path,
                            is_indexed=False,
                            status=BazarrCorrelationStatus.TEMPORARY_ERROR,
                            http_status=res.status_code,
                            error_message=f"HTTP {res.status_code} Server Error",
                        )
                    elif res.status_code == 200:
                        try:
                            m_data = res.json()
                        except Exception:
                            return BazarrMediaInfo(
                                media_type="movie",
                                radarr_id=radarr_id,
                                video_path=video_path,
                                is_indexed=False,
                                status=BazarrCorrelationStatus.TEMPORARY_ERROR,
                                error_message="Malformed JSON response from /api/movies",
                            )
                        movies = m_data.get("data", []) if isinstance(m_data, dict) else (
                            m_data if isinstance(m_data, list) else []
                        )
                        for m in movies:
                            if not isinstance(m, dict):
                                continue
                            m_radarr_id = m.get("radarrId")
                            m_path = m.get("path", "")

                            # Match if radarrId matches or path correlates
                            if (radarr_id and m_radarr_id == radarr_id) or paths_correlate(norm_video_path, m_path) or (m_path and os.path.basename(m_path) == base_name):
                                info = BazarrMediaInfo(
                                    media_type="movie",
                                    radarr_id=m_radarr_id or radarr_id,
                                    bazarr_id=m.get("id"),
                                    title=m.get("title", ""),
                                    year=m.get("year"),
                                    video_path=video_path,
                                    bazarr_path=m_path,
                                    is_indexed=True,
                                    status=BazarrCorrelationStatus.INDEXED,
                                    metadata=m,
                                )
                                self._media_cache[norm_video_path] = info
                                return info
                        if radarr_id or media_type == "movie":
                            return BazarrMediaInfo(
                                media_type="movie",
                                radarr_id=radarr_id,
                                video_path=video_path,
                                is_indexed=False,
                                status=BazarrCorrelationStatus.NOT_INDEXED,
                            )
                    elif 400 <= res.status_code < 500 and res.status_code != 404:
                        return BazarrMediaInfo(
                            media_type="movie",
                            radarr_id=radarr_id,
                            video_path=video_path,
                            is_indexed=False,
                            status=BazarrCorrelationStatus.TEMPORARY_ERROR,
                            http_status=res.status_code,
                            error_message=f"HTTP {res.status_code} Client Error",
                        )
                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    return BazarrMediaInfo(
                        media_type="movie",
                        radarr_id=radarr_id,
                        video_path=video_path,
                        is_indexed=False,
                        status=BazarrCorrelationStatus.TEMPORARY_ERROR,
                        error_message=f"Network error / timeout querying /api/movies: {e}",
                    )
                except Exception as e:
                    logger.debug(f"Bazarr movie correlation error: {e}")
                    return BazarrMediaInfo(
                        media_type="movie",
                        radarr_id=radarr_id,
                        video_path=video_path,
                        is_indexed=False,
                        status=BazarrCorrelationStatus.TEMPORARY_ERROR,
                        error_message=str(e),
                    )

            # --- 2. Sonarr Series & Episode lookup ---
            if sonarr_series_id or sonarr_episode_id or media_type == "episode" or not radarr_id:
                try:
                    s_params = {"seriesid[]": str(sonarr_series_id)} if sonarr_series_id else {}
                    s_res = await client.get(f"{clean_url}/api/series", headers=headers, params=s_params)
                    if s_res.status_code in (401, 403):
                        return BazarrMediaInfo(
                            media_type="episode",
                            sonarr_series_id=sonarr_series_id,
                            sonarr_episode_id=sonarr_episode_id,
                            video_path=video_path,
                            is_indexed=False,
                            status=BazarrCorrelationStatus.AUTH_ERROR,
                            http_status=s_res.status_code,
                            error_message=f"HTTP {s_res.status_code} Auth Error",
                        )
                    elif s_res.status_code >= 500:
                        return BazarrMediaInfo(
                            media_type="episode",
                            sonarr_series_id=sonarr_series_id,
                            sonarr_episode_id=sonarr_episode_id,
                            video_path=video_path,
                            is_indexed=False,
                            status=BazarrCorrelationStatus.TEMPORARY_ERROR,
                            http_status=s_res.status_code,
                            error_message=f"HTTP {s_res.status_code} Server Error",
                        )
                    elif s_res.status_code == 200:
                        try:
                            s_data = s_res.json()
                        except Exception:
                            return BazarrMediaInfo(
                                media_type="episode",
                                sonarr_series_id=sonarr_series_id,
                                sonarr_episode_id=sonarr_episode_id,
                                video_path=video_path,
                                is_indexed=False,
                                status=BazarrCorrelationStatus.TEMPORARY_ERROR,
                                error_message="Malformed JSON response from /api/series",
                            )
                        all_series = s_data.get("data", []) if isinstance(s_data, dict) else (
                            s_data if isinstance(s_data, list) else []
                        )

                        for series in all_series:
                            if not isinstance(series, dict):
                                continue
                            s_id = series.get("sonarrSeriesId")
                            s_path = series.get("path", "")

                            # Match series by ID or path containment
                            series_match = (
                                (sonarr_series_id and s_id == sonarr_series_id)
                                or (s_path and (s_path in norm_video_path or os.path.basename(s_path) in norm_video_path or paths_correlate(norm_video_path, s_path)))
                            )
                            if not series_match:
                                continue

                            # Query episodes for this series
                            ep_res = await client.get(
                                f"{clean_url}/api/episodes",
                                headers=headers,
                                params={"seriesid[]": str(s_id)},
                            )
                            if ep_res.status_code in (401, 403):
                                return BazarrMediaInfo(
                                    media_type="episode",
                                    sonarr_series_id=s_id,
                                    sonarr_episode_id=sonarr_episode_id,
                                    video_path=video_path,
                                    is_indexed=False,
                                    status=BazarrCorrelationStatus.AUTH_ERROR,
                                    http_status=ep_res.status_code,
                                    error_message=f"HTTP {ep_res.status_code} Auth Error",
                                )
                            elif ep_res.status_code >= 500:
                                return BazarrMediaInfo(
                                    media_type="episode",
                                    sonarr_series_id=s_id,
                                    sonarr_episode_id=sonarr_episode_id,
                                    video_path=video_path,
                                    is_indexed=False,
                                    status=BazarrCorrelationStatus.TEMPORARY_ERROR,
                                    http_status=ep_res.status_code,
                                    error_message=f"HTTP {ep_res.status_code} Server Error",
                                )
                            elif ep_res.status_code == 200:
                                try:
                                    ep_data = ep_res.json()
                                except Exception:
                                    return BazarrMediaInfo(
                                        media_type="episode",
                                        sonarr_series_id=s_id,
                                        sonarr_episode_id=sonarr_episode_id,
                                        video_path=video_path,
                                        is_indexed=False,
                                        status=BazarrCorrelationStatus.TEMPORARY_ERROR,
                                        error_message="Malformed JSON response from /api/episodes",
                                    )
                                episodes = ep_data.get("data", []) if isinstance(ep_data, dict) else (
                                    ep_data if isinstance(ep_data, list) else []
                                )
                                for ep in episodes:
                                    if not isinstance(ep, dict):
                                        continue
                                    e_id = ep.get("sonarrEpisodeId")
                                    ep_path = ep.get("path", "")
                                    if (sonarr_episode_id and e_id == sonarr_episode_id) or paths_correlate(norm_video_path, ep_path) or (ep_path and os.path.basename(ep_path) == base_name):
                                        info = BazarrMediaInfo(
                                            media_type="episode",
                                            sonarr_series_id=s_id,
                                            sonarr_episode_id=e_id,
                                            bazarr_id=ep.get("id"),
                                            title=series.get("title", ""),
                                            video_path=video_path,
                                            bazarr_path=ep_path,
                                            is_indexed=True,
                                            status=BazarrCorrelationStatus.INDEXED,
                                            metadata={"series": series, "episode": ep},
                                        )
                                        self._media_cache[norm_video_path] = info
                                        return info
                    elif 400 <= s_res.status_code < 500 and s_res.status_code != 404:
                        return BazarrMediaInfo(
                            media_type="episode",
                            sonarr_series_id=sonarr_series_id,
                            sonarr_episode_id=sonarr_episode_id,
                            video_path=video_path,
                            is_indexed=False,
                            status=BazarrCorrelationStatus.TEMPORARY_ERROR,
                            http_status=s_res.status_code,
                            error_message=f"HTTP {s_res.status_code} Client Error",
                        )
                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    return BazarrMediaInfo(
                        media_type="episode",
                        sonarr_series_id=sonarr_series_id,
                        sonarr_episode_id=sonarr_episode_id,
                        video_path=video_path,
                        is_indexed=False,
                        status=BazarrCorrelationStatus.TEMPORARY_ERROR,
                        error_message=f"Network error / timeout querying /api/series: {e}",
                    )
                except Exception as e:
                    logger.debug(f"Bazarr episode correlation error: {e}")
                    return BazarrMediaInfo(
                        media_type="episode",
                        sonarr_series_id=sonarr_series_id,
                        sonarr_episode_id=sonarr_episode_id,
                        video_path=video_path,
                        is_indexed=False,
                        status=BazarrCorrelationStatus.TEMPORARY_ERROR,
                        error_message=str(e),
                    )

        # Media not currently indexed in Bazarr
        return BazarrMediaInfo(
            media_type=media_type or "movie",
            radarr_id=radarr_id,
            sonarr_series_id=sonarr_series_id,
            sonarr_episode_id=sonarr_episode_id,
            video_path=video_path,
            is_indexed=False,
            status=BazarrCorrelationStatus.NOT_INDEXED,
        )

    # ── Job Polling ─────────────────────────────────────────────────────────

    async def poll_system_jobs(
        self,
        bazarr_url: str,
        bazarr_api_key: str,
        timeout: float = 3.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> BazarrJobsPollResult:
        """
        Poll Bazarr's real job state via GET /api/system/jobs.
        Returns explicit BazarrJobsPollResult distinguishing KNOWN_IDLE, ACTIVE, and UNKNOWN.
        """
        if not bazarr_url or not bazarr_api_key:
            return BazarrJobsPollResult(
                status=BazarrJobPollStatus.UNKNOWN,
                error="missing_config",
            )
        clean_url = bazarr_url.rstrip("/")
        headers = {"X-API-KEY": bazarr_api_key}

        async def _fetch(c: httpx.AsyncClient) -> BazarrJobsPollResult:
            res = await c.get(f"{clean_url}/api/system/jobs", headers=headers)
            if res.status_code == 200:
                data = res.json()
                raw_jobs = data.get("data", []) if isinstance(data, dict) else (
                    data if isinstance(data, list) else []
                )
                jobs = []
                for rj in raw_jobs:
                    if not isinstance(rj, dict):
                        continue
                    j_id = str(rj.get("job_id") or rj.get("id") or "")
                    j_name = rj.get("job_name") or rj.get("name") or ""
                    j_status = (rj.get("status") or "running").lower()
                    j_prog = bool(rj.get("is_progress", False))
                    j_msg = rj.get("progress_message") or ""

                    # Classify job type
                    j_type = "other"
                    matched_lang = None
                    matched_file = None

                    name_lower = j_name.lower()
                    msg_lower = j_msg.lower()

                    if "syncing" in name_lower or "syncing" in msg_lower:
                        j_type = "sync"
                        matched_file = j_name.replace("Syncing", "").replace("syncing", "").strip()
                    elif "search" in name_lower or "searching" in msg_lower or "download_specific_subtitles" in j_name:
                        j_type = "search"

                    jobs.append(BazarrJobInfo(
                        job_id=j_id,
                        job_name=j_name,
                        status=j_status,
                        last_run_time=rj.get("last_run_time"),
                        is_progress=j_prog,
                        progress_value=int(rj.get("progress_value", 0)),
                        progress_max=int(rj.get("progress_max", 100)),
                        progress_message=j_msg,
                        job_type=j_type,
                        matched_language=matched_lang,
                        matched_file=matched_file,
                    ))

                has_active = any(j.is_active for j in jobs)
                return BazarrJobsPollResult(
                    status=BazarrJobPollStatus.ACTIVE if has_active else BazarrJobPollStatus.KNOWN_IDLE,
                    jobs=jobs,
                )
            else:
                logger.debug(f"Bazarr jobs poll non-200: {res.status_code}")
                return BazarrJobsPollResult(
                    status=BazarrJobPollStatus.UNKNOWN,
                    error=f"http_{res.status_code}",
                )

        try:
            if client is not None:
                return await _fetch(client)
            else:
                async with httpx.AsyncClient(timeout=timeout) as c:
                    return await _fetch(c)
        except Exception as e:
            logger.debug(f"Failed to poll Bazarr jobs: {e}")
            return BazarrJobsPollResult(
                status=BazarrJobPollStatus.UNKNOWN,
                error=str(e),
            )

    def classify_jobs_for_target(
        self,
        jobs_or_poll_res: Any,
        video_path: str,
        target_lang: str,
        media_info: Optional[BazarrMediaInfo] = None,
    ) -> Tuple[List[BazarrJobInfo], List[BazarrJobInfo]]:
        """
        Classifies active jobs into:
          1. Target search jobs matching this video and language
          2. Target sync jobs matching this target subtitle file / language
        """
        jobs: List[BazarrJobInfo] = []
        if isinstance(jobs_or_poll_res, BazarrJobsPollResult):
            jobs = jobs_or_poll_res.jobs
        elif isinstance(jobs_or_poll_res, list):
            jobs = jobs_or_poll_res

        search_jobs: List[BazarrJobInfo] = []
        sync_jobs: List[BazarrJobInfo] = []

        base_name = os.path.basename(video_path)
        base_stem, _ = os.path.splitext(base_name)
        lang_norm = normalize_language_code(target_lang, default=target_lang).lower()
        title_str = media_info.title if media_info and media_info.title else base_stem

        clean_stem = re.sub(r'[\._]', ' ', base_stem).strip().lower()
        clean_title = re.sub(r'[\._]', ' ', title_str).strip().lower()
        bazarr_base = os.path.basename(media_info.bazarr_path).lower() if (media_info and media_info.bazarr_path) else ""
        clean_bazarr_base = re.sub(r'[\._]', ' ', bazarr_base).strip().lower() if bazarr_base else ""

        for job in jobs:
            if not job.is_active:
                continue

            j_text = f"{job.job_name} {job.progress_message}".strip()
            j_text_lower = j_text.lower()
            j_clean = re.sub(r'[\._]', ' ', j_text_lower)

            # --- Target Sync Job Matching ---
            if job.job_type == "sync" or "syncing" in j_text_lower:
                is_file_or_title_match = (
                    base_stem.lower() in j_text_lower
                    or clean_stem in j_clean
                    or title_str.lower() in j_text_lower
                    or clean_title in j_clean
                    or (bool(bazarr_base) and (bazarr_base in j_text_lower or clean_bazarr_base in j_clean))
                )
                if is_file_or_title_match:
                    job_langs = _extract_job_language_codes(j_text, ignore_title=title_str)
                    if job.matched_language:
                        norm_matched = normalize_language_code(job.matched_language, default=job.matched_language).lower()
                        job_langs.add(norm_matched)

                    if not job_langs:
                        if _language_matches_job_text(lang_norm, j_text):
                            sync_jobs.append(job)
                    elif lang_norm in job_langs:
                        sync_jobs.append(job)

            # --- Target Search Job Matching ---
            elif job.job_type == "search" or "searching" in j_text_lower or "searched" in j_text_lower:
                is_file_or_title_match = (
                    title_str.lower() in j_text_lower
                    or clean_title in j_clean
                    or base_stem.lower() in j_text_lower
                    or clean_stem in j_clean
                    or (bool(bazarr_base) and (bazarr_base in j_text_lower or clean_bazarr_base in j_clean))
                )
                if is_file_or_title_match:
                    job_langs = _extract_job_language_codes(j_text, ignore_title=title_str)
                    if not job_langs or lang_norm in job_langs:
                        search_jobs.append(job)
                elif _language_matches_job_text(lang_norm, j_text) and any(x in j_text_lower for x in ["download_specific_subtitles", "search_subtitles"]) and not any(other_t in j_text_lower for other_t in [".mkv", ".mp4", ".avi"]):
                    search_jobs.append(job)

        return search_jobs, sync_jobs

    def is_job_conclusively_unrelated(
        self,
        job: BazarrJobInfo,
        video_path: str,
        target_lang: str,
        media_info: Optional[BazarrMediaInfo] = None,
    ) -> bool:
        """
        Determines whether an active Bazarr job is conclusively unrelated to this
        target media and language operation.

        Returns True ONLY if the job is definitely not related to this operation:
          - Non-subtitle system maintenance (e.g. backup, health check, update series)
          - Subtitle work explicitly for a completely different media file/title
          - Subtitle work explicitly for a different language with no language match

        Returns False if the job could potentially be an unclassified subtitle search,
        download, or sync operation for this target (e.g. generic 'Downloading Subtitles',
        'download_specific_subtitles', 'Searching Subtitles' without explicit conflicting media).
        """
        if not job.is_active:
            return True

        j_text = f"{job.job_name} {job.progress_message}".strip()
        j_text_lower = j_text.lower()
        j_clean = re.sub(r'[\._]', ' ', j_text_lower)

        base_name = os.path.basename(video_path)
        base_stem, _ = os.path.splitext(base_name)
        lang_norm = normalize_language_code(target_lang, default=target_lang).lower()
        title_str = media_info.title if media_info and media_info.title else base_stem
        clean_stem = re.sub(r'[\._]', ' ', base_stem).strip().lower()
        clean_title = re.sub(r'[\._]', ' ', title_str).strip().lower()
        bazarr_base = os.path.basename(media_info.bazarr_path).lower() if (media_info and media_info.bazarr_path) else ""
        clean_bazarr_base = re.sub(r'[\._]', ' ', bazarr_base).strip().lower() if bazarr_base else ""

        # 1. Non-subtitle system maintenance tasks
        pure_system_jobs = [
            "backup", "health", "analytics", "clean_cache", "clear_cache",
            "disk_space", "update_series", "update_movies", "rss_sync", "tasks"
        ]
        is_system_maintenance = (
            any(k in j_text_lower for k in pure_system_jobs)
            and not any(sub_k in j_text_lower for sub_k in ["sub", "sync", "download", "search"])
        )
        if is_system_maintenance:
            return True

        # 2. Check if job explicitly mentions another language with no overlap with ours
        job_langs = _extract_job_language_codes(j_text, ignore_title=title_str)
        if job.matched_language:
            job_langs.add(normalize_language_code(job.matched_language, default=job.matched_language).lower())

        if job_langs and lang_norm not in job_langs:
            # Explicitly for a different language (e.g. French vs Swedish)
            return True

        # 3. Check if job explicitly mentions a distinct media file/title that is not ours
        has_media_ref = bool(re.search(r'\.(mkv|mp4|avi|ts|m4v|iso)\b', j_text_lower))
        if has_media_ref:
            is_our_media = (
                base_stem.lower() in j_text_lower
                or clean_stem in j_clean
                or title_str.lower() in j_text_lower
                or clean_title in j_clean
                or (bool(bazarr_base) and (bazarr_base in j_text_lower or clean_bazarr_base in j_clean))
            )
            if not is_our_media:
                return True

        # If job is an unclassified subtitle task (e.g. generic "Downloading Subtitles",
        # "download_specific_subtitles", "Searching Subtitles") without conflicting media/language,
        # it is ambiguous and cannot be conclusively ruled out.
        return False

    def target_has_correlated_active_work(
        self,
        jobs_or_poll_res: Any,
        video_path: str,
        target_lang: str,
        media_info: Optional[BazarrMediaInfo] = None,
    ) -> Tuple[bool, List[BazarrJobInfo], List[BazarrJobInfo]]:
        """
        Determines whether Bazarr has active search or sync jobs specifically
        correlated to this target video and language.
        """
        search_jobs, sync_jobs = self.classify_jobs_for_target(
            jobs_or_poll_res, video_path, target_lang, media_info=media_info
        )
        return bool(search_jobs or sync_jobs), search_jobs, sync_jobs

    async def target_has_write_risk(
        self,
        poll_res: Any,
        video_path: str,
        target_lang: str,
        candidate_path: Optional[str] = None,
        media_info: Optional[BazarrMediaInfo] = None,
    ) -> Tuple[bool, str]:
        """
        Evaluates whether there is concrete target-specific evidence that THIS exact
        target file is actively being written or modified.
        """
        norm_poll = _normalize_poll_result(poll_res)
        if norm_poll.status == BazarrJobPollStatus.UNKNOWN:
            return True, "bazarr_lifecycle_unknown"

        search_jobs, sync_jobs = self.classify_jobs_for_target(
            norm_poll, video_path, target_lang, media_info=media_info
        )
        if sync_jobs:
            return True, "bazarr_target_syncing"
        if search_jobs:
            return True, "bazarr_target_searching"

        if candidate_path and os.path.exists(candidate_path):
            stable = await wait_for_file_stability(candidate_path, timeout_sec=0.2, interval_sec=0.025)
            if not stable:
                return True, "target_file_unstable"

        return False, "target_clear"

    def evaluate_target_idle_status(
        self,
        poll_res: Any,
        video_path: str,
        target_lang: str,
        search_accepted: bool = False,
        media_info: Optional[BazarrMediaInfo] = None,
    ) -> Tuple[bool, List[BazarrJobInfo], List[BazarrJobInfo], List[BazarrJobInfo]]:
        """
        Target-scoped evaluation of whether Bazarr is idle for the given media and language.
        A target is authoritatively idle when poll status is not UNKNOWN and no correlated
        search or sync jobs exist for this target.
        """
        norm_poll = _normalize_poll_result(poll_res)
        if norm_poll.status == BazarrJobPollStatus.UNKNOWN:
            return False, [], [], []

        search_jobs, sync_jobs = self.classify_jobs_for_target(
            norm_poll, video_path, target_lang, media_info=media_info
        )
        if search_jobs or sync_jobs:
            return False, search_jobs, sync_jobs, []

        return True, [], [], []

    # ── Idempotent Target Search Trigger ────────────────────────────────────

    async def trigger_or_attach_target_search(
        self,
        video_path: str,
        target_lang: str,
        bazarr_url: str,
        bazarr_api_key: str,
        radarr_id: Optional[int] = None,
        sonarr_series_id: Optional[int] = None,
        sonarr_episode_id: Optional[int] = None,
        media_type: Optional[str] = None,
        job_id: Optional[int] = None,
        event_source: Optional[str] = None,
        timeout: float = 15.0,
        readiness_timeout: Optional[float] = None,
    ) -> BazarrResult:
        """
        Trigger a Bazarr target search idempotently.
        If a search operation was already recently triggered or is active in Bazarr's jobs,
        attaches to that existing operation instead of triggering a duplicate.
        Concurrency:
          - self._lock protects ONLY in-memory state (self._operations and self._in_flight_triggers).
          - No network calls, job polling, correlation, or sleeps occur under the global lock.
          - Multiple requests for DIFFERENT media progress concurrently.
          - Simultaneous requests for the SAME media attach to the in-flight operation.
        """
        lang_norm = normalize_language_code(target_lang, default=target_lang)
        op_key = self._get_op_key(video_path, lang_norm, arr_id=radarr_id or sonarr_episode_id, media_type=media_type)

        loop = asyncio.get_running_loop()
        in_flight_fut: Optional[asyncio.Future[BazarrResult]] = None
        is_owner = False
        owner_fut: Optional[asyncio.Future[BazarrResult]] = None

        # 1. Inspect and claim in-memory coordinator state under global lock
        async with self._lock:
            now = time.monotonic()
            existing_op = self._operations.get(op_key)

            # Check if an identical trigger is already in-flight right now
            if op_key in self._in_flight_triggers:
                in_flight_fut = self._in_flight_triggers[op_key]
            # Attach if an operation is genuinely active in Bazarr or very recently triggered (< 5s)
            elif existing_op and (now - existing_op.trigger_time < 5.0) and existing_op.state not in (
                BazarrLifecycleState.FINALIZED_WITH_TARGET,
                BazarrLifecycleState.FINALIZED_NO_TARGET,
                BazarrLifecycleState.FAILED,
            ):
                if job_id:
                    from app.core.db import append_job_log
                    append_job_log(job_id, f"Bazarr target search: attached to active operation for {lang_norm.upper()} (no duplicate trigger)")
                is_corr = bool(existing_op.media_info and existing_op.media_info.is_indexed)
                return BazarrResult(
                    code=BazarrResultCode.TRIGGERED,
                    language=lang_norm,
                    detail="attached_to_existing",
                    http_status=200,
                    media_correlated=is_corr,
                )
            else:
                # We are the owner for this op_key
                is_owner = True
                owner_fut = loop.create_future()
                self._in_flight_triggers[op_key] = owner_fut

        # 2. If another caller is currently running the trigger for this op_key, wait for its result
        if in_flight_fut is not None:
            try:
                res = await asyncio.shield(in_flight_fut)
                if res.code in (BazarrResultCode.TRIGGERED, BazarrResultCode.ACCEPTED):
                    if job_id:
                        from app.core.db import append_job_log
                        append_job_log(job_id, f"Bazarr target search: attached to in-flight operation for {lang_norm.upper()} (no duplicate trigger)")
                    return BazarrResult(
                        code=BazarrResultCode.TRIGGERED,
                        language=lang_norm,
                        detail="attached_to_existing",
                        http_status=200,
                        media_correlated=res.media_correlated,
                    )
                return res
            except Exception as e:
                return BazarrResult(
                    code=BazarrResultCode.TEMPORARY_ERROR,
                    language=lang_norm,
                    detail=f"In-flight operation error: {e}",
                )

        # 3. Owner execution: NO global lock is held across network awaits/sleeps!
        result: Optional[BazarrResult] = None
        try:
            # Check if there is an older active operation (> 5s and < 120s) that might have finished
            now = time.monotonic()
            existing_op = self._operations.get(op_key)
            if existing_op and (now - existing_op.trigger_time < 120.0) and existing_op.state not in (
                BazarrLifecycleState.FINALIZED_WITH_TARGET,
                BazarrLifecycleState.FINALIZED_NO_TARGET,
                BazarrLifecycleState.FAILED,
            ):
                active_jobs = await self.poll_system_jobs(bazarr_url, bazarr_api_key)
                search_jobs, _ = self.classify_jobs_for_target(active_jobs, video_path, lang_norm)
                if search_jobs:
                    if job_id:
                        from app.core.db import append_job_log
                        append_job_log(job_id, f"Bazarr target search: attached to active operation for {lang_norm.upper()} (no duplicate trigger)")
                    is_corr = bool(existing_op.media_info and existing_op.media_info.is_indexed)
                    result = BazarrResult(
                        code=BazarrResultCode.TRIGGERED,
                        language=lang_norm,
                        detail="attached_to_existing",
                        http_status=200,
                        media_correlated=is_corr,
                    )
                    return result
                else:
                    existing_op.state = BazarrLifecycleState.FINALIZED_NO_TARGET

            # Check Bazarr's real jobs queue before sending a new trigger
            active_jobs = await self.poll_system_jobs(bazarr_url, bazarr_api_key)
            search_jobs, _ = self.classify_jobs_for_target(active_jobs, video_path, lang_norm)
            if search_jobs:
                if job_id:
                    from app.core.db import append_job_log
                    append_job_log(job_id, f"Bazarr target search: found active job '{search_jobs[0].job_name}' — attaching")
                op = BazarrOperation(
                    op_key=op_key,
                    video_path=video_path,
                    target_lang=lang_norm,
                    state=BazarrLifecycleState.SEARCHING,
                    is_search_triggered=True,
                    trigger_time=time.monotonic(),
                    active_jobs=search_jobs,
                )
                async with self._lock:
                    self._operations[op_key] = op
                result = BazarrResult(
                    code=BazarrResultCode.TRIGGERED,
                    language=lang_norm,
                    detail="attached_to_active_bazarr_job",
                    http_status=200,
                    media_correlated=False,
                )
                return result

            # Perform Media Correlation
            media_info = await self.correlate_media(
                video_path=video_path,
                bazarr_url=bazarr_url,
                bazarr_api_key=bazarr_api_key,
                radarr_id=radarr_id,
                sonarr_series_id=sonarr_series_id,
                sonarr_episode_id=sonarr_episode_id,
                media_type=media_type,
            )

            # Distinguish AUTH_ERROR and TEMPORARY_ERROR immediately (must not masquerade as indexing delay)
            if media_info.status == BazarrCorrelationStatus.AUTH_ERROR:
                if job_id:
                    from app.core.db import append_job_log
                    append_job_log(job_id, f"Bazarr correlation AUTH_ERROR ({media_info.error_message})")
                result = BazarrResult(
                    code=BazarrResultCode.AUTH_ERROR,
                    language=lang_norm,
                    detail=media_info.error_message or "Bazarr authentication failed during media correlation",
                    http_status=media_info.http_status or 401,
                )
                return result

            if media_info.status == BazarrCorrelationStatus.TEMPORARY_ERROR:
                if job_id:
                    from app.core.db import append_job_log
                    append_job_log(job_id, f"Bazarr correlation TEMPORARY_ERROR ({media_info.error_message})")
                result = BazarrResult(
                    code=BazarrResultCode.TEMPORARY_ERROR,
                    language=lang_norm,
                    detail=media_info.error_message or "Bazarr temporary error during media correlation",
                    http_status=media_info.http_status or 500,
                )
                return result

            # Bounded retry loop ONLY if media is genuinely not yet indexed (NOT_INDEXED)
            if not media_info.is_indexed:
                from app.core.db import get_setting
                max_readiness_wait = (
                    readiness_timeout
                    if readiness_timeout is not None
                    else float(get_setting("bazarr_readiness_timeout_sec", "12.0"))
                )
                if max_readiness_wait > 0:
                    t_correlate_start = time.monotonic()
                    if job_id:
                        from app.core.db import append_job_log
                        append_job_log(
                            job_id,
                            f"Bazarr correlation: WAITING_FOR_MEDIA (indexing in progress, window={max_readiness_wait:.1f}s)...",
                        )

                    while (time.monotonic() - t_correlate_start) < max_readiness_wait:
                        await asyncio.sleep(0.5)
                        media_info = await self.correlate_media(
                            video_path=video_path,
                            bazarr_url=bazarr_url,
                            bazarr_api_key=bazarr_api_key,
                            radarr_id=radarr_id,
                            sonarr_series_id=sonarr_series_id,
                            sonarr_episode_id=sonarr_episode_id,
                            media_type=media_type,
                        )
                        if media_info.is_indexed:
                            if job_id:
                                from app.core.db import append_job_log
                                append_job_log(job_id, f"Bazarr correlation: media matched ({media_info.title})")
                            break
                        if media_info.status in (BazarrCorrelationStatus.AUTH_ERROR, BazarrCorrelationStatus.TEMPORARY_ERROR):
                            break

            if media_info.status == BazarrCorrelationStatus.AUTH_ERROR:
                result = BazarrResult(
                    code=BazarrResultCode.AUTH_ERROR,
                    language=lang_norm,
                    detail=media_info.error_message or "Bazarr authentication failed during correlation retry",
                    http_status=media_info.http_status or 401,
                )
                return result

            if media_info.status == BazarrCorrelationStatus.TEMPORARY_ERROR:
                result = BazarrResult(
                    code=BazarrResultCode.TEMPORARY_ERROR,
                    language=lang_norm,
                    detail=media_info.error_message or "Bazarr temporary error during correlation retry",
                    http_status=media_info.http_status or 500,
                )
                return result

            if not media_info.is_indexed:
                eff_source = event_source
                if not eff_source and job_id:
                    from app.core.db import get_job_by_id
                    _j = get_job_by_id(job_id)
                    if _j:
                        eff_source = _j.get("event_source")

                is_arr_or_retry = (
                    (eff_source and str(eff_source).upper() in ("SONARR", "RADARR", "WEBHOOK", "ARR", "RETRY"))
                    or bool(radarr_id or sonarr_series_id or sonarr_episode_id)
                )

                if is_arr_or_retry:
                    result = BazarrResult(
                        code=BazarrResultCode.WAITING_FOR_MEDIA,
                        language=lang_norm,
                        detail=f"Video path not yet indexed in Bazarr (WAITING_FOR_MEDIA): {video_path}",
                        http_status=404,
                    )
                    return result
                else:
                    result = BazarrResult(
                        code=BazarrResultCode.MEDIA_NOT_FOUND,
                        language=lang_norm,
                        detail=f"Video path not yet indexed in Bazarr: {video_path}",
                        http_status=404,
                    )
                    return result

            # Send Search Trigger to Bazarr
            clean_url = bazarr_url.rstrip("/")
            headers = {"X-API-KEY": bazarr_api_key} if bazarr_api_key else {}
            headers["Content-Type"] = "application/json"

            try:
                bazarr_lang = get_bazarr_language_code(target_lang, default=lang_norm)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    if media_info.media_type == "movie" and media_info.radarr_id:
                        patch_url = f"{clean_url}/api/movies/subtitles"
                        patch_params = {
                            "radarrid": media_info.radarr_id,
                            "language": bazarr_lang,
                            "forced": "False",
                            "hi": "False",
                        }
                    elif media_info.media_type == "episode" and media_info.sonarr_series_id and media_info.sonarr_episode_id:
                        patch_url = f"{clean_url}/api/episodes/subtitles"
                        patch_params = {
                            "seriesid": media_info.sonarr_series_id,
                            "episodeid": media_info.sonarr_episode_id,
                            "language": bazarr_lang,
                            "forced": "False",
                            "hi": "False",
                        }
                    else:
                        result = BazarrResult(
                            code=BazarrResultCode.MEDIA_NOT_FOUND,
                            language=lang_norm,
                            detail="Missing Radarr/Sonarr ID for Bazarr search trigger",
                            http_status=404,
                        )
                        return result

                    res = await client.patch(patch_url, headers=headers, params=patch_params)

                    if res.status_code in (200, 201, 204):
                        op = BazarrOperation(
                            op_key=op_key,
                            video_path=video_path,
                            target_lang=lang_norm,
                            media_info=media_info,
                            state=BazarrLifecycleState.SEARCHING,
                            is_search_triggered=True,
                            trigger_time=time.monotonic(),
                        )
                        async with self._lock:
                            self._operations[op_key] = op
                        result = BazarrResult(
                            code=BazarrResultCode.TRIGGERED,
                            language=lang_norm,
                            detail="accepted",
                            http_status=res.status_code,
                            media_correlated=True,
                        )
                        return result
                    elif res.status_code in (401, 403):
                        result = BazarrResult(
                            code=BazarrResultCode.AUTH_ERROR,
                            language=lang_norm,
                            detail=f"HTTP {res.status_code} Auth Error",
                            http_status=res.status_code,
                        )
                        return result
                    elif res.status_code == 404:
                        result = BazarrResult(
                            code=BazarrResultCode.MEDIA_NOT_FOUND,
                            language=lang_norm,
                            detail=f"HTTP 404 Media Not Found",
                            http_status=res.status_code,
                        )
                        return result
                    elif res.status_code == 409:
                        result = BazarrResult(
                            code=BazarrResultCode.CONFLICT,
                            language=lang_norm,
                            detail=f"HTTP 409 Conflict in Bazarr",
                            http_status=res.status_code,
                        )
                        return result
                    elif 400 <= res.status_code < 500:
                        result = BazarrResult(
                            code=BazarrResultCode.CLIENT_ERROR,
                            language=lang_norm,
                            detail=f"HTTP {res.status_code} Client Error",
                            http_status=res.status_code,
                        )
                        return result
                    else:
                        result = BazarrResult(
                            code=BazarrResultCode.TEMPORARY_ERROR,
                            language=lang_norm,
                            detail=f"HTTP {res.status_code} Server Error",
                            http_status=res.status_code,
                        )
                        return result
            except httpx.TimeoutException:
                result = BazarrResult(code=BazarrResultCode.TEMPORARY_ERROR, language=lang_norm, detail="Request timeout")
                return result
            except Exception as e:
                result = BazarrResult(code=BazarrResultCode.TEMPORARY_ERROR, language=lang_norm, detail=str(e))
                return result
        except Exception as e:
            result = BazarrResult(code=BazarrResultCode.TEMPORARY_ERROR, language=lang_norm, detail=str(e))
            return result
        finally:
            if is_owner:
                async with self._lock:
                    self._in_flight_triggers.pop(op_key, None)
                    if owner_fut and not owner_fut.done():
                        if result is not None:
                            owner_fut.set_result(result)
                        else:
                            owner_fut.set_result(
                                BazarrResult(
                                    code=BazarrResultCode.TEMPORARY_ERROR,
                                    language=lang_norm,
                                    detail="Trigger operation cancelled or aborted",
                                )
                            )

    # ── Authoritative Target Lifecycle Coordinator ──────────────────────────

    async def coordinate_target(
        self,
        video_path: str,
        target_lang: str,
        bazarr_url: str,
        bazarr_api_key: str,
        max_wait_seconds: float = DEFAULT_HYBRID_BAZARR_MAX_WAIT_SEC,
        candidate_stability_sec: float = DEFAULT_CANDIDATE_STABILITY_SEC,
        quiescence_sec: float = DEFAULT_BAZARR_QUIESCENCE_SEC,
        container_tracks: Optional[Dict[str, Any]] = None,
        primary_audio_lang: Optional[str] = None,
        provided_source: Optional[SubtitleSource] = None,
        job_id: Optional[int] = None,
        auto_repair: bool = True,
        find_external_subtitle_fn=None,
        pre_trigger_snapshot: Optional[TargetSnapshot] = None,
        search_accepted: bool = False,
        media_correlated: bool = False,
    ) -> Tuple[BazarrLifecycleState, Optional[str], Optional[TrustResult]]:
        """
        Coordinates the complete lifecycle of a Bazarr target download:
        1. Polls Bazarr jobs every ~0.4s to detect active search / sync jobs.
        2. Detects candidate presence on disk.
        3. If candidate exists while search or sync job is active or UNKNOWN, marks it PROVISIONAL.
        4. When search and sync finish and state is KNOWN_IDLE, performs final file stability & generation evaluation.
        5. Executes SubtitleTrustEngine against exact final generation.
        6. Returns (FinalState, CandidatePath, TrustResult).
        """
        lang_norm = normalize_language_code(target_lang, default=target_lang)
        op_key = self._get_op_key(video_path, lang_norm)
        find_fn = find_external_subtitle_fn or find_external_subtitle
        trust_engine = SubtitleTrustEngine()

        t_start = time.monotonic()
        last_logged_state: Optional[str] = None

        def _log(msg: str):
            if job_id:
                from app.core.db import append_job_log
                append_job_log(job_id, msg)
            logger.info(msg)

        evaluated_gen: Optional[str] = None
        last_trust_res: Optional[TrustResult] = None
        last_gen_change_time: float = t_start
        last_seen_gen: Optional[str] = None
        last_poll_status: BazarrJobPollStatus = BazarrJobPollStatus.UNKNOWN

        async with httpx.AsyncClient(timeout=3.0) as http_client:
            while (time.monotonic() - t_start) < max_wait_seconds:
                now = time.monotonic()

                # 1. Query Bazarr's real jobs queue
                raw_poll = await self.poll_system_jobs(bazarr_url, bazarr_api_key, client=http_client)
                poll_res = _normalize_poll_result(raw_poll)
                last_poll_status = poll_res.status

                candidate_p = find_fn(video_path, lang_norm)
                candidate_snap = capture_target_snapshot(candidate_p) if candidate_p else None

                # Detect generation change
                if candidate_snap and candidate_snap.exists:
                    if last_seen_gen != candidate_snap.generation_id:
                        last_seen_gen = candidate_snap.generation_id
                        last_gen_change_time = now
                        evaluated_gen = None
                        last_trust_res = None

                # --- UNKNOWN lifecycle state handling ---
                if poll_res.status == BazarrJobPollStatus.UNKNOWN:
                    state_log = f"Bazarr lifecycle: UNKNOWN ({poll_res.error or 'transient_error'})"
                    if last_logged_state != state_log:
                        _log(state_log)
                        last_logged_state = state_log
                    if not (candidate_snap and candidate_snap.exists):
                        return BazarrLifecycleState.UNKNOWN, None, None
                    # If candidate exists on disk but Bazarr status is UNKNOWN, inspect provisionally
                    stable = await wait_for_file_stability(
                        candidate_p,
                        min_stability_sec=candidate_stability_sec,
                        timeout_sec=0.2,
                        interval_sec=0.02
                    )
                    if stable:
                        _tres = await trust_engine.evaluate_candidate(
                            video_path=video_path,
                            candidate_path=candidate_p,
                            target_lang=lang_norm,
                            origin=CandidateOrigin.EXTERNAL,
                            container_tracks=container_tracks,
                            primary_audio_lang=primary_audio_lang,
                            provided_source=provided_source,
                            job_id=job_id,
                            auto_repair=auto_repair,
                            allow_ai_audit=True,
                        )
                        return BazarrLifecycleState.UNKNOWN, candidate_p, _tres
                    return BazarrLifecycleState.UNKNOWN, candidate_p, None

                search_jobs, sync_jobs = self.classify_jobs_for_target(
                    poll_res, video_path, lang_norm
                )

                # --- State Determination ---
                if sync_jobs:
                    current_state = BazarrLifecycleState.SYNCING
                    state_log = "Bazarr target sync: RUNNING"
                    if last_logged_state != state_log:
                        _log(state_log)
                        _log("Waiting for Bazarr finalization")
                        last_logged_state = state_log
                    await asyncio.sleep(0.4)
                    continue

                if search_jobs:
                    if candidate_snap and candidate_snap.exists:
                        current_state = BazarrLifecycleState.TARGET_APPEARED
                        state_log = "Bazarr candidate appeared: PROVISIONAL"
                    else:
                        current_state = BazarrLifecycleState.SEARCHING
                        state_log = "Bazarr target search: SEARCHING"

                    if last_logged_state != state_log:
                        _log(state_log)
                        last_logged_state = state_log
                    await asyncio.sleep(0.4)
                    continue

                if candidate_p and candidate_snap and candidate_snap.exists:
                    # Step 1: Stability check on disk
                    stable = await wait_for_file_stability(
                        candidate_p,
                        min_stability_sec=candidate_stability_sec,
                        timeout_sec=min(0.2, max(0.04, max_wait_seconds - (now - t_start))),
                        interval_sec=0.02
                    )
                    if not stable:
                        await asyncio.sleep(0.02)
                        continue

                    # Step 2: Quiescence check (ensure file wasn't just written a few milliseconds ago)
                    time_since_gen_change = now - last_gen_change_time
                    if time_since_gen_change < quiescence_sec:
                        await asyncio.sleep(0.02)
                        continue

                    # Step 3: Evaluate Trust against exact finalized generation
                    cur_snap = capture_target_snapshot(candidate_p)
                    if evaluated_gen != cur_snap.generation_id:
                        b_prov = BazarrProvenance(
                            video_path=video_path,
                            target_lang=lang_norm,
                            search_accepted=search_accepted,
                            pre_trigger_snapshot=pre_trigger_snapshot,
                            is_finalized=True,
                            is_quiescent=True,
                            media_correlated=media_correlated,
                            poll_state=BazarrJobPollStatus.KNOWN_IDLE,
                            candidate_snapshot=cur_snap,
                        )
                        _tres = await trust_engine.evaluate_candidate(
                            video_path=video_path,
                            candidate_path=candidate_p,
                            target_lang=lang_norm,
                            origin=CandidateOrigin.BAZARR,
                            container_tracks=container_tracks,
                            primary_audio_lang=primary_audio_lang,
                            provided_source=provided_source,
                            job_id=job_id,
                            auto_repair=auto_repair,
                            allow_ai_audit=True,
                            bazarr_provenance=b_prov,
                        )

                        # Verify generation did not mutate during Trust evaluation (unless modified by safe auto-repair)
                        post_eval_snap = capture_target_snapshot(candidate_p)
                        if _tres.repaired_path is not None or post_eval_snap.generation_id == cur_snap.generation_id:
                            evaluated_gen = post_eval_snap.generation_id
                            last_trust_res = _tres
                        else:
                            # Mutated during evaluation — loop again
                            last_seen_gen = post_eval_snap.generation_id
                            last_gen_change_time = time.monotonic()
                            evaluated_gen = None
                            last_trust_res = None
                            continue

                    if last_trust_res and last_trust_res.passed:
                        _log(f"Bazarr coordination: finalized target verified (PASS, score={last_trust_res.score}/100)")
                        return (BazarrLifecycleState.FINALIZED_WITH_TARGET, candidate_p, last_trust_res)
                    elif last_trust_res:
                        if time_since_gen_change >= quiescence_sec:
                            _log(f"Bazarr coordination: finalized target rejected by Trust Engine ({last_trust_res.decision.value}: {'; '.join(last_trust_res.reasons)})")
                            return (BazarrLifecycleState.FINALIZED_WITH_TARGET, candidate_p, last_trust_res)

                else:
                    # No candidate file on disk and no search/sync jobs active
                    if (now - t_start) >= min(0.25, max_wait_seconds):
                        _log("Bazarr target search: completed (no target subtitle found)")
                        return (BazarrLifecycleState.FINALIZED_NO_TARGET, None, None)

                await asyncio.sleep(0.04)

        # Timeout reached
        candidate_p = find_fn(video_path, lang_norm)
        if candidate_p and os.path.exists(candidate_p):
            if last_poll_status == BazarrJobPollStatus.UNKNOWN:
                return (BazarrLifecycleState.TIMED_OUT, candidate_p, None)

            cur_snap = capture_target_snapshot(candidate_p)
            b_prov = BazarrProvenance(
                video_path=video_path,
                target_lang=lang_norm,
                search_accepted=search_accepted,
                pre_trigger_snapshot=pre_trigger_snapshot,
                is_finalized=False,
                is_quiescent=False,
                media_correlated=media_correlated,
                poll_state=last_poll_status,
                candidate_snapshot=cur_snap,
            )
            _tres = await trust_engine.evaluate_candidate(
                video_path=video_path,
                candidate_path=candidate_p,
                target_lang=lang_norm,
                origin=CandidateOrigin.BAZARR,
                container_tracks=container_tracks,
                primary_audio_lang=primary_audio_lang,
                provided_source=provided_source,
                job_id=job_id,
                auto_repair=auto_repair,
                allow_ai_audit=True,
                bazarr_provenance=b_prov,
            )
            return (BazarrLifecycleState.FINALIZED_WITH_TARGET, candidate_p, _tres)

        return (BazarrLifecycleState.TIMED_OUT, None, None)

    # ── Publication Ownership Gate ──────────────────────────────────────────

    async def acquire_publication_ownership(
        self,
        video_path: str,
        target_lang: str,
        bazarr_url: str,
        bazarr_api_key: str,
        container_tracks: Optional[Dict[str, Any]] = None,
        primary_audio_lang: Optional[str] = None,
        provided_source: Optional[SubtitleSource] = None,
        job_id: Optional[int] = None,
        timeout_sec: float = 5.0,
        find_external_subtitle_fn=None,
        # Current-run provenance context: supplied by pipeline when Babel triggered Bazarr
        # in this job. Must be None/False if Bazarr was not triggered by this run.
        pre_trigger_snapshot: Optional[TargetSnapshot] = None,
        search_accepted: bool = False,
        media_correlated: bool = False,
        media_info: Optional[BazarrMediaInfo] = None,
    ) -> PublicationOwnershipResult:
        """
        Publication Ownership Hard Invariant Gate:
        Babel MUST NEVER replace/publish movie.<target>.srt while a correlated
        Bazarr job may still write/sync that target or while lifecycle state is UNKNOWN.

        1. Queries current Bazarr jobs.
        2. If lifecycle is UNKNOWN, defers publication (does NOT clobber).
        3. Ensures no correlated target search or sync is active.
        4. If active, waits for completion up to timeout_sec.
        5. If Bazarr writes a new healthy target during wait:
           - If current-run provenance is available (search_accepted + media_correlated +
             pre_trigger_snapshot + KNOWN_IDLE + file quiescent + generation proven new),
             constructs truthful BazarrProvenance so Trust Engine can attempt safe
             global-offset repair (LOW_COVERAGE path) before deciding.
           - Adopts the candidate if Trust returns PASS or PASS_WITH_WARNINGS.
        6. If still actively writing after timeout, denies ownership (defer=True).
        7. If quiescent and verified, grants ownership (granted=True).

        SAFETY INVARIANTS (never weakened by this change):
        - pre_trigger_snapshot must have been captured BEFORE the Bazarr trigger this run.
        - is_quiescent is only synthesised as True after independently confirming file
          stability AND the quiescence window via wait_for_candidate_quiescence().
        - Pre-existing targets whose generation matches pre_trigger_snapshot NEVER receive
          strong Bazarr provenance (enforced by BazarrProvenance.is_strong_current_run()).
        - UNKNOWN lifecycle NEVER becomes KNOWN_IDLE (poll_state is from a real poll).
        - ACTIVE/SYNCING Bazarr lifecycle NEVER transitions to ownership here; the loop
          continues until KNOWN_IDLE is confirmed or timeout expires.
        """
        lang_norm = normalize_language_code(target_lang, default=target_lang)
        find_fn = find_external_subtitle_fn or find_external_subtitle
        trust_engine = SubtitleTrustEngine()

        # Import quiescence helper — required for late-path provenance
        from app.core.trust_engine import (
            wait_for_candidate_quiescence,
            DEFAULT_BAZARR_QUIESCENCE_SEC,
        )

        t_start = time.monotonic()
        last_status = BazarrJobPollStatus.UNKNOWN
        search_jobs = []
        sync_jobs = []

        async with httpx.AsyncClient(timeout=3.0) as http_client:
            while (time.monotonic() - t_start) < timeout_sec:
                raw_poll = await self.poll_system_jobs(bazarr_url, bazarr_api_key, client=http_client)
                poll_res = _normalize_poll_result(raw_poll)
                last_status = poll_res.status

                if poll_res.status == BazarrJobPollStatus.UNKNOWN:
                    if job_id:
                        from app.core.db import append_job_log
                        append_job_log(job_id, f"Publication ownership: Bazarr lifecycle is UNKNOWN ({poll_res.error}). Deferring publication.")
                    await asyncio.sleep(0.4)
                    continue

                has_work, search_jobs, sync_jobs = self.target_has_correlated_active_work(
                    poll_res, video_path, lang_norm, media_info=media_info
                )

                if has_work:
                    if job_id:
                        from app.core.db import append_job_log
                        job_desc = sync_jobs[0].job_name if sync_jobs else search_jobs[0].job_name
                        append_job_log(job_id, f"Publication ownership: Bazarr active job in progress ({job_desc}). Waiting for finalization...")
                    await asyncio.sleep(0.4)
                    continue

                # ── Target-Authoritatively Idle: no correlated or ambiguous Bazarr jobs active ──
                # Check if a candidate appeared on disk.
                candidate_p = find_fn(video_path, lang_norm)
                proven_prov: Optional[BazarrProvenance] = None
                proven_snap: Optional[TargetSnapshot] = None
                if candidate_p and os.path.exists(candidate_p):
                    stable = await wait_for_file_stability(candidate_p, timeout_sec=0.5, interval_sec=0.03)
                    if stable:
                        # ── Try to construct truthful current-run BazarrProvenance ──
                        # Only possible when Babel triggered Bazarr this run AND:
                        #   • search was explicitly accepted by Bazarr API
                        #   • media was correlated in Bazarr's DB
                        #   • pre_trigger_snapshot was captured before the trigger
                        #   • current candidate generation is proven new vs pre-trigger
                        # BazarrProvenance.is_strong_current_run() enforces all invariants
                        # including the generation-changed check.
                        b_prov: Optional[BazarrProvenance] = None
                        if (
                            search_accepted
                            and media_correlated
                            and pre_trigger_snapshot is not None
                        ):
                            # Establish real quiescence: file generation must remain
                            # unchanged for the full quiescence window. This is the same
                            # quiescence proof used by coordinate_target() and is
                            # NEVER synthesised without actually observing it.
                            remaining_sec = max(0.5, timeout_sec - (time.monotonic() - t_start))
                            quiescence_sec = float(get_setting(
                                "bazarr_quiescence_seconds",
                                get_setting("bazarr_quiescence_sec", str(DEFAULT_BAZARR_QUIESCENCE_SEC))
                            ))
                            is_quiescent, q_snap = await wait_for_candidate_quiescence(
                                candidate_p,
                                quiescence_sec=quiescence_sec,
                                timeout_sec=min(remaining_sec, quiescence_sec + 0.5),
                                interval_sec=0.025,
                            )
                            if is_quiescent and q_snap.exists and q_snap.size > 0:
                                # Re-poll once more after quiescence to confirm still target-idle
                                re_poll = await self.poll_system_jobs(bazarr_url, bazarr_api_key, client=http_client)
                                re_poll_res = _normalize_poll_result(re_poll)
                                re_idle, _, _, _ = self.evaluate_target_idle_status(
                                    re_poll_res, video_path, lang_norm, search_accepted=search_accepted
                                )
                                if re_idle and re_poll_res.status != BazarrJobPollStatus.UNKNOWN:
                                    b_prov = BazarrProvenance(
                                        video_path=video_path,
                                        target_lang=lang_norm,
                                        search_accepted=True,
                                        pre_trigger_snapshot=pre_trigger_snapshot,
                                        is_finalized=True,
                                        is_quiescent=True,  # proven above by wait_for_candidate_quiescence
                                        media_correlated=True,
                                        poll_state=BazarrJobPollStatus.KNOWN_IDLE,
                                        candidate_snapshot=q_snap,
                                    )
                                    proven_prov = b_prov
                                    proven_snap = q_snap
                                    if job_id:
                                        from app.core.db import append_job_log
                                        append_job_log(
                                            job_id,
                                            "Publication ownership: current-run Bazarr provenance established "
                                            "(target-authoritatively idle + quiescent + generation proven new). "
                                            "Evaluating with full Trust Engine (auto_repair=True)."
                                        )
                                else:
                                    # Target is not idle upon re-poll (e.g. active jobs or ambiguous jobs detected)
                                    if job_id:
                                        from app.core.db import append_job_log
                                        append_job_log(job_id, "Publication ownership: active Bazarr work detected upon re-poll. Continuing wait...")
                                    await asyncio.sleep(0.4)
                                    continue

                        tres = await trust_engine.evaluate_candidate(
                            video_path=video_path,
                            candidate_path=candidate_p,
                            target_lang=lang_norm,
                            origin=CandidateOrigin.BAZARR,
                            container_tracks=container_tracks,
                            primary_audio_lang=primary_audio_lang,
                            provided_source=provided_source,
                            job_id=job_id,
                            auto_repair=True,
                            allow_ai_audit=False,
                            bazarr_provenance=b_prov,
                        )
                        if tres.passed:
                            if job_id:
                                from app.core.db import append_job_log
                                repair_note = ""
                                if tres.repair and tres.repair.get("applied_shift_sec") is not None:
                                    repair_note = f" (offset-repaired {tres.repair['applied_shift_sec']:+.2f}s)"
                                append_job_log(
                                    job_id,
                                    f"Publication ownership: Bazarr final target verified healthy "
                                    f"(score={tres.score}/100{repair_note}). Preserving human subtitle."
                                )
                            return PublicationOwnershipResult(
                                granted=False,
                                reason="bazarr_target_passed",
                                adopted=True,
                                trust_result=tres,
                                proven_bazarr_provenance=proven_prov,
                                proven_candidate_snapshot=proven_snap,
                            )

                # Quiescent and verified (no active Bazarr worker, no healthy target adopted)
                return PublicationOwnershipResult(
                    granted=True,
                    reason="quiescent_and_verified",
                    proven_bazarr_provenance=proven_prov,
                    proven_candidate_snapshot=proven_snap,
                )

        # Timeout reached and Bazarr is still active or unknown
        if last_status == BazarrJobPollStatus.UNKNOWN:
            if job_id:
                from app.core.db import append_job_log
                append_job_log(job_id, "Publication ownership: Bazarr lifecycle remained UNKNOWN past timeout. Defensively deferring publication.")
            return PublicationOwnershipResult(
                granted=False,
                reason="bazarr_lifecycle_unknown",
                defer=True,
            )

        if job_id:
            from app.core.db import append_job_log
            append_job_log(job_id, "Publication ownership: Bazarr still actively writing after timeout. Refusing to clobber active worker.")
        return PublicationOwnershipResult(
            granted=False,
            reason="bazarr_actively_writing",
            defer=True,
        )


bazarr_coordinator = BazarrCoordinator()
