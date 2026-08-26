"""
Source Resolver for Babel v2.3.43-beta.

Provides:
- SubtitleSource: explicit representation of a resolved source subtitle
- BazarrResult: structured result from Bazarr API calls (not fire-and-forget)
- BazarrResultCode: enumeration of possible Bazarr outcomes
- SourceResolver: target-first, source-flexible subtitle resolution with
  deadline-based Bazarr polling and proper fallback ordering

Design invariants:
  1. A healthy target subtitle always wins — no AI needed.
  2. source_language == target_language → AI is NEVER dispatched.
  3. Audio language is a prioritisation signal only; it NEVER blocks a job.
  4. Bazarr target search and source preparation run concurrently.
  5. All source candidates are validated before use.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import srt

from app.core.languages import normalize_language_code, get_language, LANGUAGES
from app.core.validator import detect_language_heuristics, evaluate_subtitle_health
from app.services.bazarr_checker import find_external_subtitle

logger = logging.getLogger("babel.source_resolver")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ordered list of fallback source languages to try from Bazarr when no local
# source is found.  Keeps a single authoritative definition (see §21 of spec).
BAZARR_SOURCE_FALLBACK_ORDER: List[str] = [
    "en", "es", "fr", "de", "pt", "it", "ja", "zh", "ko",
    "ru", "nl", "pl", "sv", "da", "no", "fi",
]

# Minimum byte-size of an SRT file to be considered non-trivial
_MIN_SRT_BYTES = 20

# Minimum number of cues for a source to be useful
# Set to 1 to accommodate test fixtures and very short scenes.
# Quality validation is done by the QA gate, not here.
_MIN_CUE_COUNT = 1

# Forced/sparse subtitle detection thresholds (cue-density and absolute count).
#
# A true forced/sign-only subtitle has very few cues spread over a long film
# (e.g. 40 cues for a 42-minute film ≈ 0.95 cues/min).
# A full dialogue subtitle — including SDH tracks with many [MUSIC] / (laughs) markers —
# has 5–15+ cues per minute of coverage.
#
# We detect sparse coverage via:
#   1. Absolute cue count: < _SPARSE_CUE_COUNT_THRESHOLD hard-rejects clearly tiny subtitles
#   2. Cue density: cues per minute of subtitle span; very low = forced/incomplete
#      Only applied when we have enough timing data (span ≥ _SPARSE_MIN_SPAN_SECONDS).
#
# SDH/noise-marker ratio (e.g. [MUSIC], [APPLAUSE]) is NOT used for hard rejection.
# A legitimate SDH track with 900 cues and 30% sound-effect markers is a COMPLETE source.
# Embedded track forced-flag is handled earlier (extract_embedded_srt skips forced tracks).
_SPARSE_CUE_DENSITY_MIN  = 2.0   # cues/minute; below this with long span → sparse/forced
_SPARSE_MIN_SPAN_SECONDS = 60.0  # only apply density check when subtitle spans ≥ 60 seconds
                                  # Short-span subtitles (scenes, test fixtures) always pass
_SPARSE_COVERAGE_MIN     = 0.25  # subtitle span / video_duration < this → low_coverage
                                  # Requires video_duration to be passed (from container_tracks)
_SPARSE_DENSITY_FULL_SAVE = 0.50 # if coverage ≥ this fraction, don't reject on density alone
                                  # Allows legitimate full-coverage low-density (art/dialogue-sparse films)

# Regex patterns retained for potential quality-signal / cleaner use, but NOT for hard-reject.
_RE_SDH_ONLY   = re.compile(r"^\s*[\(\[][^\)\]]+[\)\]]\s*$")  # e.g. (laughs), [door slams]
_RE_MUSIC_ONLY = re.compile(r"^\s*[♪♬#]+\s*$")               # music-note-only lines


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class SourceOrigin(str, Enum):
    EMBEDDED = "embedded"
    EXTERNAL = "external"
    BAZARR   = "bazarr"


@dataclass
class SubtitleSource:
    """Explicit representation of a resolved source subtitle."""
    language: str          # ISO 639-1 canonical code, e.g. "en"
    origin: SourceOrigin
    path: str              # Absolute path on disk
    content: str           # Raw SRT text
    cues: List[Any]        # Parsed srt.Subtitle objects
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def language_name(self) -> str:
        lang_obj = get_language(self.language)
        return lang_obj.display_name if lang_obj else self.language.upper()


class BazarrResultCode(str, Enum):
    TRIGGERED       = "TRIGGERED"       # Search accepted / download queued
    ACCEPTED        = "ACCEPTED"        # Alias for TRIGGERED
    MEDIA_NOT_FOUND = "MEDIA_NOT_FOUND" # Path could not be matched in Bazarr
    AUTH_ERROR      = "AUTH_ERROR"      # 401 / 403 — bad API key / forbidden
    CONFIG_ERROR    = "CONFIG_ERROR"    # Bad URL, missing key, other config error
    TEMPORARY_ERROR = "TEMPORARY_ERROR" # 5xx / network timeout / transient
    TIMEOUT         = "TIMEOUT"         # Local poll deadline exceeded
    DISABLED        = "DISABLED"        # Bazarr integration not enabled
    NOOP            = "NOOP"            # Already had the subtitle — no action taken


@dataclass
class BazarrResult:
    """Structured result from a Bazarr search trigger."""
    code: BazarrResultCode
    language: str = ""
    detail: str = ""
    http_status: Optional[int] = None

    @property
    def is_transient(self) -> bool:
        return self.code in (BazarrResultCode.TEMPORARY_ERROR, BazarrResultCode.TIMEOUT)

    @property
    def is_permanent(self) -> bool:
        return self.code in (BazarrResultCode.AUTH_ERROR, BazarrResultCode.CONFIG_ERROR,
                             BazarrResultCode.MEDIA_NOT_FOUND)

    @property
    def was_accepted(self) -> bool:
        return self.code in (BazarrResultCode.TRIGGERED, BazarrResultCode.ACCEPTED)


# ---------------------------------------------------------------------------
# Source validation
# ---------------------------------------------------------------------------

def _read_file_safe(path: str) -> Optional[str]:
    """Read a file with UTF-8 fallback to windows-1252."""
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="windows-1252") as f:
                return f.read()
        except Exception:
            return None
    except Exception:
        return None


def _validate_source_candidate(
    path: str,
    declared_language: str,
    video_duration_seconds: Optional[float] = None,
) -> Tuple[bool, str, Optional[str], List[Any]]:
    """
    Validate a source subtitle candidate.

    Args:
        path:                    Absolute path to the SRT file.
        declared_language:       ISO 639-1 code the subtitle is expected to be in.
        video_duration_seconds:  Total video duration in seconds (from container_tracks).
                                 When provided, enables coverage-ratio check which catches
                                 clustered forced subtitles that have high local density.

    Returns:
        (ok: bool, reason: str, actual_language: str | None, cues: list)

    Invariants checked:
    - File is readable
    - File size > minimum
    - Parseable as SRT
    - Contains enough cues (>= _MIN_CUE_COUNT)
    - Not sparse/forced (via coverage ratio and/or cue density)
    - Actual language is reasonably compatible with declared language
    """
    if not os.path.exists(path):
        return False, "file_not_found", None, []

    try:
        size = os.path.getsize(path)
    except OSError:
        return False, "file_unreadable", None, []

    if size < _MIN_SRT_BYTES:
        return False, f"too_small ({size} bytes)", None, []

    content = _read_file_safe(path)
    if not content:
        return False, "cannot_read", None, []

    try:
        cues = list(srt.parse(content))
    except Exception as e:
        return False, f"parse_error: {e}", None, []

    if len(cues) < _MIN_CUE_COUNT:
        return False, f"too_few_cues ({len(cues)})", None, []

    # Sparse / forced-only detection — two independent signals.
    #
    # Rationale:
    #   - SDH/noise markers ([MUSIC], [APPLAUSE]) are NOT a sign of forced content.
    #     A complete SDH track with 30%+ sound markers is a valid source.
    #   - Embedded forced tracks already filtered upstream (extract_embedded_srt skips them).
    #   - Short-span subtitles (<60s): short scenes and test fixtures pass through.
    #     Their quality is assessed by the QA gate, not here.
    #
    # Signal 1 — COVERAGE (requires video_duration_seconds from container_tracks):
    #   subtitle_span / video_duration < _SPARSE_COVERAGE_MIN → low_coverage.
    #   Catches clustered forced subtitles (40 cues in min 10-14 of a 42-min film)
    #   that have high *local* density but tiny video coverage.
    #
    # Signal 2 — DENSITY (fallback when video_duration is not available):
    #   cues/minute of SRT span < _SPARSE_CUE_DENSITY_MIN.
    #   Guarded by _SPARSE_DENSITY_FULL_SAVE: if coverage >= 50% of video, density
    #   alone cannot reject — a dialogue-sparse but complete film is a valid source.
    real_cues = [c for c in cues if c.content and c.content.strip() and c.content.strip() != "<i></i>"]
    if len(real_cues) < _MIN_CUE_COUNT:
        return False, f"empty ({len(real_cues)} real cues)", None, []

    if cues:
        first_start  = cues[0].start.total_seconds()
        last_end     = cues[-1].end.total_seconds()
        span_seconds = last_end - first_start

        if span_seconds >= _SPARSE_MIN_SPAN_SECONDS:
            video_dur = video_duration_seconds or 0.0
            coverage  = (span_seconds / video_dur) if video_dur > 0 else None

            # Signal 1: coverage — independent of density
            if coverage is not None and coverage < _SPARSE_COVERAGE_MIN:
                return False, (
                    f"low_coverage ({span_seconds/60:.1f}min SRT span "
                    f"= {coverage:.0%} of {video_dur/60:.1f}min video "
                    f"< {_SPARSE_COVERAGE_MIN:.0%} threshold)"
                ), None, []

            # Signal 2: density — not applied if subtitle clearly covers most of the film
            density = len(real_cues) / (span_seconds / 60.0)
            is_full_coverage = coverage is not None and coverage >= _SPARSE_DENSITY_FULL_SAVE
            if density < _SPARSE_CUE_DENSITY_MIN and not is_full_coverage:
                return False, (
                    f"sparse_forced ({len(real_cues)} cues over {span_seconds/60:.1f}min "
                    f"= {density:.1f} cues/min < {_SPARSE_CUE_DENSITY_MIN} threshold, "
                    f"coverage={f'{coverage:.0%}' if coverage is not None else 'unknown'})"
                ), None, []

    # Language validation — sample the first 50 cues
    sample_text = " ".join(c.content for c in cues[:50] if c.content)
    if len(sample_text.strip()) >= 30:
        detected = detect_language_heuristics(sample_text, expected_language=declared_language)
        detected_code = detected.get("lang", "unknown")
        detected_conf = detected.get("confidence", 0.0)

        # If we detect a different language with high confidence, report it
        if (
            detected_code not in ("unknown", "und")
            and detected_conf > 0.85
            and not _languages_compatible(detected_code, declared_language)
        ):
            # Return the detected language as actual_language so the caller
            # can use it as the real source language
            return True, "mislabeled", detected_code, cues

        actual_lang = detected_code if detected_code not in ("unknown", "und") else declared_language
    else:
        actual_lang = declared_language

    return True, "ok", normalize_language_code(actual_lang, default=declared_language), cues


def _languages_compatible(a: str, b: str) -> bool:
    """Return True if two language codes are the same or closely related."""
    a_norm = normalize_language_code(a)
    b_norm = normalize_language_code(b)
    if a_norm == b_norm:
        return True
    # BCS family
    bcs = {"sr", "hr", "bs"}
    if a_norm in bcs and b_norm in bcs:
        return True
    return False


# ---------------------------------------------------------------------------
# Bazarr API
# ---------------------------------------------------------------------------

async def trigger_bazarr_search(
    video_path: str,
    language: str,
    bazarr_url: str,
    bazarr_api_key: str,
    timeout: float = 15.0,
) -> BazarrResult:
    """
    Trigger a Bazarr subtitle search for a specific language.

    Returns a structured BazarrResult — never a bare None or fire-and-forget.

    HTTP status semantics:
      401/403 → AUTH_ERROR (actionable, permanent)
      404     → MEDIA_NOT_FOUND (path not in Bazarr)
      5xx     → TEMPORARY_ERROR (transient)
      200/201 → TRIGGERED
    """
    import httpx
    from pathlib import Path

    if not bazarr_api_key:
        return BazarrResult(
            code=BazarrResultCode.CONFIG_ERROR,
            language=language,
            detail="Bazarr API key not configured",
        )

    headers = {"X-API-KEY": bazarr_api_key}
    lang_norm = normalize_language_code(language, default=language)

    async def _patch(client, url, params) -> BazarrResult:
        try:
            r = await client.patch(url, headers=headers, params=params)
            if r.status_code in (200, 201, 204):
                return BazarrResult(code=BazarrResultCode.TRIGGERED, language=lang_norm,
                                    detail="accepted", http_status=r.status_code)
            elif r.status_code in (401, 403):
                return BazarrResult(code=BazarrResultCode.AUTH_ERROR, language=lang_norm,
                                    detail=f"HTTP {r.status_code}", http_status=r.status_code)
            elif r.status_code == 404:
                return BazarrResult(code=BazarrResultCode.MEDIA_NOT_FOUND, language=lang_norm,
                                    detail=f"HTTP 404", http_status=r.status_code)
            elif r.status_code >= 500:
                return BazarrResult(code=BazarrResultCode.TEMPORARY_ERROR, language=lang_norm,
                                    detail=f"HTTP {r.status_code}", http_status=r.status_code)
            else:
                return BazarrResult(code=BazarrResultCode.TRIGGERED, language=lang_norm,
                                    detail=f"HTTP {r.status_code}", http_status=r.status_code)
        except httpx.TimeoutException:
            return BazarrResult(code=BazarrResultCode.TEMPORARY_ERROR, language=lang_norm,
                                detail="timeout during patch")
        except Exception as e:
            return BazarrResult(code=BazarrResultCode.TEMPORARY_ERROR, language=lang_norm,
                                detail=str(e))

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # --- Movies ---
            try:
                m_res = await client.get(f"{bazarr_url}/api/movies", headers=headers)
            except httpx.TimeoutException:
                return BazarrResult(code=BazarrResultCode.TEMPORARY_ERROR, language=lang_norm,
                                    detail="timeout fetching movies")
            except Exception as e:
                return BazarrResult(code=BazarrResultCode.TEMPORARY_ERROR, language=lang_norm,
                                    detail=str(e))

            if m_res.status_code in (401, 403):
                return BazarrResult(code=BazarrResultCode.AUTH_ERROR, language=lang_norm,
                                    detail=f"HTTP {m_res.status_code} on /api/movies",
                                    http_status=m_res.status_code)
            if m_res.status_code >= 500:
                return BazarrResult(code=BazarrResultCode.TEMPORARY_ERROR, language=lang_norm,
                                    detail=f"HTTP {m_res.status_code} on /api/movies",
                                    http_status=m_res.status_code)

            if m_res.status_code == 200:
                try:
                    m_json = m_res.json()
                    movies = m_json.get("data", []) if isinstance(m_json, dict) else (
                        m_json if isinstance(m_json, list) else [])
                except Exception:
                    movies = []
                norm_target = os.path.normpath(video_path)
                target_parts = norm_target.lstrip("/").split("/")
                for m in movies:
                    if not isinstance(m, dict):
                        continue
                    m_raw = m.get("path", "")
                    if not m_raw:
                        continue
                    m_path = os.path.normpath(m_raw)
                    m_parts = m_path.lstrip("/").split("/")
                    # Match 1: exact normalized path
                    # Match 2: basename only (filename unique within library)
                    # Match 3: path-suffix — last min(3, len) components match
                    #   Handles container vs host prefix differences, e.g.
                    #   Babel: /mnt/movies/MovieTitle/Movie.mkv
                    #   Bazarr: /media/Movies/MovieTitle/Movie.mkv
                    suffix_depth = min(3, len(target_parts), len(m_parts))
                    suffix_match = (
                        suffix_depth >= 1 and
                        target_parts[-suffix_depth:] == m_parts[-suffix_depth:]
                    )
                    if m_path == norm_target or os.path.basename(m_path) == os.path.basename(norm_target) or suffix_match:
                        r_id = m.get("radarrId")
                        if r_id:
                            logger.info(f"Bazarr: triggering movie search radarrId={r_id} lang={lang_norm}")
                            return await _patch(client, f"{bazarr_url}/api/movies/subtitles", {
                                "radarrid": r_id,
                                "language": lang_norm,
                                "forced": "False",
                                "hi": "False",
                            })

            # --- Series / Episodes ---
            try:
                s_res = await client.get(f"{bazarr_url}/api/series", headers=headers)
            except httpx.TimeoutException:
                return BazarrResult(code=BazarrResultCode.TEMPORARY_ERROR, language=lang_norm,
                                    detail="timeout fetching series")
            except Exception as e:
                return BazarrResult(code=BazarrResultCode.TEMPORARY_ERROR, language=lang_norm,
                                    detail=str(e))

            if s_res.status_code in (401, 403):
                return BazarrResult(code=BazarrResultCode.AUTH_ERROR, language=lang_norm,
                                    detail=f"HTTP {s_res.status_code} on /api/series",
                                    http_status=s_res.status_code)
            if s_res.status_code >= 500:
                return BazarrResult(code=BazarrResultCode.TEMPORARY_ERROR, language=lang_norm,
                                    detail=f"HTTP {s_res.status_code} on /api/series",
                                    http_status=s_res.status_code)

            if s_res.status_code == 200:
                try:
                    s_json = s_res.json()
                    all_series = s_json.get("data", []) if isinstance(s_json, dict) else (
                        s_json if isinstance(s_json, list) else [])
                except Exception:
                    all_series = []

                target_p = Path(video_path).resolve()
                for series in all_series:
                    if not isinstance(series, dict):
                        continue
                    raw_series_path = series.get("path", "")
                    if not raw_series_path:
                        continue
                    series_p = Path(raw_series_path).resolve()
                    is_match = False
                    try:
                        is_match = target_p.is_relative_to(series_p)
                    except Exception:
                        is_match = False

                    if not (is_match or series_p.name == target_p.parent.name
                            or series_p.name == target_p.parent.parent.name):
                        continue

                    s_id = series.get("sonarrSeriesId")
                    if not s_id:
                        continue

                    # Get episodes for this series
                    try:
                        ep_res = await client.get(
                            f"{bazarr_url}/api/episodes",
                            headers=headers,
                            params={"seriesid[]": s_id},
                        )
                    except httpx.TimeoutException:
                        return BazarrResult(code=BazarrResultCode.TEMPORARY_ERROR, language=lang_norm,
                                            detail="timeout fetching episodes")
                    except Exception as e:
                        return BazarrResult(code=BazarrResultCode.TEMPORARY_ERROR, language=lang_norm,
                                            detail=str(e))

                    if ep_res.status_code in (401, 403):
                        return BazarrResult(code=BazarrResultCode.AUTH_ERROR, language=lang_norm,
                                            detail=f"HTTP {ep_res.status_code} on /api/episodes",
                                            http_status=ep_res.status_code)
                    if ep_res.status_code >= 500:
                        return BazarrResult(code=BazarrResultCode.TEMPORARY_ERROR, language=lang_norm,
                                            detail=f"HTTP {ep_res.status_code} on /api/episodes",
                                            http_status=ep_res.status_code)

                    if ep_res.status_code == 200:
                        try:
                            ep_json = ep_res.json()
                            episodes = ep_json.get("data", []) if isinstance(ep_json, dict) else (
                                ep_json if isinstance(ep_json, list) else [])
                        except Exception:
                            episodes = []

                        for ep in episodes:
                            if not isinstance(ep, dict):
                                continue
                            ep_raw_path = ep.get("path", "")
                            if not ep_raw_path:
                                continue
                            ep_p = Path(ep_raw_path).resolve()
                            if ep_p == target_p or ep_p.name == target_p.name:
                                e_id = ep.get("sonarrEpisodeId")
                                if e_id:
                                    logger.info(
                                        f"Bazarr: triggering episode search "
                                        f"series={s_id} episode={e_id} lang={lang_norm}"
                                    )
                                    return await _patch(client, f"{bazarr_url}/api/episodes/subtitles", {
                                        "seriesid": s_id,
                                        "episodeid": e_id,
                                        "language": lang_norm,
                                        "forced": "False",
                                        "hi": "False",
                                    })

                    # Found the right series — if episode not matched, report it
                    logger.warning(
                        f"Bazarr: series matched ({series.get('title', '?')} id={s_id}) "
                        f"but episode not found for path: {video_path}"
                    )
                    return BazarrResult(
                        code=BazarrResultCode.MEDIA_NOT_FOUND,
                        language=lang_norm,
                        detail=f"Series matched (id={s_id}) but episode not found. "
                               f"Video path: {video_path}",
                    )

            # Nothing matched
            return BazarrResult(
                code=BazarrResultCode.MEDIA_NOT_FOUND,
                language=lang_norm,
                detail=f"Video path could not be matched to any Bazarr movie or series: {video_path}",
            )

    except asyncio.CancelledError:
        raise
    except Exception as e:
        return BazarrResult(
            code=BazarrResultCode.TEMPORARY_ERROR,
            language=lang_norm,
            detail=str(e),
        )


# ---------------------------------------------------------------------------
# Source Resolver
# ---------------------------------------------------------------------------

class SourceResolver:
    """
    Resolves the best usable source subtitle for AI translation.

    Priority (local sources first, then Bazarr fallback):
      1. Embedded subtitle matching primary audio language
      2. Embedded subtitle for best-available source language (e.g. English)
      3. External subtitle matching primary audio language
      4. External subtitle for best-available source language
      5. Other local external subtitles by fallback order
      6. Bazarr source fallback (total budget shared across all languages)

    Never returns a source whose language matches the target language
    (that would be a direct use, not a translation source).
    """

    def __init__(
        self,
        video_path: str,
        container_tracks: Optional[Dict[str, Any]],
        primary_audio_lang: str,
        target_languages: List[str],        # list of target ISO codes
        bazarr_url: str,
        bazarr_api_key: str,
        enable_bazarr: bool,
        extract_source_embedded: bool,
        source_search_deadline: float,      # total seconds budget for Bazarr source search
        source_poll_interval: float = 3.0,
        job_id: Optional[int] = None,
        find_external_subtitle_fn=None,     # injectable for testability (defaults to bazarr_checker impl)
    ):
        self.video_path = video_path
        self.container_tracks = container_tracks
        self.primary_audio_lang = normalize_language_code(primary_audio_lang, default="und")
        self.target_languages = [normalize_language_code(t) for t in target_languages]
        self.bazarr_url = bazarr_url
        self.bazarr_api_key = bazarr_api_key
        self.enable_bazarr = enable_bazarr
        self.extract_source_embedded = extract_source_embedded
        self.source_search_deadline = source_search_deadline
        self.source_poll_interval = source_poll_interval
        self.job_id = job_id
        # Cancellation event for terminating running extraction subprocesses
        self.cancel_event = threading.Event()
        # Use caller-provided find function (allows test mocking via pipeline namespace)
        self._find_external_subtitle = find_external_subtitle_fn or find_external_subtitle
        # Video duration from container metadata — enables coverage-ratio check in validation
        self.video_duration_seconds: Optional[float] = (
            float(container_tracks["duration"])
            if container_tracks and container_tracks.get("duration")
            else None
        )

    def cancel(self) -> None:
        """Signal cancellation to terminate any in-flight extraction subprocesses."""
        self.cancel_event.set()

    def _log(self, msg: str) -> None:
        if self.job_id:
            from app.core.db import append_job_log
            append_job_log(self.job_id, msg)
        logger.info(msg)

    def _is_usable_for_any_target(self, lang_code: str) -> bool:
        """Return True if lang_code could be used as source for at least one remaining target."""
        norm = normalize_language_code(lang_code, default=lang_code)
        return not all(_languages_compatible(norm, t) for t in self.target_languages)

    async def resolve(self) -> Optional[SubtitleSource]:
        """
        Resolve the best available source subtitle.

        Returns SubtitleSource or None if nothing usable was found.
        """
        from app.core.extractor import extract_embedded_srt, inspect_mkv_tracks
        from app.services.pipeline import _safe_extract_embedded_srt
        import uuid

        base_path, _ = os.path.splitext(self.video_path)

        # --- Build priority list of candidate languages ---
        candidate_langs = self._build_candidate_language_list()

        # --- PRIORITY 1 & 2: Embedded subtitles ---
        if self.extract_source_embedded and self.container_tracks is not None:
            for lang_code in candidate_langs:
                if not self._is_usable_for_any_target(lang_code):
                    continue

                tmp = f"{base_path}.temp_src.{lang_code}.{uuid.uuid4().hex}.srt"
                keep_tmp = False
                try:
                    self._log(f"Source Resolver: attempting extraction of embedded {lang_code.upper()} track...")
                    def _safe_call_extractor(_vp=self.video_path, _out=tmp, _pl=lang_code, _ti=self.container_tracks, _ce=self.cancel_event):
                        try:
                            return _safe_extract_embedded_srt(_vp, _out, preferred_lang=_pl, tracks_info=_ti, cancel_event=_ce)
                        except TypeError:
                            try:
                                return _safe_extract_embedded_srt(_vp, _out, preferred_lang=_pl, tracks_info=_ti)
                            except TypeError:
                                return _safe_extract_embedded_srt(_vp, _out, preferred_lang=_pl)

                    extracted = await asyncio.to_thread(_safe_call_extractor)
                    if extracted and os.path.exists(tmp):
                        ok, reason, actual_lang, cues = _validate_source_candidate(
                            tmp, lang_code,
                            video_duration_seconds=self.video_duration_seconds,
                        )
                        content = _read_file_safe(tmp)
                        if ok and content:
                            effective_lang = actual_lang or lang_code
                            if reason == "mislabeled":
                                self._log(
                                    f"Source Resolver: embedded track declared as '{lang_code}' "
                                    f"but detected as '{effective_lang}'. Using detected language."
                                )
                            else:
                                self._log(
                                    f"Source Resolver: embedded {effective_lang} source selected "
                                    f"(origin=embedded, {len(cues)} cues)"
                                )
                            keep_tmp = True
                            return SubtitleSource(
                                language=effective_lang,
                                origin=SourceOrigin.EMBEDDED,
                                path=tmp,
                                content=content,
                                cues=cues,
                                metadata={"declared_language": lang_code, "reason": reason},
                            )
                        elif not ok:
                            self._log(
                                f"Source Resolver: embedded {lang_code} track rejected: {reason}"
                            )
                finally:
                    if not keep_tmp and os.path.exists(tmp):
                        try:
                            os.remove(tmp)
                        except Exception:
                            pass

        # --- PRIORITY 3, 4, 5: External subtitles on disk ---
        for lang_code in candidate_langs:
            if not self._is_usable_for_any_target(lang_code):
                continue
            external = self._find_external_subtitle(self.video_path, lang_code)
            if external and os.path.exists(external):
                ok, reason, actual_lang, cues = _validate_source_candidate(
                    external, lang_code,
                    video_duration_seconds=self.video_duration_seconds,
                )
                content = _read_file_safe(external)
                if ok and content:
                    effective_lang = actual_lang or lang_code
                    if reason == "mislabeled":
                        self._log(
                            f"Source Resolver: external '{lang_code}' subtitle detected as "
                            f"'{effective_lang}'. Using detected language."
                        )
                    else:
                        self._log(
                            f"Source Resolver: external {effective_lang} source selected "
                            f"(origin=external, {len(cues)} cues, path={os.path.basename(external)})"
                        )
                    return SubtitleSource(
                        language=effective_lang,
                        origin=SourceOrigin.EXTERNAL,
                        path=external,
                        content=content,
                        cues=cues,
                        metadata={"declared_language": lang_code, "reason": reason},
                    )
                elif not ok:
                    self._log(
                        f"Source Resolver: external {lang_code} subtitle rejected: {reason}"
                    )

        # --- PRIORITY 6: Bazarr source fallback ---
        if not self.enable_bazarr or not self.bazarr_api_key:
            return None

        return await self._resolve_via_bazarr(base_path, candidate_langs)

    async def _resolve_via_bazarr(
        self,
        base_path: str,
        candidate_langs: List[str],
    ) -> Optional[SubtitleSource]:
        """
        Try to get a source subtitle from Bazarr with a shared deadline budget.
        Tries languages in priority order; stops when deadline is reached.
        """
        deadline = time.monotonic() + self.source_search_deadline
        tried_langs: List[str] = []

        # Filter candidate langs to those not yet tried (local search was exhausted)
        # and to those usable for at least one remaining target
        bazarr_langs = [
            l for l in candidate_langs
            if self._is_usable_for_any_target(l)
        ]

        if not bazarr_langs:
            return None

        self._log(
            f"Source Resolver: no local source found. "
            f"Bazarr source fallback budget: {self.source_search_deadline}s "
            f"for languages: {bazarr_langs}"
        )

        for lang_code in bazarr_langs:
            if time.monotonic() >= deadline:
                self._log(
                    f"Source Resolver: Bazarr source search budget exhausted "
                    f"({self.source_search_deadline}s). Tried: {tried_langs}"
                )
                return None

            # Trigger Bazarr search for this language
            result = await trigger_bazarr_search(
                self.video_path, lang_code,
                self.bazarr_url, self.bazarr_api_key,
            )

            if result.code == BazarrResultCode.AUTH_ERROR:
                self._log(
                    f"Source Resolver: Bazarr API rejected request for source '{lang_code}': "
                    f"HTTP auth error ({result.detail}). Check Bazarr API key."
                )
                return None  # Permanent — no point trying other languages

            if result.code == BazarrResultCode.MEDIA_NOT_FOUND:
                self._log(
                    f"Source Resolver: media entity could not be matched in Bazarr for source '{lang_code}'. "
                    f"Video path: {self.video_path}. {result.detail}"
                )
                # Media not found is also permanent per video — stop
                return None

            if result.code == BazarrResultCode.TEMPORARY_ERROR:
                self._log(
                    f"Source Resolver: Bazarr transient error for source '{lang_code}': {result.detail}"
                )
                tried_langs.append(lang_code)
                continue

            if result.was_accepted:
                self._log(
                    f"Source Resolver: Bazarr source search for '{lang_code}' accepted. "
                    f"Polling (budget remaining: {round(deadline - time.monotonic(), 1)}s)..."
                )
                tried_langs.append(lang_code)

                # Poll filesystem until deadline or file appears
                found = None
                while time.monotonic() < deadline:
                    found = self._find_external_subtitle(self.video_path, lang_code)
                    if found:
                        break
                    remaining = deadline - time.monotonic()
                    await asyncio.sleep(min(self.source_poll_interval, max(0.5, remaining)))

                # Final filesystem check
                if not found:
                    found = self._find_external_subtitle(self.video_path, lang_code)

                if found:
                    ok, reason, actual_lang, cues = _validate_source_candidate(
                        found, lang_code,
                        video_duration_seconds=self.video_duration_seconds,
                    )
                    content = _read_file_safe(found)
                    if ok and content:
                        effective_lang = actual_lang or lang_code
                        self._log(
                            f"Source Resolver: Bazarr provided {effective_lang} source "
                            f"({len(cues)} cues, path={os.path.basename(found)})"
                        )
                        return SubtitleSource(
                            language=effective_lang,
                            origin=SourceOrigin.BAZARR,
                            path=found,
                            content=content,
                            cues=cues,
                            metadata={"declared_language": lang_code, "reason": reason},
                        )
                    else:
                        self._log(
                            f"Source Resolver: Bazarr-provided {lang_code} subtitle rejected: {reason}"
                        )
                else:
                    self._log(
                        f"Source Resolver: no usable {lang_code} source found within "
                        f"current source-search budget. Trying next fallback..."
                    )

        return None

    def _build_candidate_language_list(self) -> List[str]:
        """
        Build an ordered, deduplicated list of candidate source languages.

        Order:
          1. Primary audio language (if not a target)
          2. English (if not already listed and not a target)
          3. BAZARR_SOURCE_FALLBACK_ORDER (skip already listed and targets)
        """
        seen: set = set()
        result: List[str] = []

        def _add(lang: str):
            norm = normalize_language_code(lang, default=lang)
            if norm and norm not in seen:
                seen.add(norm)
                result.append(norm)

        # Primary audio first (strong signal)
        if self.primary_audio_lang and self.primary_audio_lang not in ("und", "unknown", ""):
            _add(self.primary_audio_lang)

        # English always a top candidate (most common subtitle source)
        _add("en")

        # Container subtitle track languages (dynamic discovery)
        if self.container_tracks and isinstance(self.container_tracks, dict):
            for trk in self.container_tracks.get("subtitles", []):
                t_lang = trk.get("language")
                if t_lang:
                    _add(t_lang)

        # Rest of fallback order
        for lang in BAZARR_SOURCE_FALLBACK_ORDER:
            _add(lang)

        # Also add all supported languages not yet included
        for lang_obj in LANGUAGES:
            _add(lang_obj.code)

        return result
