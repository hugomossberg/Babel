"""Bounded model listing, separate from inference and saved model configuration.

Live means listed by the endpoint, not proof of inference access or compatibility.
Offline suggestions are never presented as installed local models.
"""
import asyncio
import hashlib
import ipaddress
import time
import weakref
from collections import OrderedDict
from urllib.parse import urlsplit

import httpx

from app.core.ai_providers import (
    filter_anthropic_models, filter_custom_models, filter_gemini_models,
    filter_ollama_models, filter_openai_models, filter_openrouter_models,
    get_default_model, get_provider_catalog,
)

MODELS_CACHE = OrderedDict()
_LOCKS = weakref.WeakKeyDictionary()
_MAX_PAGES = 20
_MAX_MODELS = 20000
_MAX_CACHE = 128


def validate_endpoint(endpoint):
    """Allow configured LAN/local services; forbid URL secrets and metadata IPs.

    Redirects are disabled separately so an endpoint cannot forward auth headers.
    Private addresses and Docker hostnames remain supported for local providers.
    """
    if not isinstance(endpoint, str) or any(c.isspace() or ord(c) < 32 for c in endpoint) or '\\' in endpoint:
        raise ValueError("Invalid endpoint")
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or '?' in endpoint or '#' in endpoint):
        raise ValueError("Invalid endpoint")
    # Accessing port validates malformed/out-of-range ports as well.
    parsed.port
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        if parsed.hostname.lower() in ("metadata.google.internal", "metadata.goog"):
            raise ValueError("Invalid endpoint")
    else:
        if address.is_link_local or address.is_multicast or address.is_unspecified:
            raise ValueError("Invalid endpoint")
    return endpoint.rstrip('/')


def cache_key(provider, key, endpoint):
    # Hash endpoint too: internal hostnames do not belong in cache diagnostics.
    scope = hashlib.sha256((endpoint + '\0' + key).encode()).hexdigest()
    return f"{provider}:{scope}"


class DiscoveryFailure(Exception):
    def __init__(self, code):
        self.code = code


async def _list_models(provider, key, endpoint):
    if provider == "deepl":
        return get_provider_catalog(provider)
    if provider not in ("ollama", "custom", "openrouter") and not key:
        raise DiscoveryFailure("missing_credentials")
    urls = {
        "gemini": "https://generativelanguage.googleapis.com/v1beta/models",
        "openai": "https://api.openai.com/v1/models",
        "anthropic": "https://api.anthropic.com/v1/models",
        "openrouter": "https://openrouter.ai/api/v1/models",
        "deepseek": "https://api.deepseek.com/models",
        "ollama": endpoint + "/api/tags",
        "custom": endpoint + "/models",
    }
    headers = {}
    if provider == "gemini":
        headers = {"x-goog-api-key": key}
    elif provider == "anthropic":
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    elif provider == "custom":
        # Match Babel's existing OpenAI-compatible client contract for local APIs.
        headers = {"Authorization": f"Bearer {key or 'dummy'}"}
    elif key:
        headers = {"Authorization": f"Bearer {key}"}
    field = "models" if provider in ("gemini", "ollama") else "data"
    filters = {
        "gemini": filter_gemini_models, "openai": filter_openai_models,
        "anthropic": filter_anthropic_models, "openrouter": filter_openrouter_models,
        "ollama": filter_ollama_models, "custom": filter_custom_models,
        "deepseek": filter_custom_models,
    }
    raw, cursors, params = [], set(), {}
    # A timeout per request plus an overall deadline bounds pathological pagination.
    async with asyncio.timeout(20):
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
            for _ in range(_MAX_PAGES):
                res = await client.get(urls[provider], headers=headers, params=params)
                if res.status_code in (401, 403):
                    raise DiscoveryFailure("authentication")
                if res.status_code != 200:
                    raise DiscoveryFailure("unavailable")
                try:
                    data = res.json()
                except ValueError:
                    raise DiscoveryFailure("invalid_response") from None
                if not isinstance(data, dict) or not isinstance(data.get(field), list):
                    raise DiscoveryFailure("invalid_response")
                page = data[field]
                if any(not isinstance(item, dict) for item in page):
                    raise DiscoveryFailure("invalid_response")
                for item in page:
                    for name in ("id", "name", "display_name", "displayName"):
                        if name in item and not isinstance(item[name], str):
                            raise DiscoveryFailure("invalid_response")
                raw.extend(page)
                if len(raw) > _MAX_MODELS:
                    raise DiscoveryFailure("invalid_response")
                cursor = None
                if provider == "gemini":
                    cursor = data.get("nextPageToken")
                    params = {"pageToken": cursor}
                elif provider == "anthropic" and data.get("has_more"):
                    cursor = data.get("last_id")
                    if not cursor:
                        raise DiscoveryFailure("invalid_response")
                    params = {"after_id": cursor}
                if not cursor:
                    return filters[provider](raw)
                if not isinstance(cursor, str) or cursor in cursors or len(cursor) > 4096:
                    raise DiscoveryFailure("invalid_response")
                cursors.add(cursor)
            raise DiscoveryFailure("invalid_response")


def _response(provider, entry, cached=False):
    result = dict(entry)
    result.update(provider=provider, default_model=get_default_model(provider), cached=cached)
    if cached and result["source"] == "live":
        result["source"] = "cache"
    return result


async def discover_models(provider, key="", endpoint="", refresh=False):
    if provider in ("custom", "ollama"):
        endpoint = validate_endpoint(endpoint)
    scope = cache_key(provider, key, endpoint)
    loop_locks = _LOCKS.setdefault(asyncio.get_running_loop(), weakref.WeakValueDictionary())
    lock = loop_locks.setdefault(scope, asyncio.Lock())
    observed = MODELS_CACHE.get(scope)
    async with lock:
        now = time.time()
        previous = MODELS_CACHE.get(scope)
        ttl = 60 if provider in ("custom", "ollama") else 3600
        # Concurrent refreshes share the result produced after they started.
        if previous and ((not refresh and now < previous[0]) or previous is not observed):
            return _response(provider, previous[1], cached=True)
        error = None
        try:
            models = await _list_models(provider, key, endpoint)
            completed_at = time.time()
            entry = {"models": models, "source": "configuration" if provider == "deepl" else "live",
                     "stale": False, "error": None, "fetched_at": completed_at,
                     "listing_verified": provider != "deepl"}
            expires = completed_at + ttl
        except (DiscoveryFailure, httpx.HTTPError, TimeoutError, ValueError, TypeError) as exc:
            if isinstance(exc, DiscoveryFailure):
                error = exc.code
            elif isinstance(exc, (httpx.TimeoutException, TimeoutError)):
                error = "timeout"
            elif isinstance(exc, (ValueError, TypeError)):
                error = "invalid_response"
            else:
                error = "unavailable"
            if previous and previous[1]["listing_verified"]:
                entry = dict(previous[1], source="cache", stale=True, error=error)
            else:
                entry = {"models": [] if provider in ("custom", "ollama") else get_provider_catalog(provider),
                         "source": "fallback", "stale": False, "error": error,
                         "fetched_at": None, "listing_verified": False}
            # Short negative cache avoids repeated failed API requests.
            expires = time.time() + 15
        MODELS_CACHE[scope] = (expires, entry)
        MODELS_CACHE.move_to_end(scope)
        while len(MODELS_CACHE) > _MAX_CACHE:
            MODELS_CACHE.popitem(last=False)
        return _response(provider, entry, cached=entry["source"] == "cache")
