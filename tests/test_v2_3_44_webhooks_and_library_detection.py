import os
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.core import db
from app.core.languages import get_language, normalize_language_code
from app.services.scanner import (
    scan_library_folders,
    is_qualifying_embedded_subtitle_track,
    _EMBEDDED_TRACKS_CACHE
)

client = TestClient(app)


# ==============================================================================
# SECTION 1: WEBHOOK DYNAMICS & SECURITY (NO RAW SECRET EXPOSURE)
# ==============================================================================

def test_api_settings_all_does_not_expose_raw_webhook_secret(monkeypatch):
    """Test that /api/settings/all does NOT expose raw webhook_secret, only a safe boolean."""
    secret_value = "super_secret_webhook_key_xyz987"
    monkeypatch.setattr("app.api.dashboard.get_setting", lambda k, d="": secret_value if k == "webhook_secret" else d)

    res = client.get("/api/settings/all")
    assert res.status_code == 200

    raw_response_text = res.text
    assert secret_value not in raw_response_text, "Raw webhook secret was leaked in /api/settings/all response!"

    data = res.json()
    assert "integrations" in data
    assert "webhook_secret" not in data["integrations"]
    assert data["integrations"].get("webhook_secret_configured") is True


def test_api_settings_all_webhook_secret_configured_false_when_empty(monkeypatch):
    """Test that /api/settings/all reports webhook_secret_configured=False when secret is empty."""
    monkeypatch.setattr("app.api.dashboard.get_setting", lambda k, d="": "" if k == "webhook_secret" else d)

    res = client.get("/api/settings/all")
    assert res.status_code == 200
    data = res.json()
    assert data["integrations"].get("webhook_secret_configured") is False


def test_index_html_webhook_ui_elements():
    """Verify index.html contains dynamic webhook functions, copy triggers, reachability notes, and correct trigger text without embedding secrets."""
    template_path = os.path.join(os.path.dirname(__file__), "..", "app", "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Dynamic bindings
    assert ":value=\"getWebhookUrl('sonarr')\"" in html
    assert ":value=\"getWebhookUrl('radarr')\"" in html
    assert "@click=\"copyWebhook('sonarr')\"" in html
    assert "@click=\"copyWebhook('radarr')\"" in html
    assert "getDockerWebhookUrl('sonarr')" in html
    assert "getDockerWebhookUrl('radarr')" in html

    # Reachability & Docker notes
    assert "Use an address that Sonarr can reach." in html
    assert "Use an address that Radarr can reach." in html
    assert "Same Docker network only:" in html

    # Servarr Setup Instructions
    assert "On File Import" in html
    assert "On File Upgrade" in html
    assert "On Download" not in html
    assert "On Grab" not in html

    # JS Implementation uses origin without appending stored secret
    assert "getWebhookUrl(type)" in html
    assert "getDockerWebhookUrl(type)" in html
    assert "window.location.origin" in html
    assert "http://babel:8765/webhook/" in html


def test_readme_servarr_instructions():
    """Verify README.md contains correct triggers and warns against On Grab."""
    readme_path = os.path.join(os.path.dirname(__file__), "..", "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        readme = f.read()

    assert "On File Import" in readme
    assert "On File Upgrade" in readme
    assert "On Download" not in readme
    assert "do not use on grab" in readme.lower() or "not use on grab" in readme.lower()


def test_backend_webhook_auth_mechanisms(monkeypatch):
    """Verify Sonarr/Radarr webhooks accept ?secret=..., X-Webhook-Secret, and Bearer headers when configured."""
    secret = "test_guard_secret_123"
    monkeypatch.setattr("app.core.db.get_setting", lambda k, d="": secret if k == "webhook_secret" else d)

    # 1. Reject without auth
    res_no_auth = client.post("/webhook/sonarr", json={"eventType": "Test"})
    assert res_no_auth.status_code == 401

    # 2. Allow with query parameter (?secret=...)
    res_query = client.post(f"/webhook/sonarr?secret={secret}", json={"eventType": "Test"})
    assert res_query.status_code == 200

    # 3. Allow with X-Webhook-Secret header
    res_header = client.post("/webhook/sonarr", headers={"X-Webhook-Secret": secret}, json={"eventType": "Test"})
    assert res_header.status_code == 200

    # 4. Allow with Bearer token
    res_bearer = client.post("/webhook/radarr", headers={"Authorization": f"Bearer {secret}"}, json={"eventType": "Test"})
    assert res_bearer.status_code == 200

    # 5. Reject invalid secret
    res_bad = client.post("/webhook/radarr", headers={"X-Webhook-Secret": "wrong_secret"}, json={"eventType": "Test"})
    assert res_bad.status_code == 401


# ==============================================================================
# SECTION 2: LANGUAGE NORMALIZATION WITH SUBTAGS
# ==============================================================================

def test_language_subtag_normalization():
    """Verify that region / script subtags map to the correct canonical language."""
    assert normalize_language_code("sv-SE") == "sv"
    assert normalize_language_code("sv_SE") == "sv"
    assert normalize_language_code("swe") == "sv"
    assert normalize_language_code("sve") == "sv"
    assert normalize_language_code("Swedish") == "sv"
    assert normalize_language_code("pt-BR") == "pt-BR"
    assert normalize_language_code("pt_BR") == "pt-BR"
    assert normalize_language_code("pt-PT") == "pt"
    assert normalize_language_code("pt_PT") == "pt"
    assert normalize_language_code("sr-Latn") == "sr"
    assert normalize_language_code("zh-CN") == "zh"
    assert normalize_language_code("zh-TW") == "zh"

    # get_language returns matching Language object
    sv_lang = get_language("sv-SE")
    assert sv_lang is not None
    assert sv_lang.code == "sv"

    pt_br_lang = get_language("pt-BR")
    assert pt_br_lang is not None
    assert pt_br_lang.code == "pt-BR"

    pt_pt_lang = get_language("pt-PT")
    assert pt_pt_lang is not None
    assert pt_pt_lang.code == "pt"


# ==============================================================================
# SECTION 3: EMBEDDED SUBTITLE TRACK QUALIFICATION
# ==============================================================================

def test_is_qualifying_embedded_subtitle_track():
    """Test qualification logic for various embedded tracks."""
    target_aliases = ["sv", "swe", "swedish"]

    # 1. Valid Swedish text track
    valid_track = {
        "id": 1,
        "language": "swe",
        "codec": "SubRip/SRT",
        "forced": False,
        "default": True,
        "title": "Swedish (SDH)"
    }
    assert is_qualifying_embedded_subtitle_track(valid_track, target_aliases) is True

    # 2. Valid Swedish with region code
    valid_region_track = {
        "id": 2,
        "language": "sv-SE",
        "codec": "S_TEXT/ASS",
        "forced": False,
        "default": False,
        "title": ""
    }
    assert is_qualifying_embedded_subtitle_track(valid_region_track, target_aliases) is True

    # 3. Non-target language (English)
    en_track = {
        "id": 3,
        "language": "eng",
        "codec": "SubRip/SRT",
        "forced": False,
        "default": True,
        "title": "English"
    }
    assert is_qualifying_embedded_subtitle_track(en_track, target_aliases) is False

    # 4. Forced track (should be rejected for full translation completeness)
    forced_track = {
        "id": 4,
        "language": "swe",
        "codec": "SubRip/SRT",
        "forced": True,
        "default": False,
        "title": "Swedish Forced"
    }
    assert is_qualifying_embedded_subtitle_track(forced_track, target_aliases) is False

    # 5. Commentary / Director track (should be rejected)
    commentary_track = {
        "id": 5,
        "language": "swe",
        "codec": "SubRip/SRT",
        "forced": False,
        "default": False,
        "title": "Director's Commentary"
    }
    assert is_qualifying_embedded_subtitle_track(commentary_track, target_aliases) is False

    # 6. Non-text codec (e.g. PGS / VobSub bitmap subtitles)
    bitmap_track = {
        "id": 6,
        "language": "swe",
        "codec": "HDMV_PGS_SUBTITLE",
        "forced": False,
        "default": True,
        "title": "Swedish PGS"
    }
    assert is_qualifying_embedded_subtitle_track(bitmap_track, target_aliases) is False

    # 7. Undefined language with Swedish in title
    und_title_track = {
        "id": 7,
        "language": "und",
        "codec": "SubRip/SRT",
        "forced": False,
        "default": False,
        "title": "Swedish Full"
    }
    assert is_qualifying_embedded_subtitle_track(und_title_track, target_aliases) is True


# ==============================================================================
# SECTION 4: SCANNER LIBRARY DETECTION (EXTERNAL, EMBEDDED, BOTH, MISSING)
# ==============================================================================

def test_scanner_detects_embedded_and_external_target_subtitles(tmp_path, monkeypatch):
    """
    Verify full matrix:
    - External only => has_target_sub=True, target_sub_source='external'
    - Embedded only => has_target_sub=True, target_sub_source='embedded'
    - Both present  => has_target_sub=True, target_sub_source='both'
    - Neither       => has_target_sub=False, target_sub_source=None
    """
    _EMBEDDED_TRACKS_CACHE.clear()
    monkeypatch.setattr("app.services.scanner._get_target_lang_codes", lambda: ["sv"])

    movies_dir = tmp_path / "movies"
    movies_dir.mkdir()

    # 1. External only
    m_ext = movies_dir / "MovieExt.mkv"
    m_ext.write_bytes(b"dummy mkv content 1")
    s_ext = movies_dir / "MovieExt.sv.srt"
    s_ext.write_text("1\n00:00:01,000 --> 00:00:02,000\nHej\n", encoding="utf-8")

    # 2. Embedded only
    m_emb = movies_dir / "MovieEmb.mkv"
    m_emb.write_bytes(b"dummy mkv content 2")

    # 3. Both external & embedded (with pre-existing embedded cache)
    m_both = movies_dir / "MovieBoth.mkv"
    m_both.write_bytes(b"dummy mkv content 3")
    s_both = movies_dir / "MovieBoth.sv.srt"
    s_both.write_text("1\n00:00:01,000 --> 00:00:02,000\nHej\n", encoding="utf-8")
    st_both = os.stat(str(m_both))
    from app.core.db import set_cached_embedded_subtitle_tracks
    set_cached_embedded_subtitle_tracks(
        str(m_both),
        int(st_both.st_size),
        getattr(st_both, "st_mtime_ns", int(st_both.st_mtime * 1e9)),
        [{"id": 0, "language": "swe", "codec": "SubRip/SRT", "forced": False, "title": "Swedish"}]
    )

    # 4. Neither (only English embedded)
    m_none = movies_dir / "MovieNone.mkv"
    m_none.write_bytes(b"dummy mkv content 4")

    # Mock inspect_mkv_tracks
    def mock_inspect(path):
        if "MovieEmb" in path or "MovieBoth" in path:
            return {
                "subtitles": [
                    {"id": 0, "language": "swe", "codec": "SubRip/SRT", "forced": False, "title": "Swedish"}
                ],
                "audio": []
            }
        elif "MovieNone" in path:
            return {
                "subtitles": [
                    {"id": 0, "language": "eng", "codec": "SubRip/SRT", "forced": False, "title": "English"}
                ],
                "audio": []
            }
        return {"subtitles": [], "audio": []}

    monkeypatch.setattr("app.core.extractor.inspect_mkv_tracks", mock_inspect)

    from app.services.scanner import embedded_prober
    scan_library_folders(str(movies_dir), category="movies")
    embedded_prober.wait_completion(timeout=5.0)

    res = scan_library_folders(str(movies_dir), category="movies")
    assert len(res) == 4

    items = {r["filename"]: r for r in res}

    # Verify External only
    assert items["MovieExt.mkv"]["has_target_sub"] is True
    assert items["MovieExt.mkv"]["has_embedded_target"] is False
    assert items["MovieExt.mkv"]["target_sub_source"] == "external"

    # Verify Embedded only
    assert items["MovieEmb.mkv"]["has_target_sub"] is True
    assert items["MovieEmb.mkv"]["has_embedded_target"] is True
    assert items["MovieEmb.mkv"]["target_sub_source"] == "embedded"

    # Verify Both
    assert items["MovieBoth.mkv"]["has_target_sub"] is True
    assert items["MovieBoth.mkv"]["has_embedded_target"] is True
    assert items["MovieBoth.mkv"]["target_sub_source"] == "both"

    # Verify Neither
    assert items["MovieNone.mkv"]["has_target_sub"] is False
    assert items["MovieNone.mkv"]["has_embedded_target"] is False
    assert items["MovieNone.mkv"]["target_sub_source"] is None


def test_scanner_handles_probe_failure_gracefully(tmp_path, monkeypatch):
    """Corrupted files or inspect_mkv_tracks exceptions must not crash the scan or break external sub detection."""
    _EMBEDDED_TRACKS_CACHE.clear()
    monkeypatch.setattr("app.services.scanner._get_target_lang_codes", lambda: ["sv"])

    movies_dir = tmp_path / "movies_err"
    movies_dir.mkdir()

    m1 = movies_dir / "Broken.mkv"
    m1.write_bytes(b"bad data")
    s1 = movies_dir / "Broken.sv.srt"
    s1.write_text("1\n00:00:01,000 --> 00:00:02,000\nHej\n", encoding="utf-8")

    m2 = movies_dir / "BrokenNoSub.mp4"
    m2.write_bytes(b"bad data 2")

    def mock_inspect_fail(path):
        raise RuntimeError("ffprobe crashed")

    monkeypatch.setattr("app.core.extractor.inspect_mkv_tracks", mock_inspect_fail)

    res = scan_library_folders(str(movies_dir), category="movies")
    assert len(res) == 2

    items = {r["filename"]: r for r in res}
    # External sub should still be detected
    assert items["Broken.mkv"]["has_target_sub"] is True
    assert items["Broken.mkv"]["target_sub_source"] == "external"

    # Missing sub remains missing without crashing
    assert items["BrokenNoSub.mp4"]["has_target_sub"] is False
    assert items["BrokenNoSub.mp4"]["target_sub_source"] is None

    from app.services.scanner import embedded_prober
    embedded_prober.wait_completion(timeout=5.0)


def test_scanner_embedded_tracks_cache(tmp_path, monkeypatch):
    """Verify that _EMBEDDED_TRACKS_CACHE avoids redundant probes and invalidates on mtime/size changes."""
    from app.services.scanner import embedded_prober
    embedded_prober.wait_completion(timeout=5.0)
    _EMBEDDED_TRACKS_CACHE.clear()
    monkeypatch.setattr("app.services.scanner._get_target_lang_codes", lambda: ["sv"])

    movies_dir = tmp_path / "cache_test"
    movies_dir.mkdir()

    vid = movies_dir / "CacheMovie.mkv"
    vid.write_bytes(b"initial content")

    probe_count = 0

    def mock_inspect(path):
        nonlocal probe_count
        probe_count += 1
        return {
            "subtitles": [
                {"id": 0, "language": "swe", "codec": "SubRip/SRT", "forced": False, "title": "Swedish"}
            ],
            "audio": []
        }

    monkeypatch.setattr("app.core.extractor.inspect_mkv_tracks", mock_inspect)

    from app.services.scanner import embedded_prober

    # First scan: enqueues background probe
    scan_library_folders(str(movies_dir), category="movies")
    embedded_prober.wait_completion(timeout=5.0)

    res1 = scan_library_folders(str(movies_dir), category="movies")
    assert probe_count == 1
    assert res1[0]["has_target_sub"] is True

    # Second scan with unchanged file: cache hit, no extra probe
    res2 = scan_library_folders(str(movies_dir), category="movies")
    assert probe_count == 1
    assert res2[0]["has_target_sub"] is True

    # Modify file: size/mtime change triggers re-probe
    vid.write_bytes(b"modified longer content")
    scan_library_folders(str(movies_dir), category="movies")
    embedded_prober.wait_completion(timeout=5.0)

    res3 = scan_library_folders(str(movies_dir), category="movies")
    assert probe_count == 2
    assert res3[0]["has_target_sub"] is True
