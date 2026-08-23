import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
import httpx

from app.main import app
from app.core.db import get_setting, set_setting
from app.core.security import mask_secret, is_masked_secret, resolve_secret_key
from app.services.bazarr_controller import bazarr_controller

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture(autouse=True)
def clean_settings():
    old_bazarr = get_setting("bazarr_api_key", "")
    old_gemini = get_setting("gemini_api_key", "")
    old_openai = get_setting("openai_api_key", "")
    old_deepl = get_setting("deepl_api_key", "")
    yield
    set_setting("bazarr_api_key", old_bazarr)
    set_setting("gemini_api_key", old_gemini)
    set_setting("openai_api_key", old_openai)
    set_setting("deepl_api_key", old_deepl)

def test_1_saved_real_bazarr_key_with_masked_incoming_key(client, monkeypatch):
    """TEST 1: Saved real Bazarr key + masked incoming key -> endpoint uses REAL DB key."""
    set_setting("bazarr_api_key", "test_real_secret_key_12345")
    
    with patch("app.services.bazarr_controller.BazarrController.get_status", new_callable=AsyncMock) as mock_get_status:
        mock_get_status.return_value = {"connected": True, "version": "v1.4.3"}
        
        resp = client.post("/api/settings/test-bazarr", json={
            "bazarr_url": "http://bazarr:6767",
            "bazarr_api_key": "••••••••2345"
        })
        
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "version": "v1.4.3"}
        mock_get_status.assert_called_once_with("http://bazarr:6767", "test_real_secret_key_12345")

def test_2_new_unmasked_bazarr_key_used(client):
    """TEST 2: New unmasked Bazarr key sent -> endpoint uses the new incoming key."""
    set_setting("bazarr_api_key", "test_old_stored_key")
    
    with patch("app.services.bazarr_controller.BazarrController.get_status", new_callable=AsyncMock) as mock_get_status:
        mock_get_status.return_value = {"connected": True, "version": "v1.4.3"}
        
        resp = client.post("/api/settings/test-bazarr", json={
            "bazarr_url": "http://bazarr:6767",
            "bazarr_api_key": "test_brand_new_key_9999"
        })
        
        assert resp.status_code == 200
        mock_get_status.assert_called_once_with("http://bazarr:6767", "test_brand_new_key_9999")

def test_3_masked_key_with_no_saved_db_key_fails_safely(client):
    """TEST 3: Masked key + no saved DB key -> clear configuration failure, no request sent."""
    set_setting("bazarr_api_key", "")
    
    with patch("app.services.bazarr_controller.BazarrController.get_status", new_callable=AsyncMock) as mock_get_status:
        resp = client.post("/api/settings/test-bazarr", json={
            "bazarr_url": "http://bazarr:6767",
            "bazarr_api_key": "••••••••0000"
        })
        
        assert resp.status_code == 400
        assert "No Bazarr API Key" in resp.json()["detail"]
        mock_get_status.assert_not_called()

def test_4_save_masked_key_preserves_stored_key(client):
    """TEST 4: Save masked key -> stored real key is NOT overwritten."""
    set_setting("bazarr_api_key", "test_original_secret_key")
    
    resp = client.post("/api/settings/integrations", json={
        "enable_bazarr_check": True,
        "bazarr_url": "http://bazarr:6767",
        "bazarr_api_key": "••••••••_key",
        "bazarr_container_name": "bazarr",
        "wait_time_seconds": 15,
        "notify_jellyfin": True,
        "jellyfin_url": "http://jellyfin:8096",
        "jellyfin_api_key": "••••••••_jelly"
    })
    
    assert resp.status_code == 200
    assert get_setting("bazarr_api_key", "") == "test_original_secret_key"

def test_5_save_new_real_key_updates_db(client):
    """TEST 5: Save new real key -> DB is updated."""
    set_setting("bazarr_api_key", "test_original_secret_key")
    
    resp = client.post("/api/settings/integrations", json={
        "enable_bazarr_check": True,
        "bazarr_url": "http://bazarr:6767",
        "bazarr_api_key": "test_newly_typed_key_5678",
        "bazarr_container_name": "bazarr",
        "wait_time_seconds": 15,
        "notify_jellyfin": True,
        "jellyfin_url": "http://jellyfin:8096",
        "jellyfin_api_key": "••••••••_jelly"
    })
    
    assert resp.status_code == 200
    assert get_setting("bazarr_api_key", "") == "test_newly_typed_key_5678"

def test_6_settings_response_masks_keys(client):
    """TEST 6: Settings response -> returns masked key, never plaintext."""
    set_setting("bazarr_api_key", "test_very_secret_bazarr_key_1234")
    set_setting("gemini_api_key", "test_very_secret_gemini_key_5678")
    
    resp = client.get("/api/settings/all")
    assert resp.status_code == 200
    data = resp.json()
    
    bazarr_masked = data["integrations"]["bazarr_api_key"]
    gemini_masked = data["ai"]["gemini_api_key"]
    
    assert bazarr_masked == "••••••••1234"
    assert gemini_masked == "••••••••5678"
    assert "test_very_secret" not in resp.text

@pytest.mark.asyncio
async def test_7_bazarr_controller_get_status_success():
    """TEST 7: Bazarr success -> reports connected and version."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = httpx.Response(200, json={"version": "1.4.3.0"})
        
        status = await bazarr_controller.get_status("http://bazarr:6767", "test_key")
        assert status["connected"] is True
        assert status["version"] == "v1.4.3.0"

@pytest.mark.asyncio
async def test_8_bazarr_controller_get_status_invalid_api_key():
    """TEST 8: Bazarr 401/403 -> safe 'Invalid API Key' error."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = httpx.Response(401)
        status_401 = await bazarr_controller.get_status("http://bazarr:6767", "wrong_key")
        assert status_401["connected"] is False
        assert status_401["message"] == "Invalid API Key"

        mock_get.return_value = httpx.Response(403)
        status_403 = await bazarr_controller.get_status("http://bazarr:6767", "wrong_key")
        assert status_403["connected"] is False
        assert status_403["message"] == "Invalid API Key"

@pytest.mark.asyncio
async def test_9_bazarr_controller_get_status_unreachable_and_timeout():
    """TEST 9: Bazarr unreachable/timeout -> safe specific errors."""
    with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("Connection refused")):
        status_conn = await bazarr_controller.get_status("http://invalid_host:6767", "test_key")
        assert status_conn["connected"] is False
        assert status_conn["message"] == "Unable to reach Bazarr"
        
    with patch("httpx.AsyncClient.get", side_effect=httpx.TimeoutException("Timeout")):
        status_timeout = await bazarr_controller.get_status("http://timeout_host:6767", "test_key")
        assert status_timeout["connected"] is False
        assert status_timeout["message"] == "Connection timed out"

def test_10_ai_secret_helpers_and_resolution():
    """TEST 10: Secret masking, checking, and resolution helpers behave consistently."""
    assert mask_secret("short") == "short"
    assert mask_secret("12345678") == "••••••••5678"
    assert mask_secret("my_super_long_secret_key_abcd") == "••••••••abcd"
    
    assert is_masked_secret("••••••••abcd") is True
    assert is_masked_secret("my_secret_key") is False
    assert is_masked_secret("") is False
    assert is_masked_secret(None) is False
    
    set_setting("gemini_api_key", "stored_gemini_key_12345")
    assert resolve_secret_key("••••••••2345", "gemini_api_key") == "stored_gemini_key_12345"
    assert resolve_secret_key("", "gemini_api_key") == "stored_gemini_key_12345"
    assert resolve_secret_key("new_gemini_key_9999", "gemini_api_key") == "new_gemini_key_9999"
