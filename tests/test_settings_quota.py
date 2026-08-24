import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.db import get_setting, set_setting
from app.core.quota import block_provider, unblock_provider

@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    db_file = str(tmp_path / "babel_settings_quota_test.db")
    import app.core.db as db_module
    import app.core.quota as quota_module

    original_db = db_module.DB_PATH
    db_module.DB_PATH = db_file
    quota_module.DB_PATH = db_file

    db_module.init_db()

    yield db_file

    db_module.DB_PATH = original_db
    quota_module.DB_PATH = original_db

@pytest.fixture
def client():
    return TestClient(app)

def test_daily_budget_saving_and_loading(client):
    response = client.post(
        "/api/settings/ai",
        json={
            "daily_request_budget_gemini": 1500,
            "daily_request_budget_openai": 500,
            "daily_request_budget_deepl": 0,
            "daily_request_budget_ollama": 100
        }
    )
    assert response.status_code == 200, response.text
    
    response = client.get("/api/settings/all")
    assert response.status_code == 200
    data = response.json()
    assert data["ai"]["daily_request_budget_gemini"] == 1500
    assert data["ai"]["daily_request_budget_openai"] == 500
    assert data["ai"]["daily_request_budget_deepl"] == 0
    assert data["ai"]["daily_request_budget_ollama"] == 100

def test_quota_endpoint(client):
    block_provider("gemini", "TEST_REASON")
    try:
        response = client.get("/api/quota")
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert "gemini" in data["providers"]
        assert data["providers"]["gemini"]["blocked"] is True
        assert data["providers"]["gemini"]["reason"] == "TEST_REASON"
    finally:
        unblock_provider("gemini")

def test_quota_unblock_endpoint(client):
    block_provider("gemini", "TEST_REASON_2")
    
    response = client.post("/api/quota/gemini/unblock")
    assert response.status_code == 200
    
    response = client.get("/api/quota")
    data = response.json()
    assert data["providers"]["gemini"]["blocked"] is False
    assert data["providers"]["gemini"]["reason"] is None
