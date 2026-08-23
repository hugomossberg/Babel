import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.db import get_setting, set_setting

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture(autouse=True)
def clean_settings():
    old_batch_size = get_setting("batch_size", "50")
    yield
    set_setting("batch_size", old_batch_size)

def test_batch_size_no_upper_bound(client):
    response = client.post(
        "/api/settings/ai",
        json={"batch_size": 400}
    )
    assert response.status_code == 200, response.text
    
    response = client.get("/api/settings/all")
    assert response.status_code == 200
    data = response.json()
    assert data["ai"]["batch_size"] == 400

def test_batch_size_lower_bound(client):
    response = client.post(
        "/api/settings/ai",
        json={"batch_size": 0}
    )
    assert response.status_code == 422
