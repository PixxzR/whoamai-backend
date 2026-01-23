from app.config import settings


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == settings.app_version
    assert "models_loaded" in data
    assert "demo_mode" in data


def test_health_returns_app_name(client):
    response = client.get("/health")
    data = response.json()
    assert data["app_name"] == settings.app_name
