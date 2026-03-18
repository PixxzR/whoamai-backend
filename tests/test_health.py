from app.config import settings


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == settings.app_version
    assert "models_loaded" in data
    assert "demo_mode" in data


def test_health_returns_uptime(client):
    response = client.get("/health")
    data = response.json()
    assert "uptime" in data
    assert isinstance(data["uptime"], (int, float))
    assert data["uptime"] >= 0
