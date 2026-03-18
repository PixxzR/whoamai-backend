def test_models_info_endpoint(client):
    response = client.get("/models/info")
    assert response.status_code == 200
    data = response.json()
    assert "strategies" in data
    assert "specialized" in data["strategies"]
    assert "multitask" in data["strategies"]
    assert "transfer" in data["strategies"]
    assert "demo_mode" in data
    assert "metrics" in data


def test_models_info_metrics_structure(client):
    response = client.get("/models/info")
    data = response.json()
    for metric in data["metrics"]:
        assert "strategy" in metric
        assert metric["strategy"] in ["specialized", "multitask", "transfer"]
