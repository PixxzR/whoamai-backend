def test_dashboard_returns_html(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "WhoAmAI Dashboard" in response.text
