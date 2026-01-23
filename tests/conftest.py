import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Client de test FastAPI."""
    with TestClient(app) as c:
        yield c
