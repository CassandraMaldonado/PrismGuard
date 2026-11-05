from fastapi.testclient import TestClient
from prism_guard.api import app

client = TestClient(app)

def test_brief_endpoint():
    assert client.get("/api/health").json() == {"status": "ok"}
    response = client.get("/api/brief?seed=41&samples=600")
    assert response.status_code == 200
    assert response.json()["experiment"]["samples"] == 600


def test_sample_validation():
    assert client.get("/api/brief?samples=50").status_code == 422
