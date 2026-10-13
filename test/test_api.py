"""Tests for the HTTP API."""

from fastapi.testclient import TestClient

from lab_in_the_loop.api import app

client = TestClient(app)


def test_health() -> None:
    """Tests the wake-up endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_start_call() -> None:
    """Tests that an empty history gives only the start proposal."""
    response = client.post("/step", json={"seed": 7, "history": []})
    assert response.status_code == 200
    body = response.json()
    assert body["measurement"] is None
    assert body["model"] is None
    assert body["proposal"]["method"] == "start"
    assert len(body["proposal"]["concentrations"]) == 10


def test_iteration() -> None:
    """Tests that one applied row gives a measurement and a model."""
    start = client.post("/step", json={"seed": 7, "history": []}).json()
    history = [start["proposal"]["concentrations"]]
    body = client.post("/step", json={"seed": 7, "history": history}).json()
    assert len(body["measurement"]["phenotype"]) == 3
    assert body["model"]["n_experiments"] == 1


def test_rejects_wrong_number_of_compounds() -> None:
    """Tests that a history row must have one value per compound."""
    response = client.post("/step", json={"seed": 7, "history": [[0.5] * 9]})
    assert response.status_code == 422


def test_cors_allows_local_page() -> None:
    """Tests that the local development page may call the API."""
    response = client.get("/health", headers={"Origin": "http://localhost:8000"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:8000"
