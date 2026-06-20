"""Tests for VRP solver FastAPI app."""

import pytest
from fastapi.testclient import TestClient

from vrp_solver_service.main import app
from vrp_solver_service.models import ConstraintsInput, POIInput, SolverRequest


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_solve_endpoint_returns_itinerary(client):
    request = SolverRequest(
        pois=[
            POIInput(id="p1", name="故宫", lat=39.9163, lng=116.3972, duration_minutes=180, ticket_price=60.0),
            POIInput(id="p2", name="天坛", lat=39.8830, lng=116.4120, duration_minutes=120, ticket_price=34.0),
        ],
        constraints=ConstraintsInput(travel_days=1, total_budget=5000),
    )
    response = client.post("/solve", json=request.model_dump())
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("optimal", "fallback")
    assert len(data["days"]) == 1
    assert len(data["days"][0]["activities"]) > 0


def test_solve_endpoint_rejects_empty_pois(client):
    response = client.post("/solve", json={"pois": [], "constraints": {"travel_days": 1}})
    assert response.status_code == 422
