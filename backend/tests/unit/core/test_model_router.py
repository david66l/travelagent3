"""Unit tests for model routing."""

from core.model_router import select_model
from core.settings import settings


def test_guest_uses_small_model():
    model = select_model(role="guest", task_type="chat")
    assert model == settings.small_model


def test_cost_circuit_forces_small_model():
    model = select_model(
        role="user",
        task_type="planning",
        cost_circuit_active=True,
    )
    assert model == settings.small_model


def test_intent_task_uses_small_model():
    model = select_model(role="user", task_type="intent")
    assert model == settings.small_model


def test_premium_allows_large_model_for_planning():
    model = select_model(role="premium", task_type="planning")
    assert model == settings.default_model


def test_itinerary_task_uses_large_model():
    model = select_model(role="premium", task_type="itinerary")
    assert model == settings.default_model


def test_repair_task_uses_repair_model():
    model = select_model(role="premium", task_type="repair")
    assert model == settings.repair_model


def test_polish_task_uses_large_model():
    model = select_model(role="user", task_type="polish")
    assert model == settings.default_model
