"""Unit tests for model routing.

Policy: every task — including intent recognition — goes to the DeepSeek Flash
cloud model (``settings.llm_model``). Cost-circuit or a missing cloud key falls
back to the local model.
"""

import pytest

from core.model_router import select_model
from core.settings import settings


@pytest.fixture
def cloud_ready(monkeypatch):
    """Ensure a DeepSeek key + local model are configured for deterministic routing."""
    monkeypatch.setattr(settings, "deepseek_api_key", "sk-test", raising=False)
    monkeypatch.setattr(settings, "local_llm_enabled", True, raising=False)
    monkeypatch.setattr(settings, "local_llm_model", "qwen2.5-7b-instruct", raising=False)
    monkeypatch.setattr(settings, "llm_model", "deepseek-v4-flash", raising=False)
    return settings


def test_intent_task_uses_cloud_model(cloud_ready):
    assert select_model(role="user", task_type="intent") == settings.llm_model


def test_chat_uses_cloud_model(cloud_ready):
    assert select_model(role="guest", task_type="chat") == settings.llm_model


def test_planning_uses_cloud_model(cloud_ready):
    assert select_model(role="user", task_type="planning") == settings.llm_model


def test_itinerary_uses_cloud_model(cloud_ready):
    assert select_model(role="premium", task_type="itinerary") == settings.llm_model


def test_repair_uses_cloud_model(cloud_ready):
    assert select_model(role="premium", task_type="repair") == settings.llm_model


def test_polish_uses_cloud_model(cloud_ready):
    assert select_model(role="user", task_type="polish") == settings.llm_model


def test_cost_circuit_falls_back_to_local(cloud_ready):
    model = select_model(role="user", task_type="planning", cost_circuit_active=True)
    assert model == settings.local_llm_model


def test_missing_cloud_key_falls_back_to_local(monkeypatch):
    monkeypatch.setattr(settings, "deepseek_api_key", "", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "", raising=False)
    monkeypatch.setattr(settings, "local_llm_enabled", True, raising=False)
    monkeypatch.setattr(settings, "local_llm_model", "qwen2.5-7b-instruct", raising=False)
    assert select_model(role="user", task_type="planning") == settings.local_llm_model
