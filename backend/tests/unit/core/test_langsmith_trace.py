"""Tests for LangSmith trace helpers."""

from core.langsmith_trace import langsmith_enabled, traceable_step


def test_traceable_step_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setattr("core.langsmith_trace._langsmith_traceable", None)

    @traceable_step("intent/test_step", run_type="chain")
    def sample(x: int) -> int:
        return x + 1

    assert sample(1) == 2
    assert langsmith_enabled() is False
