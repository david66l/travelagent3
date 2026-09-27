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


def test_traceable_step_stays_disabled_when_key_exists_but_tracing_is_off(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")

    assert langsmith_enabled() is False


def test_traceable_step_wraps_only_when_tracing_is_explicitly_enabled(monkeypatch):
    calls = []

    def fake_traceable(*, name, run_type):
        calls.append((name, run_type))

        def decorate(fn):
            return fn

        return decorate

    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr("core.langsmith_trace._langsmith_traceable", fake_traceable)

    @traceable_step("intent/test_enabled", run_type="chain")
    def sample(x: int) -> int:
        return x + 1

    assert sample(1) == 2
    assert calls == [("intent/test_enabled", "chain")]
    assert langsmith_enabled() is True
