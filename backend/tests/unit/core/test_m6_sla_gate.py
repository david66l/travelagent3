"""SLA target constants and performance gate wiring (M6)."""

from pathlib import Path

from core.settings import settings


def test_performance_targets_documented_in_prd():
    root = Path(__file__).resolve().parents[4]
    prd = (root / "PRD_AI全栈高并发改造.md").read_text(encoding="utf-8")
    assert "首次响应时间" in prd and "< 100ms" in prd
    assert "行程草稿生成" in prd and "< 500ms" in prd
    assert "SSE 首 Token" in prd and "< 500ms" in prd


def test_performance_gate_script_exists():
    root = Path(__file__).resolve().parents[4]
    script = root / "scripts" / "run_performance_gate.sh"
    assert script.is_file()


def test_cost_circuit_breaker_defaults_enabled():
    assert settings.cost_circuit_breaker_enabled
    assert settings.cost_circuit_breaker_daily_tokens > 0
