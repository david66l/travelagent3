from core.output_safety import (
    append_risk_hints,
    filter_sensitive_locations,
    sanitize_assistant_output,
)


def test_sensitive_location_replaced():
    text = "推荐军事禁区一日游"
    assert "该地点暂不适合推荐" in filter_sensitive_locations(text)


def test_risk_hint_appended():
    text = "行程包含高原徒步"
    out = append_risk_hints(text)
    assert "安全提示" in out


def test_sanitize_pipeline():
    out = sanitize_assistant_output("军事禁区高原潜水")
    assert "该地点暂不适合推荐" in out
    assert "安全提示" in out
