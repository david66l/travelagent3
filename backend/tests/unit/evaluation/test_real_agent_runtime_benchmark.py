from scripts.benchmark_real_agent_runtime import build_report


def _row(case_id: str, mode: str, tokens: int, passed: bool) -> dict:
    return {
        "case_id": case_id,
        "mode": mode,
        "execution_mode": "policy_driven",
        "destination": "北京",
        "hard_pass": passed,
        "total_tokens": tokens,
        "model_calls": 1,
        "tool_calls": 5,
        "episode_steps": 9,
        "latency_ms": 1000,
        "solve_status": "fallback",
        "violation_codes": [],
    }


def test_real_runtime_report_pairs_by_case_and_calculates_reduction():
    actual = [
        _row("a", "real_agent_runtime", 500, True),
        _row("b", "real_agent_runtime", 500, True),
    ]
    pure = [_row("a", "pure_agent", 1000, False), _row("b", "pure_agent", 1000, True)]
    cases = {
        "a": {"travel_days": 2},
        "b": {"travel_days": 3},
    }
    report = build_report(actual, pure, cases)
    assert report["paired_tasks"] == 2
    assert report["execution_mode"] == "policy_driven"
    assert report["paired_delta"]["pure_to_actual_token_ratio"] == 2
    assert report["paired_delta"]["actual_token_reduction_vs_pure_percent"] == 50
    assert report["real_agent_runtime"]["hard_pass_rate"] == 1
