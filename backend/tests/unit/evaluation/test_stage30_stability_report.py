from scripts.build_stage30_stability_report import summarize_repetitions


def _run(case_id: str, repetition: int, action: str, success: bool = True):
    return {
        "case_id": case_id,
        "repetition": repetition,
        "expected_action": action,
        "observed_actions": [action],
        "success": success,
        "http_error": None,
    }


def test_repetition_summary_detects_action_volatility():
    runs = [
        _run("a", 0, "search_pois"),
        _run("b", 0, "abort"),
        _run("a", 1, "search_pois"),
        _run("b", 1, "abort", success=False),
    ]
    runs[-1]["observed_actions"] = ["propose_tradeoff"]

    result = summarize_repetitions(runs, 2)

    assert result["minimum_success_rate"] == 0.5
    assert result["maximum_success_rate"] == 1.0
    assert result["stable_action_cases"] == 1
    assert result["volatile_case_ids"] == ["b"]
