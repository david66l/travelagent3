from scripts.build_stage34_simulated_hard_report import _paired, _policy_visible_router


def _row(case_id: str, action: str, success: bool, latency: float) -> dict:
    return {
        "case_id": case_id,
        "expected_action": action,
        "allowed_actions": [action],
        "observed_actions": [action] if success else ["ask_user"],
        "observed_arguments": {},
        "success": success,
        "policy_contract_success": success,
        "inference_metrics": {"request_latency_ms": latency},
    }


def test_paired_counts_candidate_and_baseline_unique_wins():
    baseline = {
        "a": _row("a", "abort", True, 1),
        "b": _row("b", "search_pois", False, 1),
    }
    candidate = {
        "a": _row("a", "abort", False, 1),
        "b": _row("b", "search_pois", True, 1),
    }

    result = _paired(candidate, baseline, success_key="success")

    assert result["candidate_only_success"] == 1
    assert result["baseline_only_success"] == 1
    assert result["difference_percentage_points"] == 0


def test_policy_visible_router_uses_sft_only_for_search():
    base = {
        "abort": _row("abort", "abort", True, 10),
        "search": _row("search", "search_pois", False, 20),
        "mixed": {
            **_row("mixed", "ask_user", False, 30),
            "allowed_actions": ["search_pois", "ask_user"],
        },
    }
    sft = {
        "abort": _row("abort", "abort", False, 30),
        "search": _row("search", "search_pois", True, 40),
        "mixed": {
            **_row("mixed", "ask_user", True, 50),
            "allowed_actions": ["search_pois", "ask_user"],
        },
    }

    result = _policy_visible_router(base, sft)

    assert result["raw_successful"] == 3
    assert result["contract_successful"] == 3
    assert result["route_counts"] == {"base": 1, "sft": 2}
    assert result["latency_ms"]["mean"] == 33.333
