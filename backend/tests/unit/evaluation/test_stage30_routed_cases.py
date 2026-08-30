import json

from scripts.build_stage30_routed_cases import split_cases


def _case(case_id: str, status: str, failures=None):
    context = {
        "capability": {"status": status},
        "failure_summary": failures or [],
    }
    return {
        "case_id": case_id,
        "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": json.dumps(context)},
        ],
    }


def test_split_cases_uses_only_policy_visible_state():
    cases = [
        _case("search", "solvable"),
        _case("tradeoff", "infeasible"),
        _case(
            "retry",
            "missing_tool",
            [{"retryable": True, "retry_budget_remaining": 1}],
        ),
        _case(
            "exhausted",
            "missing_tool",
            [{"retryable": False, "retry_budget_remaining": 0}],
        ),
    ]

    student, teacher = split_cases(cases)

    assert [case["case_id"] for case in student] == ["search", "retry"]
    assert [case["case_id"] for case in teacher] == ["tradeoff", "exhausted"]
