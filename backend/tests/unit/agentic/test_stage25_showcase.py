import pytest

from scripts.build_stage25_showcase import select_successful_example


def test_select_successful_example_is_deterministic_and_sanitized():
    metrics = {
        "model": "student-4b",
        "completion_tokens": 12,
        "request_latency_ms": 123.4,
        "ttft_ms": 50.0,
        "tpot_ms": 4.0,
    }
    rows = [
        {
            "case_id": "b",
            "family": "search",
            "repetition": 0,
            "success": True,
            "expected_action": "search_pois",
            "observed_actions": ["search_pois"],
            "observed_arguments": {"city": "Shanghai"},
            "inference_metrics": metrics,
            "raw_prompt": "must not leak",
        },
        {
            "case_id": "a",
            "family": "search",
            "repetition": 0,
            "success": True,
            "expected_action": "search_pois",
            "observed_actions": ["search_pois"],
            "observed_arguments": {"city": "Beijing"},
            "inference_metrics": metrics,
        },
    ]

    result = select_successful_example(rows, "search")

    assert result["case_id"] == "a"
    assert result["model"] == "student-4b"
    assert result["completion_tokens"] == 12
    assert "raw_prompt" not in result


def test_select_successful_example_rejects_missing_family():
    with pytest.raises(ValueError, match="no successful example"):
        select_successful_example([], "tradeoff")
