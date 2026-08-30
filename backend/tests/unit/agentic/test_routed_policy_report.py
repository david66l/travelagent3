from scripts.build_routed_policy_report import build_routed_report


def _run(case_id, family, repetition, *, success=True, latency=10.0):
    return {
        "case_id": case_id,
        "family": family,
        "repetition": repetition,
        "success": success,
        "action_mismatch": not success,
        "argument_mismatch": False,
        "http_error": None,
        "inference_metrics": {
            "model": family,
            "backend": "vllm-http",
            "thinking_mode": "disabled",
            "completion_tokens": 5,
            "request_latency_ms": latency,
        },
    }


def test_routed_report_uses_student_for_frequent_families_and_teacher_for_tradeoff():
    student = [
        _run("search", "search", 0),
        _run("recovery", "recovery", 0),
        _run("tradeoff", "tradeoff", 0, success=False),
    ]
    teacher = [_run("tradeoff", "tradeoff", 0, latency=20.0)]

    report, runs = build_routed_report(student, teacher)

    assert report["status"] == "passed"
    assert report["summary"]["successful_runs"] == 3
    assert report["summary"]["route_counts"] == {"student": 2, "teacher": 1}
    assert report["summary"]["inference"]["completion_tokens"]["total"] == 15
    assert [row["route_target"] for row in runs] == ["student", "student", "teacher"]
