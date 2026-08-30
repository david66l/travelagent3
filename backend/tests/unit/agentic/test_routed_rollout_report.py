from scripts.build_routed_rollout_report import build_routed_rollout_report


def _candidate(task_id, family, *, success=True, tokens=10, latency=20.0):
    return {
        "task_id": task_id,
        "family": family,
        "score": {
            "successful": success,
            "episode_reward": 1.0 if success else 0.0,
            "policy_steps": 1,
            "completion_tokens": tokens,
            "request_latency_ms": latency,
        },
    }


def test_routed_rollout_report_replaces_student_tradeoff_with_teacher():
    student = [
        _candidate("search", "search"),
        _candidate("tradeoff", "tradeoff", success=False),
    ]
    teacher = [_candidate("tradeoff", "tradeoff", tokens=20, latency=30.0)]

    report, rows = build_routed_rollout_report(student, teacher)

    assert report["status"] == "passed"
    assert report["summary"]["successful_tasks"] == 2
    assert report["summary"]["completion_tokens"] == 30
    assert report["summary"]["route_tasks"] == {"student": 1, "teacher": 1}
    assert {row["task_id"]: row["route_target"] for row in rows} == {
        "search": "student",
        "tradeoff": "teacher",
    }
