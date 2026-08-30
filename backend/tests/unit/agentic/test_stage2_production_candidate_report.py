import json

from scripts.build_stage2_production_candidate_report import summarize


def _decision(task_id: str, successes: int, group_size: int = 4) -> dict:
    return {
        "task_id": task_id,
        "group_size": group_size,
        "success_rate": successes / group_size,
    }


def test_summarize_preserves_decision_class_counts():
    report = {
        "checkpoint": "candidate",
        "temperature": 0.2,
        "seed": 211,
        "group_size": 4,
        "decisions": [
            _decision("task-semantic-clarification", 4),
            _decision("task-terminal-injection", 3),
            _decision("task-actionable-tradeoff", 4),
            _decision("task-necessary-abort", 2),
        ],
        "behavior_gate": {
            "successful_rollouts": 13,
            "rollouts": 16,
            "success_rate": 13 / 16,
            "invalid_actions": 0,
            "policy_output_errors": 0,
            "policy_argument_errors": 0,
        },
    }

    result = summarize(json.loads(json.dumps(report)))

    assert result["per_class"]["semantic-clarification"]["successful_rollouts"] == 4
    assert result["per_class"]["terminal-injection"]["successful_rollouts"] == 3
    assert result["per_class"]["actionable-tradeoff"]["successful_rollouts"] == 4
    assert result["per_class"]["necessary-abort"]["successful_rollouts"] == 2
