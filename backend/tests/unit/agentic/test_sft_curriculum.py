import json

from agentic.sft_curriculum import (
    action_family,
    collect_keys,
    minimize_row_context,
    target_call,
)


def _row(action: str = "get_poi_detail") -> dict:
    context = {
        "trajectory_id": "[CURRENT_TRAJECTORY]",
        "goal_version": 1,
        "plan_version": 1,
        "original_request": "plan a trip",
        "current_subtask": {
            "task_id": "collect_poi_details",
            "goal": "collect details",
            "status": "running",
            "required_facts": ["candidate_poi_ids"],
            "allowed_actions": [action],
        },
        "allowed_actions": [action],
        "relevant_facts": [
            {
                "key": "candidate_poi_ids",
                "value": ["poi-secret-1"],
            }
        ],
        "remaining_steps": 10,
        "remaining_tasks": 6,
    }
    return {
        "example_id": "example:1",
        "scenario_id": "scenario:1",
        "trajectory_id": "trajectory:1",
        "step_index": 3,
        "split": "train",
        "quality_label": "validated_plan",
        "source": "synthetic",
        "environment_version": "test.v1",
        "policy_name": "Teacher",
        "policy_version": "v1",
        "messages": [
            {"role": "system", "content": "system", "tool_calls": []},
            {"role": "user", "content": json.dumps(context), "tool_calls": []},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": action, "arguments": {}},
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": action,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            }
        ],
    }


def test_minimize_row_removes_controller_argument_lures_but_preserves_target():
    row = _row()
    minimized = minimize_row_context(row)
    context = json.loads(minimized["messages"][1]["content"])

    assert target_call(minimized) == ("get_poi_detail", {})
    assert context["controller_hydrates_arguments"] is True
    assert "candidate_poi_ids" not in json.dumps(context)
    assert "poi-secret-1" not in json.dumps(context)


def test_collect_keys_and_recovery_family():
    context = {
        "failure_summary": [{"code": "UPSTREAM_TIMEOUT"}],
        "relevant_facts": [{"key": "candidate_poi_ids"}],
    }

    assert {"failure_summary", "code", "candidate_poi_ids", "key"} <= collect_keys(context)
    assert action_family("search_pois", context) == "recovery"
