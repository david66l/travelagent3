import json

from scripts.build_sft_v2_curriculum import _deduplicate_model_payloads, _prepare


def _row(example_id: str) -> dict:
    context = {
        "trajectory_id": "[CURRENT_TRAJECTORY]",
        "goal_version": 1,
        "plan_version": 1,
        "current_subtask": {
            "task_id": "collect_poi_details",
            "goal": "Collect details",
            "required_facts": ["candidate_poi_ids"],
            "allowed_actions": ["get_poi_detail"],
        },
        "allowed_actions": ["get_poi_detail"],
        "relevant_facts": [{"key": "candidate_poi_ids", "value": ["poi-secret"]}],
    }
    return {
        "schema_version": "agent-policy-sft.v2",
        "example_id": example_id,
        "scenario_id": f"scenario:{example_id}",
        "trajectory_id": f"trajectory:{example_id}",
        "step_index": 1,
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
                        "function": {
                            "name": "get_poi_detail",
                            "arguments": {},
                        },
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_poi_detail",
                    "description": "details",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            }
        ],
    }


def test_prepare_preserves_empty_target_and_removes_controller_state():
    prepared = _prepare(_row("one"), "train", "minimal", minimize=True)
    context = json.loads(prepared["messages"][1]["content"])
    arguments = prepared["messages"][-1]["tool_calls"][0]["function"]["arguments"]

    assert arguments == {}
    assert context["controller_hydrates_arguments"] is True
    assert "candidate_poi_ids" not in json.dumps(context)


def test_deduplicate_uses_model_visible_payload_not_audit_ids():
    first = _prepare(_row("one"), "train", "minimal", minimize=True)
    second = _prepare(_row("two"), "train", "minimal", minimize=True)

    assert len(_deduplicate_model_payloads([first, second])) == 1
