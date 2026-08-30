import json

from scripts.build_stage35_isolated_action_preferences import isolate_action


def test_isolate_action_reduces_controller_contract_and_tools():
    row = {
        "example_id": "example-1",
        "scenario_id": "scenario-1",
        "trajectory_id": "trajectory-1",
        "environment_version": "test.v1",
        "messages": [
            {"role": "system", "content": "Call one tool."},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "allowed_actions": ["ask_user", "abort"],
                        "current_subtask": {"allowed_actions": ["ask_user", "abort"]},
                        "original_request": "无法安全继续",
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "abort",
                            "arguments": {"reason": "unsafe"},
                        },
                    }
                ],
            },
        ],
        "tools": [
            {"type": "function", "function": {"name": "ask_user"}},
            {"type": "function", "function": {"name": "abort"}},
        ],
    }

    isolated = isolate_action(row)
    payload = json.loads(isolated["messages"][-2]["content"])

    assert payload["allowed_actions"] == ["abort"]
    assert payload["current_subtask"]["allowed_actions"] == ["abort"]
    assert [tool["function"]["name"] for tool in isolated["tools"]] == ["abort"]
    assert isolated["messages"][-1] == row["messages"][-1]
    assert json.loads(row["messages"][-2]["content"])["allowed_actions"] == [
        "ask_user",
        "abort",
    ]
