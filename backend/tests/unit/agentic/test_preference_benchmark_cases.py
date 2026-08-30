import json

from agentic.distillation import TeacherPreferencePair
from agentic.policy import AGENT_TOOL_POLICY_SYSTEM_PROMPT
from agentic.policy_actions import policy_action_schemas
from evaluation.inference_benchmark import VLLMBenchmarkCase
from scripts.build_preference_benchmark_cases import build


def test_preference_benchmark_uses_current_prompt_and_action_only_for_clarification(
    tmp_path,
):
    pair = TeacherPreferencePair(
        pair_id="pref-1",
        task_id="task-1",
        family="clarification",
        context_hash="context-1",
        messages=[
            {"role": "system", "content": "stale prompt"},
            {"role": "user", "content": "{}"},
        ],
        tools=policy_action_schemas(["ask_user"]),
        chosen={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "ask_user",
                        "arguments": {"question": "Which dates?"},
                    },
                }
            ],
        },
        rejected={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "abort", "arguments": {"reason": "stop"}},
                }
            ],
        },
        chosen_trajectory_id="chosen",
        rejected_trajectory_id="rejected",
        reason_codes=["VERIFIER_SUCCESS_OVER_FAILURE"],
        reward_margin=1.0,
    )
    source = tmp_path / "test.jsonl"
    output = tmp_path / "benchmark" / "cases.jsonl"
    source.write_text(pair.model_dump_json() + "\n", encoding="utf-8")

    manifest = build(source, output)

    case = VLLMBenchmarkCase(**json.loads(output.read_text(encoding="utf-8")))
    assert manifest["cases"] == 1
    assert case.messages[0]["content"] == AGENT_TOOL_POLICY_SYSTEM_PROMPT
    assert case.expected_action == "ask_user"
    assert case.expected_arguments is None
