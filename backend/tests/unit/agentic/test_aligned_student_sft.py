import json

from agentic.distillation import TeacherPreferencePair
from agentic.policy import AGENT_TOOL_POLICY_SYSTEM_PROMPT
from agentic.policy_actions import policy_action_schemas
from agentic.sft_dataset import SFTExample
from scripts.build_aligned_student_sft import build


def _example(*, action: str, quality: str, scenario_id: str = "clar-1") -> SFTExample:
    arguments = {} if action == "capability_check" else {"question": "old question"}
    return SFTExample(
        example_id=f"old:{action}",
        scenario_id=scenario_id,
        trajectory_id=f"trajectory:{action}",
        step_index=0,
        split="train",
        quality_label=quality,
        source="teacher",
        environment_version="old-env",
        policy_name="teacher",
        policy_version="old",
        messages=[
            {"role": "system", "content": "old prompt"},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "capability": {"status": "needs_user"},
                        "missing_information": ["budget_range"],
                    }
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": action, "arguments": arguments},
                    }
                ],
            },
        ],
        tools=policy_action_schemas(["capability_check", "ask_user"]),
    )


def _clarification_pair() -> TeacherPreferencePair:
    context = json.dumps(
        {
            "capability": {"status": "needs_user"},
            "missing_information": ["budget_range"],
        }
    )
    return TeacherPreferencePair(
        pair_id="pref-clar-1",
        task_id="clar-1",
        family="clarification",
        context_hash="context-1",
        messages=[
            {"role": "system", "content": "old prompt"},
            {"role": "user", "content": context},
        ],
        tools=policy_action_schemas(["capability_check", "ask_user"]),
        chosen={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "ask_user",
                        "arguments": {"question": "请补充您的旅行预算范围。"},
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
                    "function": {"name": "capability_check", "arguments": {}},
                }
            ],
        },
        chosen_trajectory_id="chosen-1",
        rejected_trajectory_id="rejected-1",
        reason_codes=["VERIFIER_SUCCESS_OVER_FAILURE"],
        reward_margin=1.0,
    )


def test_aligned_student_sft_replaces_stale_clarification(tmp_path):
    source = tmp_path / "source"
    preferences = tmp_path / "preferences"
    output = tmp_path / "output"
    source.mkdir()
    preferences.mkdir()
    (source / "train.jsonl").write_text(
        _example(action="capability_check", quality="clarification").model_dump_json() + "\n",
        encoding="utf-8",
    )
    for split in ("validation", "test"):
        (source / f"{split}.jsonl").write_text("", encoding="utf-8")
        (preferences / f"{split}.jsonl").write_text("", encoding="utf-8")
    (preferences / "train.jsonl").write_text(
        _clarification_pair().model_dump_json() + "\n", encoding="utf-8"
    )

    result = build(source, preferences, output)

    row = SFTExample(**json.loads((output / "train.jsonl").read_text(encoding="utf-8")))
    assert result["action_counts"] == {"ask_user": 1}
    assert row.messages[0].content == AGENT_TOOL_POLICY_SYSTEM_PROMPT
    assert row.messages[-1].tool_calls[0].function.name == "ask_user"
    assert result["scenario_split_overlap"] == 0
