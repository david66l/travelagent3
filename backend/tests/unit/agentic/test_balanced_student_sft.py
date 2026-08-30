import json

from agentic.policy_actions import policy_action_schemas
from agentic.sft_dataset import SFTExample
from scripts import build_balanced_student_sft as balanced


def _example(index: int, environment: str, split: str) -> SFTExample:
    action = "ask_user" if environment == "verified-preference.v1" else "search_pois"
    arguments = {"question": "budget"} if action == "ask_user" else {"keywords": ["art"]}
    return SFTExample(
        example_id=f"example-{split}-{environment}-{index}",
        scenario_id=f"scenario-{split}-{environment}-{index}",
        trajectory_id=f"trajectory-{split}-{environment}-{index}",
        step_index=0,
        split=split,
        quality_label="clarification" if action == "ask_user" else "validated_plan",
        source="teacher",
        environment_version=environment,
        policy_name="teacher",
        policy_version="v1",
        messages=[
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "{}"},
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
        tools=policy_action_schemas([action]),
    )


def test_balanced_sft_caps_generic_search_and_retains_special_actions(tmp_path, monkeypatch):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    monkeypatch.setattr(
        balanced,
        "DEFAULT_TRAIN_LIMITS",
        {"travel-curriculum.v1": 1, "verified-preference.v1": 10},
    )
    train = [
        _example(0, "travel-curriculum.v1", "train"),
        _example(1, "travel-curriculum.v1", "train"),
        _example(0, "verified-preference.v1", "train"),
    ]
    (source / "train.jsonl").write_text(
        "\n".join(item.model_dump_json() for item in train) + "\n", encoding="utf-8"
    )
    for split in ("validation", "test"):
        row = _example(0, "travel-curriculum.v1", split)
        (source / f"{split}.jsonl").write_text(row.model_dump_json() + "\n", encoding="utf-8")

    result = balanced.build(source, output)

    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert result["selected_train_environment_counts"] == {
        "travel-curriculum.v1": 1,
        "verified-preference.v1": 1,
    }
    assert result["scenario_split_overlap"] == 0
