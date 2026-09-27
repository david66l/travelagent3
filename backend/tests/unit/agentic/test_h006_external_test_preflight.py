import json

from agentic.sft_dataset import DatasetManifest, SFTExample, SFTMessage, SFTToolCall, SFTToolFunction
from agentic.training import preflight_sft_dataset


def write_dataset(path):
    path.mkdir()
    example = SFTExample(
        example_id="example", scenario_id="source", trajectory_id="trajectory", step_index=0,
        split="train", quality_label="validated_plan", source="synthetic",
        environment_version="test", policy_name="test", policy_version="test",
        messages=[SFTMessage(role="system", content="system"), SFTMessage(role="user", content="user"),
                  SFTMessage(role="assistant", tool_calls=[SFTToolCall(
                      function=SFTToolFunction(name="search_pois", arguments={"keywords": ["museum"]}))])],
        tools=[],
    )
    validation = example.model_copy(deep=True, update={
        "example_id": "validation", "scenario_id": "shadow", "split": "validation",
    })
    validation.messages[1].content = "different shadow state"
    for name, rows in (("train", [example]), ("validation", [validation]), ("test", [])):
        (path / f"{name}.jsonl").write_text("".join(
            json.dumps(row.model_dump(mode="json")) + "\n" for row in rows
        ), encoding="utf-8")
    manifest = DatasetManifest(
        dataset_version="internal", candidate_episodes=2, accepted_episodes=2,
        rejected_episodes=0, exported_examples=2,
        split_examples={"train": 1, "validation": 1, "test": 0},
        source_episodes={"synthetic": 2}, quality_episodes={"validated_plan": 2},
        rejection_codes={}, environment_versions=["test"], policy_versions=["test"],
        split_group_overlap=False,
    )
    (path / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")


def test_external_holdout_must_be_explicit(tmp_path):
    dataset = tmp_path / "dataset"
    write_dataset(dataset)
    blocked = preflight_sft_dataset(
        dataset, minimum_train_examples=1, require_dependencies=False,
    )
    assert "TEST_SPLIT_EMPTY" in blocked.errors
    internal = preflight_sft_dataset(
        dataset, minimum_train_examples=1, require_dependencies=False, require_test_split=False,
    )
    assert internal.ready is True
    assert internal.warnings == ["TEST_EVALUATION_IS_EXTERNAL_AND_NOT_LOADED_BY_TRAINER"]
