"""Tests for dependency-light Agent Policy training preflight."""

import json

from agentic.sft_dataset import DatasetManifest, SFTExample, SFTMessage
from agentic.training import preflight_sft_dataset, to_conversational_prompt_completion


def _example(split: str, index: int) -> SFTExample:
    return SFTExample(
        example_id=f"trajectory:{index}",
        scenario_id=f"scenario-{index}",
        trajectory_id=f"trajectory-{index}",
        step_index=0,
        split=split,
        quality_label="validated_plan",
        source="teacher",
        environment_version="env-v1",
        policy_name="teacher",
        policy_version="v1",
        messages=[
            SFTMessage(role="system", content="policy"),
            SFTMessage(role="user", content="context"),
            SFTMessage(role="assistant", content='{"action":"get_weather","arguments":{}}'),
        ],
    )


def _dataset(tmp_path, *, overlap: bool = False):
    rows = {
        "train": [_example("train", 1)],
        "validation": [_example("validation", 2)],
        "test": [_example("test", 3)],
    }
    for split, examples in rows.items():
        (tmp_path / f"{split}.jsonl").write_text(
            "\n".join(item.model_dump_json() for item in examples) + "\n",
            encoding="utf-8",
        )
    manifest = DatasetManifest(
        dataset_version="sft-test",
        candidate_episodes=3,
        accepted_episodes=3,
        rejected_episodes=0,
        exported_examples=3,
        split_examples={"train": 1, "validation": 1, "test": 1},
        source_episodes={"teacher": 3},
        quality_episodes={"validated_plan": 3},
        rejection_codes={},
        environment_versions=["env-v1"],
        policy_versions=["teacher:v1"],
        split_group_overlap=overlap,
    )
    (tmp_path / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def test_conversational_rows_train_only_on_assistant_completion():
    row = _example("train", 1)

    converted = to_conversational_prompt_completion([row.model_dump(mode="json")])

    assert [message["role"] for message in converted[0]["prompt"]] == ["system", "user"]
    assert converted[0]["completion"][0]["role"] == "assistant"
    assert json.loads(converted[0]["completion"][0]["content"])["action"] == "get_weather"


def test_preflight_can_validate_data_without_gpu_dependencies(tmp_path):
    _dataset(tmp_path)

    report = preflight_sft_dataset(
        tmp_path,
        minimum_train_examples=1,
        require_dependencies=False,
    )

    assert report.ready is True
    assert report.dataset_version == "sft-test"
    assert report.train_examples == 1
    assert report.validation_examples == 1
    assert report.test_examples == 1


def test_preflight_blocks_small_or_leaking_dataset(tmp_path):
    _dataset(tmp_path, overlap=True)

    report = preflight_sft_dataset(
        tmp_path,
        minimum_train_examples=3000,
        require_dependencies=False,
    )

    assert report.ready is False
    assert "DATASET_SPLIT_GROUP_OVERLAP" in report.errors
    assert any(error.startswith("TRAIN_EXAMPLES_BELOW_MINIMUM") for error in report.errors)
