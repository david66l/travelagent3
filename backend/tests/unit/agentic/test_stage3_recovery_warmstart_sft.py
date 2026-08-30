import json

from agentic.sft_dataset import DatasetManifest
from scripts.build_stage3_multiturn_rl_corpus import build as build_source
from scripts.build_stage3_recovery_warmstart_sft import build


def _fake_replay(path, counts):
    path.mkdir()
    for split, count in counts.items():
        rows = []
        for index in range(count):
            rows.append(
                {
                    "example_id": f"replay-{split}-{index}",
                    "scenario_id": f"replay-scenario-{split}-{index}",
                    "trajectory_id": f"replay-trajectory-{split}-{index}",
                    "step_index": 0,
                    "split": split,
                    "quality_label": "validated_plan",
                    "source": "synthetic",
                    "environment_version": "replay.v1",
                    "policy_name": "teacher",
                    "policy_version": "v1",
                    "messages": [
                        {"role": "system", "content": "policy"},
                        {"role": "user", "content": f"request-{split}-{index}"},
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {"name": "finish", "arguments": {}},
                                }
                            ],
                        },
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "finish",
                                "description": "finish",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        }
                    ],
                }
            )
        (path / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )


async def test_stage3_warmstart_uses_verified_history_and_keeps_blind_test_external(tmp_path):
    source = tmp_path / "source"
    build_source(
        source,
        start_index=90000,
        train_count=8,
        validation_count=4,
        test_count=4,
    )
    replay = tmp_path / "replay"
    _fake_replay(replay, {"train": 8, "validation": 2, "test": 2})
    output = tmp_path / "output"

    manifest = await build(source, replay, output)

    assert isinstance(manifest, DatasetManifest)
    assert manifest.split_examples == {"train": 16, "validation": 4, "test": 4}
    derivation = json.loads((output / "derivation.json").read_text(encoding="utf-8"))
    assert derivation["external_blind_test_used"] is False
    recovery = [
        json.loads(line)
        for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if "stage3-recovery-warmstart" in line
    ]
    assert recovery
    assert [message["role"] for message in recovery[0]["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
