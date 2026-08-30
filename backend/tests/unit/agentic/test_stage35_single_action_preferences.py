import json
from pathlib import Path

from scripts.build_stage35_single_action_preferences import (
    ACTIONS,
    SPLITS,
    _target_split,
    build,
)


def _row(action: str, scenario_id: str, index: int) -> dict:
    arguments = {
        "search_pois": {"keywords": ["历史"]},
        "ask_user": {"question": "请补充预算。"},
        "propose_tradeoff": {"reason": "时间冲突", "options": ["减少景点"]},
        "abort": {"reason": "无安全可行方案"},
    }[action]
    return {
        "schema_version": "agent-policy-sft.v2",
        "example_id": f"example-{action}-{index}",
        "scenario_id": scenario_id,
        "trajectory_id": f"trajectory-{action}-{index}",
        "step_index": 0,
        "split": "train",
        "quality_label": "validated_plan",
        "source": "teacher",
        "environment_version": "unit-test.v1",
        "policy_name": "teacher",
        "policy_version": "v1",
        "messages": [
            {"role": "system", "content": "Call exactly one function."},
            {
                "role": "user",
                "content": json.dumps(
                    {"original_request": f"独立训练请求 {action} {index}"},
                    ensure_ascii=False,
                ),
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": action, "arguments": arguments},
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": action,
                    "description": action,
                    "parameters": {"type": "object"},
                },
            }
        ],
    }


def _scenario_for(split: str, action: str, start: int) -> tuple[str, int]:
    index = start
    while True:
        scenario = f"scenario-{action}-{index}"
        if _target_split(scenario) == split:
            return scenario, index + 1
        index += 1


def test_builds_balanced_group_safe_single_over_duplicate_preferences(tmp_path: Path):
    rows = []
    cursor = 0
    for action in ACTIONS:
        for split in SPLITS:
            scenario, cursor = _scenario_for(split, action, cursor)
            rows.append(_row(action, scenario, cursor))
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    forbidden = tmp_path / "forbidden.jsonl"
    forbidden.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"original_request": "完全无关的冻结评测请求"},
                            ensure_ascii=False,
                        ),
                    }
                ]
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build(
        source,
        tmp_path / "output",
        forbidden_files=[forbidden],
        per_action=3,
    )

    assert manifest["status"] == "passed"
    assert manifest["preference_pairs"] == 12
    assert manifest["split_counts"] == {"train": 4, "validation": 4, "test": 4}
    assert manifest["scenario_split_overlap"] is False
    training_manifest = json.loads(
        (tmp_path / "output" / "preferences" / "manifest.json").read_text(encoding="utf-8")
    )
    assert training_manifest["preference_evidence_policy"] == (
        "verifier_success_or_deterministic_single_action_contract"
    )
    assert training_manifest["run_scope_constraint"].startswith("targeted_smoke_only")
    for split in SPLITS:
        pairs = [
            json.loads(line)
            for line in (tmp_path / "output" / "preferences" / f"{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert {pair["family"] for pair in pairs} == {
            f"single_action_{action}" for action in ACTIONS
        }
        assert all(len(pair["chosen"]["tool_calls"]) == 1 for pair in pairs)
        assert all(len(pair["rejected"]["tool_calls"]) == 2 for pair in pairs)
        assert all(
            pair["rejected"]["tool_calls"][0] == pair["rejected"]["tool_calls"][1] for pair in pairs
        )


def test_rejects_missing_frozen_evaluation_requests(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    source.write_text("", encoding="utf-8")
    forbidden = tmp_path / "forbidden.jsonl"
    forbidden.write_text("{}\n", encoding="utf-8")

    try:
        build(source, tmp_path / "output", forbidden_files=[forbidden], per_action=3)
    except ValueError as exc:
        assert "no auditable user requests" in str(exc)
    else:
        raise AssertionError("expected frozen-evaluation audit failure")
