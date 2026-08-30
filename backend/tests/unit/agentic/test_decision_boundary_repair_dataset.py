from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from agentic.sft_dataset import SFTExample, SFTMessage, SFTToolCall, SFTToolFunction


SCRIPT = (
    Path(__file__).resolve().parents[4] / "scripts" / "build_decision_boundary_repair_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("decision_boundary_repair", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _example(example_id: str, scenario_id: str, split: str, action: str) -> SFTExample:
    request = f"request-{example_id}"
    return SFTExample(
        example_id=example_id,
        scenario_id=scenario_id,
        trajectory_id=f"trajectory-{example_id}",
        step_index=0,
        split=split,
        quality_label="safe_termination",
        source="synthetic",
        environment_version="test.v1",
        policy_name="test",
        policy_version="v1",
        messages=[
            SFTMessage(role="system", content="system"),
            SFTMessage(role="user", content=json.dumps({"original_request": request})),
            SFTMessage(
                role="assistant",
                tool_calls=[
                    SFTToolCall(function=SFTToolFunction(name=action, arguments={"x": "y"}))
                ],
            ),
        ],
        tools=[],
    )


def _write_dataset(root: Path, actions: tuple[str, ...], prefix: str) -> None:
    root.mkdir(parents=True)
    total = 0
    for split in MODULE.SPLITS:
        rows = []
        for action in actions:
            for index in range(2):
                total += 1
                rows.append(
                    _example(f"{prefix}-{total}", f"{prefix}-scenario-{total}", split, action)
                )
        (root / f"{split}.jsonl").write_text(
            "\n".join(row.model_dump_json() for row in rows) + "\n", encoding="utf-8"
        )


def _holdout(path: Path, request: str = "independent") -> None:
    path.write_text(
        json.dumps({"task": {"task_id": "holdout-1", "user_request": request}}) + "\n",
        encoding="utf-8",
    )


def test_builds_balanced_leakage_free_curriculum(tmp_path: Path) -> None:
    base, abort, output = tmp_path / "base", tmp_path / "abort", tmp_path / "output"
    _write_dataset(base, ("propose_tradeoff", "ask_user", "search_pois"), "base")
    _write_dataset(abort, ("abort",), "abort")
    holdout = tmp_path / "holdout.jsonl"
    _holdout(holdout)

    report = MODULE.build(base, abort, output, holdout)

    assert report["holdout_contamination"]["passed"] is True
    assert report["selected_counts"]["train"] == {
        "abort": 2,
        "propose_tradeoff": 2,
        "replay": 2,
    }
    assert sum(1 for _ in (output / "train.jsonl").open(encoding="utf-8")) == 6


def test_rejects_exact_holdout_request_overlap(tmp_path: Path) -> None:
    base, abort, output = tmp_path / "base", tmp_path / "abort", tmp_path / "output"
    _write_dataset(base, ("propose_tradeoff", "ask_user"), "base")
    _write_dataset(abort, ("abort",), "abort")
    holdout = tmp_path / "holdout.jsonl"
    _holdout(holdout, "request-abort-1")

    with pytest.raises(ValueError, match="decision holdout contamination"):
        MODULE.build(base, abort, output, holdout)
