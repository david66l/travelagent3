"""Training preference merges stay unique, strong and deterministic."""

import importlib.util
import json
from pathlib import Path

from agentic.distillation import TeacherPreferencePair


SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "merge_verified_preferences.py"
SPEC = importlib.util.spec_from_file_location("merge_verified_preferences", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _pair(index: int, family: str) -> TeacherPreferencePair:
    return TeacherPreferencePair(
        pair_id=f"pref-{index}",
        task_id=f"task-{index}",
        family=family,
        context_hash=f"context-{index}",
        messages=[{"role": "user", "content": f"context {index}"}],
        tools=[],
        chosen={"role": "assistant", "content": f"chosen {index}"},
        rejected={"role": "assistant", "content": f"rejected {index}"},
        chosen_trajectory_id=f"chosen-{index}",
        rejected_trajectory_id=f"rejected-{index}",
        reason_codes=["VERIFIER_SUCCESS_OVER_FAILURE"],
        reward_margin=1.0,
    )


def _write_source(path: Path, pairs: list[TeacherPreferencePair]) -> None:
    path.mkdir()
    (path / "preference_pairs.jsonl").write_text(
        "\n".join(json.dumps(pair.model_dump(mode="json")) for pair in pairs) + "\n",
        encoding="utf-8",
    )


def test_merge_builds_disjoint_deterministic_splits(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "output"
    _write_source(first, [_pair(index, "search") for index in range(8)])
    _write_source(second, [_pair(index, "recovery") for index in range(8, 16)])

    manifest = MODULE.merge([first, second], output)

    assert manifest["status"] == "passed"
    assert manifest["unique_pairs"] == 16
    assert manifest["family_counts"] == {"search": 8, "recovery": 8}
    assert manifest["split_counts"] == {"validation": 2, "test": 2, "train": 12}
    split_ids = {
        name: {
            json.loads(line)["pair_id"]
            for line in (output / f"{name}.jsonl").read_text().splitlines()
        }
        for name in ("train", "validation", "test")
    }
    assert not split_ids["train"] & split_ids["validation"]
    assert not split_ids["train"] & split_ids["test"]
    assert not split_ids["validation"] & split_ids["test"]
