"""Build a focused, leakage-audited SFT curriculum for termination boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.sft_dataset import DatasetManifest, SFTExample  # noqa: E402


SPLITS = ("train", "validation", "test")
REPLAY_ACTIONS = frozenset({"ask_user", "search_pois"})


def _load(path: Path) -> list[SFTExample]:
    return [
        SFTExample.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _action(example: SFTExample) -> str:
    calls = example.messages[-1].tool_calls
    if len(calls) != 1:
        raise ValueError(f"{example.example_id} must contain exactly one target call")
    return calls[0].function.name


def _stable(rows: list[SFTExample], salt: str) -> list[SFTExample]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{salt}:{row.example_id}".encode()).hexdigest(),
    )


def _request(example: SFTExample) -> str:
    for message in example.messages:
        if message.role != "user" or not message.content:
            continue
        try:
            payload = json.loads(message.content)
        except json.JSONDecodeError:
            return message.content.strip()
        return str(payload.get("original_request") or "").strip()
    return ""


def _holdout_keys(path: Path) -> tuple[set[str], set[str]]:
    task_ids: set[str] = set()
    requests: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task = row.get("task", row)
        task_ids.add(str(task.get("task_id") or ""))
        request = str(task.get("user_request") or task.get("original_request") or "").strip()
        if request:
            requests.add(request)
    task_ids.discard("")
    return task_ids, requests


def build(
    base_dir: Path,
    abort_dir: Path,
    output_dir: Path,
    forbidden_holdout: Path,
) -> dict[str, object]:
    output: dict[str, list[SFTExample]] = {}
    selected_counts: dict[str, dict[str, int]] = {}

    for split in SPLITS:
        base = _load(base_dir / f"{split}.jsonl")
        repairs = _load(abort_dir / f"{split}.jsonl")
        abort_rows = [row for row in repairs if _action(row) == "abort"]
        tradeoff_rows = [row for row in base if _action(row) == "propose_tradeoff"]
        replay_rows = [row for row in base if _action(row) in REPLAY_ACTIONS]
        per_class = min(len(abort_rows), len(tradeoff_rows), len(replay_rows))
        if per_class == 0:
            raise ValueError(f"{split} has no complete abort/tradeoff/replay curriculum")
        chosen = (
            _stable(abort_rows, f"{split}:abort")[:per_class]
            + _stable(tradeoff_rows, f"{split}:tradeoff")[:per_class]
            + _stable(replay_rows, f"{split}:replay")[:per_class]
        )
        output[split] = _stable(chosen, f"{split}:output")
        selected_counts[split] = {
            "abort": per_class,
            "propose_tradeoff": per_class,
            "replay": per_class,
        }

    scenario_sets = {
        split: {row.scenario_id for row in rows} for split, rows in output.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = scenario_sets[left] & scenario_sets[right]
        if overlap:
            raise ValueError(f"scenario split leakage between {left} and {right}: {len(overlap)}")

    all_rows = [row for split in SPLITS for row in output[split]]
    example_ids = [row.example_id for row in all_rows]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("duplicate example IDs in focused curriculum")
    prompts = [row.messages[1].content or "" for row in all_rows]
    if len(prompts) != len(set(prompts)):
        raise ValueError("duplicate model-visible prompts in focused curriculum")

    forbidden_ids, forbidden_requests = _holdout_keys(forbidden_holdout)
    id_overlap = sorted({row.scenario_id for row in all_rows} & forbidden_ids)
    request_overlap = sorted({_request(row) for row in all_rows if _request(row)} & forbidden_requests)
    if id_overlap or request_overlap:
        raise ValueError(
            "decision holdout contamination: "
            f"task_ids={len(id_overlap)}, exact_requests={len(request_overlap)}"
        )

    digest_payload = {
        split: [row.example_id for row in output[split]] for split in SPLITS
    }
    version = "sft-decision-boundary-repair-" + hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    manifest = DatasetManifest(
        dataset_version=version,
        candidate_episodes=len(all_rows),
        accepted_episodes=len(all_rows),
        rejected_episodes=0,
        exported_examples=len(all_rows),
        split_examples={split: len(output[split]) for split in SPLITS},
        source_episodes=dict(Counter(row.source for row in all_rows)),
        quality_episodes=dict(Counter(row.quality_label for row in all_rows)),
        rejection_codes={},
        environment_versions=sorted({row.environment_version for row in all_rows}),
        policy_versions=sorted(
            {f"{row.policy_name}:{row.policy_version}" for row in all_rows}
        ),
        split_group_overlap=False,
        excluded_policy_steps=0,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (output_dir / f"{split}.jsonl").write_text(
            "\n".join(row.model_dump_json() for row in output[split]) + "\n",
            encoding="utf-8",
        )
    (output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    report: dict[str, object] = {
        "schema_version": "decision-boundary-repair-derivation.v1",
        "dataset_version": version,
        "base_dir": str(base_dir),
        "abort_dir": str(abort_dir),
        "forbidden_holdout": str(forbidden_holdout),
        "curriculum_ratio": "abort:propose_tradeoff:replay = 1:1:1 per split",
        "selected_counts": selected_counts,
        "action_counts": dict(Counter(_action(row) for row in all_rows)),
        "holdout_contamination": {
            "task_id_overlap": len(id_overlap),
            "exact_request_overlap": len(request_overlap),
            "passed": True,
        },
    }
    (output_dir / "derivation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--abort-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forbidden-holdout", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.base_dir, args.abort_dir, args.output_dir, args.forbidden_holdout),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
