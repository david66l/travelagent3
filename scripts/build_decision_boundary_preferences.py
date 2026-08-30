"""Build bidirectional DPO pairs from a leakage-audited boundary SFT curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.sft_dataset import SFTExample  # noqa: E402


SPLITS = ("train", "validation", "test")
EVIDENCE_POLICY = "verifier_success_or_deterministic_decision_boundary_contract"
REASON_CODE = "DECISION_BOUNDARY_CONTRACT_OVER_OPPOSITE_ACTION"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def _load(path: Path) -> list[SFTExample]:
    return [
        SFTExample.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _action(example: SFTExample) -> str:
    calls = example.messages[-1].tool_calls
    if len(calls) != 1:
        raise ValueError(f"{example.example_id} must contain one target call")
    return calls[0].function.name


def _boundary(example: SFTExample) -> bool:
    content = example.messages[1].content or ""
    context = json.loads(content)
    actionable = (context.get("capability") or {}).get("actionable_alternatives")
    expected = _action(example)
    return (actionable is True and expected == "propose_tradeoff") or (
        actionable is False and expected == "abort"
    )


def _response(example: SFTExample) -> dict[str, Any]:
    return example.messages[-1].model_dump(mode="json", exclude_none=True)


def _opposite(example: SFTExample) -> dict[str, Any]:
    chosen = _response(example)
    chosen_call = chosen["tool_calls"][0]
    arguments = chosen_call["function"]["arguments"]
    reason = str(arguments.get("reason") or "当前条件不可满足")
    rejected = deepcopy(chosen)
    if _action(example) == "abort":
        rejected["tool_calls"][0]["function"] = {
            "name": "propose_tradeoff",
            "arguments": {
                "reason": reason,
                "options": ["放宽已经确认的硬约束"],
            },
        }
    else:
        rejected["tool_calls"][0]["function"] = {
            "name": "abort",
            "arguments": {"reason": reason},
        }
    return rejected


def _pair(example: SFTExample, split: str) -> dict[str, Any]:
    messages = [
        message.model_dump(mode="json", exclude_none=True)
        for message in example.messages[:-1]
    ]
    tools = [
        deepcopy(tool)
        if isinstance(tool, dict)
        else tool.model_dump(mode="json", exclude_none=True)
        for tool in example.tools
    ]
    context_hash = _canonical_hash({"messages": messages, "tools": tools})
    chosen = _response(example)
    rejected = _opposite(example)
    action = _action(example)
    pair_id = "boundary-pref-" + _canonical_hash(
        {"context_hash": context_hash, "chosen": chosen, "rejected": rejected}
    )[:20]
    return {
        "schema_version": "teacher-preference-pair.v1",
        "pair_id": pair_id,
        "task_id": example.scenario_id,
        "family": f"decision_boundary_{action}",
        "context_hash": context_hash,
        "messages": messages,
        "tools": tools,
        "chosen": chosen,
        "rejected": rejected,
        "chosen_trajectory_id": example.trajectory_id,
        "rejected_trajectory_id": f"contract-negative:{example.trajectory_id}",
        "reason_codes": [REASON_CODE],
        "reward_margin": 1.0,
        "split": split,
    }


def _holdout_keys(path: Path) -> tuple[set[str], set[str]]:
    task_ids: set[str] = set()
    requests: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task = row.get("task", row)
        task_id = str(task.get("task_id") or "")
        request = str(task.get("user_request") or "").strip()
        if task_id:
            task_ids.add(task_id)
        if request:
            requests.add(request)
    return task_ids, requests


def build(source_dir: Path, output_dir: Path, forbidden_holdout: Path) -> dict[str, Any]:
    output: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        examples = [
            example
            for example in _load(source_dir / f"{split}.jsonl")
            if _action(example) in {"abort", "propose_tradeoff"}
        ]
        invalid = [example.example_id for example in examples if not _boundary(example)]
        if invalid:
            raise ValueError(f"invalid model-visible boundary contracts: {invalid[:3]}")
        output[split] = sorted(
            [_pair(example, split) for example in examples],
            key=lambda row: row["pair_id"],
        )

    all_pairs = [pair for split in SPLITS for pair in output[split]]
    pair_ids = [pair["pair_id"] for pair in all_pairs]
    contexts = [pair["context_hash"] for pair in all_pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("duplicate preference pair IDs")
    if len(contexts) != len(set(contexts)):
        raise ValueError("duplicate or cross-split preference contexts")

    forbidden_ids, forbidden_requests = _holdout_keys(forbidden_holdout)
    task_overlap = {pair["task_id"] for pair in all_pairs} & forbidden_ids
    pair_requests = set()
    for pair in all_pairs:
        context = json.loads(pair["messages"][1]["content"])
        request = str(context.get("original_request") or "").strip()
        if request:
            pair_requests.add(request)
    request_overlap = pair_requests & forbidden_requests
    if task_overlap or request_overlap:
        raise ValueError(
            "decision holdout contamination: "
            f"task_ids={len(task_overlap)}, exact_requests={len(request_overlap)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (output_dir / f"{split}.jsonl").write_text(
            "\n".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                for row in output[split]
            )
            + "\n",
            encoding="utf-8",
        )
    version = "preference-decision-boundary-" + _canonical_hash(
        {split: [row["pair_id"] for row in output[split]] for split in SPLITS}
    )[:16]
    manifest = {
        "schema_version": "verified-preference-dataset.v1",
        "status": "passed",
        "dataset_version": version,
        "requires_verifier_success_over_failure": False,
        "preference_evidence_policy": EVIDENCE_POLICY,
        "deterministic_contract": (
            "actionable_alternatives=true requires propose_tradeoff; false requires abort"
        ),
        "unique_pairs": len(all_pairs),
        "unique_contexts": len(set(contexts)),
        "split_counts": {split: len(output[split]) for split in SPLITS},
        "family_counts": dict(Counter(row["family"] for row in all_pairs)),
        "reason_counts": {REASON_CODE: len(all_pairs)},
        "frozen_holdout_task_overlap": len(task_overlap),
        "frozen_holdout_exact_request_overlap": len(request_overlap),
        "errors": [],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forbidden-holdout", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.source_dir, args.output_dir, args.forbidden_holdout),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
