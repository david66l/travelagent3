"""Greedy native-decoding screen for SFT checkpoint tool protocol validity."""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.local_policy import LocalCheckpointAgentPolicy  # noqa: E402
from agentic.policy import PolicyOutputError  # noqa: E402
from agentic.sft_curriculum import policy_context, target_call, target_schema  # noqa: E402


DECISION_ACTIONS = {"abort", "ask_user", "propose_tradeoff", "search_pois"}


def select_empty_argument_challenges(
    dataset_dir: Path,
    *,
    per_action: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split in ("validation", "test"):
        path = dataset_dir / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            action, expected_arguments = target_call(row)
            schema = target_schema(row, action)
            if expected_arguments or schema.get("properties") or schema.get("required"):
                continue
            groups[action].append(row)
    selected: list[dict[str, Any]] = []
    for action in sorted(groups):
        ordered = sorted(
            groups[action],
            key=lambda row: _hash(str(row.get("example_id") or "")),
        )
        selected.extend(ordered[:per_action])
    return selected


def decision_family(row: dict[str, Any]) -> str | None:
    """Classify model-owned decisions while excluding controller transitions."""
    action, _arguments = target_call(row)
    if action not in DECISION_ACTIONS:
        return None
    if action == "search_pois":
        context = policy_context(row)
        return "recovery" if context.get("failure_summary") else "search"
    if action == "ask_user":
        return "clarification"
    return "tradeoff"


def select_decision_challenges(
    dataset_dir: Path,
    *,
    per_family: int,
) -> list[dict[str, Any]]:
    """Select frozen hard decisions from validation/test, balanced by family."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split in ("validation", "test"):
        path = dataset_dir / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            family = decision_family(row)
            if family is not None:
                groups[family].append(row)
    selected: list[dict[str, Any]] = []
    for family in sorted(groups):
        ordered = sorted(
            groups[family],
            key=lambda row: _hash(str(row.get("example_id") or "")),
        )
        selected.extend(ordered[:per_family])
    return selected


async def evaluate_checkpoint(
    checkpoint: str,
    rows: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    repeats: int,
    do_sample: bool,
    temperature: float,
) -> dict[str, Any]:
    policy = LocalCheckpointAgentPolicy(
        checkpoint,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        load_in_4bit=True,
        structured_decoding="native",
    )
    results: list[dict[str, Any]] = []
    try:
        for row in rows:
            expected_action, expected_arguments = target_call(row)
            context = policy_context(row)
            for sample_index in range(repeats):
                policy.set_rollout_seed(
                    int(_hash(f"{row['example_id']}:{sample_index}")[:8], 16)
                )
                try:
                    action = await policy.propose_from_history(
                        list(row["messages"][:-1]),
                        tools=list(row["tools"]),
                        allowed_actions=list(context.get("allowed_actions") or []),
                    )
                    action_matched = action.action == expected_action
                    exact_matched = (
                        action.action == expected_action
                        and action.arguments == expected_arguments
                    )
                    results.append(
                        {
                            "example_id": row["example_id"],
                            "sample_index": sample_index,
                            "expected_action": expected_action,
                            "actual_action": action.action,
                            "actual_arguments": action.arguments,
                            "action_matched": action_matched,
                            "passed": exact_matched,
                            "error_code": None,
                        }
                    )
                except PolicyOutputError as exc:
                    results.append(
                        {
                            "example_id": row["example_id"],
                            "sample_index": sample_index,
                            "expected_action": expected_action,
                            "actual_action": None,
                            "actual_arguments": None,
                            "action_matched": False,
                            "passed": False,
                            "error_code": exc.code,
                            "raw_output": exc.raw_output,
                        }
                    )
    finally:
        policy.close()
        gc.collect()

    passed = sum(item["passed"] for item in results)
    action_passed = sum(item["action_matched"] for item in results)
    return {
        "checkpoint": checkpoint,
        "rows": len(results),
        "passed": passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "action_passed": action_passed,
        "action_pass_rate": action_passed / len(results) if results else 0.0,
        "exact_match_by_family": _rate_by_family(results, rows, key="passed"),
        "action_match_by_family": _rate_by_family(
            results,
            rows,
            key="action_matched",
        ),
        "failures_by_action": dict(
            Counter(item["expected_action"] for item in results if not item["passed"])
        ),
        "failure_codes": dict(
            Counter(item["error_code"] or "WRONG_VALID_CALL" for item in results if not item["passed"])
        ),
        "results": results,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.selection_mode == "decision":
        rows = select_decision_challenges(
            args.dataset_dir,
            per_family=args.per_action,
        )
    else:
        rows = select_empty_argument_challenges(
            args.dataset_dir,
            per_action=args.per_action,
        )
    arms = []
    for checkpoint in args.checkpoints:
        arm = await evaluate_checkpoint(
            checkpoint,
            rows,
            max_new_tokens=args.max_new_tokens,
            repeats=args.repeats,
            do_sample=args.do_sample,
            temperature=args.temperature,
        )
        arms.append(arm)
        print(
            f"{checkpoint}: {arm['passed']}/{arm['rows']} "
            f"exact ({arm['pass_rate']:.2%}), "
            f"action={arm['action_passed']}/{arm['rows']} "
            f"({arm['action_pass_rate']:.2%})",
            flush=True,
        )
    return {
        "schema_version": "agent-policy-sft-protocol-screen.v1",
        "scope": "greedy native-decoding isolated protocol screen",
        "selection_mode": args.selection_mode,
        "dataset_dir": str(args.dataset_dir),
        "per_action": args.per_action,
        "repeats": args.repeats,
        "decoding": "sampled-native" if args.do_sample else "greedy-native",
        "temperature": args.temperature if args.do_sample else None,
        "selected_rows": len(rows),
        "selected_actions": dict(Counter(target_call(row)[0] for row in rows)),
        "selected_families": dict(
            Counter(decision_family(row) or "controller" for row in rows)
        ),
        "arms": arms,
    }


def _rate_by_family(
    results: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    key: str,
) -> dict[str, dict[str, float | int]]:
    family_by_id = {
        str(row.get("example_id") or ""): decision_family(row) or "controller"
        for row in rows
    }
    totals: Counter[str] = Counter()
    passed: Counter[str] = Counter()
    for item in results:
        family = family_by_id[str(item["example_id"])]
        totals[family] += 1
        if item[key]:
            passed[family] += 1
    return {
        family: {
            "passed": passed[family],
            "rows": total,
            "rate": passed[family] / total,
        }
        for family, total in sorted(totals.items())
    }


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-action", type=int, default=4)
    parser.add_argument(
        "--selection-mode",
        choices=("empty_schema", "decision"),
        default="empty_schema",
        help="Evaluate empty-schema protocol rows or balanced model-owned decisions.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    if args.per_action < 1:
        parser.error("--per-action must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.do_sample and args.temperature <= 0:
        parser.error("--temperature must be positive for sampled decoding")
    report = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
