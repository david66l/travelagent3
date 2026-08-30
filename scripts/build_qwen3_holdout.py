"""Freeze leakage-audited regular, hard and adversarial Qwen3 policy holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.grpo_training import GRPOCorpusRow, load_grpo_corpus  # noqa: E402
from agentic.policy import AGENT_TOOL_POLICY_SYSTEM_PROMPT  # noqa: E402
from agentic.policy_actions import policy_action_schemas  # noqa: E402
from agentic.trl_environment import TRL_ENVIRONMENT_FACTORIES  # noqa: E402
from evaluation.inference_benchmark import VLLMBenchmarkCase  # noqa: E402


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def task_family(row: GRPOCorpusRow) -> str:
    if row.task.missing_slots:
        return "clarification"
    if row.task.feasibility_report.get("feasible", True) is False:
        return "tradeoff"
    search = row.snapshot.tool_responses.get("search_pois") or []
    if search and (search[0].error_code or search[0].data_source == "unavailable"):
        return "recovery"
    return "search"


def route_and_actions(row: GRPOCorpusRow) -> tuple[str, list[str]]:
    family = task_family(row)
    if family == "clarification":
        return "clarification", ["ask_user"]
    if family == "tradeoff":
        return "tradeoff", ["propose_tradeoff", "abort"]
    return "search", ["search_pois"]


def complexity_score(row: GRPOCorpusRow) -> int:
    serialized = json.dumps(
        {"task": row.task.model_dump(mode="json"), "snapshot": row.snapshot.model_dump(mode="json")},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    family = task_family(row)
    family_weight = {"search": 0, "clarification": 200, "recovery": 350, "tradeoff": 400}
    return (
        len(serialized)
        + 200 * len(row.task.missing_slots)
        + family_weight[family]
    )


def stable_rank(row: GRPOCorpusRow, *, seed: int) -> str:
    return canonical_hash({"seed": seed, "task_id": row.task.task_id})


def stratified_take(
    rows: Iterable[GRPOCorpusRow],
    *,
    total: int,
    key,
    allowed_families: set[str] | None = None,
) -> list[GRPOCorpusRow]:
    buckets: dict[str, list[GRPOCorpusRow]] = defaultdict(list)
    for row in rows:
        family = task_family(row)
        if allowed_families is None or family in allowed_families:
            buckets[family].append(row)
    for family in buckets:
        buckets[family].sort(key=key)
    selected: list[GRPOCorpusRow] = []
    families = sorted(buckets)
    while len(selected) < total and any(buckets.values()):
        for family in families:
            if buckets[family] and len(selected) < total:
                selected.append(buckets[family].pop(0))
    if len(selected) != total:
        raise ValueError(f"holdout capacity {len(selected)} is smaller than requested {total}")
    return selected


def benchmark_case(row: GRPOCorpusRow) -> tuple[VLLMBenchmarkCase, str, str]:
    route, allowed_actions = route_and_actions(row)
    environment = TRL_ENVIRONMENT_FACTORIES[route](audit_enabled=False)
    started = False
    try:
        initial = environment.reset(
            task=row.task.model_dump(mode="json"),
            snapshot=row.snapshot.model_dump(mode="json"),
        )
        started = True
    finally:
        # reset starts the same bounded Agent Loop used by GRPO. Finalize the
        # prompt-only session immediately so holdout building never leaks one
        # event-loop thread or file descriptor per candidate.
        if started:
            environment.get_reward()
    messages = [
        {"role": "system", "content": AGENT_TOOL_POLICY_SYSTEM_PROMPT},
        {"role": "user", "content": initial},
    ]
    tools = policy_action_schemas(allowed_actions)
    case = VLLMBenchmarkCase(
        case_id=row.task.task_id,
        family=task_family(row),
        messages=messages,
        tools=tools,
        allowed_actions=allowed_actions,
        expected_action=allowed_actions[0] if len(allowed_actions) == 1 else None,
    )
    model_payload_hash = canonical_hash({"messages": messages, "tools": tools})
    scenario_signature = canonical_hash(
        {
            "task": row.task.model_dump(mode="json", exclude={"task_id"}),
            "snapshot": row.snapshot.model_dump(mode="json"),
        }
    )
    return case, model_payload_hash, scenario_signature


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--regular", type=int, default=100)
    parser.add_argument("--hard", type=int, default=100)
    parser.add_argument("--adversarial", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    if min(args.regular, args.hard, args.adversarial) < 1:
        parser.error("all split sizes must be positive")

    rows = load_grpo_corpus(args.validation_corpus)
    unique: dict[str, GRPOCorpusRow] = {}
    for row in rows:
        source_hash = canonical_hash(
            {
                "task": row.task.model_dump(mode="json"),
                "snapshot": row.snapshot.model_dump(mode="json"),
            }
        )
        unique.setdefault(source_hash, row)
    candidates = list(unique.values())

    regular = stratified_take(
        candidates,
        total=args.regular,
        key=lambda row: stable_rank(row, seed=args.seed),
    )
    used = {row.task.task_id for row in regular}
    remaining = [row for row in candidates if row.task.task_id not in used]
    hard = stratified_take(
        remaining,
        total=args.hard,
        key=lambda row: (-complexity_score(row), stable_rank(row, seed=args.seed)),
    )
    used.update(row.task.task_id for row in hard)
    remaining = [row for row in candidates if row.task.task_id not in used]
    adversarial = stratified_take(
        remaining,
        total=args.adversarial,
        key=lambda row: (-complexity_score(row), stable_rank(row, seed=args.seed)),
        allowed_families={"recovery", "tradeoff"},
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_rows = {"regular": regular, "hard": hard, "adversarial": adversarial}
    manifest_splits: dict[str, Any] = {}
    all_payload_hashes: list[str] = []
    all_signatures: list[str] = []
    all_ids: list[str] = []
    for split, selected in split_rows.items():
        exported = []
        payload_hashes = []
        signatures = []
        for row in selected:
            case, payload_hash, signature = benchmark_case(row)
            exported.append(case.model_dump(mode="json"))
            payload_hashes.append(payload_hash)
            signatures.append(signature)
            all_ids.append(case.case_id)
        write_jsonl(args.output_dir / f"{split}.jsonl", exported)
        all_payload_hashes.extend(payload_hashes)
        all_signatures.extend(signatures)
        manifest_splits[split] = {
            "rows": len(exported),
            "families": dict(Counter(case["family"] for case in exported)),
            "unique_model_payloads": len(set(payload_hashes)),
            "unique_scenario_signatures": len(set(signatures)),
        }

    expected_total = args.regular + args.hard + args.adversarial
    if len(set(all_ids)) != expected_total:
        raise ValueError("HOLDOUT_TASK_ID_OVERLAP")
    if len(set(all_payload_hashes)) != expected_total:
        raise ValueError("HOLDOUT_MODEL_PAYLOAD_OVERLAP")
    if len(set(all_signatures)) != expected_total:
        raise ValueError("HOLDOUT_SCENARIO_SIGNATURE_OVERLAP")

    manifest = {
        "schema_version": "qwen3-policy-holdout.v1",
        "status": "frozen-before-teacher-generation",
        "source": str(args.validation_corpus),
        "source_scope": "official validation only; forbidden for gradient updates",
        "source_sha256": hashlib.sha256(args.validation_corpus.read_bytes()).hexdigest(),
        "seed": args.seed,
        "rows": expected_total,
        "unique_task_ids": len(set(all_ids)),
        "unique_model_payloads": len(set(all_payload_hashes)),
        "unique_scenario_signatures": len(set(all_signatures)),
        "splits": manifest_splits,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
