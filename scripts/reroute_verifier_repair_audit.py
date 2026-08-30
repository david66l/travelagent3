"""Replay audited actions through the current verifier-repair reward contract.

This changes neither model outputs nor task membership.  It is used when the
verifier reward decomposition changes: previously sampled actions are replayed
through the production environment, exact-success invariants are checked, and
only groups with useful on-policy variance are exported for GRPO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.grpo import GRPOGroupAuditor  # noqa: E402
from agentic.grpo_training import (  # noqa: E402
    GRPOCorpusRow,
    load_grpo_corpus,
    to_trl_environment_rows,
)
from agentic.trl_environment import build_trl_environment_factories  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reroute(corpus_file: Path, rollouts_file: Path, output_dir: Path) -> dict[str, Any]:
    corpus = {row.task.task_id: row for row in load_grpo_corpus(corpus_file)}
    sampled = _read_jsonl(rollouts_file)
    if not sampled:
        raise ValueError("audit rollout file is empty")
    factories = build_trl_environment_factories("react")
    replayed_by_task: dict[str, list[Any]] = defaultdict(list)
    replay_rows: list[dict[str, Any]] = []

    for sampled_row in sampled:
        task_id = str(sampled_row.get("task_id") or "")
        row = corpus.get(task_id)
        if row is None:
            raise ValueError(f"sampled task is absent from corpus: {task_id}")
        actions = list(sampled_row.get("actions") or [])
        if len(actions) > 1:
            raise ValueError(f"decision-state rollout has multiple sampled actions: {task_id}")
        converted = to_trl_environment_rows([row])[0]
        environment = factories[converted["environment"]](audit_enabled=False)
        environment.reset(**converted)
        if actions:
            action = actions[0]
            environment._act(str(action.get("action") or ""), dict(action.get("arguments") or {}))
        score = environment.get_reward()
        replayed = environment.rollout_record
        if replayed is None:
            raise RuntimeError("replayed decision did not produce a reward")
        old_exact = float(sampled_row.get("reward") or -1) == 1.0
        new_exact = replayed.reward.gate_status == "passed"
        if old_exact != new_exact:
            raise ValueError(f"exact-success invariant changed during reward replay: {task_id}")
        replayed_by_task[task_id].append(replayed)
        contract = row.snapshot.hidden_test_facts["grpo_decision_state"]
        replay_rows.append(
            {
                "task_id": task_id,
                "sample_index": sampled_row.get("sample_index"),
                "target_action": contract["target_action"],
                "old_reward": sampled_row.get("reward"),
                "reward": score,
                "gate_status": replayed.reward.gate_status,
                "actions": actions,
                "audit_metrics": replayed.reward.audit_metrics,
            }
        )

    auditor = GRPOGroupAuditor()
    decisions = [
        auditor.evaluate(f"reward-replay:{task_id}", replayed_by_task[task_id])
        for task_id in sorted(replayed_by_task)
    ]
    route_by_task = {decision.task_id: decision.route for decision in decisions}
    routed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task_id, route in route_by_task.items():
        routed[route].append(corpus[task_id].model_dump(mode="json"))

    output_dir.mkdir(parents=True, exist_ok=True)
    for route in ("grpo_update", "sft_repair", "evaluation", "reject"):
        _write_jsonl(output_dir / f"{route}.jsonl", routed[route])
    _write_jsonl(output_dir / "replayed_rollouts.jsonl", replay_rows)
    _write_jsonl(
        output_dir / "group_decisions.jsonl",
        [decision.model_dump(mode="json") for decision in decisions],
    )
    by_target: dict[str, Counter[str]] = defaultdict(Counter)
    for row in replay_rows:
        by_target[str(row["target_action"])][str(row["reward"])] += 1
    report = {
        "schema_version": "verifier-repair-reward-replay.v1",
        "status": "passed",
        "corpus_file": corpus_file.as_posix(),
        "corpus_sha256": _sha256(corpus_file),
        "rollouts_file": rollouts_file.as_posix(),
        "rollouts_sha256": _sha256(rollouts_file),
        "sampled_rollouts": len(replay_rows),
        "independent_tasks": len(replayed_by_task),
        "exact_successes_preserved": sum(row["gate_status"] == "passed" for row in replay_rows),
        "routes": dict(Counter(decision.route for decision in decisions)),
        "route_target_counts": {
            route: dict(
                Counter(
                    str(row.snapshot.hidden_test_facts["grpo_decision_state"]["target_action"])
                    for row in (
                        GRPOCorpusRow(**payload) for payload in routed[route]
                    )
                )
            )
            for route in ("grpo_update", "sft_repair", "evaluation", "reject")
        },
        "reward_distribution_by_target": {
            target: dict(sorted(counts.items())) for target, counts in sorted(by_target.items())
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--rollouts-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            reroute(args.corpus_file, args.rollouts_file, args.output_dir),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
