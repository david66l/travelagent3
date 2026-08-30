"""Build a compact Stage 3 GRPO curriculum from audited non-zero-variance groups.

GRPO cannot learn from a group whose sampled rewards are all identical.  This
builder keeps only exact audited tasks that produced both successful and failed
rollouts under the frozen exploration protocol.  It does not promote the wider
message-template stratum, because task-level learnability did not generalize to
every member of that stratum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.environment import environment_fingerprint  # noqa: E402
from agentic.grpo_training import load_grpo_corpus, preflight_grpo_corpus  # noqa: E402


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    path.write_text(
        "\n".join(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def _stratum(row: Any) -> tuple[str, bool]:
    metadata = row.snapshot.hidden_test_facts.get("stage3_multiturn_recovery")
    if not isinstance(metadata, dict):
        raise ValueError(f"{row.task.task_id} lacks Stage 3 recovery metadata")
    return str(metadata["message_template_sha256"]), bool(metadata["cross_tool"])


def build(
    source_dir: Path,
    audit_report: Path,
    output_dir: Path,
    *,
    minimum_success_rate: float = 0.1,
    maximum_success_rate: float = 0.9,
    expand_to_strata: bool = False,
) -> dict[str, Any]:
    if not 0 <= minimum_success_rate < maximum_success_rate <= 1:
        raise ValueError("success-rate bounds must satisfy 0 <= min < max <= 1")
    report = json.loads(audit_report.read_text(encoding="utf-8"))
    decisions = report.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("audit report contains no group decisions")

    source_train = load_grpo_corpus(source_dir / "train.jsonl")
    source_validation = load_grpo_corpus(source_dir / "validation.jsonl")
    by_id = {row.task.task_id: row for row in source_train}
    selected_decisions = [
        decision
        for decision in decisions
        if decision.get("zero_variance") is False
        and minimum_success_rate
        <= float(decision.get("success_rate", -1))
        <= maximum_success_rate
    ]
    if not selected_decisions:
        raise ValueError("audit contains no task inside the requested variance band")
    unknown = sorted(
        str(decision.get("task_id"))
        for decision in selected_decisions
        if decision.get("task_id") not in by_id
    )
    if unknown:
        raise ValueError(f"audited tasks are absent from source train: {unknown[:3]}")

    seed_rows = [by_id[str(decision["task_id"])] for decision in selected_decisions]
    selected_strata = {_stratum(row) for row in seed_rows}
    train = (
        [row for row in source_train if _stratum(row) in selected_strata]
        if expand_to_strata
        else seed_rows
    )
    train_ids = {row.task.task_id for row in train}
    validation_ids = {row.task.task_id for row in source_validation}
    if train_ids & validation_ids:
        raise ValueError("train and validation task IDs overlap")
    fingerprints = [
        environment_fingerprint(row.task, row.snapshot)
        for row in [*train, *source_validation]
    ]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("train and validation environment fingerprints overlap")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "train.jsonl", train)
    _write_jsonl(output_dir / "validation.jsonl", source_validation)
    preflight = preflight_grpo_corpus(
        output_dir,
        minimum_train_tasks=1,
        require_dependencies=False,
    )
    if not preflight.ready:
        raise ValueError(f"variance curriculum failed preflight: {preflight.errors}")

    route_counts = Counter(str(item.get("route")) for item in selected_decisions)
    manifest = {
        "schema_version": "stage3-exact-variance-grpo.v1",
        "source_dir": str(source_dir),
        "audit_report": str(audit_report),
        "audit_report_sha256": hashlib.sha256(audit_report.read_bytes()).hexdigest(),
        "selection_unit": (
            "recovery message template hash x cross-tool flag"
            if expand_to_strata
            else "exact audited task"
        ),
        "selection_rule": (
            "expand audited non-zero-variance seeds to their training strata"
            if expand_to_strata
            else "non-zero reward variance inside the configured success band"
        ),
        "expanded_to_strata": expand_to_strata,
        "selected_strata": [
            {"message_template_sha256": digest, "cross_tool": cross_tool}
            for digest, cross_tool in sorted(selected_strata)
        ],
        "success_rate_band": [minimum_success_rate, maximum_success_rate],
        "selected_audit_routes": dict(sorted(route_counts.items())),
        "selected_tasks": [
            {
                "task_id": str(item["task_id"]),
                "success_rate": float(item["success_rate"]),
                "route": str(item.get("route")),
            }
            for item in selected_decisions
        ],
        "counts": {"train": len(train), "validation": len(source_validation)},
        "preflight": preflight.model_dump(mode="json"),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-success-rate", type=float, default=0.1)
    parser.add_argument("--maximum-success-rate", type=float, default=0.9)
    parser.add_argument("--expand-to-strata", action="store_true")
    args = parser.parse_args()
    manifest = build(
        args.source_dir,
        args.audit_report,
        args.output_dir,
        minimum_success_rate=args.minimum_success_rate,
        maximum_success_rate=args.maximum_success_rate,
        expand_to_strata=args.expand_to_strata,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
