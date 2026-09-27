"""Rescore saved curriculum rollouts at their evaluator-owned decision boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.grpo_training import GRPOCorpusRow, load_grpo_corpus  # noqa: E402
from agentic.training import load_jsonl  # noqa: E402
from scripts.audit_model_curriculum import (  # noqa: E402
    decision_loop_metadata,
    episode_outcome_metrics,
    external_evidence_metadata,
    target_decision_metrics,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enrich_rollouts(
    rollout_rows: list[dict[str, Any]],
    corpus_rows: list[GRPOCorpusRow],
) -> list[dict[str, Any]]:
    """Attach evaluator targets from the immutable corpus without changing actions."""
    source_by_id = {row.task.task_id: row for row in corpus_rows}
    enriched: list[dict[str, Any]] = []
    for rollout in rollout_rows:
        task_id = str(rollout.get("task_id") or "")
        source = source_by_id.get(task_id)
        if source is None:
            raise ValueError(f"rollout task is absent from corpus: {task_id}")
        item = deepcopy(rollout)
        item["external_evidence"] = external_evidence_metadata(source)
        item["decision_loop"] = decision_loop_metadata(source)
        enriched.append(item)
    return enriched


def rescore(audit_dir: Path, corpus_file: Path, output_file: Path) -> dict[str, Any]:
    report_file = audit_dir / "report.json"
    rollouts_file = audit_dir / "rollouts.jsonl"
    report = json.loads(report_file.read_text(encoding="utf-8"))
    rollouts = enrich_rollouts(
        load_jsonl(rollouts_file),
        load_grpo_corpus(corpus_file),
    )
    result = {
        "schema_version": "curriculum-target-decision-rescore.v1",
        "scope": "offline rescore of saved actions; no model generation",
        "checkpoint": report.get("checkpoint"),
        "checkpoint_adapter_sha256": report.get("checkpoint_adapter_sha256"),
        "source_audit_report": str(report_file),
        "source_audit_report_sha256": _sha256(report_file),
        "source_rollouts": str(rollouts_file),
        "source_rollouts_sha256": _sha256(rollouts_file),
        "corpus_file": str(corpus_file),
        "corpus_sha256": _sha256(corpus_file),
        "original_behavior_gate": report.get("behavior_gate"),
        "original_behavior_gate_semantics": (
            "capability reward gate; not complete end-to-end task success"
        ),
        "target_decision_gate": target_decision_metrics(rollouts),
        "episode_outcomes": episode_outcome_metrics(rollouts),
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            rescore(args.audit_dir, args.corpus_file, args.output_file),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
