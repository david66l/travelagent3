"""Audit whether verifier-repair evaluation can be solved by shallow shortcuts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = ROOT / "backend" / "src"
for candidate in (str(ROOT), str(BACKEND_SRC)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from agentic.grpo_training import GRPOCorpusRow, load_grpo_corpus  # noqa: E402
from agentic.verifier_repair import (  # noqa: E402
    constraint_handling_for_violation,
    parse_constraint_flexibility,
)


_ACTION_CUES = {
    "retry_solve": ("重试", "重算", "再算", "换策略", "切换算法", "有界重排"),
    "propose_tradeoff": ("让我选择", "给我选项", "先问我", "让我决定"),
    "abort": ("停止", "终止", "结束规划", "不要继续", "不要编造", "无解就"),
}


def _target(row: GRPOCorpusRow) -> str:
    return str(row.snapshot.hidden_test_facts["grpo_decision_state"]["target_action"])


def _majority_feature_accuracy(
    rows: list[GRPOCorpusRow],
    feature: Callable[[GRPOCorpusRow], str],
) -> float:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[feature(row)][_target(row)] += 1
    predictions = {
        key: sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]
        for key, counter in counts.items()
    }
    return sum(predictions[feature(row)] == _target(row) for row in rows) / len(rows)


def _lexical_prediction(row: GRPOCorpusRow) -> str:
    text = row.task.user_request
    scores = {
        action: sum(cue in text for cue in cues)
        for action, cues in _ACTION_CUES.items()
    }
    best = max(scores.values())
    if best == 0:
        return "abort"
    winners = sorted(action for action, score in scores.items() if score == best)
    return winners[0]


def _deterministic_contract_router_prediction(row: GRPOCorpusRow) -> str:
    """Apply the production controller rule without using a learned policy."""
    hard_violations = row.snapshot.tool_responses["validate_itinerary"][0].data.get(
        "hard_violations"
    )
    if not isinstance(hard_violations, list) or len(hard_violations) != 1:
        return "abort"
    contract = parse_constraint_flexibility(
        row.task.slots.get("constraint_flexibility")
    )
    if contract is None:
        return "abort"
    handling = constraint_handling_for_violation(
        str(hard_violations[0].get("code") or ""),
        contract,
    )
    return {
        "solver_adjustable": "retry_solve",
        "relaxable": "propose_tradeoff",
        "locked": "abort",
    }.get(handling, "abort")


def audit(rows: list[GRPOCorpusRow], *, maximum_shortcut_accuracy: float = 0.5) -> dict[str, Any]:
    if not rows:
        raise ValueError("shortcut audit requires at least one row")
    lexical_accuracy = sum(_lexical_prediction(row) == _target(row) for row in rows) / len(rows)
    code_accuracy = _majority_feature_accuracy(
        rows,
        lambda row: str(
            row.snapshot.tool_responses["validate_itinerary"][0]
            .data["hard_violations"][0]["code"]
        ),
    )
    contract_accuracy = _majority_feature_accuracy(
        rows,
        lambda row: json.dumps(
            row.task.slots.get("constraint_flexibility") or {},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    accuracies = {
        "direct_action_cue": lexical_accuracy,
        "violation_code_majority": code_accuracy,
        "contract_majority": contract_accuracy,
    }
    deterministic_router_accuracy = sum(
        _deterministic_contract_router_prediction(row) == _target(row) for row in rows
    ) / len(rows)
    return {
        "schema_version": "verifier-repair-shortcut-audit.v2",
        "rows": len(rows),
        "source_clusters": len(
            {
                (
                    row.snapshot.hidden_test_facts["grpo_decision_state"]["source_task_id"],
                    row.snapshot.hidden_test_facts["grpo_decision_state"][
                        "source_snapshot_version"
                    ],
                )
                for row in rows
            }
        ),
        "target_counts": dict(sorted(Counter(_target(row) for row in rows).items())),
        "accuracies": {key: round(value, 6) for key, value in accuracies.items()},
        "strong_baselines": {
            "deterministic_contract_router": round(deterministic_router_accuracy, 6),
        },
        "maximum_shortcut_accuracy": maximum_shortcut_accuracy,
        "passed": all(value <= maximum_shortcut_accuracy for value in accuracies.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--maximum-shortcut-accuracy", type=float, default=0.5)
    args = parser.parse_args()
    report = audit(
        load_grpo_corpus(args.corpus_file),
        maximum_shortcut_accuracy=args.maximum_shortcut_accuracy,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
