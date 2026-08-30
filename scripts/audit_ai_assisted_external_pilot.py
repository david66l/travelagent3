"""Audit the AI-assisted pilot for calibration use, never external claims."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))
sys.path.insert(0, str(REPO_ROOT))

from evaluation.external_benchmark import (  # noqa: E402
    BenchmarkSource,
    BenchmarkSplit,
    ExternalBenchmarkCase,
    ForbiddenCorpusDocument,
    audit_training_contamination,
    normalized_prompt_hash,
)
from scripts.audit_external_benchmark import read_forbidden_documents  # noqa: E402


DEFAULT_FORBIDDEN_CORPORA = (
    "ml/agentic/datasets/qwen3-stage19-holdout-v1/regular.jsonl",
    "ml/agentic/datasets/qwen3-stage19-holdout-v1/hard.jsonl",
    "ml/agentic/datasets/qwen3-stage19-holdout-v1/adversarial.jsonl",
    "ml/agentic/datasets/qwen3-stage20-teacher-sft-reverified-v3/sft/train.jsonl",
    "ml/agentic/datasets/qwen3-stage21-student-sft-balanced-v1/train.jsonl",
    "ml/agentic/datasets/qwen3-stage22-preferences-balanced-v1/train.jsonl",
)

_PHONE = re.compile(r"(?<![A-Za-z0-9])(?:\+?86[- ]?)?1[3-9]\d{9}(?![A-Za-z0-9])")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_ID_CARD = re.compile(r"(?<![A-Za-z0-9])\d{17}[\dXx](?![A-Za-z0-9])")


def read_cases(path: Path) -> list[ExternalBenchmarkCase]:
    return [
        ExternalBenchmarkCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def contains_pii(case: ExternalBenchmarkCase) -> bool:
    payload = case.model_dump_json()
    return any(pattern.search(payload) for pattern in (_PHONE, _EMAIL, _ID_CARD))


def audit_pilot(
    cases: list[ExternalBenchmarkCase], documents: list[ForbiddenCorpusDocument]
) -> dict[str, Any]:
    contamination = audit_training_contamination(cases, documents)
    action_counts = Counter(case.annotations[0].primary_action for case in cases)
    family_counts = Counter(case.group_keys.request_family for case in cases)
    gates = {
        "cases_30": len(cases) == 30,
        "dev_only": all(case.split == BenchmarkSplit.DEV for case in cases),
        "unique_case_ids": len({case.case_id for case in cases}) == len(cases),
        "unique_normalized_prompts": len({normalized_prompt_hash(case) for case in cases})
        == len(cases),
        "synthetic_provenance_explicit": all(
            case.provenance.authoring_method == "simulated"
            and not case.provenance.template_independent
            and case.provenance.author_group == "codex-ai-assisted-draft"
            for case in cases
        ),
        "single_draft_annotation_only": all(len(case.annotations) == 1 for case in cases),
        "tool_failure_6": sum(
            case.source == BenchmarkSource.TOOL_FAILURE for case in cases
        )
        == 6,
        "long_context_3": sum(
            case.source == BenchmarkSource.LONG_CONTEXT_REPLAN for case in cases
        )
        == 3,
        "action_coverage": (
            action_counts["ask_user"] >= 5
            and action_counts["search_pois"] >= 5
            and action_counts["propose_tradeoff"] >= 5
            and action_counts["abort"] >= 3
        ),
        "no_pii": not any(contains_pii(case) for case in cases),
        "forbidden_corpora_registered": bool(documents),
        "no_training_contamination": contamination["passed"],
    }
    return {
        "schema_version": "travel-agent-ai-assisted-pilot-audit.v1",
        "status": "passed_for_schema_calibration" if all(gates.values()) else "blocked",
        "passed": all(gates.values()),
        "eligible_for_external_claim": False,
        "gates": gates,
        "counts": {
            "cases": len(cases),
            "families": dict(family_counts),
            "primary_actions": dict(action_counts),
        },
        "training_contamination": contamination,
        "claim_boundary": (
            "Passing this audit permits evaluator calibration only. It does not establish "
            "independent authorship, double annotation, or an external benchmark result."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(
            "ml/agentic/datasets/external-benchmark-v1/ai-assisted-pilot-v1/cases.jsonl"
        ),
    )
    parser.add_argument("--forbidden-corpus", type=Path, action="append", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "ml/agentic/datasets/external-benchmark-v1/ai-assisted-pilot-v1/audit.json"
        ),
    )
    args = parser.parse_args()
    corpus_paths = args.forbidden_corpus or [Path(path) for path in DEFAULT_FORBIDDEN_CORPORA]
    report = audit_pilot(read_cases(args.cases), read_forbidden_documents(corpus_paths))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "gates": report["gates"]}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
