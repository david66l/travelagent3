"""Freeze policy-visible 4B/8B subsets for Stage30 stability benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_stage29_model_comparison_report import route_to_teacher


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_cases(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    student = []
    teacher = []
    for case in cases:
        context = json.loads(case["messages"][1]["content"])
        (teacher if route_to_teacher(context) else student).append(case)
    return student, teacher


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "ml/agentic/datasets/external-benchmark-v1/"
            "deepseek-v4-flash-stage29-v1/vllm-cases.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "ml/agentic/datasets/external-benchmark-v1/"
            "deepseek-v4-flash-stage29-v1/stage30-routed-v1"
        ),
    )
    args = parser.parse_args()
    cases = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(cases) != 150 or len({case["case_id"] for case in cases}) != 150:
        raise ValueError("Stage30 requires the frozen 150 unique Stage29 cases")
    student, teacher = split_cases(cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("student-cases.jsonl", student), ("teacher-cases.jsonl", teacher)):
        (args.output_dir / name).write_text(
            "\n".join(json.dumps(case, ensure_ascii=False) for case in values) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "schema_version": "travel-agent-stage30-routed-cases.v1",
        "source_cases": len(cases),
        "student_cases": len(student),
        "teacher_cases": len(teacher),
        "route_policy": (
            "8B for infeasible/unsafe or exhausted non-retryable tool state; "
            "Stage28 DPO 4B otherwise"
        ),
        "uses_gold_label": False,
        "source_content_sha256": _canonical_hash(cases),
        "student_case_ids_sha256": _canonical_hash(
            sorted(case["case_id"] for case in student)
        ),
        "teacher_case_ids_sha256": _canonical_hash(
            sorted(case["case_id"] for case in teacher)
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
