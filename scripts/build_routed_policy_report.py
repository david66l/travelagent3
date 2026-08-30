"""Compose a frozen student/teacher routing result from per-model HTTP runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from core.inference_metrics import summarize_inference_metrics


STUDENT_FAMILIES = frozenset({"clarification", "search", "recovery"})


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["case_id"]), int(row["repetition"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_routed_report(
    student_runs: list[dict[str, Any]],
    teacher_runs: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select bounded families from the student and tradeoff from the teacher."""
    expected = {_key(row) for row in student_runs}
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    for target, rows in (("student", student_runs), ("teacher", teacher_runs)):
        for row in rows:
            family = str(row["family"])
            if (target == "student") != (family in STUDENT_FAMILIES):
                continue
            key = _key(row)
            if key in selected:
                raise ValueError(f"duplicate routed run: {key}")
            selected[key] = {**row, "route_target": target}

    missing = sorted(expected - set(selected))
    extra = sorted(set(selected) - expected)
    if missing or extra:
        raise ValueError(f"routed coverage mismatch: missing={missing[:5]} extra={extra[:5]}")

    routed = [selected[key] for key in sorted(selected)]
    metrics = [row["inference_metrics"] for row in routed if row.get("inference_metrics")]
    report = {
        "schema_version": "routed-policy-evaluation.v1",
        "status": "passed" if all(row.get("success") for row in routed) else "failed",
        "routing_contract": {
            "student_families": sorted(STUDENT_FAMILIES),
            "teacher_families": ["tradeoff"],
            "execution_mode": "sequential-model-replay",
        },
        "summary": {
            "runs": len(routed),
            "successful_runs": sum(bool(row.get("success")) for row in routed),
            "action_mismatches": sum(bool(row.get("action_mismatch")) for row in routed),
            "argument_mismatches": sum(bool(row.get("argument_mismatch")) for row in routed),
            "http_errors": sum(bool(row.get("http_error")) for row in routed),
            "route_counts": {
                "student": sum(row["route_target"] == "student" for row in routed),
                "teacher": sum(row["route_target"] == "teacher" for row in routed),
            },
            "inference": summarize_inference_metrics(metrics),
        },
    }
    return report, routed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-runs", type=Path, required=True)
    parser.add_argument("--teacher-runs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report, runs = build_routed_report(
        load_jsonl(args.student_runs),
        load_jsonl(args.teacher_runs),
    )
    report["sources"] = {
        "student_runs": str(args.student_runs),
        "student_runs_sha256": _sha256(args.student_runs),
        "teacher_runs": str(args.teacher_runs),
        "teacher_runs_sha256": _sha256(args.teacher_runs),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runs.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in runs),
        encoding="utf-8",
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
