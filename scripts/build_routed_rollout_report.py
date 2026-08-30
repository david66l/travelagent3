"""Compose a frozen multi-step rollout report for student/teacher routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any


STUDENT_FAMILIES = frozenset({"clarification", "search", "recovery"})


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_routed_rollout_report(
    student_candidates: list[dict[str, Any]],
    teacher_candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected = {str(row["task_id"]) for row in student_candidates}
    selected: dict[str, dict[str, Any]] = {}
    for target, rows in (("student", student_candidates), ("teacher", teacher_candidates)):
        for row in rows:
            family = str(row["family"])
            if (target == "student") != (family in STUDENT_FAMILIES):
                continue
            task_id = str(row["task_id"])
            if task_id in selected:
                raise ValueError(f"duplicate routed task: {task_id}")
            selected[task_id] = {**row, "route_target": target}
    if set(selected) != expected:
        missing = sorted(expected - set(selected))
        extra = sorted(set(selected) - expected)
        raise ValueError(f"routed coverage mismatch: missing={missing[:5]} extra={extra[:5]}")

    rows = [selected[key] for key in sorted(selected)]
    scores = [row["score"] for row in rows]
    report = {
        "schema_version": "routed-rollout-evaluation.v1",
        "status": "passed" if all(score["successful"] for score in scores) else "failed",
        "execution_mode": "sequential-model-replay",
        "student_families": sorted(STUDENT_FAMILIES),
        "teacher_families": ["tradeoff"],
        "summary": {
            "tasks": len(rows),
            "successful_tasks": sum(bool(score["successful"]) for score in scores),
            "mean_episode_reward": round(fmean(score["episode_reward"] for score in scores), 6),
            "policy_steps": sum(int(score["policy_steps"]) for score in scores),
            "completion_tokens": sum(int(score["completion_tokens"]) for score in scores),
            "request_latency_ms": round(sum(float(score["request_latency_ms"]) for score in scores), 3),
            "mean_episode_request_latency_ms": round(
                fmean(float(score["request_latency_ms"]) for score in scores), 3
            ),
            "route_tasks": {
                "student": sum(row["route_target"] == "student" for row in rows),
                "teacher": sum(row["route_target"] == "teacher" for row in rows),
            },
            "route_policy_steps": {
                "student": sum(
                    int(row["score"]["policy_steps"])
                    for row in rows
                    if row["route_target"] == "student"
                ),
                "teacher": sum(
                    int(row["score"]["policy_steps"])
                    for row in rows
                    if row["route_target"] == "teacher"
                ),
            },
        },
    }
    return report, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-candidates", type=Path, required=True)
    parser.add_argument("--teacher-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report, rows = build_routed_rollout_report(
        load_jsonl(args.student_candidates), load_jsonl(args.teacher_candidates)
    )
    report["sources"] = {
        "student_candidates": str(args.student_candidates),
        "teacher_candidates": str(args.teacher_candidates),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "routed_candidates.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
