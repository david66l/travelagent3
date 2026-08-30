"""Create a content-free, balanced authoring packet for the 30-case pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


PILOT_PLAN = (
    ("authorized_real_or_simulated", 12),
    ("human_original_constraint", 9),
    ("tool_failure", 6),
    ("long_context_replan", 3),
)

FAMILIES = (
    "clarification",
    "candidate_search",
    "budget_tradeoff",
    "schedule_conflict",
    "accessibility",
    "family_pacing",
    "tool_recovery",
    "in_trip_replan",
)
CITIES = ("north", "east", "south", "southwest", "northwest", "multi_city")
DATE_PATTERNS = ("weekday", "weekend", "holiday", "cross_month", "weather_sensitive")
DIFFICULTIES = ("L2", "L3", "L3", "L4")
FAULT_TYPES = ("empty_result", "timeout", "rate_limit", "stale_data", "invalid_argument")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_assignments(seed: int = 20260815) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sources = [source for source, count in PILOT_PLAN for _ in range(count)]
    rng.shuffle(sources)
    assignments = []
    for index, source in enumerate(sources, start=1):
        fault = rng.choice(FAULT_TYPES) if source == "tool_failure" else None
        authoring_method = (
            "authorized_real"
            if source == "authorized_real_or_simulated" and index % 2 == 0
            else "human_original"
        )
        assignments.append(
            {
                "assignment_id": f"pilot-v1-{index:03d}",
                "source": source,
                "author_group": "external-writer-a" if index % 2 else "external-writer-b",
                "authoring_method_required": authoring_method,
                "target_family": FAMILIES[(index - 1) % len(FAMILIES)],
                "difficulty": DIFFICULTIES[(index - 1) % len(DIFFICULTIES)],
                "city_cluster": CITIES[(index - 1) % len(CITIES)],
                "date_pattern": DATE_PATTERNS[(index - 1) % len(DATE_PATTERNS)],
                "fault_type": fault,
                "required_dimensions": [
                    "at_least_two_constraints",
                    "explicit_success_boundary",
                    "natural_user_expression",
                ],
                "prohibited_inputs": [
                    "existing_training_prompt",
                    "existing_holdout_prompt",
                    "teacher_model_rewrite",
                    "personally_identifiable_information",
                ],
                "submission_status": "awaiting_independent_author",
            }
        )
    return assignments


def build_manifest(assignments: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    return {
        "schema_version": "travel-agent-external-pilot-authoring.v1",
        "status": "awaiting_independent_authors",
        "seed": seed,
        "assignments": len(assignments),
        "source_counts": dict(Counter(item["source"] for item in assignments)),
        "author_group_counts": dict(Counter(item["author_group"] for item in assignments)),
        "content_free": all("messages" not in item for item in assignments),
        "assignment_sha256": canonical_hash(assignments),
        "claim_boundary": (
            "This packet is a quota and blinding plan, not an external benchmark dataset."
        ),
    }


def render_author_packet(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# TravelAgent External Benchmark Pilot 作者任务包",
            "",
            f"> 任务槽位：{manifest['assignments']}  ",
            "> 状态：等待独立作者；本文件不包含现有训练或测试题。",
            "",
            "## 作者要求",
            "",
            "- 不查看 TravelAgent 的训练数据、内部 holdout、生成脚本或模型输出。",
            "- 按 assignment 的高层维度自行构思真实自然的旅行请求，不做模板同义改写。",
            "- 不包含姓名、手机号、订单号、精确住址、API key 或其他隐私信息。",
            "- 填写授权依据、是否去标识化、预期动作、硬约束和四档 outcome rubric。",
            "- 作者 A/B 不互看原始题目；标注阶段再由不同人员独立双标。",
            "",
            "## 提交方式",
            "",
            "每个 assignment 单独提交 `ExternalBenchmarkCase` JSON。Pilot 先进入 Dev，",
            "通过 schema、污染检测和标注校准后再决定是否纳入正式 100 条 Dev。",
            "400 条 Sealed Test 使用新的 assignment，不从 Pilot 复制或改写。",
            "",
            f"Assignment SHA-256：`{manifest['assignment_sha256']}`",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/agentic/datasets/external-benchmark-v1/pilot-authoring-v1"),
    )
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    assignments = build_assignments(args.seed)
    manifest = build_manifest(assignments, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "assignments.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in assignments) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "AUTHOR_PACKET.md").write_text(
        render_author_packet(manifest), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
