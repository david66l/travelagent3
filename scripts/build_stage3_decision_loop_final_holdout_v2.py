"""Build the frozen, orthogonal Stage 3 decision-loop final holdout v2.

The holdout uses fresh source tasks and template families, and verifies that
its model-visible failures, task identities, and environment fingerprints do
not overlap any V3 development split.
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from agentic.environment import environment_fingerprint  # noqa: E402
from agentic.grpo_training import load_grpo_corpus  # noqa: E402
from build_stage3_decision_loop_curriculum import (  # noqa: E402
    derive_decision_loop_case,
)
from build_stage3_decision_loop_curriculum_v3 import (  # noqa: E402
    _TEMPLATES as V3_TEMPLATES,
)
from build_stage3_multiturn_rl_corpus import _next_retry_source  # noqa: E402


SCHEMA_VERSION = "stage3-decision-loop-final-holdout.v2"
FACTOR_SCHEDULE = "orthogonal-latin-rotation.v2"
_STRATA = (
    ("change_arguments", "explicit_instruction"),
    ("retry_same_arguments", "explicit_instruction"),
    ("change_arguments", "diagnostic_evidence"),
    ("retry_same_arguments", "diagnostic_evidence"),
)

_HOLDOUT_TEMPLATES = {
    "change_arguments/explicit_instruction": (
        "请将后续检索限定为“{target}”，同时撤下“{drop}”这一条件。",
        "再次查询时只发送“{target}”；“{drop}”不应出现在新参数中。",
        "下一轮把“{target}”设为唯一关键词，并从组合里剔除“{drop}”。",
        "调整请求为单项“{target}”，不再沿用“{drop}”。",
    ),
    "retry_same_arguments/explicit_instruction": (
        "条件审核没有发现问题；节点恢复后请完整复用刚才的关键词集合。",
        "无需重写这次请求，待临时中断解除后按原参数再次提交。",
        "本轮仅执行链路异常；下一次调用应保持所有查询项不变。",
        "输入内容已经确认可用，请把同一份参数重新投递一次。",
    ),
    "change_arguments/diagnostic_evidence": (
        "主题聚合结果表明“{drop}”扩大了无关分支，而“{target}”仍贴合核心意图。",
        "语义检索报告将“{drop}”识别为扩散源，“{target}”的精度表现仍然稳定。",
        "候选聚类受到“{drop}”的宽泛含义干扰，“{target}”仍落在高相关区域。",
        "相关度审计发现“{drop}”引入大面积偏题结果，“{target}”本身保持聚焦。",
    ),
    "retry_same_arguments/diagnostic_evidence": (
        "请求在参数审核后进入运行阶段，随后因短时资源抖动没有产出结果。",
        "服务已经登记当前查询，候选汇总环节临时中止，返回内容为空。",
        "输入项顺利通过预处理，执行集群之后发生可恢复的瞬时故障。",
        "查询成功进入计算流程，但结果写回前连接短暂断开。",
    ),
}


def _factor_index(local_index: int) -> int:
    return (local_index % 4 + local_index // 4) % 4


def _failure_messages(rows: list[Any]) -> set[str]:
    return {
        str(response.fallback_reason)
        for row in rows
        for response in row.snapshot.tool_responses.get("search_pois") or []
        if response.fallback_reason
    }


def _template_literals() -> set[str]:
    return {template for templates in _HOLDOUT_TEMPLATES.values() for template in templates}


def _v3_template_literals() -> set[str]:
    return {
        template
        for split_templates in V3_TEMPLATES.values()
        for templates in split_templates.values()
        for template in templates
    }


def _load_development_rows(development_dir: Path) -> list[Any]:
    paths = [development_dir / f"{split}.jsonl" for split in ("train", "validation", "test")]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing V3 development splits: {missing}")
    return [row for path in paths for row in load_grpo_corpus(path)]


def _build_rows(*, start_index: int, count: int) -> tuple[list[Any], int]:
    rows: list[Any] = []
    cursor = start_index
    occurrences: Counter[int] = Counter()
    for local_index in range(count):
        cursor, source = _next_retry_source(cursor)
        factor_index = _factor_index(local_index)
        scenario, evidence_style = _STRATA[factor_index]
        occurrence = occurrences[factor_index]
        occurrences[factor_index] += 1
        target_position = (occurrence // 4) % 2
        key = f"{scenario}/{evidence_style}"
        templates = _HOLDOUT_TEMPLATES[key]
        template_index = (occurrence // 8 + occurrence % 4) % len(templates)
        template = templates[template_index]
        change_templates = _HOLDOUT_TEMPLATES[f"change_arguments/{evidence_style}"]
        retry_templates = _HOLDOUT_TEMPLATES[f"retry_same_arguments/{evidence_style}"]
        row = derive_decision_loop_case(
            source,
            ordinal=200000 + local_index,
            scenario=scenario,
            evidence_style=evidence_style,
            change_message_template=(
                template if scenario == "change_arguments" else change_templates[template_index]
            ),
            timeout_message=(
                template
                if scenario == "retry_same_arguments"
                else retry_templates[template_index]
            ),
            target_index=target_position,
        )
        row.task.task_id += "-final-holdout-v2"
        row.task.template_family += "-final-holdout-v2"
        row.snapshot.environment_version = SCHEMA_VERSION
        row.snapshot.snapshot_version = SCHEMA_VERSION
        row.snapshot.state_id += "-final-holdout-v2"
        metadata = row.snapshot.hidden_test_facts["decision_loop_curriculum"]
        metadata.update(
            {
                "holdout_tier": "final_v2",
                "factor_schedule": FACTOR_SCHEDULE,
                "target_position": target_position,
                "failure_template_id": f"final-v2:{key}:{template_index}",
                "failure_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            }
        )
        rows.append(row)
    return rows, cursor


def _coverage(rows: list[Any]) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        metadata = row.snapshot.hidden_test_facts["decision_loop_curriculum"]
        grouped[f"{metadata['scenario']}/{metadata['evidence_style']}"].append(row)
    coverage: dict[str, Any] = {}
    for stratum, items in sorted(grouped.items()):
        metadata = [item.snapshot.hidden_test_facts["decision_loop_curriculum"] for item in items]
        coverage[stratum] = {
            "tasks": len(items),
            "cities": dict(sorted(Counter(item.task.slots["destination"] for item in items).items())),
            "target_positions": dict(
                sorted(Counter(str(item["target_position"]) for item in metadata).items())
            ),
            "template_ids": dict(
                sorted(Counter(item["failure_template_id"] for item in metadata).items())
            ),
            "template_sha256": sorted({item["failure_template_sha256"] for item in metadata}),
        }
    return coverage


def _validate_coverage(coverage: dict[str, Any], *, tasks_per_stratum: int) -> None:
    expected_strata = {f"{scenario}/{style}" for scenario, style in _STRATA}
    if set(coverage) != expected_strata:
        raise ValueError("final holdout does not contain all four semantic strata")
    for stratum, facts in coverage.items():
        if facts["tasks"] != tasks_per_stratum:
            raise ValueError(f"{stratum} has an unexpected task count")
        if len(facts["cities"]) != 4:
            raise ValueError(f"{stratum} does not cover all four cities")
        if set(facts["target_positions"]) != {"0", "1"}:
            raise ValueError(f"{stratum} does not balance target position")
        if len(facts["template_ids"]) < 4:
            raise ValueError(f"{stratum} does not cover four fresh templates")


def _write_jsonl(path: Path, rows: list[Any]) -> str:
    path.write_text(
        "\n".join(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    output_dir: Path,
    *,
    development_dir: Path,
    start_index: int = 120000,
    count: int = 128,
) -> dict[str, Any]:
    if count < 128 or count % 128:
        raise ValueError("final holdout v2 count must be a positive multiple of 128")
    if _template_literals() & _v3_template_literals():
        raise ValueError("final holdout template literals overlap V3 curriculum")

    development_rows = _load_development_rows(development_dir)
    rows, cursor = _build_rows(start_index=start_index, count=count)
    tasks_per_stratum = count // 4
    coverage = _coverage(rows)
    _validate_coverage(coverage, tasks_per_stratum=tasks_per_stratum)

    development_task_ids = {row.task.task_id for row in development_rows}
    task_ids = {row.task.task_id for row in rows}
    development_fingerprints = {
        environment_fingerprint(row.task, row.snapshot) for row in development_rows
    }
    fingerprints = {environment_fingerprint(row.task, row.snapshot) for row in rows}
    development_messages = _failure_messages(development_rows)
    messages = _failure_messages(rows)
    development_source_ids = {
        str(row.snapshot.hidden_test_facts["decision_loop_curriculum"].get("source_task_id"))
        for row in development_rows
    }
    source_ids = {
        str(row.snapshot.hidden_test_facts["decision_loop_curriculum"].get("source_task_id"))
        for row in rows
    }
    development_template_ids = {
        str(row.snapshot.hidden_test_facts["decision_loop_curriculum"].get("failure_template_id"))
        for row in development_rows
    }
    template_ids = {
        str(row.snapshot.hidden_test_facts["decision_loop_curriculum"]["failure_template_id"])
        for row in rows
    }
    leakage = {
        "task_overlap": sorted(task_ids & development_task_ids),
        "environment_fingerprint_overlap": sorted(fingerprints & development_fingerprints),
        "failure_message_overlap": sorted(messages & development_messages),
        "source_task_overlap": sorted(source_ids & development_source_ids),
        "failure_template_id_overlap": sorted(template_ids & development_template_ids),
        "template_literal_overlap": sorted(_template_literals() & _v3_template_literals()),
    }
    if any(leakage.values()):
        raise ValueError(f"final holdout v2 leaks V3 development data: {leakage}")
    if len(task_ids) != count or len(fingerprints) != count:
        raise ValueError("final holdout v2 contains duplicate tasks or environments")

    output_dir.mkdir(parents=True, exist_ok=True)
    test_path = output_dir / "test.jsonl"
    test_sha256 = _write_jsonl(test_path, rows)
    development_manifest = development_dir / "manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scope": "frozen orthogonal final holdout; never use for training or checkpoint selection",
        "development_dir": str(development_dir),
        "development_manifest_sha256": (
            hashlib.sha256(development_manifest.read_bytes()).hexdigest()
            if development_manifest.is_file()
            else None
        ),
        "factor_schedule": FACTOR_SCHEDULE,
        "start_index": start_index,
        "next_unused_index": cursor,
        "count": count,
        "coverage": coverage,
        "test_sha256": test_sha256,
        "leakage": leakage,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=120000)
    parser.add_argument("--count", type=int, default=128)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.output_dir,
                development_dir=args.development_dir,
                start_index=args.start_index,
                count=args.count,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
