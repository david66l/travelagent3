"""Freeze a never-trained final holdout for decision-loop recovery."""

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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from agentic.environment import environment_fingerprint  # noqa: E402
from agentic.grpo_training import load_grpo_corpus  # noqa: E402
from build_stage3_decision_loop_curriculum import derive_decision_loop_case  # noqa: E402
from build_stage3_multiturn_rl_corpus import _next_retry_source  # noqa: E402


SCHEMA_VERSION = "stage3-decision-loop-final-holdout.v1"

_EXPLICIT_CHANGE = (
    "本次组合检索范围失控；下一次仅提交“{target}”，并去除“{drop}”。",
    "请把搜索条件缩减到“{target}”一个词，不再携带“{drop}”。",
)
_EXPLICIT_TIMEOUT = (
    "请求参数已确认有效；这是一次瞬态执行超时，请原封不动重放该查询。",
    "无需调整查询条件，本轮只因上游短暂故障失败；使用同一参数再次调用。",
)
_DIAGNOSTIC_CHANGE = (
    "召回分析：“{drop}”对应候选高度发散，“{target}”仍具有可靠选择性；本轮无结果。",
    "查询质量门显示“{drop}”引入主要噪声，“{target}”的相关性信号正常；候选被丢弃。",
    "检索画像中“{drop}”覆盖面异常宽，“{target}”仍能区分用户偏好；当前组合未通过。",
)
_DIAGNOSTIC_TIMEOUT = (
    "参数检查和请求接收均已完成，候选计算节点随后瞬时超时，因此结果为空。",
    "服务端已接受当前查询条件，但检索执行链路临时中断，没有生成候选。",
    "本次请求通过输入校验，进入执行队列后遇到短暂网络故障，未返回结果。",
)


def _failure_messages(rows: list[Any]) -> set[str]:
    return {
        str(response.fallback_reason)
        for row in rows
        for response in row.snapshot.tool_responses.get("search_pois") or []
        if response.fallback_reason
    }


def build(
    output_dir: Path,
    *,
    development_dir: Path,
    start_index: int = 90000,
    count: int = 128,
) -> dict[str, Any]:
    if count < 4 or count % 4:
        raise ValueError("final holdout count must be a positive multiple of four")
    development_rows = [
        *load_grpo_corpus(development_dir / "train.jsonl"),
        *load_grpo_corpus(development_dir / "validation.jsonl"),
        *load_grpo_corpus(development_dir / "test.jsonl"),
    ]
    development_ids = {row.task.task_id for row in development_rows}
    development_fingerprints = {
        environment_fingerprint(row.task, row.snapshot) for row in development_rows
    }
    development_messages = _failure_messages(development_rows)

    rows = []
    cursor = start_index
    for ordinal in range(count):
        cursor, source = _next_retry_source(cursor)
        scenario = "change_arguments" if ordinal % 2 == 0 else "retry_same_arguments"
        evidence_style = (
            "explicit_instruction" if (ordinal // 2) % 2 == 0 else "diagnostic_evidence"
        )
        change_templates = (
            _EXPLICIT_CHANGE if evidence_style == "explicit_instruction" else _DIAGNOSTIC_CHANGE
        )
        timeout_messages = (
            _EXPLICIT_TIMEOUT
            if evidence_style == "explicit_instruction"
            else _DIAGNOSTIC_TIMEOUT
        )
        row = derive_decision_loop_case(
            source,
            ordinal=ordinal,
            scenario=scenario,
            evidence_style=evidence_style,
            change_message_template=change_templates[ordinal % len(change_templates)],
            timeout_message=timeout_messages[ordinal % len(timeout_messages)],
        )
        row.task.task_id = f"{row.task.task_id}-final-holdout"
        row.task.template_family = f"{row.task.template_family}-final-holdout"
        row.snapshot.environment_version = SCHEMA_VERSION
        row.snapshot.snapshot_version = SCHEMA_VERSION
        row.snapshot.state_id = f"{row.snapshot.state_id}-final-holdout"
        row.snapshot.hidden_test_facts["decision_loop_curriculum"]["holdout_tier"] = (
            "final"
        )
        rows.append(row)

    ids = {row.task.task_id for row in rows}
    fingerprints = {environment_fingerprint(row.task, row.snapshot) for row in rows}
    messages = _failure_messages(rows)
    if len(ids) != len(rows) or len(fingerprints) != len(rows):
        raise ValueError("final holdout contains duplicate tasks or environments")
    if ids & development_ids:
        raise ValueError("final holdout task IDs overlap development data")
    if fingerprints & development_fingerprints:
        raise ValueError("final holdout environments overlap development data")
    if messages & development_messages:
        raise ValueError("final holdout failure messages overlap development data")

    output_dir.mkdir(parents=True, exist_ok=True)
    test_path = output_dir / "test.jsonl"
    test_path.write_text(
        "\n".join(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    strata = Counter(
        (
            row.snapshot.hidden_test_facts["decision_loop_curriculum"]["scenario"],
            row.snapshot.hidden_test_facts["decision_loop_curriculum"]["evidence_style"],
        )
        for row in rows
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scope": "frozen final holdout; never use for training or checkpoint selection",
        "development_dir": str(development_dir),
        "start_index": start_index,
        "next_unused_index": cursor,
        "count": len(rows),
        "strata": {
            f"{scenario}/{style}": value
            for (scenario, style), value in sorted(strata.items())
        },
        "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "development_task_overlap": [],
        "development_environment_overlap": [],
        "development_failure_message_overlap": [],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=90000)
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
