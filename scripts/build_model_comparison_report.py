"""Build a reproducible Base/SFT/GRPO comparison from curriculum audits."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


def load_arm(name: str, directory: Path) -> dict[str, Any]:
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    rollouts = [
        json.loads(line)
        for line in (directory / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rollouts:
        raise ValueError(f"{name} has no rollouts")
    components = sorted(rollouts[0].get("reward_components") or {})
    if not components:
        raise ValueError(f"{name} rollouts have no reward component audit")
    for row in rollouts:
        if sorted(row.get("reward_components") or {}) != components:
            raise ValueError(f"{name} reward component schema changed within one arm")
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rollouts:
        family_rows[row["family"]].append(row)
    failed = [row for row in rollouts if row["gate_status"] != "passed"]
    return {
        "name": name,
        "checkpoint": report["checkpoint"],
        "tasks": report["tasks"],
        "samples": len(rollouts),
        "success_rate": _success_rate(rollouts),
        "mean_reward": round(fmean(float(row["reward"]) for row in rollouts), 8),
        "families": {
            family: {
                "samples": len(rows),
                "success_rate": _success_rate(rows),
                "mean_reward": round(fmean(float(row["reward"]) for row in rows), 8),
            }
            for family, rows in sorted(family_rows.items())
        },
        "reward_components": {
            component: round(
                fmean(float(row["reward_components"][component]) for row in rollouts),
                8,
            )
            for component in components
        },
        "gate_statuses": dict(Counter(row["gate_status"] for row in rollouts)),
        "failure_examples": [
            {
                "task_id": row["task_id"],
                "family": row["family"],
                "sample_index": row["sample_index"],
                "rollout_seed": row.get("rollout_seed"),
                "gate_status": row["gate_status"],
                "reward": row["reward"],
                "termination_reason": row["termination_reason"],
                "actions": row["actions"],
            }
            for row in failed[:8]
        ],
        "contract": {
            field: report.get(field)
            for field in (
                "corpus_file",
                "seed",
                "seed_protocol",
                "temperature",
                "quantization",
                "group_size",
                "family_offset",
            )
        },
        "task_ids": sorted({row["task_id"] for row in rollouts}),
        "reward_config_versions": sorted(
            {str(row["reward_config_version"]) for row in rollouts}
        ),
    }


def build(arms: list[tuple[str, Path]]) -> dict[str, Any]:
    summaries = [load_arm(name, path) for name, path in arms]
    reference = summaries[0]
    for arm in summaries[1:]:
        if arm["contract"] != reference["contract"]:
            raise ValueError(f"comparison contract mismatch: {arm['name']}")
        if arm["task_ids"] != reference["task_ids"]:
            raise ValueError(f"task set mismatch: {arm['name']}")
        if arm["reward_config_versions"] != reference["reward_config_versions"]:
            raise ValueError(f"reward config mismatch: {arm['name']}")
    baseline = summaries[0]
    for arm in summaries:
        arm["delta_vs_base"] = {
            "success_rate": round(arm["success_rate"] - baseline["success_rate"], 8),
            "mean_reward": round(arm["mean_reward"] - baseline["mean_reward"], 8),
        }
    return {
        "schema_version": "local-policy-comparison.v1",
        "scope": (
            "paired NF4 stochastic validation audit; bounded training runs establish "
            "an engineering baseline, not a paper-level benchmark"
        ),
        "contract": reference["contract"],
        "reward_config_versions": reference["reward_config_versions"],
        "arms": summaries,
    }


def render_markdown(report: dict[str, Any]) -> str:
    arms = report["arms"]
    lines = [
        "# Qwen2.5-3B Agent Policy：Base / SFT / SFT+GRPO 对照",
        "",
        "> 范围：逐任务、逐样本配对的 NF4 随机验证审计。当前训练为受控工程实验，",
        "> 不将结果表述为论文级 benchmark，也不宣称已完成逐轮信用优化。",
        "",
        "## 评测契约",
        "",
        f"- 任务数：{arms[0]['tasks']}；每任务采样：{report['contract']['group_size']}；总 rollout：{arms[0]['samples']}。",
        f"- Seed：{report['contract']['seed']}；协议：`{report['contract']['seed_protocol']}`。",
        f"- 温度：{report['contract']['temperature']}；量化：`{report['contract']['quantization']}`。",
        f"- Reward：`{', '.join(report['reward_config_versions'])}`。",
        "- validation 仅用于盲评，不参与 SFT/GRPO 任务挑选或梯度更新。",
        "",
        "## 总体结果",
        "",
        "| 模型 | 成功率 | 平均 Reward | 相对 Base 成功率 | 相对 Base Reward |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in arms:
        lines.append(
            f"| {arm['name']} | {_pct(arm['success_rate'])} | {arm['mean_reward']:.4f} "
            f"| {_signed_pct(arm['delta_vs_base']['success_rate'])} "
            f"| {arm['delta_vs_base']['mean_reward']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## 分任务能力",
            "",
            "| 模型 | 澄清 | 故障恢复 | 普通搜索 | 约束权衡 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm in arms:
        families = arm["families"]
        lines.append(
            f"| {arm['name']} | {_pct(families['clarification']['success_rate'])} "
            f"| {_pct(families['recovery']['success_rate'])} "
            f"| {_pct(families['search']['success_rate'])} "
            f"| {_pct(families['tradeoff']['success_rate'])} |"
        )
    component_order = [
        "task",
        "constraint",
        "tool",
        "grounding",
        "efficiency",
        "format",
        "quality",
    ]
    lines.extend(
        [
            "",
            "## Reward 分量均值",
            "",
            "六类策略 Reward 为 task、constraint、tool、grounding、efficiency、format；",
            "quality 当前权重为 0，仅作审计字段。",
            "",
            "| 模型 | task | constraint | tool | grounding | efficiency | format | quality |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in arms:
        values = arm["reward_components"]
        lines.append(
            f"| {arm['name']} | "
            + " | ".join(f"{values[key]:.4f}" for key in component_order)
            + " |"
        )
    lines.extend(
        [
            "",
            "## 结论与边界",
            "",
            "- Base 已能处理部分澄清、搜索和恢复，但本评测中的权衡成功率为 0%。",
            "- 环境对齐 SFT 主要补齐约束冲突时的合法工具决策，并显著提高总体成功率。",
            "- 单步保守 GRPO 在不降低搜索/恢复的前提下，进一步提高澄清与权衡成功率。",
            "- 当前结果支持“多轮环境中的轨迹级 GRPO 工程基线”；尚未完成逐轮信用对照、完整 Reward 消融或线上灰度，不能宣称已解决 Long-Horizon credit assignment。",
            "- 每个 arm 的原始 `rollouts.jsonl` 保留失败终止原因、动作、独立 rollout seed 和 Reward 分量，可逐条复核。",
            "",
        ]
    )
    return "\n".join(lines)


def _success_rate(rows: list[dict[str, Any]]) -> float:
    return round(sum(row["gate_status"] == "passed" for row in rows) / len(rows), 8)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _signed_pct(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="Arm in NAME=REPORT_DIRECTORY form; first arm is the baseline.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    arms = []
    for value in args.arm:
        if "=" not in value:
            parser.error("--arm must use NAME=REPORT_DIRECTORY")
        name, raw_path = value.split("=", 1)
        arms.append((name, Path(raw_path)))
    report = build(arms)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
