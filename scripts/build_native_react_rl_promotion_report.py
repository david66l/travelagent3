"""Build the frozen SFT/GRPO promotion evidence for the native ReAct policy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DECISION_REPORTS = {
    "sft": "decision-state-history-validation-bridge-v1",
    "grpo": "decision-state-history-validation-grpo-kl001-lr5e7-v3",
}
FULL_LOOP_SEGMENTS = [
    (
        "full-history-validation-bridge-controller-owned-v2",
        "full-history-validation-grpo-kl001-first8-8x4-v5",
    ),
    (
        "full-history-validation-bridge-offset8-12x4-v3",
        "full-history-validation-grpo-kl001-offset8-12x4-v5",
    ),
    (
        "full-history-validation-bridge-offset20-3x4-v4",
        "full-history-validation-grpo-kl001-offset20-3x4-v5",
    ),
]


def _load_report(root: Path, name: str) -> dict[str, Any]:
    return json.loads((root / name / "report.json").read_text(encoding="utf-8"))


def _rollout_outcomes(root: Path, name: str) -> dict[tuple[str, int], bool]:
    rows = (root / name / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
    return {
        (row["task_id"], int(row["sample_index"])): row["gate_status"] == "passed"
        for row in map(json.loads, rows)
    }


def _exact_mcnemar(losses: int, wins: int) -> float:
    discordant = losses + wins
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(losses, wins) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _paired(root: Path, before: str, after: str) -> dict[str, Any]:
    left = _rollout_outcomes(root, before)
    right = _rollout_outcomes(root, after)
    if left.keys() != right.keys():
        missing_after = sorted(left.keys() - right.keys())
        missing_before = sorted(right.keys() - left.keys())
        raise ValueError(
            "paired rollout keys differ: "
            f"missing_after={missing_after[:5]}, missing_before={missing_before[:5]}"
        )
    keys = sorted(left)
    regressions = sum(left[key] and not right[key] for key in keys)
    improvements = sum(not left[key] and right[key] for key in keys)
    task_ids = sorted({task_id for task_id, _ in keys})
    task_deltas = []
    sample_counts = []
    for task_id in task_ids:
        task_keys = [key for key in keys if key[0] == task_id]
        sample_counts.append(len(task_keys))
        task_deltas.append(
            sum(right[key] for key in task_keys) - sum(left[key] for key in task_keys)
        )
    task_improvements = sum(delta > 0 for delta in task_deltas)
    task_regressions = sum(delta < 0 for delta in task_deltas)
    task_ties = sum(delta == 0 for delta in task_deltas)
    task_p = _exact_mcnemar(task_regressions, task_improvements)
    return {
        "paired_rollouts": len(keys),
        "before_passes": sum(left[key] for key in keys),
        "after_passes": sum(right[key] for key in keys),
        "paired_improvements": improvements,
        "paired_regressions": regressions,
        "rollout_level_exact_mcnemar_p": _exact_mcnemar(regressions, improvements),
        "independent_tasks": len(task_ids),
        "samples_per_task_min": min(sample_counts),
        "samples_per_task_max": max(sample_counts),
        "task_level_improvements": task_improvements,
        "task_level_regressions": task_regressions,
        "task_level_ties": task_ties,
        "task_level_exact_sign_test_p": task_p,
        "task_level_significant_at_0_05": task_p < 0.05,
    }


def _aggregate(root: Path, pairs: list[tuple[str, str]]) -> dict[str, Any]:
    totals = {
        "rollouts": 0,
        "sft_passes": 0,
        "grpo_passes": 0,
        "sft_policy_output_errors": 0,
        "grpo_policy_output_errors": 0,
        "sft_weighted_latency_ms": 0.0,
        "grpo_weighted_latency_ms": 0.0,
        "paired_improvements": 0,
        "paired_regressions": 0,
        "independent_tasks": 0,
        "task_level_improvements": 0,
        "task_level_regressions": 0,
        "task_level_ties": 0,
    }
    for sft_name, grpo_name in pairs:
        sft = _load_report(root, sft_name)
        grpo = _load_report(root, grpo_name)
        paired = _paired(root, sft_name, grpo_name)
        count = int(sft["behavior_gate"]["rollouts"])
        if count != int(grpo["behavior_gate"]["rollouts"]):
            raise ValueError(f"rollout count mismatch: {sft_name} vs {grpo_name}")
        totals["rollouts"] += count
        totals["sft_passes"] += int(sft["behavior_gate"]["successful_rollouts"])
        totals["grpo_passes"] += int(grpo["behavior_gate"]["successful_rollouts"])
        totals["sft_policy_output_errors"] += int(sft["behavior_gate"]["policy_output_errors"])
        totals["grpo_policy_output_errors"] += int(grpo["behavior_gate"]["policy_output_errors"])
        totals["sft_weighted_latency_ms"] += count * float(sft["rollout_latency"]["mean_ms"])
        totals["grpo_weighted_latency_ms"] += count * float(grpo["rollout_latency"]["mean_ms"])
        totals["paired_improvements"] += int(paired["paired_improvements"])
        totals["paired_regressions"] += int(paired["paired_regressions"])
        totals["independent_tasks"] += int(paired["independent_tasks"])
        totals["task_level_improvements"] += int(paired["task_level_improvements"])
        totals["task_level_regressions"] += int(paired["task_level_regressions"])
        totals["task_level_ties"] += int(paired["task_level_ties"])
    count = totals["rollouts"]
    return {
        **{key: value for key, value in totals.items() if "weighted" not in key},
        "sft_success_rate": totals["sft_passes"] / count,
        "grpo_success_rate": totals["grpo_passes"] / count,
        "sft_mean_latency_ms": round(totals["sft_weighted_latency_ms"] / count, 3),
        "grpo_mean_latency_ms": round(totals["grpo_weighted_latency_ms"] / count, 3),
        "rollout_level_exact_mcnemar_p": _exact_mcnemar(
            int(totals["paired_regressions"]), int(totals["paired_improvements"])
        ),
        "task_level_exact_sign_test_p": _exact_mcnemar(
            int(totals["task_level_regressions"]),
            int(totals["task_level_improvements"]),
        ),
    }


def build(reports_root: Path) -> dict[str, Any]:
    decision = _paired(reports_root, DECISION_REPORTS["sft"], DECISION_REPORTS["grpo"])
    decision["before_success_rate"] = decision["before_passes"] / decision["paired_rollouts"]
    decision["after_success_rate"] = decision["after_passes"] / decision["paired_rollouts"]
    decision["absolute_lift_pp"] = round(
        100 * (decision["after_success_rate"] - decision["before_success_rate"]), 3
    )
    full_loop = _aggregate(reports_root, FULL_LOOP_SEGMENTS)
    return {
        "schema_version": "native-react-rl-promotion.v2",
        "scope": "frozen heldout checkpoint pilot evidence",
        "sft_generalist": "qwen3-1.7b-native-react-sft-decision-bridge-step3-v1",
        "grpo_specialist": "qwen3-1.7b-native-react-grpo-decision-kl001-lr5e7-step1-seed06-v3",
        "training": {
            "method": "decision-state GRPO",
            "optimizer_steps": 1,
            "learning_rate": 5e-7,
            "beta": 0.01,
            "group_size": 8,
            "reward_source": "programmatic tool/state verifier",
        },
        "decision_state_holdout": decision,
        "full_agent_loop_holdout": full_loop,
        "promotion": {
            "global_replacement": False,
            "decision_specialist": True,
            "deployment_status": "shadow_candidate",
            "production_promotion_supported": False,
            "route": "verified poi_candidate_set present and poi_detail_set absent",
            "fallback": "SFT generalist on specialist inference or validation error",
            "reason": (
                "GRPO shows a positive pilot signal on four of six independent decision "
                "tasks, with two ties and no task-level regressions, while preserving all "
                "92 paired full-loop rollout outcomes. The task-level sign test is not "
                "significant, so this supports shadow evaluation only, not a production "
                "promotion or a generalization claim."
            ),
        },
        "limitations": [
            "The scoped holdout contains 6 tasks with 8 stochastic samples each.",
            "The full-loop holdout contains 23 tasks with 4 samples each.",
            "Repeated samples from one task are correlated and cannot be counted as independent tasks.",
            "The reported full-loop comparison evaluates each checkpoint as a whole arm, not the routed SFT-plus-specialist policy.",
            "The archived SFT and GRPO reports do not prove an identical quantization and serving contract.",
            "Latency is diagnostic on one RTX 4080 SUPER 32GB host, not a serving SLA.",
        ],
    }


def _markdown(report: dict[str, Any]) -> str:
    decision = report["decision_state_holdout"]
    full = report["full_agent_loop_holdout"]
    return f"""# Native ReAct GRPO 晋升报告

## 结论

保留 SFT 作为通用策略。GRPO checkpoint 仅作为 `get_poi_detail` 决策专家的 Shadow 候选；专家异常时自动回退 SFT。现有证据不足以支持生产晋升或跨任务泛化结论。

## 冻结评测

| 指标 | SFT | GRPO | 结论 |
|---|---:|---:|---|
| 决策状态重复采样硬通过率 | {decision['before_passes']}/{decision['paired_rollouts']} ({decision['before_success_rate']:.2%}) | {decision['after_passes']}/{decision['paired_rollouts']} ({decision['after_success_rate']:.2%}) | +{decision['absolute_lift_pp']:.2f}pp（诊断） |
| 完整 Agent Loop 硬通过率 | {full['sft_passes']}/{full['rollouts']} ({full['sft_success_rate']:.2%}) | {full['grpo_passes']}/{full['rollouts']} ({full['grpo_success_rate']:.2%}) | 无回归 |
| 完整 Loop 参数/格式错误 | {full['sft_policy_output_errors']} | {full['grpo_policy_output_errors']} | 持平 |
| 完整 Loop 平均延迟 | {full['sft_mean_latency_ms']:.1f} ms | {full['grpo_mean_latency_ms']:.1f} ms | 诊断指标 |

决策状态只有 **{decision['independent_tasks']} 个独立任务**，每题重复采样 {decision['samples_per_task_min']} 次。按独立任务聚合后，{decision['task_level_improvements']} 题改善、{decision['task_level_regressions']} 题退化、{decision['task_level_ties']} 题持平，双侧精确符号检验 p={decision['task_level_exact_sign_test_p']:.3f}，**未达到 0.05 显著性标准**。

如果把同一任务的重复采样错误地视为彼此独立，会得到 {decision['paired_improvements']} 个采样级改善、{decision['paired_regressions']} 个采样级退化和 McNemar p={decision['rollout_level_exact_mcnemar_p']:.5f}。该数值只描述采样稳定性，不能用于跨任务泛化声明。完整 Loop 的 {full['independent_tasks']} 个任务、{full['rollouts']} 次重复采样结果完全一致。

## 训练与工程策略

- 在真实工具历史之后截取决策状态，避免静态提示与生产 ReAct 状态不一致。
- 使用程序化 Verifier 奖励；group size=8、学习率 5e-7、KL beta=0.01。
- 用 vLLM multi-LoRA 在同一 1.7B 基座挂载 SFT/GRPO，按已验证状态路由，无需常驻两份基座模型。
- 模型只选择动作；POI 身份等可信参数由控制器从 ledger 注入，冗余模型参数在权限边界丢弃。

## 边界

这是一条值得继续验证的正向工程信号，不是统计上已经确认的泛化提升。下一步必须扩大独立任务数、城市和故障类型，并直接评测“SFT 通用策略 + GRPO 专家 + fallback”的真实路由系统。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.reports_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
