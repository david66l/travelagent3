"""Combine orthogonal planning and post-training ablations without mixing denominators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(
    planning: dict[str, Any],
    runtime: dict[str, Any],
    rl_gain: dict[str, Any],
) -> dict[str, Any]:
    pure = planning["pure_agent"]
    verified = planning["verified_planner"]
    real_runtime = runtime["real_agent_runtime"]
    rl_candidate_rate = float(rl_gain["candidate_success_rate"])
    rl_gain_value = float(rl_gain["absolute_gain"])
    gate_errors = []
    if int(planning["paired_tasks"]) < 30:
        gate_errors.append("PLANNING_ABLATION_TOO_SMALL")
    if float(verified["hard_pass_rate"]) < 0.90:
        gate_errors.append("VERIFIED_PLANNER_BELOW_TARGET")
    if float(real_runtime["hard_pass_rate"]) < 0.95:
        gate_errors.append("REAL_RUNTIME_BELOW_TARGET")
    if int(rl_gain["paired_rollouts"]) < 128:
        gate_errors.append("RL_EVALUATION_TOO_SMALL")
    if rl_gain_value < 0.02:
        gate_errors.append("RL_GAIN_BELOW_TARGET")
    if rl_candidate_rate < 0.85:
        gate_errors.append("RL_CANDIDATE_BELOW_TARGET")
    if not bool((rl_gain.get("gate") or {}).get("passed")):
        gate_errors.append("RL_EVIDENCE_GATE_FAILED")

    return {
        "schema_version": "stage3-two-axis-fullchain-evidence.v1",
        "methodology": (
            "two orthogonal paired ablations; success rates from different suites "
            "are never treated as one three-arm ranking"
        ),
        "planning_axis": {
            "paired_tasks": int(planning["paired_tasks"]),
            "model": planning["model"],
            "pure_agent_hard_pass_rate": float(pure["hard_pass_rate"]),
            "agent_plus_cpsat_hard_pass_rate": float(verified["hard_pass_rate"]),
            "hard_pass_gain": float(verified["hard_pass_rate"])
            - float(pure["hard_pass_rate"]),
            "pure_agent_mean_tokens": float(pure["mean_total_tokens"]),
            "agent_plus_cpsat_mean_tokens": float(verified["mean_total_tokens"]),
            "token_reduction_percent": float(
                planning["paired_delta"]["verified_token_reduction_vs_pure_percent"]
            ),
        },
        "repository_runtime_axis": {
            "paired_tasks": int(runtime["paired_tasks"]),
            "execution_mode": runtime["execution_mode"],
            "real_runtime_hard_pass_rate": float(real_runtime["hard_pass_rate"]),
            "real_runtime_mean_tokens": float(real_runtime["mean_total_tokens"]),
            "real_runtime_mean_model_calls": float(real_runtime["mean_model_calls"]),
            "real_runtime_mean_tool_calls": float(real_runtime["mean_tool_calls"]),
        },
        "post_training_recovery_axis": {
            "tasks": int(rl_gain["tasks"]),
            "paired_rollouts": int(rl_gain["paired_rollouts"]),
            "sft_success_rate": float(rl_gain["baseline_success_rate"]),
            "grpo_success_rate": rl_candidate_rate,
            "absolute_gain": rl_gain_value,
            "relative_error_reduction": float(rl_gain["relative_error_reduction"]),
            "exact_mcnemar_two_sided_p": float(
                rl_gain["exact_mcnemar_two_sided_p"]
            ),
            "task_cluster_bootstrap_95ci": list(
                rl_gain["task_cluster_bootstrap_95ci"]
            ),
            "fullchain_success_definition": (
                "recovered Agent Loop trajectory reaches solver and deterministic verifier hard pass"
            ),
        },
        "gate": {"passed": not gate_errors, "errors": gate_errors},
        "limitations": [
            "The planning and recovery axes use different frozen suites and are not a direct three-arm ranking.",
            "The planning suite uses frozen synthetic replay rather than production traffic.",
            "The RL claim is scoped to fault-recovery trajectories.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    planning = report["planning_axis"]
    runtime = report["repository_runtime_axis"]
    recovery = report["post_training_recovery_axis"]
    return "\n".join(
        [
            "# TravelAgent 两轴全链路证据",
            "",
            "> 规划架构与后训练分别做同题配对；不同测试集的成功率不混成一个三臂排名。",
            "",
            "## 规划架构轴",
            "",
            f"- {planning['paired_tasks']} 个同题任务：纯 Agent 硬通过率 "
            f"{planning['pure_agent_hard_pass_rate'] * 100:.2f}%，Agent+CP-SAT 为 "
            f"{planning['agent_plus_cpsat_hard_pass_rate'] * 100:.2f}%。",
            f"- 平均 Token 从 {planning['pure_agent_mean_tokens']:.2f} 降至 "
            f"{planning['agent_plus_cpsat_mean_tokens']:.2f}，降低 "
            f"{planning['token_reduction_percent']:.2f}%。",
            "",
            "## 仓库真实运行时",
            "",
            f"- {runtime['paired_tasks']} 个任务，硬通过率 "
            f"{runtime['real_runtime_hard_pass_rate'] * 100:.2f}%，平均模型调用 "
            f"{runtime['real_runtime_mean_model_calls']:.2f} 次。",
            "",
            "## 后训练恢复轴",
            "",
            f"- {recovery['paired_rollouts']} 条配对恢复轨迹：SFT "
            f"{recovery['sft_success_rate'] * 100:.2f}%，GRPO "
            f"{recovery['grpo_success_rate'] * 100:.2f}%，提升 "
            f"{recovery['absolute_gain'] * 100:.2f}pp。",
            f"- 失败率相对下降 {recovery['relative_error_reduction'] * 100:.2f}%，"
            f"McNemar p={recovery['exact_mcnemar_two_sided_p']:.6f}。",
            "",
            f"- 总门禁：{'通过' if report['gate']['passed'] else '未通过'}。",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-report", type=Path, required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--rl-gain-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    report = build_report(
        _load(args.planning_report),
        _load(args.runtime_report),
        _load(args.rl_gain_report),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
