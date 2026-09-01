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
    downstream_repair: dict[str, Any] | None = None,
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
    if runtime.get("execution_mode") != "react":
        gate_errors.append("REAL_RUNTIME_NOT_REACT")
    if int(rl_gain["paired_rollouts"]) < 128:
        gate_errors.append("RL_EVALUATION_TOO_SMALL")
    if rl_gain_value < 0.02:
        gate_errors.append("RL_GAIN_BELOW_TARGET")
    if rl_candidate_rate < 0.85:
        gate_errors.append("RL_CANDIDATE_BELOW_TARGET")
    if not bool((rl_gain.get("gate") or {}).get("passed")):
        gate_errors.append("RL_EVIDENCE_GATE_FAILED")
    if downstream_repair is None:
        gate_errors.append("DOWNSTREAM_REPAIR_EVIDENCE_MISSING")
    else:
        if downstream_repair.get("schema_version") != "verifier-repair-downstream-gain.v1":
            gate_errors.append("DOWNSTREAM_REPAIR_SCHEMA_INVALID")
        if not bool((downstream_repair.get("gate") or {}).get("passed")):
            gate_errors.append("DOWNSTREAM_REPAIR_GATE_FAILED")
        if int(downstream_repair.get("paired_tasks") or 0) < 16:
            gate_errors.append("DOWNSTREAM_REPAIR_TOO_SMALL")
        if float(downstream_repair.get("specialist_activation_rate") or 0.0) < 0.8:
            gate_errors.append("VERIFIER_REPAIR_SPECIALIST_ACTIVATION_TOO_LOW")
        if int(downstream_repair.get("scope_violations") or 0) > 0:
            gate_errors.append("VERIFIER_REPAIR_SPECIALIST_SCOPE_VIOLATION")

    source_cluster_ci = rl_gain.get("source_cluster_bootstrap_95ci")
    source_cluster_p = rl_gain.get("source_cluster_randomization_two_sided_p")
    if source_cluster_ci is None or source_cluster_p is None:
        gate_errors.append("SOURCE_CLUSTER_STATISTICS_MISSING")

    return {
        "schema_version": "stage3-two-axis-fullchain-evidence.v2",
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
        "post_training_decision_axis": {
            "tasks": int(rl_gain["tasks"]),
            "independent_source_clusters": int(
                rl_gain.get("independent_source_clusters") or 0
            ),
            "paired_rollouts": int(rl_gain["paired_rollouts"]),
            "sft_success_rate": float(rl_gain["baseline_success_rate"]),
            "grpo_success_rate": rl_candidate_rate,
            "absolute_gain": rl_gain_value,
            "relative_error_reduction": float(rl_gain["relative_error_reduction"]),
            "exact_mcnemar_two_sided_p": float(
                rl_gain["exact_mcnemar_two_sided_p"]
            ),
            "source_cluster_randomization_two_sided_p": (
                float(source_cluster_p) if source_cluster_p is not None else None
            ),
            "source_cluster_bootstrap_95ci": (
                list(source_cluster_ci) if source_cluster_ci is not None else None
            ),
            "decision_success_definition": (
                "the bounded review-state policy selects the expected repair action and "
                "passes its argument contract; downstream solve and validation are separate"
            ),
        },
        "downstream_repair_axis": downstream_repair or {"status": "missing"},
        "gate": {"passed": not gate_errors, "errors": gate_errors},
        "limitations": [
            "The planning and recovery axes use different frozen suites and are not a direct three-arm ranking.",
            "The planning suite uses frozen synthetic replay rather than production traffic.",
            "The one-decision RL metric does not prove downstream repair success by itself.",
            "End-to-end claims require an exercised verifier-repair specialist route and a fresh solve-validate result.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    planning = report["planning_axis"]
    runtime = report["repository_runtime_axis"]
    recovery = report["post_training_decision_axis"]
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
            "## 后训练决策轴",
            "",
            f"- {recovery['paired_rollouts']} 条配对 review 决策 rollout：SFT "
            f"{recovery['sft_success_rate'] * 100:.2f}%，GRPO "
            f"{recovery['grpo_success_rate'] * 100:.2f}%，提升 "
            f"{recovery['absolute_gain'] * 100:.2f}pp。",
            f"- 失败率相对下降 {recovery['relative_error_reduction'] * 100:.2f}%，"
            f"source-state randomization p="
            f"{recovery['source_cluster_randomization_two_sided_p']}。",
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
    parser.add_argument("--downstream-repair-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    report = build_report(
        _load(args.planning_report),
        _load(args.runtime_report),
        _load(args.rl_gain_report),
        _load(args.downstream_repair_report),
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
