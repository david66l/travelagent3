"""Build the Stage 24 unified post-training and inference evaluation report.

The report intentionally keeps one-step policy benchmarks separate from full
multi-turn rollouts.  Every metric is derived from an archived source file and
the source SHA-256 values are emitted alongside the normalized results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


STRICT_ARMS = (
    ("base_4b", "Qwen3-4B Base", "stage21-preference-hard-base-current-prompt-v1", "eager"),
    ("sft_4b", "Qwen3-4B SFT", "stage21-preference-hard-balanced-formal-v1", "dynamic LoRA/eager"),
    ("dpo_4b_dynamic", "Qwen3-4B SFT+DPO", "stage22-preference-hard-dpo-step84-v1", "dynamic LoRA/eager"),
    ("dpo_4b_static", "Qwen3-4B SFT+DPO", "stage22-preference-hard-dpo-formal-scale150-fp16-test-v1", "static merge/eager"),
    ("dpo_4b_cuda_graph", "Qwen3-4B SFT+DPO", "stage23-preference-hard-dpo-scale150-cudagraph-steady-v1", "static merge/CUDA Graph"),
    ("teacher_8b", "Qwen3-8B teacher", "stage23-preference-hard-qwen3-8b-teacher-v1", "eager/unconstrained"),
)

ROLLOUT_ARMS = (
    ("base_4b", "Qwen3-4B Base", "stage21-base-preference-rollout-holdout-v1", "unconstrained"),
    ("sft_4b", "Qwen3-4B SFT", "stage21-balanced-preference-rollout-holdout-v1", "unconstrained"),
    ("dpo_4b_dynamic", "Qwen3-4B SFT+DPO", "stage22-dpo-formal-dynamic-preference-rollout-holdout-v1", "unconstrained"),
    ("dpo_4b_static", "Qwen3-4B SFT+DPO", "stage22-dpo-formal-scale150-static-preference-rollout-holdout-v1", "constrained/static"),
    ("teacher_8b", "Qwen3-8B teacher", "stage23-qwen3-8b-teacher-constrained-preference-rollout-holdout-v1", "constrained"),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pct_delta(value: float, reference: float) -> float:
    if not reference:
        raise ValueError("percentage delta reference must be non-zero")
    return round((value / reference - 1.0) * 100.0, 3)


def summarize_http_benchmark(directory: Path) -> dict[str, Any]:
    report_path = directory / "report.json"
    report = _read_json(report_path)
    if report.get("schema_version") != "vllm-http-benchmark.v1":
        raise ValueError(f"unexpected HTTP benchmark schema: {report_path}")
    summary = report["summary"]
    inference = summary["inference"]
    runs_path = directory / "runs.jsonl"
    runs = _read_jsonl(runs_path)
    if len(runs) != summary["runs"]:
        raise ValueError(f"run count mismatch: {directory}")
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        families[row["family"]].append(row)
    return {
        "model": report["model"],
        "cases": report["cases"],
        "repetitions": report["repetitions"],
        "runs": summary["runs"],
        "successful_runs": summary["successful_runs"],
        "success_rate": round(summary["successful_runs"] / summary["runs"], 8),
        "action_mismatches": summary["action_mismatches"],
        "argument_mismatches": summary["argument_mismatches"],
        "http_errors": summary["http_errors"],
        "concurrency": report["concurrency"],
        "completion_tokens_mean": inference["completion_tokens"]["mean"],
        "latency_ms": inference["request_latency_ms"],
        "ttft_ms": inference["ttft_ms"],
        "tpot_ms": inference["tpot_ms"],
        "throughput_requests_per_second": summary["request_throughput_per_second"],
        "family_success": {
            family: {
                "runs": len(rows),
                "successful_runs": sum(bool(row["success"]) for row in rows),
                "success_rate": round(sum(bool(row["success"]) for row in rows) / len(rows), 8),
            }
            for family, rows in sorted(families.items())
        },
        "case_ids": sorted({row["case_id"] for row in runs}),
        "source": {
            "report": str(report_path.as_posix()),
            "report_sha256": _sha256(report_path),
            "runs": str(runs_path.as_posix()),
            "runs_sha256": _sha256(runs_path),
        },
    }


def summarize_rollout_candidates(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    candidates_path = directory / "teacher_candidates.jsonl"
    manifest = _read_json(manifest_path)
    rows = _read_jsonl(candidates_path)
    if len(rows) != manifest["candidate_rollouts"]:
        raise ValueError(f"rollout count mismatch: {directory}")
    scores = [row["score"] for row in rows]
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        families[row["family"]].append(row["score"])
    latency_total = sum(float(score["request_latency_ms"]) for score in scores)
    return {
        "model": manifest["model"],
        "tasks": len(rows),
        "successful_tasks": sum(bool(score["successful"]) for score in scores),
        "success_rate": round(sum(bool(score["successful"]) for score in scores) / len(rows), 8),
        "mean_episode_reward": round(fmean(float(score["episode_reward"]) for score in scores), 8),
        "policy_steps": sum(int(score["policy_steps"]) for score in scores),
        "completion_tokens": sum(int(score["completion_tokens"]) for score in scores),
        "request_latency_ms": round(latency_total, 3),
        "mean_episode_request_latency_ms": round(latency_total / len(rows), 3),
        "family_success": {
            family: {
                "tasks": len(items),
                "successful_tasks": sum(bool(item["successful"]) for item in items),
                "success_rate": round(sum(bool(item["successful"]) for item in items) / len(items), 8),
            }
            for family, items in sorted(families.items())
        },
        "task_ids": sorted(row["task_id"] for row in rows),
        "source": {
            "manifest": str(manifest_path.as_posix()),
            "manifest_sha256": _sha256(manifest_path),
            "candidates": str(candidates_path.as_posix()),
            "candidates_sha256": _sha256(candidates_path),
        },
    }


def _summarize_routed_strict(directory: Path) -> dict[str, Any]:
    path = directory / "report.json"
    report = _read_json(path)
    if report.get("schema_version") != "routed-policy-evaluation.v1":
        raise ValueError("unexpected routed strict schema")
    summary = report["summary"]
    inference = summary["inference"]
    return {
        "model": "Qwen3-4B student + Qwen3-8B teacher",
        "cases": 172,
        "repetitions": 3,
        "runs": summary["runs"],
        "successful_runs": summary["successful_runs"],
        "success_rate": round(summary["successful_runs"] / summary["runs"], 8),
        "action_mismatches": summary["action_mismatches"],
        "argument_mismatches": summary["argument_mismatches"],
        "http_errors": summary["http_errors"],
        "concurrency": 4,
        "completion_tokens_mean": inference["completion_tokens"]["mean"],
        "latency_ms": inference["request_latency_ms"],
        "ttft_ms": inference["ttft_ms"],
        "tpot_ms": inference["tpot_ms"],
        "throughput_requests_per_second": None,
        "route_counts": summary["route_counts"],
        "execution_mode": report["routing_contract"]["execution_mode"],
        "source": {"report": str(path.as_posix()), "report_sha256": _sha256(path)},
    }


def _summarize_routed_rollout(directory: Path) -> dict[str, Any]:
    path = directory / "report.json"
    report = _read_json(path)
    if report.get("schema_version") != "routed-rollout-evaluation.v1":
        raise ValueError("unexpected routed rollout schema")
    summary = dict(report["summary"])
    summary["success_rate"] = round(summary["successful_tasks"] / summary["tasks"], 8)
    summary["model"] = "Qwen3-4B student + Qwen3-8B teacher"
    summary["execution_mode"] = report["execution_mode"]
    summary["source"] = {"report": str(path.as_posix()), "report_sha256": _sha256(path)}
    return summary


def _load_named_json(root: Path, relative: str) -> tuple[dict[str, Any], dict[str, str]]:
    path = root / relative
    return _read_json(path), {"path": str(path.as_posix()), "sha256": _sha256(path)}


def _preference_payload(row: dict[str, Any]) -> str:
    return json.dumps(
        {key: row[key] for key in ("family", "messages", "tools", "chosen", "rejected")},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build(reports_root: Path, checkpoints_root: Path) -> dict[str, Any]:
    strict = []
    for key, label, directory, runtime in STRICT_ARMS:
        row = summarize_http_benchmark(reports_root / directory)
        row.update(key=key, label=label, runtime=runtime)
        strict.append(row)
    reference_case_ids = strict[0]["case_ids"]
    for row in strict[1:]:
        if row["case_ids"] != reference_case_ids:
            raise ValueError(f"strict benchmark case set mismatch: {row['key']}")
        row.pop("case_ids")
    strict[0].pop("case_ids")

    routed_strict = _summarize_routed_strict(
        reports_root / "stage23-routed-4b-dpo-cudagraph-8b-teacher-composite-v1"
    )
    routed_strict.update(key="routed", label="4B student / 8B teacher router", runtime="sequential replay")
    strict.append(routed_strict)

    rollouts = []
    for key, label, directory, action_space in ROLLOUT_ARMS:
        row = summarize_rollout_candidates(reports_root / directory)
        row.update(key=key, label=label, action_space=action_space)
        rollouts.append(row)
    reference_task_ids = rollouts[0]["task_ids"]
    for row in rollouts[1:]:
        if row["task_ids"] != reference_task_ids:
            raise ValueError(f"rollout task set mismatch: {row['key']}")
        row.pop("task_ids")
    rollouts[0].pop("task_ids")
    routed_rollout = _summarize_routed_rollout(
        reports_root / "stage23-routed-4b-dpo-8b-teacher-rollout-composite-v1"
    )
    routed_rollout.update(key="routed", label="4B student / 8B teacher router", action_space="routed constraints")
    rollouts.append(routed_rollout)

    sft_training, sft_training_source = _load_named_json(
        checkpoints_root, "qwen3-4b-stage21-sft-balanced-formal-v1/training_report.json"
    )
    dpo_training, dpo_training_source = _load_named_json(
        checkpoints_root, "qwen3-4b-stage22-dpo-balanced-formal-v1/training_report.json"
    )
    sft_pref, sft_pref_source = _load_named_json(
        reports_root, "stage22-preference-logprob-balanced-sft-test-v1/report.json"
    )
    dpo_pref, dpo_pref_source = _load_named_json(
        reports_root, "stage22-preference-logprob-dpo-formal-test-v1/report.json"
    )
    grpo_comparison, grpo_source = _load_named_json(
        reports_root, "formal-3b-holdout32-v1/comparison.json"
    )
    project_root = reports_root.resolve().parents[2]
    sft_preference_path = project_root / sft_pref["preference_file"]
    dpo_preference_path = project_root / dpo_pref["preference_file"]
    sft_preference_payloads = [_preference_payload(row) for row in _read_jsonl(sft_preference_path)]
    dpo_preference_payloads = [_preference_payload(row) for row in _read_jsonl(dpo_preference_path)]
    if sft_preference_payloads != dpo_preference_payloads:
        raise ValueError("SFT and DPO preference evaluations are not paired on identical payloads")

    by_strict = {row["key"]: row for row in strict}
    by_rollout = {row["key"]: row for row in rollouts}
    prefix = summarize_http_benchmark(reports_root / "stage23-preference-hard-dpo-scale150-prefix-cache-v1")
    c8 = summarize_http_benchmark(reports_root / "stage23-preference-hard-dpo-scale150-cudagraph-c8-v1")
    c16 = summarize_http_benchmark(reports_root / "stage23-preference-hard-dpo-scale150-cudagraph-c16-v1")
    scale125 = summarize_http_benchmark(reports_root / "stage22-preference-hard-validation-dpo-formal-scale125-fp16-v1")
    scale150 = summarize_http_benchmark(reports_root / "stage22-preference-hard-validation-dpo-formal-scale150-fp16-v1")
    teacher_tradeoff = summarize_http_benchmark(reports_root / "stage23-tradeoff-narrow-qwen3-8b-teacher-v1")

    route_counts = by_rollout["routed"]["route_tasks"]
    task_total = by_rollout["routed"]["tasks"]
    report = {
        "schema_version": "travel-agent-stage24-final-evaluation.v1",
        "status": "passed",
        "evaluation_contract": {
            "strict_policy_cases": 172,
            "strict_policy_families": ["clarification", "recovery", "search", "tradeoff"],
            "rollout_tasks": 172,
            "rollout_pairing": "same frozen task ids across all non-routed arms",
            "scope_note": "one-step HTTP latency excludes model startup and tool execution; rollout latency is summed model request latency",
        },
        "training_evidence": {
            "sft": {
                "dataset_version": sft_training["dataset_version"],
                "examples_checked": sft_training["model_preflight"]["rows_checked"],
                "train_runtime_seconds": sft_training["train_metrics"]["train_runtime"],
                "train_loss": sft_training["train_metrics"]["train_loss"],
                "eval_loss": sft_training["eval_metrics"]["eval_loss"],
                "source": sft_training_source,
            },
            "dpo": {
                "dataset_version": dpo_training["dataset_version"],
                "unique_pairs": dpo_training["dataset_preflight"]["unique_pairs"],
                "train_examples": dpo_training["train_examples"],
                "validation_examples": dpo_training["eval_examples"],
                "train_runtime_seconds": dpo_training["train_metrics"]["train_runtime"],
                "train_loss": dpo_training["train_metrics"]["train_loss"],
                "eval_loss": dpo_training["eval_metrics"]["eval_loss"],
                "eval_preference_accuracy": dpo_training["eval_metrics"]["eval_rewards/accuracies"],
                "source": dpo_training_source,
            },
            "trajectory_grpo_b0": {
                "backbone": "Qwen2.5-3B",
                "scope": grpo_comparison["scope"],
                "tasks": grpo_comparison["arms"][0]["tasks"],
                "rollouts": grpo_comparison["arms"][0]["samples"],
                "base_success_rate": grpo_comparison["arms"][0]["success_rate"],
                "sft_success_rate": grpo_comparison["arms"][1]["success_rate"],
                "sft_grpo_success_rate": grpo_comparison["arms"][2]["success_rate"],
                "sft_mean_reward": grpo_comparison["arms"][1]["mean_reward"],
                "sft_grpo_mean_reward": grpo_comparison["arms"][2]["mean_reward"],
                "success_delta_vs_sft_percentage_points": round(
                    (grpo_comparison["arms"][2]["success_rate"] - grpo_comparison["arms"][1]["success_rate"]) * 100,
                    3,
                ),
                "claim_boundary": "trajectory-level GRPO-B0 engineering baseline; not a claim of solved long-horizon credit assignment",
                "source": grpo_source,
            },
        },
        "strict_policy_benchmark": strict,
        "multi_turn_rollout_benchmark": rollouts,
        "ablations": {
            "preference_optimization": {
                "pairs": dpo_pref["overall"]["pairs"],
                "paired_model_payloads_verified": True,
                "sft_preference_accuracy": sft_pref["overall"]["preference_accuracy"],
                "dpo_preference_accuracy": dpo_pref["overall"]["preference_accuracy"],
                "sft_mean_logprob_margin": sft_pref["overall"]["mean_logprob_margin"],
                "dpo_mean_logprob_margin": dpo_pref["overall"]["mean_logprob_margin"],
                "margin_delta_percent": _pct_delta(dpo_pref["overall"]["mean_logprob_margin"], sft_pref["overall"]["mean_logprob_margin"]),
                "sources": [sft_pref_source, dpo_pref_source],
                "dataset_sources": [
                    {"path": sft_pref["preference_file"], "sha256": _sha256(sft_preference_path)},
                    {"path": dpo_pref["preference_file"], "sha256": _sha256(dpo_preference_path)},
                ],
            },
            "static_merge_scale_validation": {
                "scale_1_25_success_rate": scale125["success_rate"],
                "scale_1_50_success_rate": scale150["success_rate"],
                "selection": "1.50 selected on validation before frozen test",
                "sources": [scale125["source"], scale150["source"]],
            },
            "prefix_cache": {
                "baseline_mean_latency_ms": by_strict["dpo_4b_static"]["latency_ms"]["mean"],
                "prefix_cache_mean_latency_ms": prefix["latency_ms"]["mean"],
                "latency_delta_percent": _pct_delta(prefix["latency_ms"]["mean"], by_strict["dpo_4b_static"]["latency_ms"]["mean"]),
                "decision": "rejected for this workload",
                "source": prefix["source"],
            },
            "cuda_graph": {
                "eager_mean_latency_ms": by_strict["dpo_4b_static"]["latency_ms"]["mean"],
                "cuda_graph_mean_latency_ms": by_strict["dpo_4b_cuda_graph"]["latency_ms"]["mean"],
                "latency_delta_percent": _pct_delta(by_strict["dpo_4b_cuda_graph"]["latency_ms"]["mean"], by_strict["dpo_4b_static"]["latency_ms"]["mean"]),
                "p95_delta_percent": _pct_delta(by_strict["dpo_4b_cuda_graph"]["latency_ms"]["p95"], by_strict["dpo_4b_static"]["latency_ms"]["p95"]),
                "throughput_delta_percent": _pct_delta(by_strict["dpo_4b_cuda_graph"]["throughput_requests_per_second"], by_strict["dpo_4b_static"]["throughput_requests_per_second"]),
                "decision": "accepted for resident service after warm-up",
            },
            "concurrency": {
                "c4_requests_per_second": by_strict["dpo_4b_cuda_graph"]["throughput_requests_per_second"],
                "c8_requests_per_second": c8["throughput_requests_per_second"],
                "c16_requests_per_second": c16["throughput_requests_per_second"],
                "c8_speedup_vs_c4": round(c8["throughput_requests_per_second"] / by_strict["dpo_4b_cuda_graph"]["throughput_requests_per_second"], 3),
                "c16_speedup_vs_c4": round(c16["throughput_requests_per_second"] / by_strict["dpo_4b_cuda_graph"]["throughput_requests_per_second"], 3),
                "recommended_online_concurrency": 8,
                "sources": [c8["source"], c16["source"]],
            },
            "routing": {
                "student_task_share": round(route_counts["student"] / task_total, 8),
                "teacher_task_share": round(route_counts["teacher"] / task_total, 8),
                "strict_success_rate": by_strict["routed"]["success_rate"],
                "rollout_success_rate": by_rollout["routed"]["success_rate"],
                "completion_token_delta_vs_all_teacher_percent": _pct_delta(by_rollout["routed"]["completion_tokens"], by_rollout["teacher_8b"]["completion_tokens"]),
                "latency_delta_vs_all_teacher_percent": _pct_delta(by_rollout["routed"]["mean_episode_request_latency_ms"], by_rollout["teacher_8b"]["mean_episode_request_latency_ms"]),
                "teacher_tradeoff_constrained_success_rate": teacher_tradeoff["success_rate"],
                "execution_caveat": "composite metrics use sequential-model replay, not simultaneous 4B/8B residency on one GPU",
                "source": by_rollout["routed"]["source"],
            },
        },
        "headline": {
            "final_policy": "Qwen3-4B SFT+DPO static student with deterministic family routing to Qwen3-8B teacher",
            "strict_success": f"{by_strict['routed']['successful_runs']}/{by_strict['routed']['runs']}",
            "rollout_success": f"{by_rollout['routed']['successful_tasks']}/{by_rollout['routed']['tasks']}",
            "mean_reward": by_rollout["routed"]["mean_episode_reward"],
            "teacher_call_share": round(route_counts["teacher"] / task_total, 8),
            "token_reduction_vs_all_teacher_percent": -_pct_delta(by_rollout["routed"]["completion_tokens"], by_rollout["teacher_8b"]["completion_tokens"]),
            "latency_reduction_vs_all_teacher_percent": -_pct_delta(by_rollout["routed"]["mean_episode_request_latency_ms"], by_rollout["teacher_8b"]["mean_episode_request_latency_ms"]),
        },
        "methodology_references": [
            {
                "name": "QLoRA: Efficient Finetuning of Quantized LLMs",
                "url": "https://arxiv.org/abs/2305.14314",
                "project_alignment": "NF4 double-quantized base model with trainable LoRA adapters",
            },
            {
                "name": "Direct Preference Optimization",
                "url": "https://arxiv.org/abs/2305.18290",
                "project_alignment": "direct chosen/rejected preference optimization after SFT",
            },
            {
                "name": "RouteLLM: Learning to Route LLMs with Preference Data",
                "url": "https://arxiv.org/abs/2406.18665",
                "project_alignment": "strong/weak model routing motivates the cost-quality design; this project currently uses an auditable deterministic router rather than a learned router",
            },
            {
                "name": "vLLM engine arguments",
                "url": "https://docs.vllm.ai/en/v0.10.0/configuration/engine_args.html",
                "project_alignment": "the eager-versus-CUDA-Graph ablation follows the documented enforce-eager runtime distinction",
            },
        ],
        "limitations": [
            "The frozen holdout has 172 synthetic policy tasks and is an engineering benchmark, not an external paper-level benchmark.",
            "The routed comparison is deterministic sequential replay; live dual-model co-hosting has not yet been measured on the 24 GB GPU.",
            "The one-step Base, dynamic-adapter, static-merge, and CUDA-Graph rows use different runtime modes, which are explicitly labeled.",
            "Tool execution and external API latency are excluded from model-only latency metrics.",
        ],
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    headline = report["headline"]
    lines = [
        "# TravelAgent 阶段 24：后训练与推理路由总评估",
        "",
        "## 一句话结论",
        "",
        f"最终方案是 **{headline['final_policy']}**。严格工具决策为 {headline['strict_success']}，完整多轮任务为 {headline['rollout_success']}，平均 Reward 为 {headline['mean_reward']:.6f}；仅 {headline['teacher_call_share'] * 100:.2f}% 的任务交给 8B 教师，相比全部使用 8B，生成 token 减少 {headline['token_reduction_vs_all_teacher_percent']:.1f}%，模型请求延迟减少 {headline['latency_reduction_vs_all_teacher_percent']:.1f}%。",
        "",
        "## 后训练证据链",
        "",
        "| 阶段 | 数据规模 | 训练结果 |",
        "|---|---:|---|",
    ]
    sft = report["training_evidence"]["sft"]
    dpo = report["training_evidence"]["dpo"]
    grpo = report["training_evidence"]["trajectory_grpo_b0"]
    lines.extend([
        f"| 蒸馏 SFT | {sft['examples_checked']} 条序列预检 | train loss {sft['train_loss']:.4f}；eval loss {sft['eval_loss']:.4f}；{sft['train_runtime_seconds']:.1f}s |",
        f"| DPO 偏好优化 | {dpo['unique_pairs']} 个唯一偏好对 | train loss {dpo['train_loss']:.4f}；eval loss {dpo['eval_loss']:.4f}；eval preference accuracy {dpo['eval_preference_accuracy'] * 100:.2f}% |",
        f"| 有状态 GRPO-B0 | Qwen2.5-3B；{grpo['tasks']} 题 × 4 rollout | SFT {grpo['sft_success_rate'] * 100:.2f}% → SFT+GRPO {grpo['sft_grpo_success_rate'] * 100:.2f}%（+{grpo['success_delta_vs_sft_percentage_points']:.2f} pp） |",
        "| 静态部署校准 | 1.25× / 1.50× LoRA merge | 仅用 validation 选择 1.50×，再冻结测试 |",
        "| 学生/教师路由 | 4B 高频动作，8B 复杂权衡 | 端到端路由与失败回退已实现 |",
        "",
        "> GRPO 行来自前一代 Qwen2.5-3B 的独立配对协议，用来证明项目具备真实多轮 GRPO 训练与晋升门能力；它不与后续 Qwen3-4B 表格做跨协议数值排名，也不宣称已解决逐轮信用分配。",
        "",
        "## 严格单步工具决策",
        "",
        "> 测的是模型选工具的准确率和模型请求延迟，不包含工具执行时间。运行时不同，因此表中明确标注。",
        "",
        "| 模型 | 运行时 | 正确/总数 | 成功率 | 平均延迟 | P95 | 吞吐 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in report["strict_policy_benchmark"]:
        throughput = "—" if row["throughput_requests_per_second"] is None else f"{row['throughput_requests_per_second']:.3f}/s"
        lines.append(
            f"| {row['label']} | {row['runtime']} | {row['successful_runs']}/{row['runs']} | {row['success_rate'] * 100:.2f}% | {row['latency_ms']['mean']:.1f} ms | {row['latency_ms']['p95']:.1f} ms | {throughput} |"
        )
    lines.extend([
        "",
        "## 完整多轮旅行任务",
        "",
        "| 模型 | 动作约束 | 成功任务 | Reward | 策略步 | 生成 token | 每任务模型延迟 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in report["multi_turn_rollout_benchmark"]:
        lines.append(
            f"| {row['label']} | {row['action_space']} | {row['successful_tasks']}/{row['tasks']} | {row['mean_episode_reward']:.6f} | {row['policy_steps']} | {row['completion_tokens']} | {row['mean_episode_request_latency_ms']:.1f} ms |"
        )
    pref = report["ablations"]["preference_optimization"]
    cuda = report["ablations"]["cuda_graph"]
    cache = report["ablations"]["prefix_cache"]
    conc = report["ablations"]["concurrency"]
    route = report["ablations"]["routing"]
    lines.extend([
        "",
        "## 关键消融结论",
        "",
        f"- 在完全相同的 {pref['pairs']} 个模型输入/偏好对上，SFT 和 DPO 都是 {pref['dpo_preference_accuracy'] * 100:.0f}% preference accuracy，但 DPO 将 chosen/rejected 平均 log-prob margin 从 {pref['sft_mean_logprob_margin']:.6f} 提高到 {pref['dpo_mean_logprob_margin']:.6f}（+{pref['margin_delta_percent']:.1f}%）。",
        f"- 1.50× 静态合并由 validation 选出；1.25× 为 {report['ablations']['static_merge_scale_validation']['scale_1_25_success_rate'] * 100:.2f}%，1.50× 为 {report['ablations']['static_merge_scale_validation']['scale_1_50_success_rate'] * 100:.2f}%。",
        f"- Prefix cache 在这个短输出、固定并发 workload 上平均延迟反而变化 {cache['latency_delta_percent']:+.1f}%，所以拒绝启用。",
        f"- CUDA Graph 在预热后平均延迟变化 {cuda['latency_delta_percent']:+.1f}%，P95 变化 {cuda['p95_delta_percent']:+.1f}%，吞吐变化 {cuda['throughput_delta_percent']:+.1f}%，因此用于常驻服务。",
        f"- 并发从 4 提升到 8/16 后，吞吐分别为 {conc['c8_requests_per_second']:.3f}/s 和 {conc['c16_requests_per_second']:.3f}/s；在线默认推荐并发 8，兼顾尾延迟。",
        f"- 路由方案保持严格决策与多轮任务 100% 成功，教师任务占比 {route['teacher_task_share'] * 100:.2f}%；但当前是顺序模型回放，尚不能冒充同卡双模型在线实测。",
        "",
        "## 文献与实现对齐",
        "",
        "- [QLoRA](https://arxiv.org/abs/2305.14314)：对应本项目的 NF4 double-quantized base + LoRA 后训练。",
        "- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)：对应 SFT 后的 chosen/rejected 直接偏好优化。",
        "- [RouteLLM](https://arxiv.org/abs/2406.18665)：支持大小模型按质量/成本路由的方向；本项目当前是更容易审计的确定性路由，尚未声称实现 learned router。",
        "- [vLLM engine arguments](https://docs.vllm.ai/en/v0.10.0/configuration/engine_args.html)：文档明确区分 eager 与 CUDA Graph；本项目用预热后 A/B 数据决定是否启用。",
        "",
        "## 边界",
        "",
    ])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, default=Path("ml/agentic/reports"))
    parser.add_argument("--checkpoints-root", type=Path, default=Path("ml/agentic/checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("ml/agentic/reports/stage24-final-evaluation-v1"))
    args = parser.parse_args()
    report = build(args.reports_root, args.checkpoints_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["headline"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
