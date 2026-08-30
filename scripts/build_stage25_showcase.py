"""Build a presentation-ready showcase from frozen TravelAgent evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROUTE_SOURCES = {
    "clarification": ("student", "stage23-preference-hard-dpo-scale150-cudagraph-steady-v1"),
    "search": ("student", "stage23-preference-hard-dpo-scale150-cudagraph-steady-v1"),
    "recovery": ("student", "stage23-preference-hard-dpo-scale150-cudagraph-steady-v1"),
    "tradeoff": ("teacher", "stage23-tradeoff-narrow-qwen3-8b-teacher-v1"),
}

ROUTE_REASONS = {
    "clarification": "缺少只能由用户提供的信息，交给高频动作学生模型追问",
    "search": "候选搜索是高频、动作空间明确的任务，交给学生模型",
    "recovery": "搜索失败后的有界重试属于学生模型蒸馏课程",
    "tradeoff": "硬约束冲突与安全终止需要更强判断，升级到 8B 教师",
}

DEMO_PROMPTS = {
    "clarification": "帮我规划 3 天西安行程，但我还没确定预算。",
    "search": "在上海找适合亲子的一日游候选景点。",
    "recovery": "上一次景点搜索为空，请调整关键词后重试。",
    "tradeoff": "预算无法同时覆盖两个必去景点，请给出可解释的取舍方案。",
}


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


def select_successful_example(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("family") == family and row.get("success")]
    if not candidates:
        raise ValueError(f"no successful example for family {family}")
    row = sorted(candidates, key=lambda item: (item["case_id"], item.get("repetition", 0)))[0]
    metrics = row["inference_metrics"]
    return {
        "case_id": row["case_id"],
        "family": family,
        "expected_action": row["expected_action"],
        "observed_actions": row["observed_actions"],
        "observed_arguments": row["observed_arguments"],
        "model": metrics["model"],
        "completion_tokens": metrics["completion_tokens"],
        "request_latency_ms": metrics["request_latency_ms"],
        "ttft_ms": metrics["ttft_ms"],
        "tpot_ms": metrics["tpot_ms"],
    }


def build(reports_root: Path) -> dict[str, Any]:
    final_path = reports_root / "stage24-final-evaluation-v1" / "report.json"
    final = _read_json(final_path)
    if final.get("status") != "passed":
        raise ValueError("Stage 24 report has not passed")

    loaded_sources: dict[str, tuple[list[dict[str, Any]], Path]] = {}
    scenarios = []
    for family, (route, directory) in ROUTE_SOURCES.items():
        if directory not in loaded_sources:
            runs_path = reports_root / directory / "runs.jsonl"
            loaded_sources[directory] = (_read_jsonl(runs_path), runs_path)
        rows, runs_path = loaded_sources[directory]
        scenarios.append(
            {
                "demo_prompt": DEMO_PROMPTS[family],
                "route": route,
                "route_reason": ROUTE_REASONS[family],
                "actual_frozen_example": select_successful_example(rows, family),
                "source": {
                    "path": str(runs_path.as_posix()),
                    "sha256": _sha256(runs_path),
                },
            }
        )

    headline = final["headline"]
    grpo = final["training_evidence"]["trajectory_grpo_b0"]
    preference = final["ablations"]["preference_optimization"]
    return {
        "schema_version": "travel-agent-stage25-showcase.v1",
        "status": "ready",
        "title": "TravelAgent：可验证后训练与大小模型路由",
        "elevator_pitch": (
            "将旅行规划的高频工具决策蒸馏到 Qwen3-4B，用 DPO 强化偏好边界，"
            "复杂权衡路由到 Qwen3-8B，并用程序化 verifier、冻结 holdout 和服务压测闭环验收。"
        ),
        "headline_metrics": {
            "strict_tool_decisions": headline["strict_success"],
            "multi_turn_tasks": headline["rollout_success"],
            "mean_reward": headline["mean_reward"],
            "teacher_task_share": headline["teacher_call_share"],
            "token_reduction_vs_all_teacher_percent": headline["token_reduction_vs_all_teacher_percent"],
            "model_latency_reduction_vs_all_teacher_percent": headline["latency_reduction_vs_all_teacher_percent"],
        },
        "post_training_evidence": {
            "distillation_sft": final["training_evidence"]["sft"],
            "dpo": {
                **final["training_evidence"]["dpo"],
                "paired_test_margin_delta_percent": preference["margin_delta_percent"],
            },
            "trajectory_grpo_b0": grpo,
        },
        "routing_demo_scenarios": scenarios,
        "failure_recovery_contract": {
            "student_failure": "同一步仅升级一次 8B 教师，并记录稳定错误码",
            "teacher_failure": "不吞异常，交给现有确定性 fallback/guard",
            "trajectory_visibility": "每个动作保存 requested target、executed target、reason、fallback、model、tokens 和 latency",
            "ui_visibility": "App 顶栏显示本轮 4B/8B 调用数与升级次数",
        },
        "deployment_config": {
            "AGENTIC_POLICY_MODE": "agent",
            "AGENTIC_POLICY_BACKEND": "api",
            "AGENTIC_POLICY_PROTOCOL": "native_tool",
            "AGENTIC_POLICY_ROUTING_ENABLED": "true",
            "AGENTIC_STUDENT_POLICY_MODEL": "travel-policy-qwen3-4b-dpo-scale150-cudagraph",
            "AGENTIC_TEACHER_POLICY_MODEL": "travel-policy-qwen3-8b-base",
            "AGENTIC_STUDENT_BASE_URL": "http://student-vllm:8000/v1",
            "AGENTIC_TEACHER_BASE_URL": "http://teacher-vllm:8000/v1",
        },
        "portfolio_claims": [
            {
                "claim": "完成 verifier-guided 蒸馏 SFT 与 DPO 后训练，并建立数据去重、split 隔离、训练预检和冻结评测。",
                "evidence": f"SFT 预检 {final['training_evidence']['sft']['examples_checked']} 条；DPO {final['training_evidence']['dpo']['unique_pairs']} 个唯一偏好对；paired margin +{preference['margin_delta_percent']:.1f}%",
                "boundary": "冻结集为项目工程 benchmark，不是公开排行榜。",
            },
            {
                "claim": "完成真实多轮 trajectory-level GRPO-B0 与 checkpoint 晋升门。",
                "evidence": f"Qwen2.5-3B 同协议 SFT {grpo['sft_success_rate'] * 100:.2f}% → SFT+GRPO {grpo['sft_grpo_success_rate'] * 100:.2f}%",
                "boundary": "不宣称已解决逐轮信用分配。",
            },
            {
                "claim": "实现 4B/8B 可配置路由、学生失败升级和动作级可观察性。",
                "evidence": f"严格决策 {headline['strict_success']}；多轮任务 {headline['rollout_success']}；教师任务占比 {headline['teacher_call_share'] * 100:.2f}%",
                "boundary": "最终对比为顺序模型回放；24GB 单卡未做双模型同时常驻。",
            },
            {
                "claim": "通过静态合并、CUDA Graph 和并发消融优化推理服务。",
                "evidence": "CUDA Graph 预热后平均延迟 -7.5%、P95 -9.7%、吞吐 +8.1%；并发 8 为在线推荐点。",
                "boundary": "模型延迟不包含外部工具 API 时间。",
            },
        ],
        "source": {"path": str(final_path.as_posix()), "sha256": _sha256(final_path)},
    }


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["headline_metrics"]
    lines = [
        f"# {report['title']}",
        "",
        report["elevator_pitch"],
        "",
        "## 演示首页数字",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| 严格工具决策 | {metrics['strict_tool_decisions']} |",
        f"| 完整多轮任务 | {metrics['multi_turn_tasks']} |",
        f"| 平均 Reward | {metrics['mean_reward']:.6f} |",
        f"| 8B 教师任务占比 | {metrics['teacher_task_share'] * 100:.2f}% |",
        f"| 相比全量 8B 的 Token | -{metrics['token_reduction_vs_all_teacher_percent']:.1f}% |",
        f"| 相比全量 8B 的模型延迟 | -{metrics['model_latency_reduction_vs_all_teacher_percent']:.1f}% |",
        "",
        "## 四个现场演示场景",
        "",
        "| 场景 | 演示输入 | 路由 | 实测动作 | Token | 模型延迟 |",
        "|---|---|---|---|---:|---:|",
    ]
    for scenario in report["routing_demo_scenarios"]:
        example = scenario["actual_frozen_example"]
        lines.append(
            f"| {example['family']} | {scenario['demo_prompt']} | {scenario['route']} | {example['observed_actions'][0]} | {example['completion_tokens']} | {example['request_latency_ms']:.1f} ms |"
        )
    lines.extend([
        "",
        "演示时点击顶栏“智能路由”区域观察本轮 4B/8B 调用数量；如果故意让学生端点返回非法工具调用，可看到一次教师升级及稳定错误码。",
        "",
        "## 面试可讲的后训练闭环",
        "",
        "```text",
        "8B verifier-guided 轨迹 → 去重/隔离 → 4B QLoRA SFT",
        "→ chosen/rejected 难例 → DPO → validation 选择静态 merge scale",
        "→ frozen 单步 + 多轮评测 → 4B/8B Router → CUDA Graph/并发消融",
        "```",
        "",
        "此外，项目在 Qwen2.5-3B 上完成了真实多轮 GRPO-B0：同协议 SFT 80.47% → SFT+GRPO 82.81%。这是轨迹级工程基线，不包装成已经解决长程信用分配。",
        "",
        "## 部署配置",
        "",
        "```env",
    ])
    lines.extend(f"{key}={value}" for key, value in report["deployment_config"].items())
    lines.extend([
        "```",
        "",
        "> 24GB 单卡不能稳定同时常驻当前 4B/8B FP16 服务。正式演示使用两个端点；只有一张卡时使用顺序服务，并明确标注不是同时在线测量。",
        "",
        "## 可陈述能力与边界",
        "",
    ])
    for item in report["portfolio_claims"]:
        lines.append(f"- {item['claim']} 证据：{item['evidence']} 边界：{item['boundary']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, default=Path("ml/agentic/reports"))
    parser.add_argument("--output-dir", type=Path, default=Path("ml/agentic/reports/stage25-showcase-v1"))
    args = parser.parse_args()
    report = build(args.reports_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "showcase.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "SHOWCASE.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["headline_metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
