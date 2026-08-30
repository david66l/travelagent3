"""Build the Stage35 repeated-tool-call post-training decision report."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_preference_logprobs import (  # noqa: E402
    summarize_preference_scores,
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _score_evidence(path: Path) -> dict[str, Any]:
    rows = _jsonl(path / "scores.jsonl")
    summary = summarize_preference_scores(rows)["overall"]
    return {
        "pairs": summary["pairs"],
        "mean_token_preferred": summary["chosen_preferred"],
        "mean_token_accuracy": summary["preference_accuracy"],
        "mean_token_margin": summary["mean_logprob_margin"],
        "sequence_preferred": summary["sequence_chosen_preferred"],
        "sequence_accuracy": summary["sequence_preference_accuracy"],
        "sequence_margin": summary["mean_sequence_logprob_margin"],
        "scores_sha256": _sha256(path / "scores.jsonl"),
    }


def _training_evidence(path: Path) -> dict[str, Any]:
    report = _json(path / "training_report.json")
    return {
        "scope": report["run_scope"],
        "train_examples": report["train_examples"],
        "eval_examples": report["eval_examples"],
        "train_loss": report["train_metrics"]["train_loss"],
        "eval_loss": report["eval_metrics"]["eval_loss"],
        "eval_reward_accuracy": report["eval_metrics"]["eval_rewards/accuracies"],
        "eval_reward_margin": report["eval_metrics"]["eval_rewards/margins"],
        "training_report_sha256": _sha256(path / "training_report.json"),
    }


def _case_evidence(path: Path, case_id: str) -> dict[str, Any]:
    run = next(row for row in _jsonl(path / "runs.jsonl") if row["case_id"] == case_id)
    return {
        "case_id": case_id,
        "expected_action": run["expected_action"],
        "observed_actions": run["observed_actions"],
        "raw_success": run["success"],
        "policy_contract_success": run["policy_contract_success"],
        "http_error": run["http_error"],
        "completion_tokens": run["inference_metrics"]["completion_tokens"],
    }


def build(repo_root: Path) -> dict[str, Any]:
    datasets = repo_root / "ml" / "agentic" / "datasets"
    reports = repo_root / "ml" / "agentic" / "reports"
    single = _json(
        datasets / "qwen3-stage35-single-action-preferences-v1" / "manifest.json"
    )
    mixed = _json(datasets / "qwen3-stage35-preferences-mixed-v1" / "manifest.json")
    isolated = _json(
        datasets / "qwen3-stage35-isolated-action-preferences-v1" / "manifest.json"
    )
    hard = _json(reports / "stage35-simulated-hard-dpo-smoke20-v1" / "report.json")
    hard_run = _case_evidence(
        reports / "stage35-simulated-hard-dpo-smoke20-v1",
        "ext-v1-stage29-ds-065",
    )
    targeted_run = _case_evidence(
        reports / "stage35-targeted-dpo-smoke20-case065-v1",
        "ext-v1-stage29-ds-065",
    )
    isolated_run = _case_evidence(
        reports / "stage35-isolated-dpo-smoke20-case065-v1",
        "ext-v1-stage29-ds-065",
    )
    return {
        "schema_version": "stage35-repeated-call-posttraining-report.v1",
        "status": "formal_dpo_rejected",
        "target_failure": "model emits the same allowed tool call twice in one response",
        "datasets": {
            "single_action_contract": {
                "dataset_version": single["dataset_version"],
                "pairs": single["preference_pairs"],
                "split_counts": single["split_counts"],
                "action_counts": single["action_counts"],
                "frozen_exact_overlap": single["accepted_frozen_exact_overlap"],
                "frozen_near_overlap": single["accepted_frozen_near_overlap"],
                "manifest_sha256": _sha256(
                    datasets
                    / "qwen3-stage35-single-action-preferences-v1"
                    / "manifest.json"
                ),
            },
            "mixed_evidence": {
                "dataset_version": mixed["dataset_version"],
                "split_counts": mixed["split_counts"],
                "evidence_counts": mixed["evidence_counts"],
                "train_single_action_ratio": mixed["train_single_action_ratio"],
                "unique_contexts": mixed["unique_contexts"],
                "manifest_sha256": _sha256(
                    datasets / "qwen3-stage35-preferences-mixed-v1" / "manifest.json"
                ),
            },
            "isolated_action_contract": {
                "dataset_version": isolated["dataset_version"],
                "split_counts": isolated["split_counts"],
                "family_counts": isolated["family_counts"],
                "frozen_exact_overlap": isolated["accepted_frozen_exact_overlap"],
                "frozen_near_overlap": isolated["accepted_frozen_near_overlap"],
                "manifest_sha256": _sha256(
                    datasets
                    / "qwen3-stage35-isolated-action-preferences-v1"
                    / "manifest.json"
                ),
            },
        },
        "logprob_evidence": {
            "stage32_sft": _score_evidence(
                reports / "stage35-single-action-sft-logprobs-v1"
            ),
            "stage32_dpo": _score_evidence(
                reports / "stage35-single-action-dpo-logprobs-v1"
            ),
            "stage35_mixed_smoke20": _score_evidence(
                reports / "stage35-single-action-dpo-smoke20-logprobs-v1"
            ),
        },
        "training_smokes": {
            "mixed_20_steps": _training_evidence(
                reports / "stage35-dpo-smoke20-training-v1"
            ),
            "targeted_20_steps": _training_evidence(
                reports / "stage35-dpo-targeted20-training-v2"
            ),
            "isolated_action_20_steps": _training_evidence(
                reports / "stage35-dpo-isolated20-training-v3"
            ),
        },
        "generation_gates": {
            "mixed_150_cases": {
                "raw_successful": hard["summary"]["successful_runs"],
                "policy_contract_successful": hard["summary"][
                    "policy_contract_successful_runs"
                ],
                "http_errors": hard["summary"]["http_errors"],
                "p95_ms": hard["summary"]["inference"]["request_latency_ms"]["p95"],
                "target_case": hard_run,
            },
            "targeted_only_target_case": targeted_run,
            "isolated_action_target_case": isolated_run,
        },
        "decision": {
            "start_formal_dpo": False,
            "promote_any_stage35_smoke": False,
            "reason": (
                "All three smoke runs reached 100% DPO eval reward accuracy, and the "
                "mixed smoke improved offline log-probability margins, but every "
                "generation gate still emitted abort twice on the unchanged target case."
            ),
            "next_method": (
                "boundary-weighted single-call SFT or an explicit one-tool-call decoding "
                "constraint, followed by the same frozen generation gate"
            ),
        },
        "limitations": [
            "The isolated preference test has 24 synthetic-negative pairs, not human judgments.",
            "Only 20 optimization steps were used per smoke; these are method probes, not release candidates.",
            "The frozen 150-case set is synthetic and does not replace live Shadow evidence.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    logprob = report["logprob_evidence"]
    smokes = report["training_smokes"]
    gates = report["generation_gates"]
    lines = [
        "# Stage35：重复工具调用后训练实验",
        "",
        "> 结论：三轮 DPO smoke 的离线指标均通过，但生成故障未修复，因此不启动正式 DPO。",
        "",
        "## 数据",
        "",
        "- 单调用偏好：240 对，四类动作各 60，冻结评测精确/近重复均为 0；",
        "- 混合偏好：1331 对，其中 verifier 证据 1091、机械合同证据 240；",
        "- 动作隔离偏好：240 对，所有上下文只暴露 chosen 动作及对应工具 schema。",
        "",
        "## Log-prob 证据（24 条独立测试）",
        "",
        "| 模型 | mean-token 偏好 | mean-token margin | sequence margin |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in (
        ("Stage32 SFT", "stage32_sft"),
        ("Stage32 DPO", "stage32_dpo"),
        ("Stage35 mixed smoke20", "stage35_mixed_smoke20"),
    ):
        item = logprob[key]
        lines.append(
            f"| {label} | {item['mean_token_preferred']}/{item['pairs']} | "
            f"{item['mean_token_margin']:+.6f} | {item['sequence_margin']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## 三轮 smoke",
            "",
            "| 实验 | 训练 loss | 验证 loss | DPO 验证准确率 | 目标生成 |",
            "| --- | ---: | ---: | ---: | --- |",
            f"| 混合偏好 20 step | {smokes['mixed_20_steps']['train_loss']:.4f} | "
            f"{smokes['mixed_20_steps']['eval_loss']:.4f} | 100% | abort × 2 |",
            f"| 专项偏好 20 step | {smokes['targeted_20_steps']['train_loss']:.4f} | "
            f"{smokes['targeted_20_steps']['eval_loss']:.4f} | 100% | abort × 2 |",
            f"| 动作隔离 20 step | {smokes['isolated_action_20_steps']['train_loss']:.4f} | "
            f"{smokes['isolated_action_20_steps']['eval_loss']:.4f} | 100% | abort × 2 |",
            "",
            "混合 smoke 在 150 条冻结难例上仍为 "
            f"{gates['mixed_150_cases']['raw_successful']}/150 原标签正确、"
            f"{gates['mixed_150_cases']['policy_contract_successful']}/150 生产合同正确；"
            "重复 `abort` 故障未消失。",
            "",
            "## 决策",
            "",
            "- 不启动正式 DPO，不晋级任何 Stage35 smoke；",
            "- 不能用 100% DPO eval reward accuracy 代替真实生成门禁；",
            "- 下一方法改为边界加权单调用 SFT，或服务端单工具调用解码约束；",
            "- 新方法必须先修复同一冻结目标例，再跑完整 150 条回归。",
            "",
            "## 边界",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.repo_root.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
