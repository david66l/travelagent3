"""Build a paired Stage33 champion/challenger Shadow promotion report."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from evaluation.policy_shadow import (  # noqa: E402
    PolicyShadowGateConfig,
    compare_policy_shadow_runs,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _readme(payload: dict[str, Any]) -> str:
    champion = payload["champion"]
    challenger = payload["challenger"]
    ci_low, ci_high = payload["success_rate_delta_ci95"]
    family_rows = "\n".join(
        "| {family} | {paired_decisions} | {champion} | {challenger} | {delta} | {divergence} |".format(
            family=row["family"],
            paired_decisions=row["paired_decisions"],
            champion=_percent(row["champion_success_rate"]),
            challenger=_percent(row["challenger_success_rate"]),
            delta=_percent(row["success_rate_delta"]),
            divergence=_percent(row["action_divergence_rate"]),
        )
        for row in payload["family_comparisons"]
    )
    checks = "\n".join(
        f"| {row['code']} | {'通过' if row['passed'] else '未通过'} | "
        f"{row['actual']} | {row['expected']} |"
        for row in payload["checks"]
    )
    decision = (
        "可以进入 Canary"
        if payload["release_eligible"]
        else "不可进入 Canary，继续保持 Shadow"
    )
    return f"""# Stage33 策略模型成对 Shadow 报告

证据来源：`{payload['evidence_source']}`  
冠军：`{payload['champion_label']}`  
挑战者：`{payload['challenger_label']}`

## 结论

{decision}。质量门禁：`{str(payload['quality_gates_passed']).lower()}`；真实 Shadow
证据：`{str(payload['canary_evidence']).lower()}`；最终晋级：
`{str(payload['release_eligible']).lower()}`。

## 总体结果

- 合同一致配对：{payload['paired_decisions']}；排除标签/控制器合同冲突：{payload['excluded_label_contract_conflicts']}；
- 冠军正确率：{_percent(champion['policy_contract_success_rate'])}；
- 挑战者正确率：{_percent(challenger['policy_contract_success_rate'])}；
- 挑战者减冠军：{_percent(payload['success_rate_delta'])}；95% CI [{_percent(ci_low)}, {_percent(ci_high)}]；
- 冠军独赢/挑战者独赢：{payload['champion_only_successes']}/{payload['challenger_only_successes']}；
- McNemar 精确检验 p={payload['mcnemar_exact_pvalue']:.6f}；
- 动作分歧率：{_percent(payload['action_divergence_rate'])}；
- P95 延迟：冠军 {champion['p95_latency_ms']:.1f} ms，挑战者 {challenger['p95_latency_ms']:.1f} ms，比值 {payload['p95_latency_ratio']:.4f}x；
- HTTP 错误：冠军 {champion['http_errors']}，挑战者 {challenger['http_errors']}。

## 分任务族

| family | 配对 | 冠军正确率 | 挑战者正确率 | 差值 | 动作分歧率 |
|---|---:|---:|---:|---:|---:|
{family_rows}

## 晋级门禁

| 门禁 | 结果 | 实际值 | 要求 |
|---|---|---:|---|
{checks}

## 解释边界

冻结测试集和授权回放只能用于离线质量验证，不得标记为线上 Canary 证据。只有来源为
`live_shadow`、完成至少 300 个合同一致配对且其余质量/延迟门禁全部通过，才允许晋级。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion-runs", type=Path, required=True)
    parser.add_argument("--challenger-runs", type=Path, required=True)
    parser.add_argument("--champion-label", required=True)
    parser.add_argument("--challenger-label", required=True)
    parser.add_argument(
        "--evidence-source",
        choices=("sealed_benchmark", "authorized_replay", "live_shadow"),
        required=True,
    )
    parser.add_argument("--minimum-pairs", type=int, default=300)
    parser.add_argument("--minimum-success-rate", type=float, default=0.98)
    parser.add_argument("--noninferiority-margin", type=float, default=0.01)
    parser.add_argument("--maximum-http-error-rate", type=float, default=0.01)
    parser.add_argument("--maximum-p95-latency-ratio", type=float, default=1.25)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = compare_policy_shadow_runs(
        _read_jsonl(args.champion_runs),
        _read_jsonl(args.challenger_runs),
        evidence_source=args.evidence_source,
        gate=PolicyShadowGateConfig(
            minimum_paired_decisions=args.minimum_pairs,
            minimum_challenger_success_rate=args.minimum_success_rate,
            noninferiority_margin=args.noninferiority_margin,
            maximum_http_error_rate=args.maximum_http_error_rate,
            maximum_p95_latency_ratio=args.maximum_p95_latency_ratio,
        ),
    ).model_dump(mode="json")
    report.update(
        {
            "champion_label": args.champion_label,
            "challenger_label": args.challenger_label,
            "sources": {
                "champion_runs": str(args.champion_runs),
                "champion_runs_sha256": _sha256(args.champion_runs),
                "challenger_runs": str(args.challenger_runs),
                "challenger_runs_sha256": _sha256(args.challenger_runs),
            },
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(_readme(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
