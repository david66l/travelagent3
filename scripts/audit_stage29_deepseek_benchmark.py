"""Audit the frozen Stage29 DeepSeek external-model benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from evaluation.external_benchmark import (  # noqa: E402
    BenchmarkSplit,
    ExternalBenchmarkCase,
    ForbiddenCorpusDocument,
    annotation_agreement,
    audit_split_isolation,
    audit_training_contamination,
    canonical_hash,
    normalized_prompt_hash,
)

from scripts.audit_external_benchmark import read_forbidden_documents  # noqa: E402


EXPECTED_ACTIONS = {
    "search_pois": 45,
    "ask_user": 35,
    "propose_tradeoff": 40,
    "abort": 30,
}
EXPECTED_SOURCES = {
    "authorized_real_or_simulated": 85,
    "tool_failure": 30,
    "long_context_replan": 35,
}


def _read_cases(path: Path) -> list[ExternalBenchmarkCase]:
    return [
        ExternalBenchmarkCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _final_action(case: ExternalBenchmarkCase) -> str:
    if case.adjudication is not None:
        return case.adjudication.primary_action
    return case.annotations[0].primary_action


def audit_stage29(
    dev: list[ExternalBenchmarkCase],
    sealed: list[ExternalBenchmarkCase],
    documents: list[ForbiddenCorpusDocument],
) -> dict[str, Any]:
    cases = dev + sealed
    source_counts = Counter(case.source.value for case in cases)
    action_counts = Counter(_final_action(case) for case in cases)
    disagreements = [
        case
        for case in cases
        if len(case.annotations) >= 2
        and case.annotations[0].primary_action != case.annotations[1].primary_action
    ]
    unresolved = [case for case in disagreements if case.adjudication is None]
    termination_by_action = {
        "search_pois": "plan",
        "ask_user": "clarification",
        "propose_tradeoff": "tradeoff",
        "abort": "safe_abort",
    }
    action_contract_mismatches = [
        case.case_id
        for case in cases
        if case.expected_termination.value != termination_by_action[_final_action(case)]
        or not case.argument_rules
        or case.argument_rules[0].action != _final_action(case)
    ]
    isolation = audit_split_isolation(dev, sealed)
    agreement = annotation_agreement(cases)
    contamination = audit_training_contamination(cases, documents)
    gates = {
        "total_150": len(cases) == 150,
        "dev_30": len(dev) == 30,
        "sealed_test_120": len(sealed) == 120,
        "source_composition_exact": dict(source_counts) == EXPECTED_SOURCES,
        "action_composition_exact": dict(action_counts) == EXPECTED_ACTIONS,
        "unique_case_ids": len({case.case_id for case in cases}) == len(cases),
        "unique_normalized_prompts": len({normalized_prompt_hash(case) for case in cases}) == len(cases),
        "all_synthetic_provenance": all(
            case.provenance.authoring_method == "simulated" for case in cases
        ),
        "double_annotation_and_kappa": agreement["passed"],
        "all_conflicts_adjudicated": not unresolved,
        "action_contract_consistent": not action_contract_mismatches,
        "split_isolation": isolation["passed"],
        "forbidden_corpora_registered": bool(documents),
        "training_contamination": contamination["passed"],
    }
    return {
        "schema_version": "travel-agent-stage29-deepseek-benchmark-audit.v1",
        "passed": all(gates.values()),
        "gates": gates,
        "counts": {
            "total": len(cases),
            "dev": len(dev),
            "sealed_test": len(sealed),
            "sources": dict(source_counts),
            "actions": dict(action_counts),
            "annotation_disagreements": len(disagreements),
            "unresolved_conflicts": len(unresolved),
        },
        "annotation_agreement": agreement,
        "split_isolation": isolation,
        "training_contamination": contamination,
        "action_contract_mismatch_case_ids": action_contract_mismatches,
        "content_sha256": canonical_hash(
            sorted(canonical_hash(case.model_dump(mode="json")) for case in cases)
        ),
        "claim_boundary": {
            "external_model_evaluation": True,
            "independent_human_benchmark": False,
            "reason": "DeepSeek V4 Flash authored and double-annotated all synthetic cases; Codex adjudicated conflicts.",
        },
    }


def _render_report(report: dict[str, Any]) -> str:
    status = "PASS" if report["passed"] else "FAIL"
    counts = report["counts"]
    agreement = report["annotation_agreement"]
    contamination = report["training_contamination"]
    isolation = report["split_isolation"]
    gate_lines = [
        f"- {'通过' if passed else '未通过'}：`{name}`"
        for name, passed in report["gates"].items()
    ]
    return "\n".join(
        [
            "# Stage29 DeepSeek 外部模型评测集审计报告",
            "",
            f"> 总结论：**{status}**",
            "",
            "## 核心结果",
            "",
            f"- 总题数：{counts['total']}（Dev {counts['dev']} / Sealed Test {counts['sealed_test']}）",
            f"- 双标 Cohen's kappa：{agreement['primary_action_kappa']}",
            f"- 标注分歧：{counts['annotation_disagreements']}；未裁决：{counts['unresolved_conflicts']}",
            f"- 训练语料精确重合：{contamination['exact_matches']}",
            f"- 训练语料近重复：{contamination['near_duplicate_matches']}",
            f"- 最高字符 5-gram Jaccard：{contamination['max_similarity']}",
            f"- Dev/Test 隔离：{'通过' if isolation['passed'] else '未通过'}",
            f"- 冻结内容 SHA-256：`{report['content_sha256']}`",
            "",
            "## 审计门",
            "",
            *gate_lines,
            "",
            "## 声明边界",
            "",
            "该数据集可用于外部模型评测，但不是独立人工 benchmark。题目由 DeepSeek V4 Flash 合成，",
            "两路盲标也来自同一模型家族，冲突由 Codex 裁决。",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--sealed-test", type=Path, required=True)
    parser.add_argument("--forbidden-corpus", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dev = _read_cases(args.dev)
    sealed = _read_cases(args.sealed_test)
    documents = read_forbidden_documents(args.forbidden_corpus)
    report = audit_stage29(dev, sealed, documents)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(_render_report(report), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "gates": report["gates"]}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
