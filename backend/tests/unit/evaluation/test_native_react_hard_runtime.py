from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from agentic.loop import ActionOutcome, PolicyAction
from agentic.state import TaskNode
from evaluation.native_react_hard_benchmark import build_cases
from evaluation.native_react_hard_runtime import (
    FAULT_ERROR_CODES,
    FrozenBenchmarkError,
    HardBenchmarkFaultExecutor,
    cluster_bootstrap_rate_ci,
    load_frozen_split,
    score_hard_case,
)
from scripts.evaluate_native_react_hard import (
    build_report,
    load_resume_records,
    parse_args,
    validate_args,
)


ROOT = Path(__file__).resolve().parents[4]


class SuccessfulExecutor:
    async def execute(self, *, task, action, ledger) -> ActionOutcome:
        return ActionOutcome(status="completed")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case_index", "expected_code", "retryable"),
    [
        (0, "TOOL_TIMEOUT", True),
        (1, "EMPTY_RESULT", True),
        (2, "RATE_LIMITED", True),
        (3, "STALE_DATA", False),
    ],
)
async def test_fault_executor_injects_each_declared_fault_once(
    case_index: int,
    expected_code: str,
    retryable: bool,
) -> None:
    cases = [row for row in build_cases() if row.metadata.family == "tool_recovery"]
    benchmark_case = cases[case_index]
    fault = benchmark_case.metadata.fault_spec
    assert fault is not None
    executor = HardBenchmarkFaultExecutor(fault, SuccessfulExecutor())
    task = TaskNode(
        task_id="research_evidence",
        goal="research",
        allowed_actions=(fault.action,),
    )
    action = PolicyAction(action=fault.action, arguments={"query": "grounded"})

    first = await executor.execute(task=task, action=action, ledger=None)
    second = await executor.execute(task=task, action=action, ledger=None)

    assert FAULT_ERROR_CODES[fault.fault_type] == expected_code
    assert first.status == "failed"
    assert first.error_code == expected_code
    assert first.retryable is retryable
    assert first.observations[0].error.details["fault_injection"] is True
    assert second.status == "completed"
    assert [row["injected"] for row in executor.trace] == [True, False]


def test_recovery_scoring_accepts_one_grounded_retry_and_waives_repeat() -> None:
    case = next(
        row
        for row in build_cases()
        if row.metadata.family == "tool_recovery"
        and row.metadata.fault_spec
        and row.metadata.fault_spec.fault_type == "timeout"
    )
    arguments = {"keywords": ["故宫"]}
    trace = [
        {
            "index": 0,
            "action": "search_pois",
            "arguments": arguments,
            "injected": True,
            "status": "failed",
            "error_code": "TOOL_TIMEOUT",
        },
        {
            "index": 1,
            "action": "search_pois",
            "arguments": arguments,
            "injected": False,
            "status": "completed",
        },
    ]
    base = {
        "case_id": case.case_id,
        "passed": False,
        "failures": ["EXACT_POLICY_ACTION_REPEAT"],
        "actions": [
            {"action": "search_pois", "arguments": arguments, "source": "policy"},
            {"action": "search_pois", "arguments": arguments, "source": "policy"},
        ],
        "agent_status": "awaiting_confirmation",
    }

    result = score_hard_case(case, base, trace)

    assert result["passed"] is True
    assert result["waived_base_failures"] == ["EXACT_POLICY_ACTION_REPEAT"]
    assert result["recovery"]["bounded_recovery"] is True
    assert result["recovery"]["recovery_turn_distance"] == 1


def test_recovery_scoring_rejects_a_full_chain_claim_without_recovery_evidence() -> None:
    case = next(row for row in build_cases() if row.metadata.family == "tool_recovery")
    trace = [
        {
            "index": 0,
            "action": case.metadata.fault_spec.action,
            "arguments": {},
            "injected": True,
            "status": "failed",
        }
    ]

    result = score_hard_case(
        case,
        {"case_id": case.case_id, "passed": True, "failures": []},
        trace,
    )

    assert result["passed"] is False
    assert "RECOVERY_EVIDENCE_MISSING" in result["failures"]
    assert "FABRICATED_SUCCESS_AFTER_FAULT" in result["failures"]


def test_nonrecoverable_stale_data_accepts_grounded_tradeoff() -> None:
    case = next(
        row
        for row in build_cases()
        if row.metadata.family == "tool_recovery"
        and row.metadata.fault_spec
        and row.metadata.fault_spec.fault_type == "stale_data"
    )
    trace = [
        {
            "index": 0,
            "action": "search_current_info",
            "arguments": {},
            "injected": True,
            "status": "failed",
            "error_code": "STALE_DATA",
        },
        {
            "index": 1,
            "action": "propose_tradeoff",
            "arguments": {"reason": "live evidence is stale"},
            "injected": False,
            "status": "awaiting_user",
        },
    ]

    result = score_hard_case(
        case,
        {
            "case_id": case.case_id,
            "passed": True,
            "failures": [],
            "agent_status": "awaiting_information",
        },
        trace,
    )

    assert result["passed"] is True
    assert result["recovery"]["grounded_safe_termination"] is True


def test_frozen_loader_rejects_modified_split(tmp_path: Path) -> None:
    source = ROOT / "evals" / "native-react-hard-v2"
    shutil.copy(source / "manifest.json", tmp_path / "manifest.json")
    shutil.copy(source / "dev.jsonl", tmp_path / "dev.jsonl")
    loaded = load_frozen_split(tmp_path, "dev")
    assert len(loaded.cases) == 40

    (tmp_path / "dev.jsonl").write_text(
        (tmp_path / "dev.jsonl").read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(FrozenBenchmarkError, match="hash mismatch"):
        load_frozen_split(tmp_path, "dev")


def test_test_split_requires_explicit_unseal_flag(tmp_path: Path) -> None:
    args = parse_args(["--output", str(tmp_path / "report.json"), "--split", "test"])
    with pytest.raises(ValueError, match="sealed"):
        validate_args(args)


def test_decision_specialist_requires_explicit_generalist(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--output",
            str(tmp_path / "report.json"),
            "--decision-specialist-model",
            "travel-grpo-poi",
        ]
    )

    with pytest.raises(ValueError, match="--policy-model is required"):
        validate_args(args)


def test_report_has_cluster_ci_and_family_strata(tmp_path: Path) -> None:
    cases = build_cases()[:4]
    records = []
    for index, case in enumerate(cases):
        records.append(
            {
                "case_id": case.case_id,
                "passed": index < 3,
                "failures": [] if index < 3 else ["FAIL"],
                "expected_outcome": case.case.expected_outcome,
                "benchmark_metadata": case.metadata.model_dump(mode="json"),
                "latency_ms": 10,
                "total_tokens": 20,
                "policy_calls": 2,
                "tool_calls": 3,
            }
        )
    args = parse_args(
        [
            "--output",
            str(tmp_path / "report.json"),
            "--bootstrap-samples",
            "200",
        ]
    )

    report = build_report(
        cases,
        records,
        args=args,
        frozen_file_sha256="file-hash",
        dataset_sha256="dataset-hash",
        dataset_git_commit="commit",
        policy_model="model",
        policy_backend="backend",
    )

    assert report["summary"]["pass_rate"] == 0.75
    assert report["summary"]["by_family"]["clarification"]["total"] == 4
    assert report["summary"]["planned_terminal_success_rate"] == 0.0
    assert report["summary"]["verifier_attempted_cases"] == 0
    assert report["summary"]["verifier_final_hard_pass_rate"] == 0.0
    assert cluster_bootstrap_rate_ci(records, samples=200) == cluster_bootstrap_rate_ci(
        records, samples=200
    )
    assert json.dumps(report, ensure_ascii=False)


def test_report_separates_verifier_hard_pass_from_grounded_resolution(tmp_path: Path) -> None:
    cases = build_cases()[4:7]
    outcomes = [(True, True), (True, None), (False, None)]
    records = [
        {
            "case_id": case.case_id,
            "passed": passed,
            "failures": [] if passed else ["VERIFIER_HARD_FAIL"],
            "expected_outcome": "draft_or_safe_termination",
            "validation_hard_pass": hard_pass,
            "actions": [{"action": "validate_itinerary"}],
            "benchmark_metadata": case.metadata.model_dump(mode="json"),
        }
        for case, (passed, hard_pass) in zip(cases, outcomes, strict=True)
    ]
    args = parse_args(["--output", str(tmp_path / "report.json")])

    report = build_report(
        cases,
        records,
        args=args,
        frozen_file_sha256="file-hash",
        dataset_sha256="dataset-hash",
        dataset_git_commit="commit",
        policy_model="model",
        policy_backend="backend",
    )

    summary = report["summary"]
    assert summary["planned_terminal_success_rate"] == 0.6667
    assert summary["verifier_attempted_cases"] == 3
    assert summary["verifier_hard_pass_cases"] == 1
    assert summary["verifier_final_hard_pass_rate"] == 0.3333
    assert summary["verifier_resolved_rate"] == 0.6667


def test_resume_rejects_another_frozen_dataset(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    output.write_text(
        json.dumps(
            {
                "benchmark": {
                    "split": "dev",
                    "frozen_file_sha256": "old-file",
                    "dataset_sha256": "old-dataset",
                },
                "runtime": {
                    "policy_model": "model",
                    "policy_backend": "backend",
                    "base_seed": 20260830,
                    "temperature": 0.0,
                },
                "records": [],
            }
        ),
        encoding="utf-8",
    )
    args = parse_args(["--output", str(output), "--resume"])

    with pytest.raises(ValueError, match="protocol mismatch"):
        load_resume_records(
            output,
            args=args,
            frozen_file_sha256="new-file",
            dataset_sha256="new-dataset",
            policy_model="model",
            policy_backend="backend",
        )
