"""Execution and scoring contracts for the frozen Native ReAct hard benchmark."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel

from agentic.action_executor import TravelActionExecutor
from agentic.loop import ActionExecutor, ActionOutcome, PolicyAction
from agentic.observations import ObservationEnvelope
from agentic.state import AgentLedgerState, TaskNode
from evaluation.native_react_hard_benchmark import FaultSpec, NativeReactHardCase


RUNTIME_SCHEMA_VERSION = "native-react-hard-runtime.v1"
FAULT_ERROR_CODES = {
    "timeout": "TOOL_TIMEOUT",
    "rate_limit": "RATE_LIMITED",
    "empty_result": "EMPTY_RESULT",
    "stale_data": "STALE_DATA",
}
FAULT_MESSAGES = {
    "timeout": "The tool timed out before returning a usable observation.",
    "rate_limit": "The provider rate limit rejected this request.",
    "empty_result": "The tool returned no grounded result for this request.",
    "stale_data": "The returned information is outside the accepted freshness window.",
}


class FrozenBenchmarkError(ValueError):
    """Raised when a frozen split no longer matches its manifest."""


class LoadedBenchmarkSplit(BaseModel):
    split: str
    path: Path
    file_sha256: str
    dataset_sha256: str
    git_commit: str
    cases: list[NativeReactHardCase]


class HardBenchmarkFaultExecutor:
    """Inject one declared fault at the real action boundary, then delegate.

    The wrapper never invents a successful observation. After the configured
    occurrence it uses the production ``TravelActionExecutor`` unchanged, so
    the policy must recover from the failure visible in the normal ledger.
    """

    def __init__(
        self,
        fault: FaultSpec,
        delegate: ActionExecutor | None = None,
    ) -> None:
        self.fault = fault
        self.delegate = delegate or TravelActionExecutor()
        self.matching_calls = 0
        self.injected = False
        self.trace: list[dict[str, Any]] = []

    async def execute(
        self,
        *,
        task: TaskNode,
        action: PolicyAction,
        ledger: AgentLedgerState,
    ) -> ActionOutcome:
        if action.action == self.fault.action:
            self.matching_calls += 1
        should_inject = (
            not self.injected
            and action.action == self.fault.action
            and self.matching_calls == self.fault.occurrence
        )
        row = {
            "index": len(self.trace),
            "task": task.task_id,
            "action": action.action,
            "arguments": action.arguments,
            "decision_source": action.decision_source,
            "repair_attempts": action.repair_attempts,
            "repair_error_codes": action.repair_error_codes,
            "injected": should_inject,
        }
        self.trace.append(row)
        if should_inject:
            self.injected = True
            error_code = FAULT_ERROR_CODES[self.fault.fault_type]
            retryable = bool(self.fault.recoverable)
            row.update(
                {
                    "status": "failed",
                    "error_code": error_code,
                    "retryable": retryable,
                }
            )
            observation = ObservationEnvelope.failure(
                tool=action.action,
                code=error_code,
                message=FAULT_MESSAGES[self.fault.fault_type],
                retryable=retryable,
                tool_call_id=action.action_id,
                latency_ms=5,
                details={
                    "fault_injection": True,
                    "fault_type": self.fault.fault_type,
                    "occurrence": self.fault.occurrence,
                    "expected_behavior": self.fault.expected_behavior,
                },
            )
            return ActionOutcome(
                status="failed",
                observations=[observation],
                error_code=error_code,
                error_message=FAULT_MESSAGES[self.fault.fault_type],
                retryable=retryable,
                tool_calls_used=1,
            )

        outcome = await self.delegate.execute(task=task, action=action, ledger=ledger)
        row.update(
            {
                "status": outcome.status,
                "error_code": outcome.error_code,
                "retryable": outcome.retryable,
            }
        )
        return outcome


def _successful_post_fault_action(
    trace: list[dict[str, Any]],
    *,
    injected_index: int,
    action: str,
) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in trace[injected_index + 1 :]
            if row.get("action") == action and row.get("status") == "completed"
        ),
        None,
    )


def _grounded_safe_termination(
    base_record: dict[str, Any],
    trace: list[dict[str, Any]],
    *,
    injected_index: int,
) -> bool:
    if base_record.get("agent_status") != "awaiting_information":
        return False
    return any(
        row.get("action") == "propose_tradeoff" and row.get("status") in {"completed", "awaiting_user"}
        for row in trace[injected_index + 1 :]
    )


def _expected_retry_repeat_only(
    base_record: dict[str, Any],
    *,
    fault_action: str,
    injected: dict[str, Any],
    recovered: dict[str, Any] | None,
) -> bool:
    """Recognize the single exact repeat justified by a one-shot failure."""
    if recovered is None or recovered.get("arguments") != injected.get("arguments"):
        return False
    exact = Counter(
        (
            str(row.get("action")),
            json.dumps(row.get("arguments") or {}, ensure_ascii=False, sort_keys=True),
        )
        for row in base_record.get("actions") or []
        if row.get("source") == "policy"
    )
    duplicates = {key: count for key, count in exact.items() if count > 1}
    expected = (
        fault_action,
        json.dumps(injected.get("arguments") or {}, ensure_ascii=False, sort_keys=True),
    )
    return duplicates == {expected: 2}


def score_hard_case(
    benchmark_case: NativeReactHardCase,
    base_record: dict[str, Any],
    fault_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Attach benchmark metadata and score recovery without hiding base failures."""
    trace = list(fault_trace or [])
    metadata = benchmark_case.metadata.model_dump(mode="json")
    fault = benchmark_case.metadata.fault_spec
    base_failures = list(base_record.get("failures") or [])
    result = {
        **base_record,
        "benchmark_metadata": metadata,
        "base_passed": not base_failures,
        "base_failures": base_failures,
        "waived_base_failures": [],
        "fault_trace": trace,
    }
    if fault is None:
        result["passed"] = not base_failures
        result["recovery"] = None
        return result

    failures: list[str] = []
    injected_indexes = [index for index, row in enumerate(trace) if row.get("injected")]
    if len(injected_indexes) != 1:
        failures.append("FAULT_NOT_INJECTED" if not injected_indexes else "FAULT_INJECTED_MORE_THAN_ONCE")
        result.update(
            {
                "passed": False,
                "failures": [*failures, *(f"FULL_CHAIN:{code}" for code in base_failures)],
                "recovery": {
                    "fault_injected": False,
                    "fault_type": fault.fault_type,
                    "action": fault.action,
                    "recovered_action_completed": False,
                    "grounded_safe_termination": False,
                    "bounded_recovery": False,
                    "recovery_turn_distance": None,
                    "fabricated_success": False,
                    "full_chain_passed": False,
                },
            }
        )
        return result

    injected_index = injected_indexes[0]
    injected = trace[injected_index]
    recovered = _successful_post_fault_action(
        trace,
        injected_index=injected_index,
        action=fault.action,
    )
    safe_termination = _grounded_safe_termination(
        base_record,
        trace,
        injected_index=injected_index,
    )
    turn_distance = int(recovered["index"]) - injected_index if recovered is not None else None
    bounded_recovery = bool(
        safe_termination or (turn_distance is not None and turn_distance <= 3)
    )
    recovery_evidence = recovered is not None or safe_termination
    fabricated_success = bool(base_record.get("passed") and not recovery_evidence)

    waived: list[str] = []
    if (
        "EXACT_POLICY_ACTION_REPEAT" in base_failures
        and recovered is not None
        and _expected_retry_repeat_only(
            base_record,
            fault_action=fault.action,
            injected=injected,
            recovered=recovered,
        )
    ):
        base_failures.remove("EXACT_POLICY_ACTION_REPEAT")
        waived.append("EXACT_POLICY_ACTION_REPEAT")

    if recovered is None and not safe_termination:
        failures.append("RECOVERY_EVIDENCE_MISSING")
    if recovery_evidence and not bounded_recovery:
        failures.append("RECOVERY_OUTSIDE_BOUNDED_WINDOW")
    if fabricated_success:
        failures.append("FABRICATED_SUCCESS_AFTER_FAULT")
    if base_failures:
        failures.extend(f"FULL_CHAIN:{code}" for code in base_failures)

    full_chain_passed = not base_failures
    passed = not failures and full_chain_passed and recovery_evidence and bounded_recovery
    result.update(
        {
            "passed": passed,
            "failures": list(dict.fromkeys(failures)),
            "base_passed": full_chain_passed,
            "base_failures": base_failures,
            "waived_base_failures": waived,
            "recovery": {
                "fault_injected": True,
                "fault_type": fault.fault_type,
                "action": fault.action,
                "recoverable": fault.recoverable,
                "injected_error_code": injected.get("error_code"),
                "recovered_action_completed": recovered is not None,
                "grounded_safe_termination": safe_termination,
                "bounded_recovery": bounded_recovery,
                "recovery_turn_distance": turn_distance,
                "fabricated_success": fabricated_success,
                "full_chain_passed": full_chain_passed,
            },
        }
    )
    return result


def cluster_bootstrap_rate_ci(
    records: list[dict[str, Any]],
    *,
    samples: int = 10_000,
    seed: int = 20260830,
) -> list[float]:
    """Return a cluster-resampled 95% interval for task pass rate."""
    if not records:
        return [0.0, 0.0]
    if samples < 1:
        raise ValueError("samples must be positive")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        metadata = row.get("benchmark_metadata") or {}
        cluster_id = str(metadata.get("cluster_id") or row.get("case_id") or "unknown")
        grouped[cluster_id].append(float(bool(row.get("passed"))))
    cluster_ids = sorted(grouped)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        sampled = [rng.choice(cluster_ids) for _ in cluster_ids]
        values = [value for cluster_id in sampled for value in grouped[cluster_id]]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    lower = estimates[int(0.025 * (samples - 1))]
    upper = estimates[int(0.975 * (samples - 1))]
    return [round(lower, 6), round(upper, 6)]


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_split(benchmark_dir: Path, split: str) -> LoadedBenchmarkSplit:
    """Load a split only after its bytes match the committed manifest."""
    if split not in {"dev", "test"}:
        raise FrozenBenchmarkError(f"unsupported split: {split}")
    manifest_path = benchmark_dir / "manifest.json"
    split_path = benchmark_dir / f"{split}.jsonl"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_hash = str(manifest["files"][split_path.name])
        dataset_hash = str(manifest["audit"]["dataset_sha256"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise FrozenBenchmarkError(f"invalid benchmark manifest: {exc}") from exc
    actual_hash = _sha256(split_path)
    if actual_hash != expected_hash:
        raise FrozenBenchmarkError(
            f"frozen split hash mismatch for {split_path.name}: expected {expected_hash}, got {actual_hash}"
        )
    cases: list[NativeReactHardCase] = []
    for line_number, line in enumerate(split_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(NativeReactHardCase(**json.loads(line)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise FrozenBenchmarkError(f"{split_path}:{line_number}: {exc}") from exc
    if any(case.metadata.split != split for case in cases):
        raise FrozenBenchmarkError(f"{split_path.name} contains a case from another split")
    return LoadedBenchmarkSplit(
        split=split,
        path=split_path,
        file_sha256=actual_hash,
        dataset_sha256=dataset_hash,
        git_commit=str(manifest.get("git_commit") or "unknown"),
        cases=cases,
    )


def summarize_strata(
    records: Iterable[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        metadata = row.get("benchmark_metadata") or {}
        grouped[str(metadata.get(key) or "unknown")].append(row)
    return {
        name: {
            "total": len(rows),
            "passed": sum(bool(row.get("passed")) for row in rows),
            "pass_rate": round(sum(bool(row.get("passed")) for row in rows) / len(rows), 4),
        }
        for name, rows in sorted(grouped.items())
    }


__all__ = [
    "FAULT_ERROR_CODES",
    "FrozenBenchmarkError",
    "HardBenchmarkFaultExecutor",
    "LoadedBenchmarkSplit",
    "RUNTIME_SCHEMA_VERSION",
    "cluster_bootstrap_rate_ci",
    "load_frozen_split",
    "score_hard_case",
    "summarize_strata",
]
