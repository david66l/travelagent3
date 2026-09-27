"""Audit donor attribution and conservative target quality without changing scores.

This is a train-only quarantine filter, not a new evaluation metric. Original
rollouts and their frozen same-stack scores remain untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from agentic.reason_quality import repair_reason_semantic_conflict  # noqa: E402


SCHEMA_VERSION = "h006-donor-target-quality.v1"
_SCHEMA_ECHO = re.compile(
    r"grounded\s+failure\s+detail|the\s+system\s+adds|"
    r"bounded[- ]retry\s+rationale|fixed\s+bounded[- ]retry",
    re.IGNORECASE,
)
_RETRY_AMBIGUITY = re.compile(
    r"(?:无法|不能|不可).{0,3}(?:修复|重试|重算)|"
    r"(?:需|需要|必须)(?:由)?(?:用户|您).{0,5}(?:调整|放宽|选择|确认)"
)
_REPAIR_ACTIONS = {"retry_solve", "propose_tradeoff", "abort"}


def target_quality_findings(record: dict[str, Any]) -> list[str]:
    """Fail closed on missing sampled-step provenance and explicit bad targets."""
    findings: set[str] = set()
    gates = record.get("gates") or {}
    if gates.get("selection_eligible") is not True:
        findings.add("UPSTREAM_SELECTION_REJECTED")
    for name in (
        "raw_model_semantic_success",
        "assembled_system_success",
        "deterministic_verifier_success",
        "model_contract_compliant",
        "no_controller_override",
        "no_policy_rejection",
    ):
        if gates.get(name) is not True:
            findings.add("UPSTREAM_COMPONENT_GATE_REJECTED")
    actions = record.get("actions") or []
    episode = ((record.get("candidate") or {}).get("rollout") or {}).get(
        "episode"
    ) or {}
    steps = {step.get("step_index"): step for step in episode.get("steps", [])}
    if not actions:
        findings.add("NO_SAMPLED_ACTION")
    if record.get("target_action") and (
        len(actions) != 1 or gates.get("exactly_one_sampled_decision") is not True
    ):
        findings.add("DECISION_CARDINALITY_INVALID")
    seen: set[Any] = set()
    for action in actions:
        index = action.get("step_index")
        if not isinstance(index, int) or index in seen:
            findings.add("SAMPLED_STEP_INDEX_INVALID")
            continue
        seen.add(index)
        step_action = (steps.get(index) or {}).get("action") or {}
        if not step_action or step_action.get("decision_source") != "policy":
            findings.add("SAMPLED_STEP_NOT_MODEL_OWNED")
        if action.get("action") != step_action.get("action"):
            findings.add("SAMPLED_ACTION_MISMATCH")
        raw = action.get("model_raw_arguments")
        if not isinstance(raw, dict) or raw != step_action.get("model_arguments"):
            findings.add("SAMPLED_RAW_ARGUMENTS_MISMATCH")
        metrics = action.get("inference_metrics") or {}
        if not metrics.get("model") or int(metrics.get("completion_tokens") or 0) <= 0:
            findings.add("SAMPLED_INFERENCE_PROVENANCE_MISSING")
        name = str(action.get("action") or "")
        reason = str((raw or {}).get("reason") or "") if isinstance(raw, dict) else ""
        if _SCHEMA_ECHO.search(reason):
            findings.add("REASON_SCHEMA_DESCRIPTION_ECHO")
        if name in _REPAIR_ACTIONS and repair_reason_semantic_conflict(reason, name):
            findings.add("REASON_ACTION_CONFLICT")
        if name == "retry_solve" and _RETRY_AMBIGUITY.search(reason):
            findings.add("RETRY_REASON_AMBIGUOUS_OR_CONTRADICTORY")
    return sorted(findings)


def audit_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    upstream_sources: set[str] = set()
    safe_sources: set[str] = set()
    safe_by_target: dict[str, set[str]] = {}
    findings: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    policy_errors: Counter[str] = Counter()
    eligible = 0
    retry_records = 0
    retry_execution_successes = 0
    for record in records:
        source = str(record["source_state_id"])
        target = str(record.get("target_action") or "normal_agent_loop")
        codes = target_quality_findings(record)
        findings.update(codes)
        if (record.get("gates") or {}).get("selection_eligible"):
            upstream_sources.add(source)
        for action in record.get("actions") or []:
            actions[str(action.get("action"))] += 1
        policy_errors.update(
            str(error.get("detail_code") or error.get("code") or "UNKNOWN")
            for error in record.get("policy_errors") or []
        )
        if target == "retry_solve":
            retry_records += 1
            retry_execution_successes += int(
                bool((record.get("gates") or {}).get("deterministic_verifier_success"))
            )
        if codes:
            continue
        eligible += 1
        safe_sources.add(source)
        safe_by_target.setdefault(target, set()).add(source)
    return {
        "rollouts": len(records),
        "upstream_successful_sources": len(upstream_sources),
        "post_audit_eligible_rollouts": eligible,
        "post_audit_successful_sources": len(safe_sources),
        "post_audit_sources_per_target": {
            name: len(values) for name, values in sorted(safe_by_target.items())
        },
        "finding_counts": dict(sorted(findings.items())),
        "sampled_action_counts": dict(sorted(actions.items())),
        "policy_error_counts": dict(sorted(policy_errors.items())),
        "retry_target_rollouts": retry_records,
        "retry_target_execution_successes": retry_execution_successes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.output}")
    inputs = {}
    for path in args.rollouts:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        inputs[str(path)] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            **audit_records(records),
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_authorized": False,
        "scope": "train-only target quarantine; frozen evaluation scores are unchanged",
        "inputs": inputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
