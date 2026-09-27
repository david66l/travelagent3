"""Read-only H-005 replay for historical verifier-repair rollout artifacts.

This command never runs a model or an environment.  It joins historical raw
policy outputs to the exact train-internal corpus, applies the production
reason assembly function, and reports raw-model and assembled-system outcomes
separately.  The source rollout file is hashed before and after replay so the
report carries explicit evidence that the historical artifact was not edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.grpo_training import load_grpo_corpus  # noqa: E402
from agentic.reason_quality import (  # noqa: E402
    REPAIR_REASON_ASSEMBLY_VERSION,
    assemble_repair_reason,
    normalize_reason_text,
    verifier_reason_quality_checks,
    verifier_reason_semantic_checks,
)
from scripts.audit_model_curriculum import verifier_repair_metadata  # noqa: E402
from scripts.audit_model_curriculum import paired_rollout_seed  # noqa: E402


SCHEMA_VERSION = "reason-assembly-offline-replay.v1"
SUPPORTED_ACTIONS = {"abort", "propose_tradeoff", "retry_solve"}
QUALITY_KEYS = (
    "grounding_match",
    "reason_specificity_match",
    "reason_language_match",
    "reason_public_language",
    "reason_action_rationale_match",
)
SEMANTIC_KEYS = (
    "grounding_match",
    "reason_specificity_match",
    "reason_language_match",
    "reason_public_language",
    "reason_action_semantic_consistent",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: Any) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _json_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"no rollout rows found: {path}")
    return rows


def _visible_violation(contract: dict[str, Any]) -> str:
    for message in reversed(list(contract.get("prompt_messages") or [])):
        try:
            payload = json.loads(str(message.get("content") or "{}"))
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            continue
        state = payload.get("policy_state")
        if not isinstance(state, dict):
            continue
        reports = [
            item
            for item in state.get("relevant_artifacts") or []
            if isinstance(item, dict) and item.get("artifact_type") == "validation_report"
        ]
        violations = reports[-1].get("violations") or [] if reports else []
        if violations and isinstance(violations[0], dict):
            return str(violations[0].get("message") or "").strip()
    return ""


def _raw_arguments(row: dict[str, Any]) -> dict[str, Any] | None:
    candidates = (
        row.get("model_raw_arguments"),
        (row.get("generation_audit") or {}).get("model_raw_arguments"),
    )
    for candidate in candidates:
        if isinstance(candidate, dict):
            return dict(candidate)
    return None


def _authorized_arguments(row: dict[str, Any]) -> dict[str, Any] | None:
    value = row.get("observed_arguments")
    return dict(value) if isinstance(value, dict) else None


def _controller_hydration_exact(
    *,
    target_action: str,
    raw_arguments: dict[str, Any],
    authorized_arguments: dict[str, Any],
    contract: dict[str, Any],
) -> bool:
    expected = dict(contract.get("controller_arguments") or {})
    if target_action == "propose_tradeoff":
        expected["options"] = list(contract.get("supervised_options") or [])
    if any(key in raw_arguments for key in expected):
        return False
    return all(authorized_arguments.get(key) == value for key, value in expected.items())


def _controller_argument_checks(
    *,
    target_action: str,
    authorized_arguments: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, bool]:
    expected = dict(contract.get("controller_arguments") or {})
    checks = {
        "controller_arguments_match": all(
            authorized_arguments.get(key) == value for key, value in expected.items()
        )
    }
    if target_action == "propose_tradeoff":
        expected_options = list(contract.get("supervised_options") or [])
        submitted_options = authorized_arguments.get("options")
        checks.update(
            {
                "options_present": isinstance(submitted_options, list)
                and bool(submitted_options),
                "option_contract_present": bool(expected_options),
                "options_supported": submitted_options == expected_options,
                "option_contract_covered": submitted_options == expected_options,
            }
        )
    return checks


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def replay_row(
    row: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Replay one historical row without executing or mutating its trajectory."""

    source_row_sha_before = _json_sha256(row)
    source_checks = dict(row.get("checks") or {})
    target_action = str(contract.get("target_action") or row.get("target_action") or "")
    observed_action = str(row.get("observed_action") or "")
    raw_arguments = _raw_arguments(row)
    authorized_arguments = _authorized_arguments(row)
    raw_reason = raw_arguments.get("reason") if raw_arguments is not None else None
    evidence = _visible_violation(contract) or "；".join(
        str(item) for item in contract.get("grounding_phrases") or []
    )
    phrases = [
        str(item)
        for item in contract.get("grounding_phrases") or []
        if str(item).strip()
    ]

    raw_quality = {key: False for key in QUALITY_KEYS}
    raw_semantic = {key: False for key in SEMANTIC_KEYS}
    if raw_reason is not None and target_action in SUPPORTED_ACTIONS:
        raw_quality = verifier_reason_quality_checks(
            reason=raw_reason,
            target_action=target_action,
            grounding_phrases=phrases,
            evidence=evidence,
        )
        raw_semantic = verifier_reason_semantic_checks(
            reason=raw_reason,
            target_action=target_action,
            grounding_phrases=phrases,
            evidence=evidence,
        )

    assembled_reason: str | None = None
    assembly_error: str | None = None
    if raw_arguments is None or authorized_arguments is None:
        assembly_error = "RAW_OR_AUTHORIZED_ARGUMENTS_UNAVAILABLE"
    elif observed_action not in SUPPORTED_ACTIONS:
        assembly_error = "UNSUPPORTED_OBSERVED_ACTION"
    else:
        try:
            assembled_reason = assemble_repair_reason(raw_reason, observed_action)
        except ValueError as exc:
            assembly_error = str(exc)

    replay_arguments = dict(authorized_arguments or {})
    if assembled_reason is not None:
        replay_arguments["reason"] = assembled_reason
    assembled_quality = {key: False for key in QUALITY_KEYS}
    if assembled_reason is not None and target_action in SUPPORTED_ACTIONS:
        assembled_quality = verifier_reason_quality_checks(
            reason=assembled_reason,
            target_action=target_action,
            grounding_phrases=phrases,
            evidence=evidence,
        )

    action_match = observed_action == target_action
    structural_valid = bool(source_checks.get("single_policy_call")) and bool(
        source_checks.get("no_policy_call_rejections")
    )
    execution_valid = bool(source_checks.get("execution_valid"))
    model_contract_compliant = bool(row.get("model_contract_compliant"))
    no_controller_override = not bool(row.get("controller_override_attempt"))
    controller_hydration_exact = False
    controller_checks = {"controller_arguments_match": False}
    if raw_arguments is not None and authorized_arguments is not None:
        controller_hydration_exact = _controller_hydration_exact(
            target_action=target_action,
            raw_arguments=raw_arguments,
            authorized_arguments=authorized_arguments,
            contract=contract,
        )
        controller_checks = _controller_argument_checks(
            target_action=target_action,
            authorized_arguments=replay_arguments,
            contract=contract,
        )

    expected_arguments = contract.get("expected_arguments")
    expected_arguments_match = isinstance(expected_arguments, dict) and all(
        replay_arguments.get(key) == value for key, value in expected_arguments.items()
    )
    system_success = (
        assembled_reason is not None
        and structural_valid
        and action_match
        and execution_valid
        and expected_arguments_match
        and all(controller_checks.values())
        and all(assembled_quality.values())
    )
    model_semantic_success = (
        raw_arguments is not None
        and structural_valid
        and model_contract_compliant
        and no_controller_override
        and controller_hydration_exact
        and action_match
        and execution_valid
        and expected_arguments_match
        and all(controller_checks.values())
        and all(raw_semantic.values())
    )
    raw_legacy_success = (
        bool(source_checks.get("decision_step_valid"))
        and float(row.get("reward") or 0.0) == 1.0
        and model_contract_compliant
        and no_controller_override
        and (
            target_action != "propose_tradeoff"
            or row.get("controller_hydration_exact") is True
        )
    )

    raw_non_reason = dict(authorized_arguments or {})
    replay_non_reason = dict(replay_arguments)
    raw_non_reason.pop("reason", None)
    replay_non_reason.pop("reason", None)
    evidence_metrics_unchanged = (
        assembled_reason is not None
        and raw_quality["grounding_match"] == assembled_quality["grounding_match"]
        and raw_quality["reason_specificity_match"]
        == assembled_quality["reason_specificity_match"]
        and all(
            (normalize_reason_text(phrase) in normalize_reason_text(raw_reason))
            == (normalize_reason_text(phrase) in normalize_reason_text(assembled_reason))
            for phrase in phrases
        )
    )
    generation_audit = row.get("generation_audit") or {}
    raw_output_sha_before = generation_audit.get("raw_output_sha256")
    raw_output_sha_after = (row.get("generation_audit") or {}).get("raw_output_sha256")
    action_before = row.get("observed_action")
    action_after = observed_action or None
    state_fingerprint_before = row.get("initial_state_fingerprint")
    state_fingerprint_after = row.get("initial_state_fingerprint")
    trace_projection_before = {
        "action": action_before,
        "arguments_without_reason": raw_non_reason,
        "initial_state_fingerprint": state_fingerprint_before,
    }
    trace_projection_after = {
        "action": action_after,
        "arguments_without_reason": replay_non_reason,
        "initial_state_fingerprint": state_fingerprint_after,
    }
    source_row_sha_after = _json_sha256(row)
    promotion_composite_success = (
        model_semantic_success
        and system_success
        and raw_non_reason == replay_non_reason
        and evidence_metrics_unchanged
        and source_row_sha_before == source_row_sha_after
    )
    return {
        "task_id": row.get("task_id"),
        "source_task_id": row.get("source_task_id"),
        "sample_index": row.get("sample_index"),
        "rollout_seed": row.get("rollout_seed"),
        "target_action": target_action,
        "observed_action": observed_action or None,
        "raw_model_legacy_success": raw_legacy_success,
        "raw_model_contract_success": model_semantic_success,
        "assembled_system_success": system_success,
        "promotion_composite_success": promotion_composite_success,
        "assembly_succeeded": assembled_reason is not None,
        "assembly_error": assembly_error,
        "raw_reason_sha256": _text_sha256(raw_reason),
        "assembled_reason_sha256": _text_sha256(assembled_reason),
        "raw_reason_quality": raw_quality,
        "raw_model_semantic_quality": raw_semantic,
        "assembled_reason_quality": assembled_quality,
        "invariants": {
            "action_before": action_before,
            "action_after": action_after,
            "action_unchanged": action_before == action_after,
            "non_reason_arguments_unchanged": raw_non_reason == replay_non_reason,
            "evidence_metrics_unchanged": evidence_metrics_unchanged,
            "state_fingerprint_before": state_fingerprint_before,
            "state_fingerprint_after": state_fingerprint_after,
            "state_fingerprint_unchanged": (
                state_fingerprint_before == state_fingerprint_after
            ),
            "source_row_sha256_before": source_row_sha_before,
            "source_row_sha256_after": source_row_sha_after,
            "source_row_unchanged": source_row_sha_before == source_row_sha_after,
            "historical_raw_output_sha256_before": raw_output_sha_before,
            "historical_raw_output_sha256_after": raw_output_sha_after,
            "historical_raw_output_unchanged": (
                raw_output_sha_before is not None
                and raw_output_sha_before == raw_output_sha_after
            ),
            "tool_trace_projection_sha256_before": _json_sha256(
                trace_projection_before
            ),
            "tool_trace_projection_sha256_after": _json_sha256(
                trace_projection_after
            ),
            "tool_trace_projection_unchanged": (
                trace_projection_before == trace_projection_after
            ),
            "full_tool_trace_status": "unverifiable_not_present_in_source_artifact",
            "tool_execution_performed": False,
        },
        "historical_raw_output_sha256": raw_output_sha_before,
        "historical_row_sha256": source_row_sha_before,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    assembly_errors = Counter(
        str(row["assembly_error"])
        for row in rows
        if row.get("assembly_error") is not None
    )
    summary = {
        "rollouts": total,
        "raw_model_legacy_success_rate": _rate(
            sum(bool(row["raw_model_legacy_success"]) for row in rows), total
        ),
        "raw_model_contract_success_rate": _rate(
            sum(bool(row["raw_model_contract_success"]) for row in rows), total
        ),
        "assembled_system_success_rate": _rate(
            sum(bool(row["assembled_system_success"]) for row in rows), total
        ),
        "promotion_composite_success_rate": _rate(
            sum(bool(row["promotion_composite_success"]) for row in rows), total
        ),
        "connector_recovered_count": sum(
            not bool(row["raw_model_legacy_success"])
            and bool(row["raw_model_contract_success"])
            and bool(row["assembled_system_success"])
            and bool(row["invariants"]["action_unchanged"])
            and bool(row["invariants"]["non_reason_arguments_unchanged"])
            and bool(row["invariants"]["evidence_metrics_unchanged"])
            for row in rows
        ),
        "assembly_success_rate": _rate(
            sum(bool(row["assembly_succeeded"]) for row in rows), total
        ),
        "action_unchanged_rate": _rate(
            sum(bool(row["invariants"]["action_unchanged"]) for row in rows), total
        ),
        "non_reason_arguments_unchanged_rate": _rate(
            sum(
                bool(row["invariants"]["non_reason_arguments_unchanged"])
                for row in rows
            ),
            total,
        ),
        "evidence_metrics_unchanged_rate": _rate(
            sum(
                bool(row["invariants"]["evidence_metrics_unchanged"])
                for row in rows
            ),
            total,
        ),
        "source_row_unchanged_rate": _rate(
            sum(bool(row["invariants"]["source_row_unchanged"]) for row in rows),
            total,
        ),
        "historical_raw_output_verifiable_rate": _rate(
            sum(
                row["invariants"]["historical_raw_output_sha256_before"] is not None
                for row in rows
            ),
            total,
        ),
        "historical_raw_output_unchanged_rate": _rate(
            sum(
                bool(row["invariants"]["historical_raw_output_unchanged"])
                for row in rows
            ),
            total,
        ),
        "tool_trace_projection_unchanged_rate": _rate(
            sum(
                bool(row["invariants"]["tool_trace_projection_unchanged"])
                for row in rows
            ),
            total,
        ),
        "full_tool_trace_verifiable_rate": 0.0,
        "tool_execution_count": sum(
            bool(row["invariants"]["tool_execution_performed"]) for row in rows
        ),
        "assembly_errors": dict(sorted(assembly_errors.items())),
    }
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_action[str(row.get("target_action") or "<none>")].append(row)
    summary["targets"] = {
        action: _summarize(action_rows)
        for action, action_rows in sorted(by_action.items())
    } if len(by_action) > 1 else {}
    return summary


def replay(
    *,
    corpus_path: Path,
    rollouts_path: Path,
    output_dir: Path,
    split: str,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    source_sha_before = _sha256(rollouts_path)
    source_report_path = rollouts_path.with_name("report.json")
    if not source_report_path.is_file():
        raise FileNotFoundError(
            "formal historical replay requires the co-located source report: "
            f"{source_report_path}"
        )
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    if not isinstance(source_report, dict):
        raise ValueError(f"historical source report is not an object: {source_report_path}")
    corpus_sha = _sha256(corpus_path)
    expected_corpus_sha = (source_report.get("corpus_sha256") or {}).get(split)
    if expected_corpus_sha != corpus_sha:
        raise ValueError(
            "historical report and replay corpus SHA-256 do not match: "
            f"{expected_corpus_sha!r} != {corpus_sha!r}"
        )
    if source_report.get("evaluated_splits") != [split]:
        raise ValueError("historical source report split contract does not match replay")
    if source_report.get("seed_protocol") != "sha256-task-sample-v1":
        raise ValueError("unsupported historical rollout seed protocol")
    samples_per_task = int(source_report.get("samples_per_task") or 0)
    if samples_per_task < 1:
        raise ValueError("historical source report has no valid samples-per-task contract")

    corpus_rows = load_grpo_corpus(corpus_path)
    contracts = {
        row.task.task_id: verifier_repair_metadata(row) for row in corpus_rows
    }
    if len(contracts) != len(corpus_rows):
        raise ValueError("corpus task ids are not unique")
    historical_rows = _load_jsonl(rollouts_path)
    expected_rollouts = len(contracts) * samples_per_task
    report_rollouts = int(
        ((source_report.get("splits") or {}).get(split) or {}).get("rollouts") or 0
    )
    if len(historical_rows) != expected_rollouts or report_rollouts != expected_rollouts:
        raise ValueError(
            "historical rollout cardinality does not match corpus and source report: "
            f"rows={len(historical_rows)}, report={report_rollouts}, "
            f"expected={expected_rollouts}"
        )
    pair_keys = [
        (
            str(row.get("task_id") or ""),
            int(row.get("sample_index") or 0),
            int(row.get("rollout_seed") or 0),
        )
        for row in historical_rows
    ]
    if len(set(pair_keys)) != len(pair_keys):
        raise ValueError("historical rollout pair keys are not unique")
    missing = sorted(
        {
            str(row.get("task_id") or "")
            for row in historical_rows
            if str(row.get("task_id") or "") not in contracts
        }
    )
    if missing:
        raise ValueError(f"historical rows are missing from replay corpus: {missing[:5]}")
    actual_task_ids = {str(row.get("task_id") or "") for row in historical_rows}
    if actual_task_ids != set(contracts):
        missing_tasks = sorted(set(contracts) - actual_task_ids)
        extra_tasks = sorted(actual_task_ids - set(contracts))
        raise ValueError(
            "historical rollout task set is incomplete: "
            f"missing={missing_tasks[:5]}, extra={extra_tasks[:5]}"
        )
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in historical_rows:
        task_id = str(row.get("task_id") or "")
        rows_by_task[task_id].append(row)
        contract = contracts[task_id]
        sample_index = int(row.get("sample_index") or 0)
        expected_seed = paired_rollout_seed(
            int(source_report["seed"]),
            task_id=task_id,
            sample_index=sample_index,
        )
        if row.get("split") != split:
            raise ValueError(f"historical rollout split mismatch for {task_id}")
        if str(row.get("target_action") or "") != str(
            contract.get("target_action") or ""
        ):
            raise ValueError(f"historical target action mismatch for {task_id}")
        if str(row.get("source_task_id") or "") != str(
            contract.get("source_task_id") or ""
        ):
            raise ValueError(f"historical source-state mismatch for {task_id}")
        if int(row.get("rollout_seed") or 0) != expected_seed:
            raise ValueError(f"historical paired seed mismatch for {task_id}")
    expected_sample_indices = set(range(samples_per_task))
    malformed_samples = sorted(
        task_id
        for task_id, task_rows in rows_by_task.items()
        if {int(row.get("sample_index") or 0) for row in task_rows}
        != expected_sample_indices
    )
    if malformed_samples:
        raise ValueError(
            "historical per-task sample indices are incomplete: "
            f"{malformed_samples[:5]}"
        )
    replay_rows = [
        replay_row(row, contracts[str(row.get("task_id") or "")])
        for row in historical_rows
    ]
    source_sha_after = _sha256(rollouts_path)
    if source_sha_after != source_sha_before:
        raise RuntimeError("historical rollout artifact changed during read-only replay")

    output_dir.mkdir(parents=True, exist_ok=True)
    replay_path = output_dir / "replay.jsonl"
    replay_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in replay_rows
        ),
        encoding="utf-8",
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "historical train-internal output replay only",
        "historical_only": True,
        "model_inference_performed": False,
        "tool_execution_performed": False,
        "production_promotion_eligible": False,
        "reason_assembly_version": REPAIR_REASON_ASSEMBLY_VERSION,
        "split": split,
        "corpus_path": str(corpus_path.resolve()),
        "corpus_sha256": corpus_sha,
        "source_rollouts_path": str(rollouts_path.resolve()),
        "source_rollouts_sha256_before": source_sha_before,
        "source_rollouts_sha256_after": source_sha_after,
        "source_report_path": str(source_report_path.resolve()),
        "source_report_sha256": _sha256(source_report_path),
        "source_validation": {
            "passed": True,
            "unique_pair_keys": len(pair_keys),
            "expected_rollouts": expected_rollouts,
            "tasks": len(rows_by_task),
            "samples_per_task": samples_per_task,
            "paired_seed_exact": True,
            "task_set_exact": True,
            "target_and_source_state_exact": True,
            "source_report_bound_to_rollouts_sha256": False,
            "source_report_binding_note": (
                "legacy v4 report did not store the rollouts SHA-256; this replay records "
                "the input SHA before and after instead"
            ),
        },
        "summary": _summarize(replay_rows),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="internal_dev")
    args = parser.parse_args()
    report = replay(
        corpus_path=args.corpus,
        rollouts_path=args.rollouts,
        output_dir=args.output_dir,
        split=args.split,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
