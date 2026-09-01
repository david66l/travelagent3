"""Fail-closed selection and locking for tradeoff-grounding SFT checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.chat_template_contract import (  # noqa: E402
    AGENT_CHAT_TEMPLATE_KWARGS,
    AGENT_CHAT_TEMPLATE_SHA256,
    AGENT_RENDER_PROTOCOL_VERSION,
)
from agentic.local_policy import parse_local_tool_call  # noqa: E402
from scripts.audit_model_curriculum import paired_rollout_seed  # noqa: E402
from scripts.evaluate_tradeoff_grounding_sft import _summarize  # noqa: E402


SCHEMA_VERSION = "tradeoff-grounding-candidate-selection.v2"
EVAL_SCHEMA_VERSION = "tradeoff-grounding-internal-eval.v3"
EXPECTED_STEPS = {"epoch1": (12, 1.0), "epoch2": (24, 2.0)}
EXPECTED_EVAL_CONTRACT = {
    "seed": 20260910,
    "seed_protocol": "sha256-task-sample-v1",
    "samples_per_task": 1,
    "temperature": 0.8,
    "max_new_tokens": 192,
    "load_in_4bit": True,
    "render_protocol": AGENT_RENDER_PROTOCOL_VERSION,
    "chat_template_sha256": AGENT_CHAT_TEMPLATE_SHA256,
    "chat_template_kwargs": dict(AGENT_CHAT_TEMPLATE_KWARGS),
    "route_schema_sha256": {
        "abort": "3ec5b45a20908e539385a228a48f86a7c305c77654cb051971c9b25730b494cb",
        "propose_tradeoff": (
            "17e15e25c38cc52df301262e00bfc81376eb527d728e8d1202bad2597bcaae09"
        ),
        "retry_solve": "ae9df6a2ba6057956b16388acd821647e7703acb7f8ec5ea7efc3d219795a19b",
    },
    "runtime_versions": {
        "python": "3.12.3",
        "platform": "Linux-5.15.0-94-generic-x86_64-with-glibc2.35",
        "torch": "2.6.0",
        "transformers": "5.12.1",
        "peft": "0.20.0",
        "bitsandbytes": "0.50.0",
    },
    "corpus_sha256": {
        "internal_dev": "bff08ede57c85009c17fa752263d8ea7e6388a519aa18be3897bb54b535d73a1"
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _load_and_validate_rollouts(
    label: str,
    report_path: Path,
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    rollouts_path = report_path.parent / "rollouts.jsonl"
    rows = [
        json.loads(line)
        for line in rollouts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 50 or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} must retain exactly 50 raw rollout objects")
    task_ids = [str(row.get("task_id") or "") for row in rows]
    source_ids = [str(row.get("source_task_id") or "") for row in rows]
    if not all(task_ids) or len(set(task_ids)) != 50:
        raise ValueError(f"{label} raw rollout task IDs are invalid")
    if not all(source_ids) or len(set(source_ids)) != 10:
        raise ValueError(f"{label} raw rollout source IDs are invalid")
    if any(row.get("split") != "internal_dev" for row in rows):
        raise ValueError(f"{label} raw rollouts include a non-development split")
    if Counter(str(row.get("target_action") or "") for row in rows) != Counter(
        {"abort": 10, "propose_tradeoff": 30, "retry_solve": 10}
    ):
        raise ValueError(f"{label} raw rollout target counts are invalid")
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_source[str(row["source_task_id"])][str(row["target_action"])] += 1
    expected_per_source = Counter(
        {"abort": 1, "propose_tradeoff": 3, "retry_solve": 1}
    )
    if any(counts != expected_per_source for counts in by_source.values()):
        raise ValueError(f"{label} raw source-state composition is invalid")
    for row in rows:
        if row.get("sample_index") != 0:
            raise ValueError(f"{label} used an unregistered sample index")
        expected_seed = paired_rollout_seed(
            int(report["seed"]),
            task_id=str(row["task_id"]),
            sample_index=0,
        )
        if int(row.get("rollout_seed", -1)) != expected_seed:
            raise ValueError(f"{label} rollout seed does not match task pairing")
        generation = row.get("generation_audit")
        if not isinstance(generation, dict):
            raise ValueError(f"{label} is missing generation audit evidence")
        raw_output = generation.get("raw_output")
        if not isinstance(raw_output, str) or hashlib.sha256(
            raw_output.encode("utf-8")
        ).hexdigest() != generation.get("raw_output_sha256"):
            raise ValueError(f"{label} raw completion hash is invalid")
        if generation.get("parser_contract") != (
            "exact-single-tool-call-envelope-or-json.v1"
        ):
            raise ValueError(f"{label} parser contract is not pinned")
        if not row.get("policy_errors"):
            action, arguments = parse_local_tool_call(raw_output)
            model_raw_arguments = row.get("model_raw_arguments")
            if (
                not isinstance(model_raw_arguments, dict)
                or action != row.get("observed_action")
                or arguments != model_raw_arguments
                or generation.get("parsed_action") != action
                or generation.get("model_raw_arguments") != model_raw_arguments
            ):
                raise ValueError(
                    f"{label} parsed completion and model-authored action differ"
                )
            # The executed action is intentionally different for v4 tradeoff:
            # the controller, not the model, injects its exact authorized list.
            # Verify that relationship independently so the audit cannot hide
            # either a parser mismatch or an authorization mismatch.
            if action == "propose_tradeoff":
                authorized_options = row.get("controller_authorized_options")
                observed_arguments = row.get("observed_arguments")
                if (
                    not isinstance(authorized_options, list)
                    or not all(
                        isinstance(item, str) and item.strip()
                        for item in authorized_options
                    )
                    or not isinstance(observed_arguments, dict)
                    or observed_arguments
                    != {
                        **{key: value for key, value in model_raw_arguments.items() if key != "options"},
                        "options": authorized_options,
                    }
                ):
                    raise ValueError(
                        f"{label} controller-authorized tradeoff payload differs from evidence"
                    )
        for inference in row.get("inference_metrics") or []:
            if inference.get("model") != report.get("checkpoint"):
                raise ValueError(f"{label} inference checkpoint evidence differs")
    recomputed = _summarize(rows)
    if recomputed != report.get("splits", {}).get("internal_dev"):
        raise ValueError(f"{label} report summary does not match raw rollouts")
    return rows


def _integer_metric(rate: float, rows: int) -> int:
    count = round(float(rate) * rows)
    if abs(float(rate) - count / rows) > 1e-9:
        raise ValueError(f"rate {rate} is not an exact count over {rows} rows")
    return count


def _check(name: str, passed: bool, observed: Any, threshold: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "threshold": threshold,
    }


def candidate_gates(
    report: dict[str, Any],
    *,
    baseline: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    split = report["splits"]["internal_dev"]
    targets = split["targets"]
    tradeoff = targets["propose_tradeoff"]
    checks = [
        _check(
            "overall_structural",
            _integer_metric(split["parse_schema_cardinality_valid_rate"], 50) == 50,
            split["parse_schema_cardinality_valid_rate"],
            "50/50",
        ),
        _check(
            "overall_policy_output_errors",
            _integer_metric(split["policy_output_error_rate"], 50) == 0,
            split["policy_output_error_rate"],
            "0/50",
        ),
        _check(
            "overall_action",
            _integer_metric(split["action_accuracy"], 50) >= 45,
            split["action_accuracy"],
            ">=45/50",
        ),
        _check(
            "overall_reason_grounding",
            _integer_metric(split["reason_grounding_rate"], 50) >= 45,
            split["reason_grounding_rate"],
            ">=45/50",
        ),
        _check(
            "overall_full_success",
            int(split["full_success_count"]) >= 43,
            split["full_success_count"],
            ">=43/50",
        ),
        _check(
            "overall_model_contract_compliance",
            _integer_metric(split["model_contract_compliance_rate"], 50) == 50,
            split["model_contract_compliance_rate"],
            "50/50",
        ),
        _check(
            "overall_controller_override_attempts",
            _integer_metric(split["controller_override_attempt_rate"], 50) == 0,
            split["controller_override_attempt_rate"],
            "0/50",
        ),
        _check(
            "overall_reward_minus_one",
            _integer_metric(split["reward_minus_one_rate"], 50) <= 2,
            split["reward_minus_one_rate"],
            "<=2/50",
        ),
        _check(
            "tradeoff_structural",
            _integer_metric(tradeoff["parse_schema_cardinality_valid_rate"], 30) == 30,
            tradeoff["parse_schema_cardinality_valid_rate"],
            "30/30",
        ),
        _check(
            "tradeoff_policy_output_errors",
            _integer_metric(tradeoff["policy_output_error_rate"], 30) == 0,
            tradeoff["policy_output_error_rate"],
            "0/30",
        ),
        _check(
            "tradeoff_execution_invalid",
            _integer_metric(tradeoff["execution_invalid_rate"], 30) == 0,
            tradeoff["execution_invalid_rate"],
            "0/30",
        ),
        _check(
            "tradeoff_action",
            _integer_metric(tradeoff["action_accuracy"], 30) >= 27,
            tradeoff["action_accuracy"],
            ">=27/30",
        ),
        _check(
            "tradeoff_reason_grounding",
            _integer_metric(tradeoff["reason_grounding_rate"], 30) >= 27,
            tradeoff["reason_grounding_rate"],
            ">=27/30",
        ),
        _check(
            "tradeoff_options_supported",
            _integer_metric(tradeoff["tradeoff_options_supported_rate"], 30) == 30,
            tradeoff["tradeoff_options_supported_rate"],
            "30/30",
        ),
        _check(
            "tradeoff_option_coverage",
            _integer_metric(
                tradeoff["tradeoff_required_option_coverage_rate"], 30
            )
            == 30,
            tradeoff["tradeoff_required_option_coverage_rate"],
            "30/30",
        ),
        _check(
            "tradeoff_model_contract_compliance",
            _integer_metric(tradeoff["model_contract_compliance_rate"], 30) == 30,
            tradeoff["model_contract_compliance_rate"],
            "30/30",
        ),
        _check(
            "tradeoff_controller_override_attempts",
            _integer_metric(tradeoff["controller_override_attempt_rate"], 30) == 0,
            tradeoff["controller_override_attempt_rate"],
            "0/30",
        ),
        _check(
            "tradeoff_controller_hydration_exact",
            _integer_metric(tradeoff["controller_hydration_exact_rate"], 30) == 30,
            tradeoff["controller_hydration_exact_rate"],
            "30/30",
        ),
        _check(
            "tradeoff_full_success",
            int(tradeoff["full_success_count"]) >= 26,
            tradeoff["full_success_count"],
            ">=26/30",
        ),
        _check(
            "tradeoff_reward_minus_one",
            _integer_metric(tradeoff["reward_minus_one_rate"], 30) <= 2,
            tradeoff["reward_minus_one_rate"],
            "<=2/30",
        ),
    ]
    for route in ("abort", "retry_solve"):
        route_metrics = targets[route]
        checks.extend(
            [
                _check(
                    f"{route}_structural",
                    _integer_metric(
                        route_metrics["parse_schema_cardinality_valid_rate"], 10
                    )
                    == 10,
                    route_metrics["parse_schema_cardinality_valid_rate"],
                    "10/10",
                ),
                _check(
                    f"{route}_policy_output_errors",
                    _integer_metric(route_metrics["policy_output_error_rate"], 10)
                    == 0,
                    route_metrics["policy_output_error_rate"],
                    "0/10",
                ),
                _check(
                    f"{route}_policy_call_rejections",
                    _integer_metric(route_metrics["policy_call_rejection_rate"], 10)
                    == 0,
                    route_metrics["policy_call_rejection_rate"],
                    "0/10",
                ),
                _check(
                    f"{route}_execution_invalid",
                    _integer_metric(route_metrics["execution_invalid_rate"], 10) == 0,
                    route_metrics["execution_invalid_rate"],
                    "0/10",
                ),
            ]
        )
        if baseline is not None:
            baseline_route = baseline["splits"]["internal_dev"]["targets"][route]
            checks.extend(
                [
                    _check(
                        f"{route}_full_success_retention",
                        int(route_metrics["full_success_count"])
                        >= int(baseline_route["full_success_count"]),
                        route_metrics["full_success_count"],
                        f">=baseline {baseline_route['full_success_count']}/10",
                    ),
                    _check(
                        f"{route}_action_retention",
                        _integer_metric(route_metrics["action_accuracy"], 10)
                        >= _integer_metric(baseline_route["action_accuracy"], 10),
                        route_metrics["action_accuracy"],
                        f">=baseline {baseline_route['action_accuracy']}",
                    ),
                ]
            )
    overall_execution_invalid = sum(
        _integer_metric(targets[route]["execution_invalid_rate"], rows)
        for route, rows in (("abort", 10), ("propose_tradeoff", 30), ("retry_solve", 10))
    )
    checks.append(
        _check(
            "overall_execution_invalid",
            overall_execution_invalid == 0,
            overall_execution_invalid,
            "0/50",
        )
    )
    return checks


def _validate_eval_report(label: str, path: Path, report: dict[str, Any]) -> None:
    if report.get("schema_version") != EVAL_SCHEMA_VERSION:
        raise ValueError(f"{label} does not use {EVAL_SCHEMA_VERSION}")
    if report.get("evaluated_splits") != ["internal_dev"]:
        raise ValueError(f"{label} is not an internal-dev-only report")
    if report.get("formal_validation_used") is not False:
        raise ValueError(f"{label} touched formal validation")
    if report.get("samples_per_task") != 1:
        raise ValueError(f"{label} must use exactly one preregistered sample per task")
    integrity = report.get("corpus_integrity", {}).get("internal_dev", {})
    if integrity.get("passed") is not True:
        raise ValueError(f"{label} corpus integrity did not pass")
    rollouts_path = path.parent / "rollouts.jsonl"
    if not rollouts_path.is_file() or len(rollouts_path.read_text(encoding="utf-8").splitlines()) != 50:
        raise ValueError(f"{label} must retain exactly 50 raw rollout records")


def _compatibility_key(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "seed",
            "seed_protocol",
            "samples_per_task",
            "temperature",
            "max_new_tokens",
            "load_in_4bit",
            "render_protocol",
            "chat_template_sha256",
            "chat_template_kwargs",
            "route_schema_sha256",
            "runtime_versions",
            "corpus_sha256",
        )
    }


def _checkpoint_state(label: str, checkpoint: Path) -> dict[str, Any] | None:
    if label == "baseline":
        return None
    state_path = checkpoint / "trainer_state.json"
    state = _load_json(state_path)
    expected_step, expected_epoch = EXPECTED_STEPS[label]
    if int(state.get("global_step", -1)) != expected_step:
        raise ValueError(f"{label} global step does not match the preregistered checkpoint")
    if float(state.get("epoch", -1)) != expected_epoch:
        raise ValueError(f"{label} epoch does not match the preregistered checkpoint")
    return {
        "trainer_state_sha256": _sha256(state_path),
        "global_step": expected_step,
        "epoch": expected_epoch,
        "max_steps": int(state.get("max_steps", -1)),
        "num_train_epochs": float(state.get("num_train_epochs", -1)),
    }


def _rollout_full_success(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("checks", {}).get("decision_step_valid"))
        and float(row.get("reward", -1)) == 1.0
        and bool(row.get("model_contract_compliant"))
        and not bool(row.get("controller_override_attempt"))
        and (
            row.get("target_action") != "propose_tradeoff"
            or row.get("controller_hydration_exact") is True
        )
    )


def paired_retention_gates(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline = {str(row["task_id"]): row for row in baseline_rows}
    candidate = {str(row["task_id"]): row for row in candidate_rows}
    if set(baseline) != set(candidate):
        raise ValueError("candidate and baseline raw rollout task sets differ")
    checks = []
    for route in ("abort", "retry_solve"):
        regressed = sorted(
            task_id
            for task_id, baseline_row in baseline.items()
            if baseline_row["target_action"] == route
            and _rollout_full_success(baseline_row)
            and not _rollout_full_success(candidate[task_id])
        )
        checks.append(
            _check(
                f"{route}_taskwise_full_success_retention",
                not regressed,
                regressed,
                "no baseline-success task may regress",
            )
        )
    return checks


def select_candidate(
    reports: dict[str, dict[str, Any]],
) -> tuple[str | None, dict[str, list[dict[str, Any]]]]:
    gates = {
        "baseline": candidate_gates(reports["baseline"], baseline=None),
        "epoch1": candidate_gates(reports["epoch1"], baseline=reports["baseline"]),
        "epoch2": candidate_gates(reports["epoch2"], baseline=reports["baseline"]),
    }
    selected = next(
        (label for label in ("baseline", "epoch1", "epoch2") if all(item["passed"] for item in gates[label])),
        None,
    )
    return selected, gates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--epoch1-report", type=Path, required=True)
    parser.add_argument("--epoch2-report", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--shadow-corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"refusing to overwrite non-empty output: {args.output_dir}")

    report_paths = {
        "baseline": args.baseline_report,
        "epoch1": args.epoch1_report,
        "epoch2": args.epoch2_report,
    }
    reports = {label: _load_json(path) for label, path in report_paths.items()}
    for label, report in reports.items():
        _validate_eval_report(label, report_paths[label], report)
    rollout_rows = {
        label: _load_and_validate_rollouts(label, report_paths[label], report)
        for label, report in reports.items()
    }
    compatibility = _compatibility_key(reports["baseline"])
    if compatibility != EXPECTED_EVAL_CONTRACT:
        raise ValueError("evaluation does not match the preregistered fixed contract")
    for label in ("epoch1", "epoch2"):
        if _compatibility_key(reports[label]) != compatibility:
            raise ValueError(f"{label} evaluation protocol differs from baseline")

    training_report = _load_json(args.training_report)
    baseline_sha = reports["baseline"]["checkpoint_adapter_sha256"]
    if training_report.get("source_adapter_sha256_before") != baseline_sha:
        raise ValueError("training parent adapter does not match baseline")
    if training_report.get("source_adapter_sha256_after") != baseline_sha:
        raise ValueError("training mutated the baseline adapter")
    if training_report.get("chat_template_sha256") != compatibility["chat_template_sha256"]:
        raise ValueError("training and evaluation chat templates differ")
    if int(training_report.get("seed", -1)) != int(compatibility["seed"]):
        raise ValueError("training and evaluation seeds differ")

    training_root = args.training_report.resolve().parent
    expected_checkpoints = {
        "epoch1": (training_root / "checkpoint-12").resolve(),
        "epoch2": (training_root / "checkpoint-24").resolve(),
    }
    if Path(reports["baseline"]["checkpoint"]).resolve() != Path(
        str(training_report.get("base_model") or "")
    ).resolve():
        raise ValueError("baseline checkpoint does not match the training parent")
    for label, expected_checkpoint in expected_checkpoints.items():
        if Path(reports[label]["checkpoint"]).resolve() != expected_checkpoint:
            raise ValueError(f"{label} checkpoint is outside the recorded training run")
    for label, report in reports.items():
        actual_sha = _sha256(
            Path(report["checkpoint"]) / "adapter_model.safetensors"
        )
        if actual_sha != report.get("checkpoint_adapter_sha256"):
            raise ValueError(f"{label} adapter SHA does not match the current checkpoint")

    checkpoint_states = {
        label: _checkpoint_state(label, Path(report["checkpoint"]))
        for label, report in reports.items()
    }
    if reports["epoch2"]["checkpoint_adapter_sha256"] != training_report.get(
        "output_adapter_sha256"
    ):
        raise ValueError("epoch2 adapter does not match the recorded final output")

    epoch1_state = _load_json(expected_checkpoints["epoch1"] / "trainer_state.json")
    epoch2_state = _load_json(expected_checkpoints["epoch2"] / "trainer_state.json")
    epoch1_history = epoch1_state.get("log_history") or []
    epoch2_history = epoch2_state.get("log_history") or []
    if epoch2_history[: len(epoch1_history)] != epoch1_history:
        raise ValueError("epoch1 trainer history is not a prefix of the final run")
    if _sha256(expected_checkpoints["epoch1"] / "adapter_config.json") != _sha256(
        expected_checkpoints["epoch2"] / "adapter_config.json"
    ):
        raise ValueError("epoch checkpoints do not share one adapter configuration")
    if _sha256(expected_checkpoints["epoch1"] / "training_args.bin") != _sha256(
        expected_checkpoints["epoch2"] / "training_args.bin"
    ):
        raise ValueError("epoch checkpoints do not share one training configuration")

    selected, gates = select_candidate(reports)
    for label in ("epoch1", "epoch2"):
        gates[label].extend(
            paired_retention_gates(rollout_rows["baseline"], rollout_rows[label])
        )
    selected = next(
        (
            label
            for label in ("baseline", "epoch1", "epoch2")
            if all(item["passed"] for item in gates[label])
        ),
        None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence = {
        label: {
            "report_path": str(path.resolve()),
            "report_sha256": _sha256(path),
            "rollouts_sha256": _sha256(path.parent / "rollouts.jsonl"),
            "checkpoint": report["checkpoint"],
            "checkpoint_adapter_sha256": report["checkpoint_adapter_sha256"],
            "checkpoint_state": checkpoint_states[label],
        }
        for label, (path, report) in {
            label: (report_paths[label], reports[label]) for label in reports
        }.items()
    }
    selection_report = {
        "schema_version": SCHEMA_VERSION,
        "status": "locked" if selected else "blocked_no_candidate_passed",
        "selection_rule": "baseline_if_all_gates_else_earliest_passing_epoch1_then_epoch2",
        "selected_label": selected,
        "gates": gates,
        "compatibility_contract": compatibility,
        "training_report_path": str(args.training_report.resolve()),
        "training_report_sha256": _sha256(args.training_report),
        "training_contract": training_report.get("training_contract"),
        "evidence": evidence,
        "formal_validation_used": False,
        "production_promotion_eligible": False,
    }
    selection_path = args.output_dir / "selection_report.json"
    selection_path.write_text(
        json.dumps(selection_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if selected is None:
        print(json.dumps(selection_report, ensure_ascii=False, indent=2))
        return 2

    lock = {
        "schema_version": "tradeoff-grounding-candidate-lock.v1",
        "status": "locked_internal_dev_only",
        "selected_label": selected,
        "checkpoint": evidence[selected]["checkpoint"],
        "checkpoint_adapter_sha256": evidence[selected]["checkpoint_adapter_sha256"],
        "parent_adapter_sha256": baseline_sha,
        "selection_report_sha256": _sha256(selection_path),
        "selection_report_path": str(selection_path.resolve()),
        "training_report_sha256": _sha256(args.training_report),
        "training_report_path": str(args.training_report.resolve()),
        "internal_eval_report_sha256": evidence[selected]["report_sha256"],
        "internal_eval_report_path": evidence[selected]["report_path"],
        "internal_rollouts_sha256": evidence[selected]["rollouts_sha256"],
        "internal_rollouts_path": str(
            (report_paths[selected].parent / "rollouts.jsonl").resolve()
        ),
        "checkpoint_state": evidence[selected]["checkpoint_state"],
        "evaluation_contract": compatibility,
        "authorized_shadow_corpus_sha256": _sha256(args.shadow_corpus),
        "shadow_claim_file": str((args.output_dir / "shadow_eval.claim.json").resolve()),
        "formal_validation_used": False,
        "production_promotion_eligible": False,
    }
    (args.output_dir / "candidate_lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selection_report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
