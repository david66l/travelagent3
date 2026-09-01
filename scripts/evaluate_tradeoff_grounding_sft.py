"""Evaluate SFT checkpoints on train-internal verifier decision states only."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.grpo_training import load_grpo_corpus  # noqa: E402
from agentic.chat_template_contract import (  # noqa: E402
    AGENT_CHAT_TEMPLATE_KWARGS,
    AGENT_CHAT_TEMPLATE_SHA256,
    AGENT_RENDER_PROTOCOL_VERSION,
)
from agentic.local_policy import LocalCheckpointAgentPolicy  # noqa: E402
from agentic.policy_actions import policy_action_schemas_for_state  # noqa: E402
from agentic.trl_environment import build_trl_environment_factories  # noqa: E402
from scripts.audit_model_curriculum import (  # noqa: E402
    paired_rollout_seed,
    rollout_action_rows,
    rollout_trl_history,
    transport_trl_environment_rows,
    verifier_repair_metadata,
)


SCHEMA_VERSION = "tradeoff-grounding-internal-eval.v4"
EXPECTED_TARGET_ACTIONS = {"abort", "propose_tradeoff", "retry_solve"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_adapter_sha256(checkpoint: str | Path) -> str | None:
    """Return adapter provenance when evaluating an adapter, or ``None`` for base models.

    ``LocalCheckpointAgentPolicy`` intentionally supports both base models and
    PEFT adapters.  Evaluation reports must preserve that distinction instead
    of failing after all rollouts merely because a base checkpoint has no
    ``adapter_model.safetensors`` file.
    """

    adapter = Path(checkpoint) / "adapter_model.safetensors"
    return _sha256(adapter) if adapter.is_file() else None


def _json_sha256(payload: Any) -> str:
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in ("torch", "transformers", "peft", "bitsandbytes"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _shadow_claim_preflight(
    args: argparse.Namespace,
    *,
    corpus_paths: dict[str, Path],
    route_schema_hashes: dict[str, str],
) -> tuple[Path, dict[str, Any]] | None:
    if "train_shadow" not in args.splits:
        if args.candidate_lock is not None:
            raise ValueError("candidate lock is only valid for train-shadow evaluation")
        return None
    if list(args.splits) != ["train_shadow"]:
        raise ValueError("train-shadow must run alone after candidate lock")
    if args.candidate_lock is None:
        raise ValueError("train-shadow evaluation requires a candidate lock")
    lock_path = args.candidate_lock.resolve()
    lock = _load_json_object(lock_path)
    if lock.get("schema_version") != "tradeoff-grounding-candidate-lock.v1":
        raise ValueError("unsupported candidate lock schema")
    if lock.get("status") != "locked_internal_dev_only":
        raise ValueError("candidate lock is not eligible for shadow evaluation")
    selection_path = lock_path.parent / "selection_report.json"
    if _sha256(selection_path) != lock.get("selection_report_sha256"):
        raise ValueError("selection report no longer matches candidate lock")
    selection = _load_json_object(selection_path)
    selected_label = str(lock.get("selected_label") or "")
    selected_evidence = selection.get("evidence", {}).get(selected_label, {})
    internal_report = Path(str(selected_evidence.get("report_path") or ""))
    if _sha256(internal_report) != lock.get("internal_eval_report_sha256"):
        raise ValueError("locked internal evaluation report changed")
    checkpoint = Path(args.checkpoint).resolve()
    if checkpoint != Path(str(lock.get("checkpoint") or "")).resolve():
        raise ValueError("requested checkpoint does not match candidate lock")
    adapter_sha = _sha256(checkpoint / "adapter_model.safetensors")
    if adapter_sha != lock.get("checkpoint_adapter_sha256"):
        raise ValueError("candidate adapter SHA does not match lock")
    shadow_sha = _sha256(corpus_paths["train_shadow"])
    if shadow_sha != lock.get("authorized_shadow_corpus_sha256"):
        raise ValueError("train-shadow corpus SHA does not match lock")
    contract = lock.get("evaluation_contract") or {}
    requested_contract = {
        "seed": args.seed,
        "seed_protocol": "sha256-task-sample-v1",
        "samples_per_task": args.samples_per_task,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "load_in_4bit": args.load_in_4bit,
        "render_protocol": AGENT_RENDER_PROTOCOL_VERSION,
        "chat_template_sha256": AGENT_CHAT_TEMPLATE_SHA256,
        "chat_template_kwargs": dict(AGENT_CHAT_TEMPLATE_KWARGS),
        "route_schema_sha256": dict(sorted(route_schema_hashes.items())),
        "runtime_versions": _runtime_versions(),
    }
    for key, value in requested_contract.items():
        if contract.get(key) != value:
            raise ValueError(f"shadow evaluation contract mismatch: {key}")
    claim_path = Path(str(lock.get("shadow_claim_file") or "")).resolve()
    if claim_path.parent != lock_path.parent:
        raise ValueError("shadow claim must stay beside the candidate lock")
    return claim_path, {
        "schema_version": "tradeoff-grounding-shadow-claim.v1",
        "status": "started",
        "candidate_lock_path": str(lock_path),
        "candidate_lock_sha256": _sha256(lock_path),
        "checkpoint_adapter_sha256": adapter_sha,
        "shadow_corpus_sha256": shadow_sha,
        "output_dir": str(args.output_dir.resolve()),
    }


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _validate_corpus(split: str, rows: list[Any]) -> dict[str, Any]:
    task_ids = [row.task.task_id for row in rows]
    contracts = [verifier_repair_metadata(row) for row in rows]
    source_ids = [str(contract.get("source_task_id") or "") for contract in contracts]
    target_counts = Counter(str(contract.get("target_action") or "") for contract in contracts)
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"{split} task IDs must be unique")
    if not all(source_ids) or len(set(source_ids)) != 10:
        raise ValueError(f"{split} must contain exactly 10 non-empty source states")
    if set(target_counts) != EXPECTED_TARGET_ACTIONS:
        raise ValueError(f"{split} must contain all three verifier-repair targets")
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for source_id, contract in zip(source_ids, contracts, strict=True):
        by_source[source_id][str(contract.get("target_action") or "")] += 1
    per_source_counts = [dict(sorted(counts.items())) for counts in by_source.values()]
    expected_per_source = per_source_counts[0]
    if set(expected_per_source) != EXPECTED_TARGET_ACTIONS:
        raise ValueError(f"{split} source-state target composition is incomplete")
    if any(counts != expected_per_source for counts in per_source_counts):
        raise ValueError(f"{split} source-state target composition is invalid")
    expected_total = Counter(
        {
            target: count * len(by_source)
            for target, count in expected_per_source.items()
        }
    )
    if target_counts != expected_total or len(rows) != sum(expected_total.values()):
        raise ValueError(f"{split} target counts do not match its source-balanced contract")
    return {
        "rows": len(rows),
        "unique_task_ids": len(set(task_ids)),
        "unique_source_task_ids": len(set(source_ids)),
        "target_counts": dict(sorted(target_counts.items())),
        "targets_per_source": expected_per_source,
        "passed": True,
    }


def _system_full_success(row: dict[str, Any]) -> bool:
    """Serving safety after authorization, deliberately separate from credit."""
    return bool(row["checks"].get("system_step_valid"))


def _full_success(row: dict[str, Any]) -> bool:
    """Count model success only when controller hydration did not mask a contract error."""
    return (
        bool(row["checks"].get("decision_step_valid"))
        and float(row["reward"]) == 1.0
        and bool(row.get("model_contract_compliant"))
        and not bool(row.get("controller_override_attempt"))
        and (
            row.get("target_action") != "propose_tradeoff"
            or row.get("controller_hydration_exact") is True
        )
    )


def _load_selected_corpora(
    args: argparse.Namespace,
) -> tuple[dict[str, Path], dict[str, list[Any]]]:
    configured_paths = {
        "internal_dev": args.internal_dev_corpus,
        "train_shadow": args.train_shadow_corpus,
    }
    missing_paths = [split for split in args.splits if configured_paths[split] is None]
    if missing_paths:
        raise ValueError(f"missing corpus paths for selected splits: {missing_paths}")
    selected_paths = {split: configured_paths[split] for split in args.splits}
    return selected_paths, {
        split: load_grpo_corpus(path) for split, path in selected_paths.items()
    }


def _route_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    tradeoff = [row for row in rows if row["target_action"] == "propose_tradeoff"]
    return {
        "rows": total,
        "full_success_count": sum(_full_success(row) for row in rows),
        "full_success_rate": _rate(sum(_full_success(row) for row in rows), total),
        "system_full_success_count": sum(_system_full_success(row) for row in rows),
        "system_full_success_rate": _rate(
            sum(_system_full_success(row) for row in rows), total
        ),
        "model_contract_compliance_rate": _rate(
            sum(bool(row.get("model_contract_compliant")) for row in rows), total
        ),
        "controller_override_attempt_rate": _rate(
            sum(bool(row.get("controller_override_attempt")) for row in rows), total
        ),
        "controller_hydration_exact_rate": (
            _rate(
                sum(row.get("controller_hydration_exact") is True for row in tradeoff),
                len(tradeoff),
            )
            if tradeoff
            else None
        ),
        "parse_schema_cardinality_valid_rate": _rate(
            sum(
                bool(row["checks"].get("single_policy_call"))
                and bool(row["checks"].get("no_policy_call_rejections"))
                for row in rows
            ),
            total,
        ),
        "action_accuracy": _rate(
            sum(bool(row["checks"].get("action_match")) for row in rows), total
        ),
        "reason_grounding_rate": _rate(
            sum(bool(row["checks"].get("grounding_match")) for row in rows), total
        ),
        "reason_specificity_rate": _rate(
            sum(bool(row["checks"].get("reason_specificity_match")) for row in rows),
            total,
        ),
        "reason_language_rate": _rate(
            sum(bool(row["checks"].get("reason_language_match")) for row in rows),
            total,
        ),
        "reason_public_language_rate": _rate(
            sum(bool(row["checks"].get("reason_public_language")) for row in rows),
            total,
        ),
        "reason_action_rationale_rate": _rate(
            sum(
                bool(row["checks"].get("reason_action_rationale_match"))
                for row in rows
            ),
            total,
        ),
        "reason_quality_rate": _rate(
            sum(
                all(
                    bool(row["checks"].get(metric))
                    for metric in (
                        "grounding_match",
                        "reason_specificity_match",
                        "reason_language_match",
                        "reason_public_language",
                        "reason_action_rationale_match",
                    )
                )
                for row in rows
            ),
            total,
        ),
        "tradeoff_options_supported_rate": (
            _rate(
                sum(bool(row["checks"].get("options_supported")) for row in tradeoff),
                len(tradeoff),
            )
            if tradeoff
            else None
        ),
        "tradeoff_required_option_coverage_rate": (
            _rate(
                sum(
                    bool(row["checks"].get("option_contract_covered"))
                    for row in tradeoff
                ),
                len(tradeoff),
            )
            if tradeoff
            else None
        ),
        "reward_minus_one_rate": _rate(
            sum(float(row["reward"]) <= -1.0 for row in rows), total
        ),
        "policy_output_error_rate": _rate(
            sum(bool(row["policy_errors"]) for row in rows), total
        ),
        "policy_call_rejection_rate": _rate(
            sum(
                not bool(row["checks"].get("no_policy_call_rejections"))
                for row in rows
            ),
            total,
        ),
        "execution_invalid_rate": _rate(
            sum(row["checks"].get("execution_valid") is False for row in rows),
            total,
        ),
        "mean_reward": sum(float(row["reward"]) for row in rows) / total if total else 0.0,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_target[row["target_action"]].append(row)
    tradeoff = by_target.get("propose_tradeoff", [])
    action_match = sum(bool(row["checks"].get("action_match")) for row in rows)
    decision_valid = sum(bool(row["checks"].get("decision_step_valid")) for row in rows)
    structural = sum(
        bool(row["checks"].get("single_policy_call"))
        and bool(row["checks"].get("no_policy_call_rejections"))
        for row in rows
    )
    grounded = sum(bool(row["checks"].get("grounding_match")) for row in rows)
    options_supported = sum(
        bool(row["checks"].get("options_supported")) for row in tradeoff
    )
    options_covered = sum(
        bool(row["checks"].get("option_contract_covered")) for row in tradeoff
    )
    reward_minus_one = sum(float(row["reward"]) <= -1.0 for row in rows)
    tool_failures = sum(bool(row["policy_errors"]) for row in rows)
    target_metrics = {}
    for target, target_rows in sorted(by_target.items()):
        target_metrics[target] = _route_summary(target_rows)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source_task_id"])].append(row)
    source_summaries = {
        source_id: _route_summary(source_rows)
        for source_id, source_rows in sorted(by_source.items())
    }
    source_macro = {
        metric: sum(float(summary[metric]) for summary in source_summaries.values())
        / len(source_summaries)
        for metric in (
            "full_success_rate",
            "action_accuracy",
            "reason_grounding_rate",
        )
    }
    summary = {
        "rollouts": total,
        "full_success_count": sum(_full_success(row) for row in rows),
        "full_success_rate": _rate(sum(_full_success(row) for row in rows), total),
        "system_full_success_count": sum(_system_full_success(row) for row in rows),
        "system_full_success_rate": _rate(
            sum(_system_full_success(row) for row in rows), total
        ),
        "model_contract_compliance_rate": _rate(
            sum(bool(row.get("model_contract_compliant")) for row in rows), total
        ),
        "controller_override_attempt_rate": _rate(
            sum(bool(row.get("controller_override_attempt")) for row in rows), total
        ),
        "controller_hydration_exact_rate": _rate(
            sum(row.get("controller_hydration_exact") is True for row in tradeoff),
            len(tradeoff),
        ),
        "parse_schema_cardinality_valid_rate": _rate(structural, total),
        "policy_output_error_rate": _rate(tool_failures, total),
        "action_accuracy": _rate(action_match, total),
        "reason_grounding_rate": _rate(grounded, total),
        "reason_specificity_rate": _rate(
            sum(bool(row["checks"].get("reason_specificity_match")) for row in rows),
            total,
        ),
        "reason_language_rate": _rate(
            sum(bool(row["checks"].get("reason_language_match")) for row in rows),
            total,
        ),
        "reason_public_language_rate": _rate(
            sum(bool(row["checks"].get("reason_public_language")) for row in rows),
            total,
        ),
        "reason_action_rationale_rate": _rate(
            sum(
                bool(row["checks"].get("reason_action_rationale_match"))
                for row in rows
            ),
            total,
        ),
        "reason_quality_rate": _rate(
            sum(
                all(
                    bool(row["checks"].get(metric))
                    for metric in (
                        "grounding_match",
                        "reason_specificity_match",
                        "reason_language_match",
                        "reason_public_language",
                        "reason_action_rationale_match",
                    )
                )
                for row in rows
            ),
            total,
        ),
        "tradeoff_options_supported_rate": _rate(options_supported, len(tradeoff)),
        "tradeoff_required_option_coverage_rate": _rate(options_covered, len(tradeoff)),
        "decision_step_valid_rate": _rate(decision_valid, total),
        "reward_minus_one_rate": _rate(reward_minus_one, total),
        "mean_reward": sum(float(row["reward"]) for row in rows) / total if total else 0.0,
        "targets": target_metrics,
        "source_macro": source_macro,
        "source_all_five_rows_full_success_rate": _rate(
            sum(
                len(source_rows) == 5 and all(_full_success(row) for row in source_rows)
                for source_rows in by_source.values()
            ),
            len(by_source),
        ),
        "observed_actions": dict(
            sorted(Counter(row.get("observed_action") or "<none>" for row in rows).items())
        ),
    }
    if decision_valid != summary["full_success_count"]:
        raise RuntimeError("model-valid and exact reward=1 success counts diverged")
    return summary


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    corpus_paths, corpora = _load_selected_corpora(args)
    corpus_integrity = {
        split: _validate_corpus(split, rows) for split, rows in corpora.items()
    }
    route_schema_hashes: dict[str, str] = {}
    for rows in corpora.values():
        for row in rows:
            contract = verifier_repair_metadata(row)
            target = str(contract.get("target_action") or "")
            allowed_actions = [
                str(action) for action in contract.get("review_allowed_actions") or []
            ]
            prompt_messages = contract.get("prompt_messages") or []
            if not prompt_messages:
                raise ValueError("verifier-repair contract is missing prompt messages")
            transition = json.loads(str(prompt_messages[-1].get("content") or ""))
            capability = dict((transition.get("policy_state") or {}).get("capability") or {})
            schema_hash = _json_sha256(
                policy_action_schemas_for_state(
                    allowed_actions,
                    capability=capability,
                )
            )
            previous = route_schema_hashes.setdefault(target, schema_hash)
            if previous != schema_hash:
                raise ValueError(f"route schema drift detected for target {target!r}")
    shadow_claim = _shadow_claim_preflight(
        args,
        corpus_paths=corpus_paths,
        route_schema_hashes=route_schema_hashes,
    )
    source_ids = {
        split: {
            str(verifier_repair_metadata(row).get("source_task_id") or "")
            for row in rows
        }
        for split, rows in corpora.items()
    }
    if len(source_ids) > 1 and (
        source_ids["internal_dev"] & source_ids["train_shadow"]
    ):
        raise ValueError("internal dev and train shadow source states overlap")
    transported = {
        split: transport_trl_environment_rows(rows)
        for split, rows in corpora.items()
    }
    policy = LocalCheckpointAgentPolicy(
        args.checkpoint,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        do_sample=True,
        temperature=args.temperature,
        load_in_4bit=args.load_in_4bit,
        structured_decoding="native",
    )
    if shadow_claim is not None:
        claim_path, claim_payload = shadow_claim
        with claim_path.open("x", encoding="utf-8") as handle:
            json.dump(claim_payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    factories = build_trl_environment_factories("react")
    rollout_rows: list[dict[str, Any]] = []
    try:
        for split, rows in corpora.items():
            transported_rows, transport_evidence = transported[split]
            for task_index, (row, transported_row) in enumerate(
                zip(rows, transported_rows, strict=True), start=1
            ):
                contract = verifier_repair_metadata(row)
                for sample_index in range(args.samples_per_task):
                    seed = paired_rollout_seed(
                        args.seed,
                        task_id=row.task.task_id,
                        sample_index=sample_index,
                    )
                    policy.set_rollout_seed(seed)
                    policy_errors: list[dict[str, Any]] = []
                    inference_metrics: list[dict[str, Any]] = []
                    rollout = await rollout_trl_history(
                        row,
                        policy,
                        execution_mode="react",
                        environment_factories=factories,
                        max_tool_calling_iterations=1,
                        policy_errors=policy_errors,
                        policy_inference_metrics=inference_metrics,
                        transported_row=transported_row,
                    )
                    actions = rollout_action_rows(
                        rollout,
                        policy_inference_metrics=inference_metrics,
                    )
                    metrics = rollout.reward.audit_metrics
                    checks = {
                        key.removeprefix("decision_"): value
                        for key, value in metrics.items()
                        if key.startswith("decision_")
                    }
                    checks["decision_step_valid"] = metrics.get(
                        "decision_step_valid", False
                    )
                    rollout_rows.append(
                        {
                            "split": split,
                            "task_id": row.task.task_id,
                            "source_task_id": contract.get("source_task_id"),
                            "target_action": contract.get("target_action"),
                            "sample_index": sample_index,
                            "rollout_seed": seed,
                            "reward": rollout.reward.episode_reward,
                            "gate_status": rollout.reward.gate_status,
                            "observed_action": actions[-1]["action"] if actions else None,
                            "observed_arguments": (
                                actions[-1]["arguments"] if actions else None
                            ),
                            "model_raw_arguments": (
                                actions[-1]["model_raw_arguments"] if actions else None
                            ),
                            "controller_override_attempt": (
                                bool(actions[-1]["controller_override_attempt"])
                                if actions
                                else bool(
                                    (policy.last_generation_audit or {}).get(
                                        "controller_override_attempt"
                                    )
                                )
                            ),
                            "model_contract_compliant": (
                                bool(actions[-1]["model_contract_compliant"])
                                if actions
                                else False
                            ),
                            "controller_hydration_exact": (
                                actions[-1]["controller_hydration_exact"]
                                if actions
                                else None
                            ),
                            "controller_authorized_options": (
                                list(contract.get("supervised_options") or [])
                                if contract.get("target_action") == "propose_tradeoff"
                                else None
                            ),
                            "checks": checks,
                            "policy_errors": policy_errors,
                            "generation_audit": policy.last_generation_audit,
                            "inference_metrics": inference_metrics,
                            "initial_state_fingerprint": rollout.initial_state_fingerprint,
                            "authority_transport": transport_evidence["schema_version"],
                        }
                    )
                print(
                    f"[{split} {task_index}/{len(rows)}] {row.task.task_id}",
                    flush=True,
                )
    finally:
        policy.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = args.output_dir / "rollouts.jsonl"
    rollouts_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rollout_rows
        ),
        encoding="utf-8",
    )
    split_reports = {
        split: _summarize([row for row in rollout_rows if row["split"] == split])
        for split in corpora
    }
    checkpoint_adapter_sha256 = _checkpoint_adapter_sha256(args.checkpoint)
    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": "selected train-internal splits only",
        "formal_validation_used": False,
        "checkpoint": args.checkpoint,
        "checkpoint_adapter_sha256": checkpoint_adapter_sha256,
        "checkpoint_kind": (
            "peft-adapter"
            if checkpoint_adapter_sha256 is not None
            else "base-model"
        ),
        "seed": args.seed,
        "seed_protocol": "sha256-task-sample-v1",
        "samples_per_task": args.samples_per_task,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "load_in_4bit": args.load_in_4bit,
        "render_protocol": policy.render_protocol_version,
        "chat_template_sha256": policy.chat_template_sha256,
        "chat_template_kwargs": policy.chat_template_kwargs,
        "route_schema_sha256": dict(sorted(route_schema_hashes.items())),
        "runtime_versions": _runtime_versions(),
        "evaluated_splits": list(corpora),
        "source_overlap_zero": (
            not bool(source_ids["internal_dev"] & source_ids["train_shadow"])
            if len(source_ids) > 1
            else None
        ),
        "corpus_integrity": corpus_integrity,
        "corpus_sha256": {
            split: _sha256(corpus_paths[split])
            for split in corpora
        },
        "splits": split_reports,
        "production_promotion_eligible": False,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if shadow_claim is not None:
        claim_path, claim_payload = shadow_claim
        claim_payload.update(
            {
                "status": "completed",
                "report_path": str(report_path.resolve()),
                "report_sha256": _sha256(report_path),
            }
        )
        claim_path.write_text(
            json.dumps(claim_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--internal-dev-corpus", type=Path)
    parser.add_argument("--train-shadow-corpus", type=Path)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("internal_dev", "train_shadow"),
        default=("internal_dev",),
        help="Evaluate only the named train-internal splits, in the given order.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--candidate-lock", type=Path)
    args = parser.parse_args()
    if args.samples_per_task < 1:
        parser.error("samples-per-task must be positive")
    report = asyncio.run(evaluate(args))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
