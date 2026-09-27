"""Compare two H-005 evaluations under one strictly identical inference stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.environment import environment_fingerprint  # noqa: E402
from agentic.grpo_training import load_grpo_corpus  # noqa: E402
from scripts.audit_model_curriculum import verifier_repair_metadata  # noqa: E402
from scripts.evaluate_tradeoff_grounding_sft import (  # noqa: E402
    effective_tokenizer_manifest_for_checkpoint,
    evaluation_code_snapshot,
    model_file_manifest,
)


SCHEMA_VERSION = "h005-same-stack-comparison.v1"
EVALUATION_SCHEMA_VERSION = "tradeoff-grounding-internal-eval.v7"
REASON_ASSEMBLY_VERSION = "repair-reason-assembly.v1"
REQUIRED_SAMPLES_PER_TASK = 4
AUTHORITY_TRANSPORT_VERSION = "canonical-grpo-audit-transport.v1"
EXPECTED_INTERNAL_DEV_CORPUS_SHA256 = (
    "fe0a06f604456f0e0d1c1725c7934b8bf95cca3ba1b9d70a30d28f36a069fc33"
)
EXPECTED_TASKS = 80
EXPECTED_SOURCES = 10
EXPECTED_TARGET_COUNTS = {"abort": 10, "propose_tradeoff": 60, "retry_solve": 10}
EXPECTED_TARGETS_PER_SOURCE = {"abort": 1, "propose_tradeoff": 6, "retry_solve": 1}
CONTRACT_KEYS = (
    "schema_version",
    "reason_assembly_version",
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
    "evaluation_code_snapshot",
    "corpus_sha256",
    "evaluated_splits",
    "runtime_versions",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected non-empty JSON-object rows: {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _raw_success(row: dict[str, Any]) -> bool:
    checks = row.get("checks") or {}
    return (
        bool(checks.get("decision_step_valid"))
        and float(row.get("reward") or 0.0) == 1.0
        and bool(row.get("model_contract_compliant"))
        and not bool(row.get("controller_override_attempt"))
        and (
            row.get("target_action") != "propose_tradeoff"
            or row.get("controller_hydration_exact") is True
        )
    )


def _semantic_success(row: dict[str, Any]) -> bool:
    return bool((row.get("checks") or {}).get("model_semantic_valid"))


def _system_success(row: dict[str, Any]) -> bool:
    return bool((row.get("checks") or {}).get("system_step_valid"))


def _pass4_success(row: dict[str, Any]) -> bool:
    return bool(row.get("pass4"))


def _rate(
    rows: list[dict[str, Any]], metric: Callable[[dict[str, Any]], bool]
) -> float:
    return sum(metric(row) for row in rows) / len(rows) if rows else 0.0


def _model_summary(rows: list[dict[str, Any]], samples_per_task: int) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task_id") or "")].append(row)
    malformed = sorted(
        task_id
        for task_id, task_rows in by_task.items()
        if len(task_rows) != samples_per_task
    )
    if malformed:
        raise ValueError(
            f"tasks do not contain exactly {samples_per_task} samples: {malformed[:5]}"
        )
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_target[str(row.get("target_action") or "<none>")].append(row)
    assembly_eligible = [
        row
        for row in rows
        if row.get("observed_action") in {"abort", "propose_tradeoff", "retry_solve"}
        and isinstance(row.get("model_raw_arguments"), dict)
        and isinstance(row.get("observed_arguments"), dict)
    ]
    hydration_rows = [
        row
        for row in rows
        if row.get("target_action") in {"propose_tradeoff", "retry_solve"}
    ]
    inference = [
        metric
        for row in rows
        for metric in row.get("inference_metrics") or []
        if isinstance(metric, dict)
    ]
    return {
        "rollouts": len(rows),
        "tasks": len(by_task),
        "raw_model_legacy_success_rate": _rate(rows, _raw_success),
        "raw_model_contract_success_rate": _rate(rows, _semantic_success),
        "assembled_system_success_rate": _rate(rows, _system_success),
        "strict_connector_recovered_count": sum(
            not _raw_success(row)
            and _semantic_success(row)
            and _system_success(row)
            and bool(row.get("reason_assembly_exact"))
            for row in rows
        ),
        "action_accuracy": sum(
            bool((row.get("checks") or {}).get("action_match")) for row in rows
        )
        / len(rows),
        "model_contract_compliance_rate": sum(
            bool(row.get("model_contract_compliant")) for row in rows
        )
        / len(rows),
        "controller_override_attempt_rate": sum(
            bool(row.get("controller_override_attempt")) for row in rows
        )
        / len(rows),
        "controller_hydration_exact_rate": (
            sum(row.get("controller_hydration_exact") is True for row in hydration_rows)
            / len(hydration_rows)
            if hydration_rows
            else None
        ),
        "reason_assembly_coverage_rate": len(assembly_eligible) / len(rows),
        "reason_assembly_exact_given_eligible_rate": (
            sum(bool(row.get("reason_assembly_exact")) for row in assembly_eligible)
            / len(assembly_eligible)
            if assembly_eligible
            else None
        ),
        "mean_completion_tokens": (
            sum(float(metric.get("completion_tokens") or 0.0) for metric in inference)
            / len(inference)
            if inference
            else None
        ),
        "mean_request_latency_ms": (
            sum(float(metric.get("request_latency_ms") or 0.0) for metric in inference)
            / len(inference)
            if inference
            else None
        ),
        "assembled_system_pass4_rate": sum(
            all(_system_success(row) for row in task_rows)
            for task_rows in by_task.values()
        )
        / len(by_task),
        "raw_model_legacy_pass4_rate": sum(
            all(_raw_success(row) for row in task_rows)
            for task_rows in by_task.values()
        )
        / len(by_task),
        "raw_model_contract_pass4_rate": sum(
            all(_semantic_success(row) for row in task_rows)
            for task_rows in by_task.values()
        )
        / len(by_task),
        "targets": {
            target: {
                "rollouts": len(target_rows),
                "raw_model_legacy_success_rate": _rate(target_rows, _raw_success),
                "raw_model_contract_success_rate": _rate(
                    target_rows, _semantic_success
                ),
                "assembled_system_success_rate": _rate(target_rows, _system_success),
            }
            for target, target_rows in sorted(by_target.items())
        },
    }


def _pair_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("task_id") or ""),
        int(row.get("sample_index") or 0),
        int(row.get("rollout_seed") or 0),
    )


def _expected_seed(base_seed: int, *, task_id: str, sample_index: int) -> int:
    payload = f"{base_seed}:{task_id}:{sample_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def _validate_arm(
    *,
    label: str,
    rows: list[dict[str, Any]],
    report: dict[str, Any],
    contracts: dict[str, dict[str, Any]],
    expected_fingerprints: dict[str, str],
    corpus_sha256: str,
) -> dict[str, Any]:
    if report.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError(f"{label} uses an unsupported evaluation schema")
    if report.get("reason_assembly_version") != REASON_ASSEMBLY_VERSION:
        raise ValueError(f"{label} uses an unsupported reason assembly contract")
    if report.get("evaluated_splits") != ["internal_dev"]:
        raise ValueError(f"{label} is not an internal-dev-only evaluation")
    if report.get("production_promotion_eligible") is not False:
        raise ValueError(f"{label} report has an invalid promotion scope")
    samples_per_task = int(report.get("samples_per_task") or 0)
    if samples_per_task != REQUIRED_SAMPLES_PER_TASK:
        raise ValueError(
            f"{label} must use exactly {REQUIRED_SAMPLES_PER_TASK} samples per task"
        )
    if (report.get("corpus_sha256") or {}).get("internal_dev") != corpus_sha256:
        raise ValueError(
            f"{label} report corpus hash does not match the supplied corpus"
        )
    expected_rows = len(contracts) * samples_per_task
    if len(rows) != expected_rows:
        raise ValueError(f"{label} row count mismatch: {len(rows)} != {expected_rows}")

    task_ids = [str(row.get("task_id") or "") for row in rows]
    if set(task_ids) != set(contracts):
        raise ValueError(f"{label} rollout task set does not match the corpus")
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = str(row.get("task_id") or "")
        source_task_id = row.get("source_task_id")
        target_action = row.get("target_action")
        state_fingerprint = row.get("initial_state_fingerprint")
        authority_transport = row.get("authority_transport")
        if not isinstance(source_task_id, str) or not source_task_id:
            raise ValueError(f"{label} has an empty source_task_id for {task_id}")
        if not isinstance(target_action, str) or not target_action:
            raise ValueError(f"{label} has an empty target_action for {task_id}")
        if not isinstance(state_fingerprint, str) or not state_fingerprint:
            raise ValueError(f"{label} has an empty state fingerprint for {task_id}")
        if authority_transport != AUTHORITY_TRANSPORT_VERSION:
            raise ValueError(f"{label} authority transport mismatch for {task_id}")
        contract = contracts[task_id]
        if source_task_id != str(contract.get("source_task_id") or ""):
            raise ValueError(
                f"{label} source state does not match corpus for {task_id}"
            )
        if target_action != str(contract.get("target_action") or ""):
            raise ValueError(
                f"{label} target action does not match corpus for {task_id}"
            )
        if state_fingerprint != expected_fingerprints[task_id]:
            raise ValueError(
                f"{label} state fingerprint does not match corpus for {task_id}"
            )
        sample_index = int(row.get("sample_index") or 0)
        if int(row.get("rollout_seed") or 0) != _expected_seed(
            int(report["seed"]), task_id=task_id, sample_index=sample_index
        ):
            raise ValueError(f"{label} paired seed mismatch for {task_id}")
        if row.get("split") != "internal_dev":
            raise ValueError(f"{label} rollout split mismatch for {task_id}")
        if row.get("reason_assembly_version") != REASON_ASSEMBLY_VERSION:
            raise ValueError(f"{label} row assembly version mismatch for {task_id}")
        inference_metrics = row.get("inference_metrics") or []
        if len(inference_metrics) != 1 or not isinstance(inference_metrics[0], dict):
            raise ValueError(
                f"{label} inference audit cardinality mismatch for {task_id}"
            )
        inference_metric = inference_metrics[0]
        inference_models = {inference_metric.get("model")}
        if inference_models != {report.get("checkpoint")}:
            raise ValueError(
                f"{label} inference model does not match report for {task_id}"
            )
        for metric_name in ("completion_tokens", "request_latency_ms"):
            metric_value = inference_metric.get(metric_name)
            if (
                not isinstance(metric_value, (int, float))
                or isinstance(metric_value, bool)
                or not math.isfinite(float(metric_value))
                or float(metric_value) < 0.0
            ):
                raise ValueError(
                    f"{label} has an invalid {metric_name} audit for {task_id}"
                )
        generation_audit = row.get("generation_audit") or {}
        prompt_sha256 = generation_audit.get("prompt_input_ids_sha256")
        if (
            not _is_sha256(prompt_sha256)
            or generation_audit.get("inference_metrics") != inference_metric
        ):
            raise ValueError(f"{label} generation audit mismatch for {task_id}")
        by_task[task_id].append(row)

    required_indices = set(range(REQUIRED_SAMPLES_PER_TASK))
    malformed = sorted(
        task_id
        for task_id, task_rows in by_task.items()
        if len(task_rows) != REQUIRED_SAMPLES_PER_TASK
        or {int(row.get("sample_index") or 0) for row in task_rows} != required_indices
    )
    if malformed:
        raise ValueError(f"{label} has incomplete four-sample tasks: {malformed[:5]}")

    expected_sources = {
        str(contract.get("source_task_id") or "") for contract in contracts.values()
    }
    if "" in expected_sources:
        raise ValueError("corpus contains an empty source-state id")
    target_counts = Counter(
        str(contract.get("target_action") or "") for contract in contracts.values()
    )
    source_target_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for contract in contracts.values():
        source_target_counts[str(contract.get("source_task_id") or "")][
            str(contract.get("target_action") or "")
        ] += 1
    if (
        len(contracts) != EXPECTED_TASKS
        or len(expected_sources) != EXPECTED_SOURCES
        or dict(sorted(target_counts.items())) != EXPECTED_TARGET_COUNTS
        or any(
            dict(sorted(source_counts.items())) != EXPECTED_TARGETS_PER_SOURCE
            for source_counts in source_target_counts.values()
        )
    ):
        raise ValueError(f"{label} corpus does not match the frozen H-005 contract")
    integrity = (report.get("corpus_integrity") or {}).get("internal_dev") or {}
    if (
        integrity.get("passed") is not True
        or int(integrity.get("rows") or 0) != len(contracts)
        or int(integrity.get("unique_task_ids") or 0) != len(contracts)
        or int(integrity.get("unique_source_task_ids") or 0) != len(expected_sources)
        or integrity.get("target_counts") != dict(sorted(target_counts.items()))
        or integrity.get("targets_per_source") != EXPECTED_TARGETS_PER_SOURCE
    ):
        raise ValueError(f"{label} corpus integrity summary does not match content")

    split_summary = (report.get("splits") or {}).get("internal_dev") or {}
    recomputed = {
        "rollouts": len(rows),
        "raw_model_legacy_full_success_rate": _rate(rows, _raw_success),
        "raw_model_contract_success_rate": _rate(rows, _semantic_success),
        "assembled_system_success_rate": _rate(rows, _system_success),
        "action_accuracy": sum(
            bool((row.get("checks") or {}).get("action_match")) for row in rows
        )
        / len(rows),
        "reason_assembly_exact_rate": sum(
            bool(row.get("reason_assembly_exact")) for row in rows
        )
        / len(rows),
    }
    for key, observed in recomputed.items():
        recorded = split_summary.get(key)
        if recorded is None or not math.isclose(
            float(recorded), float(observed), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"{label} report summary does not match rollouts for {key}: "
                f"{recorded!r} != {observed!r}"
            )
    return {
        "passed": True,
        "rows": len(rows),
        "tasks": len(by_task),
        "source_states": len(expected_sources),
        "target_counts": dict(sorted(target_counts.items())),
        "four_sample_indices_exact": True,
        "derived_seeds_exact": True,
        "report_summary_recomputed": True,
        "fingerprints_recomputed_from_corpus": True,
        "authority_transport_exact": True,
        "inference_audit_coverage_rate": 1.0,
    }


def _mcnemar_exact(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    metric: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    both_pass = baseline_only = candidate_only = both_fail = 0
    for baseline, candidate in pairs:
        baseline_pass = metric(baseline)
        candidate_pass = metric(candidate)
        if baseline_pass and candidate_pass:
            both_pass += 1
        elif baseline_pass:
            baseline_only += 1
        elif candidate_pass:
            candidate_only += 1
        else:
            both_fail += 1
    discordant = baseline_only + candidate_only
    if discordant:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(baseline_only, candidate_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    return {
        "both_pass": both_pass,
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "both_fail": both_fail,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cluster_bootstrap_delta(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    metric: Callable[[dict[str, Any]], bool],
    *,
    seed: int,
    rounds: int,
) -> dict[str, Any]:
    clusters: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in pairs:
        clusters[str(pair[0].get("source_task_id") or "")].append(pair)
    cluster_ids = sorted(clusters)
    if not cluster_ids:
        raise ValueError("paired rows contain no source-state clusters")
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(rounds):
        sampled = [rng.choice(cluster_ids) for _ in cluster_ids]
        sampled_pairs = [
            pair for cluster_id in sampled for pair in clusters[cluster_id]
        ]
        deltas.append(
            sum(
                metric(candidate) - metric(baseline)
                for baseline, candidate in sampled_pairs
            )
            / len(sampled_pairs)
        )
    point = sum(
        metric(candidate) - metric(baseline) for baseline, candidate in pairs
    ) / len(pairs)
    return {
        "clusters": len(cluster_ids),
        "pairs": len(pairs),
        "point_delta": point,
        "ci95": [_quantile(deltas, 0.025), _quantile(deltas, 0.975)],
        "rounds": rounds,
        "seed": seed,
    }


def _pass4_pairs(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    samples_per_task: int,
    metric: Callable[[dict[str, Any]], bool],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in pairs:
        grouped[str(pair[0].get("task_id") or "")].append(pair)
    output: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for task_id, task_pairs in sorted(grouped.items()):
        if len(task_pairs) != samples_per_task:
            raise ValueError(
                f"paired task {task_id!r} does not contain {samples_per_task} samples"
            )
        baseline_template, candidate_template = task_pairs[0]
        output.append(
            (
                {
                    **baseline_template,
                    "pass4": all(metric(pair[0]) for pair in task_pairs),
                },
                {
                    **candidate_template,
                    "pass4": all(metric(pair[1]) for pair in task_pairs),
                },
            )
        )
    return output


def _paired_metric_report(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    metric: Callable[[dict[str, Any]], bool],
    *,
    samples_per_task: int,
    seed: int,
    rounds: int,
) -> dict[str, Any]:
    pass4_pairs = _pass4_pairs(pairs, samples_per_task, metric)
    return {
        "descriptive_rollout_mcnemar": _mcnemar_exact(pairs, metric),
        "source_state_clustered_bootstrap": _cluster_bootstrap_delta(
            pairs, metric, seed=seed, rounds=rounds
        ),
        "pass4": {
            "task_level_mcnemar": _mcnemar_exact(pass4_pairs, _pass4_success),
            "source_state_clustered_bootstrap": _cluster_bootstrap_delta(
                pass4_pairs,
                _pass4_success,
                seed=seed + 1,
                rounds=rounds,
            ),
        },
    }


def _h005_gate(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "reason_assembly_coverage_100pct": summary["reason_assembly_coverage_rate"]
        == 1.0,
        "reason_assembly_exact_given_eligible_100pct": (
            summary["reason_assembly_exact_given_eligible_rate"] == 1.0
        ),
        "model_contract_compliance_100pct": (
            summary["model_contract_compliance_rate"] == 1.0
        ),
        "controller_override_attempt_0pct": (
            summary["controller_override_attempt_rate"] == 0.0
        ),
        "controller_hydration_exact_100pct": (
            summary["controller_hydration_exact_rate"] == 1.0
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _inference_value(row: dict[str, Any], key: str) -> float:
    return float(row["inference_metrics"][0][key])


def _completion_tokens(row: dict[str, Any]) -> float:
    return _inference_value(row, "completion_tokens")


def _request_latency_ms(row: dict[str, Any]) -> float:
    return _inference_value(row, "request_latency_ms")


def compare(
    *,
    corpus_path: Path,
    baseline_dir: Path,
    candidate_dir: Path,
    expected_baseline_adapter_sha256: str,
    expected_candidate_adapter_sha256: str,
    output_path: Path,
    bootstrap_seed: int,
    bootstrap_rounds: int,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output_path}")
    corpus_sha256 = _sha256(corpus_path)
    if corpus_sha256 != EXPECTED_INTERNAL_DEV_CORPUS_SHA256:
        raise ValueError("supplied corpus is not the frozen H-005 internal-dev corpus")
    corpus_rows = load_grpo_corpus(corpus_path)
    contracts = {row.task.task_id: verifier_repair_metadata(row) for row in corpus_rows}
    if len(contracts) != len(corpus_rows):
        raise ValueError("corpus task ids are not unique")
    expected_fingerprints = {
        row.task.task_id: environment_fingerprint(row.task, row.snapshot)
        for row in corpus_rows
    }
    baseline_report = _load_json(baseline_dir / "report.json")
    candidate_report = _load_json(candidate_dir / "report.json")
    current_code_snapshot = evaluation_code_snapshot()
    if baseline_report.get("evaluation_code_snapshot") != current_code_snapshot:
        raise ValueError("baseline evaluator code snapshot does not match current code")
    if candidate_report.get("evaluation_code_snapshot") != current_code_snapshot:
        raise ValueError(
            "candidate evaluator code snapshot does not match current code"
        )
    baseline_checkpoint = baseline_report.get("checkpoint")
    candidate_checkpoint = candidate_report.get("checkpoint")
    if not isinstance(baseline_checkpoint, str) or not baseline_checkpoint:
        raise ValueError("baseline checkpoint identity is empty")
    if not isinstance(candidate_checkpoint, str) or not candidate_checkpoint:
        raise ValueError("candidate checkpoint identity is empty")
    baseline_adapter_path = Path(baseline_checkpoint) / "adapter_model.safetensors"
    candidate_adapter_path = Path(candidate_checkpoint) / "adapter_model.safetensors"
    if not baseline_adapter_path.is_file():
        raise ValueError(f"baseline adapter file is missing: {baseline_adapter_path}")
    if not candidate_adapter_path.is_file():
        raise ValueError(f"candidate adapter file is missing: {candidate_adapter_path}")
    baseline_adapter_sha256 = baseline_report.get("checkpoint_adapter_sha256")
    candidate_adapter_sha256 = candidate_report.get("checkpoint_adapter_sha256")
    if (
        not isinstance(baseline_adapter_sha256, str)
        or not baseline_adapter_sha256
        or baseline_adapter_sha256 != expected_baseline_adapter_sha256
    ):
        raise ValueError("baseline adapter SHA-256 does not match the locked identity")
    if (
        not isinstance(candidate_adapter_sha256, str)
        or not candidate_adapter_sha256
        or candidate_adapter_sha256 != expected_candidate_adapter_sha256
    ):
        raise ValueError("candidate adapter SHA-256 does not match the locked identity")
    if baseline_adapter_sha256 == candidate_adapter_sha256:
        raise ValueError("baseline and candidate adapters must be different")
    if _sha256(baseline_adapter_path) != expected_baseline_adapter_sha256:
        raise ValueError("baseline adapter file does not match the locked identity")
    if _sha256(candidate_adapter_path) != expected_candidate_adapter_sha256:
        raise ValueError("candidate adapter file does not match the locked identity")
    if baseline_checkpoint == candidate_checkpoint:
        raise ValueError("baseline and candidate checkpoints must be different")
    baseline_model_manifest = baseline_report.get("effective_model_manifest") or {}
    candidate_model_manifest = candidate_report.get("effective_model_manifest") or {}
    expected_baseline_manifest = {
        "schema_version": "effective-model-manifest.v1",
        "model_files": model_file_manifest(baseline_checkpoint),
        "tokenizer": effective_tokenizer_manifest_for_checkpoint(baseline_checkpoint),
    }
    expected_candidate_manifest = {
        "schema_version": "effective-model-manifest.v1",
        "model_files": model_file_manifest(candidate_checkpoint),
        "tokenizer": effective_tokenizer_manifest_for_checkpoint(candidate_checkpoint),
    }
    if baseline_model_manifest != expected_baseline_manifest:
        raise ValueError("baseline effective model manifest does not match local files")
    if candidate_model_manifest != expected_candidate_manifest:
        raise ValueError(
            "candidate effective model manifest does not match local files"
        )
    if (
        baseline_model_manifest["model_files"]["base_model"]
        != candidate_model_manifest["model_files"]["base_model"]
    ):
        raise ValueError("baseline and candidate base model identities differ")
    if baseline_model_manifest["tokenizer"] != candidate_model_manifest["tokenizer"]:
        raise ValueError("baseline and candidate effective tokenizer identities differ")
    contract_equality = {
        key: baseline_report.get(key) == candidate_report.get(key)
        for key in CONTRACT_KEYS
    }
    if not all(contract_equality.values()):
        mismatches = [key for key, equal in contract_equality.items() if not equal]
        raise ValueError(f"evaluation stacks are not identical: {mismatches}")
    baseline_rows = _load_jsonl(baseline_dir / "rollouts.jsonl")
    candidate_rows = _load_jsonl(candidate_dir / "rollouts.jsonl")
    baseline_validation = _validate_arm(
        label="baseline",
        rows=baseline_rows,
        report=baseline_report,
        contracts=contracts,
        expected_fingerprints=expected_fingerprints,
        corpus_sha256=corpus_sha256,
    )
    candidate_validation = _validate_arm(
        label="candidate",
        rows=candidate_rows,
        report=candidate_report,
        contracts=contracts,
        expected_fingerprints=expected_fingerprints,
        corpus_sha256=corpus_sha256,
    )
    baseline_by_key = {_pair_key(row): row for row in baseline_rows}
    candidate_by_key = {_pair_key(row): row for row in candidate_rows}
    if len(baseline_by_key) != len(baseline_rows) or len(candidate_by_key) != len(
        candidate_rows
    ):
        raise ValueError("rollout pair keys are not unique")
    if set(baseline_by_key) != set(candidate_by_key):
        raise ValueError("baseline and candidate rollout pair keys differ")
    pairs = [
        (baseline_by_key[key], candidate_by_key[key]) for key in sorted(baseline_by_key)
    ]
    base_seed = int(baseline_report["seed"])
    paired_seed_exact = all(
        int(baseline.get("rollout_seed") or 0)
        == _expected_seed(
            base_seed,
            task_id=str(baseline.get("task_id") or ""),
            sample_index=int(baseline.get("sample_index") or 0),
        )
        for baseline, _ in pairs
    )
    pair_invariants = {
        "paired_seed_exact": paired_seed_exact,
        "source_task_id_exact": all(
            baseline.get("source_task_id") == candidate.get("source_task_id")
            for baseline, candidate in pairs
        ),
        "target_action_exact": all(
            baseline.get("target_action") == candidate.get("target_action")
            for baseline, candidate in pairs
        ),
        "initial_state_fingerprint_exact": all(
            baseline.get("initial_state_fingerprint")
            == candidate.get("initial_state_fingerprint")
            for baseline, candidate in pairs
        ),
        "authority_transport_exact": all(
            baseline.get("authority_transport") == candidate.get("authority_transport")
            for baseline, candidate in pairs
        ),
        "prompt_input_ids_sha256_exact": all(
            (baseline.get("generation_audit") or {}).get("prompt_input_ids_sha256")
            == (candidate.get("generation_audit") or {}).get("prompt_input_ids_sha256")
            for baseline, candidate in pairs
        ),
    }
    if not all(pair_invariants.values()):
        failed = [key for key, passed in pair_invariants.items() if not passed]
        raise ValueError(f"paired rollout invariants failed: {failed}")
    samples_per_task = int(baseline_report["samples_per_task"])
    baseline_summary = _model_summary(baseline_rows, samples_per_task)
    candidate_summary = _model_summary(candidate_rows, samples_per_task)
    baseline_gate = _h005_gate(baseline_summary)
    candidate_gate = _h005_gate(candidate_summary)
    targets = sorted(
        {str(baseline.get("target_action") or "") for baseline, _ in pairs}
    )
    paired_by_target = {}
    for target_index, target in enumerate(targets):
        target_pairs = [
            pair for pair in pairs if str(pair[0].get("target_action") or "") == target
        ]
        target_seed = bootstrap_seed + 100 + target_index * 10
        paired_by_target[target] = {
            "paired_rollouts": len(target_pairs),
            "paired_tasks": len(
                {str(pair[0].get("task_id") or "") for pair in target_pairs}
            ),
            "paired_raw_model_legacy": _paired_metric_report(
                target_pairs,
                _raw_success,
                samples_per_task=samples_per_task,
                seed=target_seed,
                rounds=bootstrap_rounds,
            ),
            "paired_raw_model_contract": _paired_metric_report(
                target_pairs,
                _semantic_success,
                samples_per_task=samples_per_task,
                seed=target_seed + 2,
                rounds=bootstrap_rounds,
            ),
            "paired_assembled_system": _paired_metric_report(
                target_pairs,
                _system_success,
                samples_per_task=samples_per_task,
                seed=target_seed + 4,
                rounds=bootstrap_rounds,
            ),
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": "train-internal same-stack comparison only",
        "production_promotion_eligible": False,
        "corpus_path": str(corpus_path.resolve()),
        "corpus_sha256": corpus_sha256,
        "baseline_report_sha256": _sha256(baseline_dir / "report.json"),
        "baseline_rollouts_sha256": _sha256(baseline_dir / "rollouts.jsonl"),
        "candidate_report_sha256": _sha256(candidate_dir / "report.json"),
        "candidate_rollouts_sha256": _sha256(candidate_dir / "rollouts.jsonl"),
        "baseline_checkpoint": baseline_report.get("checkpoint"),
        "baseline_adapter_sha256": baseline_report.get("checkpoint_adapter_sha256"),
        "candidate_checkpoint": candidate_report.get("checkpoint"),
        "candidate_adapter_sha256": candidate_report.get("checkpoint_adapter_sha256"),
        "effective_model_identity": {
            "shared_base_model": baseline_model_manifest["model_files"]["base_model"],
            "shared_tokenizer": baseline_model_manifest["tokenizer"],
            "baseline_adapter": baseline_model_manifest["model_files"]["adapter"],
            "candidate_adapter": candidate_model_manifest["model_files"]["adapter"],
        },
        "contract_equality": contract_equality,
        "pair_invariants": pair_invariants,
        "baseline_validation": baseline_validation,
        "candidate_validation": candidate_validation,
        "paired_rollouts": len(pairs),
        "paired_tasks": len(
            {str(baseline.get("task_id") or "") for baseline, _ in pairs}
        ),
        "statistics_contract": {
            "formal_interval": "paired source-state-clustered bootstrap 95% CI",
            "descriptive_only": "rollout/task McNemar exact test",
            "delta_direction": "candidate minus baseline",
        },
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "h005_safety_gate": {
            "baseline": baseline_gate,
            "candidate": candidate_gate,
            "passed": baseline_gate["passed"] and candidate_gate["passed"],
        },
        "paired_raw_model_legacy": _paired_metric_report(
            pairs,
            _raw_success,
            samples_per_task=samples_per_task,
            seed=bootstrap_seed,
            rounds=bootstrap_rounds,
        ),
        "paired_raw_model_contract": _paired_metric_report(
            pairs,
            _semantic_success,
            samples_per_task=samples_per_task,
            seed=bootstrap_seed + 2,
            rounds=bootstrap_rounds,
        ),
        "paired_assembled_system": _paired_metric_report(
            pairs,
            _system_success,
            samples_per_task=samples_per_task,
            seed=bootstrap_seed + 4,
            rounds=bootstrap_rounds,
        ),
        "paired_by_target": paired_by_target,
        "paired_cost_observation": {
            "completion_tokens_candidate_minus_baseline": _cluster_bootstrap_delta(
                pairs,
                _completion_tokens,
                seed=bootstrap_seed + 1_000,
                rounds=bootstrap_rounds,
            ),
            "request_latency_ms_candidate_minus_baseline": _cluster_bootstrap_delta(
                pairs,
                _request_latency_ms,
                seed=bootstrap_seed + 1_001,
                rounds=bootstrap_rounds,
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--expected-baseline-adapter-sha256", required=True)
    parser.add_argument("--expected-candidate-adapter-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260910)
    parser.add_argument("--bootstrap-rounds", type=int, default=10_000)
    args = parser.parse_args()
    if args.bootstrap_rounds < 1:
        parser.error("bootstrap-rounds must be positive")
    report = compare(
        corpus_path=args.corpus,
        baseline_dir=args.baseline_dir,
        candidate_dir=args.candidate_dir,
        expected_baseline_adapter_sha256=args.expected_baseline_adapter_sha256,
        expected_candidate_adapter_sha256=args.expected_candidate_adapter_sha256,
        output_path=args.output,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_rounds=args.bootstrap_rounds,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
