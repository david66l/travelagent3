"""Freeze and audit one H006 internal full-Agent-Loop diagnostic run.

This utility never opens promotion-val or sealed payloads.  It snapshots the
complete local runtime surface before/after the run and distinguishes model
business failures from infrastructure-invalid results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = str(ROOT / "backend" / "src")
if BACKEND_SRC not in sys.path:
    sys.path.insert(0, BACKEND_SRC)


SCHEMA_VERSION = "h006-full-agent-loop-run-audit.v1"
FULL_LOOP_SCHEMA = "full-agent-loop.v3"
RECOVERY_SCHEMA = "full-agent-loop-recovery.v1"
RECOVERY_SCORING_CONTRACT = "recovery-injection-accounting.v2"
FULL_LOOP_BENCHMARK_SHA256 = (
    "6c49db74b80141979b56a93a5e287f959b1abe93a12eda43c2448fd74e0257d4"
)
RECOVERY_BENCHMARK_SHA256 = (
    "cabb73f84624a9f762dbedffe0cec34f22f183131381738f0d3a8de7f15e7037"
)
EXPECTED_INTENT_MODEL = "deepseek-v4-flash"
EXPECTED_INTENT_BACKEND = "cloud-openai-compatible"
RUNTIME_ROOTS = ("backend/src", "backend/tests", "scripts")
RUNTIME_FILES = ("backend/pyproject.toml", "backend/uv.lock")
PACKAGE_NAMES = (
    "openai",
    "pydantic",
    "SQLAlchemy",
    "redis",
    "ortools",
    "httpx",
    "langsmith",
)
MANIFEST_SCHEMA_VERSION = "h006-full-agent-loop-runtime-manifest.v1"
SNAPSHOT_MANIFEST_FIELDS = (
    "run_id",
    "git_head",
    "git_status_sha256",
    "git_patch_sha256",
    "runtime_file_count",
    "runtime_combined_sha256",
    "files",
    "python",
    "safe_settings",
    "benchmark_contract",
    "scope_boundary",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(repo_root: Path, *args: str, check: bool = True) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _runtime_paths(repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    for root_name in RUNTIME_ROOTS:
        root = repo_root / root_name
        paths.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    paths.extend(
        repo_root / name for name in RUNTIME_FILES if (repo_root / name).is_file()
    )
    return sorted(set(paths), key=lambda item: item.relative_to(repo_root).as_posix())


def _safe_settings(repo_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(repo_root / "backend" / "src"))
    try:
        from core.settings import settings
        from langsmith.utils import tracing_is_enabled

        behavior_fields = {
            "llm_model",
            "llm_temperature",
            "llm_max_tokens",
            "llm_timeout",
            "local_llm_enabled",
            "local_llm_model",
            "vllm_enabled",
            "vllm_max_retries",
            "search_engine",
            "embedding_provider",
            "tool_timeout_seconds",
            "tool_max_retries",
            "agentic_tool_timeout_seconds",
            "crawl_rate_limit",
            "crawl_max_retries",
            "crawl_timeout",
            "local_cache_ttl_seconds",
            "seed_data_dir",
            "seed_cities",
            "langsmith_tracing",
            "langsmith_project",
        }
        behavior_fields.update(
            name
            for name in type(settings).model_fields
            if name.startswith(
                (
                    "agentic_",
                    "cache_ttl_",
                    "circuit_breaker_",
                    "cost_circuit_breaker_",
                    "prompt_compress_",
                )
            )
        )
        behavior = {}
        for name in sorted(behavior_fields):
            value = getattr(settings, name)
            # Environment files can be malformed (for example two assignments
            # accidentally joined on one line). Never serialize any string
            # value into an audit artifact; a digest still detects drift.
            behavior[name] = (
                {
                    "sha256": _sha256_bytes(value.encode("utf-8")),
                    "length": len(value),
                }
                if isinstance(value, str)
                else value
            )
        endpoint_fingerprints = {
            name: _sha256_bytes(str(getattr(settings, name)).encode("utf-8"))
            for name in (
                "database_url",
                "redis_url",
                "redis_redlock_urls",
                "openai_base_url",
                "local_llm_url",
                "vllm_base_url",
                "vrp_solver_url",
                "langsmith_endpoint",
            )
        }
        return {
            "behavior": behavior,
            "endpoint_sha256": endpoint_fingerprints,
            "credential_presence": {
                "deepseek_api_key": bool(settings.deepseek_api_key),
                "openai_api_key": bool(settings.openai_api_key),
                "tavily_api_key": bool(settings.tavily_api_key),
                "amap_key": bool(settings.amap_key),
                "weather_key": bool(settings.weather_key),
                "langsmith_api_key": bool(settings.langsmith_api_key),
                "langchain_api_key": bool(os.environ.get("LANGCHAIN_API_KEY")),
            },
            "runtime_flags": {
                "langsmith_traceable_decorator_active": bool(
                    settings.langsmith_tracing and settings.langsmith_api_key
                ),
                "langsmith_sdk_tracing_enabled": bool(tracing_is_enabled()),
            },
        }
    finally:
        sys.path.pop(0)


def build_snapshot(repo_root: Path, *, phase: str, run_id: str) -> dict[str, Any]:
    paths = _runtime_paths(repo_root)
    files = {
        path.relative_to(repo_root).as_posix(): _sha256(path) for path in paths
    }
    combined = "\n".join(f"{path}\0{digest}" for path, digest in files.items())
    scope = [*RUNTIME_ROOTS, *RUNTIME_FILES]
    status = _run_git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *scope,
    )
    patch = _run_git(
        repo_root,
        "diff",
        "--binary",
        "HEAD",
        "--",
        *scope,
    )
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "runtime_snapshot",
        "phase": phase,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root.resolve()),
        "git_head": _run_git(repo_root, "rev-parse", "HEAD").decode().strip(),
        "git_status_sha256": _sha256_bytes(status),
        "git_status_lines": status.decode("utf-8", errors="replace").splitlines(),
        "git_patch_sha256": _sha256_bytes(patch),
        "runtime_file_count": len(files),
        "runtime_combined_sha256": _sha256_bytes(combined.encode("utf-8")),
        "files": files,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
            "packages": versions,
        },
        "safe_settings": _safe_settings(repo_root),
        "benchmark_contract": {
            "full_loop_sha256": FULL_LOOP_BENCHMARK_SHA256,
            "full_loop_cases": {"core": 10, "expanded": 20},
            "recovery_sha256": RECOVERY_BENCHMARK_SHA256,
            "recovery_cases": 8,
            "recovery_group_size": 4,
        },
        "scope_boundary": {
            "internal_development_diagnostic_only": True,
            "promotion_val_opened": False,
            "sealed_160_opened": False,
            "production_promotion_eligible": False,
        },
    }


def build_expected_manifest(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        **{name: snapshot[name] for name in SNAPSHOT_MANIFEST_FIELDS},
    }


def validate_snapshot_manifest(
    snapshot: dict[str, Any], expected: dict[str, Any]
) -> list[str]:
    mismatches = []
    if expected.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        mismatches.append("schema_version")
    mismatches.extend(
        name
        for name in SNAPSHOT_MANIFEST_FIELDS
        if snapshot.get(name) != expected.get(name)
    )
    return mismatches


def _truthy_runtime_errors(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return [str(value)] if value else []
    return [f"{key}:{item}" for key, item in value.items() if item]


def _intent_anomalies(records: Iterable[dict[str, Any]]) -> list[str]:
    anomalies: list[str] = []
    for row in records:
        row_id = str(row.get("case_id") or row.get("rollout_key") or "unknown")
        intent = row.get("intent") or {}
        inference = row.get("intent_inference") or {}
        if intent.get("parse_source") != "llm":
            anomalies.append(f"{row_id}:intent_parse_source={intent.get('parse_source')}")
        if inference.get("model") != EXPECTED_INTENT_MODEL:
            anomalies.append(f"{row_id}:intent_model={inference.get('model')}")
        if inference.get("backend") != EXPECTED_INTENT_BACKEND:
            anomalies.append(f"{row_id}:intent_backend={inference.get('backend')}")
    return anomalies


def _record_infrastructure_anomalies(records: Iterable[dict[str, Any]]) -> list[str]:
    anomalies: list[str] = []
    for row in records:
        row_id = str(row.get("rollout_key") or row.get("case_id") or "unknown")
        if row.get("error"):
            anomalies.append(f"{row_id}:error={row['error']}")
        if row.get("agent_status") == "exception":
            anomalies.append(f"{row_id}:agent_status=exception")
        for code in row.get("failures") or []:
            if str(code).startswith("EXCEPTION:"):
                anomalies.append(f"{row_id}:{code}")
        if row.get("runtime_error"):
            anomalies.append(f"{row_id}:runtime_error={row['runtime_error']}")
        anomalies.extend(
            f"{row_id}:runtime_error={item}"
            for item in _truthy_runtime_errors(row.get("runtime_errors"))
        )
    return anomalies


def _frozen_full_case_ids(suite: str) -> list[str]:
    from evaluation.full_agent_loop_benchmark import build_frozen_cases

    return [case.case_id for case in build_frozen_cases() if case.suite == suite]


def _frozen_recovery_case_ids() -> list[str]:
    from evaluation.full_agent_loop_recovery import build_recovery_cases

    return [case.case_id for case in build_recovery_cases()]


def validate_full_report(
    report: dict[str, Any],
    *,
    suite: str,
    expected_total: int,
    policy_model: str,
    policy_backend: str,
) -> dict[str, Any]:
    records = list(report.get("records") or [])
    expected_case_ids = _frozen_full_case_ids(suite)
    record_case_ids = [str(row.get("case_id") or "") for row in records]
    checks = {
        "schema": report.get("schema_version") == FULL_LOOP_SCHEMA,
        "benchmark_hash": report.get("benchmark_hash") == FULL_LOOP_BENCHMARK_SHA256,
        "policy_model": report.get("policy_model") == policy_model,
        "policy_backend": report.get("policy_backend") == policy_backend,
        "intent_model": report.get("intent_model") == EXPECTED_INTENT_MODEL,
        "summary_total": (report.get("summary") or {}).get("total") == expected_total,
        "record_count": len(records) == expected_total,
        "frozen_suite_size": len(expected_case_ids) == expected_total,
        "selected_case_ids_exact": report.get("selected_case_ids")
        == expected_case_ids,
        "record_case_ids_exact": record_case_ids == expected_case_ids,
    }
    anomalies = [
        *_record_infrastructure_anomalies(records),
        *_intent_anomalies(records),
    ]
    return {
        "checks": checks,
        "infrastructure_anomalies": anomalies,
        "infrastructure_valid": all(checks.values()) and not anomalies,
        "business_passed": int((report.get("summary") or {}).get("passed") or 0),
        "business_total": expected_total,
        "business_pass_rate": (report.get("summary") or {}).get("pass_rate"),
    }


def validate_recovery_report(
    report: dict[str, Any],
    *,
    run_id: str,
    policy_model: str,
    policy_backend: str,
) -> dict[str, Any]:
    expected_total = 32
    records = list(report.get("records") or [])
    expected_case_ids = _frozen_recovery_case_ids()
    expected_rollout_keys = [
        f"{case_id}::sample-{sample_index}"
        for case_id in expected_case_ids
        for sample_index in range(4)
    ]
    injection_statuses = [
        str((row.get("recovery") or {}).get("injection_status") or "missing")
        for row in records
    ]
    allowed_injection_statuses = {
        "injected_once",
        "not_reached_due_to_policy_failure",
        "not_reached_due_to_runtime_error",
        "injector_error",
    }
    reported_status_counts = dict((report.get("summary") or {}).get("injection_status_counts") or {})
    actual_status_counts = dict(Counter(injection_statuses))
    checks = {
        "schema": report.get("schema_version") == RECOVERY_SCHEMA,
        "scoring_contract": report.get("scoring_contract_version")
        == RECOVERY_SCORING_CONTRACT,
        "benchmark_hash": report.get("benchmark_hash") == RECOVERY_BENCHMARK_SHA256,
        "policy_model": report.get("policy_model") == policy_model,
        "policy_backend": report.get("policy_backend") == policy_backend,
        "intent_model": report.get("intent_model") == EXPECTED_INTENT_MODEL,
        "rollout_id": report.get("rollout_id") == run_id,
        "base_seed": report.get("base_seed") == 421,
        "group_size": report.get("group_size") == 4,
        "temperature": abs(float(report.get("temperature") or 0.0) - 0.8) < 1e-9,
        "summary_total": (report.get("summary") or {}).get("total") == expected_total,
        "record_count": len(records) == expected_total,
        "selected_case_ids_exact": report.get("selected_case_ids")
        == expected_case_ids,
        "rollout_keys_exact": [row.get("rollout_key") for row in records]
        == expected_rollout_keys,
        "injection_status_accounted": all(
            status in allowed_injection_statuses for status in injection_statuses
        ),
        "fault_injected_consistent": all(
            bool((row.get("recovery") or {}).get("fault_injected"))
            == (status == "injected_once")
            for row, status in zip(records, injection_statuses, strict=True)
        ),
        "injection_status_summary_exact": reported_status_counts == actual_status_counts,
        "no_injector_error": "injector_error" not in injection_statuses,
        "no_unclassified_non_reach": "missing" not in injection_statuses,
    }
    anomalies = [
        *_record_infrastructure_anomalies(records),
        *_intent_anomalies(records),
    ]
    return {
        "checks": checks,
        "infrastructure_anomalies": anomalies,
        "infrastructure_valid": all(checks.values()) and not anomalies,
        "business_passed": int((report.get("summary") or {}).get("passed") or 0),
        "business_total": expected_total,
        "pass_rate": (report.get("summary") or {}).get("pass_rate"),
        "first_try_recovery_rate": (report.get("summary") or {}).get(
            "first_try_recovery_rate"
        ),
        "full_chain_pass_rate": (report.get("summary") or {}).get(
            "full_chain_pass_rate"
        ),
        "strata": (report.get("summary") or {}).get("strata"),
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain one JSON object")
        rows.append(value)
    return rows


def _expected_full_episode_ids(suite: str, rollout_id: str) -> list[str]:
    from evaluation.full_agent_loop_benchmark import build_frozen_cases

    expected: list[str] = []
    for case in build_frozen_cases():
        if case.suite != suite or case.expected_outcome == "clarification":
            continue
        phases = ("initial", "revised") if case.expected_outcome == "revision" else ("initial",)
        expected.extend(f"{case.case_id}:{rollout_id}:{phase}" for phase in phases)
    return sorted(expected)


def _expected_recovery_episode_ids(run_id: str) -> list[str]:
    return sorted(
        f"{case_id}:{run_id}:421:{sample_index}:initial"
        for case_id in _frozen_recovery_case_ids()
        for sample_index in range(4)
    )


def validate_episode_sidecar(path: Path, expected_ids: list[str], *, recovery: bool) -> dict[str, Any]:
    from agentic.sft_dataset import EpisodeCandidate

    parsed = [EpisodeCandidate.model_validate(row) for row in _load_jsonl(path)]
    scenario_ids = [row.scenario_id for row in parsed]
    observations = [
        observation
        for candidate in parsed
        for step in candidate.episode.steps
        for observation in step.observations
    ]

    def has_model_policy_terminal(candidate: EpisodeCandidate) -> bool:
        return any(
            event.event_type == "episode_terminated"
            and event.payload.get("reason") == "policy_error_fallback"
            and event.payload.get("failure_class") == "model_policy_failure"
            for event in candidate.episode.events
        )

    injected_counts = [
        sum(
            bool(
                observation.error
                and observation.error.details.get("fault_injection") is True
            )
            for step in candidate.episode.steps
            for observation in step.observations
        )
        for candidate in parsed
    ]
    checks = {
        "record_count": len(parsed) == len(expected_ids),
        "scenario_ids_exact": scenario_ids == expected_ids,
        "unique_scenario_ids": len(set(scenario_ids)) == len(scenario_ids),
        "episodes_have_steps": all(candidate.episode.steps for candidate in parsed),
        "observations_present": bool(observations),
    }
    if recovery:
        checks.pop("episodes_have_steps")
        checks.pop("observations_present")
        checks["steps_or_explicit_policy_terminal"] = all(
            candidate.episode.steps or has_model_policy_terminal(candidate)
            for candidate in parsed
        )
        checks["injection_or_explicit_policy_non_reach"] = all(
            injected_count == 1
            or (injected_count == 0 and has_model_policy_terminal(candidate))
            for candidate, injected_count in zip(parsed, injected_counts, strict=True)
        )
        checks["observations_or_all_policy_terminals"] = bool(observations) or all(
            has_model_policy_terminal(candidate) for candidate in parsed
        )
    return {
        "checks": checks,
        "valid": all(checks.values()),
        "records": len(parsed),
        "observations": len(observations),
    }


def validate_server_cleanup(value: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "cleanup_passed": value.get("cleanup_passed") is True,
        "process_group_empty": value.get("process_group_empty") is True,
        "remote_port_released": value.get("remote_port_released") is True,
        "gpu_memory_recovered": value.get("gpu_memory_recovered") is True,
    }
    return {"checks": checks, "valid": all(checks.values())}


def build_postflight(
    run_dir: Path,
    *,
    run_id: str,
    policy_model: str,
    policy_backend: str,
) -> dict[str, Any]:
    required = {
        name: run_dir / name
        for name in (
            "run-contract.json",
            "runtime-pre.json",
            "runtime-post.json",
            "remote-model-manifest.txt",
            "remote-process.json",
            "server-health.json",
            "server-cleanup.json",
            "monitor.jsonl",
            "controller.log",
            "stage-processes.json",
            "test-result.json",
            "preflight-tests.stdout.log",
            "preflight-tests.stderr.log",
            "core-10.json",
            "core-10-episodes.jsonl",
            "core-10.stdout.log",
            "core-10.stderr.log",
            "expanded-20.json",
            "expanded-20-episodes.jsonl",
            "expanded-20.stdout.log",
            "expanded-20.stderr.log",
            "recovery-8x4.json",
            "recovery-8x4-episodes.jsonl",
            "recovery-8x4.stdout.log",
            "recovery-8x4.stderr.log",
            "vllm.stdout.log",
            "vllm.stderr.log",
        )
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required run artifacts: {missing}")

    pre = _load_json(required["runtime-pre.json"])
    post = _load_json(required["runtime-post.json"])
    run_contract = _load_json(required["run-contract.json"])
    remote_process = _load_json(required["remote-process.json"])
    server_health = _load_json(required["server-health.json"])
    server_cleanup = _load_json(required["server-cleanup.json"])
    stages = _load_json(required["stage-processes.json"])
    tests = _load_json(required["test-result.json"])
    core = validate_full_report(
        _load_json(required["core-10.json"]),
        suite="core",
        expected_total=10,
        policy_model=policy_model,
        policy_backend=policy_backend,
    )
    expanded = validate_full_report(
        _load_json(required["expanded-20.json"]),
        suite="expanded",
        expected_total=20,
        policy_model=policy_model,
        policy_backend=policy_backend,
    )
    recovery = validate_recovery_report(
        _load_json(required["recovery-8x4.json"]),
        run_id=run_id,
        policy_model=policy_model,
        policy_backend=policy_backend,
    )
    episodes = {
        "core": validate_episode_sidecar(
            required["core-10-episodes.jsonl"],
            _expected_full_episode_ids("core", f"{run_id}-core"),
            recovery=False,
        ),
        "expanded": validate_episode_sidecar(
            required["expanded-20-episodes.jsonl"],
            _expected_full_episode_ids("expanded", f"{run_id}-expanded"),
            recovery=False,
        ),
        "recovery": validate_episode_sidecar(
            required["recovery-8x4-episodes.jsonl"],
            _expected_recovery_episode_ids(run_id),
            recovery=True,
        ),
    }
    runtime_unchanged = all(
        pre.get(name) == post.get(name)
        for name in (
            "git_head",
            "git_status_sha256",
            "git_patch_sha256",
            "runtime_file_count",
            "runtime_combined_sha256",
            "files",
            "safe_settings",
        )
    )
    stage_rows = list(stages.get("stages") or [])
    stage_processes_valid = (
        len(stage_rows) == 3
        and [row.get("name") for row in stage_rows]
        == ["core-10", "expanded-20", "recovery-8x4"]
        and all(row.get("exit_code") == 0 for row in stage_rows)
        and not any(bool(row.get("timed_out")) for row in stage_rows)
    )
    monitor_rows = _load_jsonl(required["monitor.jsonl"])
    monitor_starts = [
        row.get("name") for row in monitor_rows if row.get("event") == "PROCESS_STARTED"
    ]
    monitor_finishes = [
        row.get("name") for row in monitor_rows if row.get("event") == "PROCESS_FINISHED"
    ]
    expected_monitored = ["preflight-tests", "core-10", "expanded-20", "recovery-8x4"]
    monitoring_valid = (
        monitor_starts == expected_monitored
        and monitor_finishes == expected_monitored
        and not any(row.get("event") == "HARD_TIMEOUT" for row in monitor_rows)
    )
    artifact_presence = {
        name: path.stat().st_size for name, path in required.items()
    }
    required_nonempty = (
        "run-contract.json",
        "remote-model-manifest.txt",
        "remote-process.json",
        "server-health.json",
        "server-cleanup.json",
        "monitor.jsonl",
        "controller.log",
        "preflight-tests.stdout.log",
        "core-10.stdout.log",
        "expanded-20.stdout.log",
        "recovery-8x4.stdout.log",
        "vllm.stdout.log",
    )
    artifacts_complete = all(artifact_presence[name] > 0 for name in required_nonempty)
    contract_valid = all(
        (
            run_contract.get("schema_version") == "h006-full-agent-loop-run-contract.v1",
            run_contract.get("run_id") == run_id,
            run_contract.get("policy_model") == policy_model,
            run_contract.get("policy_backend") == policy_backend,
            run_contract.get("intent_model") == EXPECTED_INTENT_MODEL,
            run_contract.get("intent_backend") == EXPECTED_INTENT_BACKEND,
            run_contract.get("external_tracing_enabled") is False,
            run_contract.get("production_promotion_eligible") is False,
            len(str(run_contract.get("expected_manifest_sha256") or "")) == 64,
        )
    )
    remote_identity_valid = all(
        (
            int(remote_process.get("pid") or 0) > 1,
            int(remote_process.get("pgid") or 0)
            == int(remote_process.get("pid") or 0),
            int(remote_process.get("start_ticks") or 0) > 0,
            remote_process.get("identity_verified") is True,
        )
    )
    server_cleanup_validation = validate_server_cleanup(server_cleanup)
    cleanup_identity_matches = all(
        (
            server_cleanup.get("remote_pid") == remote_process.get("pid"),
            server_cleanup.get("remote_pgid") == remote_process.get("pgid"),
            server_cleanup.get("remote_start_ticks")
            == remote_process.get("start_ticks"),
        )
    )
    server_cleanup_valid = (
        server_cleanup_validation["valid"] and cleanup_identity_matches
    )
    infrastructure_pass = all(
        (
            runtime_unchanged,
            artifacts_complete,
            contract_valid,
            remote_identity_valid,
            server_health.get("health_passed") is True,
            server_health.get("model_alias_present") is True,
            server_health.get("tool_smoke_passed") is True,
            server_health.get("remote_manifest_verified") is True,
            server_cleanup_valid,
            tests.get("exit_code") == 0,
            tests.get("timed_out") is False,
            stage_processes_valid,
            monitoring_valid,
            core["infrastructure_valid"],
            expanded["infrastructure_valid"],
            recovery["infrastructure_valid"],
            all(item["valid"] for item in episodes.values()),
        )
    )
    full_loop_business_pass = (
        core["business_passed"] == 10 and expanded["business_passed"] == 20
    )
    artifact_manifest = {
        path.relative_to(run_dir).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "postflight.json"
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "postflight",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "internal full-Agent-Loop smoke/development diagnostic only",
        "verification_status": (
            "VALIDATED_INTERNAL_DIAGNOSTIC"
            if infrastructure_pass
            else "INVALID_INFRASTRUCTURE"
        ),
        "infrastructure_pass": infrastructure_pass,
        "runtime_unchanged": runtime_unchanged,
        "artifacts_complete": artifacts_complete,
        "artifact_presence": artifact_presence,
        "run_contract_valid": contract_valid,
        "remote_identity_valid": remote_identity_valid,
        "server_cleanup_valid": server_cleanup_valid,
        "server_cleanup": server_cleanup_validation,
        "cleanup_identity_matches": cleanup_identity_matches,
        "stage_processes_valid": stage_processes_valid,
        "monitoring_valid": monitoring_valid,
        "full_loop_business_pass": full_loop_business_pass,
        "core": core,
        "expanded": expanded,
        "recovery": recovery,
        "episodes": episodes,
        "artifact_manifest": artifact_manifest,
        "promotion_boundary": {
            "eligible_to_prepare_promotion_campaign": False,
            "production_promotion_eligible": False,
            "reason": (
                "This internal diagnostic has no paired promotion baseline, power analysis, "
                "or authorized promotion-val access."
            ),
        },
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--repo-root", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.add_argument("--phase", choices=("pre", "post"), required=True)
    snapshot.add_argument("--run-id", required=True)
    snapshot.add_argument("--expected-manifest", type=Path)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--repo-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--run-id", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--repo-root", type=Path, required=True)
    verify.add_argument("--expected-manifest", type=Path, required=True)
    verify.add_argument("--run-id", required=True)

    postflight = subparsers.add_parser("postflight")
    postflight.add_argument("--run-dir", type=Path, required=True)
    postflight.add_argument("--output", type=Path, required=True)
    postflight.add_argument("--run-id", required=True)
    postflight.add_argument("--policy-model", required=True)
    postflight.add_argument("--policy-backend", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "verify":
        snapshot = build_snapshot(
            args.repo_root.resolve(), phase="verify", run_id=args.run_id
        )
        expected = _load_json(args.expected_manifest.resolve())
        mismatches = validate_snapshot_manifest(snapshot, expected)
        if mismatches:
            raise RuntimeError(
                "runtime differs from independently reviewed manifest: "
                + ",".join(mismatches)
            )
        print(json.dumps({"kind": "runtime_manifest_verification", "verified": True}))
        return
    if args.command == "snapshot":
        value = build_snapshot(
            args.repo_root.resolve(), phase=args.phase, run_id=args.run_id
        )
        if args.expected_manifest:
            expected = _load_json(args.expected_manifest.resolve())
            mismatches = validate_snapshot_manifest(value, expected)
            if mismatches:
                raise RuntimeError(
                    "runtime differs from independently reviewed manifest: "
                    + ",".join(mismatches)
                )
    elif args.command == "manifest":
        snapshot = build_snapshot(
            args.repo_root.resolve(), phase="manifest", run_id=args.run_id
        )
        value = build_expected_manifest(snapshot)
    else:
        value = build_postflight(
            args.run_dir.resolve(),
            run_id=args.run_id,
            policy_model=args.policy_model,
            policy_backend=args.policy_backend,
        )
    _write_json(args.output.resolve(), value)
    print(json.dumps({key: value.get(key) for key in ("kind", "verification_status", "run_id") if key in value}))


if __name__ == "__main__":
    main()
