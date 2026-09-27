from __future__ import annotations

from pathlib import Path

from agentic.loop import AgentLoopEvent, PolicyAction, PolicyContext
from agentic.observations import ObservationEnvelope
from agentic.sft_dataset import EpisodeCandidate
from agentic.trajectory import AgentEpisode, TrajectoryStep
from scripts.audit_h006_full_agent_loop_run import (
    FULL_LOOP_BENCHMARK_SHA256,
    RECOVERY_BENCHMARK_SHA256,
    SNAPSHOT_MANIFEST_FIELDS,
    build_expected_manifest,
    validate_full_report,
    validate_episode_sidecar,
    validate_recovery_report,
    validate_server_cleanup,
    validate_snapshot_manifest,
)
from evaluation.full_agent_loop_benchmark import build_frozen_cases
from evaluation.full_agent_loop_recovery import build_recovery_cases


POLICY_MODEL = "travel-h006-checkpoint32"
POLICY_BACKEND = "http://127.0.0.1:18000/v1"
INTENT_MODEL = "deepseek-v4-flash"


def _intent() -> tuple[dict, dict]:
    return (
        {"parse_source": "llm"},
        {"model": INTENT_MODEL, "backend": "cloud-openai-compatible"},
    )


def _full_report(suite: str) -> dict:
    case_ids = [case.case_id for case in build_frozen_cases() if case.suite == suite]
    total = len(case_ids)
    records = []
    for case_id in case_ids:
        intent, inference = _intent()
        records.append(
            {
                "case_id": case_id,
                "passed": True,
                "failures": [],
                "error": None,
                "intent": intent,
                "intent_inference": inference,
            }
        )
    return {
        "schema_version": "full-agent-loop.v3",
        "benchmark_hash": FULL_LOOP_BENCHMARK_SHA256,
        "selected_case_ids": [row["case_id"] for row in records],
        "policy_model": POLICY_MODEL,
        "policy_backend": POLICY_BACKEND,
        "intent_model": INTENT_MODEL,
        "summary": {"total": total, "passed": total, "pass_rate": 1.0},
        "records": records,
    }


def test_full_report_validation_rejects_hidden_case_exception() -> None:
    report = _full_report("core")

    clean = validate_full_report(
        report,
        suite="core",
        expected_total=10,
        policy_model=POLICY_MODEL,
        policy_backend=POLICY_BACKEND,
    )
    assert clean["infrastructure_valid"] is True

    report["records"][0]["error"] = "HTTP 502"
    invalid = validate_full_report(
        report,
        suite="core",
        expected_total=10,
        policy_model=POLICY_MODEL,
        policy_backend=POLICY_BACKEND,
    )
    assert invalid["infrastructure_valid"] is False
    assert "HTTP 502" in invalid["infrastructure_anomalies"][0]


def test_full_report_validation_rejects_intent_fallback() -> None:
    report = _full_report("expanded")
    report["records"][3]["intent"]["parse_source"] = "deterministic_fallback"

    result = validate_full_report(
        report,
        suite="expanded",
        expected_total=20,
        policy_model=POLICY_MODEL,
        policy_backend=POLICY_BACKEND,
    )

    assert result["infrastructure_valid"] is False
    assert any("deterministic_fallback" in row for row in result["infrastructure_anomalies"])


def test_full_report_rejects_self_consistent_wrong_intent_identity() -> None:
    report = _full_report("core")
    report["intent_model"] = "wrong-intent-model"
    for record in report["records"]:
        record["intent_inference"]["model"] = "wrong-intent-model"

    result = validate_full_report(
        report,
        suite="core",
        expected_total=10,
        policy_model=POLICY_MODEL,
        policy_backend=POLICY_BACKEND,
    )

    assert result["infrastructure_valid"] is False
    assert result["checks"]["intent_model"] is False
    assert any("wrong-intent-model" in row for row in result["infrastructure_anomalies"])


def test_full_report_validation_rejects_runtime_error_and_case_drift() -> None:
    report = _full_report("core")
    report["records"][0]["runtime_errors"] = {
        "initial": "tool transport failed",
        "revised": None,
    }
    report["selected_case_ids"] = list(reversed(report["selected_case_ids"]))

    result = validate_full_report(
        report,
        suite="core",
        expected_total=10,
        policy_model=POLICY_MODEL,
        policy_backend=POLICY_BACKEND,
    )

    assert result["infrastructure_valid"] is False
    assert result["checks"]["selected_case_ids_exact"] is False
    assert any("tool transport failed" in row for row in result["infrastructure_anomalies"])


def test_reviewed_runtime_manifest_detects_any_frozen_field_change() -> None:
    snapshot = {name: f"value-{index}" for index, name in enumerate(SNAPSHOT_MANIFEST_FIELDS)}
    manifest = build_expected_manifest(snapshot)

    assert validate_snapshot_manifest(snapshot, manifest) == []

    changed = {**snapshot, "runtime_combined_sha256": "different"}
    assert validate_snapshot_manifest(changed, manifest) == ["runtime_combined_sha256"]


def test_wrapper_disables_current_and_legacy_langsmith_upload_paths() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    wrapper = (repo_root / "scripts" / "run_h006_full_agent_loop_eval.ps1").read_text(
        encoding="utf-8"
    )

    for name in (
        "LANGSMITH_TRACING",
        "LANGSMITH_TRACING_V2",
        "LANGCHAIN_TRACING",
        "LANGCHAIN_TRACING_V2",
    ):
        assert '[Environment]::SetEnvironmentVariable($name, "false", "Process")' in wrapper
        assert f'"{name}"' in wrapper
    for name in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        assert f'"{name}"' in wrapper
    assert '[Environment]::SetEnvironmentVariable($name, "", "Process")' in wrapper
    assert '"-p", "no:cacheprovider", "-W", "error"' in wrapper
    assert '$expectedRunId = "h006-full-agent-loop-v1-20260904-r2"' in wrapper
    assert "if ($RunId -ne $expectedRunId)" in wrapper


def test_reviewed_runtime_manifest_is_bound_to_run_id() -> None:
    snapshot = {name: f"value-{index}" for index, name in enumerate(SNAPSHOT_MANIFEST_FIELDS)}
    manifest = build_expected_manifest(snapshot)

    changed = {**snapshot, "run_id": "another-run"}

    assert manifest["run_id"] == snapshot["run_id"]
    assert validate_snapshot_manifest(changed, manifest) == ["run_id"]


def test_recovery_episode_sidecar_requires_exact_id_and_injected_observation(tmp_path) -> None:
    scenario_id = "falr-v1-test:run-1:421:0:initial"
    observation = ObservationEnvelope.failure(
        tool="search_pois",
        code="TOOL_TIMEOUT",
        message="injected",
        retryable=True,
        details={"fault_injection": True},
    )
    context = PolicyContext(
        trajectory_id="trajectory-1",
        goal_version=1,
        plan_version=1,
        original_request="plan a trip",
        current_subtask={"task_id": "research"},
        hard_constraints={},
        soft_preferences={},
        relevant_fact_refs=[],
        relevant_artifact_refs=[],
        failure_summary=[],
        remaining_tasks=1,
        remaining_steps=1,
        allowed_actions=["search_pois"],
    )
    candidate = EpisodeCandidate(
        scenario_id=scenario_id,
        source="teacher",
        template_family="native-react:recovery",
        city="上海",
        episode=AgentEpisode(
            trajectory_id="trajectory-1",
            environment_version="test",
            validator_version="v1",
            policy_name="checkpoint-32",
            policy_version="v1",
            initial_state={},
            steps=[
                TrajectoryStep(
                    step_index=0,
                    task_id="research",
                    context=context,
                    action=PolicyAction(action="search_pois"),
                    observations=[observation],
                    state_before_hash="before",
                    state_after_hash="after",
                )
            ],
        ),
    )
    sidecar = tmp_path / "recovery.jsonl"
    sidecar.write_text(candidate.model_dump_json() + "\n", encoding="utf-8")

    valid = validate_episode_sidecar(sidecar, [scenario_id], recovery=True)
    wrong_id = validate_episode_sidecar(sidecar, [scenario_id + "-wrong"], recovery=True)

    assert valid["valid"] is True
    assert valid["observations"] == 1
    assert wrong_id["valid"] is False


def test_recovery_sidecar_accepts_zero_step_explicit_model_policy_terminal(tmp_path) -> None:
    scenario_id = "falr-v1-test:run-1:421:0:initial"
    candidate = EpisodeCandidate(
        scenario_id=scenario_id,
        source="teacher",
        template_family="native-react:recovery",
        city="上海",
        episode=AgentEpisode(
            trajectory_id="trajectory-policy-failure",
            environment_version="test",
            validator_version="v1",
            policy_name="checkpoint-32",
            policy_version="v1",
            initial_state={},
            status="failed",
            termination_reason="policy_error_fallback",
            events=[
                AgentLoopEvent(
                    sequence=1,
                    event_type="episode_terminated",
                    payload={
                        "status": "failed",
                        "reason": "policy_error_fallback",
                        "error": "PolicyOutputError: emitted two tool calls",
                        "failure_class": "model_policy_failure",
                        "error_type": "PolicyOutputError",
                        "error_code": "TOOL_CALL_SHAPE_ERROR",
                        "policy_output_summary": {
                            "tool_call_count": 2,
                            "actions": ["search_pois", "ask_user"],
                        },
                    },
                )
            ],
        ),
    )
    sidecar = tmp_path / "recovery-policy-failure.jsonl"
    sidecar.write_text(candidate.model_dump_json() + "\n", encoding="utf-8")

    result = validate_episode_sidecar(sidecar, [scenario_id], recovery=True)

    assert result["valid"] is True
    assert result["checks"]["steps_or_explicit_policy_terminal"] is True
    assert result["checks"]["injection_or_explicit_policy_non_reach"] is True


def test_server_cleanup_requires_empty_group_port_and_gpu_recovery() -> None:
    clean = {
        "cleanup_passed": True,
        "process_group_empty": True,
        "remote_port_released": True,
        "gpu_memory_recovered": True,
    }

    assert validate_server_cleanup(clean)["valid"] is True

    clean["process_group_empty"] = False
    assert validate_server_cleanup(clean)["valid"] is False


def test_recovery_validation_accepts_policy_non_reach_but_rejects_injector_error() -> None:
    records = []
    recovery_cases = build_recovery_cases()
    for case in recovery_cases:
        for sample_index in range(4):
            intent, inference = _intent()
            records.append(
                {
                    "case_id": case.case_id,
                    "rollout_key": f"{case.case_id}::sample-{sample_index}",
                    "passed": True,
                    "failures": [],
                    "error": None,
                    "intent": intent,
                    "intent_inference": inference,
                    "recovery": {
                        "fault_injected": True,
                        "injection_status": "injected_once",
                    },
                }
            )
    report = {
        "schema_version": "full-agent-loop-recovery.v1",
        "scoring_contract_version": "recovery-injection-accounting.v2",
        "benchmark_hash": RECOVERY_BENCHMARK_SHA256,
        "selected_case_ids": [case.case_id for case in recovery_cases],
        "policy_model": POLICY_MODEL,
        "policy_backend": POLICY_BACKEND,
        "intent_model": INTENT_MODEL,
        "rollout_id": "run-1",
        "base_seed": 421,
        "group_size": 4,
        "temperature": 0.8,
        "summary": {
            "total": 32,
            "passed": 32,
            "pass_rate": 1.0,
            "first_try_recovery_rate": 1.0,
            "full_chain_pass_rate": 1.0,
            "injection_status_counts": {"injected_once": 32},
            "strata": {},
        },
        "records": records,
    }

    clean = validate_recovery_report(
        report,
        run_id="run-1",
        policy_model=POLICY_MODEL,
        policy_backend=POLICY_BACKEND,
    )
    assert clean["infrastructure_valid"] is True

    report["records"][0]["recovery"] = {
        "fault_injected": False,
        "injection_status": "not_reached_due_to_policy_failure",
    }
    report["records"][0]["model_policy_failures"] = {
        "initial": {"classification": "model_policy_failure"}
    }
    report["summary"]["injection_status_counts"] = {
        "injected_once": 31,
        "not_reached_due_to_policy_failure": 1,
    }
    policy_non_reach = validate_recovery_report(
        report,
        run_id="run-1",
        policy_model=POLICY_MODEL,
        policy_backend=POLICY_BACKEND,
    )
    assert policy_non_reach["infrastructure_valid"] is True

    report["records"][0]["recovery"] = {
        "fault_injected": False,
        "injection_status": "injector_error",
    }
    report["summary"]["injection_status_counts"] = {
        "injected_once": 31,
        "injector_error": 1,
    }
    invalid = validate_recovery_report(
        report,
        run_id="run-1",
        policy_model=POLICY_MODEL,
        policy_backend=POLICY_BACKEND,
    )
    assert invalid["infrastructure_valid"] is False
    assert invalid["checks"]["no_injector_error"] is False
