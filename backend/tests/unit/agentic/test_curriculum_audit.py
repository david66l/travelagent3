from agentic.grpo_training import GRPOCorpusRow
from agentic.policy import PolicyOutputError
from tests.unit.agentic.test_environment import _snapshot, _task

from scripts.audit_model_curriculum import (
    _route_and_actions,
    behavior_gate_metrics,
    boundary_stratum,
    decision_loop_behavior_metrics,
    decision_loop_metadata,
    paired_rollout_seed,
    rollout_latency_metrics,
    rollout_trl_history,
    select_boundary_stratified,
    select_stratified,
    select_verifier_repair_stratified,
    task_family,
)


def test_paired_rollout_seed_is_stable_and_sample_specific():
    assert paired_rollout_seed(44, task_id="task-a", sample_index=0) == paired_rollout_seed(
        44, task_id="task-a", sample_index=0
    )
    assert paired_rollout_seed(44, task_id="task-a", sample_index=0) != paired_rollout_seed(
        44, task_id="task-a", sample_index=1
    )


def _row(kind: str, index: int) -> GRPOCorpusRow:
    task = _task().model_copy(update={"task_id": f"{kind}-{index}"})
    snapshot = _snapshot().model_copy(update={"state_id": f"{kind}-state-{index}"})
    if kind == "clarification":
        task.missing_slots = ["budget"]
    elif kind == "tradeoff":
        task.feasibility_report = {"feasible": False}
    elif kind == "recovery":
        snapshot.tool_responses["search_pois"][0].data = None
        snapshot.tool_responses["search_pois"][0].data_source = "unavailable"
        snapshot.tool_responses["search_pois"][0].error_code = "UPSTREAM_TIMEOUT"
    return GRPOCorpusRow(task=task, snapshot=snapshot)


def test_task_family_and_stratified_selection():
    rows = [
        _row(kind, index)
        for kind in ("search", "clarification", "tradeoff", "recovery")
        for index in range(3)
    ]

    selected = select_stratified(rows, per_family=2)
    offset_selected = select_stratified(rows, per_family=1, offset_per_family=2)

    assert len(selected) == 8
    assert {row.task.task_id for row in offset_selected} == {
        "search-2",
        "clarification-2",
        "tradeoff-2",
        "recovery-2",
    }
    assert {task_family(row) for row in selected} == {
        "search",
        "clarification",
        "tradeoff",
        "recovery",
    }
    routes = {task_family(row): _route_and_actions(row) for row in selected}
    assert routes["search"] == ("search", ["search_pois"])
    assert routes["recovery"] == ("search", ["search_pois"])
    assert routes["clarification"] == ("clarification", ["ask_user"])
    assert routes["tradeoff"] == ("tradeoff", ["propose_tradeoff", "abort"])


def test_stratified_selection_can_be_pre_filtered_to_one_family():
    rows = [
        _row(kind, index)
        for kind in ("search", "clarification", "tradeoff", "recovery")
        for index in range(3)
    ]

    selected = select_stratified(
        [row for row in rows if task_family(row) == "clarification"],
        per_family=2,
    )

    assert [row.task.task_id for row in selected] == [
        "clarification-0",
        "clarification-1",
    ]


def test_verifier_repair_selection_supports_disjoint_target_offsets():
    rows = []
    for target in ("abort", "propose_tradeoff", "retry_solve"):
        for index in range(4):
            row = _row("search", index).model_copy(deep=True)
            row.task.task_id = f"{target}-{index}"
            row.snapshot.hidden_test_facts["grpo_decision_state"] = {
                "schema_version": "react-verifier-repair-decision.v1",
                "target_action": target,
            }
            rows.append(row)

    selected = select_verifier_repair_stratified(
        rows,
        per_target=2,
        offset_per_target=2,
    )

    assert {row.task.task_id for row in selected} == {
        "abort-2",
        "abort-3",
        "propose_tradeoff-2",
        "propose_tradeoff-3",
        "retry_solve-2",
        "retry_solve-3",
    }


def test_boundary_selection_is_balanced_by_status_and_actionability():
    rows = []
    for kind in ("infeasible", "unsafe", "missing_tool"):
        for variant in ("actionable_tradeoff", "necessary_abort"):
            for index in range(3):
                row = _row("tradeoff", index).model_copy(deep=True)
                row.task.task_id = f"{kind}-{variant}-{index}"
                row.snapshot.hidden_test_facts["decision_boundary_training"] = {
                    "boundary_kind": kind,
                    "variant": variant,
                }
                rows.append(row)

    selected = select_boundary_stratified(rows, per_cell=2)

    assert len(selected) == 12
    assert all(
        sum(boundary_stratum(row) == cell for row in selected) == 2
        for cell in {
            ("infeasible", "actionable_tradeoff"),
            ("infeasible", "necessary_abort"),
            ("unsafe", "actionable_tradeoff"),
            ("unsafe", "necessary_abort"),
            ("missing_tool", "actionable_tradeoff"),
            ("missing_tool", "necessary_abort"),
        }
    )


def test_decision_loop_metadata_and_breakdown_keep_orthogonal_factors_visible():
    row = _row("recovery", 7)
    row.task.slots["destination"] = "广州"
    row.snapshot.hidden_test_facts["decision_loop_curriculum"] = {
        "scenario": "change_arguments",
        "evidence_style": "diagnostic_evidence",
        "target_position": 1,
    }

    assert decision_loop_metadata(row) == {
        "scenario": "change_arguments",
        "evidence_style": "diagnostic_evidence",
        "target_position": 1,
    }

    successful = {
        "city": "广州",
        "decision_loop": decision_loop_metadata(row),
        "gate_status": "passed",
        "reward": 1,
        "actions": [{}, {}],
        "policy_errors": [],
    }
    failed = {
        **successful,
        "gate_status": "task_failed",
        "reward": -1,
        "actions": [{}],
        "policy_errors": [{"code": "POLICY_OUTPUT_ERROR"}],
    }

    metrics = decision_loop_behavior_metrics([successful, failed])

    assert metrics["scenario"]["change_arguments"] == {
        "rollouts": 2,
        "successful_rollouts": 1,
        "success_rate": 0.5,
        "mean_policy_actions": 1.5,
        "policy_error_rate": 0.5,
    }
    assert metrics["evidence_style"]["diagnostic_evidence"]["success_rate"] == 0.5
    assert metrics["target_position"]["1"]["success_rate"] == 0.5
    assert metrics["city"]["广州"]["success_rate"] == 0.5
    joint = "change_arguments|diagnostic_evidence|1"
    assert metrics["scenario_evidence_position"][joint]["success_rate"] == 0.5


def test_decision_loop_metadata_is_empty_for_non_stage3_rows():
    assert decision_loop_metadata(_row("search", 1)) == {}


class _InvalidHistoryPolicy:
    async def propose_from_history(self, messages, *, tools, allowed_actions):
        raise PolicyOutputError("invalid sampled completion")


async def test_trl_history_audit_scores_invalid_completion_as_failed_rollout():
    row = _row("search", 99)
    errors = []

    rollout = await rollout_trl_history(
        row,
        _InvalidHistoryPolicy(),
        policy_errors=errors,
    )

    assert rollout.episode.status == "failed"
    assert rollout.episode.termination_reason == "rollout_truncated"
    assert rollout.reward.gate_status == "task_failed"
    components = rollout.reward.components.model_dump(mode="json")
    assert set(components) == {
        "task",
        "constraint",
        "format",
        "tool",
        "grounding",
        "efficiency",
        "quality",
    }
    assert components["task"] == -1.0
    assert components["constraint"] == -1.0
    assert errors == [
        {
            "code": "POLICY_OUTPUT_ERROR",
            "message": "invalid sampled completion",
            "raw_output": None,
        }
    ]


def test_behavior_gate_reports_empty_and_invalid_policy_actions():
    metrics = behavior_gate_metrics(
        [
            {"gate_status": "passed", "reward": 0.9, "actions": [{"error_code": None}]},
            {"gate_status": "task_failed", "reward": -1, "actions": []},
            {
                "gate_status": "task_failed",
                "reward": -1,
                "actions": [{"error_code": "SNAPSHOT_ARGUMENT_MISMATCH"}],
            },
            {
                "gate_status": "task_failed",
                "reward": -1,
                "actions": [],
                "policy_errors": [
                    {
                        "code": "POLICY_ARGUMENT_INVALID",
                        "raw_output": (
                            '<tool_call>{"name":"search_pois","arguments":'
                            '{"trusted_city":"北京","max_results":10}}'
                            "</tool_call>"
                        ),
                    }
                ],
            },
        ]
    )

    assert metrics == {
        "rollouts": 4,
        "successful_rollouts": 1,
        "success_rate": 1 / 4,
        "empty_action_rollouts": 2,
        "empty_action_rate": 1 / 2,
        "invalid_actions": 1,
        "invalid_action_rate": 1 / 4,
        "policy_output_errors": 1,
        "policy_output_error_rate": 1 / 4,
        "policy_argument_errors": 1,
        "policy_argument_error_rate": 1 / 4,
        "unknown_argument_errors": 1,
        "unknown_argument_error_rate": 1 / 4,
        "protected_argument_errors": 1,
        "protected_argument_error_rate": 1 / 4,
    }


def test_rollout_latency_metrics_reports_interpolated_percentiles():
    metrics = rollout_latency_metrics(
        [
            {"latency_ms": 100.0},
            {"latency_ms": 200.0},
            {"latency_ms": 300.0},
            {"latency_ms": 400.0},
        ]
    )

    assert metrics == {
        "rollouts": 4,
        "mean_ms": 250.0,
        "p50_ms": 250.0,
        "p95_ms": 385.0,
    }
