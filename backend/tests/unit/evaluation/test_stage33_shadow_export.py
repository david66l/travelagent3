from agentic.loop import (
    PolicyAction,
    PolicyContext,
    PolicyRouteTrace,
    PolicyShadowTrace,
)
from agentic.trajectory import AgentEpisode, TrajectoryStep
from scripts.export_stage33_policy_shadow_observations import export_observations


def test_export_shadow_observations_omits_context_and_blocks_release_use():
    context = PolicyContext(
        trajectory_id="trajectory-private",
        goal_version=1,
        plan_version=1,
        original_request="private trip request",
        current_subtask={"task_id": "search"},
        hard_constraints={},
        soft_preferences={},
        relevant_fact_refs=[],
        relevant_artifact_refs=[],
        failure_summary=[],
        remaining_tasks=1,
        remaining_steps=2,
        allowed_actions=["search_pois", "abort"],
    )
    action = PolicyAction(
        action="search_pois",
        route_trace=PolicyRouteTrace(
            requested_target="student",
            executed_target="student",
            family="search",
            reason="bounded curriculum",
        ),
        shadow_trace=PolicyShadowTrace(
            candidate_model="dpo-shadow",
            status="completed",
            action="abort",
        ),
    )
    episode = AgentEpisode(
        trajectory_id="trajectory-private",
        environment_version="test",
        validator_version="test",
        policy_name="champion",
        policy_version="sft",
        initial_state={},
        steps=[
            TrajectoryStep(
                step_index=0,
                task_id="search",
                context=context,
                action=action,
                state_before_hash="before",
                state_after_hash="after",
            )
        ],
    )

    rows, manifest = export_observations([episode.model_dump(mode="json")])

    assert len(rows) == 1
    assert rows[0]["champion_action"] == "search_pois"
    assert rows[0]["challenger_action"] == "abort"
    assert rows[0]["action_divergent"] is True
    assert rows[0]["challenger_outcome_observed"] is False
    assert "private trip request" not in str(rows)
    assert "trajectory-private" not in str(rows)
    assert manifest["release_gate_eligible"] is False
    assert manifest["action_divergences"] == 1
