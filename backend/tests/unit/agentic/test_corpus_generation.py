"""Curriculum fixtures must be valid and executable, not stitched transcripts."""

import pytest

from agentic.loop import PolicyAction

from agentic.corpus_generation import CurriculumTeacherPolicy, build_curriculum_case
from agentic.environment import TravelAgentEnvironment
from agentic.policy import ControllerFirstPolicy
from agentic.trajectory import EpisodeReplayVerifier
from scripts.generate_agentic_corpus import generate


async def test_curriculum_case_executes_through_production_loop():
    task, snapshot = build_curriculum_case(19)

    rollout = await TravelAgentEnvironment(task, snapshot).rollout(
        ControllerFirstPolicy(CurriculumTeacherPolicy())
    )

    assert rollout.episode.termination_reason == "awaiting_user"
    assert EpisodeReplayVerifier().verify(rollout.episode) == []
    assert rollout.reward.gate_status == "passed"
    assert rollout.reward.episode_reward > 0
    assert len(rollout.episode.steps) == 11
    policy_steps = [
        step for step in rollout.episode.steps if step.action.decision_source == "policy"
    ]
    assert [step.task_id for step in policy_steps] == [
        "search_candidates",
        "search_candidates",
    ]


async def test_three_day_curriculum_detail_snapshot_matches_plannable_candidates():
    task, snapshot = build_curriculum_case(2)

    rollout = await TravelAgentEnvironment(task, snapshot).rollout(
        ControllerFirstPolicy(CurriculumTeacherPolicy())
    )

    detail_step = next(
        step for step in rollout.episode.steps if step.action.action == "get_poi_detail"
    )
    assert task.slots["travel_days"] == 3
    assert detail_step.verification["task_status"] == "succeeded"
    assert all(observation.ok for observation in detail_step.observations)
    assert rollout.episode.termination_reason == "awaiting_user"
    assert rollout.reward.gate_status == "passed"


async def test_policy_driven_teacher_owns_every_nominal_dag_action():
    task, snapshot = build_curriculum_case(0)

    rollout = await TravelAgentEnvironment(task, snapshot).rollout(CurriculumTeacherPolicy())

    assert rollout.reward.gate_status == "passed"
    assert [step.action.action for step in rollout.episode.steps] == [
        "capability_check",
        "get_weather",
        "search_pois",
        "accept_candidates",
        "get_poi_detail",
        "get_route_matrix",
        "solve_itinerary",
        "validate_itinerary",
        "accept_itinerary",
        "compose_draft",
        "finish",
    ]
    assert all(step.action.decision_source == "policy" for step in rollout.episode.steps)


async def test_generator_respects_non_overlapping_start_index():
    candidates, metadata = await generate(
        2,
        2,
        start_index=900,
        execution_mode="policy_driven",
    )

    assert [candidate.scenario_id for candidate in candidates] == [
        build_curriculum_case(900)[0].task_id,
        build_curriculum_case(901)[0].task_id,
    ]
    assert metadata["start_index"] == 900
    assert metadata["stop_index_exclusive"] == 902


async def test_curriculum_covers_clarification_tradeoff_and_retry():
    outcomes = {}
    for index, name in ((6, "missing"), (7, "retry"), (8, "infeasible")):
        task, snapshot = build_curriculum_case(index)
        rollout = await TravelAgentEnvironment(task, snapshot).rollout(
            ControllerFirstPolicy(CurriculumTeacherPolicy())
        )
        outcomes[name] = rollout

    assert outcomes["missing"].episode.steps[-1].action.action == "ask_user"
    assert outcomes["missing"].episode.termination_reason == "awaiting_user"
    assert outcomes["infeasible"].episode.steps[-1].action.action == "propose_tradeoff"
    assert outcomes["infeasible"].reward.gate_status == "passed"
    search_steps = [
        step for step in outcomes["retry"].episode.steps if step.action.action == "search_pois"
    ]
    assert len(search_steps) == 2
    assert search_steps[0].observations[0].ok is False
    assert search_steps[1].observations[0].ok is True


async def test_necessary_abort_is_visible_and_wrong_tradeoff_fails_verifier():
    task, snapshot = build_curriculum_case(8)
    task.task_id += "-necessary-abort"
    task.user_request += " 所有约束均不可调整，也不要提供替代方案。"
    task.feasibility_report.update(
        {
            "status": "unsafe",
            "actionable_alternatives": False,
            "alternatives": [],
            "reasons": ["硬约束互相冲突且用户拒绝全部放宽"],
        }
    )

    successful = await TravelAgentEnvironment(task, snapshot).rollout(
        ControllerFirstPolicy(CurriculumTeacherPolicy())
    )

    class WrongTradeoffPolicy(CurriculumTeacherPolicy):
        async def propose(self, context):
            if context.capability.get("status") in {"infeasible", "unsafe", "missing_tool"}:
                return PolicyAction(
                    action="propose_tradeoff",
                    arguments={"reason": "尝试放宽约束", "options": ["调整要求"]},
                )
            return await super().propose(context)

    rejected = await TravelAgentEnvironment(task, snapshot).rollout(
        ControllerFirstPolicy(WrongTradeoffPolicy())
    )

    assert successful.episode.steps[-1].context.capability["actionable_alternatives"] is False
    assert successful.episode.steps[-1].context.capability["status"] == "unsafe"
    assert len(successful.episode.steps) == 1
    assert successful.episode.steps[-1].action.action == "abort"
    assert successful.reward.gate_status == "passed"
    assert rejected.episode.steps[-1].action.action == "propose_tradeoff"
    assert rejected.reward.gate_status == "task_failed"
    assert rejected.reward.audit_metrics["termination_action_mismatch"] is True
    assert "TERMINATION_ACTION_MISMATCH" in rejected.reward.gate_reasons


@pytest.mark.parametrize(
    ("actionable", "action", "expected_gate"),
    [
        (True, "propose_tradeoff", "passed"),
        (True, "abort", "task_failed"),
        (False, "abort", "passed"),
        (False, "propose_tradeoff", "task_failed"),
    ],
)
async def test_terminal_decision_reward_covers_all_four_contract_quadrants(
    actionable,
    action,
    expected_gate,
):
    task, snapshot = build_curriculum_case(8)
    task.task_id += f"-quadrant-{actionable}-{action}"
    evidence = "预算和行程天数均已锁定，当前要求无法同时满足"
    alternatives = ["提高预算", "减少行程天数"] if actionable else []
    task.feasibility_report.update(
        {
            "status": "infeasible",
            "reasons": [evidence],
            "actionable_alternatives": actionable,
            "alternatives": alternatives,
        }
    )

    class FixedTerminalPolicy(CurriculumTeacherPolicy):
        async def propose(self, context):
            if context.capability.get("status") == "infeasible":
                return PolicyAction(
                    action=action,
                    arguments=(
                        {"reason": evidence, "options": alternatives}
                        if action == "propose_tradeoff"
                        else {"reason": evidence}
                    ),
                )
            return await super().propose(context)

    rollout = await TravelAgentEnvironment(task, snapshot).rollout(
        ControllerFirstPolicy(FixedTerminalPolicy())
    )

    assert rollout.reward.gate_status == expected_gate
    assert rollout.reward.reward_config_version == "hierarchical-b0.v2"
    assert rollout.reward.audit_metrics["termination_contract_incomplete"] is False
    assert rollout.reward.audit_metrics["termination_argument_mismatch"] is (
        not actionable and action == "propose_tradeoff"
    )
    assert rollout.reward.audit_metrics["termination_action_mismatch"] is (
        (actionable and action == "abort") or (not actionable and action == "propose_tradeoff")
    )


async def test_generic_tradeoff_text_cannot_hack_terminal_reward():
    task, snapshot = build_curriculum_case(8)

    class GenericTradeoffPolicy(CurriculumTeacherPolicy):
        async def propose(self, context):
            if context.capability.get("status") == "infeasible":
                return PolicyAction(
                    action="propose_tradeoff",
                    arguments={"reason": "需要调整", "options": ["调整要求"]},
                )
            return await super().propose(context)

    rollout = await TravelAgentEnvironment(task, snapshot).rollout(
        ControllerFirstPolicy(GenericTradeoffPolicy())
    )

    assert rollout.episode.steps[-1].action.action == "propose_tradeoff"
    assert rollout.reward.gate_status == "task_failed"
    assert rollout.reward.audit_metrics["termination_action_mismatch"] is False
    assert rollout.reward.audit_metrics["termination_argument_mismatch"] is True
    assert "TERMINATION_ARGUMENT_MISMATCH" in rollout.reward.gate_reasons
    assert rollout.reward.turn_rewards[-1].grounding == -1
    assert "ARGUMENT_NOT_GROUNDED" in rollout.reward.turn_rewards[-1].signals
