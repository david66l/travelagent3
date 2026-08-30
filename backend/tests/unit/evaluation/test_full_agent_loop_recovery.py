from __future__ import annotations

import pytest

from agentic.loop import ActionOutcome, PolicyAction
from agentic.state import TaskNode
from evaluation.full_agent_loop_recovery import (
    OneShotFaultExecutor,
    build_recovery_cases,
    score_recovery,
)


class SuccessfulExecutor:
    async def execute(self, *, task, action, ledger) -> ActionOutcome:
        return ActionOutcome()


@pytest.mark.asyncio
async def test_one_shot_fault_fails_once_then_delegates():
    case = build_recovery_cases()[0]
    executor = OneShotFaultExecutor(case.fault, SuccessfulExecutor())
    task = TaskNode(
        task_id="research_evidence",
        goal="research",
        allowed_actions=("search_pois",),
    )
    first = await executor.execute(
        task=task,
        action=PolicyAction(
            action="search_pois",
            arguments={"keywords": ["历史文化", "美食"]},
        ),
        ledger=None,
    )
    second = await executor.execute(
        task=task,
        action=PolicyAction(
            action="search_pois",
            arguments={"keywords": ["历史文化"]},
        ),
        ledger=None,
    )

    assert first.status == "failed"
    assert first.error_code == "QUERY_TOO_BROAD"
    assert first.observations[0].error.details["fault_injection"] is True
    assert second.status == "completed"
    assert [row["injected"] for row in executor.trace] == [True, False]


def test_score_recovery_requires_direct_first_try_and_full_chain_pass():
    case = build_recovery_cases()[0]
    trace = [
        {
            "index": 0,
            "action": "search_pois",
            "arguments": {"keywords": ["历史", "文化", "美食"]},
            "injected": True,
            "repair_attempts": 0,
        },
        {
            "index": 1,
            "action": "search_pois",
            "arguments": {"keywords": ["历史", "文化"]},
            "injected": False,
            "repair_attempts": 0,
        },
    ]

    result = score_recovery(case, {"passed": True, "failures": []}, trace)

    assert result["passed"] is True
    assert result["recovery"]["first_try_recovery"] is True
    assert result["recovery"]["full_chain_passed"] is True


def test_score_retry_rejects_changed_arguments_and_policy_repair():
    case = build_recovery_cases()[4]
    trace = [
        {
            "index": 0,
            "action": "search_pois",
            "arguments": {"keywords": ["历史建筑", "城市公园"]},
            "injected": True,
            "repair_attempts": 0,
        },
        {
            "index": 1,
            "action": "search_pois",
            "arguments": {"keywords": ["历史建筑"]},
            "injected": False,
            "repair_attempts": 1,
        },
    ]

    result = score_recovery(case, {"passed": True, "failures": []}, trace)

    assert result["passed"] is False
    assert "RECOVERY_ARGUMENT_RETRY_INCORRECT" in result["failures"]
    assert "RECOVERY_REQUIRED_POLICY_REPAIR" in result["failures"]


def test_score_retry_waives_the_single_expected_full_chain_repeat():
    case = build_recovery_cases()[4]
    arguments = {"keywords": ["历史建筑", "城市公园"]}
    trace = [
        {
            "index": 0,
            "action": "search_pois",
            "arguments": arguments,
            "injected": True,
            "repair_attempts": 0,
        },
        {
            "index": 1,
            "action": "search_pois",
            "arguments": arguments,
            "injected": False,
            "repair_attempts": 0,
        },
    ]
    base = {
        "passed": False,
        "failures": ["EXACT_POLICY_ACTION_REPEAT"],
        "actions": [
            {"action": "search_pois", "arguments": arguments, "source": "policy"},
            {"action": "search_pois", "arguments": arguments, "source": "policy"},
        ],
    }

    result = score_recovery(case, base, trace)

    assert result["passed"] is True
    assert result["base_passed"] is True
    assert result["waived_base_failures"] == ["EXACT_POLICY_ACTION_REPEAT"]


def test_recovery_benchmark_balances_four_semantic_strata():
    cases = build_recovery_cases()
    strata = {(case.fault.scenario, case.fault.evidence_style) for case in cases}

    assert len(cases) == 8
    assert strata == {
        ("change_arguments", "explicit_instruction"),
        ("change_arguments", "diagnostic_evidence"),
        ("retry_same_arguments", "explicit_instruction"),
        ("retry_same_arguments", "diagnostic_evidence"),
    }
