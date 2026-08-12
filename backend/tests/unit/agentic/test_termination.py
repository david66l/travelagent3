"""Tests for the global completion guard."""

from agentic.termination import CompletionGuard


def test_enforce_requires_validator_report():
    decision = CompletionGuard(mode="enforce").evaluate(None)

    assert decision.allowed is False
    assert decision.would_block is True
    assert decision.blocks[0].code == "VALIDATOR_NOT_RUN"


def test_shadow_reports_failed_constraints_without_blocking():
    decision = CompletionGuard(mode="shadow").evaluate(
        {
            "hard_pass": False,
            "hard_violations": [{"code": "OVER_BUDGET", "message": "budget exceeded"}],
        }
    )

    assert decision.allowed is True
    assert decision.would_block is True
    assert decision.blocks[0].details["violation_codes"] == ["OVER_BUDGET"]


def test_enforce_accepts_programmatic_hard_pass():
    decision = CompletionGuard(mode="enforce").evaluate({"hard_pass": True, "hard_violations": []})

    assert decision.allowed is True
    assert decision.would_block is False
    assert decision.validator_version == "travel-validator.v1"
