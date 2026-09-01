import pytest

from agentic.reason_quality import (
    build_grounded_repair_reason,
    verifier_reason_quality_checks,
)


EVIDENCE = "成都3日行程第1天，宽窄巷子尚未结束就与人民公园重叠45分钟。"
PHRASES = ["宽窄巷子尚未结束", "人民公园重叠45分钟"]


@pytest.mark.parametrize("target", ["retry_solve", "propose_tradeoff", "abort"])
def test_teacher_reason_passes_every_quality_gate(target):
    reason = build_grounded_repair_reason(EVIDENCE, target)

    checks = verifier_reason_quality_checks(
        reason=reason,
        target_action=target,
        grounding_phrases=PHRASES,
        evidence=EVIDENCE,
    )

    assert all(checks.values())
    assert not reason.startswith(EVIDENCE.rstrip("。"))
    assert reason.endswith(EVIDENCE)


def test_one_copied_phrase_cannot_pass_grounding_or_specificity():
    checks = verifier_reason_quality_checks(
        reason="宽窄巷子尚未结束，建议调整。",
        target_action="retry_solve",
        grounding_phrases=PHRASES,
        evidence=EVIDENCE,
    )

    assert checks["grounding_match"] is False
    assert checks["reason_specificity_match"] is False


def test_english_wrapper_with_copied_chinese_entities_fails_language_gate():
    checks = verifier_reason_quality_checks(
        reason=(
            "Verification failed because 宽窄巷子尚未结束 and "
            "人民公园重叠45分钟, so retry the solver."
        ),
        target_action="retry_solve",
        grounding_phrases=PHRASES,
        evidence=EVIDENCE,
    )

    assert checks["grounding_match"] is True
    assert checks["reason_language_match"] is False


def test_correct_evidence_without_action_rationale_is_not_full_quality():
    checks = verifier_reason_quality_checks(
        reason=EVIDENCE,
        target_action="retry_solve",
        grounding_phrases=PHRASES,
        evidence=EVIDENCE,
    )

    assert checks["grounding_match"] is True
    assert checks["reason_action_rationale_match"] is False


def test_internal_action_name_is_rejected_from_user_visible_reason():
    checks = verifier_reason_quality_checks(
        reason=f"{EVIDENCE} 因此调用 retry_solve 重算。",
        target_action="retry_solve",
        grounding_phrases=PHRASES,
        evidence=EVIDENCE,
    )

    assert checks["reason_public_language"] is False


def test_teacher_reason_has_deterministic_but_nonconstant_wording():
    first = build_grounded_repair_reason(EVIDENCE, "retry_solve")
    assert first == build_grounded_repair_reason(EVIDENCE, "retry_solve")
    variants = {
        build_grounded_repair_reason(
            f"成都3日行程第1天，活动A与活动B重叠{minutes}分钟。",
            "retry_solve",
        )
        for minutes in range(20, 80)
    }
    assert len({reason.split("：", 1)[0] for reason in variants}) >= 4
