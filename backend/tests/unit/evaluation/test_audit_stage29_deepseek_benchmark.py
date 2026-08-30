from scripts.audit_stage29_deepseek_benchmark import EXPECTED_ACTIONS, audit_stage29
from scripts.build_stage29_deepseek_benchmark import (
    ACTIONS,
    _build_case,
    build_assignments,
)
from evaluation.external_benchmark import IndependentAnnotation


def test_stage29_audit_rejects_incomplete_fixture():
    assignment = build_assignments()[0]
    item = {
        **assignment,
        "case_id": assignment["assignment_id"],
        "request": "周末想找一个安静的室内展馆。",
        "request_family": "quiet_indoor",
        "constraint_kind": "indoor_quiet",
        "hard_constraint_description": "候选必须为安静室内场所。",
        "missing_information": [],
        "capability_status": "solvable",
        "actionable_alternatives": False,
        "alternatives": [],
        "search_keywords": ["安静室内展馆"],
        "failure_summary": [],
        "remaining_steps": 4,
    }
    # The full audit must fail a one-case fixture even though schema validation passes.
    annotations = [
        IndependentAnnotation(
            annotator_id=annotator,
            primary_action=assignment["target_action"],
            allowed_actions=list(ACTIONS),
        )
        for annotator in ("a", "b")
    ]
    case = _build_case(item, assignment["target_action"], annotations, None)

    report = audit_stage29([case], [], [])

    assert report["passed"] is False
    assert report["gates"]["total_150"] is False
    assert report["gates"]["forbidden_corpora_registered"] is False
    assert sum(EXPECTED_ACTIONS.values()) == 150
