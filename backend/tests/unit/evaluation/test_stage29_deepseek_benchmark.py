from collections import Counter

from scripts.build_stage29_deepseek_benchmark import (
    ACTION_COUNTS,
    _validate_authored,
    build_assignments,
    find_conflicts,
)


def test_stage29_plan_is_balanced_and_has_frozen_split_counts():
    assignments = build_assignments()

    assert len(assignments) == 150
    assert Counter(item["split"] for item in assignments) == {
        "sealed_test": 120,
        "dev": 30,
    }
    assert Counter(item["target_action"] for item in assignments) == ACTION_COUNTS
    assert len({item["assignment_id"] for item in assignments}) == 150


def test_authored_case_enforces_termination_boundary():
    assignment = next(item for item in build_assignments() if item["target_action"] == "abort")
    authored = {
        "assignment_id": assignment["assignment_id"],
        "request": "官方已关闭且没有替代入口，仍要求今天进入。",
        "request_family": "fixed_closure",
        "constraint_kind": "closure_no_alternative",
        "hard_constraint_description": "官方关闭，且没有安全合法替代方案。",
        "missing_information": [],
        "capability_status": "unsafe",
        "actionable_alternatives": False,
        "alternatives": [],
        "search_keywords": [],
    }

    result = _validate_authored(authored, assignment)

    assert result["author_target_action"] == "abort"
    assert result["actionable_alternatives"] is False


def test_conflicts_include_annotator_or_author_disagreement():
    case = {
        "case_id": "stage29-ds-001",
        "request": "预算冲突但可以减少一个可选项目。",
        "hard_constraint_description": "当前预算不能覆盖全部项目。",
        "capability_status": "infeasible",
        "actionable_alternatives": True,
        "alternatives": ["减少可选项目"],
        "missing_information": [],
        "failure_summary": [],
        "remaining_steps": 4,
        "author_target_action": "propose_tradeoff",
    }
    annotations_a = [
        {
            "case_id": case["case_id"],
            "primary_action": "propose_tradeoff",
            "confidence": "high",
            "reason": "有替代方案",
        }
    ]
    annotations_b = [
        {
            "case_id": case["case_id"],
            "primary_action": "abort",
            "confidence": "medium",
            "reason": "误判为不可行",
        }
    ]

    conflicts = find_conflicts([case], annotations_a, annotations_b)

    assert len(conflicts) == 1
    assert conflicts[0]["adjudication"] is None
