from agentic.grpo_training import GRPOCorpusRow
from scripts.build_aligned_tradeoff_repair_dataset import (
    _split,
    _teacher_action,
    task_family,
)


def _row(index: int, *, feasible: bool = False, missing: list[str] | None = None):
    return GRPOCorpusRow(
        task={
            "task_id": f"task-{index}",
            "template_family": "test",
            "difficulty": "L3",
            "seed": index,
            "user_request": "请规划3天上海行程，预算3000元",
            "slots": {"destination": "上海", "travel_days": 3, "budget_range": 3000},
            "profile": {"interests": ["美食"]},
            "missing_slots": missing or [],
            "feasibility_report": {
                "feasible": feasible,
                "reasons": [] if feasible else ["预算不足以覆盖指定天数"],
            },
        },
        snapshot={
            "environment_version": "test-v1",
            "snapshot_version": "test-v1",
            "state_id": f"state-{index}",
            "tool_responses": {},
        },
    )


def test_tradeoff_teacher_matches_runtime_route_and_tools():
    row = _row(1)

    route, tools, arguments = _teacher_action(row)

    assert task_family(row) == "tradeoff"
    assert route == "tradeoff"
    assert tools == ["propose_tradeoff", "abort"]
    assert arguments["reason"] == "预算不足以覆盖指定天数"
    assert 1 < len(arguments["options"]) <= 3


def test_internal_split_is_stable_and_disjoint():
    rows = [_row(index) for index in range(40)]

    first = _split(rows)
    second = _split(list(reversed(rows)))

    assert {key: [row.task.task_id for row in value] for key, value in first.items()} == {
        key: [row.task.task_id for row in value] for key, value in second.items()
    }
    assert {key: len(value) for key, value in first.items()} == {
        "validation": 6,
        "test": 6,
        "train": 28,
    }
