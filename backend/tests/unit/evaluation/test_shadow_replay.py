import json

import pytest

from evaluation.shadow_replay import (
    AuthorizedReplayCase,
    load_authorized_replay_cases,
    replay_case_state,
    replay_scenario_id,
)


def _case(case_id: str = "case-001") -> AuthorizedReplayCase:
    return AuthorizedReplayCase(
        case_id=case_id,
        destination="上海",
        start_date="2026-09-01",
        end_date="2026-09-02",
        travel_days=2,
        budget=3000,
        interests=["历史", "美食"],
    )


def test_replay_scenario_id_is_deterministic_and_batch_scoped():
    first = replay_scenario_id(deployment_id="stage31", batch_id="batch-a", case_id="case-001")
    second = replay_scenario_id(deployment_id="stage31", batch_id="batch-a", case_id="case-001")
    other = replay_scenario_id(deployment_id="stage31", batch_id="batch-b", case_id="case-001")

    assert first == second
    assert first != other
    assert len(first) <= 64


def test_replay_case_state_is_solvable_and_contains_no_identity_fields():
    state = replay_case_state(_case())

    assert state["feasibility_report"]["status"] == "solvable"
    assert state["slots"]["destination"] == "上海"
    assert "user_id" not in state
    assert "session_id" not in state


def test_replay_case_rejects_date_range_mismatch():
    with pytest.raises(ValueError, match="date range"):
        AuthorizedReplayCase(
            case_id="case-001",
            destination="上海",
            start_date="2026-09-01",
            end_date="2026-09-03",
            travel_days=2,
            budget=3000,
            interests=["历史"],
        )


def test_replay_loader_rejects_duplicate_case_ids(tmp_path):
    path = tmp_path / "cases.jsonl"
    row = json.dumps(_case().model_dump(mode="json"), ensure_ascii=False)
    path.write_text(f"{row}\n{row}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unique"):
        load_authorized_replay_cases(path)
