"""Tests for version-bound human approval of resumable drafts."""

from datetime import UTC, datetime, timedelta

import pytest

from graph.approval import (
    ApprovalValidationError,
    issue_pending_approval,
    public_approval,
    validate_pending_approval,
)


def _state() -> dict:
    return {
        "user_id": "user-1",
        "itinerary": [
            {
                "day_number": 1,
                "activities": [{"poi_name": "故宫", "start_time": "09:00"}],
            }
        ],
        "agent_ledger": {
            "goal": {"goal_version": 3},
            "task_graph": {"plan_version": 5},
        },
    }


def test_pending_approval_binds_user_versions_and_exact_itinerary():
    now = datetime(2026, 8, 29, 8, tzinfo=UTC)
    state = _state()
    pending = issue_pending_approval(state, now=now)
    state["pending_approval"] = pending

    validated = validate_pending_approval(
        state,
        public_approval(pending),
        action="confirm",
        user_id="user-1",
        now=now + timedelta(minutes=1),
    )

    assert validated["approval_id"] == pending["approval_id"]
    assert validated["goal_version"] == 3
    assert validated["plan_version"] == 5
    assert "user_id" not in public_approval(pending)


def test_pending_approval_rejects_changed_itinerary():
    now = datetime(2026, 8, 29, 8, tzinfo=UTC)
    state = _state()
    pending = issue_pending_approval(state, now=now)
    state["pending_approval"] = pending
    state["itinerary"][0]["activities"][0]["start_time"] = "10:00"

    with pytest.raises(ApprovalValidationError, match="已发生变化") as exc:
        validate_pending_approval(
            state,
            public_approval(pending),
            action="confirm",
            user_id="user-1",
            now=now + timedelta(minutes=1),
        )

    assert exc.value.code == "APPROVAL_STALE"


def test_pending_approval_rejects_expired_or_cross_user_request():
    now = datetime(2026, 8, 29, 8, tzinfo=UTC)
    state = _state()
    pending = issue_pending_approval(state, now=now, ttl=timedelta(minutes=5))
    state["pending_approval"] = pending

    with pytest.raises(ApprovalValidationError) as cross_user:
        validate_pending_approval(
            state,
            public_approval(pending),
            action="modify",
            user_id="user-2",
            now=now + timedelta(minutes=1),
        )
    assert cross_user.value.code == "APPROVAL_USER_MISMATCH"

    with pytest.raises(ApprovalValidationError) as expired:
        validate_pending_approval(
            state,
            public_approval(pending),
            action="confirm",
            user_id="user-1",
            now=now + timedelta(minutes=6),
        )
    assert expired.value.code == "APPROVAL_EXPIRED"
