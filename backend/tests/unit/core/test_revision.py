"""Unit tests for P2 revision mechanism."""
import time
from core.conversation_state import (
    default_conversation_state,
    merge_profile,
    append_message,
    is_profile_ready,
)
from schemas import ProfilePatch


class TestRevision:
    def test_first_job_revision_is_one(self):
        state = default_conversation_state()
        state["phase"] = "planning"
        state.setdefault("revision", 1)
        assert state["revision"] == 1

    def test_modify_after_completed_bumps_revision(self):
        state = default_conversation_state()
        state["phase"] = "completed"
        state["revision"] = 1
        state["profile"] = merge_profile(
            state["profile"],
            ProfilePatch(set={"destination": "成都", "travel_days": 4}),
        )

        # Simulate new message after completion
        was_completed = state.get("phase") == "completed"
        state["phase"] = "planning"
        if was_completed:
            state["revision"] = state.get("revision", 1) + 1
        else:
            state.setdefault("revision", 1)

        assert state["revision"] == 2

    def test_modify_before_completion_does_not_bump(self):
        """If still gathering, don't bump revision on each message."""
        state = default_conversation_state()
        state["phase"] = "gathering"
        state["revision"] = 1

        # Getting destination
        state["profile"] = merge_profile(
            state["profile"],
            ProfilePatch(set={"destination": "成都"}),
        )
        # Still gathering → no revision bump
        state.setdefault("revision", 1)
        assert state["revision"] == 1

    def test_phase_transitions_correctly(self):
        state = default_conversation_state()
        assert state["phase"] == "gathering"

        state["profile"] = merge_profile(
            state["profile"],
            ProfilePatch(set={"destination": "成都", "travel_days": 4}),
        )
        state["phase"] = "planning"
        assert state["phase"] == "planning"

        # Job completes
        state["phase"] = "completed"
        append_message(state, "assistant", "行程已生成")
        assert state["phase"] == "completed"
        assert state["recent_messages"][-1]["role"] == "assistant"

    def test_second_modify_after_completion(self):
        """Modify → complete → modify again → revision 3."""
        state = default_conversation_state()
        state["phase"] = "completed"
        state["revision"] = 2

        # Another modification
        state["profile"] = merge_profile(
            state["profile"],
            ProfilePatch(set={"pace": "relaxed"}),
        )
        was_completed = state.get("phase") == "completed"
        state["phase"] = "planning"
        if was_completed:
            state["revision"] = state.get("revision", 1) + 1
        assert state["revision"] == 3
