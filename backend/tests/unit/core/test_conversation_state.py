"""Unit tests for ConversationState – P0."""

import time
from core.conversation_state import (
    default_conversation_state,
    merge_profile,
    append_message,
    is_profile_ready,
    MAX_RECENT_MESSAGES,
)
from schemas import ProfilePatch


class TestDefaultState:
    def test_schema_version(self):
        state = default_conversation_state()
        assert state["schema_version"] == 1

    def test_phase_is_gathering(self):
        assert default_conversation_state()["phase"] == "gathering"

    def test_profile_all_fields_present(self):
        p = default_conversation_state()["profile"]
        assert "destination" in p
        assert "travel_days" in p
        assert "interests" in p
        assert "avoid" in p
        assert isinstance(p["interests"], list)
        assert p["destination"] is None

    def test_turn_starts_at_zero(self):
        assert default_conversation_state()["turn"] == 0

    def test_revision_starts_at_one(self):
        assert default_conversation_state()["revision"] == 1


class TestMergeProfile:
    def test_set_scalar(self):
        profile = default_conversation_state()["profile"]
        patch = ProfilePatch(set={"destination": "成都", "travel_days": 4})
        merged = merge_profile(profile, patch)
        assert merged["destination"] == "成都"
        assert merged["travel_days"] == 4

    def test_add_to_list(self):
        profile = default_conversation_state()["profile"]
        patch = ProfilePatch(add={"interests": ["自然", "历史"]})
        merged = merge_profile(profile, patch)
        assert "自然" in merged["interests"]
        assert "历史" in merged["interests"]

    def test_add_deduplicates(self):
        profile = default_conversation_state()["profile"]
        profile["interests"] = ["自然"]
        patch = ProfilePatch(add={"interests": ["自然", "美食"]})
        merged = merge_profile(profile, patch)
        assert merged["interests"] == ["自然", "美食"]

    def test_remove_from_list(self):
        profile = default_conversation_state()["profile"]
        profile["interests"] = ["自然", "历史", "美食"]
        patch = ProfilePatch(remove={"interests": ["历史"]})
        merged = merge_profile(profile, patch)
        assert merged["interests"] == ["自然", "美食"]

    def test_clear_field(self):
        profile = default_conversation_state()["profile"]
        profile["interests"] = ["自然", "历史"]
        patch = ProfilePatch(clear=["interests"])
        merged = merge_profile(profile, patch)
        assert merged["interests"] == []

    def test_overwrite_budget(self):
        profile = default_conversation_state()["profile"]
        profile["budget_range"] = 5000
        # User says "budget 3000" → set
        patch = ProfilePatch(set={"budget_range": 3000})
        merged = merge_profile(profile, patch)
        assert merged["budget_range"] == 3000

    def test_avoid_list_works(self):
        profile = default_conversation_state()["profile"]
        patch = ProfilePatch(add={"avoid": ["博物馆", "爬山"]})
        merged = merge_profile(profile, patch)
        assert "博物馆" in merged["avoid"]
        assert "爬山" in merged["avoid"]

    def test_remove_from_avoid(self):
        profile = default_conversation_state()["profile"]
        profile["avoid"] = ["博物馆", "爬山"]
        patch = ProfilePatch(remove={"avoid": ["爬山"]})
        merged = merge_profile(profile, patch)
        assert merged["avoid"] == ["博物馆"]

    def test_none_value_in_set_is_ignored(self):
        profile = default_conversation_state()["profile"]
        profile["destination"] = "成都"
        patch = ProfilePatch(set={"destination": None})
        merged = merge_profile(profile, patch)
        assert merged["destination"] == "成都"  # unchanged


class TestAppendMessage:
    def test_appends_message(self):
        state = default_conversation_state()
        append_message(state, "user", "成都4天")
        assert len(state["recent_messages"]) == 1
        assert state["recent_messages"][0]["role"] == "user"
        assert "ts" in state["recent_messages"][0]

    def test_truncates_to_max(self):
        state = default_conversation_state()
        for i in range(15):
            append_message(state, "user", f"msg{i}")
        assert len(state["recent_messages"]) == MAX_RECENT_MESSAGES
        # Should keep the last 10
        assert state["recent_messages"][0]["content"] == "msg5"
        assert state["recent_messages"][-1]["content"] == "msg14"


class TestIsProfileReady:
    def test_not_ready_when_empty(self):
        assert not is_profile_ready(default_conversation_state()["profile"])

    def test_ready_when_destination_and_days_present(self):
        profile = {"destination": "成都", "travel_days": 4}
        assert is_profile_ready(profile)

    def test_not_ready_without_days(self):
        profile = {"destination": "成都", "travel_days": None}
        assert not is_profile_ready(profile)
