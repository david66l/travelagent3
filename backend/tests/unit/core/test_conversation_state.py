"""Unit tests for ConversationState – P0 (split personal / trip profile)."""

from core.conversation_state import (
    default_conversation_state,
    merge_profile,
    append_message,
    is_profile_ready,
    flatten_profile,
    MAX_RECENT_MESSAGES,
)
from schemas import ProfilePatch


def _p(state):
    """Shorthand: profile portion of a full state dict."""
    return state["profile"]


def _trip(profile):
    return profile["trip"]


def _personal(profile):
    return profile["personal"]


class TestDefaultState:
    def test_schema_version(self):
        state = default_conversation_state()
        assert state["schema_version"] == 1

    def test_phase_is_gathering(self):
        assert default_conversation_state()["phase"] == "gathering"

    def test_profile_has_personal_and_trip(self):
        p = _p(default_conversation_state())
        assert "personal" in p
        assert "trip" in p

    def test_trip_fields_present(self):
        t = _trip(default_conversation_state()["profile"])
        assert t["destination"] is None
        assert t["travel_days"] is None
        assert t["travelers_count"] is None
        assert t["travelers_type"] is None
        assert isinstance(t["special_requests"], list)

    def test_personal_fields_present(self):
        per = _personal(default_conversation_state()["profile"])
        assert isinstance(per["interests"], list)
        assert isinstance(per["food_preferences"], list)
        assert isinstance(per["avoid"], list)
        assert per["pace"] == "moderate"

    def test_turn_starts_at_zero(self):
        assert default_conversation_state()["turn"] == 0

    def test_revision_starts_at_one(self):
        assert default_conversation_state()["revision"] == 1


class TestMergeProfile:
    # -- trip-scoped fields --

    def test_set_trip_scalar(self):
        profile = default_conversation_state()["profile"]
        patch = ProfilePatch(set={"destination": "成都", "travel_days": 4})
        merged = merge_profile(profile, patch)
        assert merged["trip"]["destination"] == "成都"
        assert merged["trip"]["travel_days"] == 4

    def test_overwrite_trip_budget(self):
        profile = default_conversation_state()["profile"]
        profile["trip"]["budget_range"] = 5000
        patch = ProfilePatch(set={"budget_range": 3000})
        merged = merge_profile(profile, patch)
        assert merged["trip"]["budget_range"] == 3000

    def test_none_value_in_set_is_ignored(self):
        profile = default_conversation_state()["profile"]
        profile["trip"]["destination"] = "成都"
        patch = ProfilePatch(set={"destination": None})
        merged = merge_profile(profile, patch)
        assert merged["trip"]["destination"] == "成都"  # unchanged

    # -- personal (list) fields --

    def test_add_to_personal_list(self):
        profile = default_conversation_state()["profile"]
        patch = ProfilePatch(add={"interests": ["自然", "历史"]})
        merged = merge_profile(profile, patch)
        assert "自然" in merged["personal"]["interests"]
        assert "历史" in merged["personal"]["interests"]

    def test_add_deduplicates(self):
        profile = default_conversation_state()["profile"]
        profile["personal"]["interests"] = ["自然"]
        patch = ProfilePatch(add={"interests": ["自然", "美食"]})
        merged = merge_profile(profile, patch)
        assert merged["personal"]["interests"] == ["自然", "美食"]

    def test_remove_from_personal_list(self):
        profile = default_conversation_state()["profile"]
        profile["personal"]["interests"] = ["自然", "历史", "美食"]
        patch = ProfilePatch(remove={"interests": ["历史"]})
        merged = merge_profile(profile, patch)
        assert merged["personal"]["interests"] == ["自然", "美食"]

    def test_clear_personal_field(self):
        profile = default_conversation_state()["profile"]
        profile["personal"]["interests"] = ["自然", "历史"]
        patch = ProfilePatch(clear=["interests"])
        merged = merge_profile(profile, patch)
        assert merged["personal"]["interests"] == []

    def test_avoid_list_works(self):
        profile = default_conversation_state()["profile"]
        patch = ProfilePatch(add={"avoid": ["博物馆", "爬山"]})
        merged = merge_profile(profile, patch)
        assert "博物馆" in merged["personal"]["avoid"]
        assert "爬山" in merged["personal"]["avoid"]

    # -- mix of trip and personal in one patch --

    def test_mixed_patch_routes_correctly(self):
        """budget → trip, food_preferences → personal, one patch."""
        profile = default_conversation_state()["profile"]
        patch = ProfilePatch(
            set={"budget_range": 8000},
            add={"food_preferences": ["火锅"]},
        )
        merged = merge_profile(profile, patch)
        assert merged["trip"]["budget_range"] == 8000
        assert "火锅" in merged["personal"]["food_preferences"]

    # -- schema migration: flat → nested --

    def test_flat_profile_migrates(self):
        """Old flat profile dict auto-migrates to personal/trip."""
        flat = {
            "destination": "北京",
            "travel_days": 3,
            "interests": ["历史"],
            "food_preferences": ["辣"],
        }
        merged = merge_profile(flat, ProfilePatch())
        assert merged["trip"]["destination"] == "北京"
        assert merged["personal"]["interests"] == ["历史"]


class TestFlattenProfile:
    def test_merges_personal_and_trip(self):
        profile = default_conversation_state()["profile"]
        profile["personal"]["food_preferences"] = ["辣"]
        profile["trip"]["destination"] = "北京"
        flat = flatten_profile(profile)
        assert flat["food_preferences"] == ["辣"]
        assert flat["destination"] == "北京"

    def test_trip_overrides_personal_on_conflict(self):
        profile = default_conversation_state()["profile"]
        profile["personal"]["pace"] = "moderate"
        # trip should never have 'pace', but if it does it wins
        profile["trip"]["pace"] = "intensive"  # shouldn't happen, but tested
        flat = flatten_profile(profile)
        assert flat["pace"] == "intensive"


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
        assert state["recent_messages"][0]["content"] == "msg5"
        assert state["recent_messages"][-1]["content"] == "msg14"


class TestIsProfileReady:
    def _complete_trip(self, **overrides):
        base = {
            "origin": "深圳",
            "destination": "成都",
            "travel_dates": "下周",
            "travel_days": 4,
            "travelers_count": 2,
            "travelers_type": "couple",
            "has_elderly": False,
            "has_children": False,
            "budget_range": 5000,
        }
        base.update(overrides)
        return base

    def test_not_ready_when_empty(self):
        assert not is_profile_ready(default_conversation_state()["profile"])

    def test_ready_when_all_required_present(self):
        profile = default_conversation_state()["profile"]
        profile["trip"].update(self._complete_trip())
        assert is_profile_ready(profile)

    def test_not_ready_without_days(self):
        profile = default_conversation_state()["profile"]
        profile["trip"].update(self._complete_trip(travel_days=None))
        del profile["trip"]["travel_days"]
        assert not is_profile_ready(profile)

    def test_ready_without_optional_child_flag(self):
        profile = default_conversation_state()["profile"]
        patch = self._complete_trip()
        patch.pop("has_children")
        profile["trip"].update(patch)
        assert is_profile_ready(profile)

    def test_flat_profile_still_works(self):
        """Old flat profile dict tolerated by is_profile_ready."""
        assert is_profile_ready(self._complete_trip())
