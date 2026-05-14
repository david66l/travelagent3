"""Tests for graph routing and builder."""

import pytest
from graph.graph import (
    _is_vague_modification,
    route_after_intent,
    route_after_update,
    route_after_confirm,
    build_graph,
)
from core.state import ItineraryState


class TestIsVagueModification:
    """Test vague modification detection."""

    def test_vague_phrases(self):
        assert _is_vague_modification("继续修改") is True
        assert _is_vague_modification("不太满意") is True
        assert _is_vague_modification("再改改") is True

    def test_specific_keywords(self):
        assert _is_vague_modification("换掉第三个景点") is False
        assert _is_vague_modification("增加一天") is False
        assert _is_vague_modification("删除这个") is False

    def test_short_input_without_keyword(self):
        assert _is_vague_modification("可以") is True

    def test_neutral_input(self):
        assert _is_vague_modification("这是可以的") is False


class TestRouteAfterIntent:
    """Test intent routing."""

    def test_needs_clarification(self):
        state = ItineraryState(needs_clarification=True)
        assert route_after_intent(state) == "collect_info"

    def test_generate_itinerary(self):
        state = ItineraryState(intent="generate_itinerary")
        assert route_after_intent(state) == "prepare_context"

    def test_modify_itinerary_vague(self):
        state = ItineraryState(intent="modify_itinerary", user_input="继续修改")
        assert route_after_intent(state) == "ask_modification"

    def test_modify_itinerary_specific(self):
        state = ItineraryState(intent="modify_itinerary", user_input="换掉酒店")
        assert route_after_intent(state) == "prepare_context"

    def test_update_preferences(self):
        state = ItineraryState(intent="update_preferences")
        assert route_after_intent(state) == "update_prefs"

    def test_query_info(self):
        state = ItineraryState(intent="query_info")
        assert route_after_intent(state) == "qa"

    def test_confirm_itinerary(self):
        state = ItineraryState(intent="confirm_itinerary")
        assert route_after_intent(state) == "confirm"

    def test_view_history(self):
        state = ItineraryState(intent="view_history")
        assert route_after_intent(state) == "qa"

    def test_chitchat(self):
        state = ItineraryState(intent="chitchat")
        assert route_after_intent(state) == "qa"

    def test_unknown_intent(self):
        state = ItineraryState(intent="unknown")
        assert route_after_intent(state) == "qa"


class TestRouteAfterUpdate:
    """Test preference update routing."""

    def test_needs_replan(self):
        state = ItineraryState(needs_replan=True)
        assert route_after_update(state) == "prepare_context"

    def test_no_replan(self):
        state = ItineraryState(needs_replan=False)
        assert route_after_update(state) == "format_output"


class TestRouteAfterConfirm:
    """Test confirmation routing."""

    def test_always_save_memory(self):
        state = ItineraryState()
        assert route_after_confirm(state) == "save_memory"


class TestBuildGraph:
    """Test graph builder."""

    def test_builds_successfully(self):
        graph = build_graph()
        assert graph is not None

    def test_has_nodes(self):
        graph = build_graph()
        assert len(graph.nodes) > 0
