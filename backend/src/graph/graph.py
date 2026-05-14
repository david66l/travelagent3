"""LangGraph StateGraph builder - orchestrates all nodes with conditional edges."""

from langgraph.graph import StateGraph, END

from core.state import ItineraryState
from graph.nodes import (
    intent_node,
    collect_info_node,
    prepare_context_node,
    poi_search_node,
    weather_node,
    budget_init_node,
    context_enrichment_node,
    planner_node,
    validation_node,
    route_node,
    apply_routes_node,
    budget_calc_node,
    proposal_node,
    update_prefs_node,
    qa_node,
    confirm_node,
    save_memory_node,
    format_output_node,
    ask_modification_node,
)


# ===== Conditional Routing Functions =====


def _is_vague_modification(user_input: str) -> bool:
    """Check if the modification request is vague (no specific change mentioned)."""
    user_input = user_input.strip()
    # Definitely vague phrases
    vague_phrases = ["继续修改", "再改", "再改改", "不太满意", "不满意", "再调整", "再优化"]
    if any(p in user_input for p in vague_phrases):
        return True
    # Specific change keywords make it non-vague
    specific_keywords = [
        "换",
        "改",
        "增加",
        "添加",
        "删掉",
        "去掉",
        "调整",
        "改为",
        "换成",
        "删除",
        "减少",
        "延长",
        "缩短",
    ]
    if any(kw in user_input for kw in specific_keywords):
        return False
    # Very short input without any specific action is vague
    if len(user_input) <= 4:
        return True
    return False


def route_after_intent(state: ItineraryState) -> str:
    """Route to next node based on intent and clarification needs."""
    if state.get("needs_clarification"):
        return "collect_info"

    intent = state.get("intent")

    # Vague modification requests should ask user for specifics first
    if intent == "modify_itinerary" and _is_vague_modification(state.get("user_input", "")):
        return "ask_modification"

    routing_map = {
        "generate_itinerary": "prepare_context",
        "modify_itinerary": "prepare_context",
        "update_preferences": "update_prefs",
        "query_info": "qa",
        "confirm_itinerary": "confirm",
        "view_history": "qa",
        "chitchat": "qa",
    }
    return routing_map.get(intent, "qa")


def route_after_update(state: ItineraryState) -> str:
    """After preference update, either replan or respond directly."""
    if state.get("needs_replan"):
        return "prepare_context"
    return "format_output"


def route_after_confirm(state: ItineraryState) -> str:
    """After confirmation, save memory and format output."""
    return "save_memory"


# ===== Graph Builder =====


def build_graph() -> StateGraph:
    """Build and return the StateGraph."""
    builder = StateGraph(ItineraryState)

    # --- Register all nodes ---
    builder.add_node("intent_node", intent_node)
    builder.add_node("collect_info_node", collect_info_node)
    builder.add_node("prepare_context_node", prepare_context_node)
    builder.add_node("poi_search_node", poi_search_node)
    builder.add_node("weather_node", weather_node)
    builder.add_node("budget_init_node", budget_init_node)
    builder.add_node("context_enrichment_node", context_enrichment_node)
    builder.add_node("planner_node", planner_node)
    builder.add_node("validation_node", validation_node)
    builder.add_node("route_node", route_node)
    builder.add_node("apply_routes_node", apply_routes_node)
    builder.add_node("budget_calc_node", budget_calc_node)
    builder.add_node("proposal_node", proposal_node)
    builder.add_node("update_prefs_node", update_prefs_node)
    builder.add_node("qa_node", qa_node)
    builder.add_node("confirm_node", confirm_node)
    builder.add_node("save_memory_node", save_memory_node)
    builder.add_node("format_output_node", format_output_node)
    builder.add_node("ask_modification_node", ask_modification_node)

    # --- Entry point ---
    builder.set_entry_point("intent_node")

    # --- Intent routing (conditional) ---
    builder.add_conditional_edges(
        "intent_node",
        route_after_intent,
        {
            "collect_info": "collect_info_node",
            "prepare_context": "prepare_context_node",
            "update_prefs": "update_prefs_node",
            "qa": "qa_node",
            "confirm": "confirm_node",
            "ask_modification": "ask_modification_node",
        },
    )

    # --- Clarification path ---
    builder.add_edge("collect_info_node", "format_output_node")

    # --- Preference update path ---
    builder.add_conditional_edges(
        "update_prefs_node",
        route_after_update,
        {
            "prepare_context": "prepare_context_node",
            "format_output": "format_output_node",
        },
    )

    # --- Generate / Modify itinerary path ---
    # Fan-out: parallel POI search, weather query, budget init, context enrichment
    builder.add_edge("prepare_context_node", "poi_search_node")
    builder.add_edge("prepare_context_node", "weather_node")
    builder.add_edge("prepare_context_node", "budget_init_node")
    builder.add_edge("prepare_context_node", "context_enrichment_node")

    # Fan-in: all parallel nodes converge to planner
    builder.add_edge("poi_search_node", "planner_node")
    builder.add_edge("weather_node", "planner_node")
    builder.add_edge("budget_init_node", "planner_node")
    builder.add_edge("context_enrichment_node", "planner_node")

    # Fan-out: parallel validation, route optimization, budget calculation
    builder.add_edge("planner_node", "validation_node")
    builder.add_edge("planner_node", "route_node")
    builder.add_edge("planner_node", "budget_calc_node")

    # Apply optimized routes back to itinerary
    builder.add_edge("validation_node", "apply_routes_node")
    builder.add_edge("route_node", "apply_routes_node")
    builder.add_edge("budget_calc_node", "apply_routes_node")

    # Fan-in: converge to proposal
    builder.add_edge("apply_routes_node", "proposal_node")

    # Proposal → output
    builder.add_edge("proposal_node", "format_output_node")

    # --- Ask modification path ---
    builder.add_edge("ask_modification_node", "format_output_node")

    # --- Q&A path ---
    builder.add_edge("qa_node", "format_output_node")

    # --- Confirmation path ---
    builder.add_edge("confirm_node", "save_memory_node")
    builder.add_edge("save_memory_node", "format_output_node")

    # --- Final output ---
    builder.add_edge("format_output_node", END)

    return builder
