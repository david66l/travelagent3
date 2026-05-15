"""LangGraph node functions - each calls an Agent."""

from core.state import ItineraryState
from core.database import async_session_maker
from core.thought_logger import log_step, thought_logger
from schemas import UserProfile, ScoredPOI, WeatherDay, DayPlan
from agents.intent_recognition import IntentRecognitionAgent
from agents.information_collection import InformationCollectionAgent
from agents.realtime_query import RealtimeQueryAgent
from agents.preference_budget import PreferenceBudgetAgent
from agents.itinerary_planner import ItineraryPlannerAgent
from agents.qa_agent import QAAgent
from agents.proposal_generation import ProposalGenerationAgent
from agents.validation import ValidationAgent
from agents.map_route import MapRouteAgent
from skills.memory_store import MemoryStoreSkill


# ===== Node Functions =====


@log_step("intent_node")
async def intent_node(state: ItineraryState) -> dict:
    """Intent recognition node."""
    agent = IntentRecognitionAgent()
    profile = None
    if state.get("user_profile"):
        profile = UserProfile(**state["user_profile"])

    result = await agent.recognize(
        user_input=state["user_input"],
        messages=state.get("messages", []),
        user_profile=profile,
    )

    return {
        "intent": result.intent,
        "intent_confidence": result.confidence,
        "user_entities": result.user_entities,
        "missing_required": result.missing_required,
        "missing_recommended": result.missing_recommended,
        "preference_changes": result.preference_changes,
        "clarification_questions": result.clarification_questions,
        "needs_clarification": len(result.missing_required) > 0,
    }


@log_step("collect_info_node")
async def collect_info_node(state: ItineraryState) -> dict:
    """Generate clarifying questions."""
    agent = InformationCollectionAgent()
    response = await agent.generate_response(
        missing_required=state.get("missing_required", []),
        missing_recommended=state.get("missing_recommended", []),
        current_info=state.get("user_entities", {}),
    )
    return {
        "assistant_response": response,
        "needs_clarification": True,
    }


@log_step("prepare_context_node")
async def prepare_context_node(state: ItineraryState) -> dict:
    """Load user memory and initialize panels."""
    user_profile = state.get("user_profile")
    if not user_profile and state.get("user_entities"):
        user_profile = UserProfile(**state["user_entities"]).model_dump()

    budget_agent = PreferenceBudgetAgent()
    panel = budget_agent.build_preference_panel(
        UserProfile(**user_profile) if user_profile else UserProfile()
    )

    return {
        "user_profile": user_profile,
        "preference_panel": panel,
    }


@log_step("poi_search_node")
async def poi_search_node(state: ItineraryState) -> dict:
    """Search POIs."""
    agent = RealtimeQueryAgent()
    profile = state.get("user_profile", {})
    city = profile.get("destination", "")
    keywords = profile.get("interests", []) + profile.get("food_preferences", [])

    pois = await agent.query_pois(city, keywords)
    return {"candidate_pois": [p.model_dump() for p in pois]}


@log_step("weather_node")
async def weather_node(state: ItineraryState) -> dict:
    """Query weather."""
    agent = RealtimeQueryAgent()
    profile = state.get("user_profile", {})
    city = profile.get("destination", "")
    dates = profile.get("travel_dates", "")

    start, end = _split_dates(dates)
    if not start:
        start = end = "2026-05-01"

    weather = await agent.query_weather(city, start, end)
    return {"weather_data": [w.model_dump() for w in weather]}


@log_step("budget_init_node")
async def budget_init_node(state: ItineraryState) -> dict:
    """Initialize budget panel."""
    agent = PreferenceBudgetAgent()
    profile = UserProfile(**state.get("user_profile", {}))
    panel = agent.init_panel(profile)
    return {"budget_panel": panel.model_dump()}


@log_step("planner_node")
async def planner_node(state: ItineraryState) -> dict:
    """Plan itinerary."""
    agent = ItineraryPlannerAgent()
    profile = UserProfile(**state.get("user_profile", {}))
    pois = [ScoredPOI(**p) for p in state.get("candidate_pois", [])]
    weather = [WeatherDay(**w) for w in state.get("weather_data", [])]
    travel_context = state.get("travel_context")

    itinerary = await agent.plan(pois, weather, profile, travel_context)
    itinerary_json = [day.model_dump() for day in itinerary]
    return {
        "current_itinerary": itinerary_json,
        "itinerary_status": "draft",
        "planning_json": {
            "trip_profile": {
                "destination": profile.destination,
                "travel_days": profile.travel_days,
                "travel_dates": profile.travel_dates,
                "travelers_count": profile.travelers_count,
                "travelers_type": profile.travelers_type,
                "budget_range": profile.budget_range,
                "food_preferences": profile.food_preferences,
                "interests": profile.interests,
                "pace": profile.pace,
                "special_requests": profile.special_requests,
            },
            "days": itinerary_json,
        },
    }


@log_step("validation_node")
async def validation_node(state: ItineraryState) -> dict:
    """Validate itinerary."""
    agent = ValidationAgent()
    profile = UserProfile(**state.get("user_profile", {}))
    itinerary = [DayPlan(**d) for d in state.get("current_itinerary", [])]

    result = await agent.validate(itinerary, profile)
    return {"validation_result": result.model_dump()}


@log_step("route_node")
async def route_node(state: ItineraryState) -> dict:
    """Optimize routes."""
    agent = MapRouteAgent()
    itinerary = [DayPlan(**d) for d in state.get("current_itinerary", [])]

    optimized = await agent.batch_optimize_routes(itinerary)
    return {"optimized_routes": {k: [a.model_dump() for a in v] for k, v in optimized.items()}}


@log_step("apply_routes_node")
async def apply_routes_node(state: ItineraryState) -> dict:
    """Apply route optimization results back to itinerary."""
    optimized = state.get("optimized_routes", {})
    itinerary = [DayPlan(**d) for d in state.get("current_itinerary", [])]

    for day in itinerary:
        key = day.day_number
        if key in optimized:
            day.activities = [Activity(**a) for a in optimized[key]]
        elif str(key) in optimized:
            day.activities = [Activity(**a) for a in optimized[str(key)]]

    return {
        "current_itinerary": [day.model_dump() for day in itinerary],
    }


@log_step("budget_calc_node")
async def budget_calc_node(state: ItineraryState) -> dict:
    """Calculate budget breakdown."""
    agent = PreferenceBudgetAgent()
    profile = UserProfile(**state.get("user_profile", {}))
    itinerary = [DayPlan(**d) for d in state.get("current_itinerary", [])]

    budget = agent.calculate_budget(itinerary, profile)
    return {"budget_panel": budget.model_dump()}


@log_step("proposal_node")
async def proposal_node(state: ItineraryState) -> dict:
    """Generate proposal text."""
    agent = ProposalGenerationAgent()
    profile = UserProfile(**state.get("user_profile", {}))
    planning_json = dict(state.get("planning_json") or {})
    planning_json["trip_profile"] = planning_json.get("trip_profile") or {
        "destination": profile.destination,
        "travel_days": profile.travel_days,
        "travel_dates": profile.travel_dates,
        "travelers_count": profile.travelers_count,
        "travelers_type": profile.travelers_type,
        "budget_range": profile.budget_range,
        "food_preferences": profile.food_preferences,
        "interests": profile.interests,
        "pace": profile.pace,
        "special_requests": profile.special_requests,
    }
    planning_json["days"] = state.get("current_itinerary", [])
    planning_json["budget_panel"] = state.get("budget_panel", {})
    planning_json["validation_result"] = state.get("validation_result", {})

    proposal = await agent.generate(planning_json)
    return {
        "proposal_text": proposal,
        "assistant_response": proposal,
        "waiting_for_confirmation": True,
    }


@log_step("update_prefs_node")
async def update_prefs_node(state: ItineraryState) -> dict:
    """Update preferences and trigger replan if needed."""
    agent = PreferenceBudgetAgent()
    profile = UserProfile(**state.get("user_profile", {}))
    changes = state.get("preference_changes", [])

    updated = agent.update_preferences(profile, changes)
    panel = agent.build_preference_panel(updated)
    budget = agent.init_panel(updated)

    needs_replan = state.get("current_itinerary") is not None

    change_text = ", ".join(f"{c['field']}改为{c['new_value']}" for c in changes)
    response = f"已更新偏好：{change_text}"
    if needs_replan:
        response += "。正在重新规划行程..."

    return {
        "user_profile": updated.model_dump(),
        "preference_panel": panel,
        "budget_panel": budget.model_dump(),
        "needs_replan": needs_replan,
        "assistant_response": response,
    }


@log_step("qa_node")
async def qa_node(state: ItineraryState) -> dict:
    """Answer user question."""
    agent = QAAgent()
    profile = state.get("user_profile", {})
    city = profile.get("destination")

    answer = await agent.answer(state["user_input"], city)
    return {"assistant_response": answer}


@log_step("confirm_node")
async def confirm_node(state: ItineraryState) -> dict:
    """Handle itinerary confirmation."""
    return {
        "itinerary_status": "confirmed",
        "assistant_response": "行程已确认！已保存到您的历史行程中。",
        "waiting_for_confirmation": False,
    }


@log_step("save_memory_node")
async def save_memory_node(state: ItineraryState) -> dict:
    """Save confirmed itinerary and conversation to database."""
    profile = state.get("user_profile", {})
    itinerary = state.get("current_itinerary", [])
    budget = state.get("budget_panel", {})

    async with async_session_maker() as db:
        try:
            # Save itinerary
            await MemoryStoreSkill.save_itinerary(
                db=db,
                session_id=state["session_id"],
                user_id=state.get("user_id", "anonymous"),
                destination=profile.get("destination", ""),
                travel_days=profile.get("travel_days", len(itinerary)),
                daily_plans=itinerary,
                preference_snapshot=profile,
                budget_snapshot=budget,
                status="confirmed",
            )
        except Exception:
            # Persistence failure should not break the flow
            pass

    return {"itinerary_status": "confirmed"}


@log_step("ask_modification_node")
async def ask_modification_node(state: ItineraryState) -> dict:
    """Ask user what they want to modify when the request is vague."""
    current_itinerary = state.get("current_itinerary", [])
    has_itinerary = len(current_itinerary) > 0

    if has_itinerary:
        response = (
            "没问题，告诉我你想调整哪些地方？比如：\n"
            "• 某天的景点想换一下\n"
            "• 预算需要调整\n"
            "• 增加或减少天数\n"
            "• 换个住宿区域\n\n"
            "请直接说你的想法，比如『第三天换个景点』或者『预算再加500』～"
        )
    else:
        response = (
            "没问题，告诉我你的具体需求，比如：\n"
            "• 目的地、天数、预算的变化\n"
            "• 想要体验的类型（美食、文化、自然风光等）\n\n"
            "请直接说你的想法～"
        )

    return {
        "assistant_response": response,
        "waiting_for_confirmation": False,
    }


@log_step("format_output")
async def format_output_node(state: ItineraryState) -> dict:
    """Format final output and append to messages."""
    response = state.get("assistant_response", "")
    if not response:
        response = "抱歉，我没有理解您的意思。能再说一遍吗？"

    # Truncate messages to prevent unbounded growth
    messages = state.get("messages", [])
    if len(messages) > 20:
        messages = messages[-20:]

    # Save thought log
    try:
        log_file = thought_logger.save(
            final_response=response,
            status="success",
        )
        if log_file:
            print(f"[ThoughtLog] Saved to {log_file}")
    except Exception:
        pass

    return {
        "messages": messages + [{"role": "assistant", "content": response}],
    }


# ===== Helpers =====


def _split_dates(dates: str) -> tuple[str, str]:
    """Split date range string into start and end dates."""
    if not dates:
        return "", ""

    for sep in [" to ", " ~ ", " - ", " 到 ", "至", "—"]:
        if sep in dates:
            parts = dates.split(sep, 1)
            return parts[0].strip(), parts[1].strip()

    # Single date
    return dates.strip(), dates.strip()


from schemas import Activity  # noqa: E402
