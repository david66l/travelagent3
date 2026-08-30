"""AgentState TypedDict for the TravelAgent LangGraph orchestration layer."""

from __future__ import annotations

import operator
from typing import Annotated, Any, NotRequired, TypedDict


def _reduce_last_str(left: str | None, right: str | None) -> str:
    """Last-wins merge for keys that parallel nodes may both touch (e.g. stage)."""
    if right is not None and str(right).strip():
        return str(right)
    if left is not None:
        return str(left)
    return ""


def _reduce_last_list(
    left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Last-wins merge for list channels (e.g. itinerary).

    Defensive guard: if two writes ever land in the same superstep, keep the
    latest non-empty value instead of raising INVALID_CONCURRENT_GRAPH_UPDATE.
    """
    if right:
        return right
    if left:
        return left
    return []


class AgentState(TypedDict):
    """Full state carried through the LangGraph execution.

    Mirrors the blueprint: perception -> understand -> profile -> retrieve
    -> plan -> fact_check -> output -> booking -> memory write-back.
    """

    # Perception layer output
    user_input: str
    messages: NotRequired[list[dict[str, Any]]]
    attachments: NotRequired[list[dict[str, Any]]]
    external_event: NotRequired[dict[str, Any] | None]

    # DemandParser output
    intent: NotRequired[str]
    confidence: NotRequired[float]
    sentiment: NotRequired[str]
    slots: NotRequired[dict[str, Any]]
    missing_slots: NotRequired[list[str]]
    clarification_questions: NotRequired[list[str]]
    disambiguation_candidates: NotRequired[list[dict[str, Any]]]
    feasibility_report: NotRequired[dict[str, Any]]

    # ProfileRecall output
    user_id: NotRequired[str]
    profile: NotRequired[dict[str, Any]]
    preference_vector: NotRequired[list[float] | None]
    inferred_slots: NotRequired[dict[str, Any]]
    is_new_user: NotRequired[bool]

    # RAG output
    poi_candidates: NotRequired[list[dict[str, Any]]]
    knowledge_results: NotRequired[list[dict[str, Any]]]
    retrieval_query: NotRequired[str]
    retrieval_empty: NotRequired[bool]
    retrieval_stats: NotRequired[dict[str, Any]]

    # Weather (fetched before planning)
    weather: NotRequired[list[dict[str, Any]]]
    weather_fetched: NotRequired[bool]
    weather_start: NotRequired[str]
    weather_end: NotRequired[str]

    # Planner output
    # Last-wins reducer: defends against concurrent writes from parallel branches
    # (see _reduce_last_list). Annotated must be top-level for LangGraph to detect it.
    itinerary: Annotated[list[dict[str, Any]], _reduce_last_list]
    budget_breakdown: NotRequired[dict[str, Any]]
    solve_status: NotRequired[str]
    solve_time_ms: NotRequired[int]
    conflict_reasons: NotRequired[list[str]]
    replan_mode: NotRequired[bool]

    # FactCheck output
    factcheck_passed: NotRequired[bool]
    validation_report: NotRequired[dict[str, Any]]
    completion_decision: NotRequired[dict[str, Any]]

    # Long-horizon Agent Loop authoritative state (shadow mode first).
    policy_mode: NotRequired[str]
    agent_ledger: NotRequired[dict[str, Any]]
    agent_episode: NotRequired[dict[str, Any]]
    current_task_id: NotRequired[str | None]
    agent_step: NotRequired[int]
    subtask_step: NotRequired[int]
    agent_status: NotRequired[str]
    agent_execution_mode: NotRequired[str]
    agent_policy_routing: NotRequired[dict[str, Any]]
    termination_reason: NotRequired[str | None]
    shadow_scenario_id: NotRequired[str]
    shadow_input_hash: NotRequired[str]
    shadow_status: NotRequired[str]
    # Accumulator: nodes return only their *new* warnings; the reducer concatenates.
    # NOTE: Annotated must be top-level (not wrapped in NotRequired) for LangGraph
    # to detect the reducer; the channel still defaults to [] when unset.
    warnings: Annotated[list[str], operator.add]
    retry_count: NotRequired[int]

    # Tool call layer
    pending_tool_calls: NotRequired[list[dict[str, Any]]]
    tool_results: NotRequired[list[dict[str, Any]]]
    allowed_tools: NotRequired[set[str]]

    # Safety / hallucination layer
    hallucination_result: NotRequired[dict[str, Any]]
    safety_result: NotRequired[dict[str, Any]]

    # Output layer
    output_markdown: NotRequired[str]
    output_pdf_url: NotRequired[str]
    output_excel_url: NotRequired[str]
    output_map_url: NotRequired[str]

    # Booking tool
    booking_options: NotRequired[dict[str, Any]]
    booking_results: NotRequired[dict[str, Any]]

    # Confirmation gate (draft → confirm/modify/reject interrupt)
    # (external_event for in-trip replanning is declared in the perception layer)
    confirm_decision: NotRequired[str | None]
    pending_change: NotRequired[dict[str, Any] | None]
    pending_approval: NotRequired[dict[str, Any] | None]

    # Session / routing context
    session_id: NotRequired[str]
    # Planning-job id (async worker path). Lets the output node stream polish
    # tokens to the Redis channel token:{job_id} that the SSE layer forwards.
    job_id: NotRequired[str]
    user_role: NotRequired[str]
    phase: NotRequired[str]
    attachments_meta: NotRequired[list[dict[str, Any]]]

    # Control flow
    next_node: NotRequired[str]
    next_action: NotRequired[str]
    loop_count: NotRequired[int]
    max_loops: NotRequired[int]
    version: NotRequired[int]
    execution_trace: NotRequired[list[str]]
    error_node: NotRequired[str]
    error_message: NotRequired[str]
    fallback_used: Annotated[list[str], operator.add]
    # Parallel branches (profile_recall ∥ weather_check) may both emit stage hints;
    # use a reducer so LangGraph does not raise INVALID_CONCURRENT_GRAPH_UPDATE.
    stage: Annotated[str, _reduce_last_str]
    # Gathering subgraph — sync back to WS conversation state
    intent_ready_message: NotRequired[str]
    conversation_sync: NotRequired[dict[str, Any]]
    _conversation_state: NotRequired[dict[str, Any]]
