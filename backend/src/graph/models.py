"""AgentState TypedDict for the TravelAgent LangGraph orchestration layer."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


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

    # Planner output
    itinerary: NotRequired[list[dict[str, Any]]]
    budget_breakdown: NotRequired[dict[str, Any]]
    solve_status: NotRequired[str]
    solve_time_ms: NotRequired[int]
    conflict_reasons: NotRequired[list[str]]
    replan_mode: NotRequired[bool]

    # FactCheck output
    factcheck_passed: NotRequired[bool]
    warnings: NotRequired[list[str]]
    retry_count: NotRequired[int]

    # Tool call layer
    pending_tool_calls: NotRequired[list[dict[str, Any]]]
    tool_results: NotRequired[list[dict[str, Any]]]

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

    # Control flow
    next_node: NotRequired[str]
    next_action: NotRequired[str]
    loop_count: NotRequired[int]
    max_loops: NotRequired[int]
    version: NotRequired[int]
    execution_trace: NotRequired[list[str]]
    error_node: NotRequired[str]
    error_message: NotRequired[str]
    fallback_used: NotRequired[list[str]]
    stage: NotRequired[str]
    # Gathering subgraph — sync back to WS conversation state
    intent_ready_message: NotRequired[str]
    conversation_sync: NotRequired[dict[str, Any]]
    _conversation_state: NotRequired[dict[str, Any]]
