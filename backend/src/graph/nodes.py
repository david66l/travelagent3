"""LangGraph async nodes for the TravelAgent orchestration layer."""

from __future__ import annotations

import logging
import json
from typing import Any

from core.langsmith_trace import traceable_step
from graph.exceptions import with_error_handling

logger = logging.getLogger(__name__)


@with_error_handling("profile")
async def profile_node(state: dict[str, Any]) -> dict[str, Any]:
    """User profile recall and memory conflict resolution."""
    from graph.node_impl import _user_memory_async

    result = await _user_memory_async(state)
    from agentic.runtime import initialize_agent_ledger
    from core.settings import settings

    projected_state = {**state, **result}
    result.update(initialize_agent_ledger(projected_state, mode=settings.agentic_policy_mode))
    return result


@with_error_handling("agent_loop")
async def agent_loop_node(state: dict[str, Any]) -> dict[str, Any]:
    """Execute one durable decide-act-observe-verify batch.

    LangGraph checkpoints the returned ledger before the router schedules the
    next batch, so a worker crash resumes after the last committed action rather
    than replaying the whole episode.
    """
    from agentic.integration import run_agent_branch

    return await run_agent_branch(state, single_step=True)


@with_error_handling("confirm_gate")
async def confirm_gate_node(state: dict[str, Any]) -> dict[str, Any]:
    """Pause for the user to confirm / modify / reject the draft itinerary.

    Uses a LangGraph dynamic ``interrupt``. The runner resumes with
    ``Command(resume={"action": "confirm"|"modify"|"reject", "change": {...}?})``.
    """
    from langgraph.types import interrupt

    decision = interrupt(
        {
            "type": "awaiting_confirm",
            "itinerary": state.get("itinerary"),
        }
    )
    return await apply_draft_decision(state, decision)


async def apply_draft_decision(state: dict[str, Any], decision: Any) -> dict[str, Any]:
    """Apply a user decision through the one agent lifecycle."""
    if isinstance(decision, str):
        decision = {"action": decision}
    decision = decision or {}
    action = decision.get("action", "confirm")

    if action == "modify":
        from agentic.runtime import revise_agent_ledger

        change = decision.get("change")
        ledger = await revise_agent_ledger(
            state["agent_ledger"], revision_reason=json.dumps(change, ensure_ascii=False)
        )
        return {
            "confirm_decision": "modify",
            "agent_ledger": ledger.model_dump(mode="json"),
            "agent_status": "running",
            "next_action": "agent_continue",
            "stage": "revision_resumed",
        }
    if action == "reject":
        reason = str(decision.get("reason") or "").strip()
        if state.get("policy_mode") == "agent" and state.get("agent_ledger"):
            if not reason:
                return {
                    "confirm_decision": "reject_needs_reason",
                    "agent_status": "awaiting_revision_reason",
                    "next_action": "clarify",
                    "clarification_questions": [
                        "这版行程哪里不合适？例如太赶、预算过高、景点不喜欢或交通不方便。"
                    ],
                    "stage": "awaiting_revision_reason",
                }
            from agentic.runtime import revise_agent_ledger

            ledger = await revise_agent_ledger(state["agent_ledger"], revision_reason=reason)
            return {
                "confirm_decision": "reject_with_reason",
                "agent_ledger": ledger.model_dump(mode="json"),
                "agent_status": "running",
                "next_action": "agent_continue",
                "stage": "revision_resumed",
            }
        raise ValueError("Agent session missing; a legacy planner cannot be resumed")
    if state.get("agent_status") != "awaiting_confirmation" or not state.get("agent_ledger"):
        return {
            "agent_status": "failed",
            "next_action": "agent_error",
            "termination_reason": "NO_VERIFIED_DRAFT_TO_CONFIRM",
        }
    agent_patch: dict[str, Any] = {}
    if (
        state.get("policy_mode") == "agent"
        and state.get("agent_status") == "awaiting_confirmation"
        and state.get("agent_ledger")
    ):
        from agentic.runtime import confirm_agent_ledger

        ledger, completion = confirm_agent_ledger(state["agent_ledger"])
        agent_patch = {
            "agent_ledger": ledger.model_dump(mode="json"),
            "agent_status": "finished",
            "termination_reason": "validated_finish",
            "completion_decision": completion.model_dump(mode="json"),
        }
    return {
        "confirm_decision": "confirm",
        "stage": "confirmed",
        "next_action": "agent_final",
        **agent_patch,
    }


@traceable_step("planning/content_safety", run_type="chain")
def _trace_content_safety(state: dict[str, Any]) -> Any:
    from agents.content_safety import ContentSafetyEngine

    return ContentSafetyEngine.check(state)


@traceable_step("planning/output_artifacts", run_type="chain")
async def _trace_output_format(
    *,
    proposal_text: str,
    itinerary: list[Any],
    city: str,
    session_id: str,
    on_token: Any = None,
) -> dict[str, Any]:
    from agents.output_format import output_format_agent

    return await output_format_agent.format(
        proposal_text=proposal_text,
        itinerary=itinerary,
        city=city,
        session_id=session_id,
        on_token=on_token,
    )


def _agent_clarification_message(ledger: Any, latest: Any) -> tuple[str, list[str]]:
    """Render the model's actual question; the renderer does not replace its decision."""
    payload = latest.payload
    question = str(payload.get("question") or payload.get("reason") or "请补充信息后继续。")
    return question, [str(item) for item in (payload.get("options") or [])]


@with_error_handling("output")
async def output_node(state: dict[str, Any]) -> dict[str, Any]:
    """Output formatting (Markdown / clarification) + multi-modal export.

    The writer may use one structured LLM call for themes and recommendation
    reasons, but the final Markdown is always rendered deterministically from
    the enriched itinerary. This prevents prose generation from changing names,
    times, prices or route order while retaining progressive client rendering.
    """
    from api.chat_runtime import publish_live_stage, publish_token
    from agents.output_format import output_format_agent
    from core.conversation_state import flatten_profile
    from graph.node_impl import _output_async

    job_id = state.get("job_id")
    session_id = state.get("session_id")
    itinerary = state.get("itinerary", []) or []
    if (
        state.get("policy_mode") == "agent"
        and state.get("agent_status") == "awaiting_information"
        and state.get("agent_ledger")
    ):
        from agentic.state import AgentLedgerState

        ledger = AgentLedgerState(**state["agent_ledger"])
        questions = [
            artifact
            for artifact in ledger.artifacts.values()
            if artifact.artifact_type in {"user_question", "propose_tradeoff"}
            and artifact.goal_version == ledger.goal.goal_version
            and artifact.plan_version == ledger.task_graph.plan_version
        ]
        latest = questions[-1] if questions else None
        blocked = next(
            (task for task in ledger.task_graph.tasks if task.status == "blocked"),
            None,
        )
        if latest is not None:
            question, options = _agent_clarification_message(ledger, latest)
            if options:
                question = f"{question}\n" + "\n".join(
                    f"{index}. {option}" for index, option in enumerate(options, start=1)
                )
            return {
                "messages": (state.get("messages") or [])
                + [
                    {
                        "role": "assistant",
                        "content": question,
                        "type": "agent_clarification",
                        "task_id": blocked.task_id if blocked else None,
                        "question_id": latest.artifact_id,
                    }
                ],
                "stage": "agent_awaiting_information",
                "next_action": "clarify",
            }
    if state.get("policy_mode") == "agent" and state.get("agent_status") == "failed":
        reason = str(state.get("termination_reason") or "AGENT_STOPPED")
        return {
            "messages": (state.get("messages") or [])
            + [
                {
                    "role": "assistant",
                    "content": (
                        "这次规划没有在安全预算内得到可验证行程，已停止执行，"
                        f"没有切换到另一套流程掩盖失败。原因：{reason}。"
                    ),
                    "type": "agent_error",
                }
            ],
            "stage": "agent_failed",
            "next_action": "agent_error",
        }
    # Some callers/tests restore an itinerary-only checkpoint created before
    # ``next_action`` became part of the state schema.  Treat an existing
    # itinerary as the itinerary path instead of accidentally formatting it as
    # a generic chat response.
    next_action = state.get("next_action") or ("fact_check" if itinerary else "respond")

    async def _on_token(chunk: str) -> None:
        if job_id:
            await publish_token(str(job_id), chunk)
        elif session_id:
            from api.chat_runtime import manager

            await manager.send_json(session_id, {"type": "token", "chunk": chunk})

    # Non-itinerary outputs (clarify / respond / infeasible / empty): single pass.
    if next_action in ("clarify", "respond", "infeasible") or not itinerary:
        return await _output_async(state)

    # ---- Itinerary path -------------------------------------------------- #
    # Content safety first (cheap, sync) so we never stream a blocked plan.
    safety_dict: dict[str, Any] | None = None
    try:
        safety = _trace_content_safety(state)
        safety_dict = safety.model_dump()
    except Exception as exc:
        logger.warning("Content safety check failed: %s", exc)
        safety = None
    if safety and not safety.passed:
        reasons = "；".join(safety.improvement_suggestions) or "内容安全校验未通过"
        return {
            "messages": (state.get("messages") or [])
            + [
                {
                    "role": "assistant",
                    "content": f"内容安全拦截：{reasons}。请调整需求后重试。",
                    "type": "safety_blocked",
                }
            ],
            "stage": "safety_blocked",
            "safety_result": safety_dict,
        }

    # Progress line only — the blinking caret appears on the first real token.
    if session_id or job_id:
        try:
            await publish_live_stage(session_id=session_id, job_id=job_id, stage="writing")
        except Exception as exc:
            logger.debug("Live stage push skipped: %s", exc)

    profile_raw = flatten_profile(state.get("profile") or {})
    city = profile_raw.get("destination") or ""
    sid = session_id or state.get("user_id") or "default"
    on_token = _on_token if (job_id or session_id) else None

    try:
        base = await _output_async(state)
    except Exception as exc:
        logger.warning("Itinerary enrichment failed: %s", exc)
        base = {
            "messages": (state.get("messages") or [])
            + [
                {
                    "role": "assistant",
                    "content": "",
                    "type": "itinerary",
                    "itinerary": itinerary,
                    "warnings": state.get("warnings", []),
                }
            ],
            "itinerary": itinerary,
            "stage": "awaiting_booking",
        }

    if safety_dict:
        base["safety_result"] = safety_dict

    enriched_itin = base.get("itinerary") or itinerary
    final_md = base["messages"][-1].get("content", "") if base.get("messages") else ""
    await output_format_agent.stream_existing_markdown(final_md, on_token)

    try:
        artifacts = await output_format_agent.build_artifacts(final_md, enriched_itin, city, sid)
    except Exception as exc:
        logger.warning("Artifact build failed: %s", exc)
        artifacts = {"pdf": None, "excel": None, "map": None}

    base["output_markdown"] = final_md
    base["output_pdf_url"] = artifacts.get("pdf")
    base["output_excel_url"] = artifacts.get("excel")
    base["output_map_url"] = artifacts.get("map")
    if base.get("messages"):
        base["messages"][-1]["content"] = final_md
        base["messages"][-1]["output_pdf_url"] = artifacts.get("pdf")
        base["messages"][-1]["output_excel_url"] = artifacts.get("excel")
        base["messages"][-1]["output_map_url"] = artifacts.get("map")

    if state.get("confirm_decision") != "confirm" and enriched_itin:
        from graph.approval import issue_pending_approval

        base["pending_approval"] = issue_pending_approval(
            {**state, **base, "itinerary": enriched_itin}
        )
    else:
        base["pending_approval"] = None

    return base


@with_error_handling("booking")
async def booking_node(state: dict[str, Any]) -> dict[str, Any]:
    """Booking tool aggregation (mock data in MVP)."""
    from graph.node_impl import _booking_tool_async

    return await _booking_tool_async(state)
