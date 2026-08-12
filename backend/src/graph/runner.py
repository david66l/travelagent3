"""High-level runner to execute the TravelAgent graph for a single user turn."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, AsyncIterator

from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from core.langsmith_trace import graph_langsmith_callbacks, patch_langsmith_root_run_for_pause
from graph.graph import get_graph
from graph.session_manager import SessionManager

logger = logging.getLogger(__name__)

# Local trace saving — writes a copy of every graph execution to disk
_LOCAL_TRACE_ENABLED = os.environ.get("LANGSMITH_API_KEY", "") != ""

try:
    from core.langsmith_local import save_local_trace_lightweight as _save_local
except ImportError:
    _LOCAL_TRACE_ENABLED = False
    def _save_local(*args: Any, **kwargs: Any) -> str:
        return ""


def _langsmith_enabled() -> bool:
    return os.environ.get("LANGSMITH_API_KEY", "") != ""


def _build_graph_config(session_id: str) -> dict[str, Any]:
    """Build the config dict for graph invocation, including LangSmith metadata."""
    config: dict[str, Any] = {"configurable": {"thread_id": session_id}}
    if _langsmith_enabled():
        config["metadata"] = {
            "langsmith_project": os.environ.get("LANGSMITH_PROJECT", "TravelAgent"),
            "session_id": session_id,
        }
        callbacks = graph_langsmith_callbacks()
        if callbacks:
            config["callbacks"] = callbacks
    return config


def _is_confirm_pause(snapshot: Any) -> bool:
    """Recognize a confirm interrupt across LangGraph checkpoint versions."""
    if not snapshot:
        return False
    if "confirm_gate" in (getattr(snapshot, "next", None) or ()):
        return True
    for task in getattr(snapshot, "tasks", None) or ():
        if getattr(task, "name", None) == "confirm_gate" and getattr(task, "interrupts", None):
            return True
    return False


async def run_graph_turn(
    session_id: str,
    user_id: str,
    user_input: str,
    messages: list[dict[str, Any]] | None = None,
    attachments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one complete graph turn and return the final state.

    This is the integration point for chat_runtime / SSE / WS to use the
    LangGraph orchestration layer instead of the legacy pipeline.
    """
    sm = SessionManager()
    state = await sm.create(session_id, user_id, user_input, messages, attachments)
    state["session_id"] = session_id

    graph = await get_graph()
    t_start = time.monotonic()
    final_state: dict[str, Any] = {}
    graph_config = _build_graph_config(session_id)
    try:
        async for event in graph.astream(state, graph_config):
            logger.debug("Graph event: %s", event)
    except GraphBubbleUp as pause:
        # Human-in-the-loop pause at confirm_gate — expected, not a failure.
        logger.info("Graph paused at interrupt for session %s", session_id)
        patch_langsmith_root_run_for_pause(pause)
    elapsed_ms = int((time.monotonic() - t_start) * 1000)

    # Recover the persisted checkpoint
    checkpoint = await graph.aget_state(_build_graph_config(session_id))
    if checkpoint and checkpoint.values:
        final_state = dict(checkpoint.values)
        await sm.save(session_id, final_state)
    else:
        final_state = state

    result = _extract_result(final_state)

    # Save local trace
    if _LOCAL_TRACE_ENABLED:
        try:
            _save_local(
                session_id=session_id,
                user_input=user_input,
                itinerary=result.get("itinerary"),
                duration_ms=elapsed_ms,
                status="success",
                warnings=final_state.get("warnings"),
            )
        except Exception:
            pass

    return result


# Real graph nodes whose start we surface to the client as granular progress.
# The frontend (stageLabels.ts) maps each of these to a specific status line
# (e.g. retrieve → "正在收集景点信息…"), so we must forward the *actual* node
# name instead of collapsing everything to a generic "planning" label.
_PROGRESS_NODES = {
    "understand",
    "profile_recall",
    "retrieve",
    "weather_check",
    "plan",
    "tool_call",
    "factcheck",
    "hallucination",
    "output",
    "booking",
    "apply_single_change",
    "replan_local",
}


def _public_stage_for_node(name: str) -> str:
    """Map LangGraph node ids to frontend-facing stage ids.

    Known planning nodes are surfaced under their real id so the client can show
    a granular, step-specific status line; everything else passes through.
    """
    if name == "gathering" or name.endswith("gathering_turn"):
        return "gathering"
    return name


async def stream_graph_events(
    session_id: str,
    user_id: str,
    user_input: str,
    messages: list[dict[str, Any]] | None = None,
    attachments: list[dict[str, Any]] | None = None,
    profile: dict[str, Any] | None = None,
    slots: dict[str, Any] | None = None,
    conversation_state: dict[str, Any] | None = None,
    action: str = "chat",
    action_payload: dict[str, Any] | None = None,
    job_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream graph events for SSE/WS.

    Yields simplified events: {type, stage, payload}.

    Event types:
      - thinking: node started
      - tool_call: tool execution batch
      - partial: intermediate node output
      - final: final itinerary with download links
      - clarify: need more user info
      - error: unrecoverable error / escalation
    """
    from core.conversation_state import flatten_profile

    sm = SessionManager()
    graph = await get_graph()
    config = _build_graph_config(session_id)

    # Decide the graph input: resume a paused interrupt, jump for an in-trip
    # event, or start a fresh turn. A checkpoint read failure must not crash the
    # turn — degrade to "not paused" and run a fresh turn.
    try:
        snapshot = await graph.aget_state(config)
    except Exception as exc:
        logger.debug("aget_state failed for %s: %s", session_id, exc)
        snapshot = None
    is_paused = _is_confirm_pause(snapshot)
    graph_input: Any

    if action in ("confirm", "modify", "reject") and is_paused:
        resume_value: dict[str, Any] = {"action": action}
        if action == "modify":
            payload = action_payload or {}
            resume_value["change"] = payload.get("change") or payload
        graph_input = Command(resume=resume_value)
        logger.info("Graph resume for %s: action=%s", session_id, action)
    elif action in ("confirm", "modify", "reject"):
        # A completed-checkpoint or older saver may no longer expose ``next``
        # even though the latest state still contains a draft. Continue from an
        # explicit node instead of silently starting a brand-new chat turn and
        # ignoring the user's button action.
        values = dict(snapshot.values) if snapshot and snapshot.values else {}
        if not values.get("itinerary"):
            yield {
                "type": "error",
                "stage": "draft_missing",
                "payload": {
                    "error": "当前没有可操作的行程草案，请先生成行程。",
                    "error_type": "DraftNotFound",
                },
            }
            return
        if action == "modify":
            payload = action_payload or {}
            graph_input = Command(
                goto="apply_single_change",
                update={
                    "confirm_decision": "modify",
                    "pending_change": payload.get("change") or payload,
                },
            )
        elif action == "reject":
            graph_input = Command(
                goto="plan",
                update={"confirm_decision": None, "stage": "rejected"},
            )
        else:
            graph_input = Command(
                goto="tool_call",
                update={"confirm_decision": "confirm", "next_action": "enrich"},
            )
        logger.warning(
            "Checkpoint for %s did not expose confirm interrupt; continued action=%s from draft",
            session_id,
            action,
        )
    elif action == "trip_event" and snapshot and snapshot.values:
        graph_input = Command(
            goto="replan_local",
            update={"external_event": action_payload or {}},
        )
        logger.info("Graph in-trip replan for %s", session_id)
    elif is_paused:
        # A paused draft only accepts explicit control actions. Silently treating
        # free text as "reject" used to discard the user's text and corrupt the
        # gathered profile. Surface a stable error instead; the client disables
        # its text box and presents confirm/modify/regenerate controls.
        yield {
            "type": "error",
            "stage": "awaiting_confirm",
            "payload": {
                "error": "当前行程草案正在等待确认，请使用确认、修改或重新生成按钮。",
                "error_type": "DraftActionRequired",
            },
        }
        return
    else:
        existing = await sm.load(session_id)
        if existing:
            state = existing
            state["user_input"] = user_input
            state["user_id"] = user_id
            state["messages"] = messages or []
            state["attachments"] = attachments or []
            state.setdefault("session_id", session_id)
            state["stage"] = "resumed"
        else:
            state = await sm.create(session_id, user_id, user_input, messages, attachments)
        state["session_id"] = session_id
        if job_id:
            state["job_id"] = job_id  # enables output-node token streaming

        if conversation_state:
            state["_conversation_state"] = conversation_state
            state["profile"] = conversation_state.get("profile") or profile or state.get("profile") or {}
            if not messages:
                recent = conversation_state.get("recent_messages", [])
                if isinstance(recent, list):
                    messages = [
                        {"role": m.get("role", "user"), "content": m.get("content", "")}
                        for m in recent[-10:]
                        if isinstance(m, dict) and m.get("role") and m.get("content")
                    ]
            state["messages"] = messages or state.get("messages") or []

        # Seed the graph checkpoint with the API-level profile when the checkpoint
        # does not already contain one. This is essential for multi-turn sessions
        # where the legacy gathering phase collected destination/days/etc.
        existing_profile = state.get("profile") or {}
        existing_flat = flatten_profile(existing_profile)
        if not existing_flat.get("destination"):
            if profile:
                state["profile"] = profile
                existing_flat = flatten_profile(profile)
        if slots:
            state["slots"] = slots
        elif not state.get("slots"):
            state["slots"] = existing_flat

        logger.info(
            "Graph turn for %s: destination=%s days=%s profile_keys=%s",
            session_id,
            existing_flat.get("destination"),
            existing_flat.get("travel_days"),
            list(existing_flat.keys()),
        )
        # Accumulator channels (reducers) are managed inside the graph; don't
        # re-seed them from the loaded mirror or they would double across turns.
        for _k in ("warnings", "fallback_used", "execution_trace"):
            state.pop(_k, None)
        graph_input = state

    output_state: dict[str, Any] | None = None
    t_start = time.monotonic()

    try:
        async for event in graph.astream_events(
            graph_input,
            _build_graph_config(session_id),
            version="v1",
        ):
            kind = event.get("event")
            name = event.get("name", "")
            raw_data = event.get("data", {})
            data = raw_data if isinstance(raw_data, dict) else {}

            if kind == "on_chain_start":
                # Only surface real graph nodes (skip LangGraph internals like
                # ChannelWrite/RunnableSeq) so each progress line is meaningful.
                if name in _PROGRESS_NODES or name == "gathering" or name.endswith("gathering_turn"):
                    yield {
                        "type": "thinking",
                        "stage": _public_stage_for_node(name),
                        "payload": {},
                    }
            elif kind == "on_chain_end":
                raw_output = data.get("output", {})
                output = raw_output if isinstance(raw_output, dict) else {}
                if "__interrupt__" in output:
                    # Don't forward the LangGraph Interrupt sentinel (not JSON
                    # serializable); the awaiting_confirm signal is emitted from
                    # the paused-state check after this loop.
                    output = {k: v for k, v in output.items() if k != "__interrupt__"}
                stage = output.get("stage", name) if isinstance(output, dict) else name
                payload = output

                if name == "tool_call":
                    yield {
                        "type": "tool_call",
                        "stage": stage,
                        "payload": {
                            "tool_results": output.get("tool_results", [])
                            if isinstance(output, dict)
                            else []
                        },
                    }
                    continue

                if name == "output":
                    if isinstance(output, dict):
                        output_state = output
                    if isinstance(output, dict) and output.get("next_action") == "clarify":
                        yield {
                            "type": "clarify",
                            "stage": output.get("stage", "gathering"),
                            "payload": output,
                        }
                        continue
                    msg = {}
                    if isinstance(output, dict) and output.get("messages"):
                        msg = output["messages"][-1]
                    yield {
                        "type": "partial",
                        "stage": stage,
                        "payload": {
                            "content": msg.get("content", "") if isinstance(msg, dict) else "",
                            "itinerary": output.get("itinerary") if isinstance(output, dict) else None,
                            "output_pdf_url": output.get("output_pdf_url") if isinstance(output, dict) else None,
                            "output_excel_url": output.get("output_excel_url") if isinstance(output, dict) else None,
                            "output_map_url": output.get("output_map_url") if isinstance(output, dict) else None,
                        },
                    }
                    continue

                if (
                    isinstance(output, dict)
                    and output.get("intent_ready_message")
                    and name == "gathering"
                ):
                    yield {
                        "type": "intent_ready",
                        "stage": "planning",
                        "payload": output,
                    }

                # As with chain-start events, expose only real graph nodes.
                # LangGraph also emits on_chain_end for internal runnable
                # sequences; forwarding those leaked misleading stages such as
                # ``completed`` before the confirm interrupt had actually been
                # reached.
                if name in _PROGRESS_NODES or name == "gathering" or name.endswith("gathering_turn"):
                    yield {"type": "stage", "stage": stage, "payload": payload}

    except GraphBubbleUp as pause:
        # confirm_gate interrupt() — normal pause for user confirmation.
        logger.info("Graph paused at interrupt for session %s", session_id)
        patch_langsmith_root_run_for_pause(pause)
    except Exception as exc:
        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        logger.exception("Graph streaming failed for session %s: %s", session_id, exc)
        if _LOCAL_TRACE_ENABLED:
            try:
                _save_local(session_id=session_id, user_input=user_input,
                    duration_ms=elapsed_ms, status="error", error=str(exc))
            except Exception:
                pass
        yield {
            "type": "error",
            "stage": "error",
            "payload": {"error": str(exc), "error_type": type(exc).__name__},
        }

    # If the graph paused at the confirmation interrupt, signal the client to
    # confirm/modify instead of emitting a final result.
    try:
        paused = await graph.aget_state(config)
        if _is_confirm_pause(paused):
            vals = dict(paused.values) if paused.values else {}
            await sm.save(session_id, vals)
            elapsed_ms = int((time.monotonic() - t_start) * 1000)
            if _LOCAL_TRACE_ENABLED:
                try:
                    _save_local(
                        session_id=session_id,
                        user_input=user_input,
                        itinerary=vals.get("itinerary"),
                        duration_ms=elapsed_ms,
                        status="paused",
                        warnings=vals.get("warnings"),
                    )
                except Exception:
                    pass
            yield {
                "type": "awaiting_confirm",
                "stage": "draft_ready",
                "payload": {
                    "itinerary": vals.get("itinerary"),
                    "warnings": vals.get("warnings", []),
                },
            }
            return
    except Exception as exc:
        logger.debug("Paused-state check failed for %s: %s", session_id, exc)

    # Final state with download links.
    elapsed_ms = int((time.monotonic() - t_start) * 1000)
    try:
        checkpoint = await graph.aget_state(_build_graph_config(session_id))
        if checkpoint and checkpoint.values:
            # A node's on_chain_end output is only its state patch. The persisted
            # checkpoint is the complete state and includes the later booking /
            # memory-writeback results. Prefer it so final responses never drop
            # tool_results, booking_results or the gathered profile.
            final_state = dict(checkpoint.values)
            await sm.save(session_id, final_state)
            result = _extract_result(final_state)

            # Save local trace
            if _LOCAL_TRACE_ENABLED:
                try:
                    _save_local(
                        session_id=session_id, user_input=user_input,
                        itinerary=result.get("itinerary"),
                        duration_ms=elapsed_ms, status="success",
                        warnings=final_state.get("warnings"),
                    )
                except Exception:
                    pass

            yield {"type": "final", "stage": result.get("stage", "completed"), "payload": result}
        elif output_state:
            final_state = dict(output_state)
            await sm.save(session_id, final_state)
            result = _extract_result(final_state)
            yield {"type": "final", "stage": result.get("stage", "completed"), "payload": result}
        else:
            yield {"type": "final", "stage": "completed", "payload": {}}
    except Exception as exc:
        logger.warning("Failed to read final checkpoint for %s: %s", session_id, exc)
        yield {"type": "final", "stage": "completed", "payload": {}}


def _extract_result(state: dict[str, Any]) -> dict[str, Any]:
    """Extract user-facing result from the final graph state."""
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else {}
    internal_stage = state.get("stage", "completed")
    # Memory write-back runs after booking and leaves the graph's technical
    # stage as ``memory_updated``. For API/UI consumers that is still a fully
    # completed turn, not a new user-visible phase.
    public_stage = (
        "completed"
        if state.get("booking_results") or internal_stage == "memory_updated"
        else internal_stage
    )
    return {
        "stage": public_stage,
        "content": last_message.get("content", ""),
        "message_type": last_message.get("type", "itinerary"),
        "itinerary": state.get("itinerary"),
        "profile": state.get("profile"),
        "output_markdown": state.get("output_markdown"),
        "output_pdf_url": state.get("output_pdf_url"),
        "output_excel_url": state.get("output_excel_url"),
        "output_map_url": state.get("output_map_url"),
        "tool_results": state.get("tool_results"),
        "booking_results": state.get("booking_results"),
        "budget_breakdown": state.get("budget_breakdown"),
        "warnings": state.get("warnings", []),
    }


class GraphRunner:
    """Reusable runner exposing both invoke and stream interfaces."""

    async def run(
        self,
        user_input: str,
        *,
        session_id: str,
        user_id: str,
        user_role: str = "guest",
        messages: list[dict[str, Any]] | None = None,
        profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one turn and return the final result."""
        # TODO: propagate profile/slots to run_graph_turn when needed
        return await run_graph_turn(
            session_id=session_id,
            user_id=user_id,
            user_input=user_input,
            messages=messages,
            attachments=None,
        )

    async def stream(
        self,
        user_input: str,
        *,
        session_id: str,
        user_id: str,
        user_role: str = "guest",
        messages: list[dict[str, Any]] | None = None,
        profile: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream events for one turn."""
        async for event in stream_graph_events(
            session_id=session_id,
            user_id=user_id,
            user_input=user_input,
            messages=messages,
            attachments=None,
            profile=profile,
        ):
            yield event


# Global runner instance.
runner = GraphRunner()

__all__ = ["run_graph_turn", "stream_graph_events", "GraphRunner", "runner"]
