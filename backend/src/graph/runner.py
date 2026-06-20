"""High-level runner to execute the TravelAgent graph for a single user turn."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, AsyncIterator

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
    return config


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
    async for event in graph.astream(state, _build_graph_config(session_id)):
        logger.debug("Graph event: %s", event)
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


def _public_stage_for_node(name: str) -> str:
    """Map LangGraph node ids to frontend-facing stage ids."""
    if name == "gathering" or name.endswith("gathering_turn"):
        return "gathering"
    planning_nodes = {
        "profile_recall",
        "retrieve",
        "weather_check",
        "plan",
        "tool_call",
        "factcheck",
        "hallucination",
        "booking",
        "understand",
    }
    if name in planning_nodes:
        return "planning"
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

    graph = await get_graph()
    output_state: dict[str, Any] | None = None
    t_start = time.monotonic()

    try:
        async for event in graph.astream_events(
            state,
            _build_graph_config(session_id),
            version="v1",
        ):
            kind = event.get("event")
            name = event.get("name", "")
            raw_data = event.get("data", {})
            data = raw_data if isinstance(raw_data, dict) else {}

            if kind == "on_chain_start":
                yield {
                    "type": "thinking",
                    "stage": _public_stage_for_node(name),
                    "payload": {},
                }
            elif kind == "on_chain_end":
                raw_output = data.get("output", {})
                output = raw_output if isinstance(raw_output, dict) else {}
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
                    and output.get("next_action") == "clarify"
                    and name == "output"
                ):
                    yield {
                        "type": "clarify",
                        "stage": output.get("stage", "gathering"),
                        "payload": output,
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

                yield {"type": "stage", "stage": stage, "payload": payload}

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

    # Final state with download links.
    elapsed_ms = int((time.monotonic() - t_start) * 1000)
    try:
        if output_state:
            final_state = dict(output_state)
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
        else:
            checkpoint = await graph.aget_state(_build_graph_config(session_id))
            if checkpoint and checkpoint.values:
                final_state = dict(checkpoint.values)
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
    return {
        "stage": state.get("stage", "completed"),
        "content": last_message.get("content", ""),
        "message_type": last_message.get("type", "itinerary"),
        "itinerary": state.get("itinerary"),
        "profile": state.get("profile"),
        "output_markdown": state.get("output_markdown"),
        "output_pdf_url": state.get("output_pdf_url"),
        "output_excel_url": state.get("output_excel_url"),
        "output_map_url": state.get("output_map_url"),
        "tool_results": state.get("tool_results"),
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
