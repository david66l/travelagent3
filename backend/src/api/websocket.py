"""WebSocket endpoint for real-time chat with the TravelAgent graph."""

import json
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.state import ItineraryState
from core.input_guard import sanitize_user_input
from core.database import async_session_maker
from core.thought_logger import thought_logger
from skills.memory_store import MemoryStoreSkill

router = APIRouter()


class ConnectionManager:
    """Manage active WebSocket connections by session."""

    def __init__(self):
        self._connections: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self._connections[session_id] = websocket

    def disconnect(self, session_id: str):
        self._connections.pop(session_id, None)

    async def send_json(self, session_id: str, data: dict):
        ws = self._connections.get(session_id)
        if ws:
            await ws.send_json(data)


manager = ConnectionManager()


@router.websocket("/ws/chat/{session_id}")
async def chat_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for session-based chat.

    Messages:
        Client -> Server: {"content": "...", "user_id": "..."}
        Server -> Client: {"assistant_message": "...", "intent": "...", ...}
    """
    await manager.connect(session_id, websocket)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send_json(
                    session_id, {"error": "Invalid JSON", "type": "error"}
                )
                continue

            content = msg.get("content", "").strip()
            if not content:
                await manager.send_json(
                    session_id, {"error": "Empty message", "type": "error"}
                )
                continue

            # Sanitize user input
            safe_content = sanitize_user_input(content)

            # Get graph from app state
            graph = websocket.app.state.graph
            user_id = msg.get("user_id", "anonymous")
            config = {"configurable": {"thread_id": session_id}}

            input_state = ItineraryState(
                session_id=session_id,
                user_id=user_id,
                user_input=safe_content,
                messages=[{"role": "user", "content": safe_content}],
            )

            # Start logging session and register WS callback for real-time status
            thought_logger.start_session(session_id, safe_content)

            async def push_status(data: dict):
                await manager.send_json(session_id, data)

            thought_logger.register_ws_callback(session_id, push_status)

            try:
                result = await graph.ainvoke(input_state, config=config)
            except Exception as e:
                thought_logger.unregister_ws_callback(session_id)
                await manager.send_json(
                    session_id,
                    {
                        "type": "error",
                        "error": f"Graph execution failed: {str(e)}",
                    },
                )
                continue

            # Save thought log (JSON + Markdown)
            try:
                thought_logger.save(
                    session_id=session_id,
                    final_response=result.get("assistant_response") or "",
                    status="success" if not result.get("error") else "error",
                )
            except Exception:
                # Log saving failure should not break the chat flow
                pass

            # Send final run_status so frontend shows "completed"
            # Must happen BEFORE unregistering the WS callback
            try:
                await thought_logger.push_status(session_id)
            except Exception:
                pass

            thought_logger.unregister_ws_callback(session_id)

            # If client disconnected while graph was running, skip sending response
            # and skip DB persistence to avoid wasted writes
            if websocket.client_state.name != "CONNECTED":
                continue

            response = _build_response(result)
            await manager.send_json(session_id, response)

            # Persist conversation to database
            try:
                async with async_session_maker() as db:
                    await MemoryStoreSkill.save_conversation(
                        db=db,
                        session_id=session_id,
                        user_message=safe_content,
                        assistant_response=result.get("assistant_response") or "",
                        intent=result.get("intent"),
                    )
            except Exception:
                # Persistence failure should not break the chat flow
                pass

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(session_id)


def _build_response(state: dict) -> dict:
    """Build WebSocket response from final state."""
    return {
        "type": "message",
        "assistant_message": state.get("assistant_response") or "",
        "intent": state.get("intent"),
        "itinerary": state.get("current_itinerary"),
        "itinerary_status": state.get("itinerary_status"),
        "budget_panel": state.get("budget_panel"),
        "preference_panel": state.get("preference_panel"),
        "validation_result": state.get("validation_result"),
        "optimized_routes": state.get("optimized_routes"),
        "needs_clarification": state.get("needs_clarification", False),
        "waiting_for_confirmation": state.get("waiting_for_confirmation", False),
        "needs_replan": state.get("needs_replan", False),
    }
