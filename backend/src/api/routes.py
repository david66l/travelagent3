"""REST API routes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import db_session, get_graph
from core.input_guard import sanitize_user_input
from core.state import ItineraryState
from schemas import ChatResponse
from skills.memory_store import MemoryStoreSkill

router = APIRouter(prefix="/api")


class ChatRequestBody(BaseModel):
    content: str
    session_id: Optional[str] = None
    user_id: Optional[str] = "anonymous"


class CreateSessionResponse(BaseModel):
    session_id: str
    message: str = "Session created"


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequestBody,
    graph=Depends(get_graph),
    db: AsyncSession = Depends(db_session),
) -> ChatResponse:
    """Process a chat message via REST API (non-WebSocket fallback)."""
    session_id = body.session_id or _generate_session_id()
    config = {"configurable": {"thread_id": session_id}}

    # Sanitize user input
    safe_content = sanitize_user_input(body.content)

    input_state = ItineraryState(
        session_id=session_id,
        user_id=body.user_id or "anonymous",
        user_input=safe_content,
        messages=[{"role": "user", "content": safe_content}],
    )

    result = await graph.ainvoke(input_state, config=config)

    # Persist conversation to database
    try:
        await MemoryStoreSkill.save_conversation(
            db=db,
            session_id=session_id,
            user_message=safe_content,
            assistant_response=result.get("assistant_response") or "",
            intent=result.get("intent"),
        )
    except Exception:
        # Persistence failure should not break the response
        pass

    return ChatResponse(
        assistant_message=result.get("assistant_response") or "",
        itinerary=result.get("current_itinerary"),
        budget_panel=result.get("budget_panel"),
        preference_panel=result.get("preference_panel"),
        intent=result.get("intent"),
        needs_clarification=result.get("needs_clarification", False),
        waiting_for_confirmation=result.get("waiting_for_confirmation", False),
    )


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session() -> CreateSessionResponse:
    """Create a new chat session."""
    return CreateSessionResponse(session_id=_generate_session_id())


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "service": "travel-agent"}


def _generate_session_id() -> str:
    import uuid
    return str(uuid.uuid4())
