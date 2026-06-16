"""Lightweight service endpoints."""

import uuid

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api")


class CreateSessionResponse(BaseModel):
    session_id: str
    message: str = "Session created"


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session() -> CreateSessionResponse:
    """Create a new chat session id."""
    return CreateSessionResponse(session_id=str(uuid.uuid4()))


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint used by Docker and CI."""
    return {"status": "ok", "service": "travel-agent"}
