"""FastAPI dependencies."""

from typing import AsyncGenerator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db


async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session."""
    async for session in get_db():
        yield session


def get_graph(request: Request):
    """Get compiled graph from app state."""
    return request.app.state.graph


def get_checkpointer(request: Request):
    """Get checkpointer from app state."""
    return request.app.state.checkpointer
