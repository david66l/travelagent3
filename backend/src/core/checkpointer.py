"""Async PostgresSaver for LangGraph state persistence."""

import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from core.settings import settings


async def create_checkpointer() -> AsyncPostgresSaver:
    """Create and setup AsyncPostgresSaver for LangGraph state persistence."""
    db_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
    conn = await psycopg.AsyncConnection.connect(db_url, autocommit=True)
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()
    return checkpointer
