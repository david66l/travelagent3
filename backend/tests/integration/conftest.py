"""Shared fixtures for integration tests."""

import pytest_asyncio
from sqlalchemy import delete
from sqlalchemy.pool import NullPool

import core.database as database
from core.database import Base, async_session_maker, reset_engine
from models.planning_job import PlanningJob, PlanningJobEvent


@pytest_asyncio.fixture(scope="function")
async def db():
    """Provide a real DB session through the app's session maker.

    Tests and production code under test must share the same
    `core.database.async_session_maker` proxy.  Using NullPool keeps
    asyncpg connections from being reused across pytest function event
    loops.
    """
    await reset_engine(poolclass=NullPool)

    async with database.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Pre-test cleanup (in case previous test failed)
    async with async_session_maker() as cleanup_session:
        await cleanup_session.execute(delete(PlanningJobEvent))
        await cleanup_session.execute(delete(PlanningJob))
        await cleanup_session.commit()

    async with async_session_maker() as session:
        yield session

    await database.engine.dispose()
