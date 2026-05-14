"""Tests for database utilities."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from core.database import get_db, init_db


class TestGetDB:
    """Test database session context manager."""

    @pytest.mark.asyncio
    async def test_get_db_yields_session(self):
        mock_session = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()

        # Mock the async context manager protocol
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("core.database.async_session_maker", return_value=mock_ctx):
            gen = get_db()
            session = await gen.__anext__()
            assert session is mock_session
            try:
                await gen.__anext__()
            except StopAsyncIteration:
                pass
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()


class TestInitDB:
    """Test database initialization."""

    @pytest.mark.asyncio
    async def test_init_db(self):
        mock_conn = MagicMock()
        mock_conn.run_sync = AsyncMock()

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_engine = MagicMock()
        mock_engine.begin = MagicMock(return_value=mock_ctx)

        with patch("core.database.engine", mock_engine):
            await init_db()
            mock_conn.run_sync.assert_called_once()
