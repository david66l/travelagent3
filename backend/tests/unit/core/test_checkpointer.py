"""Tests for checkpointer creation."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from core.checkpointer import create_checkpointer


class TestCreateCheckpointer:
    """Test Postgres checkpointer setup."""

    @pytest.mark.asyncio
    async def test_create_checkpointer(self):
        mock_conn = MagicMock()
        mock_checkpointer = MagicMock()
        mock_checkpointer.setup = AsyncMock()

        # AsyncPostgresSaver validates conn type in __init__, so we patch the whole class
        class MockSaver:
            def __init__(self, conn):
                pass
            setup = AsyncMock()

        with patch("psycopg.AsyncConnection.connect", AsyncMock(return_value=mock_conn)):
            with patch("core.checkpointer.AsyncPostgresSaver", MockSaver):
                result = await create_checkpointer()
                assert isinstance(result, MockSaver)
                MockSaver.setup.assert_called_once()
