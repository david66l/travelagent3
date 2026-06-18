"""Authorization bypass regression tests (PRD §13.5)."""

import pytest
from uuid import uuid4

from core.exceptions import NotFoundException
from api.v1.chat import _ensure_conversation


@pytest.mark.security
@pytest.mark.asyncio
async def test_ensure_conversation_rejects_other_user():
    user_id = uuid4()
    other_user_id = uuid4()
    conversation_id = uuid4()

    class FakeConversation:
        user_id = other_user_id

    class FakeService:
        async def get(self, cid):
            if cid == conversation_id:
                return FakeConversation()
            return None

    class FakeUser:
        id = user_id

    with pytest.raises(NotFoundException):
        await _ensure_conversation(conversation_id, FakeUser(), FakeService())
