from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from starlette.requests import Request

from api.v1 import agent_chat


@pytest.mark.asyncio
async def test_agent_stream_uses_graph_runner() -> None:
    user_id = uuid4()
    conversation_id = uuid4()
    service = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(user_id=user_id)))
    user = SimpleNamespace(id=user_id, role="guest")
    request = Request({"type": "http", "headers": []})
    calls: list[dict[str, object]] = []

    async def fake_stream(user_input: str, **kwargs: object):
        calls.append({"user_input": user_input, **kwargs})
        yield {"type": "thinking", "stage": "plan"}

    with (
        patch.object(agent_chat.redis_client, "get_json", new=AsyncMock(return_value=None)),
        patch.object(agent_chat.graph_runner, "stream", new=fake_stream),
    ):
        response = await agent_chat.agent_stream(
            request=request,
            conversation_id=conversation_id,
            timeout=60,
            user=user,
            service=service,
        )
        body = "".join([chunk async for chunk in response.body_iterator])

    assert calls == [
        {
            "user_input": "",
            "session_id": str(conversation_id),
            "user_id": str(user_id),
            "user_role": "guest",
            "messages": [],
            "profile": {},
        }
    ]
    assert "event: node" in body
    assert "event: done" in body
