"""Unit coverage for the durable chat-to-PlanningJob handoff."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from starlette.requests import Request

from api.v1.chat import chat_message
from api.v1.schemas import ChatMessageRequest


@pytest.mark.asyncio
async def test_chat_message_commits_job_before_dispatch_and_returns_job_id():
    conversation_id = uuid4()
    user = MagicMock(id=uuid4(), role="guest")
    service = MagicMock()
    service.add_message = AsyncMock()
    parser = MagicMock()
    parser.parse_many = AsyncMock(return_value=[])
    repo = MagicMock()
    db = MagicMock()
    db.commit = AsyncMock()
    body = ChatMessageRequest(
        conversation_id=conversation_id,
        content="杭州三天",
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/chat/message",
            "headers": [],
            "client": ("127.0.0.1", 12345),
        }
    )

    with patch("api.v1.chat._ensure_conversation", new=AsyncMock()):
        with patch("api.v1.chat.RateLimitCostController") as controller_cls:
            controller_cls.return_value.check_request_allowed = AsyncMock()
            with patch(
                "api.v1.chat.create_chat_planning_job",
                new=AsyncMock(return_value="job-123"),
            ):
                with patch("api.v1.chat.enqueue_planning_job") as enqueue:
                    response = await chat_message(
                        request,
                        body,
                        user,
                        service,
                        parser,
                        repo,
                        db,
                    )

    assert response.status_code == 202
    assert json.loads(response.body)["data"]["job_id"] == "job-123"
    db.commit.assert_awaited_once()
    enqueue.assert_called_once_with("job-123")
