"""Direct unit tests for v1 route handlers.

FastAPI's TestClient does not reliably attribute coverage to the original
async endpoint functions in this project's pytest-asyncio setup.  Calling the
route handlers directly with real (or minimal) collaborators covers the
endpoint bodies without going through the ASGI transport layer.
"""

import json
from uuid import uuid4

import pytest

from api.v1.conversations import (
    archive_conversation,
    create_conversation,
    create_message,
    get_conversation,
    list_conversations,
    list_messages,
)
from api.v1.itineraries import create_itinerary, list_itineraries, set_favorite
from api.v1.planning_jobs import (
    create_planning_job,
    get_planning_job,
    update_planning_job_status,
)
from api.v1.schemas import (
    CreateConversationRequest,
    CreateItineraryRequest,
    CreateMessageRequest,
    CreatePlanningJobRequest,
    UpdateFavoriteRequest,
    UpdatePlanningJobRequest,
)
from api.v1.users import get_me, get_my_profile, get_user, update_my_profile
from core.exceptions import NotFoundException
from repositories.v1 import (
    ConversationRepository,
    ItineraryRepository,
    MessageRepository,
    PlanningJobRepository,
    UserProfileRepository,
    UserRepository,
)
from services import (
    ConversationService,
    ItineraryService,
    PlanningJobService,
    UserService,
)

pytestmark = pytest.mark.asyncio


def _body(response):
    """Decode a FastAPI JSONResponse body."""
    return json.loads(response.body)


async def _persist_guest(db):
    """Create and persist a guest user for FK-valid tests."""
    repo = UserRepository(db)
    return await repo.create_user(email=f"guest_{uuid4()}@local")


class TestUserRoutes:
    async def test_get_me(self):
        from models import User

        user = User(id=uuid4(), email="guest_test@local", role="guest")
        response = await get_me(user)
        assert response.status_code == 200
        assert _body(response)["data"]["id"] == str(user.id)

    async def test_get_my_profile_no_profile(self, db):
        user = await _persist_guest(db)
        service = UserService(UserRepository(db), UserProfileRepository(db))
        response = await get_my_profile(user, service)
        assert response.status_code == 200
        assert _body(response).get("data") is None

    async def test_get_my_profile_after_update(self, db):
        user = await _persist_guest(db)
        service = UserService(UserRepository(db), UserProfileRepository(db))
        await service.update_profile(user.id, preferences={"pace": "relaxed"})
        response = await get_my_profile(user, service)
        assert response.status_code == 200
        assert _body(response)["data"]["preferences"]["pace"] == "relaxed"

    async def test_update_my_profile(self, db):
        user = await _persist_guest(db)
        service = UserService(UserRepository(db), UserProfileRepository(db))
        body = type(
            "Body",
            (),
            {"personal": {}, "preferences": {"budget": 5000}, "frequent_destinations": []},
        )()
        response = await update_my_profile(body, user, service)
        assert response.status_code == 200
        assert _body(response)["data"]["preferences"]["budget"] == 5000

    async def test_get_user_found(self, db):
        user = await _persist_guest(db)
        service = UserService(UserRepository(db), UserProfileRepository(db))
        response = await get_user(user.id, service)
        assert response.status_code == 200
        assert _body(response)["data"]["id"] == str(user.id)

    async def test_get_user_not_found(self, db):
        service = UserService(UserRepository(db), UserProfileRepository(db))
        with pytest.raises(NotFoundException):
            await get_user(uuid4(), service)


class TestConversationRoutes:
    async def test_list_conversations_empty(self, db):
        user = await _persist_guest(db)
        service = ConversationService(ConversationRepository(db), MessageRepository(db))
        response = await list_conversations(
            status=None, limit=20, offset=0, user=user, service=service
        )
        assert response.status_code == 200
        assert _body(response)["data"] == []

    async def test_list_conversations_with_status(self, db):
        user = await _persist_guest(db)
        repo = ConversationRepository(db)
        await repo.create_conversation(user.id, title="Trip")
        service = ConversationService(repo, MessageRepository(db))
        response = await list_conversations(
            status="active", limit=10, offset=0, user=user, service=service
        )
        assert response.status_code == 200
        assert len(_body(response)["data"]) == 1

    async def test_create_conversation(self, db):
        user = await _persist_guest(db)
        service = ConversationService(ConversationRepository(db), MessageRepository(db))
        body = CreateConversationRequest(title="New", state_snapshot={"a": 1})
        response = await create_conversation(body, user, service)
        assert response.status_code == 201
        data = _body(response)["data"]
        assert data["title"] == "New"
        assert data["user_id"] == str(user.id)

    async def test_get_conversation_found(self, db):
        user = await _persist_guest(db)
        repo = ConversationRepository(db)
        conv = await repo.create_conversation(user.id, title="Found")
        service = ConversationService(repo, MessageRepository(db))
        response = await get_conversation(conv.id, user, service)
        assert response.status_code == 200
        assert _body(response)["data"]["id"] == str(conv.id)

    async def test_get_conversation_not_found(self, db):
        user = await _persist_guest(db)
        service = ConversationService(ConversationRepository(db), MessageRepository(db))
        with pytest.raises(NotFoundException):
            await get_conversation(uuid4(), user, service)

    async def test_get_conversation_cross_user(self, db):
        user_a = await _persist_guest(db)
        user_b = await _persist_guest(db)
        repo = ConversationRepository(db)
        conv = await repo.create_conversation(user_a.id, title="Private")
        service = ConversationService(repo, MessageRepository(db))
        with pytest.raises(NotFoundException):
            await get_conversation(conv.id, user_b, service)

    async def test_archive_conversation(self, db):
        user = await _persist_guest(db)
        repo = ConversationRepository(db)
        conv = await repo.create_conversation(user.id, title="Archive")
        service = ConversationService(repo, MessageRepository(db))
        response = await archive_conversation(conv.id, user, service)
        assert response.status_code == 200
        assert _body(response)["message"] == "Conversation archived"

    async def test_archive_conversation_not_found(self, db):
        user = await _persist_guest(db)
        service = ConversationService(ConversationRepository(db), MessageRepository(db))
        with pytest.raises(NotFoundException):
            await archive_conversation(uuid4(), user, service)

    async def test_list_messages(self, db):
        user = await _persist_guest(db)
        repo = ConversationRepository(db)
        conv = await repo.create_conversation(user.id, title="Msgs")
        msg_repo = MessageRepository(db)
        await msg_repo.create_message(conv.id, "user", "hello")
        service = ConversationService(repo, msg_repo)
        response = await list_messages(conv.id, 100, 0, user, service)
        assert response.status_code == 200
        assert len(_body(response)["data"]) == 1

    async def test_list_messages_not_found(self, db):
        user = await _persist_guest(db)
        service = ConversationService(ConversationRepository(db), MessageRepository(db))
        with pytest.raises(NotFoundException):
            await list_messages(uuid4(), 100, 0, user, service)

    async def test_create_message(self, db):
        user = await _persist_guest(db)
        repo = ConversationRepository(db)
        conv = await repo.create_conversation(user.id, title="Msg")
        service = ConversationService(repo, MessageRepository(db))
        body = CreateMessageRequest(role="user", content="hi", token_count=1)
        response = await create_message(conv.id, body, user, service)
        assert response.status_code == 201
        assert _body(response)["data"]["content"] == "hi"

    async def test_create_message_not_found(self, db):
        user = await _persist_guest(db)
        service = ConversationService(ConversationRepository(db), MessageRepository(db))
        body = CreateMessageRequest(role="user", content="hi")
        with pytest.raises(NotFoundException):
            await create_message(uuid4(), body, user, service)


class TestItineraryRoutes:
    async def test_list_itineraries(self, db):
        user = await _persist_guest(db)
        conv_repo = ConversationRepository(db)
        conv = await conv_repo.create_conversation(user.id, title="Itin")
        itin_repo = ItineraryRepository(db)
        await itin_repo.create_itinerary(
            user_id=user.id,
            conversation_id=conv.id,
            destination="杭州",
            days=3,
        )
        service = ItineraryService(itin_repo, conv_repo)
        response = await list_itineraries(limit=20, offset=0, user=user, service=service)
        assert response.status_code == 200
        assert len(_body(response)["data"]) == 1

    async def test_create_itinerary(self, db):
        user = await _persist_guest(db)
        conv_repo = ConversationRepository(db)
        conv = await conv_repo.create_conversation(user.id, title="Itin")
        service = ItineraryService(ItineraryRepository(db), conv_repo)
        body = CreateItineraryRequest(
            conversation_id=conv.id,
            destination="杭州",
            days=3,
            content={"day1": []},
        )
        response = await create_itinerary(body, user, service)
        assert response.status_code == 201
        assert _body(response)["data"]["destination"] == "杭州"

    async def test_set_favorite(self, db):
        user = await _persist_guest(db)
        conv_repo = ConversationRepository(db)
        conv = await conv_repo.create_conversation(user.id, title="Itin")
        itin_repo = ItineraryRepository(db)
        itin = await itin_repo.create_itinerary(
            user_id=user.id,
            conversation_id=conv.id,
            destination="杭州",
            days=3,
        )
        service = ItineraryService(itin_repo, conv_repo)
        body = UpdateFavoriteRequest(is_favorite=True)
        response = await set_favorite(itin.id, body, user, service)
        assert response.status_code == 200
        assert _body(response)["message"] == "Favorite status updated"

    async def test_set_favorite_not_found(self, db):
        user = await _persist_guest(db)
        conv_repo = ConversationRepository(db)
        service = ItineraryService(ItineraryRepository(db), conv_repo)
        body = UpdateFavoriteRequest(is_favorite=True)
        with pytest.raises(NotFoundException):
            await set_favorite(uuid4(), body, user, service)


class TestPlanningJobRoutes:
    async def test_create_planning_job(self, db):
        user = await _persist_guest(db)
        conv_repo = ConversationRepository(db)
        conv = await conv_repo.create_conversation(user.id, title="Job")
        service = PlanningJobService(PlanningJobRepository(db), conv_repo)
        body = CreatePlanningJobRequest(
            conversation_id=conv.id,
            input_requirements={"destination": "杭州"},
        )
        response = await create_planning_job(body, user, service)
        assert response.status_code == 201
        assert _body(response)["data"]["conversation_id"] == str(conv.id)

    async def test_get_planning_job(self, db):
        user = await _persist_guest(db)
        conv_repo = ConversationRepository(db)
        conv = await conv_repo.create_conversation(user.id, title="Job")
        service = PlanningJobService(PlanningJobRepository(db), conv_repo)
        job = await service.create(user_uuid=user.id, conversation_id=conv.id)
        response = await get_planning_job(job.id, user, service)
        assert response.status_code == 200
        assert _body(response)["data"]["id"] == job.id

    async def test_get_planning_job_not_found(self, db):
        user = await _persist_guest(db)
        service = PlanningJobService(PlanningJobRepository(db), ConversationRepository(db))
        with pytest.raises(NotFoundException):
            await get_planning_job("missing-id", user, service)

    async def test_update_planning_job_status(self, db):
        user = await _persist_guest(db)
        conv_repo = ConversationRepository(db)
        conv = await conv_repo.create_conversation(user.id, title="Job")
        service = PlanningJobService(PlanningJobRepository(db), conv_repo)
        job = await service.create(user_uuid=user.id, conversation_id=conv.id)
        body = UpdatePlanningJobRequest(status="completed")
        response = await update_planning_job_status(job.id, body, user, service)
        assert response.status_code == 200
        assert _body(response)["message"] == "Status updated"

    async def test_update_planning_job_status_not_found(self, db):
        user = await _persist_guest(db)
        service = PlanningJobService(PlanningJobRepository(db), ConversationRepository(db))
        body = UpdatePlanningJobRequest(status="completed")
        with pytest.raises(NotFoundException):
            await update_planning_job_status("missing-id", body, user, service)
