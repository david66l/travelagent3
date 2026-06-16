"""Tests for v1 application services."""

import pytest

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
    MessageService,
    PlanningJobService,
    UserService,
)

pytestmark = pytest.mark.asyncio


class TestUserService:
    async def test_get_or_create_guest(self, db):
        user_repo = UserRepository(db)
        profile_repo = UserProfileRepository(db)
        service = UserService(user_repo, profile_repo)

        user = await service.get_or_create_guest("device-1")
        assert user.role == "guest"
        assert user.email == "guest_device-1@local"

        same = await service.get_or_create_guest("device-1")
        assert same.id == user.id

    async def test_update_and_get_profile(self, db):
        user_repo = UserRepository(db)
        profile_repo = UserProfileRepository(db)
        service = UserService(user_repo, profile_repo)

        user = await user_repo.create_user(email="svc@example.com", role="user")
        profile = await service.update_profile(
            user.id,
            personal={"name": "Test"},
            preferences={"pace": "relaxed"},
        )
        assert profile.personal["name"] == "Test"

        fetched = await service.get_profile(user.id)
        assert fetched.preferences["pace"] == "relaxed"


class TestConversationService:
    async def test_create_and_list(self, db):
        user_repo = UserRepository(db)
        user = await user_repo.create_user(email="conv-svc@example.com", role="user")

        service = ConversationService(ConversationRepository(db), MessageRepository(db))
        conv = await service.create(user.id, title="Service Test")
        assert conv.user_id == user.id

        conversations = await service.list_by_user(user.id)
        assert len(conversations) == 1

    async def test_archive(self, db):
        user_repo = UserRepository(db)
        user = await user_repo.create_user(email="archive-svc@example.com", role="user")

        service = ConversationService(ConversationRepository(db), MessageRepository(db))
        conv = await service.create(user.id)
        ok = await service.archive(conv.id)
        assert ok is True

        fetched = await service.get(conv.id)
        assert fetched.status == "archived"

    async def test_add_message(self, db):
        user_repo = UserRepository(db)
        user = await user_repo.create_user(email="msg-svc@example.com", role="user")

        service = ConversationService(ConversationRepository(db), MessageRepository(db))
        conv = await service.create(user.id)
        message = await service.add_message(conv.id, "user", "hello", token_count=2)
        assert message.conversation_id == conv.id

        messages = await service.get_messages(conv.id)
        assert len(messages) == 1


class TestMessageService:
    async def test_create_rejects_missing_conversation(self, db):
        from uuid import uuid4

        service = MessageService(MessageRepository(db), ConversationRepository(db))
        with pytest.raises(ValueError):
            await service.create(uuid4(), "user", "hello")


class TestItineraryService:
    async def test_create_and_list(self, db):
        user_repo = UserRepository(db)
        user = await user_repo.create_user(email="itin-svc@example.com", role="user")
        conv_repo = ConversationRepository(db)
        conv = await conv_repo.create_conversation(user.id)

        service = ItineraryService(ItineraryRepository(db), conv_repo)
        itin = await service.create(
            user_id=user.id,
            conversation_id=conv.id,
            destination="杭州",
            days=3,
        )
        assert itin.destination == "杭州"

        listed = await service.list_by_user(user.id)
        assert len(listed) == 1

    async def test_set_favorite(self, db):
        user_repo = UserRepository(db)
        user = await user_repo.create_user(email="fav-svc@example.com", role="user")
        conv_repo = ConversationRepository(db)
        conv = await conv_repo.create_conversation(user.id)

        service = ItineraryService(ItineraryRepository(db), conv_repo)
        itin = await service.create(
            user_id=user.id,
            conversation_id=conv.id,
            destination="杭州",
            days=3,
        )
        ok = await service.set_favorite(itin.id, True)
        assert ok is True


class TestPlanningJobService:
    async def test_create_and_update(self, db):
        user_repo = UserRepository(db)
        user = await user_repo.create_user(email="job-svc@example.com", role="user")
        conv_repo = ConversationRepository(db)
        conv = await conv_repo.create_conversation(user.id)

        service = PlanningJobService(PlanningJobRepository(db), conv_repo)
        job = await service.create(
            user_uuid=user.id,
            conversation_id=conv.id,
            input_requirements={"destination": "杭州"},
        )
        assert job.status == "pending"

        ok = await service.update_status(job.id, "running")
        assert ok is True

        ok = await service.update_result(
            job.id, result={"itinerary": {}}, token_usage={"total": 10}
        )
        assert ok is True

        fetched = await service.get(job.id)
        assert fetched.result == {"itinerary": {}}
