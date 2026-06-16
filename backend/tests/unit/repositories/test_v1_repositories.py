"""Tests for v1 repositories."""

import pytest

from repositories.v1 import (
    UserRepository,
    UserProfileRepository,
    ConversationRepository,
    MessageRepository,
    ItineraryRepository,
    PlanningJobRepository,
)

pytestmark = pytest.mark.asyncio


class TestUserRepository:
    async def test_create_and_get(self, db):
        repo = UserRepository(db)
        user = await repo.create_user(email="test@example.com", role="user")
        assert user.id is not None
        assert user.email == "test@example.com"

        fetched = await repo.get_by_id(user.id)
        assert fetched is not None
        assert fetched.email == "test@example.com"

    async def test_get_by_email(self, db):
        repo = UserRepository(db)
        await repo.create_user(email="lookup@example.com", role="user")

        found = await repo.get_by_email("lookup@example.com")
        assert found is not None
        assert found.email == "lookup@example.com"

        not_found = await repo.get_by_email("missing@example.com")
        assert not_found is None


class TestUserProfileRepository:
    async def test_create_or_update(self, db):
        user_repo = UserRepository(db)
        user = await user_repo.create_user(email="profile@example.com", role="user")

        profile_repo = UserProfileRepository(db)
        profile = await profile_repo.create_or_update(
            user.id,
            personal={"interests": ["food"]},
            preferences={"pace": "relaxed"},
        )
        assert profile.user_id == user.id
        assert profile.personal["interests"] == ["food"]

        # Update
        updated = await profile_repo.create_or_update(
            user.id,
            preferences={"pace": "fast"},
        )
        assert updated.preferences["pace"] == "fast"
        assert updated.personal["interests"] == ["food"]  # unchanged


class TestConversationRepository:
    async def test_create_and_list(self, db):
        user_repo = UserRepository(db)
        user = await user_repo.create_user(email="conv@example.com", role="user")

        conv_repo = ConversationRepository(db)
        conv = await conv_repo.create_conversation(
            user.id,
            title="杭州3日游",
            state_snapshot={"destination": "杭州"},
        )
        assert conv.user_id == user.id
        assert conv.title == "杭州3日游"

        conversations = await conv_repo.get_by_user(user.id)
        assert len(conversations) == 1
        assert conversations[0].title == "杭州3日游"

    async def test_archive(self, db):
        user_repo = UserRepository(db)
        user = await user_repo.create_user(email="archive@example.com", role="user")

        conv_repo = ConversationRepository(db)
        conv = await conv_repo.create_conversation(user.id)

        rows = await conv_repo.archive(conv.id)
        assert rows == 1

        await db.refresh(conv)
        assert conv.status == "archived"
        assert conv.archived_at is not None


class TestMessageRepository:
    async def test_create_and_list(self, db):
        user_repo = UserRepository(db)
        user = await user_repo.create_user(email="msg@example.com", role="user")

        conv_repo = ConversationRepository(db)
        conv = await conv_repo.create_conversation(user.id)

        msg_repo = MessageRepository(db)
        msg = await msg_repo.create_message(
            conv.id,
            role="user",
            content="我想去杭州",
            token_count=5,
        )
        assert msg.conversation_id == conv.id
        assert msg.token_count == 5

        messages = await msg_repo.get_by_conversation(conv.id)
        assert len(messages) == 1
        assert messages[0].content == "我想去杭州"

    async def test_token_count(self, db):
        user_repo = UserRepository(db)
        user = await user_repo.create_user(email="tokens@example.com", role="user")

        conv_repo = ConversationRepository(db)
        conv = await conv_repo.create_conversation(user.id)

        msg_repo = MessageRepository(db)
        await msg_repo.create_message(conv.id, "user", "msg1", token_count=10)
        await msg_repo.create_message(conv.id, "assistant", "msg2", token_count=20)

        total = await msg_repo.get_total_token_count(conv.id)
        assert total == 30


class TestItineraryRepository:
    async def test_create_and_list(self, db):
        user_repo = UserRepository(db)
        user = await user_repo.create_user(email="itin@example.com", role="user")

        conv_repo = ConversationRepository(db)
        conv = await conv_repo.create_conversation(user.id)

        itin_repo = ItineraryRepository(db)
        itin = await itin_repo.create_itinerary(
            user_id=user.id,
            conversation_id=conv.id,
            destination="杭州",
            days=3,
            content={"days": []},
        )
        assert itin.destination == "杭州"
        assert itin.days == 3

        itineraries = await itin_repo.get_by_user(user.id)
        assert len(itineraries) == 1

    async def test_favorite(self, db):
        user_repo = UserRepository(db)
        user = await user_repo.create_user(email="fav@example.com", role="user")

        conv_repo = ConversationRepository(db)
        conv = await conv_repo.create_conversation(user.id)

        itin_repo = ItineraryRepository(db)
        itin = await itin_repo.create_itinerary(
            user_id=user.id,
            conversation_id=conv.id,
            destination="杭州",
            days=3,
        )

        rows = await itin_repo.set_favorite(itin.id, True)
        assert rows == 1

        await db.refresh(itin)
        assert itin.is_favorite is True


class TestPlanningJobRepository:
    async def test_create_and_get(self, db):
        user_repo = UserRepository(db)
        user = await user_repo.create_user(email="job@example.com", role="user")

        conv_repo = ConversationRepository(db)
        conv = await conv_repo.create_conversation(user.id)

        job_repo = PlanningJobRepository(db)
        job = await job_repo.create_job(
            conversation_id=conv.id,
            user_uuid=user.id,
            input_requirements={"destination": "杭州"},
        )
        assert job.conversation_id == conv.id
        assert job.user_uuid == user.id
        assert job.queue_name == "default"
        assert job.status == "pending"
        assert job.user_feedback.get("profile", {}).get("trip", {}).get("destination") == "杭州"

        fetched = await job_repo.get_by_id(job.id)
        assert fetched is not None
        assert fetched.id == job.id

    async def test_get_by_queue_and_status(self, db):
        user_repo = UserRepository(db)
        user = await user_repo.create_user(email="queue@example.com", role="user")

        conv_repo = ConversationRepository(db)
        conv = await conv_repo.create_conversation(user.id)

        job_repo = PlanningJobRepository(db)
        await job_repo.create_job(
            conversation_id=conv.id,
            user_uuid=user.id,
            queue_name="planning",
        )

        jobs = await job_repo.get_by_queue_and_status("planning", "pending")
        assert len(jobs) == 1

        jobs = await job_repo.get_by_queue_and_status("poi", "pending")
        assert len(jobs) == 0

    async def test_update_status(self, db):
        user_repo = UserRepository(db)
        user = await user_repo.create_user(email="status@example.com", role="user")

        conv_repo = ConversationRepository(db)
        conv = await conv_repo.create_conversation(user.id)

        job_repo = PlanningJobRepository(db)
        job = await job_repo.create_job(
            conversation_id=conv.id,
            user_uuid=user.id,
        )

        rows = await job_repo.update_status(
            job.id,
            "completed",
            result={"itinerary": {}},
            token_usage={"total": 100},
            latency_ms=1234,
        )
        assert rows == 1

        await db.refresh(job)
        assert job.status == "completed"
        assert job.result == {"itinerary": {}}
        assert job.token_usage == {"total": 100}
        assert job.latency_ms == 1234
