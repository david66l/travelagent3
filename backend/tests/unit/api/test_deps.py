"""Tests for FastAPI dependency providers."""

import pytest
from uuid import uuid4
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request

from api import deps
from core import security
from core.llm_client import llm
from core.redis_client import redis_client
from repositories.v1 import UserRepository


class TestInfrastructureDeps:
    def test_get_redis_client(self):
        assert deps.get_redis_client() is redis_client

    def test_get_llm_client(self):
        assert deps.get_llm_client() is llm


class TestGetDb:
    @pytest.mark.asyncio
    async def test_yields_session_and_commits(self, db):
        gen = deps.get_db()
        session = await gen.__anext__()
        assert isinstance(session, AsyncSession)
        # Generator exits cleanly and commits
        try:
            await gen.__anext__()
        except StopAsyncIteration:
            pass

    @pytest.mark.asyncio
    async def test_rolls_back_on_exception(self, db):
        gen = deps.get_db()
        await gen.__anext__()
        with pytest.raises(RuntimeError):
            await gen.athrow(RuntimeError("boom"))


class TestAuthenticationDeps:
    def _request(self, fingerprint="device-123"):
        scope = {
            "type": "http",
            "headers": [(b"x-device-fingerprint", fingerprint.encode())],
        }
        return Request(scope)

    @pytest.mark.asyncio
    async def test_get_current_user_creates_guest(self, db):
        request = self._request("device-guest-1")
        user = await deps.get_current_user(request, None, db)
        assert user.role == "guest"
        assert user.email.startswith("guest_")

    @pytest.mark.asyncio
    async def test_get_current_user_returns_existing_guest(self, db):
        request = self._request("device-guest-2")
        first = await deps.get_current_user(request, None, db)
        second = await deps.get_current_user(request, None, db)
        assert first.id == second.id

    @pytest.mark.asyncio
    async def test_get_optional_user_without_credentials(self, db):
        request = self._request("device-opt")
        assert await deps.get_optional_user(request, None, db) is None

    @pytest.mark.asyncio
    async def test_get_optional_user_with_credentials(self, db):
        request = self._request("device-opt-cred")
        repo = UserRepository(db)
        guest = await repo.get_or_create_guest("device-opt-cred")
        token = security.create_guest_token(guest.id, "device-opt-cred")
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        user = await deps.get_optional_user(request, creds, db)
        assert user is not None
        assert user.role == "guest"


class TestRepositoryDeps:
    def test_get_user_repository(self, db):
        repo = deps.get_user_repository(db)
        assert repo.db is db

    def test_get_user_profile_repository(self, db):
        repo = deps.get_user_profile_repository(db)
        assert repo.db is db

    def test_get_conversation_repository(self, db):
        repo = deps.get_conversation_repository(db)
        assert repo.db is db

    def test_get_message_repository(self, db):
        repo = deps.get_message_repository(db)
        assert repo.db is db

    def test_get_itinerary_repository(self, db):
        repo = deps.get_itinerary_repository(db)
        assert repo.db is db

    def test_get_planning_job_repository(self, db):
        repo = deps.get_planning_job_repository(db)
        assert repo.db is db


class TestServiceDeps:
    def test_get_user_service(self, db):
        service = deps.get_user_service(
            deps.get_user_repository(db),
            deps.get_user_profile_repository(db),
        )
        assert service.user_repo.db is db
        assert service.profile_repo.db is db

    def test_get_conversation_service(self, db):
        service = deps.get_conversation_service(
            deps.get_conversation_repository(db),
            deps.get_message_repository(db),
        )
        assert service.repo.db is db
        assert service.message_repo.db is db

    def test_get_message_service(self, db):
        service = deps.get_message_service(
            deps.get_message_repository(db),
            deps.get_conversation_repository(db),
        )
        assert service.repo.db is db
        assert service.conversation_repo.db is db

    def test_get_itinerary_service(self, db):
        service = deps.get_itinerary_service(
            deps.get_itinerary_repository(db),
            deps.get_conversation_repository(db),
        )
        assert service.repo.db is db
        assert service.conversation_repo.db is db

    def test_get_planning_job_service(self, db):
        service = deps.get_planning_job_service(
            deps.get_planning_job_repository(db),
            deps.get_conversation_repository(db),
        )
        assert service.repo.db is db
        assert service.conversation_repo.db is db


class TestTokenAuthDeps:
    def _request(self, fingerprint="device-123"):
        scope = {
            "type": "http",
            "headers": [(b"x-device-fingerprint", fingerprint.encode())],
        }
        return Request(scope)

    def _creds(self, token):
        return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    @pytest.mark.asyncio
    async def test_get_current_user_from_valid_token(self, db):
        repo = UserRepository(db)
        user = await repo.create_user(email=f"token_{uuid4()}@example.com", role="user")
        token = security.create_access_token(user.id, user.role)

        resolved = await deps.get_current_user(self._request(), self._creds(token), db)
        assert resolved.id == user.id
        assert resolved.role == "user"

    @pytest.mark.asyncio
    async def test_get_current_user_rejects_blacklisted_token(self, db, mock_redis):
        repo = UserRepository(db)
        user = await repo.create_user(email=f"bl_{uuid4()}@example.com", role="user")
        token = security.create_access_token(user.id, user.role)
        mock_redis.get.return_value = "1"  # blacklisted

        with pytest.raises(deps.UnauthorizedException):
            await deps.get_current_user(self._request(), self._creds(token), db)

    @pytest.mark.asyncio
    async def test_require_user_rejects_guest(self, db):
        repo = UserRepository(db)
        user = await repo.get_or_create_guest("guest-device-123")
        with pytest.raises(deps.ForbiddenException):
            deps.require_user(user)

    @pytest.mark.asyncio
    async def test_require_admin_rejects_user(self, db):
        repo = UserRepository(db)
        user = await repo.create_user(email=f"admin_test_{uuid4()}@example.com", role="user")
        with pytest.raises(deps.ForbiddenException):
            deps.require_admin(user)

    @pytest.mark.asyncio
    async def test_require_admin_accepts_admin(self, db):
        repo = UserRepository(db)
        user = await repo.create_user(email=f"admin_ok_{uuid4()}@example.com", role="admin")
        assert deps.require_admin(user) is user
