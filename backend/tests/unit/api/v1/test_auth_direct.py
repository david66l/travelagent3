"""Direct async tests for auth endpoint functions.

pytest-cov under-reports async endpoint bodies when routes are exercised
through TestClient.  These tests call the endpoint functions directly with
injected real services to record full coverage.
"""

import json

import pytest
from fastapi.security import HTTPAuthorizationCredentials

from api.v1 import auth
from api.v1.schemas import (
    GuestTokenRequest,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    UpgradeRequest,
)
from core import security
from repositories.v1 import UserRepository, UserProfileRepository
from services.user_service import UserService


def _creds(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def _body(response):
    return json.loads(response.body)


def _service(db):
    return UserService(UserRepository(db), UserProfileRepository(db))


class TestGuestToken:
    @pytest.mark.asyncio
    async def test_create_guest_token(self, db):
        service = _service(db)
        response = await auth.create_guest_token(
            GuestTokenRequest(device_fingerprint="fp-direct"),
            service=service,
        )
        body = _body(response)
        assert body["data"]["role"] == "guest"
        assert body["data"]["token_type"] == "bearer"


class TestRegister:
    @pytest.mark.asyncio
    async def test_register(self, db):
        service = _service(db)
        response = await auth.register(
            RegisterRequest(email="direct@example.com", password="secret123"),
            service=service,
        )
        body = _body(response)
        assert response.status_code == 201
        assert body["data"]["role"] == "user"
        assert "refresh_token" in body["data"]


class TestLogin:
    @pytest.mark.asyncio
    async def test_login_by_email(self, db):
        service = _service(db)
        await service.register_user(email="login@example.com", password="secret123")
        response = await auth.login(
            LoginRequest(email="login@example.com", password="secret123"),
            service=service,
        )
        body = _body(response)
        assert response.status_code == 200
        assert body["data"]["role"] == "user"

    @pytest.mark.asyncio
    async def test_login_by_phone(self, db):
        service = _service(db)
        await service.register_user(
            email="phone@example.com", phone="13800138000", password="secret123"
        )
        response = await auth.login(
            LoginRequest(phone="13800138000", password="secret123"),
            service=service,
        )
        body = _body(response)
        assert response.status_code == 200
        assert body["data"]["role"] == "user"

    @pytest.mark.asyncio
    async def test_login_missing_email_and_phone(self, db):
        service = _service(db)
        with pytest.raises(Exception):
            await auth.login(
                LoginRequest(password="secret123"),
                service=service,
            )


class TestRefresh:
    @pytest.mark.asyncio
    async def test_refresh(self, db):
        service = _service(db)
        user = await service.register_user(email="refresh@example.com", password="secret123")
        refresh = security.create_refresh_token(user.id, user.role)
        response = await auth.refresh(
            RefreshRequest(refresh_token=refresh),
            service=service,
        )
        body = _body(response)
        assert response.status_code == 200
        assert body["data"]["role"] == "user"
        assert body["data"].get("refresh_token") is None

    @pytest.mark.asyncio
    async def test_refresh_with_access_token_fails(self, db):
        service = _service(db)
        user = await service.register_user(email="refresh2@example.com", password="secret123")
        access = security.create_access_token(user.id, user.role)
        with pytest.raises(Exception):
            await auth.refresh(
                RefreshRequest(refresh_token=access),
                service=service,
            )


class TestUpgrade:
    @pytest.mark.asyncio
    async def test_upgrade(self, db):
        service = _service(db)
        guest = await service.user_repo.get_or_create_guest("upgrade-direct")
        token = security.create_guest_token(guest.id, "upgrade-direct")
        response = await auth.upgrade(
            UpgradeRequest(email="upgraded@example.com", password="newpass123"),
            credentials=_creds(token),
            service=service,
        )
        body = _body(response)
        assert response.status_code == 200
        assert body["data"]["role"] == "user"

    @pytest.mark.asyncio
    async def test_upgrade_requires_guest_token(self, db):
        service = _service(db)
        user = await service.register_user(email="reg@example.com", password="secret123")
        token = security.create_access_token(user.id, user.role)
        with pytest.raises(Exception):
            await auth.upgrade(
                UpgradeRequest(email="up2@example.com", password="newpass123"),
                credentials=_creds(token),
                service=service,
            )

    @pytest.mark.asyncio
    async def test_upgrade_missing_credentials(self, db):
        service = _service(db)
        with pytest.raises(Exception):
            await auth.upgrade(
                UpgradeRequest(email="up3@example.com", password="newpass123"),
                credentials=None,
                service=service,
            )


class TestLogout:
    @pytest.mark.asyncio
    async def test_logout(self, db):
        service = _service(db)
        user = await service.register_user(email="logout@example.com", password="secret123")
        token = security.create_access_token(user.id, user.role)
        response = await auth.logout(
            credentials=_creds(token),
            service=service,
        )
        body = _body(response)
        assert response.status_code == 200
        assert body["message"] == "Logged out successfully"

    @pytest.mark.asyncio
    async def test_logout_missing_credentials(self, db):
        service = _service(db)
        with pytest.raises(Exception):
            await auth.logout(credentials=None, service=service)
