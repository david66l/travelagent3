"""Tests for v1 authentication endpoints."""

from uuid import uuid4


def _headers(fingerprint=None):
    headers = {}
    if fingerprint:
        headers["X-Device-Fingerprint"] = fingerprint
    return headers


class TestGuestAuth:
    def test_create_guest_token(self, client):
        fp = str(uuid4())
        response = client.post(
            "/api/v1/auth/guest",
            json={"device_fingerprint": fp},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["role"] == "guest"
        assert data["token_type"] == "bearer"
        assert "access_token" in data

    def test_guest_token_accesses_me(self, client):
        fp = str(uuid4())
        guest_response = client.post(
            "/api/v1/auth/guest",
            json={"device_fingerprint": fp},
        )
        token = guest_response.json()["data"]["access_token"]

        me_response = client.get(
            "/api/v1/users/me",
            headers={**_headers(fp), "Authorization": f"Bearer {token}"},
        )
        assert me_response.status_code == 200
        assert me_response.json()["data"]["role"] == "guest"

    def test_guest_token_device_mismatch(self, client):
        fp = str(uuid4())
        guest_response = client.post(
            "/api/v1/auth/guest",
            json={"device_fingerprint": fp},
        )
        token = guest_response.json()["data"]["access_token"]

        me_response = client.get(
            "/api/v1/users/me",
            headers={"X-Device-Fingerprint": "other-device", "Authorization": f"Bearer {token}"},
        )
        assert me_response.status_code == 403
        assert me_response.json()["error"]["code"] == "DEVICE_MISMATCH"


class TestRegisterLogin:
    def test_register_and_login(self, client):
        email = f"user_{uuid4()}@example.com"
        password = "secret123"

        register_response = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": password},
        )
        assert register_response.status_code == 201
        tokens = register_response.json()["data"]
        assert tokens["role"] == "user"
        assert "access_token" in tokens
        assert "refresh_token" in tokens

        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": password},
        )
        assert login_response.status_code == 200
        assert login_response.json()["data"]["role"] == "user"

    def test_login_wrong_password(self, client):
        email = f"user_{uuid4()}@example.com"
        password = "secret123"
        client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": password},
        )

        response = client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": "wrong"},
        )
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHORIZED"

    def test_register_duplicate_email(self, client):
        email = f"user_{uuid4()}@example.com"
        password = "secret123"
        client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": password},
        )
        response = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": "another123"},
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "CONFLICT"


class TestRefresh:
    def test_refresh_token(self, client):
        email = f"user_{uuid4()}@example.com"
        password = "secret123"
        register_response = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": password},
        )
        refresh_token = register_response.json()["data"]["refresh_token"]

        response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["role"] == "user"
        assert "access_token" in data
        assert data.get("refresh_token") is not None
        assert data["refresh_token"] != refresh_token


class TestUpgrade:
    def test_upgrade_guest_to_user(self, client):
        fp = str(uuid4())
        guest_response = client.post(
            "/api/v1/auth/guest",
            json={"device_fingerprint": fp},
        )
        guest_token = guest_response.json()["data"]["access_token"]

        upgrade_response = client.post(
            "/api/v1/auth/upgrade",
            headers={"Authorization": f"Bearer {guest_token}"},
            json={
                "email": f"upgraded_{uuid4()}@example.com",
                "password": "newpass123",
            },
        )
        assert upgrade_response.status_code == 200
        data = upgrade_response.json()["data"]
        assert data["role"] == "user"
        assert "refresh_token" in data

        # New access token should identify as user.
        new_token = data["access_token"]
        me_response = client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {new_token}"},
        )
        assert me_response.status_code == 200
        assert me_response.json()["data"]["role"] == "user"


class TestLogout:
    async def test_logout_revokes_token(self, client, mock_redis):
        email = f"user_{uuid4()}@example.com"
        password = "secret123"
        register_response = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": password},
        )
        access_token = register_response.json()["data"]["access_token"]

        logout_response = client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert logout_response.status_code == 200

        # Simulate Redis containing the blacklisted token hash.
        mock_redis.get.return_value = "1"
        me_response = client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert me_response.status_code == 401
        assert me_response.json()["error"]["code"] == "TOKEN_REVOKED"
