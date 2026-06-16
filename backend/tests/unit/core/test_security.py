"""Tests for core security utilities."""

from datetime import timedelta
from uuid import uuid4

import pytest

from core import security
from core.exceptions import UnauthorizedException


class TestPasswordHashing:
    def test_hash_and_verify(self):
        hashed = security.hash_password("secret123")
        assert security.verify_password("secret123", hashed) is True
        assert security.verify_password("wrong", hashed) is False

    def test_hash_is_different_each_time(self):
        h1 = security.hash_password("secret123")
        h2 = security.hash_password("secret123")
        assert h1 != h2


class TestTokens:
    def test_access_token_round_trip(self):
        user_id = uuid4()
        token = security.create_access_token(user_id, role="user")
        payload = security.decode_token(token)
        assert payload["sub"] == str(user_id)
        assert payload["role"] == "user"
        assert payload["type"] == "access"
        assert "exp" in payload

    def test_guest_token_contains_fingerprint(self):
        user_id = uuid4()
        token = security.create_guest_token(user_id, "fp-123")
        payload = security.decode_token(token)
        assert payload["type"] == "guest"
        assert payload["device_fingerprint"] == "fp-123"

    def test_expired_token_raises(self):
        user_id = uuid4()
        # Manually create an already-expired token by patching the clock used
        # inside create_access_token is hard; instead decode an expired token.
        expired = security.jwt.encode(
            {
                "sub": str(user_id),
                "role": "user",
                "type": "access",
                "exp": security._now() - timedelta(seconds=1),
                "iat": security._now() - timedelta(minutes=31),
            },
            security.settings.jwt_secret,
            algorithm=security.settings.jwt_algorithm,
        )
        with pytest.raises(UnauthorizedException):
            security.decode_token(expired)

    def test_invalid_token_raises(self):
        with pytest.raises(UnauthorizedException):
            security.decode_token("not-a-token")

    def test_malformed_token_missing_sub(self):
        token = security.jwt.encode(
            {"type": "access"},
            security.settings.jwt_secret,
            algorithm=security.settings.jwt_algorithm,
        )
        with pytest.raises(UnauthorizedException):
            security.decode_token(token)


class TestBlacklist:
    async def test_blacklist_blocks_token(self, mock_redis):
        user_id = uuid4()
        token = security.create_access_token(user_id)
        assert await security.is_token_blacklisted(token) is False
        await security.blacklist_token(token)
        # Simulate Redis now containing the blacklisted hash.
        mock_redis.get.return_value = "1"
        assert await security.is_token_blacklisted(token) is True
