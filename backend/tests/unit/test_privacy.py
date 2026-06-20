"""Unit tests for the privacy module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from privacy import PrivacyGuard, delete_all_user_data


class TestPrivacyGuard:
    def test_encrypt_decrypt_roundtrip_with_string_key(self):
        guard = PrivacyGuard("my-secret-key")
        plaintext = "user@example.com"
        token = guard.encrypt_pii(plaintext)

        assert token != plaintext
        assert isinstance(token, str)
        # URL-safe base64 should not contain standard base64 padding chars
        assert "=" not in token
        assert guard.decrypt_pii(token) == plaintext

    def test_encrypt_decrypt_roundtrip_with_bytes_key(self):
        key = b"\x00" * 32
        guard = PrivacyGuard(key)
        plaintext = "sensitive PII"
        token = guard.encrypt_pii(plaintext)

        assert guard.decrypt_pii(token) == plaintext
        # Two encryptions of the same plaintext must yield different tokens
        # because of the random nonce.
        token2 = guard.encrypt_pii(plaintext)
        assert token2 != token

    def test_decrypt_invalid_token_raises(self):
        guard = PrivacyGuard("another-key")
        with pytest.raises(ValueError):
            guard.decrypt_pii("not-valid-base64!!!")

    def test_key_derivation_normalizes_string_to_32_bytes(self):
        guard = PrivacyGuard("short")
        assert isinstance(guard._key, bytes)
        assert len(guard._key) == 32


class TestMaskLogs:
    def test_masks_mobile_phone(self):
        text = "请联系13800138000"
        assert PrivacyGuard.mask_logs(text) == "请联系****"

    def test_masks_email(self):
        text = "Send to alice.test+tag@example.com.cn please"
        assert PrivacyGuard.mask_logs(text) == "Send to **** please"

    def test_masks_id_card(self):
        text = "身份证110101199001011234"
        assert PrivacyGuard.mask_logs(text) == "身份证****"

    def test_masks_id_card_with_x(self):
        text = "身份证11010119900101123X"
        assert PrivacyGuard.mask_logs(text) == "身份证****"

    def test_masks_bank_card(self):
        text = "银行卡6222021234567890123"
        assert PrivacyGuard.mask_logs(text) == "银行卡****"

    def test_masks_multiple_pii_pieces(self):
        text = (
            "联系13800138000或 test@example.com，"
            "身份证110101199001011234，银行卡6222021234567890123"
        )
        masked = PrivacyGuard.mask_logs(text)
        assert "13800138000" not in masked
        assert "test@example.com" not in masked
        assert "110101199001011234" not in masked
        assert "6222021234567890123" not in masked
        assert masked.count("****") == 4


class TestDeleteAllUserData:
    async def test_deletes_rows_and_redis_keys(self, monkeypatch):
        user_id = "user-123"
        session_keys = [f"session:{user_id}:a", f"session:{user_id}:b"]
        state_keys = [f"state:{user_id}:c"]
        all_redis_keys = session_keys + state_keys

        def _scan_side_effect(cursor, match, count):
            if match == f"session:{user_id}:*":
                return (0, session_keys)
            if match == f"state:{user_id}:*":
                return (0, state_keys)
            return (0, [])

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock(rowcount=1))
        mock_session.commit = AsyncMock()

        import privacy

        mock_redis = AsyncMock()
        mock_redis.scan = AsyncMock(side_effect=_scan_side_effect)
        mock_redis.delete = AsyncMock()
        monkeypatch.setattr(privacy, "redis_client", mock_redis)

        counts = await delete_all_user_data(user_id, mock_session)

        assert mock_session.execute.call_count == 9
        assert mock_session.commit.called

        # Each tracked table should report its deleted row count.
        expected_tables = {
            "user_profile_vectors",
            "user_profiles",
            "user_trip_history",
            "user_modification_log",
            "planning_log",
            "itineraries",
            "conversations",
            "planning_jobs",
            "users",
        }
        assert set(counts.keys()) == expected_tables
        assert all(count == 1 for count in counts.values())

        mock_redis.scan.assert_any_call(cursor=0, match=f"session:{user_id}:*", count=100)
        mock_redis.scan.assert_any_call(cursor=0, match=f"state:{user_id}:*", count=100)
        assert mock_redis.delete.call_count == len(all_redis_keys)
        for key in all_redis_keys:
            mock_redis.delete.assert_any_call(key)
