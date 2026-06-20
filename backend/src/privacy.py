"""Privacy helpers: PII encryption, log masking, and user data deletion."""

import base64
import hashlib
import os
import re
from typing import Union

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy import delete

from core.redis_client import redis_client
from models import (
    Conversation,
    Itinerary,
    PlanningJob,
    PlanningLog,
    User,
    UserModificationLog,
    UserProfile,
    UserProfileVector,
    UserTripHistory,
)

# Regex patterns for common PII in log lines.  Order matters: longer/more
# specific patterns (ID card, bank card) are applied before shorter ones
# (mobile phone) to avoid partial masking.
_ID_CARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_BANK_CARD_RE = re.compile(r"(?<!\d)\d{16,19}(?!\d)")
_PHONE_RE = re.compile(r"1[3-9]\d{9}")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Models that contain a user_id column (or the users table itself) and should
# be purged when deleting all data for a single user.  Order respects foreign
# key dependencies so child rows are removed before parents.
_USER_DATA_MODELS = (
    UserProfileVector,
    UserProfile,
    UserTripHistory,
    UserModificationLog,
    PlanningLog,
    Itinerary,
    Conversation,
    PlanningJob,
    User,
)


class PrivacyGuard:
    """AES-256-GCM encryption for PII with URL-safe base64 tokens."""

    _NONCE_SIZE = 12
    _KEY_SIZE = 32

    def __init__(self, key: Union[str, bytes]) -> None:
        if isinstance(key, str):
            key = hashlib.sha256(key.encode("utf-8")).digest()
        if not isinstance(key, bytes):
            raise TypeError("Encryption key must be bytes or str")
        if len(key) != self._KEY_SIZE:
            raise ValueError(f"Encryption key must be {self._KEY_SIZE} bytes")
        self._key = key
        self._aesgcm = AESGCM(self._key)

    def encrypt_pii(self, plaintext: str) -> str:
        """Encrypt plaintext and return urlsafe base64 ``nonce + tag + ciphertext``."""
        if not isinstance(plaintext, str):
            raise TypeError("plaintext must be str")
        nonce = os.urandom(self._NONCE_SIZE)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        # ciphertext already has the auth tag appended
        token = base64.urlsafe_b64encode(nonce + ciphertext).rstrip(b"=").decode("ascii")
        return token

    def decrypt_pii(self, token: str) -> str:
        """Decrypt a token produced by ``encrypt_pii``."""
        if not isinstance(token, str):
            raise TypeError("token must be str")
        try:
            # Restore optional padding stripped during encryption.
            padded = token.encode("ascii") + b"=" * (-len(token) % 4)
            raw = base64.urlsafe_b64decode(padded)
        except Exception as exc:
            raise ValueError("Invalid encrypted token") from exc
        if len(raw) < self._NONCE_SIZE + 16:
            raise ValueError("Invalid encrypted token")
        nonce = raw[: self._NONCE_SIZE]
        ciphertext = raw[self._NONCE_SIZE :]
        try:
            plaintext = self._aesgcm.decrypt(nonce, ciphertext, None)
        except InvalidTag as exc:
            raise ValueError("Invalid encrypted token") from exc
        return plaintext.decode("utf-8")

    @staticmethod
    def mask_logs(text: str) -> str:
        """Mask PII in log text: Chinese IDs, bank cards, phones, emails."""
        if not isinstance(text, str):
            raise TypeError("text must be str")
        text = _ID_CARD_RE.sub("****", text)
        text = _BANK_CARD_RE.sub("****", text)
        text = _PHONE_RE.sub("****", text)
        text = _EMAIL_RE.sub("****", text)
        return text


async def delete_all_user_data(user_id: str, db_session) -> dict[str, int]:
    """Delete all database rows and Redis keys belonging to ``user_id``.

    Redis keys matching ``session:{user_id}:*`` and ``state:{user_id}:*`` are
    removed.  Returns a mapping of ``table_name -> deleted_row_count``.
    """
    counts: dict[str, int] = {}

    for model in _USER_DATA_MODELS:
        if model is User:
            stmt = delete(User).where(User.id == user_id)
        else:
            stmt = delete(model).where(model.user_id == user_id)
        result = await db_session.execute(stmt)
        counts[model.__tablename__] = getattr(result, "rowcount", 0) or 0

    await db_session.commit()

    for pattern in (f"session:{user_id}:*", f"state:{user_id}:*"):
        cursor = 0
        while True:
            cursor, keys = await redis_client.scan(cursor=cursor, match=pattern, count=100)
            for key in keys:
                await redis_client.delete(key)
            if cursor == 0:
                break

    return counts
