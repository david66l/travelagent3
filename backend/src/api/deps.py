"""FastAPI dependency injection providers.

All business-layer dependencies are exposed here so routers and services
receive their collaborators through explicit injection rather than global
singletons.  This makes the codebase testable and allows the same service
logic to run inside HTTP requests, Celery workers, or management scripts.
"""

from typing import AsyncGenerator, Optional

from fastapi import Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import async_session_maker
from core.exceptions import ForbiddenException, UnauthorizedException
from core.llm_client import LLMClient, llm as global_llm
from core.redis_client import RedisClient, redis_client as global_redis
from core.security import (
    decode_token,
    is_token_blacklisted,
    is_user_banned,
)
from models import User
from repositories.v1 import (
    ConversationRepository,
    ItineraryRepository,
    MessageRepository,
    PlanningJobRepository,
    UserProfileRepository,
    UserRepository,
)
from services.conversation_service import ConversationService
from services.itinerary_service import ItineraryService
from services.message_service import MessageService
from services.planning_job_service import PlanningJobService
from services.user_service import UserService

# --------------------------------------------------------------------------- #
# Infrastructure dependencies
# --------------------------------------------------------------------------- #


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async DB session and handle commit/rollback/close."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_redis_client() -> RedisClient:
    """Provide the shared Redis client.

    In the future this can be replaced by a connection-pool aware client or
    a cluster client without changing callers.
    """
    return global_redis


def get_llm_client() -> LLMClient:
    """Provide the shared LLM client."""
    return global_llm


# --------------------------------------------------------------------------- #
# Authentication placeholder
# --------------------------------------------------------------------------- #

security = HTTPBearer(auto_error=False)


async def _resolve_token_user(
    token: str,
    request: Request,
    db: AsyncSession,
) -> User:
    """Decode a JWT, validate it, and load the corresponding user."""
    if await is_token_blacklisted(token):
        raise UnauthorizedException("Token has been revoked", code="TOKEN_REVOKED")

    payload = decode_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise UnauthorizedException("Malformed token", code="TOKEN_INVALID")

    if await is_user_banned(user_id):
        raise UnauthorizedException("Token has been revoked", code="TOKEN_REVOKED")

    repo = UserRepository(db)
    user = await repo.get_by_id(user_id)
    if user is None:
        raise UnauthorizedException("User not found", code="TOKEN_INVALID")

    # Guest tokens are bound to a device fingerprint.
    if payload.get("type") == "guest" or payload.get("role") == "guest":
        expected = payload.get("device_fingerprint")
        actual = request.headers.get("X-Device-Fingerprint")
        if expected and expected != actual:
            raise ForbiddenException("Device mismatch", code="DEVICE_MISMATCH")

    return user


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Resolve the current user.

    If a Bearer token is provided it is decoded and validated.  If no token is
    provided, a guest user is created or fetched from the device fingerprint
    for backward compatibility with P2 clients.
    """
    if credentials is not None and credentials.credentials:
        return await _resolve_token_user(credentials.credentials, request, db)

    device_fingerprint = request.headers.get("X-Device-Fingerprint", "anonymous")
    repo = UserRepository(db)
    return await repo.get_or_create_guest(device_fingerprint)


async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """Optional user resolution.  Returns None when no token is provided.

    Invalid tokens raise UnauthorizedException rather than returning None.
    """
    if credentials is None or not credentials.credentials:
        return None
    return await _resolve_token_user(credentials.credentials, request, db)


def require_user(user: User = Depends(get_current_user)) -> User:
    """Require an authenticated non-guest user."""
    if user.role == "guest":
        raise ForbiddenException("请先登录", code="AUTH_MISSING")
    return user


def require_admin(user: User = Depends(require_user)) -> User:
    """Require an admin user."""
    if user.role != "admin":
        raise ForbiddenException("Admin access denied")
    return user


# --------------------------------------------------------------------------- #
# Repository dependencies
# --------------------------------------------------------------------------- #


def get_user_repository(db: AsyncSession = Depends(get_db)) -> UserRepository:
    return UserRepository(db)


def get_user_profile_repository(
    db: AsyncSession = Depends(get_db),
) -> UserProfileRepository:
    return UserProfileRepository(db)


def get_conversation_repository(
    db: AsyncSession = Depends(get_db),
) -> ConversationRepository:
    return ConversationRepository(db)


def get_message_repository(
    db: AsyncSession = Depends(get_db),
) -> MessageRepository:
    return MessageRepository(db)


def get_itinerary_repository(
    db: AsyncSession = Depends(get_db),
) -> ItineraryRepository:
    return ItineraryRepository(db)


def get_planning_job_repository(
    db: AsyncSession = Depends(get_db),
) -> PlanningJobRepository:
    return PlanningJobRepository(db)


# --------------------------------------------------------------------------- #
# Service dependencies
# --------------------------------------------------------------------------- #


def get_user_service(
    repo: UserRepository = Depends(get_user_repository),
    profile_repo: UserProfileRepository = Depends(get_user_profile_repository),
) -> UserService:
    return UserService(repo, profile_repo)


def get_conversation_service(
    repo: ConversationRepository = Depends(get_conversation_repository),
    message_repo: MessageRepository = Depends(get_message_repository),
) -> ConversationService:
    return ConversationService(repo, message_repo)


def get_message_service(
    repo: MessageRepository = Depends(get_message_repository),
    conversation_repo: ConversationRepository = Depends(get_conversation_repository),
) -> MessageService:
    return MessageService(repo, conversation_repo)


def get_itinerary_service(
    repo: ItineraryRepository = Depends(get_itinerary_repository),
    conversation_repo: ConversationRepository = Depends(get_conversation_repository),
) -> ItineraryService:
    return ItineraryService(repo, conversation_repo)


def get_planning_job_service(
    repo: PlanningJobRepository = Depends(get_planning_job_repository),
    conversation_repo: ConversationRepository = Depends(get_conversation_repository),
) -> PlanningJobService:
    return PlanningJobService(repo, conversation_repo)
