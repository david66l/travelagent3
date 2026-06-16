"""v1 API routers."""

from fastapi import APIRouter

from api.v1.auth import router as auth_router
from api.v1.chat import router as chat_router
from api.v1.conversations import router as conversations_router
from api.v1.dead_letters import router as dead_letters_router
from api.v1.itineraries import router as itineraries_router
from api.v1.planning_jobs import router as planning_jobs_router
from api.v1.users import router as users_router

v1_router = APIRouter(prefix="/api/v1")

v1_router.include_router(auth_router)
v1_router.include_router(chat_router)
v1_router.include_router(dead_letters_router)
v1_router.include_router(conversations_router)
v1_router.include_router(itineraries_router)
v1_router.include_router(planning_jobs_router)
v1_router.include_router(users_router)

__all__ = ["v1_router"]
