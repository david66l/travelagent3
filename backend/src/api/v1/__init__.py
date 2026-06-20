"""v1 API routers."""

from fastapi import APIRouter

from api.v1.agent_chat import router as agent_router
from api.v1.auth import router as auth_router
from api.v1.bookings import router as bookings_router
from api.v1.chat import router as chat_router
from api.v1.conversations import router as conversations_router
from api.v1.dead_letters import router as dead_letters_router
from api.v1.analytics import router as analytics_router
from api.v1.downloads import router as downloads_router
from api.v1.itineraries import router as itineraries_router
from api.v1.metrics import router as metrics_router
from api.v1.planning_jobs import router as planning_jobs_router
from api.v1.users import router as users_router

from api.v1.webhooks import router as webhooks_router

v1_router = APIRouter(prefix="/api/v1")

v1_router.include_router(agent_router)
v1_router.include_router(auth_router)
v1_router.include_router(metrics_router)
v1_router.include_router(chat_router)
v1_router.include_router(dead_letters_router)
v1_router.include_router(downloads_router)
v1_router.include_router(analytics_router)
v1_router.include_router(conversations_router)
v1_router.include_router(itineraries_router)
v1_router.include_router(planning_jobs_router)
v1_router.include_router(users_router)
v1_router.include_router(bookings_router)
v1_router.include_router(webhooks_router)

__all__ = ["v1_router"]
