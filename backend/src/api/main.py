"""FastAPI application entry point."""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend/src is on path (works regardless of project location)
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from core.database import init_db  # noqa: E402
from core.settings import settings  # noqa: E402
from core.redis_client import redis_client  # noqa: E402
from api.health import router as health_router  # noqa: E402
from api.websocket import router as ws_router  # noqa: E402
from api.v1 import v1_router  # noqa: E402
from middleware import setup_exception_handlers  # noqa: E402
from middleware.rate_limit import setup_rate_limit  # noqa: E402
from worker.planning_worker import start_worker, stop_worker  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    await init_db()
    await redis_client.connect()
    if settings.planning_executor == "embedded":
        await start_worker()

    yield

    if settings.planning_executor == "embedded":
        await stop_worker()
    await redis_client.disconnect()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="TravelAgent API",
        description="AI Travel Planning Agent",
        version="2.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting
    setup_rate_limit(app)

    # Global exception handlers
    setup_exception_handlers(app)

    # Routers
    app.include_router(health_router)
    app.include_router(ws_router)
    app.include_router(v1_router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
    )
