"""FastAPI application entry point."""

import asyncio
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend/src is on path (works regardless of project location)
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from core.database import init_db
from core.settings import settings
from core.checkpointer import create_checkpointer
from core.redis_client import redis_client
from graph.graph import build_graph
from api.routes import router as api_router
from api.websocket import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    # Startup
    await init_db()
    await redis_client.connect()
    checkpointer = await create_checkpointer()
    graph = build_graph().compile(checkpointer=checkpointer)

    app.state.checkpointer = checkpointer
    app.state.graph = graph

    yield

    # Shutdown: graceful close with timeout
    try:
        await asyncio.wait_for(checkpointer.conn.close(), timeout=5.0)
    except asyncio.TimeoutError:
        pass
    await redis_client.disconnect()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="TravelAgent API",
        description="AI Travel Planning Agent powered by LangGraph",
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

    # Routers
    app.include_router(api_router)
    app.include_router(ws_router)

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
