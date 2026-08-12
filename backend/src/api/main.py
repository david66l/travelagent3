"""FastAPI application entry point."""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Ensure backend/src is on path (works regardless of project location)
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from core.database import init_db  # noqa: E402
from core.settings import settings  # noqa: E402
from core.redis_client import redis_cache_client, redis_client  # noqa: E402
from core.redlock import redlock  # noqa: E402
from api.health import router as health_router  # noqa: E402
from api.websocket import router as ws_router  # noqa: E402
from api.v1 import v1_router  # noqa: E402
from middleware import setup_exception_handlers  # noqa: E402
from middleware.metrics_middleware import MetricsMiddleware  # noqa: E402
from middleware.request_context import RequestContextMiddleware  # noqa: E402
from core.tracing import setup_tracing  # noqa: E402
from worker.planning_worker import start_worker, stop_worker  # noqa: E402
from graph.graph import build_graph, set_graph  # noqa: E402
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    await init_db()

    # LangSmith tracing — auto-instruments all LangGraph/LangChain calls
    if settings.langsmith_api_key:
        import os as _os

        _os.environ["LANGSMITH_TRACING"] = "true"
        _os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        _os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        _os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        logger.info(
            "LangSmith tracing enabled: project=%s endpoint=%s",
            settings.langsmith_project,
            settings.langsmith_endpoint,
        )

    # LangGraph persistent checkpoints. The async context manager must stay open
    # for the whole application lifetime so the psycopg connection pool remains
    # valid. See: https://langchain-ai.github.io/langgraph/concepts/persistence/
    async with AsyncPostgresSaver.from_conn_string(settings.database_url_sync) as checkpointer:
        await checkpointer.setup()  # idempotent: creates checkpoint tables
        graph = build_graph(checkpointer=checkpointer)
        app.state.graph = graph
        set_graph(graph)
        logger.info("TravelAgent graph compiled with AsyncPostgresSaver")

        await redis_client.connect()
        await redis_cache_client.connect()
        await redlock.connect()
        if settings.planning_executor == "embedded":
            await start_worker()

        # Preload the 1.3GB BGE embedding model in the background so the first
        # planning request doesn't pay the ~10s cold-load cost inside RAG retrieval.
        from data.embedding import warmup_embedder

        warmup_task = asyncio.create_task(warmup_embedder())

        yield

        warmup_task.cancel()

        if settings.planning_executor == "embedded":
            await stop_worker()
        await redis_client.disconnect()
        await redis_cache_client.disconnect()
        await redlock.disconnect()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="TravelAgent API",
        description="AI Travel Planning Agent",
        version="2.0.0",
        lifespan=lifespan,
    )

    # Request context + metrics (M5)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RequestContextMiddleware)

    # Global exception handlers
    setup_exception_handlers(app)

    # Routers
    app.include_router(health_router)
    app.include_router(ws_router)
    app.include_router(v1_router)

    setup_tracing(app)

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
