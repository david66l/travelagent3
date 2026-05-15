from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import Pool

from .settings import settings


def _create_engine(*, poolclass: type[Pool] | None = None):
    kwargs = {
        "echo": settings.debug,
        "future": True,
        "pool_pre_ping": True,
    }
    if poolclass is not None:
        kwargs["poolclass"] = poolclass
    else:
        kwargs["pool_size"] = 20
        kwargs["max_overflow"] = 10

    return create_async_engine(
        settings.database_url,
        **kwargs,
    )


class _SessionMakerProxy:
    """Proxy object so that `reset_engine()` updates all references.

    Modules that import `async_session_maker` via::

        from core.database import async_session_maker

    hold a reference to this proxy.  When `reset_engine()` replaces
    the inner maker all those references automatically see the new
    engine's session factory — no import rewiring needed.
    """

    def __init__(self, eng):
        self._maker = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False)

    def __call__(self, **kwargs):
        return self._maker(**kwargs)

    def reset(self, eng):
        self._maker = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False)


engine = _create_engine()
async_session_maker = _SessionMakerProxy(engine)

Base = declarative_base()


async def reset_engine(*, poolclass: type[Pool] | None = None):
    """Recreate engine & session maker, binding them to the current event loop.

    Call once from the test conftest (inside a running event loop) so
    the asyncpg pool is bound to pytest-asyncio's loop rather than an
    import-time or stale loop.
    """
    global engine
    old = engine
    engine = _create_engine(poolclass=poolclass)
    async_session_maker.reset(engine)
    await old.dispose()


async def get_db():
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
