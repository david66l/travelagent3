"""Base repository with common CRUD operations."""

from typing import Generic, TypeVar, Type, Optional, Sequence
from uuid import UUID

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """Generic repository providing basic CRUD operations."""

    def __init__(self, db: AsyncSession, model: Type[ModelT]):
        self.db = db
        self.model = model

    async def get_by_id(self, id_: UUID | str) -> Optional[ModelT]:
        """Get a single record by primary key."""
        result = await self.db.execute(select(self.model).where(self.model.id == id_))
        return result.scalar_one_or_none()

    async def get_many(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[ModelT]:
        """Get a list of records with pagination."""
        result = await self.db.execute(select(self.model).limit(limit).offset(offset))
        return result.scalars().all()

    async def create(self, obj: ModelT) -> ModelT:
        """Create a new record."""
        self.db.add(obj)
        await self.db.flush()
        await self.db.refresh(obj)
        return obj

    async def update(self, id_: UUID | str, **kwargs) -> int:
        """Update a record by primary key. Returns number of rows updated."""
        result = await self.db.execute(
            update(self.model).where(self.model.id == id_).values(**kwargs)
        )
        return result.rowcount

    async def delete(self, id_: UUID | str) -> int:
        """Delete a record by primary key. Returns number of rows deleted."""
        result = await self.db.execute(delete(self.model).where(self.model.id == id_))
        return result.rowcount
