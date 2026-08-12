"""
数据仓库 — Agent 唯一数据查询入口。

所有查询走本地 PostgreSQL，不调外部 API。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import text

from core.database import async_session_maker
from data.collectors.amap import RawPOI

logger = logging.getLogger(__name__)


@dataclass
class Attraction:
    id: UUID
    name: str
    city: str
    category: str
    lat: float
    lng: float
    ticket_price: Optional[float] = None
    open_time: Optional[str] = None
    close_time: Optional[str] = None
    duration_minutes: int = 120
    walk_intensity: int = 3
    tags: list[str] = field(default_factory=list)
    suitable_for: list[str] = field(default_factory=list)
    source: str = ""


class DataRepository:
    """Agent 数据查询入口。"""

    async def search_attractions(
        self,
        city: str,
        *,
        max_walk_intensity: Optional[int] = None,
        wheelchair_only: bool = False,
        avoid_tags: Optional[list[str]] = None,
        limit: int = 50,
    ) -> list[Attraction]:
        """结构化查询景点 — 过滤轮椅/步行强度/标签。"""
        async with async_session_maker() as db:
            sql = """
                SELECT id, name, city, category, lat, lng,
                       ticket_price, open_time, close_time,
                       duration_minutes, walk_intensity,
                       tags, suitable_for, source
                FROM attractions
                WHERE city = :city AND status != 'deprecated'
            """
            params: dict = {"city": city}

            if max_walk_intensity is not None:
                sql += " AND walk_intensity <= :max_walk"
                params["max_walk"] = max_walk_intensity

            if wheelchair_only:
                sql += " AND wheelchair_accessible = true"

            if avoid_tags:
                sql += " AND NOT (tags && :avoid_tags::text[])"
                params["avoid_tags"] = avoid_tags

            sql += " ORDER BY ticket_price NULLS LAST LIMIT :limit"
            params["limit"] = limit

            result = await db.execute(text(sql), params)
            rows = result.mappings().all()
            return [Attraction(**dict(r)) for r in rows]

    async def search_restaurants(
        self,
        city: str,
        *,
        food_prefs: Optional[list[str]] = None,
        avoid_foods: Optional[list[str]] = None,
        max_price: Optional[float] = None,
        limit: int = 20,
    ) -> list[dict]:
        """查询餐厅 — 按口味/忌口/价格过滤。"""
        async with async_session_maker() as db:
            sql = """
                SELECT * FROM restaurants
                WHERE city = :city AND status != 'deprecated'
            """
            params: dict = {"city": city}

            if food_prefs:
                sql += " AND tags && :food_prefs::text[]"
                params["food_prefs"] = food_prefs

            if avoid_foods:
                sql += " AND NOT (tags && :avoid_foods::text[])"
                params["avoid_foods"] = avoid_foods

            if max_price is not None:
                sql += " AND avg_price <= :max_price"
                params["max_price"] = max_price

            sql += " ORDER BY rating DESC NULLS LAST LIMIT :limit"
            params["limit"] = limit

            result = await db.execute(text(sql), params)
            rows = result.mappings().all()
            return [dict(r) for r in rows]

    async def search_knowledge(
        self,
        query_text: str,
        *,
        city: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """语义检索攻略知识库（pgvector 原生向量检索）。"""
        from data.embedding import get_embedder

        embedder = await get_embedder()
        embedding = await embedder.aencode_single(query_text)

        async with async_session_maker() as db:
            result = await db.execute(
                text("""
                    SELECT content, content_type,
                           1 - (embedding <=> (:embedding)::vector) AS similarity
                    FROM knowledge_tips
                    WHERE ((:city)::text IS NULL OR city = :city)
                    ORDER BY embedding <=> (:embedding)::vector
                    LIMIT :top_k
                """),
                {"embedding": str(embedding), "city": city, "top_k": top_k},
            )
            rows = result.mappings().all()
            return [dict(r) for r in rows]

    async def upsert_attraction(self, raw: RawPOI) -> UUID:
        """增量更新景点 — 只更新动态字段。"""
        async with async_session_maker() as db:
            result = await db.execute(
                text("SELECT id FROM attractions WHERE name = :name AND city = :city"),
                {"name": raw.name, "city": raw.city},
            )
            existing = result.mappings().first()

            if existing:
                await db.execute(
                    text("""
                        UPDATE attractions
                        SET ticket_price = :ticket_price,
                            open_time = :open_time,
                            close_time = :close_time,
                            source_updated_at = :source_updated_at
                        WHERE id = :id
                    """),
                    {
                        "ticket_price": raw.ticket_price,
                        "open_time": raw.open_time,
                        "close_time": raw.close_time,
                        "source_updated_at": raw.source_updated_at,
                        "id": existing["id"],
                    },
                )
                await db.commit()
                return existing["id"]

            new_id = uuid4()
            await db.execute(
                text("""
                    INSERT INTO attractions (id, name, city, category, lat, lng,
                        address, ticket_price, open_time, close_time,
                        tags, source, source_updated_at)
                    VALUES (:id, :name, :city, :category, :lat, :lng,
                        :address, :ticket_price, :open_time, :close_time,
                        :tags, :source, :source_updated_at)
                """),
                {
                    "id": new_id,
                    "name": raw.name,
                    "city": raw.city,
                    "category": raw.category,
                    "lat": raw.lat,
                    "lng": raw.lng,
                    "address": raw.address,
                    "ticket_price": raw.ticket_price,
                    "open_time": raw.open_time,
                    "close_time": raw.close_time,
                    "tags": raw.tags,
                    "source": raw.source,
                    "source_updated_at": raw.source_updated_at,
                },
            )
            await db.commit()
            return new_id

    async def upsert_knowledge_tip(self, city: str, content: str, content_type: str) -> None:
        """入库攻略贴士并生成向量（pgvector）。"""
        from data.embedding import get_embedder

        embedder = await get_embedder()
        embedding = await embedder.aencode_single(content)

        async with async_session_maker() as db:
            await db.execute(
                text("""
                    INSERT INTO knowledge_tips (city, content, content_type, embedding)
                    VALUES (:city, :content, :content_type, (:embedding)::vector)
                """),
                {
                    "city": city,
                    "content": content,
                    "content_type": content_type,
                    "embedding": str(embedding),
                },
            )
            await db.commit()


# 全局单例
repo = DataRepository()
