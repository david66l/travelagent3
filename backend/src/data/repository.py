"""
数据仓库 — Agent 唯一数据查询入口。

所有查询走本地 PostgreSQL，不调外部 API。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

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
            query = """
                SELECT id, name, city, category, lat, lng,
                       ticket_price, open_time, close_time,
                       duration_minutes, walk_intensity,
                       tags, suitable_for, source
                FROM attractions
                WHERE city = $1 AND status != 'deprecated'
            """
            params = [city]

            if max_walk_intensity is not None:
                params.append(max_walk_intensity)
                query += f" AND walk_intensity <= ${len(params)}"

            if wheelchair_only:
                query += " AND wheelchair_accessible = true"

            if avoid_tags:
                params.append(avoid_tags)
                query += f" AND NOT (tags && ${len(params)}::text[])"

            query += " ORDER BY ticket_price NULLS LAST LIMIT $" + str(len(params) + 1)
            params.append(limit)

            rows = await db.fetch(query, *params)
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
            query = """
                SELECT * FROM restaurants
                WHERE city = $1 AND status != 'deprecated'
            """
            params = [city]

            if food_prefs:
                params.append(food_prefs)
                query += f" AND tags && ${len(params)}::text[]"

            if avoid_foods:
                params.append(avoid_foods)
                query += f" AND NOT (tags && ${len(params)}::text[])"

            if max_price is not None:
                params.append(max_price)
                query += f" AND avg_price <= ${len(params)}"

            query += " ORDER BY rating DESC NULLS LAST LIMIT $" + str(len(params) + 1)
            params.append(limit)

            rows = await db.fetch(query, *params)
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
        embedding = embedder.encode_single(query_text)

        async with async_session_maker() as db:
            rows = await db.fetch(
                """
                SELECT content, content_type,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM knowledge_tips
                WHERE ($2::text IS NULL OR city = $2)
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                embedding,
                city,
                top_k,
            )
            return [dict(r) for r in rows]

    async def upsert_attraction(self, raw: RawPOI) -> UUID:
        """增量更新景点 — 只更新动态字段。"""
        async with async_session_maker() as db:
            # 查找已存在记录
            existing = await db.fetchrow(
                "SELECT id FROM attractions WHERE name = $1 AND city = $2",
                raw.name, raw.city,
            )

            if existing:
                # 只更新动态字段
                await db.execute(
                    """
                    UPDATE attractions
                    SET ticket_price = $1, open_time = $2, close_time = $3,
                        source_updated_at = $4
                    WHERE id = $5
                    """,
                    raw.ticket_price, raw.open_time, raw.close_time,
                    raw.source_updated_at, existing["id"],
                )
                return existing["id"]
            else:
                # 新 POI 完整入库
                new_id = uuid4()
                await db.execute(
                    """
                    INSERT INTO attractions (id, name, city, category, lat, lng,
                        address, ticket_price, open_time, close_time,
                        tags, source, source_updated_at)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                    """,
                    new_id, raw.name, raw.city, raw.category,
                    raw.lat, raw.lng, raw.address,
                    raw.ticket_price, raw.open_time, raw.close_time,
                    raw.tags, raw.source, raw.source_updated_at,
                )
                return new_id

    async def upsert_knowledge_tip(
        self, city: str, content: str, content_type: str
    ) -> None:
        """入库攻略贴士并生成向量（pgvector）。"""
        from data.embedding import get_embedder

        embedder = await get_embedder()
        embedding = embedder.encode_single(content)

        async with async_session_maker() as db:
            await db.execute(
                """
                INSERT INTO knowledge_tips (city, content, content_type, embedding)
                VALUES ($1, $2, $3, $4::vector)
                """,
                city, content, content_type, embedding,
            )


# 全局单例
repo = DataRepository()
