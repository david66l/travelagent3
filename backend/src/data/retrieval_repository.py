"""RAG retrieval repository — structured + vector + BM25 hybrid search over attractions.

Uses SQLAlchemy async API so it works with the project's AsyncSession fixture.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import text

from core.database import async_session_maker
from data.embedding import get_embedder
from models.poi import POI

logger = logging.getLogger(__name__)


class RetrievalRepository:
    """Query attractions with structured filters, vector similarity and full-text search."""

    VECTOR_DIM = 1024

    async def search_structured(
        self,
        city: str,
        *,
        max_walk_intensity: Optional[int] = None,
        wheelchair_only: bool = False,
        avoid_tags: Optional[list[str]] = None,
        suitable_for: Optional[list[str]] = None,
        limit: int = 50,
    ) -> list[POI]:
        """Structured SQL pre-filtering over attractions."""
        sql = """
            SELECT id, name, city, lat, lng, address,
                   ticket_price, open_time, close_time,
                   duration_minutes, walk_intensity,
                   queue_time_avg, indoor_outdoor,
                   need_reservation, reservation_advance_days,
                   tags, description, suitable_for,
                   accessibility, wheelchair_accessible,
                   spot_tags, season_restriction, temp_closure_dates
            FROM attractions
            WHERE city = :city AND status != 'deprecated'
        """
        params: dict[str, Any] = {"city": city}

        if max_walk_intensity is not None:
            sql += " AND walk_intensity <= :max_walk"
            params["max_walk"] = max_walk_intensity

        if wheelchair_only:
            sql += " AND wheelchair_accessible = true"

        if avoid_tags:
            sql += " AND NOT (tags && :avoid_tags)"
            params["avoid_tags"] = avoid_tags

        if suitable_for:
            sql += " AND suitable_for && :suitable_for"
            params["suitable_for"] = suitable_for

        sql += " ORDER BY rating DESC NULLS LAST LIMIT :limit"
        params["limit"] = limit

        async with async_session_maker() as db:
            result = await db.execute(text(sql), params)
            rows = result.mappings().all()

        return [POI.from_attraction_row(dict(r)) for r in rows]

    async def search_vector(
        self,
        query_text: str,
        city: Optional[str] = None,
        *,
        limit: int = 50,
        ef_search: int = 128,
    ) -> list[POI]:
        """Semantic search over attraction description vectors via pgvector."""
        embedder = await get_embedder()
        embedding = await embedder.aencode_single(query_text)

        # asyncpg expects the vector literal as a string.
        embedding_str = str(embedding)

        sql = """
            SELECT id, name, city, lat, lng, address,
                   ticket_price, open_time, close_time,
                   duration_minutes, walk_intensity,
                   queue_time_avg, indoor_outdoor,
                   need_reservation, reservation_advance_days,
                   tags, description, suitable_for,
                   accessibility, wheelchair_accessible,
                   spot_tags, season_restriction, temp_closure_dates,
                   1 - (description_vector <=> (:embedding)::vector) AS similarity
            FROM attractions
            WHERE description_vector IS NOT NULL
              AND status != 'deprecated'
        """
        params: dict[str, Any] = {"embedding": embedding_str}

        if city:
            sql += " AND city = :city"
            params["city"] = city

        sql += " ORDER BY description_vector <=> (:embedding)::vector LIMIT :limit"
        params["limit"] = limit

        async with async_session_maker() as db:
            # hnsw.ef_search must be set in a separate statement; asyncpg does not
            # support parameters in SET LOCAL, so embed the integer safely.
            await db.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
            result = await db.execute(text(sql), params)
            rows = result.mappings().all()

        pois = [POI.from_attraction_row(dict(r)) for r in rows]
        for poi in pois:
            poi.source = "vector"
        return pois

    async def search_bm25(
        self,
        query_text: str,
        city: Optional[str] = None,
        *,
        limit: int = 50,
    ) -> list[POI]:
        """Portable lexical search that works for Chinese without zhparser.

        PostgreSQL's built-in parsers treat a contiguous Chinese phrase as one
        token, so tsvector matching misses names such as ``成都大熊猫基地`` for
        the query ``成都 熊猫``. Match the safely bound whitespace-separated
        terms against the searchable text and use the match count as rank.
        """
        sql = """
            SELECT id, name, city, lat, lng, address,
                   ticket_price, open_time, close_time,
                   duration_minutes, walk_intensity,
                   queue_time_avg, indoor_outdoor,
                   need_reservation, reservation_advance_days,
                   tags, description, suitable_for,
                   accessibility, wheelchair_accessible,
                   spot_tags, season_restriction, temp_closure_dates,
                   (
                       SELECT count(*)::float
                       FROM unnest(regexp_split_to_array(trim(:query), E'\\s+')) AS term
                       WHERE concat_ws(
                           ' ', name, description, array_to_string(tags, ' '),
                           array_to_string(spot_tags, ' ')
                       ) ILIKE '%' || term || '%'
                   ) AS rank
            FROM attractions
            WHERE EXISTS (
                SELECT 1
                FROM unnest(regexp_split_to_array(trim(:query), E'\\s+')) AS term
                WHERE concat_ws(
                    ' ', name, description, array_to_string(tags, ' '),
                    array_to_string(spot_tags, ' ')
                ) ILIKE '%' || term || '%'
            )
              AND status != 'deprecated'
        """
        params: dict[str, Any] = {"query": query_text}

        if city:
            sql += " AND city = :city"
            params["city"] = city

        sql += " ORDER BY rank DESC LIMIT :limit"
        params["limit"] = limit

        async with async_session_maker() as db:
            result = await db.execute(text(sql), params)
            rows = result.mappings().all()

        pois = [POI.from_attraction_row(dict(r)) for r in rows]
        for poi in pois:
            poi.source = "bm25"
        return pois

    async def get_popular_attractions(
        self,
        city: str,
        *,
        limit: int = 15,
    ) -> list[POI]:
        """Fallback: return top-rated attractions in the city."""
        return await self.search_structured(city, limit=limit)

    async def update_attraction_embedding(
        self,
        attraction_id: str,
        embedding: list[float],
    ) -> None:
        """Write description vector back to attractions table."""
        async with async_session_maker() as db:
            await db.execute(
                text(
                    "UPDATE attractions SET description_vector = (:embedding)::vector WHERE id = :id"
                ),
                {"embedding": str(embedding), "id": attraction_id},
            )
            await db.commit()


# 全局单例
retrieval_repo = RetrievalRepository()
