"""Seed local PostgreSQL with built-in CITY_DEFAULTS POIs.

Usage:
    cd backend
    .venv/bin/python scripts/seed_city_defaults.py

This inserts attractions/restaurants from skills/city_data.py and computes
BGE vectors + tsvector search vectors so structured/vector/BM25 retrieval all work.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from datetime import datetime, time, timezone
from typing import Any

from sqlalchemy import text

from core.database import async_session_maker
from data.embedding import get_embedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_recommended_hours(hours_str: str | None) -> int:
    """把 '2-3小时' / '半天' 等建议时长解析为分钟。"""
    if not hours_str:
        return 120
    s = str(hours_str)
    m = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", s)
    if m:
        return int((float(m.group(1)) + float(m.group(2))) / 2 * 60)
    m = re.search(r"(\d+(?:\.\d+)?)\s*小时?", s)
    if m:
        return int(float(m.group(1)) * 60)
    if "半天" in s:
        return 240
    if "全天" in s:
        return 480
    return 120


def _time_window(best_time: str | None) -> tuple[time, time]:
    """根据最佳游玩时间推断开放时段。"""
    if best_time == "上午":
        return datetime.strptime("08:00", "%H:%M").time(), datetime.strptime(
            "14:00", "%H:%M"
        ).time()
    if best_time == "下午":
        return datetime.strptime("12:00", "%H:%M").time(), datetime.strptime(
            "21:00", "%H:%M"
        ).time()
    if best_time == "傍晚":
        return datetime.strptime("16:00", "%H:%M").time(), datetime.strptime(
            "22:00", "%H:%M"
        ).time()
    if best_time == "晚上":
        return datetime.strptime("18:00", "%H:%M").time(), datetime.strptime(
            "23:00", "%H:%M"
        ).time()
    if best_time == "全天":
        return datetime.strptime("08:00", "%H:%M").time(), datetime.strptime(
            "23:00", "%H:%M"
        ).time()
    return datetime.strptime("08:00", "%H:%M").time(), datetime.strptime("22:00", "%H:%M").time()


def _walk_intensity(poi: Any) -> int:
    """根据类型/标签粗略估计体力强度。"""
    tags = {t.lower() for t in (getattr(poi, "tags", []) or [])}
    indoor_outdoor = (getattr(poi, "indoor_outdoor", "") or "").lower()
    if "登山" in tags or "徒步" in tags or "户外" in tags:
        return 4
    if "公园" in tags or "自然" in tags or indoor_outdoor == "outdoor":
        return 3
    if indoor_outdoor == "indoor":
        return 1
    return 3


def _embedding_text(poi: Any) -> str:
    """组合用于生成向量的文本。"""
    parts = [getattr(poi, "name", "")]
    desc = getattr(poi, "description", "")
    if desc:
        parts.append(desc)
    tags = getattr(poi, "tags", []) or []
    if tags:
        parts.append(" ".join(tags))
    area = getattr(poi, "area", "")
    if area:
        parts.append(area)
    return " ".join(parts).strip() or poi.name


# ---------------------------------------------------------------------------
# Insert logic
# ---------------------------------------------------------------------------


async def _insert_attractions(db, city: str, pois: list[Any], embedder) -> int:
    """批量插入/更新景点，返回写入条数。"""
    if not pois:
        return 0

    texts = [_embedding_text(p) for p in pois]
    embeddings = embedder.encode(texts)

    inserted = 0
    for poi, embedding in zip(pois, embeddings):
        open_time, close_time = _time_window(getattr(poi, "best_time", None))
        duration = _parse_recommended_hours(getattr(poi, "recommended_hours", None))
        location = getattr(poi, "location", None)
        lat = getattr(location, "lat", 0.0) if location else 0.0
        lng = getattr(location, "lng", 0.0) if location else 0.0
        tags = list(getattr(poi, "tags", []) or [])
        area = getattr(poi, "area", "") or ""
        description = getattr(poi, "description", "") or ""
        text_for_search = " ".join([poi.name, description, " ".join(tags), area])

        await db.execute(
            text(
                """
                INSERT INTO attractions (
                    id, name, city, category, ticket_price, open_time, close_time,
                    rating, lat, lng, address, duration_minutes, walk_intensity,
                    indoor_outdoor, tags, description, description_vector, search_vector,
                    source, status, created_at
                ) VALUES (
                    :id, :name, :city, :category, :ticket_price,
                    :open_time, :close_time,
                    :rating, :lat, :lng, :address, :duration_minutes, :walk_intensity,
                    :indoor_outdoor, :tags, :description,
                    (:embedding)::vector,
                    to_tsvector('simple', :search_text),
                    :source, :status, :created_at
                )
                ON CONFLICT (name, city) DO UPDATE SET
                    category = EXCLUDED.category,
                    ticket_price = EXCLUDED.ticket_price,
                    open_time = EXCLUDED.open_time,
                    close_time = EXCLUDED.close_time,
                    rating = EXCLUDED.rating,
                    lat = EXCLUDED.lat,
                    lng = EXCLUDED.lng,
                    address = EXCLUDED.address,
                    duration_minutes = EXCLUDED.duration_minutes,
                    walk_intensity = EXCLUDED.walk_intensity,
                    indoor_outdoor = EXCLUDED.indoor_outdoor,
                    tags = EXCLUDED.tags,
                    description = EXCLUDED.description,
                    description_vector = EXCLUDED.description_vector,
                    search_vector = EXCLUDED.search_vector,
                    source = EXCLUDED.source,
                    status = EXCLUDED.status,
                    created_at = EXCLUDED.created_at
                """
            ),
            {
                "id": str(uuid.uuid4()),
                "name": poi.name,
                "city": city,
                "category": poi.category,
                "ticket_price": getattr(poi, "ticket_price", None),
                "open_time": open_time,
                "close_time": close_time,
                "rating": min(getattr(poi, "score", 0.5) * 5, 5.0),
                "lat": lat,
                "lng": lng,
                "address": area,
                "duration_minutes": duration,
                "walk_intensity": _walk_intensity(poi),
                "indoor_outdoor": getattr(poi, "indoor_outdoor", None),
                "tags": tags,
                "description": description,
                "embedding": str(embedding),
                "search_text": text_for_search,
                "source": "city_defaults",
                "status": "active",
                "created_at": datetime.now(timezone.utc),
            },
        )
        inserted += 1
    return inserted


async def _insert_restaurants(db, city: str, pois: list[Any]) -> int:
    """批量插入/更新餐厅，返回写入条数。"""
    if not pois:
        return 0

    inserted = 0
    for poi in pois:
        location = getattr(poi, "location", None)
        lat = getattr(location, "lat", 0.0) if location else 0.0
        lng = getattr(location, "lng", 0.0) if location else 0.0
        tags = list(getattr(poi, "tags", []) or [])
        area = getattr(poi, "area", "") or ""
        open_time, close_time = _time_window(getattr(poi, "best_time", None))

        await db.execute(
            text(
                """
                INSERT INTO restaurants (
                    id, name, city, cuisine, avg_price, open_time, close_time,
                    lat, lng, address, tags, rating, source, status, created_at
                ) VALUES (
                    :id, :name, :city, :cuisine, :avg_price,
                    :open_time, :close_time,
                    :lat, :lng, :address, :tags, :rating, :source, :status, :created_at
                )
                ON CONFLICT (name, city) DO UPDATE SET
                    cuisine = EXCLUDED.cuisine,
                    avg_price = EXCLUDED.avg_price,
                    open_time = EXCLUDED.open_time,
                    close_time = EXCLUDED.close_time,
                    lat = EXCLUDED.lat,
                    lng = EXCLUDED.lng,
                    address = EXCLUDED.address,
                    tags = EXCLUDED.tags,
                    rating = EXCLUDED.rating,
                    source = EXCLUDED.source,
                    status = EXCLUDED.status,
                    created_at = EXCLUDED.created_at
                """
            ),
            {
                "id": str(uuid.uuid4()),
                "name": poi.name,
                "city": city,
                "cuisine": tags[0] if tags else None,
                "avg_price": getattr(poi, "ticket_price", None),
                "open_time": open_time,
                "close_time": close_time,
                "lat": lat,
                "lng": lng,
                "address": area,
                "tags": tags,
                "rating": min(getattr(poi, "score", 0.5) * 5, 5.0),
                "source": "city_defaults",
                "status": "active",
                "created_at": datetime.now(timezone.utc),
            },
        )
        inserted += 1
    return inserted


async def main():
    from skills.city_data import CITY_DEFAULTS

    embedder = await get_embedder()
    total_attractions = 0
    total_restaurants = 0

    async with async_session_maker() as db:
        for city, pois in CITY_DEFAULTS.items():
            attractions = [p for p in pois if p.category == "attraction"]
            restaurants = [p for p in pois if p.category == "restaurant"]
            logger.info(
                "Seeding %s: %d attractions, %d restaurants",
                city,
                len(attractions),
                len(restaurants),
            )

            attr_count = await _insert_attractions(db, city, attractions, embedder)
            rest_count = await _insert_restaurants(db, city, restaurants)
            total_attractions += attr_count
            total_restaurants += rest_count

        await db.commit()

    logger.info(
        "Done. Inserted/updated %d attractions and %d restaurants across %d cities.",
        total_attractions,
        total_restaurants,
        len(CITY_DEFAULTS),
    )


if __name__ == "__main__":
    asyncio.run(main())
