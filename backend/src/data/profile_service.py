"""
用户画像服务 — 长期偏好存储 + Embedding 更新。

职责:
  - 读取用户画像（供 Agent 规划使用）
  - 更新用户画像（行程结束后学习偏好）
  - 生成/更新偏好向量

实现注意:
  - user_profile_vectors 使用 UUID 主键并带 pgvector 扩展，直接用 asyncpg
    的 positional parameter 可以避免 SQLAlchemy + asyncpg 在 named parameter
    和 jsonb/vector 类型上的兼容性问题。
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import asyncpg

from core.settings import settings

logger = logging.getLogger(__name__)


class ProfileService:
    """用户画像读写服务。"""

    @staticmethod
    def _dsn() -> str:
        """Return a sync-style PostgreSQL DSN for asyncpg."""
        return settings.database_url_sync

    async def get_profile(self, user_id: str) -> dict:
        """读取用户画像 — 包含结构化字段和向量。"""
        conn: asyncpg.Connection | None = None
        try:
            conn = await asyncpg.connect(self._dsn())
            row = await conn.fetchrow(
                """
                SELECT profile_json, visited_cities, favorite_spots,
                       liked_foods, avoided_foods, avg_daily_budget,
                       preferred_transport, preferred_accommodation
                FROM user_profile_vectors
                WHERE user_id = $1
                """,
                user_id,
            )
            if row:
                raw_json = row["profile_json"]
                if isinstance(raw_json, str):
                    raw_json = json.loads(raw_json or "{}")
                profile = dict(raw_json or {})
                profile["visited_cities"] = row["visited_cities"] or []
                profile["favorite_spots"] = row["favorite_spots"] or []
                profile["liked_foods"] = row["liked_foods"] or []
                profile["avoided_foods"] = row["avoided_foods"] or []
                profile["avg_daily_budget"] = (
                    float(row["avg_daily_budget"]) if row["avg_daily_budget"] else None
                )
                profile["preferred_transport"] = row["preferred_transport"] or ""
                profile["preferred_accommodation"] = row["preferred_accommodation"] or ""
                return profile
            return {}
        except Exception as exc:
            logger.warning("Failed to get profile for %s: %s", user_id, exc)
            return {}
        finally:
            if conn:
                await conn.close()

    async def update_profile(
        self,
        user_id: str,
        *,
        visited_cities: Optional[list[str]] = None,
        favorite_spots: Optional[list[str]] = None,
        avoid_spots: Optional[list[str]] = None,
        liked_foods: Optional[list[str]] = None,
        avoided_foods: Optional[list[str]] = None,
        trip_budget: Optional[float] = None,
        profile_updates: Optional[dict] = None,
    ) -> None:
        """增量更新用户画像 — 只更新传入的字段。"""

        conn: asyncpg.Connection | None = None
        try:
            conn = await asyncpg.connect(self._dsn())
            async with conn.transaction():
                existing = await conn.fetchrow(
                    "SELECT * FROM user_profile_vectors WHERE user_id = $1", user_id
                )

                if existing:
                    merged: dict = {}
                    if visited_cities:
                        merged["visited_cities"] = list(
                            set((existing["visited_cities"] or []) + visited_cities)
                        )
                    if favorite_spots:
                        merged["favorite_spots"] = list(
                            set((existing["favorite_spots"] or []) + favorite_spots)
                        )
                    if avoid_spots:
                        merged["avoid_spots"] = list(
                            set((existing["avoid_spots"] or []) + avoid_spots)
                        )
                    if liked_foods:
                        merged["liked_foods"] = list(
                            set((existing["liked_foods"] or []) + liked_foods)
                        )
                    if avoided_foods:
                        merged["avoided_foods"] = list(
                            set((existing["avoided_foods"] or []) + avoided_foods)
                        )
                    if trip_budget is not None:
                        old_count = existing["trip_count"] or 0
                        old_avg = float(existing["avg_daily_budget"] or 0)
                        new_avg = (old_avg * old_count + trip_budget) / (old_count + 1)
                        merged["avg_daily_budget"] = new_avg
                        merged["trip_count"] = old_count + 1

                    if profile_updates:
                        raw_existing = existing["profile_json"]
                        if isinstance(raw_existing, str):
                            raw_existing = json.loads(raw_existing or "{}")
                        existing_profile = dict(raw_existing or {})
                        existing_profile.update(profile_updates)
                        merged["profile_json"] = json.dumps(existing_profile)

                    if merged:
                        # Build positional SET clause dynamically.
                        columns = list(merged.keys())
                        set_clause = ", ".join(
                            f"{col} = ${i + 2}::jsonb"
                            if col == "profile_json"
                            else f"{col} = ${i + 2}"
                            for i, col in enumerate(columns)
                        )
                        values = [merged[col] for col in columns]
                        await conn.execute(
                            f"""
                            UPDATE user_profile_vectors
                            SET {set_clause}, updated_at = NOW()
                            WHERE user_id = $1
                            """,
                            user_id,
                            *values,
                        )
                else:
                    await conn.execute(
                        """
                        INSERT INTO user_profile_vectors
                            (user_id, profile_json, visited_cities, favorite_spots,
                             avoid_spots, liked_foods, avoided_foods, avg_daily_budget, trip_count)
                        VALUES ($1, $2::jsonb, $3, $4, $5, $6, $7, $8, $9)
                        """,
                        user_id,
                        json.dumps(profile_updates or {}),
                        visited_cities or [],
                        favorite_spots or [],
                        avoid_spots or [],
                        liked_foods or [],
                        avoided_foods or [],
                        trip_budget or 0,
                        1,
                    )
        except Exception as exc:
            logger.warning("Memory update failed: %s", exc)
            return
        finally:
            if conn:
                await conn.close()

        # 生成/更新向量（异步，不阻塞主流程）
        try:
            await self._update_embedding(user_id)
        except Exception as exc:
            logger.warning("Failed to update embedding for user %s: %s", user_id, exc)

    async def _update_embedding(self, user_id: str) -> None:
        """生成偏好向量 — 从结构化字段拼接文本 → bge encode。"""
        profile = await self.get_profile(user_id)

        parts = []
        if profile.get("visited_cities"):
            parts.append(f"去过: {', '.join(profile['visited_cities'][:10])}")
        if profile.get("favorite_spots"):
            parts.append(f"喜欢: {', '.join(profile['favorite_spots'][:10])}")
        if profile.get("liked_foods"):
            parts.append(f"爱吃: {', '.join(profile['liked_foods'][:10])}")
        if profile.get("avoided_foods"):
            parts.append(f"忌口: {', '.join(profile['avoided_foods'][:10])}")

        embedding_text = "。".join(parts)
        if not embedding_text:
            return

        from data.embedding import get_embedder

        embedder = await get_embedder()
        embedding = await embedder.aencode_single(embedding_text)

        conn: asyncpg.Connection | None = None
        try:
            conn = await asyncpg.connect(self._dsn())
            await conn.execute(
                "UPDATE user_profile_vectors SET preference_embedding = $1::vector WHERE user_id = $2",
                str(embedding),
                user_id,
            )
        finally:
            if conn:
                await conn.close()


# 全局单例
profile_service = ProfileService()
