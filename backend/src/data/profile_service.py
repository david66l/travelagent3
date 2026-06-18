"""
用户画像服务 — 长期偏好存储 + Embedding 更新。

职责:
  - 读取用户画像（供 Agent 规划使用）
  - 更新用户画像（行程结束后学习偏好）
  - 生成/更新偏好向量
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from core.database import async_session_maker

logger = logging.getLogger(__name__)


class ProfileService:
    """用户画像读写服务。"""

    async def get_profile(self, user_id: str) -> dict:
        """读取用户画像 — 包含结构化字段和向量。"""
        async with async_session_maker() as db:
            row = await db.fetchrow(
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
                profile = dict(row["profile_json"] or {})
                profile["visited_cities"] = row["visited_cities"] or []
                profile["favorite_spots"] = row["favorite_spots"] or []
                profile["liked_foods"] = row["liked_foods"] or []
                profile["avoided_foods"] = row["avoided_foods"] or []
                profile["avg_daily_budget"] = float(row["avg_daily_budget"]) if row["avg_daily_budget"] else None
                profile["preferred_transport"] = row["preferred_transport"] or ""
                profile["preferred_accommodation"] = row["preferred_accommodation"] or ""
                return profile
            return {}

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

        # 读取现有数据
        async with async_session_maker() as db:
            existing = await db.fetchrow(
                "SELECT * FROM user_profile_vectors WHERE user_id = $1", user_id
            )

            if existing:
                # 合并更新
                updates = {}
                if visited_cities:
                    merged = list(set((existing["visited_cities"] or []) + visited_cities))
                    updates["visited_cities"] = merged
                if favorite_spots:
                    merged = list(set((existing["favorite_spots"] or []) + favorite_spots))
                    updates["favorite_spots"] = merged
                if avoid_spots:
                    merged = list(set((existing["avoid_spots"] or []) + avoid_spots))
                    updates["avoid_spots"] = merged
                if liked_foods:
                    merged = list(set((existing["liked_foods"] or []) + liked_foods))
                    updates["liked_foods"] = merged
                if avoided_foods:
                    merged = list(set((existing["avoided_foods"] or []) + avoided_foods))
                    updates["avoided_foods"] = merged
                if trip_budget is not None:
                    old_count = existing["trip_count"] or 0
                    old_avg = float(existing["avg_daily_budget"] or 0)
                    new_avg = (old_avg * old_count + trip_budget) / (old_count + 1)
                    updates["avg_daily_budget"] = new_avg
                    updates["trip_count"] = old_count + 1

                if profile_updates:
                    existing_profile = dict(existing["profile_json"] or {})
                    existing_profile.update(profile_updates)
                    updates["profile_json"] = existing_profile

                if updates:
                    set_clause = ", ".join(f"{k} = ${i+1}" for i, k in enumerate(updates.keys()))
                    values = list(updates.values())
                    values.append(user_id)
                    await db.execute(
                        f"UPDATE user_profile_vectors SET {set_clause}, updated_at = NOW() WHERE user_id = ${len(values)}",
                        *values,
                    )
            else:
                # 新用户，创建记录
                merged_cities = visited_cities or []
                merged_spots = favorite_spots or []
                merged_avoid = avoid_spots or []
                merged_liked = liked_foods or []
                merged_avoided = avoided_foods or []

                await db.execute(
                    """
                    INSERT INTO user_profile_vectors
                        (user_id, profile_json, visited_cities, favorite_spots,
                         avoid_spots, liked_foods, avoided_foods, avg_daily_budget, trip_count)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    user_id,
                    profile_updates or {},
                    merged_cities,
                    merged_spots,
                    merged_avoid,
                    merged_liked,
                    merged_avoided,
                    trip_budget or 0,
                    1,
                )

        # 生成/更新向量（异步，不阻塞主流程）
        try:
            await self._update_embedding(user_id)
        except Exception as exc:
            logger.warning("Failed to update embedding for user %s: %s", user_id, exc)

    async def _update_embedding(self, user_id: str) -> None:
        """生成偏好向量 — 从结构化字段拼接文本 → bge encode。"""
        profile = await self.get_profile(user_id)

        # 构建描述文本
        parts = []
        if profile.get("visited_cities"):
            parts.append(f"去过: {', '.join(profile['visited_cities'][:10])}")
        if profile.get("favorite_spots"):
            parts.append(f"喜欢: {', '.join(profile['favorite_spots'][:10])}")
        if profile.get("liked_foods"):
            parts.append(f"爱吃: {', '.join(profile['liked_foods'][:10])}")
        if profile.get("avoided_foods"):
            parts.append(f"忌口: {', '.join(profile['avoided_foods'][:10])}")

        text = "。".join(parts)
        if not text:
            return

        from data.embedding import get_embedder

        embedder = await get_embedder()
        embedding = embedder.encode_single(text)

        async with async_session_maker() as db:
            await db.execute(
                "UPDATE user_profile_vectors SET preference_embedding = $1::vector WHERE user_id = $2",
                embedding,
                user_id,
            )


# 全局单例
profile_service = ProfileService()
