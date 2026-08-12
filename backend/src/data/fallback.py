"""
兜底实时检索 — Layer 4 数据流水线。

仅在本地 PostgreSQL 无匹配结果时触发高德 API，
结果自动写回本地库，下次直走本地查询。
"""

from __future__ import annotations

import logging
from typing import Optional

from core.settings import settings

logger = logging.getLogger(__name__)


class FallbackSearcher:
    """兜底检索器：本地 → 高德 API → 写回本地。"""

    def __init__(self):
        self._collector = None

    async def _get_collector(self):
        if self._collector is None:
            from data.collectors.amap import AmapCollector

            # 统一从 settings 读取高德 Key
            key = settings.amap_key
            if not key:
                logger.warning("AMAP_KEY not configured in settings")
                return None

            self._collector = AmapCollector(key)

        return self._collector

    async def search_attractions(
        self, city: str, interests: Optional[list[str]] = None
    ) -> list[dict]:
        """搜索景点：本地优先，缺失时走高德。"""
        from data.repository import repo

        # 1. 尝试本地库
        local = await repo.search_attractions(city, limit=20)
        if local:
            logger.debug("Local hit: %d attractions for %s", len(local), city)
            return [
                {
                    "name": a.name,
                    "category": a.category,
                    "price": a.ticket_price,
                    "tags": a.tags,
                    "source": "local",
                }
                for a in local
            ]

        # 2. 本地缺失 → 高德 API
        logger.info("Local miss for %s, falling back to Amap API", city)
        collector = await self._get_collector()
        if not collector:
            logger.warning("No Amap key configured, returning empty")
            return []

        try:
            types = "风景名胜|公园广场|寺庙道观|纪念馆"
            raw_pois = await collector.search_pois(city, types=types)
        except Exception as exc:
            logger.warning("Amap API failed for %s: %s", city, exc)
            return []

        if not raw_pois:
            return []

        # 3. 写回本地库（下次直走本地）
        upserted = 0
        for raw in raw_pois:
            try:
                await repo.upsert_attraction(raw)
                upserted += 1
            except Exception as exc:
                logger.debug("Failed to upsert %s: %s", raw.name, exc)

        logger.info("Amap returned %d POIs for %s, upserted %d", len(raw_pois), city, upserted)

        return [
            {
                "name": r.name,
                "category": r.category,
                "price": r.ticket_price,
                "tags": r.tags,
                "source": "amap",
            }
            for r in raw_pois
        ]

    async def search_restaurants(
        self, city: str, food_prefs: Optional[list[str]] = None
    ) -> list[dict]:
        """搜索餐厅：本地优先。"""
        from data.repository import repo

        local = await repo.search_restaurants(city, food_prefs=food_prefs, limit=20)
        if local:
            return local

        # 高德搜索
        collector = await self._get_collector()
        if not collector:
            return []

        types = "中餐厅|外国餐厅|小吃快餐店"
        raw_pois = await collector.search_pois(city, types=types)

        for raw in raw_pois:
            try:
                await repo.upsert_attraction(raw)
            except Exception:
                pass

        return [
            {
                "name": r.name,
                "category": r.category,
                "price": r.ticket_price,
                "tags": r.tags,
                "source": "amap",
            }
            for r in raw_pois
        ]


# 全局单例
fallback_searcher = FallbackSearcher()
