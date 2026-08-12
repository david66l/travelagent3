"""Retrieval fallback strategies when RAG returns empty or too few results."""

from __future__ import annotations

import logging
from typing import Any, Optional

from data.retrieval_repository import RetrievalRepository, retrieval_repo
from models.poi import POI
from models.travel_slots import TravelSlots

logger = logging.getLogger(__name__)


class RetrievalFallback:
    """Fallback for empty RAG results.

    Strategies (applied in order):
    1. Relax structured filters (drop suitable_for, avoid_tags, wheelchair_only).
    2. Return popular attractions in the destination city.
    3. Return a hard-coded safety set for well-known cities.
    """

    def __init__(self, repo: Optional[RetrievalRepository] = None):
        self._repo = repo or retrieval_repo

    async def fallback(
        self,
        slots: TravelSlots,
        profile: Optional[dict[str, Any]] = None,
        *,
        min_results: int = 15,
    ) -> dict[str, Any]:
        """Try progressively relaxed strategies and return fallback POIs + metadata."""
        city = slots.destination
        if not city:
            return {
                "poi_candidates": [],
                "retrieval_empty": True,
                "fallback_reason": "no_destination",
            }

        pois: list[POI] = []
        reasons: list[str] = []

        # Strategy 1: relax structured filters
        try:
            pois = await self._repo.search_structured(
                city,
                max_walk_intensity=None,
                wheelchair_only=False,
                avoid_tags=None,
                suitable_for=None,
                limit=min_results,
            )
            if len(pois) >= min_results:
                reasons.append("relaxed_structured_filters")
                return self._build_result(pois, reasons)
        except Exception as exc:
            logger.warning("Fallback strategy 1 failed: %s", exc)

        # Strategy 2: popular attractions in city
        try:
            popular = await self._repo.get_popular_attractions(city, limit=min_results)
            pois = self._merge_unique(pois, popular)
            if len(pois) >= min_results:
                reasons.append("popular_city_attractions")
                return self._build_result(pois, reasons)
        except Exception as exc:
            logger.warning("Fallback strategy 2 failed: %s", exc)

        # Strategy 3: hard-coded safety set for known cities
        safety = self._safety_set(city)
        pois = self._merge_unique(pois, safety)
        if pois:
            reasons.append("hardcoded_safety_set")
            return self._build_result(pois, reasons)

        return {
            "poi_candidates": [],
            "retrieval_empty": True,
            "fallback_reason": "all_strategies_failed",
        }

    @staticmethod
    def _merge_unique(base: list[POI], extra: list[POI]) -> list[POI]:
        """Merge two POI lists, deduplicating by spot_id."""
        seen = {p.spot_id for p in base}
        result = list(base)
        for poi in extra:
            if poi.spot_id not in seen:
                seen.add(poi.spot_id)
                result.append(poi)
        return result

    @staticmethod
    def _build_result(pois: list[POI], reasons: list[str]) -> dict[str, Any]:
        return {
            "poi_candidates": pois,
            "retrieval_empty": False,
            "fallback_reason": ";".join(reasons),
            "fallback_used": True,
        }

    @staticmethod
    def _safety_set(city: str) -> list[POI]:
        """Hard-coded POIs for major Chinese cities (last-resort fallback)."""
        safety_data: dict[str, list[dict[str, Any]]] = {
            "北京": [
                {
                    "spot_id": "beijing-001",
                    "spot_name": "故宫博物院",
                    "lat": 39.9163,
                    "lng": 116.3972,
                    "tags": ["历史", "世界文化遗产"],
                },
                {
                    "spot_id": "beijing-002",
                    "spot_name": "八达岭长城",
                    "lat": 40.3590,
                    "lng": 116.0200,
                    "tags": ["历史", "登山"],
                },
                {
                    "spot_id": "beijing-003",
                    "spot_name": "天坛公园",
                    "lat": 39.8830,
                    "lng": 116.4120,
                    "tags": ["历史", "公园"],
                },
                {
                    "spot_id": "beijing-004",
                    "spot_name": "颐和园",
                    "lat": 39.9990,
                    "lng": 116.2750,
                    "tags": ["历史", "园林"],
                },
                {
                    "spot_id": "beijing-005",
                    "spot_name": "南锣鼓巷",
                    "lat": 39.9370,
                    "lng": 116.4030,
                    "tags": ["美食", "逛街"],
                },
            ],
            "上海": [
                {
                    "spot_id": "shanghai-001",
                    "spot_name": "外滩",
                    "lat": 31.2397,
                    "lng": 121.4998,
                    "tags": ["夜景", "历史"],
                },
                {
                    "spot_id": "shanghai-002",
                    "spot_name": "东方明珠",
                    "lat": 31.2397,
                    "lng": 121.4998,
                    "tags": ["地标", "观光"],
                },
                {
                    "spot_id": "shanghai-003",
                    "spot_name": "豫园",
                    "lat": 31.2270,
                    "lng": 121.4920,
                    "tags": ["园林", "美食"],
                },
            ],
            "成都": [
                {
                    "spot_id": "chengdu-001",
                    "spot_name": "宽窄巷子",
                    "lat": 30.6680,
                    "lng": 104.0550,
                    "tags": ["美食", "文化"],
                },
                {
                    "spot_id": "chengdu-002",
                    "spot_name": "武侯祠",
                    "lat": 30.6420,
                    "lng": 104.0470,
                    "tags": ["历史", "三国"],
                },
                {
                    "spot_id": "chengdu-003",
                    "spot_name": "大熊猫繁育基地",
                    "lat": 30.7340,
                    "lng": 104.1470,
                    "tags": ["亲子", "动物"],
                },
            ],
        }
        rows = safety_data.get(city, [])
        return [POI(city=city, spot_type="attraction", **row) for row in rows]
