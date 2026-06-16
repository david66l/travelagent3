"""Price Query Skill - query ticket/meal/hotel prices."""

import logging

from core.settings import settings
from schemas import PriceInfo, ToolResult
from tools.base import Tool

logger = logging.getLogger(__name__)


class PriceQuerySkill(Tool):
    """Query prices for POIs (tickets, meals, hotels)."""

    name = "price"
    timeout = 3.0
    retries = 1
    cache_ttl = settings.cache_ttl_price

    async def query_price(
        self,
        poi_name: str,
        city: str,
        price_type: str,  # ticket / meal / hotel
    ) -> PriceInfo:
        """Backward-compatible entry point returning a PriceInfo."""
        result = await self.run({"poi_name": poi_name, "city": city, "price_type": price_type})
        data = result.data
        if isinstance(data, PriceInfo):
            return data
        if isinstance(data, dict):
            return PriceInfo(**data)
        return PriceInfo(
            poi_name=poi_name,
            price_type=price_type,
            data_source="unavailable",
            is_fallback=True,
            fallback_reason="price lookup returned no data",
        )

    async def execute(self, params: dict) -> ToolResult:
        """Try real price API; fallback to static estimates."""
        poi_name = params["poi_name"]
        city = params["city"]
        price_type = params["price_type"]

        if settings.amap_key or settings.tavily_api_key:
            try:
                info = await self._fetch_price_api(poi_name, city, price_type)
                return ToolResult(
                    data=info,
                    data_source="api",
                    confidence=0.8,
                )
            except Exception as exc:
                logger.warning("Price API failed for %s/%s: %s", poi_name, price_type, exc)

        info = self._static_estimate(poi_name, price_type)
        return ToolResult(
            data=info,
            data_source="fallback",
            confidence=0.6,
            is_fallback=True,
            fallback_reason="price api unavailable, using static estimate",
        )

    async def _fetch_price_api(self, poi_name: str, city: str, price_type: str) -> PriceInfo:
        """OTA/price platform API stub (replace with real endpoint)."""
        raise NotImplementedError("real price API not configured")

    def _static_estimate(self, poi_name: str, price_type: str) -> PriceInfo:
        price_ranges = {
            "ticket": "50-200元",
            "meal": "80-200元/人",
            "hotel": "300-800元/晚",
        }
        return PriceInfo(
            poi_name=poi_name,
            price_type=price_type,
            price_range=price_ranges.get(price_type),
            source="",
            data_source="fallback",
            confidence=0.6,
            is_fallback=True,
            fallback_reason="static price estimate",
        )
