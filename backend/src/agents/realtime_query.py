"""Real-time Query Agent - parallel POI/weather/price queries."""

import asyncio
from typing import Optional

from schemas import ScoredPOI, WeatherDay, PriceInfo, UserProfile
from skills.poi_search import POISearchSkill
from skills.weather_query import WeatherQuerySkill
from skills.price_query import PriceQuerySkill


class RealtimeQueryAgent:
    """Query real-time information via search and crawling."""

    def __init__(self):
        self.poi_skill = POISearchSkill()
        self.weather_skill = WeatherQuerySkill()
        self.price_skill = PriceQuerySkill()

    async def query_all(
        self,
        city: str,
        dates: list[str],
        keywords: list[str],
        profile: UserProfile,
    ) -> tuple[list[ScoredPOI], list[WeatherDay], dict[str, PriceInfo]]:
        """Parallel query POIs, weather, and prices."""
        # POI search and weather query in parallel
        poi_task = self.poi_skill.search_pois(city, keywords)
        weather_task = self.weather_skill.query(
            city, dates[0], dates[-1] if len(dates) > 1 else dates[0]
        )

        pois, weather = await asyncio.gather(poi_task, weather_task)

        # Prices are queried after POIs are known
        prices = await self._query_prices(city, pois[:10])

        return pois, weather, prices

    async def query_pois(
        self,
        city: str,
        keywords: list[str],
    ) -> list[ScoredPOI]:
        """Search for POIs."""
        return await self.poi_skill.search_pois(city, keywords)

    async def query_weather(
        self,
        city: str,
        start_date: str,
        end_date: str,
    ) -> list[WeatherDay]:
        """Query weather forecast."""
        return await self.weather_skill.query(city, start_date, end_date)

    async def _query_prices(
        self,
        city: str,
        pois: list[ScoredPOI],
    ) -> dict[str, PriceInfo]:
        """Query prices for top POIs."""
        price_tasks = [
            self.price_skill.query_price(poi.name, city, "ticket")
            for poi in pois
            if poi.category == "attraction"
        ]
        if not price_tasks:
            return {}

        prices = await asyncio.gather(*price_tasks)
        return {p.poi_name: p for p in prices}
