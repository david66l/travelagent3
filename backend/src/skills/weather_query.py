"""Weather Query Skill - search and parse weather data with Redis caching."""

import logging
import random
from datetime import datetime, timedelta


from core.settings import settings
from schemas import ToolResult, WeatherDay
from tools.base import Tool

logger = logging.getLogger(__name__)


class WeatherQuerySkill(Tool):
    """Query weather forecast for a city and date range."""

    name = "weather"
    timeout = 3.0
    retries = 2
    cache_ttl = settings.cache_ttl_weather

    async def query(
        self,
        city: str,
        start_date: str,
        end_date: str,
    ) -> list[WeatherDay]:
        """Backward-compatible entry point returning a list of WeatherDay."""
        result = await self.run({"city": city, "start_date": start_date, "end_date": end_date})
        data = result.data or []
        if not isinstance(data, list):
            return []
        days = []
        for w in data:
            day = w if isinstance(w, WeatherDay) else WeatherDay(**w)
            # Propagate the result-level data source when items were cached without it.
            if getattr(day, "is_fallback", True):
                day.data_source = result.data_source
                day.is_fallback = result.is_fallback
            days.append(day)
        return days

    async def execute(self, params: dict) -> ToolResult:
        """Try real weather API; fallback to simulated data."""
        city = params["city"]
        start_date = params["start_date"]
        end_date = params["end_date"]

        if settings.weather_key:
            try:
                days = await self._fetch_weather_api(city, start_date, end_date)
                return ToolResult(
                    data=days,
                    data_source="api",
                    confidence=0.9,
                    latency_ms=0,
                )
            except Exception as exc:
                logger.warning("Weather API failed for %s: %s", city, exc)

        return ToolResult(
            data=self._simulate_weather(start_date, end_date),
            data_source="fallback",
            confidence=0.7,
            is_fallback=True,
            fallback_reason="weather api unavailable, using simulated data",
        )

    async def _fetch_weather_api(
        self, city: str, start_date: str, end_date: str
    ) -> list[WeatherDay]:
        """QWeather API stub (replace with real endpoint when available)."""
        # Placeholder implementation that always raises so we exercise fallback.
        raise NotImplementedError("real weather API not configured")

    def _simulate_weather(self, start_date: str, end_date: str) -> list[WeatherDay]:
        """Generate deterministic-looking simulated weather."""
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            return []

        weather_days = []
        conditions = ["晴", "多云", "阴", "小雨", "中雨"]
        current = start
        while current <= end:
            high = random.randint(20, 32)
            low = high - random.randint(5, 12)
            weather_days.append(
                WeatherDay(
                    date=current.strftime("%Y-%m-%d"),
                    condition=random.choice(conditions),
                    temp_high=high,
                    temp_low=low,
                    precipitation_chance=random.randint(0, 40),
                    recommendation=self._recommend(conditions[0], high, low),
                    data_source="fallback",
                    confidence=0.7,
                    is_fallback=True,
                    fallback_reason="simulated weather fallback",
                )
            )
            current += timedelta(days=1)
        return weather_days

    def _recommend(self, condition: str, high: int, low: int) -> str:
        if "雨" in condition:
            return "有雨，建议携带雨具"
        if high > 30:
            return "气温较高，注意防晒补水"
        if low < 15:
            return "早晚温差大，建议带外套"
        return "天气适宜出行"
