"""Weather Query Skill - search and parse weather data with Redis caching."""

from schemas import WeatherDay
from skills.web_search import WebSearchSkill
from core.redis_client import redis_client


class WeatherQuerySkill:
    """Query weather forecast for a city and date range."""

    def __init__(self):
        self.search_skill = WebSearchSkill()

    async def query(
        self,
        city: str,
        start_date: str,
        end_date: str,
    ) -> list[WeatherDay]:
        """Query weather for the given date range with Redis caching."""
        cache_key = f"weather:{city}:{start_date}:{end_date}"

        # Try cache first
        try:
            cached = await redis_client.get_json(cache_key)
            if cached:
                return [WeatherDay(**w) for w in cached]
        except Exception:
            pass

        # Fallback: search + simulated data (MVP)
        query = f"{city} 天气预报 {start_date} 到 {end_date}"
        _results = await self.search_skill.search(query, top_n=3)

        # For MVP, return simulated weather data
        # In production, would crawl weather pages and parse
        import random
        from datetime import datetime, timedelta

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
                )
            )
            current += timedelta(days=1)

        # Cache result
        try:
            await redis_client.set_json(
                cache_key,
                [w.model_dump() for w in weather_days],
                ttl=1800,  # 30 minutes
            )
        except Exception:
            pass

        return weather_days

    def _recommend(self, condition: str, high: int, low: int) -> str:
        if "雨" in condition:
            return "有雨，建议携带雨具"
        if high > 30:
            return "气温较高，注意防晒补水"
        if low < 15:
            return "早晚温差大，建议带外套"
        return "天气适宜出行"
