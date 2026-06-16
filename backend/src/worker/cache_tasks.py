"""Celery tasks for Redis cache warmup (PRD §4.8.3)."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from core.celery_app import celery_app
from core.settings import settings
from worker.memory_tasks import _ensure_redis, _run_async

logger = logging.getLogger(__name__)


async def _warm_poi_cities() -> dict[str, int]:
    from skills.poi_search import POISearchSkill

    skill = POISearchSkill()
    warmed = 0
    for city in settings.seed_cities_list[:settings.cache_warm_top_n_cities]:
        try:
            await skill.search_pois(city, [])
            warmed += 1
        except Exception as exc:
            logger.warning("POI warmup failed for %s: %s", city, exc)
    return {"poi_cities": warmed}


async def _warm_weather_cities() -> dict[str, int]:
    from skills.weather_query import WeatherQuerySkill

    skill = WeatherQuerySkill()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    warmed = 0
    for city in settings.seed_cities_list[:settings.cache_warm_top_n_cities]:
        try:
            await skill.query(city, tomorrow, tomorrow)
            warmed += 1
        except Exception as exc:
            logger.warning("Weather warmup failed for %s: %s", city, exc)
    return {"weather_cities": warmed}


@celery_app.task(bind=True, max_retries=2, default_retry_delay=300)  # type: ignore[untyped-decorator]
def warm_top_cities_cache(self: Any) -> dict[str, int]:
    """Warm POI and next-day weather caches for top seed cities."""
    try:
        _run_async(_ensure_redis())
        poi_stats = _run_async(_warm_poi_cities())
        weather_stats = _run_async(_warm_weather_cities())
        result = {**poi_stats, **weather_stats}
        logger.info("Cache warmup complete: %s", result)
        return result
    except Exception as exc:
        logger.exception("warm_top_cities_cache failed")
        raise self.retry(exc=exc)
