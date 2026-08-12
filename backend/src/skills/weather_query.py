"""Weather Query Skill — 高德 + Open-Meteo + geography fallback.

Primary:   高德天气 API (free 5000 calls/day, Chinese cities, no extra key).
Secondary: Open-Meteo free API (no key, global coverage, p50 < 50ms).
Tertiary:  QWeather API when ``settings.weather_key`` is configured.
Fallback:  Latitude + month-driven estimation (never pure random).

高德天气 docs: https://lbs.amap.com/api/webservice/guide/api/weatherinfo
Open-Meteo docs: https://open-meteo.com/en/docs
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from datetime import datetime, timedelta
from typing import Optional

import httpx

from core.settings import settings
from schemas import ToolResult, WeatherDay
from tools.base import Tool

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# City → (lat, lng) lookup. Avoids extra geocoding HTTP round-trip.
# --------------------------------------------------------------------------- #
CITY_COORDS: dict[str, tuple[float, float]] = {
    "北京": (39.9042, 116.4074),  "上海": (31.2304, 121.4737),
    "广州": (23.1291, 113.2644),  "深圳": (22.5431, 114.0579),
    "成都": (30.5728, 104.0668),  "杭州": (30.2741, 120.1551),
    "西安": (34.3416, 108.9398),  "重庆": (29.4316, 106.9123),
    "苏州": (31.2990, 120.5853),  "南京": (32.0603, 118.7969),
    "厦门": (24.4798, 118.0894),  "青岛": (36.0671, 120.3826),
    "大理": (25.5916, 100.2299),  "丽江": (26.8721, 100.2299),
    "三亚": (18.2528, 109.5120),  "长沙": (28.2282, 112.9388),
    "武汉": (30.5928, 114.3055),  "昆明": (25.0389, 102.7183),
    "桂林": (25.2736, 110.2900),  "拉萨": (29.6500, 91.1000),
    "济南": (36.6509, 117.0116),
}

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
AMAP_WEATHER_URL = "https://restapi.amap.com/v3/weather/weatherInfo"

# 高德城市名称 → adcode (行政区划代码)
_CITY_ADCODE: dict[str, str] = {
    "北京": "110000", "上海": "310000", "广州": "440100", "深圳": "440300",
    "成都": "510100", "杭州": "330100", "西安": "610100", "重庆": "500000",
    "苏州": "320500", "南京": "320100", "厦门": "350200", "青岛": "370200",
    "大理": "532901", "丽江": "530700", "三亚": "460200", "长沙": "430100",
    "武汉": "420100", "昆明": "530100", "桂林": "450300", "拉萨": "540100",
    "济南": "370100", "郑州": "410100", "天津": "120000", "合肥": "340100",
    "福州": "350100", "南昌": "360100", "贵阳": "520100", "兰州": "620100",
    "哈尔滨": "230100", "长春": "220100", "沈阳": "210100",
}

# WMO Weather interpretation codes → Chinese description
# https://open-meteo.com/en/docs
_WMO_CODE_MAP: dict[int, str] = {
    0: "晴", 1: "晴", 2: "晴间多云", 3: "多云",
    45: "雾", 48: "雾凇",
    51: "小雨", 53: "中雨", 55: "大雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "阵雨", 81: "中阵雨", 82: "大阵雨",
    85: "阵雪", 86: "大阵雪",
    95: "雷阵雨", 96: "冰雹雷暴", 99: "强冰雹雷暴",
}


class WeatherQuerySkill(Tool):
    """Query weather forecast for a city and date range.

    Tries Open-Meteo first (free, no key), then QWeather (keyed),
    then falls back to geography-based estimation.
    """

    name = "weather"
    timeout = 5.0  # Open-Meteo is fast but give it headroom
    retries = 2
    cache_ttl = settings.cache_ttl_weather

    def __init__(self):
        super().__init__()
        self._http: Optional[httpx.AsyncClient] = None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(8.0),
                headers={"User-Agent": "TravelAgent2/2.0"},
            )
        return self._http

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

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
            if getattr(day, "is_fallback", True):
                day.data_source = result.data_source
                day.is_fallback = result.is_fallback
            days.append(day)
        return days

    async def execute(self, params: dict) -> ToolResult:
        """Try Open-Meteo → QWeather → geography fallback."""
        city = params["city"]
        start_date = params["start_date"]
        end_date = params["end_date"]

        # 1. 高德天气 (free 5000 calls/day, Chinese cities, no extra key)
        if settings.amap_key:
            try:
                days = await self._fetch_amap(city, start_date, end_date)
                if days:
                    return ToolResult(
                        data=days, data_source="api", confidence=0.90, latency_ms=0,
                    )
            except asyncio.TimeoutError:
                logger.warning("AMap weather timeout for %s", city)
            except Exception as exc:
                logger.warning("AMap weather failed for %s: %s", city, exc)

        # 2. Open-Meteo (free, no key, global)
        coords = self._resolve_coords(city)
        if coords is not None:
            try:
                days = await self._fetch_open_meteo(city, *coords, start_date, end_date)
                if days:
                    return ToolResult(
                        data=days, data_source="api", confidence=0.85, latency_ms=0,
                    )
            except asyncio.TimeoutError:
                logger.warning("Open-Meteo timeout for %s", city)
            except Exception as exc:
                logger.warning("Open-Meteo failed for %s: %s", city, exc)

        # 3. QWeather (if key configured)
        if settings.weather_key:
            try:
                days = await self._fetch_qweather(city, start_date, end_date)
                if days:
                    return ToolResult(
                        data=days, data_source="api", confidence=0.9, latency_ms=0,
                    )
            except Exception as exc:
                logger.warning("QWeather failed for %s: %s", city, exc)

        # 3. Geography-based fallback
        return ToolResult(
            data=self._geography_fallback(city, coords, start_date, end_date),
            data_source="fallback",
            confidence=0.6,
            is_fallback=True,
            fallback_reason="weather api unavailable, using geography-based estimation",
        )

    # ------------------------------------------------------------------ #
    # Open-Meteo (free, no API key)
    # ------------------------------------------------------------------ #

    async def _fetch_open_meteo(
        self, city: str, lat: float, lng: float,
        start_date: str, end_date: str,
    ) -> list[WeatherDay]:
        """Fetch daily forecast from Open-Meteo and map to WeatherDay."""
        client = await self._get_http()
        params = {
            "latitude": lat, "longitude": lng,
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max,wind_speed_10m_max"
            ),
            "timezone": "Asia/Shanghai",
            "start_date": start_date, "end_date": end_date,
            "forecast_days": 16,
        }
        resp = await client.get(OPEN_METEO_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})

        results: list[WeatherDay] = []
        dates = daily.get("time", [])
        codes = daily.get("weather_code", [])
        highs = daily.get("temperature_2m_max", [])
        lows = daily.get("temperature_2m_min", [])
        precips = daily.get("precipitation_probability_max", [])
        winds = daily.get("wind_speed_10m_max", [])

        for i, date_str in enumerate(dates):
            code = int(codes[i]) if i < len(codes) else 0
            condition = _WMO_CODE_MAP.get(code, "多云")
            high = int(round(highs[i])) if i < len(highs) else 25
            low = int(round(lows[i])) if i < len(lows) else 15
            precip = int(precips[i]) if i < len(precips) else 0
            wind = round(winds[i], 1) if i < len(winds) else None

            results.append(WeatherDay(
                date=date_str, condition=condition,
                temp_high=high, temp_low=low,
                precipitation_chance=precip, wind_speed=wind,
                recommendation=self._recommend(condition, high, low, precip),
                data_source="api", confidence=0.85, is_fallback=False,
            ))

        logger.info("Open-Meteo: %d days for %s (%s–%s)", len(results), city, start_date, end_date)
        return results

    # ------------------------------------------------------------------ #
    # 高德天气 (free, key already in settings.amap_key)
    # ------------------------------------------------------------------ #

    async def _fetch_amap(
        self, city: str, start_date: str, end_date: str,
    ) -> list[WeatherDay]:
        """Fetch weather from 高德地图 API (4-day forecast, free 5000 calls/day).

        Docs: https://lbs.amap.com/api/webservice/guide/api/weatherinfo
        """
        adcode = self._resolve_adcode(city)
        if not adcode:
            return []

        from core.amap_rate import amap_rate_gate

        client = await self._get_http()
        params = {
            "key": settings.amap_key,
            "city": adcode,
            "extensions": "all",  # "base" = today, "all" = 4-day forecast
        }
        await amap_rate_gate()
        resp = await client.get(AMAP_WEATHER_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "1":
            logger.warning("AMap weather API error: %s", data.get("info"))
            return []

        try:
            s = datetime.strptime(start_date, "%Y-%m-%d")
            e = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            return []

        results: list[WeatherDay] = []
        for forecast in data.get("forecasts", []):
            for cast in forecast.get("casts", []):
                date_str = cast.get("date", "")
                if not date_str:
                    continue
                try:
                    d = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    continue
                if not (s <= d <= e):
                    continue

                day_weather = cast.get("dayweather", "多云")
                day_temp = int(float(cast.get("daytemp_float", cast.get("daytemp", 25))))
                night_temp = int(float(cast.get("nighttemp_float", cast.get("nighttemp", 15))))

                results.append(WeatherDay(
                    date=date_str,
                    condition=day_weather,
                    temp_high=day_temp,
                    temp_low=night_temp,
                    precipitation_chance=_rain_from_weather(day_weather),
                    wind_speed=_parse_wind_level(cast.get("daypower", "")),
                    recommendation=self._recommend(
                        day_weather, day_temp, night_temp,
                        _rain_from_weather(day_weather),
                    ),
                    data_source="api", confidence=0.90, is_fallback=False,
                ))

        if results:
            logger.info("AMap weather: %d days for %s (%s–%s)",
                        len(results), city, start_date, end_date)
        return results

    @staticmethod
    def _resolve_adcode(city: str) -> Optional[str]:
        """Look up 高德 adcode for *city*."""
        if city in _CITY_ADCODE:
            return _CITY_ADCODE[city]
        normalized = city.rstrip("市省")
        if normalized in _CITY_ADCODE:
            return _CITY_ADCODE[normalized]
        for name, code in _CITY_ADCODE.items():
            if name in city or city in name:
                return code
        return None

    # ------------------------------------------------------------------ #
    # QWeather (keyed, higher confidence)
    # ------------------------------------------------------------------ #

    async def _fetch_qweather(
        self, city: str, start_date: str, end_date: str
    ) -> list[WeatherDay]:
        """QWeather API — requires ``settings.weather_key``."""
        # QWeather 7-day forecast endpoint
        # Docs: https://dev.qweather.com/docs/api/weather/weather-daily-forecast/
        city_id = await self._resolve_qweather_city_id(city)
        if not city_id:
            return []

        client = await self._get_http()
        url = "https://devapi.qweather.com/v7/weather/7d"
        params = {
            "location": city_id,
            "key": settings.weather_key,
        }
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != "200":
            logger.warning("QWeather API error: %s", data.get("code"))
            return []

        results: list[WeatherDay] = []
        try:
            s = datetime.strptime(start_date, "%Y-%m-%d")
            e = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            return []

        for day_data in data.get("daily", []):
            date_str = day_data.get("fxDate", "")
            if not date_str:
                continue
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue
            if not (s <= d <= e):
                continue

            results.append(WeatherDay(
                date=date_str,
                condition=day_data.get("textDay", "多云"),
                temp_high=int(day_data.get("tempMax", 25)),
                temp_low=int(day_data.get("tempMin", 15)),
                precipitation_chance=int(day_data.get("pop", 0)),
                wind_speed=self._parse_wind_speed(day_data.get("windSpeedDay", "")),
                recommendation=self._recommend(
                    day_data.get("textDay", ""),
                    int(day_data.get("tempMax", 25)),
                    int(day_data.get("tempMin", 15)),
                    int(day_data.get("pop", 0)),
                ),
                data_source="api", confidence=0.9, is_fallback=False,
            ))

        if results:
            logger.info("QWeather: %d days for %s", len(results), city)
        return results

    async def _resolve_qweather_city_id(self, city: str) -> Optional[str]:
        """Resolve city name → QWeather LocationID."""
        # QWeather city lookup
        # Docs: https://dev.qweather.com/docs/api/geo/city-lookup/
        client = await self._get_http()
        url = "https://geoapi.qweather.com/v2/city/lookup"
        params = {"location": city, "key": settings.weather_key}
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") == "200" and data.get("location"):
                return data["location"][0]["id"]
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_wind_speed(raw: str) -> Optional[int]:
        """Parse QWeather wind speed like '3-4级' → average km/h."""
        import re
        m = re.findall(r"(\d+)", raw)
        if m:
            vals = [int(x) for x in m]
            return sum(vals) // len(vals)
        return None

    # ------------------------------------------------------------------ #
    # Geography-based fallback
    # ------------------------------------------------------------------ #

    def _geography_fallback(
        self, city: str, coords: Optional[tuple[float, float]],
        start_date: str, end_date: str,
    ) -> list[WeatherDay]:
        """Latitude + month-driven estimation. Marked '（估算数据）'."""
        lat = coords[0] if coords else 30.0
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            return []

        results: list[WeatherDay] = []
        current = start
        while current <= end:
            high, low = _estimate_temp(lat, current.month)
            high += random.randint(-3, 3)
            low += random.randint(-2, 2)
            condition = _estimate_condition(lat, current.month)
            precip = _estimate_precip(lat, current.month)

            results.append(WeatherDay(
                date=current.strftime("%Y-%m-%d"),
                condition=condition,
                temp_high=high, temp_low=low,
                precipitation_chance=precip,
                recommendation=self._recommend(condition, high, low, precip) + "（估算数据）",
                data_source="fallback", confidence=0.6, is_fallback=True,
                fallback_reason="geography-based estimation",
            ))
            current += timedelta(days=1)

        logger.info("Geography fallback: %d days for %s (lat=%.1f)", len(results), city, lat)
        return results

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_coords(city: str) -> Optional[tuple[float, float]]:
        if city in CITY_COORDS:
            return CITY_COORDS[city]
        normalized = city.rstrip("市省")
        if normalized in CITY_COORDS:
            return CITY_COORDS[normalized]
        for name, coords in CITY_COORDS.items():
            if name in city or city in name:
                return coords
        return None

    @staticmethod
    def _recommend(condition: str, high: int, low: int, precip: int) -> str:
        parts: list[str] = []
        if "雨" in condition or "雪" in condition:
            parts.append("有降水，建议携带雨具")
        elif precip > 50:
            parts.append("降水概率较高，建议备伞")
        if high > 32:
            parts.append("高温天气，注意防晒补水")
        elif high < 10:
            parts.append("气温较低，注意保暖")
        if low < 8:
            parts.append("早晚寒冷，建议带厚外套")
        elif 8 <= low < 15:
            parts.append("早晚偏凉，建议带薄外套")
        if abs(high - low) > 12:
            parts.append("昼夜温差大，注意衣物增减")
        if not parts:
            parts.append("天气适宜出行")
        return "；".join(parts)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _rain_from_weather(condition: str) -> int:
    """Estimate precipitation probability from Chinese weather description."""
    if any(kw in condition for kw in ("雨", "雷")):
        return 70
    if any(kw in condition for kw in ("雪",)):
        return 60
    if "阴" in condition:
        return 30
    if "多云" in condition:
        return 15
    return 5


def _parse_wind_level(power: str) -> Optional[int]:
    """Parse AMap wind power like '1-3' → average level."""
    import re
    m = re.findall(r"(\d+)", power)
    if m:
        vals = [int(x) for x in m]
        return sum(vals) // len(vals)
    return None


# --------------------------------------------------------------------------- #
# Climate estimation — latitude × month, calibrated against China climate data
# --------------------------------------------------------------------------- #

def _estimate_temp(lat: float, month: int) -> tuple[int, int]:
    """Return (high, low) based on latitude and month.

    Calibration points:
      lat 18 (Sanya)   → Jul high ~33, Jan low ~18
      lat 30 (Hangzhou) → Jul high ~33, Jan low ~3
      lat 40 (Beijing)  → Jul high ~31, Jan low ~-8
    """
    phase = (month - 7) / 6 * math.pi
    amp_high = (lat - 15) * 0.55
    amp_low = (lat - 15) * 0.45
    base_high = 33 - (lat - 20) * 0.15
    base_low = 24 - (lat - 20) * 0.50

    high = int(round(base_high - amp_high * (1 - math.cos(phase)) / 2))
    low = int(round(base_low - amp_low * (1 - math.cos(phase)) / 2))
    return high, low


def _estimate_condition(lat: float, month: int) -> str:
    """Plausible weather condition by latitude and season."""
    is_summer = 5 <= month <= 9
    r = random.random() * 100

    if lat < 25:       # tropical / south
        t = (15, 40, 70, 90) if is_summer else (25, 60, 85, 95)
    elif lat < 32:     # central
        t = (20, 50, 75, 90) if is_summer else (30, 65, 80, 95)
    else:              # north
        t = (30, 65, 85, 95) if is_summer else (40, 75, 90, 97)

    if r < t[0]:       return "晴"
    elif r < t[1]:     return "多云"
    elif r < t[2]:     return "小雨" if is_summer or lat < 32 else "阴"
    elif r < t[3]:     return "中雨" if is_summer else "小雪"
    else:              return "雷阵雨" if is_summer else "中雨"


def _estimate_precip(lat: float, month: int) -> int:
    if lat < 25:
        base = 45 if 5 <= month <= 9 else 25
    elif lat < 32:
        base = 35 if 5 <= month <= 9 else 20
    else:
        base = 25 if 5 <= month <= 9 else 10
    return max(0, min(100, base + random.randint(-15, 15)))
