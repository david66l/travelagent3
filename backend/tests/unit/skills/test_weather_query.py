"""Tests for WeatherQuerySkill."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from skills.weather_query import WeatherQuerySkill


class TestWeatherQuerySkill:
    """Test weather query and caching."""

    def setup_method(self):
        self.skill = WeatherQuerySkill()

    @pytest.mark.asyncio
    async def test_query_basic(self, mock_redis):
        mock_redis.get_json = AsyncMock(return_value=None)
        mock_redis.set_json = AsyncMock()
        result = await self.skill.query("北京", "2026-05-01", "2026-05-03")
        assert len(result) == 3  # 3 days
        for day in result:
            assert day.date is not None
            assert day.condition is not None
            assert day.temp_high >= day.temp_low
            assert day.data_source == "fallback"
            assert day.is_fallback is True

    @pytest.mark.asyncio
    async def test_query_cache_hit(self, mock_redis):
        cached = {
            "data": [
                {
                    "date": "2026-05-01",
                    "condition": "晴",
                    "temp_high": 25,
                    "temp_low": 15,
                    "precipitation_chance": 0,
                },
            ],
            "data_source": "api",
        }
        mock_redis.get_json = AsyncMock(return_value=cached)
        result = await self.skill.query("北京", "2026-05-01", "2026-05-01")
        assert len(result) == 1
        assert result[0].condition == "晴"
        assert result[0].data_source == "api"

    @pytest.mark.asyncio
    async def test_query_invalid_date(self, mock_redis):
        mock_redis.get_json = AsyncMock(return_value=None)
        result = await self.skill.query("北京", "invalid", "invalid")
        assert result == []

    @pytest.mark.asyncio
    async def test_query_single_day(self, mock_redis):
        mock_redis.get_json = AsyncMock(return_value=None)
        mock_redis.set_json = AsyncMock()
        result = await self.skill.query("上海", "2026-06-01", "2026-06-01")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_open_meteo_date_range_does_not_send_conflicting_forecast_days(self):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "daily": {
                "time": ["2026-09-02"],
                "weather_code": [0],
                "temperature_2m_max": [30],
                "temperature_2m_min": [22],
                "precipitation_probability_max": [10],
                "wind_speed_10m_max": [8.5],
            }
        }
        client = AsyncMock()
        client.get.return_value = response
        self.skill._http = client

        result = await self.skill._fetch_open_meteo(
            "上海", 31.2304, 121.4737, "2026-09-02", "2026-09-02"
        )

        assert len(result) == 1
        assert result[0].wind_speed == 8
        params = client.get.await_args.kwargs["params"]
        assert params["start_date"] == "2026-09-02"
        assert params["end_date"] == "2026-09-02"
        assert "forecast_days" not in params

    def test_recommend_rain(self):
        rec = self.skill._recommend("小雨", 20, 15)
        assert "雨具" in rec

    def test_recommend_hot(self):
        rec = self.skill._recommend("晴", 35, 25)
        assert "防晒" in rec or "高温" in rec

    def test_recommend_cold(self):
        rec = self.skill._recommend("晴", 10, 5)
        assert "外套" in rec

    def test_recommend_normal(self):
        rec = self.skill._recommend("多云", 22, 15)
        assert "适宜" in rec or "出行" in rec
