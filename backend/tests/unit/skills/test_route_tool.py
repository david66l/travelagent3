"""Tests for RouteTool."""

import pytest
from unittest.mock import AsyncMock, patch

from schemas import Location, RouteInfo
from skills.route_tool import RouteTool


class TestRouteTool:
    def setup_method(self):
        self.tool = RouteTool()

    @pytest.mark.asyncio
    async def test_route_fallback(self, mock_redis):
        mock_redis.get_json = AsyncMock(return_value=None)
        mock_redis.set_json = AsyncMock()
        origin = Location(lat=31.2397, lng=121.4998)
        destination = Location(lat=31.1413, lng=121.6618)
        result = await self.tool.route(origin, destination)
        assert isinstance(result, RouteInfo)
        assert result.distance_m > 0
        assert result.duration_min > 0
        assert result.data_source == "fallback"
        assert result.is_fallback is True

    @pytest.mark.asyncio
    async def test_route_api_path(self, mock_redis):
        mock_redis.get_json = AsyncMock(return_value=None)
        mock_redis.set_json = AsyncMock()
        from schemas import ToolResult

        api_route = RouteInfo(
            origin=Location(lat=0.0, lng=0.0),
            destination=Location(lat=1.0, lng=1.0),
            distance_m=1000,
            duration_min=10,
            mode="transit",
            data_source="api",
            is_fallback=False,
        )
        with patch.object(
            self.tool,
            "execute",
            AsyncMock(return_value=ToolResult(data=api_route, data_source="api")),
        ):
            result = await self.tool.route(Location(lat=0.0, lng=0.0), Location(lat=1.0, lng=1.0))
            assert result.data_source == "api"
            assert result.is_fallback is False
