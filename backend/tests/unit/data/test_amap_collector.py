"""Tests for AMap provider normalization at the data boundary."""

import pytest
from unittest.mock import AsyncMock

from data.collectors.amap import AmapCollector


@pytest.mark.parametrize(
    ("amap_type", "expected"),
    [
        ("餐饮服务;中餐厅;苏帮菜", "restaurant"),
        ("风景名胜;公园广场;公园", "attraction"),
        ("住宿服务;宾馆酒店", "hotel"),
        ("购物服务;商场", "shopping"),
    ],
)
@pytest.mark.asyncio
async def test_amap_normalize_uses_full_type_hierarchy(amap_type, expected):
    collector = AmapCollector("test-key")
    try:
        result = collector._normalize(
            "苏州",
            {
                "name": "测试地点",
                "location": "120.6,31.3",
                "type": amap_type,
                "address": "测试路",
            },
        )
    finally:
        await collector.close()

    assert result is not None
    assert result.category == expected


@pytest.mark.asyncio
async def test_amap_search_respects_total_result_limit():
    collector = AmapCollector("test-key")
    collector._search_page = AsyncMock(return_value=[object(), object(), object()])
    try:
        result = await collector.search_pois("苏州", types="风景名胜|公园广场", limit=3)
    finally:
        await collector.close()

    assert len(result) == 3
    collector._search_page.assert_awaited_once_with("苏州", "", "风景名胜", limit=3)
