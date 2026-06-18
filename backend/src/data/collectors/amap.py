"""
高德地图 POI 采集器 — Layer 1 多源数据采集。

每天凌晨增量同步，每周全量刷新。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# 高德 POI 类型映射
AMAP_TYPE_CATEGORY = {
    "风景名胜": "attraction",
    "公园广场": "attraction",
    "寺庙道观": "attraction",
    "纪念馆": "attraction",
    "中餐厅": "restaurant",
    "外国餐厅": "restaurant",
    "小吃快餐店": "restaurant",
    "咖啡厅": "restaurant",
    "宾馆酒店": "hotel",
    "经济型酒店": "hotel",
    "青年旅舍": "hotel",
    "购物中心": "shopping",
}


@dataclass
class RawPOI:
    """标准化 POI 数据结构。"""
    name: str
    city: str
    category: str
    lat: float
    lng: float
    address: str = ""
    ticket_price: Optional[float] = None
    open_time: Optional[str] = None
    close_time: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    source: str = "amap"
    source_updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AmapCollector:
    """高德地图 POI API 采集器。"""

    BASE_URL = "https://restapi.amap.com/v3"

    def __init__(self, api_key: str):
        self._key = api_key
        self._client = httpx.AsyncClient(timeout=10.0)

    async def search_pois(
        self, city: str, keywords: str = "", types: str = ""
    ) -> list[RawPOI]:
        """搜索城市 POI。"""
        all_pois: list[RawPOI] = []

        # 分类型搜索（高德一次最多返回 1000 条，分类型可获取更多）
        search_types = types.split("|") if types else [
            "风景名胜|公园广场|寺庙道观|纪念馆",
            "中餐厅|外国餐厅|小吃快餐店",
        ]

        for stype in search_types:
            pois = await self._search_page(city, keywords, stype)
            all_pois.extend(pois)

        return all_pois

    async def _search_page(
        self, city: str, keywords: str, types: str
    ) -> list[RawPOI]:
        """分页搜索 POI。"""
        pois: list[RawPOI] = []
        page = 1

        while page <= 5:  # 最多 5 页，避免超量
            params = {
                "key": self._key,
                "keywords": keywords or "",
                "types": types,
                "city": city,
                "citylimit": "true",
                "offset": 20,
                "page": page,
                "extensions": "all",
                "output": "json",
            }

            try:
                resp = await self._client.get(
                    f"{self.BASE_URL}/place/text", params=params
                )
                data = resp.json()

                if data.get("status") != "1":
                    break

                for poi in data.get("pois", []):
                    raw = self._normalize(city, poi)
                    if raw:
                        pois.append(raw)

                # 检查是否还有下一页
                total = int(data.get("count", 0))
                if page * 20 >= total or page * 20 >= 100:
                    break
                page += 1

            except Exception as exc:
                logger.warning("Amap API error for %s page %d: %s", city, page, exc)
                break

        return pois

    def _normalize(self, city: str, raw: dict) -> Optional[RawPOI]:
        """高德 POI → 标准化 RawPOI。"""
        name = raw.get("name", "").strip()
        if not name:
            return None

        # 坐标
        location = raw.get("location", "0,0")
        try:
            lng_str, lat_str = location.split(",")
            lng, lat = float(lng_str), float(lat_str)
        except (ValueError, AttributeError):
            return None

        # 分类映射
        amap_type = raw.get("type", "").split(";")[0] if raw.get("type") else ""
        category = AMAP_TYPE_CATEGORY.get(amap_type, "attraction")

        # 地址
        address = raw.get("address", "")

        # 标签
        tags = []
        type_parts = (raw.get("type") or "").split(";")
        for part in type_parts:
            part = part.split("|")
            tags.extend([t for t in part if t and t not in tags])

        # 门票（高德部分景点有返利价格字段）
        ticket_price = None
        biz_ext = raw.get("biz_ext", {})
        if isinstance(biz_ext, dict):
            cost_str = biz_ext.get("cost", "")
            if cost_str:
                try:
                    ticket_price = float(cost_str)
                except (ValueError, TypeError):
                    pass

        # 营业时间
        open_time = None
        close_time = None
        biz_info = raw.get("biz_ext", {})
        if isinstance(biz_info, dict):
            rating = biz_info.get("rating")
            # 高德没有直接返回 open/close time，需要通过详情接口获取
            # 这里先留空，后续通过详情接口补全

        return RawPOI(
            name=name,
            city=city,
            category=category,
            lat=lat,
            lng=lng,
            address=address or "",
            ticket_price=ticket_price,
            open_time=open_time,
            close_time=close_time,
            tags=tags[:5],  # 最多 5 个标签
            source="amap",
            source_updated_at=datetime.now(timezone.utc),
        )

    async def close(self):
        await self._client.aclose()
