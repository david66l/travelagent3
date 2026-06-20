"""Price Query Skill — tiered estimation by city level + POI category.

Fallback estimation uses:
- City tier (T1–T4) for cost-of-living adjustment
- POI category + tags for type-specific pricing
- Known price anchors for major attractions (from city_data)

When Tavily/Amap is configured, the estimate is enriched with web search.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from core.settings import settings
from schemas import PriceInfo, ToolResult
from tools.base import Tool

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# City tier → cost multiplier (relative to T2 baseline = 1.0)
# Based on 2025 China city consumption level data.
# --------------------------------------------------------------------------- #
CITY_TIER: dict[str, int] = {
    "北京": 1, "上海": 1, "广州": 1, "深圳": 1,
    "杭州": 1, "成都": 2, "重庆": 2, "武汉": 2, "南京": 2,
    "苏州": 2, "西安": 2, "长沙": 2, "天津": 2, "郑州": 2,
    "东莞": 2, "青岛": 2, "宁波": 2, "厦门": 2, "昆明": 2,
    "大连": 2, "福州": 2, "合肥": 2, "无锡": 2, "沈阳": 2,
    "济南": 2, "哈尔滨": 3, "长春": 3, "石家庄": 3,
    "南宁": 3, "贵阳": 3, "南昌": 3, "太原": 3, "兰州": 3,
    "海口": 3, "三亚": 2, "大理": 3, "丽江": 3, "桂林": 3,
    "拉萨": 3, "乌鲁木齐": 3,
}

_TIER_MULTIPLIER = {1: 1.5, 2: 1.0, 3: 0.7, None: 1.0}

# --------------------------------------------------------------------------- #
# Base prices by category (T2 city baseline, in CNY)
# --------------------------------------------------------------------------- #
_BASE_PRICES: dict[str, dict] = {
    "ticket": {
        # 5A / landmark attractions
        "5A级景区":  (80, 200),
        "主题公园":  (150, 500),
        "博物馆":    (0, 60),
        "自然风光":  (30, 120),
        "历史古迹":  (30, 100),
        "寺庙":      (0, 50),
        "default":   (30, 150),
    },
    "meal": {
        "高端餐厅":  (200, 600),
        "特色餐厅":  (80, 200),
        "小吃":      (10, 40),
        "夜市":      (20, 60),
        "早茶":      (30, 80),
        "火锅":      (80, 150),
        "快餐":      (20, 40),
        "default":   (40, 120),
    },
    "hotel": {
        "经济型":    (100, 250),
        "舒适型":    (250, 500),
        "高档型":    (500, 1000),
        "豪华型":    (1000, 3000),
        "民宿":      (150, 400),
        "default":   (200, 600),
    },
}

# --------------------------------------------------------------------------- #
# Known price anchors for major attractions (authoritative, from official sites)
# --------------------------------------------------------------------------- #
_KNOWN_TICKETS: dict[str, tuple[int, int]] = {
    "故宫": (40, 60),          "天坛": (15, 34),
    "颐和园": (20, 30),        "长城": (35, 45),
    "兵马俑": (120, 120),      "大雁塔": (40, 50),
    "外滩": (0, 0),            "东方明珠": (199, 299),
    "豫园": (30, 40),           "上海迪士尼": (475, 799),
    "灵隐寺": (30, 45),        "西湖": (0, 0),
    "武侯祠": (50, 60),        "杜甫草堂": (50, 60),
    "大熊猫繁育基地": (55, 55), "青城山": (80, 90),
    "都江堰": (80, 90),        "趵突泉": (40, 45),
    "千佛山": (28, 30),        "大明湖": (0, 0),
    "中山陵": (0, 0),           "夫子庙": (0, 0),
    "鼓浪屿": (30, 50),        "张家界": (225, 248),
    "九寨沟": (169, 220),      "黄山": (190, 230),
    "桂林漓江": (100, 215),    "布达拉宫": (100, 200),
}


class PriceQuerySkill(Tool):
    """Query prices using tiered city model + known anchors."""

    name = "price"
    timeout = 3.0
    retries = 1
    cache_ttl = settings.cache_ttl_price

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def query_price(
        self, poi_name: str, city: str, price_type: str,
    ) -> PriceInfo:
        result = await self.run({
            "poi_name": poi_name, "city": city, "price_type": price_type,
        })
        data = result.data
        if isinstance(data, PriceInfo):
            return data
        if isinstance(data, dict):
            return PriceInfo(**data)
        return PriceInfo(
            poi_name=poi_name, price_type=price_type,
            data_source="unavailable", is_fallback=True,
            fallback_reason="price lookup returned no data",
        )

    async def execute(self, params: dict) -> ToolResult:
        poi_name = params["poi_name"]
        city = params["city"]
        price_type = params["price_type"]

        # 1. Web search enrichment (optional, non-blocking)
        if settings.tavily_api_key:
            try:
                info = await self._fetch_price_api(poi_name, city, price_type)
                if info and not info.is_fallback:
                    return ToolResult(data=info, data_source="api", confidence=0.85)
            except Exception:
                pass

        # 2. Structured estimate
        info = self._structured_estimate(poi_name, city, price_type)
        return ToolResult(
            data=info,
            data_source="built_in",
            confidence=0.75,
            is_fallback=True,
            fallback_reason="structured price model (city tier + category)",
        )

    # ------------------------------------------------------------------ #
    # Web search enhancement
    # ------------------------------------------------------------------ #

    async def _fetch_price_api(
        self, poi_name: str, city: str, price_type: str,
    ) -> Optional[PriceInfo]:
        """Try Tavily search for real price information."""
        try:
            from skills.tavily_search import TavilySearchSkill
            tavily = TavilySearchSkill()
            query = f"{city} {poi_name} "
            if price_type == "ticket":
                query += "门票价格 2025"
            elif price_type == "meal":
                query += "人均消费"
            else:
                query += "住宿价格"

            results, answer = await tavily.search_with_context(query, top_n=3)
            if answer:
                range_ = _extract_price_range(answer)
                if range_:
                    return PriceInfo(
                        poi_name=poi_name, price_type=price_type,
                        price_range=range_, currency="CNY",
                        source="tavily", data_source="api",
                        confidence=0.85, is_fallback=False,
                    )
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ #
    # Structured estimation
    # ------------------------------------------------------------------ #

    def _structured_estimate(
        self, poi_name: str, city: str, price_type: str,
    ) -> PriceInfo:
        """Estimate price using city tier + POI category + known anchors."""
        tier = CITY_TIER.get(city)
        multiplier = _TIER_MULTIPLIER[tier] if tier else _TIER_MULTIPLIER[None]

        # Check known anchors first
        if price_type == "ticket" and poi_name in _KNOWN_TICKETS:
            lo, hi = _KNOWN_TICKETS[poi_name]
            return PriceInfo(
                poi_name=poi_name, price_type=price_type,
                price_range=f"¥{lo}-{hi}",
                currency="CNY", source="official",
                data_source="built_in", confidence=0.9,
                is_fallback=True,
                fallback_reason="known ticket price anchor",
            )

        # Category-based estimate
        sub = _classify_poi(poi_name, price_type)
        base_lo, base_hi = _BASE_PRICES.get(price_type, {}).get(
            sub, _BASE_PRICES.get(price_type, {}).get("default", (50, 200))
        )
        lo = max(0, int(base_lo * multiplier))
        hi = max(lo + 10, int(base_hi * multiplier))

        return PriceInfo(
            poi_name=poi_name, price_type=price_type,
            price_range=f"¥{lo}-{hi}",
            currency="CNY", source="",
            data_source="built_in", confidence=0.7,
            is_fallback=True,
            fallback_reason=f"structured estimate ({_tier_label(tier)} · {sub})",
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _classify_poi(name: str, price_type: str) -> str:
    """Classify a POI into a sub-category for pricing."""
    n = name
    if price_type == "ticket":
        if any(kw in n for kw in ("博物馆", "博物院", "展览")):
            return "博物馆"
        if any(kw in n for kw in ("乐园", "欢乐谷", "迪士尼", "海洋馆", "动物园", "植物园")):
            return "主题公园"
        if any(kw in n for kw in ("山", "湖", "海", "公园", "森林", "溪", "峡谷", "瀑布", "岛")):
            return "自然风光"
        if any(kw in n for kw in ("寺", "庙", "宫", "祠", "观", "教堂", "塔")):
            return "寺庙" if "宫" not in n else "历史古迹"
        if any(kw in n for kw in ("遗址", "古城", "城墙", "故居", "陵", "旧址")):
            return "历史古迹"
        if any(kw in n for kw in ("长城", "故宫", "兵马俑", "颐和园", "天坛", "布达拉宫")):
            return "5A级景区"
        return "default"

    if price_type == "meal":
        if any(kw in n for kw in ("小吃", "串", "面", "粉", "包")):
            return "小吃"
        if any(kw in n for kw in ("火锅", "涮")):
            return "火锅"
        if any(kw in n for kw in ("早茶", "点心", "饮茶")):
            return "早茶"
        if any(kw in n for kw in ("夜市", "大排档", "路边")):
            return "夜市"
        if any(kw in n for kw in ("酒店", "私房", "米其林", "高端", "会所")):
            return "高端餐厅"
        return "default"

    if price_type == "hotel":
        if any(kw in n for kw in ("快捷", "如家", "汉庭", "7天", "锦江之星", "青旅")):
            return "经济型"
        if any(kw in n for kw in ("民宿", "客栈")):
            return "民宿"
        if any(kw in n for kw in ("希尔顿", "万豪", "洲际", "凯悦", "香格里拉", "四季", "丽思")):
            return "豪华型"
        if any(kw in n for kw in ("喜来登", "威斯汀", "皇冠", "铂尔曼")):
            return "高档型"
        return "default"

    return "default"


def _tier_label(tier: Optional[int]) -> str:
    if tier == 1:
        return "一线城市"
    if tier == 2:
        return "新一线/二线"
    if tier == 3:
        return "三四线"
    return "未知"


def _extract_price_range(text: str) -> Optional[str]:
    """Extract price range like '50-200元' or '¥120' from text."""
    # Pattern: "50-200元" or "人均80-150" or "门票120元"
    m = re.search(r"(?:¥|￥)?(\d+)\s*[-~到至]\s*(\d+)\s*(?:元|块|人民币)?", text)
    if m:
        return f"¥{m.group(1)}-{m.group(2)}"
    m = re.search(r"(?:¥|￥)(\d+)\s*(?:元)?", text)
    if m:
        return f"¥{m.group(1)}"
    return None
