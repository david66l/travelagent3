"""Canonical city-name aliases shared by retrieval and provider tools."""

from __future__ import annotations

_CITY_ALIASES = {
    "shanghai": "上海",
    "beijing": "北京",
    "guangzhou": "广州",
    "chengdu": "成都",
    "hangzhou": "杭州",
    "xian": "西安",
    "xi'an": "西安",
    "chongqing": "重庆",
    "shenzhen": "深圳",
    "nanjing": "南京",
    "suzhou": "苏州",
    "wuhan": "武汉",
    "xiamen": "厦门",
}


def canonical_city_name(city: str) -> str:
    """Map stable English aliases and optional Chinese 市 suffixes."""
    stripped = str(city or "").strip()
    if not stripped:
        return ""
    return _CITY_ALIASES.get(stripped.casefold(), stripped.removesuffix("市"))
