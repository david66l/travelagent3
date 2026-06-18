"""Output compliance filtering (PRD §4.9.2)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "sensitive_locations.json"

_SENSITIVE_CONTENT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(色情|裸体|赌博)\b",
        r"\b(恐怖组织|恐怖袭击)\b",
    )
)

_RISK_HINTS: dict[str, str] = {
    "高原": "注意高反/体力要求",
    "登山": "注意高反/体力要求",
    "潜水": "注意水上安全，请在专业指导下进行",
    "漂流": "注意水上安全",
    "夜爬": "注意夜间安全",
    "边境": "提前了解当地政策",
    "无人区": "请在专业指导下进行",
}


def _load_sensitive_locations() -> list[str]:
    if not _DATA_PATH.exists():
        return []
    try:
        payload = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
        return [str(x) for x in payload.get("locations", [])]
    except (json.JSONDecodeError, OSError):
        return []


def filter_sensitive_locations(text: str) -> str:
    """Replace sensitive POI/region mentions."""
    if not text:
        return text
    result = text
    for location in _load_sensitive_locations():
        if location and location in result:
            result = result.replace(location, "该地点暂不适合推荐")
    return result


def filter_sensitive_content(text: str) -> str:
    """Remove obvious policy-violating phrases."""
    if not text:
        return text
    result = text
    for pattern in _SENSITIVE_CONTENT_PATTERNS:
        result = pattern.sub("[内容已过滤]", result)
    return result


def append_risk_hints(text: str) -> str:
    """Add travel risk reminders when keywords appear."""
    if not text:
        return text
    hints: list[str] = []
    for keyword, hint in _RISK_HINTS.items():
        if keyword in text and hint not in hints:
            hints.append(hint)
    if not hints:
        return text
    footer = "；".join(hints)
    return f"{text}\n\n【安全提示】{footer}"


def sanitize_assistant_output(text: str) -> str:
    """Full output pipeline for assistant-visible text."""
    cleaned = filter_sensitive_content(filter_sensitive_locations(text))
    return append_risk_hints(cleaned)


def sanitize_itinerary_payload(payload: Any) -> Any:
    """Walk string fields in itinerary JSON-like structures."""
    if isinstance(payload, str):
        return sanitize_assistant_output(payload)
    if isinstance(payload, list):
        return [sanitize_itinerary_payload(item) for item in payload]
    if isinstance(payload, dict):
        return {k: sanitize_itinerary_payload(v) for k, v in payload.items()}
    return payload
