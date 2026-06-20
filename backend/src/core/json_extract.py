"""Extract JSON payloads from LLM text (markdown fences, prose wrappers, etc.)."""

from __future__ import annotations

import json
import re


def extract_json_text(content: str) -> str:
    """Return the most likely JSON object/array substring from *content*."""
    text = (content or "").strip()
    if not text:
        return "{}"

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    if text.startswith("{") or text.startswith("["):
        return text

    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start != -1 and obj_end > obj_start:
        return text[obj_start : obj_end + 1]

    arr_start = text.find("[")
    arr_end = text.rfind("]")
    if arr_start != -1 and arr_end > arr_start:
        return text[arr_start : arr_end + 1]

    return text


def loads_json(content: str) -> dict | list:
    """Parse JSON from raw LLM output, stripping common wrappers first."""
    return json.loads(extract_json_text(content))
