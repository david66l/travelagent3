"""Unit tests for JSON extraction from LLM output."""

from core.json_extract import extract_json_text, loads_json


def test_extract_from_markdown_fence():
    raw = '说明如下：\n```json\n{"intent": "chitchat", "confidence": 0.8}\n```'
    assert loads_json(raw)["intent"] == "chitchat"


def test_extract_embedded_object():
    raw = '结果：{"destination": "成都", "travel_days": 4} 完毕'
    data = loads_json(raw)
    assert data["destination"] == "成都"
    assert data["travel_days"] == 4


def test_extract_empty_returns_empty_object_string():
    assert extract_json_text("") == "{}"
