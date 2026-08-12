"""Disambiguation engine for ambiguous travel demand slots."""

from __future__ import annotations

import logging
from typing import Any, Optional

from models.travel_slots import TravelSlots

logger = logging.getLogger(__name__)


class DisambiguationEngine:
    """Detect ambiguity in slots and generate clarification questions."""

    # MVP fallback candidates for vague destination words.
    VAGUE_DESTINATION_MAP: dict[str, list[dict[str, str]]] = {
        "南方": [
            {"value": "厦门", "reason": "6月淡季机票便宜，海边休闲"},
            {"value": "成都", "reason": "美食多，节奏慢，适合吃喝逛"},
            {"value": "三亚", "reason": "海边度假，亲子友好"},
        ],
        "北方": [
            {"value": "北京", "reason": "历史文化浓厚，景点集中"},
            {"value": "西安", "reason": "古都美食，性价比高"},
            {"value": "青岛", "reason": "海滨城市，啤酒海鲜"},
        ],
        "海边": [
            {"value": "三亚", "reason": "国内热带海滨首选"},
            {"value": "厦门", "reason": "文艺小资，海边步道"},
            {"value": "青岛", "reason": "欧式建筑+海滨，避暑佳选"},
        ],
        "周边游": [
            {"value": "杭州", "reason": "西湖+周边古镇，交通便利"},
            {"value": "苏州", "reason": "园林水乡，上海周边首选"},
            {"value": "南京", "reason": "历史名城，高铁1小时可达"},
        ],
        "好玩的地方": [
            {"value": "成都", "reason": "吃喝玩乐综合体验好"},
            {"value": "重庆", "reason": "山城夜景+火锅，出片率高"},
            {"value": "长沙", "reason": "美食+夜生活，年轻人热门"},
        ],
    }

    @classmethod
    def analyze(
        cls,
        slots: TravelSlots,
        raw_input: str,
        candidates_from_llm: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Return disambiguation result: {has_ambiguity, field, candidates, question}."""
        # Destination ambiguity
        dest_result = cls._check_destination(slots, raw_input, candidates_from_llm)
        if dest_result["has_ambiguity"]:
            return dest_result

        # Budget ambiguity
        budget_result = cls._check_budget(slots, raw_input)
        if budget_result["has_ambiguity"]:
            return budget_result

        # Days ambiguity
        days_result = cls._check_days(slots, raw_input)
        if days_result["has_ambiguity"]:
            return days_result

        # People ambiguity
        people_result = cls._check_people(slots, raw_input)
        if people_result["has_ambiguity"]:
            return people_result

        return {"has_ambiguity": False, "field": None, "candidates": [], "question": ""}

    @classmethod
    def _check_destination(
        cls,
        slots: TravelSlots,
        raw_input: str,
        candidates_from_llm: Optional[list[dict[str, Any]]],
    ) -> dict[str, Any]:
        vague_words = set(cls.VAGUE_DESTINATION_MAP.keys())
        matched = [w for w in vague_words if w in raw_input]

        if not matched and not candidates_from_llm:
            return {
                "has_ambiguity": False,
                "field": "destination",
                "candidates": [],
                "question": "",
            }

        candidates = candidates_from_llm or []
        if matched:
            for word in matched:
                for c in cls.VAGUE_DESTINATION_MAP[word]:
                    if c["value"] not in {x.get("value") for x in candidates}:
                        candidates.append(c)

        return {
            "has_ambiguity": True,
            "field": "destination",
            "candidates": candidates[:5],
            "question": "您说的目的地比较宽泛，以下几个城市您更倾向哪个？",
        }

    @classmethod
    def _check_budget(cls, slots: TravelSlots, raw_input: str) -> dict[str, Any]:
        vague_budget = {"便宜", "性价比高", "不贵", "省钱", "经济"}
        if slots.total_budget is None and slots.budget_per_person is None:
            if any(w in raw_input for w in vague_budget):
                return {
                    "has_ambiguity": True,
                    "field": "budget",
                    "candidates": [],
                    "question": "您方便说一下大致预算范围吗？比如人均 2000-3000 元？",
                }
        return {"has_ambiguity": False, "field": "budget", "candidates": [], "question": ""}

    @classmethod
    def _check_days(cls, slots: TravelSlots, raw_input: str) -> dict[str, Any]:
        vague_days = {"几天", "待几天", "玩几天", "过两天", "过几天"}
        if slots.travel_days is None and any(w in raw_input for w in vague_days):
            return {
                "has_ambiguity": True,
                "field": "travel_days",
                "candidates": [],
                "question": "您计划出行多少天呢？",
            }
        return {"has_ambiguity": False, "field": "travel_days", "candidates": [], "question": ""}

    @classmethod
    def _check_people(cls, slots: TravelSlots, raw_input: str) -> dict[str, Any]:
        if slots.travelers_count is None and "几个人" in raw_input:
            return {
                "has_ambiguity": True,
                "field": "travelers",
                "candidates": [],
                "question": "一共几位出行呢？",
            }
        return {"has_ambiguity": False, "field": "travelers", "candidates": [], "question": ""}
