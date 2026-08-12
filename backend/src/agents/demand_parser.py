"""Demand Parser Agent — turn raw user input into structured TravelSlots.

The agent uses a lightweight LLM (task_type="intent" → 7B/small model) to directly
output a ``SlotParseOutput``. A deterministic fallback parser takes over when the
LLM call fails or returns invalid JSON. After parsing, a rule engine verifies
required slots and the disambiguation engine detects vague inputs.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from agents.disambiguation import DisambiguationEngine
from core.llm_client import llm
from models.travel_slots import SlotParseOutput, TravelSlots
from schemas import UserProfile

logger = logging.getLogger(__name__)


_DEMAND_PARSER_PROMPT = """你是旅行规划助手的槽位解析专家。分析用户输入，判断意图并提取结构化旅行需求。

## 意图类型（intent，必填）
- generate_itinerary — 想规划/生成行程（如"去成都玩4天"）
- modify_itinerary — 修改已有行程（如"第三天换个景点"）
- update_preferences — 只更新偏好，不涉及新行程（如"我喜欢吃辣"）
- query_info — 询问信息（如"成都有什么好吃的"）
- confirm_itinerary — 确认当前行程（如"确认行程"）
- view_history — 查看历史行程
- chitchat — 闲聊/打招呼（如"你好"）

## 规划必填槽位（8 项，缺一不可，缺则 missing_slots 列出并在 clarifying_question 中追问）
1. origin — 出发城市
2. destination — 目的地城市
3. travel_dates — 出行日期原文（用户说了才填，如"明天"、"5月1日"；未提到必须为 null）
4. travel_days — 旅行天数（整数）
5. travelers_count — 出行人数（整数，未提到必须为 null，禁止默认为 1）
6. has_elderly — 是否有老人同行：true / false（必须明确，未提到则 null 并追问）
7. has_children — 是否有儿童：true / false（必须明确，未提到则 null 并追问）
8. total_budget — 总预算（元，整数）

## 补充信息（有则提取，缺失不 blocking）
- travel_companion（出行类型，可选）、interests, food_prefs, pace 等

## 输出格式（只输出 JSON，不要 markdown，不要解释）
{
    "intent": "generate_itinerary",
    "confidence": 0.0,
    "sentiment": "positive | neutral | negative | urgent",
    "slots": {
        "origin": null,
        "destination": null,
        "travel_dates": null,
        "travel_days": null,
        "travelers_count": null,
        "has_children": null,
        "total_budget": null,
        "interests": [],
        "food_prefs": [],
        "pace": null,
        "travel_companion": null,
        "has_elderly": null
    },
    "missing_slots": ["destination", "travel_days"],
    "clarifying_question": "您需要把以下信息告诉我：出发城市、出行日期、玩几天、几个人、有没有带老人、有没有带小孩、预算多少。"
}

## 提取规则
1. 先判断 intent；query_info / chitchat / update_preferences / view_history 时不填 missing_slots
2. generate_itinerary / modify_itinerary：缺哪项就把字段名放进 missing_slots；clarifying_question 必须以「您需要把以下信息告诉我：」或「您需要把这个信息告诉我：」开头，一次性列出全部仍缺的必填项（用中文标签，顿号分隔，顺序：出发城市→目的地→出行日期→玩几天→几个人→有没有带老人→有没有带小孩→预算多少）
3. "女朋友/男朋友/情侣/夫妻" → travelers_count=2；可同时填 travel_companion=couple（可选，非必填）
4. "带爸妈/父母/老人" → has_elderly=true；若明确人数则填 travelers_count
5. "没有老人/不带父母/父母不去" → has_elderly=false
6. "带小孩/孩子/亲子" → has_children=true；"没有小孩/不带娃" → has_children=false
7. 未提及是否带老人 → has_elderly=null（不要猜 false）；未提及是否带小孩 → has_children=null（不要猜 false）
8. "人均2000，4人" → total_budget=8000
9. 当前用户画像里已有的字段视为已填，不要重复追问
10. 不要编造未提及的信息；travel_companion 为可选字段，缺了不要追问

## 多轮信息收集（重要）
- 「当前用户画像」里已有 destination / travel_days 等时，用户本轮只补充一个字段（如「5000块」「明天从济南出发」「2个人」「不带娃」），intent 仍为 generate_itinerary，只更新对应 slots
- 不要把补槽误判为 update_preferences 或 chitchat
- missing_slots 与 clarifying_question 只针对「画像 + 本轮」仍缺的必填项，不要重复问已有字段
- 用户只说金额（如「5000」「5000块」）→ total_budget=5000，intent=generate_itinerary
"""


class DemandParserAgent:
    """Parse user demand into structured TravelSlots."""

    # All required before planning — must match is_profile_ready()
    REQUIRED_SLOTS = (
        "origin",
        "destination",
        "travel_dates",
        "travel_days",
        "travelers_count",
        "has_elderly",
        "has_children",
        "total_budget",
    )

    SLOT_TO_PROFILE = {
        "origin": "origin",
        "destination": "destination",
        "travel_dates": "travel_dates",
        "travel_days": "travel_days",
        "travelers_count": "travelers_count",
        "travel_companion": "travelers_type",
        "has_elderly": "has_elderly",
        "has_children": "has_children",
        "total_budget": "budget_range",
    }

    # Map slot names to human-readable Chinese labels
    SLOT_LABELS = {
        "origin": "出发城市",
        "destination": "目的地",
        "travel_dates": "出行日期",
        "travel_days": "玩几天",
        "travelers_count": "几个人",
        "has_elderly": "有没有带老人",
        "has_children": "有没有带小孩",
        "total_budget": "预算多少",
    }

    async def parse(
        self,
        user_input: str,
        messages: list[dict[str, str]],
        user_profile: Optional[UserProfile] = None,
        known_profile: Optional[dict] = None,
    ) -> SlotParseOutput:
        """Parse input into SlotParseOutput using small LLM or deterministic fallback."""
        profile_str = user_profile.model_dump_json(exclude_none=True) if user_profile else "{}"
        prompt_messages = [
            {"role": "system", "content": _DEMAND_PARSER_PROMPT},
            {"role": "system", "content": f"当前用户画像：{profile_str}"},
            *messages[-10:],
            {"role": "user", "content": user_input},
        ]

        try:
            parsed = await llm.structured_call(
                messages=prompt_messages,
                response_model=SlotParseOutput,
                temperature=0.3,
                task_type="intent",
            )
            # Rule-based sentiment override when LLM returns neutral
            if parsed.sentiment == "neutral":
                parsed.sentiment = self._detect_sentiment(user_input)
        except Exception as exc:
            logger.warning("LLM demand parsing failed; using deterministic fallback: %s", exc)
            parsed = self._fallback_parse(user_input)

        parsed = self._normalize_intent_during_gathering(parsed, known_profile, user_input)
        parsed = self._enrich_slots_from_text(parsed, user_input)

        if known_profile:
            self._merge_known_profile_into_slots(parsed, known_profile)

        # Rule-based validation for required slots
        parsed = self._apply_required_rules(parsed, user_input)

        # Resolve natural language dates to concrete dates
        if parsed.slots.travel_dates:
            resolved = self._resolve_date(parsed.slots.travel_dates)
            if resolved:
                parsed.slots.travel_dates = resolved

        # Disambiguation for vague inputs
        disambiguation = DisambiguationEngine.analyze(parsed.slots, user_input)
        parsed.disambiguation = disambiguation
        if disambiguation.get("has_ambiguity") and disambiguation.get("question"):
            parsed.clarifying_question = disambiguation["question"]
            ambiguous_field = disambiguation.get("field")
            if (
                ambiguous_field in self.REQUIRED_SLOTS
                and ambiguous_field not in parsed.missing_slots
            ):
                parsed.missing_slots.append(ambiguous_field)

        return parsed

    @classmethod
    def _normalize_intent_during_gathering(
        cls,
        parsed: SlotParseOutput,
        known_profile: Optional[dict],
        user_input: str,
    ) -> SlotParseOutput:
        """Keep slot-filling turns on the planning path while profile is incomplete."""
        if not known_profile or cls.profile_is_complete(known_profile):
            return parsed
        if parsed.intent not in ("chitchat", "query_info", "update_preferences", "view_history"):
            return parsed
        if parsed.intent == "chitchat" and cls._is_greeting_only(user_input):
            return parsed
        if cls._looks_like_slot_fill(user_input, parsed):
            parsed.intent = "generate_itinerary"
        return parsed

    @staticmethod
    def _is_greeting_only(text: str) -> bool:
        stripped = text.strip().lower()
        return stripped in (
            "你好",
            "嗨",
            "hello",
            "hi",
            "在吗",
            "在不在",
            "早上好",
            "晚上好",
        )

    @classmethod
    def _looks_like_slot_fill(cls, user_input: str, parsed: SlotParseOutput) -> bool:
        """True when the utterance likely adds a travel slot, not a new topic."""
        text = user_input.strip()
        slot_data = parsed.slots.model_dump(exclude_none=True, exclude_defaults=True)
        if slot_data:
            return True
        if re.fullmatch(r"\d{3,6}\s*[元块]?", text):
            return True
        if cls._extract_budget(text) is not None:
            return True
        if cls._extract_origin(text) is not None:
            return True
        if cls._extract_travel_dates(text) is not None:
            return True
        if cls._extract_travel_days(text) is not None:
            return True
        if re.search(r"^\d+\s*个?人$", text):
            return True
        if any(w in text for w in ("没有小孩", "不带娃", "带小孩", "带孩子", "有小孩")):
            return True
        if any(w in text for w in ("没有老人", "不带老人", "带老人", "带爸妈", "带父母", "有老人")):
            return True
        return False

    @classmethod
    def _enrich_slots_from_text(cls, parsed: SlotParseOutput, user_input: str) -> SlotParseOutput:
        """Fill empty slots from deterministic extractors when the small LLM misses them."""
        slots = parsed.slots
        if cls._slot_is_empty(slots.origin, field="origin"):
            origin = cls._extract_origin(user_input)
            if origin:
                slots.origin = origin
        if cls._slot_is_empty(slots.travel_dates, field="travel_dates"):
            dates = cls._extract_travel_dates(user_input)
            if dates:
                slots.travel_dates = dates
        if cls._slot_is_empty(slots.total_budget, field="total_budget"):
            budget = cls._extract_budget(user_input)
            if budget is not None:
                slots.total_budget = budget
        if cls._slot_is_empty(slots.travel_days, field="travel_days"):
            days = cls._extract_travel_days(user_input)
            if days is not None:
                slots.travel_days = days
        if cls._slot_is_empty(slots.travelers_count, field="travelers_count"):
            count = cls._extract_travelers_count(user_input)
            if count is not None:
                slots.travelers_count = count
        if cls._slot_is_empty(slots.travel_companion, field="travel_companion"):
            companion = cls._extract_companion(user_input)
            if companion is not None:
                slots.travel_companion = companion
        if slots.has_elderly is None:
            elderly = cls._extract_has_elderly(user_input)
            if elderly is not None:
                slots.has_elderly = elderly
        if slots.has_children is None:
            no_child_keywords = ("没有小孩", "不带小孩", "没带孩子", "没有孩子", "不带娃")
            child_keywords = ("带小孩", "带孩子", "有小孩")
            if any(word in user_input for word in no_child_keywords):
                slots.has_children = False
            elif any(word in user_input for word in child_keywords):
                slots.has_children = True
        return parsed

    @classmethod
    def _merge_known_profile_into_slots(cls, parsed: SlotParseOutput, flat_profile: dict) -> None:
        """Fill empty slots from accumulated session profile (multi-turn gathering)."""
        for slot_key, profile_key in cls.SLOT_TO_PROFILE.items():
            if profile_key not in flat_profile:
                continue
            value = flat_profile[profile_key]
            current = getattr(parsed.slots, slot_key)
            if not cls._slot_is_empty(current, field=slot_key) or cls._slot_is_empty(
                value, field=slot_key
            ):
                continue
            setattr(parsed.slots, slot_key, value)

    @classmethod
    def missing_from_profile(cls, flat_profile: dict) -> list[str]:
        """Return REQUIRED slot keys still absent from a flattened profile dict."""
        missing: list[str] = []
        for slot_key in cls.REQUIRED_SLOTS:
            profile_key = cls.SLOT_TO_PROFILE.get(slot_key, slot_key)
            if profile_key not in flat_profile:
                missing.append(slot_key)
                continue
            if cls._slot_is_empty(flat_profile[profile_key], field=slot_key):
                missing.append(slot_key)
        return missing

    @classmethod
    def profile_is_complete(cls, flat_profile: dict) -> bool:
        """True when every REQUIRED slot is present in a flattened profile dict."""
        return not cls.missing_from_profile(flat_profile)

    @classmethod
    def build_clarification_questions(
        cls,
        flat_profile: dict,
        *,
        existing: list[str] | None = None,
    ) -> list[str]:
        """Build user-facing clarification questions from merged profile gaps."""
        missing = cls.missing_from_profile(flat_profile)
        if missing:
            return [cls._build_clarifying_question(missing, flat_profile.get("destination"))]
        return [q for q in (existing or []) if q]

    @classmethod
    def _apply_required_rules(
        cls, parsed: SlotParseOutput, user_input: str = ""
    ) -> SlotParseOutput:
        """Rebuild missing_slots from authoritative rules; ignore LLM extras."""
        if parsed.intent in ("chitchat", "query_info", "update_preferences", "view_history"):
            parsed.missing_slots = []
            parsed.clarifying_question = None
            return parsed

        missing = [
            field
            for field in cls.REQUIRED_SLOTS
            if cls._slot_is_empty(getattr(parsed.slots, field), field=field)
        ]
        parsed.missing_slots = missing

        if parsed.missing_slots:
            parsed.clarifying_question = cls._build_clarifying_question(
                parsed.missing_slots, parsed.slots.destination
            )
        else:
            parsed.clarifying_question = None
        return parsed

    @staticmethod
    def _slot_is_empty(value, *, field: str = "") -> bool:
        if field in ("has_children", "has_elderly"):
            return value is None
        return value is None or value == "" or value == []

    @classmethod
    def _build_clarifying_question(
        cls, missing: list[str], destination: Optional[str] = None
    ) -> str:
        """Ask for all missing required slots in one message."""
        if not missing:
            return ""

        ordered = [field for field in cls.REQUIRED_SLOTS if field in missing]
        labels = [cls.SLOT_LABELS.get(field, field) for field in ordered]
        if len(labels) == 1:
            return f"您需要把这个信息告诉我：{labels[0]}。"
        return f"您需要把以下信息告诉我：{'、'.join(labels)}。"

    @classmethod
    def _fallback_parse(cls, user_input: str) -> SlotParseOutput:
        """Minimal deterministic fallback when the small LLM is unavailable."""
        no_child_keywords = ("没有小孩", "不带小孩", "没带孩子", "没有孩子", "不带娃")
        child_keywords = ("带小孩", "带孩子", "有小孩")
        if any(word in user_input for word in no_child_keywords):
            has_children: bool | None = False
        elif any(word in user_input for word in child_keywords):
            has_children = True
        else:
            has_children = None

        has_elderly = cls._extract_has_elderly(user_input)

        slots = TravelSlots(
            destination=cls._extract_destination(user_input),
            travel_days=cls._extract_travel_days(user_input),
            travel_dates=cls._extract_travel_dates(user_input),
            travelers_count=cls._extract_travelers_count(user_input),
            total_budget=cls._extract_budget(user_input),
            has_children=has_children,
            has_elderly=has_elderly,
            origin=cls._extract_origin(user_input),
        )
        parsed = SlotParseOutput(
            intent=cls._classify_intent(user_input),
            confidence=0.45,
            sentiment=cls._detect_sentiment(user_input),
            slots=slots,
        )
        parsed = cls._apply_required_rules(parsed, user_input)
        if not any(
            getattr(slots, field) is not None
            for field in ("destination", "travel_days", "total_budget", "origin", "travel_dates")
        ):
            parsed.clarifying_question = "抱歉，我这边暂时没听清楚，能再说一下您的出行计划吗？"
        return parsed

    # ----------------------------------------------------------------------- #
    # Minimal fallback extractors (LLM-down only)
    # ----------------------------------------------------------------------- #

    @staticmethod
    def _extract_destination(text: str) -> Optional[str]:
        # Match "去/到XX玩/旅游/游/几天" — non-greedy, captures nearest city.
        pattern = r"(?:去|到|来|飞|前往|想去)\s*([\u4e00-\u9fa5]{2,8}?)(?:玩|旅游|旅行|游|逛|出差|待|住|的|\d+天)"
        match = re.search(pattern, text)
        if match:
            city = match.group(1)
            if city not in ("出发", "回来", "回去"):
                return city
        # First-occurrence fallback for well-known cities.
        cities = [
            "北京",
            "上海",
            "广州",
            "深圳",
            "成都",
            "杭州",
            "西安",
            "重庆",
            "苏州",
            "南京",
            "厦门",
            "青岛",
            "大理",
            "丽江",
            "三亚",
            "长沙",
            "武汉",
            "昆明",
            "桂林",
            "拉萨",
        ]
        best, best_pos = None, len(text)
        for city in cities:
            pos = text.find(city)
            if pos != -1 and pos < best_pos:
                best, best_pos = city, pos
        return best

    @staticmethod
    def _extract_origin(text: str) -> Optional[str]:
        """Extract departure city from '从北京出发' '济南出发' '深圳飞'."""
        non_city = {"明天", "后天", "今天", "大后天", "出发", "预算"}
        # Pattern 1: "从/由 XX 出发/走/去/来/飞"
        pattern = r"(?:从|由)\s*([\u4e00-\u9fa5]{2,8}?)(?:出发|走|去|来|飞)"
        match = re.search(pattern, text)
        if match:
            city = match.group(1)
            if city not in non_city:
                return city
        # Pattern 2: "XX出发" anywhere in the utterance
        match = re.search(r"([\u4e00-\u9fa5]{2,8})出发", text)
        if match:
            city = match.group(1)
            if city not in non_city:
                return city
        # Pattern 3: "XX飞" at end
        match = re.search(r"([\u4e00-\u9fa5]{2,8})飞$", text)
        if match:
            city = match.group(1)
            if city not in non_city:
                return city
        return None

    @staticmethod
    def _extract_travel_days(text: str) -> Optional[int]:
        patterns = [
            r"(\d+)\s*[天日]",
            r"玩\s*(\d+)\s*[天日]",
            r"(\d+)\s*day",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                days = int(match.group(1))
                if 1 <= days <= 30:
                    return days
        return None

    @staticmethod
    def _extract_travel_dates(text: str) -> Optional[str]:
        match = re.search(
            r"(今天|明天|后天|大后天|下?周[一二三四五六日]|\d{1,2}月\d{1,2}[日号]|\d{4}-\d{2}-\d{2})",
            text,
        )
        return match.group(1) if match else None

    @staticmethod
    def _extract_companion(text: str) -> Optional[str]:
        """Extract companion type only when the user explicitly mentions it."""
        companion_map = {
            "独自": "alone",
            "一个人": "alone",
            "自己一个人": "alone",
            "情侣": "couple",
            "夫妻": "couple",
            "两口子": "couple",
            "女朋友": "couple",
            "男朋友": "couple",
            "女友": "couple",
            "男友": "couple",
            "亲子": "family",
            "带孩子": "family",
            "带孩子去": "family",
            "家庭": "family",
            "一家人": "family",
            "朋友": "friends",
            "闺蜜": "friends",
            "兄弟": "friends",
            "同学": "friends",
            "父母": "parents",
            "老人": "parents",
            "爸妈": "parents",
            "带爸妈": "parents",
            "同事": "colleagues",
        }
        for keyword, value in companion_map.items():
            if keyword in text:
                return value
        return None

    @staticmethod
    def _extract_has_elderly(text: str) -> Optional[bool]:
        no_elderly = ("没有老人", "不带老人", "没带老人", "父母不去", "不带父母", "不带爸妈")
        elderly = ("带老人", "有老人", "老人同行", "带爸妈", "带父母", "陪父母", "腿脚不便")
        if any(word in text for word in no_elderly):
            return False
        if any(word in text for word in elderly):
            return True
        return None

    @staticmethod
    def _extract_travelers_count(text: str) -> Optional[int]:
        # Explicit numbers
        match = re.search(r"(\d+)\s*个?人", text)
        if match:
            return int(match.group(1))
        # Common phrases
        if any(w in text for w in ("一个人", "独自", " solo ")):
            return 1
        if any(
            w in text
            for w in ("两个人", "情侣", "夫妻", "我们俩", "女朋友", "男朋友", "女友", "男友")
        ):
            return 2
        if any(w in text for w in ("带爸妈", "带父母", "一家三口")):
            return 3
        if any(w in text for w in ("一家四口", "带两个孩子")):
            return 4
        return None

    @staticmethod
    def _extract_budget(text: str) -> Optional[float]:
        match = re.search(r"预算\s*(\d+)(?:\s*[元块])?", text)
        if match:
            return float(match.group(1))
        match = re.search(r"(\d{3,6})\s*[元块]", text)
        if match:
            return float(match.group(1))
        if re.fullmatch(r"\d{3,6}", text.strip()):
            return float(text.strip())
        return None

    @staticmethod
    def _classify_intent(text: str) -> str:
        """Minimal intent for LLM-down fallback only."""
        stripped = text.strip().lower()
        if stripped in ("你好", "嗨", "hello", "hi", "在吗", "在不在"):
            return "chitchat"
        if re.search(r"(?:去|到|玩|游|旅游|旅行|行程|\d+\s*[天日])", text):
            return "generate_itinerary"
        if DemandParserAgent._extract_budget(text) is not None:
            return "generate_itinerary"
        return "generate_itinerary"

    @staticmethod
    def _detect_sentiment(text: str) -> str:
        urgent = {"急", "马上", "立刻", "今晚", "明天必须", "赶紧", "救命"}
        negative = {"差", "烂", "坑", "失望", "后悔", "不满意", "生气", "烦", "投诉", "退"}
        positive = {"期待", "开心", "兴奋", "满意", "喜欢", "完美", "棒", "赞"}

        if any(word in text for word in urgent):
            return "urgent"
        if any(word in text for word in negative):
            return "negative"
        if any(word in text for word in positive):
            return "positive"
        return "neutral"

    @staticmethod
    def _resolve_date(date_str: str) -> Optional[str]:
        """Resolve natural language dates to concrete dates (MVP)."""
        from datetime import datetime, timedelta

        today = datetime.now()
        weekday_map = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6}

        if date_str == "今天":
            return today.strftime("%Y-%m-%d")
        if date_str == "明天":
            return (today + timedelta(days=1)).strftime("%Y-%m-%d")
        if date_str == "后天":
            return (today + timedelta(days=2)).strftime("%Y-%m-%d")
        if date_str.startswith("下周"):
            weekday_char = date_str[-1]
            target_weekday = weekday_map.get(weekday_char)
            if target_weekday is not None:
                # Move to next Monday then add target weekday
                days_until_next_mon = (7 - today.weekday()) % 7 or 7
                next_mon = today + timedelta(days=days_until_next_mon)
                target = next_mon + timedelta(days=target_weekday)
                return target.strftime("%Y-%m-%d")

        # Normalize common absolute Chinese dates so weather APIs, weekday
        # closure constraints and itinerary cards all receive ISO dates. Keep a
        # possible end date using the same ``|`` separator accepted downstream.
        absolute = re.findall(
            r"(?:(\d{4})\s*[年/-])?\s*(\d{1,2})\s*[月/-]\s*(\d{1,2})\s*日?",
            date_str,
        )
        if absolute:
            resolved: list[str] = []
            inherited_year: int | None = None
            for year_text, month_text, day_text in absolute[:2]:
                year = int(year_text) if year_text else inherited_year or today.year
                try:
                    value = datetime(year, int(month_text), int(day_text))
                except ValueError:
                    return None
                if not year_text and inherited_year is None and value.date() < today.date():
                    value = value.replace(year=year + 1)
                    year = value.year
                inherited_year = year
                resolved.append(value.strftime("%Y-%m-%d"))
            return "|".join(resolved)
        return date_str
