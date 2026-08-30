"""Demand Parser Agent — turn raw user input into structured TravelSlots.

The agent uses a lightweight LLM (task_type="intent" → 7B/small model) to directly
output a ``SlotParseOutput``. A deterministic fallback parser takes over when the
LLM call fails or returns invalid JSON. After parsing, a rule engine verifies
required slots and the disambiguation engine detects vague inputs.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field

from agents.disambiguation import DisambiguationEngine
from core.llm_client import llm
from models.travel_slots import RevisionParseOutput, SlotParseOutput, TravelSlots
from schemas import UserProfile

logger = logging.getLogger(__name__)


class IntercityTransportAudit(BaseModel):
    """Focused semantic check for a conditional origin requirement."""

    explicit_request: bool = False
    modes: list[Literal["flight", "train", "bus", "ferry"]] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


_INTERCITY_TRANSPORT_AUDIT_PROMPT = """你只判断用户是否明确要求查询或乘坐城际班次，并输出 JSON。
城际班次包括飞机、普通火车、高铁、长途汽车、轮渡；市内地铁、公交、出租车不算。
若明确要求，explicit_request=true，并把方式映射为 flight/train/bus/ferry；
若只是普通地说去某地旅行，explicit_request=false。不得根据目的地自行猜测。"""


_DEMAND_PARSER_PROMPT = """你是旅行规划助手的槽位解析专家。分析用户输入，判断意图并提取结构化旅行需求。

## 意图类型（intent，必填）
- generate_itinerary — 想规划/生成行程（如"去成都玩4天"）
- modify_itinerary — 修改已有行程（如"第三天换个景点"）
- update_preferences — 只更新偏好，不涉及新行程（如"我喜欢吃辣"）
- query_info — 询问信息（如"成都有什么好吃的"）
- confirm_itinerary — 确认当前行程（如"确认行程"）
- view_history — 查看历史行程
- chitchat — 闲聊/打招呼（如"你好"）

## 用户必填槽位（只追问用户拥有、且当前任务确实必需的信息）
1. destination — 目的地城市
2. travel_days — 需要规划的旅行天数

origin、travel_dates、travelers_count、has_elderly、has_children、total_budget 都应尽量提取，
但普通行程缺少它们时不要阻塞。若用户明确要求查询航班/火车，后续 Agent 会按需追问
origin；演出时间地点、天气和营业时间属于工具可发现事实，禁止让用户代替系统查询。

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
        "travel_companion": null,
        "has_elderly": null,
        "has_children": null,
        "has_pregnant": null,
        "has_wheelchair": null,
        "total_budget": null,
        "interests": [],
        "food_prefs": [],
        "food_taboos": [],
        "must_visit": [],
        "must_not_visit": [],
        "pace": null,
        "max_walk_minutes": null,
        "max_transit_minutes": null,
        "fatigue_preference": null,
        "avoid_crowds": null,
        "transport_preference": null,
        "intent_kind": "itinerary",
        "event_query": null,
        "transport_modes_requested": [],
        "information_needs": [],
        "current_info_queries": []
    },
    "missing_slots": ["destination", "travel_days"],
    "clarifying_question": "您需要把以下信息告诉我：目的地、玩几天。"
}

## 提取规则
1. 先判断 intent；query_info / chitchat / update_preferences / view_history 时不填 missing_slots
2. generate_itinerary / modify_itinerary：只把 destination、travel_days 中缺失的字段放进 missing_slots；clarifying_question 一次性列出它们
3. "女朋友/男朋友/情侣/夫妻" → travelers_count=2；可同时填 travel_companion=couple（可选，非必填）
4. "带爸妈/父母/老人" → has_elderly=true；若明确人数则填 travelers_count
5. "没有老人/不带父母/父母不去" → has_elderly=false
6. "带小孩/孩子/亲子" → has_children=true；"没有小孩/不带娃" → has_children=false
7. 未提及是否带老人 → has_elderly=null（不要猜 false）；未提及是否带小孩 → has_children=null（不要猜 false）
8. "人均2000，4人" → total_budget=8000
9. 当前用户画像里已有的字段视为已填，不要重复追问
10. 不要编造未提及的信息；travel_companion 为可选字段，缺了不要追问
11. 识别工具需求靠语义而不是字面关键词：
    - 只有行程依赖具体演出、比赛、展览、节庆等具有确定日期/场地的公开活动时，intent_kind=event_trip，填写 event_query，information_needs 加 event
    - 景点是否开放/营业不是 event：使用 opening_hours；询问临时闭馆使用 closure；intent_kind 保持 itinerary，event_query=null
    - 花期、红叶、雪季等自然季节状态不是 event：使用 seasonal_activity；只有用户明确说的是某个节庆/展会时才使用 event
    - 用户明确要求查询航班、火车、汽车或轮渡班次时，填写 transport_modes_requested，information_needs 加 transport；枚举值只能是 flight/train/bus/ferry，高铁也写 train
    - “市内坐地铁/公交/出租车”等只是 transport_preference，不是城际班次查询：地铁/公交/公共交通=public，打车=taxi，步行=walk，租车自驾=rental_car，组合使用=mixed；不得填写 transport_modes_requested，也不得添加 information_needs=transport
    - 天气、营业/闭馆、餐厅营业、季节状态等时效事实，分别加入 information_needs，并把要核实的完整问题写入 current_info_queries
    - 用户需要寻找、推荐或安排餐厅时使用 restaurant；只有已经给出具体店名、仅核实该店营业时间时才使用 opening_hours
    - pace 只能是 relaxed/moderate/intensive；“轻松/不要太赶”=relaxed，“紧凑/多安排几个地方”=intensive
    - 只是表达交通偏好（如“市内尽量地铁”）不等于查询城际班次
12. event_query 和 current_info_queries 要保留实体和限定条件，不能只写“活动”或“查一下”
13. interests、food_prefs 等开放文本标签保留用户原语言和核心表达，不要擅自翻译成英文
14. “必须去/一定要去/必去”后的地点放入 must_visit；“不要去/不安排/排除”后的地点放入 must_not_visit，不能降级成普通 interests
15. “忌口/过敏/不能吃”提取到 food_taboos；饮食偏好仍放 food_prefs，两者不能混淆
16. 明确出现孕妇或轮椅使用者时，分别设置 has_pregnant=true、has_wheelchair=true；不得被画像中的默认 false 覆盖
17. 用户明确给出步行或通勤分钟上限时原样提取到 max_walk_minutes/max_transit_minutes，不得替换成系统默认值
18. “疲劳度低/不耐累”设置 fatigue_preference=low；“避开人群/不喜欢拥挤”设置 avoid_crowds=true

## 多轮信息收集（重要）
- 「当前用户画像」里已有 destination / travel_days 等时，用户本轮只补充一个字段（如「5000块」「明天从济南出发」「2个人」「不带娃」），intent 仍为 generate_itinerary，只更新对应 slots
- 不要把补槽误判为 update_preferences 或 chitchat
- missing_slots 与 clarifying_question 只针对「画像 + 本轮」仍缺的必填项，不要重复问已有字段
- 用户只说金额（如「5000」「5000块」）→ total_budget=5000，intent=generate_itinerary
"""


_REVISION_PARSER_PROMPT = """你是旅行 Agent 的修改意图解析器。结合当前目标与用户对草案的反馈，输出结构化修改操作。

你只负责理解用户语义，不负责直接改行程、搜索事实或判断约束是否可行。Controller 会对你的输出做字段白名单、类型和边界校验。

## intent
- revise_itinerary：用户给出了可执行的修改意见
- clarify_revision：反馈含糊或存在多个明显解释，必须追问
- accept_itinerary：用户实际是在接受当前行程
- start_new_trip：用户明确放弃当前目的地并开始另一次旅行

## operations
每项为 {"field": 字段, "operation": "set|add|remove|clear", "value": 值}。
允许字段：origin, destination, travel_days, start_date, end_date, budget_range,
must_visit, must_not_visit, mobility_constraints, max_transit_minutes, intent_kind,
event_query, transport_modes_requested, information_needs, current_info_queries,
interests, food_preferences, transport_preference, hotel_preference, pace,
travelers_type, travelers_count, has_children, has_elderly, avoid_pois。

规则：
1. “太赶/想轻松些”应转成 pace=relaxed；只有用户给出明确天数时才修改 travel_days。
2. “不要博物馆/删掉外滩”等负向要求，使用 must_not_visit 或 avoid_pois 的 add/remove，不要塞进普通文字备注。
3. 新增演唱会、比赛、展览等具有确定时间地点的活动时，设置 intent_kind=event_trip、event_query，并给 information_needs 添加 event。
4. 新增航班/火车等城际班次要求时，设置 transport_modes_requested，并给 information_needs 添加 transport。transport_modes_requested 的值只能是 flight/train/bus/ferry。
5. 需要最新营业时间、闭馆、餐厅、天气或季节性信息时，修改 information_needs 和 current_info_queries。
6. 不要根据常识补造数值、日期、地点；确实无法唯一理解时 needs_clarification=true，并给出一个具体问题。
7. affected_domains 只从 research, candidates, transport, schedule, budget, presentation 中选择。
8. 只输出 JSON，不要 markdown，不要解释。
9. 市内交通偏好使用 transport_preference=set，值只能是 public/taxi/walk/rental_car/mixed/any；“公共交通/地铁/公交”统一为 public。用户用“不用查/不是要查/不需要查”等任何否定表达排除航班、火车或车次查询时，禁止 add/set transport_modes_requested 或 information_needs=transport；若当前目标里已有，可用 remove/clear 删除。
10. 景点营业/闭馆和自然花期不是 event；分别使用 opening_hours/closure/seasonal_activity。event 只表示需要核实日期和场馆的具体公开活动。
11. 用户一句话可能同时包含多项修改。对“但、同时、另外、还要”等连接的每个可执行子要求都生成独立 operation；输出前检查明确出现的天数、金额、增删地点和偏好没有遗漏。
12. 例如“总预算控制在4500以内，但酒店品质别降低”必须同时输出 budget_range=set=4500 和 hotel_preference=set="preserve_quality" 两项，不能因为第二项没有给数值就省略。
13. pace 的值只能是 relaxed/moderate/intensive；“太赶/轻松些”=relaxed，“紧凑/多安排几个地方”=intensive，禁止输出 slow/fast。
14. 用户要求系统先查询活动日期和地点时，这是 research 操作，不是让用户澄清；仍应输出 event_trip、event_query 和 information_needs=event。
15. interests、food_preferences 等开放文本值保留用户原语言和核心表达，不要擅自翻译成英文。
16. 只有用户明确说要放弃当前行程，并给出另一趟旅行或新目的地时才使用 start_new_trip；“换个方案/重新安排/再优化”但没有说明改什么，属于 clarify_revision，必须询问不满意之处。

输出结构：
{
  "intent": "revise_itinerary",
  "confidence": 0.0,
  "operations": [],
  "affected_domains": [],
  "needs_clarification": false,
  "clarification_question": null
}
"""


class DemandParserAgent:
    """Parse user demand into structured TravelSlots."""

    # All required before planning — must match is_profile_ready()
    REQUIRED_SLOTS = ("destination", "travel_days")

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
        # Do not show schema defaults (for example has_wheelchair=false or a
        # 180-minute walking limit) as if they were remembered user facts. They
        # anchor the intent model and can overwrite explicit constraints in the
        # current utterance.
        profile_str = (
            user_profile.model_dump_json(exclude_none=True, exclude_defaults=True)
            if user_profile
            else "{}"
        )
        prompt_messages = [
            {"role": "system", "content": _DEMAND_PARSER_PROMPT},
            {
                "role": "system",
                "content": (
                    f"系统当前日期：{date.today().isoformat()}。"
                    "解析相对日期和缺少年份的未来旅行日期时必须以此为准；"
                    "用户没有提供或无法可靠推出的年份不得自行编造。"
                ),
            },
            {"role": "system", "content": f"当前用户画像：{profile_str}"},
            *messages[-10:],
            {"role": "user", "content": user_input},
        ]

        try:
            parsed = await llm.structured_call(
                messages=prompt_messages,
                response_model=SlotParseOutput,
                temperature=0.0,
                task_type="intent",
            )
            parsed.parse_source = "llm"
            # Rule-based sentiment override when LLM returns neutral
            if parsed.sentiment == "neutral":
                parsed.sentiment = self._detect_sentiment(user_input)
        except Exception as exc:
            logger.warning("LLM demand parsing failed; using deterministic fallback: %s", exc)
            parsed = self._fallback_parse(user_input)
            parsed.parse_source = "deterministic_fallback"

        parsed = self._normalize_intent_during_gathering(parsed, known_profile, user_input)
        parsed = self._enrich_slots_from_text(parsed, user_input)
        parsed = await self._audit_conditional_transport_requirement(parsed, user_input)

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

        parsed.token_usage = int(llm.last_token_usage or 0)
        return parsed

    @staticmethod
    async def _audit_conditional_transport_requirement(
        parsed: SlotParseOutput,
        user_input: str,
    ) -> SlotParseOutput:
        """Use a focused model check when the broad parser may have missed a mode.

        This deliberately stays model-based: conditional intent is not inferred
        from a keyword list.  The second call only runs on an already-incomplete
        gathering turn, so ordinary complete requests do not pay a second-call
        latency and token tax.
        """
        if parsed.intent not in {"generate_itinerary", "modify_itinerary"}:
            return parsed
        if parsed.slots.origin or parsed.slots.transport_modes_requested:
            return parsed
        if parsed.slots.destination and parsed.slots.travel_days is not None:
            return parsed

        primary_tokens = int(llm.last_token_usage or 0)
        try:
            audit = await llm.structured_call(
                messages=[
                    {"role": "system", "content": _INTERCITY_TRANSPORT_AUDIT_PROMPT},
                    {"role": "user", "content": user_input},
                ],
                response_model=IntercityTransportAudit,
                temperature=0.0,
                task_type="intent",
            )
            if not isinstance(audit, IntercityTransportAudit):
                payload = audit.model_dump(mode="json") if hasattr(audit, "model_dump") else audit
                audit = IntercityTransportAudit.model_validate(payload)
            audit_tokens = int(llm.last_token_usage or 0)
            llm.last_token_usage = primary_tokens + audit_tokens
        except Exception as exc:
            llm.last_token_usage = primary_tokens
            logger.warning("Intercity transport intent audit failed: %s", exc)
            return parsed

        if audit.explicit_request and audit.confidence >= 0.65 and audit.modes:
            parsed.slots.transport_modes_requested = list(dict.fromkeys(audit.modes))
            if "transport" not in parsed.slots.information_needs:
                parsed.slots.information_needs.append("transport")
        return parsed

    async def parse_revision(
        self,
        revision_reason: str,
        *,
        current_goal: dict[str, object],
        recent_messages: list[dict[str, str]] | None = None,
    ) -> RevisionParseOutput:
        """Interpret draft feedback with the intent model and a typed output contract.

        Unlike initial slot extraction, revision parsing deliberately has no keyword
        fallback: silently guessing a constraint change is less safe than preserving
        the feedback verbatim and re-planning without an unverified mutation.
        """
        messages = [
            {"role": "system", "content": _REVISION_PARSER_PROMPT},
            {
                "role": "system",
                "content": f"系统当前日期：{date.today().isoformat()}。不得编造日期或年份。",
            },
            {
                "role": "system",
                "content": (
                    "当前旅行目标（JSON）："
                    + json.dumps(current_goal, ensure_ascii=False, default=str)
                ),
            },
            *(recent_messages or [])[-6:],
            {"role": "user", "content": revision_reason},
        ]
        return await llm.structured_call(
            messages=messages,
            response_model=RevisionParseOutput,
            temperature=0.0,
            task_type="intent",
        )

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
        required_slots = list(cls.REQUIRED_SLOTS)
        if flat_profile.get("transport_modes_requested"):
            required_slots.append("origin")
        for slot_key in required_slots:
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

        required_slots = list(cls.REQUIRED_SLOTS)
        if parsed.slots.transport_modes_requested:
            required_slots.append("origin")
        missing = [
            field
            for field in required_slots
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

        ordered = [field for field in (*cls.REQUIRED_SLOTS, "origin") if field in missing]
        labels = [cls.SLOT_LABELS.get(field, field) for field in ordered]
        if not labels:
            return ""
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
