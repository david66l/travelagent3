"""Intent Recognition Agent - LLM-driven intent classification."""

import logging
import re
from datetime import datetime, timedelta
from typing import Optional

from core.llm_client import llm
from schemas import IntentResult, UserProfile

logger = logging.getLogger(__name__)


INTENT_PROMPT = """你是旅行Agent系统的意图识别专家。分析用户输入，判断意图并提取关键信息。

## 意图类型
1. generate_itinerary - 用户想要生成行程（如"我想去成都玩4天"）
2. modify_itinerary - 用户想要修改已有行程（如"第三天换个景点"）
3. update_preferences - 用户更新偏好（如"我喜欢吃辣"）
4. query_info - 用户询问信息（如"成都有什么好吃的"）
5. confirm_itinerary - 用户确认行程（如"确认行程"）
6. view_history - 用户查看历史行程（如"看我之前的行程"）
7. chitchat - 闲聊（如"你好"）

## 关键信息
- destination: 目的地城市
- travel_days: 旅行天数（整数）
- travel_dates: 旅行日期。可以是自然语言表达，如"下周一"、"下周"、"5月1日"、"这周五"。不要强制转换为YYYY-MM-DD，保持原样即可。
- travelers_count: 出行人数
- travelers_type: 同行类型（独自/情侣/亲子/朋友/父母）
- budget_range: 预算范围（数字，单位元）
- food_preferences: 饮食偏好列表（如["辣","清淡","海鲜"]）
- interests: 兴趣标签列表（如["历史","自然","拍照","美食"]）
- pace: 节奏（relaxed轻松/moderate适中/intensive紧凑）
- accommodation_preference: 住宿偏好
- special_requests: 特殊要求列表

## 偏好作用域（重要）
区分两种偏好，提取时请注意语义：
- 本次旅行限定：destination、travel_days、travel_dates、travelers_count、travelers_type、budget_range、special_requests。例如"这次和朋友去"中 travelers_type="朋友"仅限本次（下次可能带父母）。
- 长期个人资料：food_preferences、interests、pace、avoid、accommodation_preference。例如"我一直喜欢吃辣"中 food_preferences=["辣"]是长期偏好，跨旅行保留。
- 模糊地带根据语义判断。例如"这次想去看看历史景点"中 interests=["历史"]更像是本次限定；"我平时喜欢自然风光"中 interests=["自然"]更像是长期偏好。如果无法判断，interests 默认归为长期。

## 输出格式
以JSON格式输出（user_entities 为扁平的键值对，不需要 scope 标记）：
{
    "missing_required": ["缺失的必需字段"],
    "missing_recommended": ["缺失的建议字段"],
    "preference_changes": [{"field": "字段", "old_value": "旧值", "new_value": "新值"}],
    "clarification_questions": ["追问问题"],
    "disambiguation_candidates": [
        {"field": "destination", "candidates": [{"value": "城市", "reason": "推荐理由"}]}
    ],
    "reasoning": "判断理由"
}

## 歧义消解规则
- 用户说模糊词（如"南方"/"周边游"/"好玩的地方"）时，给出 disambiguation_candidates
- 每个候选附带简洁推荐理由（≤15字）
- "南方" → 推荐 厦门（6月淡季机票便宜）/ 成都（美食多节奏慢）/ 三亚（海边度假）
- "性价比高" → 不生成候选，而是追问预算范围

必需字段：destination, travel_days
建议字段：travel_dates, travelers_count, budget_range, travelers_type

## 日期提取规则
- 用户说"下周去" → travel_dates = "下周"
- 用户说"下周一出发" → travel_dates = "下周一"
- 用户说"5月1号到5号" → travel_dates = "5月1日-5月5日"
- 只要能提取到任何日期相关信息，travel_dates 就不算缺失
"""


class IntentRecognitionAgent:
    """Recognize user intent and extract entities."""

    async def recognize(
        self,
        user_input: str,
        messages: list[dict[str, str]],
        user_profile: Optional[UserProfile] = None,
    ) -> IntentResult:
        """Recognize intent from user input."""
        profile_str = user_profile.model_dump_json() if user_profile else "{}"

        prompt_messages = [
            {"role": "system", "content": INTENT_PROMPT},
            {"role": "system", "content": f"当前用户画像：{profile_str}"},
            *messages[-10:],  # Last 10 messages for context
            {"role": "user", "content": user_input},
        ]

        try:
            raw_result = await llm.structured_call(
                messages=prompt_messages,
                response_model=IntentResult,
                temperature=0.3,
            )
            result = self._coerce_result(raw_result)
        except Exception as exc:
            logger.warning(
                "LLM intent recognition failed; using deterministic fallback: %s",
                exc,
            )
            result = self._fallback_result(user_input)

        # Resolve natural language dates to concrete dates
        if result.user_entities.get("travel_dates"):
            resolved = self._resolve_date(result.user_entities["travel_dates"])
            if resolved:
                result.user_entities["travel_dates"] = resolved

        # Re-evaluate missing_required after date resolution
        if result.user_entities.get("travel_dates") and "travel_dates" in result.missing_required:
            result.missing_required.remove("travel_dates")

        # Detect preference changes by comparing with current profile
        if user_profile and result.user_entities:
            result.preference_changes = self._detect_changes(result.user_entities, user_profile)

        # Add clarifying questions if confidence is low
        if result.confidence < 0.7 and not result.clarification_questions:
            result.clarification_questions = ["能再详细说说您的需求吗？"]

        return result

    def _coerce_result(self, raw_result) -> IntentResult:
        """Convert LLM structured output to IntentResult or fall back on invalid data."""
        if isinstance(raw_result, IntentResult):
            return raw_result
        try:
            return IntentResult.model_validate(raw_result)
        except Exception as exc:
            raise ValueError(f"invalid intent payload: {raw_result!r}") from exc

    def _fallback_result(self, user_input: str) -> IntentResult:
        """Best-effort deterministic parser used when the LLM output is invalid.

        This is intentionally conservative: it only extracts enumerable facts from
        the user's text and marks confidence low so downstream logic can keep
        going without pretending the parse was perfect.
        """
        entities: dict = {}

        destination = self._extract_destination(user_input)
        if destination:
            entities["destination"] = destination

        travel_days = self._extract_travel_days(user_input)
        if travel_days:
            entities["travel_days"] = travel_days

        budget_range = self._extract_budget(user_input)
        if budget_range:
            entities["budget_range"] = budget_range

        travelers_count = self._extract_travelers_count(user_input)
        if travelers_count:
            entities["travelers_count"] = travelers_count

        pace = self._extract_pace(user_input)
        if pace:
            entities["pace"] = pace

        interests = self._extract_interests(user_input)
        if interests:
            entities["interests"] = interests

        food_preferences = self._extract_food_preferences(user_input)
        if food_preferences:
            entities["food_preferences"] = food_preferences

        missing_required = [
            field for field in ("destination", "travel_days") if field not in entities
        ]
        clarification_questions = []
        if "destination" in missing_required:
            clarification_questions.append("您想去哪个目的地？")
        if "travel_days" in missing_required:
            clarification_questions.append("计划玩几天？")

        return IntentResult(
            intent=self._classify_intent(user_input),
            confidence=0.55 if not missing_required else 0.35,
            user_entities=entities,
            missing_required=missing_required,
            missing_recommended=[
                field
                for field in ("travel_dates", "budget_range", "travelers_count")
                if field not in entities
            ],
            clarification_questions=clarification_questions,
            reasoning="LLM structured output invalid; used deterministic text fallback.",
        )

    def _classify_intent(self, text: str) -> str:
        if any(token in text for token in ("旅游", "旅行", "行程", "路线", "规划", "游", "玩")):
            return "generate_itinerary"
        if any(token in text for token in ("修改", "换", "改成", "不要", "删掉")):
            return "modify_itinerary"
        if any(token in text for token in ("确认", "就这样", "可以了")):
            return "confirm_itinerary"
        if any(token in text for token in ("之前", "历史", "上次")):
            return "view_history"
        if any(token in text for token in ("喜欢", "偏好", "预算", "节奏")) and not any(
            token in text for token in ("旅游", "旅行", "行程", "游", "玩")
        ):
            return "update_preferences"
        if any(token in text for token in ("什么", "推荐", "怎么去", "多少钱")) and not any(
            token in text for token in ("行程", "路线", "规划", "游", "玩")
        ):
            return "query_info"
        return "chitchat"

    def _extract_destination(self, text: str) -> Optional[str]:
        for city in sorted(self._known_destinations(), key=len, reverse=True):
            if city and (city in text or f"{city}市" in text):
                return city

        patterns = [
            r"(?:去|到|在)([\u4e00-\u9fff]{2,8})(?:玩|旅游|旅行|度假|自由行)",
            r"(?:去|到|在)([\u4e00-\u9fff]{2,8})(?:\d+|[一二两三四五六七八九十]+)[日天]",
            r"([\u4e00-\u9fff]{2,8})(?:\d+|[一二两三四五六七八九十]+)[日天]游",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return self._normalize_destination(match.group(1))
        return None

    def _known_destinations(self) -> set[str]:
        try:
            from skills.poi_search import CITY_FALLBACK_POIS

            return set(CITY_FALLBACK_POIS.keys())
        except Exception:
            return set()

    def _normalize_destination(self, value: str) -> str:
        value = value.strip("，。,.、 ")
        for suffix in ("市", "省"):
            if value.endswith(suffix) and len(value) > len(suffix) + 1:
                value = value[: -len(suffix)]
        return value

    def _extract_travel_days(self, text: str) -> Optional[int]:
        patterns = [
            r"([0-9]{1,2}|[一二两三四五六七八九十]{1,3})\s*(?:天|日)(?:游|行程|旅行|旅游)?",
            r"(?:玩|旅行|旅游|规划)\s*([0-9]{1,2}|[一二两三四五六七八九十]{1,3})\s*(?:天|日)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                days = self._parse_int(match.group(1))
                if days and 1 <= days <= 30:
                    return days
        return None

    def _extract_budget(self, text: str) -> Optional[float]:
        patterns = [
            r"(?:预算|花费|费用|人均)\s*([0-9]+(?:\.[0-9]+)?)\s*([万千kK]?)",
            r"([0-9]+(?:\.[0-9]+)?)\s*([万千kK]?)\s*(?:元|块|人民币)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            value = float(match.group(1))
            unit = match.group(2)
            if unit == "万":
                value *= 10000
            elif unit in ("千", "k", "K"):
                value *= 1000
            return int(value) if value.is_integer() else value
        return None

    def _extract_travelers_count(self, text: str) -> Optional[int]:
        match = re.search(r"([0-9]{1,2}|[一二两三四五六七八九十]{1,3})\s*(?:个人|人|位)", text)
        if not match:
            return None
        count = self._parse_int(match.group(1))
        return count if count and 1 <= count <= 50 else None

    def _extract_pace(self, text: str) -> Optional[str]:
        if any(
            token in text
            for token in ("轻松", "不赶", "不要太赶", "慢一点", "休闲", "老人", "父母")
        ):
            return "relaxed"
        if any(token in text for token in ("紧凑", "多逛", "特种兵", "效率")):
            return "intensive"
        return None

    def _extract_interests(self, text: str) -> list[str]:
        mapping = {
            "历史": ("历史", "古迹", "博物馆", "文化", "人文"),
            "自然": ("自然", "山", "湖", "海", "公园", "徒步"),
            "美食": ("美食", "吃", "小吃", "餐厅", "夜市"),
            "拍照": ("拍照", "出片", "摄影", "打卡"),
            "亲子": ("亲子", "孩子", "儿童"),
            "购物": ("购物", "商场", "逛街"),
        }
        return [
            label for label, tokens in mapping.items() if any(token in text for token in tokens)
        ]

    def _extract_food_preferences(self, text: str) -> list[str]:
        tokens = (
            "辣",
            "清淡",
            "海鲜",
            "火锅",
            "烧烤",
            "甜品",
            "素食",
            "面食",
            "烤鸭",
            "本地菜",
        )
        return [token for token in tokens if token in text]

    def _parse_int(self, token: str) -> Optional[int]:
        if token.isdigit():
            return int(token)

        digit_map = {
            "零": 0,
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
        }
        if token in digit_map:
            return digit_map[token]
        if "十" in token:
            left, _, right = token.partition("十")
            tens = digit_map.get(left, 1) if left else 1
            ones = digit_map.get(right, 0) if right else 0
            return tens * 10 + ones
        return None

    def _resolve_date(self, date_str: str) -> Optional[str]:
        """Resolve natural language Chinese dates to YYYY-MM-DD or range.

        Examples:
            "下周一" → "2026-04-27"
            "下周" → "2026-04-27"
            "明天" → "2026-04-29"
            "5月1日" → "2026-05-01"
            "5月1号到5号" → "2026-05-01 to 2026-05-05"
        """
        if not date_str or not isinstance(date_str, str):
            return None

        date_str = date_str.strip()
        today = datetime.now().date()

        # Weekday names
        weekdays = {
            "一": 0,
            "周一": 0,
            "星期一": 0,
            "二": 1,
            "周二": 1,
            "星期二": 1,
            "三": 2,
            "周三": 2,
            "星期三": 2,
            "四": 3,
            "周四": 3,
            "星期四": 3,
            "五": 4,
            "周五": 4,
            "星期五": 4,
            "六": 5,
            "周六": 5,
            "星期六": 5,
            "日": 6,
            "周日": 6,
            "星期日": 6,
            "天": 6,
            "周天": 6,
        }

        # Pattern: 下周一、下周二
        m = re.match(
            r"下[周\s]*([一二三四五六日天]|周一|周二|周三|周四|周五|周六|周日|周天|星期[一二三四五六日])",
            date_str,
        )
        if m:
            wd_name = m.group(1)
            target_wd = weekdays.get(wd_name)
            if target_wd is not None:
                days_ahead = (target_wd - today.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                days_ahead += 7  # next week
                return (today + timedelta(days=days_ahead)).isoformat()

        # Pattern: 下周（泛指）
        if "下周" in date_str:
            next_monday = today + timedelta(days=(7 - today.weekday()) % 7 or 7)
            return next_monday.isoformat()

        # Pattern: 本周X / 这周五
        m = re.match(
            r"[这本]周\s*([一二三四五六日天]|周一|周二|周三|周四|周五|周六|周日|周天|星期[一二三四五六日])",
            date_str,
        )
        if m:
            wd_name = m.group(1)
            target_wd = weekdays.get(wd_name)
            if target_wd is not None:
                days_ahead = (target_wd - today.weekday()) % 7
                if days_ahead == 0:
                    return today.isoformat()
                return (today + timedelta(days=days_ahead)).isoformat()

        # Pattern: 明天
        if "明天" in date_str or "明日" in date_str:
            return (today + timedelta(days=1)).isoformat()

        # Pattern: 后天
        if "后天" in date_str:
            return (today + timedelta(days=2)).isoformat()

        # Pattern: X月X日 到 X月X日 / X月X号-X月X号 (range first!)
        m = re.search(
            r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]\s*[~\-到至]\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]",
            date_str,
        )
        if m:
            m1, d1, m2, d2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            year = today.year
            try:
                start = datetime(year, m1, d1).date()
                end = datetime(year, m2, d2).date()
                if start < today:
                    start = datetime(year + 1, m1, d1).date()
                    end = datetime(year + 1, m2, d2).date()
                return f"{start.isoformat()} to {end.isoformat()}"
            except ValueError:
                pass

        # Pattern: X月X日 到 X日（同月）
        m = re.search(
            r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]\s*[~\-到至]\s*(\d{1,2})\s*[日号]", date_str
        )
        if m:
            month, d1, d2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
            year = today.year
            try:
                start = datetime(year, month, d1).date()
                end = datetime(year, month, d2).date()
                if start < today:
                    start = datetime(year + 1, month, d1).date()
                    end = datetime(year + 1, month, d2).date()
                return f"{start.isoformat()} to {end.isoformat()}"
            except ValueError:
                pass

        # Pattern: X月X日 或 X月X号 (single date after ranges)
        m = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]", date_str)
        if m:
            month, day = int(m.group(1)), int(m.group(2))
            year = today.year
            try:
                d = datetime(year, month, day).date()
                if d < today:
                    d = datetime(year + 1, month, day).date()
                return d.isoformat()
            except ValueError:
                pass

        # Pattern: YYYY-MM-DD already
        if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
            return date_str

        # If we can't parse but it's not empty, keep original
        return date_str if date_str else None

    def _detect_changes(self, entities: dict, profile: UserProfile) -> list[dict]:
        """Detect preference changes by comparing new entities with current profile."""
        changes = []
        field_map = {
            "food_preferences": profile.food_preferences,
            "interests": profile.interests,
            "pace": profile.pace,
            "accommodation_preference": profile.accommodation_preference,
            "budget_range": profile.budget_range,
            "travel_dates": profile.travel_dates,
            "travel_days": profile.travel_days,
            "destination": profile.destination,
        }

        for field, old_value in field_map.items():
            if field in entities:
                new_value = entities[field]
                if new_value != old_value:
                    changes.append(
                        {
                            "field": field,
                            "old_value": str(old_value) if old_value is not None else None,
                            "new_value": str(new_value) if new_value is not None else None,
                        }
                    )

        return changes
