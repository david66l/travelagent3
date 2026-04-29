"""Intent Recognition Agent - LLM-driven intent classification."""

import json
import re
from datetime import datetime, timedelta
from typing import Optional

from core.llm_client import llm
from schemas import IntentResult, UserProfile


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

## 输出格式
以JSON格式输出：
{
    "intent": "意图类型",
    "confidence": 0.0-1.0,
    "user_entities": {"提取的实体键值对"},
    "missing_required": ["缺失的必需字段"],
    "missing_recommended": ["缺失的建议字段"],
    "preference_changes": [{"field": "字段", "old_value": "旧值", "new_value": "新值"}],
    "clarification_questions": ["追问问题"],
    "reasoning": "判断理由"
}

必需字段：destination, travel_days, travel_dates
建议字段：travelers_count, budget_range, travelers_type

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

        result = await llm.structured_call(
            messages=prompt_messages,
            response_model=IntentResult,
            temperature=0.3,
        )

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
            result.preference_changes = self._detect_changes(
                result.user_entities, user_profile
            )

        # Add clarifying questions if confidence is low
        if result.confidence < 0.7 and not result.clarification_questions:
            result.clarification_questions = ["能再详细说说您的需求吗？"]

        return result

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
            "一": 0, "周一": 0, "星期一": 0,
            "二": 1, "周二": 1, "星期二": 1,
            "三": 2, "周三": 2, "星期三": 2,
            "四": 3, "周四": 3, "星期四": 3,
            "五": 4, "周五": 4, "星期五": 4,
            "六": 5, "周六": 5, "星期六": 5,
            "日": 6, "周日": 6, "星期日": 6, "天": 6, "周天": 6,
        }

        # Pattern: 下周一、下周二
        m = re.match(r"下[周\s]*([一二三四五六日天]|周一|周二|周三|周四|周五|周六|周日|周天|星期[一二三四五六日])", date_str)
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
        m = re.match(r"[这本]周\s*([一二三四五六日天]|周一|周二|周三|周四|周五|周六|周日|周天|星期[一二三四五六日])", date_str)
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
        m = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]\s*[~\-到至]\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]", date_str)
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
        m = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]\s*[~\-到至]\s*(\d{1,2})\s*[日号]", date_str)
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

    def _detect_changes(
        self, entities: dict, profile: UserProfile
    ) -> list[dict]:
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
                    changes.append({
                        "field": field,
                        "old_value": str(old_value) if old_value is not None else None,
                        "new_value": str(new_value) if new_value is not None else None,
                    })

        return changes
