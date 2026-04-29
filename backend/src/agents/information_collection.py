"""Information Collection Agent - generate natural clarifying questions."""

from core.llm_client import llm


class InformationCollectionAgent:
    """Generate natural follow-up questions when required info is missing."""

    async def generate_questions(
        self,
        missing_required: list[str],
        missing_recommended: list[str],
        current_info: dict,
    ) -> list[str]:
        """Generate 1-2 natural clarifying questions."""
        field_labels = {
            "destination": "目的地",
            "travel_days": "旅行天数",
            "travel_dates": "出行日期",
            "travelers_count": "出行人数",
            "travelers_type": "同行类型",
            "budget_range": "预算范围",
            "food_preferences": "饮食偏好",
            "interests": "兴趣爱好",
            "pace": "旅行节奏",
        }

        questions = []

        # Prioritize required fields
        for field in missing_required[:2]:
            label = field_labels.get(field, field)
            if field == "destination":
                questions.append("想去哪里旅行呢？比如成都、杭州、西安这些热门城市？")
            elif field == "travel_days":
                questions.append("计划玩几天？3天、5天还是更久？")
            elif field == "travel_dates":
                questions.append("大概什么时候出发？可以告诉我具体日期或者月份~")
            elif field == "budget_range":
                questions.append("预算大概多少？2000/5000/8000，或者告诉我一个范围都可以")
            else:
                questions.append(f"能告诉我您的{label}吗？")

        # If no required fields missing, ask recommended fields
        if not questions and missing_recommended:
            for field in missing_recommended[:1]:
                label = field_labels.get(field, field)
                if field == "travelers_type":
                    questions.append("这次是和谁一起旅行？独自、情侣、亲子还是朋友？")
                elif field == "budget_range":
                    questions.append("方便透露一下预算范围吗？这样我能更好地推荐~")
                else:
                    questions.append(f"顺便问一下，您的{label}是什么？")

        return questions[:2]

    async def generate_response(
        self,
        missing_required: list[str],
        missing_recommended: list[str],
        current_info: dict,
    ) -> str:
        """Generate a natural response with clarifying questions."""
        questions = await self.generate_questions(
            missing_required, missing_recommended, current_info
        )

        if not questions:
            return "信息已经很完整了，马上为您规划行程！"

        # Use LLM to generate a natural-sounding message
        prompt = f"""你是旅行顾问。用户的信息还不完整，需要追问以下问题。请用自然、友好的语气组织成一段话，最多包含2个问题：

需要了解的信息：{questions}

要求：
1. 语气像朋友聊天，不要像填表
2. 提供选项减少用户的输入成本
3. 如果用户说"不知道"，用默认值并告知

直接输出追问文本，不要其他内容。"""

        response = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.strip()
