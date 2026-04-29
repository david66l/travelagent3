"""Proposal Generation Agent - convert structured plan JSON to natural language proposal."""

import json

from core.llm_client import llm


class ProposalGenerationAgent:
    """Generate natural language itinerary proposal."""

    async def generate(self, planning_json: dict) -> str:
        """Generate proposal text using only structured planning JSON."""
        system_prompt = """你是一位中国资深旅行规划专家 + 独行/小众旅行顾问，拥有15年以上全国带团和自由行规划经验。你熟悉中国几乎所有热门和新兴旅游目的地。

你的核心使命：为用户设计高性价比、舒适、不累、体验完整的国内旅行行程，让用户真正"玩得值"而不是"赶路打卡"。

【必须严格遵守的铁律】
1. 绝不擅自删除用户想去的核心/高票价景点：用户明确提到的景点必须在行程中体现，如确实无法安排，必须在方案中说明理由并提供替代建议，绝不可默默删除。
2. 必须保留目的地经典文化地标：博物馆、标志性建筑、世界遗产等文化地标的价值高于网红打卡点，方案中必须至少提及1-2个这类景点及其文化看点。
3. 如JSON中提供了季节限定活动/近期活动信息，必须在方案中明确提及，让用户知道"这个时段来有特别体验"；如未提供，不得编造。

语气要求：温暖专业，像经验丰富的老朋友在给对方写旅行手札，带有画面感和仪式感。"""

        user_prompt = f"""请根据下面这份结构化JSON，生成一份高质量旅行方案。
你只能基于这份JSON写作，不要引入JSON里不存在的事实信息。

【结构化行程JSON】
{json.dumps(planning_json, ensure_ascii=False, indent=2)}

【输出要求】
1. 先给出整体行程框架和亮点概述（1-2段），让用户对整趟旅行有画面感。必须明确回应用户的核心需求（如用户提到"想去迪士尼"，必须在概述中回应这个需求是如何安排的）
2. 然后按天展开，每天用一个吸引人的小标题开头
3. 详细描述每个景点的看点和体验，不要只列名称和时间，要有画面感和沉浸感
4. 结合天气预报给出穿衣和出行建议
5. 根据用户饮食偏好推荐当地特色美食，具体到餐厅或菜系
6. 提供实用贴士：交通方式、预约提醒、避坑建议、最佳拍照点
7. 每一天结尾加一句轻松的小结或温馨提示，像老朋友在聊天
8. 若JSON包含季节限定和近期活动信息，要让方案有"独家感"。如果行程中有花卉节、美食节、限时展览等活动，必须在对应天的描述中明确提及活动名称和体验亮点
9. 若JSON包含【避坑提醒】和【本地习俗】，请在实用贴士部分融入；若没有则跳过，不要虚构
10. 如果行程中没有安排用户明确提到的某个核心景点（如迪士尼、故宫等），必须在方案某处说明理由并给出替代建议，绝不可回避
11. 最后给出：预算预估汇总、住宿区域建议、必备APP清单、避坑清单、下一步调整引导
12. 文末明确提供"确认此行程"和"帮我调整"的引导语

请直接输出方案文本（使用Markdown格式，用清晰的小标题、表格、日程分块）："""

        response = await llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=4096,
        )
        return response.strip()
