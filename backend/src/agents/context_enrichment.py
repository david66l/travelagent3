"""Context Enrichment Agent - multi-dimensional travel context gathering with Redis caching."""

import asyncio
from datetime import datetime, timedelta

from core.llm_client import llm
from core.redis_client import redis_client
from schemas import TravelContext
from skills.tavily_search import UnifiedSearchSkill


class ContextEnrichmentAgent:
    """Gather rich travel context from 6 parallel search dimensions."""

    def __init__(self):
        self.search_skill = UnifiedSearchSkill()

    async def enrich(
        self,
        city: str,
        travel_days: int,
        travel_dates: str,
    ) -> TravelContext:
        """Execute 6 parallel searches and extract structured context with caching."""
        # Handle empty dates gracefully
        if not travel_dates or travel_dates.strip() == "":
            start = datetime.now() + timedelta(days=7)
            travel_dates = start.strftime("%Y-%m-%d")

        year, month = self._extract_year_month(travel_dates)
        cache_key = f"context:{city}:{year}{month}"

        # Try cache first
        try:
            cached = await redis_client.get_json(cache_key)
            if cached:
                return TravelContext(**cached)
        except Exception:
            pass

        # Build 6 search queries with their target dimension
        queries = [
            (f"{city} 旅游攻略 {travel_days}天 {year}", "route"),
            (f"{city} 美食 必吃 餐厅 推荐 {year}", "food"),
            (f"{city} 交通 地铁 出行 {year}", "transport"),
            (f"{city} 住宿 酒店 推荐 区域 {year}", "accommodation"),
            (f"{city} 最新活动 展览 节日 {year} {month}月", "events"),
            (f"{city} 旅游 避坑 注意事项 攻略 {year}", "tips"),
        ]

        # Parallel search
        tasks = [self._search_and_extract(q, dimension) for q, dimension in queries]
        results = await asyncio.gather(*tasks)

        # Merge all partial contexts into one
        merged = TravelContext()
        for partial in results:
            self._merge_into(merged, partial)

        # Cache result
        try:
            await redis_client.set_json(
                cache_key,
                merged.model_dump(),
                ttl=86400,  # 24 hours
            )
        except Exception:
            pass

        return merged

    async def _search_and_extract(self, query: str, dimension: str) -> TravelContext:
        """Search one query and extract relevant dimension."""
        search_results = await self.search_skill.search(query, top_n=6)
        if not search_results:
            return TravelContext()

        # Combine top 4 result snippets into text
        texts = []
        for r in search_results[:4]:
            if r.title or r.snippet:
                texts.append(f"标题：{r.title}\n摘要：{r.snippet}")
        combined_text = "\n\n".join(texts)

        if not combined_text:
            return TravelContext()

        return await self._extract_dimension(combined_text, dimension)

    async def _extract_dimension(self, text: str, dimension: str) -> TravelContext:
        """Extract one dimension from search text using LLM."""
        dimension_prompts = {
            "route": """从以下搜索结果中提取经典路线建议。

要求：
- 提取具体的路线框架（如"跨越5区的深度漫游"、"黄浦区+静安区+长宁区"等行政区规划路线）
- 提取推荐的游玩顺序逻辑
- 提取行程起始点建议

返回JSON格式：{"route_suggestions": "提取的路线建议文本"}""",
            "food": """从以下搜索结果中提取推荐的特色美食/餐厅信息。

要求：
- 每个餐厅包含：name(名称), cuisine_type(菜系), area(区域), price_range(人均消费), note(特色说明)
- 优先提取本地人推荐、必吃榜、米其林等口碑餐厅
- 提取时令美食信息

返回JSON格式：
{"food_specialties": [{"name":"...","cuisine_type":"...","area":"...","price_range":"...","note":"..."}]}""",
            "transport": """从以下搜索结果中提取交通出行提示。

要求：
- 地铁最新调整、首末班车时刻
- 机场联络线、机场大巴等交通信息
- 市内交通便利建议
- 打车/网约车提示

返回JSON格式：{"transport_tips": "提取的交通提示文本"}""",
            "accommodation": """从以下搜索结果中提取住宿推荐信息。

要求：
- 推荐住宿区域及特色（如"外滩南京东路生活便利"、"北外滩性价比之选"）
- 不同区域的优缺点对比
- 住宿类型建议（酒店/民宿/青旅）

返回JSON格式：{"accommodation_areas": "提取的住宿建议文本"}""",
            "events": """从以下搜索结果中提取近期活动/展览/节日信息。

要求：
- 每个活动包含：name(名称), date_range(日期范围), location(地点), description(简介), type(类型：展览/节日/演出/市集/其他)
- 提取季节限定亮点（如花卉节、限定市集、时令活动）
- 关注活动的具体日期是否在旅行期间

返回JSON格式：
{
  "upcoming_events": [{"name":"...","date_range":"...","location":"...","description":"...","type":"..."}],
  "seasonal_highlights": "季节限定亮点描述"
}""",
            "tips": """从以下搜索结果中提取避坑提醒和本地实用信息。

要求：
- 避坑提醒：宰客陷阱、消费陷阱、非法拉客等警告
- 实用小贴士：外滩亮灯时间、景点预约要求、排队攻略等
- 本地习俗/礼仪/文化注意事项

返回JSON格式：
{
  "pitfall_tips": ["tip1", "tip2", "tip3"],
  "local_customs": "本地习俗和礼仪说明"
}""",
        }

        prompt = f"""{dimension_prompts[dimension]}

搜索结果：
{text[:3000]}

只返回JSON，不要任何其他内容。如果没有相关信息，返回对应空值。"""

        try:
            response = await llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            import json

            # Clean markdown code block wrappers
            cleaned = response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                start = 0
                end = len(lines)
                for i, line in enumerate(lines):
                    if line.strip().startswith("```"):
                        if start == 0:
                            start = i + 1
                        else:
                            end = i
                            break
                cleaned = "\n".join(lines[start:end]).strip()

            data = json.loads(cleaned)
            return TravelContext(**data)
        except Exception:
            return TravelContext()

    @staticmethod
    def _merge_into(target: TravelContext, source: TravelContext) -> None:
        """Merge source context into target with deduplication."""
        # String fields: append if new and not already contained
        if source.route_suggestions and source.route_suggestions not in target.route_suggestions:
            target.route_suggestions += source.route_suggestions + "\n"
        if (
            source.accommodation_areas
            and source.accommodation_areas not in target.accommodation_areas
        ):
            target.accommodation_areas += source.accommodation_areas + "\n"
        if source.transport_tips and source.transport_tips not in target.transport_tips:
            target.transport_tips += source.transport_tips + "\n"
        if (
            source.seasonal_highlights
            and source.seasonal_highlights not in target.seasonal_highlights
        ):
            target.seasonal_highlights += source.seasonal_highlights + "\n"
        if source.local_customs and source.local_customs not in target.local_customs:
            target.local_customs += source.local_customs + "\n"

        # List fields: deduplicate by key field
        if source.upcoming_events:
            existing_event_names = {e.get("name") for e in target.upcoming_events}
            for e in source.upcoming_events:
                if e.get("name") and e.get("name") not in existing_event_names:
                    target.upcoming_events.append(e)
                    existing_event_names.add(e.get("name"))

        if source.food_specialties:
            existing_food_names = {f.get("name") for f in target.food_specialties}
            for f in source.food_specialties:
                if f.get("name") and f.get("name") not in existing_food_names:
                    target.food_specialties.append(f)
                    existing_food_names.add(f.get("name"))

        if source.pitfall_tips:
            existing_tips = set(target.pitfall_tips)
            for tip in source.pitfall_tips:
                if tip and tip not in existing_tips:
                    target.pitfall_tips.append(tip)
                    existing_tips.add(tip)

    @staticmethod
    def _extract_year_month(travel_dates: str) -> tuple[str, str]:
        """Extract year and month from travel_dates.

        Handles formats:
        - '2027-05-01 to 2027-05-07'
        - '2027-05-01'
        """
        try:
            first_date = travel_dates.split("to")[0].strip()
            parts = first_date.split("-")
            return parts[0], parts[1]
        except Exception:
            now = datetime.now()
            return str(now.year), f"{now.month:02d}"
