"""POI Search Skill - Tavily answer-driven structured POI extraction with Redis caching."""

import asyncio
import hashlib
import logging
from typing import Optional

from core.llm_client import llm
from core.redis_client import redis_client
from schemas import ScoredPOI
from skills.tavily_search import TavilySearchSkill, SearchResult

logger = logging.getLogger(__name__)


# Minimal fallback POIs per city (name + category only)
CITY_FALLBACK_POIS: dict[str, list[dict]] = {
    "北京": [
        {"name": "故宫", "category": "attraction"},
        {"name": "长城", "category": "attraction"},
        {"name": "天坛", "category": "attraction"},
        {"name": "颐和园", "category": "attraction"},
        {"name": "王府井", "category": "attraction"},
        {"name": "798艺术区", "category": "attraction"},
        {"name": "南锣鼓巷", "category": "attraction"},
        {"name": "鸟巢/水立方", "category": "attraction"},
        {"name": "什刹海", "category": "attraction"},
        {"name": "国家博物馆", "category": "attraction"},
        {"name": "景山公园", "category": "attraction"},
        {"name": "北海公园", "category": "attraction"},
        {"name": "前门大街", "category": "attraction"},
        {"name": "雍和宫", "category": "attraction"},
        {"name": "恭王府", "category": "attraction"},
        {"name": "全聚德", "category": "restaurant"},
        {"name": "四季民福", "category": "restaurant"},
        {"name": "护国寺小吃", "category": "restaurant"},
        {"name": "方砖厂炸酱面", "category": "restaurant"},
        {"name": "姚记炒肝", "category": "restaurant"},
    ],
    "上海": [
        {"name": "外滩", "category": "attraction"},
        {"name": "东方明珠", "category": "attraction"},
        {"name": "豫园", "category": "attraction"},
        {"name": "南京路步行街", "category": "attraction"},
        {"name": "田子坊", "category": "attraction"},
        {"name": "新天地", "category": "attraction"},
        {"name": "陆家嘴", "category": "attraction"},
        {"name": "上海博物馆", "category": "attraction"},
        {"name": "城隍庙", "category": "attraction"},
        {"name": "武康路", "category": "attraction"},
        {"name": "1933老场坊", "category": "attraction"},
        {"name": "上海迪士尼", "category": "attraction"},
        {"name": "思南公馆", "category": "attraction"},
        {"name": "M50创意园", "category": "attraction"},
        {"name": "七宝老街", "category": "attraction"},
        {"name": "南翔馒头店", "category": "restaurant"},
        {"name": "小杨生煎", "category": "restaurant"},
        {"name": "老正兴菜馆", "category": "restaurant"},
        {"name": "新天地美食", "category": "restaurant"},
        {"name": "云南路美食街", "category": "restaurant"},
    ],
    "广州": [
        {"name": "广州塔", "category": "attraction"},
        {"name": "陈家祠", "category": "attraction"},
        {"name": "沙面", "category": "attraction"},
        {"name": "北京路步行街", "category": "attraction"},
        {"name": "上下九步行街", "category": "attraction"},
        {"name": "越秀公园", "category": "attraction"},
        {"name": "珠江夜游", "category": "attraction"},
        {"name": "广东省博物馆", "category": "attraction"},
        {"name": "白云山", "category": "attraction"},
        {"name": "长隆欢乐世界", "category": "attraction"},
        {"name": "石室圣心大教堂", "category": "attraction"},
        {"name": "红砖厂", "category": "attraction"},
        {"name": "永庆坊", "category": "attraction"},
        {"name": "花城广场", "category": "attraction"},
        {"name": "华南植物园", "category": "attraction"},
        {"name": "点都德", "category": "restaurant"},
        {"name": "陶陶居", "category": "restaurant"},
        {"name": "广州酒家", "category": "restaurant"},
        {"name": "银记肠粉", "category": "restaurant"},
        {"name": "陈添记", "category": "restaurant"},
    ],
    "成都": [
        {"name": "宽窄巷子", "category": "attraction"},
        {"name": "锦里", "category": "attraction"},
        {"name": "大熊猫繁育基地", "category": "attraction"},
        {"name": "武侯祠", "category": "attraction"},
        {"name": "杜甫草堂", "category": "attraction"},
        {"name": "春熙路", "category": "attraction"},
        {"name": "青城山", "category": "attraction"},
        {"name": "都江堰", "category": "attraction"},
        {"name": "文殊院", "category": "attraction"},
        {"name": "人民公园", "category": "attraction"},
        {"name": "九眼桥", "category": "attraction"},
        {"name": "金沙遗址", "category": "attraction"},
        {"name": "太古里", "category": "attraction"},
        {"name": "黄龙溪古镇", "category": "attraction"},
        {"name": "东郊记忆", "category": "attraction"},
        {"name": "蜀大侠火锅", "category": "restaurant"},
        {"name": "小龙坎", "category": "restaurant"},
        {"name": "陈麻婆豆腐", "category": "restaurant"},
        {"name": "钟水饺", "category": "restaurant"},
        {"name": "龙抄手", "category": "restaurant"},
    ],
    "杭州": [
        {"name": "西湖", "category": "attraction"},
        {"name": "灵隐寺", "category": "attraction"},
        {"name": "宋城", "category": "attraction"},
        {"name": "西溪湿地", "category": "attraction"},
        {"name": "河坊街", "category": "attraction"},
        {"name": "千岛湖", "category": "attraction"},
        {"name": "龙井村", "category": "attraction"},
        {"name": "断桥残雪", "category": "attraction"},
        {"name": "雷峰塔", "category": "attraction"},
        {"name": "楼外楼", "category": "restaurant"},
        {"name": "知味小笼", "category": "restaurant"},
        {"name": "外婆家", "category": "restaurant"},
    ],
    "西安": [
        {"name": "兵马俑", "category": "attraction"},
        {"name": "大雁塔", "category": "attraction"},
        {"name": "古城墙", "category": "attraction"},
        {"name": "回民街", "category": "attraction"},
        {"name": "华清宫", "category": "attraction"},
        {"name": "大唐芙蓉园", "category": "attraction"},
        {"name": "陕西历史博物馆", "category": "attraction"},
        {"name": "钟楼", "category": "attraction"},
        {"name": "鼓楼", "category": "attraction"},
        {"name": "肉夹馍", "category": "restaurant"},
        {"name": "羊肉泡馍", "category": "restaurant"},
        {"name": "凉皮", "category": "restaurant"},
    ],
    "重庆": [
        {"name": "洪崖洞", "category": "attraction"},
        {"name": "解放碑", "category": "attraction"},
        {"name": "磁器口", "category": "attraction"},
        {"name": "长江索道", "category": "attraction"},
        {"name": "武隆天坑", "category": "attraction"},
        {"name": "朝天门", "category": "attraction"},
        {"name": "鹅岭二厂", "category": "attraction"},
        {"name": "南山一棵树", "category": "attraction"},
        {"name": "四川美术学院", "category": "attraction"},
        {"name": "重庆火锅", "category": "restaurant"},
        {"name": "小面", "category": "restaurant"},
        {"name": "酸辣粉", "category": "restaurant"},
    ],
    "深圳": [
        {"name": "世界之窗", "category": "attraction"},
        {"name": "欢乐谷", "category": "attraction"},
        {"name": "大梅沙", "category": "attraction"},
        {"name": "华侨城", "category": "attraction"},
        {"name": "深圳湾公园", "category": "attraction"},
        {"name": "莲花山公园", "category": "attraction"},
        {"name": "东门老街", "category": "attraction"},
        {"name": "海上世界", "category": "attraction"},
        {"name": "大鹏所城", "category": "attraction"},
        {"name": "潮汕牛肉火锅", "category": "restaurant"},
        {"name": "椰子鸡", "category": "restaurant"},
        {"name": "肠粉", "category": "restaurant"},
    ],
    "南京": [
        {"name": "中山陵", "category": "attraction"},
        {"name": "夫子庙", "category": "attraction"},
        {"name": "秦淮河", "category": "attraction"},
        {"name": "明孝陵", "category": "attraction"},
        {"name": "总统府", "category": "attraction"},
        {"name": "鸡鸣寺", "category": "attraction"},
        {"name": "玄武湖", "category": "attraction"},
        {"name": "南京博物院", "category": "attraction"},
        {"name": "老门东", "category": "attraction"},
        {"name": "鸭血粉丝汤", "category": "restaurant"},
        {"name": "盐水鸭", "category": "restaurant"},
        {"name": "小笼包", "category": "restaurant"},
    ],
    "厦门": [
        {"name": "鼓浪屿", "category": "attraction"},
        {"name": "南普陀寺", "category": "attraction"},
        {"name": "厦门大学", "category": "attraction"},
        {"name": "环岛路", "category": "attraction"},
        {"name": "曾厝垵", "category": "attraction"},
        {"name": "中山路步行街", "category": "attraction"},
        {"name": "胡里山炮台", "category": "attraction"},
        {"name": "园林植物园", "category": "attraction"},
        {"name": "沙坡尾", "category": "attraction"},
        {"name": "沙茶面", "category": "restaurant"},
        {"name": "海蛎煎", "category": "restaurant"},
        {"name": "花生汤", "category": "restaurant"},
    ],
    "青岛": [
        {"name": "栈桥", "category": "attraction"},
        {"name": "八大关", "category": "attraction"},
        {"name": "崂山", "category": "attraction"},
        {"name": "五四广场", "category": "attraction"},
        {"name": "青岛啤酒博物馆", "category": "attraction"},
        {"name": "小鱼山", "category": "attraction"},
        {"name": "信号山公园", "category": "attraction"},
        {"name": "金沙滩", "category": "attraction"},
        {"name": "劈柴院", "category": "attraction"},
        {"name": "青岛啤酒", "category": "restaurant"},
        {"name": "海鲜大咖", "category": "restaurant"},
        {"name": "鲅鱼饺子", "category": "restaurant"},
    ],
}


class POISearchSkill:
    """Search and extract structured POI data using Tavily answer + JSON mode."""

    def __init__(self):
        self.tavily = TavilySearchSkill()
        # Keep extraction bounded to avoid long-tail latency from oversized completions.
        self.extract_max_tokens = 1200

    async def search_pois(
        self,
        city: str,
        keywords: list[str],
        category: Optional[str] = None,
    ) -> list[ScoredPOI]:
        """Search POIs for a city with given keywords."""
        # Build cache key
        keywords_str = "|".join(sorted(keywords)) if keywords else "_"
        keywords_hash = hashlib.md5(keywords_str.encode()).hexdigest()[:12]
        cache_key = f"pois:{city}:{keywords_hash}"

        # Try cache first
        try:
            cached = await redis_client.get_json(cache_key)
            if cached:
                logger.info(f"POI cache hit for {city}")
                return [ScoredPOI(**p) for p in cached]
        except Exception:
            pass

        # Build multiple queries for broader coverage
        queries = [
            f"{city} 必去景点 旅游攻略 推荐",
            f"{city} 美食 餐厅 特色小吃 推荐",
        ]
        if keywords:
            queries.append(f"{city} {' '.join(keywords[:3])} 推荐 攻略")
        if category:
            queries.append(f"{city} {category} 推荐")

        # Search all queries with Tavily context (results + answer)
        all_results: list[SearchResult] = []
        all_answers: list[str] = []
        search_tasks = [self.tavily.search_with_context(q, top_n=6) for q in queries]
        search_outputs = await asyncio.gather(*search_tasks)
        for results, answer in search_outputs:
            all_results.extend(results)
            if answer:
                all_answers.append(answer)

        # Deduplicate results by URL
        seen_urls: set[str] = set()
        unique_results: list[SearchResult] = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique_results.append(r)

        all_pois: list[ScoredPOI] = []

        # PRIMARY: Extract from Tavily answers (most reliable content source)
        if all_answers:
            combined_answer = "\n\n".join(all_answers)
            answer_pois = await self._extract_pois_from_answer(combined_answer, city, keywords)
            logger.info(f"Extracted {len(answer_pois)} POIs from Tavily answers")
            all_pois.extend(answer_pois)

        # SECONDARY: Extract from search snippets (no crawl needed)
        snippets = [r.snippet for r in unique_results if r.snippet]
        if snippets and len(all_pois) < 10:
            snippet_pois = await self._extract_from_snippets(snippets, city, keywords)
            logger.info(f"Extracted {len(snippet_pois)} POIs from snippets")
            all_pois.extend(snippet_pois)

        # FALLBACK: Built-in city default POIs if still too few
        if len(all_pois) < 8:
            fallback = self._get_fallback_pois(city)
            logger.info(f"Using {len(fallback)} fallback POIs for {city}")
            all_pois.extend(fallback)

        # Deduplicate by name
        seen_names: set[str] = set()
        unique_pois: list[ScoredPOI] = []
        for poi in all_pois:
            name_key = poi.name.strip()
            if name_key and name_key not in seen_names and len(name_key) >= 2:
                seen_names.add(name_key)
                unique_pois.append(poi)

        # Score and sort
        scored = self._score_pois(unique_pois, keywords)
        scored.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"Final POI count for {city}: {len(scored)}")

        # Cache result
        try:
            await redis_client.set_json(
                cache_key,
                [p.model_dump() for p in scored[:40]],
                ttl=3600,  # 1 hour
            )
        except Exception:
            pass

        return scored[:40]

    async def _extract_pois_from_answer(
        self, answer: str, city: str, keywords: list[str]
    ) -> list[ScoredPOI]:
        """Extract structured POIs from Tavily's AI-generated answer."""
        prompt = f"""从以下关于{city}的旅游信息中提取景点和餐厅。

要求：
1. 返回 JSON 对象，包含一个 "pois" 数组
2. 每个 POI 包含以下字段：
   - name: 名称（必须）
   - category: "attraction"（景点）或 "restaurant"（餐厅）
   - description: 简介（50字以内）
   - tags: 标签列表，如 ["夜景","历史","亲子","美食"]
   - best_time: 最佳游览时间，如 "上午","下午","傍晚","全天","晚上"
   - area: 所在区域/商圈，如 "朝阳区","东城区"
   - recommended_hours: 建议时长，如 "1-2小时","半天","2-3小时"
   - indoor_outdoor: "indoor" / "outdoor" / "mixed"
   - ticket_price: 门票价格（数字，单位元，免费填0，未知填null）
   - open_time: 开放时间，如 "08:30-17:00"
3. 尽量提取完整信息，不确定的字段填 null
4. 不要输出任何其他内容，只返回 JSON
5. 最多返回25个POI，信息不确定时宁缺毋滥，不要为了凑数量编造。

信息内容：
{answer[:6000]}

示例输出：
{{
  "pois": [
    {{
      "name": "故宫博物院",
      "category": "attraction",
      "description": "明清皇宫，世界文化遗产",
      "tags": ["历史", "文化", "皇家"],
      "best_time": "上午",
      "area": "东城区",
      "recommended_hours": "半天",
      "indoor_outdoor": "mixed",
      "ticket_price": 60,
      "open_time": "08:30-17:00"
    }}
  ]
}}"""

        # Try JSON mode first (most reliable)
        try:
            data = await llm.json_chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=self.extract_max_tokens,
            )
            pois_data = data.get("pois", []) if isinstance(data, dict) else []
            if isinstance(pois_data, list) and len(pois_data) > 0:
                return self._parse_poi_items(pois_data)
        except Exception as e:
            logger.warning(f"JSON mode extraction from answer failed: {e}")

        # Don't do a second LLM call here; avoid cascading latency in POI stage.
        return []

    async def _extract_from_snippets(
        self, snippets: list[str], city: str, keywords: list[str]
    ) -> list[ScoredPOI]:
        """Extract POIs from search result snippets."""
        if not snippets:
            return []

        combined = "\n".join(snippets[:10])
        prompt = f"""从以下关于{city}的搜索结果摘要中提取景点和餐厅。

要求：
1. 返回 JSON 对象，包含一个 "pois" 数组
2. 每个 POI 包含：name, category, description, tags, best_time, area, recommended_hours, indoor_outdoor, ticket_price
3. category 只能是 attraction 或 restaurant
4. 信息不完整时合理推断，不确定填 null
5. 只返回 JSON，不要其他内容
6. 最多返回20个POI，避免冗长输出

摘要：
{combined[:2000]}

示例：{{"pois":[{{"name":"故宫","category":"attraction","description":"...","tags":["历史"],"best_time":"上午","area":"东城区","recommended_hours":"半天","indoor_outdoor":"mixed","ticket_price":60}}]}}"""

        # Try JSON mode first
        try:
            data = await llm.json_chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=self.extract_max_tokens,
            )
            pois_data = data.get("pois", []) if isinstance(data, dict) else []
            if isinstance(pois_data, list) and len(pois_data) > 0:
                return self._parse_poi_items(pois_data)
        except Exception as e:
            logger.warning(f"JSON mode extraction from snippets failed: {e}")

        # Don't do a second LLM call here; avoid cascading latency in POI stage.
        return []

    def _get_fallback_pois(self, city: str) -> list[ScoredPOI]:
        """Return built-in fallback POIs when extraction fails."""
        fallback_data = CITY_FALLBACK_POIS.get(city, [])
        pois = []
        for item in fallback_data:
            pois.append(
                ScoredPOI(
                    name=item["name"],
                    category=item["category"],
                    description="",
                    score=0.5,
                    tags=[],
                )
            )
        return pois

    @staticmethod
    def _clean_json_response(response: str) -> str:
        """Clean LLM response to extract valid JSON."""
        text = response.strip()

        # Remove markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            start = 0
            end = len(lines)
            for i, line in enumerate(lines):
                if line.strip().startswith("```"):
                    if start == 0:
                        start = i + 1
                    else:
                        end = i
                        break
            text = "\n".join(lines[start:end]).strip()

        # Find JSON boundaries
        start_brace = text.find("{")
        start_bracket = text.find("[")
        start_candidates = [x for x in [start_brace, start_bracket] if x != -1]
        if not start_candidates:
            return ""
        start_idx = min(start_candidates)

        if text[start_idx] == "{":
            end_idx = text.rfind("}")
        else:
            end_idx = text.rfind("]")

        if end_idx != -1 and end_idx > start_idx:
            return text[start_idx : end_idx + 1]
        return ""

    def _parse_poi_items(self, items: list[dict]) -> list[ScoredPOI]:
        """Parse list of dicts into ScoredPOI objects."""
        pois = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "").strip()
            if not name or len(name) < 2:
                continue

            # Parse ticket_price
            ticket = item.get("ticket_price")
            try:
                if ticket is not None and ticket != "":
                    ticket = float(ticket)
                else:
                    ticket = None
            except (ValueError, TypeError):
                ticket = None

            # Normalize time_constraint
            tc = item.get("time_constraint", "flexible")
            if tc not in ("flexible", "morning_only", "afternoon_only", "evening_only"):
                tc = "flexible"

            # Normalize indoor_outdoor
            io = item.get("indoor_outdoor")
            if io and io.lower() not in ("indoor", "outdoor", "mixed"):
                io = None

            pois.append(
                ScoredPOI(
                    name=name,
                    category=item.get("category", "attraction"),
                    description=item.get("description", ""),
                    tags=item.get("tags", []) if isinstance(item.get("tags"), list) else [],
                    score=0.5,
                    highlights=item.get("highlights"),
                    best_time=item.get("best_time"),
                    area=item.get("area"),
                    recommended_hours=item.get("recommended_hours"),
                    indoor_outdoor=io,
                    open_time=item.get("open_time"),
                    ticket_price=ticket,
                    time_constraint=tc,
                )
            )
        return pois

    def _score_pois(self, pois: list[ScoredPOI], keywords: list[str]) -> list[ScoredPOI]:
        """Score POIs based on keyword match and richness of data."""
        keyword_set = set(k.lower() for k in keywords)
        for poi in pois:
            score = 0.4
            text = f"{poi.name} {poi.description} {' '.join(poi.tags)} {poi.highlights or ''} {poi.area or ''}".lower()
            matches = sum(1 for kw in keyword_set if kw in text)
            score += min(matches * 0.12, 0.4)

            # Bonus for rich data
            if poi.recommended_hours:
                score += 0.05
            if poi.area:
                score += 0.03
            if poi.best_time:
                score += 0.03
            if poi.indoor_outdoor:
                score += 0.02
            if poi.ticket_price is not None:
                score += 0.02

            poi.score = min(score, 1.0)
        return pois
