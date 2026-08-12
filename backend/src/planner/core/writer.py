"""Phase 2C Writer — enrich itinerary with LLM-generated prose.

Phase 2C uses a day-batch enrichment strategy: one LLM call generates the day
theme and all activity recommendation reasons for that day.  This is cheaper
than per-activity calls because the shared context (system prompt, user
profile) is paid for once per day.

If the day-batch call fails or returns invalid data, the pipeline falls back to
per-activity LLM enrichment (with per-activity validation and retry).  If a
single activity still cannot be enriched, a template fallback is used for that
activity only — other activities keep their enriched prose.

Protected fields (poi_name, start_time, end_time, duration_min, ticket_price,
location lat/lng) can never be mutated by enrichment.
"""

import asyncio
import logging
from copy import deepcopy
from typing import Any, Optional

from core.langsmith_trace import traceable_step
from core.llm_client import llm
from schemas import Activity, DayPlan, UserProfile
from planner.core.fact_guard import activity_fields_match, protected_field_differences

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Template fallback (single-activity, formerly _REASON_TEMPLATES)
# --------------------------------------------------------------------------- #

_REASON_TEMPLATES: dict[str, str] = {
    # 北京景点
    "故宫": "世界文化遗产，明清两代皇宫，中华文明的象征",
    "天坛": "明清代帝王祭天场所，建筑学杰作",
    "颐和园": "中国现存最大的皇家园林，昆明湖与万寿山相映成趣",
    "长城": "世界七大奇迹之一，中华民族的脊梁",
    "南锣鼓巷": "老北京胡同文化代表，美食与文艺小店林立",
    "798艺术区": "工业遗址改造的当代艺术圣地",
    "鸟巢": "2008年奥运会主体育场，现代建筑奇迹",
    "雍和宫": "北京最大的藏传佛教寺院，香火鼎盛",
    "圆明园": "清代皇家园林遗址，历史沧桑与湖光山色交织",
    "恭王府": "清代规模最大的一座王府，半部清代史尽在其中",
    "北海公园": "中国现存最古老最完整的皇家园林之一",
    "什刹海": "老北京水乡风貌，酒吧与胡同文化交融的休闲胜地",
    "前门大街": "北京中轴线上的老字号商业街，京味文化聚集地",
    # 上海景点
    "外滩": "万国建筑博览群，上海最经典的城市名片",
    "东方明珠": "上海地标性建筑，俯瞰浦江两岸的绝佳位置",
    "豫园": "明代江南园林代表作，感受老上海的风雅",
    "南京路步行街": "中华商业第一街，购物者的天堂",
    "田子坊": "石库门里弄里的创意艺术区，适合闲逛拍照",
    "新天地": "石库门与现代时尚的完美融合",
    "上海博物馆": "馆藏丰富的综合性博物馆，历史爱好者必去",
    "上海迪士尼": "中国内地首座迪士尼主题乐园，亲子与童话梦想之地",
    "朱家角古镇": "沪上水乡古镇，小桥流水人家的江南韵味",
    "上海中心大厦": "中国第一高楼，云端俯瞰魔都天际线",
    "城隍庙": "上海老城厢文化地标，小吃与民俗风情汇集",
    # 杭州景点
    "西湖": "中国十大风景名胜之一，湖光山色与人文古迹交相辉映",
    "灵隐寺": "千年古刹，江南禅宗五山之一，飞来峰石刻艺术瑰宝",
    "雷峰塔": "西湖十景之一雷峰夕照，登塔俯瞰西湖全景",
    "三潭印月": "西湖第一胜境，人民币一元纸币背面图案",
    "断桥残雪": "西湖十景之首，白娘子与许仙相遇之地",
    "龙井村": "中国十大名茶之首龙井茶原产地，茶园梯田美不胜收",
    "西溪湿地": "中国首个国家湿地公园，城市中的天然氧吧",
    "岳王庙": "纪念民族英雄岳飞，精忠报国的历史见证",
    "九溪烟树": "西湖新十景，溪水潺潺古道清幽",
    "千岛湖": "天下第一秀水，千岛错落湖光山色如诗如画",
    "宋城": "大型宋代文化主题公园，千古情演出震撼人心",
    "河坊街": "杭州历史商业街区，老字号与江南小吃云集",
    # 南京景点
    "中山陵": "中国近代建筑史上第一陵，392级台阶象征三民主义与五权宪法",
    "夫子庙": "中国四大文庙之一，秦淮河畔千年儒学圣地，夜游灯会最是迷人",
    "秦淮河": "六朝金粉地，十里秦淮河，夜游画舫最是南京风情",
    "老门东": "明清老城南风貌区，青砖黛瓦间藏着南京最地道的市井烟火",
    "南京大牌档": "南京餐饮名片，一站式品尝盐水鸭、鸭血粉丝等地道金陵美食",
    "盐水鸭": "金陵名菜之首，皮白肉嫩肥而不腻，有六朝风味白门佳品之誉",
    "鸭血粉丝汤": "南京街头灵魂小吃，鸭血嫩滑粉丝爽口，一碗暖到心底",
    "小笼包": "皮薄汤鲜肉馅饱满，轻轻提慢慢移，先开窗后喝汤",
    "明孝陵": "明清皇家第一陵，世界文化遗产，石象路秋色尤为动人",
    "总统府": "中国近代史遗址博物馆，见证民国风云变幻",
    "鸡鸣寺": "南朝第一寺，春日樱花大道堪称南京最美风景",
    "玄武湖": "江南三大名湖之一，皇家园林湖泊，城市中心的绿洲",
    "牛首山": "佛顶骨舍利供奉地，现代佛宫艺术与山水园林相得益彰",
    # 成都景点
    "宽窄巷子": "成都三大历史文化保护区之一，青砖黛瓦间品地道川西民居风情",
    "锦里": "西蜀第一街，三国文化主题商业街，夜游灯火璀璨最是巴蜀韵味",
    "武侯祠": "中国唯一君臣合祀祠庙，三国迷必访的蜀汉文化圣地",
    "大熊猫基地": "全球最大的大熊猫繁育研究基地，近距离观察国宝的憨态可掬",
    "杜甫草堂": "诗圣杜甫流寓成都时的故居，中国文学史上的圣地",
    "春熙路": "成都最繁华的商业中心，潮流购物与地道美食并存",
    "太古里": "开放式低密街区，传统川西建筑与现代时尚完美融合",
    "青城山": "道教发源地之一，天下幽之誉，避暑修仙的清凉胜地",
    "都江堰": "世界文化遗产，两千年前的水利工程至今造福天府之国",
    "文殊院": "成都市中心千年古刹，香火鼎盛的佛教圣地",
    "人民公园": "成都慢生活缩影，鹤鸣茶社里喝一碗盖碗茶",
    "九眼桥": "成都夜生活地标，锦江两岸灯火与酒吧街相映成趣",
    # 西安景点
    "兵马俑": "世界第八大奇迹，秦始皇陵陪葬坑，两千年前的军团依然震撼人心",
    "大雁塔": "唐代玄奘法师为保存佛经而建，西安的文化地标",
    "回民街": "西安著名美食街，肉夹馍羊肉泡馍biangbiang面一站吃遍",
    "城墙": "中国现存规模最大保存最完整的古代城垣，骑行一圈穿越千年",
    "钟楼": "西安城市中心，晨钟暮鼓六百年，登楼俯瞰古城四方",
    "大唐不夜城": "盛唐文化主题步行街，灯火璀璨梦回长安",
    "大唐芙蓉园": "中国第一个全方位展示盛唐风貌的大型皇家园林式文化主题公园",
    "华清宫": "唐玄宗与杨贵妃爱情故事发生地，温泉文化与历史遗迹交融",
    "陕西历史博物馆": "华夏珍宝库，周秦汉唐文明的一站式沉浸体验",
    "小雁塔": "唐代佛教建筑艺术瑰宝，晨钟暮鼓位列关中八景",
    "永兴坊": "陕西非遗美食文化街区，摔碗酒与百余种小吃汇聚",
    "碑林博物馆": "中国书法艺术宝库，历代名碑石刻荟萃",
    # 厦门景点
    "鼓浪屿": "海上花园，万国建筑博览，厦门最经典的城市名片",
    "南普陀寺": "闽南佛教圣地，背山面海，千年古刹",
    "厦门大学": "中国最美大学之一，芙蓉湖畔书香与海景交融",
    "环岛路": "黄金海岸线，骑行观海，厦门最浪漫的滨海大道",
    "曾厝垵": "文艺渔村蜕变的小清新聚落，适合闲逛拍照",
    "中山路步行街": "厦门最繁华的商业街，骑楼建筑与闽南风情交织",
    "胡里山炮台": "现存世界最大古炮，见证近代海防历史",
    "沙茶面": "厦门特色面食，沙茶酱香浓，一碗地道闽南味",
    "集美学村": "嘉庚精神发源地，龙舟池畔闽南建筑与学府气息交融",
    "厦门园林植物园": "热带雨林区与多肉植物区，网红打卡与自然科普兼具",
}


_MEAL_LABELS = {"午餐", "晚餐", "早餐", "下午茶", "夜宵", "用餐", "早午餐"}


def _normalize_poi_name(name: str) -> str:
    """Normalize a POI name for batch-result matching.

    The day-batch LLM is asked to echo poi_name verbatim, but in practice it
    often drops the meal label (returns "南翔馒头店" for "午餐 · 南翔馒头店") or
    tweaks spacing. Normalizing both sides lets the batch result map back to the
    activity instead of triggering an expensive per-activity LLM fallback.
    """
    n = str(name).strip()
    for sep in (" · ", "·", " - ", "·"):
        if sep in n:
            label, _, rest = n.partition(sep)
            if label.strip() in _MEAL_LABELS and rest.strip():
                n = rest.strip()
                break
    return n.replace(" ", "").replace("\u3000", "")


def _template_reason(poi_name: str, category: str, tags: list[str] | None = None) -> str:
    """Single-activity fallback when LLM enrichment fails."""
    if poi_name in _REASON_TEMPLATES:
        return _REASON_TEMPLATES[poi_name]
    if category == "restaurant":
        cuisine = "、".join(tags[:2]) if tags else ""
        # poi_name may carry a concrete venue as "午餐 · 老吉士酒家".
        if " · " in poi_name:
            label, venue = poi_name.split(" · ", 1)
            taste = f"{cuisine}风味" if cuisine else "地道风味"
            return f"{label}就近选在{venue}，品尝{taste}，补充体力继续行程"
        meal_label = poi_name if poi_name in ("午餐", "晚餐") else "用餐"
        return f"{meal_label}时段，就近寻一家口碑餐厅，品味当地{cuisine or '特色'}美食"
    if category == "attraction":
        tag_part = "、".join(tags[:2]) if tags else "探索"
        return f"{poi_name}，适合{tag_part}的好去处"
    return f"体验{poi_name}的当地特色"


def _template_theme(day_activities: list[Activity]) -> str:
    """Rule-based day theme when LLM is unavailable."""
    tags: set[str] = set()
    for a in day_activities:
        for t in a.tags or []:
            tags.add(t)

    if "历史" in tags and "文化" in tags:
        return "历史文化之旅"
    if "园林" in tags or "湖景" in tags:
        return "园林湖景休闲"
    if "登山" in tags:
        return "户外探索"
    if "文艺" in tags or "艺术" in tags:
        return "文艺漫游"
    if "美食" in tags:
        return "美食寻味"
    if "夜景" in tags:
        return "都市夜色"
    if "购物" in tags:
        return "购物休闲"
    return "精彩探索"


_GENERIC_DRAFT = {"推荐游览", "值得一游", "口碑推荐", "推荐去看看", "推荐"}


def _prefill_day(day: DayPlan) -> set[str]:
    """Prefill template/draft reasons in place; return prefilled poi_names.

    Known POIs use their curated template; a draft that already carries a
    non-generic reason is kept. Both skip the LLM entirely.
    """
    prefilled: set[str] = set()
    for act in day.activities:
        reason = (act.recommendation_reason or "").strip()
        if act.poi_name in _REASON_TEMPLATES:
            act.recommendation_reason = _REASON_TEMPLATES[act.poi_name]
            prefilled.add(act.poi_name)
        elif reason and not any(g in reason for g in _GENERIC_DRAFT) and len(reason) >= 4:
            prefilled.add(act.poi_name)
    if prefilled:
        day.has_prefilled = True
    return prefilled


# --------------------------------------------------------------------------- #
# All-days single-call enrichment (primary path)
# --------------------------------------------------------------------------- #

_ALL_DAYS_TIMEOUT = 45.0  # seconds — one call covers the whole trip

_BUILD_ALL_DAYS_PROMPT = """请为以下多天行程生成每天的主题和每个景点的推荐语。

用户画像：{travelers_type}，兴趣是{interests}，节奏偏好{pace}

行程（共{n_days}天）：
{days_block}

要求：
1. 每天 theme：4-8个字的中文主题，概括当天行程特色
2. 每个景点必须原样返回 poi_name，并给出 recommendation_reason（20-40字推荐语）和 tags（2-3个标签）
3. 推荐语根据用户画像个性化——亲子游强调"适合带孩子"，情侣游强调"浪漫"，历史爱好者强调"文化底蕴"
4. 仅输出 JSON，不要其他内容

返回 JSON 格式：
{{"days": [
  {{"day_number": 1, "theme": "...", "activities": [
    {{"poi_name": "...", "recommendation_reason": "...", "tags": ["...", "..."]}}
  ]}}
]}}"""


@traceable_step("planning/writer_llm_all_days", run_type="llm")
async def _llm_enrich_all_days(
    itinerary: list[DayPlan],
    profile: UserProfile,
    skip_map: dict[int, set[str]],
) -> Optional[dict]:
    """One LLM call for the whole trip (theme + reasons for every day).

    The total prose for a trip is small, so a single round-trip avoids the
    per-day call overhead (and any cloud concurrency limits). Returns the parsed
    JSON dict, or None on failure so the caller can fall back to per-day batches.
    """
    interests = "、".join(profile.interests) if profile.interests else "无特殊偏好"
    travelers_type = profile.travelers_type or "普通游客"
    pace = profile.pace or "适中"

    day_blocks: list[str] = []
    total = 0
    for day in itinerary:
        skip = skip_map.get(day.day_number, set())
        lines: list[str] = []
        for act in day.activities:
            if act.poi_name in skip:
                continue
            time_str = (
                f" {act.start_time}-{act.end_time}" if act.start_time and act.end_time else ""
            )
            lines.append(f"- {time_str} {act.poi_name} [{act.category}]")
            total += 1
        if lines:
            day_blocks.append(f"第{day.day_number}天：\n" + "\n".join(lines))

    if total == 0:
        return None

    prompt = _BUILD_ALL_DAYS_PROMPT.format(
        travelers_type=travelers_type,
        interests=interests,
        pace=pace,
        n_days=len(itinerary),
        days_block="\n\n".join(day_blocks),
    )
    # Scale output budget with activity count (reasons are short, ~25 tokens each).
    max_tokens = min(4096, 512 + 96 * total)

    try:
        result = await asyncio.wait_for(
            llm.json_chat(
                messages=[
                    {"role": "system", "content": "你是一个专业旅行文案写手，只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
                task_type="output_format",
            ),
            timeout=_ALL_DAYS_TIMEOUT,
        )
        return result if isinstance(result, dict) else None
    except asyncio.TimeoutError:
        logger.warning("LLM all-days enrichment timeout (%d activities)", total)
        return None
    except Exception as exc:
        logger.warning("LLM all-days enrichment failed: %s", exc)
        return None


@traceable_step("planning/writer_all_days", run_type="chain")
async def _enrich_all_days(
    itinerary: list[DayPlan],
    profile: UserProfile,
    prefill_map: dict[int, set[str]],
) -> bool:
    """Enrich every day from a single LLM call. Returns False to fall back."""
    result = await _llm_enrich_all_days(itinerary, profile, prefill_map)
    if not result:
        return False

    raw_days = result.get("days")
    if not isinstance(raw_days, list) or not raw_days:
        return False

    by_num: dict[int, dict] = {}
    for d in raw_days:
        if isinstance(d, dict) and d.get("day_number") is not None:
            try:
                by_num[int(d["day_number"])] = d
            except (TypeError, ValueError):
                continue
    if not by_num:
        return False

    # Only mutate after we know the result is usable, so a False return leaves the
    # days untouched for a clean per-day fallback.
    for day in itinerary:
        d = by_num.get(day.day_number, {})
        _apply_day_result(
            day,
            d.get("theme"),
            d.get("activities", []),
            prefill_map.get(day.day_number, set()),
        )
    return True


# --------------------------------------------------------------------------- #
# Day-batch LLM enrichment
# --------------------------------------------------------------------------- #

_BATCH_ENRICHMENT_TIMEOUT = 30.0  # seconds — batch output is longer

_BUILD_DAY_ENRICHMENT_PROMPT = """请为以下一日行程生成主题和每个景点的推荐语。

用户画像：{travelers_type}，兴趣是{interests}，节奏偏好{pace}

一日行程：
{activities}

要求：
1. theme: 4-8个字的中文主题名，能概括当天行程特色
2. 每个景点必须原样返回 poi_name，并给出 recommendation_reason（20-40字推荐语）和 tags（2-3个标签）
3. 推荐语根据用户画像个性化——比如亲子游强调"适合带孩子"，情侣游强调"浪漫"，历史爱好者强调"文化底蕴"
4. 仅输出 JSON，不要其他内容

返回 JSON 格式：
{{
  "theme": "...",
  "activities": [
    {{"poi_name": "...", "recommendation_reason": "...", "tags": ["...", "..."]}},
    {{"poi_name": "...", "recommendation_reason": "...", "tags": ["...", "..."]}}
  ]
}}"""


@traceable_step("planning/writer_llm_day_batch", run_type="llm")
async def _llm_enrich_day_batch(
    day: DayPlan,
    profile: UserProfile,
    skip_names: Optional[set[str]] = None,
) -> Optional[dict]:
    """Call LLM to enrich a whole day (theme + all activities). Returns parsed JSON dict or None."""
    skip_names = skip_names or set()
    if not day.activities:
        return None

    # If all activities are already prefilled, skip the LLM call entirely.
    if all(a.poi_name in skip_names for a in day.activities):
        return None

    interests = "、".join(profile.interests) if profile.interests else "无特殊偏好"
    travelers_type = profile.travelers_type or "普通游客"
    pace = profile.pace or "适中"

    activity_lines = []
    for act in day.activities:
        if act.poi_name in skip_names:
            continue
        time_str = f" {act.start_time}-{act.end_time}" if act.start_time and act.end_time else ""
        activity_lines.append(f"- {time_str} {act.poi_name} [{act.category}]")

    prompt = _BUILD_DAY_ENRICHMENT_PROMPT.format(
        travelers_type=travelers_type,
        interests=interests,
        pace=pace,
        activities="\n".join(activity_lines),
    )

    try:
        result = await asyncio.wait_for(
            llm.json_chat(
                messages=[
                    {"role": "system", "content": "你是一个专业旅行文案写手，只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1024,
                task_type="output_format",
            ),
            timeout=_BATCH_ENRICHMENT_TIMEOUT,
        )
        if not isinstance(result, dict):
            return None
        return result
    except asyncio.TimeoutError:
        logger.warning("LLM day-batch enrichment timeout for day %d", day.day_number)
        return None
    except Exception as exc:
        logger.warning("LLM day-batch enrichment failed for day %d: %s", day.day_number, exc)
        return None


def _find_category(poi_name: str, day: DayPlan) -> str:
    for a in day.activities:
        if a.poi_name == poi_name:
            return a.category
    return "attraction"


def _find_tags(poi_name: str, day: DayPlan) -> list[str]:
    for a in day.activities:
        if a.poi_name == poi_name:
            return a.tags or []
    return []


GENERIC_PATTERNS = ["推荐游览", "值得一游", "口碑推荐", "推荐", "不错的地方"]


def _apply_day_result(
    day: DayPlan,
    theme: Any,
    raw_items: Any,
    skip_names: Optional[set[str]] = None,
) -> None:
    """Apply an LLM theme + activity items to a day in place.

    Shared by the all-days and per-day batch paths. Matching is name-normalized
    (meal labels / spacing tolerated); uncovered activities, empty reasons and
    Fact-Guard violations fall back to a rule-based template (never an extra LLM
    round-trip). Protected fields are guaranteed unchanged.
    """
    skip_names = skip_names or set()

    theme_str = str(theme or "").strip()
    if theme_str and 2 <= len(theme_str) <= 12:
        day.theme = theme_str

    if not isinstance(raw_items, list):
        raw_items = []

    # P3: quality guard — replace generic LLM output with template fallback
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        reason = str(item.get("recommendation_reason", "")).strip()
        if any(p in reason for p in GENERIC_PATTERNS):
            poi_name = str(item.get("poi_name", "")).strip()
            item["recommendation_reason"] = _template_reason(
                poi_name,
                _find_category(poi_name, day),
                _find_tags(poi_name, day),
            )

    # Index results by exact and normalized name so meal labels / spacing tweaks
    # from the LLM still resolve to the right activity.
    result_by_name: dict[str, dict] = {}
    for item in raw_items:
        if isinstance(item, dict) and item.get("poi_name"):
            raw_name = str(item["poi_name"]).strip()
            result_by_name[raw_name] = item
            result_by_name.setdefault(_normalize_poi_name(raw_name), item)

    def _lookup(name: str) -> Optional[dict]:
        return result_by_name.get(name) or result_by_name.get(_normalize_poi_name(name))

    new_activities: list[Activity] = []

    for activity in day.activities:
        if activity.poi_name in skip_names:
            new_activities.append(activity)
            continue

        item = _lookup(activity.poi_name)
        reason = str(item.get("recommendation_reason", "")).strip() if item else ""
        raw_tags = item.get("tags", []) if item else []
        tags = list(raw_tags) if isinstance(raw_tags, list) else []

        candidate = deepcopy(activity)
        if reason:
            candidate.recommendation_reason = reason
            candidate.tags = list(set((activity.tags or []) + tags))
            # Fact Guard: LLM must not mutate protected fields. On violation, fall
            # back to a rule-based template rather than another LLM round-trip.
            if not activity_fields_match(activity, candidate):
                changed_fields = protected_field_differences(activity, candidate)
                logger.warning(
                    "Batch Fact Guard failed for %s fields=%s — using template",
                    activity.poi_name,
                    ",".join(changed_fields) or "unknown",
                )
                candidate = deepcopy(activity)
                candidate.recommendation_reason = _template_reason(
                    activity.poi_name, activity.category, activity.tags
                )
        else:
            # Not covered by the batch (or empty reason) → template, no extra LLM.
            candidate.recommendation_reason = _template_reason(
                activity.poi_name, activity.category, activity.tags
            )

        new_activities.append(candidate)

    if new_activities:
        day.activities = new_activities


@traceable_step("planning/writer_day_batch", run_type="chain")
async def _enrich_day_batch(
    day: DayPlan,
    profile: UserProfile,
    skip_names: Optional[set[str]] = None,
) -> bool:
    """Try to enrich an entire day with one LLM call.

    On success, mutates ``day`` in place and returns True.
    On failure (the batch LLM call itself failed), leaves ``day`` unchanged and
    returns False so the caller can fall back to per-activity enrichment.
    """
    skip_names = skip_names or set()
    batch_result = await _llm_enrich_day_batch(day, profile, skip_names=skip_names)
    if not batch_result:
        return False

    _apply_day_result(
        day,
        batch_result.get("theme"),
        batch_result.get("activities", []),
        skip_names,
    )
    # The batch call (+ template fallback for stragglers) fully handled the day;
    # returning True prevents a redundant second per-activity LLM pass.
    return True


# --------------------------------------------------------------------------- #
# Per-activity LLM enrichment (fallback)
# --------------------------------------------------------------------------- #

_ENRICHMENT_TIMEOUT = 10.0  # seconds — single-activity fallback

_BUILD_ENRICHMENT_PROMPT = """请为以下景点写一句简短的中文推荐语。

景点：{poi_name}
类型：{category}
用户画像：{travelers_type}，兴趣是{interests}，节奏偏好{pace}

要求：
1. 仅输出一句推荐语（20-40字），不提价格或具体时间
2. 根据用户画像个性化——比如亲子游强调"适合带孩子"，情侣游强调"浪漫"
3. 如果该景点适合特定时间段（如夜景、早茶），可在推荐语中自然提及

返回 JSON 格式：
{{"recommendation_reason": "...", "tags": ["标签1", "标签2"]}}"""


async def _llm_enrich_activity(
    activity: Activity,
    profile: UserProfile,
) -> Optional[tuple[str, list[str]]]:
    """Call LLM to enrich a single activity.  Returns (reason, tags) or None on failure."""
    interests = "、".join(profile.interests) if profile.interests else "无特殊偏好"
    travelers_type = profile.travelers_type or "普通游客"
    pace = profile.pace or "适中"

    prompt = _BUILD_ENRICHMENT_PROMPT.format(
        poi_name=activity.poi_name,
        category=activity.category,
        travelers_type=travelers_type,
        interests=interests,
        pace=pace,
    )

    try:
        result = await asyncio.wait_for(
            llm.json_chat(
                messages=[
                    {"role": "system", "content": "你是一个专业旅行文案写手，只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=256,
                task_type="output_format",
            ),
            timeout=_ENRICHMENT_TIMEOUT,
        )
        reason = str(result.get("recommendation_reason", "")).strip()
        tags = list(result.get("tags", [])) if isinstance(result.get("tags"), list) else []
        if not reason:
            return None
        return reason, tags
    except asyncio.TimeoutError:
        logger.warning("LLM enrichment timeout for %s", activity.poi_name)
        return None
    except Exception as exc:
        logger.warning("LLM enrichment failed for %s: %s", activity.poi_name, exc)
        return None


@traceable_step("planning/writer_activity_enrich", run_type="chain")
async def _enrich_activity_with_retry(
    activity: Activity,
    profile: UserProfile,
    max_retries: int = 2,
) -> Activity:
    """Enrich a single activity with LLM, validate, retry on mutation, fallback."""
    # LLM not available or enrichment skipped — use template
    enriched = deepcopy(activity)

    for attempt in range(max_retries + 1):
        llm_result = await _llm_enrich_activity(activity, profile)
        if llm_result is None:
            # LLM call failed — try again or fallback
            if attempt < max_retries:
                continue
            enriched.recommendation_reason = _template_reason(
                activity.poi_name, activity.category, activity.tags
            )
            return enriched

        reason, tags = llm_result
        enriched.recommendation_reason = reason
        enriched.tags = list(set((activity.tags or []) + tags))

        # Validate: LLM must not mutate protected fields
        if activity_fields_match(activity, enriched):
            return enriched

        changed_fields = protected_field_differences(activity, enriched)
        logger.warning(
            "Fact Guard failed for %s fields=%s (attempt %d/%d)",
            activity.poi_name,
            ",".join(changed_fields) or "unknown",
            attempt + 1,
            max_retries + 1,
        )
        # Reset for retry
        enriched = deepcopy(activity)

    # All retries exhausted — single-activity template fallback
    enriched.recommendation_reason = _template_reason(
        activity.poi_name, activity.category, activity.tags
    )
    return enriched


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


@traceable_step("planning/writer_enrich", run_type="chain")
async def enrich(
    itinerary: list[DayPlan],
    profile: UserProfile,
) -> tuple[list[DayPlan], str]:
    """Enrich itinerary with LLM-generated themes and recommendation reasons.

    Strategy:
    1. Prefill known POIs with high-quality templates, skipping the LLM call
       for those activities entirely.
    2. Try to enrich each day with a single LLM call (theme + all activities).
       This is cheaper because the shared context is paid for once per day.
    3. If the day-batch call fails for a day, fall back to per-activity LLM
       enrichment with per-activity validation and retry.
    4. If a single activity still cannot be enriched, use a template fallback
       for that activity only.

    Returns (enriched_itinerary, proposal_text).  If the entire enrichment
    process fails, returns the original itinerary with a fallback proposal.
    """
    try:
        enriched = deepcopy(itinerary)
        if not enriched:
            return enriched, _build_proposal(enriched, profile)

        # P0: prefill known/draft POIs (no LLM) and record what each day still needs.
        prefill_map = {day.day_number: _prefill_day(day) for day in enriched}
        pending = [
            d for d in enriched if len(prefill_map.get(d.day_number, set())) < len(d.activities)
        ]

        if pending:
            # Primary: ONE LLM call for the whole trip. The prose is small, so a
            # single round-trip beats N per-day calls (and dodges cloud
            # concurrency limits). Fall back to per-day batches only if it fails.
            all_ok = await _enrich_all_days(pending, profile, prefill_map)
            if not all_ok:
                await asyncio.gather(*(_enrich_one_day(day, profile) for day in pending))

        # Fully-prefilled days still need a rule-based theme.
        for d in enriched:
            if not d.theme:
                d.theme = _template_theme(d.activities)

        proposal = _build_proposal(enriched, profile)
        return enriched, proposal

    except Exception:
        logger.exception("Enrichment failed — returning original itinerary")
        return itinerary, _fallback_proposal(itinerary, profile)


@traceable_step("planning/writer_enrich_day", run_type="chain")
async def _enrich_one_day(day: DayPlan, profile: UserProfile) -> None:
    """Enrich a single day in place (per-day fallback when the all-days call fails)."""
    prefilled = _prefill_day(day)

    # If every activity is prefilled, just generate a rule-based theme.
    if len(prefilled) == len(day.activities):
        if not day.theme:
            day.theme = _template_theme(day.activities)
        return

    # One LLM call per day (only for non-prefilled activities).
    batch_ok = await _enrich_day_batch(day, profile, skip_names=prefilled)

    if not batch_ok:
        # Fallback path: per-activity LLM (parallel) + rule-based theme
        if not day.theme:
            day.theme = _template_theme(day.activities)
        todo = [i for i, a in enumerate(day.activities) if a.poi_name not in prefilled]
        results = await asyncio.gather(
            *(_enrich_activity_with_retry(day.activities[i], profile) for i in todo)
        )
        for i, enriched_act in zip(todo, results):
            day.activities[i] = enriched_act


# --------------------------------------------------------------------------- #
# Proposal text
# --------------------------------------------------------------------------- #


def _build_proposal(itinerary: list[DayPlan], profile: UserProfile) -> str:
    """Build a rich markdown proposal from the enriched itinerary."""
    lines: list[str] = []
    dest = profile.destination or "目的地"
    days = profile.travel_days or len(itinerary)

    # Header
    lines.append(f"# {dest} {days}日游行程方案\n")

    if profile.travel_dates:
        lines.append(f"**出行日期**: {profile.travel_dates}")
    if profile.travelers_type:
        lines.append(f"**出行类型**: {profile.travelers_type}")
    if profile.interests:
        lines.append(f"**兴趣偏好**: {'、'.join(profile.interests)}")
    lines.append("")

    # Budget — total_cost per day already includes tickets + food + city transit.
    # Label the scope so the figure is not mistaken for an all-in trip cost.
    total_cost = sum(day.total_cost for day in itinerary)
    lines.append(f"**预估费用（门票+餐饮+市内交通）**: ¥{total_cost:.0f}")
    lines.append("> 不含住宿与往返大交通，为模拟参考价")
    if profile.budget_range:
        ratio = total_cost / profile.budget_range
        if ratio <= 1.0:
            lines.append(f"✅ 在预算 ¥{profile.budget_range:.0f} 之内")
        elif ratio <= 1.2:
            lines.append(f"⚠️ 略超预算 ¥{profile.budget_range:.0f}（+{ratio - 1:.0%}）")
        else:
            lines.append(f"❌ 超出预算 ¥{profile.budget_range:.0f} 的 20%")
    lines.append("")

    # Day-by-day
    for day in itinerary:
        lines.append(f"## 第{day.day_number}天" + (f" — {day.theme}" if day.theme else ""))
        if day.date:
            lines.append(f"**日期**: {day.date}")
        lines.append("")

        for i, act in enumerate(day.activities, 1):
            time_str = ""
            if act.start_time and act.end_time:
                time_str = f" ({act.start_time}-{act.end_time})"

            cost_str = ""
            if act.ticket_price:
                cost_str = f" — ¥{act.ticket_price:.0f}"
            elif act.meal_cost:
                cost_str = f" — ¥{act.meal_cost:.0f}"

            lines.append(f"{i}. **{act.poi_name}**{time_str}{cost_str}")
            if act.recommendation_reason:
                lines.append(f"   _{act.recommendation_reason}_")
        if day.budget_note:
            lines.append(day.budget_note)
        lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"*由 TravelAgent 自动生成 · {dest} {days}日行程*")
    lines.append("")

    return "\n".join(lines)


def _fallback_proposal(itinerary: list[DayPlan], profile: UserProfile) -> str:
    """Minimal proposal used when enrichment fails entirely."""
    dest = profile.destination or "目的地"
    days = profile.travel_days or len(itinerary)
    lines = [
        f"# {dest} {days}日游行程方案\n",
        f"**预估费用（门票+餐饮+市内交通）**: ¥{sum(d.total_cost for d in itinerary):.0f}\n",
    ]
    for day in itinerary:
        lines.append(f"## 第{day.day_number}天")
        for act in day.activities:
            t = f" ({act.start_time}-{act.end_time})" if act.start_time else ""
            lines.append(f"- **{act.poi_name}**{t}")
        lines.append("")
    return "\n".join(lines)
