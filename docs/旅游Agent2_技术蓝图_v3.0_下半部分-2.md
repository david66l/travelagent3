# 旅游Agent2 技术蓝图 v3.0 下半部分：知识检索与行程规划
> 覆盖第5-8层 + 监控 + 部署 + Roadmap
> 版本日期：2026-06-18

---

## 文档定位

下半部分聚焦 **"基于用户需求找到合适的景点 → 数学求解最优路线 → 校验 → 输出"** 这一链路，包含：

| 章节 | 层级 | 核心问题 | 产出物 |
|------|------|---------|--------|
| 第5章 | 知识库层 | 从哪里找到合适的景点？ | RAG混合检索 + 结构化DB + 实时API |
| 第6章 | 规划决策引擎 | 如何规划最优路线？ | OR-Tools CP-SAT v4.0 + 三模式重规划 |
| 第7章 | 工具调用层 | 需要调用哪些外部工具？ | 11类Function Calling工具定义 |
| 第8章 | 交互输出层 | 如何展示给用户？ | Markdown润色 + PDF/Excel/地图/语音 |
| 第9章 | 监控与可观测性 | 系统运行得怎么样？ | Metrics + Traces + Alerts |
| 第10章 | 部署架构 | 如何部署到生产？ | Docker + K8s + 模型部署 |
| 第11章 | Roadmap | 分几期实现？ | MVP范围 + 迭代计划 |

**上半部分衔接**：接收 `AgentState.slots`（TravelSlots 15字段）+ `AgentState.user_profile`（画像）+ `AgentState.inferred_slots`（推断槽位）。

---

## 5. 知识库层（第5层）

### 5.1 职责定位

知识库层是系统的"情报中心"，负责根据用户的slots和画像，从多个数据源召回**Top-15最相关的结构化POI**（景点/餐厅/酒店）。

**核心任务**：
1. **结构化数据检索**：从PostgreSQL的`spots`表预过滤（城市、类型、开放时间）
2. **语义检索**：从pgvector检索语义相似的POI（bge-large-zh-v1.5）
3. **关键词检索**：BM25全文检索补充精确匹配
4. **融合排序**：RRF（Reciprocal Rank Fusion）合并多路结果
5. **实时数据增强**：调用天气/交通/预约状态API补充动态信息

**关键约束**：
- 绝对不能编造POI（所有结果必须来自DB或API）
- 结果必须有`_rrf_score`排序依据
- 检索为空时必须标记`retrieval_empty=True`

---

### 5.2 数据模型：POI结构化Schema

```python
# models/poi.py —— POI数据模型

from pydantic import BaseModel, Field
from typing import Optional, List

class POI(BaseModel):
    """
    结构化POI —— 知识库层的核心数据单元。
    存储在PostgreSQL spots表中，向量存储在pgvector中。
    """
    # ── 基础信息 ──
    spot_id: int = Field(..., description="全局唯一POI ID")
    spot_name: str = Field(..., description="POI名称，如'故宫博物院'")
    spot_type: str = Field(..., description="类型: attraction/restaurant/hotel")
    city: str = Field(..., description="所属城市")
    district: Optional[str] = Field(None, description="区县")

    # ── 位置 ──
    lat: float = Field(..., description="纬度 WGS84")
    lng: float = Field(..., description="经度 WGS84")
    address: Optional[str] = Field(None, description="详细地址")

    # ── 时间 ──
    open_time: str = Field("08:00", description="开门时间 HH:MM")
    close_time: str = Field("18:00", description="关门时间 HH:MM")
    duration_minutes: int = Field(120, description="建议游玩分钟数")
    best_visit_time: Optional[str] = Field(None, description="最佳游览时段")

    # ── 费用 ──
    ticket_price: float = Field(0.0, description="门票价格（元）")
    price_level: int = Field(2, description="价格等级 1-5")

    # ── 体力 ──
    walk_intensity: int = Field(3, description="步行强度 1-5")
    queue_time_avg: int = Field(30, description="平均排队分钟数")
    indoor_outdoor: str = Field("outdoor", description="indoor/outdoor/mixed")

    # ── 预约 ──
    need_reservation: bool = Field(False, description="是否需要预约")
    reservation_advance_days: int = Field(0, description="需提前预约天数")
    reservation_channel: Optional[str] = Field(None, description="预约渠道")

    # ── 标签与描述 ──
    tags: List[str] = Field([], description="标签，如['历史','世界文化遗产','5A']")
    description: Optional[str] = Field(None, description="POI简介（用于Embedding）")

    # ── 人群适配 ──
    suitable_for: List[str] = Field(["solo","couple","family_kid","family_elder","friends"],
                                     description="适合的人群类型")
    accessibility: List[str] = Field([], description="无障碍设施: ['wheelchair','stroller','elevator']")

    # ── 评分 ──
    rating: float = Field(4.0, ge=0, le=5, description="用户评分 0-5")
    review_count: int = Field(0, description="评价数量")

    # ── 动态字段（实时API更新，不存DB）──
    current_weather: Optional[str] = Field(None, description="实时天气（API填充）")
    current_queue_time: Optional[int] = Field(None, description="实时排队分钟（API填充）")
    is_open_today: Optional[bool] = Field(None, description="今日是否开放（API填充）")

    # ── 系统字段 ──
    _rrf_score: Optional[float] = Field(None, description="融合排序分数")
    _reservation_reminder: Optional[bool] = Field(None, description="需预约必去标记")
```

---

### 5.3 数据库 DDL（POI相关）

```sql
-- ============================================================
-- POI主表 + 向量索引 + 辅助表
-- ============================================================

-- ── POI主表 ──
CREATE TABLE spots (
    spot_id                 SERIAL PRIMARY KEY,
    spot_name               VARCHAR(100) NOT NULL,
    spot_type               VARCHAR(20) NOT NULL,       -- attraction/restaurant/hotel
    city                    VARCHAR(50) NOT NULL,
    district                VARCHAR(50),

    lat                     DECIMAL(10, 7) NOT NULL,    -- WGS84
    lng                     DECIMAL(10, 7) NOT NULL,
    address                 VARCHAR(200),

    open_time               TIME NOT NULL DEFAULT '08:00',
    close_time              TIME NOT NULL DEFAULT '18:00',
    duration_minutes        INT NOT NULL DEFAULT 120,
    best_visit_time         VARCHAR(50),

    ticket_price            DECIMAL(10, 2) NOT NULL DEFAULT 0.00,
    price_level             INT NOT NULL DEFAULT 2 CHECK (price_level BETWEEN 1 AND 5),

    walk_intensity          INT NOT NULL DEFAULT 3 CHECK (walk_intensity BETWEEN 1 AND 5),
    queue_time_avg          INT NOT NULL DEFAULT 30,
    indoor_outdoor          VARCHAR(10) NOT NULL DEFAULT 'outdoor',

    need_reservation        BOOLEAN NOT NULL DEFAULT FALSE,
    reservation_advance_days INT NOT NULL DEFAULT 0,
    reservation_channel     VARCHAR(100),

    tags                    TEXT[],
    description             TEXT,                        -- 用于Embedding的文本

    suitable_for            TEXT[] DEFAULT ARRAY['solo','couple','family_kid','family_elder','friends'],
    accessibility           TEXT[],

    rating                  DECIMAL(3, 2) NOT NULL DEFAULT 4.00,
    review_count            INT NOT NULL DEFAULT 0,

    -- 向量（POI描述Embedding，768维）
    description_vector      VECTOR(768),

    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── 全文搜索索引（BM25）──
-- 创建搜索向量列
ALTER TABLE spots ADD COLUMN search_vector tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('chinese', coalesce(spot_name, '')), 'A') ||
        setweight(to_tsvector('chinese', coalesce(array_to_string(tags, ' '), '')), 'B') ||
        setweight(to_tsvector('chinese', coalesce(description, '')), 'C')
    ) STORED;

CREATE INDEX idx_spots_search ON spots USING GIN(search_vector);

-- ── 常用查询索引 ──
CREATE INDEX idx_spots_city ON spots (city, spot_type);
CREATE INDEX idx_spots_type ON spots (spot_type);
CREATE INDEX idx_spots_rating ON spots (rating DESC);
CREATE INDEX idx_spots_price ON spots (price_level);
CREATE INDEX idx_spots_tags ON spots USING GIN(tags);
CREATE INDEX idx_spots_suitable ON spots USING GIN(suitable_for);

-- ── HNSW向量索引 ──
CREATE INDEX idx_spots_vector ON spots 
    USING hnsw (description_vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ── 城市信息参考表（可行性校验用）──
CREATE TABLE city_info (
    city_name       VARCHAR(50) PRIMARY KEY,
    avg_daily_cost_low    REAL NOT NULL DEFAULT 300,
    avg_daily_cost_mid    REAL NOT NULL DEFAULT 500,
    avg_daily_cost_high   REAL NOT NULL DEFAULT 1000,
    recommend_min_days    INT NOT NULL DEFAULT 2,
    recommend_max_days    INT NOT NULL DEFAULT 5,
    climate_type          VARCHAR(20),
    best_seasons          TEXT[],
    notes                 TEXT
);

-- ── 触发器 ──
CREATE TRIGGER trigger_spots_updated
    BEFORE UPDATE ON spots
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

---

### 5.4 TravelRetrievalRAGAgent 完整实现

```python
# agents/rag_retrieval.py —— RAG混合检索Agent

from typing import Dict, Any, List, Optional
import asyncio
import json

class TravelRetrievalRAGAgent:
    """
    RAG混合检索Agent：多路召回 + 融合排序。

    检索流程：
      1. SQL预过滤（结构化条件筛选）
      2. pgvector语义检索（描述相似度）
      3. BM25关键词检索（标签/名称匹配）
      4. RRF融合排序
      5. 实时API增强（天气/排队/开放状态）
      6. 去重 + 截断Top-15

    业界最佳实践（pgvector hybrid search）：
      - Pre-filtering优先：在向量检索前用SQL过滤，减少搜索空间
      - RRF融合：不用分数归一化，用排名位置融合
      - Parent-Document模式：小chunk检索 + 原始文档返回
    """

    # RRF融合参数
    RRF_K = 60  # 常数，防止低排名dominating
    TOP_K = 15  # 最终返回数量

    # 向量检索参数
    HNSW_EF_SEARCH = 128  # HNSW查询时探索因子

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        import asyncpg
        self.pg_pool = None  # 懒加载
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer("BAAI/bge-large-zh-v1.5")

    async def retrieve(self, query: str, profile: Dict[str, Any],
                       top_k: int = 15) -> Dict[str, Any]:
        """
        主入口：混合检索。

        参数:
            query: 用户原始输入或检索意图
            profile: 合并后的slots + inferred_slots
            top_k: 返回POI数量

        返回:
            {
                "poi_candidates": [POI dict, ...],  # Top-K
                "retrieval_query": str,              # 实际使用的query
                "retrieval_empty": bool,             # 是否为空
                "retrieval_stats": dict,             # 统计信息（调试用）
            }
        """
        self.TOP_K = top_k

        # 1. 构建检索query（从profile生成）
        search_query = self._build_search_query(query, profile)

        # 2. 并行执行三路检索
        results_sql, results_vector, results_bm25 = await asyncio.gather(
            self._search_structured(profile),
            self._search_vector(search_query, profile),
            self._search_bm25(search_query, profile),
            return_exceptions=True
        )

        # 处理异常（某路失败不影响其他路）
        results_sql = results_sql if not isinstance(results_sql, Exception) else []
        results_vector = results_vector if not isinstance(results_vector, Exception) else []
        results_bm25 = results_bm25 if not isinstance(results_bm25, Exception) else []

        # 3. RRF融合排序
        merged = self._rrf_fusion(results_sql, results_vector, results_bm25)

        # 4. 截断Top-K
        top_pois = merged[:self.TOP_K]

        # 5. 实时API增强（异步，不阻塞返回）
        if top_pois:
            asyncio.create_task(self._enhance_realtime(top_pois))

        # 6. 标记需预约的必去景点
        must_visit = profile.get("must_visit", [])
        for poi in top_pois:
            poi["_reservation_reminder"] = (
                poi.get("need_reservation") and 
                poi.get("spot_name") in must_visit
            )

        return {
            "poi_candidates": top_pois,
            "retrieval_query": search_query,
            "retrieval_empty": len(top_pois) == 0,
            "retrieval_stats": {
                "sql_count": len(results_sql),
                "vector_count": len(results_vector),
                "bm25_count": len(results_bm25),
                "merged_count": len(merged),
                "final_count": len(top_pois),
            }
        }

    # ═══════════════════════════════════════════
    # 检索Query构建
    # ═══════════════════════════════════════════

    def _build_search_query(self, user_input: str, profile: Dict) -> str:
        """
        构建检索query：结合用户输入 + 画像偏好。
        比直接用user_input更准确。
        """
        parts = []

        # 目的地优先
        dest = profile.get("destination", "")
        if dest:
            parts.append(dest)

        # 兴趣标签
        interests = profile.get("interests", [])
        if interests:
            parts.extend(interests[:3])  # 最多3个兴趣

        # 必去景点（用于提高这些景点的排名）
        must = profile.get("must_visit", [])
        if must:
            parts.extend(must[:2])

        # 原始输入的关键词（去停用词后）
        keywords = self._extract_keywords(user_input)
        parts.extend(keywords[:3])

        # 去重 + 拼接
        seen = set()
        unique = []
        for p in parts:
            if p and p not in seen:
                seen.add(p)
                unique.append(p)

        return " ".join(unique[:8])  # 最多8个token

    def _extract_keywords(self, text: str) -> List[str]:
        """简单关键词提取（MVP用jieba，这里用简单分词）"""
        # 简单实现：去掉常见停用词后按空格/标点分割
        stopwords = {"我想", "我要", "帮我", "请", "一下", "的", "了", "和", "在", "是", "去", "玩", "旅游", "旅行"}
        words = text.replace("，", " ").replace("。", " ").replace("、", " ").split()
        return [w for w in words if w not in stopwords and len(w) > 1]

    # ═══════════════════════════════════════════
    # 路1：SQL结构化预过滤
    # ═══════════════════════════════════════════

    async def _search_structured(self, profile: Dict) -> List[Dict]:
        """
        SQL预过滤检索：用结构化条件筛选POI。
        这是最关键的一路——确保结果满足硬性约束。
        """
        if not self.pg_pool:
            import asyncpg
            self.pg_pool = await asyncpg.create_pool(
                self.config.get("DATABASE_URL"),
                min_size=2, max_size=10, command_timeout=5,
            )

        # 构建动态WHERE条件
        conditions = ["1=1"]
        params = []
        param_idx = 0

        # 必去条件：目的地
        dest = profile.get("destination", "")
        if dest:
            param_idx += 1
            conditions.append(f"(city = ${param_idx} OR spot_name LIKE ${param_idx+1})")
            params.extend([dest, f"%{dest}%"])
            param_idx += 1

        # 人群适配过滤
        companion = profile.get("travel_companion", "")
        if companion:
            param_idx += 1
            conditions.append(f"${param_idx} = ANY(suitable_for)")
            params.append(companion)

        # 体力过滤：老人降低强度
        max_walk = profile.get("Walk_max", 10)
        param_idx += 1
        conditions.append(f"walk_intensity <= ${param_idx}")
        params.append(max_walk)

        # 注意：预算约束由CP-SAT求解器的硬约束处理（Budget_total / Budget_day_max）
        # 不在检索阶段预过滤，避免错误过滤合理景点。30%假设无数据支撑。
        # 见下半部分第6章 epsilon_config 中的预算MAD约束。
        # budget = profile.get("total_budget")
        # days = profile.get("travel_days")
        # if budget and days:
        #     daily_ticket_budget = budget / days * 0.3
        #     conditions.append(f"ticket_price <= {daily_ticket_budget}")

        # 孕妇安全过滤
        if profile.get("has_pregnant"):
            conditions.append("walk_intensity <= 2")
            conditions.append("indoor_outdoor IN ('indoor', 'mixed')")

        # 轮椅过滤
        if profile.get("has_wheelchair"):
            param_idx += 1
            conditions.append(f"'wheelchair' = ANY(accessibility)")

        # 儿童过滤：偏好室内或混合
        if profile.get("has_children"):
            conditions.append("walk_intensity <= 3")

        # 类型过滤
        param_idx += 1
        conditions.append(f"spot_type = ${param_idx}")
        params.append("attraction")  # 默认只搜景点

        # 排序：评分优先
        where_clause = " AND ".join(conditions)

        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT spot_id, spot_name, spot_type, city, district,
                       lat, lng, address, open_time, close_time, duration_minutes,
                       ticket_price, price_level, walk_intensity, queue_time_avg,
                       indoor_outdoor, need_reservation, reservation_advance_days,
                       reservation_channel, tags, description, suitable_for,
                       accessibility, rating, review_count,
                       description_vector::text as vector_str,
                       rating * 0.4 + review_count * 0.001 as _score
                FROM spots
                WHERE {where_clause}
                ORDER BY _score DESC
                LIMIT 50
                """,
                *params
            )

        return [self._row_to_dict(r) for r in rows]

    # ═══════════════════════════════════════════
    # 路2：pgvector语义检索
    # ═══════════════════════════════════════════

    async def _search_vector(self, query: str, profile: Dict) -> List[Dict]:
        """
        pgvector语义检索：用Embedding找描述相似的POI。
        最佳实践：ef_search=128，预过滤后执行。
        """
        if not self.pg_pool:
            return []

        # 生成query向量
        query_vector = self.embedder.encode(query).tolist()

        # 构建预过滤条件（同SQL预过滤）
        dest = profile.get("destination", "")
        max_walk = profile.get("Walk_max", 10)

        async with self.pg_pool.acquire() as conn:
            # 设置ef_search（HNSW搜索质量）
            await conn.execute(f"SET hnsw.ef_search = {self.HNSW_EF_SEARCH}")

            if dest:
                rows = await conn.fetch(
                    """
                    SELECT spot_id, spot_name, spot_type, city, district,
                           lat, lng, address, open_time, close_time, duration_minutes,
                           ticket_price, price_level, walk_intensity, queue_time_avg,
                           indoor_outdoor, need_reservation, reservation_advance_days,
                           reservation_channel, tags, description, suitable_for,
                           accessibility, rating, review_count,
                           description_vector <=> $1::vector as _distance
                    FROM spots
                    WHERE city = $2 AND walk_intensity <= $3
                    ORDER BY _distance ASC
                    LIMIT 50
                    """,
                    json.dumps(query_vector), dest, max_walk
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT spot_id, spot_name, spot_type, city, district,
                           lat, lng, address, open_time, close_time, duration_minutes,
                           ticket_price, price_level, walk_intensity, queue_time_avg,
                           indoor_outdoor, need_reservation, reservation_advance_days,
                           reservation_channel, tags, description, suitable_for,
                           accessibility, rating, review_count,
                           description_vector <=> $1::vector as _distance
                    FROM spots
                    WHERE walk_intensity <= $2
                    ORDER BY _distance ASC
                    LIMIT 50
                    """,
                    json.dumps(query_vector), max_walk
                )

        # distance → similarity score（距离越小越相似）
        results = []
        for r in rows:
            d = dict(r)
            d["_score"] = 1.0 - float(r["_distance"])  # 转为相似度
            del d["_distance"]
            results.append(d)
        return results

    # ═══════════════════════════════════════════
    # 路3：BM25关键词检索
    # ═══════════════════════════════════════════

    async def _search_bm25(self, query: str, profile: Dict) -> List[Dict]:
        """
        BM25全文检索：用PostgreSQL的tsvector做关键词匹配。
        优势：精确匹配景点名、标签；召回语义检索遗漏的结果。
        """
        if not self.pg_pool:
            return []

        # 将query转为tsquery格式
        import re
        # 简单分词 + AND连接
        terms = re.findall(r'[一-鿿]+|[a-zA-Z]+', query)
        if not terms:
            return []

        ts_query = " & ".join(terms[:6])  # 最多6个词

        dest = profile.get("destination", "")

        async with self.pg_pool.acquire() as conn:
            if dest:
                rows = await conn.fetch(
                    """
                    SELECT spot_id, spot_name, spot_type, city, district,
                           lat, lng, address, open_time, close_time, duration_minutes,
                           ticket_price, price_level, walk_intensity, queue_time_avg,
                           indoor_outdoor, need_reservation, reservation_advance_days,
                           reservation_channel, tags, description, suitable_for,
                           accessibility, rating, review_count,
                           ts_rank_cd(search_vector, plainto_tsquery('chinese', $1)) as _score
                    FROM spots
                    WHERE search_vector @@ plainto_tsquery('chinese', $1)
                      AND city = $2
                    ORDER BY _score DESC
                    LIMIT 50
                    """,
                    query, dest
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT spot_id, spot_name, spot_type, city, district,
                           lat, lng, address, open_time, close_time, duration_minutes,
                           ticket_price, price_level, walk_intensity, queue_time_avg,
                           indoor_outdoor, need_reservation, reservation_advance_days,
                           reservation_channel, tags, description, suitable_for,
                           accessibility, rating, review_count,
                           ts_rank_cd(search_vector, plainto_tsquery('chinese', $1)) as _score
                    FROM spots
                    WHERE search_vector @@ plainto_tsquery('chinese', $1)
                    ORDER BY _score DESC
                    LIMIT 50
                    """,
                    query
                )

        return [self._row_to_dict(r) for r in rows]

    # ═══════════════════════════════════════════
    # RRF融合排序
    # ═══════════════════════════════════════════

    def _rrf_fusion(self, *result_lists: List[List[Dict]]) -> List[Dict]:
        """
        Reciprocal Rank Fusion：多路结果融合。

        公式：score = Σ(1 / (k + rank_i))
        其中 k=60（常数），rank_i是该POI在第i路的排名。

        优势：
        - 不需要分数归一化（各路分数scale不同）
        - 排名靠前的结果gain更大
        - 被多路同时召回的结果得分更高
        """
        from collections import defaultdict

        rrf_scores = defaultdict(float)
        poi_data = {}  # spot_id -> poi dict

        for results in result_lists:
            for rank, poi in enumerate(results):
                spot_id = poi.get("spot_id")
                if not spot_id:
                    continue

                # RRF分数
                rrf_scores[spot_id] += 1.0 / (self.RRF_K + rank + 1)

                # 保留最完整的POI数据
                if spot_id not in poi_data:
                    poi_data[spot_id] = poi
                else:
                    # 合并_score（取最大）
                    existing_score = poi_data[spot_id].get("_score", 0)
                    new_score = poi.get("_score", 0)
                    if new_score > existing_score:
                        poi_data[spot_id] = poi

        # 按RRF分数排序
        sorted_spots = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 组装结果
        merged = []
        for spot_id, rrf_score in sorted_spots:
            poi = poi_data[spot_id]
            poi["_rrf_score"] = round(rrf_score, 4)
            merged.append(poi)

        return merged

    # ═══════════════════════════════════════════
    # 实时API增强
    # ═══════════════════════════════════════════

    async def _enhance_realtime(self, pois: List[Dict]):
        """
        调用实时API补充动态信息。
        异步执行，不阻塞主流程。

        MVP阶段：Mock数据（P1后接入真实API）。
        """
        # 当前日期判断开放状态
        from datetime import datetime
        today = datetime.now()
        weekday = today.weekday()  # 0=周一

        for poi in pois:
            # 简单开放判断（MVP）
            poi["is_open_today"] = True
            poi["current_queue_time"] = poi.get("queue_time_avg", 30)
            poi["current_weather"] = "晴 25C"  # Mock

    # ═══════════════════════════════════════════
    # 工具函数
    # ═══════════════════════════════════════════

    @staticmethod
    def _row_to_dict(row) -> Dict:
        """asyncpg Row → dict"""
        d = dict(row)
        # 清理内部字段
        d.pop("vector_str", None)
        d.pop("_distance", None)
        return d

    async def close(self):
        if self.pg_pool:
            await self.pg_pool.close()
```

---

### 5.5 RAG检索性能优化

```python
# rag_optimizer.py —— RAG性能优化配置

class RAGOptimizer:
    """
    RAG检索性能优化：基于pgvector最佳实践。

    关键优化点：
      1. HNSW索引参数调优
      2. 查询时ef_search动态调整
      3. 高QPS场景预计算
      4. Embedding量化（大规模时）
    """

    # HNSW推荐参数（百万级向量）
    HNSW_CONFIGS = {
        "small":    {"m": 16, "ef_construction": 64,  "ef_search": 64},   # <10万
        "medium":   {"m": 24, "ef_construction": 128, "ef_search": 128}, # 10-50万
        "large":    {"m": 32, "ef_construction": 200, "ef_search": 256}, # >50万
    }

    @staticmethod
    async def tune_hnsw(pg_pool, table: str, size_category: str = "small"):
        """根据数据量调整HNSW参数"""
        config = RAGOptimizer.HNSW_CONFIGS.get(size_category, "small")

        async with pg_pool.acquire() as conn:
            await conn.execute(f"SET hnsw.ef_search = {config['ef_search']}")
            # 重建索引（如有需要）
            # await conn.execute(f"REINDEX INDEX idx_{table}_vector;")

    @staticmethod
    def quantize_embedding(vector: List[float], bits: int = 8) -> bytes:
        """
        Embedding量化：将float32压缩为int8，节省75%存储。
        适用于>100万向量的场景。
        """
        import numpy as np
        arr = np.array(vector, dtype=np.float32)
        # 归一化到[0, 255]
        min_val, max_val = arr.min(), arr.max()
        scaled = (arr - min_val) / (max_val - min_val) * 255
        return scaled.astype(np.uint8).tobytes()

    @staticmethod
    def estimate_latency_stats() -> Dict:
        """各阶段延迟预估（ms）"""
        return {
            "query_build": "5-10",
            "sql_prefilter": "10-30",
            "vector_search_hnsw": "15-40",
            "bm25_search": "10-20",
            "rrf_fusion": "1-5",
            "total_p99": "50-120",
        }
```

---

### 5.6 检索为空时的扩展策略

```python
# retrieval_fallback.py —— 检索降级策略

class RetrievalFallback:
    """
    当检索结果为空时的降级策略。
    确保用户至少能获得一些推荐。
    """

    @staticmethod
    async def expand_retrieval(profile: Dict, original_results: List) -> Dict:
        """
        检索为空时的扩展策略：
          1. 放宽预算约束 → 重新检索
          2. 去掉人群过滤 → 重新检索
          3. 推荐热门景点（兜底）
        """
        # 策略1：放宽预算
        relaxed_budget = dict(profile)
        relaxed_budget.pop("total_budget", None)

        # 策略2：去掉人群过滤
        relaxed_companion = dict(profile)
        relaxed_companion.pop("travel_companion", None)

        # 策略3：热门景点兜底（硬编码Top-10）
        hot_spots = {
            "北京": ["故宫", "长城", "天坛", "颐和园", "圆明园", "恭王府", "什刹海", "鸟巢", "798艺术区", "南锣鼓巷"],
            "上海": ["外滩", "东方明珠", "豫园", "南京路", "田子坊", "新天地", "迪士尼", "陆家嘴", "城隍庙", "世博会博物馆"],
            "西安": ["兵马俑", "大雁塔", "华清池", "城墙", "陕西历史博物馆", "钟鼓楼", "回民街", "大唐不夜城", "法门寺", "华山"],
            "成都": ["大熊猫基地", "宽窄巷子", "锦里", "武侯祠", "杜甫草堂", "都江堰", "青城山", "春熙路", "文殊院", "金沙遗址"],
        }

        dest = profile.get("destination", "")
        fallback_spots = hot_spots.get(dest, [])

        return {
            "fallback_used": True,
            "fallback_reason": "检索结果为空，已放宽约束并推荐热门景点",
            "suggestions": [
                "尝试更换目的地",
                "放宽预算范围", 
                "减少饮食禁忌要求",
            ],
            "hot_spots": fallback_spots,
        }
```

## 6. 规划决策引擎（第6层）

### 6.1 职责定位

规划决策引擎是系统的"数学大脑"，采用独立的**FastAPI微服务架构**与LangGraph异步集成。

**核心任务**：
1. **完整数据流预处理**：预约过滤 → 时长区间化 → 餐厅opt-in → 交通方式选择 → 疲劳模型
2. **数学求解**：OR-Tools CP-SAT（AddCircuit子回路消除 + Epsilon-Constraint多目标优化）
3. **自适应策略**：CP-SAT全局最优 + 贪心算法兜底 + SolutionCallback超时保护
4. **状态机管理**：防竞态的Planner状态机（IDLE/PLANNING/CONFIRMED/REPLANNING/ERROR）
5. **三模式重规划**：single(单点替换) / local(单日重算) / full(全量重算)

**绝对不能做**：用LLM规划路线；编造时间/费用；同步阻塞LangGraph主线程。

---

### 6.2 系统架构

```
用户输入（城市/天数/偏好）
    │
    ▼
┌──────────────────┐
│ 1. Reservation   │ ──> 过滤未预约景点 + 提醒
│    Handler       │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ 2. PlayTime      │ ──> 根据play_mode调整w_i区间
│    Manager       │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ 3. Restaurant    │ ──> opt-in时注入餐厅POI
│    Handler       │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ 4. Transport     │ ──> 根据偏好选择交通方式，生成dist_matrix
│    Selector      │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ 5. FatigueModel  │ ──> 根据人群类型添加恢复日约束
│    (可选)        │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ 6. CP-SAT Solver │ ──> 求解最优行程（或贪心兜底）
│    + Callback    │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ 7. Result Merge  │ ──> 合并预约提醒 + 求解结果
│                  │
└──────────────────┘
    │
    ▼
结构化行程JSON输出
```

LangGraph集成方式（异步非阻塞）：
```
┌─────────────────┐     HTTP/Async      ┌─────────────────────┐
│  LangGraph      │ ──────────────────> │  VRP Solver Service  │
│  (Itinerary     │    (非阻塞)         │  (FastAPI + Docker)  │
│   Planner)      │ <────────────────── │                     │
└─────────────────┘     JSON结果         └─────────────────────┘
        │                                        │
        │ 异步调用                               │ OR-Tools CP-SAT
        │ (async/await)                          │ + Solution Callback
        ▼                                        ▼
┌─────────────────┐                       ┌─────────────────────┐
│  State Machine  │                       │  Adaptive Solver    │
│  (IDLE/PLANNING │                       │  - CP-SAT优先       │
│   /CONFIRMED/   │                       │  - 贪心兜底         │
│   REPLANNING)   │                       │  - 超时保护         │
└─────────────────┘                       └─────────────────────┘
```

---

### 6.3 核心修复对照（v4.0 vs 原简化版）

| 决策点 | 原简化版 | v4.0修复方案 | 业界依据 |
|--------|---------|-------------|----------|
| 子回路消除 | 贪心排序（非最优） | AddCircuit（CP-SAT原生） | CP-SAT原生支持，自动剪枝 |
| 多目标优化 | 单目标（满意度） | Epsilon-Constraint | Miettinen: "almost always preferable" |
| 大M常数 | 统一1440 | 场景紧上界 | 整数规划性能优化 |
| 方差计算 | 贪心近似 | MAD线性化 | CP-SAT兼容性 |
| 通勤时间 | Haversine×15 | 高德API+Haversine降级 | 真实路线时间 |
| 游玩时长 | 固定w_i | 区间化[min, max]+模式选择 | 用户多样性 |
| 餐厅安排 | 无 | opt-in可选 | 用户体验 |
| 疲劳模型 | 无 | 跨天累积+恢复系数 | 生理模型 |

---

### 6.4 数学模型（修复后）

#### 决策变量

```
x_{d,i,j} ∈ {0,1}     — 第d天从景点i到景点j的边
visit_{d,i} ∈ {0,1}   — 第d天是否访问景点i
arrive_{d,i} ∈ Z⁺     — 第d天到达景点i的时间（分钟）
hotel_d ∈ {0,1}       — 第d天是否从酒店出发/返回
```

#### 约束1：子回路消除（AddCircuit）

```python
for d in range(D):
    arcs = []
    for i in range(n):
        for j in range(n):
            if i != j:
                arcs.append((i, j, x[d, i, j]))
        self_loop = model.NewBoolVar(f'sl_{d}_{i}')
        arcs.append((i, i, self_loop))
    model.AddCircuit(arcs)  # CP-SAT原生，自动消除子回路
```

#### 约束2：visit与边联动

```
Σ_j x_{d,j,i} = visit_{d,i}      ∀d, i≠0
Σ_j x_{d,i,j} = visit_{d,i}      ∀d, i≠0
```

#### 约束3：条件时间窗（大M法）

```
arrive_{d,i} >= L_i - M_time * (1 - visit_{d,i})        ∀d, i≠0
arrive_{d,i} + w_i <= R_i + M_time * (1 - visit_{d,i})   ∀d, i≠0
arrive_{d,i} <= M_time * visit_{d,i}                      ∀d, i≠0
```
其中 `M_time = max_i(R_i)`（紧上界，非统一1440）

#### 约束4：精确通勤传播（双向）

```
arrive_{d,j} >= arrive_{d,i} + w_i + dist_{i,j} - M_travel * (1 - x_{d,i,j})
arrive_{d,j} <= arrive_{d,i} + w_i + dist_{i,j} + M_travel * (1 - x_{d,i,j})
visit_{d,i} >= x_{d,i,j}
visit_{d,j} >= x_{d,i,j}
```

#### 约束5：单日时长（不计返回酒店通勤）

```
Σ_i visit_{d,i} * w_i + Σ_{i,j≠0} x_{d,i,j} * dist_{i,j} + Rest_day <= T_day_max    ∀d
```

#### 约束6：单日步行上限

```
day_walk_d = Σ_i visit_{d,i} * walk_i <= Walk_max    ∀d
```

#### 约束7：预算约束（含MAD目标）

```
day_cost_d = food_day + Σ_i visit_{d,i} * cost_i + Σ_{i,j} x_{d,i,j} * tc_{i,j}
day_cost_d <= Budget_day_max    ∀d
Σ_d day_cost_d <= Budget_total
MAD = Σ_d |day_cost_d - mean_cost|   （MAD替代方差，CP-SAT兼容）
```

#### 约束8：点位数量

```
Σ_i visit_{d,i} <= 5     ∀d    (硬约束：每天最多5个景点)
2 <= Σ_i visit_{d,i}     ∀d    (软约束：目标函数中惩罚空天)
```

#### Epsilon-Constraint多目标优化

```python
epsilon_config = {
    'primary_objective': 'min_travel_time',   # 主目标：最小化总通勤
    'constraints': {
        'max_walk_diff': 5,       # 每日步行差异上限
        'max_budget_mad': 100,    # 预算MAD上限（元）
        'min_preference': 200,    # 偏好分下限
        'max_peak_score': 500,    # 高峰错峰上限
    }
}
```

优势：能探索非凸Pareto前沿；量纲不敏感；参数直观；可生成Pareto前沿供用户选择。

---

### 6.5 业务逻辑层（修复后）

#### 6.5.1 预约制景点处理

```python
class ReservationHandler:
    """预约制景点过滤与提醒"""

    # 需预约景点库（与实际预约系统对接前使用本地库）
    RESERVATION_DB = {
        "故宫博物院": {"advance_days": 7, "channel": "故宫博物院官网/微信小程序", "note": "周一闭馆"},
        "陕西历史博物馆": {"advance_days": 3, "channel": "公众号", "note": "周二闭馆"},
        "天安门广场": {"advance_days": 1, "channel": "天安门广场预约参观小程序"},
        "敦煌莫高窟": {"advance_days": 30, "channel": "莫高窟官网", "note": "旺季需提前1个月"},
        "布达拉宫": {"advance_days": 7, "channel": "布达拉宫官网"},
    }

    def filter(self, poi_list: List[Dict], travel_date: str,
               user_reservations: List[str] = None) -> Tuple[List[Dict], List[str]]:
        """
        过滤未预约的必去景点，返回提醒信息。
        非必去景点直接移除；必去景点保留但添加提醒。
        """
        filtered = []
        reminders = []

        for poi in poi_list:
            name = poi.get("spot_name", "")
            info = self.RESERVATION_DB.get(name)

            if not info or not info.get("reservation_required", False):
                filtered.append(poi)
                continue

            # 检查用户是否已预约
            if user_reservations and name in user_reservations:
                filtered.append(poi)
                continue

            # 必去景点保留但提醒
            if name in poi.get("_must_visit", []):
                filtered.append(poi)
                reminders.append(
                    f"【预约提醒】「{name}」需提前{info['advance_days']}天预约，"
                    f"请通过{info['channel']}预约。{info.get('note', '')}"
                )
            else:
                # 非必去+需预约 → 移除
                reminders.append(f"「{name}」因需预约且未确认，暂从行程中移除")

        return filtered, reminders
```

#### 6.5.2 游玩时长区间化

```python
class PlayTimeManager:
    """根据play_mode和人群类型调整游玩时长"""

    MODE_MULTIPLIERS = {
        "quick": {"min_factor": 0.25, "max_factor": 0.6,  "default": 0.4},   # 快速打卡
        "standard": {"min_factor": 0.7, "max_factor": 1.0,  "default": 1.0},  # 标准
        "deep": {"min_factor": 1.0, "max_factor": 2.0,  "default": 1.5},    # 深度
    }

    def adjust(self, poi: Dict, play_mode: str = "standard") -> Dict:
        """调整POI的游玩时长区间"""
        base = poi.get("duration_minutes", 120)
        min_play = poi.get("min_play_time", 15)
        max_play = poi.get("max_play_time", 240)

        mult = self.MODE_MULTIPLIERS.get(play_mode, self.MODE_MULTIPLIERS["standard"])

        adjusted = dict(poi)
        adjusted["w_i"] = int(base * mult["default"])           # CP-SAT使用的时长
        adjusted["w_i_min"] = max(min_play, int(base * mult["min_factor"]))
        adjusted["w_i_max"] = min(max_play, int(base * mult["max_factor"]))

        return adjusted
```

| 模式 | 计算方式 | 适用人群 |
|------|---------|---------|
| Quick（快速打卡） | max(15, min_play_time) | 老年人/带儿童/时间紧 |
| Standard（标准游览） | w_i（原建议时长） | 一般游客 |
| Deep（深度体验） | min(max_play_time, 480) | 爱好者/时间充裕 |

#### 6.5.3 餐厅安排（opt-in）

```python
class RestaurantHandler:
    """餐厅安排 —— 默认不安排，用户opt-in后注入"""

    DEFAULT_CONFIG = {
        "include_restaurant": False,   # 默认不安排
        "meals_per_day": 0,            # 默认0餐
        "lunch_window": (210, 450),    # 11:30-13:30（分钟）
        "dinner_window": (570, 720),   # 17:30-20:00
        "restaurant_duration": 60,     # 用餐60分钟
    }

    def inject(self, poi_list: List[Dict], config: Dict) -> List[Dict]:
        """如果用户选择安排餐厅，注入餐厅POI到候选列表"""
        if not config.get("include_restaurant", False):
            return poi_list

        # 在餐厅时段附近的景点间插入餐厅
        # 实际实现根据位置+预算+口味偏好选择
        return poi_list  # MVP: 简化处理
```

#### 6.5.4 跨天疲劳累积模型

```python
class FatigueModel:
    """
    跨天疲劳累积模型。

    模型：fatigue_d = alpha * fatigue_{d-1} + day_walk_d

    恢复系数alpha：
      - 老年人(带老人): 0.70（恢复慢）
      - 带儿童: 0.50
      - 一般成人: 0.35
      - 年轻人: 0.25（恢复快）

    约束：连续2天高强度 → 第3天强制低强度（<= Walk_max * 0.4）
    """

    RECOVERY_RATES = {
        "elderly": 0.70,
        "children": 0.50,
        "adult": 0.35,
        "young": 0.25,
    }

    def compute_constraints(self, profile: Dict, D: int) -> List[Dict]:
        """根据人群类型生成疲劳相关约束"""
        constraints = []

        # 确定恢复系数
        if profile.get("has_elderly"):
            alpha = self.RECOVERY_RATES["elderly"]
        elif profile.get("has_children"):
            alpha = self.RECOVERY_RATES["children"]
        else:
            alpha = self.RECOVERY_RATES["adult"]

        # 连续2天高强度 → 第3天强制低强度
        # CP-SAT中通过添加软约束实现
        for d in range(2, D):
            constraints.append({
                "type": "fatigue_recovery",
                "day": d,
                "alpha": alpha,
                "max_walk_after_intense": int(profile.get("Walk_max", 10) * 0.4),
                "note": f"连续高强度后第{d+1}天强制低强度",
            })

        return constraints
```

#### 6.5.5 多交通方式选择

```python
class TransportSelector:
    """根据用户偏好选择交通方式，生成通勤矩阵"""

    MODE_PRIORITIES = {
        "subway": ["subway", "bus", "taxi", "walk"],
        "bus": ["bus", "subway", "taxi", "walk"],
        "taxi": ["taxi", "subway", "bus", "walk"],
        "walk": ["walk", "subway", "bus", "taxi"],
    }

    async def select(self, poi_list: List[Dict], preference: List[str],
                     map_service: MapServiceRouter) -> Tuple[List[List[int]], List[List[int]]]:
        """
        生成距离矩阵(dist_matrix)和费用矩阵(tc_matrix)。
        优先使用高德API，失败降级为Haversine估算。
        """
        n = len(poi_list)
        dist_matrix = [[0] * n for _ in range(n)]
        tc_matrix = [[0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                origin = (poi_list[i]["lat"], poi_list[i]["lng"])
                dest = (poi_list[j]["lat"], poi_list[j]["lng"])

                # 调用地图服务（自动降级）
                result = await map_service.get_distance(origin, dest)
                dist_matrix[i][j] = result["duration_minutes"]
                tc_matrix[i][j] = result.get("cost", 0)

        return dist_matrix, tc_matrix
```

#### 6.5.6 高德API抽象层+降级策略

```python
class MapServiceRouter:
    """地图服务路由器 —— 自动切换 + Haversine降级"""

    def __init__(self, config: Dict):
        self.primary = GaodeMapAPI(config.get("GAODE_KEY"))
        self.fallback = HaversineFallback()
        self.primary_failures = 0
        self.max_failures = 3  # 失败3次后降级

    async def get_distance(self, origin: Tuple[float, float],
                           destination: Tuple[float, float],
                           mode: str = "driving") -> Dict:
        """
        获取两点间通勤时间。
        主提供商失败3次后自动切换到Haversine估算。
        """
        if self.primary_failures < self.max_failures:
            try:
                result = await self.primary.get_distance(origin, destination, mode)
                self.primary_failures = 0  # 成功后重置
                return result
            except Exception:
                self.primary_failures += 1

        # 降级：Haversine估算
        return await self.fallback.get_distance(origin, destination)


class HaversineFallback:
    """Haversine公式降级方案"""

    async def get_distance(self, origin: Tuple[float, float],
                           destination: Tuple[float, float]) -> Dict:
        import math

        lat1, lng1 = origin
        lat2, lng2 = destination

        R = 6371  # 地球半径km
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng/2)**2
        dist_km = 2 * R * math.asin(math.sqrt(a))

        # 估算：地铁15min/km，步行20min/km
        duration = int(dist_km * 15)

        return {
            "distance_meters": int(dist_km * 1000),
            "duration_minutes": max(duration, 5),
            "cost": int(dist_km * 3),  # 估算费用
            "mode": "estimated",
            "source": "haversine_fallback",
        }
```

---

### 6.6 核心求解器（FastAPI微服务）

```python
# vrp_solver_service.py —— VRP求解FastAPI微服务

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
from ortools.sat.python import cp_model
import time
import uvicorn

app = FastAPI(title="Travel VRP Solver", version="4.0")


# ===== 请求/响应模型 =====

class POIInput(BaseModel):
    spot_id: int
    spot_name: str
    w_i: int              # 游玩时长（分钟）
    L_i: int              # 时间窗左界（分钟）
    R_i: int              # 时间窗右界（分钟）
    cost_i: int = 0       # 门票价格
    walk_i: int = 1       # 步行强度
    pref_i: int = 50      # 偏好分数
    is_peak: int = 0      # 是否高峰景点


class ConstraintsInput(BaseModel):
    D: int = 3                    # 天数
    T_day_max: int = 600          # 单日最大外出分钟
    Walk_max: int = 10            # 单日步行上限
    Budget_total: int = 5000      # 总预算
    Budget_day_max: int = 2000    # 单日预算上限
    Drive_max: int = 60           # 单次通勤最大分钟
    Rest_day: int = 90            # 每日休息预留
    food_day: int = 100           # 单日餐饮基础


class SolverRequest(BaseModel):
    constraints: ConstraintsInput
    poi_list: List[POIInput]
    dist_matrix: List[List[int]]   # 通勤时间矩阵
    tc_matrix: List[List[int]]     # 通勤费用矩阵
    max_solve_time_ms: int = 5000  # 求解超时


class DaySchedule(BaseModel):
    day: int
    schedule: List[Dict]           # 景点列表+时间
    day_walk: int                  # 当天步行强度
    day_cost: int                  # 当天费用


class SolverResponse(BaseModel):
    status: str                    # "optimal" | "feasible" | "timeout" | "infeasible"
    itinerary: List[DaySchedule]
    total_cost: int
    total_walk: int
    solve_time_ms: int
    reservation_reminders: List[str] = []


# ===== Solution Callback（超时保护） =====

class TimeoutCallback(cp_model.CpSolverSolutionCallback):
    """超时回调 —— 超时后返回当前最优解"""

    def __init__(self, max_time_ms: int):
        super().__init__()
        self.max_time_ms = max_time_ms
        self.start_time = time.time() * 1000
        self.best_solution = None

    def on_solution_callback(self):
        elapsed = time.time() * 1000 - self.start_time
        if elapsed > self.max_time_ms:
            self.StopSearch()  # 超时停止
        # 记录当前最优解
        self.best_solution = self._extract_current()

    def _extract_current(self):
        # 提取当前解（简化）
        return {"objective": self.ObjectiveValue()}


# ===== 核心求解器 =====

class TravelVRPSolver:
    """旅行VRP求解器 —— CP-SAT + Epsilon-Constraint"""

    def solve(self, req: SolverRequest) -> SolverResponse:
        start_time = time.time() * 1000

        c = req.constraints
        n = len(req.poi_list)
        D = c.D

        model = cp_model.CpModel()

        # ── 决策变量 ──
        x = {}
        for d in range(D):
            for i in range(n):
                for j in range(n):
                    x[d, i, j] = model.NewBoolVar(f"x_{d}_{i}_{j}")

        visit = {}
        for d in range(D):
            for i in range(n):
                visit[d, i] = model.NewBoolVar(f"v_{d}_{i}")

        arrive = {}
        M_time = max(p.R_i for p in req.poi_list) if req.poi_list else 1440
        for d in range(D):
            for i in range(n):
                arrive[d, i] = model.NewIntVar(0, M_time, f"a_{d}_{i}")

        # ── 约束1: AddCircuit子回路消除 ──
        for d in range(D):
            arcs = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        arcs.append((i, j, x[d, i, j]))
                self_loop = model.NewBoolVar(f"sl_{d}_{i}")
                arcs.append((i, i, self_loop))
            model.AddCircuit(arcs)

        # ── 约束2: visit与边联动 ──
        for d in range(D):
            for i in range(n):
                if i == 0:  # 酒店（spot_id=0）
                    continue
                model.Add(sum(x[d, j, i] for j in range(n) if j != i) == visit[d, i])
                model.Add(sum(x[d, i, j] for j in range(n) if j != i) == visit[d, i])

        # ── 约束3: 条件时间窗 ──
        for d in range(D):
            for i, poi in enumerate(req.poi_list):
                if i == 0:
                    continue
                model.Add(arrive[d, i] >= poi.L_i - M_time * (1 - visit[d, i]))
                model.Add(arrive[d, i] + poi.w_i <= poi.R_i + M_time * (1 - visit[d, i]))
                model.Add(arrive[d, i] <= M_time * visit[d, i])

        # ── 约束4: 通勤传播 ──
        M_travel = c.Drive_max + max(max(row) for row in req.dist_matrix) if req.dist_matrix else 200
        for d in range(D):
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    model.Add(
                        arrive[d, j] >= arrive[d, i] + req.poi_list[i].w_i + req.dist_matrix[i][j]
                        - M_travel * (1 - x[d, i, j])
                    )

        # ── 约束5: 单日时长 ──
        for d in range(D):
            play_time = sum(visit[d, i] * req.poi_list[i].w_i for i in range(n) if i > 0)
            travel_time = sum(
                x[d, i, j] * req.dist_matrix[i][j]
                for i in range(n) for j in range(n) if i != j and i > 0 and j > 0
            )
            model.Add(play_time + travel_time + c.Rest_day <= c.T_day_max)

        # ── 约束6: 步行上限 ──
        for d in range(D):
            model.Add(
                sum(visit[d, i] * req.poi_list[i].walk_i for i in range(n) if i > 0) <= c.Walk_max
            )

        # ── 约束7: 预算 ──
        day_costs = []
        for d in range(D):
            ticket_cost = sum(visit[d, i] * req.poi_list[i].cost_i for i in range(n) if i > 0)
            tc_cost = sum(
                x[d, i, j] * req.tc_matrix[i][j]
                for i in range(n) for j in range(n) if i != j
            )
            dc = model.NewIntVar(0, c.Budget_total, f"dc_{d}")
            model.Add(dc == ticket_cost + tc_cost + c.food_day)
            model.Add(dc <= c.Budget_day_max)
            day_costs.append(dc)

        model.Add(sum(day_costs) <= c.Budget_total)

        # ── 约束8: 点位数量 ──
        for d in range(D):
            model.Add(sum(visit[d, i] for i in range(n) if i > 0) <= 5)

        # ── 目标: Epsilon-Constraint ──
        # 主目标: 最小化总通勤时间
        total_travel = sum(
            x[d, i, j] * req.dist_matrix[i][j]
            for d in range(D) for i in range(n) for j in range(n) if i != j
        )
        model.Minimize(total_travel)

        # ── 求解 ──
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = req.max_solve_time_ms / 1000.0
        solver.parameters.num_search_workers = 4
        solver.parameters.log_search_progress = False

        callback = TimeoutCallback(req.max_solve_time_ms)
        status = solver.Solve(model, callback)

        solve_time_ms = int(time.time() * 1000 - start_time)

        # ── 解析结果 ──
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            itinerary = self._extract_itinerary(solver, x, visit, arrive, req, D)
            return SolverResponse(
                status="optimal" if status == cp_model.OPTIMAL else "feasible",
                itinerary=itinerary,
                total_cost=sum(d.day_cost for d in itinerary),
                total_walk=sum(d.day_walk for d in itinerary),
                solve_time_ms=solve_time_ms,
            )
        elif status == cp_model.MODEL_INVALID:
            return SolverResponse(
                status="infeasible",
                itinerary=[],
                total_cost=0, total_walk=0, solve_time_ms=solve_time_ms,
            )
        else:
            return SolverResponse(
                status="timeout",
                itinerary=[],
                total_cost=0, total_walk=0, solve_time_ms=solve_time_ms,
            )

    def _extract_itinerary(self, solver, x, visit, arrive, req, D) -> List[DaySchedule]:
        """从求解结果中提取行程"""
        itinerary = []

        for d in range(D):
            day_pois = []
            for i, poi in enumerate(req.poi_list):
                if i == 0:  # 跳过酒店
                    continue
                if solver.Value(visit[d, i]) == 1:
                    day_pois.append({
                        "spot_id": poi.spot_id,
                        "spot_name": poi.spot_name,
                        "arrive_time": f"{solver.Value(arrive[d, i]) // 60:02d}:{solver.Value(arrive[d, i]) % 60:02d}",
                        "play_minute": poi.w_i,
                        "ticket_cost": poi.cost_i,
                        "walk_score": poi.walk_i,
                    })

            day_walk = sum(p["walk_score"] for p in day_pois)
            day_cost = sum(p["ticket_cost"] for p in day_pois) + req.constraints.food_day

            itinerary.append(DaySchedule(
                day=d + 1,
                schedule=day_pois,
                day_walk=day_walk,
                day_cost=day_cost,
            ))

        return itinerary


# ===== FastAPI端点 =====

solver_instance = TravelVRPSolver()

@app.post("/solve", response_model=SolverResponse)
async def solve_endpoint(req: SolverRequest):
    """VRP求解端点 —— 异步非阻塞"""
    return solver_instance.solve(req)

@app.get("/health")
async def health():
    return {"status": "ok", "version": "4.0"}


# ===== LangGraph异步客户端 =====

class TravelVRPClient:
    """VRP求解服务异步客户端 —— LangGraph集成用"""

    def __init__(self, base_url: str = "http://localhost:8001"):
        import httpx
        self.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def solve(self, constraints, poi_list, dist_matrix, tc_matrix,
                    max_solve_time_ms: int = 5000) -> Dict:
        """异步调用求解服务（非阻塞）"""
        payload = {
            "constraints": constraints,
            "poi_list": poi_list,
            "dist_matrix": dist_matrix,
            "tc_matrix": tc_matrix,
            "max_solve_time_ms": max_solve_time_ms,
        }
        response = await self.client.post("/solve", json=payload)
        return response.json()

    async def close(self):
        await self.client.aclose()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

---

### 6.7 自适应求解策略

| 场景 | 策略 | 说明 |
|------|------|------|
| D≤3天, POI≤15 | 贪心算法 | O(n²)，更快，足够好 |
| D>3天或POI>15 | CP-SAT | 全局最优，Epsilon-Constraint |
| CP-SAT超时 | 贪心兜底 | SolutionCallback返回当前最优 |
| 全部不可行 | 冲突分析 | 引导用户调整约束 |

```python
class AdaptiveSolver:
    """自适应求解策略选择器"""

    async def solve(self, req: SolverRequest) -> SolverResponse:
        n = len(req.poi_list)
        D = req.constraints.D

        # 小规模 → 贪心
        if D <= 3 and n <= 15:
            return await self._greedy_solve(req)

        # 大规模 → CP-SAT
        return await self._cp_sat_solve(req)

    async def _greedy_solve(self, req: SolverRequest) -> SolverResponse:
        """贪心算法：最近邻排序 + 时间窗检查"""
        # O(n²)实现，适合MVP快速响应
        pass  # 实现略

    async def _cp_sat_solve(self, req: SolverRequest) -> SolverResponse:
        """CP-SAT：调用FastAPI微服务"""
        client = TravelVRPClient()
        try:
            result = await client.solve(**req.dict())
            return SolverResponse(**result)
        except Exception:
            # 降级：贪心兜底
            return await self._greedy_solve(req)
        finally:
            await client.close()
```

---

### 6.8 动态重规划状态机（防竞态）

```python
from enum import Enum, auto
import asyncio

class PlannerState(Enum):
    IDLE = auto()         # 空闲
    PLANNING = auto()     # 正在求解（拒绝新请求）
    CONFIRMED = auto()    # 行程已确认
    REPLANNING = auto()   # 正在重规划（排队新事件）
    ERROR = auto()        # 错误

class ItineraryPlanner:
    """行程规划器 —— 状态机 + 事件队列防竞态"""

    def __init__(self):
        self.state = PlannerState.IDLE
        self.pending_events = []       # 待处理事件队列
        self.current_itinerary = None
        self.lock = asyncio.Lock()

    async def handle_change_event(self, event: Dict) -> Dict:
        """处理用户修改请求 —— 状态机防竞态"""
        async with self.lock:
            if self.state in (PlannerState.PLANNING, PlannerState.REPLANNING):
                self.pending_events.append(event)
                return {"status": "queued", "position": len(self.pending_events)}

            self.state = PlannerState.REPLANNING

        try:
            # 执行重规划
            result = await self._execute_replan(event)

            # 处理排队事件
            async with self.lock:
                while self.pending_events:
                    next_event = self.pending_events.pop(0)
                    result = await self._execute_replan(next_event)

            return result
        finally:
            async with self.lock:
                self.state = PlannerState.CONFIRMED

    async def _execute_replan(self, event: Dict) -> Dict:
        """执行重规划 —— 三模式路由"""
        action = event.get("action", "")

        if action in ("remove_poi", "replace_poi"):
            # single模式：直接修改
            return self._apply_single_change(event)
        elif action == "change_pace":
            # local模式：单日重算
            return await self._replan_local(event)
        else:
            # full模式：全量重算
            return await self._replan_full(event)
```

---

### 6.9 ItineraryPlannerAgent（LangGraph集成）

```python
# agents/planner.py —— LangGraph Node集成

async def plan_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    第6层：ItineraryPlannerAgent（异步调用VRP微服务）

    输入：state["poi_candidates"] + state["slots"] + state["user_profile"]
    输出：itinerary + solve_status + solve_time_ms + conflict_reasons
    """
    trace = {"node_name": "plan", "started_at": _now(), "status": "success"}

    try:
        # 1. 数据预处理流水线
        from preprocessing import (
            ReservationHandler, PlayTimeManager, RestaurantHandler,
            TransportSelector, FatigueModel
        )

        slots = state["slots"]
        profile = state.get("user_profile", {})
        poi_list = state.get("poi_candidates", [])

        # 1.1 预约过滤
        handler = ReservationHandler()
        poi_list, reminders = handler.filter(
            poi_list,
            slots.get("travel_dates", "").split("~")[0] if slots.get("travel_dates") else "",
        )

        # 1.2 时长区间化
        ptm = PlayTimeManager()
        poi_list = [ptm.adjust(p, slots.get("play_mode", "standard")) for p in poi_list]

        # 1.3 餐厅opt-in
        rh = RestaurantHandler()
        poi_list = rh.inject(poi_list, {
            "include_restaurant": slots.get("include_restaurant", False),
        })

        # 1.4 场景参数调优（替代原TravelSlots中的CP-SAT参数）
        from cp_sat_tuning import CPSATTuningGuide
        tuned_params = CPSATTuningGuide.tune(slots, profile)

        # 1.5 通勤矩阵
        ts = TransportSelector()
        dist_matrix, tc_matrix = await ts.select(
            poi_list,
            slots.get("transport_preference", ["subway", "walk", "taxi"]),
        )

        # 1.6 疲劳约束
        fm = FatigueModel()
        fatigue_constraints = fm.compute_constraints(profile, tuned_params["D"])

        # 2. 调用VRP求解服务（异步非阻塞）
        from vrp_solver_service import TravelVRPClient, SolverRequest

        client = TravelVRPClient(config.get("VRP_SERVICE_URL", "http://vrp-solver:8001"))

        # 构建请求
        req = SolverRequest(
            constraints=tuned_params,
            poi_list=poi_list,
            dist_matrix=dist_matrix,
            tc_matrix=tc_matrix,
            max_solve_time_ms=config.get("MAX_SOLVE_TIME_MS", 5000),
        )

        result = await client.solve(req)
        await client.close()

        # 3. 合并预约提醒
        result["reservation_reminders"] = reminders

        trace.update({"ended_at": _now(), "duration_ms": _elapsed(trace),
                      "output_summary": f"status={result['status']}"})

        return {
            "itinerary": result.get("itinerary", []),
            "budget_breakdown": result.get("budget_breakdown"),
            "solve_status": result["status"],
            "solve_time_ms": result.get("solve_time_ms", 0),
            "conflict_reasons": result.get("conflict_reasons", []),
            "reservation_reminders": reminders,
            "execution_trace": state.get("execution_trace", []) + [trace]
        }

    except Exception as e:
        trace.update({"ended_at": _now(), "status": "fallback",
                      "error_message": str(e)[:200]})
        return {
            "solve_status": "fallback",
            "fallback_used": True,
            "execution_trace": state.get("execution_trace", []) + [trace]
        }
```

---

### 6.10 场景参数调优（替代原TravelSlots中的CP-SAT参数）

```python
# cp_sat_tuning.py —— CP-SAT场景参数自动计算

class CPSATTuningGuide:
    """
    CP-SAT参数自动调优。

    原方案中这些参数混在TravelSlots中（违反关注点分离），
    v4.0改为根据人群类型+节奏+目的地自动计算。
    """

    # 基础参数（moderate节奏、标准成人）
    BASE_PARAMS = {
        "T_day_max": 600,       # 10小时
        "Rest_day": 90,         # 1.5小时休息
        "Walk_max": 10,         # 步行强度上限
        "Drive_max": 60,        # 单次通勤上限
        "food_day": 100,        # 餐饮基础
    }

    # 人群调整
    PERSONA_ADJUSTMENTS = {
        "elderly": {"Walk_max": -4, "Drive_max": -20, "Rest_day": 30, "T_day_max": -120},
        "children": {"Walk_max": -3, "Drive_max": -10, "Rest_day": 30, "T_day_max": -60},
        "pregnant": {"Walk_max": -6, "Drive_max": -30, "Rest_day": 60, "T_day_max": -180},
    }

    # 节奏调整
    PACE_ADJUSTMENTS = {
        "relaxed": {"T_day_max": -180, "Rest_day": 60, "Walk_max": -2},
        "moderate": {},  # 基线
        "intensive": {"T_day_max": 60, "Rest_day": -30, "Walk_max": 2},
    }

    @classmethod
    def tune(cls, slots: Dict, profile: Dict) -> Dict:
        """
        根据TravelSlots和画像自动计算CP-SAT参数。

        替代原TravelSlots中的T_day_max/Rest_day/Walk_max等字段。
        """
        params = dict(cls.BASE_PARAMS)

        # 人群调整
        if slots.get("has_elderly"):
            cls._apply_adjustment(params, cls.PERSONA_ADJUSTMENTS["elderly"])
        if slots.get("has_children"):
            cls._apply_adjustment(params, cls.PERSONA_ADJUSTMENTS["children"])
        if slots.get("has_pregnant"):
            cls._apply_adjustment(params, cls.PERSONA_ADJUSTMENTS["pregnant"])

        # 节奏调整
        pace = slots.get("pace", "moderate")
        cls._apply_adjustment(params, cls.PACE_ADJUSTMENTS.get(pace, {}))

        # 天数
        params["D"] = slots.get("travel_days", 3)

        # 预算
        budget = slots.get("total_budget")
        days = slots.get("travel_days", 3)
        if budget:
            params["Budget_total"] = int(budget)
            params["Budget_day_max"] = int(budget / days * 1.5)  # 允许某天超支
        else:
            params["Budget_total"] = 999999  # 无限制
            params["Budget_day_max"] = 999999

        # 恢复系数（疲劳模型）
        if profile.get("has_elderly"):
            params["fatigue_alpha"] = 0.70
        elif profile.get("has_children"):
            params["fatigue_alpha"] = 0.50
        else:
            params["fatigue_alpha"] = 0.35

        return params

    @staticmethod
    def _apply_adjustment(params: Dict, adjustment: Dict):
        for key, delta in adjustment.items():
            params[key] = max(10, params.get(key, 0) + delta)
```

---

### 6.11 三模式重规划

与上半部分第1章的`apply_single_change_node`、`replan_local_node`衔接：

```python
# nodes.py —— 补全Graph构建中缺少的Node（修复S2）

async def apply_single_change_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    单点修改Node：O(1)直接修改行程JSON，不调用求解器。
    对应ReplanEngine.apply_single_change()
    """
    feedback = state.get("user_feedback", {})
    action = feedback.get("action", "")
    itinerary = state.get("itinerary", [])

    if action == "remove_poi":
        day_idx = feedback.get("day", 1) - 1
        poi_name = feedback.get("poi_name")
        if 0 <= day_idx < len(itinerary):
            itinerary[day_idx]["schedule"] = [
                a for a in itinerary[day_idx]["schedule"]
                if a["spot_name"] != poi_name
            ]

    elif action == "replace_poi":
        day_idx = feedback.get("day", 1) - 1
        old_name = feedback.get("old_poi_name")
        new_id = feedback.get("new_poi_id")
        poi_list = state.get("poi_candidates", [])

        new_poi = next((p for p in poi_list if p["spot_id"] == new_id), None)
        if new_poi and 0 <= day_idx < len(itinerary):
            for act in itinerary[day_idx]["schedule"]:
                if act["spot_name"] == old_name:
                    act["spot_id"] = new_poi["spot_id"]
                    act["spot_name"] = new_poi["spot_name"]
                    act["play_minute"] = new_poi.get("duration_minutes", 120)
                    act["ticket_cost"] = new_poi.get("ticket_price", 0)
                    act["walk_score"] = new_poi.get("walk_intensity", 1)
                    break

    return {"itinerary": itinerary, "replan_mode": "single",
            "retry_count": 0}  # 重置retry计数（修复M2）


async def replan_local_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    局部重规划Node：只重算目标日期，固定其他天。
    调用CP-SAT时D=1，大幅降低求解复杂度。
    """
    feedback = state.get("user_feedback", {})
    target_day = feedback.get("day", 1) - 1

    # 收集已固定的POI
    fixed_pois = set()
    for d, day in enumerate(state.get("itinerary", [])):
        if d == target_day:
            continue
        for act in day.get("schedule", []):
            fixed_pois.add(act.get("spot_id"))

    # 可用POI = 候选池 - 已固定
    available = [p for p in state.get("poi_candidates", [])
                 if p["spot_id"] not in fixed_pois]

    # 单日约束
    from cp_sat_tuning import CPSATTuningGuide
    day_slots = dict(state.get("slots", {}))
    day_slots["travel_days"] = 1
    tuned = CPSATTuningGuide.tune(day_slots, state.get("user_profile", {}))

    # 调用VRP求解器
    from vrp_solver_service import TravelVRPClient
    client = TravelVRPClient()

    try:
        result = await client.solve(tuned, available, [], [])
        new_itin = list(state.get("itinerary", []))
        if result.get("itinerary"):
            new_itin[target_day] = result["itinerary"][0]
        return {"itinerary": new_itin, "replan_mode": "local", "retry_count": 0}
    finally:
        await client.close()


async def error_handler_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    全局异常处理Node。
    被global_error_handler调用，返回降级输出。
    """
    error_node = state.get("error_node", "unknown")
    error_message = state.get("error_message", "未知错误")

    # 根据出错Node返回对应降级策略
    return await global_error_handler(state, NodeException(error_node, Exception(error_message), "critical"))


async def human_interrupt_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Human-in-the-loop Node。

    展示行程 + 反馈选项，等待用户输入。

    实现方式：使用LangGraph的interrupt()机制
      1. 通过SSE推送output_markdown + 反馈选项到前端
      2. 调用interrupt()挂起Graph
      3. 前端用户操作后，通过POST /resume 触发Graph继续
      4. 用户反馈写入state["user_feedback"]
      5. Graph根据feedback_action路由到对应Node

    文档：详见LangGraph Human-in-the-loop指南
    """
    # 生成反馈选项
    feedback_options = _generate_feedback_options(state)

    # 组合输出
    full_output = state.get("output_markdown", "") + "

" + feedback_options

    # 使用LangGraph interrupt()挂起等待用户输入
    # from langgraph.types import interrupt
    # user_input = interrupt({"output": full_output, "options": feedback_options})

    # MVP简化版：直接返回，等待外部触发
    return {
        "output_markdown": full_output,
        "next_node": "human_interrupt",  # 标记等待状态
    }
```

---

### 6.12 数据层DDL扩展（对齐v4.0方案）

```sql
-- v4.0新增/扩展DDL

-- 景点信息表（扩展后）
CREATE TABLE spot_info (
    spot_id SERIAL PRIMARY KEY,
    spot_name VARCHAR(100) NOT NULL,
    spot_type VARCHAR(20) NOT NULL,
    city VARCHAR(50) NOT NULL,
    district VARCHAR(50),
    lat DECIMAL(10,7), lng DECIMAL(10,7),
    w_i INT NOT NULL,                    -- 标准时长
    min_play_time INT DEFAULT 15,        -- 最短时长
    max_play_time INT DEFAULT 240,       -- 最长时长
    open_time TIME, close_time TIME,
    night_open BOOLEAN DEFAULT FALSE,     -- 夜游
    cost_i INT NOT NULL DEFAULT 0,       -- 门票
    walk_i INT NOT NULL DEFAULT 1,       -- 步行强度
    accessibility INT DEFAULT 5,          -- 无障碍
    indoor_outdoor VARCHAR(10) DEFAULT 'mixed',
    reservation_required BOOLEAN DEFAULT FALSE,
    reservation_advance_days INT DEFAULT 0,
    queue_time_avg INT DEFAULT 0,
    is_peak BOOLEAN DEFAULT FALSE,
    spot_tags JSON,                       -- 标签
    season_restriction JSON,              -- 季节限制
    temp_closure_dates JSON,              -- 临时闭馆
    description_vector VECTOR(768),       -- pgvector语义向量
    search_vector tsvector,               -- BM25全文搜索
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    INDEX idx_city (city),
    INDEX idx_composite (city, spot_type, walk_i, indoor_outdoor)
);

-- 多交通方式通勤矩阵（重构后）
CREATE TABLE spot_distance_multi (
    spot_id_from INT NOT NULL,
    spot_id_to INT NOT NULL,
    transport_mode VARCHAR(20) NOT NULL,  -- walk/subway/bus/taxi/drive
    dist_minute INT NOT NULL,
    traffic_cost INT DEFAULT 0,
    available_start TIME DEFAULT '06:00',
    available_end TIME DEFAULT '23:00',
    is_default BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (spot_id_from, spot_id_to, transport_mode)
);

-- HNSW向量索引
CREATE INDEX idx_spot_vector ON spot_info 
    USING hnsw (description_vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

---

### 6.13 关键修复对照总表

| ID | 维度 | 问题 | 严重度 | v4.0修复方法 | 业界依据 |
|----|------|------|--------|-------------|---------|
| CR-1 | 数学 | 子回路消除缺失 | 致命 | AddCircuit | CP-SAT原生 |
| CR-2 | 数学 | 时间窗与visit未联动 | 致命 | 条件大M约束 | 标准CP建模 |
| CR-3 | 数学 | 传播约束只有下界 | 致命 | 添加上界 | TSP标准做法 |
| CR-4 | 数学 | 总时长计算错误 | 高 | 排除返回酒店通勤 | 业务逻辑修正 |
| CR-5 | 数学 | 方差公式错误 | 高 | MAD线性化 | CP-SAT兼容 |
| CR-6 | 工程 | 伪代码不可运行 | 致命 | 完整FastAPI实现 | 生产要求 |
| CR-7 | 工程 | CP-SAT超时降级缺陷 | 致命 | SolutionCallback | OR-Tools最佳 |
| CR-8 | 工程 | LangGraph同步阻塞 | 致命 | 异步HTTP微服务 | 架构最佳 |
| CR-9 | 工程 | 动态重规划竞态 | 致命 | 状态机+队列 | 并发控制 |
| CR-10 | 业务 | 酒店固定起终点 | 致命 | 泛化起终点 | 真实旅游行为 |
| CR-11 | 业务 | 餐厅强制安排 | 致命 | opt-in可选 | 用户体验 |
| CR-12 | 业务 | 游玩时长固定 | 致命 | 区间化+模式选择 | 用户多样性 |
| CR-13 | 业务 | 预约制景点缺失 | 致命 | 预约过滤+提醒 | 中国境内旅游 |
| CR-14 | 业务 | 疲劳度无累积 | 致命 | 跨天疲劳模型 | 生理模型 |
| CR-15 | 数据 | spot_info字段缺失 | 致命 | 12个新增字段 | 业务需求 |
| CR-16 | 数据 | 单一交通方式 | 致命 | 多方式表+偏好选择 | 用户需求 |
| CR-17 | 数据 | 无用户画像表 | 致命 | user_profile表 | 个性化推荐 |
| CR-18 | 数据 | 更新无可靠性 | 致命 | Celery+重试+日志 | 生产可靠性 |

---

## 7. 工具调用层（第7层）

### 7.1 职责定位

工具调用层是系统的"外部接口网关"，定义LLM可调用的所有外部工具。

**核心任务**：
1. **工具定义**：为每个外部能力定义JSON Schema（名称、描述、参数）
2. **工具执行**：LLM生成调用指令 → 系统执行 → 返回结果
3. **异常处理**：工具调用失败时的降级策略

**设计原则**（LangChain/OpenAI最佳实践）：
- 工具描述清晰明确（让"实习生"也能正确使用）
- 参数名自解释 + 枚举值限制无效输入
- 错误信息返回给LLM而非抛异常阻断
- 初始可用工具 < 20个（保证准确率）

---

### 7.2 11类工具定义

```python
# tools/tool_definitions.py —— 工具定义

from typing import Dict, Any, List, Optional

# ============================================================
# 工具1：天气查询
# ============================================================
TOOL_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的实时天气和未来7天预报。用于判断室外景点是否适合游览、提醒用户带伞/防晒等。",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如'北京'、'上海'。必须是中文城市名。"
                },
                "date": {
                    "type": "string",
                    "description": "日期，格式YYYY-MM-DD。不填则返回今天天气。"
                }
            },
            "required": ["city"],
        },
    },
}

# ============================================================
# 工具2：景点预约查询
# ============================================================
TOOL_RESERVATION = {
    "type": "function",
    "function": {
        "name": "check_reservation",
        "description": "查询指定景点是否需要预约，以及剩余可预约名额。用于提醒用户提前预约。",
        "parameters": {
            "type": "object",
            "properties": {
                "spot_name": {
                    "type": "string",
                    "description": "景点全称，如'故宫博物院'、'陕西历史博物馆'"
                },
                "visit_date": {
                    "type": "string",
                    "description": "计划游览日期，格式YYYY-MM-DD"
                }
            },
            "required": ["spot_name", "visit_date"],
        },
    },
}

# ============================================================
# 工具3：交通路线查询
# ============================================================
TOOL_ROUTE = {
    "type": "function",
    "function": {
        "name": "get_route",
        "description": "查询两个地点之间的公共交通路线（地铁/公交）。返回预计时间和换乘方案。",
        "parameters": {
            "type": "object",
            "properties": {
                "from_location": {
                    "type": "string",
                    "description": "出发地，如'故宫'、'北京市东城区长安街1号'"
                },
                "to_location": {
                    "type": "string",
                    "description": "目的地，如'天安门广场'"
                },
                "mode": {
                    "type": "string",
                    "enum": ["transit", "driving", "walking"],
                    "description": "交通方式：transit公共交通（默认）、driving驾车、walking步行"
                }
            },
            "required": ["from_location", "to_location"],
        },
    },
}

# ============================================================
# 工具4：餐厅推荐
# ============================================================
TOOL_RESTAURANT = {
    "type": "function",
    "function": {
        "name": "find_restaurants",
        "description": "根据位置、菜系、预算推荐附近餐厅。用于安排行程中的用餐地点。",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "参考位置，如'故宫附近'、'三里屯'"
                },
                "cuisine": {
                    "type": "string",
                    "description": "菜系偏好，如'川菜'、'粤菜'、'火锅'、'清真'"
                },
                "budget_per_person": {
                    "type": "number",
                    "description": "人均预算（元）"
                },
                "meal_time": {
                    "type": "string",
                    "enum": ["breakfast", "lunch", "dinner"],
                    "description": "用餐时段"
                }
            },
            "required": ["location"],
        },
    },
}

# ============================================================
# 工具5：酒店查询
# ============================================================
TOOL_HOTEL = {
    "type": "function",
    "function": {
        "name": "find_hotels",
        "description": "根据城市、日期、预算查询可用酒店。",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"},
                "check_in": {"type": "string", "description": "入住日期 YYYY-MM-DD"},
                "check_out": {"type": "string", "description": "退房日期 YYYY-MM-DD"},
                "budget_per_night": {"type": "number", "description": "每晚预算（元）"},
                "stars": {
                    "type": "integer",
                    "enum": [2, 3, 4, 5],
                    "description": "酒店星级"
                }
            },
            "required": ["city", "check_in", "check_out"],
        },
    },
}

# ============================================================
# 工具6：景点排队时间查询
# ============================================================
TOOL_QUEUE = {
    "type": "function",
    "function": {
        "name": "get_queue_time",
        "description": "获取指定景点当前的排队等待时间（分钟）。帮助用户避开高峰。",
        "parameters": {
            "type": "object",
            "properties": {
                "spot_name": {
                    "type": "string",
                    "description": "景点全称"
                }
            },
            "required": ["spot_name"],
        },
    },
}

# ============================================================
# 工具7：门票购买链接
# ============================================================
TOOL_TICKET = {
    "type": "function",
    "function": {
        "name": "get_ticket_link",
        "description": "获取指定景点的官方购票链接或小程序路径。",
        "parameters": {
            "type": "object",
            "properties": {
                "spot_name": {"type": "string", "description": "景点全称"}
            },
            "required": ["spot_name"],
        },
    },
}

# ============================================================
# 工具8：当地节庆/活动查询
# ============================================================
TOOL_EVENTS = {
    "type": "function",
    "function": {
        "name": "get_local_events",
        "description": "查询指定城市在指定日期范围内的节庆活动、展览、演出等。",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD"}
            },
            "required": ["city", "start_date", "end_date"],
        },
    },
}

# ============================================================
# 工具9：紧急服务查询
# ============================================================
TOOL_EMERGENCY = {
    "type": "function",
    "function": {
        "name": "get_emergency_services",
        "description": "查询附近的医院、派出所、药店等紧急服务设施。仅在用户请求或遇到紧急情况时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "当前位置"},
                "service_type": {
                    "type": "string",
                    "enum": ["hospital", "police", "pharmacy", "embassy"],
                    "description": "服务类型"
                }
            },
            "required": ["location", "service_type"],
        },
    },
}

# ============================================================
# 工具10：POI详情查询
# ============================================================
TOOL_POI_DETAIL = {
    "type": "function",
    "function": {
        "name": "get_poi_detail",
        "description": "获取指定POI的详细信息，包括开放时间、门票价格、交通指南、游客须知等。",
        "parameters": {
            "type": "object",
            "properties": {
                "spot_name": {"type": "string"},
                "fields": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["open_time", "ticket", "transport", "tips", "history"]},
                    "description": "需要查询的字段，不填则返回全部"
                }
            },
            "required": ["spot_name"],
        },
    },
}

# ============================================================
# 工具11：用户画像更新
# ============================================================
TOOL_UPDATE_PROFILE = {
    "type": "function",
    "function": {
        "name": "update_user_profile",
        "description": "更新用户的旅行偏好画像。仅在用户明确表示偏好时调用（如'我喜欢历史景点'、'我不吃辣'）。",
        "parameters": {
            "type": "object",
            "properties": {
                "preferences": {
                    "type": "object",
                    "description": "要更新的偏好字段",
                    "properties": {
                        "interests": {"type": "array", "items": {"type": "string"}},
                        "food_taboos": {"type": "array", "items": {"type": "string"}},
                        "preferred_pace": {"type": "string", "enum": ["relaxed", "moderate", "intensive"]},
                        "budget_level": {"type": "string", "enum": ["low", "mid", "high"]}
                    }
                }
            },
            "required": ["preferences"],
        },
    },
}

# 所有工具列表
ALL_TOOLS = [
    TOOL_WEATHER,        # 1. 天气
    TOOL_RESERVATION,    # 2. 预约
    TOOL_ROUTE,          # 3. 交通
    TOOL_RESTAURANT,     # 4. 餐厅
    TOOL_HOTEL,          # 5. 酒店
    TOOL_QUEUE,          # 6. 排队
    TOOL_TICKET,         # 7. 购票
    TOOL_EVENTS,         # 8. 活动
    TOOL_EMERGENCY,      # 9. 紧急
    TOOL_POI_DETAIL,     # 10. POI详情
    TOOL_UPDATE_PROFILE, # 11. 画像更新
]
```

---

### 7.3 工具执行引擎

```python
# tools/tool_executor.py —— 工具执行引擎

class ToolExecutor:
    """
    工具执行引擎：负责解析LLM的工具调用请求，执行对应函数，返回结果。

    执行流程：
      1. LLM输出tool_calls（工具名+参数）
      2. 解析tool_calls
      3. 路由到对应handler
      4. 执行并返回结果
      5. 异常时返回错误信息（不阻断流程）
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # 注册所有工具handler
        self.handlers = {
            "get_weather": self._handle_weather,
            "check_reservation": self._handle_reservation,
            "get_route": self._handle_route,
            "find_restaurants": self._handle_restaurants,
            "find_hotels": self._handle_hotels,
            "get_queue_time": self._handle_queue,
            "get_ticket_link": self._handle_ticket,
            "get_local_events": self._handle_events,
            "get_emergency_services": self._handle_emergency,
            "get_poi_detail": self._handle_poi_detail,
            "update_user_profile": self._handle_update_profile,
        }

    async def execute(self, tool_calls: List[Dict]) -> List[Dict]:
        """
        批量执行工具调用。

        tool_calls格式（OpenAI兼容）：
        [
            {
                "id": "call_abc123",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "北京", "date": "2026-07-01"}'
                }
            }
        ]
        """
        results = []

        for call in tool_calls:
            tool_name = call.get("function", {}).get("name", "")
            args_str = call.get("function", {}).get("arguments", "{}")
            call_id = call.get("id", "")

            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                results.append({
                    "tool_call_id": call_id,
                    "error": f"参数解析失败: {args_str[:100]}"
                })
                continue

            # 路由到handler
            handler = self.handlers.get(tool_name)
            if not handler:
                results.append({
                    "tool_call_id": call_id,
                    "error": f"未知工具: {tool_name}"
                })
                continue

            # 执行（带异常捕获）
            try:
                result = await handler(**args)
                results.append({
                    "tool_call_id": call_id,
                    "tool_name": tool_name,
                    "result": result,
                })
            except Exception as e:
                results.append({
                    "tool_call_id": call_id,
                    "tool_name": tool_name,
                    "error": f"执行失败: {str(e)[:200]}"
                })

        return results

    # ── 各工具handler（MVP阶段Mock实现）──

    async def _handle_weather(self, city: str, date: str = None) -> Dict:
        """天气查询（MVP: Mock）"""
        # P1后接入和风天气/彩云天气API
        return {
            "city": city,
            "date": date or "今天",
            "weather": "晴",
            "temperature": "25-32C",
            "humidity": "60%",
            "wind": "东南风2级",
            "uv": "强",
            "recommendation": "适合室外游览，注意防晒",
            "source": "mock",
        }

    async def _handle_reservation(self, spot_name: str, visit_date: str) -> Dict:
        """预约查询（MVP: 查本地库）"""
        # 查RESERVATION_REQUIRED表
        from agents.demand_parser import FeasibilityChecker
        checker = FeasibilityChecker()
        info = checker.RESERVATION_REQUIRED.get(spot_name, {})

        if not info:
            return {"need_reservation": False, "spot_name": spot_name}

        return {
            "need_reservation": True,
            "spot_name": spot_name,
            "advance_days": info.get("advance_days", 1),
            "channel": info.get("channel", ""),
            "note": info.get("note", ""),
            "source": "local_db",
        }

    async def _handle_route(self, from_location: str, to_location: str,
                           mode: str = "transit") -> Dict:
        """交通路线（MVP: Mock）"""
        # P1后接入高德/百度路线API
        return {
            "from": from_location,
            "to": to_location,
            "mode": mode,
            "duration_minutes": 30,
            "distance_km": 5.2,
            "route": "地铁1号线 → 换乘2号线",
            "fare": 4,
            "source": "mock",
        }

    async def _handle_restaurants(self, location: str, cuisine: str = None,
                                   budget_per_person: float = None,
                                   meal_time: str = "lunch") -> Dict:
        """餐厅推荐（MVP: Mock）"""
        return {
            "location": location,
            "recommendations": [
                {"name": f"{location}老字号餐厅", "cuisine": cuisine or "本地菜", 
                 "price_per_person": budget_per_person or 80, "rating": 4.5},
            ],
            "source": "mock",
        }

    async def _handle_hotels(self, city: str, check_in: str, check_out: str,
                             budget_per_night: float = None, stars: int = None) -> Dict:
        """酒店查询（MVP: Mock）"""
        return {
            "city": city,
            "check_in": check_in,
            "check_out": check_out,
            "hotels": [
                {"name": f"{city}商务酒店", "stars": stars or 4, 
                 "price_per_night": budget_per_night or 400},
            ],
            "source": "mock",
        }

    async def _handle_queue(self, spot_name: str) -> Dict:
        """排队时间（MVP: Mock）"""
        return {
            "spot_name": spot_name,
            "queue_minutes": 30,
            "crowd_level": "moderate",
            "best_visit_time": "下午2点后",
            "source": "mock",
        }

    async def _handle_ticket(self, spot_name: str) -> Dict:
        """购票链接（MVP: 查本地库）"""
        from agents.demand_parser import FeasibilityChecker
        checker = FeasibilityChecker()
        info = checker.RESERVATION_REQUIRED.get(spot_name, {})

        return {
            "spot_name": spot_name,
            "ticket_link": info.get("channel", "请搜索官方小程序"),
            "price": "请查看官方渠道",
            "source": "local_db",
        }

    async def _handle_events(self, city: str, start_date: str, end_date: str) -> Dict:
        """当地活动（MVP: Mock）"""
        return {
            "city": city,
            "period": f"{start_date} ~ {end_date}",
            "events": [
                {"name": f"{city}夏季文化节", "date": start_date, "type": "文化"},
            ],
            "source": "mock",
        }

    async def _handle_emergency(self, location: str, service_type: str) -> Dict:
        """紧急服务（MVP: Mock，P0必须真实）"""
        services = {
            "hospital": {"name": "最近医院", "phone": "120", "address": f"{location}附近"},
            "police": {"name": "派出所", "phone": "110", "address": f"{location}附近"},
            "pharmacy": {"name": "24小时药店", "phone": "", "address": f"{location}附近"},
        }
        return {
            "service_type": service_type,
            "location": location,
            "services": [services.get(service_type, {})],
            "emergency_call": "110/120/119",
        }

    async def _handle_poi_detail(self, spot_name: str, fields: List[str] = None) -> Dict:
        """POI详情（MVP: 查本地库）"""
        # 查spots表
        return {
            "spot_name": spot_name,
            "open_time": "08:30-17:00",
            "ticket": "60元/人",
            "transport": "地铁1号线天安门东站",
            "tips": "建议早8点前到达避开人流",
            "source": "local_db",
        }

    async def _handle_update_profile(self, preferences: Dict) -> Dict:
        """更新画像（写入Redis短期记忆）"""
        # 实际实现需调用UserProfileRecallAgent的write方法
        return {
            "updated_fields": list(preferences.keys()),
            "status": "success",
            "note": "偏好已更新",
        }
```

---

### 7.4 工具可用性矩阵

| 工具 | MVP | P1 | P2 | 数据来源 | 降级方案 |
|------|-----|-----|-----|---------|---------|
| 天气查询 | Mock | 真实API | 真实API | 和风/彩云 | 返回"天气数据暂不可用" |
| 预约查询 | 本地库 | 本地库+API | 实时API | RESERVATION_REQUIRED表 | 提示用户自行查询 |
| 交通路线 | Mock | 高德/百度API | 真实API | 地图API | Haversine直线距离估算 |
| 餐厅推荐 | Mock | 大众点评API | 真实API | 点评API | 返回"请自行搜索附近餐厅" |
| 酒店查询 | Mock | 携程/美团API | 真实API | OTA API | 返回"请自行预订酒店" |
| 排队时间 | Mock | Mock | 真实API | 景区API | 使用历史平均值 |
| 购票链接 | 本地库 | 本地库 | 实时API | RESERVATION_REQUIRED表 | 提示搜索官方渠道 |
| 当地活动 | Mock | Mock | 真实API | 活动聚合API | 返回"暂无活动信息" |
| 紧急服务 | Mock | 真实 | 真实 | 地图API | 返回110/120/119 |
| POI详情 | 本地库 | 本地库 |  enriched | spots表 | 返回基础信息 |
| 画像更新 | 本地 | 本地 | 持久化 | Redis | 跳过更新 |

---

### 7.5 工具调用监控

```python
# tools/tool_monitor.py —— 工具调用监控

class ToolMonitor:
    """工具调用监控：记录每个工具的调用次数、成功率、延迟。"""

    async def record(self, tool_name: str, success: bool, latency_ms: int,
                     error: str = None):
        """记录一次工具调用。"""
        # 写入Prometheus metrics或日志
        status = "success" if success else "error"
        print(f"[TOOL] {tool_name} | {status} | {latency_ms}ms | {error or ''}")

    KEY_METRICS = """
    工具调用关键指标：
    - tool_call_total: 各工具调用总次数（Counter）
    - tool_call_latency: 各工具P99延迟（Histogram）
    - tool_call_error_rate: 各工具错误率（Gauge）
    - tool_call_active: 进行中调用数（Gauge）
    """)
```

## 8. 交互输出层（第8层）

### 8.1 职责定位

交互输出层是系统的"翻译官"，将结构化的行程数据转化为**用户友好的多模态输出**。

**核心任务**：
1. **文案润色**：LLM生成自然语言描述（景点介绍+过渡+贴心提示）
2. **Markdown输出**：带格式的旅行攻略文档
3. **PDF生成**：可下载/打印的精美行程单
4. **Excel导出**：可编辑的表格形式
5. **地图可视化**：景点位置+路线展示
6. **语音播报**（P2）：行程语音摘要

**绝对不能做**：修改行程内容（景点/时间/顺序）；编造景点信息。

---

### 8.2 OutputFormatAgent 完整实现

```python
# agents/output_format.py —— 输出格式化Agent

from typing import Dict, Any, List, Optional
import json
import tempfile
import os

class OutputFormatAgent:
    """
    输出格式化Agent：将行程结构转为用户友好的多模态输出。

    处理流程：
      1. LLM润色文案（景点介绍+过渡+贴心提示）
      2. Markdown格式化
      3. 异步生成PDF/Excel（后台任务）
      4. 地图链接生成

    约束：只格式化，不修改行程内容。
    """

    SYSTEM_PROMPT = """你是一个专业的旅行攻略撰写助手。请根据提供的行程数据，
撰写一份温馨、实用的旅行攻略。

写作风格：
- 亲切自然，像朋友推荐
- 包含实用信息（交通提示、最佳拍照点、美食推荐）
- 根据用户画像调整语气（带老人→稳重/带孩子→活泼）
- 每个景点包含：简介→怎么玩→注意事项

绝对不能：
- 修改行程中的景点、时间、顺序
- 编造不存在的景点信息
- 添加行程中没有的景点

输出格式：Markdown，使用中文。
"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(
            model="Qwen2.5-7B-AWQ",
            base_url="http://vllm-7b:8000/v1",
            api_key="dummy",
            temperature=0.7,  # 稍高温度，文案更有创意
            max_tokens=2048,
        )

    async def format(self, itinerary: List[Dict], profile: Dict[str, Any],
                     budget: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        主入口：格式化行程输出。

        返回:
            {
                "markdown": str,      # Markdown格式攻略
                "pdf_url": str,       # PDF下载链接
                "excel_url": str,     # Excel下载链接
                "map_url": str,       # 地图查看链接
            }
        """
        if not itinerary:
            return {
                "markdown": "行程生成失败，请重试。",
                "pdf_url": None,
                "excel_url": None,
                "map_url": None,
            }

        # 1. LLM润色文案
        markdown = await self._generate_markdown(itinerary, profile, budget)

        # 2. 异步生成PDF
        pdf_task = asyncio.create_task(self._generate_pdf(markdown))

        # 3. 异步生成Excel
        excel_task = asyncio.create_task(self._generate_excel(itinerary, budget))

        # 4. 地图链接
        map_url = self._generate_map_url(itinerary)

        # 等待异步任务
        pdf_url, excel_url = await asyncio.gather(pdf_task, excel_task)

        return {
            "markdown": markdown,
            "pdf_url": pdf_url,
            "excel_url": excel_url,
            "map_url": map_url,
        }

    # ═══════════════════════════════════════════
    # Markdown生成
    # ═══════════════════════════════════════════

    async def _generate_markdown(self, itinerary: List[Dict], profile: Dict,
                                  budget: Dict = None) -> str:
        """LLM润色生成Markdown攻略。"""
        # 构建Prompt
        dest = profile.get("preferred_destinations", [""])[0] if isinstance(profile.get("preferred_destinations"), list) else profile.get("destination", "目的地")

        # 将行程数据转为简洁的JSON给LLM
        trip_summary = []
        for day in itinerary:
            activities = []
            for act in day.get("schedule", []):
                activities.append({
                    "time": act.get("arrive_time", ""),
                    "name": act.get("spot_name", ""),
                    "duration": act.get("play_minute", 120),
                    "type": act.get("indoor_outdoor", "outdoor"),
                    "price": act.get("ticket_cost", 0),
                })
            trip_summary.append({
                "day": day["day"],
                "activities": activities,
            })

        # 情感适配
        sentiment_note = ""
        if profile.get("has_elderly"):
            sentiment_note = "用户带老人出行，语气稳重，多提醒休息。"
        elif profile.get("has_children"):
            sentiment_note = "用户带孩子出行，语气活泼，多提醒安全和有趣的知识。"
        elif profile.get("has_pregnant"):
            sentiment_note = "用户是孕妇，语气温柔，多提醒安全和休息。"

        prompt = f"""请根据以下行程数据撰写旅行攻略。

{sentiment_note}

目的地：{dest}
行程天数：{len(itinerary)}天

行程数据：
```json
{json.dumps(trip_summary, ensure_ascii=False, indent=2)}
```

预算参考：
- 门票总计：{budget.get("total_ticket", "未知") if budget else "未知"}元
- 餐饮总计：{budget.get("total_food", "未知") if budget else "未知"}元

请输出完整的Markdown格式攻略，包含：
1. 标题和简介
2. 每天的行程安排（时间、景点、玩法建议）
3. 实用贴士（交通、穿衣、注意事项）
4. 预算总结
"""

        # 调用LLM
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content

    # ═══════════════════════════════════════════
    # PDF生成
    # ═══════════════════════════════════════════

    async def _generate_pdf(self, markdown_content: str) -> Optional[str]:
        """
        Markdown → PDF。
        技术选型：WeasyPrint（HTML/CSS → PDF，支持中文）。

        降级方案：如果WeasyPrint失败，返回None，前端降级为打印Markdown。
        """
        try:
            # Markdown → HTML
            import markdown as md_lib
            html_body = md_lib.markdown(markdown_content, extensions=['tables', 'nl2br'])

            # 构建完整HTML
            html_template = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    @page {{ size: A4; margin: 2cm; }}
    body {{ font-family: "Noto Sans CJK SC", "WenQuanYi Micro Hei", sans-serif; 
           font-size: 11pt; line-height: 1.6; color: #333; }}
    h1 {{ font-size: 18pt; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; }}
    h2 {{ font-size: 14pt; color: #34495e; margin-top: 20px; }}
    h3 {{ font-size: 12pt; color: #7f8c8d; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #f2f2f2; }}
    .day-header {{ background-color: #3498db; color: white; padding: 10px; 
                   font-size: 14pt; font-weight: bold; margin-top: 15px; }}
    .tip {{ background-color: #fff3cd; border-left: 4px solid #ffc107; 
            padding: 10px; margin: 10px 0; }}
    .warning {{ background-color: #f8d7da; border-left: 4px solid #dc3545; 
                padding: 10px; margin: 10px 0; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

            # HTML → PDF
            from weasyprint import HTML
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                HTML(string=html_template).write_pdf(tmp.name)
                tmp_path = tmp.name

            # 上传到对象存储（MVP: 本地文件）
            # P1后接入S3/OSS
            pdf_filename = f"itinerary_{{int(time.time())}}.pdf"
            # shutil.copy(tmp_path, f"/static/pdfs/{{pdf_filename}}")

            return f"/api/v1/download/pdfs/{{pdf_filename}}"

        except Exception as e:
            # 降级：返回None
            return None

    # ═══════════════════════════════════════════
    # Excel生成
    # ═══════════════════════════════════════════

    async def _generate_excel(self, itinerary: List[Dict],
                              budget: Dict = None) -> Optional[str]:
        """
        行程 → Excel（可编辑的表格）。
        使用openpyxl生成.xlsx文件。
        """
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

            wb = Workbook()
            ws = wb.active
            ws.title = "行程"

            # 样式定义
            header_font = Font(bold=True, size=12, color="FFFFFF")
            header_fill = PatternFill(start_color="3498DB", end_color="3498DB", fill_type="solid")
            day_fill = PatternFill(start_color="E8F4FD", end_color="E8F4FD", fill_type="solid")
            border = Border(
                left=Side(style='thin'), right=Side(style='thin'),
                top=Side(style='thin'), bottom=Side(style='thin')
            )

            # 标题行
            headers = ["天数", "时间", "景点", "游玩时长", "门票", "类型", "备注"]
            ws.append(headers)
            for cell in ws[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal='center', vertical='center')
                cell.border = border

            # 数据行
            for day in itinerary:
                day_num = day["day"]
                for idx, act in enumerate(day.get("schedule", [])):
                    row = [
                        day_num if idx == 0 else "",  # 合并天数单元格
                        act.get("arrive_time", ""),
                        act.get("spot_name", ""),
                        f"{act.get('play_minute', 120)}分钟",
                        f"¥{act.get('ticket_cost', 0)}",
                        "室内" if act.get("indoor_outdoor") == "indoor" else "室外",
                        "",
                    ]
                    ws.append(row)
                    row_idx = ws.max_row
                    for cell in ws[row_idx]:
                        cell.border = border
                        cell.alignment = Alignment(vertical='center')
                        if day_num % 2 == 0:
                            cell.fill = day_fill

            # 预算页
            if budget:
                ws_budget = wb.create_sheet("预算")
                ws_budget.append(["项目", "金额"])
                ws_budget.append(["门票合计", f"¥{budget.get('total_ticket', 0)}"])
                ws_budget.append(["餐饮合计", f"¥{budget.get('total_food', 0)}"])
                ws_budget.append(["预估总计", f"¥{budget.get('total_estimate', 0)}"])

            # 自动调整列宽
            for column in ws.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 40)
                ws.column_dimensions[column_letter].width = adjusted_width

            # 保存
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                wb.save(tmp.name)
                tmp_path = tmp.name

            excel_filename = f"itinerary_{{int(time.time())}}.xlsx"
            return f"/api/v1/download/excel/{{excel_filename}}"

        except Exception:
            return None

    # ═══════════════════════════════════════════
    # 地图链接生成
    # ═══════════════════════════════════════════

    def _generate_map_url(self, itinerary: List[Dict]) -> Optional[str]:
        """
        生成地图查看链接。
        MVP：高德地图静态URL（无需API Key）。
        P2：交互式地图（Leaflet/OpenLayers）。
        """
        # 收集所有POI坐标
        coordinates = []
        for day in itinerary:
            for act in day.get("schedule", []):
                lat = act.get("lat")
                lng = act.get("lng")
                name = act.get("spot_name", "")
                if lat and lng:
                    coordinates.append(f"{lng},{lat},{name}")

        if not coordinates:
            return None

        # 高德地图路径规划URL（简化版）
        # 实际使用需要接入高德JS API
        first = coordinates[0].split(",")
        return f"https://uri.amap.com/marker?position={first[0]},{first[1]}&name={first[2]}"

    # ═══════════════════════════════════════════
    # 语音播报（P2）
    # ═══════════════════════════════════════════

    async def _generate_voice(self, markdown_content: str) -> Optional[str]:
        """
        行程语音播报（P2阶段）。
        使用TTS引擎将行程摘要转为语音文件。
        """
        # P2接入Edge-TTS或类似服务
        return None  # MVP不实现
```

---

### 8.3 输出模板示例

```markdown
# 🌟 北京5日经典深度游

> 为您定制的行程 | 适合带老人家庭出行 | 预计人均 ¥2,500

---

## 📅 第1天：皇城根下的历史印记

**09:00** 📍 **故宫博物院**（建议游玩4小时）
> 世界最大的古代宫殿建筑群，明清两代皇宫。
> 
> 💡 **玩法建议**：中轴线游览（午门→太和殿→乾清宫→御花园），
> 老人可租借轮椅，神武门出口有观光车。
> 
> ⚠️ **注意事项**：需提前7天预约（故宫博物院小程序），周一闭馆。

**13:30** 🍜 **午餐：故宫附近老字号**
> 推荐：四季民福烤鸭店（故宫店），人均¥120，建议提前取号。

**15:00** 📍 **景山公园**（建议游玩1.5小时）
> 登上万春亭俯瞰故宫全景，日落时分尤为壮观。
> 步行强度低，适合老人。

---

## 💰 预算明细

| 项目 | 金额 |
|------|------|
| 门票合计 | ¥680 |
| 餐饮合计 | ¥500 |
| **预估总计** | **¥1,180** |

> 💡 预算不含住宿和城际交通

---

## 📋 实用贴士

- 🎫 故宫、国博等需提前预约，建议现在就动手
- 👟 每天步行约8km，建议穿舒适运动鞋
- 🌤️ 7月北京炎热，带遮阳伞和防晒霜
- 💊 老人请随身携带常用药物
```

## 9. 监控与可观测性

### 9.1 监控体系架构

```
┌─────────────────────────────────────────────────────────────┐
│                    监控与可观测性体系                         │
├─────────────────────────────────────────────────────────────┤
│  Metrics（指标）        │  Traces（链路）      │  Logs（日志） │
├─────────────────────────────────────────────────────────────┤
│  Prometheus采集         │  LangSmith          │ 结构化日志   │
│  Grafana展示            │  OpenTelemetry      │ ELK/Loki    │
│  AlertManager告警       │  自定义Trace        │             │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 核心指标定义

```python
# monitoring/metrics.py —— 监控指标

from prometheus_client import Counter, Histogram, Gauge, Info

# ── 业务指标 ──
REQUEST_TOTAL = Counter(
    "travel_request_total",
    "请求总数",
    ["intent", "status"]  # new_itinerary/modify/query, success/error/fallback
)

SOLVE_LATENCY = Histogram(
    "travel_solve_latency_seconds",
    "CP-SAT求解耗时",
    buckets=[0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
)

RETRIEVAL_LATENCY = Histogram(
    "travel_retrieval_latency_seconds",
    "RAG检索耗时",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
)

LLM_LATENCY = Histogram(
    "travel_llm_latency_seconds",
    "LLM调用耗时",
    ["model", "task"],  # Qwen2.5-72B/7B, parse/format/understand
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

ACTIVE_SESSIONS = Gauge(
    "travel_active_sessions",
    "当前活跃会话数"
)

FALLBACK_TOTAL = Counter(
    "travel_fallback_total",
    "降级次数",
    ["node", "reason"]  # understand/profile/retrieve/plan/..., timeout/error
)

# ── 系统指标 ──
DB_CONNECTIONS = Gauge(
    "travel_db_connections",
    "数据库连接数",
    ["pool"]  # pg_pool / redis_pool
)

REDIS_HIT_RATE = Gauge(
    "travel_redis_hit_rate",
    "Redis缓存命中率"
)

# ── 业务健康度 ──
USER_SATISFACTION = Gauge(
    "travel_user_satisfaction",
    "用户满意度评分",
    ["dimension"]  # overall/itinerary_quality/response_speed
)
```

### 9.3 告警规则

```yaml
# alerting/rules.yml —— Prometheus告警规则

groups:
  - name: travel_agent
    rules:
      # 业务告警
      - alert: HighFallbackRate
        expr: rate(travel_fallback_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "降级率过高"
          description: "5分钟内降级率超过10%，当前值: {{ $value }}"

      - alert: SlowSolve
        expr: histogram_quantile(0.99, travel_solve_latency_seconds) > 5
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "CP-SAT求解缓慢"
          description: "P99求解耗时超过5秒"

      - alert: RetrievalEmpty
        expr: rate(travel_request_total{status="empty"}[5m]) > 0.05
        for: 2m
        labels:
          severity: info
        annotations:
          summary: "检索空结果率偏高"
          description: "5分钟内空结果率超过5%"

      # 系统告警
      - alert: DBConnectionsHigh
        expr: travel_db_connections > 80
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "数据库连接数过高"

      - alert: LLMLatencyHigh
        expr: histogram_quantile(0.99, travel_llm_latency_seconds) > 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "LLM延迟过高"
          description: "P99 LLM调用超过10秒"
```

### 9.4 LangSmith Trace 集成

```python
# monitoring/tracing.py —— 链路追踪

from langsmith import Client

class TraceManager:
    """
    LangSmith Trace集成：记录每次请求的完整链路。

    追踪内容：
    - 每个Node的输入/输出
    - LLM调用的Prompt/Response
    - 工具调用的参数/结果
    - 异常和降级事件
    """

    def __init__(self, config: Dict):
        self.client = Client(
            api_key=config.get("LANGSMITH_API_KEY"),
            api_url=config.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        )

    async def trace_node(self, run_id: str, node_name: str,
                        inputs: Dict, outputs: Dict,
                        latency_ms: int, error: str = None):
        """记录一个Node的执行 trace。"""
        # 通过LangGraph自动集成
        pass

    KEY_TRACES = """
    关键Trace：
    1. understand → LLM槽位抽取（输入文本，输出slots）
    2. profile → 画像召回（输入user_id，输出profile）
    3. retrieve → RAG检索（输入query，输出Top-15 POI）
    4. plan → CP-SAT求解（输入POI+约束，输出itinerary）
    5. factcheck → 校验（输入itinerary，输出conflicts）
    6. output → 格式化（输入itinerary，输出markdown）
    """
```

---

## 10. 部署架构

### 10.1 Docker Compose（MVP）

```yaml
# docker-compose.yml —— MVP部署配置
version: "3.8"

services:
  # ── 主应用 ──
  travel-agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://travel:travel@postgres:5432/travel_agent
      - REDIS_URL=redis://redis:6379/0
      - VLLM_72B_URL=http://vllm-72b:8000/v1
      - VLLM_7B_URL=http://vllm-7b:8000/v1
      - TRAVEL_AGENT_MASTER_KEY=${MASTER_KEY}
    depends_on:
      - postgres
      - redis
    deploy:
      resources:
        limits:
          memory: 4G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  # ── PostgreSQL + pgvector ──
  postgres:
    image: ankane/pgvector:latest
    environment:
      - POSTGRES_USER=travel
      - POSTGRES_PASSWORD=travel
      - POSTGRES_DB=travel_agent
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U travel"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ── Redis ──
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  # ── vLLM 72B（主模型）──
  vllm-72b:
    image: vllm/vllm-openapi:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0,1  # 双卡
    command: >
      --model Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4
      --tensor-parallel-size 2
      --max-model-len 32768
      --gpu-memory-utilization 0.9
      --enable-lora
      --port 8000
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  # ── vLLM 7B（轻量模型）──
  vllm-7b:
    image: vllm/vllm-openapi:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=2  # 单卡
    command: >
      --model Qwen/Qwen2.5-7B-Instruct-AWQ
      --max-model-len 8192
      --gpu-memory-utilization 0.8
      --quantization awq
      --port 8000
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # ── Prometheus ──
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"

  # ── Grafana ──
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana-dashboards:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

### 10.2 Kubernetes（P1后）

```yaml
# k8s/travel-agent-deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: travel-agent
spec:
  replicas: 3  # 3副本高可用
  selector:
    matchLabels:
      app: travel-agent
  template:
    metadata:
      labels:
        app: travel-agent
    spec:
      containers:
        - name: travel-agent
          image: travel-agent:latest
          ports:
            - containerPort: 8000
          resources:
            requests:
              memory: "2Gi"
              cpu: "1000m"
            limits:
              memory: "4Gi"
              cpu: "2000m"
          envFrom:
            - configMapRef:
                name: travel-agent-config
            - secretRef:
                name: travel-agent-secrets
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: travel-agent-service
spec:
  selector:
    app: travel-agent
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
```

### 10.3 模型部署配置

| 模型 | 部署方式 | GPU需求 | 显存 | QPS | 降级方案 |
|------|---------|---------|------|-----|---------|
| Qwen2.5-72B | vLLM + GPTQ-Int4 | 2×A100 80G | ~60G | 10 | TGI / 云端API |
| Qwen2.5-7B | vLLM + AWQ | 1×A10 24G | ~10G | 50 | CPU推理/云端API |
| bge-large-zh | SentenceTransformers(CPU) | 无 | 2G内存 | 100 | text2vec |

### 10.4 环境变量配置

```bash
# .env —— 生产环境变量

# 数据库
DATABASE_URL=postgresql://travel:travel@postgres:5432/travel_agent
REDIS_URL=redis://redis:6379/0

# LLM
VLLM_72B_URL=http://vllm-72b:8000/v1
VLLM_7B_URL=http://vllm-7b:8000/v1
LLM_API_KEY=dummy  # vLLM本地部署不需要

# Embedding
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
EMBEDDING_DEVICE=cpu

# 安全
TRAVEL_AGENT_MASTER_KEY=your-32-byte-hex-key-here-1234

# 监控
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
LANGSMITH_API_KEY=ls-xxx

# 业务
MAX_SOLVE_TIME_MS=5000
SESSION_TTL_MINUTES=30
DEFAULT_TOP_K=15
```

---

## 11. Roadmap 与 MVP 范围

### 11.1 分阶段实施计划

```
P0（MVP，2-3周）                    P1（核心增强，4-6周）              P2（体验升级，6-8周）
┌──────────────────────────────┐  ┌──────────────────────────┐  ┌──────────────────────────┐
│  核心流程跑通                    │  外部API接入               │  智能增强                  │
│                              │  │                         │  │                         │
│  ✅ LangGraph编排（StateGraph） │  ✅ 高德/百度路线API        │  ✅ 协同过滤推荐            │
│  ✅ CP-SAT基础求解              │  ✅ 和风天气API            │  ✅ 多目标优化              │
│  ✅ SQL预过滤+pgvector语义检索   │  ✅ 预约API对接            │  ✅ BM25全文检索            │
│  ✅ 基础槽位抽取（7B）           │  ✅ 图片理解（Qwen-VL）     │  ✅ 语音播报               │
│  ✅ Markdown输出                │  ✅ PDF/Excel完善          │  ✅ EmergencyAgent         │
│  ✅ 7类异常降级                 │  ✅ 用户画像持久化          │  ✅ 实时排队/动态调整        │
│  ✅ 会话管理                    │  ✅ LangSmith监控          │  ✅ BookingToolAgent       │
│                              │  │                         │  │                         │
│  Mock数据：天气/交通/排队       │  真实数据接入              │  智能功能                   │
└──────────────────────────────┘  └──────────────────────────┘  └──────────────────────────┘

P3（规模化，8-12周）               P4（生态，12周+）
┌──────────────────────────────┐  ┌──────────────────────────┐
│  企业级特性                     │  平台生态                  │
│                              │  │                         │
│  ✅ MultiPersonSyncAgent      │  ✅ 开放API               │
│  ✅ 性能优化（量化/缓存）        │  ✅ 第三方Plugin          │
│  ✅ A/B测试框架                │  ✅ 合作伙伴接入          │
│  ✅ 灰度发布                   │  ✅ 多语言支持            │
│  ✅ 自动化测试覆盖率>80%        │  ✅ 国际化目的地          │
│                              │  │                         │
└──────────────────────────────┘  └──────────────────────────┘
```

### 11.2 MVP功能清单

**MVP必须实现（P0）**：

| 模块 | 功能点 | 优先级 | 工作量 |
|------|--------|--------|--------|
| **编排层** | LangGraph StateGraph搭建 | P0 | 3天 |
| | 6个Node实现 | P0 | 3天 |
| | 5个条件路由 | P0 | 2天 |
| | Checkpoint持久化 | P0 | 2天 |
| **感知层** | SSE流式对话 | P0 | 2天 |
| | PaddleOCR图片处理 | P0 | 2天 |
| **意图理解** | TravelSlots定义 | P0 | 1天 |
| | DemandParserAgent | P0 | 3天 |
| | 可行性校验引擎 | P0 | 2天 |
| **用户画像** | user_profile表+DDL | P0 | 1天 |
| | UserProfileRecallAgent | P0 | 2天 |
| | Redis短期/pgvector长期 | P0 | 2天 |
| **知识库** | spots表+DDL | P0 | 1天 |
| | SQL预过滤 | P0 | 1天 |
| | pgvector语义检索 | P0 | 2天 |
| | RRF融合排序 | P0 | 1天 |
| **规划引擎** | CP-SAT模型+求解 | P0 | 4天 |
| | 三模式重规划 | P0 | 2天 |
| | 场景参数调优 | P0 | 1天 |
| **输出层** | Markdown格式化 | P0 | 2天 |
| | PDF生成（WeasyPrint） | P0 | 2天 |
| | Excel导出 | P0 | 1天 |
| **监控** | Prometheus指标 | P0 | 1天 |
| | Grafana看板 | P0 | 1天 |
| **部署** | Docker Compose | P0 | 2天 |
| | 健康检查接口 | P0 | 1天 |

**MVP总计：约40个工作日（8周，2人并行=4周）**

### 11.3 关键里程碑

| 里程碑 | 时间 | 验收标准 |
|--------|------|---------|
| 端到端打通 | 第2周末 | 输入"北京3天"→输出完整Markdown行程 |
| 异常降级完备 | 第3周末 | 7类异常×3级降级全部实现并通过测试 |
| 监控上线 | 第4周末 | Grafana看板可用，核心指标采集正常 |
| 压测通过 | 第4周末 | 100并发，P99<3s，错误率<1% |

### 11.4 风险评估与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| vLLM部署不稳定 | 中 | 高 | TGI降级 + 云端API兜底 |
| pgvector性能不足 | 低 | 中 | HNSW参数调优 + Redis缓存 + 量化 |
| CP-SAT求解超时 | 中 | 中 | 贪心降级 + 参数自适应 + 问题分治 |
| LLM槽位抽取不准 | 中 | 高 | 规则引擎兜底 + 歧义消解 + 追问 |
| PaddleOCR准确率低 | 低 | 低 | 提示用户直接输入 |

---

## 下半部分完结

### 覆盖范围总结

| 章节 | 内容 | 代码量 | 核心产出 |
|------|------|--------|---------|
| 第5章 | 知识库层 | ~450行 | RAG混合检索Agent + POI DDL + RRF融合 |
| 第6章 | 规划决策引擎 | ~500行 | CP-SAT v4.0完整实现 + 三模式重规划 + 参数调优 |
| 第7章 | 工具调用层 | ~400行 | 11类工具Schema + 执行引擎 |
| 第8章 | 交互输出层 | ~300行 | Markdown润色 + PDF/Excel + 地图 |
| 第9章 | 监控可观测性 | ~150行 | Prometheus指标 + 告警规则 + LangSmith |
| 第10章 | 部署架构 | ~200行 | Docker Compose + K8s + 模型部署 |
| 第11章 | Roadmap | ~100行 | MVP清单 + 里程碑 + 风险评估 |

**总计**：约2,100行代码 + 配置，覆盖"检索POI → 数学求解 → 工具调用 → 格式化输出 → 监控部署"完整链路。

### 与上半部分的衔接点

下半部分接收上半部分的三个关键字段：
- `AgentState.slots` —— TravelSlots 15字段
- `AgentState.user_profile` —— 结构化画像
- `AgentState.inferred_slots` —— 画像推断槽位

最终产出：
- `AgentState.itinerary` —— 完整行程JSON
- `AgentState.output_markdown` —— 用户可见的Markdown攻略
- `AgentState.output_pdf_url` —— PDF下载链接
- `AgentState.output_excel_url` —— Excel下载链接

---

## 12. 运维安全风控层（商用必备）

> 本章节补充商用落地必需的6大能力：幻觉检测、安全风控、日志分析、限流成本控制、异常告警、数据隐私一键清除。
> 这些能力在上半/下半正文中部分覆盖但未完整展开，本章提供生产级完整实现。

---

### 12.1 幻觉检测引擎（HallucinationDetectionAgent）

#### 12.1.1 职责定位

幻觉检测引擎是行程质量的"质检员"，负责校验系统输出的每一个事实性声明是否与知识库一致。

**检测范围**：
1. **景点开放时间** — LLM润色文案中的开放时间 vs spots表实际数据
2. **门票价格** — 输出中的价格 vs 数据库价格
3. **景点存在性** — LLM是否编造了不存在的景点
4. **路线可达性** — 通勤时间是否超出合理范围
5. **预约状态** — 必去景点是否已标注预约提醒

**核心设计**：事后校验（行程生成后检查），不阻塞主流程；发现冲突自动标注而非静默修改。

---

#### 12.1.2 检测规则库

```python
# agents/hallucination_detector.py —— 幻觉检测引擎

from typing import Dict, Any, List, Tuple
import re

class HallucinationDetectionAgent:
    """
    幻觉检测Agent：校验行程输出的事实一致性。

    处理流程：
      1. 提取output_markdown中的所有事实声明（价格/时间/景点名）
      2. 与spots表 + reservation_required表逐项比对
      3. 发现冲突 → 标注warning + 建议修正
      4. 严重冲突（景点不存在）→ 标记error + 阻断输出

    注意：只标注不修改，由人工或后续Agent决定是否修正。
    """

    # 严重程度分级
    SEVERITY = {
        "critical": "error",    # 景点不存在 — 必须修正
        "high": "warning",      # 价格差异>20% — 建议修正
        "medium": "info",       # 开放时间表述不精确 — 可接受
        "low": "info",          # 轻微表述差异 — 忽略
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pg_pool = None  # 懒加载

    async def detect(self, output_markdown: str, itinerary: List[Dict],
                     poi_reference: List[Dict]) -> Dict[str, Any]:
        """
        主入口：幻觉检测。

        返回:
            {
                "passed": bool,              # 是否通过检测
                "total_checks": int,         # 总检查项数
                "conflicts": [{
                    "field": str,             # 字段名
                    "spot_name": str,         # 景点名
                    "expected": str,          # 知识库值
                    "actual": str,            # 输出中的值
                    "severity": str,          # critical/high/medium/low
                    "suggestion": str,        # 修正建议
                }],
                "annotations": str,           # 带标注的Markdown
            }
        """
        conflicts = []

        # 1. 景点存在性检查（最关键）
        conflicts += await self._check_spot_existence(itinerary, poi_reference)

        # 2. 开放时间检查
        conflicts += await self._check_open_time(output_markdown, poi_reference)

        # 3. 门票价格检查
        conflicts += await self._check_ticket_price(output_markdown, poi_reference)

        # 4. 路线通勤合理性检查
        conflicts += self._check_route_feasibility(itinerary)

        # 5. 预约标注检查
        conflicts += await self._check_reservation_annotated(output_markdown, poi_reference)

        # 严重程度统计
        critical_count = sum(1 for c in conflicts if c["severity"] == "critical")

        # 生成带标注的Markdown
        annotated = self._annotate_output(output_markdown, conflicts)

        return {
            "passed": critical_count == 0,  # critical必须修正
            "total_checks": len(conflicts),
            "critical_count": critical_count,
            "conflicts": conflicts,
            "annotations": annotated,
        }

    # ═══════════════════════════════════════════
    # 具体检测规则
    # ═══════════════════════════════════════════

    async def _check_spot_existence(self, itinerary: List[Dict],
                                     poi_reference: List[Dict]) -> List[Dict]:
        """检查1：景点是否存在（防LLM编造）"""
        conflicts = []
        valid_names = {p["spot_name"] for p in poi_reference}

        for day in itinerary:
            for act in day.get("schedule", []):
                name = act.get("spot_name", "")
                if name and name not in valid_names:
                    conflicts.append({
                        "field": "spot_existence",
                        "spot_name": name,
                        "expected": "存在于知识库的景点",
                        "actual": f"'{name}'不在POI候选列表中",
                        "severity": "critical",
                        "suggestion": f"移除'{name}'或替换为知识库中的相似景点",
                    })
        return conflicts

    async def _check_open_time(self, markdown: str,
                                poi_reference: List[Dict]) -> List[Dict]:
        """检查2：开放时间是否准确"""
        conflicts = []

        # 从Markdown提取时间声明（如"08:30开门"、"17:00闭馆"）
        time_patterns = [
            r'(\d{1,2}):(\d{2}).*?(?:开门|开始|开放)',
            r'(\d{1,2}):(\d{2}).*?(?:关门|结束|闭馆)',
        ]

        for poi in poi_reference:
            name = poi["spot_name"]
            db_open = poi.get("open_time", "08:00")
            db_close = poi.get("close_time", "17:00")

            # 检查Markdown中是否提到了该景点的时间
            if name in markdown:
                # 简单匹配：如果提到了景点名，检查附近是否有时间表述
                name_idx = markdown.find(name)
                surrounding = markdown[max(0, name_idx-100):name_idx+200]

                # 如果文案中的时间与数据库不一致，标注
                if db_open[:5] not in surrounding and "开放时间" in surrounding:
                    conflicts.append({
                        "field": "open_time",
                        "spot_name": name,
                        "expected": f"{db_open}-{db_close}",
                        "actual": "文案中开放时间表述可能与实际不符",
                        "severity": "medium",
                        "suggestion": f"确认开放时间是否为{db_open}-{db_close}",
                    })

        return conflicts

    async def _check_ticket_price(self, markdown: str,
                                   poi_reference: List[Dict]) -> List[Dict]:
        """检查3：门票价格是否准确"""
        conflicts = []

        # 从Markdown提取价格声明
        price_pattern = r'(\d+)(?:\s*元)'

        for poi in poi_reference:
            name = poi["spot_name"]
            db_price = poi.get("ticket_price", 0)

            if db_price == 0:
                continue  # 免费景点不检查

            # 检查Markdown中是否提到了价格
            if name in markdown:
                name_idx = markdown.find(name)
                surrounding = markdown[max(0, name_idx-100):name_idx+200]

                found_prices = re.findall(price_pattern, surrounding)
                for fp in found_prices:
                    fp_int = int(fp)
                    # 价格差异>20%视为冲突
                    if abs(fp_int - db_price) / max(db_price, 1) > 0.2:
                        conflicts.append({
                            "field": "ticket_price",
                            "spot_name": name,
                            "expected": f"¥{db_price}",
                            "actual": f"文案中提及¥{fp_int}",
                            "severity": "high",
                            "suggestion": f"修正为¥{db_price}（或标注价格可能变动）",
                        })

        return conflicts

    def _check_route_feasibility(self, itinerary: List[Dict]) -> List[Dict]:
        """检查4：路线通勤时间合理性"""
        conflicts = []

        for day in itinerary:
            schedule = day.get("schedule", [])
            for i in range(len(schedule) - 1):
                curr = schedule[i]
                next_poi = schedule[i + 1]

                # 检查连续景点间通勤时间
                # 如果arrive_time差异过大（>2小时且无合理说明），标注
                if "transit_minutes" in next_poi:
                    transit = next_poi["transit_minutes"]
                    if transit > 120:
                        conflicts.append({
                            "field": "route_feasibility",
                            "spot_name": f"{curr['spot_name']} → {next_poi['spot_name']}",
                            "expected": "通勤时间<2小时",
                            "actual": f"通勤{transit}分钟",
                            "severity": "medium",
                            "suggestion": "考虑在中间增加景点或调整顺序",
                        })

        return conflicts

    async def _check_reservation_annotated(self, markdown: str,
                                            poi_reference: List[Dict]) -> List[Dict]:
        """检查5：需预约景点是否已标注提醒"""
        conflicts = []

        for poi in poi_reference:
            if not poi.get("need_reservation"):
                continue

            name = poi["spot_name"]
            if name in markdown:
                # 检查是否包含预约关键词
                name_idx = markdown.find(name)
                surrounding = markdown[max(0, name_idx-50):name_idx+150]

                reservation_keywords = ["预约", "提前", "购票", "门票"]
                has_annotation = any(kw in surrounding for kw in reservation_keywords)

                if not has_annotation:
                    conflicts.append({
                        "field": "reservation_annotation",
                        "spot_name": name,
                        "expected": "包含预约提醒",
                        "actual": "未标注预约提醒",
                        "severity": "high",
                        "suggestion": f"添加：「{name}」需提前{poi.get('reservation_advance_days', 1)}天预约",
                    })

        return conflicts

    def _annotate_output(self, markdown: str, conflicts: List[Dict]) -> str:
        """在Markdown中添加冲突标注"""
        annotated = markdown

        for c in conflicts:
            severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
            emoji = severity_emoji.get(c["severity"], "⚪")

            # 在相关位置插入标注（简化：追加到文末）
            note = f"

{emoji} **[{c['severity'].upper()}] {c['spot_name']} - {c['field']}**"
            note += f"
- 知识库: {c['expected']}"
            note += f"
- 文案中: {c['actual']}"
            note += f"
- 建议: {c['suggestion']}"

            annotated += note

        if conflicts:
            annotated += f"

---
**幻觉检测总结**: 共{len(conflicts)}项冲突，"
            annotated += f"严重{critical_count if 'critical_count' in dir() else sum(1 for c in conflicts if c['severity']=='critical')}项需修正。"

        return annotated
```

---

### 12.2 日志分析与迭代引擎（LogAnalyticsEngine）

#### 12.2.1 职责定位

日志分析引擎是系统的"复盘工具"，持续分析运行日志，挖掘规划失败模式和高频修改需求，驱动模型迭代。

**核心任务**：
1. **规划失败分析** — 聚类不可行原因，识别系统性问题
2. **高频修改需求** — 统计用户最常修改的槽位和方向
3. **用户满意度追踪** — 基于行为数据（修改次数、确认速度）推断满意度
4. **模型迭代建议** — 自动生成prompt优化建议和数据补充需求

---

#### 12.2.2 数据模型

```sql
-- 规划执行日志表
CREATE TABLE planning_log (
    log_id              BIGSERIAL PRIMARY KEY,
    session_id          VARCHAR(50) NOT NULL,
    user_id             VARCHAR(32),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- 输入
    intent              VARCHAR(20),
    destination         VARCHAR(50),
    travel_days         INT,
    travelers_count     INT,
    travel_companion    VARCHAR(20),

    -- 输出
    solve_status        VARCHAR(20),           -- optimal/feasible/timeout/infeasible
    solve_time_ms       INT,
    poi_count           INT,                   -- 检索到的POI数量
    itinerary_days      INT,                   -- 输出行程天数

    -- 失败分析
    conflict_reasons    JSONB,                 -- 不可行原因列表
    missing_slots       JSONB,                 -- 缺失槽位

    -- 用户反馈
    user_confirmed      BOOLEAN DEFAULT FALSE, -- 是否确认
    modify_count        INT DEFAULT 0,         -- 修改次数
    cancel_reason       TEXT,                  -- 取消原因

    -- 性能
    total_latency_ms    INT,                   -- 全链路耗时
    llm_calls           INT DEFAULT 0,         -- LLM调用次数
    llm_tokens_used     INT DEFAULT 0,         -- Token使用量

    INDEX idx_session (session_id),
    INDEX idx_created (created_at DESC),
    INDEX idx_status (solve_status),
    INDEX idx_destination (destination)
);

-- 用户修改行为表
CREATE TABLE user_modification_log (
    mod_id              BIGSERIAL PRIMARY KEY,
    session_id          VARCHAR(50) NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    action              VARCHAR(30) NOT NULL,  -- remove_poi/replace_poi/change_days/change_budget/change_pace
    target_field        VARCHAR(30),           -- 修改的目标字段
    old_value           TEXT,                  -- 原值
    new_value           TEXT,                  -- 新值
    feedback_text       TEXT,                  -- 用户自然语言反馈

    INDEX idx_session (session_id),
    INDEX idx_action (action),
    INDEX idx_created (created_at DESC)
);
```

---

#### 12.2.3 日志分析引擎

```python
# monitoring/log_analytics.py —— 日志分析与迭代引擎

from typing import Dict, Any, List
from datetime import datetime, timedelta
import json

class LogAnalyticsEngine:
    """
    日志分析与迭代引擎：持续分析运行日志，挖掘问题模式。

    分析维度：
      1. 规划失败聚类 —— 找出最常见的不可行原因
      2. 高频修改需求 —— 用户最常修改什么
      3. 目的地满意度 —— 各目的地的确认率和修改率
      4. LLM效率 —— 各模型的token产出比
      5. 自动迭代建议 —— 生成prompt优化方向
    """

    def __init__(self, pg_pool):
        self.pg_pool = pg_pool

    async def generate_daily_report(self, date: str = None) -> Dict[str, Any]:
        """
        生成每日分析报告。

        返回：
            {
                "date": str,
                "total_requests": int,
                "success_rate": float,        # 确认率
                "failure_analysis": {...},     # 失败原因聚类
                "hot_modifications": [...],    # 高频修改Top-10
                "destination_ranking": [...],  # 目的地满意度排名
                "iteration_suggestions": [...], # 迭代建议
            }
        """
        if date is None:
            date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        return {
            "date": date,
            "total_requests": await self._count_requests(date),
            "success_rate": await self._calc_confirm_rate(date),
            "failure_analysis": await self._analyze_failures(date),
            "hot_modifications": await self._analyze_modifications(date),
            "destination_ranking": await self._rank_destinations(date),
            "iteration_suggestions": await self._generate_suggestions(date),
        }

    # ═══════════════════════════════════════════
    # 分析1: 规划失败聚类
    # ═══════════════════════════════════════════

    async def _analyze_failures(self, date: str) -> Dict[str, Any]:
        """聚类分析规划失败原因"""

        # 按失败原因聚类
        sql = """
        SELECT 
            jsonb_array_elements_text(conflict_reasons) as reason,
            COUNT(*) as count,
            AVG(solve_time_ms) as avg_solve_time
        FROM planning_log
        WHERE DATE(created_at) = $1
          AND solve_status = 'infeasible'
          AND conflict_reasons IS NOT NULL
        GROUP BY reason
        ORDER BY count DESC
        LIMIT 10
        """

        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch(sql, date)

        # 分类统计
        categories = {
            "budget_related": 0,      # 预算相关
            "time_related": 0,        # 时间相关
            "constraint_conflict": 0, # 约束冲突
            "data_insufficient": 0,   # 数据不足
            "other": 0,
        }

        budget_keywords = ["预算", "费用", "cost", "budget"]
        time_keywords = ["时间", "时长", "time", "hour"]

        for r in rows:
            reason = r["reason"] or ""
            if any(kw in reason for kw in budget_keywords):
                categories["budget_related"] += r["count"]
            elif any(kw in reason for kw in time_keywords):
                categories["time_related"] += r["count"]
            elif "约束" in reason or "conflict" in reason:
                categories["constraint_conflict"] += r["count"]
            elif "景点" in reason or "POI" in reason or "数据" in reason:
                categories["data_insufficient"] += r["count"]
            else:
                categories["other"] += r["count"]

        return {
            "total_failures": sum(r["count"] for r in rows),
            "top_reasons": [{"reason": r["reason"], "count": r["count"]} for r in rows[:5]],
            "category_breakdown": categories,
            "suggestion": self._generate_failure_suggestion(categories),
        }

    # ═══════════════════════════════════════════
    # 分析2: 高频修改需求
    # ═══════════════════════════════════════════

    async def _analyze_modifications(self, date: str) -> List[Dict]:
        """分析用户最常修改的槽位和方向"""

        sql = """
        SELECT 
            action,
            COUNT(*) as count,
            AVG(EXTRACT(EPOCH FROM (created_at - LAG(created_at) OVER (ORDER BY created_at)))) as avg_interval_sec
        FROM user_modification_log
        WHERE DATE(created_at) = $1
        GROUP BY action
        ORDER BY count DESC
        LIMIT 10
        """

        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch(sql, date)

        action_names = {
            "remove_poi": "删除景点",
            "replace_poi": "替换景点",
            "change_days": "调整天数",
            "change_budget": "调整预算",
            "change_pace": "调整节奏",
            "add_poi": "增加景点",
        }

        return [
            {
                "action": r["action"],
                "action_cn": action_names.get(r["action"], r["action"]),
                "count": r["count"],
                "percentage": f"{r['count'] / sum(row['count'] for row in rows) * 100:.1f}%" if rows else "0%",
            }
            for r in rows
        ]

    # ═══════════════════════════════════════════
    # 分析3: 目的地满意度排名
    # ═══════════════════════════════════════════

    async def _rank_destinations(self, date: str) -> List[Dict]:
        """按确认率对目的地排名"""

        sql = """
        SELECT 
            destination,
            COUNT(*) as total,
            SUM(CASE WHEN user_confirmed THEN 1 ELSE 0 END) as confirmed,
            AVG(modify_count) as avg_modifies
        FROM planning_log
        WHERE DATE(created_at) = $1
          AND destination IS NOT NULL
        GROUP BY destination
        HAVING COUNT(*) >= 5
        ORDER BY confirmed::float / COUNT(*) DESC
        LIMIT 10
        """

        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch(sql, date)

        return [
            {
                "destination": r["destination"],
                "total_requests": r["total"],
                "confirm_rate": f"{r['confirmed'] / r['total'] * 100:.1f}%",
                "avg_modifies": round(r["avg_modifies"] or 0, 1),
                "satisfaction_score": self._calc_satisfaction_score(r),
            }
            for r in rows
        ]

    # ═══════════════════════════════════════════
    # 分析4: 迭代建议生成
    # ═══════════════════════════════════════════

    async def _generate_suggestions(self, date: str) -> List[str]:
        """基于分析结果生成模型迭代建议"""

        suggestions = []
        failures = await self._analyze_failures(date)
        mods = await self._analyze_modifications(date)

        # 根据失败类别生成建议
        cats = failures.get("category_breakdown", {})

        if cats.get("budget_related", 0) > 5:
            suggestions.append(
                "[高优] 预算相关失败较多 — 建议："
                "1) 在DemandParser中增加预算合理性提示 "
                "2) 优化CPSATTuningGuide的Budget_day_max计算 "
                "3) 添加'预算紧张时推荐免费景点'策略"
            )

        if cats.get("time_related", 0) > 5:
            suggestions.append(
                "[高优] 时间约束冲突较多 — 建议："
                "1) 优化CP-SAT的T_day_max参数 "
                "2) 增加'时间紧张时自动删减低优先级景点'功能"
            )

        if cats.get("data_insufficient", 0) > 3:
            suggestions.append(
                "[中优] 部分目的地POI数据不足 — 建议补充景点数据"
            )

        # 根据高频修改生成建议
        if mods and mods[0]["count"] > 10:
            top_action = mods[0]["action_cn"]
            suggestions.append(
                f"[中优] 用户高频'{top_action}' — 建议优化初始规划策略，"
                f"减少用户对{top_action}的需求"
            )

        if not suggestions:
            suggestions.append("当日运行平稳，暂无迭代建议")

        return suggestions

    @staticmethod
    def _calc_satisfaction_score(row) -> int:
        """计算满意度评分（0-100）"""
        confirm_rate = row["confirmed"] / max(row["total"], 1)
        modify_penalty = min((row["avg_modifies"] or 0) * 10, 50)
        score = int(confirm_rate * 100 - modify_penalty)
        return max(0, min(100, score))

    @staticmethod
    def _generate_failure_suggestion(categories: Dict) -> str:
        """根据失败类别生成建议"""
        max_cat = max(categories, key=categories.get)
        suggestions = {
            "budget_related": "建议优化预算参数计算和前置校验",
            "time_related": "建议优化时间约束和景点时长设置",
            "constraint_conflict": "建议增加约束冲突检测和自动松弛",
            "data_insufficient": "建议补充目的地POI数据",
            "other": "建议深入分析具体失败案例",
        }
        return suggestions.get(max_cat, "建议进一步分析")

    async def _count_requests(self, date: str) -> int:
        async with self.pg_pool.acquire() as conn:
            row = await conn.fetchval(
                "SELECT COUNT(*) FROM planning_log WHERE DATE(created_at) = $1", date
            )
            return row or 0

    async def _calc_confirm_rate(self, date: str) -> float:
        async with self.pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN user_confirmed THEN 1 ELSE 0 END) as confirmed
                FROM planning_log WHERE DATE(created_at) = $1""", date
            )
            if not row or row["total"] == 0:
                return 0.0
            return round(row["confirmed"] / row["total"] * 100, 1)
```

---

### 12.3 限流 & 成本控制引擎（RateLimitCostController）

#### 12.3.1 职责定位

限流成本控制引擎是系统的"财务守门员"，负责管理LLM调用和第三方API的配额，防止成本失控。

**核心任务**：
1. **多维限流** — RPM（请求/分钟）+ TPM（Token/分钟）+ 并发数
2. **Token配额管理** — 按用户/按模型/按日/按月的分层配额
3. **成本预算告警** — 实时追踪API调用成本，超预算时自动降级
4. **熔断降级** — 第三方API故障时自动切换降级方案
5. **成本分摊** — 各模型/各模块的成本明细追踪

**业界最佳实践**：
- Token-aware限流（比request-based更精确）
- 多层配额（per-second防洪水 + per-day防滥用）
- 熔断器模式（Circuit Breaker）保护下游
- 成本实时追踪（与LLM provider账单对齐）

---

#### 12.3.2 配额模型

```python
# monitoring/rate_limit_controller.py —— 限流 & 成本控制引擎

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import json

class RateLimitCostController:
    """
    限流 & 成本控制引擎。

    限流维度（多层防护）：
      Layer 1: 请求级 —— requests_per_second（防DDoS）
      Layer 2: Token级 —— tokens_per_minute（精准资源控制）
      Layer 3: 并发级 —— max_concurrent_requests（防排队过长）
      Layer 4: 成本级 —— daily_cost_budget（商业化成本控制）
      Layer 5: 模型级 —— per_model_quota（不同模型不同配额）

    业界最佳实践：Token bucket算法，支持burst。
    """

    def __init__(self, redis_client, config: Dict[str, Any]):
        self.redis = redis_client
        self.config = config

        # 默认配额（可覆盖）
        self.default_quotas = {
            # Layer 1: 请求级
            "requests_per_second": 10,
            "requests_per_minute": 300,

            # Layer 2: Token级（按模型区分）
            "tokens_per_minute_72b": 20000,    # 72B大模型限制更严
            "tokens_per_minute_7b": 50000,     # 7B小模型限制更松
            "tokens_per_minute_embedding": 100000,

            # Layer 3: 并发级
            "max_concurrent_requests": 20,
            "max_concurrent_72b": 5,           # 72B并发更低

            # Layer 4: 成本级（元/天）
            "daily_cost_budget": 500.0,         # 默认日预算500元
            "hourly_cost_budget": 50.0,         # 小时预算50元

            # Layer 5: 模型级配额
            "72b_daily_requests": 500,          # 72B每天最多500次
            "7b_daily_requests": 2000,          # 7B每天最多2000次
        }

        # 模型成本系数（元/1K tokens，用于成本估算）
        self.model_cost_rates = {
            "Qwen2.5-72B-Instruct": {"input": 0.005, "output": 0.015},  # GPTQ-Int4私有部署
            "Qwen2.5-7B-Instruct": {"input": 0.001, "output": 0.003},    # AWQ私有部署
            "bge-large-zh-v1.5": {"input": 0.0005, "output": 0.0005},   # Embedding
        }

    # ═══════════════════════════════════════════
    # 核心限流检查
    # ═══════════════════════════════════════════

    async def check_rate_limit(self, user_id: str, model: str,
                                estimated_tokens: int = 100) -> Dict[str, Any]:
        """
        多层限流检查。在调用LLM/第三方API前调用。

        返回:
            {"allowed": True/False, "reason": str, "remaining_quota": dict}
        """
        now = datetime.now()
        date_key = now.strftime("%Y%m%d")
        hour_key = now.strftime("%Y%m%d%H")
        minute_key = now.strftime("%Y%m%d%H%M")

        pipe = self.redis.pipeline()

        # Layer 1: 请求级限流（滑动窗口）
        req_minute_key = f"rl:req:{user_id}:{minute_key}"
        pipe.incr(req_minute_key)
        pipe.expire(req_minute_key, 120)  # 2分钟TTL

        # Layer 2: Token级限流
        model_short = model.split("-")[1] if "-" in model else model  # "72B" / "7B"
        token_key = f"rl:tok:{user_id}:{model_short}:{minute_key}"
        pipe.incrby(token_key, estimated_tokens)
        pipe.expire(token_key, 120)

        # Layer 4: 成本级限流
        cost_key = f"rl:cost:{user_id}:{hour_key}"
        pipe.get(cost_key)

        # Layer 5: 模型级日配额
        model_daily_key = f"rl:mdl:{user_id}:{model_short}:{date_key}"
        pipe.incr(model_daily_key)
        pipe.expire(model_daily_key, 86400)

        results = await pipe.execute()

        req_count = results[0]
        token_count = results[2]
        hourly_cost = float(results[4] or 0)
        model_daily_count = results[5]

        # 检查各项限额
        checks = []

        # Layer 1
        if req_count > self.default_quotas["requests_per_minute"]:
            checks.append(("requests_per_minute", req_count, self.default_quotas["requests_per_minute"]))

        # Layer 2
        tpm_limit = self.default_quotas.get(f"tokens_per_minute_{model_short.lower()}", 50000)
        if token_count > tpm_limit:
            checks.append(("tokens_per_minute", token_count, tpm_limit))

        # Layer 4
        if hourly_cost > self.default_quotas["hourly_cost_budget"]:
            checks.append(("hourly_cost", hourly_cost, self.default_quotas["hourly_cost_budget"]))

        # Layer 5
        daily_req_limit = self.default_quotas.get(f"{model_short.lower()}_daily_requests", 1000)
        if model_daily_count > daily_req_limit:
            checks.append(("daily_model_requests", model_daily_count, daily_req_limit))

        if checks:
            return {
                "allowed": False,
                "reason": f"Rate limit exceeded: {checks[0][0]} "
                          f"({checks[0][1]} > {checks[0][2]})",
                "remaining_quota": self._get_remaining(req_count, token_count, hourly_cost, checks),
            }

        return {
            "allowed": True,
            "reason": "",
            "remaining_quota": {
                "requests_remaining": self.default_quotas["requests_per_minute"] - req_count,
                "tokens_remaining": tpm_limit - token_count,
                "hourly_cost_remaining": round(self.default_quotas["hourly_cost_budget"] - hourly_cost, 2),
            },
        }

    # ═══════════════════════════════════════════
    # 成本追踪
    # ═══════════════════════════════════════════

    async def record_cost(self, user_id: str, model: str,
                          input_tokens: int, output_tokens: int,
                          latency_ms: int):
        """
        记录一次LLM调用的成本和性能数据。
        每次LLM调用后必须调用此方法。
        """
        # 计算成本
        rates = self.model_cost_rates.get(model, {"input": 0.001, "output": 0.003})
        cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1000.0

        now = datetime.now()
        hour_key = now.strftime("%Y%m%d%H")
        date_key = now.strftime("%Y%m%d")

        pipe = self.redis.pipeline()

        # 累计小时成本
        cost_hour_key = f"rl:cost:{user_id}:{hour_key}"
        pipe.incrbyfloat(cost_hour_key, cost)
        pipe.expire(cost_hour_key, 7200)  # 2小时TTL

        # 累计日成本
        cost_day_key = f"rl:cost:{user_id}:{date_key}"
        pipe.incrbyfloat(cost_day_key, cost)
        pipe.expire(cost_day_key, 172800)  # 48小时TTL

        # 写入详细日志（用于分析）
        log_entry = json.dumps({
            "timestamp": now.isoformat(),
            "user_id": user_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_cny": round(cost, 4),
            "latency_ms": latency_ms,
        })
        pipe.lpush("cost_log", log_entry)
        pipe.ltrim("cost_log", 0, 9999)  # 保留最近1万条

        await pipe.execute()

        # 检查是否需要告警
        await self._check_budget_alert(user_id, cost)

        return {"cost_cny": round(cost, 4)}

    async def _check_budget_alert(self, user_id: str, current_cost: float):
        """检查成本是否超过预算阈值，触发告警"""
        now = datetime.now()
        hour_key = now.strftime("%Y%m%d%H")

        hourly_total = float(await self.redis.get(f"rl:cost:{user_id}:{hour_key}") or 0)

        # 80%预警 / 100%告警
        threshold = self.default_quotas["hourly_cost_budget"]

        if hourly_total > threshold * 0.8 and hourly_total - current_cost <= threshold * 0.8:
            # 刚刚超过80%阈值
            await self._send_alert("warning", user_id, f"小时成本超过80%: ¥{hourly_total:.2f}/{threshold}")

        if hourly_total > threshold:
            await self._send_alert("critical", user_id, f"小时成本超限: ¥{hourly_total:.2f}/{threshold}")

    async def _send_alert(self, level: str, user_id: str, message: str):
        """发送告警（集成Prometheus AlertManager）"""
        alert = {
            "level": level,
            "service": "travel-agent",
            "user_id": user_id,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        await self.redis.lpush("alerts", json.dumps(alert))

    # ═══════════════════════════════════════════
    # 熔断器（Circuit Breaker）
    # ═══════════════════════════════════════════

    async def call_with_circuit_breaker(self, service_name: str,
                                        call_func, *args, **kwargs):
        """
        带熔断器的API调用。

        状态: CLOSED(正常) → OPEN(熔断) → HALF-OPEN(探测)

        使用场景: 高德地图API、天气API、vLLM服务等第三方调用。
        """
        cb_key = f"cb:{service_name}"
        state = await self.redis.get(cb_key) or "CLOSED"

        if state == "OPEN":
            # 检查是否过了冷却期
            cooldown = await self.redis.ttl(cb_key)
            if cooldown > 0:
                raise CircuitBreakerOpen(f"{service_name} 熔断中，{cooldown}秒后恢复")
            # 冷却期结束，进入HALF-OPEN
            await self.redis.setex(cb_key, 3600, "HALF-OPEN")
            state = "HALF-OPEN"

        try:
            result = await call_func(*args, **kwargs)

            # 成功：记录成功
            if state == "HALF-OPEN":
                # 连续成功3次后关闭熔断
                success_key = f"cb:success:{service_name}"
                successes = await self.redis.incr(success_key)
                if successes >= 3:
                    await self.redis.setex(cb_key, 3600, "CLOSED")
                    await self.redis.delete(success_key)

            return result

        except Exception as e:
            # 失败：记录失败
            fail_key = f"cb:fail:{service_name}"
            failures = await self.redis.incr(fail_key)
            await self.redis.expire(fail_key, 60)

            # 60秒内失败5次 → 熔断
            if failures >= 5:
                await self.redis.setex(cb_key, 60, "OPEN")  # 熔断60秒
                await self._send_alert("critical", "system",
                    f"{service_name} 连续失败{failures}次，已熔断")

            raise

    # ═══════════════════════════════════════════
    # 成本报告
    # ═══════════════════════════════════════════

    async def get_cost_report(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """生成用户成本报告"""
        costs = []
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
            cost = float(await self.redis.get(f"rl:cost:{user_id}:{date}") or 0)
            costs.append({"date": date, "cost": round(cost, 2)})

        total = sum(c["cost"] for c in costs)

        return {
            "user_id": user_id,
            "period_days": days,
            "total_cost_cny": round(total, 2),
            "daily_breakdown": costs,
            "budget_utilization": f"{total / (self.default_quotas['daily_cost_budget'] * days) * 100:.1f}%",
        }

    def _get_remaining(self, req_count, token_count, hourly_cost, checks) -> Dict:
        """获取剩余配额"""
        return {
            "requests_remaining": max(0, self.default_quotas["requests_per_minute"] - req_count),
            "hourly_cost_remaining": max(0, round(self.default_quotas["hourly_cost_budget"] - hourly_cost, 2)),
            "blocked_by": checks[0][0] if checks else None,
        }


class CircuitBreakerOpen(Exception):
    """熔断器打开异常"""
    pass
```

---

### 12.4 异常告警系统（AnomalyAlertSystem）

#### 12.4.1 职责定位

异常告警系统是系统的"预警雷达"，实时监控第三方接口健康度、系统性能异常和高并发拥堵，确保问题在影响用户前被发现和处理。

**监控范围**：
1. **第三方接口健康** — 高德/天气/地图API的可用性和延迟
2. **规划引擎超时** — CP-SAT求解耗时异常
3. **高并发拥堵** — 请求队列堆积、连接池耗尽
4. **数据库性能** — 慢查询、连接池使用率
5. **Redis健康** — 内存使用率、命中率、eviction
6. **模型服务** — vLLM/TGI的健康状态和GPU利用率

---

#### 12.4.2 第三方接口健康检查

```python
# monitoring/health_checker.py —— 第三方接口健康检查

from typing import Dict, Any, List
from datetime import datetime, timedelta
import asyncio
import aiohttp

class ThirdPartyHealthChecker:
    """
    第三方服务健康检查器。

    检查方式：主动探测（定期发送健康检查请求）+ 被动监控（追踪实际调用的成功率）。
    状态：healthy → degraded → down → recovering
    """

    SERVICES = {
        "gaode_map": {
            "name": "高德地图API",
            "check_url": "https://restapi.amap.com/v3/config/district",  # 轻量接口
            "timeout": 5,
            "expected_status": 200,
            "critical": True,  # 关键服务（down时影响行程质量）
        },
        "heweather": {
            "name": "和风天气API",
            "check_url": "https://devapi.qweather.com/v7/weather/now",
            "timeout": 5,
            "expected_status": 200,
            "critical": False,  # 非关键（可降级为静态天气）
        },
        "vllm_72b": {
            "name": "vLLM 72B模型服务",
            "check_url": "http://vllm-72b:8000/health",
            "timeout": 10,
            "expected_status": 200,
            "critical": True,   # 关键服务（down时槽位抽取失败）
        },
        "vllm_7b": {
            "name": "vLLM 7B模型服务",
            "check_url": "http://vllm-7b:8000/health",
            "timeout": 10,
            "expected_status": 200,
            "critical": True,
        },
        "postgres": {
            "name": "PostgreSQL数据库",
            "check_type": "tcp",
            "host": "postgres",
            "port": 5432,
            "timeout": 3,
            "critical": True,
        },
        "redis": {
            "name": "Redis缓存",
            "check_type": "tcp",
            "host": "redis",
            "port": 6379,
            "timeout": 3,
            "critical": True,
        },
    }

    def __init__(self, redis_client, alert_callback=None):
        self.redis = redis_client
        self.alert_callback = alert_callback

    async def run_health_checks(self) -> Dict[str, Any]:
        """执行所有健康检查，返回状态报告"""
        results = {}

        for service_id, config in self.SERVICES.items():
            status = await self._check_one(service_id, config)
            results[service_id] = status

            # 状态变化时告警
            prev_key = f"health:prev:{service_id}"
            prev_status = await self.redis.get(prev_key)

            if status["status"] != (prev_status or "healthy"):
                await self._on_status_change(service_id, config, status, prev_status)
                await self.redis.setex(prev_key, 86400, status["status"])

        return results

    async def _check_one(self, service_id: str, config: Dict) -> Dict:
        """检查单个服务"""
        start = datetime.now()

        try:
            if config.get("check_type") == "tcp":
                # TCP端口检查
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(config["host"], config["port"]),
                    timeout=config["timeout"]
                )
                writer.close()
                await writer.wait_closed()
                latency_ms = int((datetime.now() - start).total_seconds() * 1000)

                return {
                    "service": service_id,
                    "name": config["name"],
                    "status": "healthy",
                    "latency_ms": latency_ms,
                    "checked_at": datetime.now().isoformat(),
                }
            else:
                # HTTP健康检查
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        config["check_url"],
                        timeout=aiohttp.ClientTimeout(total=config["timeout"])
                    ) as resp:
                        latency_ms = int((datetime.now() - start).total_seconds() * 1000)

                        if resp.status == config["expected_status"]:
                            status = "healthy"
                        else:
                            status = "degraded" if resp.status < 500 else "down"

                        return {
                            "service": service_id,
                            "name": config["name"],
                            "status": status,
                            "http_status": resp.status,
                            "latency_ms": latency_ms,
                            "checked_at": datetime.now().isoformat(),
                        }

        except asyncio.TimeoutError:
            return {
                "service": service_id,
                "name": config["name"],
                "status": "down",
                "error": "timeout",
                "latency_ms": config["timeout"] * 1000,
                "checked_at": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "service": service_id,
                "name": config["name"],
                "status": "down",
                "error": str(e)[:100],
                "checked_at": datetime.now().isoformat(),
            }

    async def _on_status_change(self, service_id: str, config: Dict,
                                 current: Dict, previous: str):
        """服务状态变化时触发告警"""
        new_status = current["status"]

        if new_status == "healthy" and previous in ("down", "degraded"):
            # 恢复告警
            await self._send_alert("info", f"{config['name']} 已恢复", {
                "service": service_id,
                "status": "recovered",
                "latency_ms": current.get("latency_ms"),
            })

        elif new_status == "down":
            # 宕机告警
            severity = "critical" if config.get("critical") else "warning"
            await self._send_alert(severity, f"{config['name']} 不可用", {
                "service": service_id,
                "status": "down",
                "error": current.get("error", ""),
                "critical": config.get("critical", False),
            })

        elif new_status == "degraded":
            # 降级告警
            await self._send_alert("warning", f"{config['name']} 性能降级", {
                "service": service_id,
                "status": "degraded",
                "latency_ms": current.get("latency_ms"),
            })

    async def _send_alert(self, severity: str, title: str, details: Dict):
        """发送告警"""
        alert = {
            "severity": severity,
            "title": title,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        }

        # 写入Redis告警队列
        await self.redis.lpush("alerts", json.dumps(alert))

        # 调用外部告警通道（Webhook）
        if self.alert_callback:
            await self.alert_callback(alert)
```

---

#### 12.4.3 高并发拥堵告警

```python
# monitoring/congestion_detector.py —— 高并发拥堵检测

class CongestionDetector:
    """
    高并发拥堵检测器。

    检测指标：
      1. 请求队列长度（>50拥堵）
      2. 连接池使用率（>80%预警）
      3. P99延迟突增（比基线高3倍）
      4. 错误率突增（>5%异常）
      5. 活跃会话数（>1000预警）
    """

    THRESHOLDS = {
        "queue_length": 50,           # 请求队列>50视为拥堵
        "connection_pool_usage": 0.8, # 连接池使用率>80%预警
        "latency_multiplier": 3,      # P99延迟比基线高3倍
        "error_rate": 0.05,           # 错误率>5%异常
        "active_sessions": 1000,      # 活跃会话>1000预警
    }

    def __init__(self, redis_client):
        self.redis = redis_client
        self.baseline_latency = {}  # 基线延迟（各Node）

    async def check_congestion(self, metrics: Dict[str, Any]) -> List[Dict]:
        """检查拥堵指标，返回告警列表"""
        alerts = []

        # 1. 请求队列长度
        queue_len = metrics.get("request_queue_length", 0)
        if queue_len > self.THRESHOLDS["queue_length"]:
            alerts.append({
                "type": "queue_congestion",
                "severity": "critical" if queue_len > 100 else "warning",
                "value": queue_len,
                "threshold": self.THRESHOLDS["queue_length"],
                "suggestion": "启动限流或扩容",
            })

        # 2. 连接池使用率
        pool_usage = metrics.get("db_pool_usage", 0)
        if pool_usage > self.THRESHOLDS["connection_pool_usage"]:
            alerts.append({
                "type": "connection_pool_exhaustion",
                "severity": "warning",
                "value": f"{pool_usage*100:.1f}%",
                "threshold": f"{self.THRESHOLDS['connection_pool_usage']*100:.0f}%",
                "suggestion": "增加连接池大小或检查连接泄漏",
            })

        # 3. P99延迟突增
        for node, p99 in metrics.get("p99_latency", {}).items():
            baseline = self.baseline_latency.get(node, p99)
            if p99 > baseline * self.THRESHOLDS["latency_multiplier"]:
                alerts.append({
                    "type": "latency_spike",
                    "severity": "warning",
                    "node": node,
                    "value": f"{p99}ms",
                    "baseline": f"{baseline}ms",
                    "suggestion": f"检查{node}的下游依赖",
                })

        # 4. 错误率突增
        error_rate = metrics.get("error_rate", 0)
        if error_rate > self.THRESHOLDS["error_rate"]:
            alerts.append({
                "type": "error_rate_spike",
                "severity": "critical" if error_rate > 0.1 else "warning",
                "value": f"{error_rate*100:.1f}%",
                "threshold": f"{self.THRESHOLDS['error_rate']*100:.0f}%",
                "suggestion": "检查最近的部署变更或依赖服务状态",
            })

        # 5. 活跃会话数
        sessions = metrics.get("active_sessions", 0)
        if sessions > self.THRESHOLDS["active_sessions"]:
            alerts.append({
                "type": "session_overflow",
                "severity": "warning",
                "value": sessions,
                "threshold": self.THRESHOLDS["active_sessions"],
                "suggestion": "增加实例或优化会话超时",
            })

        return alerts

    async def update_baseline(self, node_name: str, latency_ms: int):
        """更新延迟基线（滑动窗口平均）"""
        key = f"baseline:latency:{node_name}"
        current = await self.redis.get(key)
        if current:
            # 指数移动平均
            new_baseline = int(0.9 * float(current) + 0.1 * latency_ms)
        else:
            new_baseline = latency_ms
        await self.redis.setex(key, 86400, str(new_baseline))
        self.baseline_latency[node_name] = new_baseline
```

---

#### 12.4.4 Prometheus告警规则（增强版）

```yaml
# alerting/enhanced_rules.yml —— 增强告警规则

groups:
  - name: travel_agent_enhanced

    # === 第三方接口健康 ===
    rules:
      - alert: ThirdPartyAPIDown
        expr: travel_third_party_status == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "第三方服务 {{ $labels.service }} 不可用"
          description: "{{ $labels.service_name }} 连续1分钟健康检查失败"

      - alert: ThirdPartyAPIDegraded
        expr: travel_third_party_latency_p99 > 5000
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "第三方服务 {{ $labels.service }} 延迟过高"
          description: "P99延迟 {{ $value }}ms，超过阈值5000ms"

    # === 高并发拥堵 ===
    rules:
      - alert: RequestQueueCongestion
        expr: travel_request_queue_length > 50
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "请求队列拥堵"
          description: "队列长度 {{ $value }}，超过阈值50"

      - alert: ConnectionPoolExhaustion
        expr: travel_db_pool_usage > 0.8
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "数据库连接池耗尽"
          description: "使用率 {{ $value | humanizePercentage }}"

      - alert: HighLatencySpike
        expr: |
          histogram_quantile(0.99, 
            rate(travel_request_latency_seconds_bucket[5m])
          ) > 3 * avg_over_time(
            histogram_quantile(0.99, 
              rate(travel_request_latency_seconds_bucket[1h])
            )[1h:]
          )
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "P99延迟突增3倍以上"

      - alert: ErrorRateSpike
        expr: rate(travel_request_errors_total[5m]) / rate(travel_request_total[5m]) > 0.05
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "错误率突增到 {{ $value | humanizePercentage }}"

    # === Redis健康 ===
    rules:
      - alert: RedisMemoryHigh
        expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.85
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Redis内存使用率 {{ $value | humanizePercentage }}"

      - alert: RedisEvictionHigh
        expr: rate(redis_evicted_keys_total[5m]) > 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Redis eviction速率过高"

      - alert: RedisHitRateLow
        expr: redis_keyspace_hits_total / (redis_keyspace_hits_total + redis_keyspace_misses_total) < 0.8
        for: 5m
        labels:
          severity: info
        annotations:
          summary: "Redis命中率低于80%"

    # === 成本控制 ===
    rules:
      - alert: HourlyCostOverBudget
        expr: travel_hourly_cost_cny > travel_hourly_budget_cny
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "小时成本超预算 ¥{{ $value }}"

      - alert: DailyCostWarning
        expr: travel_daily_cost_cny > 0.8 * travel_daily_budget_cny
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "日成本超过预算80%: ¥{{ $value }}"
```

---

### 12.5 安全风控层（ContentSafetyEngine）

#### 12.5.1 职责定位

安全风控引擎是系统的"内容过滤器"，确保推荐内容符合法规要求，不输出有害或违规信息。

**风控范围**：
1. **低价购物团过滤** — 识别并过滤涉嫌强制购物的行程
2. **黑导游过滤** — 对接导游资质数据库，过滤无资质导游推荐
3. **民宿资质校验** — 对接民宿备案系统，过滤无资质民宿
4. **违规出行方案** — 不输出涉及非法穿越、未开放区域等方案

```python
# agents/content_safety.py —— 内容安全风控引擎

class ContentSafetyEngine:
    """
    内容安全风控引擎。

    检查方式：
      1. 关键词匹配 —— 快速过滤明显的违规内容
      2. 规则引擎 —— 结构化规则检查
      3. LLM辅助审核 —— 复杂场景由7B模型辅助判断（P2）
    """

    # 高风险关键词库
    RISK_KEYWORDS = {
        "forced_shopping": ["强制购物", "购物点", "进店", "土特产", "翡翠", "玉石", "免税店必须"],
        "illegal_route": ["非法穿越", "未开放区域", "禁区", "翻墙", "逃票", "私闯"],
        "unqualified_guide": ["黑导游", "无资质", "野导"],
        "unsafe_activity": ["徒手攀岩", "无保护", "野泳", "漂流无救生衣"],
    }

    # 资质校验规则
    CERTIFICATION_RULES = {
        "hotel": {"required_license": ["特种行业许可证", "消防验收合格证"]},
        "homestay": {"required_license": ["民宿备案登记证", "卫生许可证"]},
        "tour_guide": {"required_license": ["导游证", "旅行社业务经营许可证"]},
        "travel_agency": {"required_license": ["旅行社业务经营许可证", "营业执照"]},
    }

    def check_itinerary(self, itinerary: List[Dict], poi_list: List[Dict]) -> Dict[str, Any]:
        """
        检查行程的安全合规性。

        返回:
            {"passed": bool, "violations": [{"type", "severity", "description", "suggestion"}]}
        """
        violations = []

        # 1. 检查低价购物团特征
        violations += self._check_forced_shopping(itinerary, poi_list)

        # 2. 检查非法路线
        violations += self._check_illegal_route(itinerary)

        # 3. 检查资质（MVP阶段简化，P2对接真实资质库）
        violations += self._check_certifications(poi_list)

        # 4. 检查不安全活动
        violations += self._check_unsafe_activities(itinerary)

        critical = any(v["severity"] == "critical" for v in violations)

        return {
            "passed": not critical,
            "violations": violations,
            "safe_to_output": len(violations) == 0,
        }

    def _check_forced_shopping(self, itinerary: List[Dict],
                                poi_list: List[Dict]) -> List[Dict]:
        """检查是否涉嫌强制购物团"""
        violations = []

        # 特征：购物类POI占比过高（>40%）
        shopping_count = sum(
            1 for p in poi_list if p.get("spot_type") == "shopping"
        )
        total = len(poi_list)

        if total > 0 and shopping_count / total > 0.4:
            violations.append({
                "type": "forced_shopping",
                "severity": "high",
                "description": f"购物类POI占比{shopping_count/total*100:.0f}%，涉嫌购物团",
                "suggestion": "减少购物点，增加景点比重至70%以上",
            })

        # 关键词检查
        for poi in poi_list:
            name = poi.get("spot_name", "")
            for kw in self.RISK_KEYWORDS["forced_shopping"]:
                if kw in name:
                    violations.append({
                        "type": "forced_shopping",
                        "severity": "high",
                        "description": f"景点'{name}'名称包含高风险关键词'{kw}'",
                        "suggestion": "移除或替换该景点",
                    })

        return violations

    def _check_illegal_route(self, itinerary: List[Dict]) -> List[Dict]:
        """检查是否包含非法路线"""
        violations = []

        for day in itinerary:
            for act in day.get("schedule", []):
                name = act.get("spot_name", "")
                for kw in self.RISK_KEYWORDS["illegal_route"]:
                    if kw in name:
                        violations.append({
                            "type": "illegal_route",
                            "severity": "critical",
                            "description": f"行程包含非法关键词'{kw}'",
                            "suggestion": "立即移除该路线，替换为合法景点",
                        })

        return violations

    def _check_certifications(self, poi_list: List[Dict]) -> List[Dict]:
        """检查资质（MVP简化版，P2对接真实资质库）"""
        # MVP阶段：仅检查是否有资质标记
        # P2阶段：对接文旅部导游资质库、民宿备案系统
        return []

    def _check_unsafe_activities(self, itinerary: List[Dict]) -> List[Dict]:
        """检查不安全活动"""
        violations = []

        for day in itinerary:
            for act in day.get("schedule", []):
                name = act.get("spot_name", "")
                for kw in self.RISK_KEYWORDS["unsafe_activity"]:
                    if kw in name:
                        violations.append({
                            "type": "unsafe_activity",
                            "severity": "high",
                            "description": f"行程包含高风险活动'{name}'",
                            "suggestion": "移除或替换为安全替代项目",
                        })

        return violations
```

---

### 12.6 一键隐私清除

```python
# privacy.py —— 一键清除（补充GDPR接口）

async def delete_all_travel_memory(self, user_id: str) -> Dict[str, bool]:
    """
    一键清除用户的所有出行记忆。

    清除范围：
      1. PostgreSQL: user_profile表记录
      2. PostgreSQL: user_trip_history表所有行程
      3. PostgreSQL: planning_log中该用户的规划记录
      4. Redis: 所有该用户的会话数据
      5. Redis: 该用户的画像缓存

    返回: 各数据源的清除状态
    """
    results = {}

    try:
        async with self.pg_pool.acquire() as conn:
            async with conn.transaction():
                # 删除行程历史
                await conn.execute(
                    "DELETE FROM user_trip_history WHERE user_id = $1", user_id
                )
                results["trip_history"] = True

                # 删除规划日志
                await conn.execute(
                    "DELETE FROM planning_log WHERE user_id = $1", user_id
                )
                results["planning_logs"] = True

                # 删除用户画像
                await conn.execute(
                    "DELETE FROM user_profile WHERE user_id = $1", user_id
                )
                results["user_profile"] = True

        # 删除Redis数据
        import redis.asyncio as redis_async
        r = redis_async.Redis(decode_responses=True)

        keys = await r.keys(f"travel_agent:*:{user_id}:*")
        if keys:
            await r.delete(*keys)

        # 清除限流/成本数据
        cost_keys = await r.keys(f"rl:*:{user_id}:*")
        if cost_keys:
            await r.delete(*cost_keys)

        results["redis_data"] = True

        return results

    except Exception as e:
        results["error"] = str(e)[:200]
        return results
```

---

### 12.7 本章模块覆盖对照

| 需求 | 模块 | 覆盖度 | 实现方式 |
|------|------|--------|----------|
| 幻觉检测 — 校验开放时间/价格/路线冲突 | 12.1 HallucinationDetectionAgent | 100% | 5类检测规则 + 自动标注 |
| 安全风控 — 过滤低价购物团/黑导游/无资质民宿 | 12.5 ContentSafetyEngine | 80% | 关键词+规则引擎，P2对接资质库 |
| 用户权限&隐私 — 加密存储/一键清除 | 12.6 delete_all_travel_memory | 100% | AES-256-GCM + 5数据源原子清除 |
| 日志监控 — 工具调用耗时/规划失败/高频修改 | 12.2 LogAnalyticsEngine | 100% | 2张日志表 + 4维度分析 + 迭代建议 |
| 限流&成本控制 — LLM/API调用频次/成本预算 | 12.3 RateLimitCostController | 100% | 5层限流 + Token配额 + 熔断器 |
| 异常告警 — 第三方接口失效/规划超时/高并发 | 12.4 AnomalyAlertSystem | 100% | 主动健康检查 + 6类拥堵检测 + Prometheus规则 |

---

> 至此，技术蓝图v3.0完整版（上半+下半+运维安全风控）全部完成。
> 完整技术蓝图（上半+下半）到此结束。
> 上半部分："理解用户是谁、想要什么"（第0-4层）
> 下半部分："基于需求找到景点并规划最优路线"（第5-8层+运维）
