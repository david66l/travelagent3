# TravelAgent2 技术蓝图：全链路自主规划旅游 Agent

> 对应 PRD 全部 8 层架构，逐层落地技术选型。

---

## 0. 总览：技术栈全景图

```
┌─────────────────────────────────────────────────────┐
│ 交互输出层: Markdown + 地图 + 表格 + PDF + 语音      │
│              Next.js 15 + Leaflet + jsPDF + TTS      │
├─────────────────────────────────────────────────────┤
│ 工具调用层: Function Calling 7 类工具                 │
│              OpenAI Tool Call + Pydantic Schema      │
├─────────────────────────────────────────────────────┤
│ 规划引擎: 约束求解 + 路线优化 + 预算分配              │
│              OR-Tools CP-SAT + 贪心 + 遗传算法       │
├─────────────────────────────────────────────────────┤
│ 知识库层: 结构化 DB + RAG 向量检索 + 实时API连接器    │
│              PostgreSQL + pgvector + Redis           │
├─────────────────────────────────────────────────────┤
│ 用户画像层: 短期上下文 + 长期向量记忆                 │
│              Redis + Chroma/Milvus                   │
├─────────────────────────────────────────────────────┤
│ 意图理解层: 槽位抽取 + 歧义消解 + 情感识别            │
│              Qwen2.5 + LoRA + LangGraph StateGraph    │
├─────────────────────────────────────────────────────┤
│ 感知层: 文本 + 图片 + PDF + 外部事件                  │
│              Next.js + OCR + WebSocket/SSE            │
└─────────────────────────────────────────────────────┘
```

### 全栈技术选型

| 层级 | 选型 | 理由 |
|---|---|---|
| **编排框架** | LangGraph | StateGraph 支持循环/分支/断点持久化，天然适配多轮Agent |
| **底层框架** | LangChain | 工具封装、RAG 检索、Memory 封装、LLM 统一调用 |
| **大模型** | Qwen2.5-72B-Instruct + 旅游 LoRA | 中文最优、支持 vLLM 私有化部署、成本可控 |
| **小模型(轻量任务)** | Qwen2.5-7B-AWQ | 意图识别/情感分析/槽位抽取，显存友好 |
| **向量库** | Chroma(开发) / Milvus(生产) | Chroma 零配置起步，Milvus 高并发商用 |
| **数据库** | PostgreSQL 16 + pgvector | 结构化数据 + 向量检索统一存储 |
| **缓存** | Redis 7 | 会话热记忆、实时数据缓存、限流 |
| **推理引擎** | vLLM | continuous batching、LoRA adapter 热加载 |
| **前端** | Next.js 15 + TypeScript + Tailwind | 当前技术栈，保留 |
| **后端** | FastAPI + Celery | 当前技术栈，保留 |
| **部署** | Docker + K8s | 当前技术栈，保留 |
| **监控** | LangSmith + Prometheus + Grafana | LLM调用链追踪 + 系统指标 |
| **搜索** | Tavily API + 本地 BM25 | 混合检索：联网 + 本地 |

---

## 0.5 多 Agent 内部架构

### 整体设计

**1 根主流程 LangGraph 状态机 + 6 个核心子 Agent + 3 个可选增值 Agent**。

```
┌──────────────────────────────────────────────────────────────────┐
│                    LangGraph StateGraph (主流程)                   │
│                                                                  │
│  START                                                           │
│    ↓                                                             │
│  ① DemandParserAgent (需求解析)                                    │
│    ├─ 槽位不全 → HumanInterrupt → 等待用户补充                      │
│    └─ 槽位完整 ↓                                                  │
│  ② UserMemoryRecallAgent (用户画像记忆)                             │
│    ↓                                                             │
│  ③ TravelRetrievalRAGAgent (知识库检索)                             │
│    ↓                                                             │
│  ④ ItineraryPlannerAgent (行程规划求解)                             │
│    ↓                                                             │
│  ⑤ FactCheckAgent (事实校验 & 幻觉拦截)                              │
│    ├─ 发现冲突 → 回到 ④ 重规划                                      │
│    └─ 校验通过 ↓                                                  │
│  HumanInterrupt (展示行程，等待用户反馈)                              │
│    ├─ 用户修改 → 回到 ① 更新需求                                    │
│    └─ 用户确认 ↓                                                  │
│  ⑥ Output&DocAgent (多模态输出 & 文档)                              │
│    ↓                                                             │
│  (可选) ⑦ BookingToolAgent (预订工具)                               │
│    ↓                                                             │
│  ② UserMemoryRecallAgent (写入本次出行偏好，更新画像)                  │
│    ↓                                                             │
│  END                                                             │
└──────────────────────────────────────────────────────────────────┘
```

### 分工原则

| 层 | 职责 |
|---|---|
| **LangGraph** | 全局流程控制、分支判断、循环重规划、Human-in-the-loop、Checkpoint 持久化 |
| **LangChain** | 每个子 Agent 的 LLM 调用、RAG、工具封装、记忆、Prompt 管理、结构化输出 |

所有子 Agent 无独立调度权，统一由 Graph 流转驱动，避免多智能体对话失控、上下文漂移。

---

### 6 个核心子 Agent

#### ① DemandParserAgent — 需求解析 Agent

| 维度 | 说明 |
|---|---|
| **职责** | 解析原始自然语言，抽取标准化槽位；识别模糊需求生成反问；区分意图类型；情感识别 |
| **LangChain 组件** | Pydantic 结构化输出解析器、FewShot Prompt、上下文记忆读取 |
| **Graph 流转** | 槽位不全 → HumanInterrupt；槽位完整 → 进入 UserMemoryRecallAgent |
| **输入** | 用户原始文本 + 历史对话 |
| **输出** | TravelSlots (结构化槽位) + intent + confidence + missing_slots |

#### ② UserMemoryRecallAgent — 用户画像记忆 Agent

| 维度 | 说明 |
|---|---|
| **职责** | 查询 PostgreSQL 画像 + pgvector 长期偏好向量；RAG 召回历史出行；自动合并偏好到当前槽位；行程结束后更新画像 |
| **LangChain 组件** | VectorStoreRetriever (pgvector)、SQL DB 工具、记忆总结 Chain |
| **输入** | user_id + 当前槽位 |
| **输出** | 合并后的完整 UserProfile + 历史偏好向量 |

#### ③ TravelRetrievalRAGAgent — 知识库检索 Agent (事实来源核心)

| 维度 | 说明 |
|---|---|
| **职责** | 生成检索 Query；联合检索结构化 DB + pgvector 攻略；多层过滤（预算/步行强度/忌口/人群/开放时间/距离）；清洗无效数据，杜绝模型编造 |
| **LangChain 组件** | SelfQueryRetriever（结构化字段过滤 + 向量混合检索）、SQLDatabaseChain、去重校验工具链 |
| **输入** | UserProfile + 偏好向量 |
| **输出** | 纯结构化实体列表（景点/酒店/餐厅，带真实门票/时长/营业时间） |

#### ④ ItineraryPlannerAgent — 行程规划求解 Agent

| 维度 | 说明 |
|---|---|
| **职责** | 基于 RAG 真实数据 + 全部约束，生成最优行程；多目标优化（减少折返/均衡体力/错峰/预算均分）；调用 OR-Tools CP-SAT；预算自动拆分 |
| **LangChain 组件** | 规划求解器 CustomTool、结构化输出解析器、约束冲突检测 Chain |
| **输入** | 结构化 POI 列表 + UserProfile + 约束权重 |
| **输出** | 标准化行程 JSON（分天/时段/交通/费用） + 预算明细 |

#### ⑤ FactCheckAgent — 事实校验 & 幻觉拦截 Agent

| 维度 | 说明 |
|---|---|
| **职责** | 遍历 Planner 输出的所有 POI ID，回查 PostgreSQL 校验真实数据；识别冲突（周二闭馆/超预算/步行超标）；生成冲突清单，触发重规划或提示用户 |
| **LangChain 组件** | SQL 查询工具链、冲突总结 Chain |
| **输入** | 行程 JSON + PostgreSQL 结构化库 |
| **输出** | 冲突清单 (conflicts[]) 或 校验通过标记 |

#### ⑥ Output&DocAgent — 多模态输出 & 文档 Agent

| 维度 | 说明 |
|---|---|
| **职责** | 结构化行程 → 自然文案；生成 Excel/PDF/打包清单；整理避坑贴士/应急电话；支持精简/详细两种风格 |
| **LangChain 组件** | 文档生成 CustomTool、多模态格式化 Chain |
| **输入** | 校验通过的行程 + 知识库贴士 |
| **输出** | 文案 + PDF + Excel + 出行清单 |

---

### 3 个可选增值 Agent

| # | Agent | 职责 | MVP 建议 |
|---|---|---|---|
| ⑦ | **BookingToolAgent** | 统一封装机票/高铁/酒店/门票/餐厅预订 Function Calling | 暂缓 |
| ⑧ | **EmergencyAssistantAgent** | 途中异常处理（暴雨/堵车/景区关闭/生病），生成替代方案 | 后期 |
| ⑨ | **MultiPersonSyncAgent** | 多人出行：合并偏好、分摊预算、统一行程 | 后期 |

### 咪娜酱补充建议

| 场景 | 是否需要 | 判断 |
|---|---|---|
| 出境游签证 Agent | ❌ | 做出境游再加，MVP 不需要 |
| 实时交通 Agent | ❌ | Planner 用静态 transit 时间已足够，动态交通属锦上添花 |
| 内容安全/注入检测 | ❌ | 作为 DemandParser 入口 filter 即可，不必独立 Agent |

> **结论**：6 核心 Agent + 3 可选 = 标准上限。再多就是过度设计。

---

### 为什么分拆 6 个子 Agent（不揉成一个）

| 价值 | 说明 |
|---|---|
| **解耦迭代** | 优化检索逻辑只改 RAGAgent，优化路线算法只改 Planner，互不影响 |
| **故障隔离** | FactCheck 出错不会破坏 DemandParser 流程 |
| **可观测性** | LangSmith 按 Agent 维度打点，清晰定位哪一步产生幻觉、哪一步规划失败 |
| **资源可控** | 每个子 Agent 可独立配置 LLM 模型、温度、token 上限：Parser 用低温度保证抽取精准，OutputAgent 高温度优化文案流畅度 |
| **便于扩展** | 后续新增出境游、签证功能，仅新增子 Agent 插入 Graph 节点即可，不用重构主流程 |

### 关键落地要点

- 所有业务智能体以 **LangChain Chain** 实现，作为 **LangGraph 的 Node**
- LangGraph **只做流程调度**，不承担任何 LLM 业务逻辑
- 依靠 **TravelRetrievalRAGAgent** 对接 PostgreSQL 结构化库，从根源解决模型编造事实
- 循环重规划、人工交互、断点恢复全部依赖 **LangGraph 原生状态机**能力，纯 LangChain 链式无法实现
- 每个 Agent **单一职责**，符合高内聚低耦合工程规范，方便后续扩展商业化预订、多人协同、应急调度
- **PlannerAgent 关键特性**：当用户修改需求或外部异常推送（暴雨、景区关闭），Graph 循环重新调用本 Agent 局部重规划，无需全流程重跑
- 控制循环次数，防止无限迭代重规划，增加迭代次数上限保护

---

## 1. 感知层

### 1.1 文本输入
- **当前**: FastAPI SSE 流式对话 ← 保留
- **增强**: 语音转文字 → `Whisper API` 或本地 `faster-whisper`

### 1.2 图片/截图输入 ⚠️ 待确认
- **选型**: Qwen2.5-VL (多模态版本) 或 GPT-4o-mini
- **场景**: 用户上传目的地照片、攻略截图、预算表格截图
- **处理流程**:
  ```
  图片 → OCR提取文字 → 结构化槽位填充 → 意图理解层
  ```
- **OCR选型**: 文本为主用 `Tesseract.js`(前端) 或 `PaddleOCR`(后端/中文更优)
  - **待确认**: 是否需要 OCR？还是直接用 VL 模型端到端？

### 1.3 PDF/攻略文档
- **选型**: `PyMuPDF` 提取文字 + 表格 → LLM 摘要 → 填充偏好槽位
- **待确认**: 用户上传的 PDF 攻略是否常见场景？初期可暂缓

### 1.4 外部事件输入
- **Webhook 端点**: FastAPI `/api/v1/webhooks/events`
- **事件类型**:
  - 天气突变: 和风天气 API webhook
  - 景区公告: 定时爬取 + 变更检测
  - 航班/高铁: 第三方 API 定时轮询 (30min间隔)
- **待确认**: 实时交通数据源用哪个？高德？百度？

---

## 2. 意图理解层

### 2.1 大模型选型

| 场景 | 模型 | 部署 |
|---|---|---|
| 复杂规划、文案润色 | Qwen2.5-72B-Instruct + 旅游 LoRA | vLLM (A100×1) |
| 意图识别、情感分析 | Qwen2.5-7B-Instruct + LoRA | vLLM (A10×1) |
| 多模态理解(图片) | Qwen2.5-VL-7B | vLLM |
| 开发/测试降级 | Qwen2.5-14B-AWQ | 本地 4090 |

### 2.2 LoRA 微调策略

```
基座: Qwen2.5-7B/72B-Instruct
数据: 5000+ 条旅游多轮对话 (合成 + 真实)
任务:
  - intent-lora:     意图分类 (7类)
  - slot-lora:       槽位抽取 (15个槽位)
  - sentiment-lora:  情感识别 (焦虑/放松/亲子)
  - planner-lora:    行程结构化输出
  - writer-lora:     文案生成 (个性化)
```

**待确认**: LoRA 训练数据从哪来？自己合成还是用公开数据集？

### 2.3 槽位抽取 Schema

```python
from pydantic import BaseModel
from typing import Optional, Literal

class TravelSlots(BaseModel):
    # 基础
    origin: Optional[str]           # 出发地
    destination: Optional[str]      # 目的地
    travel_days: Optional[int]      # 天数
    travel_dates: Optional[str]     # 日期范围
    
    # 人群 (⚠️ 新增)
    travelers_count: int = 1
    has_elderly: bool = False       # 有老人
    has_children: bool = False      # 有儿童
    has_pregnant: bool = False      # 孕妇
    has_wheelchair: bool = False    # 轮椅需求
    
    # 预算
    total_budget: Optional[float]
    budget_per_person: Optional[float]
    
    # 偏好
    interests: list[str]            # [自然, 历史, 美食, 拍照, ...]
    food_prefs: list[str]           # [辣, 清淡, 海鲜, ...]
    food_taboos: list[str]          # [不吃辣, 海鲜过敏, ...] ← ⚠️ 新增
    
    # 约束 (⚠️ 新增)
    pace: Literal["relaxed", "moderate", "intensive"] = "moderate"
    max_walk_minutes: int = 180     # 日最大步行时长(分钟)
    max_transit_minutes: int = 120  # 日最大车程(分钟)
    avoid_crowds: bool = False      # 避开人流
    prefer_morning: bool = False    # 早起型
    
    # 出行方式 (⚠️ 新增)
    transport_mode: Literal["self_drive", "public", "mixed", "any"] = "any"
```

### 2.4 歧义消解

```
触发条件: 置信度 < 0.7 或 槽位模糊
消解策略:
  1. 模糊词识别: "南方" → {广州, 深圳, 厦门, 三亚...}
     → 给出 Top-3 候选 + 简洁推荐理由
  2. "性价比高" → 给出该目的地 3 档预算参考
  3. "周边游" → 根据用户画像中的常驻城市推断出发地

输出格式:
  {
    "type": "clarification",
    "field": "destination",
    "candidates": [
      {"value": "厦门", "reason": "6月淡季，机票便宜，适合短途"},
      {"value": "成都", "reason": "美食多，节奏慢，预算友好"},
      {"value": "大理", "reason": "自然风光好，适合放松"}
    ]
  }
```

### 2.5 编排框架：LangGraph StateGraph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    # 用户输入
    user_input: str
    messages: list[dict]
    
    # 槽位
    slots: TravelSlots
    missing_slots: list[str]
    
    # 意图
    intent: str                    # new_itinerary / modify / query / book / emergency
    confidence: float
    
    # 规划结果
    itinerary: list[dict]
    budget: dict
    
    # 工具调用
    tool_calls: list[dict]
    tool_results: list[dict]
    
    # 记忆
    user_profile: dict             # 长期画像
    session_context: dict          # 短期上下文
    
    # 控制流
    next_action: str               # clarify / plan / call_tool / respond / error
    loop_count: int

# LangGraph 流程
graph = StateGraph(AgentState)

graph.add_node("perceive", perceive_node)       # 感知层
graph.add_node("understand", understand_node)   # 意图理解层
graph.add_node("load_profile", profile_node)    # 加载画像
graph.add_node("load_knowledge", knowledge_node) # 加载知识
graph.add_node("plan", planner_node)            # 规划引擎
graph.add_node("call_tools", tools_node)        # 工具调用
graph.add_node("format_output", output_node)    # 格式化输出
graph.add_node("dynamic_replan", replan_node)   # 动态重规划

# 条件边
graph.add_conditional_edges("understand", route_after_understand, {
    "clarify": "format_output",
    "plan": "load_profile",
    "query": "load_knowledge",
})
graph.add_conditional_edges("plan", route_after_plan, {
    "need_tools": "call_tools",
    "done": "format_output",
})
```

**待确认**: LangGraph 是否太重？当前 FastAPI 手动编排是否够用？初期可保留现有 pipeline，逐步迁移到 LangGraph。

---

## 3. 用户画像与记忆层

### 3.1 短期上下文（当前已有，保留）
- Redis 哈希: `session:{session_id}:state`
- TTL: 30min，每次访问续期
- 内容: 当前草稿行程、已确认参数、最近10条消息

### 3.2 长期向量记忆 ⚠️ 待确认

| 选型 | Chroma | Milvus |
|---|---|---|
| 门槛 | 零配置，pip install | 需 Docker 部署 |
| 性能 | < 100万向量 | > 1000万向量 |
| 场景 | 开发/小规模 | 生产/商用 |
| **建议** | 先用 Chroma，验证效果后迁移 Milvus |

### 3.3 用户画像 Schema

```python
class UserProfileVector:
    user_id: str
    # 出行习惯
    preferred_transport: list[str]    # ["高铁", "飞机"]
    preferred_accommodation: str      # "民宿" / "经济型" / "五星"
    avg_budget_per_day: float         # 日均预算
    
    # 饮食特征 (从历史行程中提取)
    liked_foods: list[str]            # ["川菜", "烧烤"]
    avoided_foods: list[str]          # ["辣", "生食"]
    
    # 体力特征
    avg_daily_steps: int              # 日均步数
    max_walk_minutes: int             # 最大步行时长
    prefers_morning: bool             # 早起型
    
    # 偏好向量 (768维 Embedding)
    preference_embedding: list[float]
    
    # 历史
    visited_cities: list[str]
    favorite_spots: list[str]
    avoid_spots: list[str]            # 踩坑记录
```

### 3.4 记忆读写策略

```
写入时机:
  - 行程确认后: 更新 visited_cities / favorite_spots
  - 用户反馈后: 更新 liked_foods / avoided_foods
  - 行程结束后: 总结偏好，重新计算 preference_embedding

读取时机:
  - 新行程开始: 召回 Top-3 相似历史行程
  - 槽位补全: 从画像推断缺失槽位（如用户不说预算→用历史平均）
  - RAG 检索: 偏好向量加权搜索结果

### 3.5 隐私隔离与安全 (PRD 3.3 新增)
- 用户记忆独立分片，不可跨用户泄露
- AES-256-GCM 加密存储敏感偏好字段
- 支持一键清除所有出行记忆（GDPR 合规）

### 3.6 行程结束自动记忆更新 (PRD 三 新增)
- 行程结束后自动触发总结流程
- 从行程中提取：新 visited_cities / favorite_spots / 预算基线
- 重新计算 preference_embedding，加权合并 (0.7×旧 + 0.3×新)
- LangGraph 新增 `trip_end_node`，行程确认后自动执行
```

**待确认**: 用什么 Embedding 模型？建议 `bge-large-zh-v1.5`（中文最优开源方案）

---

## 4. 知识库层

### 4.1 结构化数据库

**选型**: PostgreSQL 16 + pgvector 扩展
**理由**: 当前已用 PG，加插件即可支持向量检索，避免多一套存储

```sql
-- 景点库
CREATE TABLE attractions (
    id UUID PRIMARY KEY,
    name VARCHAR(200),
    city VARCHAR(50),
    category VARCHAR(50),       -- 自然/人文/美食/购物/娱乐
    open_time TIME,
    close_time TIME,
    ticket_price DECIMAL,
    duration_minutes INT,       -- 推荐游览时长
    need_reservation BOOL,
    wheelchair_accessible BOOL,
    peak_hours VARCHAR(50),     -- 人流高峰时段
    best_season VARCHAR(50),    -- 最佳游览季节
    lat DOUBLE PRECISION,
    lng DOUBLE PRECISION,
    tags TEXT[],                -- [历史, 拍照, 亲子, ...]
    description TEXT,           -- 200字简介
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 餐饮库
CREATE TABLE restaurants (
    id UUID PRIMARY KEY,
    name VARCHAR(200),
    city VARCHAR(50),
    cuisine VARCHAR(50),        -- 川菜/粤菜/日料/...
    avg_price DECIMAL,
    open_time TIME,
    close_time TIME,
    lat DOUBLE PRECISION,
    lng DOUBLE PRECISION,
    tags TEXT[],                -- [辣, 清淡, 海鲜, 排队, ...]
    signature_dishes TEXT[],    -- 招牌菜
    rating DECIMAL
);

-- 酒店库
CREATE TABLE hotels (
    id UUID PRIMARY KEY,
    name VARCHAR(200),
    city VARCHAR(50),
    district VARCHAR(100),
    price_range VARCHAR(20),    -- budget/mid/luxury
    has_elevator BOOL,
    has_breakfast BOOL,
    has_parking BOOL,
    child_friendly BOOL,
    cancel_policy VARCHAR(100),  -- 退改政策
    distance_to_center_km FLOAT, -- 距市中心距离
    lat DOUBLE PRECISION,
    lng DOUBLE PRECISION
);

-- 交通枢纽库 (PRD 4.1 新增)
CREATE TABLE transport_hubs (
    id UUID PRIMARY KEY,
    city VARCHAR(50),
    name VARCHAR(200),         -- 成都东站 / 双流机场
    hub_type VARCHAR(20),      -- airport / railway / bus / subway
    lat DOUBLE PRECISION,
    lng DOUBLE PRECISION,
    lines TEXT[],              -- 线路（地铁1号线、成渝高铁...）
);

-- 城市基础库 (PRD 4.1 新增)
CREATE TABLE city_info (
    id UUID PRIMARY KEY,
    city VARCHAR(50) UNIQUE,
    climate VARCHAR(100),      -- 亚热带季风 / 温带大陆性
    best_season VARCHAR(50),   -- 3-5月、9-11月
    main_districts TEXT[],     -- [锦江区, 青羊区, ...]
    transport_hubs JSONB,      -- {airport: "双流", railway: "成都东站"}
    peak_months TEXT[],        -- [7, 8] 旅游旺季月份
);

-- 餐饮库补充字段
ALTER TABLE restaurants ADD COLUMN queue_time_min INT;    -- 排队时长
ALTER TABLE restaurants ADD COLUMN cancel_policy VARCHAR(50);  -- 退改政策

-- 酒店库补充字段
ALTER TABLE hotels ADD COLUMN cancel_policy VARCHAR(100);      -- 退改政策
ALTER TABLE hotels ADD COLUMN distance_to_center_km FLOAT;     -- 距市中心距离
```

**待确认**: 结构化数据从哪来？建议方案：
1. 手动标注 Top-20 城市核心景点(每个城市 20-50 个)
2. 爬虫补充(携程/马蜂窝公开数据)
3. 第三方 API(高德 POI 搜索)

### 4.2 RAG 向量检索

```
流程:
  用户需求 → bge-large-zh 向量化
  → pgvector 余弦相似度检索 Top-5
  → 召回: 关联攻略、避坑指南、季节贴士
  → 注入 LLM Prompt 上下文
  → LLM 基于检索结果生成回答

检索策略:
  - 混合检索: 向量相似度(0.6) + BM25关键词(0.4)
  - 分城市索引: 按 destination 过滤，减少检索范围
  - 时效性加权: 近期攻略权重 > 历史攻略
```

### 4.3 实时数据连接器

| 数据源 | 更新频率 | 用途 |
|---|---|---|
| 和风天气 API | 1h | 天气预报、极端天气预警 |
| 高德 POI API | 24h | 景点/餐厅/酒店信息补充 |
| 景区官网 | 12h | 闭园/维修/限流公告 |
| 12306/航司 API | 实时(按需) | 高铁/机票查询 |

---

## 5. 规划决策引擎（核心）

### 5.1 约束求解器

**选型**: Google OR-Tools CP-SAT
**理由**: 
- 生产级约束求解，支持多目标优化
- Python 原生接口
- 支持整数规划、布尔约束、加权目标

```python
from ortools.sat.python import cp_model

def solve_itinerary(pois, constraints, profile):
    model = cp_model.CpModel()
    
    # 变量: poi_i 是否安排在第 d 天第 s 个时段
    X = {}
    for i, poi in enumerate(pois):
        for d in range(profile.travel_days):
            for s in range(MAX_SLOTS_PER_DAY):  # 每天最多 6 个时段
                X[(i, d, s)] = model.NewBoolVar(f'x_{i}_{d}_{s}')
    
    # 约束1: 每个POI最多安排一次
    for i in range(len(pois)):
        model.Add(sum(X[(i,d,s)] for d in range(days) for s in range(slots)) <= 1)
    
    # 约束2: 每天总时长不超过可用时间
    for d in range(days):
        model.Add(sum(
            X[(i,d,s)] * pois[i].duration_minutes
            for i in range(len(pois)) for s in range(slots)
        ) <= DAY_AVAILABLE_MINUTES)
    
    # 约束3: 预算不超限
    model.Add(sum(
        X[(i,d,s)] * pois[i].cost
        for i,d,s in product(...)
    ) <= profile.total_budget)
    
    # 约束4: must_see 必须包含
    for must_poi in must_see_list:
        model.Add(sum(X[(i,d,s)] for d,s in product(...)) == 1)
    
    # 约束5: 每天不超过 max_pois_per_day
    # 约束6: 同category每天不超过2个
    # 约束7: 体力约束(日步行总时长)
    
    # 目标: 最大化偏好匹配度 + 最小化总车程
    model.Maximize(
        sum(X[(i,d,s)] * pois[i].score * PREFERENCE_WEIGHT for ...)
        - sum(transit_cost * TRANSIT_WEIGHT)
    )
    
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    return extract_schedule(solver, X)
```

**待确认**: OR-Tools 是否太重？初期可用改进版贪心（当前方案的约束感知版本），后续替换为 CP-SAT。

### 5.2 路线优化器

```
轻度行程(≤3天, ≤10个POI): 贪心 NN + 2-opt (保留当前方案)
重度行程(>3天 或 >10个POI): 遗传算法
```

**遗传算法方案**:
```python
import random
from deap import base, creator, tools

# 编码: 每段基因 = (poi_index, day, slot)
# 适应度: 偏好匹配度 - 车程惩罚 - 体力惩罚
# 选择: 锦标赛选择
# 交叉: 单点交叉(保持约束)
# 变异: 随机交换两个POI的时段
# 终止: 100代 或 适应度收敛
```

### 5.3 预算分配器

```python
class BudgetAllocator:
    def allocate(self, total_budget, profile):
        # 根据用户画像自动分配比例
        ratios = {
            "transport": 0.25,    # 交通 (含机票/高铁)
            "accommodation": 0.35, # 住宿
            "food": 0.20,         # 餐饮
            "tickets": 0.10,      # 门票
            "reserve": 0.10,      # 备用金
        }
        # 根据出行方式调整
        if profile.transport_mode == "self_drive":
            ratios["transport"] = 0.15  # 自驾省交通费
        
        return {
            k: round(total_budget * v, 2)
            for k, v in ratios.items()
        }
```

### 5.4 动态重规划

```python
class DynamicReplanner:
    triggers = {
        "weather_alert": "暴雨/台风 → 替换室内景点",
        "attraction_closed": "景区闭园 → 删除+替换",
        "flight_cancelled": "航班取消 → 调整Day1行程",
        "user_modify": "增删景点/改天数 → 局部重算",
    }
    
    def replan(self, original_itinerary, trigger_event):
        if trigger_event.type == "weather_alert":
            affected_pois = [p for p in original_itinerary 
                           if p.is_outdoor and p.day == trigger_event.day]
            alternatives = self.knowledge_base.query_indoor(
                city=trigger_event.city, 
                budget=remaining_budget
            )
            return self.replace_pois(original_itinerary, affected_pois, alternatives)
```

### 5.5 多约束权重自定义 (PRD 四.1 新增)

用户可自由设置优化优先级，引擎调整目标函数权重：

```python
class OptimizationWeights:
    """用户可配权重 —— 省钱/省力/打卡/美食优先。"""
    weights: dict[str, float] = {
        "cost_save": 0.25,      # 省钱优先
        "effort_save": 0.25,    # 省力优先（减少步行/车程）
        "poi_count": 0.25,      # 打卡景点数量优先
        "food_quality": 0.25,   # 美食优先
    }
    
    def apply(self, profile: UserProfile) -> dict[str, float]:
        """根据用户画像自动调整权重。"""
        w = dict(self.weights)
        if profile.has_elderly or profile.has_children:
            w["effort_save"] = 0.5
            w["poi_count"] = 0.1
        if "美食" in (profile.interests or []):
            w["food_quality"] = 0.4
            w["cost_save"] = 0.1
        return w
```

### 5.6 人群专属模板引擎 (PRD 四.2 新增)

```python
class PersonaRules:
    """人群约束规则库"""
    
    RULES = {
        "elderly": {
            "max_walk_minutes": 120,
            "max_transit_minutes": 60,
            "avoid_morning_rush": True,
            "require_rest_after_3h": True,
            "prefer_elevator": True,
            "prefer_flat_terrain": True,
        },
        "children": {
            "max_walk_minutes": 90,
            "require_playground_nearby": True,
            "include_kids_activities": True,
            "avoid_night_activities": True,
            "require_frequent_breaks": True,
        },
        "couple": {
            "prefer_scenic_spots": True,
            "prefer_night_view": True,
            "prefer_atmosphere_dining": True,
            "avoid_rush": True,
        },
        "hiking": {
            "match_hiking_duration": True,
            "recommend_gear": True,
            "prefer_nature_lodging": True,
        },
    }
    
    def apply(self, profile: UserProfile) -> dict:
        rules = {}
        if profile.has_elderly:
            rules.update(self.RULES["elderly"])
        if profile.has_children:
            rules.update(self.RULES["children"])
        if profile.travelers_type == "couple":
            rules.update(self.RULES["couple"])
        return rules
```

### 5.7 可行性校验 + 折中方案 (PRD 5.1 新增)

```python
def feasibility_check(profile: UserProfile) -> list[str]:
    """前置校验：检测不可行约束组合，给出折中方案。"""
    conflicts = []
    
    # 例：5天三亚 + 预算3000 + 五星酒店 = 不可行
    if profile.budget_range and profile.travel_days:
        daily = profile.budget_range / profile.travel_days
        if daily < 500:  # 五星酒店日均至少500
            conflicts.append(
                f"预算{profile.budget_range}元住{profile.travel_days}天五星酒店不可行，"
                f"建议：1) 缩短至{int(profile.budget_range/500)}天 "
                f"2) 降级为经济型酒店(¥200/晚) "
                f"3) 提供民宿替代方案"
            )
    return conflicts
```

### 5.8 热门景点错峰安排 (PRD 5.2 新增)

```python
def avoid_peak_hours(schedule: list[DayPlan], pois: list[ScoredPOI]):
    """将热门景点安排在非高峰时段（早8点前或下午4点后）。"""
    for day in schedule:
        for act in day.activities:
            poi = next((p for p in pois if p.name == act.poi_name), None)
            if poi and getattr(poi, 'peak_hours', None):
                # 调整到错峰时段
                if act.start_time and act.start_time < "10:00":
                    continue  # 已经是好时段
                act.start_time = "08:00"
                act.end_time = "10:00"
```

### 5.9 多轮交互迭代回路 (PRD 三 新增)

LangGraph 增加 feedback 节点，用户反馈后局部重算而非全量重跑。

```python
# agent/graph.py 新增节点
def _make_feedback_node():
    """用户反馈节点 —— 局部修改行程，不重跑全链路。"""
    async def feedback_node(state: TravelAgentState) -> dict:
        feedback = state.get("user_feedback", {})
        action = feedback.get("action", "")
        
        if action == "remove_poi":
            # 只删除指定POI，不重跑planner
            poi_name = feedback.get("poi_name")
            for day in state["itinerary"]:
                day["activities"] = [a for a in day["activities"] if a["poi_name"] != poi_name]
        
        if action == "replace_poi":
            # 局部替换一个POI
            ...
        
        if action == "change_days":
            # 增减天数
            ...
        
        return {"itinerary": state["itinerary"], "next_action": "output"}
    
    return feedback_node

# 图路由: output → (if user feedback) → feedback → planner(局部) → output
builder.add_node("feedback", _make_feedback_node())
builder.add_conditional_edges("output", route_after_output, {
    "feedback": "feedback",
    "end": END,
})
builder.add_edge("feedback", "planner")  # 局部重规划
            return self.replace_pois(original_itinerary, affected_pois, alternatives)
```

---

## 6. 工具调用层

### 6.1 工具协议

```python
from pydantic import BaseModel
from typing import Literal

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict  # JSON Schema
    requires_auth: bool = False

# 7 类工具注册表
TOOLS = {
    "transport": [
        ToolDefinition(name="search_flights", ...),
        ToolDefinition(name="search_trains", ...),
        ToolDefinition(name="estimate_fare", ...),  # 打车预估
    ],
    "accommodation": [
        ToolDefinition(name="search_hotels", ...),
        ToolDefinition(name="check_availability", ...),
    ],
    "dining": [
        ToolDefinition(name="search_restaurants", ...),
        ToolDefinition(name="reserve_table", ...),
    ],
    "attractions": [
        ToolDefinition(name="search_attractions", ...),
        ToolDefinition(name="check_tickets", ...),
    ],
    "maps": [
        ToolDefinition(name="get_route", ...),
        ToolDefinition(name="get_distance_matrix", ...),
    ],
    "weather": [
        ToolDefinition(name="get_weather_forecast", ...),
        ToolDefinition(name="get_weather_alert", ...),
    ],
    "documents": [
        ToolDefinition(name="generate_pdf_itinerary", ...),
        ToolDefinition(name="generate_budget_excel", ...),
        ToolDefinition(name="generate_packing_list", ...),
    ],
    # PRD 6 新增
    "rental": [
        ToolDefinition(name="rent_car", ...),            # 租车
    ],
    "guide": [
        ToolDefinition(name="book_guide", ...),          # 导游预约
    ],
    "navigation": [
        ToolDefinition(name="get_navigation", ...),      # 定位导航
    ],
    "policy": [
        ToolDefinition(name="check_holiday_policy", ...),# 节假日政策
    ],
}
```

### 6.2 工具调用流程

```
LLM 输出: { "tool": "search_hotels", "params": { "city": "西安", "budget": 300 } }
         ↓
ToolExecutor.execute("search_hotels", params)
         ↓
结构化结果返回 LLM → 注入规划上下文
```

**待确认**: Function Calling 用 OpenAI 格式还是 LangChain Tool？建议保持 OpenAI-compatible 格式（当前 vLLM 已支持），便于切换模型。

---

## 7. 交互输出层

### 7.1 Markdown 行程 (当前已有)

### 7.2 地图可视化
- **选型**: Leaflet.js + OpenStreetMap
- **集成**: 前端渲染，后端提供经纬度数组
- **每**: 每日路线画线 + POI 标注

### 7.3 预算表格
- **选型**: 前端 HTML Table + 导出 xlsx
- **后端**: 提供结构化 JSON → 前端渲染

### 7.4 PDF 导出
- **选型**: `jsPDF`(前端) 或 `WeasyPrint`(后端)
- **待确认**: 前端还是后端生成？

### 7.5 语音输出
- **选型**: Edge TTS (免费) 或 ElevenLabs (高质)
- **当前**: Hermes 已有 TTS 能力，可复用

### 7.6 交互式修改
- **选型**: 前端拖拽排序 (dnd-kit) + 点击替换
- **待确认**: 优先级暂缓

### 7.7 图片攻略输出 (PRD 7 新增)
- **选型**: AI 图片生成 (DALL-E / Stable Diffusion) → 生成目的地攻略插图
- **场景**: 每日行程配一张 AI 生成的景点氛围图
- **实现**: 后端调用 Image Gen API → 返回 URL → 前端嵌入行程卡片

---

## 8. 运维监控层

| 模块 | 选型 | 说明 |
|---|---|---|
| LLM追踪 | LangSmith / Phoenix | 免费额度够用，可视化调用链 |
| 指标监控 | Prometheus + Grafana | 当前已有 |
| 日志 | structlog → Loki | 结构化日志 |
| 幻觉检测 | 规则引擎 + LLM校验 | 价格/时间/地点与知识库比对 |
| 安全风控 | 规则 + LLM检测 | 过滤违规推荐 |
| 成本控制 | Token配额 + API计数 | 当前已有 |
| 隐私加密 | AES-256-GCM + 一键清除 | 记忆数据加密存储，用户可控删除 |
| 自动记忆更新 | 行程结束 trigger | 总结偏好 → 更新 user_profile_vectors |

---

## 9. 实施路线

| 阶段 | 内容 | 预计 |
|---|---|---|
| P0 | 调度器重构(约束内嵌) ← 当前 | 进行中 |
| P1 | 知识库建设(PG结构化数据 + pgvector) | 2周 |
| P2 | 槽位升级(新增人群/约束/忌口) | 1周 |
| P3 | RAG检索接入 | 1周 |
| P4 | 用户画像向量化(Chroma) | 1周 |
| P5 | LangGraph编排迁移 | 2周 |
| P6 | 工具调用层(Function Calling) | 2周 |
| P7 | 动态重规划 | 2周 |
| P8 | 多模态输出(地图/PDF/表格) | 2周 |
| P9 | 交通库 + 城市库 + 退改政策 | 1周 |
| P10 | 多约束权重 + 人群模板 + 错峰调度 | 2周 |
| P11 | 可行性校验 + 折中方案 | 1周 |
| P12 | 多轮交互迭代回路 (feedback node) | 1周 |
| P13 | 租车/导游/导航/节假日 工具 | 1周 |
| P14 | 图片攻略 + 语音输出 | 1周 |
| P15 | 隐私加密 + 自动记忆更新 | 1周 |

---

## 10. 待确认清单 ⚠️

| # | 问题 | 选项 |
|---|---|---|
| 1 | 多模态(图片)初期做不做？ | 做 / 暂缓 |
| 2 | Embedding 模型用哪个？ | bge-large-zh-v1.5 / text2vec |
| 3 | 向量库 Chroma 还是直接 pgvector？ | Chroma / pgvector |
| 4 | 结构化数据从哪来？ | 手动标注 / 爬虫 / 第三方API |
| 5 | 规划引擎用 OR-Tools 还是改进贪心？ | OR-Tools / 贪心升级 |
| 6 | PDF/地图输出前端还是后端？ | 前端 / 后端 |
| 7 | LangGraph 迁移现在做还是等稳定后？ | 现在 / 后期 |
| 8 | 实时交通和预订API初期接入吗？ | 接入 / 暂缓 |
