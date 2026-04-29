# 旅行 Agent 系统 - 技术方案 v2（严格对齐 v4 设计文档）

> **目标**：让 AI 能按此方案逐步理解和实现，与 v4 设计文档严格对齐
> **原则**：说清楚"做什么"、"为什么"、"怎么做"，不贴冗余代码

---

## 一、架构总览

### 1.1 核心架构图

```
用户输入（自然语言）
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  ① Intent Recognition Agent（LLM 驱动，唯一串行入口）        │
│     - 理解用户意图（7 种类型）                                │
│     - 提取关键信息实体                                        │
│     - 判断关键信息是否缺失，决定追问或进入流程                │
│     - 检测偏好变更                                            │
└──────────────────────────┬──────────────────────────────────┘
                           │ IntentResult（意图 + 实体 + 变更）
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  ② StateGraph（LangGraph 状态图，声明式编排）                │
│     - 根据 intent 条件路由到对应子图/节点                     │
│     - fan-out/fan-in 管理并行执行（无需 asyncio.gather）      │
│     - 状态在节点间自动传递（TypedDict State）                 │
│     - interrupt 处理人机交互（等待用户确认）                  │
│     - Checkpointer 自动持久化对话状态                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   ┌─────────┐      ┌──────────┐       ┌──────────┐
   │ generate│      │ update_  │       │  q_a     │
   │流程     │      │preferences│      │ 流程     │
   └────┬────┘      └────┬─────┘       └────┬─────┘
        │                │                  │
        ▼                ▼                  ▼
   Agent 并行调用    Preference &      Q&A Agent
   （见下文）        Budget Agent      （RAG + 搜索）
└─────────────────────────────────────────────────────────────┘

Agent 层（10 个 Agent，能并行尽量并行）
├─ Intent Recognition Agent（串行，入口）
├─ Information Collection Agent（串行，追问补全）
├─ Realtime Query Agent（并行，POI/天气/价格查询）
├─ Preference & Budget Agent（串行，偏好管理 + 预算计算）
├─ Itinerary Planner Agent（串行，行程规划算法）
├─ Q&A Agent（可并行，问答）
├─ Memory Management Agent（后台异步，记忆持久化）
├─ Proposal Generation Agent（串行，方案文本生成）
├─ Validation Agent（并行，约束校验）
└─ Map & Route Agent（并行，路线优化）

Skill 层（8 个 Skill，供 Agent 调用）
├─ web_search ────────── 网络搜索（DuckDuckGo）
├─ web_crawler ───────── 网页爬取（aiohttp + BeautifulSoup）
├─ poi_search ────────── POI 搜索整合（搜索 + 爬取 + LLM 提取）
├─ weather_query ─────── 天气查询
├─ route_calculation ─── 路线计算（距离矩阵 + 时间估算）
├─ price_query ───────── 价格查询（门票/餐饮/酒店）
├─ memory_store ──────── 记忆存储
└─ memory_retrieve ───── 记忆检索

基础设施层
├─ PostgreSQL ────────── 持久化存储（对话/行程/偏好/预算）
├─ Redis ─────────────── 热点数据缓存 + 速率限制
├─ StateGraph ────────── LangGraph 状态图（声明式编排 + 条件路由）
├─ Checkpointer ──────── 对话状态持久化（PostgreSQL，自动 checkpoint）
├─ EventBus ──────────── 前端推送事件总线（预算/偏好更新推送 WebSocket）
└─ LLMClient ─────────── LLM 调用封装（chat + structured_call）
```

### 1.2 设计原则（与 v4 一致）

1. **LLM 做判断，代码做执行**
   - LLM 负责：意图识别、偏好变更检测、方案文本生成、追问措辞
   - 代码负责：约束校验、路径计算、数据聚合、状态持久化、流程编排

2. **能并行尽量并行（LangGraph fan-out/fan-in）**
   - 无依赖的节点通过 Graph `fan-out` 并行执行（如 POI 搜索 / 天气查询 / 预算初始化同时启动）
   - 多个上游节点汇聚到同一节点时，LangGraph 自动 `fan-in`（等待全部完成后才执行下游）
   - Agent 内部多个 Skill 调用仍用 `asyncio.gather`

3. **LLM 不直接操作数据，通过 Agent 接口**
   - LLM 输出意图和参数 → Graph Node 调用 Agent → Agent 操作数据/调用 Skill

4. **记忆是核心资产**
   - 每条对话记录、每次行程、每次偏好变更都要持久化
   - 新对话加载历史偏好，驱动个性化

5. **事件驱动解耦**
   - Agent 间通信走 StateGraph edges（条件路由 + 子图调用），不再通过 EventBus
   - EventBus 保留用于前端推送（预算/偏好更新实时推送到 WebSocket）
   - PreferenceChanged 通过 Graph 条件边触发重规划子图

---

### 1.3 LangGraph State 与 Graph 结构

**State（TypedDict）**：Graph 中所有节点共享的状态对象，节点只读取需要的字段，返回要修改的字段。

```python
class ItineraryState(TypedDict):
    session_id: str
    user_id: str
    user_input: str
    messages: Annotated[list[dict], add]   # 累积，operator.add
    
    # Intent 结果
    intent: str | None
    intent_confidence: float
    user_entities: dict
    missing_required: list[str]
    preference_changes: list[dict]
    clarification_questions: list[str]
    
    # 用户画像与查询结果
    user_profile: dict | None
    candidate_pois: list[dict]
    weather_data: list[dict]
    price_data: dict
    
    # 行程与校验
    current_itinerary: list[dict] | None
    itinerary_status: str
    validation_result: dict | None
    optimized_routes: dict
    
    # 面板与输出
    budget_panel: dict | None
    preference_panel: dict | None
    assistant_response: str | None
    proposal_text: str | None
    
    # 控制流
    needs_clarification: bool
    needs_replan: bool
    waiting_for_confirmation: bool
    error: str | None
```

**Graph 拓扑**：

```
用户输入 ──▶ intent_node ──▶ route_intent（条件边）
                              ├─ generate ──▶ [poi_search / weather / budget_init] 并行 fan-out
                              │                  ↓ 自动 fan-in（全部完成后）
                              │               planner_node
                              │                  ↓
                              │               [validation / route_optimize / budget_calc] 并行 fan-out
                              │                  ↓ 自动 fan-in
                              │               proposal_node
                              │                  ↓
                              │               interrupt（等待用户确认）
                              │                  ↓ resume
                              │               save_memory_node
                              │
                              ├─ update_prefs ──▶ update_prefs_node
                              │                      ↓（若 needs_replan）
                              │                   planner_node（复用）
                              │
                              ├─ qa ──▶ qa_node
                              └─ confirm ──▶ save_memory_node
```

**并行执行的实现**：LangGraph 中一个节点的多个出边连接的下游节点会自动并行执行；多个上游节点指向同一下游节点时，自动等待全部完成后才执行（无需手写 `asyncio.gather`）。

---

## 二、10 个 Agent 职责定义

### Agent 1：Intent Recognition Agent（意图识别 Agent）

**职责**：系统的"大脑入口"。理解用户说什么、想干什么、给了什么信息、改了什么偏好。

**输入**：
- 用户原始输入（当前轮）
- 当前会话的对话历史（最近 10 轮）
- 当前 State 中的用户画像

**输出（IntentResult）**：
```
intent: 7 种之一（generate_itinerary / modify_itinerary / update_preferences /
                   query_info / confirm_itinerary / view_history / chitchat）
confidence: 0-1 置信度
user_entities: 提取的实体（目的地/天数/预算/偏好等）
missing_required: 缺失的必需字段（destination / travel_days / travel_dates）
missing_recommended: 缺失的重要字段（人数/预算/同行类型）
preference_changes: 检测到的偏好变更列表（字段/旧值/新值）
clarification_questions: 需要追问的问题（最多 2 个）
reasoning: LLM 判断理由
```

**实现要点**：
- 使用 LLM structured_call（JSON mode，temperature=0.3）
- System Prompt 要包含：7 种意图定义、关键信息定义、判断规则、输出格式
- 提取实体后，与当前 State 中的 user_profile 对比，检测偏好变更
- 如果 confidence < 0.7，追加 clarifying 追问

**关键代码逻辑**：
```python
# 1. 构建 Prompt（系统提示 + 当前画像 + 对话历史 + 用户输入）
messages = [
    {"role": "system", "content": INTENT_PROMPT},
    {"role": "system", "content": f"当前用户画像：{profile}"},
    *history_messages,
    {"role": "user", "content": user_input}
]

# 2. LLM 结构化输出
result = await llm.structured_call(messages, IntentResult, temperature=0.3)

# 3. 后处理：对比新旧偏好，生成 preference_changes
changes = detect_preference_changes(result.user_entities, profile)
result.preference_changes = changes
```

---

### Agent 2：Information Collection Agent（信息收集 Agent）

**职责**：当关键信息缺失时，生成自然的追问，收集用户信息。

**输入**：
- missing_required 列表
- missing_recommended 列表
- 当前已收集的部分信息

**输出**：
- 1-2 个自然语言追问问题（带选项减少输入成本）

**实现要点**：
- 追问问题要自然，像旅行顾问在问，不要像表单
- 提供选项："预算大概多少？2000/5000/8000，或者告诉我范围"
- 用户说"不知道"时用默认值并告知

---

### Agent 3：Realtime Query Agent（实时信息查询 Agent）

**职责**：通过网络爬虫和搜索，获取最新的景点、天气、价格信息。

**输入**：查询参数（城市、日期、关键词列表）

**输出**：
- 候选 POI 列表（20-30 个）
- 天气预报（按天）
- 景点价格信息

**内部并行策略**：
```python
# 同时发起 3 个查询（互不影响）
tasks = [
    query_pois(city, interests, food_prefs),     # 景点 + 美食
    query_weather(city, dates),                   # 天气
    query_prices(city, poi_names)                 # 价格（POI 查完后再查）
]
pois, weather = await asyncio.gather(tasks[0], tasks[1])
# prices 在 pois 返回后单独查
```

**实现要点**：
- 使用 DuckDuckGo 搜索（无需 API Key）
- 搜索结果 URL 用 aiohttp 异步爬取
- 爬取结果用 LLM 提取结构化 POI 数据
- 本地缓存（Redis）减少重复爬取，TTL 按数据类型设置
- 控制爬取频率（1 req/s），轮换 User-Agent
- 爬取失败时降级：使用本地种子数据

---

### Agent 4：Preference & Budget Agent（偏好与预算管理 Agent）

**职责**：
- 管理用户偏好（动态更新，新偏好覆盖旧偏好）
- 计算预算面板（分项估算 + 进度条）
- 检测偏好变更，更新 State，Graph 条件边自动触发重规划

**输入**：偏好更新指令 或 行程数据

**输出**：更新后的 UserProfile / BudgetPanel

**关键逻辑**：
```python
# 偏好更新
async def update_preferences(session_id, changes):
    for change in changes:
        setattr(profile, change.field, change.new_value)
        # 记录变更历史
        profile.preference_history.append({timestamp, field, old, new})
    
    # 保存到数据库
    await db.save(profile)
    
    # 更新 State，Graph 条件边自动触发重规划子图
    # （不需要 event_bus.publish，LangGraph 中通过 state["needs_replan"] 控制流）

# 预算计算
def calculate_budget(itinerary, profile):
    breakdown = {accommodation, meals, transport, tickets, shopping, buffer}
    # 住宿：基于城市消费水平 × 天数（一线 1.3，二线 1.0，三线 0.8）
    # 餐饮：基于用餐安排 × 人均
    # 交通：基于路线距离估算
    # Buffer：一线 15%，二线 12%，三线 10%
    return BudgetPanel
```

---

### Agent 5：Itinerary Planner Agent（行程规划 Agent）

**职责**：核心算法执行。将候选 POI 转化为可执行的每日行程。

**输入**：候选 POI 列表 + 用户偏好 + 约束条件 + 天气数据

**输出**：每日行程列表（DayPlan[]）

**算法流程（6 层）**：

```
Step 1: POI 偏好打分
  - 根据用户兴趣、饮食偏好、节奏、同行人类型给每个 POI 打分
  - 公式：base_score + interest_match * 0.2 + food_match * 0.3 + pace_match * 0.1

Step 2: 区域聚类（DBSCAN）
  - 输入：POI 地理坐标
  - 参数：eps 按城市类型动态（密集城区 1.0km / 一般城区 1.5km / 郊区 3.0km）
  - 输出：若干地理簇 + 噪声点

Step 3: 时间窗口硬约束标记
  - morning_only（上午独有）：标记为固定位置
  - afternoon_only（下午独有）：标记为固定位置
  - evening_only（晚上独有）：标记为固定位置
  - flexible（灵活）：参与后续优化

Step 4: 日分配（多目标贪心）
  - must_include 景点优先分配到指定日期
  - 时间约束景点分配到合适时段
  - 大簇优先分配到独立天
  - 小簇合并（距离合理 + 偏好互补）
  - 噪声点填充到有空余的天

Step 5: 日内路线优化（2-opt）
  - 分离固定点（时间约束）和灵活点
  - 灵活点用 2-opt 优化路线距离
  - 考虑从酒店出发并返回酒店（Closed TSP）
  - 混合交通：同一区域步行，跨区域公共交通

Step 6: 时间表生成 + 用餐插入
  - 每天 09:00 开始
  - 插入交通时间（基于路线计算）
  - 午餐窗口 11:30-13:30，晚餐 17:30-19:30
  - 距离上一餐 >= 3.5 小时才插入
  - 在下一个景点附近搜索匹配偏好的餐厅
  - 检查开放时间，不开放的景点标记为冲突（保留但标注）
```

**实现要点**：
- DBSCAN 使用 sklearn.cluster.DBSCAN，距离用 haversine
- 2-opt 设置最大迭代次数，防止超时
- 用餐搜索调用 poi_search Skill（关键词：城市 + "餐厅" + 偏好）

---

### Agent 6：Q&A Agent（问答 Agent）

**职责**：独立回答用户的非行程规划问题。不触发行程更新。

**输入**：用户问题 + RAG 知识库检索结果 + 网络搜索结果

**输出**：自然语言回答

**实现要点**：
```python
async def answer(question, city=None):
    # 并行查询
    rag_results = await rag_retriever.search(question, city)
    web_results = await web_search(f"{city} {question}")
    
    # LLM 整合生成回答
    prompt = f"基于以下信息回答：\n知识库：{rag_results}\n网络：{web_results}"
    answer = await llm.chat(prompt, temperature=0.7)
    
    return answer
```

---

### Agent 7：Memory Management Agent（记忆管理 Agent）

**职责**：后台异步管理所有用户相关的持久化数据。不阻塞主流程。

**管理的 4 类记忆**：

```
1. 对话记忆（Conversation Memory）
   - 每轮对话：用户输入 + Agent 回复 + 意图 + 使用的 Tools
   - 表：conversations

2. 历史行程记忆（Itinerary Memory）
   - 每次确认的行程：完整每日安排 + 每天活动 + 费用
   - 表：itineraries

3. 偏好记忆（Preference Memory）
   - 每次行程的偏好快照
   - 偏好变更历史（什么时间改了什么）
   - 长期偏好模式（从多次行程总结）

4. 预算记忆（Budget Memory）
   - 每次行程的预算设置 vs 实际花费
   - 城市消费水平参考
```

**核心方法**：
```python
# 保存对话轮次（每次用户-Assistant 交互后调用）
async def save_conversation_turn(session_id, user_msg, assistant_msg, intent, tools)

# 保存行程（用户确认后调用）
async def save_itinerary(session_id, itinerary, profile, budget, confirmed=False)

# 获取用户完整记忆（新对话时调用）
async def get_user_memory(user_id):
    recent_itineraries = 最近 3 次确认的行程
    preference_patterns = 从多次行程总结的偏好模式
    recent_conversations = 最近 50 轮对话
    return UserMemory

# 提取偏好模式（从 history 中总结）
def extract_preference_patterns(itineraries):
    # 统计：最常选的节奏、最喜欢的口味、平均预算、常去目的地
    return {preferred_pace, preferred_food, avg_budget, favorite_cities}
```

**调用时机**：
- `save_conversation_turn`：每轮对话结束后，后台 async 调用
- `save_itinerary`：用户点击"确认行程"后
- `get_user_memory`：新对话开始时，加载历史偏好到 Prompt

---

### Agent 8：Proposal Generation Agent（方案生成 Agent）

**职责**：将行程数据转化为自然语言方案，生成推荐理由、预算明细、风险提示。

**输入**：行程数据 + 用户偏好 + 预算面板 + 天气 + 校验结果

**输出**：带格式的自然语言方案文本

**实现要点**：
- 使用 LLM（temperature=0.7，max_tokens=4096）
- Prompt 包含：行程概览、每日安排、预算明细、天气提示
- 要求语气友好，像旅行顾问
- 输出末尾提供"确认行程"和"继续修改"的引导

---

### Agent 9：Validation Agent（校验 Agent）

**职责**：对生成的行程进行多维度并行校验。

**输入**：行程草案 + 用户画像

**输出**：ValidationResult（是否通过 + 各维度得分 + 修改建议）

**校验维度（并行执行）**：
```python
async def validate(itinerary, profile):
    tasks = [
        check_budget(itinerary, profile),           # 预算是否超支
        check_time_feasibility(itinerary),           # 时间是否合理
        check_poi_existence(itinerary),              # POI 是否真实存在（抽样 30%）
        check_opening_hours(itinerary),              # 是否在开放时间内
        check_preference_coverage(itinerary, profile) # 偏好是否被覆盖
    ]
    results = await asyncio.gather(*tasks)
    return ValidationResult
```

**实现要点**：
- POI 真实性校验：抽样 30% 的景点，通过 web_search 验证是否存在
- 预算校验：总花费 vs 预算上限
- 开放时间校验：每个景点的 open_time / close_time
- 偏好覆盖度：用户兴趣中有多少被行程覆盖

---

### Agent 10：Map & Route Agent（地图与路线 Agent）

**职责**：地理计算、路线优化。

**输入**：POI 坐标列表 + 酒店位置 + 交通模式

**输出**：最优路线 + 每段交通时间

**核心方法**：
```python
# 获取 POI 坐标（优先本地数据库，缺失时搜索）
async def get_coordinates(poi_name, city)

# 计算距离矩阵
async def get_distance_matrix(points, mode="transit")

# 2-opt 路线优化
def two_opt(route, dist_matrix):
    improved = True
    while improved:
        improved = False
        for i in range(n-1):
            for j in range(i+2, n):
                delta = calc_2opt_delta(dist_matrix, route, i, j)
                if delta < 0:
                    reverse_segment(route, i+1, j)
                    improved = True
    return route

# 日内路线优化（供 Itinerary Planner 调用）
async def optimize_daily_route(activities, hotel_location):
    # 1. 获取所有坐标
    # 2. 计算距离矩阵
    # 3. 2-opt 优化
    # 4. 返回优化后的顺序 + 交通时间
```

**实现要点**：
- 坐标优先查本地数据库（seed_data），缺失时通过搜索获取
- 距离计算用 haversine 公式（经纬度球面距离）
- 2-opt 设置最大迭代次数（如 100 次）防止超时
- 步行距离 < 1km 用步行时间，> 1km 用公共交通时间估算

---

## 三、8 个 Skill 定义

Skill 是 Agent 可调用的原子能力。通过函数 + Pydantic 参数定义。

| # | Skill | 职责 | 输入 | 输出 |
|---|-------|------|------|------|
| 1 | **web_search** | 网络搜索 | query, top_n | SearchResult[] |
| 2 | **web_crawler** | 网页爬取 | url, extract_fields | CrawledPage |
| 3 | **poi_search** | POI 搜索整合 | city, keywords, category | ScoredPOI[] |
| 4 | **weather_query** | 天气查询 | city, start_date, end_date | WeatherDay[] |
| 5 | **route_calculation** | 路线计算 | origin, destination, mode | RouteInfo |
| 6 | **price_query** | 价格查询 | poi_name, city, price_type | PriceInfo |
| 7 | **memory_store** | 记忆存储 | memory_type, data | StoreResult |
| 8 | **memory_retrieve** | 记忆检索 | memory_type, query, filters | MemoryRecord[] |

**Skill 调用方式**：
```python
# Agent 内部直接调用 Skill 函数
# 不通过 LLM function calling，减少延迟
results = await poi_search_skill.search_pois(city="成都", keywords=["美食", "拍照"])
```

---

## 四、EventBus 事件设计

### 4.1 事件类型

```python
EVENT_TYPES = {
    # 用户相关
    "UserInputReceived",        # 收到用户输入
    "IntentRecognized",         # 意图识别完成
    "PreferenceChanged",        # 偏好变更 ⭐ 核心事件
    "BudgetUpdated",            # 预算更新
    
    # 行程相关
    "ItineraryGenerated",       # 行程生成完成
    "ItineraryConfirmed",       # 行程确认
    "ItineraryModified",        # 行程修改
    
    # 数据相关
    "POIDataLoaded",           # POI 数据加载完成
    "WeatherDataLoaded",       # 天气数据加载完成
}
```

### 4.2 核心事件订阅关系

```python
# PreferenceChanged 事件订阅者
event_bus.subscribe("PreferenceChanged", ItineraryPlannerAgent.on_preference_changed)
event_bus.subscribe("PreferenceChanged", PreferenceBudgetAgent.on_preference_changed)
event_bus.subscribe("PreferenceChanged", MemoryManagementAgent.on_preference_changed)

# ItineraryGenerated 事件订阅者
event_bus.subscribe("ItineraryGenerated", ValidationAgent.on_itinerary_generated)
event_bus.subscribe("ItineraryGenerated", PreferenceBudgetAgent.on_itinerary_generated)
event_bus.subscribe("ItineraryGenerated", ProposalGenerationAgent.on_itinerary_generated)
event_bus.subscribe("ItineraryGenerated", MemoryManagementAgent.on_itinerary_generated)

# ItineraryConfirmed 事件订阅者
event_bus.subscribe("ItineraryConfirmed", MemoryManagementAgent.on_itinerary_confirmed)
```

**LangGraph 中的变化**：
- 上述订阅关系在 LangGraph 中由 Graph edges 替代（如 `ItineraryGenerated` 后自动路由到 validation_node / route_node / budget_calc_node）
- EventBus 保留用于**前端推送**（`BudgetUpdated`、`PreferenceUpdated` 推送到 WebSocket）
- Agent 间不再通过 EventBus 通信，改为 State 传递 + 条件边路由

**为什么用事件而非直接调用**：
- 解耦：发布者不知道有哪些订阅者
- 可扩展：新增订阅者不需要修改发布者
- 异步：Memory Agent 等后台任务不阻塞主流程

---

## 五、核心流程设计

### 5.1 流程 A：从零生成行程

```
用户输入 → "我想去成都玩 4 天，预算 3000，喜欢吃辣"

Step 1: Intent Recognition Agent（串行）
  - 意图：generate_itinerary
  - 提取：destination=成都, travel_days=4, budget=3000, food=[辣]
  - 缺失：travel_dates（追问）
  - 如果有缺失 → collect_info_node 生成追问 → Graph 到达 END，等待用户回复
  - 如果完整 → 进入 Step 2

Step 2: 并行查询（Graph fan-out，3 个节点并行执行）
  ├─ poi_search_node: search_pois(city=成都, keywords=[美食,拍照])
  │   输出：20-30 个候选 POI → 写入 state["candidate_pois"]
  ├─ weather_node: query_weather(city=成都, dates=[...])
  │   输出：4 天天气预报 → 写入 state["weather_data"]
  └─ budget_init_node: init_panel(profile)
      输出：PreferencePanel 初始数据 → 写入 state["budget_panel"]

  LangGraph 自动 fan-in（3 个节点都完成后）→ 进入 Step 3

Step 3: Itinerary Planner Agent（串行，依赖 Step 2 结果）
  - 输入：候选 POI + 天气 + 偏好
  - 执行：POI 打分 → 区域聚类 → 时间约束标记 → 日分配 → 2-opt 优化 → 时间表生成 + 用餐插入
  - 输出：DayPlan[]（4 天行程草案）

Step 4: 并行校验 + 路线精确计算（Graph fan-out，4 个节点并行执行）
  ├─ validation_node: validate(itinerary, profile)
  │   内部并行：预算/时间/POI真实性/开放时间/偏好覆盖
  │   输出：ValidationResult → 写入 state["validation_result"]
  ├─ route_node: batch_optimize_routes(itinerary, hotel)
  │   每天路线并行优化
  │   输出：精确路线 + 交通时间 → 写入 state["optimized_routes"]
  ├─ budget_calc_node: calculate_budget_breakdown(itinerary, profile)
  │   输出：BudgetPanel（分项明细）→ 写入 state["budget_panel"]
  └─ memory_node: save_conversation_turn(...) [后台异步]
      不等待，不阻塞（Node 返回空 dict，不阻断 fan-in）

  LangGraph 自动 fan-in（4 个节点都完成后）→ 进入 Step 5

Step 5: Proposal Generation Agent（串行，依赖 Step 3/4 结果）
  - 输入：行程 + 预算 + 天气 + 校验结果
  - LLM 生成自然语言方案（temperature=0.7）
  - 输出：方案文本 → 写入 state["proposal_text"]

Step 6: 输出给用户 + interrupt 等待确认
  - 自然语言描述行程
  - 中间面板：行程时间线（可展开每日详情）
  - 右侧面板：预算进度条 + 偏好标签
  - 提供 [确认行程] / [继续修改] 按钮
  - **LangGraph interrupt**：Graph 执行暂停，状态自动 checkpoint

Step 7: 用户点击"确认行程"
  - WebSocket 收到用户回复，调用 `Command(resume="confirm")`
  - Graph 从 checkpoint 恢复，路由到 save_memory_node
  - Memory Management Agent 保存为 confirmed 行程
  - 左侧"当前行程"变为可点击
```

### 5.2 流程 B：对话中更新偏好

```
用户输入 → "对了，我其实也爱吃甜品"

Step 1: Intent Recognition Agent
  - 意图：update_preferences
  - 检测到：food_preferences += [甜品]

Step 2: Preference & Budget Agent
  - 追加偏好
  - 更新 PreferencePanel（右侧实时更新）
  - 发布 PreferenceChanged 事件

Step 3: Graph 条件边触发重规划（子图调用）
  ├─ update_prefs_node 设置 state["needs_replan"] = True
  ├─ 条件边检测到 needs_replan → 路由到 planner_node（复用生成子图）
  │   - 重新搜索甜品店
  │   - 增量更新行程（在每天下午插入甜品店）
  ├─ budget_calc_node（并行）
  │   - 重新计算预算（甜品增加费用）
  │   - 更新 BudgetPanel
  └─ memory_node（并行，后台异步）
      - 记录偏好变更历史

Step 4: Proposal Generation Agent
  - 生成更新后的方案
  - 高亮变更部分（"已为你新增甜品偏好..."）

Step 5: 输出给用户
  - 展示更新后的行程和预算
  - 提供 [确认行程] 按钮
```

### 5.3 流程 C：问答模式

```
用户输入 → "成都有什么必吃的美食？"

Step 1: Intent Recognition Agent
  - 意图：query_info
  - 不是行程相关操作

Step 2: 并行启动
  ├─ Q&A Agent
  │   - 检索 RAG 知识库
  │   - 网络搜索补充
  │   - LLM 生成回答
  └─ Memory Management Agent（后台异步记录对话）

Step 3: 输出给用户
  - 自然语言回答
  - 不触发行程更新
  - 不更新预算/偏好面板
```

---

### 5.4 循环终止条件（AgentLoopTerminator）

**职责**：判断单轮对话何时结束。LLM 判断 + 代码校验结合，避免对话无限循环。

**设计原则**：对话型 Agent 的终止需要理解语境，不能完全靠确定性代码。

```python
class AgentLoopTerminator:
    """
    循环终止：LLM 判断 + 代码校验结合。
    """

    def should_stop(self, state: ItineraryState, llm_decision: dict) -> bool:
        # 条件 1（LLM 判断）：LLM 认为本轮对话已完成
        llm_thinks_complete = llm_decision.get("is_conversation_complete", False)
        
        # 条件 2（代码校验）：没有待执行的工具/Agent 调用
        no_pending_calls = len(state.get("pending_agent_calls", [])) == 0
        
        # 条件 3（代码校验）：输出草稿已准备
        response_ready = state.get("assistant_response") is not None
        
        # 条件 4（代码校验）：非阻塞状态（不在追问中）
        not_waiting_input = not state.get("needs_clarification", False)
        
        # 终止条件：LLM 认为完成 且 代码校验全部通过
        return llm_thinks_complete and no_pending_calls and response_ready and not_waiting_input
```

**4 个终止条件**：

| # | 条件 | 类型 | 说明 |
|---|------|------|------|
| 1 | `llm_thinks_complete` | LLM 判断 | LLM 认为本轮对话已完成 |
| 2 | `no_pending_calls` | 代码校验 | 没有待执行的 Agent/Skill 调用 |
| 3 | `response_ready` | 代码校验 | 输出草稿已准备（assistant_response 不为空）|
| 4 | `not_waiting_input` | 代码校验 | 非阻塞状态，不在追问/等待用户确认中 |

**与 LangGraph interrupt 的协作**：
- `interrupt` 处理**人机交互暂停**（如等待用户确认行程）
- `AgentLoopTerminator` 处理**对话轮次终止**（判断本轮是否已完成）
- 两者互补：`interrupt` 暂停时 `not_waiting_input=False`，不会误终止

---

## 六、数据模型设计（核心字段）

### 6.1 用户画像（UserProfile）

```python
class UserProfile:
    # 必需关键信息
    destination: str | None          # 目的地城市
    travel_days: int | None          # 旅行天数
    travel_dates: str | None         # 旅行日期

    # 重要信息
    travelers_count: int = 1         # 出行人数
    travelers_type: str | None       # 同行类型（独自/情侣/亲子/朋友/父母）
    budget_range: float | None       # 预算范围

    # 偏好信息
    food_preferences: list[str]      # 饮食偏好（辣/清淡/海鲜/甜品...）
    interests: list[str]             # 兴趣标签（历史/自然/拍照/美食...）
    pace: str = "moderate"           # 节奏（relaxed/moderate/intensive）
    accommodation_preference: str | None  # 住宿偏好
    special_requests: list[str]      # 特殊要求

    # 历史
    preference_history: list[dict]   # 偏好变更历史
```

### 6.2 行程（Itinerary / DayPlan / Activity）

```python
class Activity:
    poi_name: str                    # 景点名称
    poi_id: str | None               # POI ID
    category: str                    # 类别（attraction/restaurant/...）
    start_time: str | None           # 开始时间 HH:MM
    end_time: str | None             # 结束时间 HH:MM
    duration_min: int                # 持续时间（分钟）
    ticket_price: float | None       # 门票价格
    meal_cost: float | None          # 餐饮费用
    transport_cost: float | None     # 交通费用
    location: Location | None        # 地理坐标
    recommendation_reason: str       # 推荐理由
    transit_from_prev: dict | None   # 从上一地点的交通信息
    time_constraint: str = "flexible" # 时间约束类型

class DayPlan:
    day_number: int                  # 第几天
    date: str | None                 # 日期
    theme: str | None                # 当日主题
    activities: list[Activity]       # 活动列表
    total_cost: float = 0            # 当日总费用
    total_walking_steps: int = 0     # 当日步行数
    total_transit_time_min: int = 0  # 当日交通时间

class ItineraryRecord:
    record_id: str
    session_id: str
    destination: str
    travel_days: int
    daily_plans: list[DayPlan]       # 每天详细行程
    preference_snapshot: dict        # 生成时的偏好快照
    budget_snapshot: dict            # 生成时的预算快照
    status: str = "draft"            # draft / confirmed / completed
```

### 6.3 面板数据（BudgetPanel / PreferencePanel）

```python
class BudgetPanel:
    total_budget: float | None       # 总预算
    spent: float = 0                 # 预计花费
    remaining: float | None          # 剩余
    breakdown: dict                  # 分项：住宿/餐饮/交通/门票/购物/buffer
    status: str                      # within_budget / over_budget

class PreferencePanel:
    destination: str | None
    travel_days: int | None
    travel_dates: str | None
    travelers_count: int | None
    travelers_type: str | None
    budget_range: float | None
    food_preferences: list[str]
    interests: list[str]
    pace: str | None
    special_requests: list[str]
```

### 6.4 数据库表设计（与 v4 一致）

```sql
-- users: 用户表
-- sessions: 会话表（destination, travel_days, status）
-- conversations: 对话记录（user_message, assistant_response, intent, tools_used）
-- itineraries: 行程表（daily_plans JSON, preference_snapshot JSON, budget_snapshot JSON）
-- preference_changes: 偏好变更记录（field, old_value, new_value, source_message）
```

---

## 七、记忆系统设计

### 7.1 4 类记忆

```
┌─────────────────────────────────────────────┐
│  1. 对话记忆（Conversation Memory）            │
│     - 表：conversations                        │
│     - 内容：每轮用户输入 + Agent回复 + 意图 + 工具 │
│     - 用途：回顾对话上下文                      │
│     - 保存时机：每轮对话结束后（后台异步）       │
├─────────────────────────────────────────────┤
│  2. 历史行程记忆（Itinerary Memory）           │
│     - 表：itineraries                          │
│     - 内容：完整行程 + 偏好快照 + 预算快照       │
│     - 用途：查看历史行程、提取偏好模式          │
│     - 保存时机：用户点击"确认行程"后            │
├─────────────────────────────────────────────┤
│  3. 偏好记忆（Preference Memory）              │
│     - 表：preference_changes                   │
│     - 内容：每次偏好变更（字段/旧值/新值/时间）  │
│     - 用途：追踪用户偏好变化、提取长期模式       │
│     - 保存时机：偏好变更时                      │
├─────────────────────────────────────────────┤
│  4. 预算记忆（Budget Memory）                  │
│     - 表：itineraries.budget_snapshot          │
│     - 内容：每次行程的预算 vs 实际花费           │
│     - 用途：了解用户消费水平、优化预算建议       │
│     - 保存时机：行程确认时                      │
└─────────────────────────────────────────────┘
```

### 7.2 记忆加载（新对话时）

```python
async def load_memory_for_session(user_id):
    # 1. 获取最近 3 次确认的行程
    recent_itineraries = await db.get_recent_itineraries(user_id, limit=3)
    
    # 2. 提取偏好模式
    patterns = extract_preference_patterns(recent_itineraries)
    #    例：{"preferred_pace": "relaxed", "preferred_food": ["辣"], "avg_budget_per_day": 800}
    
    # 3. 获取最近 50 轮对话
    recent_conversations = await db.get_recent_conversations(user_id, limit=50)
    
    # 4. 格式化为 LLM Prompt 的一部分
    memory_text = format_memory_for_llm(patterns, recent_itineraries)
    
    return memory_text
```

### 7.3 记忆在 Intent Recognition 中的使用

```python
# Intent Recognition Agent 的 Prompt 中注入历史记忆
messages = [
    {"role": "system", "content": INTENT_PROMPT},
    {"role": "system", "content": f"用户历史偏好：{memory_text}"},
    ...
]
```

---

## 八、RAG 知识库设计（MVP 简化版）

### 8.1 职责边界

| 信息类型 | 获取方式 | 原因 |
|---------|---------|------|
| POI 介绍、攻略文本 | RAG | 变化慢，适合离线索引 |
| 城市文化、注意事项 | RAG | 变化慢，适合文档检索 |
| 实时天气 | Web Search | 实时数据 |
| 实时价格 | Web Search | 可能有波动 |
| POI 坐标 | 本地数据库 | 需要精确 |

### 8.2 MVP 简化实现

由于 1-2 人团队资源有限，MVP 阶段 RAG 做简化：

```python
class SimpleRAG:
    """简化版 RAG：基于文本匹配 + 本地数据"""
    
    def __init__(self):
        self.poi_database = {}      # 本地 POI 数据（种子数据）
        self.guide_database = {}    # 城市攻略文本
    
    async def search(self, query, city=None, top_k=5):
        # Phase 1: 本地数据匹配
        local_results = self._search_local(query, city)
        
        # Phase 2: 网络搜索补充（如果本地结果不足）
        if len(local_results) < top_k:
            web_results = await web_search(f"{city} {query}")
            local_results.extend(web_results)
        
        return local_results[:top_k]
    
    def _search_local(self, query, city):
        # 简单关键词匹配（后续可升级为向量检索）
        results = []
        for poi in self.poi_database.get(city, []):
            score = self._keyword_match(query, poi)
            if score > 0.3:
                results.append((poi, score))
        return sorted(results, key=lambda x: x[1], reverse=True)
```

### 8.3 Phase 2 升级路径（Hybrid Retrieval）

MVP 阶段使用简化版 RAG（8.2 节）。Phase 2 升级为完整 Hybrid Retrieval Pipeline：

```python
class HybridRetriever:
    """混合检索：Vector Search + BM25 + Cross-Encoder Rerank"""

    async def search(self, query: str, filters: dict | None = None, top_k: int = 10):
        # 1. Vector Search（向量检索）
        vector_results = await self.vector_store.search(
            embedding=self.embed_model.encode(query),
            filters=filters,
            top_k=top_k * 3
        )

        # 2. BM25 Search（关键词检索，jieba 中文分词）
        bm25_results = await self.bm25_index.search(
            query=query,
            filters=filters,
            top_k=top_k * 3
        )

        # 3. RRF 融合（Reciprocal Rank Fusion）
        candidates = self._rrf_merge(
            vector_results, bm25_results,
            vector_weight=0.6, bm25_weight=0.4
        )

        # 4. Cross-Encoder Rerank（重排序）
        reranked = await self.reranker.rerank(
            query=query,
            documents=candidates,
            top_k=top_k
        )

        return reranked
```

**3 个知识库**：

| 知识库 | 类型 | Chunk 策略 | 更新频率 |
|--------|------|-----------|---------|
| POI 知识库 | `type: "poi"` | 一个 POI = 一个完整 Chunk（不拆分） | 每周 |
| 城市攻略库 | `type: "guide"` | 按景点/主题拆分，每段保留城市上下文 | 每月 |
| 用户评论库 | `type: "review_summary"` | 按情感聚类（好评/差评/建议） | 每周 |

**技术选型**：
- 向量数据库：Qdrant
- Embedding 模型：bge-m3（中文/英文/日文，无需分词）
- Reranker：bge-reranker-v2-m3
- BM25 分词：jieba

**数据更新策略**：
```python
UPDATE_SCHEDULE = {
    "poi_basic": "weekly",       # 每周刷新 POI 基础信息
    "guide": "monthly",          # 每月更新攻略
    "review_summary": "weekly",  # 每周更新评论摘要
}
```

---

## 九、评估体系设计

### 9.1 评估指标（8 个维度）

V4 要求对生成的行程进行 8 维度量化评估。

| 维度 | 指标 | 计算方式 | 权重 | 阈值 |
|------|------|---------|------|------|
| 偏好匹配度 | preference_coverage | 用户偏好中被行程覆盖的比例 | 0.20 | >= 0.8 |
| 路线合理性 | route_efficiency | 1 - (实际交通时间 / 总可用时间) | 0.15 | >= 0.5 |
| 时间可行性 | time_feasibility | 所有活动是否在开放时间内 | 0.15 | = 1.0 |
| 预算合理性 | budget_compliance | max(0, 1 - 超支比例) | 0.15 | >= 0.9 |
| 舒适度 | comfort_score | 基于步行量/景点数/时长的综合分 | 0.10 | >= 0.7 |
| 真实性 | factuality | 可通过工具验证的 POI 比例（抽样 30%）| 0.15 | >= 0.95 |
| 可解释性 | explainability | 有推荐理由的活动占比 | 0.05 | >= 0.9 |
| 多样性 | diversity | 活动类别的 Shannon 熵 | 0.05 | >= 0.6 |

### 9.2 ItineraryEvaluator

```python
class ItineraryEvaluator:
    """行程质量评估器：8 维度加权评分。"""

    def evaluate(self, itinerary: list[DayPlan], profile: UserProfile) -> EvalResult:
        scores = {}

        # 1. 偏好匹配度
        user_prefs = set(profile.interests + profile.food_preferences)
        covered = set()
        for day in itinerary:
            for activity in day.activities:
                covered.update(set(activity.get("tags", [])) & user_prefs)
        scores["preference_coverage"] = len(covered) / len(user_prefs) if user_prefs else 1.0

        # 2. 路线合理性
        total_transit = sum(day.get("total_transit_time_min", 0) for day in itinerary)
        total_available = len(itinerary) * 8 * 60  # 每天 8 小时
        scores["route_efficiency"] = max(0, 1 - total_transit / total_available) if total_available > 0 else 0

        # 3. 时间可行性
        time_violations = 0
        total_activities = 0
        for day in itinerary:
            for activity in day.activities:
                total_activities += 1
                # 检查开放时间
                if activity.get("open_time") and activity.get("close_time"):
                    if not self._is_within_opening_hours(activity):
                        time_violations += 1
        scores["time_feasibility"] = 1 - (time_violations / total_activities) if total_activities > 0 else 1.0

        # 4. 预算合理性
        total_cost = sum(day.get("total_cost", 0) for day in itinerary)
        max_budget = profile.budget_range or float('inf')
        if max_budget > 0:
            over_ratio = max(0, (total_cost - max_budget) / max_budget)
            scores["budget_compliance"] = max(0, 1 - over_ratio)
        else:
            scores["budget_compliance"] = 1.0

        # 5. 舒适度
        comfort_scores = []
        for day in itinerary:
            day_comfort = 1.0
            if day.get("total_walking_steps", 0) > 15000:
                day_comfort -= 0.3
            if len(day.activities) > 6:
                day_comfort -= 0.3
            if day.get("total_transit_time_min", 0) > 120:
                day_comfort -= 0.2
            comfort_scores.append(max(0, day_comfort))
        scores["comfort_score"] = sum(comfort_scores) / len(comfort_scores) if comfort_scores else 1.0

        # 6. 真实性（抽样 30% 验证）
        verified = 0
        total = 0
        for day in itinerary:
            for activity in day.activities:
                total += 1
                if random.random() < 0.3 or total <= 1:
                    if self._verify_poi_exists(activity["poi_name"]):
                        verified += 1
                else:
                    verified += 1  # 未抽样默认通过
        scores["factuality"] = verified / total if total > 0 else 1.0

        # 7. 可解释性
        explained = sum(
            1 for day in itinerary
            for a in day.activities
            if a.get("recommendation_reason")
        )
        scores["explainability"] = explained / total if total > 0 else 1.0

        # 8. 多样性（Shannon 熵）
        category_counts = {}
        for day in itinerary:
            for a in day.activities:
                cat = a.get("category", "other")
                category_counts[cat] = category_counts.get(cat, 0) + 1
        if category_counts:
            import math
            total_c = sum(category_counts.values())
            entropy = -sum((c / total_c) * math.log2(c / total_c) for c in category_counts.values())
            max_entropy = math.log2(len(category_counts))
            scores["diversity"] = entropy / max_entropy if max_entropy > 0 else 0
        else:
            scores["diversity"] = 0

        # 加权总分
        weights = {
            "preference_coverage": 0.20, "route_efficiency": 0.15,
            "time_feasibility": 0.15, "budget_compliance": 0.15,
            "comfort_score": 0.10, "factuality": 0.15,
            "explainability": 0.05, "diversity": 0.05,
        }
        total_score = sum(scores[k] * weights[k] for k in weights)

        # 关键失败项
        critical_failures = [
            k for k in ["time_feasibility", "factuality", "budget_compliance"]
            if scores[k] < 0.8
        ]

        return EvalResult(
            scores=scores,
            total_score=total_score,
            passed=total_score >= 0.75 and len(critical_failures) == 0,
            critical_failures=critical_failures,
        )
```

### 9.3 ConfidenceCalculator

```python
class ConfidenceCalculator:
    """
    信心值计算：基于客观指标，不由 LLM 自我评估。
    """

    def calculate(self, eval_result: EvalResult, state: ItineraryState) -> float:
        # 基础评估分
        base_score = eval_result.total_score

        # 数据可靠性系数（基于实时数据占比）
        live_data_ratio = self._calc_live_data_ratio(state)
        reliability_factor = 0.7 + 0.3 * live_data_ratio

        # 约束满足度
        hard_constraints_met = len(eval_result.critical_failures) == 0
        constraint_factor = 1.0 if hard_constraints_met else 0.5

        confidence = base_score * reliability_factor * constraint_factor
        return min(1.0, max(0.0, confidence))
```

**公式**：`confidence = base_score * reliability_factor * constraint_factor`

| 因子 | 范围 | 说明 |
|------|------|------|
| `base_score` | 0.0–1.0 | ItineraryEvaluator 加权总分 |
| `reliability_factor` | 0.7–1.0 | `0.7 + 0.3 * live_data_ratio` |
| `constraint_factor` | 0.5 / 1.0 | 有硬约束失败时 0.5，否则 1.0 |

**调用时机**：行程生成后、方案展示前，将 confidence 附加到 proposal 中展示给用户。

---

## 十、前端设计

### 9.1 三栏布局

```
┌──────────────────────────────────────────────────────────────┐
│ 顶部导航                                                     │
├──────────┬──────────────────────────────┬────────────────────┤
│          │                              │                    │
│  左侧栏   │         中间栏                │      右侧栏        │
│          │                              │                    │
│ [+ 新建  │  对话区 / 行程时间线           │  ┌──────────────┐ │
│  对话]   │                              │  │   预算面板    │ │
│          │  🤖 你好！想去哪里旅行？       │  │  总预算: xxx │ │
│ ───────  │                              │  │  已分配: xxx │ │
│          │  👤 我想去成都玩4天           │  │  剩余: xxx   │ │
│ 对话列表  │                              │  │  ████████░░  │ │
│ 💬 成都  │  🤖 行程已生成...             │  │              │ │
│    4日游 │                              │  ├──────────────┤ │
│          │  [Day 1] [Day 2] [Day 3]...  │  │   偏好面板    │ │
│ ───────  │                              │  │  目的地: 成都 │ │
│ [当前    │  [✅ 确认行程]                │  │  天数: 4     │ │
│  行程]   │                              │  │  饮食: 辣 🌶️ │ │
│ ───────  │                              │  │  兴趣: 拍照   │ │
│ 历史行程  │                              │  └──────────────┘ │
│ 📋 成都  │                              │                    │
│    3日游 │                              │                    │
└──────────┴──────────────────────────────┴────────────────────┘
```

### 9.2 状态流转

```
用户输入 → Intent Recognition → 右侧面板实时更新
                │
                ├── generate ──▶ 并行查询 ──▶ 规划 ──▶ 校验 ──▶ 方案生成
                │                                                    │
                │                                                    ▼
                │                                              显示行程 + 面板
                │                                                    │
                │                                              [确认行程]
                │                                                    │
                │                                              保存为当前行程
                │                                                    │
                │                                              左侧"当前行程"可点击
                │
                ├── update_preferences ──▶ 更新面板 ──▶ Graph 条件边 ──▶ 重规划子图
                │                                                    │
                │                                              显示更新后行程
                │
                └── q_a ──▶ 回答（不触发行程/面板更新）
```

### 9.3 WebSocket 消息类型

```typescript
// 客户端 → 服务端
{ type: "user_message", content: "...", session_id: "..." }
{ type: "confirm_itinerary", session_id: "..." }

// 服务端 → 客户端
{ type: "assistant_message", content: "...", metadata: { intent, has_itinerary } }
{ type: "itinerary_update", data: DayPlan[] }
{ type: "budget_update", data: BudgetPanel }
{ type: "preference_update", data: PreferencePanel }
{ type: "error", content: "..." }
```

---

## 十一、实现路线图（AI 可逐步执行）

### Phase 1：基础设施（第 1-2 天）

**目标**：项目骨架 + 数据库 + 基础模块可运行

**Step 1：项目初始化**
- 创建 backend / frontend 目录结构
- 配置 pyproject.toml / package.json
- 配置 docker-compose.yml（PostgreSQL + Redis）
- 创建 .env 模板

**Step 2：配置与数据库**
- 实现 settings.py（Pydantic Settings，从环境变量读取）
- 实现 database.py（SQLAlchemy async engine + session）
- 实现 models.py（5 个 SQLAlchemy ORM 模型）
- 实现 schemas.py（所有 Pydantic 模型）
- 执行 init_db() 创建表

**Step 3：核心基础设施**
- 实现 redis_client.py（Redis 连接 + 热点缓存）
- 实现 llm_client.py（chat + structured_call）
- 实现 event_bus.py（EventBus 类，**仅用于前端 WebSocket 推送**）
- 配置 PostgresSaver Checkpointer（LangGraph 状态持久化）

**Step 4：种子数据收集（可并行）**
- 实现 `seed_data_collector.py`
- 20 个目标城市：`北京、上海、广州、深圳、成都、杭州、西安、重庆、苏州、南京、厦门、青岛、大理、丽江、三亚、长沙、武汉、昆明、桂林、拉萨`
- 每城收集：Top 50 景点（名称/类型/简介/最佳时间）+ Top 30 餐厅（名称/类型/特色）+ 城市基本信息（消费水平/气候/交通）
- 并行收集：景点、餐厅、城市信息同时爬取
- 数据保存到 `backend/seed_data/` 目录（JSON 格式）

**验收标准**：
- `docker-compose up -d` 启动 postgres + redis
- `/health` 接口返回 ok
- 数据库 5 张表已创建
- LLMClient 能正常调用（测试一个简单的 structured_call）
- SeedDataCollector 能为至少 5 个城市生成种子数据文件

---

### Phase 2：Skill 层（第 3-4 天）

**目标**：8 个 Skill 全部可用

**Step 4：搜索与爬取 Skill**
- web_search.py：DuckDuckGo 搜索（无需 API Key）
- web_crawler.py：aiohttp 异步爬取 + BeautifulSoup 解析 + 速率限制

**Step 5：POI 与天气 Skill**
- poi_search.py：搜索 + 爬取 + LLM 提取结构化 POI
- weather_query.py：搜索天气网站 + 解析天气数据

**Step 6：路线与价格 Skill**
- route_calculation.py：haversine 距离 + 交通时间估算
- price_query.py：搜索门票/餐饮价格

**Step 7：记忆 Skill**
- memory_store.py：保存对话/行程/偏好到数据库
- memory_retrieve.py：从数据库检索记忆

**验收标准**：
- `web_search.search("成都 景点")` 返回搜索结果
- `web_crawler.crawl(url)` 返回页面正文
- `poi_search.search_pois("成都", ["美食"])` 返回 POI 列表
- `weather_query.query("成都", "2026-05-01", "2026-05-04")` 返回天气

---

### Phase 3：Agent 层（第 5-8 天）

**目标**：10 个 Agent 全部实现，可独立运行

**Step 8：Intent Recognition Agent**
- 编写 System Prompt（7 种意图 + 关键信息定义 + 输出格式）
- 实现 recognize() 方法
- 实现 preference change detection
- **测试**：输入"我想去成都玩4天预算3000"，验证输出 intent=generate，entities 正确

**Step 9：Information Collection Agent**
- 实现追问生成逻辑
- **测试**：输入"我想去旅行"（无目的地），验证返回追问"想去哪里？"

**Step 10：Realtime Query Agent**
- 实现内部并行的 query_pois / query_weather / query_prices
- **测试**：输入城市+日期，验证并行返回 POI + 天气 + 价格

**Step 11：Preference & Budget Agent**
- 实现 update_preferences()（覆盖更新 + 设置 state["needs_replan"] 标记）
- 实现 calculate_budget()（分项估算 + 城市系数）
- **测试**：更新偏好后验证面板数据正确

**Step 12：Itinerary Planner Agent**
- 实现 6 层算法（打分 → 聚类 → 时间约束 → 日分配 → 2-opt → 时间表 + 用餐）
- **测试**：输入 20 个 POI + 3 天，验证生成合理的 DayPlan[]

**Step 13：Q&A Agent**
- 实现 answer()（RAG 检索 + 网络搜索 + LLM 生成）
- **测试**：问"成都有什么必吃的美食？"，验证返回合理回答

**Step 14：Memory Management Agent**
- 实现 save_conversation_turn()
- 实现 save_itinerary()
- 实现 get_user_memory()
- 实现 extract_preference_patterns()
- **测试**：保存对话后从数据库能读出

**Step 15：Proposal Generation Agent**
- 编写 Prompt（行程概览 + 预算 + 天气 → 自然语言）
- **测试**：输入行程数据，验证生成通顺的方案文本

**Step 16：Validation Agent**
- 实现 5 个并行校验（预算/时间/POI真实性/开放时间/偏好覆盖）
- **测试**：生成有问题的行程，验证检测出 violations

**Step 17：Map & Route Agent**
- 实现 get_coordinates() + get_distance_matrix()
- 实现 two_opt() 路线优化
- **测试**：输入 5 个坐标点，验证优化后路线更短

**验收标准**：
- 每个 Agent 有独立的单元测试
- Intent Recognition 准确率 > 80%（人工测试 10 条）
- Planner 生成的行程每天 3-5 个景点

---

### Phase 4：编排与 API（第 9-10 天）

**Step 18：StateGraph 编译**
- 定义 `ItineraryState` TypedDict
- 实现各 Node 函数（调用对应 Agent，复用 Phase 3 的 Agent 类）
- 构建主图：`intent_node` → `route_intent` 条件边 → 各子图/节点
- 构建 generate_subgraph：fan-out 并行查询 → planner → fan-out 并行校验 → proposal → interrupt
- 配置 PostgresSaver Checkpointer
- 编译：`graph = builder.compile(checkpointer=checkpointer)`

**Step 19：FastAPI + WebSocket**
- main.py： lifespan 管理 + CORS
- REST API：/health, /api/sessions, /api/chat
- WebSocket：/ws/chat（消息路由 + 状态同步）
- 集成 LangGraph interrupt：收到用户回复时 `Command(resume=...)` 恢复 Graph

**验收标准**：
- REST API 返回完整 ChatResponse（含 itinerary + budget + preference）
- WebSocket 实时推送 budget_update / preference_update
- 完整用户流程：输入 → 行程 → 确认，全部通过

---

### Phase 5：前端（第 11-13 天）

**目标**：三栏布局 + 对话 + 行程展示 + 面板更新

**Step 20：前端基础**
- Zustand store（session / messages / itinerary / panels）
- useWebSocket hook（连接/发送/消息处理）

**Step 21：三栏布局**
- LeftSidebar（新建对话 / 会话列表 / 当前行程 / 历史行程）
- MainContent（ChatInterface / ItineraryView 切换）
- RightPanel（BudgetPanel + PreferencePanel）

**Step 22：对话与行程**
- ChatInterface（消息列表 / 输入框 / 加载状态 / 行程卡片）
- ItineraryView（时间线 / 每日活动 / 确认按钮）

**Step 23：面板**
- BudgetPanel（进度条 / 分项明细）
- PreferencePanel（基础信息 / 饮食标签 / 兴趣标签）

**验收标准**：
- 输入消息 → 后端返回 → 右侧面板实时更新
- 点击"当前行程"切换视图
- 点击"确认行程"后左侧入口可点击

---

### Phase 6：集成测试（第 14-15 天）

**目标**：完整用户流程跑通，Bug 修复

**测试场景**：
1. "我想去成都玩 4 天，预算 3000，喜欢吃辣"
   - 验证：提取实体正确，生成 4 天行程，预算面板显示
2. "对了，帮我加几个甜品店"
   - 验证：偏好面板更新，行程重新规划
3. "第三天轻松一点"
   - 验证：只修改 Day 3，其他天不变
4. "确认行程"
   - 验证：保存成功，左侧"当前行程"可查看

**性能优化**：
- Redis 缓存热门城市 POI（TTL 7 天）
- 爬虫频率限制（1 req/s）
- LLM 调用超时处理（10s）

---

## 十二、关键技术决策

### 11.1 为什么用 LangGraph？

- **声明式并行编排**：`fan-out/fan-in` 替代手写 `asyncio.gather()`，并行逻辑可视化
- **状态自动管理**：State 在节点间传递，Checkpointer 自动持久化，省掉 StateManager ~150 行代码
- **人机交互优雅**：`interrupt` 替代手动循环终止条件，用户确认行程场景天然支持
- **对话恢复**：`thread_id` 自动关联历史状态，新对话无需手动加载上下文
- **与 V2 Agent/Skill 完全兼容**：Graph 只负责编排，Agent 内部逻辑零改动

### 11.2 为什么 DuckDuckGo 而不是 Google / Bing？

- 无需 API Key，零成本
- 足够覆盖国内旅行信息
- 后续可无缝切换到 SerpAPI

### 11.3 为什么 Intent Recognition 用 LLM 而不是规则？

- 用户表达多样："想去成都" / "成都怎么样" / "安排个成都行程"
- 规则无法覆盖所有变体
- LLM 能结合上下文推断（如"还是轻松点" → 更新 pace）

### 11.4 为什么 2-opt 而不是 OR-Tools？

- MVP 阶段景点数通常 < 8，2-opt 足够
- OR-Tools 引入依赖复杂
- Phase 4 可升级

### 11.5 为什么 RAG MVP 简化？

- 1-2 人团队，优先核心流程
- 网络搜索 + 本地数据可覆盖 80% 需求
- Phase 2 引入向量检索

### 11.6 为什么需要 8 维度评估体系？

- V4 要求量化行程质量，不能仅依赖 LLM 自我评估（易产生幻觉）
- 8 维度覆盖用户最关心的方面：偏好匹配、路线合理、时间可行、预算合规、舒适、真实、可解释、多样
- `ConfidenceCalculator` 将评估分转化为用户可理解的信心值（0-1）
- 关键失败项（时间/真实性/预算不达标）触发自动重规划

### 11.7 为什么循环终止需要 LLM 判断 + 代码校验？

- 纯代码判断无法处理复杂对话语境（如用户说"再看看"是否算完成？）
- 纯 LLM 判断可能过早终止（LLM 认为完成了但还有 pending 调用未执行）
- 结合方案：LLM 提供语义判断（是否对话完成），代码提供硬约束（无 pending 调用、输出已准备、非阻塞状态）
- 4 个条件必须同时满足，确保安全终止

---

## 十三、风险与缓解

| 风险 | 缓解策略 |
|------|---------|
| LLM 意图识别错误 | 多轮对话纠正；confidence < 0.7 时主动确认 |
| LLM 幻觉（推荐不存在景点）| Validation Agent 抽样校验；标注数据来源 |
| 爬虫被反爬 | 控制频率 1 req/s；轮换 UA；本地缓存兜底 |
| 网络搜索结果质量差 | 多源交叉验证；LLM 标注不确定性 |
| 种子数据覆盖不足 | MVP 支持 20 个热门城市；非覆盖城市明确提示 |
| 1-2 人进度延迟 | 严格按 Phase 执行；每周验收；边缘功能延后 |

---

**文档结束**

> 本方案与 v4 设计文档严格对齐。每个模块的职责、输入输出、流程都经过逐一核对。按 Phase 1-6 逐步执行
