# 大厂级智能旅行 Agent 系统设计文档

> **文档版本**：v2.0  
> **最后更新**：2026-04-26  
> **适用范围**：面向 Codex / 自动化开发 Agent 的完整技术规格说明  
> **定位**：一个面向自由行用户的多 Agent 协作式旅行决策系统，采用 Agent Loop + MCP + Multi-Agent 架构，具备多约束求解、实时工具调用、动态重规划与闭环评估能力。

---

## 目录

1. [产品定位与用户画像](#一产品定位与用户画像)
2. [核心产品价值](#二核心产品价值)
3. [核心场景拆解](#三核心场景拆解)
4. [系统总体架构](#四系统总体架构)
5. [Agent 编排与通信协议](#五agent-编排与通信协议)
6. [模块详细设计](#六模块详细设计)
7. [数据模型与 Schema 定义](#七数据模型与-schema-定义)
8. [约束求解与路径优化算法](#八约束求解与路径优化算法)
9. [RAG 知识库系统](#九rag-知识库系统)
10. [评估体系设计](#十评估体系设计)
11. [Human-in-the-Loop 协作机制](#十一human-in-the-loop-协作机制)
12. [前端交互设计规范](#十二前端交互设计规范)
13. [完整用户流程设计](#十三完整用户流程设计)
14. [技术栈选型与项目结构](#十四技术栈选型与项目结构)
15. [MVP 分阶段交付计划](#十五mvp-分阶段交付计划)
16. [竞品分析与差异化](#十六竞品分析与差异化)
17. [风险与缓解策略](#十七风险与缓解策略)
18. [附录](#十八附录)

---

## 一、产品定位与用户画像

### 1.1 一句话定位

**旅行 Agent 是一个面向自由行用户的多 Agent 协作式旅行决策系统。它不是"AI 帮你写攻略"，而是一个能够理解用户多维约束、调用真实外部工具获取实时数据、通过运筹优化算法生成可执行行程、并在多轮对话与旅途中持续动态调整的智能旅行规划引擎。**

核心差异点：

- **不是文本生成，而是约束求解**：行程规划本质是带时间窗口的多约束优化问题（VRPTW 变体），不能只靠 LLM 生成自然语言。
- **不是单次输出，而是持续对话**：支持多轮修改、局部重规划、旅途中实时调整。
- **不是信息搬运，而是决策辅助**：整合碎片化信息源，帮用户在时间/预算/偏好/舒适度之间做出最优权衡。

### 1.2 目标用户画像

#### 用户类型 A：自由行新手

**特征**：第一次去某个目的地，不知道去哪、怎么安排、花多少钱。  
**核心痛点**：信息过载，不知道如何从小红书/携程/Google Maps/天气/门票等碎片信息中整合出一个可执行方案。  
**Agent 价值**：一句话输入需求 → 输出完整可执行行程，包含每日安排、路线、预算、替代方案。

#### 用户类型 B：懒人型用户

**特征**：有旅行经验但不想花时间做攻略。  
**核心痛点**：攻略制作过程太繁琐，需要跨多个平台查信息、比较、排列组合。  
**Agent 价值**：自动完成需求理解 → 偏好抽取 → 行程生成 → 预算估算 → 路线安排 → 方案解释的全流程。  
**典型输入**：`"我想去日本玩 7 天，预算 1 万，别太累，喜欢动漫和美食。"`

#### 用户类型 C：高要求用户

**特征**：需求多、约束复杂，需要精细规划。  
**核心痛点**：人脑难以同时处理多个约束条件的权衡。  
**Agent 价值**：多约束并行求解 + 多轮精细调整。  
**典型输入**：`"我想去东京 6 天，第一天晚上到，第三天想去镰仓，不能太累，酒店最好住新宿附近，预算 9000，想吃几家本地人推荐的店。"`

#### 用户类型 D：特定群体旅行

**特征**：亲子/情侣/老人同行，对舒适度要求高。  
**核心痛点**：通用攻略不考虑同行人的体力/兴趣差异。  
**Agent 价值**：根据同行人画像自动调整行程强度、路线复杂度、餐厅选择、休息时间。  
**关注因素**：是否太累、是否适合老人小孩、路线是否绕、是否需要频繁换酒店、餐厅是否家庭友好、是否有充足休息时间。

#### 用户类型 E：出境游用户

**特征**：出境游涉及更多外围信息需求。  
**额外需求**：签证提醒、汇率换算、通信卡/交通卡建议、语言翻译、入境注意事项、当地规则与法律、安全提醒。  
**Agent 价值**：在行程规划之上叠加出行辅助信息层。

---

## 二、核心产品价值

### 2.1 解决信息过载问题

用户做旅游攻略需要跨 10+ 个平台查信息（小红书、携程、飞猪、大众点评、Google Maps、天气应用、门票平台、酒店平台、交通查询、签证信息）。Agent 通过 MCP 工具标准化接口统一接入这些数据源，用 LLM 整合理解，输出结构化方案。

### 2.2 解决多约束决策困难

旅行规划是一个典型的多目标优化问题。用户需要在以下因素之间权衡：

- 想去的地方数量 vs. 可用时间
- 路线顺畅度 vs. 景点优先级
- 体验丰富度 vs. 预算限制
- 活动密度 vs. 舒适度
- 户外活动 vs. 天气风险
- 热门景点 vs. 个人偏好

Agent 的核心能力是将这个多目标决策问题建模为约束满足问题（CSP），通过算法求解而非 LLM 猜测。

### 2.3 支持行程动态变化

真实旅行中频繁出现变化：天气变化、景点临时关闭、用户睡过头、临时想换地方、预算超支、交通延误、同行人体力不支。Agent 必须支持局部重规划（只改变受影响部分，保留已确认部分），而不是每次全量重新生成。

### 2.4 打通从规划到执行的闭环

行程生成只是旅行周期的一小部分。大厂级产品需要考虑完整链路：

```
灵感探索 → 需求明确 → 行程规划 → 方案对比 → 确认行程
→ 预订执行（酒店/门票/交通）→ 行前提醒 → 旅途伴随
→ 实时调整 → 旅后回顾 → 偏好积累 → 下次旅行推荐
```

MVP 聚焦"需求明确 → 行程规划 → 方案对比 → 确认行程"核心链路，但架构设计需要为全链路预留扩展点。

---

## 三、核心场景拆解

### 场景 1：从零生成旅行计划

**用户输入**：`"我想去大阪玩 5 天，预算 8000，喜欢美食、动漫和购物，不想太累。"`

**Agent 处理流程**：

```
1. 约束抽取（Constraint Extraction）
   - destination: 大阪
   - duration: 5 天
   - budget: 8000 CNY
   - preferences: [美食, 动漫, 购物]
   - pace: relaxed（每日景点 ≤ 4，步行 ≤ 12000 步）

2. 信息缺口检测（Gap Detection）
   - 缺少：出发城市、人数、出发日期、是否含机票、酒店偏好
   - 策略：对非关键信息使用默认值，仅追问 1-2 个关键问题
   - 输出追问："大概几号出发？这次几个人去？我先按不含机票帮你做个初版。"

3. 数据获取（Tool Calling via MCP）
   - 调用 POI MCP Server：检索大阪美食/动漫/购物相关 POI
   - 调用 Weather MCP Server：查询目标日期天气预报
   - 调用 Map MCP Server：获取 POI 地理坐标与区域信息
   - 调用 Attraction MCP Server：获取开放时间、门票价格、游玩时长

4. 约束求解（Constraint Solving）
   - 区域聚类：将 POI 按地理位置聚类为日维度组
   - 时间窗口适配：根据开放时间排列每日顺序
   - 路径优化：最小化每日移动距离
   - 预算适配：确保总费用 ≤ 8000
   - 强度控制：确保每日活动强度 ≤ relaxed 阈值

5. 行程生成与解释（Response Generation）
   - 输出：每日行程 + 推荐理由 + 交通路线 + 预算明细 + 替代方案 + 风险提醒
```

**期望输出结构**：

```
总体方案概览
├── Day 1：抵达 + 轻度探索
│   ├── 下午：道顿堀（美食 + 夜景，强度：低）
│   ├── 推荐理由：适合抵达日，交通便利，不赶时间
│   ├── 交通：从关西机场乘南海电铁直达难波，约 50 分钟
│   └── 预算：交通 ¥920 + 餐饮 ¥3000
├── Day 2：市区文化 + 美食路线
│   ├── 上午：大阪城公园（文化，强度：中）
│   ├── 中午：黑门市场（美食，强度：低）
│   ├── 下午：日本桥动漫街（动漫，强度：低）
│   ├── 晚上：道顿堀夜景（自由活动）
│   ├── 推荐理由：四个点均在中央区，步行可达，路线不绕
│   └── 预算：门票 ¥600 + 餐饮 ¥4000 + 购物 ¥2000
├── ...
├── 预算总计：¥7,650（余量 ¥350）
├── 可替代方案：A省钱版 / B深度体验版
└── 风险提醒：Day 3 有降雨概率 60%，已准备室内替代方案
```

### 场景 2：多轮修改行程

**用户输入**：`"第三天太累了，轻松一点。"`

**Agent 处理流程**：

```
1. 上下文理解
   - 识别修改意图：reduce_intensity
   - 定位目标：Day 3
   - 获取当前 Day 3 状态：5 个景点，预估步行 18000 步

2. 局部重规划
   - 保留：已确认的高优先级景点
   - 删除/替换：低优先级或高体力消耗景点
   - 重新计算：时间安排、交通路线、预算

3. 变更影响分析
   - 输出：修改了什么、保留了什么、为什么这样改、是否影响其他天
   - 示例："删除了天保山摩天轮（步行距离远），替换为梅田 Loft 购物（与你购物偏好匹配，步行量减少约 5000 步）。Day 4 不受影响。"
```

**关键原则**：局部修改，不全量重写。只改 Day 3，保留其他天，明确标注变更点。

### 场景 3：预算优化

**用户输入**：`"这个方案太贵了，帮我压到 6000 以内。"`

**Agent 处理流程**：

```
1. 预算结构分析
   当前总计：¥8,200
   ├── 酒店：¥3,500（5晚 × ¥700/晚）
   ├── 餐饮：¥2,500
   ├── 门票：¥1,200
   ├── 交通：¥800
   └── 购物预算：¥200

2. 优化策略生成（按影响/牺牲比排序）
   策略 A：换酒店区域（新宿 → 日本桥），节省 ¥1,500，体验损失小
   策略 B：减少高价餐厅（2家 → 1家），节省 ¥800
   策略 C：替换收费景点为免费景点，节省 ¥600
   策略 D：优化交通（部分出租 → 地铁），节省 ¥300

3. 方案输出
   推荐组合：A + B，总预算降至 ¥5,900
   解释取舍："酒店从新宿换到日本桥，到主要景点的交通时间增加约 10 分钟，但价格便宜 ¥300/晚。高价餐厅从 2 家减为 1 家，保留了评分最高的那家。"
```

### 场景 4：天气影响调整

**用户输入**：`"如果第二天下雨怎么办？"`

**Agent 处理流程**：

```
1. 调用 Weather MCP Server 获取天气数据
2. 分析 Day 2 户外项目：大阪城公园（户外）、黑门市场（室内）
3. 替换方案：大阪城公园 → 大阪海游馆（室内，与美食偏好弱相关但亲子友好）
4. 路线重算：新路线是否需要调整交通方式
5. 输出：雨天方案 + 与原方案的对比 + 预算变化
```

### 场景 5：路线优化

**用户输入**：`"这个路线是不是太绕了？"`

**Agent 处理流程**：

```
1. 获取每个 POI 的经纬度坐标（Map MCP Server）
2. 计算当前路线总移动距离与时间
3. 运行路径优化算法（贪心 / 2-opt / OR-Tools）
4. 按区域聚类重新分配 POI 到各天
5. 输出：优化前 vs 优化后的路线对比（总距离、总交通时间、跨区域次数）
```

### 场景 6：个性化推荐

**用户输入**：`"我不喜欢网红景点，想要本地人常去的地方。"`

**Agent 处理流程**：

```
1. 识别偏好调整：tourist_popular → local_favorite
2. 调整 RAG 检索策略：
   - 降权：游客评论占比高的 POI
   - 升权：本地评论多、小红书标签含"小众"/"本地推荐"的 POI
3. 重新召回候选 POI 并替换
4. 输出：推荐理由需引用具体数据来源（"本地用户评分 4.8，游客占比 < 20%"）
```

### 场景 7：同行人约束

**用户输入**：`"我和爸妈一起去，他们走不了太多路。"`

**Agent 处理流程**：

```
1. 更新用户画像：companions = [senior]
2. 自动调整约束参数：
   - max_daily_steps: 15000 → 8000
   - max_daily_pois: 5 → 3
   - transfer_complexity: any → simple（避免多次换乘）
   - transport_preference: walk → taxi/subway_direct
   - rest_time: 0 → 60min（午休）
   - restaurant_type: any → comfortable_seating
3. 在调整后的约束下重新规划
```

### 场景 8：旅行中实时调整

**用户输入**：`"我现在在新宿，下午还有 4 小时，推荐附近能去的地方。"`

**Agent 处理流程**：

```
1. 获取当前位置：新宿（lat: 35.6896, lng: 139.7006）
2. 识别可用时间窗口：4 小时（约 14:00-18:00）
3. 调用 Map MCP Server：nearby_search(radius=3km)
4. 过滤：
   - 当前时间可入场的景点
   - 符合用户偏好的 POI
   - 交通时间 ≤ 20 分钟
5. 约束求解：在 4 小时内可串联的最优组合
6. 输出：半日行程 + 步行路线 + 预计完成时间
```

---

## 四、系统总体架构

### 4.1 架构设计理念

本系统采用 **Agent Loop + Multi-Agent + MCP** 架构，核心设计原则如下：

| 原则 | 说明 |
|------|------|
| Agent Loop 替代 Pipeline | 采用感知-推理-行动-观察循环，而非线性流水线，支持动态决策和错误恢复 |
| Multi-Agent 协作 | 将规划、推荐、预算、交通等能力拆分为独立 Sub-Agent，由 Orchestrator 协调 |
| MCP 标准化工具接口 | 所有外部工具（地图、天气、POI、酒店）均通过 MCP Server 暴露标准化接口 |
| State 驱动的多轮对话 | 以结构化 State 对象维护用户画像、当前行程、约束条件和对话历史 |
| 算法 + LLM 互补 | 结构化规划用算法求解，自然语言理解和生成用 LLM |
| 可观测性优先 | 每个决策步骤可追溯（Glass Box 而非 Black Box） |

### 4.2 总体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Client Layer                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐                 │
│  │   Web App    │  │  Mobile App  │  │  Chat Interface │                │
│  └──────┬──────┘  └──────┬───────┘  └───────┬────────┘                 │
│         └────────────────┼──────────────────┘                          │
│                          ▼                                              │
│              ┌──────────────────────┐                                   │
│              │    API Gateway        │                                   │
│              │  (WebSocket + REST)   │                                   │
│              └──────────┬───────────┘                                   │
└─────────────────────────┼───────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Orchestration Layer                                │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Orchestrator Agent                              │  │
│  │  ┌──────────────────────────────────────────────────────────────┐ │  │
│  │  │                    Agent Loop                                 │ │  │
│  │  │                                                               │ │  │
│  │  │   ┌──────────┐    ┌──────────┐    ┌──────────┐              │ │  │
│  │  │   │ Perceive │───▶│  Reason  │───▶│   Act    │              │ │  │
│  │  │   │          │    │(ReAct /  │    │(Delegate │              │ │  │
│  │  │   │ - Parse  │    │ Plan &   │    │ to Sub-  │              │ │  │
│  │  │   │   input  │    │ Execute) │    │ Agents / │              │ │  │
│  │  │   │ - Read   │    │          │    │ Call     │              │ │  │
│  │  │   │   state  │    │          │    │ Tools)   │              │ │  │
│  │  │   └──────────┘    └──────────┘    └────┬─────┘              │ │  │
│  │  │        ▲                               │                     │ │  │
│  │  │        │         ┌──────────┐          │                     │ │  │
│  │  │        └─────────│ Observe  │◀─────────┘                     │ │  │
│  │  │                  │ - Eval   │                                │ │  │
│  │  │                  │   result │                                │ │  │
│  │  │                  │ - Update │                                │ │  │
│  │  │                  │   state  │                                │ │  │
│  │  │                  └──────────┘                                │ │  │
│  │  └──────────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                          │                                              │
│            ┌─────────────┼─────────────┐                               │
│            ▼             ▼             ▼                                │
│  ┌──────────────┐ ┌────────────┐ ┌──────────────┐                     │
│  │  State Store  │ │  Guardrail │ │ Trace Logger │                     │
│  │  (Redis /     │ │  (Input /  │ │ (Decision    │                     │
│  │   Session)    │ │   Output   │ │  Audit Trail)│                     │
│  │              │ │   Safety)  │ │              │                     │
│  └──────────────┘ └────────────┘ └──────────────┘                     │
└─────────────────────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼────────────────────┐
        ▼                 ▼                    ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│ Sub-Agent:   │  │ Sub-Agent:   │  │ Sub-Agent:       │
│ Itinerary    │  │ Budget       │  │ Recommendation   │
│ Planner      │  │ Optimizer    │  │ Engine           │
│              │  │              │  │                  │
│ - Constraint │  │ - Cost       │  │ - POI Ranking    │
│   Solver     │  │   Analysis   │  │ - Personalized   │
│ - Route      │  │ - Trade-off  │  │   Filtering      │
│   Optimizer  │  │   Suggestion │  │ - RAG Retrieval  │
│ - Schedule   │  │              │  │                  │
│   Fitter     │  │              │  │                  │
└──────┬───────┘  └──────┬───────┘  └────────┬─────────┘
       │                 │                    │
       └─────────────────┼────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       Tool Layer (MCP Servers)                          │
│                                                                         │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐          │
│  │ Map MCP    │ │Weather MCP │ │  POI MCP   │ │ Hotel MCP  │          │
│  │ Server     │ │ Server     │ │  Server    │ │ Server     │          │
│  │            │ │            │ │            │ │            │          │
│  │- geocode   │ │- forecast  │ │- search    │ │- search    │          │
│  │- distance  │ │- is_outdoor│ │- detail    │ │- price     │          │
│  │- route     │ │  _friendly │ │- hours     │ │- area      │          │
│  │- nearby    │ │            │ │- reviews   │ │- amenities │          │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘          │
│                                                                         │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐          │
│  │Transport   │ │ Booking    │ │ Calendar   │ │ Currency   │          │
│  │MCP Server  │ │ MCP Server │ │ MCP Server │ │ MCP Server │          │
│  │            │ │            │ │            │ │            │          │
│  │- timetable │ │- reserve   │ │- export    │ │- convert   │          │
│  │- fare      │ │- cancel    │ │- ical      │ │- rates     │          │
│  │- realtime  │ │- confirm   │ │            │ │            │          │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Knowledge Layer (RAG)                               │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   Hybrid Retrieval Pipeline                      │   │
│  │                                                                   │   │
│  │  Query ──▶ ┌──────────┐ ──▶ ┌──────────┐ ──▶ ┌──────────┐      │   │
│  │            │ Embedding │     │  BM25    │     │ Reranker │      │   │
│  │            │ Search    │     │  Search  │     │ (Cross-  │      │   │
│  │            │ (语义)    │     │ (精确)   │     │ Encoder) │      │   │
│  │            └──────────┘     └──────────┘     └──────────┘      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ POI 知识库    │  │ 城市攻略库    │  │ 用户评论库    │                  │
│  │ (结构化)     │  │ (文档)       │  │ (摘要)       │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Agent Loop 核心流程

传统 Pipeline 架构的问题是：每一步的输出固定传递给下一步，无法根据中间结果动态调整策略。Agent Loop 解决了这个问题：

```python
# Agent Loop 伪代码
class OrchestratorAgent:
    async def run(self, user_message: str) -> str:
        # 更新 State
        self.state.add_message(role="user", content=user_message)
        
        while not self.should_stop():
            # 1. Perceive：理解当前状态和用户意图
            context = self.build_context(self.state)
            
            # 2. Reason：LLM 决定下一步行动
            #    - 可能是调用某个 Sub-Agent
            #    - 可能是调用某个 MCP Tool
            #    - 可能是直接回复用户
            #    - 可能是追问用户
            action = await self.llm.decide_next_action(
                context=context,
                available_actions=self.get_available_actions(),
                state=self.state
            )
            
            # 3. Act：执行行动
            result = await self.execute_action(action)
            
            # 4. Observe：观察结果，更新状态
            self.state.update(action=action, result=result)
            
            # 5. Evaluate：检查是否需要继续循环
            #    - 结果是否满足约束？
            #    - 是否需要调用更多工具？
            #    - 是否需要修正计划？
            if self.is_response_ready():
                break
        
        return self.generate_response(self.state)
```

与传统 Pipeline 的关键区别：

| 维度 | 传统 Pipeline | Agent Loop |
|------|--------------|------------|
| 控制流 | 预定义的线性流程 | 动态决策，LLM 决定下一步 |
| 错误恢复 | 需要为每种错误预定义处理逻辑 | 自然回退，重新推理 |
| 扩展性 | 增加场景需要修改路由逻辑 | 增加工具/Sub-Agent 即可，无需改控制流 |
| 多步推理 | 需要显式编排 | 自然涌现 |
| 复杂度 | O(场景数 × 步骤数) | O(工具数) |

---

## 五、Agent 编排与通信协议

### 5.1 MCP（Model Context Protocol）集成

所有外部工具通过 MCP Server 暴露标准化接口。MCP 是 Anthropic 于 2024 年 11 月发布的开放标准，已被 OpenAI、Google、Microsoft、Amazon 等主要 AI 提供商采纳，截至 2026 年 2 月月 SDK 下载量已超过 9700 万次。

#### MCP Server 定义示例

```python
# map_mcp_server.py
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("travel-map")

@server.tool()
async def search_poi(
    city: str,
    keyword: str,
    category: str | None = None,
    radius_km: float = 10.0
) -> list[dict]:
    """
    在指定城市搜索兴趣点。
    
    Args:
        city: 城市名称（如 "大阪"、"东京"）
        keyword: 搜索关键词（如 "动漫"、"拉面"）
        category: POI 类别过滤（可选，如 "restaurant"、"attraction"、"shopping"）
        radius_km: 搜索半径（千米）
    
    Returns:
        POI 列表，包含 name, lat, lng, category, rating, review_count, 
        opening_hours, estimated_duration, ticket_price
    """
    results = await google_places_api.search(
        location=geocode(city),
        query=keyword,
        type=category,
        radius=radius_km * 1000
    )
    return [format_poi(r) for r in results]

@server.tool()
async def get_distance_matrix(
    origins: list[dict],  # [{name, lat, lng}, ...]
    destinations: list[dict],
    mode: str = "transit"  # "transit" | "driving" | "walking"
) -> dict:
    """
    计算多个起终点之间的距离和时间矩阵。
    用于路径优化算法的输入。
    
    Returns:
        {
            "matrix": [[{distance_m, duration_s}, ...], ...],
            "mode": "transit"
        }
    """
    return await google_maps_api.distance_matrix(origins, destinations, mode)

@server.tool()
async def get_route(
    origin: dict,       # {lat, lng}
    destination: dict,  # {lat, lng}
    mode: str = "transit",
    departure_time: str | None = None
) -> dict:
    """
    获取两点之间的详细路线。
    
    Returns:
        {
            "distance_m": 5200,
            "duration_s": 1800,
            "steps": [...],
            "transit_details": {...}  # 仅 transit 模式
        }
    """
    return await google_maps_api.directions(origin, destination, mode, departure_time)

@server.tool()
async def nearby_search(
    location: dict,     # {lat, lng}
    radius_m: int = 3000,
    category: str | None = None,
    open_now: bool = True
) -> list[dict]:
    """
    搜索指定位置附近的 POI，支持按当前开放状态过滤。
    用于旅途中实时推荐场景。
    """
    return await google_places_api.nearby(location, radius_m, category, open_now)
```

```python
# weather_mcp_server.py
from mcp.server import Server

server = Server("travel-weather")

@server.tool()
async def get_forecast(
    city: str,
    dates: list[str]  # ["2026-05-01", "2026-05-02", ...]
) -> list[dict]:
    """
    获取指定城市多天天气预报。
    
    Returns:
        [{
            "date": "2026-05-01",
            "condition": "rain",        # sunny | cloudy | rain | snow | storm
            "temp_high_c": 22,
            "temp_low_c": 15,
            "rain_probability": 0.7,
            "is_outdoor_friendly": false,
            "recommendation": "建议安排室内活动"
        }, ...]
    """
    return await openweather_api.forecast(geocode(city), dates)

@server.tool()
async def check_outdoor_suitability(
    city: str,
    date: str,
    activity_type: str  # "walking" | "hiking" | "cycling" | "sightseeing"
) -> dict:
    """
    判断指定日期是否适合特定户外活动。
    
    Returns:
        {
            "suitable": false,
            "reason": "降雨概率 70%，不建议户外步行观光",
            "alternative_suggestion": "indoor_attraction"
        }
    """
    forecast = await get_forecast(city, [date])
    return evaluate_outdoor_suitability(forecast[0], activity_type)
```

#### MCP 配置

```json
{
  "mcpServers": {
    "travel-map": {
      "command": "python",
      "args": ["mcp_servers/map_server.py"],
      "env": {
        "GOOGLE_MAPS_API_KEY": "${GOOGLE_MAPS_API_KEY}"
      }
    },
    "travel-weather": {
      "command": "python",
      "args": ["mcp_servers/weather_server.py"],
      "env": {
        "OPENWEATHER_API_KEY": "${OPENWEATHER_API_KEY}"
      }
    },
    "travel-poi": {
      "command": "python",
      "args": ["mcp_servers/poi_server.py"]
    },
    "travel-hotel": {
      "command": "python",
      "args": ["mcp_servers/hotel_server.py"]
    },
    "travel-transport": {
      "command": "python",
      "args": ["mcp_servers/transport_server.py"]
    },
    "travel-booking": {
      "command": "python",
      "args": ["mcp_servers/booking_server.py"]
    },
    "travel-currency": {
      "command": "python",
      "args": ["mcp_servers/currency_server.py"]
    }
  }
}
```

### 5.2 Multi-Agent 协作架构

系统采用 Orchestrator-Worker 模式，由一个 Orchestrator Agent 协调多个专职 Sub-Agent：

```
┌─────────────────────────────────────────────────────┐
│                 Orchestrator Agent                    │
│                                                       │
│  职责：                                               │
│  - 理解用户意图                                       │
│  - 分解任务为子任务                                    │
│  - 分配子任务给 Sub-Agent                             │
│  - 整合 Sub-Agent 结果                                │
│  - 生成最终回复                                       │
│  - 管理对话状态                                       │
│                                                       │
│  决策能力（LLM 驱动，非硬编码路由）：                   │
│  - 判断需要调用哪些 Sub-Agent                         │
│  - 判断是否需要追问用户                                │
│  - 判断结果是否满足要求                                │
│  - 判断是否需要重新规划                                │
└──────────┬──────────┬──────────┬────────────────────┘
           │          │          │
     ┌─────▼────┐┌────▼─────┐┌──▼───────────┐
     │Itinerary ││ Budget   ││Recommendation│
     │Planner   ││Optimizer ││ Engine       │
     │Agent     ││ Agent    ││ Agent        │
     │          ││          ││              │
     │输入：    ││输入：    ││输入：        │
     │- POI列表 ││- 行程    ││- 用户偏好   │
     │- 约束    ││- 预算    ││- 目的地     │
     │- 偏好    ││  上限    ││- 已选POI    │
     │          ││          ││              │
     │输出：    ││输出：    ││输出：        │
     │- 日程表  ││- 优化    ││- POI推荐    │
     │- 路线    ││  方案    ││  列表       │
     │- 时间表  ││- 取舍    ││- 排序理由   │
     │          ││  分析    ││              │
     │工具：    ││工具：    ││工具：        │
     │- Map MCP ││- Hotel   ││- POI MCP    │
     │- Transport│ MCP     ││- RAG        │
     │  MCP     ││- Currency││              │
     │          ││  MCP     ││              │
     └──────────┘└──────────┘└──────────────┘
```

#### Sub-Agent 间的协作流程示例

以"从零生成行程"为例：

```
Orchestrator 接收用户请求
    │
    ├── Step 1: 调用 Recommendation Agent
    │   └── "根据用户偏好（美食/动漫/购物）推荐大阪 15-20 个候选 POI"
    │   └── 返回：带评分和标签的 POI 列表
    │
    ├── Step 2: 调用 Itinerary Planner Agent
    │   └── "用这 20 个候选 POI，在 5 天、relaxed 节奏的约束下规划行程"
    │   └── 返回：5 天行程草案 + 每日路线
    │
    ├── Step 3: 调用 Budget Optimizer Agent
    │   └── "评估这份行程的预算，确保 ≤ 8000"
    │   └── 返回：预算明细 + 是否超标 + 优化建议
    │
    ├── Step 4: Orchestrator 检查结果
    │   ├── 如果预算超标 → 要求 Budget Agent 提供优化方案
    │   ├── 如果路线不优 → 要求 Planner Agent 重新优化路径
    │   └── 如果满足所有约束 → 进入下一步
    │
    └── Step 5: Orchestrator 整合结果，生成用户回复
```

### 5.3 A2A（Agent-to-Agent）协议预留

A2A 协议是 Google 于 2025 年 4 月发布的开放标准，用于标准化不同 AI Agent 之间的发现、通信和协作。目前已捐赠给 Linux Foundation 下的 Agentic AI Foundation（AAIF）。

在本系统的高级版本中，A2A 可用于：

- **与外部 Agent 协作**：例如与航空公司的预订 Agent、酒店集团的 Agent 进行跨组织协作
- **与第三方推荐 Agent 协作**：例如与美食推荐 Agent、本地导游 Agent 交换信息

当前 MVP 暂不实现 A2A，但在 Sub-Agent 间通信中采用兼容 A2A Task 模型的消息格式，降低未来迁移成本：

```python
# A2A-compatible task message format
@dataclass
class AgentTask:
    task_id: str
    sender: str              # "orchestrator"
    receiver: str            # "itinerary_planner"
    action: str              # "generate_itinerary"
    payload: dict            # 输入参数
    constraints: dict        # 约束条件
    callback_url: str | None # 异步回调地址（A2A 场景使用）
    
@dataclass
class AgentTaskResult:
    task_id: str
    status: str              # "completed" | "failed" | "partial"
    result: dict             # 输出结果
    confidence: float        # 0.0 - 1.0
    trace: list[str]         # 决策链路日志
```

---

## 六、模块详细设计

### 6.1 Orchestrator Agent

#### 职责
- 解析用户输入，理解意图和约束
- 管理对话状态（State）
- 决定调用哪些 Sub-Agent 和 MCP Tools
- 整合结果，生成用户回复
- 处理异常和降级

#### 核心设计：LLM 驱动的动态决策（非硬编码路由）

传统做法是用 if-else 路由意图到固定流程。本系统让 LLM 直接基于工具描述和当前状态做决策：

```python
ORCHESTRATOR_SYSTEM_PROMPT = """
你是一个旅行规划系统的 Orchestrator。你的职责是理解用户需求，
调用合适的 Sub-Agent 和工具来完成任务。

## 可用的 Sub-Agent

1. **recommendation_agent**: 根据用户偏好推荐 POI
   - 输入：用户偏好、目的地、已选/已排除 POI
   - 输出：带评分的 POI 推荐列表

2. **itinerary_planner_agent**: 生成/修改行程
   - 输入：POI 列表、约束条件（天数/节奏/时间窗口）
   - 输出：按天组织的行程 + 路线

3. **budget_optimizer_agent**: 预算分析与优化
   - 输入：行程方案、预算上限
   - 输出：预算明细、是否超标、优化方案

## 决策原则

- 信息不足时，先用默认值生成草案，只追问最关键的 1-2 个问题
- 修改行程时，只改受影响的部分，不全量重写
- 每次回复都包含推荐理由和可调整的选项
- 如果 Sub-Agent 返回的结果不满足约束，重新调用并说明需要调整的方向
- 保持对话自然流畅，不暴露内部技术细节

## 当前状态
{state_json}
"""
```

**为什么不用预定义意图分类？**

预定义意图列表存在两个根本问题：
1. **覆盖不全**：用户的表达方式无限多样，预定义列表永远无法穷举
2. **多意图冲突**：用户一句话可能包含多个意图（"第三天轻松一点，顺便加个温泉"同时包含 `reduce_intensity` 和 `add_activity`）

LLM 直接做 tool use / function calling 的决策，意图识别是隐式的——LLM 理解了用户想做什么，直接选择对应的工具/Agent 来执行。意图标签仅作为日志和分析用途保留。

### 6.2 State Manager

State 是整个系统的记忆核心，维护跨轮次的完整上下文。

#### State Schema

```typescript
interface TravelState {
  // 会话元数据
  session_id: string;
  created_at: string;
  updated_at: string;
  turn_count: number;
  
  // 用户画像（渐进式收集）
  user_profile: {
    budget: number | null;               // 总预算（当地货币）
    budget_currency: string;             // "CNY" | "JPY" | "USD" ...
    budget_includes_flight: boolean;     // 预算是否含机票
    travel_days: number | null;
    destination: string | null;
    destinations: string[];              // 多城市行程
    departure_city: string | null;
    departure_date: string | null;       // ISO 8601
    return_date: string | null;
    travelers_count: number;             // 默认 1
    companions: CompanionType[];         // ["senior", "child", "couple"]
    preferences: string[];              // ["美食", "动漫", "购物"]
    anti_preferences: string[];         // ["人多", "网红景点"]
    pace: "intensive" | "moderate" | "relaxed";
    mobility: "full" | "limited" | "wheelchair";
    accommodation_preference: string | null;  // "新宿附近"、"便宜"
  };
  
  // 当前行程（结构化）
  current_itinerary: DayPlan[] | null;
  
  // 约束条件（由用户输入 + 同行人画像推导）
  constraints: {
    max_budget: number | null;
    max_daily_pois: number;              // 由 pace 推导
    max_daily_steps: number;             // 由 pace + mobility 推导
    max_daily_transit_time_min: number;  // 单日最大交通时间
    preferred_transport: string[];       // ["subway", "taxi", "walk"]
    avoid_keywords: string[];            // ["太累", "频繁换乘"]
    must_include_pois: string[];         // 用户指定必去的景点
    must_avoid_pois: string[];           // 用户指定不去的景点
    dietary_restrictions: string[];      // 饮食限制
    rest_time_min: number;               // 午休时间（分钟）
  };
  
  // 交互历史
  confirmed_items: string[];   // 用户明确同意的推荐
  rejected_items: string[];    // 用户明确拒绝的推荐
  modification_history: Modification[];  // 修改记录
  
  // 对话历史（最近 N 轮，完整历史存数据库）
  recent_messages: Message[];
}

interface DayPlan {
  day_number: number;
  date: string | null;
  theme: string;               // "市区文化 + 美食路线"
  activities: Activity[];
  total_walking_steps: number;
  total_transit_time_min: number;
  total_cost: number;
  weather_risk: string | null;
  alternative_plan: DayPlan | null;  // 雨天备选
}

interface Activity {
  id: string;
  poi_name: string;
  poi_id: string;              // POI 知识库 ID
  category: string;
  time_slot: string;           // "morning" | "afternoon" | "evening" | "night"
  start_time: string;          // "09:00"
  end_time: string;            // "11:00"
  duration_min: number;
  location: { lat: number; lng: number };
  cost: number;
  transport_from_prev: {
    mode: string;
    duration_min: number;
    cost: number;
    description: string;
  } | null;
  recommendation_reason: string;
  is_outdoor: boolean;
  intensity: "low" | "medium" | "high";
  is_confirmed: boolean;       // 用户是否已确认
  source: string;              // "rag" | "tool" | "llm"
}

interface Modification {
  timestamp: string;
  target: string;              // "day_3" | "budget" | "route"
  action: string;              // "reduce_intensity" | "add_poi" | "remove_poi"
  before_snapshot: any;
  after_snapshot: any;
  reason: string;
}
```

#### State 更新策略

```python
class StateManager:
    """
    状态管理器。核心原则：
    1. 渐进式收集：不强求一次性获取所有信息
    2. 局部更新：修改只影响变更部分
    3. 不可变历史：每次修改保存 before/after 快照
    4. 约束推导：从用户画像自动推导约束参数
    """
    
    def update_from_user_input(self, parsed_input: dict):
        """从用户输入中提取信息，渐进更新 State"""
        # 只更新用户新提供的字段，不覆盖已有信息
        for key, value in parsed_input.items():
            if value is not None:
                self.state.user_profile[key] = value
        
        # 触发约束推导
        self._derive_constraints()
    
    def _derive_constraints(self):
        """根据用户画像自动推导约束参数"""
        profile = self.state.user_profile
        constraints = self.state.constraints
        
        # pace → 每日景点数和步数上限
        pace_config = {
            "intensive": {"max_pois": 6, "max_steps": 25000},
            "moderate":  {"max_pois": 4, "max_steps": 18000},
            "relaxed":   {"max_pois": 3, "max_steps": 12000},
        }
        config = pace_config.get(profile.pace, pace_config["moderate"])
        constraints.max_daily_pois = config["max_pois"]
        constraints.max_daily_steps = config["max_steps"]
        
        # companions → 进一步降低强度
        if "senior" in profile.companions:
            constraints.max_daily_pois = min(constraints.max_daily_pois, 3)
            constraints.max_daily_steps = min(constraints.max_daily_steps, 8000)
            constraints.rest_time_min = max(constraints.rest_time_min, 60)
            constraints.preferred_transport = ["taxi", "subway_direct"]
        
        if "child" in profile.companions:
            constraints.max_daily_pois = min(constraints.max_daily_pois, 4)
            constraints.avoid_keywords.append("酒吧")
    
    def apply_modification(self, mod: Modification):
        """应用行程修改，保存历史快照"""
        mod.before_snapshot = deepcopy(self._get_target(mod.target))
        self._apply(mod)
        mod.after_snapshot = deepcopy(self._get_target(mod.target))
        self.state.modification_history.append(mod)
```

### 6.3 Itinerary Planner Agent

这是系统的核心 Sub-Agent，负责将候选 POI 列表在多维约束下排列为可执行行程。

#### 规划流程

```
候选 POI 列表（来自 Recommendation Agent）
    │
    ▼
┌──────────────────────┐
│ Step 1: 区域聚类      │  将 POI 按地理位置分组
│ (K-Means / DBSCAN)   │  每组代表一个"半日/全日"活动区域
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Step 2: 日分配        │  将聚类结果分配到各天
│ (Bin Packing)         │  考虑：时间容量、开放日期、用户指定
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Step 3: 日内排序      │  每天内部的景点顺序优化
│ (TSP / 2-opt)         │  最小化移动距离，遵守时间窗口
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Step 4: 时间表生成    │  为每个活动分配具体时间段
│ (Interval Scheduling) │  插入交通时间、用餐时间、休息时间
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Step 5: 约束校验      │  检查所有约束是否满足
│ (Constraint Check)    │  不满足则回退调整
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Step 6: LLM 润色      │  生成自然语言描述和推荐理由
│ (Natural Language)    │  
└──────────────────────┘
```

详细算法设计见 [第八章：约束求解与路径优化算法](#八约束求解与路径优化算法)。

### 6.4 Budget Optimizer Agent

#### 职责
- 计算行程总成本（分项：住宿/餐饮/交通/门票/购物）
- 判断是否超预算
- 生成优化方案（多个策略，按影响/牺牲比排序）
- 支持多档方案对比（省钱版 / 舒适版 / 豪华版）

#### 预算估算模型

```python
class BudgetModel:
    """
    分项预算估算。数据来源优先级：
    1. MCP Tool 实时查询（酒店价格、门票价格）
    2. RAG 知识库（历史价格数据）
    3. 基于城市的统计均值（兜底）
    """
    
    async def estimate(self, itinerary: list[DayPlan], profile: UserProfile) -> BudgetBreakdown:
        accommodation = await self._estimate_accommodation(itinerary, profile)
        meals = self._estimate_meals(itinerary, profile)
        transport = await self._estimate_transport(itinerary)
        tickets = self._sum_ticket_costs(itinerary)
        shopping = self._estimate_shopping(profile)
        
        return BudgetBreakdown(
            accommodation=accommodation,
            meals=meals,
            transport=transport,
            tickets=tickets,
            shopping=shopping,
            total=sum([accommodation, meals, transport, tickets, shopping]),
            buffer_ratio=0.1,  # 10% 余量
            confidence="medium",  # low | medium | high
        )
    
    async def optimize(
        self, 
        current_budget: BudgetBreakdown, 
        target: float
    ) -> list[OptimizationStrategy]:
        """
        生成优化策略列表，按「节省金额 / 体验损失」比值降序排列。
        """
        strategies = []
        gap = current_budget.total - target
        
        # 策略 1: 酒店降级
        if current_budget.accommodation > self.city_avg_accommodation * 1.2:
            saving = current_budget.accommodation * 0.3
            strategies.append(OptimizationStrategy(
                name="酒店区域/档次调整",
                saving=saving,
                impact_score=0.2,  # 对体验影响低
                description="从核心商圈换到交通便利的次核心区域",
                efficiency=saving / 0.2  # 节省/影响 比
            ))
        
        # 策略 2: 餐饮调整
        # 策略 3: 交通方式优化
        # 策略 4: 免费景点替换
        # ...
        
        # 按效率降序排列
        strategies.sort(key=lambda s: s.efficiency, reverse=True)
        
        # 贪心选择：累积节省达到目标即止
        selected = []
        cumulative_saving = 0
        for s in strategies:
            selected.append(s)
            cumulative_saving += s.saving
            if cumulative_saving >= gap:
                break
        
        return selected
```

### 6.5 Recommendation Agent

#### 职责
- 根据用户偏好从知识库中召回候选 POI
- 个性化排序（考虑偏好匹配度、评分、小众程度、时间适配性）
- 支持反向过滤（排除用户不喜欢的类型）

#### 推荐流程

```python
class RecommendationAgent:
    async def recommend(
        self,
        destination: str,
        preferences: list[str],
        anti_preferences: list[str],
        companions: list[str],
        count: int = 20
    ) -> list[ScoredPOI]:
        
        # 1. 多路召回
        rag_results = await self.rag.hybrid_search(
            query=f"{destination} {'  '.join(preferences)}",
            filters={"city": destination},
            top_k=50
        )
        
        tool_results = await self.poi_mcp.search_poi(
            city=destination,
            keywords=preferences,
            count=30
        )
        
        # 2. 合并去重
        candidates = self._merge_and_deduplicate(rag_results, tool_results)
        
        # 3. 多维评分
        scored = []
        for poi in candidates:
            score = self._compute_score(poi, preferences, anti_preferences, companions)
            scored.append(ScoredPOI(poi=poi, score=score))
        
        # 4. 排序 + 截取
        scored.sort(key=lambda x: x.score.total, reverse=True)
        return scored[:count]
    
    def _compute_score(self, poi, prefs, anti_prefs, companions) -> POIScore:
        """
        多维评分模型：
        - preference_match: 与用户偏好的匹配度（0-1）
        - quality: 基于评分和评论数的质量分（0-1）
        - uniqueness: 小众程度（游客占比低 = 高分）（0-1）
        - companion_fit: 与同行人的适配度（0-1）
        - anti_preference_penalty: 与反偏好匹配则扣分（0-1）
        
        total = w1*preference_match + w2*quality + w3*uniqueness 
                + w4*companion_fit - w5*anti_preference_penalty
        """
        return POIScore(
            preference_match=self._calc_pref_match(poi.tags, prefs),
            quality=self._calc_quality(poi.rating, poi.review_count),
            uniqueness=self._calc_uniqueness(poi.tourist_ratio),
            companion_fit=self._calc_companion_fit(poi.suitable_for, companions),
            anti_preference_penalty=self._calc_anti_pref(poi.tags, anti_prefs),
        )
```

### 6.6 Guardrail Module

#### 输入安全

```python
class InputGuardrail:
    """
    输入安全检查，防止：
    1. Prompt injection（通过用户输入操纵 Agent 行为）
    2. 非旅行相关请求（引导回旅行话题）
    3. 敏感地区/活动请求（安全提醒）
    """
    
    def check(self, user_input: str) -> GuardrailResult:
        # 1. Prompt injection 检测
        if self._detect_injection(user_input):
            return GuardrailResult(
                safe=False,
                reason="detected_injection",
                action="reject"
            )
        
        # 2. 话题边界检测
        if not self._is_travel_related(user_input):
            return GuardrailResult(
                safe=True,
                reason="off_topic",
                action="redirect",
                message="我是旅行规划助手，让我们聊聊你的旅行计划吧！"
            )
        
        # 3. 敏感目的地/活动检测
        if self._contains_sensitive_content(user_input):
            return GuardrailResult(
                safe=True,
                reason="sensitive_content",
                action="warn",
                message="该地区/活动存在安全风险，建议查阅最新旅行安全提示。"
            )
        
        return GuardrailResult(safe=True, action="pass")
```

#### 输出安全

```python
class OutputGuardrail:
    """
    输出质量与安全检查：
    1. 幻觉检测：推荐的 POI 是否真实存在
    2. 信息时效性：价格/开放时间是否为最新数据
    3. 安全性：是否包含不安全的建议
    4. 偏见检测：是否存在文化/地域偏见
    """
    
    async def check(self, response: AgentResponse) -> OutputCheckResult:
        issues = []
        
        # 1. POI 真实性验证
        for activity in response.activities:
            if activity.source == "llm":  # 非工具/RAG 来源
                exists = await self.poi_mcp.verify_existence(activity.poi_name)
                if not exists:
                    issues.append(Issue(
                        type="hallucination",
                        severity="high",
                        target=activity.poi_name,
                        action="remove_or_replace"
                    ))
        
        # 2. 数据时效性标注
        for activity in response.activities:
            if activity.data_freshness_days > 30:
                issues.append(Issue(
                    type="stale_data",
                    severity="low",
                    target=activity.poi_name,
                    action="add_disclaimer"
                ))
        
        return OutputCheckResult(issues=issues, pass_rate=len(issues) == 0)
```

---

## 七、数据模型与 Schema 定义

### 7.1 POI 数据模型

```typescript
interface POI {
  // 基础信息
  id: string;                    // 全局唯一 ID
  name: string;                  // 中文名称
  name_local: string;            // 当地语言名称
  name_en: string;               // 英文名称
  
  // 地理信息
  city: string;
  district: string;              // 区/街道
  address: string;
  location: { lat: number; lng: number };
  nearby_station: string;        // 最近车站
  
  // 分类与标签
  category: POICategory;         // "attraction" | "restaurant" | "shopping" | "hotel" | "transport_hub"
  subcategory: string;           // "temple" | "museum" | "ramen" | "department_store"
  tags: string[];                // ["历史", "拍照", "免费", "亲子友好", "小众"]
  themes: string[];              // ["美食", "文化", "动漫", "自然"]
  
  // 游玩信息
  recommended_duration_min: number;   // 建议游玩时长
  best_time_of_day: string[];         // ["morning", "afternoon"]
  best_season: string[];              // ["spring", "autumn"]
  intensity: "low" | "medium" | "high";
  is_outdoor: boolean;
  
  // 适配信息
  suitable_for: string[];        // ["亲子", "情侣", "老人", "独行"]
  not_suitable_for: string[];    // ["行动不便"]
  accessibility: "full" | "partial" | "none";
  
  // 费用信息
  ticket_price: number | null;   // 门票价格（当地货币）
  ticket_currency: string;
  is_free: boolean;
  average_meal_cost: number | null;  // 餐厅平均消费
  
  // 开放信息
  opening_hours: {
    [day: string]: { open: string; close: string } | "closed"
  };
  closed_dates: string[];        // 特殊闭馆日
  reservation_required: boolean;
  
  // 质量信号
  rating: number;                // 0-5
  review_count: number;
  tourist_ratio: number;         // 游客占比 0-1
  local_popularity: number;      // 本地人气 0-1
  
  // 数据来源与时效
  sources: string[];             // ["google_places", "tabelog", "rag_knowledge"]
  last_verified: string;         // ISO 8601
  data_confidence: "high" | "medium" | "low";
  
  // 语义描述（用于 RAG embedding）
  description: string;
  tips: string;
  nearby_pois: string[];         // 关联推荐
}

type POICategory = "attraction" | "restaurant" | "shopping" | "hotel" | "transport_hub" | "entertainment";
```

### 7.2 行程输出 Schema

```typescript
interface ItineraryResponse {
  // 总览
  summary: {
    destination: string;
    total_days: number;
    total_budget: number;
    budget_currency: string;
    pace_description: string;
    highlights: string[];        // 行程亮点摘要
  };
  
  // 每日计划
  days: DayPlan[];
  
  // 预算明细
  budget_breakdown: {
    accommodation: { total: number; per_night: number; hotel_area: string };
    meals: { total: number; per_day: number; highlights: string[] };
    transport: { total: number; breakdown: { mode: string; cost: number }[] };
    tickets: { total: number; items: { name: string; cost: number }[] };
    shopping: { estimated: number };
    total: number;
    buffer: number;
    vs_budget: number;           // 与预算的差额（正数=节省）
  };
  
  // 替代方案
  alternatives: {
    budget_friendly: ItinerarySummary;
    comfort: ItinerarySummary;
    deep_experience: ItinerarySummary;
  } | null;
  
  // 风险提醒
  risks: {
    weather: string[];
    closure: string[];
    crowd: string[];
    safety: string[];
  };
  
  // 出行辅助
  travel_tips: {
    visa: string | null;
    currency: { rate: number; tips: string };
    sim_card: string;
    transport_card: string;
    language: string[];
    customs: string[];
    emergency_contacts: { name: string; number: string }[];
  } | null;
  
  // 可交互选项
  adjustable_options: string[];  // ["想调整哪天的行程？", "需要加入温泉吗？", "要看看更省钱的方案吗？"]
}
```

---

## 八、约束求解与路径优化算法

### 8.1 问题建模

旅行规划可以建模为 **带时间窗口的多约束车辆路径问题（VRPTW 变体）**：

**形式化定义**：

```
给定：
  - POI 集合 P = {p1, p2, ..., pn}，每个 POI 有：
    - 地理坐标 (lat_i, lng_i)
    - 游玩时长 d_i
    - 开放时间窗口 [o_i, c_i]
    - 门票费用 cost_i
    - 强度等级 intensity_i
    - 偏好匹配度 pref_i
  - 天数 D
  - 每天可用时间窗口 [T_start, T_end]
  - 约束集合 C:
    - 总预算 B_max
    - 每日最大景点数 N_max
    - 每日最大步行量 S_max
    - 每日最大交通时间 TR_max
  - 距离/时间矩阵 dist[i][j], time[i][j]

目标：
  找到 POI 的一个子集 P' ⊆ P 和将 P' 分配到 D 天的方案，使得：
  1. 最大化总偏好匹配度 Σ pref_i
  2. 最小化总移动距离 Σ dist
  3. 满足所有约束 C
```

这是一个 NP-hard 问题，精确求解在实际场景下不可行。我们采用分层启发式方法。

### 8.2 分层求解算法

#### Layer 1: 区域聚类（Spatial Clustering）

将候选 POI 按地理位置聚类，使得同一天的景点尽量在同一区域。

```python
from sklearn.cluster import DBSCAN
import numpy as np

def cluster_pois(pois: list[POI], num_days: int) -> list[list[POI]]:
    """
    使用 DBSCAN 对 POI 进行地理聚类。
    
    DBSCAN 相比 K-Means 的优势：
    1. 不需要预设簇数（K-Means 需要 K=天数，但一天可能跨区）
    2. 能识别噪声点（孤立景点可以灵活分配）
    3. 基于密度，更适合地理空间分布
    
    参数选择：
    - eps: 1.5km（步行可达范围），使用 haversine 距离
    - min_samples: 2（至少 2 个景点形成一个区域）
    """
    coords = np.array([[p.location.lat, p.location.lng] for p in pois])
    
    # haversine 距离，eps 单位为弧度
    clustering = DBSCAN(
        eps=1.5 / 6371,  # 1.5km / 地球半径
        min_samples=2,
        metric='haversine'
    ).fit(np.radians(coords))
    
    clusters = {}
    noise_pois = []
    for poi, label in zip(pois, clustering.labels_):
        if label == -1:
            noise_pois.append(poi)
        else:
            clusters.setdefault(label, []).append(poi)
    
    return list(clusters.values()), noise_pois
```

#### Layer 2: 日分配（Day Assignment）

将聚类结果分配到各天，同时满足每天的时间和强度约束。

```python
def assign_clusters_to_days(
    clusters: list[list[POI]],
    noise_pois: list[POI],
    num_days: int,
    constraints: Constraints
) -> list[list[POI]]:
    """
    使用贪心 + Bin Packing 策略：
    1. 大簇优先分配到独立天
    2. 小簇合并到同一天（如果距离合理）
    3. 噪声点填充到有空余的天
    4. 遵守每天最大景点数和时间约束
    """
    days = [[] for _ in range(num_days)]
    day_durations = [0] * num_days  # 每天已分配时长
    
    # 按簇大小降序排列
    clusters.sort(key=len, reverse=True)
    
    for cluster in clusters:
        cluster_duration = sum(p.recommended_duration_min for p in cluster)
        
        # 找到剩余容量最大的天
        best_day = min(range(num_days), key=lambda d: day_durations[d])
        max_capacity = constraints.max_daily_duration_min
        
        if day_durations[best_day] + cluster_duration <= max_capacity:
            days[best_day].extend(cluster)
            day_durations[best_day] += cluster_duration
        else:
            # 簇太大，需要拆分
            remaining = cluster[:]
            while remaining:
                best_day = min(range(num_days), key=lambda d: day_durations[d])
                available = max_capacity - day_durations[best_day]
                to_add = []
                for poi in remaining[:]:
                    if day_durations[best_day] + poi.recommended_duration_min <= max_capacity:
                        to_add.append(poi)
                        remaining.remove(poi)
                        day_durations[best_day] += poi.recommended_duration_min
                if not to_add:
                    break
                days[best_day].extend(to_add)
    
    # 分配噪声点
    for poi in noise_pois:
        best_day = min(range(num_days), key=lambda d: day_durations[d])
        if day_durations[best_day] + poi.recommended_duration_min <= max_capacity:
            days[best_day].append(poi)
            day_durations[best_day] += poi.recommended_duration_min
    
    return days
```

#### Layer 3: 日内路径优化（Intra-day TSP）

每天内部的景点访问顺序优化，使用 2-opt 改进法求解近似 TSP：

```python
async def optimize_daily_route(
    pois: list[POI],
    map_mcp: MapMCPClient,
    start_location: dict | None = None  # 酒店位置
) -> list[POI]:
    """
    使用 2-opt 算法优化日内路线。
    
    2-opt 算法：
    1. 从初始路线开始（最近邻启发式）
    2. 尝试每对边的交换
    3. 如果交换后总距离减少，则接受
    4. 重复直到无法继续改进
    
    时间复杂度：O(n²) per iteration，n 通常 < 10，完全可行。
    """
    if len(pois) <= 2:
        return pois
    
    # 获取距离矩阵
    all_points = ([start_location] if start_location else []) + \
                 [{"lat": p.location.lat, "lng": p.location.lng} for p in pois]
    
    dist_matrix = await map_mcp.get_distance_matrix(
        origins=all_points,
        destinations=all_points,
        mode="transit"
    )
    
    # 最近邻启发式构建初始路线
    n = len(pois)
    visited = [False] * n
    route = []
    current = 0  # 从第一个点开始（或从酒店开始）
    
    for _ in range(n):
        best_next = None
        best_dist = float('inf')
        for j in range(n):
            if not visited[j]:
                d = dist_matrix.matrix[current][j + (1 if start_location else 0)].duration_s
                if d < best_dist:
                    best_dist = d
                    best_next = j
        visited[best_next] = True
        route.append(best_next)
        current = best_next + (1 if start_location else 0)
    
    # 2-opt 改进
    improved = True
    while improved:
        improved = False
        for i in range(len(route) - 1):
            for j in range(i + 2, len(route)):
                # 计算交换前后的距离变化
                delta = self._calc_2opt_delta(dist_matrix, route, i, j, start_location)
                if delta < 0:
                    # 反转 i+1 到 j 之间的路线
                    route[i+1:j+1] = reversed(route[i+1:j+1])
                    improved = True
    
    return [pois[i] for i in route]
```

#### Layer 4: 时间窗口适配（Time Window Scheduling）

将优化后的路线映射到具体时间，考虑开放时间、用餐时间、休息时间：

```python
def schedule_day(
    ordered_pois: list[POI],
    day_start: str,            # "09:00"
    day_end: str,              # "21:00"
    constraints: Constraints,
    transport_times: list[int]  # 相邻景点间交通时间（分钟）
) -> list[ScheduledActivity]:
    """
    将有序 POI 列表映射为带时间的日程表。
    
    规则：
    1. 遵守景点开放时间窗口
    2. 午餐时间 11:30-13:30，晚餐时间 17:30-19:30
    3. 如有 rest_time 约束，在下午安排休息
    4. 如果某景点无法在开放时间内到达，跳过并标记
    """
    schedule = []
    current_time = parse_time(day_start)
    
    for i, poi in enumerate(ordered_pois):
        # 加入交通时间
        if i > 0:
            transit_min = transport_times[i - 1]
            current_time += timedelta(minutes=transit_min)
            schedule.append(TransitActivity(
                duration_min=transit_min,
                start_time=format_time(current_time - timedelta(minutes=transit_min)),
                end_time=format_time(current_time)
            ))
        
        # 检查是否需要插入用餐
        if self._should_insert_meal(current_time, schedule):
            meal = self._find_nearby_restaurant(poi.location)
            meal_duration = 60  # 分钟
            schedule.append(MealActivity(
                restaurant=meal,
                start_time=format_time(current_time),
                duration_min=meal_duration
            ))
            current_time += timedelta(minutes=meal_duration)
        
        # 检查景点开放时间
        if not self._is_open_at(poi, current_time):
            # 尝试延后
            open_time = parse_time(poi.opening_hours[day_of_week]["open"])
            if open_time > current_time and open_time < parse_time(day_end):
                # 插入等待或调整
                current_time = open_time
            else:
                continue  # 跳过，标记为不可安排
        
        # 安排景点
        end_time = current_time + timedelta(minutes=poi.recommended_duration_min)
        
        schedule.append(ScheduledActivity(
            poi=poi,
            start_time=format_time(current_time),
            end_time=format_time(end_time),
            duration_min=poi.recommended_duration_min
        ))
        
        current_time = end_time
    
    return schedule
```

#### Layer 5: 多约束校验

```python
class ConstraintChecker:
    """
    对生成的行程进行全面的约束校验。
    返回违反的约束列表及严重程度。
    """
    
    def check(self, itinerary: list[DayPlan], constraints: Constraints) -> list[Violation]:
        violations = []
        
        # 1. 预算约束
        total_cost = sum(day.total_cost for day in itinerary)
        if constraints.max_budget and total_cost > constraints.max_budget:
            violations.append(Violation(
                type="budget_exceeded",
                severity="high",
                detail=f"总预算 {total_cost} 超出限制 {constraints.max_budget}",
                suggestion="调用 BudgetOptimizer 优化"
            ))
        
        # 2. 每日强度约束
        for day in itinerary:
            if len(day.activities) > constraints.max_daily_pois:
                violations.append(Violation(
                    type="too_many_pois",
                    severity="medium",
                    detail=f"Day {day.day_number} 有 {len(day.activities)} 个景点，超出 {constraints.max_daily_pois} 限制",
                    suggestion="删除低优先级景点"
                ))
            
            if day.total_walking_steps > constraints.max_daily_steps:
                violations.append(Violation(
                    type="too_much_walking",
                    severity="medium",
                    detail=f"Day {day.day_number} 步行 {day.total_walking_steps} 步，超出 {constraints.max_daily_steps}",
                    suggestion="替换为低强度景点或增加出租车"
                ))
        
        # 3. 时间窗口约束
        for day in itinerary:
            for activity in day.activities:
                if not self._is_within_opening_hours(activity):
                    violations.append(Violation(
                        type="outside_opening_hours",
                        severity="high",
                        detail=f"{activity.poi_name} 在安排时间 {activity.start_time} 不开放",
                        suggestion="调整时间或换天"
                    ))
        
        # 4. 路线合理性
        for day in itinerary:
            if day.total_transit_time_min > constraints.max_daily_transit_time_min:
                violations.append(Violation(
                    type="too_much_transit",
                    severity="medium",
                    detail=f"Day {day.day_number} 交通时间 {day.total_transit_time_min} 分钟",
                    suggestion="重新进行区域聚类"
                ))
        
        # 5. 偏好覆盖度
        covered_themes = set()
        for day in itinerary:
            for a in day.activities:
                covered_themes.update(a.themes)
        
        missing_prefs = set(constraints.user_preferences) - covered_themes
        if missing_prefs:
            violations.append(Violation(
                type="preference_not_covered",
                severity="low",
                detail=f"用户偏好 {missing_prefs} 未在行程中体现",
                suggestion="增加相关类型的 POI"
            ))
        
        return violations
```

### 8.3 高级版本：使用 OR-Tools 求解

对于需要精确优化的场景（如商业版本），可以引入 Google OR-Tools：

```python
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

def solve_with_ortools(
    distance_matrix: list[list[int]],
    time_windows: list[tuple[int, int]],
    durations: list[int],
    num_days: int,
    max_time_per_day: int
) -> list[list[int]]:
    """
    使用 Google OR-Tools 求解带时间窗口的 VRP 问题。
    
    将多天旅行建模为多车辆路径问题：
    - 每天 = 一辆车
    - 酒店 = depot
    - 景点 = 客户
    - 时间 = 容量约束
    """
    manager = pywrapcp.RoutingIndexManager(
        len(distance_matrix),
        num_days,  # num_vehicles = num_days
        0  # depot = hotel
    )
    
    routing = pywrapcp.RoutingModel(manager)
    
    # 距离回调
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # 时间窗口约束
    time_dimension = 'Time'
    routing.AddDimension(
        transit_callback_index,
        60,  # 最大等待时间（分钟）
        max_time_per_day,
        False,
        time_dimension
    )
    time_dim = routing.GetDimensionOrDie(time_dimension)
    
    for location_idx, (start, end) in enumerate(time_windows):
        if location_idx == 0:
            continue  # depot
        index = manager.NodeToIndex(location_idx)
        time_dim.CumulVar(index).SetRange(start, end)
    
    # 求解
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 10
    
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        return extract_routes(manager, routing, solution, num_days)
    else:
        # 降级到启发式方法
        return fallback_heuristic(distance_matrix, num_days)
```

---

## 九、RAG 知识库系统

### 9.1 数据管线设计

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG Data Pipeline                            │
│                                                                   │
│  数据源                  处理                   存储               │
│  ┌──────────┐          ┌──────────┐          ┌──────────┐       │
│  │ 城市攻略  │───┐      │ 清洗 +   │          │ Vector   │       │
│  │ (编辑撰写)│   │      │ 结构化   │          │ Store    │       │
│  └──────────┘   │      │ 提取     │     ┌───▶│(Qdrant / │       │
│  ┌──────────┐   │      └──────────┘     │    │ Pinecone)│       │
│  │ POI 数据  │───┤           │           │    └──────────┘       │
│  │ (API 抓取)│   │           ▼           │                       │
│  └──────────┘   │      ┌──────────┐     │    ┌──────────┐       │
│  ┌──────────┐   ├─────▶│ Chunk +  │─────┤    │ BM25     │       │
│  │ 用户评论  │───┤      │ Embed    │     ├───▶│ Index    │       │
│  │ (摘要化) │   │      └──────────┘     │    │(Elastic) │       │
│  └──────────┘   │                       │    └──────────┘       │
│  ┌──────────┐   │                       │                       │
│  │ 签证/交通 │───┘                       │    ┌──────────┐       │
│  │ 政策文档  │                           └───▶│ Metadata │       │
│  └──────────┘                                │ Store    │       │
│                                               │ (PG)    │       │
│  ┌───────────────────────┐                   └──────────┘       │
│  │ 增量更新 (Daily Cron)  │                                      │
│  │ - API 价格/状态刷新    │                                      │
│  │ - 新评论摘要            │                                      │
│  │ - 过期数据标记          │                                      │
│  └───────────────────────┘                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 RAG vs Tool 的职责边界

| 信息类型 | 获取方式 | 原因 |
|---------|---------|------|
| POI 基础介绍、攻略文本、游玩建议 | RAG | 变化慢，适合离线索引 |
| 用户评论摘要、本地推荐 | RAG | 经过预处理，适合语义检索 |
| 签证政策、文化礼仪、注意事项 | RAG | 变化慢，适合文档检索 |
| 实时天气预报 | Tool (MCP) | 实时数据，不能缓存 |
| 实时交通状况 | Tool (MCP) | 实时数据 |
| 酒店实时价格与可用性 | Tool (MCP) | 实时数据 |
| 景点当前开放状态 | Tool (MCP) | 可能临时变化 |
| 汇率 | Tool (MCP) | 实时数据 |
| POI 地理坐标与距离 | Tool (MCP) | 需要精确计算 |

### 9.3 Chunk 策略

```python
class TravelChunker:
    """
    旅游领域的 Chunk 策略。
    
    核心原则：一个 POI = 一个完整 Chunk。
    不拆分单个 POI 的信息，因为：
    1. POI 信息是高度关联的（名称+位置+时间+费用+评价）
    2. 拆分后检索到部分信息（只有名字没有位置）会导致幻觉
    3. POI 信息量通常 < 500 token，不需要拆分
    
    城市攻略类长文档按章节拆分，每章保留城市上下文。
    """
    
    def chunk_poi(self, poi: POI) -> Document:
        """POI 整体作为一个 Chunk"""
        text = self._format_poi_as_text(poi)
        return Document(
            text=text,
            metadata={
                "type": "poi",
                "city": poi.city,
                "category": poi.category,
                "tags": poi.tags,
                "poi_id": poi.id,
                "last_verified": poi.last_verified
            }
        )
    
    def chunk_guide(self, guide_text: str, city: str) -> list[Document]:
        """城市攻略按章节拆分，每节保留城市元数据"""
        sections = self._split_by_sections(guide_text)
        return [
            Document(
                text=f"[城市：{city}] {section}",
                metadata={"type": "guide", "city": city, "section": title}
            )
            for title, section in sections
        ]
```

### 9.4 检索策略

```python
class HybridRetriever:
    """
    混合检索：Vector Search + BM25 + Cross-Encoder Rerank
    
    为什么需要混合检索？
    - Vector Search 擅长语义匹配（"好吃的地方" → 美食相关 POI）
    - BM25 擅长精确匹配（"黑门市场" → 精确命中该 POI）
    - Cross-Encoder Rerank 提升最终排序质量
    
    多语言处理：
    - Embedding 模型选用 multilingual-e5-large（支持中日英）
    - BM25 使用 ICU tokenizer（支持 CJK 分词）
    """
    
    async def search(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 10
    ) -> list[RetrievedDocument]:
        
        # 1. Vector Search
        vector_results = await self.vector_store.search(
            embedding=self.embed_model.encode(query),
            filters=filters,
            top_k=top_k * 3  # 多召回
        )
        
        # 2. BM25 Search
        bm25_results = await self.bm25_index.search(
            query=query,
            filters=filters,
            top_k=top_k * 3
        )
        
        # 3. 合并去重
        candidates = self._merge_results(
            vector_results, 
            bm25_results,
            vector_weight=0.6,
            bm25_weight=0.4
        )
        
        # 4. Cross-Encoder Rerank
        reranked = await self.reranker.rerank(
            query=query,
            documents=candidates,
            top_k=top_k
        )
        
        return reranked
```

### 9.5 数据质量保证

```python
class DataQualityManager:
    """
    数据质量保证策略：
    1. 时效性管理：标记数据年龄，超期自动刷新
    2. 来源追踪：每条数据标注来源，支持溯源
    3. 冲突检测：多源数据冲突时的裁决策略
    4. 合规审计：确保数据使用符合平台 ToS
    """
    
    def check_freshness(self, doc: Document) -> FreshnessStatus:
        age_days = (datetime.now() - doc.metadata["last_verified"]).days
        
        if doc.metadata["type"] == "poi":
            if age_days > 90:
                return FreshnessStatus.STALE   # 需要刷新
            elif age_days > 30:
                return FreshnessStatus.AGING   # 标注 "信息可能有变化"
            else:
                return FreshnessStatus.FRESH
        
        if doc.metadata["type"] == "policy":  # 签证政策等
            if age_days > 30:
                return FreshnessStatus.STALE
            else:
                return FreshnessStatus.FRESH
    
    async def refresh_stale_data(self):
        """定时任务：刷新过期数据"""
        stale_docs = await self.find_stale_documents()
        for doc in stale_docs:
            if doc.metadata["type"] == "poi":
                fresh_data = await self.poi_mcp.get_poi_detail(doc.metadata["poi_id"])
                if fresh_data:
                    await self.update_document(doc, fresh_data)
                else:
                    await self.mark_unavailable(doc)
```

---

## 十、评估体系设计

### 10.1 评估指标总表

| 维度 | 指标 | 计算方式 | 权重 | 阈值 |
|------|------|---------|------|------|
| 偏好匹配度 | preference_coverage | 用户偏好中被行程覆盖的比例 | 0.20 | ≥ 0.8 |
| 路线合理性 | route_efficiency | 1 - (实际距离 / 最优距离) | 0.15 | ≥ 0.7 |
| 时间可行性 | time_feasibility | 所有活动是否在开放时间内 | 0.15 | = 1.0 |
| 预算合理性 | budget_compliance | max(0, 1 - 超支比例) | 0.15 | ≥ 0.9 |
| 舒适度 | comfort_score | 基于步行量/景点数/时长的综合分 | 0.10 | ≥ 0.7 |
| 真实性 | factuality | 所有 POI 可通过工具验证的比例 | 0.15 | ≥ 0.95 |
| 可解释性 | explainability | 有推荐理由的活动占比 | 0.05 | ≥ 0.9 |
| 多样性 | diversity | 活动类别的 Shannon 熵 | 0.05 | ≥ 0.6 |

### 10.2 评估实现

```python
class ItineraryEvaluator:
    """
    行程质量评估器。
    在 Agent Loop 的 Observe 阶段调用，用于判断是否需要进一步优化。
    """
    
    def evaluate(self, itinerary: list[DayPlan], state: TravelState) -> EvalResult:
        scores = {}
        
        # 1. 偏好匹配度
        user_prefs = set(state.user_profile.preferences)
        covered = set()
        for day in itinerary:
            for activity in day.activities:
                covered.update(set(activity.themes) & user_prefs)
        scores["preference_coverage"] = len(covered) / len(user_prefs) if user_prefs else 1.0
        
        # 2. 路线合理性
        total_transit = sum(day.total_transit_time_min for day in itinerary)
        total_activity = sum(
            sum(a.duration_min for a in day.activities)
            for day in itinerary
        )
        # 交通时间占比：越低越好
        transit_ratio = total_transit / (total_transit + total_activity) if (total_transit + total_activity) > 0 else 0
        scores["route_efficiency"] = max(0, 1 - transit_ratio * 2)  # 交通时间超过活动时间50%则为0
        
        # 3. 时间可行性
        time_violations = 0
        total_activities = 0
        for day in itinerary:
            for activity in day.activities:
                total_activities += 1
                if not self._is_within_opening_hours(activity):
                    time_violations += 1
        scores["time_feasibility"] = 1 - (time_violations / total_activities) if total_activities > 0 else 1.0
        
        # 4. 预算合理性
        total_cost = sum(day.total_cost for day in itinerary)
        max_budget = state.constraints.max_budget
        if max_budget:
            over_ratio = max(0, (total_cost - max_budget) / max_budget)
            scores["budget_compliance"] = max(0, 1 - over_ratio)
        else:
            scores["budget_compliance"] = 1.0
        
        # 5. 舒适度
        comfort_scores = []
        for day in itinerary:
            day_comfort = 1.0
            if day.total_walking_steps > state.constraints.max_daily_steps:
                day_comfort -= 0.3
            if len(day.activities) > state.constraints.max_daily_pois:
                day_comfort -= 0.3
            if day.total_transit_time_min > state.constraints.max_daily_transit_time_min:
                day_comfort -= 0.2
            comfort_scores.append(max(0, day_comfort))
        scores["comfort_score"] = sum(comfort_scores) / len(comfort_scores) if comfort_scores else 1.0
        
        # 6. 真实性
        verified = sum(
            1 for day in itinerary
            for a in day.activities
            if a.source in ("tool", "rag")
        )
        total = sum(len(day.activities) for day in itinerary)
        scores["factuality"] = verified / total if total > 0 else 1.0
        
        # 7. 可解释性
        explained = sum(
            1 for day in itinerary
            for a in day.activities
            if a.recommendation_reason
        )
        scores["explainability"] = explained / total if total > 0 else 1.0
        
        # 8. 多样性（Shannon 熵）
        category_counts = {}
        for day in itinerary:
            for a in day.activities:
                category_counts[a.category] = category_counts.get(a.category, 0) + 1
        if category_counts:
            total_c = sum(category_counts.values())
            entropy = -sum(
                (c/total_c) * math.log2(c/total_c)
                for c in category_counts.values()
            )
            max_entropy = math.log2(len(category_counts))
            scores["diversity"] = entropy / max_entropy if max_entropy > 0 else 0
        else:
            scores["diversity"] = 0
        
        # 加权总分
        weights = {
            "preference_coverage": 0.20,
            "route_efficiency": 0.15,
            "time_feasibility": 0.15,
            "budget_compliance": 0.15,
            "comfort_score": 0.10,
            "factuality": 0.15,
            "explainability": 0.05,
            "diversity": 0.05,
        }
        total_score = sum(scores[k] * weights[k] for k in weights)
        
        # 判断是否通过
        critical_failures = [
            k for k in ["time_feasibility", "factuality", "budget_compliance"]
            if scores[k] < 0.8
        ]
        
        return EvalResult(
            scores=scores,
            total_score=total_score,
            passed=total_score >= 0.75 and len(critical_failures) == 0,
            critical_failures=critical_failures,
            improvement_suggestions=self._generate_suggestions(scores)
        )
```

### 10.3 自动化测试用例

```python
# eval/test_cases.py

TEST_CASES = [
    {
        "name": "basic_osaka_5days",
        "input": "我想去大阪玩 5 天，预算 8000，喜欢美食和动漫",
        "expected": {
            "destination": "大阪",
            "days": 5,
            "budget_max": 8000,
            "must_cover_themes": ["美食", "动漫"],
            "min_preference_coverage": 0.8,
            "max_budget_usage": 1.0,
            "min_factuality": 0.95,
        }
    },
    {
        "name": "senior_friendly_tokyo",
        "input": "和爸妈去东京 4 天，他们走不了太多路",
        "expected": {
            "destination": "东京",
            "days": 4,
            "max_daily_steps": 8000,
            "max_daily_pois": 3,
            "must_have_rest_time": True,
        }
    },
    {
        "name": "budget_optimization",
        "input": "东京 5 天，预算 6000，越省越好",
        "expected": {
            "destination": "东京",
            "days": 5,
            "budget_max": 6000,
            "should_include_free_attractions": True,
            "min_budget_compliance": 0.95,
        }
    },
    {
        "name": "multi_constraint",
        "input": "东京 6 天，第一天晚上到，第三天想去镰仓，酒店住新宿，预算 9000，不要太累，想吃本地推荐",
        "expected": {
            "destination": "东京",
            "days": 6,
            "day_1_type": "arrival_half_day",
            "day_3_must_include": "�的仓",
            "accommodation_area": "新宿",
            "budget_max": 9000,
            "pace": "relaxed",
            "restaurant_type": "local_favorite",
        }
    },
    {
        "name": "weather_adaptation",
        "input": "如果第二天下雨怎么办？",
        "precondition": "已有 5 天行程，Day 2 有户外活动",
        "expected": {
            "should_replace_outdoor": True,
            "replacement_type": "indoor",
            "other_days_unchanged": True,
        }
    },
    {
        "name": "realtime_nearby",
        "input": "我现在在新宿，下午还有 4 小时，推荐附近能去的地方",
        "expected": {
            "should_use_location": True,
            "max_travel_time_min": 20,
            "results_within_time_budget": True,
        }
    },
]
```

---

## 十一、Human-in-the-Loop 协作机制

### 11.1 信心阈值与降级策略

```python
class ConfidenceManager:
    """
    根据 Agent 对自身输出的信心等级，决定交互策略。
    """
    
    HIGH_CONFIDENCE = 0.8     # 直接输出
    MEDIUM_CONFIDENCE = 0.5   # 输出 + 请求确认
    LOW_CONFIDENCE = 0.3      # 输出多选方案
    
    def decide_interaction(self, confidence: float, context: str) -> InteractionMode:
        if confidence >= self.HIGH_CONFIDENCE:
            return InteractionMode.DIRECT_OUTPUT
            # "我为你安排了这样的行程：..."
        
        elif confidence >= self.MEDIUM_CONFIDENCE:
            return InteractionMode.OUTPUT_WITH_CONFIRMATION
            # "我建议这样安排，你觉得怎么样？如果不合适可以告诉我调整方向。"
        
        elif confidence >= self.LOW_CONFIDENCE:
            return InteractionMode.MULTIPLE_OPTIONS
            # "我不太确定你更喜欢哪种风格，这里有两个方案：
            #  方案 A：紧凑型，5 天覆盖 15 个景点
            #  方案 B：轻松型，5 天覆盖 8 个景点
            #  你更倾向哪个？"
        
        else:
            return InteractionMode.ASK_CLARIFICATION
            # "你提到想要'特别的体验'，能具体说说是什么类型的吗？
            #  比如：文化体验、冒险活动、美食探索、自然风光？"
```

### 11.2 追问策略

```python
class ClarificationStrategy:
    """
    追问策略设计原则：
    1. 一次最多追问 2 个问题
    2. 信息不全也可以先生成草案
    3. 追问的问题要有明确选项，减少用户思考负担
    4. 关键信息（目的地/天数）必须确认，次要信息用默认值
    """
    
    REQUIRED_FIELDS = ["destination", "travel_days"]
    OPTIONAL_FIELDS_WITH_DEFAULTS = {
        "budget": None,                    # 不设限
        "travelers_count": 1,
        "pace": "moderate",
        "companions": [],
        "accommodation_preference": None,
        "departure_city": None,
        "budget_includes_flight": False,
    }
    
    def get_clarification_questions(self, state: TravelState) -> list[str] | None:
        missing_required = [
            f for f in self.REQUIRED_FIELDS
            if getattr(state.user_profile, f) is None
        ]
        
        if missing_required:
            # 必须信息缺失，但友好地问
            return self._format_required_questions(missing_required[:2])
        
        # 所有必须信息已有，可以先生成草案
        # 在草案末尾附带可选问题
        return None  # 不需要追问，直接生成
    
    def _format_required_questions(self, missing: list[str]) -> list[str]:
        templates = {
            "destination": "你想去哪个城市/国家？",
            "travel_days": "大概玩几天？",
        }
        return [templates[f] for f in missing if f in templates]
```

### 11.3 关键操作确认机制

```python
class ConfirmationGate:
    """
    需要用户显式确认的操作（防止 Agent 自动执行不可逆操作）。
    """
    
    REQUIRES_CONFIRMATION = [
        "booking_hotel",        # 预订酒店
        "booking_ticket",       # 购买门票
        "booking_transport",    # 预订交通
        "finalize_itinerary",   # 确认最终行程
        "major_plan_change",    # 大幅度行程变更（> 2 天受影响）
    ]
    
    SKIP_CONFIRMATION = [
        "add_poi",              # 增加景点
        "remove_poi",           # 删除景点
        "adjust_time",          # 调整时间
        "weather_adaptation",   # 天气适配（自动替换）
        "route_optimization",   # 路线优化
    ]
```

---

## 十二、前端交互设计规范

### 12.1 行程展示设计

```
┌─────────────────────────────────────────────────────┐
│  📍 大阪 5 天行程                          [编辑] [导出] │
│                                                       │
│  总预算：¥7,650 / ¥8,000          节奏：轻松          │
│  ████████████████████░░░░ 95.6%                      │
│                                                       │
│  ┌─── Day 1 ─── 抵达 + 轻度探索 ──────────────────┐ │
│  │                                                   │ │
│  │  🏨 14:00  抵达关西机场                          │ │
│  │     │  🚃 南海电铁 → 难波（50min, ¥920）        │ │
│  │     ▼                                             │ │
│  │  🍜 16:00  道顿堀美食街                          │ │
│  │     📝 适合抵达日，交通便利，不赶时间             │ │
│  │     💰 餐饮 ¥3,000                               │ │
│  │     ⏱️ 3 小时 | 强度：低                         │ │
│  │                                     [替换] [删除] │ │
│  └───────────────────────────────────────────────────┘ │
│                                                       │
│  ┌─── Day 2 ─── 市区文化 + 美食路线 ──────────────┐ │
│  │  ...                                              │ │
│  └───────────────────────────────────────────────────┘ │
│                                                       │
│  ⚠️ 风险提醒                                         │
│  Day 3 降雨概率 60%，已准备室内替代方案 [查看]       │
│                                                       │
│  💡 你可以说：                                        │
│  "第三天轻松一点" | "加个温泉" | "看看省钱版"       │
└─────────────────────────────────────────────────────┘
```

### 12.2 分层展示策略

```
Level 1：总体方案概览（默认展开）
  - 目的地、天数、预算、节奏
  - 每日主题一句话摘要
  - 预算使用进度条

Level 2：每日行程（点击展开）
  - 时间线视图
  - 每个活动的名称 + 推荐理由 + 费用
  - 交通方式和时间

Level 3：详细信息（点击具体活动展开）
  - 景点详细介绍
  - 开放时间
  - 地图位置
  - 用户评论摘要
  - 替代方案
```

### 12.3 交互式修改

```
支持的交互方式：
1. 自然语言修改："第三天太累了" → Agent 自动理解并调整
2. 直接操作：点击 [替换] 按钮 → 弹出推荐替代列表
3. 拖拽排序：拖拽活动卡片调整顺序（前端触发路线重算）
4. 方案对比：左右对比 A/B 方案
5. 地图联动：点击地图上的 POI 标记 → 加入行程
```

---

## 十三、完整用户流程设计

### 13.1 首次使用流程

```
用户打开应用
    │
    ▼
┌──────────────────────────┐
│ 欢迎语 + 引导提示         │
│ "说说你的旅行想法吧！    │
│  比如：去日本 7 天，      │
│  预算 1 万，喜欢美食"    │
└──────────┬───────────────┘
           ▼
用户输入自然语言需求
    │
    ▼
┌──────────────────────────┐
│ Orchestrator Agent        │
│ 1. 解析约束               │
│ 2. 检测信息缺口           │
│ 3. 缺口小 → 用默认值     │
│    缺口大 → 追问 1-2 题  │
└──────────┬───────────────┘
           ▼
    ┌──────┴──────┐
    │ 信息足够？   │
    ├── Yes ──────┤
    │             ▼
    │    ┌────────────────────┐
    │    │ 并行调用 Sub-Agents │
    │    │ + MCP Tools         │
    │    └────────┬───────────┘
    │             ▼
    │    ┌────────────────────┐
    │    │ 约束求解 + 行程生成 │
    │    └────────┬───────────┘
    │             ▼
    │    ┌────────────────────┐
    │    │ 评估校验            │
    │    │ score < 0.75?       │
    │    │ → 自动修正并重试    │
    │    └────────┬───────────┘
    │             ▼
    │    ┌────────────────────┐
    │    │ 输出行程 + 预算     │
    │    │ + 推荐理由 + 风险   │
    │    │ + 可调整选项        │
    │    └────────────────────┘
    │
    ├── No ───────┤
    │             ▼
    │    ┌────────────────────┐
    │    │ 追问最多 2 个问题   │
    │    │ 提供选项 减少输入   │
    │    └────────┬───────────┘
    │             ▼
    │         用户回答
    │             │
    └─────────────┘
```

### 13.2 多轮修改流程

```
用户发送修改请求
    │
    ▼
┌────────────────────────────────┐
│ Orchestrator 理解修改意图       │
│ - 修改目标（哪天？哪个活动？） │
│ - 修改方向（轻松？省钱？替换？）│
│ - 影响范围（局部 or 全局）     │
└──────────┬─────────────────────┘
           ▼
    ┌──────┴──────┐
    │ 局部修改？   │
    ├── Yes ──────┤
    │             ▼
    │    ┌────────────────────┐
    │    │ 只修改受影响的 Day  │
    │    │ 保留其他天不变      │
    │    │ 保存 before/after   │
    │    └────────┬───────────┘
    │             ▼
    │    ┌────────────────────┐
    │    │ 输出：              │
    │    │ - 修改了什么        │
    │    │ - 保留了什么        │
    │    │ - 为什么这样改      │
    │    │ - 是否影响其他天    │
    │    └────────────────────┘
    │
    ├── No（全局修改）──┤
    │                   ▼
    │    ┌────────────────────┐
    │    │ 重新运行完整规划    │
    │    │ 但保留已确认的部分  │
    │    └────────────────────┘
    └───────────────────────────
```

---

## 十四、技术栈选型与项目结构

### 14.1 技术栈

| 层级 | 技术选型 | 理由 |
|------|---------|------|
| LLM | Claude Sonnet 4 (主) / GPT-4o (备) | 工具调用能力强、长上下文、结构化输出稳定 |
| Agent 框架 | LangGraph / mcp-agent | 原生支持 Agent Loop + MCP，Temporal 持久化 |
| MCP SDK | @modelcontextprotocol/sdk (TS) / mcp (Python) | 官方 SDK，标准化工具接口 |
| 向量数据库 | Qdrant | 支持过滤、稀疏向量、快速部署 |
| BM25 索引 | Elasticsearch / Meilisearch | CJK 分词支持、全文检索 |
| Embedding | multilingual-e5-large | 中日英多语言支持 |
| Reranker | bge-reranker-v2-m3 | 多语言 Cross-Encoder |
| 状态存储 | Redis (Session) + PostgreSQL (持久) | 快速读写 + 持久化 |
| 路径优化 | Google OR-Tools / 自实现 2-opt | 工业级约束求解 |
| 地图 API | Google Maps Platform | 全球覆盖、距离矩阵、路线规划 |
| 天气 API | OpenWeatherMap | 免费额度充足、API 简洁 |
| 后端框架 | FastAPI | 异步支持好、WebSocket 原生支持 |
| 前端框架 | Next.js + React | SSR + 交互性 |
| 部署 | Docker + Kubernetes | 可伸缩 |
| 可观测性 | LangSmith / Langfuse | Agent 决策链路追踪 |

### 14.2 项目目录结构

```
travel-agent/
├── README.md
├── docker-compose.yml
├── pyproject.toml
│
├── src/
│   ├── agents/                          # Agent 层
│   │   ├── orchestrator.py              # Orchestrator Agent（核心入口）
│   │   ├── itinerary_planner.py         # 行程规划 Sub-Agent
│   │   ├── budget_optimizer.py          # 预算优化 Sub-Agent
│   │   ├── recommendation_engine.py     # 推荐引擎 Sub-Agent
│   │   ├── agent_loop.py               # Agent Loop 框架
│   │   └── prompts/                     # System Prompts
│   │       ├── orchestrator_prompt.py
│   │       ├── planner_prompt.py
│   │       ├── budget_prompt.py
│   │       └── recommendation_prompt.py
│   │
│   ├── mcp_servers/                     # MCP Server 层
│   │   ├── map_server.py                # 地图 MCP Server
│   │   ├── weather_server.py            # 天气 MCP Server
│   │   ├── poi_server.py                # POI MCP Server
│   │   ├── hotel_server.py              # 酒店 MCP Server
│   │   ├── transport_server.py          # 交通 MCP Server
│   │   ├── booking_server.py            # 预订 MCP Server
│   │   └── currency_server.py           # 汇率 MCP Server
│   │
│   ├── planner/                         # 规划算法层
│   │   ├── constraint_solver.py         # 约束求解器
│   │   ├── route_optimizer.py           # 路径优化（2-opt / OR-Tools）
│   │   ├── spatial_clustering.py        # 区域聚类（DBSCAN）
│   │   ├── day_assigner.py              # 日分配（Bin Packing）
│   │   ├── schedule_builder.py          # 时间表生成
│   │   └── constraint_checker.py        # 约束校验
│   │
│   ├── rag/                             # RAG 知识库层
│   │   ├── data_pipeline/
│   │   │   ├── loader.py                # 数据加载
│   │   │   ├── chunker.py               # 分块策略
│   │   │   ├── embedder.py              # 向量化
│   │   │   └── quality_manager.py       # 数据质量管理
│   │   ├── retriever.py                 # 混合检索器
│   │   ├── reranker.py                  # Cross-Encoder 重排
│   │   └── vector_store.py              # 向量数据库接口
│   │
│   ├── state/                           # 状态管理层
│   │   ├── state_manager.py             # 状态管理器
│   │   ├── schemas.py                   # 数据模型定义
│   │   └── session_store.py             # 会话存储（Redis）
│   │
│   ├── guardrails/                      # 安全与质量层
│   │   ├── input_guardrail.py           # 输入安全检查
│   │   ├── output_guardrail.py          # 输出质量检查
│   │   └── confidence_manager.py        # 信心阈值管理
│   │
│   ├── eval/                            # 评估层
│   │   ├── evaluator.py                 # 评估器
│   │   ├── metrics.py                   # 指标计算
│   │   ├── test_cases.py                # 测试用例
│   │   └── benchmark.py                 # 基准测试
│   │
│   ├── api/                             # API 层
│   │   ├── main.py                      # FastAPI 入口
│   │   ├── routes/
│   │   │   ├── chat.py                  # WebSocket 聊天接口
│   │   │   ├── itinerary.py             # REST 行程接口
│   │   │   └── export.py               # 行程导出接口
│   │   └── middleware/
│   │       ├── auth.py
│   │       └── rate_limit.py
│   │
│   └── config/
│       ├── settings.py                  # 配置管理
│       └── mcp_config.json              # MCP Server 配置
│
├── frontend/                            # 前端
│   ├── src/
│   │   ├── components/
│   │   │   ├── Chat/                    # 对话界面
│   │   │   ├── Itinerary/               # 行程展示
│   │   │   ├── Map/                     # 地图组件
│   │   │   └── Budget/                  # 预算展示
│   │   └── pages/
│   └── package.json
│
├── data/                                # 数据
│   ├── knowledge_base/                  # 知识库原始数据
│   │   ├── cities/                      # 城市攻略
│   │   ├── pois/                        # POI 数据
│   │   └── policies/                    # 签证/交通政策
│   └── eval/                            # 评估数据
│
├── scripts/                             # 脚本
│   ├── seed_knowledge_base.py           # 知识库初始化
│   ├── run_eval.py                      # 运行评估
│   └── benchmark.py                     # 性能基准测试
│
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/
```

---

## 十五、MVP 分阶段交付计划

### Phase 1: 核心可用（Week 1-2）

**目标**：单轮行程生成 + 1-2 个真实工具

**交付物**：
- Orchestrator Agent + Agent Loop 基础框架
- 单个 LLM 直接生成行程（ReAct 模式，尚无独立 Sub-Agent）
- 天气 MCP Server（接入 OpenWeatherMap）
- 地图 MCP Server（接入 Google Maps，仅距离计算）
- 简单的 State Manager（内存存储）
- 基础前端对话界面
- 结构化行程输出

**验收标准**：
- 输入 "去大阪 5 天" 能输出结构化行程
- 行程包含真实天气数据
- 景点之间的交通时间基于真实距离

### Phase 2: 多轮 + 算法（Week 3-4）

**目标**：多轮对话 + 约束求解 + RAG

**交付物**：
- 多轮对话支持（State 持久化到 Redis）
- 局部修改能力（只改指定天，不全量重写）
- DBSCAN 区域聚类 + 2-opt 路径优化
- 预算估算与校验
- 基础 RAG（本地 POI 知识库 + 向量检索）
- 评估模块（至少 4 个核心指标）

**验收标准**：
- "第三天太累了" 能只修改 Day 3
- 路线优化后总交通时间明显减少
- 预算不超出用户限制
- 评估分数 ≥ 0.75

### Phase 3: Sub-Agent + 完善（Week 5-6）

**目标**：Multi-Agent 架构 + 工具完善 + 前端完善

**交付物**：
- 拆分独立 Sub-Agent（Planner / Budget / Recommendation）
- POI MCP Server（接入 Google Places）
- 酒店 MCP Server（价格区间查询）
- 方案对比功能（省钱版 / 舒适版）
- Guardrail 模块（输入/输出安全检查）
- 前端行程可视化（时间线 + 地图）
- 完整评估测试套件

**验收标准**：
- Sub-Agent 间能正确协作
- 生成的行程通过所有约束校验
- 支持方案对比
- 6 个测试用例全部通过

### Phase 4: 高级功能（Week 7-8+）

**交付物**：
- OR-Tools 精确路径优化
- 实时位置推荐（附近搜索）
- 天气自适应（自动替换户外活动）
- 行程导出（iCal / PDF）
- 出行辅助信息（签证/汇率/交通卡）
- 可观测性（LangSmith 决策链路追踪）
- A2A 协议预留

---

## 十六、竞品分析与差异化

### 16.1 竞品对比

| 维度 | Expedia AI | Booking.com AI | Trip.com AI | 本系统 |
|------|-----------|---------------|------------|--------|
| 行程生成 | ✅ 基础 | ✅ 基础 | ✅ 基础 | ✅ 多约束求解 |
| 多轮修改 | ❌ | 部分 | 部分 | ✅ 局部重规划 |
| 路线优化 | ❌ | ❌ | ❌ | ✅ TSP/2-opt |
| 实时数据 | ✅ 自有平台 | ✅ 自有平台 | ✅ 自有平台 | ✅ MCP 开放接入 |
| 预算优化 | 部分 | 部分 | 部分 | ✅ 多策略对比 |
| 可解释性 | ❌ | ❌ | ❌ | ✅ 每个推荐有理由 |
| 评估体系 | 不公开 | 不公开 | 不公开 | ✅ 8 维度量化评估 |
| 开放架构 | ❌ 封闭 | ❌ 封闭 | ❌ 封闭 | ✅ MCP 标准化 |

### 16.2 差异化策略

1. **算法深度**：竞品主要依赖 LLM 直接生成，本系统将规划问题建模为约束满足问题，用算法求解。
2. **开放架构**：通过 MCP 标准化工具接口，可以灵活接入任何数据源，不依赖单一平台。
3. **可解释性**：每个推荐都有明确的数据来源和推荐理由，可追溯可验证。
4. **评估驱动**：有量化评估体系，可以持续度量和改进行程质量。
5. **局部重规划**：修改行程时只影响变更部分，不破坏用户已确认的内容。

---

## 十七、风险与缓解策略

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|---------|
| LLM 幻觉（推荐不存在的景点） | 高 | 高 | 所有 POI 通过 MCP Tool 或 RAG 验证，`source` 字段追踪来源 |
| 实时数据不准确 | 中 | 中 | 数据加时间戳，超期自动刷新，输出中标注数据时效性 |
| 多轮对话状态丢失 | 中 | 高 | Redis 持久化 + PostgreSQL 备份，每轮保存完整 State 快照 |
| API 调用失败（地图/天气） | 中 | 中 | 重试机制 + 降级策略（使用 RAG 缓存数据替代） |
| 路径优化算法性能 | 低 | 中 | OR-Tools 设置 10s 超时，降级到贪心算法 |
| Token 消耗过高 | 中 | 中 | Sub-Agent 使用较小模型，仅 Orchestrator 使用强模型 |
| 用户恶意输入 | 低 | 中 | Input Guardrail + Prompt Injection 检测 |
| 数据合规风险（爬取用户评论） | 中 | 高 | 仅使用公开 API 数据，评论使用摘要而非原文，标注数据来源 |

---

## 十八、附录

### 附录 A：Prompt 模板示例

#### Orchestrator System Prompt

```
你是一个专业的旅行规划助手的协调器。你的职责是：

1. 理解用户的旅行需求，从自然语言中提取结构化约束
2. 调用合适的工具和子系统来获取信息、规划行程
3. 确保行程满足所有约束条件
4. 用自然、友好的语言与用户交流

## 你可以使用的工具

[由 MCP 动态注入工具列表]

## 你可以委派的子任务

- recommend_pois: 根据偏好推荐景点
- plan_itinerary: 生成/修改行程
- optimize_budget: 优化预算
- evaluate_itinerary: 评估行程质量

## 核心原则

- 信息不足时，先用合理默认值生成草案，然后在末尾询问关键信息
- 修改行程时，只改受影响部分，明确说明变更内容和影响
- 每个推荐都附带理由
- 预算不超出用户限制
- 路线不绕路
- 节奏符合用户要求
- 景点在安排时间内开放

## 当前状态
{state_json}

## 回复格式
用自然语言回复，行程部分使用结构化格式。
不要暴露内部技术细节（如 Agent 名称、MCP Server 名称）。
```

### 附录 B：API 接口定义

```yaml
# WebSocket Chat API
ws://api.travel-agent.com/ws/chat

# 连接后的消息格式
# Client → Server
{
  "type": "user_message",
  "content": "我想去大阪玩 5 天",
  "session_id": "abc123"
}

# Server → Client (流式)
{
  "type": "assistant_message",
  "content": "好的，我来帮你规划...",  // 逐字流式输出
  "metadata": {
    "tools_used": ["weather_mcp", "poi_mcp"],
    "confidence": 0.85,
    "eval_score": 0.82
  }
}

# Server → Client (行程结构化数据)
{
  "type": "itinerary_update",
  "data": {
    // ItineraryResponse schema
  }
}

# REST API
GET  /api/v1/itinerary/{session_id}        # 获取当前行程
POST /api/v1/itinerary/{session_id}/export  # 导出行程（ical/pdf）
GET  /api/v1/itinerary/{session_id}/eval    # 获取评估结果
```

### 附录 C：关键设计决策记录 (ADR)

**ADR-001: Agent Loop vs Pipeline**
- 决策：采用 Agent Loop（ReAct 模式）替代线性 Pipeline
- 原因：Pipeline 无法处理动态场景，每增加一个意图需要修改路由逻辑
- 权衡：Agent Loop 的行为更难预测，需要更强的 Guardrail 和评估

**ADR-002: MCP vs 自定义工具接口**
- 决策：采用 MCP 标准化工具接口
- 原因：行业标准，跨框架复用，社区生态支持
- 权衡：增加了协议层开销，但带来了可扩展性和互操作性

**ADR-003: Multi-Agent vs 单体 Agent**
- 决策：Orchestrator + 多个 Sub-Agent
- 原因：职责分离，可独立迭代，不同 Agent 可使用不同大小的模型优化成本
- 权衡：增加了系统复杂度和通信开销

**ADR-004: 2-opt vs OR-Tools**
- 决策：MVP 使用 2-opt，高级版本引入 OR-Tools
- 原因：2-opt 实现简单，对 n < 10 的场景够用；OR-Tools 可处理更复杂约束
- 权衡：2-opt 不保证全局最优，但对旅行规划场景影响有限

**ADR-005: 局部修改 vs 全量重生成**
- 决策：默认局部修改，仅在用户明确要求或多天受影响时全量重生成
- 原因：用户心智模型期望"只改我说的那天"，全量重生成会破坏已确认的计划
- 权衡：局部修改可能导致全局次优，但用户体验更好

---

**文档结束**

> 本文档旨在作为完整的技术规格说明提交给 Codex 或自动化开发 Agent，包含足够的细节来指导代码实现。所有代码片段均为伪代码/示例代码，具体实现时需根据实际框架和 API 进行调整。
