# 智能旅行 Agent 系统设计文档

> **文档版本**：v4.0
> **最后更新**：2026-04-27
> **定位**：面向中国境内自由行用户的 LLM 驱动多 Agent 行程规划系统。以 LLM 为决策大脑，代码层提供结构化执行与校验能力，支持持续对话、偏好渐进式收集、实时信息查询与动态行程调整。

---

## 目录

1. [关键设计决策与约束](#一关键设计决策与约束)
2. [产品定位与用户画像](#二产品定位与用户画像)
3. [核心场景与 MVP 边界](#三核心场景与-mvp-边界)
4. [系统总体架构：LLM 主导 + 代码辅助](#四系统总体架构llm-主导--代码辅助)
5. [Agent 体系与编排](#五agent-体系与编排)
6. [Agent 间通信协议与 Skill 定义](#六agent-间通信协议与-skill-定义)
7. [记忆系统设计](#七记忆系统设计)
8. [模块详细设计](#八模块详细设计)
9. [数据层设计：网络爬虫为主](#九数据层设计网络爬虫为主)
10. [约束求解与路径优化](#十约束求解与路径优化)
11. [RAG 知识库系统](#十一rag-知识库系统)
12. [评估体系设计](#十二评估体系设计)
13. [前端交互设计：三栏布局](#十三前端交互设计三栏布局)
14. [完整用户流程设计](#十四完整用户流程设计)
15. [技术栈选型与项目结构](#十五技术栈选型与项目结构)
16. [MVP 分阶段交付计划](#十六mvp-分阶段交付计划)
17. [风险与缓解策略](#十七风险与缓解策略)

---

## 一、关键设计决策与约束

### 1.1 场景约束（已确认）

| 约束项 | 决策 | 理由 |
|--------|------|------|
| 旅行范围 | 中国境内城市 | 团队熟悉度高，数据源可控，无需汇率/签证逻辑 |
| 城市数量 | MVP 仅支持单城市 | 聚焦核心能力，跨城交通后续扩展 |
| 数据源 | 网络爬虫 + 搜索引擎 + 公开网页 | 不依赖付费 API，快速启动，覆盖更广 |
| 前端形态 | Web 端（Next.js），响应式适配移动端 | 一套代码覆盖多端 |
| 开发团队 | 1-2 人，全栈 | 人力约束直接决定 MVP 范围 |

### 1.2 架构决策（v4 核心变更）

| 决策项 | v3 设计 | **v4 设计** | 变更理由 |
|--------|---------|-------------|---------|
| 控制架构 | 双层：确定性状态机 + LLM | **LLM 主导意图识别 + 代码辅助执行** | 状态机无法应对用户表达多样性，LLM 意图识别更准确、更灵活 |
| 意图识别 | 状态机固定规则匹配 | **LLM 做意图识别，提取关键信息，决定流程分支** | 用户表达多样，固定规则覆盖不全 |
| 信息追问 | 状态机缺失字段判断 | **LLM 判断关键信息是否缺失，自然语言追问** | 追问更自然，可结合上下文推断 |
| 偏好管理 | 代码硬编码约束 | **LLM 判断偏好更新，动态覆盖旧偏好** | 用户在对话中随时修改偏好，代码无法预判 |
| Agent 拆分 | 3 个 Sub-Agent（串行为主） | **8+ 个 Agent，能并行尽量并行** | 提升性能，职责更清晰 |
| 数据源 | API 为主 | **网络爬虫 + 搜索为主，API 为辅** | 降低依赖，扩大覆盖 |
| 循环终止 | 确定性条件检查 | **LLM 判断对话是否完成 + 代码校验输出完整性** | 对话型 Agent 需要理解语境 |
| 记忆管理 | 简单 Session State | **结构化记忆系统：对话记录 + 历史行程 + 偏好 + 预算** | 长期记忆支撑个性化 |

### 1.3 关键信息定义（LLM 判断缺失时追问）

以下信息被定义为"关键信息"，LLM 在意图识别后检查是否齐全，缺失则自然追问：

```
必需关键信息（缺一则无法生成行程）：
  - destination: 目的地城市
  - travel_days: 旅行天数
  - travel_dates: 旅行日期（或"下周"、"下个月"等相对时间）

重要信息（缺失时 LLM 追问，用户可跳过）：
  - travelers_count: 出行人数
  - budget_range: 预算范围（或"没预算"）
  - travelers_type: 同行人类型（独自/情侣/亲子/父母/朋友）

偏好信息（LLM 从对话中主动提取，不主动追问）：
  - food_preferences: 饮食偏好（如：吃辣、清淡、海鲜）
  - pace: 节奏（紧凑/适中/轻松）
  - interests: 兴趣标签（如：历史、自然、拍照、美食、购物）
  - accommodation_preference: 住宿偏好（如：市中心、安静、性价比）
  - special_requests: 特殊要求（如：必去景点、避开人流）
```

**追问原则**：
1. 一次最多追问 2 个问题
2. 用自然语言追问，不带技术术语
3. 提供选项减少输入成本（如："预算大概多少？2000/5000/8000，或者告诉我范围"）
4. 用户说"不知道"或"没想好"时，LLM 用默认值并告知用户

---

## 二、产品定位与用户画像

### 2.1 一句话定位

**旅行 Agent 是一个以 LLM 为大脑、多 Agent 协作的智能行程规划系统。它像一位经验丰富的旅行顾问：通过自然语言对话理解你的需求，在线搜索实时信息，结合你的历史偏好，生成可执行的个性化行程。**

核心差异点：
- **LLM 是大脑，不是文本生成器**：意图识别、偏好理解、行程评价都由 LLM 主导
- **持续对话，渐进式理解**：每一轮对话都在丰富用户画像，不是一次性收集
- **实时信息在线查询**：通过网络爬虫获取最新景点、价格、天气信息
- **记忆驱动个性化**：记住你的每一次行程和偏好，越用越懂你

### 2.2 目标用户画像

#### 用户类型 A：自由行新手
**特征**：第一次去某个城市，不知道去哪、怎么安排。
**Agent 价值**：自然对话输入需求 → Agent 主动追问 → 输出完整行程。

#### 用户类型 B：懒人型用户
**特征**：有旅行经验但不想花时间做攻略。
**典型输入**：`"我想去成都玩 4 天，预算 3000，喜欢吃辣，别太赶。"`
**Agent 价值**：一句话说完需求，Agent 自动完成后续所有步骤。

#### 用户类型 C：精细规划用户
**特征**：需求多、约束复杂。
**典型输入**：`"我想去北京 5 天，第一天中午到，第三天想去长城，酒店住二环内，预算 4000。"`
**Agent 价值**：理解复杂约束，生成精确到时间的行程。

#### 用户类型 D：老用户（有历史行程）
**特征**：使用过 Agent，有历史偏好。
**Agent 价值**：自动加载历史偏好，"还是按照之前的风格来"即可生成。

---

## 三、核心场景与 MVP 边界

### 3.1 核心场景（MVP 内）

| 场景 | 优先级 | 说明 |
|------|--------|------|
| 从零生成单城市行程 | P0 | 核心能力，LLM 主导意图识别 → 信息收集 → 行程生成 |
| 多轮修改行程 | P0 | 支持自然语言修改，LLM 判断修改意图和影响范围 |
| 实时信息查询（天气/景点/价格） | P0 | 网络爬虫获取，展示在行程中 |
| 预算估算 | P0 | LLM 结合爬虫数据估算，右侧面板实时展示 |
| 偏好渐进式收集 | P0 | 对话中持续收集，右侧面板实时更新 |
| 历史行程回顾 | P1 | 左侧面板"历史行程"入口 |
| 行程确认与固化 | P1 | 用户确认后保存为"当前行程" |

### 3.2 v4 场景设计：从零生成行程

**用户输入**：`"我想去成都玩 4 天，预算 3000，喜欢吃辣和拍照，不想太累。"`

**处理流程**：

```
1. 意图识别 Agent（LLM 主导）
   - 意图：generate_itinerary（生成行程）
   - 关键信息提取：
     * destination: 成都 ✓
     * travel_days: 4 ✓
     * budget: 3000 ✓
     * food_preferences: [辣] ✓
     * interests: [拍照] ✓
     * pace: relaxed（推断自"不想太累"）✓
   - 关键信息完整度：100%（destination + travel_days 齐全）
   - 决定：进入行程生成流程

2. 并行 Agent 调用（能并行尽量并行）
   ├─ 实时信息查询 Agent ──▶ 爬虫搜索：成都热门景点、美食、拍照打卡点
   ├─ 实时信息查询 Agent ──▶ 爬虫搜索：成都未来 4 天天气预报
   ├─ 偏好管理 Agent ──▶ 更新用户偏好（吃辣、拍照、轻松）到右侧面板
   └─ 预算管理 Agent ──▶ 初始化预算面板：总预算 3000，待分配
   
   全部完成后 → 继续

3. 行程规划 Agent
   - 输入：候选 POI 列表 + 用户偏好 + 约束条件
   - 区域聚类 → 日分配 → 日内排序 → 时间表生成
   - 输出：每日行程草案

4. 并行校验（Agent 并行执行）
   ├─ 预算管理 Agent ──▶ 分项估算，更新右侧面板
   ├─ 偏好覆盖检查 Agent ──▶ 检查偏好是否被覆盖
   └─ 约束校验（代码层）──▶ 硬约束检查
   
   全部完成后 → 继续

5. 方案生成 Agent
   - 将行程转化为自然语言描述
   - 生成推荐理由、预算明细、风险提示
   - 输出：带"确认行程"按钮的方案

6. 用户确认 → 保存为"当前行程"
```

### 3.3 v4 场景设计：对话中更新偏好

**用户输入**：`"对了，我其实也爱吃甜品，帮我加几个甜品店"`

**处理流程**：

```
1. 意图识别 Agent
   - 意图：update_preferences（更新偏好）
   - 检测到新增偏好：food_preferences += [甜品]
   - 判断：非关键信息更新，无需追问

2. 偏好管理 Agent
   - 用新偏好覆盖/追加旧偏好
   - 更新右侧面板：饮食偏好 = [辣, 甜品]
   - 触发：偏好变更事件

3. 行程重规划 Agent（触发更新）
   - 接收到偏好变更事件
   - 判断影响范围：饮食相关（全局，影响每天用餐安排）
   - 重新搜索成都甜品店 → 插入到每日行程中
   - 更新预算面板（甜品增加预算）

4. 输出新方案
   - "已为你更新偏好：新增甜品爱好。重新规划后，每天下午加入了当地口碑甜品店。"
   - 展示更新后的行程和预算
   - 提供"确认行程"按钮
```

---

## 四、系统总体架构：LLM 主导 + 代码辅助

### 4.1 架构核心：LLM 是大脑

```
┌──────────────────────────────────────────────────────────────────────┐
│                          用户输入（自然语言）                           │
└──────────────────────────────────┬───────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│              第一层：LLM 意图识别 Agent（核心大脑）                      │
│                                                                       │
│  职责：                                                               │
│  - 理解用户自然语言意图（生成行程 / 修改行程 / 更新偏好 / 问答 等）      │
│  - 提取关键信息（目的地、天数、人数、预算、偏好等）                      │
│  - 判断关键信息是否缺失，自然语言追问                                  │
│  - 决定进入哪个流程分支                                               │
│  - 判断用户是否更新了偏好，触发偏好变更事件                            │
│                                                                       │
│  输出：IntentResult（意图类型 + 提取的实体 + 缺失字段 + 置信度）        │
└──────────────────────────────────┬───────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│              第二层：Orchestrator（轻量编排，代码实现）                   │
│                                                                       │
│  职责：                                                               │
│  - 根据意图类型，调度对应的 Agent 组合                                  │
│  - 管理 Agent 间的并行/串行执行                                       │
│  - 聚合各 Agent 结果                                                  │
│  - 不替代 LLM 决策，只做流程编排                                      │
└──────────────────────────────────┬───────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│              Agent Pool（8+ 个 Agent，能并行尽量并行）                   │
│                                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ 意图识别    │  │ 信息收集    │  │ 实时查询    │  │ 偏好&预算   │ │
│  │ Agent       │  │ Agent       │  │ Agent       │  │ 管理 Agent  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ 行程规划    │  │ 方案生成    │  │ 问答 Agent  │  │ 记忆管理    │ │
│  │ Agent       │  │ Agent       │  │             │  │ Agent       │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
│  ┌─────────────┐  ┌─────────────┐                                    │
│  │ 校验 Agent  │  │ 地图&路线   │                                    │
│  │             │  │ Agent       │                                    │
│  └─────────────┘  └─────────────┘                                    │
└──────────────────────────────────┬───────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│              Tool Layer（Skill 实现）                                   │
│                                                                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │
│  │ Web Search  │ │ Web Crawler │ │ Map Tool    │ │ Weather     │    │
│  │ Skill       │ │ Skill       │ │ Skill       │ │ Query Skill │    │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                    │
│  │ POI Search  │ │ Price Query │ │ Route Calc  │                    │
│  │ Skill       │ │ Skill       │ │ Skill       │                    │
│  └─────────────┘ └─────────────┘ └─────────────┘                    │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 架构设计原则

```
原则 1：LLM 做判断，代码做执行
  - LLM 判断：用户意图、偏好是否更新、信息是否完整、行程是否合理
  - 代码执行：约束校验、路径计算、数据聚合、状态持久化

原则 2：能并行尽量并行
  - 无依赖的 Agent 同时启动
  - 示例：POI 搜索、天气查询、预算初始化可以并行

原则 3：LLM 不直接操作数据，通过 Agent 接口
  - LLM 输出意图和参数 → Orchestrator 调用 Agent → Agent 操作数据
  - 避免 LLM 直接修改数据库带来的不可控

原则 4：记忆是核心资产
  - 每条对话都记录
  - 每次行程都保存
  - 每次偏好变更都追踪
  - 历史数据驱动个性化

原则 5：自然语言是主要交互方式
  - 用户用自然语言表达需求
  - Agent 用自然语言回复
  - 右侧面板是信息的可视化，不是必填表单
```

### 4.3 循环终止条件

```python
class AgentLoopTerminator:
    """
    循环终止：LLM 判断 + 代码校验结合。
    对话型 Agent 的终止需要理解语境，不能完全靠确定性代码。
    """

    def should_stop(self, state: TravelState, llm_decision: LLMDecision) -> bool:
        # 条件 1（LLM 判断）：LLM 认为本轮对话已完成
        llm_thinks_complete = llm_decision.is_conversation_complete
        
        # 条件 2（代码校验）：没有待执行的工具/Agent 调用
        no_pending_calls = len(state.pending_agent_calls) == 0
        
        # 条件 3（代码校验）：输出草稿已准备
        response_ready = state.response_draft is not None
        
        # 条件 4（代码校验）：非阻塞状态（不在追问中）
        not_waiting_input = state.awaiting_user_input == False
        
        # 终止条件：LLM 认为完成 且 代码校验通过
        return llm_thinks_complete and no_pending_calls and response_ready and not_waiting_input
```

---

## 五、Agent 体系与编排

### 5.1 Agent 总览

| Agent | 职责 | 输入 | 输出 | 并行性 |
|-------|------|------|------|--------|
| **Intent Recognition Agent** | 理解用户意图，提取实体，判断信息完整性 | 用户原始输入 + 对话历史 + 当前 State | IntentResult（意图 + 实体 + 缺失字段） | 串行（第一个执行） |
| **Information Collection Agent** | 收集和补全用户信息 | 缺失字段列表 + 用户回答 | UserProfile（完整用户信息） | 串行 |
| **Real-time Query Agent** | 查询实时信息（天气、景点、价格、线路） | 查询参数（城市、日期、关键词） | 实时数据（POI 列表 / 天气 / 价格） | **可并行** |
| **Preference & Budget Agent** | 管理用户偏好和预算，检测偏好变更 | 用户输入 / 偏好更新指令 | 更新后的偏好 + 预算面板数据 | 串行 |
| **Itinerary Planner Agent** | 行程规划算法执行 | 候选 POI + 约束条件 + 偏好 | 行程草案（每日活动列表） | 串行（依赖 Real-time Agent） |
| **Q&A Agent** | 回答用户关于景点、城市、交通的问题 | 用户问题 + RAG 知识库 | 自然语言回答 | **可并行** |
| **Memory Management Agent** | 管理对话记录、历史行程、偏好历史 | 新对话 / 新行程 / 偏好变更 | 持久化到数据库 | 后台异步 |
| **Proposal Generation Agent** | 将行程转化为自然语言方案 | 行程数据 + 推荐理由 + 预算 | 带格式的方案文本 | 串行 |
| **Validation Agent** | 约束校验、POI 真实性检查 | 行程草案 + 约束条件 | 校验结果 + 修改建议 | **可并行** |
| **Map & Route Agent** | 地理计算、路线优化 | POI 坐标列表 + 交通模式 | 最优路线 + 交通时间 | **可并行** |

### 5.2 Agent 编排流程

#### 流程 A：从零生成行程

```
用户输入
    │
    ▼
┌────────────────────────────────┐
│ 1. Intent Recognition Agent    │
│    - 识别意图：generate        │
│    - 提取信息                  │
│    - 检查关键信息完整性        │
│    - 缺失？→ 生成追问 → 结束   │
└────────────┬───────────────────┘
             │ 信息完整
             ▼
┌────────────────────────────────┐
│ 2. 并行启动（Step 1）          │
│    ├─ Real-time Query Agent    │
│    │  ├─ 搜索景点信息           │
│    │  ├─ 搜索天气预报           │
│    │  └─ 搜索价格信息           │
│    ├─ Preference & Budget Agent│
│    │  └─ 初始化偏好+预算面板    │
│    └─ Memory Management Agent  │
│       └─ 记录对话（后台）        │
└────────────┬───────────────────┘
             │ 全部完成
             ▼
┌────────────────────────────────┐
│ 3. Itinerary Planner Agent     │
│    - 区域聚类                   │
│    - 日分配                     │
│    - 路线优化                   │
│    - 时间表生成                 │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│ 4. 并行校验（Step 2）          │
│    ├─ Validation Agent         │
│    │  └─ 约束校验               │
│    ├─ Budget Agent             │
│    │  └─ 分项估算，更新面板     │
│    └─ Map & Route Agent        │
│       └─ 路线精确计算           │
└────────────┬───────────────────┘
             │ 全部完成
             ▼
┌────────────────────────────────┐
│ 5. Proposal Generation Agent   │
│    - 生成自然语言方案           │
│    - 生成推荐理由               │
│    - 生成预算明细               │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│ 6. 输出给用户                   │
│    - 自然语言描述行程           │
│    - 中间面板：行程时间线        │
│    - 右侧面板：预算 + 偏好       │
│    - "确认行程"按钮             │
└────────────────────────────────┘
```

#### 流程 B：对话中更新偏好

```
用户输入："对了，帮我加几个甜品店"
    │
    ▼
┌────────────────────────────────┐
│ 1. Intent Recognition Agent    │
│    - 意图：update_preferences  │
│    - 新增偏好：food += 甜品     │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│ 2. Preference & Budget Agent   │
│    - 追加偏好：food_preferences│
│    - 更新右侧面板              │
│    - 发布"PreferenceChanged"事件│
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│ 3. 触发重规划（监听事件）       │
│    ├─ Real-time Query Agent    │
│    │  └─ 搜索甜品店             │
│    ├─ Itinerary Planner Agent  │
│    │  └─ 重新规划（增量更新）    │
│    └─ Budget Agent             │
│       └─ 更新预算面板           │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│ 4. Proposal Generation Agent   │
│    - 生成更新后的方案           │
│    - 高亮变更部分               │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│ 5. 输出给用户                   │
│    - "已为你新增甜品偏好，重新   │
│       规划后每天下午加入了甜品店"│
│    - 展示更新后的行程            │
│    - "确认行程"按钮             │
└────────────────────────────────┘
```

#### 流程 C：问答模式

```
用户输入："成都有什么必吃的美食？"
    │
    ▼
┌────────────────────────────────┐
│ 1. Intent Recognition Agent    │
│    - 意图：q_a                 │
│    - 不是行程相关操作           │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│ 2. 并行启动                     │
│    ├─ Q&A Agent                │
│    │  └─ 检索 RAG + 网络搜索    │
│    │     生成回答               │
│    └─ Memory Management Agent  │
│       └─ 记录对话（后台）        │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│ 3. 输出给用户                   │
│    - 自然语言回答               │
│    - 不触发行程更新             │
└────────────────────────────────┘
```

### 5.3 Agent 详细设计

#### Agent 1：Intent Recognition Agent（意图识别 Agent）

```python
INTENT_RECOGNITION_SYSTEM_PROMPT = """
你是一个旅行规划系统的意图识别专家。你的职责是分析用户的输入，判断用户的意图，并提取关键信息。

## 可识别的意图类型
1. generate_itinerary - 生成新行程（用户表达出行意愿）
2. modify_itinerary - 修改已有行程（调整某天、替换景点等）
3. update_preferences - 更新偏好（新增兴趣、调整预算等）
4. query_info - 查询信息（问天气、问景点、问价格等）
5. confirm_itinerary - 确认行程（用户说"就这样吧"、"确定了"等）
6. view_history - 查看历史行程
7. chitchat - 闲聊

## 关键信息定义（提取到 user_entities 中）
- destination: 目的地城市（必需）
- travel_days: 旅行天数（必需）
- travel_dates: 旅行日期（必需，可以是"下周"、"五一"等）
- travelers_count: 出行人数
- budget_range: 预算范围
- travelers_type: 同行人类型（独自/情侣/亲子/朋友/父母）
- food_preferences: 饮食偏好列表
- interests: 兴趣标签列表
- pace: 节奏（relaxed/moderate/intensive）
- special_requests: 特殊要求

## 判断规则
1. 如果缺少 destination 或 travel_days 或 travel_dates，必须标记为缺失
2. 判断用户是否更新了偏好（与当前 state 中的偏好对比）
3. 如果用户表达模糊（如"想去个暖和的地方"），标注为 fuzzy_destination
4. 置信度 < 0.7 时，标注需要确认

## 输出格式（严格 JSON）
{
  "intent": "generate_itinerary",
  "confidence": 0.95,
  "user_entities": {
    "destination": "成都",
    "travel_days": 4,
    "travel_dates": "2026-05-01",
    "budget_range": 3000,
    "food_preferences": ["辣"],
    "interests": ["拍照", "美食"],
    "pace": "relaxed"
  },
  "missing_required": [],
  "missing_recommended": ["travelers_count"],
  "preference_changes": [],
  "is_fuzzy": false,
  "clarification_questions": [],
  "reasoning": "用户明确表达了去成都玩4天的意愿，提供了预算和偏好"
}
"""

class IntentRecognitionAgent:
    """
    意图识别 Agent：LLM 驱动，是系统的"大脑入口"。
    """
    
    async def recognize(
        self,
        user_input: str,
        conversation_history: list[Message],
        current_state: TravelState
    ) -> IntentResult:
        """
        识别用户意图。
        
        流程：
        1. 构建 Prompt（系统提示 + 当前 State + 对话历史 + 用户输入）
        2. 调用 LLM，要求结构化输出
        3. 解析输出，与当前 State 对比，检测偏好变更
        4. 返回 IntentResult
        """
        prompt = self._build_prompt(user_input, conversation_history, current_state)
        response = await self.llm.structured_call(
            prompt=prompt,
            schema=IntentResultSchema,
            temperature=0.3  # 意图识别要低温度，稳定输出
        )
        
        # 后处理：与当前偏好对比，检测变更
        result = IntentResult(**response)
        result.preference_changes = self._detect_preference_changes(
            result.user_entities, current_state.user_profile
        )
        
        return result
    
    def _detect_preference_changes(
        self, new_entities: dict, current_profile: UserProfile
    ) -> list[PreferenceChange]:
        """
        检测用户是否更新了偏好。
        对比新提取的实体与当前保存的偏好，找出差异。
        """
        changes = []
        
        # 对比 food_preferences
        new_food = set(new_entities.get("food_preferences", []))
        old_food = set(current_profile.food_preferences or [])
        added = new_food - old_food
        removed = old_food - new_food
        if added or removed:
            changes.append(PreferenceChange(
                field="food_preferences",
                old_value=list(old_food),
                new_value=list(new_food),
                change_type="update"
            ))
        
        # 对比 interests
        new_interests = set(new_entities.get("interests", []))
        old_interests = set(current_profile.interests or [])
        if new_interests != old_interests:
            changes.append(PreferenceChange(
                field="interests",
                old_value=list(old_interests),
                new_value=list(new_interests),
                change_type="update"
            ))
        
        # 对比 budget
        new_budget = new_entities.get("budget_range")
        old_budget = current_profile.budget_range
        if new_budget is not None and new_budget != old_budget:
            changes.append(PreferenceChange(
                field="budget_range",
                old_value=old_budget,
                new_value=new_budget,
                change_type="update"
            ))
        
        # 对比 pace
        new_pace = new_entities.get("pace")
        old_pace = current_profile.pace
        if new_pace is not None and new_pace != old_pace:
            changes.append(PreferenceChange(
                field="pace",
                old_value=old_pace,
                new_value=new_pace,
                change_type="update"
            ))
        
        return changes
```

#### Agent 2：Real-time Query Agent（实时信息查询 Agent）

```python
class RealtimeQueryAgent:
    """
    实时信息查询 Agent：通过网络爬虫和搜索获取最新信息。
    内部可并行发起多个查询。
    """
    
    async def query_pois(
        self,
        city: str,
        interests: list[str],
        food_preferences: list[str] | None = None
    ) -> list[ScoredPOI]:
        """
        查询城市 POI 信息。
        
        策略：
        1. 同时发起多个网络搜索（景点、美食、特色体验）
        2. 爬虫抓取详细页面获取评分、价格、开放时间
        3. 去重 + 评分排序
        4. 返回 Top N 候选 POI
        """
        # 并行搜索
        tasks = [
            self.web_search(f"{city} 必去景点 推荐 攻略 2026"),
            self.web_search(f"{city} 美食推荐 {','.join(food_preferences or [])}"),
            self.web_search(f"{city} 拍照打卡 网红景点"),
        ]
        
        search_results = await asyncio.gather(*tasks)
        
        # 并行爬取详情页
        poi_candidates = []
        for result in search_results:
            crawl_tasks = [self.web_crawler(url) for url in result.urls[:10]]
            pages = await asyncio.gather(*crawl_tasks)
            poi_candidates.extend(self._extract_pois_from_pages(pages))
        
        # 去重 + 评分
        return self._deduplicate_and_score(poi_candidates, interests)
    
    async def query_weather(self, city: str, dates: list[str]) -> list[WeatherDay]:
        """
        查询天气预报。
        数据来源：搜索引擎查询天气预报网站
        """
        search_query = f"{city} 天气预报 {dates[0]} {dates[-1]}"
        result = await self.web_search(search_query)
        
        # 抓取天气详情
        weather_data = await self.web_crawler(result.top_url)
        return self._parse_weather(weather_data, dates)
    
    async def query_prices(
        self,
        city: str,
        poi_names: list[str]
    ) -> dict[str, PriceInfo]:
        """
        查询景点门票价格。
        并行查询每个 POI 的价格信息。
        """
        tasks = [
            self.web_search(f"{city} {poi_name} 门票价格 2026")
            for poi_name in poi_names
        ]
        results = await asyncio.gather(*tasks)
        
        prices = {}
        for poi_name, result in zip(poi_names, results):
            prices[poi_name] = self._extract_price(result)
        
        return prices
```

#### Agent 3：Preference & Budget Agent（偏好与预算管理 Agent）

```python
class PreferenceBudgetAgent:
    """
    偏好与预算管理 Agent：
    - 管理用户偏好（动态更新）
    - 管理预算（分项估算、实时面板）
    - 检测偏好变更事件，触发重规划
    """
    
    async def update_preferences(
        self,
        session_id: str,
        changes: list[PreferenceChange]
    ) -> UserProfile:
        """
        更新用户偏好。用新偏好覆盖旧偏好。
        同时触发事件通知其他 Agent。
        """
        state = await self.state_manager.load(session_id)
        profile = state.user_profile
        
        for change in changes:
            # 用新值覆盖旧值
            setattr(profile, change.field, change.new_value)
            
            # 记录变更历史
            profile.preference_history.append({
                "timestamp": datetime.now().isoformat(),
                "field": change.field,
                "old": change.old_value,
                "new": change.new_value
            })
        
        await self.state_manager.save(state)
        
        # 发布偏好变更事件
        await self.event_bus.publish("PreferenceChanged", {
            "session_id": session_id,
            "changes": changes
        })
        
        return profile
    
    async def calculate_budget_breakdown(
        self,
        itinerary: list[DayPlan],
        profile: UserProfile
    ) -> BudgetPanel:
        """
        计算预算分项，生成右侧面板数据。
        """
        breakdown = {
            "accommodation": 0,
            "meals": 0,
            "transport": 0,
            "tickets": 0,
            "shopping": 0,
            "buffer": 0
        }
        
        # 分项估算
        for day in itinerary:
            breakdown["tickets"] += sum(a.ticket_price for a in day.activities if a.ticket_price)
            breakdown["meals"] += day.meal_cost or 0
            breakdown["transport"] += day.transport_cost or 0
        
        # 住宿估算（基于城市消费水平）
        breakdown["accommodation"] = self._estimate_accommodation(profile)
        
        total = sum(breakdown.values())
        budget_limit = profile.budget_range
        
        return BudgetPanel(
            total_budget=budget_limit,
            spent=total,
            remaining=budget_limit - total if budget_limit else None,
            breakdown=breakdown,
            status="within_budget" if (not budget_limit or total <= budget_limit) else "over_budget"
        )
    
    def get_panel_data(self, profile: UserProfile) -> PreferencePanel:
        """
        生成右侧面板的偏好展示数据。
        """
        return PreferencePanel(
            destination=profile.destination,
            travel_days=profile.travel_days,
            travel_dates=profile.travel_dates,
            travelers_count=profile.travelers_count,
            travelers_type=profile.travelers_type,
            budget_range=profile.budget_range,
            food_preferences=profile.food_preferences or [],
            interests=profile.interests or [],
            pace=profile.pace,
            special_requests=profile.special_requests or []
        )
```

#### Agent 4：Memory Management Agent（记忆管理 Agent）

```python
class MemoryManagementAgent:
    """
    记忆管理 Agent：管理所有用户相关的持久化数据。
    后台异步执行，不阻塞主流程。
    
    记忆类型：
    1. 对话记录：每条用户和 Agent 的对话
    2. 历史行程：每次确认的行程详情
    3. 历史偏好：每次行程的偏好快照
    4. 历史预算：每次行程的预算使用情况
    """
    
    async def save_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        intent: str,
        tools_used: list[str]
    ):
        """保存一轮对话记录。"""
        turn = ConversationTurn(
            session_id=session_id,
            user_message=user_message,
            assistant_response=assistant_response,
            intent=intent,
            tools_used=tools_used,
            timestamp=datetime.now().isoformat()
        )
        await self.db.conversations.insert_one(turn.to_dict())
    
    async def save_itinerary(
        self,
        session_id: str,
        itinerary: list[DayPlan],
        profile: UserProfile,
        budget: BudgetPanel,
        confirmed: bool = False
    ) -> str:
        """
        保存行程到历史。
        包含完整的行程信息、偏好快照、预算使用情况。
        """
        itinerary_record = ItineraryRecord(
            record_id=generate_id(),
            session_id=session_id,
            destination=profile.destination,
            travel_days=profile.travel_days,
            travel_dates=profile.travel_dates,
            
            # 每天的详细行程
            daily_plans=[day.to_dict() for day in itinerary],
            
            # 偏好快照（保存这个行程是基于什么偏好生成的）
            preference_snapshot={
                "food_preferences": profile.food_preferences,
                "interests": profile.interests,
                "pace": profile.pace,
                "special_requests": profile.special_requests,
                "travelers_type": profile.travelers_type
            },
            
            # 预算使用情况
            budget_snapshot={
                "total_budget": budget.total_budget,
                "spent": budget.spent,
                "breakdown": budget.breakdown
            },
            
            # 元数据
            created_at=datetime.now().isoformat(),
            confirmed=confirmed,
            status="confirmed" if confirmed else "draft"
        )
        
        await self.db.itineraries.insert_one(itinerary_record.to_dict())
        return itinerary_record.record_id
    
    async def get_user_memory(self, user_id: str) -> UserMemory:
        """
        获取用户的完整记忆（用于新对话时加载历史偏好）。
        """
        # 获取最近 3 次历史行程
        recent_itineraries = await self.db.itineraries.find(
            {"user_id": user_id}
        ).sort("created_at", -1).limit(3).to_list()
        
        # 提取历史偏好模式
        historical_preferences = self._extract_preference_patterns(recent_itineraries)
        
        # 获取所有对话记录
        conversations = await self.db.conversations.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(50).to_list()
        
        return UserMemory(
            recent_itineraries=recent_itineraries,
            historical_preferences=historical_preferences,
            recent_conversations=conversations
        )
    
    def _extract_preference_patterns(
        self, itineraries: list[ItineraryRecord]
    ) -> dict:
        """
        从历史行程中提取偏好模式。
        例如：用户每次都选轻松节奏、每次都吃辣、预算逐渐升高...
        """
        patterns = {
            "preferred_pace": Counter(),
            "preferred_food": Counter(),
            "preferred_interests": Counter(),
            "avg_budget_per_day": 0,
            "favorite_destinations": Counter()
        }
        
        for it in itineraries:
            snapshot = it.preference_snapshot
            patterns["preferred_pace"][snapshot.get("pace")] += 1
            patterns["favorite_destinations"][it.destination] += 1
            for food in snapshot.get("food_preferences", []):
                patterns["preferred_food"][food] += 1
            for interest in snapshot.get("interests", []):
                patterns["preferred_interests"][interest] += 1
        
        return patterns
```

#### Agent 5：Itinerary Planner Agent（行程规划 Agent）

```python
class ItineraryPlannerAgent:
    """
    行程规划 Agent：核心算法执行层。
    输入：候选 POI + 用户偏好 + 约束条件
    输出：每日行程草案
    """
    
    async def plan(
        self,
        pois: list[ScoredPOI],
        profile: UserProfile,
        weather: list[WeatherDay] | None = None
    ) -> list[DayPlan]:
        """
        生成行程。
        
        流程：
        1. 根据偏好给 POI 打分排序
        2. 区域聚类（DBSCAN）
        3. 时间窗口标记
        4. 日分配（考虑 must-include + 天气 + 偏好）
        5. 日内路线优化（2-opt）
        6. 时间表生成（含用餐）
        """
        # Step 1: POI 打分（基于偏好匹配度）
        scored_pois = self._score_pois_by_preferences(pois, profile)
        
        # Step 2: 区域聚类
        clusters, noise = self.cluster_pois(scored_pois, profile.destination)
        
        # Step 3: 时间约束标记
        time_constrained = self._mark_time_constraints(scored_pois)
        
        # Step 4: 日分配（LLM 辅助 + 算法结合）
        days = self.assign_to_days(
            clusters, noise, profile.travel_days,
            profile, time_constrained, weather
        )
        
        # Step 5: 日内路线优化
        for day in days:
            day.activities = await self.optimize_daily_route(
                day.activities, profile
            )
        
        # Step 6: 时间表生成
        for i, day in enumerate(days):
            day.date = self._get_date(profile.travel_dates, i)
            day.activities = self.build_schedule(
                day.activities, profile, day.date, weather[i] if weather else None
            )
        
        return days
    
    def _score_pois_by_preferences(
        self, pois: list[ScoredPOI], profile: UserProfile
    ) -> list[ScoredPOI]:
        """
        根据用户偏好给 POI 打分。
        结合 LLM 判断和规则打分。
        """
        for poi in pois:
            score = poi.base_score
            
            # 兴趣匹配
            for interest in profile.interests or []:
                if interest in poi.themes or interest in poi.tags:
                    score += 0.2
            
            # 饮食偏好匹配（餐厅类 POI）
            if poi.category == "restaurant" and profile.food_preferences:
                for food in profile.food_preferences:
                    if food in poi.tags:
                        score += 0.3
            
            # 节奏匹配
            if profile.pace == "relaxed" and poi.intensity == "low":
                score += 0.1
            elif profile.pace == "intensive" and poi.intensity == "high":
                score += 0.1
            
            # 同行人类型匹配
            if profile.travelers_type == "亲子" and "亲子" in poi.suitable_for:
                score += 0.2
            elif profile.travelers_type == "父母" and "老人" in poi.suitable_for:
                score += 0.2
            
            poi.preference_score = min(1.0, score)
        
        return sorted(pois, key=lambda p: p.preference_score, reverse=True)
```

#### Agent 6：Q&A Agent（问答 Agent）

```python
QA_SYSTEM_PROMPT = """
你是一个旅行问答专家。回答用户关于旅行目的地、景点、美食、交通等方面的问题。

## 回答原则
1. 基于 RAG 知识库和网络搜索结果回答
2. 如果不确定，明确说明"根据现有信息..."
3. 回答要简洁、实用，避免冗长
4. 如果用户问的是行程相关（如"帮我规划"），转交给 Intent Recognition Agent
5. 不要编造信息，不要推荐不存在的景点

## 信息来源优先级
1. RAG 知识库（已验证的 POI 数据）
2. 网络搜索结果（实时信息）
3. 通用旅行知识（谨慎使用，标注不确定性）
"""

class QAAgent:
    """
    问答 Agent：回答用户的非行程规划问题。
    可独立并行执行。
    """
    
    async def answer(self, question: str, city: str | None = None) -> QAResponse:
        """
        回答用户问题。
        
        流程：
        1. 同时查询 RAG 和网络搜索
        2. 整合结果
        3. LLM 生成自然语言回答
        """
        # 并行查询
        rag_task = self.rag_retriever.search(question, filters={"city": city} if city else None)
        web_task = self.web_search(f"{city} {question}" if city else question)
        
        rag_results, web_results = await asyncio.gather(rag_task, web_task)
        
        # LLM 生成回答
        prompt = f"""
        用户问题：{question}
        
        知识库结果：
        {self._format_rag_results(rag_results)}
        
        网络搜索结果：
        {self._format_web_results(web_results)}
        
        请基于以上信息回答用户问题。如果不确定，请明确说明。
        """
        
        answer = await self.llm.call(prompt, temperature=0.7)
        
        return QAResponse(
            answer=answer,
            sources=[r.source for r in rag_results[:3]],
            confidence=self._calc_confidence(rag_results, web_results)
        )
```

#### Agent 7：Map & Route Agent（地图与路线 Agent）

```python
class MapRouteAgent:
    """
    地图与路线 Agent：
    - 地理坐标查询
    - 距离矩阵计算
    - 日内路线优化
    - 交通方式建议
    """
    
    async def optimize_daily_route(
        self,
        activities: list[Activity],
        hotel_location: dict | None = None
    ) -> list[Activity]:
        """
        优化日内路线（2-opt 算法）。
        可并行执行（不同天的优化互不依赖）。
        """
        if len(activities) <= 2:
            return activities
        
        # 获取坐标
        coords = []
        for activity in activities:
            coord = await self.get_coordinates(activity.poi_name, activity.city)
            coords.append(coord)
        
        # 计算距离矩阵
        dist_matrix = await self._calc_distance_matrix(coords)
        
        # 2-opt 优化
        route = self._two_opt(range(len(activities)), dist_matrix)
        
        return [activities[i] for i in route]
    
    async def get_coordinates(self, poi_name: str, city: str) -> dict:
        """
        获取 POI 地理坐标。
        优先从本地数据库查，缺失时通过地图 API 或搜索获取。
        """
        # 先查本地
        cached = await self.db.pois.find_one({"name": poi_name, "city": city})
        if cached and cached.get("location"):
            return cached["location"]
        
        # 通过搜索获取
        result = await self.web_search(f"{city} {poi_name} 地址 坐标")
        return self._extract_coordinates(result)
```

#### Agent 8：Validation Agent（校验 Agent）

```python
class ValidationAgent:
    """
    校验 Agent：对生成的行程进行多维度校验。
    可并行执行。
    """
    
    async def validate(
        self,
        itinerary: list[DayPlan],
        profile: UserProfile
    ) -> ValidationResult:
        """
        全面校验行程。
        """
        # 并行校验多个维度
        tasks = [
            self._check_budget(itinerary, profile),
            self._check_time_feasibility(itinerary),
            self._check_poi_existence(itinerary),
            self._check_opening_hours(itinerary),
            self._check_preference_coverage(itinerary, profile),
        ]
        
        results = await asyncio.gather(*tasks)
        
        return ValidationResult(
            passed=all(r.passed for r in results),
            checks={r.check_name: r for r in results},
            overall_score=sum(r.score for r in results) / len(results)
        )
    
    async def _check_poi_existence(self, itinerary: list[DayPlan]) -> CheckResult:
        """
        检查 POI 真实性。
        抽样验证（每轮验证 30%），避免全部验证太慢。
        """
        all_pois = []
        for day in itinerary:
            for activity in day.activities:
                all_pois.append(activity.poi_name)
        
        # 抽样
        sample_size = max(1, int(len(all_pois) * 0.3))
        sample = random.sample(all_pois, min(sample_size, len(all_pois)))
        
        # 并行验证
        verify_tasks = [self._verify_poi_exists(name) for name in sample]
        results = await asyncio.gather(*verify_tasks)
        
        verified_count = sum(results)
        return CheckResult(
            check_name="poi_existence",
            passed=verified_count == len(sample),
            score=verified_count / len(sample) if sample else 1.0,
            details={"verified": verified_count, "total_sampled": len(sample)}
        )
```

### 5.4 Agent 间事件驱动机制

```python
class EventBus:
    """
    Agent 间事件总线：基于发布-订阅模式。
    Agent 通过事件通知其他 Agent，解耦调用关系。
    """
    
    def __init__(self):
        self.subscribers: dict[str, list[Callable]] = {}
    
    def subscribe(self, event_type: str, handler: Callable):
        """订阅事件。"""
        self.subscribers.setdefault(event_type, []).append(handler)
    
    async def publish(self, event_type: str, payload: dict):
        """发布事件，异步通知所有订阅者。"""
        handlers = self.subscribers.get(event_type, [])
        await asyncio.gather(*[h(payload) for h in handlers])

# 事件类型定义
EVENT_TYPES = {
    # 用户相关
    "UserInputReceived",        # 收到用户输入
    "IntentRecognized",         # 意图识别完成
    "PreferenceChanged",        # 偏好变更
    "BudgetUpdated",            # 预算更新
    
    # 行程相关
    "ItineraryGenerated",       # 行程生成完成
    "ItineraryConfirmed",       # 行程确认
    "ItineraryModified",        # 行程修改
    
    # 数据相关
    "POIDataLoaded",           # POI 数据加载完成
    "WeatherDataLoaded",       # 天气数据加载完成
    
    # 系统相关
    "AgentCallCompleted",      # Agent 调用完成
    "ValidationCompleted",     # 校验完成
}

# 典型事件订阅关系
"""
PreferenceChanged 事件订阅者：
  - ItineraryPlannerAgent: 触发重规划
  - BudgetAgent: 更新预算估算
  - MemoryAgent: 记录偏好变更历史

ItineraryGenerated 事件订阅者：
  - ValidationAgent: 触发校验
  - BudgetAgent: 计算预算面板
  - ProposalAgent: 生成方案文本
  - MemoryAgent: 保存行程草稿

ItineraryConfirmed 事件订阅者：
  - MemoryAgent: 保存为确认行程
  - StateManager: 更新当前行程状态
"""
```

---

## 六、Agent 间通信协议与 Skill 定义

### 6.1 通信协议

```python
@dataclass
class AgentMessage:
    """Agent 间消息格式。"""
    message_id: str           # 消息唯一 ID
    correlation_id: str       # 关联 ID（用于追踪同一请求链）
    sender: str               # 发送 Agent
    receiver: str             # 接收 Agent（"*" 表示广播）
    message_type: str         # 消息类型：request / response / event
    action: str               # 具体动作
    payload: dict             # 负载数据
    timestamp: str            # ISO 格式时间戳
    deadline_ms: int = 30000  # 超时时间

@dataclass
class AgentTaskResult:
    """Agent 任务执行结果。"""
    task_id: str
    status: str               # "completed" | "failed" | "partial"
    result: dict
    confidence: float         # 0.0 - 1.0
    execution_time_ms: int
    trace: list[str]          # 执行轨迹
    fallback_used: bool       # 是否使用了降级策略
```

### 6.2 Skill 定义

Skill 是 Agent 可调用的能力单元，通过 function calling 实现。

```python
# Skill 1: Web Search Skill
@skill
def web_search(
    query: str,
    top_n: int = 10,
    source_filter: list[str] | None = None
) -> SearchResult:
    """
    网络搜索 Skill。通过搜索引擎获取最新信息。
    
    Args:
        query: 搜索关键词
        top_n: 返回结果数量
        source_filter: 来源过滤（如 ["xiaohongshu", "dianping"]）
    
    Returns:
        搜索结果列表，包含标题、摘要、URL、来源
    """

# Skill 2: Web Crawler Skill
@skill
def web_crawler(
    url: str,
    extract_fields: list[str] | None = None
) -> CrawlResult:
    """
    网页爬取 Skill。抓取指定网页内容并提取结构化信息。
    
    Args:
        url: 目标网页 URL
        extract_fields: 需要提取的字段（如 ["title", "price", "rating"]）
    
    Returns:
        提取的结构化数据
    """

# Skill 3: POI Search Skill
@skill
def search_pois(
    city: str,
    keywords: list[str],
    category: str | None = None,
    top_n: int = 20
) -> list[POI]:
    """
    POI 搜索 Skill。搜索城市内符合条件的兴趣点。
    
    Args:
        city: 城市名称
        keywords: 关键词列表（如 ["火锅", "夜景"]）
        category: 类别过滤（attraction/restaurant/shopping/...）
        top_n: 返回数量
    
    Returns:
        POI 列表
    """

# Skill 4: Weather Query Skill
@skill
def query_weather(
    city: str,
    start_date: str,
    end_date: str
) -> list[WeatherDay]:
    """
    天气查询 Skill。查询指定日期范围的天气预报。
    
    Args:
        city: 城市名称
        start_date: 开始日期（YYYY-MM-DD）
        end_date: 结束日期（YYYY-MM-DD）
    
    Returns:
        每日天气数据
    """

# Skill 5: Route Calculation Skill
@skill
def calculate_route(
    origin: dict,           # {lat, lng} 或 {name, city}
    destination: dict,
    mode: str = "transit"   # "walking" | "transit" | "driving"
) -> RouteInfo:
    """
    路线计算 Skill。计算两点之间的路线和时间。
    
    Args:
        origin: 起点
        destination: 终点
        mode: 交通方式
    
    Returns:
        路线信息（距离、时间、交通方式建议）
    """

# Skill 6: Price Query Skill
@skill
def query_price(
    poi_name: str,
    city: str,
    price_type: str = "ticket"  # "ticket" | "meal" | "hotel"
) -> PriceInfo:
    """
    价格查询 Skill。查询 POI 的价格信息。
    
    Args:
        poi_name: POI 名称
        city: 城市
        price_type: 价格类型
    
    Returns:
        价格信息（当前价格、价格区间、备注）
    """

# Skill 7: Memory Store Skill
@skill
def store_memory(
    memory_type: str,       # "conversation" | "itinerary" | "preference"
    data: dict
) -> StoreResult:
    """
    记忆存储 Skill。将数据持久化到记忆系统。
    
    Args:
        memory_type: 记忆类型
        data: 要存储的数据
    
    Returns:
        存储结果（成功/失败、记录 ID）
    """

# Skill 8: Memory Retrieve Skill
@skill
def retrieve_memory(
    memory_type: str,
    query: str | None = None,
    filters: dict | None = None,
    limit: int = 10
) -> list[dict]:
    """
    记忆检索 Skill。从记忆系统中检索数据。
    
    Args:
        memory_type: 记忆类型
        query: 检索关键词
        filters: 过滤条件
        limit: 返回数量
    
    Returns:
        记忆记录列表
    """
```

---

## 七、记忆系统设计

### 7.1 记忆类型总览

```
┌─────────────────────────────────────────────────────────────────┐
│                      用户记忆系统                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 对话记忆（Conversation Memory）                              │
│     ├── 每条用户输入（原始文本）                                  │
│     ├── 每条 Agent 回复（原始文本）                               │
│     ├── 意图识别结果                                            │
│     ├── 使用的 Tools/Skills                                     │
│     └── 时间戳                                                  │
│                                                                 │
│  2. 历史行程记忆（Itinerary Memory）                             │
│     ├── 每次确认的行程（完整每日安排）                            │
│     ├── 每天的具体活动（时间、地点、费用）                         │
│     ├── 行程总预算 vs 实际花费                                   │
│     ├── 预算分项明细（住宿/餐饮/交通/门票）                        │
│     └── 时间戳 + 是否确认                                        │
│                                                                 │
│  3. 偏好记忆（Preference Memory）                                │
│     ├── 每次行程的偏好快照（生成该行程时的完整偏好）                │
│     ├── 偏好变更历史（什么时间在什么对话中改了什么）                │
│     ├── 长期偏好模式（从多次行程中总结的稳定偏好）                  │
│     └── 当前生效偏好（最新偏好）                                  │
│                                                                 │
│  4. 预算记忆（Budget Memory）                                    │
│     ├── 每次行程的预算设置                                       │
│     ├── 每次行程的实际花费                                       │
│     ├── 预算超支/结余记录                                        │
│     └── 城市消费水平参考（基于历史数据）                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 记忆数据模型

```python
# 1. 对话记录
@dataclass
class ConversationTurn:
    turn_id: str
    session_id: str
    user_id: str
    
    user_message: str           # 用户原始输入
    assistant_response: str     # Agent 回复
    
    intent: str                 # 识别的意图
    intent_confidence: float    # 意图置信度
    extracted_entities: dict    # 提取的实体
    
    tools_used: list[str]       # 使用的 Skills
    agents_involved: list[str]  # 参与的 Agents
    
    timestamp: str
    
    # 前端展示用
    display_type: str           # "text" | "itinerary" | "budget_update" | "preference_update"

# 2. 历史行程
@dataclass
class ItineraryRecord:
    record_id: str
    session_id: str
    user_id: str
    
    # 基本信息
    destination: str
    travel_days: int
    travel_dates: str
    
    # 每日详细行程
    daily_plans: list[DailyPlanRecord]
    
    # 偏好快照（生成该行程时的偏好状态）
    preference_snapshot: PreferenceSnapshot
    
    # 预算快照
    budget_snapshot: BudgetSnapshot
    
    # 状态
    status: str                 # "draft" | "confirmed" | "completed" | "cancelled"
    confirmed_at: str | None
    
    created_at: str

@dataclass
class DailyPlanRecord:
    day_number: int
    date: str
    theme: str                  # 当日主题
    
    activities: list[ActivityRecord]
    
    # 当日统计
    total_cost: float
    total_walking_steps: int
    total_transit_time_min: int

@dataclass
class ActivityRecord:
    poi_name: str
    poi_id: str | None
    category: str
    
    start_time: str
    end_time: str
    duration_min: int
    
    cost: float
    ticket_price: float | None
    
    location: dict              # {lat, lng}
    
    recommendation_reason: str  # 推荐理由
    
    # 交通信息
    transit_from_prev: dict | None  # 从上一个地点的交通信息

# 3. 偏好快照
@dataclass
class PreferenceSnapshot:
    # 基础信息
    travelers_count: int
    travelers_type: str
    
    # 偏好
    food_preferences: list[str]
    interests: list[str]
    pace: str
    accommodation_preference: str | None
    special_requests: list[str]
    
    # 预算
    budget_range: float | None
    
    # 时间戳
    captured_at: str

# 4. 偏好变更记录
@dataclass
class PreferenceChangeRecord:
    change_id: str
    session_id: str
    
    field: str                  # 变更的字段名
    old_value: any
    new_value: any
    
    source_message: str         # 触发变更的用户消息
    turn_id: str                # 关联的对话轮次
    
    timestamp: str

# 5. 预算快照
@dataclass
class BudgetSnapshot:
    total_budget: float | None
    actual_spent: float
    
    breakdown: dict             # {"accommodation": x, "meals": y, ...}
    
    overspent: bool
    overspent_amount: float
    
    captured_at: str
```

### 7.3 记忆的加载与使用

```python
class MemoryLoader:
    """
    记忆加载器：新对话开始时加载用户历史记忆。
    """
    
    async def load_for_session(self, user_id: str, session_id: str) -> LoadedMemory:
        """
        为新会话加载记忆。
        
        加载内容：
        1. 最近 3 次历史行程（用于了解用户风格）
        2. 长期偏好模式（从多次行程总结）
        3. 最近 10 轮对话（如果是同一会话继续）
        """
        # 历史行程
        recent_itineraries = await self.db.itineraries.find(
            {"user_id": user_id, "status": "confirmed"}
        ).sort("created_at", -1).limit(3).to_list()
        
        # 偏好模式
        preference_patterns = await self._calculate_preference_patterns(user_id)
        
        # 当前会话历史（如果存在）
        session_history = await self.db.conversations.find(
            {"session_id": session_id}
        ).sort("timestamp", 1).to_list()
        
        return LoadedMemory(
            recent_itineraries=recent_itineraries,
            preference_patterns=preference_patterns,
            session_history=session_history
        )
    
    def _format_for_llm(self, memory: LoadedMemory) -> str:
        """
        将记忆格式化为 LLM Prompt 的一部分。
        供 Intent Recognition Agent 和 Planner Agent 使用。
        """
        parts = []
        
        if memory.preference_patterns:
            parts.append("## 用户历史偏好模式")
            if memory.preference_patterns.get("preferred_pace"):
                pace = memory.preference_patterns["preferred_pace"].most_common(1)[0][0]
                parts.append(f"- 用户通常选择 {pace} 的节奏")
            
            if memory.preference_patterns.get("preferred_food"):
                foods = [f[0] for f in memory.preference_patterns["preferred_food"].most_common(3)]
                parts.append(f"- 用户喜欢的口味：{', '.join(foods)}")
            
            if memory.preference_patterns.get("avg_budget_per_day"):
                budget = memory.preference_patterns["avg_budget_per_day"]
                parts.append(f"- 用户历史日均预算约 {budget} 元")
        
        if memory.recent_itineraries:
            parts.append("\n## 最近行程")
            for it in memory.recent_itineraries[:2]:
                parts.append(f"- {it['destination']} {it['travel_days']} 天（{it['created_at'][:10]}）")
        
        return "\n".join(parts)
```

---

## 八、模块详细设计

### 8.1 Orchestrator（轻量编排器）

```python
class Orchestrator:
    """
    Orchestrator：轻量编排器，代码实现。
    职责是根据意图类型调度 Agent，管理并行/串行执行。
    不做决策，只做流程编排。
    """
    
    def __init__(self):
        self.agents = {}
        self.event_bus = EventBus()
        self._register_event_handlers()
    
    async def process(self, session_id: str, user_input: str) -> OrchestratorResult:
        """
        处理用户输入的主入口。
        """
        # 1. 加载当前 State
        state = await self.state_manager.load(session_id)
        
        # 2. 意图识别（必须串行，第一步）
        intent_result = await self.agents["intent"].recognize(
            user_input=user_input,
            conversation_history=state.recent_messages,
            current_state=state
        )
        
        # 3. 根据意图类型调度
        if intent_result.intent == "generate_itinerary":
            return await self._handle_generate_itinerary(session_id, state, intent_result)
        
        elif intent_result.intent == "modify_itinerary":
            return await self._handle_modify_itinerary(session_id, state, intent_result)
        
        elif intent_result.intent == "update_preferences":
            return await self._handle_update_preferences(session_id, state, intent_result)
        
        elif intent_result.intent == "query_info":
            return await self._handle_qa(session_id, state, intent_result)
        
        elif intent_result.intent == "confirm_itinerary":
            return await self._handle_confirm_itinerary(session_id, state)
        
        elif intent_result.intent == "chitchat":
            return await self._handle_chitchat(session_id, state, user_input)
        
        else:
            return await self._handle_unknown(session_id, state, user_input)
    
    async def _handle_generate_itinerary(
        self, session_id: str, state: TravelState, intent: IntentResult
    ) -> OrchestratorResult:
        """
        处理生成行程请求。
        """
        # 检查关键信息
        if intent.missing_required:
            # 信息缺失，生成追问
            questions = self._generate_clarification_questions(intent.missing_required)
            return OrchestratorResult(
                response=questions,
                state_updates={"awaiting_user_input": True}
            )
        
        # 更新用户画像
        state.user_profile = self._merge_profile(state.user_profile, intent.user_entities)
        
        # Step 1: 并行查询（POI + 天气 + 偏好面板初始化）
        query_tasks = [
            self.agents["realtime"].query_pois(
                city=state.user_profile.destination,
                interests=state.user_profile.interests or [],
                food_preferences=state.user_profile.food_preferences
            ),
            self.agents["realtime"].query_weather(
                city=state.user_profile.destination,
                dates=self._get_dates(state.user_profile)
            ),
            self.agents["preference_budget"].get_panel_data(state.user_profile)
        ]
        
        pois, weather, preference_panel = await asyncio.gather(*query_tasks)
        
        # Step 2: 行程规划（串行，依赖 Step 1 结果）
        itinerary = await self.agents["planner"].plan(
            pois=pois,
            profile=state.user_profile,
            weather=weather
        )
        
        # Step 3: 并行校验 + 预算计算
        validation_tasks = [
            self.agents["validation"].validate(itinerary, state.user_profile),
            self.agents["preference_budget"].calculate_budget_breakdown(
                itinerary, state.user_profile
            ),
            self.agents["map_route"].batch_optimize_routes(itinerary, state.user_profile)
        ]
        
        validation, budget_panel, optimized_itinerary = await asyncio.gather(*validation_tasks)
        
        # Step 4: 方案生成（串行，依赖 Step 2/3）
        proposal = await self.agents["proposal"].generate(
            itinerary=optimized_itinerary,
            profile=state.user_profile,
            budget=budget_panel,
            weather=weather,
            validation=validation
        )
        
        # Step 5: 保存状态（后台异步）
        asyncio.create_task(self._save_generation_state(
            session_id, state, itinerary, budget_panel, preference_panel
        ))
        
        return OrchestratorResult(
            response=proposal.text,
            itinerary=optimized_itinerary,
            budget_panel=budget_panel,
            preference_panel=preference_panel,
            validation_result=validation,
            state_updates={
                "current_itinerary": optimized_itinerary,
                "current_budget": budget_panel,
                "current_proposal": proposal
            }
        )
    
    async def _handle_update_preferences(
        self, session_id: str, state: TravelState, intent: IntentResult
    ) -> OrchestratorResult:
        """
        处理偏好更新。
        更新偏好 → 触发重规划（如果已有行程）。
        """
        # 更新偏好
        updated_profile = await self.agents["preference_budget"].update_preferences(
            session_id=session_id,
            changes=intent.preference_changes
        )
        
        # 如果有当前行程，触发重规划
        if state.current_itinerary:
            # 增量重规划（只更新受影响的部分）
            new_itinerary = await self.agents["planner"].replan_with_new_preferences(
                current_itinerary=state.current_itinerary,
                changes=intent.preference_changes,
                profile=updated_profile
            )
            
            # 重新计算预算
            budget_panel = await self.agents["preference_budget"].calculate_budget_breakdown(
                new_itinerary, updated_profile
            )
            
            # 生成更新方案
            proposal = await self.agents["proposal"].generate_update(
                old_itinerary=state.current_itinerary,
                new_itinerary=new_itinerary,
                changes=intent.preference_changes,
                budget=budget_panel
            )
            
            return OrchestratorResult(
                response=proposal.text,
                itinerary=new_itinerary,
                budget_panel=budget_panel,
                preference_panel=self.agents["preference_budget"].get_panel_data(updated_profile),
                state_updates={
                    "current_itinerary": new_itinerary,
                    "current_budget": budget_panel
                }
            )
        
        # 没有当前行程，只更新偏好面板
        return OrchestratorResult(
            response="好的，已记录你的偏好更新。告诉我你想去哪里玩，我来帮你规划行程！",
            preference_panel=self.agents["preference_budget"].get_panel_data(updated_profile)
        )
```

### 8.2 State 管理

```python
@dataclass
class TravelState:
    """
    旅行状态（单会话内）。
    """
    session_id: str
    user_id: str
    
    # 用户画像（渐进式收集）
    user_profile: UserProfile
    
    # 当前行程（草案或已确认）
    current_itinerary: list[DayPlan] | None
    
    # 当前预算面板
    current_budget: BudgetPanel | None
    
    # 当前偏好面板
    current_preferences: PreferencePanel | None
    
    # 对话历史（最近 20 轮）
    recent_messages: list[Message]
    
    # 运行状态
    awaiting_user_input: bool = False
    pending_agent_calls: list[str] = None
    response_draft: str | None = None
    
    # 元数据
    created_at: str
    updated_at: str
    turn_count: int = 0

class StateManager:
    """
    状态管理器：管理单会话内的状态。
    使用 Redis 做快速读写，PostgreSQL 做持久化。
    """
    
    async def load(self, session_id: str) -> TravelState:
        """从 Redis 加载 State。"""
        data = await self.redis.get(f"state:{session_id}")
        if data:
            return TravelState(**json.loads(data))
        
        # 新会话，创建默认 State
        return self._create_default_state(session_id)
    
    async def save(self, state: TravelState):
        """保存 State 到 Redis。"""
        state.updated_at = datetime.now().isoformat()
        state.turn_count += 1
        
        await self.redis.setex(
            f"state:{state.session_id}",
            timedelta(days=7),
            json.dumps(state, default=str)
        )
```

---

## 九、数据层设计：网络爬虫为主

### 9.1 数据源策略（v4 变更）

| 数据类型 | v3 来源 | **v4 来源** | 获取方式 |
|---------|---------|------------|---------|
| POI 列表/推荐 | 高德 API | **网络搜索 + 爬虫** | 搜索"成都必去景点"→ 爬取详情页 |
| POI 详情（评分/评价） | 大众点评 API | **爬虫抓取** | 爬取大众点评/携程/小红书页面 |
| 门票价格 | 携程/美团 API | **搜索 + 爬虫** | 搜索"XX景点门票价格" |
| 天气数据 | 彩云天气 API | **搜索 + 爬虫** | 搜索"成都天气预报" |
| 地理坐标 | 高德 API | **本地数据库 + 搜索** | 优先本地，缺失时搜索 |
| 攻略文本 | 小红书/马蜂窝 爬取 | **网络搜索 + RAG** | 搜索后存入 RAG |
| 路线/距离 | 高德 API | **地图工具 + 估算** | 优先本地坐标计算 |

### 9.2 爬虫架构

```python
class WebCrawlerManager:
    """
    爬虫管理器：统一的网络数据获取层。
    """
    
    async def search_and_extract(
        self,
        query: str,
        extract_schema: dict,
        top_n: int = 10
    ) -> list[ExtractedData]:
        """
        搜索并提取结构化数据。
        
        流程：
        1. 调用搜索引擎 API（或模拟搜索）
        2. 获取 Top N 结果 URL
        3. 并行爬取每个 URL
        4. 用 LLM 提取结构化数据
        5. 去重、校验
        """
        # 搜索
        search_results = await self.search_engine.search(query, top_n=top_n)
        
        # 并行爬取
        crawl_tasks = [self._crawl_page(url) for url in search_results.urls]
        pages = await asyncio.gather(*crawl_tasks, return_exceptions=True)
        
        # 提取结构化数据
        extraction_tasks = [
            self._extract_with_llm(page.content, extract_schema)
            for page in pages if not isinstance(page, Exception)
        ]
        extracted = await asyncio.gather(*extraction_tasks)
        
        # 去重
        return self._deduplicate(extracted)
    
    async def _crawl_page(self, url: str) -> CrawledPage:
        """爬取单个页面。"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers={"User-Agent": self._get_random_ua()},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                html = await response.text()
                
                # 解析正文（去除广告、导航等）
                content = self._extract_main_content(html)
                
                return CrawledPage(url=url, content=content, status=response.status)
    
    async def _extract_with_llm(self, content: str, schema: dict) -> ExtractedData:
        """
        用 LLM 从网页内容中提取结构化数据。
        """
        prompt = f"""
        从以下网页内容中提取结构化信息。
        
        提取格式：
        {json.dumps(schema, ensure_ascii=False, indent=2)}
        
        网页内容：
        {content[:8000]}  # 截断避免超长
        
        请输出严格的 JSON 格式。如果某字段找不到信息，输出 null。
        """
        
        result = await self.llm.structured_call(prompt, schema)
        return ExtractedData(**result)

# 爬虫配置（遵守 robots.txt 和频率限制）
CRAWLER_CONFIG = {
    "rate_limit": {
        "requests_per_second": 1,      # 每秒最多 1 个请求
        "delay_between_requests": 1.0,  # 请求间隔 1 秒
    },
    "timeout": 10,                     # 单个请求超时 10 秒
    "max_retries": 3,                  # 失败重试 3 次
    "respect_robots_txt": True,        # 遵守 robots.txt
    "user_agents": [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36...",
        # ... 轮换 User-Agent
    ]
}
```

### 9.3 种子数据爬取策略

```python
class SeedDataCollector:
    """
    种子数据收集器：MVP 阶段爬取 20 个热门城市的基础数据。
    """
    
    TARGET_CITIES = [
        "北京", "上海", "广州", "深圳", "成都",
        "杭州", "西安", "重庆", "苏州", "南京",
        "厦门", "青岛", "大理", "丽江", "三亚",
        "长沙", "武汉", "昆明", "桂林", "拉萨"
    ]
    
    async def collect_city_data(self, city: str) -> CitySeedData:
        """
        收集单个城市的数据。
        
        收集内容：
        1. Top 50 景点（名称、类型、简介）
        2. Top 30 餐厅（名称、类型、特色）
        3. 城市基本信息（消费水平、气候、交通）
        """
        # 并行收集
        tasks = [
            self._collect_attractions(city),
            self._collect_restaurants(city),
            self._collect_city_info(city)
        ]
        
        attractions, restaurants, city_info = await asyncio.gather(*tasks)
        
        return CitySeedData(
            city=city,
            attractions=attractions,
            restaurants=restaurants,
            city_info=city_info,
            collected_at=datetime.now().isoformat()
        )
    
    async def _collect_attractions(self, city: str) -> list[AttractionSeed]:
        """收集城市景点。"""
        query = f"{city} 必去景点 排行榜 推荐"
        results = await self.crawler.search_and_extract(
            query=query,
            extract_schema={
                "attractions": [
                    {
                        "name": "景点名称",
                        "category": "景点类型（历史/自然/现代等）",
                        "highlights": "主要看点",
                        "best_time": "最佳游览时间"
                    }
                ]
            },
            top_n=5  # 搜索前 5 个结果页面
        )
        
        return results[0].attractions[:50] if results else []
```

### 9.4 本地缓存策略

```python
class DataCache:
    """
    数据缓存：减少对网络爬取的依赖，提升响应速度。
    """
    
    CACHE_TTL = {
        "poi_list": timedelta(days=7),        # POI 列表缓存 7 天
        "poi_detail": timedelta(days=3),      # POI 详情缓存 3 天
        "weather": timedelta(hours=6),        # 天气缓存 6 小时
        "price": timedelta(days=1),           # 价格缓存 1 天
        "route": timedelta(days=1),           # 路线缓存 1 天
    }
    
    async def get_or_fetch(
        self,
        cache_key: str,
        fetch_func: Callable,
        ttl: timedelta
    ):
        """先查缓存，未命中再爬取。"""
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        # 缓存未命中，爬取
        data = await fetch_func()
        
        # 写入缓存
        await self.cache.set(cache_key, data, ttl=ttl)
        
        return data
```

---

## 十、约束求解与路径优化

### 10.1 问题建模

与 v3 保持一致，旅行规划建模为带时间窗口的多约束优化问题。

### 10.2 分层求解算法

与 v3 保持一致，包含：
1. 区域聚类（DBSCAN，动态 eps）
2. 时间窗口硬约束标记
3. 日分配（多目标贪心）
4. 日内路径优化（2-opt）
5. 时间表生成（含用餐插入）
6. 约束校验

**v4 新增**：LLM 辅助的日分配决策。
在日分配阶段，当算法无法确定某个簇应该放在哪天时，调用 LLM 做决策。

```python
def assign_to_days_with_llm_fallback(
    clusters: list[list[POI]],
    num_days: int,
    profile: UserProfile,
    weather: list[WeatherDay] | None = None
) -> list[list[POI]]:
    """
    日分配：算法为主，LLM 辅助处理模糊情况。
    """
    days = [[] for _ in range(num_days)]
    
    # 算法分配（大部分情况）
    for cluster in clusters:
        best_day = _find_best_day_by_algorithm(days, cluster, profile)
        
        if best_day is not None:
            days[best_day].extend(cluster)
        else:
            # 算法无法确定，LLM 辅助决策
            llm_decision = await llm.decide_day_assignment(
                cluster=cluster,
                current_days=days,
                weather=weather,
                profile=profile
            )
            days[llm_decision.day_index].extend(cluster)
    
    return days
```

---

## 十一、RAG 知识库系统

### 11.1 RAG 职责

| 信息类型 | 获取方式 | 原因 |
|---------|---------|------|
| POI 详细介绍、游玩建议 | RAG | 变化慢，适合离线索引 |
| 城市攻略、文化背景 | RAG | 变化慢，适合文档检索 |
| 用户评论摘要 | RAG | 经过预处理，适合语义检索 |
| 实时天气预报 | Web Search | 实时数据 |
| 实时价格 | Web Search | 可能有波动 |
| 景点当前开放状态 | Web Search | 可能临时变化 |

### 11.2 数据更新策略

```python
class RAGDataUpdater:
    """
    RAG 数据更新：基于爬虫结果定期更新向量数据库。
    """
    
    UPDATE_SCHEDULE = {
        "poi_basic": "weekly",       # 每周刷新 POI 基础信息
        "guide": "monthly",          # 每月更新攻略
        "review_summary": "weekly",  # 每周更新评论摘要
    }
    
    async def run_update(self):
        """
        增量更新：
        1. 爬取最新数据
        2. 与旧数据对比
        3. 有变化时更新 embedding
        4. 删除的数据标记为 unavailable
        """
        pass
```

---

## 十二、评估体系设计

### 12.1 评估指标

与 v3 保持一致，8 个维度评估：
- 偏好匹配度（preference_coverage）
- 路线合理性（route_efficiency）
- 时间可行性（time_feasibility）
- 预算合理性（budget_compliance）
- 舒适度（comfort_score）
- 真实性（factuality）
- 可解释性（explainability）
- 多样性（diversity）

### 12.2 信心值计算

```python
class ConfidenceCalculator:
    """
    信心值计算：基于客观指标，不由 LLM 自我评估。
    """
    
    def calculate(self, eval_result: EvalResult, state: TravelState) -> float:
        # 基础评估分
        base_score = eval_result.total_score
        
        # 数据可靠性系数
        live_data_ratio = self._calc_live_data_ratio(state)
        reliability_factor = 0.7 + 0.3 * live_data_ratio
        
        # 约束满足度
        hard_constraints_met = len(eval_result.critical_failures) == 0
        constraint_factor = 1.0 if hard_constraints_met else 0.5
        
        confidence = base_score * reliability_factor * constraint_factor
        return min(1.0, max(0.0, confidence))
```

---

## 十三、前端交互设计：三栏布局

### 13.1 总体布局

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  顶部导航栏                                                                  │
│  [Logo] 旅行 Agent                              [用户头像]                    │
├──────────┬──────────────────────────────────────────┬───────────────────────┤
│          │                                          │                       │
│  左侧栏   │              中间栏                      │       右侧栏          │
│          │                                          │                       │
│ [新建    │  ┌────────────────────────────────────┐  │  ┌─────────────────┐  │
│  对话]   │  │  对话区域                            │  │  │    预算面板      │  │
│          │  │                                    │  │  │                 │  │
│ ───────  │  │  🤖 你好！想去哪里旅行？             │  │  │  总预算: ¥3,000 │  │
│          │  │                                    │  │  │  已分配: ¥2,650 │  │
│ [对话    │  │  👤 我想去成都玩 4 天               │  │  │  剩余: ¥350     │  │
│  列表]   │  │                                    │  │  │                 │  │
│          │  │  🤖 好的！成都 4 天行程已生成...     │  │  │  ━━━━━━━━━━━━   │  │
│ 对话 1   │  │                                    │  │  │  住宿 ¥1,200    │  │
│ 对话 2   │  │  [行程卡片 - Day 1]                 │  │  │  餐饮 ¥800      │  │
│ 对话 3   │  │  09:00  宽窄巷子                     │  │  │  交通 ¥300      │  │
│ ...      │  │  12:00  午餐：火锅                   │  │  │  门票 ¥350      │  │
│          │  │  ...                                │  │  │                 │  │
│ ───────  │  │                                    │  │  └─────────────────┘  │
│          │  │  [行程卡片 - Day 2]                 │  │                       │
│ [当前    │  │  ...                                │  │  ┌─────────────────┐  │
│  行程]   │  │                                    │  │  │    偏好面板      │  │
│          │  │  [✅ 确认行程]  [🔄 重新规划]        │  │  │                 │  │
│ ───────  │  │                                    │  │  │  目的地: 成都    │  │
│          │  └────────────────────────────────────┘  │  │  天数: 4 天      │  │
│ [历史    │                                          │  │  人数: 2 人      │  │
│  行程]   │  当点击"当前行程"时，中间栏切换为：     │  │  预算: ¥3,000    │  │
│          │                                          │  │  饮食: 辣、甜品  │  │
│ 行程 1   │  ┌────────────────────────────────────┐  │  │  兴趣: 拍照、美食│  │
│ 行程 2   │  │  行程详情（时间线视图）               │  │  │  节奏: 轻松      │  │
│ ...      │  │                                    │  │  │                 │  │
│          │  │  Day 1  ━━━━━━━━━━━━━━━━━━━━━━━━━  │  │  │  [编辑偏好]      │  │
│          │  │  09:00  宽窄巷子                     │  │  └─────────────────┘  │
│          │  │        ↓ 步行 10 分钟               │  │                       │
│          │  │  11:00  人民公园喝茶                 │  │                       │
│          │  │        ↓ 地铁 15 分钟               │  │                       │
│          │  │  12:30  午餐：陈麻婆豆腐             │  │                       │
│          │  │  ...                                │  │                       │
│          │  │                                    │  │                       │
│          │  │  Day 2  ━━━━━━━━━━━━━━━━━━━━━━━━━  │  │                       │
│          │  │  ...                                │  │                       │
│          │  └────────────────────────────────────┘  │  │                       │
│          │                                          │  │                       │
└──────────┴──────────────────────────────────────────┴───────────────────────┘
```

### 13.2 左侧面板

```
┌──────────┐
│ 左侧栏    │
├──────────┤
│ [+ 新建  │
│  对话]   │
├──────────┤
│ 对话列表  │
│ ──────── │
│ 💬 成都  │
│    4日游 │
│ 💬 北京  │
│    攻略问│
│ 💬 新的  │
│    对话  │
├──────────┤
│ [当前行程]│
│ （点击后 │
│  中间栏  │
│  显示行程│
│  详情）  │
├──────────┤
│ 历史行程  │
│ ──────── │
│ 📋 成都  │
│    3日游 │
│    2026  │
│    -03   │
│ 📋 杭州  │
│    2日游 │
│    2026  │
│    -02   │
└──────────┘
```

**交互规则**：
1. 点击"新建对话"：创建新 session，中间栏清空对话，右侧面板清空
2. 点击对话列表中的某个对话：加载该 session 的对话历史和状态
3. 点击"当前行程"：中间栏切换到行程时间线视图（仅当有已确认行程时可用）
4. 点击历史行程：中间栏显示该历史行程的详情（只读）

### 13.3 右侧面板

#### 上半部分：预算面板

```
┌─────────────────┐
│    预算面板      │
├─────────────────┤
│  总预算: ¥3,000 │
│  已分配: ¥2,650 │
│  剩余: ¥350     │
│                 │
│  ████████████░░ │
│  88.3%          │
│                 │
│  ━━━━━━━━━━━━━  │
│  住宿   ¥1,200  │
│  餐饮     ¥800  │
│  交通     ¥300  │
│  门票     ¥350  │
│                 │
│  ⚠️ 剩余较少，  │
│     建议控制购物│
└─────────────────┘
```

#### 下半部分：偏好面板

```
┌─────────────────┐
│    偏好面板      │
├─────────────────┤
│  目的地: 成都    │
│  天数: 4 天      │
│  日期: 5/1-5/4  │
│  人数: 2 人      │
│  类型: 情侣      │
│  预算: ¥3,000    │
│  ─────────────  │
│  饮食偏好        │
│  🌶️ 辣  🍰 甜品 │
│  ─────────────  │
│  兴趣爱好        │
│  📷 拍照  🍜 美食│
│  ─────────────  │
│  节奏: 轻松      │
│  ─────────────  │
│  特殊要求        │
│  "不想太累"      │
│                 │
│  [编辑偏好]      │
└─────────────────┘
```

**动态更新规则**：
1. 对话过程中，Intent Recognition Agent 提取到新信息 → 右侧面板实时更新
2. 用户说"预算改到 5000" → 预算面板数字变为 5000，进度条重算
3. 用户说"再加个甜品爱好" → 偏好面板的饮食偏好增加"甜品"标签
4. 行程生成后 → 预算面板显示分项明细
5. 用户确认行程后 → 当前偏好锁定，作为该行程的偏好快照保存

### 13.4 行程确认流程

```
用户："我觉得这个行程不错"
    │
    ▼
中间栏显示：[✅ 确认行程] 按钮
    │
    ▼
用户点击"确认行程"
    │
    ▼
系统：
1. 保存当前行程为"已确认"
2. 左侧面板"当前行程"变为可点击
3. 点击"当前行程"中间栏显示行程时间线
4. 历史行程中新增一条记录
5. 记忆系统保存完整行程 + 偏好快照 + 预算快照
    │
    ▼
Agent 回复："行程已确认！你可以在左侧'当前行程'查看详情，也可以继续对话修改。"
```

### 13.5 前端技术方案

```
技术栈：
- 框架：Next.js 14 + React 18
- 样式：Tailwind CSS + shadcn/ui
- 状态管理：Zustand（轻量，适合本应用）
- 实时通信：WebSocket（对话流式输出）
- 地图：高德地图 JS API

状态设计（Zustand）：
interface AppState {
  // 会话
  currentSessionId: string | null;
  sessions: Session[];
  
  // 对话
  messages: Message[];
  isLoading: boolean;
  
  // 行程
  currentItinerary: Itinerary | null;
  currentProposal: Proposal | null;
  
  // 右侧面板
  budgetPanel: BudgetPanel | null;
  preferencePanel: PreferencePanel | null;
  
  // 历史
  historyItineraries: ItineraryRecord[];
}

WebSocket 消息类型：
type WSMessage =
  | { type: "user_message"; content: string }
  | { type: "assistant_stream"; delta: string }
  | { type: "assistant_done"; fullResponse: string }
  | { type: "itinerary_update"; data: Itinerary }
  | { type: "budget_update"; data: BudgetPanel }
  | { type: "preference_update"; data: PreferencePanel }
  | { type: "intent_recognized"; intent: string; entities: object }
  | { type: "error"; message: string };
```

---

## 十四、完整用户流程设计

### 14.1 首次使用流程

```
用户打开应用
    │
    ▼
┌────────────────────────────────────────┐
│ 欢迎页面                                │
│                                         │
│ "你好！我是你的旅行规划助手。           │
│  告诉我你想去哪里、玩几天，             │
│  我来帮你规划完美行程！"                │
│                                         │
│ [开始对话]                              │
└────────────────────────────────────────┘
    │
    ▼
中间栏显示输入框，右侧面板为空
    │
    ▼
用户输入："我想去成都玩 4 天，预算 3000"
    │
    ▼
Intent Recognition Agent：
- 意图：generate_itinerary
- 提取：destination=成都, travel_days=4, budget=3000
- 关键信息完整（有 destination + travel_days）
    │
    ▼
并行启动：
├─ Real-time Agent：搜索成都景点、美食、天气
├─ Preference Agent：初始化偏好面板
│  （显示：目的地=成都，天数=4，预算=3000）
└─ Memory Agent：记录对话
    │
    ▼
行程规划 Agent 生成行程草案
    │
    ▼
并行校验 + 预算计算
    │
    ▼
右侧面板更新：
├─ 预算面板：分项估算（住宿、餐饮、交通、门票）
└─ 偏好面板：自动推断 pace=moderate
    │
    ▼
中间栏显示：
├─ 自然语言行程描述
├─ 每日行程卡片（可展开）
├─ [✅ 确认行程] 按钮
└─ [🔄 重新规划] 按钮
    │
    ▼
用户点击"确认行程"
    │
    ▼
系统保存行程，左侧面板"当前行程"变为可点击
```

### 14.2 多轮修改流程

```
用户："第三天轻松一点，少点景点"
    │
    ▼
Intent Recognition Agent：
- 意图：modify_itinerary
- 修改类型：reduce_intensity
- 影响范围：Day 3（单日）
    │
    ▼
行程规划 Agent：
- 只修改 Day 3
- 减少景点数量（5个 → 3个）
- 替换高强度景点为低强度
- 保留已确认的高优先级景点
    │
    ▼
预算 Agent：重新计算（Day 3 预算减少）
右侧面板预算更新
    │
    ▼
中间栏显示：
├─ "已为你调整 Day 3："
├─ "- 删除了武侯祠（需要步行 5000 步）"
├─ "- 替换为人民公园喝茶（轻松休闲）"
├─ "- Day 1/2/4 不受影响"
├─ 新行程展示
└─ [✅ 确认行程] [🔄 再改改]
```

### 14.3 偏好更新流程

```
用户："对了，我其实还喜欢吃日料，帮我加几个"
    │
    ▼
Intent Recognition Agent：
- 意图：update_preferences
- 新增偏好：food_preferences += [日料]
    │
    ▼
Preference Agent：
- 更新偏好面板：饮食偏好 = [辣, 日料]
- 触发 PreferenceChanged 事件
    │
    ▼
Itinerary Planner Agent（监听事件）：
- 重新搜索成都日料店
- 插入到每日行程中（午餐/晚餐时段）
    │
    ▼
预算 Agent：
- 日料增加预算
- 更新预算面板
    │
    ▼
中间栏显示：
├─ "已为你新增日料偏好！"
├─ "重新规划后，每天的晚餐加入了成都口碑日料店"
├─ 更新后的行程展示
└─ [✅ 确认行程]
```

---

## 十五、技术栈选型与项目结构

### 15.1 技术栈

| 层级 | 技术选型 | 理由 |
|------|---------|------|
| LLM | Claude Sonnet 4 / 通义千问 / DeepSeek | 工具调用强、中文好、成本可控 |
| Agent 框架 | 自研轻量框架（基于 asyncio + EventBus） | 本系统 Agent 间需要高度自定义的并行编排，现有框架过重 |
| Skill 实现 | Python 函数 + Pydantic  schema | 轻量、类型安全 |
| 向量数据库 | Qdrant | 支持过滤、快速部署 |
| 爬虫 | aiohttp + BeautifulSoup / Playwright | 异步、轻量 |
| 搜索引擎 | DuckDuckGo API / SerpAPI | 无需认证、免费额度充足 |
| 状态存储 | Redis (Session) + PostgreSQL (持久) | 快速读写 + 关系型数据持久化 |
| 路径优化 | 自实现 2-opt | MVP 足够 |
| 地图 | 高德地图 JS API（前端）+ 本地坐标计算（后端） | 国内场景 |
| 后端框架 | FastAPI | 异步支持好、WebSocket 原生 |
| 前端框架 | Next.js + React + Tailwind CSS + shadcn/ui | SSR + 响应式 + 组件库 |
| 部署 | Docker + Railway / Render | 低成本快速部署 |

### 15.2 项目目录结构

```
travel-agent/
├── README.md
├── docker-compose.yml
├── pyproject.toml
│
├── backend/                             # 后端
│   ├── src/
│   │   ├── main.py                      # FastAPI 入口
│   │   │
│   │   ├── agents/                      # Agent 层
│   │   │   ├── __init__.py
│   │   │   ├── intent_recognition.py    # 意图识别 Agent
│   │   │   ├── information_collection.py # 信息收集 Agent
│   │   │   ├── realtime_query.py        # 实时查询 Agent
│   │   │   ├── preference_budget.py     # 偏好与预算 Agent
│   │   │   ├── itinerary_planner.py     # 行程规划 Agent
│   │   │   ├── qa_agent.py              # 问答 Agent
│   │   │   ├── memory_management.py     # 记忆管理 Agent
│   │   │   ├── proposal_generation.py   # 方案生成 Agent
│   │   │   ├── validation.py            # 校验 Agent
│   │   │   └── map_route.py             # 地图与路线 Agent
│   │   │
│   │   ├── core/                        # 核心控制层
│   │   │   ├── orchestrator.py          # Orchestrator 编排器
│   │   │   ├── state_manager.py         # 状态管理器
│   │   │   ├── event_bus.py             # 事件总线
│   │   │   └── schemas.py               # 数据模型定义
│   │   │
│   │   ├── skills/                      # Skill 实现
│   │   │   ├── __init__.py
│   │   │   ├── web_search.py            # 网络搜索
│   │   │   ├── web_crawler.py           # 网页爬取
│   │   │   ├── poi_search.py            # POI 搜索
│   │   │   ├── weather_query.py         # 天气查询
│   │   │   ├── route_calculation.py     # 路线计算
│   │   │   ├── price_query.py           # 价格查询
│   │   │   └── memory_store.py          # 记忆存取
│   │   │
│   │   ├── planner/                     # 规划算法层
│   │   │   ├── constraint_solver.py     # 约束求解器
│   │   │   ├── route_optimizer.py       # 路径优化（2-opt）
│   │   │   ├── spatial_clustering.py    # 区域聚类
│   │   │   ├── day_assigner.py          # 日分配
│   │   │   ├── schedule_builder.py      # 时间表生成
│   │   │   └── constraint_checker.py    # 约束校验
│   │   │
│   │   ├── memory/                      # 记忆系统
│   │   │   ├── conversation_store.py    # 对话记录存储
│   │   │   ├── itinerary_store.py       # 行程记录存储
│   │   │   ├── preference_store.py      # 偏好记录存储
│   │   │   └── memory_loader.py         # 记忆加载器
│   │   │
│   │   ├── rag/                         # RAG 知识库
│   │   │   ├── retriever.py             # 混合检索器
│   │   │   ├── vector_store.py          # 向量数据库接口
│   │   │   └── data_updater.py          # 数据更新
│   │   │
│   │   ├── crawler/                     # 爬虫层
│   │   │   ├── search_engine.py         # 搜索引擎接口
│   │   │   ├── page_crawler.py          # 页面爬取
│   │   │   ├── data_extractor.py        # 数据提取（LLM）
│   │   │   └── cache_manager.py         # 缓存管理
│   │   │
│   │   ├── data/                        # 数据层
│   │   │   ├── seed_data/               # 种子数据
│   │   │   │   └── cities/              # 城市数据
│   │   │   └── seed_collector.py        # 种子数据收集器
│   │   │
│   │   ├── api/                         # API 层
│   │   │   ├── routes/
│   │   │   │   ├── chat.py              # WebSocket 聊天接口
│   │   │   │   ├── itinerary.py         # 行程 REST 接口
│   │   │   │   └── memory.py            # 记忆接口
│   │   │   └── middleware/
│   │   │       ├── auth.py
│   │   │       └── rate_limit.py
│   │   │
│   │   └── config/
│   │       └── settings.py              # 配置管理
│   │
│   └── tests/
│       ├── unit/
│       ├── integration/
│       └── e2e/
│
├── frontend/                            # 前端
│   ├── src/
│   │   ├── app/                         # Next.js App Router
│   │   │   ├── page.tsx                 # 主页面（三栏布局）
│   │   │   ├── layout.tsx
│   │   │   └── globals.css
│   │   │
│   │   ├── components/
│   │   │   ├── layout/
│   │   │   │   ├── LeftSidebar.tsx      # 左侧栏
│   │   │   │   ├── MainContent.tsx      # 中间栏
│   │   │   │   └── RightPanel.tsx       # 右侧栏
│   │   │   │
│   │   │   ├── chat/
│   │   │   │   ├── ChatInput.tsx        # 聊天输入框
│   │   │   │   ├── MessageList.tsx      # 消息列表
│   │   │   │   ├── MessageBubble.tsx    # 消息气泡
│   │   │   │   └── ItineraryCard.tsx    # 行程卡片
│   │   │   │
│   │   │   ├── itinerary/
│   │   │   │   ├── TimelineView.tsx     # 时间线视图
│   │   │   │   ├── DayPlanCard.tsx      # 每日计划卡片
│   │   │   │   ├── ActivityItem.tsx     # 活动项
│   │   │   │   └── ConfirmButton.tsx    # 确认按钮
│   │   │   │
│   │   │   ├── panels/
│   │   │   │   ├── BudgetPanel.tsx      # 预算面板
│   │   │   │   ├── PreferencePanel.tsx  # 偏好面板
│   │   │   │   └── PanelItem.tsx        # 面板项
│   │   │   │
│   │   │   └── sidebar/
│   │   │       ├── NewChatButton.tsx    # 新建对话按钮
│   │   │       ├── SessionList.tsx      # 对话列表
│   │   │       ├── CurrentItinerary.tsx # 当前行程入口
│   │   │       └── HistoryList.tsx      # 历史行程列表
│   │   │
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts          # WebSocket 钩子
│   │   │   ├── useChat.ts               # 对话状态管理
│   │   │   └── useItinerary.ts          # 行程状态管理
│   │   │
│   │   ├── stores/
│   │   │   └── appStore.ts              # Zustand 全局状态
│   │   │
│   │   ├── types/
│   │   │   ├── chat.ts                  # 对话类型
│   │   │   ├── itinerary.ts             # 行程类型
│   │   │   └── api.ts                   # API 类型
│   │   │
│   │   └── lib/
│   │       ├── api.ts                   # API 客户端
│   │       └── utils.ts                 # 工具函数
│   │
│   ├── package.json
│   └── next.config.js
│
└── scripts/
    ├── seed_data_collection.py          # 种子数据收集
    └── run_eval.py                      # 评估脚本
```

---

## 十六、MVP 分阶段交付计划

### Phase 1: 核心可用（Week 1-3）

**目标**：单轮单城市行程生成，CLI / Postman 可验证

**范围**：
- 意图识别 Agent（LLM 驱动）
- 实时查询 Agent（基础网络搜索）
- 行程规划 Agent（基础算法）
- 偏好与预算 Agent（基础面板数据）
- 方案生成 Agent
- 基础三栏前端（仅中间栏对话 + 右侧面板静态展示）
- 种子数据：手工整理 5 个城市的 Top 20 POI

**不做**：
- 多轮修改
- 记忆持久化（仅内存）
- 爬虫系统（手工数据 + 简单搜索）
- 历史行程
- 方案确认流程

**验收标准**：
- 输入"去成都 4 天 预算 3000" → 输出结构化行程 JSON
- 右侧面板展示提取的偏好和预算
- 行程包含真实 POI

### Phase 2: 多轮对话 + 记忆（Week 4-6）

**目标**：支持多轮修改、偏好更新、记忆持久化

**交付物**：
- 多轮对话支持（State 持久化到 Redis）
- 偏好更新检测 + 触发重规划
- 记忆管理 Agent（对话记录 + 历史行程）
- 左侧面板（对话列表、历史行程）
- 当前行程确认流程
- 爬虫系统（基础版）
- 种子数据扩展到 20 城 × 50 POI

**验收标准**：
- "第三天轻松一点"能修改 Day 3
- "再加个甜品爱好"能更新偏好并重规划
- 确认后的行程可在"当前行程"查看

### Phase 3: 体验完善（Week 7-10）

**目标**：完整三栏交互、实时信息查询、评估体系

**交付物**：
- 完整三栏前端（时间线视图、地图联动）
- 实时信息查询（天气、价格）
- RAG 知识库（基础版）
- 问答 Agent
- 评估模块
- 地图与路线 Agent
- 行程导出（JSON）

**验收标准**：
- 前端可展示完整行程时间线
- 天气数据显示在行程中
- 支持"成都有什么好吃的"问答

### Phase 4: 高级功能（Week 11+）

**交付物**：
- 路线优化（2-opt 精确化）
- 雨天备选方案
- 用餐推荐（基于位置和偏好）
- 行程导出（PDF / iCal）
- 可观测性（决策链路追踪）
- 更多城市数据覆盖
- 移动端体验优化

---

## 十七、风险与缓解策略

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|---------|
| LLM 意图识别错误 | 中 | 高 | 1. 多轮对话纠正<br>2. 置信度低时主动确认<br>3. 用户可随时说"不对，我是想..." |
| LLM 幻觉（推荐不存在景点） | 高 | 高 | 1. 所有 POI 来自网络搜索验证<br>2. Validation Agent 抽样校验<br>3. 标注数据来源 |
| 爬虫被反爬 | 中 | 中 | 1. 控制频率（1 req/s）<br>2. 轮换 User-Agent<br>3. 本地缓存兜底<br>4. 失败时切换数据源 |
| 网络搜索结果质量差 | 中 | 中 | 1. 多源交叉验证<br>2. LLM 提取时标注不确定性<br>3. 用户可反馈纠错 |
| 爬虫数据合规风险 | 中 | 高 | 1. 遵守 robots.txt<br>2. 控制请求频率<br>3. 仅获取公开可见数据<br>4. 数据标注来源 |
| 多轮对话状态丢失 | 中 | 高 | 1. Redis 持久化<br>2. 每轮保存 State<br>3. 版本化 State |
| Token 消耗过高 | 中 | 中 | 1. 意图识别用轻量模型<br>2. 结构化输出减少 Token<br>3. 缓存 LLM 响应 |
| 种子数据覆盖不足 | 中 | 高 | 1. MVP 只支持 20 个热门城市<br>2. 非覆盖城市时明确提示<br>3. 爬虫逐步扩展 |
| 1-2 人团队进度延迟 | 高 | 高 | 1. 严格范围控制<br>2. 优先核心路径<br>3. 每周验收调整 |

---

## 附录

### 附录 A：v3 -> v4 关键变更总结

| 变更项 | v3 设计 | v4 设计 | 变更理由 |
|--------|---------|---------|---------|
| 控制架构 | 确定性状态机 + LLM | **LLM 主导意图识别 + 代码辅助执行** | 状态机无法覆盖用户表达多样性 |
| 意图识别 | 状态机规则匹配 | **LLM 做意图识别、实体提取、缺失判断** | 用户表达多样，LLM 更准确 |
| 信息追问 | 状态机缺失字段判断 | **LLM 自然语言追问** | 追问更自然，可结合上下文 |
| 偏好管理 | 代码硬编码 | **LLM 检测偏好变更，动态覆盖** | 用户随时改偏好，代码无法预判 |
| Agent 数量 | 3 个（串行为主） | **10 个（能并行尽量并行）** | 性能 + 职责清晰 |
| 数据源 | API 为主 | **网络爬虫 + 搜索为主** | 降低依赖，扩大覆盖 |
| 记忆系统 | 简单 Session | **结构化记忆：对话 + 行程 + 偏好 + 预算** | 长期记忆驱动个性化 |
| 交互方式 | 纯对话 | **三栏布局：对话 + 行程 + 面板** | 信息可视化，用户体验更好 |
| 行程确认 | 无确认流程 | **明确确认流程，保存历史** | 用户有确定感，可追溯 |
| 前端 | 无（Phase 1） | **三栏交互前端** | 用户体验核心 |

### 附录 B：Agent 调用关系图

```
用户输入
    │
    ▼
Intent Recognition Agent（串行）
    │
    ├── intent=generate ──▶ Orchestrator
    │                         │
    │                         ├── 并行: Real-time Query Agent
    │                         │         ├─ web_search
    │                         │         ├─ web_crawler
    │                         │         └─ query_weather
    │                         │
    │                         ├── 并行: Preference & Budget Agent
    │                         │         └─ init_panel
    │                         │
    │                         ├── 串行: Itinerary Planner Agent
    │                         │         └─ plan
    │                         │
    │                         ├── 并行: Validation Agent
    │                         │         ├─ check_budget
    │                         │         ├─ check_time
    │                         │         └─ check_poi_exists
    │                         │
    │                         ├── 并行: Map & Route Agent
    │                         │         └─ optimize_routes
    │                         │
    │                         ├── 并行: Budget Agent
    │                         │         └─ calculate_breakdown
    │                         │
    │                         └── 串行: Proposal Generation Agent
    │                                   └─ generate
    │
    ├── intent=update_preferences ──▶ Preference & Budget Agent
    │                                   │
    │                                   ├── 更新偏好面板
    │                                   ├── 发布 PreferenceChanged 事件
    │                                   └── 触发 Itinerary Planner（如果已有行程）
    │
    ├── intent=modify ──▶ Itinerary Planner Agent（增量更新）
    │
    ├── intent=q_a ──▶ Q&A Agent（独立并行）
    │
    └── intent=confirm ──▶ Memory Management Agent（保存确认行程）
```

### 附录 C：关键设计决策记录 (ADR)

**ADR-001: LLM 主导意图识别**
- 决策：第一层控制由 LLM 做意图识别，而非状态机
- 原因：用户表达方式多样，状态机规则覆盖不全，LLM 更灵活准确

**ADR-002: Agent 并行编排**
- 决策：无依赖的 Agent 并行执行
- 原因：提升响应速度，改善用户体验

**ADR-003: 网络爬虫为主的数据源**
- 决策：网络爬虫 + 搜索为主，API 为辅
- 原因：降低对付费 API 的依赖，扩大数据覆盖

**ADR-004: 三栏前端布局**
- 决策：左侧导航 + 中间内容 + 右侧面板
- 原因：信息可视化，用户可实时看到提取的偏好和预算

**ADR-005: 结构化记忆系统**
- 决策：独立的记忆管理 Agent，持久化对话 + 行程 + 偏好 + 预算
- 原因：长期记忆支撑个性化，历史行程可追溯

---

**文档结束**

> 本文档是 v3 版本的架构升级版本，核心变更为：LLM 主导意图识别、Agent 并行编排、网络爬虫数据源、三栏交互、结构化记忆系统。所有设计决策均面向 1-2 人团队，MVP 优先核心路径。
