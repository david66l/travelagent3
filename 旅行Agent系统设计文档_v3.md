# 智能旅行 Agent 系统设计文档

> **文档版本**：v3.0
> **最后更新**：2026-04-27
> **适用范围**：面向自动化开发 Agent 的完整技术规格说明
> **定位**：面向中国境内自由行用户的单 Agent 行程规划系统，采用双层控制架构（确定性保底 + LLM 优化），具备约束求解、动态重规划与数据可靠性保障能力。

---

## 目录

1. [关键设计决策与约束](#一关键设计决策与约束)
2. [产品定位与用户画像](#二产品定位与用户画像)
3. [核心场景与 MVP 边界](#三核心场景与-mvp-边界)
4. [系统总体架构](#四系统总体架构)
5. [Agent 编排与通信协议](#五agent-编排与通信协议)
6. [模块详细设计](#六模块详细设计)
7. [数据层设计](#七数据层设计)
8. [约束求解与路径优化算法](#八约束求解与路径优化算法)
9. [RAG 知识库系统](#九rag-知识库系统)
10. [评估体系设计](#十评估体系设计)
11. [Human-in-the-Loop 协作机制](#十一human-in-the-loop-协作机制)
12. [前端交互设计规范](#十二前端交互设计规范)
13. [完整用户流程设计](#十三完整用户流程设计)
14. [技术栈选型与项目结构](#十四技术栈选型与项目结构)
15. [MVP 分阶段交付计划](#十五mvp-分阶段交付计划)
16. [风险与缓解策略](#十六风险与缓解策略)

---

## 一、关键设计决策与约束

本章记录所有对架构有深远影响的设计决策。这些决策基于对 v2 版本问题的反思，并在落地前经团队确认。

### 1.1 场景约束（已确认）

| 约束项 | 决策 | 理由 |
|--------|------|------|
| 旅行范围 | 中国境内城市 | 团队熟悉度高，数据源可控，无需汇率/签证逻辑 |
| 城市数量 | MVP 仅支持单城市 | 1-2 人团队无法同时做好跨城交通和城内规划 |
| 数据源 | 公开数据爬取 + 免费 API 补充 | 控制成本，快速启动 |
| 前端形态 | Web 端（Next.js），响应式适配移动端 | 团队技术栈统一，一套代码覆盖多端 |
| 开发团队 | 1-2 人，全栈 | 人力约束直接决定 MVP 范围 |

### 1.2 架构决策

| 决策项 | 决策 | 理由 |
|--------|------|------|
| 控制架构 | **双层控制**：确定性状态机保底 + LLM 决策优化 | v2 完全依赖 LLM 决策，循环终止不可靠，关键路径无保底 |
| State 管理 | 版本化 State + 乐观锁 | v2 未处理并发修改和状态回滚 |
| MCP 可靠性 | Circuit Breaker + 分级降级 + 本地缓存 | v2 缺少统一的超时和降级策略 |
| 约束求解 | 时间窗口作为硬约束前置，2-opt 仅对灵活点优化 | v2 的 2-opt 忽略时间窗口，导致景点被静默跳过 |
| 局部修改 | 定义明确的影响半径（单日内），修改后触发约束重校验 | v2 未定义影响范围和回退机制 |

### 1.3 数据决策

| 决策项 | 决策 | 理由 |
|--------|------|------|
| POI 来源 | 高德地图 API（坐标 + 基础信息）+ 大众点评（价格 + 评价）+ 小红书（攻略文本） | 国内数据覆盖好，API 有免费额度 |
| 冷启动 | MVP 手工录入 20 个核心城市的 Top 100 POI | 爬取系统需要时间开发，手工种子数据可立即验证核心算法 |
| 预算货币 | 统一人民币（CNY） | 国内游无需汇率转换 |
| 数据更新 | 冷数据（POI 基础信息）月度刷新，热数据（价格/开放状态）实时查询 | 平衡数据新鲜度和 API 调用成本 |

---

## 二、产品定位与用户画像

### 2.1 一句话定位

**旅行 Agent 是一个面向中国境内自由行用户的智能行程规划系统。它不是"AI 帮你搜攻略"，而是一个能够理解用户的时间/预算/体力/偏好约束，通过算法求解生成可执行行程，并支持多轮局部调整的规划引擎。**

核心差异点：

- **不是文本生成，而是约束求解**：行程规划本质是带时间窗口的多约束优化问题，不能只靠 LLM 生成自然语言。
- **不是单次输出，而是持续对话**：支持多轮修改、局部重规划。
- **不是信息搬运，而是决策辅助**：整合多源数据，帮用户在时间/预算/偏好/舒适度之间做最优权衡。

### 2.2 目标用户画像

#### 用户类型 A：自由行新手

**特征**：第一次去某个城市，不知道去哪、怎么安排、花多少钱。
**核心痛点**：信息过载，从大众点评/小红书/高德地图等碎片信息中整合可执行方案。
**Agent 价值**：一句话输入需求 → 输出完整可执行行程。

#### 用户类型 B：懒人型用户

**特征**：有旅行经验但不想花时间做攻略。
**核心痛点**：攻略制作过程太繁琐，需要跨多个平台查信息、比较、排列组合。
**Agent 价值**：自动完成需求理解 → 偏好抽取 → 行程生成 → 预算估算 → 路线安排 → 方案解释。
**典型输入**：`"我想去成都玩 4 天，预算 3000，喜欢吃辣，别太赶。"`

#### 用户类型 C：精细规划用户

**特征**：需求多、约束复杂，需要精细规划。
**核心痛点**：人脑难以同时处理多个约束条件的权衡。
**典型输入**：`"我想去北京 5 天，第一天中午到，第三天想去长城，酒店最好住二环内，预算 4000，想吃本地人推荐的店。"`

#### 用户类型 D：同行人约束

**特征**：亲子/老人同行，对舒适度要求高。
**核心痛点**：通用攻略不考虑同行人的体力/兴趣差异。
**典型输入**：`"我和爸妈一起去西安，他们走不了太多路。"`

---

## 三、核心场景与 MVP 边界

### 3.1 核心场景（MVP 内）

| 场景 | 优先级 | 说明 |
|------|--------|------|
| 从零生成单城市行程 | P0 | 核心能力，必须可用 |
| 多轮修改行程（单日调整） | P0 | 支持"第三天轻松一点"这类修改 |
| 预算估算与超标提醒 | P0 | 基础预算计算 |
| 路线优化（减少绕行） | P1 | 2-opt 路径优化 |
| 雨天备选方案 | P1 | 天气 MCP 接入后的增值功能 |

### 3.2 后续版本场景（MVP 外）

| 场景 | 计划版本 |
|------|---------|
| 多城市行程（如成都 + 重庆） | v2.0 |
| 旅途中实时位置推荐 | v2.0 |
| 酒店/门票预订对接 | v2.0 |
| 旅后回顾与偏好积累 | v3.0 |
| 行程导出（iCal / PDF） | v1.5 |

### 3.3 场景 1 详细设计：从零生成行程

**用户输入**：`"我想去成都玩 4 天，预算 3000，喜欢吃辣和拍照，不想太累。"`

**处理流程**：

```
1. 信息完整性检查（确定性状态机）
   - destination: 成都 ✓
   - duration: 4 天 ✓
   - budget: 3000 CNY ✓
   - preferences: [美食(辣), 拍照] ✓
   - pace: relaxed（每日景点 ≤ 3，步行 ≤ 12000 步）✓
   - 信息完整度 100% → 进入生成流程

2. 数据获取（Tool Calling via MCP）
   - 调用 POI MCP Server：检索成都美食/拍照相关 POI
   - 调用 Weather MCP Server：查询目标日期天气预报
   - 调用 Map MCP Server：获取 POI 地理坐标
   - 调用 Attraction MCP Server：获取开放时间、门票价格、游玩时长

3. 约束求解（分层启发式算法）
   - 区域聚类：将 POI 按地理位置聚类为日维度组
   - 时间窗口硬约束前置：标记必须上午/下午去的景点
   - 日分配：将聚类结果分配到各天，满足每日时间和强度约束
   - 日内排序：对时间灵活点使用 2-opt 优化路线
   - 时间表生成：插入交通时间、用餐时间、休息时间
   - 约束校验：检查所有约束，不满足则回退调整

4. 行程输出
   - 输出：每日行程 + 推荐理由 + 交通路线 + 预算明细
   - 如果存在约束无法满足（如某景点时间冲突），明确告知用户并提供替代方案
```

### 3.4 场景 2 详细设计：多轮修改行程

**用户输入**：`"第三天太累了，轻松一点。"`

**处理流程**：

```
1. 修改意图解析（确定性解析层）
   - 修改类型：reduce_intensity
   - 影响范围：Day 3（单日，不扩散）
   - 触发约束重推导：max_daily_pois 5 → 3，max_daily_steps 18000 → 12000

2. 局部重规划（仅影响 Day 3）
   - 保留：已确认的高优先级景点
   - 删除/替换：低优先级或高体力消耗景点
   - 重新计算：时间安排、交通路线、预算
   - 约束校验：确保修改后的 Day 3 满足所有约束
   - 全局检查：Day 3 的变化是否导致其他天的约束被破坏（如预算）

3. 变更说明
   - 明确告知用户：修改了什么、保留了什么、为什么这样改、是否影响其他天
   - 示例："删除了武侯祠到锦里的远途步行（减少约 5000 步），替换为附近的人民公园喝茶（与你拍照偏好匹配）。Day 2 和 Day 4 不受影响。"
```

**关键原则**：局部修改，明确影响半径。单日修改不扩散，但修改后需做全局约束校验（如预算是否仍满足）。

---

## 四、系统总体架构

### 4.1 架构核心：双层控制

v2 的问题在于所有控制流都交给 LLM 决策，导致循环终止不可靠、关键路径无保底。v3 引入**双层控制架构**：

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户输入                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 第一层：确定性状态机（保底层）                      │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ 信息完整性   │───▶│ 约束硬校验   │───▶│ 流程路由     │      │
│  │ 检查         │    │（必须满足）  │    │（固定路径）  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
│  职责：                                                         │
│  - 检查 destination / days / budget 等关键信息是否齐全           │
│  - 检查约束是否可被满足（如预算不能为负数）                     │
│  - 决定进入哪个固定流程分支（生成 / 修改 / 追问）               │
│  - 如果信息缺失，直接追问，不进入 LLM 决策                       │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              第二层：LLM 驱动的 Agent Loop（优化层）              │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                 │
│  │ Perceive │───▶│  Reason  │───▶│   Act    │                 │
│  │          │    │(ReAct)   │    │(Tool/   │                 │
│  │ - Parse  │    │          │    │ Sub-    │                 │
│  │   input  │    │          │    │ Agent)  │                 │
│  └──────────┘    └──────────┘    └────┬─────┘                 │
│        ▲                              │                        │
│        │         ┌──────────┐         │                        │
│        └─────────│ Observe  │◀────────┘                        │
│                  │ - Eval   │                                   │
│                  │   result │                                   │
│                  │ - Update │                                   │
│                  │   state  │                                   │
│                  └──────────┘                                   │
│                                                                 │
│  职责：                                                         │
│  - 理解用户自然语言意图                                          │
│  - 决定调用哪个工具 / Sub-Agent                                  │
│  - 生成自然语言回复                                              │
│  - 循环终止由确定性条件控制（非 LLM 判断）                       │
└─────────────────────────────────────────────────────────────────┘
```

**关键区别**：

| 维度 | v2（纯 LLM 决策） | v3（双层控制） |
|------|------------------|---------------|
| 信息完整性 | LLM 决定是否需要追问 | 状态机强制检查，缺失直接追问 |
| 循环终止 | LLM 判断 `is_response_ready()` | 确定性条件（工具调用完成 + 约束满足 + 输出已生成） |
| 约束满足 | LLM "应该"会检查 | 状态机硬校验，不满足直接返回错误 |
| 异常处理 | LLM 自己恢复 | 状态机定义降级路径 |

### 4.2 总体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Client Layer                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐                 │
│  │   Web App    │  │  Mobile Web  │  │  Chat Interface │                │
│  │  (Next.js)   │  │  (响应式)    │  │  (WebSocket)   │                │
│  └──────┬──────┘  └──────┬───────┘  └───────┬────────┘                 │
│         └────────────────┼──────────────────┘                          │
│                          ▼                                              │
│              ┌──────────────────────┐                                   │
│              │    API Gateway        │                                   │
│              │  (FastAPI + WebSocket)│                                   │
│              └──────────┬───────────┘                                   │
└─────────────────────────┼───────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Orchestration Layer                                  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │              确定性状态机（Deterministic State Machine）            │  │
│  │                                                                   │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │  │
│  │  │ Input      │  │ Info       │  │ Constraint │  │ Route      │ │  │
│  │  │ Guardrail  │──▶│ Completeness│──▶│ Hard Check │──▶│ Dispatcher │ │  │
│  │  │            │  │ Check      │  │            │  │            │ │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘ │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │              Agent Loop（LLM 驱动）                                 │  │
│  │                                                                   │  │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐                   │  │
│  │  │ Perceive │───▶│  Reason  │───▶│   Act    │                   │  │
│  │  └──────────┘    └──────────┘    └────┬─────┘                   │  │
│  │       ▲                               │                         │  │
│  │       └──────────┐ ┌──────────┐       │                         │  │
│  │                  └─│ Observe  │◀──────┘                         │  │
│  │                    └──────────┘                                 │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│            ┌─────────────────┼────────────────────┐                    │
│            ▼                 ▼                    ▼                    │
│  ┌──────────────┐ ┌────────────┐ ┌──────────────┐                     │
│  │  State Store  │ │  Guardrail │ │ Trace Logger │                     │
│  │  (Redis /     │ │  (Output   │ │ (Decision    │                     │
│  │   Session)    │ │   Safety)  │ │  Audit Trail)│                     │
│  └──────────────┘ └────────────┘ └──────────────┘                     │
└─────────────────────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼────────────────────┐
        ▼                 ▼                    ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│ Sub-Agent:   │  │ Sub-Agent:   │  │ Sub-Agent:       │
│ Itinerary    │  │ Budget       │  │ Recommendation   │
│ Planner      │  │ Optimizer    │  │ Engine           │
└──────┬───────┘  └──────┬───────┘  └────────┬─────────┘
       │                 │                    │
       └─────────────────┼────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              Tool Layer（MCP Servers + Reliability Layer）                │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  MCP Reliability Framework                       │   │
│  │                                                                   │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │   │
│  │  │ Circuit    │  │ Timeout &  │  │ Fallback   │  │ Local      │ │   │
│  │  │ Breaker    │  │ Retry      │  │ Strategy   │  │ Cache      │ │   │
│  │  │            │  │            │  │            │  │            │ │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐          │
│  │ Map MCP    │ │Weather MCP │ │  POI MCP   │ │ Hotel MCP  │          │
│  │ Server     │ │ Server     │ │  Server    │ │ Server     │          │
│  │ (高德/百度) │ │ (彩云天气) │ │ (高德+点评)│ │ (携程/美团)│          │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Knowledge Layer（RAG）                               │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   Hybrid Retrieval Pipeline                      │   │
│  │  Query ──▶ Embedding Search + BM25 Search ──▶ Cross-Encoder     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ POI 知识库    │  │ 城市攻略库    │  │ 用户评论库    │                  │
│  │ (结构化)     │  │ (文档)       │  │ (摘要)       │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 循环终止条件（确定性）

v2 的循环终止由 LLM 判断，不可靠。v3 的终止条件是**纯确定性**的：

```python
class AgentLoopTerminator:
    """
    循环终止条件：所有条件必须同时满足才能终止。
    这些条件由确定性代码检查，不由 LLM 判断。
    """

    def should_stop(self, state: TravelState) -> bool:
        # 条件 1：至少生成了一版行程（或已确认无法生成）
        has_itinerary = state.current_itinerary is not None
        
        # 条件 2：本轮所需的工具调用全部完成（无待调用）
        no_pending_tools = len(state.pending_tool_calls) == 0
        
        # 条件 3：约束硬校验通过（或已明确告知用户无法满足）
        constraint_result = self.hard_constraint_check(state)
        constraints_ok = constraint_result.passed or constraint_result.user_notified
        
        # 条件 4：输出已生成（自然语言回复已准备好）
        response_ready = state.response_draft is not None
        
        return has_itinerary and no_pending_tools and constraints_ok and response_ready
    
    def hard_constraint_check(self, state: TravelState) -> ConstraintCheckResult:
        """
        硬约束校验：这些约束不满足，行程不能输出。
        """
        violations = []
        
        # 硬约束 1：预算不能超过上限（如果有上限）
        if state.constraints.max_budget:
            total_cost = sum(day.total_cost for day in state.current_itinerary or [])
            if total_cost > state.constraints.max_budget:
                violations.append("budget_exceeded")
        
        # 硬约束 2：所有景点必须在开放时间内
        for day in state.current_itinerary or []:
            for activity in day.activities:
                if not self._is_within_opening_hours(activity):
                    violations.append(f"{activity.poi_name}_outside_hours")
        
        # 硬约束 3：must_include 景点必须被安排
        if state.current_itinerary:
            included = set()
            for day in state.current_itinerary:
                for a in day.activities:
                    included.add(a.poi_id)
            for must in state.constraints.must_include_pois:
                if must not in included:
                    violations.append(f"must_include_missing_{must}")
        
        return ConstraintCheckResult(
            passed=len(violations) == 0,
            violations=violations,
            user_notified=state.user_notified_of_violations
        )
```

---

## 五、Agent 编排与通信协议

### 5.1 MCP 集成 + 可靠性框架

v2 的 MCP 只有接口定义，没有可靠性设计。v3 为每个 MCP Server 包装可靠性层：

```python
class ReliableMCPClient:
    """
    带可靠性保障的 MCP 客户端包装器。
    每个工具调用都经过：超时控制 → 重试 → 降级 → 缓存。
    """
    
    def __init__(self, server_name: str, client: MCPClient):
        self.server_name = server_name
        self.client = client
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,      # 5 次失败后开启断路器
            recovery_timeout=60,      # 60 秒后尝试恢复
        )
        self.cache = LocalCache(ttl=300)  # 5 分钟缓存
    
    async def call(self, tool_name: str, params: dict) -> ToolResult:
        cache_key = f"{tool_name}:{hash(str(params))}"
        
        # 1. 检查缓存
        cached = self.cache.get(cache_key)
        if cached:
            return ToolResult(data=cached, source="cache")
        
        # 2. 检查断路器
        if self.circuit_breaker.is_open():
            return await self._fallback(tool_name, params)
        
        # 3. 调用（带超时和重试）
        try:
            result = await asyncio.wait_for(
                self._call_with_retry(tool_name, params),
                timeout=10.0  # 10 秒超时
            )
            self.circuit_breaker.record_success()
            self.cache.set(cache_key, result)
            return ToolResult(data=result, source="live")
            
        except asyncio.TimeoutError:
            self.circuit_breaker.record_failure()
            return await self._fallback(tool_name, params)
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            return await self._fallback(tool_name, params)
    
    async def _fallback(self, tool_name: str, params: dict) -> ToolResult:
        """
        分级降级策略：
        1. 先用缓存数据（即使已过期）
        2. 再用 RAG 知识库中的历史数据
        3. 最后返回降级提示，标记数据可能不准确
        """
        # 尝试用过期缓存
        stale = self.cache.get_stale(f"{tool_name}:{hash(str(params))}", max_age=86400)
        if stale:
            return ToolResult(data=stale, source="cache_stale", warning="数据可能已过期")
        
        # 尝试 RAG 兜底
        rag_result = await self._query_rag_fallback(tool_name, params)
        if rag_result:
            return ToolResult(data=rag_result, source="rag_fallback", warning="基于历史数据估算")
        
        # 完全失败
        return ToolResult(
            data=None,
            source="failed",
            error=f"{self.server_name}.{tool_name} 暂时不可用，请稍后重试"
        )
```

#### MCP Server 定义示例

```python
# map_mcp_server.py - 基于高德/百度地图
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("travel-map")

@server.tool()
async def search_poi(
    city: str,
    keyword: str,
    category: str | None = None,
    radius_km: float = 5.0
) -> list[dict]:
    """
    在指定城市搜索兴趣点。
    
    Args:
        city: 城市名称（如 "成都"、"北京"）
        keyword: 搜索关键词（如 "火锅"、"古镇"）
        category: POI 类别过滤（可选，如 "餐饮"、"景点"、"购物"）
        radius_km: 搜索半径（千米），默认 5km
    
    Returns:
        POI 列表，包含 name, lat, lng, category, rating, review_count,
        opening_hours, estimated_duration, ticket_price
    """
    # 优先查询本地缓存/数据库，缺失时调用高德 API
    results = await amap_api.search_poi(
        city=city,
        keywords=keyword,
        types=category,
        radius=radius_km * 1000
    )
    return [format_poi(r) for r in results]

@server.tool()
async def get_distance_matrix(
    origins: list[dict],
    destinations: list[dict],
    mode: str = "transit"
) -> dict:
    """
    计算多个起终点之间的距离和时间矩阵。
    mode: "transit" | "driving" | "walking" | "riding"
    
    注意：高德 Distance Matrix API 有配额限制，结果会被缓存。
    """
    return await amap_api.distance_matrix(origins, destinations, mode)

@server.tool()
async def get_route(
    origin: dict,
    destination: dict,
    mode: str = "transit",
    strategy: str = "fastest"
) -> dict:
    """获取两点之间的详细路线。"""
    return await amap_api.direction(origin, destination, mode, strategy)
```

```python
# poi_mcp_server.py - 整合高德 + 大众点评
@server.tool()
async def get_poi_detail(poi_id: str) -> dict:
    """
    获取 POI 详细信息，整合多个数据源：
    - 高德地图：坐标、地址、分类
    - 大众点评：评分、人均消费、营业时间
    - 本地数据库：游玩时长、最佳时间、适合人群
    """
    # 并行查询多个数据源
    amap_task = amap_api.get_detail(poi_id)
    dianping_task = dianping_api.get_detail(poi_id)
    local_task = local_db.get_poi(poi_id)
    
    amap_data, dianping_data, local_data = await asyncio.gather(
        amap_task, dianping_task, local_task,
        return_exceptions=True
    )
    
    return merge_poi_data(amap_data, dianping_data, local_data)
```

### 5.2 Multi-Agent 协作架构

v2 的 Sub-Agent 协作流程是串行的，v3 改为**按需并行**：

```
Orchestrator 接收用户请求
    │
    ├── Step 1: 确定性状态机检查
    │   └── 信息完整？→ 进入生成流程 / 追问
    │
    ├── Step 2: 并行调用（独立任务）
    │   ├── Recommendation Agent ──▶ "推荐 15-20 个候选 POI"
    │   ├── Weather MCP ───────────▶ "查询目标日期天气"
    │   └── 等待全部完成后继续
    │
    ├── Step 3: 串行调用（有依赖关系）
    │   └── Itinerary Planner Agent ──▶ "用候选 POI 规划行程"
    │
    ├── Step 4: 并行校验
    │   ├── Budget Optimizer Agent ──▶ "预算校验"
    │   ├── Constraint Checker ──────▶ "约束硬校验"
    │   └── 等待全部完成后继续
    │
    ├── Step 5: 确定性检查
    │   ├── 预算超标？→ 要求 Budget Agent 提供优化方案
    │   ├── 约束违反？→ 要求 Planner Agent 重新规划
    │   ├── 都满足？→ 进入下一步
    │
    └── Step 6: Orchestrator 整合结果，生成用户回复
```

**并行规则**：
- 无依赖的任务可并行：POI 推荐、天气查询、地图数据获取
- 有依赖的任务必须串行：行程规划依赖候选 POI 列表
- 校验任务可并行：预算校验和约束校验互不依赖

### 5.3 Sub-Agent 通信协议

```python
@dataclass
class AgentTask:
    task_id: str
    sender: str
    receiver: str
    action: str
    payload: dict
    constraints: dict
    deadline_ms: int = 30000  # 默认 30 秒超时
    
@dataclass
class AgentTaskResult:
    task_id: str
    status: str  # "completed" | "failed" | "partial"
    result: dict
    confidence: float  # 0.0 - 1.0，由客观指标计算
    trace: list[str]
    fallback_used: bool  # 是否使用了降级数据
```

---

## 六、模块详细设计

### 6.1 确定性状态机（Deterministic State Machine）

**新增模块**，解决 v2 完全依赖 LLM 决策的问题。

```python
class DeterministicStateMachine:
    """
    确定性状态机：负责关键路径的保底控制。
    所有判断都是代码逻辑，不调用 LLM。
    """
    
    # 关键信息完整度检查
    REQUIRED_FIELDS = ["destination", "travel_days", "budget"]
    OPTIONAL_FIELDS_WITH_DEFAULTS = {
        "travelers_count": 1,
        "pace": "moderate",
        "companions": [],
        "departure_city": None,
    }
    
    def check_completeness(self, state: TravelState) -> CompletenessResult:
        """检查信息是否足够生成行程。"""
        missing = []
        for field in self.REQUIRED_FIELDS:
            if getattr(state.user_profile, field) is None:
                missing.append(field)
        
        if missing:
            return CompletenessResult(
                complete=False,
                missing_fields=missing,
                action="ask_clarification"
            )
        
        return CompletenessResult(complete=True, action="proceed")
    
    def route_flow(self, state: TravelState, user_input: str) -> FlowDecision:
        """决定进入哪个处理分支。"""
        
        # 分支 1：首次规划（无现有行程）
        if state.current_itinerary is None:
            completeness = self.check_completeness(state)
            if not completeness.complete:
                return FlowDecision(action="ask_clarification", questions=completeness.missing_fields)
            return FlowDecision(action="generate_itinerary")
        
        # 分支 2：已有行程，判断是修改还是新需求
        # 简单启发式：如果提到"天"、"行程"、"安排"等词，视为修改
        if any(kw in user_input for kw in ["天", "行程", "安排", "改", "换", "删", "加"]):
            return FlowDecision(action="modify_itinerary")
        
        # 分支 3：其他（天气查询、预算询问等）
        return FlowDecision(action="general_query")
    
    def hard_constraint_check(self, state: TravelState) -> HardCheckResult:
        """
        硬约束校验：这些约束不满足，行程不能输出给用户。
        返回的 violations 会明确告知用户。
        """
        violations = []
        itinerary = state.current_itinerary
        
        if not itinerary:
            return HardCheckResult(passed=False, violations=["no_itinerary_generated"])
        
        # 硬约束 1：预算
        if state.constraints.max_budget:
            total = sum(day.total_cost for day in itinerary)
            if total > state.constraints.max_budget:
                violations.append({
                    "type": "budget_exceeded",
                    "detail": f"总费用 {total} 元，超出预算 {state.constraints.max_budget} 元",
                    "severity": "high"
                })
        
        # 硬约束 2：must_include 景点
        included = set()
        for day in itinerary:
            for a in day.activities:
                included.add(a.poi_id)
        for must_id in state.constraints.must_include_pois:
            if must_id not in included:
                violations.append({
                    "type": "must_include_missing",
                    "detail": f"必去景点 {must_id} 未被安排",
                    "severity": "high"
                })
        
        # 硬约束 3：开放时间
        for day in itinerary:
            for activity in day.activities:
                if not self._is_within_opening_hours(activity, day.date):
                    violations.append({
                        "type": "outside_opening_hours",
                        "detail": f"{activity.poi_name} 在 {activity.start_time} 不开放",
                        "severity": "high"
                    })
        
        return HardCheckResult(
            passed=len(violations) == 0,
            violations=violations
        )
```

### 6.2 State Manager（版本化）

v2 的 State 缺少并发控制和版本管理。v3 引入版本化：

```python
@dataclass
class TravelState:
    # 会话元数据
    session_id: str
    version: int = 1  # 乐观锁版本号
    created_at: str
    updated_at: str
    turn_count: int
    
    # 用户画像（渐进式收集）
    user_profile: UserProfile
    
    # 当前行程
    current_itinerary: list[DayPlan] | None
    
    # 约束条件
    constraints: Constraints
    
    # 交互历史
    confirmed_items: list[str]
    rejected_items: list[str]
    modification_history: list[Modification]
    
    # 对话历史（最近 20 轮，完整历史存数据库）
    recent_messages: list[Message]
    
    # 运行时状态（不持久化）
    pending_tool_calls: list[str]  # 当前待完成的工具调用
    response_draft: str | None     # 待输出的回复草稿
    user_notified_of_violations: bool = False

class StateManager:
    """
    状态管理器。核心原则：
    1. 版本控制：每次更新 version + 1，并发修改时检测版本冲突
    2. 不可变历史：每次修改保存 before/after 快照
    3. 局部更新：修改只影响变更部分
    4. 约束推导：从用户画像自动推导约束参数
    """
    
    def __init__(self, redis_client: Redis, db_client: Postgres):
        self.redis = redis_client
        self.db = db_client
    
    async def load(self, session_id: str) -> TravelState:
        """从 Redis 加载 State，Redis miss 时从数据库加载。"""
        data = await self.redis.get(f"state:{session_id}")
        if data:
            return TravelState(**json.loads(data))
        # 从数据库加载完整历史
        return await self._load_from_db(session_id)
    
    async def save(self, state: TravelState, expected_version: int | None = None) -> bool:
        """
        保存 State，支持乐观锁。
        expected_version: 如果提供，只有当当前版本等于 expected_version 时才保存。
        """
        if expected_version is not None and state.version != expected_version:
            # 版本冲突，加载最新版本后重试
            latest = await self.load(state.session_id)
            return False, latest
        
        state.version += 1
        state.updated_at = datetime.now().isoformat()
        
        # 写入 Redis（TTL 7 天）
        await self.redis.setex(
            f"state:{state.session_id}",
            timedelta(days=7),
            json.dumps(state, default=str)
        )
        
        # 每 10 轮持久化到数据库
        if state.turn_count % 10 == 0:
            await self._persist_to_db(state)
        
        return True, state
    
    async def apply_modification(self, session_id: str, mod: Modification) -> tuple[bool, TravelState]:
        """
        应用行程修改，保存历史快照。
        如果版本冲突，返回 False 和最新 State。
        """
        state = await self.load(session_id)
        current_version = state.version
        
        # 保存修改前快照
        mod.before_snapshot = deepcopy(state.current_itinerary)
        mod.timestamp = datetime.now().isoformat()
        
        # 应用修改
        self._apply_mod_to_state(state, mod)
        
        # 保存修改后快照
        mod.after_snapshot = deepcopy(state.current_itinerary)
        state.modification_history.append(mod)
        
        success, new_state = await self.save(state, expected_version=current_version)
        return success, new_state
```

### 6.3 Orchestrator Agent

```python
ORCHESTRATOR_SYSTEM_PROMPT = """
你是一个旅行规划系统的 Orchestrator。你的职责是理解用户需求，调用合适的工具和子系统来完成任务。

## 可用的 Sub-Agent

1. **recommendation_agent**: 根据用户偏好推荐 POI
2. **itinerary_planner_agent**: 生成/修改行程
3. **budget_optimizer_agent**: 预算分析与优化

## 决策原则

- 信息不足时，状态机会自动追问，你不会收到不完整的信息
- 修改行程时，只改受影响的部分（单日），不全量重写
- 每个推荐都附带理由
- 如果 Sub-Agent 返回的结果不满足约束，重新调用并说明调整方向
- 保持对话自然流畅，不暴露内部技术细节
- 如果使用了降级数据（如缓存过期数据），必须明确告知用户

## 当前状态
{state_json}

## 本轮任务
{task_description}
"""
```

### 6.4 Itinerary Planner Agent

```python
class ItineraryPlannerAgent:
    """
    行程规划 Sub-Agent。
    核心流程：区域聚类 → 时间窗口硬约束标记 → 日分配 → 日内排序 → 时间表生成 → 约束校验。
    """
    
    async def plan(
        self,
        pois: list[ScoredPOI],
        num_days: int,
        constraints: Constraints,
        hotel_location: dict | None = None
    ) -> list[DayPlan]:
        
        # Step 1: 区域聚类（DBSCAN，eps 按城市密度动态调整）
        clusters, noise = self.cluster_pois(pois, constraints)
        
        # Step 2: 时间窗口硬约束标记
        # 标记哪些景点必须在上午/下午/晚上去
        time_constrained = self._mark_time_constraints(pois)
        
        # Step 3: 日分配（考虑偏好、must-include、时间窗口）
        days = self.assign_to_days(clusters, noise, num_days, constraints, time_constrained)
        
        # Step 4: 日内排序（时间约束点固定，灵活点用 2-opt）
        for day in days:
            day.activities = await self.optimize_daily_route(
                day.activities,
                hotel_location,
                constraints
            )
        
        # Step 5: 时间表生成
        for day in days:
            day.activities = self.build_schedule(
                day.activities,
                constraints,
                day.date
            )
        
        # Step 6: 约束校验
        violations = self.constraint_checker.check(days, constraints)
        if violations:
            # 尝试自动修复
            days = self.attempt_auto_fix(days, violations, constraints)
            # 仍有 violations？记录下来，让 Orchestrator 告知用户
        
        return days
```

### 6.5 Budget Optimizer Agent

```python
class BudgetOptimizerAgent:
    """
    预算优化 Sub-Agent。
    所有预算统一为人民币（CNY）。
    """
    
    async def estimate(self, itinerary: list[DayPlan], profile: UserProfile) -> BudgetBreakdown:
        """
        分项预算估算。数据来源优先级：
        1. MCP Tool 实时查询（酒店价格、门票价格）
        2. RAG 知识库（历史价格数据）
        3. 基于城市的统计均值（兜底）
        
        注意：所有金额统一为 CNY，无需汇率转换。
        """
        accommodation = await self._estimate_accommodation(itinerary, profile)
        meals = self._estimate_meals(itinerary, profile)
        transport = await self._estimate_transport(itinerary)
        tickets = self._sum_ticket_costs(itinerary)
        shopping = self._estimate_shopping(profile)
        
        # 城市消费水平系数（一线 1.2，二线 1.0，三线 0.8）
        city_factor = self._get_city_consumption_factor(profile.destination)
        
        total = (accommodation + meals + transport + tickets + shopping) * city_factor
        
        # 动态 buffer：一线城市 15%，二线 12%，三线 10%
        buffer_ratio = {1: 0.15, 2: 0.12, 3: 0.10}.get(city_tier, 0.12)
        
        return BudgetBreakdown(
            accommodation=accommodation,
            meals=meals,
            transport=transport,
            tickets=tickets,
            shopping=shopping,
            total=total,
            buffer=total * buffer_ratio,
            confidence="medium"
        )
```

### 6.6 Guardrail Module

```python
class InputGuardrail:
    """输入安全检查。"""
    
    def check(self, user_input: str) -> GuardrailResult:
        # 1. Prompt injection 检测（关键词 + 模式匹配）
        injection_patterns = [
            r"ignore previous instructions",
            r"system prompt",
            r"you are now",
            r"DAN mode",
            r"角色扮演",
            r"忽略.*指令",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                return GuardrailResult(safe=False, reason="detected_injection", action="reject")
        
        # 2. 话题边界检测（Embedding 相似度）
        travel_similarity = self._calc_travel_similarity(user_input)
        if travel_similarity < 0.3:
            return GuardrailResult(
                safe=True, reason="off_topic", action="redirect",
                message="我是旅行规划助手，让我帮你规划旅行吧！"
            )
        
        # 3. 敏感目的地/活动检测
        if self._contains_sensitive_content(user_input):
            return GuardrailResult(
                safe=True, reason="sensitive_content", action="warn",
                message="该地区/活动存在安全风险，建议查阅最新旅行安全提示。"
            )
        
        return GuardrailResult(safe=True, action="pass")

class OutputGuardrail:
    """输出质量检查（同步执行，在返回用户前完成）。"""
    
    async def check(self, response: AgentResponse) -> OutputCheckResult:
        issues = []
        
        # 1. POI 真实性验证（异步预取 + 同步校验）
        for activity in response.activities:
            if activity.source == "llm":
                exists = await self.poi_mcp.verify_existence(activity.poi_name)
                if not exists:
                    issues.append(Issue(
                        type="hallucination", severity="high",
                        target=activity.poi_name, action="remove_or_replace"
                    ))
        
        # 2. 数据时效性标注
        for activity in response.activities:
            if activity.data_freshness_days > 30:
                issues.append(Issue(
                    type="stale_data", severity="low",
                    target=activity.poi_name, action="add_disclaimer"
                ))
        
        # 3. 数据降级提醒
        if response.fallback_data_used:
            issues.append(Issue(
                type="fallback_data", severity="medium",
                target="global", action="add_fallback_notice"
            ))
        
        return OutputCheckResult(issues=issues, pass_rate=len(issues) == 0)
```

---

## 七、数据层设计

v2 完全没有数据层设计。v3 新增独立章节。

### 7.1 数据源矩阵

| 数据类型 | 来源 | 获取方式 | 更新频率 | 可靠性 |
|---------|------|---------|---------|--------|
| POI 坐标/地址 | 高德地图 API | API 调用 | 实时 | 高 |
| POI 评分/评论数 | 大众点评 API | API 调用 | 实时 | 中 |
| 门票价格 | 携程/美团 API | API 调用 | 实时 | 中 |
| 开放时间 | 高德 + 手工校验 | API + 手工 | 月度 | 高 |
| 游玩时长 | 手工录入 + 用户反馈 | 手工 | 季度 | 高 |
| 攻略文本 | 小红书/马蜂窝 | 爬取 + 编辑 | 月度 | 中 |
| 天气数据 | 彩云天气 API | API 调用 | 实时 | 高 |
| 距离/路线 | 高德地图 API | API 调用 | 实时 | 高 |

### 7.2 冷启动策略

**MVP 阶段**：手工种子数据

```
覆盖城市：20 个核心城市（一线 + 热门旅游城）
每个城市：Top 100 POI（手工录入）
数据字段：name, lat, lng, category, rating, ticket_price, opening_hours, recommended_duration_min, tags
录入时间：约 2-3 天（2 人并行）
```

**种子数据来源**：
- 高德地图搜索 + 大众点评详情页（人工整理）
- 携程景点榜单
- 小红书热门攻略（人工提炼）

**Phase 2**：自动化数据补充
- 开发爬虫系统，自动补充每个城市的 POI 到 500-1000 个
- 建立数据质量审核流程

### 7.3 数据质量保障

```python
class DataQualityManager:
    """数据质量保证。"""
    
    FRESHNESS_RULES = {
        "poi_basic": {"stale_days": 90, "aging_days": 30},
        "poi_price": {"stale_days": 30, "aging_days": 7},
        "poi_hours": {"stale_days": 30, "aging_days": 14},
        "policy": {"stale_days": 30, "aging_days": 7},
    }
    
    def check_freshness(self, poi: POI) -> FreshnessStatus:
        """检查数据新鲜度。"""
        age_days = (datetime.now() - poi.last_verified).days
        
        if poi.data_source == "api":
            if age_days > 7:
                return FreshnessStatus.AGING
        elif poi.data_source == "manual":
            if age_days > 30:
                return FreshnessStatus.AGING
            if age_days > 90:
                return FreshnessStatus.STALE
        
        return FreshnessStatus.FRESH
    
    async def refresh_priority_data(self):
        """
        定时任务：优先刷新高频访问城市的热门 POI。
        策略：
        1. 根据查询日志确定高频 POI
        2. 调用 API 刷新价格和开放状态
        3. 手工录入的数据标记为"待校验"
        """
        pass
```

### 7.4 POI 数据模型（简化版）

```typescript
interface POI {
  id: string;
  name: string;
  name_en: string | null;
  
  city: string;
  district: string;
  address: string;
  location: { lat: number; lng: number };
  
  category: "attraction" | "restaurant" | "shopping" | "hotel" | "entertainment";
  subcategory: string;
  tags: string[];
  themes: string[];
  
  recommended_duration_min: number;
  best_time_of_day: ("morning" | "afternoon" | "evening")[];
  intensity: "low" | "medium" | "high";
  is_outdoor: boolean;
  
  suitable_for: string[];
  accessibility: "full" | "partial" | "none";
  
  ticket_price: number | null;  // CNY
  is_free: boolean;
  average_meal_cost: number | null;  // CNY，仅餐厅
  
  opening_hours: {
    [day: string]: { open: string; close: string } | "closed" | "24h"
  };
  closed_dates: string[];  // 特殊闭馆日，如 "2026-05-01"
  reservation_required: boolean;
  
  rating: number;  // 0-5
  review_count: number;
  
  data_source: "amap" | "dianping" | "manual" | "mixed";
  last_verified: string;
  data_confidence: "high" | "medium" | "low";
  
  description: string;
  tips: string;
}
```

---

## 八、约束求解与路径优化算法

v2 的算法层有 4 个严重问题。v3 逐一修正。

### 8.1 问题建模

旅行规划建模为**带时间窗口的多约束优化问题**：

```
给定：
  - POI 集合 P = {p1, p2, ..., pn}，每个 POI 有：
    - 地理坐标 (lat_i, lng_i)
    - 游玩时长 d_i
    - 开放时间窗口 [o_i, c_i]（可能因日期不同）
    - 门票费用 cost_i
    - 强度等级 intensity_i
    - 偏好匹配度 pref_i
    - 时间约束类型 time_constraint ∈ {flexible, morning_only, afternoon_only, evening_only}
  - 天数 D
  - 每天可用时间窗口 [T_start, T_end]
  - 约束集合 C:
    - 总预算 B_max
    - 每日最大景点数 N_max
    - 每日最大步行量 S_max
    - 每日最大交通时间 TR_max
    - must_include 景点集合 M
  - 距离/时间矩阵 dist[i][j], time[i][j]（按交通模式）

目标：
  找到 POI 子集 P' ⊆ P 和分配到 D 天的方案，使得：
  1. 最大化总偏好匹配度 Σ pref_i
  2. 最小化总移动距离 Σ dist
  3. 满足所有约束 C
  4. must_include 景点必须被安排
```

### 8.2 分层求解算法（修正版）

#### Layer 1: 区域聚类（Spatial Clustering）

v2 问题：eps=1.5km 一刀切。
v3 修正：按城市类型动态调整 eps。

```python
def cluster_pois(pois: list[POI], city_type: str) -> tuple[list[list[POI]], list[POI]]:
    """
    使用 DBSCAN 对 POI 进行地理聚类。
    
    eps 按城市类型动态调整：
    - 密集城区（如北京二环内、上海内环）：eps = 1.0km
    - 一般城区（如成都市区、杭州市区）：eps = 1.5km
    - 郊区/景区（如长城、千岛湖）：eps = 3.0km
    
    min_samples = 2（至少 2 个景点形成一个区域）
    """
    eps_map = {
        "dense": 1.0,
        "normal": 1.5,
        "suburban": 3.0,
    }
    eps = eps_map.get(city_type, 1.5)
    
    coords = np.array([[p.location.lat, p.location.lng] for p in pois])
    clustering = DBSCAN(
        eps=eps / 6371,  # 转换为弧度
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

#### Layer 2: 时间窗口硬约束标记

v2 缺失：未区分时间灵活/不灵活的景点。
v3 新增：在聚类后、日分配前，先标记时间约束。

```python
def mark_time_constraints(pois: list[POI]) -> dict[str, str]:
    """
    标记每个 POI 的时间约束类型。
    返回: {poi_id: constraint_type}
    constraint_type: "flexible" | "morning_only" | "afternoon_only" | "evening_only"
    """
    constraints = {}
    for poi in pois:
        if "morning" in poi.best_time_of_day and len(poi.best_time_of_day) == 1:
            constraints[poi.id] = "morning_only"
        elif "evening" in poi.best_time_of_day and len(poi.best_time_of_day) == 1:
            constraints[poi.id] = "evening_only"
        else:
            constraints[poi.id] = "flexible"
    return constraints
```

#### Layer 3: 日分配（Day Assignment）

v2 问题：只按簇大小分配，未考虑偏好和 must-include。
v3 修正：多目标贪心分配。

```python
def assign_to_days(
    clusters: list[list[POI]],
    noise_pois: list[POI],
    num_days: int,
    constraints: Constraints,
    time_constraints: dict[str, str]
) -> list[list[POI]]:
    """
    将聚类结果分配到各天。
    
    分配策略（按优先级排序）：
    1. must_include 景点优先分配到用户指定日期（如"第三天去长城"）
    2. 时间约束景点分配到合适时段（morning_only → 上午有容量的天）
    3. 大簇优先分配到独立天
    4. 小簇合并（如果距离合理且偏好互补）
    5. 噪声点填充到有空余的天
    """
    days = [[] for _ in range(num_days)]
    day_durations = [0] * num_days
    day_morning_capacity = [1] * num_days  # 每天上午容量（标记是否已用）
    max_capacity = constraints.max_daily_duration_min
    
    # Step 1: 处理 must_include
    for must_id in constraints.must_include_pois:
        poi = find_poi_by_id(must_id)
        target_day = constraints.must_include_day_map.get(must_id)  # 用户指定的天
        if target_day is not None and target_day < num_days:
            days[target_day].append(poi)
            day_durations[target_day] += poi.recommended_duration_min
    
    # Step 2: 处理时间约束景点
    for poi in [p for cluster in clusters for p in cluster] + noise_pois:
        if poi.id in [m for m in constraints.must_include_pois]:
            continue  # 已处理
        
        tc = time_constraints.get(poi.id, "flexible")
        if tc == "morning_only":
            # 找到上午有容量且剩余时间最多的天
            candidates = [d for d in range(num_days) if day_morning_capacity[d] > 0]
            if candidates:
                best_day = max(candidates, key=lambda d: max_capacity - day_durations[d])
                days[best_day].append(poi)
                day_durations[best_day] += poi.recommended_duration_min
                day_morning_capacity[best_day] = 0
        # afternoon_only / flexible 类似处理...
    
    # Step 3: 普通簇分配
    remaining_clusters = [c for c in clusters if not any(p.id in [x.id for x in sum(days, [])] for p in c)]
    remaining_clusters.sort(key=len, reverse=True)
    
    for cluster in remaining_clusters:
        cluster_duration = sum(p.recommended_duration_min for p in cluster)
        best_day = min(range(num_days), key=lambda d: day_durations[d])
        
        if day_durations[best_day] + cluster_duration <= max_capacity:
            days[best_day].extend(cluster)
            day_durations[best_day] += cluster_duration
        else:
            # 簇太大，拆分填充
            for poi in cluster:
                best_day = min(range(num_days), key=lambda d: day_durations[d])
                if day_durations[best_day] + poi.recommended_duration_min <= max_capacity:
                    days[best_day].append(poi)
                    day_durations[best_day] += poi.recommended_duration_min
    
    return days
```

#### Layer 4: 日内路径优化（Intra-day TSP with Time Windows）

v2 问题：只优化距离，忽略时间窗口；只考虑单向。
v3 修正：时间约束点固定位置，灵活点用 2-opt；考虑返回酒店。

```python
async def optimize_daily_route(
    activities: list[Activity],
    hotel_location: dict | None,
    constraints: Constraints,
    map_mcp: ReliableMCPClient
) -> list[Activity]:
    """
    优化日内路线。
    
    策略：
    1. 时间约束点（morning_only/afternoon_only）固定位置，不参与优化
    2. 灵活点用 2-opt 优化
    3. 考虑从酒店出发并返回酒店（Closed TSP）
    4. 混合交通：步行段用步行距离，跨区域用公共交通时间
    """
    if len(activities) <= 2:
        return activities
    
    # 分离固定点和灵活点
    fixed = [a for a in activities if a.time_constraint != "flexible"]
    flexible = [a for a in activities if a.time_constraint == "flexible"]
    
    # 如果所有点都是固定的，无需优化
    if not flexible:
        return sorted(activities, key=lambda a: a.start_time or "")
    
    # 构建点列表（酒店 + 所有灵活点 + 酒店）
    points = ([hotel_location] if hotel_location else []) + \
             [{"lat": a.location.lat, "lng": a.location.lng} for a in flexible] + \
             ([hotel_location] if hotel_location else [])
    
    # 获取距离矩阵（按交通模式）
    # 同一区域内（<1km）用步行，其他用公共交通
    dist_matrix = await map_mcp.call("get_distance_matrix", {
        "origins": points,
        "destinations": points,
        "mode": "transit"  # 公共交通时间
    })
    
    # 2-opt 优化（仅对灵活点）
    n = len(flexible)
    route = list(range(n))  # 初始顺序
    
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                delta = calc_2opt_delta(dist_matrix, route, i, j)
                if delta < 0:
                    route[i+1:j+1] = reversed(route[i+1:j+1])
                    improved = True
    
    # 合并固定点和灵活点
    # 固定点按时间窗口排序，灵活点按优化后的顺序插入
    result = []
    fixed_sorted = sorted(fixed, key=lambda a: a.earliest_start or "")
    flexible_ordered = [flexible[i] for i in route]
    
    # 简单合并：先放 morning_only 固定点，然后灵活点，然后 afternoon_only 固定点
    for f in fixed_sorted:
        if f.time_constraint == "morning_only":
            result.append(f)
    for fl in flexible_ordered:
        result.append(fl)
    for f in fixed_sorted:
        if f.time_constraint == "afternoon_only":
            result.append(f)
    
    return result
```

#### Layer 5: 时间表生成（Schedule Builder）

v2 问题：直接 `continue` 跳过无法安排的景点。
v3 修正：尝试回退，无法安排时明确告知。

```python
def build_schedule(
    ordered_activities: list[Activity],
    day_start: str,  # "09:00"
    day_end: str,    # "21:00"
    constraints: Constraints,
    transport_times: list[int],
    date: str
) -> list[ScheduledActivity]:
    """
    将有序活动列表映射为带时间的日程表。
    
    规则：
    1. 遵守景点开放时间窗口（硬约束）
    2. 午餐 11:30-13:30，晚餐 17:30-19:30
    3. 如有 rest_time 约束，在下午安排休息
    4. 如果某景点无法在开放时间内到达：尝试延后 → 尝试提前 → 标记为冲突
    5. 冲突景点不删除，保留在结果中但标记为"需调整"
    """
    schedule = []
    current_time = parse_time(day_start)
    unscheduled = []  # 记录无法安排的景点
    
    for i, activity in enumerate(ordered_activities):
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
        meal = check_and_insert_meal(current_time, activity.location, schedule, constraints)
        if meal:
            schedule.append(meal)
            current_time += timedelta(minutes=meal.duration_min)
        
        # 检查景点开放时间
        if not is_open_at(activity, current_time, date):
            # 尝试延后
            open_time = get_open_time(activity, date)
            if open_time and open_time > current_time and open_time < parse_time(day_end):
                wait_min = (open_time - current_time).seconds // 60
                if wait_min <= 60:  # 最多等 1 小时
                    schedule.append(WaitActivity(duration_min=wait_min))
                    current_time = open_time
                else:
                    # 等太久，标记为冲突
                    unscheduled.append({
                        "activity": activity,
                        "reason": f"开放时间 {format_time(open_time)} 与当前时间 {format_time(current_time)} 差距过大"
                    })
                    continue
            else:
                # 今天无法安排
                unscheduled.append({
                    "activity": activity,
                    "reason": f"在 {date} 不开放或已关门"
                })
                continue
        
        # 安排景点
        end_time = current_time + timedelta(minutes=activity.duration_min)
        if end_time > parse_time(day_end):
            unscheduled.append({
                "activity": activity,
                "reason": f"预计结束时间 {format_time(end_time)} 超出当天结束时间"
            })
            continue
        
        schedule.append(ScheduledActivity(
            activity=activity,
            start_time=format_time(current_time),
            end_time=format_time(end_time),
            duration_min=activity.duration_min
        ))
        current_time = end_time
    
    # 将未安排的景点以特殊形式加入结果（不删除）
    for item in unscheduled:
        schedule.append(UnscheduledActivity(
            activity=item["activity"],
            reason=item["reason"],
            suggestion=f"建议调整到 {suggest_alternative_day(item['activity'])}"
        ))
    
    return schedule
```

#### Layer 6: 约束校验（Constraint Checker）

```python
class ConstraintChecker:
    """对生成的行程进行全面约束校验。"""
    
    def check(self, itinerary: list[DayPlan], constraints: Constraints) -> list[Violation]:
        violations = []
        
        # 1. 预算约束
        total_cost = sum(day.total_cost for day in itinerary)
        if constraints.max_budget and total_cost > constraints.max_budget:
            violations.append(Violation(
                type="budget_exceeded",
                severity="high",
                detail=f"总费用 {total_cost} 元，超出预算 {constraints.max_budget} 元",
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
                    detail=f"Day {day.day_number} 步行 {day.total_walking_steps} 步",
                    suggestion="替换为低强度景点或增加出租车"
                ))
        
        # 3. 时间窗口约束
        for day in itinerary:
            for activity in day.activities:
                if isinstance(activity, ScheduledActivity):
                    if not self._is_within_opening_hours(activity, day.date):
                        violations.append(Violation(
                            type="outside_opening_hours",
                            severity="high",
                            detail=f"{activity.poi_name} 在 {activity.start_time} 不开放",
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
                if hasattr(a, 'themes'):
                    covered_themes.update(a.themes)
        
        missing_prefs = set(constraints.user_preferences) - covered_themes
        if missing_prefs:
            violations.append(Violation(
                type="preference_not_covered",
                severity="low",
                detail=f"用户偏好 {missing_prefs} 未在行程中体现",
                suggestion="增加相关类型的 POI"
            ))
        
        # 6. must_include 检查
        included = set()
        for day in itinerary:
            for a in day.activities:
                if hasattr(a, 'poi_id'):
                    included.add(a.poi_id)
        for must in constraints.must_include_pois:
            if must not in included:
                violations.append(Violation(
                    type="must_include_missing",
                    severity="high",
                    detail=f"必去景点 {must} 未被安排",
                    suggestion="强制分配到某一天"
                ))
        
        return violations
    
    def attempt_auto_fix(self, itinerary: list[DayPlan], violations: list[Violation], constraints: Constraints) -> list[DayPlan]:
        """
        尝试自动修复部分 violations。
        可修复的：
        - too_many_pois: 删除当天评分最低的景点
        - preference_not_covered: 在空白天添加对应类型景点
        不可自动修复的（需告知用户）：
        - budget_exceeded
        - must_include_missing
        """
        for v in violations:
            if v.type == "too_many_pois":
                day_num = extract_day_number(v.detail)
                day = itinerary[day_num - 1]
                # 删除评分最低的 non-must-include 景点
                removable = [a for a in day.activities if a.poi_id not in constraints.must_include_pois]
                if removable:
                    lowest = min(removable, key=lambda a: a.poi_score)
                    day.activities.remove(lowest)
        
        return itinerary
```

### 8.3 用餐插入逻辑（新增）

v2 完全没有定义用餐逻辑。v3 补充：

```python
def check_and_insert_meal(
    current_time: datetime,
    next_location: dict,
    schedule: list[Activity],
    constraints: Constraints
) -> MealActivity | None:
    """
    判断是否需要插入用餐。
    
    规则：
    1. 午餐窗口 11:30-13:30，晚餐窗口 17:30-19:30
    2. 距离上一餐 >= 3.5 小时
    3. 在下一个景点附近搜索匹配用户偏好的餐厅
    4. 如果用户指定了某家餐厅，优先安排
    """
    last_meal_time = get_last_meal_time(schedule)
    hours_since_last_meal = (current_time - last_meal_time).seconds / 3600 if last_meal_time else 999
    
    is_lunch_window = time_in_range(current_time, "11:30", "13:30")
    is_dinner_window = time_in_range(current_time, "17:30", "19:30")
    
    if hours_since_last_meal < 3:
        return None  # 上一餐时间太近
    
    if not (is_lunch_window or is_dinner_window):
        return None  # 不在用餐窗口
    
    meal_type = "lunch" if is_lunch_window else "dinner"
    
    # 在下一个景点附近搜索餐厅
    restaurant = find_nearby_restaurant(
        location=next_location,
        meal_type=meal_type,
        preferences=constraints.dietary_preferences,
        budget_range=constraints.meal_budget_per_person
    )
    
    if restaurant:
        return MealActivity(
            restaurant=restaurant,
            start_time=format_time(current_time),
            duration_min=60,
            meal_type=meal_type
        )
    
    return None
```

---

## 九、RAG 知识库系统

### 9.1 RAG vs Tool 的职责边界

| 信息类型 | 获取方式 | 原因 |
|---------|---------|------|
| POI 基础介绍、攻略文本、游玩建议 | RAG | 变化慢，适合离线索引 |
| 用户评论摘要、本地推荐 | RAG | 经过预处理，适合语义检索 |
| 城市消费水平、文化礼仪、注意事项 | RAG | 变化慢，适合文档检索 |
| 实时天气预报 | Tool (MCP) | 实时数据，不能缓存 |
| 实时交通状况 | Tool (MCP) | 实时数据 |
| 酒店实时价格与可用性 | Tool (MCP) | 实时数据 |
| 景点当前开放状态 | Tool (MCP) | 可能临时变化 |
| POI 地理坐标与距离 | Tool (MCP) | 需要精确计算 |
| 门票价格 | Tool (MCP) 为主 | 可能有波动，但 RAG 兜底 |

### 9.2 Chunk 策略

```python
class TravelChunker:
    """
    旅游领域的 Chunk 策略。

    核心原则：
    1. 一个 POI = 一个完整 Chunk（不拆分，因为 POI 信息高度关联）
    2. 城市攻略按景点拆分，每段保留城市上下文
    3. 评论摘要按情感聚类（好评/差评/建议）
    """

    def chunk_poi(self, poi: POI) -> Document:
        """POI 整体作为一个 Chunk。"""
        text = self._format_poi_as_text(poi)
        return Document(
            text=text,
            metadata={
                "type": "poi",
                "city": poi.city,
                "category": poi.category,
                "tags": poi.tags,
                "poi_id": poi.id,
                "last_verified": poi.last_verified,
                "data_source": poi.data_source
            }
        )

    def chunk_guide(self, guide_text: str, city: str) -> list[Document]:
        """城市攻略按景点/主题拆分。"""
        sections = self._split_by_sections(guide_text)
        return [
            Document(
                text=f"[城市：{city}] {section}",
                metadata={"type": "guide", "city": city, "section": title}
            )
            for title, section in sections
        ]
```

### 9.3 检索策略

```python
class HybridRetriever:
    """
    混合检索：Vector Search + BM25 + Cross-Encoder Rerank。

    多语言处理：
    - Embedding 模型选用 bge-m3（支持中文、英文、日文，无需分词）
    - BM25 使用 jieba 分词（中文）
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
            top_k=top_k * 3
        )

        # 2. BM25 Search
        bm25_results = await self.bm25_index.search(
            query=query,
            filters=filters,
            top_k=top_k * 3
        )

        # 3. 合并去重（RRF 融合）
        candidates = self._rrf_merge(
            vector_results, bm25_results,
            vector_weight=0.6, bm25_weight=0.4
        )

        # 4. Cross-Encoder Rerank
        reranked = await self.reranker.rerank(
            query=query,
            documents=candidates,
            top_k=top_k
        )

        return reranked

    def _rrf_merge(self, vector_results, bm25_results, k=60):
        """Reciprocal Rank Fusion 融合两种检索结果。"""
        scores = {}
        for rank, doc in enumerate(vector_results):
            scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
        for rank, doc in enumerate(bm25_results):
            scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
        # 按分数排序
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 9.4 数据更新策略

```python
class RAGDataUpdater:
    """RAG 数据定时更新。"""

    UPDATE_SCHEDULE = {
        "poi_basic": "weekly",      # 每周刷新 POI 基础信息
        "poi_price": "daily",       # 每天刷新价格
        "guide": "monthly",         # 每月更新攻略
        "review_summary": "weekly", # 每周更新评论摘要
    }

    async def run_update(self):
        """
        增量更新策略：
        1. 优先更新高频查询城市的数据
        2. 新数据与旧数据对比，只有变化时才更新 embedding
        3. 删除的数据标记为 unavailable，不直接删除（保留历史）
        """
        pass
```

---

## 十、评估体系设计

v2 的评估指标公式存在设计缺陷。v3 修正。

### 10.1 评估指标总表（修正版）

| 维度 | 指标 | 计算方式 | 权重 | 阈值 |
|------|------|---------|------|------|
| 偏好匹配度 | preference_coverage | 用户偏好中被行程覆盖的比例 | 0.20 | >= 0.8 |
| 路线合理性 | route_efficiency | 1 - (实际交通时间 / 总可用时间) | 0.15 | >= 0.5 |
| 时间可行性 | time_feasibility | 所有活动是否在开放时间内 | 0.15 | = 1.0 |
| 预算合理性 | budget_compliance | max(0, 1 - 超支比例) | 0.15 | >= 0.9 |
| 舒适度 | comfort_score | 基于步行量/景点数/时长的综合分 | 0.10 | >= 0.7 |
| 真实性 | factuality | 可通过工具验证的 POI 比例 | 0.15 | >= 0.95 |
| 可解释性 | explainability | 有推荐理由的活动占比 | 0.05 | >= 0.9 |
| 多样性 | diversity | 活动类别的 Shannon 熵 | 0.05 | >= 0.6 |

**v2 -> v3 修正说明**：

1. **route_efficiency**：v2 公式为 `1 - transit_ratio * 2`，交通时间超过 50% 直接为 0。v3 改为 `1 - (实际交通时间 / 总可用时间)`，按目的地动态判断阈值（一线城市交通时间长，阈值放宽到 0.5）。
2. **factuality**：v2 只看 `source` 字段。v3 改为抽样做 POI 存在性校验（调用地图 API 验证）。
3. **新增 confidence 计算方式**：不再依赖 LLM 自我评估，改为基于客观指标。

### 10.2 信心值计算（新增）

```python
class ConfidenceCalculator:
    """
    计算 Agent 对输出结果的信心值。
    基于客观指标，不由 LLM 自我评估。
    """

    def calculate(self, eval_result: EvalResult, state: TravelState) -> float:
        """
        信心值 = 加权评估分 * 数据可靠性系数 * 约束满足度
        """
        # 1. 基础评估分
        base_score = eval_result.total_score

        # 2. 数据可靠性系数（使用了多少实时数据 vs 缓存数据）
        live_data_ratio = self._calc_live_data_ratio(state)
        reliability_factor = 0.7 + 0.3 * live_data_ratio  # 范围 0.7-1.0

        # 3. 约束满足度
        hard_constraints_met = len(eval_result.critical_failures) == 0
        constraint_factor = 1.0 if hard_constraints_met else 0.5

        confidence = base_score * reliability_factor * constraint_factor
        return min(1.0, max(0.0, confidence))

    def _calc_live_data_ratio(self, state: TravelState) -> float:
        """计算实时数据占比。"""
        total = 0
        live = 0
        for day in state.current_itinerary or []:
            for activity in day.activities:
                total += 1
                if activity.data_source in ("api", "live"):
                    live += 1
        return live / total if total > 0 else 0.5
```

### 10.3 评估实现

```python
class ItineraryEvaluator:
    """行程质量评估器。"""

    def evaluate(self, itinerary: list[DayPlan], state: TravelState) -> EvalResult:
        scores = {}

        # 1. 偏好匹配度
        user_prefs = set(state.user_profile.preferences)
        covered = set()
        for day in itinerary:
            for activity in day.activities:
                if hasattr(activity, 'themes'):
                    covered.update(set(activity.themes) & user_prefs)
        scores["preference_coverage"] = len(covered) / len(user_prefs) if user_prefs else 1.0

        # 2. 路线合理性（修正版）
        total_transit = sum(day.total_transit_time_min for day in itinerary)
        total_available = sum(
            (parse_time(day.end_time) - parse_time(day.start_time)).seconds // 60
            for day in itinerary
        )
        if total_available > 0:
            transit_ratio = total_transit / total_available
            # 一线城市放宽阈值
            city_tier = get_city_tier(state.user_profile.destination)
            threshold = {1: 0.5, 2: 0.4, 3: 0.35}.get(city_tier, 0.4)
            scores["route_efficiency"] = max(0, 1 - transit_ratio / threshold)
        else:
            scores["route_efficiency"] = 0

        # 3. 时间可行性
        time_violations = 0
        total_activities = 0
        for day in itinerary:
            for activity in day.activities:
                if isinstance(activity, ScheduledActivity):
                    total_activities += 1
                    if not self._is_within_opening_hours(activity, day.date):
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

        # 6. 真实性（修正版：抽样校验）
        verified = 0
        total = 0
        for day in itinerary:
            for activity in day.activities:
                if isinstance(activity, ScheduledActivity):
                    total += 1
                    # 抽样验证（每轮验证 30%，至少 1 个）
                    if random.random() < 0.3 or total <= 1:
                        if self._verify_poi_exists(activity.poi_name, activity.location):
                            verified += 1
                    else:
                        verified += 1  # 未抽样的默认算通过
        scores["factuality"] = verified / total if total > 0 else 1.0

        # 7. 可解释性
        explained = sum(
            1 for day in itinerary
            for a in day.activities
            if hasattr(a, 'recommendation_reason') and a.recommendation_reason
        )
        scores["explainability"] = explained / total if total > 0 else 1.0

        # 8. 多样性
        category_counts = {}
        for day in itinerary:
            for a in day.activities:
                cat = getattr(a, 'category', 'other')
                category_counts[cat] = category_counts.get(cat, 0) + 1
        if category_counts:
            total_c = sum(category_counts.values())
            entropy = -sum(
                (c / total_c) * math.log2(c / total_c)
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
            improvement_suggestions=self._generate_suggestions(scores)
        )
```

---

## 十一、Human-in-the-Loop 协作机制

### 11.1 信心阈值与降级策略

```python
class ConfidenceManager:
    """根据信心等级决定交互策略。"""

    HIGH_CONFIDIDENCE = 0.8     # 直接输出
    MEDIUM_CONFIDENCE = 0.5     # 输出 + 请求确认
    LOW_CONFIDENCE = 0.3        # 输出多选方案

    def decide_interaction(self, confidence: float, has_violations: bool) -> InteractionMode:
        # 有关键 violations 时，即使信心高也要确认
        if has_violations:
            return InteractionMode.OUTPUT_WITH_CONFIRMATION

        if confidence >= self.HIGH_CONFIDIDENCE:
            return InteractionMode.DIRECT_OUTPUT
        elif confidence >= self.MEDIUM_CONFIDENCE:
            return InteractionMode.OUTPUT_WITH_CONFIRMATION
        elif confidence >= self.LOW_CONFIDENCE:
            return InteractionMode.MULTIPLE_OPTIONS
        else:
            return InteractionMode.ASK_CLARIFICATION
```

### 11.2 追问策略

```python
class ClarificationStrategy:
    """
    追问策略：
    1. 一次最多追问 2 个问题
    2. 信息不全也可以先生成草案（用默认值）
    3. 追问的问题要有明确选项
    4. 预算设为建议追问（用户说"没预算"才设为 None）
    """

    REQUIRED_FIELDS = ["destination", "travel_days"]
    RECOMMENDED_FIELDS = ["budget"]  # 强烈建议提供，但不是必须
    OPTIONAL_FIELDS_WITH_DEFAULTS = {
        "travelers_count": 1,
        "pace": "moderate",
        "companions": [],
        "departure_city": None,
    }

    def get_clarification_questions(self, state: TravelState) -> list[str] | None:
        missing_required = [
            f for f in self.REQUIRED_FIELDS
            if getattr(state.user_profile, f) is None
        ]

        if missing_required:
            return self._format_required_questions(missing_required[:2])

        # 建议追问预算
        if state.user_profile.budget is None:
            return ["大概预算范围是多少？（比如 2000-5000 元，或回复'没预算'）"]

        return None
```

### 11.3 关键操作确认机制

```python
class ConfirmationGate:
    """需要用户显式确认的操作。"""

    REQUIRES_CONFIRMATION = [
        "finalize_itinerary",   # 确认最终行程
        "major_plan_change",    # 大幅度变更（> 2 天受影响）
    ]

    SKIP_CONFIRMATION = [
        "add_poi",
        "remove_poi",
        "adjust_time",
        "weather_adaptation",
        "route_optimization",
        "reduce_intensity",     # 单日强度调整
    ]
```

---

## 十二、前端交互设计规范

### 12.1 行程展示设计

```
┌─────────────────────────────────────────────────────┐
│  成都 4 天行程                             [编辑] [导出] │
│                                                       │
│  总预算：¥2,650 / ¥3,000          节奏：轻松          │
│  ████████████████████░░░░ 88.3%                      │
│                                                       │
│  ┌─── Day 1 ─── 抵达 + 轻度探索 ──────────────────┐ │
│  │                                                   │ │
│  │  14:00  抵达成都东站                             │ │
│  │     │  地铁2号线 → 春熙路（30min, ¥4）          │ │
│  │     ▼                                             │ │
│  │  15:00  春熙路 / IFS 打卡                         │ │
│  │     适合抵达日，交通便利，不赶时间               │ │
│  │     免费 | 2 小时 | 强度：低                     │ │
│  │                                     [替换] [删除] │ │
│  │     ▼                                             │ │
│  │  18:00  晚餐：蜀大侠火锅（春熙路店）             │ │
│  │     人均 ¥120，与吃辣偏好匹配                    │ │
│  └───────────────────────────────────────────────────┘ │
│                                                       │
│  ┌─── Day 2 ─── 熊猫 + 文化 ──────────────────────┐ │
│  │  ...                                              │ │
│  └───────────────────────────────────────────────────┘ │
│                                                       │
│  ⚠️ 风险提醒                                         │
│  Day 3 降雨概率 60%，已准备室内替代方案 [查看]       │
│                                                       │
│  💡 你可以说：                                       │
│  "第三天轻松一点" | "加个川菜馆" | "看看省钱版"     │
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
3. 方案对比：左右对比 A/B 方案
4. 地图联动：点击地图上的 POI 标记 → 加入行程（P1）
```

---

## 十三、完整用户流程设计

### 13.1 首次使用流程（含降级路径）

```
用户打开应用
    │
    ▼
┌──────────────────────────┐
│ 欢迎语 + 引导提示         │
│ "想去哪玩？几天？预算？"  │
└──────────┬───────────────┘
           ▼
用户输入自然语言需求
    │
    ▼
┌──────────────────────────┐
│ 确定性状态机              │
│ 1. 解析约束               │
│ 2. 检查关键信息完整性     │
│ 3. 信息完整 → 进入生成    │
│   信息缺失 → 追问 1-2 题  │
└──────────┬───────────────┘
           ▼
    ┌──────┴──────┐
    │ 信息足够？   │
    ├── Yes ──────┤
    │             ▼
    │    ┌────────────────────┐
    │    │ 并行调用 Sub-Agents │
    │    │ + MCP Tools         │
    │    │                     │
    │    │ 降级路径：          │
    │    │ - 天气 API 失败 →   │
    │    │   用历史平均天气数据 │
    │    │ - POI API 失败 →    │
    │    │   用本地种子数据     │
    │    └────────┬───────────┘
    │             ▼
    │    ┌────────────────────┐
    │    │ 约束求解 + 行程生成 │
    │    └────────┬───────────┘
    │             ▼
    │    ┌────────────────────┐
    │    │ 硬约束校验          │
    │    │ violations?         │
    │    │ → 尝试自动修复      │
    │    │ → 仍有 violations?  │
    │    │   告知用户并建议     │
    │    └────────┬───────────┘
    │             ▼
    │    ┌────────────────────┐
    │    │ 评估打分            │
    │    │ score < 0.75?       │
    │    │ → 自动修正并重试    │
    │    └────────┬───────────┘
    │             ▼
    │    ┌────────────────────┐
    │    │ 输出行程 + 预算     │
    │    │ + 推荐理由 + 风险   │
    │    │ + 可调整选项        │
    │    │ + 降级数据提醒（如有）│
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

### 13.2 多轮修改流程（含影响半径）

```
用户发送修改请求
    │
    ▼
┌────────────────────────────────┐
│ Orchestrator 理解修改意图       │
│ - 修改类型（轻松？省钱？替换？）│
│ - 影响范围判定                  │
└──────────┬─────────────────────┘
           ▼
    ┌──────┴──────┐
    │ 影响范围     │
    ├── 单日 ─────┤
    │             ▼
    │    ┌────────────────────┐
    │    │ 只修改目标 Day      │
    │    │ 保留其他天不变      │
    │    │ 保存 before/after   │
    │    └────────┬───────────┘
    │             ▼
    │    ┌────────────────────┐
    │    │ 全局约束校验        │
    │    │ （预算是否仍满足？） │
    │    └────────┬───────────┘
    │             ▼
    │    ┌────────────────────┐
    │    │ 输出变更说明        │
    │    │ - 修改了什么        │
    │    │ - 保留了什么        │
    │    │ - 为什么这样改      │
    │    │ - 是否影响预算      │
    │    └────────────────────┘
    │
    ├── 跨天 ─────┤
    │             ▼
    │    ┌────────────────────┐
    │    │ 询问用户确认        │
    │    │ "这将影响 X 天的   │
    │    │  行程安排，确认吗？" │
    │    └────────┬───────────┘
    │             ▼
    │         用户确认
    │             ▼
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
| LLM | Claude Sonnet 4 / 通义千问 / DeepSeek | 工具调用能力强、中文理解好、成本可控 |
| Agent 框架 | LangGraph / 自研轻量框架 | LangGraph 原生支持 Agent Loop；如过重则自研 |
| MCP SDK | @modelcontextprotocol/sdk (Python) | 官方 SDK |
| 向量数据库 | Qdrant / Milvus | 支持过滤、稀疏向量、快速部署 |
| BM25 索引 | Meilisearch / Elasticsearch | CJK 分词支持 |
| Embedding | bge-m3 | 中文效果最佳，无需分词 |
| Reranker | bge-reranker-v2-m3 | 多语言 Cross-Encoder |
| 状态存储 | Redis (Session) + PostgreSQL (持久) | 快速读写 + 持久化 |
| 路径优化 | 自实现 2-opt（MVP）/ OR-Tools（后续） | MVP 阶段自实现足够 |
| 地图 API | 高德地图 API | 国内覆盖好、中文支持、免费额度充足 |
| 天气 API | 彩云天气 / 和风天气 | 国内覆盖、免费额度 |
| 后端框架 | FastAPI | 异步支持好、WebSocket 原生支持 |
| 前端框架 | Next.js + React + Tailwind CSS | SSR + 响应式 |
| 部署 | Docker + Railway / Render | 低成本快速部署 |
| 可观测性 | Langfuse / 自研日志 | Agent 决策链路追踪 |

### 14.2 项目目录结构

```
travel-agent/
├── README.md
├── docker-compose.yml
├── pyproject.toml
│
├── src/
│   ├── core/                            # 核心控制层（新增）
│   │   ├── deterministic_fsm.py         # 确定性状态机
│   │   ├── agent_loop.py                # Agent Loop 框架
│   │   └── loop_terminator.py           # 循环终止条件
│   │
│   ├── agents/                          # Agent 层
│   │   ├── orchestrator.py              # Orchestrator Agent
│   │   ├── itinerary_planner.py         # 行程规划 Sub-Agent
│   │   ├── budget_optimizer.py          # 预算优化 Sub-Agent
│   │   ├── recommendation_engine.py     # 推荐引擎 Sub-Agent
│   │   └── prompts/                     # System Prompts
│   │
│   ├── mcp_servers/                     # MCP Server 层
│   │   ├── map_server.py                # 地图 MCP（高德）
│   │   ├── weather_server.py            # 天气 MCP
│   │   ├── poi_server.py                # POI MCP（高德+点评）
│   │   └── reliability/                 # MCP 可靠性框架（新增）
│   │       ├── circuit_breaker.py
│   │       ├── fallback_manager.py
│   │       └── cache_manager.py
│   │
│   ├── planner/                         # 规划算法层
│   │   ├── constraint_solver.py         # 约束求解器
│   │   ├── route_optimizer.py           # 路径优化（2-opt）
│   │   ├── spatial_clustering.py        # 区域聚类（动态 eps）
│   │   ├── day_assigner.py              # 日分配（多目标）
│   │   ├── schedule_builder.py          # 时间表生成
│   │   └── constraint_checker.py        # 约束校验
│   │
│   ├── state/                           # 状态管理层
│   │   ├── state_manager.py             # 状态管理器（版本化）
│   │   ├── schemas.py                   # 数据模型定义
│   │   └── session_store.py             # 会话存储
│   │
│   ├── rag/                             # RAG 知识库层
│   │   ├── retriever.py                 # 混合检索器
│   │   ├── data_updater.py              # 数据更新（新增）
│   │   └── vector_store.py              # 向量数据库接口
│   │
│   ├── guardrails/                      # 安全与质量层
│   │   ├── input_guardrail.py           # 输入安全检查
│   │   ├── output_guardrail.py          # 输出质量检查
│   │   └── confidence_manager.py        # 信心阈值管理
│   │
│   ├── eval/                            # 评估层
│   │   ├── evaluator.py                 # 评估器
│   │   ├── metrics.py                   # 指标计算
│   │   └── test_cases.py                # 测试用例
│   │
│   ├── data/                            # 数据层（新增）
│   │   ├── seed_data/                   # 手工种子数据
│   │   │   ├── cities/                  # 20 城市 Top 50 POI
│   │   │   └── consumption_factors.json # 城市消费水平系数
│   │   └── quality_manager.py           # 数据质量管理
│   │
│   ├── api/                             # API 层
│   │   ├── main.py                      # FastAPI 入口
│   │   ├── routes/
│   │   │   ├── chat.py                  # WebSocket 聊天接口
│   │   │   └── itinerary.py             # REST 行程接口
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
│   │   │   └── Map/                     # 地图组件
│   │   └── pages/
│   └── package.json
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
└── scripts/
    ├── seed_data_loader.py              # 种子数据导入
    └── run_eval.py                      # 运行评估
```

---

## 十五、MVP 分阶段交付计划

v2 的排期过于乐观（Phase 1 仅 2 周）。v3 基于 1-2 人团队重新评估。

### Phase 1: 核心可用（Week 1-3）

**目标**：单轮单城市行程生成，命令行可验证

**范围收缩说明**：
- 不做前端（用 CLI / Postman 验证）
- 不做多轮对话（State 存内存）
- 不做 Sub-Agent 拆分（单体 Agent）
- 不做 RAG（直接用种子数据）
- 不做预算优化（只估算，不优化）

**交付物**：
- 确定性状态机（信息完整性检查 + 流程路由）
- 单体 Agent + Agent Loop（LLM 生成行程框架）
- 种子数据加载（20 城市 × Top 50 POI）
- 地图 MCP Server（高德，仅距离计算）
- 基础约束校验（预算、景点数、时间窗口）
- 结构化行程输出（JSON）

**验收标准**：
- 输入 "去成都 4 天 预算 3000" → 输出结构化 JSON 行程
- 行程包含真实 POI（来自种子数据）
- 景点间交通时间基于真实距离（高德 API）
- 预算估算在合理范围

---

### Phase 2: 算法增强 + 多轮（Week 4-6）

**目标**：约束求解 + 多轮局部修改 + State 持久化

**交付物**：
- DBSCAN 区域聚类（动态 eps）
- 2-opt 路径优化（考虑时间窗口）
- 多轮对话支持（State 持久化到 Redis）
- 局部修改能力（单日调整，影响半径控制）
- 基础前端对话界面（Next.js，仅文本交互）
- MCP 可靠性框架（Circuit Breaker + 降级）
- 天气 MCP Server 接入

**验收标准**：
- "第三天轻松一点" 能只修改 Day 3
- 路线优化后总交通时间减少
- 天气数据可显示在行程中
- API 失败时有降级提示

---

### Phase 3: 体验完善（Week 7-10）

**目标**：前端可视化 + Sub-Agent 拆分 + 评估体系

**交付物**：
- 拆分为独立 Sub-Agent（Planner / Budget / Recommendation）
- 前端行程可视化（时间线 + 地图 + 预算进度条）
- 交互式修改（替换按钮、方案对比）
- 评估模块（6 个核心指标）
- RAG 基础版（POI 知识库 + 向量检索）
- Guardrail 模块
- 数据自动更新（定时刷新价格/状态）

**验收标准**：
- Sub-Agent 间正确协作
- 前端可展示完整行程时间线
- 支持拖拽/按钮修改
- 6 个测试用例全部通过

---

### Phase 4: 高级功能（Week 11+）

**交付物**：
- OR-Tools 精确路径优化
- 雨天备选方案自动生成
- 用餐推荐（基于位置和偏好）
- 行程导出（iCal / PDF）
- 出行辅助信息（交通建议、当地规则）
- 可观测性（决策链路追踪）
- 更多城市数据覆盖

---

## 十六、风险与缓解策略

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|---------|
| LLM 幻觉（推荐不存在的景点） | 高 | 高 | 1. 所有 POI 来自种子数据或 API 验证<br>2. Output Guardrail 抽样校验<br>3. `source` 字段追踪数据来源 |
| 地图 API 配额耗尽 | 中 | 高 | 1. 距离矩阵结果缓存 24h<br>2. Circuit Breaker 降级到直线距离估算<br>3. 配额监控告警 |
| 爬取数据合规风险 | 中 | 高 | 1. 遵守 robots.txt<br>2. 控制请求频率（<= 1 req/s）<br>3. 仅获取公开可见数据<br>4. 用户评论用摘要而非原文 |
| 多轮对话状态丢失 | 中 | 高 | 1. Redis 持久化 + PostgreSQL 备份<br>2. 每轮保存 State 快照<br>3. 版本化 State 防止覆盖 |
| 路径优化算法性能 | 低 | 中 | 1. 2-opt 设置最大迭代次数<br>2. 景点数 > 10 时启用贪心算法<br>3. 异步计算不阻塞响应 |
| Token 消耗过高 | 中 | 中 | 1. Sub-Agent 使用较小模型<br>2. 结构化输出减少 Token<br>3. 缓存 LLM 响应 |
| 用户恶意输入 | 低 | 中 | 1. Input Guardrail（关键词 + Embedding）<br>2. Prompt Injection 检测 |
| 种子数据覆盖不足 | 中 | 高 | 1. MVP 只支持 20 个热门城市<br>2. 用户选择非覆盖城市时明确提示<br>3. Phase 2 逐步扩展 |
| 1-2 人团队进度延迟 | 高 | 高 | 1. 严格的范围控制（Phase 1 不做前端）<br>2. 优先核心路径，边缘功能延后<br>3. 每周验收，及时调整范围 |

---

## 附录

### 附录 A：v2 -> v3 关键变更总结

| 变更项 | v2 设计 | v3 设计 | 原因 |
|--------|---------|---------|------|
| 控制架构 | 纯 LLM 决策 | 双层：确定性状态机 + LLM | v2 循环终止不可靠，关键路径无保底 |
| 循环终止 | LLM 判断 `is_response_ready()` | 确定性条件检查 | LLM 可能无限循环或过早退出 |
| State 管理 | 无版本控制 | 乐观锁 + 版本号 | 并发修改冲突 |
| MCP 可靠性 | 无 | Circuit Breaker + 降级 + 缓存 | API 失败时系统不可用 |
| 2-opt 算法 | 忽略时间窗口 | 时间约束点固定，灵活点优化 | 景点被静默跳过 |
| 无法安排景点 | `continue` 跳过 | 标记为冲突，告知用户 | 用户不知必去景点被删 |
| 预算 | 固定 10% buffer | 按城市动态调整 | 一线城市 buffer 不足 |
| 用餐逻辑 | 未定义 | 明确规则（用餐窗口 + 间隔 + 偏好） | 午餐晚餐安排混乱 |
| 信心值 | LLM 自我评估 | 基于客观指标计算 | LLM 过度自信 |
| 数据源 | Google Places | 高德 + 大众点评 | 国内场景 |
| 冷启动 | 未提及 | 手工种子数据 20 城 × 50 POI | 需要立即可用的数据 |
| MVP 排期 | 2 周 Phase 1 | 3 周 Phase 1 | 1-2 人团队，范围收缩 |
| 多城市 | Schema 支持但未设计 | MVP 明确不支持 | 降低复杂度 |
| 评估公式 | route_efficiency 设计不当 | 按城市动态阈值 | 一线城市交通时间天然长 |

### 附录 B：API 接口定义

```yaml
# WebSocket Chat API
ws://api.travel-agent.com/ws/chat

# Client -> Server
{
  "type": "user_message",
  "content": "我想去成都玩 4 天",
  "session_id": "abc123"
}

# Server -> Client (流式)
{
  "type": "assistant_message",
  "content": "好的，我来帮你规划...",
  "metadata": {
    "tools_used": ["weather_mcp", "poi_mcp"],
    "confidence": 0.85,
    "eval_score": 0.82,
    "fallback_used": false
  }
}

# Server -> Client (行程结构化数据)
{
  "type": "itinerary_update",
  "data": {
    // ItineraryResponse schema
  }
}

# Server -> Client (降级提醒)
{
  "type": "disclaimer",
  "content": "部分数据基于缓存，可能不是最新"
}

# REST API
GET  /api/v1/itinerary/{session_id}        # 获取当前行程
GET  /api/v1/itinerary/{session_id}/eval    # 获取评估结果
```

### 附录 C：关键设计决策记录 (ADR)

**ADR-001: Agent Loop vs Pipeline vs 双层控制**
- v2 决策：纯 Agent Loop
- v3 变更：双层控制（确定性状态机 + Agent Loop）
- 原因：纯 LLM 决策在关键路径上不可靠，需要确定性保底

**ADR-002: 国内数据源**
- 决策：高德 + 大众点评 + 小红书
- 原因：MVP 聚焦国内游，国内 API 覆盖更好、中文支持更强

**ADR-003: 单城市限制**
- 决策：MVP 只支持单城市
- 原因：1-2 人团队无法同时做好城内规划和跨城交通
- 后续：v2.0 支持多城市

**ADR-004: 种子数据冷启动**
- 决策：手工录入 20 城 × 50 POI
- 原因：爬取系统开发需要时间，种子数据可立即验证核心算法
- 后续：开发爬虫后逐步扩展

**ADR-005: 2-opt 而非 OR-Tools（MVP）**
- 决策：MVP 自实现 2-opt
- 原因：单城市场景下景点数通常 < 8，2-opt 足够；OR-Tools 引入依赖复杂
- 后续：商业版本引入 OR-Tools

---

**文档结束**

> 本文档是 v2 版本的问题修正和架构升级版本。所有设计决策均基于用户确认的约束（国内游、单城市、1-2 人团队、公开数据源）。具体实现时需根据实际框架和 API 进行调整。


