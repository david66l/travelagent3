# 旅行 Agent 系统 - 技术方案 v3（LangGraph 编排层）

> **目标**：用 LangGraph 重写编排层（Orchestrator + 事件流），保留 V2 中所有 Agent、Skill、数据模型定义不变
> **原则**：说清楚 Graph 结构、State 流转、节点职责、路由逻辑，不贴冗余代码

---

## 一、与 V2 方案的关系

```
V2 方案（当前）                     V3 方案（LangGraph）
┌─────────────────┐               ┌─────────────────────┐
│  Orchestrator   │    替换为     │   StateGraph        │
│  （代码编排）    │ ───────────▶ │   （声明式图）       │
├─────────────────┤               ├─────────────────────┤
│  StateManager   │    替换为     │   State（TypedDict）│
│  + Redis 缓存   │ ───────────▶ │   + Checkpointer    │
├─────────────────┤               ├─────────────────────┤
│  EventBus       │    缩小为     │   仅前端推送        │
│  （Agent 间通信）│ ───────────▶ │   Graph edges 替代  │
├─────────────────┤               ├─────────────────────┤
│  10 个 Agent    │    保留       │   10 个 Agent       │
│  8 个 Skill     │    保留       │   8 个 Skill        │
│  数据模型        │    保留       │   数据模型           │
│  数据库表        │    保留       │   数据库表           │
└─────────────────┘               └─────────────────────┘
```

**什么变了**：
- Orchestrator 类被删除，改为 StateGraph 声明式定义
- StateManager 被 State + Checkpointer 替代
- Agent 间通信从 EventBus 改为 Graph edges
- 循环终止、对话持久化由 Checkpointer 内置处理

**什么没变**：
- Agent 内部逻辑完全不变（ItineraryPlanner 的 6 层算法、Validation 的 5 维校验等）
- Skill 层完全不变（web_search、poi_search 等 8 个 Skill）
- Pydantic 数据模型完全不变（UserProfile、DayPlan、BudgetPanel 等）
- 数据库表结构完全不变
- 前端交互完全不变

---

## 二、State 定义（全局状态）

LangGraph 的 State 是一个 TypedDict，所有节点读写同一个 State 对象。State 在节点间自动传递，不需要手动管理。

```python
from typing import TypedDict, Annotated
from operator import add

# ============ 辅助类型定义（与 V2 完全一致）============
# 以下模型直接复用 V2 的 Pydantic 模型，此处仅标注字段用途

class ItineraryState(TypedDict):
    """
    全局状态对象。Graph 中所有节点读写此 State。
    节点只需读取自己需要的字段，修改自己负责的字段。
    """

    # ═══════════════════════════════════════════
    # 第一层：会话与输入（由入口节点写入）
    # ═══════════════════════════════════════════
    session_id: str
    user_id: str
    user_input: str                          # 当前轮用户输入
    messages: Annotated[list[dict], add]     # 对话历史（累积，operator.add）

    # ═══════════════════════════════════════════
    # 第二层：Intent 识别结果（由 intent_node 写入）
    # ═══════════════════════════════════════════
    intent: str | None                       # 7 种意图之一
    intent_confidence: float
    user_entities: dict                      # 提取的实体
    missing_required: list[str]              # 缺失必需字段
    missing_recommended: list[str]           # 缺失建议字段
    preference_changes: list[dict]           # 偏好变更列表
    clarification_questions: list[str]       # 追问问题

    # ═══════════════════════════════════════════
    # 第三层：用户画像（由 preference_node 维护）
    # ═══════════════════════════════════════════
    user_profile: dict | None                # UserProfile 的 dict 形式

    # ═══════════════════════════════════════════
    # 第四层：实时查询结果（由并行查询节点写入）
    # ═══════════════════════════════════════════
    candidate_pois: list[dict]               # ScoredPOI[]
    weather_data: list[dict]                 # WeatherDay[]
    price_data: dict                         # poi_name -> PriceInfo

    # ═══════════════════════════════════════════
    # 第五层：行程数据（由 planner 节点写入）
    # ═══════════════════════════════════════════
    current_itinerary: list[dict] | None     # DayPlan[]
    itinerary_status: str                    # "draft" / "confirmed"

    # ═══════════════════════════════════════════
    # 第六层：校验与优化（由并行校验节点写入）
    # ═══════════════════════════════════════════
    validation_result: dict | None           # ValidationResult
    optimized_routes: dict                   # day_number -> route_info

    # ═══════════════════════════════════════════
    # 第七层：面板数据（由 budget/preference 节点维护）
    # ═══════════════════════════════════════════
    budget_panel: dict | None                # BudgetPanel
    preference_panel: dict | None            # PreferencePanel

    # ═══════════════════════════════════════════
    # 第八层：输出（由 proposal/qa/collect 节点写入）
    # ═══════════════════════════════════════════
    assistant_response: str | None           # 最终返回给用户的消息
    proposal_text: str | None                # 方案文本（Markdown）

    # ═══════════════════════════════════════════
    # 控制流标记（节点间通信的信号）
    # ═══════════════════════════════════════════
    needs_clarification: bool                # 是否需要追问
    needs_replan: bool                       # 是否需要重规划
    waiting_for_confirmation: bool           # 是否等待用户确认（interrupt）
    error: str | None                        # 错误信息
```

**关于 `Annotated[list[dict], add]`**：
- `messages` 字段使用 `operator.add` 合并策略
- 每个节点返回的 `messages` 增量会自动追加到列表，而不是覆盖
- 其他字段默认是"覆盖"策略（后写的覆盖先写的）

---

## 三、Graph 拓扑结构

### 3.1 主图（Main Graph）

```
                    ┌─────────────────────────────────────────────┐
                    │              MAIN GRAPH                      │
                    │                                              │
  用户输入 ────────▶ │  ┌─────────────┐    ┌─────────────────┐    │
                    │  │ intent_node │───▶│  route_intent   │    │
                    │  │  意图识别    │    │  条件路由边      │    │
                    │  └─────────────┘    └────────┬────────┘    │
                    │                              │              │
                    │         ┌────────────────────┼────────────────────┐
                    │         │                    │                    │
                    │    [intent=                 │               [intent=
                    │    generate]                │               chitchat]
                    │         │                    │                    │
                    │         ▼                    ▼                    ▼
                    │  ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
                    │  │  generate_   │    │ update_prefs│    │   qa_node   │
                    │  │  subgraph    │    │   _node     │    │             │
                    │  │  行程生成子图 │    │  偏好更新   │    │   问答      │
                    │  └──────┬───────┘    └──────┬──────┘    └──────┬─────┘
                    │         │                   │                  │
                    │         └───────────────────┴──────────────────┘
                    │                              │
                    │                              ▼
                    │                    ┌─────────────────┐
                    │                    │   format_output │
                    │                    │   格式化输出     │
                    │                    │  （统一出口）    │
                    │                    └────────┬────────┘
                    │                              │
                    │                              ▼
                    │                    ┌─────────────────┐
                    │                    │  END（返回用户）  │
                    │                    └─────────────────┘
                    │
                    │  ┌─────────────────────────────────────────┐
                    │  │  其他分支（由 route_intent 路由）        │
                    │  │                                          │
                    │  │  [missing_required] ──▶ collect_info_node│
                    │  │  [intent=confirm] ────▶ confirm_node     │
                    │  │  [intent=view_history]▶ history_node     │
                    │  └─────────────────────────────────────────┘
                    └─────────────────────────────────────────────┘
```

### 3.2 行程生成子图（Generate Subgraph）

这是系统最核心的子图，对应 V2 的"流程 A：从零生成行程"。

```
              ┌─────────────────────────────────────────────────────────────┐
              │                  GENERATE SUBGRAPH                           │
              │                                                              │
  从主图进入 ──▶│  ┌──────────────────┐                                       │
              │  │  prepare_context │  加载用户记忆，初始化面板               │
              │  │   准备上下文     │                                       │
              │  └────────┬─────────┘                                       │
              │           │                                                  │
              │           ▼                                                  │
              │  ┌──────────────────┐    ┌──────────────┐    ┌────────────┐ │
              │  │  poi_search_node │    │ weather_node │    │ budget_    │ │
              │  │   搜索景点美食   │    │   查询天气   │    │ init_node  │ │
              │  │                  │    │              │    │ 初始化预算  │ │
              │  └────────┬─────────┘    └──────┬───────┘    └─────┬──────┘ │
              │           │                      │                  │        │
              │           └──────────────────────┴──────────────────┘        │
              │                              │                               │
              │                    ┌─────────▼─────────┐                     │
              │                    │   parallel_join   │  自动汇聚（LangGraph │
              │                    │   并行结果汇聚    │  等所有上游完成）    │
              │                    └─────────┬─────────┘                     │
              │                              │                               │
              │                              ▼                               │
              │                    ┌──────────────────┐                      │
              │                    │  planner_node    │  6层算法生成行程      │
              │                    │  行程规划         │                      │
              │                    └────────┬─────────┘                      │
              │                             │                                │
              │                             ▼                                │
              │  ┌─────────────────┐  ┌──────────────┐  ┌────────────────┐  │
              │  │ validation_node │  │ route_node   │  │ budget_calc_   │  │
              │  │   多维度校验    │  │  2-opt优化   │  │    node        │  │
              │  │                 │  │              │  │  计算预算明细   │  │
              │  └────────┬────────┘  └──────┬───────┘  └────────┬───────┘  │
              │           │                   │                   │          │
              │           └───────────────────┴───────────────────┘          │
              │                              │                               │
              │                    ┌─────────▼─────────┐                     │
              │                    │   parallel_join   │                      │
              │                    │   并行结果汇聚    │                      │
              │                    └─────────┬─────────┘                     │
              │                              │                               │
              │                              ▼                               │
              │                    ┌──────────────────┐                      │
              │                    │  proposal_node   │  LLM生成方案文本     │
              │                    │  方案生成         │                      │
              │                    └────────┬─────────┘                      │
              │                             │                                │
              │                             ▼                                │
              │                    ┌──────────────────┐                      │
              │                    │  interrupt_node  │  中断等待用户确认    │
              │                    │  等待确认         │  （人机交互点）      │
              │                    └──────────────────┘                      │
              │                                                              │
              │  如果用户确认 ──▶ confirm_node ──▶ save_to_memory ──▶ END   │
              │  如果用户修改 ──▶ 返回 planner_node（增量修改）              │
              └─────────────────────────────────────────────────────────────┘
```

**并行执行的实现方式**：

在 LangGraph 中，一个节点的多个出边连接的下游节点会自动并行执行。不需要写 `asyncio.gather()`。

```python
# 并行查询：poi_search_node / weather_node / budget_init_node 并行执行
# 这三个节点都完成后，才会执行 parallel_join
# 同理：validation_node / route_node / budget_calc_node 并行执行
```

---

## 四、节点（Node）定义

每个 Node 是一个 Python 函数，接收 State，返回 State 的部分更新。

### 4.1 主图节点

| 节点名 | 职责 | 读取 State | 写入 State | 对应 V2 Agent |
|--------|------|-----------|-----------|--------------|
| `intent_node` | 意图识别、实体提取、偏好变更检测 | `user_input`, `messages`, `user_profile` | `intent`, `user_entities`, `preference_changes`, `missing_required`, `clarification_questions` | IntentRecognitionAgent |
| `route_intent` | 条件路由函数（不是节点，是 edge） | `intent`, `missing_required` | 无 | Orchestrator 的分发逻辑 |
| `collect_info_node` | 生成追问问题 | `missing_required`, `missing_recommended` | `assistant_response`, `needs_clarification` | InformationCollectionAgent |
| `update_prefs_node` | 更新偏好、发布变更 | `preference_changes`, `user_profile` | `user_profile`, `preference_panel`, `needs_replan` | PreferenceBudgetAgent |
| `qa_node` | 回答用户问题 | `user_input`, `candidate_pois` | `assistant_response` | QAAgent |
| `confirm_node` | 处理确认行程 | `current_itinerary` | `itinerary_status="confirmed"` | Orchestrator + MemoryManagementAgent |
| `history_node` | 返回历史行程 | `user_id` | `assistant_response` | MemoryManagementAgent |
| `format_output` | 统一格式化最终输出 | `assistant_response`, `current_itinerary`, `budget_panel`, `preference_panel` | `messages`（追加 assistant 消息） | 无（纯代码） |

### 4.2 Generate Subgraph 节点

| 节点名 | 职责 | 读取 State | 写入 State | 对应 V2 Agent/Skill |
|--------|------|-----------|-----------|-------------------|
| `prepare_context_node` | 加载历史记忆，初始化面板 | `user_id`, `session_id` | `user_profile`, `preference_panel`, `budget_panel` | MemoryManagementAgent.get_user_memory() |
| `poi_search_node` | 并行：搜索 POI | `user_profile` | `candidate_pois` | RealtimeQueryAgent + poi_search Skill |
| `weather_node` | 并行：查询天气 | `user_profile` | `weather_data` | RealtimeQueryAgent + weather_query Skill |
| `budget_init_node` | 并行：初始化预算面板 | `user_profile` | `budget_panel` | PreferenceBudgetAgent |
| `parallel_join_1` | 汇聚并行结果（空操作或检查） | - | - | 无（LangGraph 自动汇聚） |
| `planner_node` | 核心规划算法 | `candidate_pois`, `weather_data`, `user_profile` | `current_itinerary` | ItineraryPlannerAgent |
| `validation_node` | 并行：多维度校验 | `current_itinerary`, `user_profile` | `validation_result` | ValidationAgent |
| `route_node` | 并行：2-opt 路线优化 | `current_itinerary` | `optimized_routes` | MapRouteAgent |
| `budget_calc_node` | 并行：计算预算明细 | `current_itinerary`, `user_profile` | `budget_panel` | PreferenceBudgetAgent |
| `parallel_join_2` | 汇聚并行结果 | - | - | 无 |
| `proposal_node` | 生成方案文本 | `current_itinerary`, `budget_panel`, `weather_data`, `validation_result` | `proposal_text`, `assistant_response` | ProposalGenerationAgent |
| `interrupt_node` | 中断等待用户确认 | `proposal_text` | `waiting_for_confirmation=True` | LangGraph interrupt |
| `save_memory_node` | 保存确认的行程 | `current_itinerary`, `user_profile` | -（写入数据库） | MemoryManagementAgent |

---

## 五、Edge 与路由逻辑

### 5.1 主图路由

```python
def route_intent(state: ItineraryState) -> str:
    """
    条件路由函数。根据 intent 和缺失字段决定走向哪个节点。
    返回值对应下一个节点的名称。
    """
    # 优先级 1：如果关键信息缺失，先收集信息
    if state["missing_required"]:
        return "collect_info_node"
    
    # 优先级 2：根据意图路由
    intent_map = {
        "generate_itinerary": "generate_subgraph",
        "modify_itinerary": "generate_subgraph",      # 修改也走生成子图（增量）
        "update_preferences": "update_prefs_node",
        "query_info": "qa_node",
        "confirm_itinerary": "confirm_node",
        "view_history": "history_node",
        "chitchat": "qa_node",                         # 闲聊走 Q&A
    }
    
    return intent_map.get(state["intent"], "qa_node")


# Graph 构建
builder = StateGraph(ItineraryState)

# 添加节点
builder.add_node("intent_node", intent_node)
builder.add_node("collect_info_node", collect_info_node)
builder.add_node("generate_subgraph", generate_subgraph)
builder.add_node("update_prefs_node", update_prefs_node)
builder.add_node("qa_node", qa_node)
builder.add_node("confirm_node", confirm_node)
builder.add_node("history_node", history_node)
builder.add_node("format_output", format_output)

# 添加边
builder.set_entry_point("intent_node")
builder.add_conditional_edges("intent_node", route_intent)
builder.add_edge("collect_info_node", "format_output")
builder.add_edge("update_prefs_node", "format_output")
builder.add_edge("qa_node", "format_output")
builder.add_edge("confirm_node", "format_output")
builder.add_edge("history_node", "format_output")
builder.add_edge("format_output", END)
```

### 5.2 Generate Subgraph 内部路由

```python
def generate_subgraph() -> StateGraph:
    """构建行程生成子图"""
    builder = StateGraph(ItineraryState)
    
    # Step 1: 准备上下文
    builder.add_node("prepare_context", prepare_context_node)
    
    # Step 2: 并行查询（fan-out）
    builder.add_node("poi_search", poi_search_node)
    builder.add_node("weather_query", weather_node)
    builder.add_node("budget_init", budget_init_node)
    
    # Step 3: 规划（串行，依赖并行查询结果）
    builder.add_node("planner", planner_node)
    
    # Step 4: 并行校验（fan-out）
    builder.add_node("validation", validation_node)
    builder.add_node("route_optimize", route_node)
    builder.add_node("budget_calc", budget_calc_node)
    
    # Step 5: 方案生成（串行）
    builder.add_node("proposal", proposal_node)
    
    # Step 6: 中断等待确认
    builder.add_node("wait_confirm", interrupt_node)
    
    # ===== 边定义 =====
    builder.set_entry_point("prepare_context")
    
    # prepare_context 后并行启动 3 个查询
    builder.add_edge("prepare_context", "poi_search")
    builder.add_edge("prepare_context", "weather_query")
    builder.add_edge("prepare_context", "budget_init")
    
    # 3 个查询都完成后，才执行 planner
    builder.add_edge("poi_search", "planner")
    builder.add_edge("weather_query", "planner")
    builder.add_edge("budget_init", "planner")
    
    # planner 后并行启动校验+优化+预算
    builder.add_edge("planner", "validation")
    builder.add_edge("planner", "route_optimize")
    builder.add_edge("planner", "budget_calc")
    
    # 3 个并行节点都完成后，才执行 proposal
    builder.add_edge("validation", "proposal")
    builder.add_edge("route_optimize", "proposal")
    builder.add_edge("budget_calc", "proposal")
    
    # proposal 后中断等待用户
    builder.add_edge("proposal", "wait_confirm")
    
    # 子图出口
    builder.add_edge("wait_confirm", END)
    
    return builder.compile()
```

**关键点**：在 LangGraph 中，当一个节点有多个下游节点时（如 `prepare_context` → `poi_search` / `weather_query` / `budget_init`），这些下游节点会自动并行执行。当多个上游节点都指向同一个下游节点时（如 `poi_search` → `planner` 且 `weather_query` → `planner`），LangGraph 会等待所有上游节点完成后才执行下游节点。

这正好替代了 V2 中的 `asyncio.gather()` 逻辑。

---

## 六、Interrupt 人机交互

V2 方案中"等待用户确认行程"需要手动处理循环终止条件。LangGraph 的 `interrupt` 机制可以更优雅地实现。

### 6.1 使用场景

```
 proposal_node 生成方案后
         │
         ▼
┌──────────────────┐
│ interrupt_node   │  ← 中断点
│ 等待用户确认     │
└──────────────────┘
         │
    暂停执行，状态自动保存到 Checkpointer
         │
    用户发送"确认"
         │
    恢复执行，继续到 save_memory_node
```

### 6.2 实现方式

```python
from langgraph.types import interrupt

def interrupt_node(state: ItineraryState) -> dict:
    """
    中断节点：生成方案后暂停，等待用户确认。
    这是一个特殊节点，执行时会抛出 interrupt，状态自动保存。
    """
    # 构建呈现给用户的方案
    proposal = state["proposal_text"]
    itinerary = state["current_itinerary"]
    budget = state["budget_panel"]
    
    # interrupt 会暂停 Graph 执行，返回给调用方
    # 调用方（FastAPI/WebSocket）将方案发送给用户，等待回复
    user_response = interrupt({
        "type": "itinerary_proposal",
        "proposal": proposal,
        "itinerary": itinerary,
        "budget": budget,
        "actions": ["confirm", "modify", "cancel"]
    })
    
    # 用户回复后，Graph 从 checkpoint 恢复，继续执行
    # user_response 是用户的选择（confirm / modify / cancel）
    if user_response == "confirm":
        return {"itinerary_status": "confirmed"}
    elif user_response == "modify":
        return {"itinerary_status": "draft", "needs_replan": True}
    else:
        return {"itinerary_status": "cancelled"}

# 在主图中，根据中断后的结果决定下一步
def route_after_interrupt(state: ItineraryState) -> str:
    if state["itinerary_status"] == "confirmed":
        return "save_memory_node"
    elif state["needs_replan"]:
        return "planner_node"  # 回到规划节点
    else:
        return END
```

### 6.3 与 FastAPI/WebSocket 集成

```python
# FastAPI WebSocket 端点
@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    await websocket.accept()
    
    # 每个会话一个 Graph 实例 + Checkpointer
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    while True:
        user_input = await websocket.receive_text()
        
        # 构建初始状态
        state = {
            "user_input": user_input,
            "messages": [{"role": "user", "content": user_input}],
        }
        
        # 运行 Graph
        async for event in graph.astream(state, config):
            
            # 情况 1：正常节点输出
            if "assistant_response" in event:
                await websocket.send_json({
                    "type": "assistant_message",
                    "content": event["assistant_response"]
                })
            
            # 情况 2：Interrupt（需要用户交互）
            elif "__interrupt__" in event:
                interrupt_data = event["__interrupt__"][0].value
                await websocket.send_json({
                    "type": "itinerary_proposal",
                    "data": interrupt_data
                })
                
                # 等待用户回复
                user_reply = await websocket.receive_text()
                
                # 恢复 Graph 执行，传入用户回复
                async for resume_event in graph.astream(
                    Command(resume=user_reply), 
                    config
                ):
                    # 处理恢复后的输出
                    ...
```

---

## 七、Checkpointer 配置

Checkpointer 替代 V2 的 StateManager，自动保存每次状态变更，支持对话恢复。

### 7.1 配置（PostgreSQL）

```python
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection

# 创建 PostgreSQL checkpointer
conn = Connection.connect("postgresql://user:pass@localhost/travel_agent")
checkpointer = PostgresSaver(conn)
checkpointer.setup()  # 创建必要的表

# 编译 Graph 时传入 checkpointer
graph = builder.compile(checkpointer=checkpointer)
```

### 7.2 Checkpointer 带来的收益

| V2 方案（手动管理） | V3 方案（Checkpointer） |
|-------------------|----------------------|
| 手动写 StateManager 类（~150 行） | 自动保存/恢复 |
| 手动处理 Redis 序列化/反序列化 | 自动处理 |
| 多轮对话需要手动加载历史 | `thread_id` 自动关联历史 |
| 中断恢复需要自己设计 | `Command(resume=...)` 内置支持 |
| 错误回滚需要自己实现 | checkpoint 自然支持回滚到上一步 |

### 7.3 对话历史管理

```python
# 新对话时，加载历史偏好
config = {"configurable": {"thread_id": thread_id}}

# 获取当前会话状态（如果有历史）
current_state = checkpointer.get(config)

if current_state:
    # 已有历史，继续对话
    # 历史 messages 自动在 State 中累积
    state_update = {"user_input": new_input}
else:
    # 新对话，加载用户记忆
    memory = await memory_agent.get_user_memory(user_id)
    state_update = {
        "user_input": new_input,
        "user_profile": memory.profile,
    }

# 运行 Graph
async for event in graph.astream(state_update, config):
    ...
```

---

## 八、EventBus 的新定位

在 V2 方案中，EventBus 负责 Agent 间通信。在 V3 方案中，Graph edges 替代了大部分 Agent 间通信，EventBus 缩小为仅负责**前端推送**。

### 8.1 EventBus 职责变化

| 场景 | V2（EventBus） | V3（Graph + EventBus） |
|------|---------------|----------------------|
| Intent 识别后分发 | EventBus 发布 `IntentRecognized` | Graph 条件边 `route_intent` |
| 偏好变更后重规划 | EventBus 发布 `PreferenceChanged` | Graph 子图调用或跳转到 planner_node |
| 行程生成后校验 | EventBus 发布 `ItineraryGenerated` | Graph fan-out 到并行校验节点 |
| 行程确认后保存 | EventBus 发布 `ItineraryConfirmed` | Graph 边连接到 save_memory_node |
| 预算更新推前端 | EventBus 发布 `BudgetUpdated` | **保留 EventBus**，WebSocket 推送 |
| 偏好更新推前端 | EventBus 发布 `PreferenceChanged` | **保留 EventBus**，WebSocket 推送 |

### 8.2 保留的 EventBus 定义

```python
# 仅保留需要推送到前端的事件
class FrontendEventBus:
    """仅用于向前端推送实时更新"""
    
    async def publish(self, event_type: str, session_id: str, data: dict):
        """通过 WebSocket 推送到指定会话"""
        await websocket_manager.send_to_session(session_id, {
            "type": event_type,
            "data": data
        })

# 使用场景：
# 1. budget_calc_node 完成后，推预算面板更新
# 2. update_prefs_node 完成后，推偏好面板更新
# 3. planner_node 完成后，推行程序列更新
```

---

## 九、节点实现模板

每个节点的实现模式统一为：**读取 State → 调用 Agent/Skill → 返回 State 更新**。

### 9.1 标准节点模板

```python
async def xxx_node(state: ItineraryState) -> dict:
    """
    节点函数模板。
    
    输入：完整 State（只读取需要的字段）
    输出：dict，包含要更新的字段（只返回修改的字段）
    
    LangGraph 会自动将返回的 dict 合并到 State 中。
    """
    # 1. 读取需要的字段
    user_input = state["user_input"]
    user_profile = state["user_profile"]
    
    # 2. 调用 Agent/Skill（复用 V2 的 Agent 类）
    agent = SomeAgent()
    result = await agent.run(user_input, user_profile)
    
    # 3. 返回 State 更新（只返回修改的字段）
    return {
        "assistant_response": result.response,
        "some_field": result.data,
    }
```

### 9.2 带副作用的节点（如保存数据库）

```python
async def save_memory_node(state: ItineraryState) -> dict:
    """
    有副作用的节点：保存到数据库。
    副作用在节点内部执行，不返回给 State。
    """
    # 读取需要保存的数据
    itinerary = state["current_itinerary"]
    profile = state["user_profile"]
    session_id = state["session_id"]
    
    # 调用 MemoryManagementAgent（复用 V2 的 Agent）
    memory_agent = MemoryManagementAgent()
    await memory_agent.save_itinerary(session_id, itinerary, profile)
    
    # 可以不返回任何更新，或返回确认标记
    return {"itinerary_status": "confirmed"}
```

### 9.3 关键节点实现示例

#### intent_node（意图识别）

```python
async def intent_node(state: ItineraryState) -> dict:
    """意图识别节点：调用 IntentRecognitionAgent"""
    agent = IntentRecognitionAgent()
    
    result = await agent.recognize(
        user_input=state["user_input"],
        messages=state["messages"],
        user_profile=state.get("user_profile")
    )
    
    return {
        "intent": result.intent,
        "intent_confidence": result.confidence,
        "user_entities": result.user_entities,
        "missing_required": result.missing_required,
        "missing_recommended": result.missing_recommended,
        "preference_changes": result.preference_changes,
        "clarification_questions": result.clarification_questions,
    }
```

#### planner_node（行程规划）

```python
async def planner_node(state: ItineraryState) -> dict:
    """行程规划节点：调用 ItineraryPlannerAgent"""
    agent = ItineraryPlannerAgent()
    
    itinerary = await agent.plan(
        pois=state["candidate_pois"],
        weather=state["weather_data"],
        profile=state["user_profile"]
    )
    
    return {
        "current_itinerary": itinerary,
        "itinerary_status": "draft",
    }
```

#### update_prefs_node（偏好更新 + 触发重规划）

```python
async def update_prefs_node(state: ItineraryState) -> dict:
    """偏好更新节点：更新偏好并触发重规划信号"""
    agent = PreferenceBudgetAgent()
    
    # 更新偏好
    new_profile = await agent.update_preferences(
        profile=state["user_profile"],
        changes=state["preference_changes"]
    )
    
    # 重新计算面板
    panel = agent.build_preference_panel(new_profile)
    
    # 如果有行程草稿，标记需要重规划
    needs_replan = state.get("current_itinerary") is not None
    
    # 推送前端更新（保留 EventBus）
    await frontend_bus.publish("PreferenceUpdated", state["session_id"], panel)
    
    return {
        "user_profile": new_profile,
        "preference_panel": panel,
        "needs_replan": needs_replan,
        "assistant_response": f"已更新偏好：{format_changes(state['preference_changes'])}",
    }
```

---

## 十、完整 Graph 编译

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver

# ═══════════════════════════════════════════
# 1. 构建主图
# ═══════════════════════════════════════════
main_builder = StateGraph(ItineraryState)

# 添加主图节点
main_builder.add_node("intent_node", intent_node)
main_builder.add_node("collect_info_node", collect_info_node)
main_builder.add_node("generate_subgraph", generate_subgraph.compile())  # 子图作为节点
main_builder.add_node("update_prefs_node", update_prefs_node)
main_builder.add_node("qa_node", qa_node)
main_builder.add_node("confirm_node", confirm_node)
main_builder.add_node("history_node", history_node)
main_builder.add_node("format_output", format_output)

# 边
main_builder.set_entry_point("intent_node")
main_builder.add_conditional_edges("intent_node", route_intent)

# 所有分支汇聚到 format_output
for node in ["collect_info_node", "update_prefs_node", "qa_node", 
             "confirm_node", "history_node", "generate_subgraph"]:
    main_builder.add_edge(node, "format_output")

main_builder.add_edge("format_output", END)

# ═══════════════════════════════════════════
# 2. 添加 Checkpointer
# ═══════════════════════════════════════════
conn = Connection.connect("postgresql://user:pass@localhost/travel_agent")
checkpointer = PostgresSaver(conn)
checkpointer.setup()

# ═══════════════════════════════════════════
# 3. 编译
# ═══════════════════════════════════════════
graph = main_builder.compile(checkpointer=checkpointer)
```

---

## 十一、V2 vs V3 对比表

| 维度 | V2 方案 | V3 方案（LangGraph） | 差异说明 |
|------|---------|---------------------|---------|
| **编排方式** | Orchestrator 类（代码编排） | StateGraph（声明式） | V3 更清晰，可视化 |
| **并行执行** | `asyncio.gather()` | Graph fan-out/fan-in | V3 自动处理，无需手写 |
| **状态管理** | StateManager + Redis | State TypedDict + Checkpointer | V3 内置持久化 |
| **对话恢复** | 手动加载/保存 | `thread_id` 自动关联 | V3 零代码实现 |
| **人机交互** | 手动循环处理 | `interrupt` 机制 | V3 更优雅 |
| **Agent 数量** | 10 个 | 10 个 | 不变 |
| **Skill 数量** | 8 个 | 8 个 | 不变 |
| **数据模型** | Pydantic | Pydantic（相同） | 不变 |
| **数据库表** | 5 张表 | 5 张表 + checkpointer 表 | 增加 2-3 张 LangGraph 系统表 |
| **EventBus** | Agent 间通信 + 前端推送 | 仅前端推送 | 缩小职责 |
| **代码量** | ~800 行（Orchestrator + StateManager） | ~400 行（Graph 定义 + 节点） | V3 减少约 50% 编排代码 |
| **学习成本** | 低（纯 Python） | 中（需理解 Graph 概念） | V3 需要 1-2 天学习 |
| **调试难度** | 低（标准 stack trace） | 中（Graph 内部错误） | V3 需要 LangGraph 调试技巧 |
| **依赖重量** | 轻（FastAPI + SQLAlchemy + Redis） | 中（+ langgraph + psycopg） | V3 增加约 30MB 依赖 |

---

## 十二、实施建议

### 12.1 如果决定用 V3（LangGraph）

**实施步骤**：

1. **Day 1**：安装依赖（langgraph, langgraph-checkpoint-postgres），理解 State/Node/Edge 概念
2. **Day 2**：定义 `ItineraryState`，实现 `intent_node` + `route_intent`，跑通主图框架
3. **Day 3**：实现 Generate Subgraph（poi_search / weather / budget 并行 + planner + validation/route/budget 并行 + proposal）
4. **Day 4**：实现 `interrupt_node` + FastAPI WebSocket 集成
5. **Day 5**：实现剩余节点（update_prefs, qa, confirm, history）+ 前端推送

**Agent/Skill 复用策略**：
- 完全复用 V2 的 Agent 类（IntentRecognitionAgent、ItineraryPlannerAgent 等）
- 完全复用 V2 的 Skill 类（WebSearchSkill、POISearchSkill 等）
- 只需把 Agent 的调用封装成 Node 函数

### 12.2 如果决定继续用 V2

LangGraph 不是必须的。V2 的 asyncio + EventBus 方案完全可行，且更轻量。

**建议用 LangGraph 的情况**：
- 团队有人熟悉 LangGraph，或愿意花 1-2 天学习
- 希望减少手写编排代码（省 ~400 行）
- 人机交互场景多（确认行程、多轮修改）
- 需要对话持久化/恢复（Checkpointer 很方便）

**建议用 V2 的情况**：
- 追求最轻量依赖
- 团队不熟悉 LangGraph，不想增加学习成本
- 对调试可控性要求高（标准 Python 调试更直接）

---

## 十三、风险与缓解

| 风险 | 缓解策略 |
|------|---------|
| LangGraph 版本更新导致 API 变更 | 锁定版本 `langgraph==0.2.x`，升级前充分测试 |
| Graph 内部错误难以调试 | 开启 LangGraph 的 `debug=True`，使用 LangSmith 追踪 |
| Interrupt 与 WebSocket 集成复杂 | 参考官方示例 `chatbot_interrupt`，封装为通用模式 |
| Checkpointer 性能瓶颈 | 高并发时用 RedisCheckpointer 替代 PostgresSaver |
| 子图嵌套过深 | 保持子图扁平，最多 2 层嵌套 |

---

**文档结束**

> 本编排层与 V2 的 Agent、Skill、数据模型完全兼容。如决定采用，只需实现 Graph 定义（约 400 行）+ Node 函数（约 300 行），即可替换 V2 的 Orchestrator + StateManager + EventBus（约 800 行）。
