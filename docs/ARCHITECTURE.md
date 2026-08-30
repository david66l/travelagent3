# 系统架构

## 概览

TravelAgent2 采用前后端分离 + 异步任务架构：

```text
Browser ──POST──> Gateway ──> FastAPI ──commit──> PostgreSQL PlanningJob
   │                                      │
   │                                      └──dispatch──> Celery/Redis
   │                                                       │
   └──SSE(last_event_id)──> FastAPI <──wake-up/pubsub── Worker/LangGraph
                                  └──replay── PostgreSQL PlanningJobEvent
```

## 后端分层

| 层 | 目录 | 职责 |
|----|------|------|
| API | `backend/src/api/` | HTTP/SSE 入口、鉴权、限流 |
| Services | `backend/src/services/` | 业务编排 |
| Agents / Planner | `backend/src/agents/`, `planner/` | 意图识别、行程规划 DAG |
| Skills / Tools | `backend/src/skills/`, `tools/` | 外部数据（搜索、天气、POI） |
| Core | `backend/src/core/` | 配置、Redis、LLM、安全、成本 |

## 核心数据流（聊天 + 规划）

1. 用户通过 `POST /api/v1/chat/message` 提交消息，FastAPI 在同一事务中保存用户消息和 `PlanningJob`
2. 事务提交后，API 把 `job_id` 派发到 Celery；Worker 通过数据库租约认领任务并执行 LangGraph
3. Worker 把公开的 Graph 事件写入 `PlanningJobEvent`，再用 Redis Pub/Sub 通知在线 SSE 连接及时拉取
4. 客户端携带 `job_id + last_event_id` 建立 SSE；断线或切换 FastAPI 副本后，从 PostgreSQL 事件日志继续重放
5. Redis 只承担缓存、队列、实时通知与热状态；关键任务状态、事件游标和最终结果以 PostgreSQL 为准

`PlanningJob` 同时承担轻量级事务 Outbox 的角色：如果事务已经提交但首次 Broker 发布失败，Celery Beat 每 30 秒重新派发超过保护窗口的 `pending/retrying` 任务。重复消息由 Worker 的原子租约去重。

## Agent Loop 与工具决策

部署默认 `AGENTIC_EXECUTION_MODE=react`。系统先由轻量意图模型通过 `SlotParseOutput` 一次性识别对话意图、旅行槽位、活动主体、城际交通方式和外部事实需求，再补齐只有用户能回答的信息并进入持久化 ReAct 研究循环。`GoalLedger` 和 `ResearchSufficiencyVerifier` 直接消费这份结构化结果，不再二次扫描“演唱会、高铁、营业时间”等关键词猜意图。模型每轮根据目标、现有 Evidence、工具 Observation、失败码和剩余预算，自主选择下一项只读工具或提出结束研究；Controller 只负责安全允许列表、可信参数注入、预算和状态写入。模型无权直接修改任务状态，也无权自行宣布证据充分或约束已经通过。

```text
结构化 LLM 意图识别 / 按需补槽
  → research_evidence ReAct 循环
      Policy 选择知识库、POI、天气、统一网页搜索、航班/火车或路线工具
      Guard + Tool Executor → Observation → Fact / Artifact Store
      Policy 根据新 Observation 再决策，或提出 finalize_research
  → ResearchSufficiencyVerifier（按意图检查覆盖与时效）
  → CP-SAT（固定演出/预约作为日期和时间硬约束）
  → ItineraryValidator 硬校验
  → accept / 补证据 / retry_solve / ask_user / tradeoff / abort
  → 只从验证产物润色 → 用户确认或带原因修改
      修改意见由同一意图模型输出受控 RevisionOperation
      Controller 校验字段白名单、类型和边界后生成新 Goal/Plan Version
```

循环中存在两个主要的非单步决策点：

- `research_evidence`：模型可动态改变工具顺序。普通行程要求城市知识、足量 POI、POI 详情和路线矩阵；近期天气、演出、交通或营业时间证据只在对应意图下追加。过早 `finalize_research` 会得到具体缺口码并继续循环；重复相同参数且得到相同证据会被判为 `REPEATED_NO_PROGRESS_ACTION`。
- `review_itinerary`：Verifier 硬通过后才能 `accept_itinerary`；未通过时可 `retry_solve` 触发局部重规划，或重新搜索触发全局重规划。重规划会提升 `plan_version`、失效旧 Solver/Verifier 产物并只重开必要子图。

活动不是有限枚举。`search_current_info` 是统一的来源搜索工具，演唱会、音乐节、展览、球赛、市集和季节活动都通过 `query` 与 `info_type=event` 调用；执行器再从网页证据中抽取日期、时间、场馆和来源，并对场馆做坐标落地。系统没有 `search_event` 或 `get_local_events` 这类按活动品类扩张的 Agent 工具。航班/火车仍单独建模，因为它们对应结构化班次接口和不同的参数、时效与验证契约。

实时搜索结果不能只作为展示文本。活动证据被转换为固定日期/时间预约；与候选 POI 精确匹配的营业时间和临时闭馆证据被转换为 `date_opening_hours/closed_dates`；来源可追溯的进出城班次被转换为首日 `daily_start_minutes` 和末日 `daily_end_minutes`。这些字段同时进入 CP-SAT 和 `ItineraryValidator`。搜索结果缺少实体匹配、明确时间或来源时，`ResearchSufficiencyVerifier` 返回 `CURRENT_INFO_NOT_PLANNABLE` 或 `TRANSPORT_SCHEDULE_NOT_PLANNABLE`，不能用“搜到链接”冒充已进入规划。

LangGraph 每执行一个串行动作或一组无副作用的并行只读动作就保存一次 checkpoint。API 与 Celery Worker 使用同一套 PostgreSQL checkpointer；用户回答澄清问题后，系统恢复原 `trajectory_id`、预算、失败记录和 Decision History，从阻塞子任务继续，而不是从头重跑。Agent 失败会显式停止，绝不静默切回旧 Planner 掩盖错误；确认后的事实核验如果发现冲突，也不会覆盖已经由 Agent + Verifier 生成的行程。

自然语言修改不是用正则提取“几天/预算/轻松”等词。意图模型输出 `set/add/remove/clear` 操作及受影响域，Controller 只接受约定字段并校验天数、预算、布尔值和枚举；模型不可用时只保留原始反馈并安全重规划，不猜测约束变更。前端按钮产生的结构化单点编辑仍可直接进入局部校验，因为它本身不是待理解的自然语言。

仓库保留旧 `controller_first` 和 `policy_driven` Task-DAG 作为消融基线。它们不能用于证明新 ReAct 链路已经跑通，也不与新架构指标混报。

新一轮 SFT/GRPO 暂停到 ReAct 真实全链路验收后再启动。训练环境必须复用上述生产 Evidence/CP-SAT/Verifier 语义，SFT 只接收真实或教师生成后经工具与硬验证通过的轨迹；GRPO 分别奖励合法工具、证据覆盖/新鲜度、硬通过、失败恢复、用户修改成功和效率。旧 DAG 训练结果只作为历史基线。

每个 GRPO rollout 的初始消息只允许 `system + 原始用户请求`，随后由环境创建新的 Ledger 并返回首个策略状态；携带 assistant/tool 教师轨迹前缀的样本会在 `reset` 时被拒绝。每轮审计保存允许动作、模型动作、工具观察、Verifier 结果和 turn reward。完成长度预算按完整 rollout 的累计工具结果计算，而不是只估算一次工具往返，避免因 token 上限过小把后续模型轮次静默截断。

## 状态所有权与恢复

| 数据 | 权威存储 | 说明 |
|------|----------|------|
| 用户消息、任务、最终结果 | PostgreSQL | 事务持久化和审计 |
| 可重放任务事件 | PostgreSQL `PlanningJobEvent` | SSE 依据 `last_event_id` 恢复 |
| Agent 执行断点 | LangGraph Checkpoint | 确认、修改和异常恢复 |
| Agent 工作记忆 | Agent Ledger | 目标/约束、任务 DAG、Facts、Artifacts、失败与动作历史 |
| 热会话与实时通知 | Redis | 可丢失缓存，不作为最终事实源 |
| FastAPI 进程内队列 | 无关键状态 | 仅承载当前连接，不能决定任务是否完成 |

## 训练—推理工具契约

线上 Agent、Teacher 轨迹、SFT/DPO 数据生成和离线评测共用工具参数模型。
快照环境按 `tool_name + normalized arguments` 选择响应；没有显式参数契约的旧快照才按顺序回放。
因此候选过滤顺序变化不会再把其他 POI 的 Observation 错配给当前工具调用。

## M6 成本优化模块

| 模块 | 作用 |
|------|------|
| `cost_circuit_breaker` | 全局日 Token / API / GPU 成本熔断 |
| `user_tier` + `token_quota` | 游客/免费/会员分级配额 |
| `model_router` | 简单意图与熔断时走小模型 |
| `prompt_compress` | 长对话截断，降低 Token |
| `external_api_tracker` | Tavily 等外部调用计数与成本估算 |

## 部署拓扑（K8s）

见 `k8s/`：backend、celery、gateway、frontend、Redis、Postgres、HPA、PDB。

可观测栈见 `monitoring/` + `docker-compose.observability.yml`。
