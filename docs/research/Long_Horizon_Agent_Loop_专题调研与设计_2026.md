# TravelAgent2 Long-Horizon Agent Loop 专题调研与设计

> 调研日期：2026-08-12  
> 研究问题：Agent 在线运行时，如何稳定完成需要多次查询、求解、校验、追问和重规划的长任务？  
> 范围：本文件研究运行时 Agent Loop；SFT、GRPO 和 Reward 的训练设计另见 `Agentic_RL_文献调研与SPEC修订建议_2026.md`。

## 1. 结论

TravelAgent2 不应采用“一个 ReAct Loop 从头想到尾”的扁平结构，也不需要为了长任务堆叠大量 Agent。建议采用：

> **确定性控制器外壳 + Planner 维护动态任务 DAG + Executor 对当前子任务运行短 ReAct + 程序 Verifier 验收 + 触发式局部重规划 + 分层状态与 Checkpoint。**

白话理解：

- 先把旅行目标列成一张可修改的任务清单；
- 一次只让模型做好当前一小步；
- 每完成一项就用工具真实结果或 Validator 打勾；
- 出错时只修改受影响的后续任务，不推倒整张行程；
- 无论模型怎么判断，程序都保存事实、进度、预算和依赖；
- 任务清单为空也不等于完成，最后必须做一次全局校验。

## 2. 为什么单纯 ReAct 不够

ReAct 的价值是让模型交替执行“判断下一步 → 调用工具 → 读取 Observation”，适合解决当前局部问题。但旅行长任务还包含它没有单独解决的控制问题：

- 十几步以后能否记住全部硬约束；
- 哪些信息已经查过、哪些还缺失；
- 两项工具结果之间有什么依赖；
- 一个子任务失败后，哪些后续步骤应失效；
- 用户新增约束时，应局部调整还是整体重来；
- Agent 自称完成时，环境是否真的满足目标。

TravelPlanner 的错误分析显示，Agent 会用错日期、重复错误调用、取回正确事实后又在结果中写错，甚至 reasoning 与 action 不一致。DeepPlanning 进一步把旅行长任务的主要失败归为信息获取失败、局部约束失败和全局组合优化失败。因此，ReAct 应放在**子任务内部**，而不是充当唯一的全局任务管理器。

## 3. 文献形成的共同模式

| 研究 | 正式状态 | 贡献 | 对 TravelAgent2 的采用方式 |
|---|---|---|---|
| ReAct | ICLR 2023 | 判断、行动、观察交替 | 作为当前子任务的微循环 |
| Reflexion | NeurIPS 2023 | 用环境反馈形成失败反思 | 只生成结构化失败摘要，且必须由环境证据支持 |
| TravelPlanner | ICML 2024 Spotlight | 真实旅行工具与多约束错误分析 | 建立工具、事实、约束和交付分层指标 |
| LATS | ICML 2024 | 搜索多个行动分支并用环境反馈选择 | 仅对高不确定、可回滚的决策做有限分支，不全程树搜索 |
| Plan-and-Act | ICML 2025 | Planner/Executor 分离、动态重规划 | 采用角色分离；改为事件触发式重规划以控制成本与抖动 |
| ACON | arXiv 2025/2026 版本 | 压缩历史同时保留任务状态 | 结构化状态为真相，历史摘要只做上下文投影 |
| DeepPlanning | ACL 2026 Long Paper | 多日旅行的主动取数、局部约束与全局优化 | 设计分阶段执行和三级 Verifier |
| TravelBench | ACL 2026 Long Paper | 多轮偏好、工具沙箱、不可解任务 | 增加能力边界检查和 `ask_user/abort` 正确性 |
| TripTide | Findings of ACL 2026 | 中断下的旅行重规划与意图保持 | 重规划评估“响应、原意保持、语义/空间/顺序适配” |
| MUSE | Findings of ACL 2026 | 子任务队列、短 ReAct、反思、重试与最终复核 | 采用任务队列/图与每子任务验证；不直接照搬多 Agent 结构 |

## 4. 目标运行时架构

```text
User Goal / External Event
          │
          ▼
┌────────────────────────────────────────────┐
│ Deterministic Orchestrator                 │
│ 权限、预算、循环、并发、Checkpoint、回退    │
└──────────────┬─────────────────────────────┘
               ▼
┌────────────────────────────────────────────┐
│ Goal & Constraint Ledger                   │
│ 原始目标、硬约束、软偏好、用户授权、能力边界 │
└──────────────┬─────────────────────────────┘
               ▼
┌────────────────────────────────────────────┐
│ Planner / Replanner                        │
│ 产生并维护带依赖、完成标准和失效范围的任务 DAG│
└──────────────┬─────────────────────────────┘
               ▼
┌────────────────────────────────────────────┐
│ Scheduler                                  │
│ 选择 READY 子任务；只并行无依赖的只读查询    │
└──────────────┬─────────────────────────────┘
               ▼
┌────────────────────────────────────────────┐
│ Bounded Executor Loop                      │
│ action → guard → tool → observation        │
│ 当前子任务最多 N 步                        │
└──────────────┬─────────────────────────────┘
               ▼
┌────────────────────────────────────────────┐
│ Verifier                                   │
│ 工具结果 / 子目标 / Solver / 全局约束验收    │
└───────┬──────────────┬──────────────┬──────┘
        │成功          │可恢复失败    │缺信息/授权
        ▼              ▼              ▼
   推进任务 DAG     局部重试/重规划   ask_user interrupt
        │
        ▼
所有任务结束 → Global Validator → 用户确认 → finish
```

模型可以同时承担 Planner 和 Executor 两种角色，但必须使用不同输入契约；首版不需要部署两个不同模型。将角色分开，是为了控制上下文和职责，不是为了增加“多 Agent”包装。

## 5. 任务 DAG，而不是一次性 Todo List

### 5.1 子任务结构

```json
{
  "id": "collect_poi_details",
  "goal": "获得入选 POI 的营业时间、票价和建议时长",
  "status": "pending|ready|running|blocked|succeeded|failed|invalidated|skipped",
  "depends_on": ["search_candidates"],
  "required_facts": ["candidate_poi_ids"],
  "allowed_actions": ["get_poi_detail"],
  "success_criteria": {
    "all_required_fields_present": true,
    "source_must_be_observation": true
  },
  "artifacts": ["poi_detail_snapshot_ids"],
  "attempts": 0,
  "max_attempts": 2,
  "failure": null,
  "invalidates_on": ["destination_changed", "travel_dates_changed"]
}
```

### 5.2 旅行规划的默认任务图

```text
G0 理解目标与能力边界
 ├─ G1 补齐必要信息 / 澄清隐含偏好
 ├─ G2 获取日期相关事实：天气、开放状态、交通可用性
 ├─ G3 搜索并筛选 POI / 酒店 / 城际交通候选
 │    └─ G4 获取候选详细事实
 └──────────────┬─────────────────────┘
                ▼
           G5 构造 SolverRequest
                ▼
           G6 确定性求解
                ▼
           G7 全局 Validator
             ├─ 通过 → G8 生成草稿 → 用户确认
             ├─ 可自动修复 → 回到受影响的 G2/G3/G4/G5
             └─ 需取舍 → ask_user → 更新约束后局部失效
```

任务图只描述“要达到什么状态”，不把具体工具参数提前写死。工具参数必须由 Executor 根据当前事实生成，否则初始计划中的错误会被机械执行到底。

## 6. 三层状态，解决长上下文漂移

### 6.1 权威结构化状态

这是唯一事实源，由程序维护，不能由 LLM 摘要覆盖：

- `goal_ledger`：原始目标、硬约束、软偏好、优先级、用户授权；
- `task_graph`：子任务、依赖、状态、尝试次数、失效原因；
- `fact_store`：事实值、来源 tool call、snapshot、时间戳、有效期；
- `artifact_store`：候选集、路线矩阵、SolverRequest/Response、Validator 报告；
- `budget_ledger`：步数、工具次数、Solver 次数、Token、时间和费用；
- `failure_ledger`：失败分类、证据、已尝试策略和禁止重复项；
- `plan_versions`：每次重规划的 diff、原因和受影响范围。

### 6.2 模型工作上下文

每一步只投影：

- 当前子任务及成功标准；
- 当前子任务依赖的事实引用；
- 相关硬约束和软偏好；
- 最近少量原始 Action/Observation；
- 失败摘要和剩余预算；
- 允许动作。

### 6.3 可压缩叙事摘要

只用于帮助模型理解历史，至少保留：

- 已完成进度；
- 关键决定及原因；
- 尚未解决的问题；
- 重要变量/实体 ID；
- 已失败的方法；
- 用户明确授权和不可修改项。

摘要与结构化状态冲突时，以结构化状态为准。每次压缩后做一致性检查；若日期、预算、must-visit、POI ID、任务状态或失败原因丢失，拒绝该摘要并回退上一个版本。

## 7. 执行、验证与进度推进

### 7.1 一个子任务内的短 Loop

```text
读取 current_subtask
  → 判断所需的一个或一批动作
  → Guard 校验
  → 执行工具
  → 写入 Fact/Artifact Store
  → Verifier 检查 success_criteria
  → succeeded / retryable_failed / blocked / terminal_failed
```

不允许模型用一句 `done` 改变任务状态。任务状态转换只能由 Verifier 或确定性控制器提交。

### 7.2 三级验证

1. **Action-level**：工具允许、参数 Schema 正确、参数有来源、没有越权或重复；
2. **Subtask-level**：该子目标的完成条件确实满足，产物可用且未过期；
3. **Global-level**：任务图已闭合，Solver/Validator 通过，原始目标和用户授权均未丢失。

### 7.3 并行边界

可以并行：

- 独立 POI 详情查询；
- 酒店、天气、城际交通等互不依赖的只读查询；
- 同一候选集上的路线矩阵分片。

必须串行：

- 依赖上一步 ID 或用户回答的动作；
- Solver 前的数据汇总；
- 任何会修改任务状态、计划版本或外部世界的动作；
- Validator 后的终止决策。

并行工具结果必须先由 Join Barrier 校验完整性，再统一写入事实库，避免谁先返回谁覆盖状态。

## 8. 触发式局部重规划

Plan-and-Act 证明动态重规划有价值，但每一步都调用 Planner 会增加成本并导致计划抖动。TravelAgent2 使用事件触发：

### 8.1 重规划触发器

- 当前子任务经过允许重试仍失败；
- Observation 与计划假设冲突；
- 必要事实为空、过期或发生变化；
- Solver 返回结构化 infeasible conflict；
- Validator 发现硬约束或全局结构问题；
- 用户修改目标、预算、日期、must-visit 或优先级；
- 外部事件导致活动/路线不可用；
- 预计剩余预算无法完成当前计划。

### 8.2 重规划范围

```text
事实刷新     → 失效直接依赖该事实的子任务及其后代
POI 替换     → 重算相关详情、路线、Solver 和 Validator
某日天气变化 → 只失效该日未开始活动及跨日资源依赖
预算变化     → 失效候选筛选、Solver、Validator
日期/城市变化→ 大范围失效，必要时新建计划版本
文案修改     → 不重跑 Solver
```

已完成、已预订和用户锁定的活动默认不可失效；若确实需要改变，必须进入 `propose_tradeoff`。

### 8.3 计划版本与防抖

每次重规划记录：

- `trigger`、`evidence`；
- `invalidated_task_ids`；
- `preserved_task_ids`；
- `changed_constraints`；
- 新旧计划 diff；
- 是否需要用户确认。

若连续两次产生同构剩余计划且没有新事实，禁止再次重规划，转入追问、降级或终止。

## 9. 失败恢复策略

失败分类与动作固定映射：

| 失败 | 首选处理 | 上限后处理 |
|---|---|---|
| 参数 Schema 错 | 同一步修正参数 | 标记策略错误并 fallback |
| 上游超时 | 幂等重试或 fallback provider | 保留缺失事实并评估能否继续 |
| 空结果 | 放宽非硬检索条件或换查询 | 询问用户/声明不可行 |
| 事实冲突 | 查询权威源并标记版本 | 请求用户选择，禁止模型猜测 |
| Solver infeasible | 根据 conflict 做最小范围重规划 | `propose_tradeoff` |
| Validator 不通过 | 失效责任子图 | 达上限后回固定流程 |
| 用户信息不足 | `ask_user` interrupt | 安全终止，不循环追问同一问题 |
| 工具缺失/越权 | 能力边界终止 | 告知可完成与不可完成部分 |

失败摘要必须包含环境证据、已尝试方法和禁止重复动作；不能让模型自由编写没有证据的“反思”。

## 10. 终止协议

`finish` 是一种申请，不是模型单方面宣告成功。控制器仅在以下条件同时成立时接受：

1. 所有 required 子任务为 `succeeded` 或经规则批准的 `skipped`；
2. 没有 `running/blocked/invalidated` 子任务；
3. 关键事实仍在有效期内；
4. Solver 产物与当前 `goal_ledger`、`fact_store`、计划版本一致；
5. Global Validator `hard_pass=true`；
6. 输出中每个关键事实能回指 Observation；
7. 涉及用户取舍或修改锁定项时已有明确授权；
8. 未超过安全、时间和成本边界。

正常终止类型应区分：

- `validated_finish`：成功完成；
- `awaiting_user`：等待必要信息或取舍；
- `partial_finish`：用户接受明确标注的部分结果；
- `unsolvable_missing_info`；
- `unsolvable_missing_tool`；
- `unsolvable_constraints`；
- `budget_exhausted_fallback`；
- `safety_abort`。

## 11. 中断、Checkpoint 与恢复

Checkpoint 应在以下边界落盘：

- Planner 创建或更新任务图后；
- 每个工具结果和子任务状态成功提交后；
- Solver/Validator 产物生成后；
- `ask_user` 或 `propose_tradeoff` 前；
- 用户确认和计划版本切换后。

恢复时按顺序检查：

1. Schema 与迁移版本；
2. 当前任务是否停在安全边界；
3. 正在执行动作是否已成功提交；
4. 工具幂等键是否已有结果；
5. 外部事实是否过期；
6. 剩余预算是否延续，而不是重置；
7. 若状态不一致，从最近安全 Checkpoint 恢复并重建模型上下文。

所有工具调用使用 `trajectory_id + task_id + action_id` 作为幂等键。恢复后不能重复执行已成功的外部动作。

## 12. 对当前 TravelAgent2 的适配判断

项目已经具备可复用地基：

- `AgentState`、LangGraph、Checkpoint 和 interrupt/resume；
- 并行的 profile/weather/retrieval 分支；
- Solver、fact check、hallucination check；
- 用户确认与局部行程重规划；
- 工具调用、错误处理和 fallback。

缺少的不是另一个宏大框架，而是以下运行时数据结构和控制节点：

1. `goal_ledger`；
2. `task_graph` 与 Scheduler；
3. `fact_store/artifact_store` 的引用式上下文；
4. `subtask_verifier`；
5. `replan_decider` 和依赖失效传播；
6. `global_completion_guard`；
7. durable budget、幂等动作和恢复一致性检查。

现有 `replan_local` 面向行程草稿修改，可以继续保留；新增的 `replan_decider` 负责更早期的“任务图剩余部分怎么调整”，两者职责不同。

## 13. 推荐实现顺序

### Loop-P0：任务账本 MVP

- 定义 Goal/Task/Fact/Artifact/Failure/PlanVersion Schema；
- 用确定性模板创建第一版旅行任务 DAG；
- Executor 每次只处理一个 `ready` 子任务；
- 程序控制状态转换和最终完成门禁。

### Loop-P1：动态恢复

- 接入 Solver conflict 和 Validator violation 到责任子图；
- 实现事件触发式局部失效与重规划；
- 支持 `ask_user` 后继续原 DAG；
- 加入幂等键、预算延续和 Checkpoint 恢复测试。

### Loop-P2：上下文与性能

- 引用式 Fact/Artifact Store；
- 结构化压缩摘要和压缩一致性测试；
- 对无依赖只读任务做受控并行；
- 用实际成功率决定是否引入有限分支搜索。

### Loop-P3：训练闭环

- 收集 Planner、Executor、Verifier、Replanner 的独立行为标签；
- SFT 先学习正确任务图和局部执行；
- GRPO 优化策略选择、恢复和工具效率；
- Reward 映射到对应子任务/turn，而不是只看最终文案。

## 14. 必做评测

除端到端成功率外，必须报告：

- `Subtask Completion Precision`：被标记成功的子任务中真实成功比例；
- `Goal Coverage`：原始目标和硬约束的覆盖比例；
- `Dependency Violation Rate`：依赖未满足却执行的比例；
- `Replan Precision`：触发重规划后确实需要改变剩余计划的比例；
- `Replan Scope Ratio`：被失效任务数 / 剩余任务数，越小且成功越好；
- `Intent Preservation`：重规划后保留用户原意和锁定项的比例；
- `Recovery Success Rate`；
- `False Finish Rate`：Agent 申请 finish 但全局门禁不通过的比例；
- `Loop/Stall Rate`；
- `Resume Consistency Rate`；
- `Peak Context Tokens` 和压缩前后任务成功差异；
- 按 1–5、6–10、11–14+ 步分桶的成功率，而不是只报平均值。

核心对照：

1. 当前固定编排；
2. 扁平 ReAct；
3. Planner + Executor 静态计划；
4. 任务 DAG + 触发式局部重规划；
5. 第 4 组 + 结构化上下文压缩。

## 15. 最终定义

在这个项目里，“Long-Horizon Agent Loop”不是模型能连续调用十次工具，而是：

> 它能在持久化目标和任务图约束下，持续获取真实信息、完成并验证子目标；环境变化或执行失败时，只重规划受影响部分；跨中断恢复后仍保持目标、事实、预算和执行幂等；最后由全局 Validator 而不是模型自报完成。

达到这个标准后，Agent Loop、SFT、GRPO 和 Reward 才真正连接成一套系统。

## 16. 官方来源

- [ReAct / ICLR 2023](https://arxiv.org/abs/2210.03629)
- [Reflexion / NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html)
- [TravelPlanner / ICML 2024](https://arxiv.org/abs/2402.01622)
- [LATS / ICML 2024](https://proceedings.mlr.press/v235/zhou24r.html)
- [Plan-and-Act / ICML 2025](https://proceedings.mlr.press/v267/erdogan25a.html)
- [ACON](https://arxiv.org/abs/2510.00615)
- [DeepPlanning / ACL 2026](https://aclanthology.org/2026.acl-long.335/)
- [TravelBench / ACL 2026](https://aclanthology.org/2026.acl-long.1347/)
- [TripTide / Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.2002/)
- [MUSE / Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.1522/)
