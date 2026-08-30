# Long-Horizon Agentic RL 最新论文精读与 TravelAgent2 技术选型

日期：2026-08-13  
调研范围：2025-2026 年多轮 Agent RL、轮级信用分配、工具失败恢复、课程学习、rollout 优化、可验证 Reward。  
离线论文库：`E:\A_Louis\大模型就业体系课\papers\agentic_rl_long_horizon`。

## 1. 结论先行

本轮共下载并解析 29 篇 PDF、664 页。对 TravelAgent2 当前问题最有价值的，不是再增加一个复杂 Reward 模型，而是按以下顺序解决：

1. **先让模型具备最低恢复能力。** 当前自适应恢复任务几乎全失败，GRPO 没有可比较轨迹。先用与真实 Agent Loop 完全一致的多轮 SFT 历史教会“读错误→改参数→重试”。
2. **训练只使用可学习任务。** 同一初始状态至少 4 个 rollout；全成功、全失败以及局部轮级零方差组不更新策略。
3. **局部有效性必须门控未来收益。** 格式错误、动作不允许、参数无依据的动作不能因为最终轨迹成功而得到正信用。
4. **只给模型自己的决策轮分配信用。** Controller 自动步骤、工具返回、系统消息和 padding 不进入策略梯度。
5. **过程与终局信号分层且分开统计。** 格式/工具/grounding/效率是过程事实；约束满足/计划质量是终局结果，不能无条件线性相加后再复制给全部轮次。
6. **先做低成本 R1，再做研究型 R2。** R1 使用环境可验证信号；SALT/TSR/TRACE/HCAPO 等放入后续公平对照，避免一次引入多个变量后无法解释收益。

## 2. 当前问题为什么发生

### 2.1 新恢复任务全失败，不是“多训练几步”能解决

模型要完成的实际决策链是：

```text
搜索多个兴趣词
  → 工具返回 QUERY_TOO_BROAD，并给出可验证的缩窄依据
  → 模型读取新的 policy_state.failure_summary
  → 只保留被指定的关键词重试
  → 后续 Controller/Solver/Validator 完成规划
```

当前 3B SFT 模型没有在完全一致的消息历史中学过这种转移，因此 32 次审计几乎全部失败。WebAgent-R1 的消融和实践指南均支持：当正确行为基本探索不到时，先做少量行为克隆/进度感知 warm-up，比直接使用稀疏 RL 更可靠。

### 2.2 第一批轮级 GRPO 为什么看似运行、实际没学

同组成功轨迹的两次搜索参数完全相同，局部奖励桶没有方差。GRPO 的核心是组内相对优势；若一组回报完全相同，标准化优势为 0，策略梯度也为 0。RAGEN、Structured Reflection、TSR、IGPO/CIGPO 都把这类现象视为必须显式处理的训练失败模式。

因此，`train_loss` 有数值、checkpoint 被写出，并不等于发生了有效学习。项目当前新增的 `effective_nonzero_credited_turns` 门禁是必要的。

### 2.3 原 R1 线性混合仍可能误奖无效动作

若使用：

```text
turn_credit = local_process_score + discounted_episode_reward
```

一条最终成功轨迹中的无效工具调用，可能因为正的终局回报而得到正信用。ToolVerse 的一致性门控以及局部/全局解耦研究指出，应先判断本轮动作在当前状态是否有效，再决定是否传播未来收益。

这里必须区分：

- **动作无效**：格式错误、工具不允许、参数不合法、参数无证据、重复成功调用。
- **环境失败**：合法调用遇到超时、上游不可用或可重试业务错误。

合法动作遇到环境失败不应被当作策略错误；模型是否能依据 Observation 正确恢复，才是下一轮需要评价的行为。

## 3. 论文方法与项目适配判断

| 方法路线 | 代表论文 | 适配性 | 决策 |
|---|---|---:|---|
| 多轮 SFT/行为克隆冷启动 | WebAgent-R1、PROGRA、Structured Reflection | 很高 | 立即采用 |
| 低方差组过滤与稳定性监控 | RAGEN、TSR、CIGPO | 很高 | 立即采用 |
| 同轮局部优势 + 门控未来优势 | ToolVerse、MT-GRPO | 很高 | 作为 R1-v2 核心 |
| 共享前缀轨迹图 | SALT | 高 | R1 稳定后做 R2 对照 |
| 树式 rollout / lookahead | TSR、TRAE、IGRPO | 中高 | 算力允许时做 R2 |
| 冻结参考模型信息增益 | TRACE、IGPO、CIGPO | 中 | 旅行规划无唯一答案，改成状态势函数后再试 |
| 事后 LLM critic / 隐式 PRM | HCAPO、iStar | 中低 | 成本高且可验证性弱，暂缓 |
| 显式自由文本反思 | Structured Reflection | 中 | 不新增私有 CoT；使用结构化 failure/progress 状态替代 |
| PPO / turn-PPO | Turn-PPO、实践指南 | 中 | GRPO 长程不稳定时的正式备选基线 |

## 4. 对 Agent Loop 的修订

### 4.1 保持现有生产循环，不建立训练专用假 Agent

生产与训练继续共享：

```text
PolicyContext
  → PolicyAction
  → Guard / Schema
  → TravelActionExecutor
  → ObservationEnvelope
  → Task/Validator transition
  → 下一 PolicyContext
```

Agent Lightning 的关键启发是执行与优化解耦，而不是更换业务 Agent。现有 `InteractiveAgentSession`、快照执行器和 TRL 环境适配器已经满足这一方向。

### 4.2 把进度感知放入结构化状态，不保存私有思维链

PROGRA 的“历史摘要 + 未来计划”与 TravelAgent 当前字段可以直接对应：

| PROGRA 信息 | TravelAgent 字段 |
|---|---|
| 用户目标摘要 | `original_request`、`hard_constraints`、`soft_preferences` |
| 已完成进度 | 当前任务状态、`relevant_artifacts`、`relevant_facts` |
| 失败诊断 | `failure_summary` |
| 下一步计划 | `current_subtask`、`remaining_tasks`、`allowed_actions` |

因此不需要强迫模型输出长篇 `<think>` 或自由文本 reflection。训练目标仍是结构化工具动作；状态本身提供足够的进度依据。

### 4.3 决策边界必须严格

- 只把 `decision_source != controller` 的步骤计为模型决策轮。
- 工具 observation 永久 mask，不参与 loss。
- 信用距离按模型决策轮计算，不能让 Controller 的自动步骤稀释折扣。
- 变长轨迹不能只按原始 `turn_index` 强行对齐；优先按初始任务、状态指纹和决策阶段分桶。

## 5. 对 SFT 的修订

### 5.1 自适应恢复数据的最低消息合同

```text
system: 生产 Agent Policy system prompt
user: reset 后的真实 policy_state
assistant: 第一次合法 search_pois 工具调用
tool: 真实 TRL transition，包含错误 observation 与新的 policy_state
assistant: 基于 failure_summary 缩窄参数后的第二次工具调用
```

要求：

- 第二次参数必须来自用户原始兴趣或工具错误中的可验证依据。
- 不把 hidden test facts 暴露给模型。
- 训练只来自官方 train 派生任务；validation/test 不参与训练。
- 加入 1:1 的既有能力 replay，降低灾难性遗忘。
- 数据同时保留成功恢复和安全终止，不把无脑重复当成恢复。

### 5.2 SFT 晋升条件

只有同时满足以下条件，才允许进入恢复场景 GRPO：

- 自适应恢复 validation 出现非零成功率。
- 4 个 rollout 的任务中存在成功/失败或局部决策差异。
- 原正式 holdout 的成功率、硬约束通过率和四类任务族无显著退化。
- 重复调用率、无依据参数率没有上升。

## 6. R1-v2：适合 TravelAgent 的轮级优势

### 6.1 轮级事实信号

对 rollout `i` 的模型决策轮 `t`，记录：

- `valid_format`：JSON/tool-call Schema 是否合法。
- `valid_action`：动作是否在当前 `allowed_actions`。
- `valid_arguments`：参数类型与业务约束是否合法。
- `grounded_arguments`：参数是否可追溯到用户请求、事实、artifact 或 observation。
- `tool_execution`：调用是否真实执行；区分策略无效与可重试环境失败。
- `state_progress`：任务是否推进、获得新事实、解除阻塞或正确进入安全等待状态。
- `duplicate_or_waste`：重复成功调用、无信息增益重试或无效循环。

### 6.2 建议的信用结构

第一层先构造局部有效性门：

```text
g_valid(i,t) =
  -1  本轮动作由模型造成不可执行/无依据错误
   0  无法可靠判断或仅遇到外部环境失败
  +1  动作合法、可执行且证据可追溯
```

第二层，在同一可比较状态桶内计算局部相对优势：

```text
A_local(i,t) = GroupNorm(r_local(i,t))
```

第三层，只对有效动作传播未来可验证收益：

```text
V_future(i,t) = 1[g_valid > 0] * Σ gamma^k * r_verified(i,t+k)
A_future(i,t) = GroupNorm(V_future(i,t))
```

最终轮级优势：

```text
if g_valid < 0:
    A_turn = negative_validity_credit
else:
    A_turn = A_local + lambda * A_future
```

终局 advantage 可以保留作为 B0 锚点，但不能把正终局分无条件覆盖到无效动作上。具体融合权重必须通过 B0/R1 对照选择，不从论文直接照抄。

### 6.3 六类 Reward 的职责重新固定

| Reward | 层级 | 用法 |
|---|---|---|
| `R_format` | 局部有效性 | 作为门控事实，不靠终局成功补偿 |
| `R_tool` | 局部有效性/进展 | 工具选择、真实执行、错误后策略变化 |
| `R_grounding` | 局部门控 | 无依据参数直接负信用；严重伪造触发 Episode gate |
| `R_efficiency` | 有效动作后的修正 | 只惩罚可证明的重复、循环和无增益调用 |
| `R_constraint` | 终局核心 | Solver/Validator 的硬约束与安全结果 |
| `R_quality` | 终局辅助/审计 | 规则可验证部分进入 Reward；主观文案质量保留离线评测 |

过程指标和终局指标分别归一化、分别记录，避免不同量纲互相冲掉。CIGPO 的分开归一化只作为设计依据，仍需本项目消融确认。

## 7. 课程与 rollout 策略

### 7.1 难度课程

```text
C0 单轮合法工具调用
 → C1 可重试 timeout，允许合理同参重试
 → C2 明确错误依据驱动的参数修复
 → C3 多工具链中的中途失败恢复
 → C4 用户补充信息与动态重规划
```

模型在某一级成功率为 0 时回到 SFT 或降低难度；成功率接近 100% 时不再用于该轮 GRPO 更新。

### 7.2 训练任务筛选

- 每个初始状态至少 `G=4` 个 rollout；算力允许时做 `G=8` 消融。
- 保存 `zero_advantage_group_ratio` 和每个 turn bucket 的方差。
- 优先训练成功率约 20%-80% 的任务。
- 任务族、城市、约束类型和链路长度都需要多样性。
- 每轮训练后重新采样，避免长期使用过期轨迹。

### 7.3 树式 rollout 放在 R2

TSR/SALT/TRAE 能提高信用精度，但会增加工程变量或 rollout 成本。顺序应是：

1. 先证明平坦 4-rollout 的 R1-v2 有非零有效信用。
2. 再用 SALT 合并相同动作/状态前缀，检查共享正确动作是否获得一致信用。
3. 最后做 best-of-N 或浅 lookahead，比较相同 GPU 预算下的收益，而不是只比较 rollout 数量。

## 8. 必做实验矩阵

| 实验 | 对照 | 主要指标 |
|---|---|---|
| E10 恢复 SFT | 原 SFT vs 恢复 SFT | 自适应恢复成功率、原 holdout 退化、重复调用 |
| E11 信用基线 | B0 整轨 GRPO vs R1-v2 | 成功率、平均 Reward、有效非零信用轮数 |
| E12 门控消融 | 线性相加 vs validity-gated | 无效动作正信用率、硬约束通过率 |
| E13 归一化消融 | 联合 vs 过程/终局分开 | zero-advantage ratio、梯度稳定性 |
| E14 group size | G=4 vs G=8 | 组内方差、样本效率、显存/时间 |
| E15 课程 | 固定最难 vs C0→C4 | 收敛速度、恢复泛化、旧能力保持 |
| E16 轨迹图 | R1-v2 vs R1-v2+SALT | 共享前缀信用一致性、训练开销 |
| E17 rollout 搜索 | naive vs best-of-N/浅 lookahead | 同 GPU 时间预算下的成功率 |

所有实验必须使用固定独立 holdout、配对 seed，至少报告 Base、SFT、B0、R1；不能只展示最优 checkpoint。

## 9. 监控与拒绝门禁

每次更新记录：

- `success_rate_by_family`
- `hard_constraint_pass_rate`
- `reward_mean/std`
- `zero_advantage_group_ratio`
- `effective_nonzero_credited_turns`
- `invalid_action_positive_credit_rate`
- `duplicate_tool_call_rate`
- `recovery_parameter_change_rate`
- `policy_entropy`
- `gradient_norm`
- `mean_model_turns`

以下情况直接拒绝 checkpoint：

- 有训练 loss，但 `effective_nonzero_credited_turns == 0`。
- 无效动作获得正信用。
- 平均 Reward 上升但硬约束通过率下降。
- 重复工具调用上升且任务成功率没有提升。
- 原正式 holdout 任一关键任务族明显退化。
- 只在训练派生任务上提升，独立 validation 无提升。

## 10. 暂不采用的方案及原因

- **不直接照搬 TRACE/IGPO**：它们依赖紧凑、可验证的标准答案；旅行计划多解。后续可把状态势函数定义为约束满足和任务进展，但必须重新证明不会奖励走捷径。
- **不立即训练 HCAPO/iStar critic**：增加模型、训练阶段与主观误差源；当前环境已有更可靠的程序事实。
- **不立即从 GRPO 全面迁移 PPO**：Turn-PPO 是重要备选，但现在先完成同一训练栈内的 B0/R1 公平对照，避免优化器变化掩盖信用分配效果。
- **不把自由文本反思作为核心资产**：TravelAgent 使用结构化失败和进度状态，避免暴露或依赖私有思维链。
- **不把最新预印本数字写进项目成果**：TCPO、ADRS、ABSeeker 等只生成实验假设，证据要由 TravelAgent 自己的配对评测产生。

## 11. 对下一步开发的直接决定

1. 修正 return-to-go：折扣距离只按模型决策轮计算。
2. 完成真实历史兼容的自适应恢复 SFT 数据构造和防泄漏测试。
3. 从当前正式 SFT checkpoint 做小步恢复 SFT，并同时评估新旧能力。
4. 只有恢复任务进入可学习区间后，才运行 B0 与 validity-gated R1-v2。
5. 将 `invalid_action_positive_credit_rate`、`zero_advantage_group_ratio` 和 `effective_nonzero_credited_turns` 纳入训练报告和候选门禁。
6. 通过 R1 后再评估 SALT/TSR，不影响当前已离线合格的 B0 候选模型。

## 12. 关键原始来源

- [TRACE](https://arxiv.org/abs/2607.13988)
- [ToolVerse](https://arxiv.org/abs/2607.15660)
- [A Practitioner's Guide to Multi-turn Agentic RL](https://arxiv.org/abs/2510.01132)
- [Turn-Level Credit Assignment](https://arxiv.org/abs/2505.11821)
- [RAGEN](https://arxiv.org/abs/2504.20073)
- [WebAgent-R1](https://aclanthology.org/2025.emnlp-main.401/)
- [Structured Reflection for Tool Recovery](https://aclanthology.org/2026.findings-acl.618/)
- [PROGRA](https://aclanthology.org/2026.findings-acl.325/)
- [IGPO](https://arxiv.org/abs/2510.14967)
- [TSR](https://arxiv.org/abs/2602.11767)
- [SALT](https://aclanthology.org/2026.findings-eacl.247/)
- [Escaping the Echo Trap](https://aclanthology.org/2026.acl-long.1636/)

其余论文、证据等级和本地文件映射见 E 盘论文库 `README.md`。
