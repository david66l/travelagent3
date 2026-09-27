# TravelAgent2 后训练路线 v2

> 2026-09-05 更新：后续执行路线已由 [最新后训练方案 v3](../docs/POSTTRAINING_PLAN.md) 接替，以小模型在 Agent Loop 中的自主决策收益为核心。本文件保留历史假设、协议和裁决；下方“当前状态”描述的是当时状态，不能用于判断最新 H008/H009 进度。

> 状态：Step 1/2/3 已完成；H-005 组件 Conditional Go；H-004 No-Go；H-006 待 Step 4 数据审计  
> 决策日期：2026-09-03  
> 适用范围：native ReAct Agent、verifier repair、工具调用与后训练晋升  
> 当前裁决：所有新 checkpoint 保持 quarantine；sealed 160 保持未解封

## 1. 决策摘要

当前最优路线不是继续修补 H-003/H-004，而是按以下顺序推进：

1. 冻结评测合同、promotion 访问规则和新的 `promotion-val-v1`；
2. H-005：把 canonical connector 从模型生成职责移到系统确定性组装；
3. H-006：从 H-001 出发，用已验证成功的 hard 轨迹做 rejection-sampled SFT，并混入 easy anchors；
4. 如果统一 checkpoint 仍不能兼顾简单题与难题，再做 H-007 安全路由；
5. 只有剩余失败确实来自多轮工具决策时，才做 H-008 真实 ReAct Agent RL；
6. 候选完全冻结并通过 promotion-val、三训练 seed、盲审和安全门后，才一次性打开 sealed 160。

禁止事项：

- 不再对 H-003/H-004 追加 DPO 数据碰碰运气；
- 不直接照搬 TravelRLAgent 的固定 13 轮、全参数训练和 60%--75% LLM judge 权重；
- 不用 internal-dev 或 sealed test 反复挑 checkpoint、调 router 阈值或调 prompt；
- 不把系统注入的固定连接词计作模型原生能力提升；
- 不以训练 reward、DPO preference accuracy 或单次采样结果替代系统成功率。

当前授权边界：

- H-005 只有在产品合同、同栈基线和双指标记账实现后才允许执行；
- H-006 在 donor 数据隔离、完整训练配置和功效分析完成前禁止启动 GPU；
- H-007/H-008 需要前一阶段的正式 No-Go 报告才能解锁。

## 2. 当前证据基线

统一使用四采样硬协议作为当前可比基线：

| checkpoint | internal-dev full success | hard challenge vs H-001 | 当前角色 |
|---|---:|---:|---|
| H-001 h48 SFT | 0.869 | baseline | 最强 generalist 实验基线 |
| H-003 r2 GRPO | 0.803 | +0.170 | hard 学习有效，但 easy 精度回退 |
| H-004 r2+DPO | 0.828 | +0.176 | 部分修复；仍未过 0.85 门 |

补充事实：

- H-001 单采样 0.887 不能替代四采样结果；后续至少四采样。
- H-003 的主要价值是证明 frontier selection 和非零方差 GRPO 可以学习；它不是可晋升模型。
- H-004 preference accuracy 100%、margin 5.15 只证明偏好目标被拟合，不证明系统质量达标。
- H-004 相对 H-003 修复约 2.5pp，但仍比 H-001 低约 4.1pp。
- 连接词是动作的确定性函数，继续消耗模型容量学习固定表面形式的收益很低。

## 3. 从 TravelRLAgent 继承的原则

### 3.1 保留

- 训练 rollout 必须执行真实工具，而不是只生成伪造的 tool trace；
- train/serve 共用 tool schema、parser、终止条件和 observation 语义；
- reward 分组件记录：contract、schema、stage、terminal state、efficiency、language quality；
- 使用 curriculum，从合同正确到单步恢复，再到完整多轮；
- LLM A/B 评审采用位置交换，降低位置偏差；
- 全量记录轨迹，能够从任一 reward 追溯到状态、动作、工具结果和 verifier 变化。

### 3.2 必须改造

- 当前任务有确定性 controller/verifier，因此确定性信号必须占 80%--90%；LLM judge 只能补充主观语言质量，权重不得超过 10%。
- 最大轮数按生产成功轨迹的 P95/P99 确定，不固定照搬 13 轮。
- 数据隔离按 `source_state_id + generator_family + template_lineage` 完成，不能只比较 query 字符串。
- 优先 LoRA/QLoRA 验证方法；只有 adapter capacity 被证据证明不足时才考虑 full tuning。
- 报告 paired confidence interval、按任务族结果和 `pass^k`，不能只报告平均 judge 分数。

## 4. 系统与训练责任边界

### 4.1 模型负责

- 在允许动作集合中选择 `action`；
- 引用真实 `evidence_refs`；
- 生成与证据一致的 `reason_detail`；
- 决定何时调用工具、重试、权衡、询问用户或终止。

### 4.2 系统负责

- JSON/schema 合法性与 action enum；
- controller-owned 字段和允许动作集合；
- `action -> canonical_connector` 的确定性映射；
- evidence id 存在性和可访问性；
- CP-SAT/verifier 的确定性结果；
- 轮数、调用预算、重复调用、越权和安全边界；
- 最终显示文本的确定性组装。

建议的语义输出合同：

```json
{
  "action": "retry | tradeoff | abort | ask_user | continue | answer",
  "evidence_refs": ["evidence-id"],
  "reason_detail": "模型生成的、有证据依据的解释"
}
```

系统组装：

```text
final_reason = canonical_connector[action] + reason_detail
```

只对 JSON、字段类型、action enum 做严格结构约束；不使用 grammar 强制整段自然语言。

### 4.3 双指标记账

每次评测必须同时记录：

- `raw_model_contract_success`：系统组装前，模型原始输出是否满足模型应承担的合同；
- `assembled_system_success`：经过确定性映射和组装后的端到端成功率。

系统注入连接词可以提升 assembled 指标，但不得反向写成模型 rationale 能力提升。

connector 映射必须来自产品动作合同，禁止根据 lexical evaluator 的接受词反推。除格式外，
还要检查 connector 与 `reason_detail` 是否语义冲突；冲突时整条结果失败，不能依靠固定前缀
掩盖错误解释。

## 5. 数据协议

### 5.1 Split

建立并冻结：

| split | 用途 | 是否允许调参 |
|---|---|---|
| train | SFT/RL 更新 | 是 |
| train-hard-donor | 由冻结 policy 生成新的 hard 成功轨迹 | 是，只能流向 train |
| train-shadow | 在线生成、frontier mining | 是 |
| internal-dev | 日常调试和失败尸检 | 是，但不再承担晋升证明 |
| router-train | Router 拟合 | 是 |
| router-calibration | Router 阈值与 abstain 校准 | 是 |
| promotion-val-v1 | 唯一晋升集 | 每个实验族只允许一次冻结 campaign |
| sealed-160 | 最终一次性验收 | 否 |

`promotion-val-v1` 暂定 200 个独立 source states，每个状态固定 4 次 rollout，覆盖：

- easy tradeoff；
- retry/abort；
- hard frontier；
- 正常旅行规划与工具链；
- 合同、安全、越权和 OOD。

200 只是初始估计。冻结前必须用 internal-dev 的 paired discordance 做功效分析，确保总体、
hard、easy 和 `pass^4` 的关键门有足够检验力；不足则在首次访问前扩容。promotion-val 的
样本内容、标签和分族结果对训练执行者保持盲态。

### 5.2.1 禁止回流清单

下列数据及其模板、状态或工具快照近亲永久禁止进入 H-006/H-008：

- 已经用于报告 H-003/H-004 `+0.170/+0.176` 的 hard challenge；
- internal-dev 及其改写、槽位替换和失败轨迹派生数据；
- promotion-val-v1；
- sealed-160；
- 从以上任一集合的 gold action、verifier 结果或人工错误分析派生的数据。

H-004 在 H-006 中只能作为冻结的 donor policy；不得直接复用它在旧 hard challenge 或
internal-dev 上已经生成的成功文本。所有 hard target 必须来自新建的 train-only
`train-hard-donor`。

### 5.2.2 Promotion 访问预算

- pilot、checkpoint 选择、早停和失败尸检只允许查看 train-shadow/internal-dev；
- promotion campaign 之前必须冻结代码 commit、数据 hash、checkpoint、训练 seed、
  inference seed、采样参数、最大步数和全部阈值；
- 每个实验族只允许一次 promotion campaign，并写 append-only 访问日志；
- H-006 campaign 同栈评测固定的 `H-001+H-005 / H-004+H-005 / H-006+H-005`；
- promotion 失败后不得依据其中的分族错误修改同一实验族后再看一次。

### 5.2 血缘隔离

manifest 至少保存：

```text
source_state_id
source_hash
template_family
generator_model
generator_prompt_hash
tool_snapshot_hash
verifier_version
split
parent_trajectory_id
```

以下任一相同均视为同一血缘族，必须放在同一 split：

- 同一个原始约束状态；
- 同模板的槽位替换或同义改写；
- 从同一失败轨迹派生；
- 同一工具快照只改表面文本。

### 5.3 H-006 数据配比

目标先生成 300--500 个独立状态，每个状态 4--6 条 on-policy 轨迹。只保留 deterministic verifier 全通过的轨迹。

训练混合：

| 类型 | 比例 | 来源 |
|---|---:|---|
| hard success | 40% | H-004 在 train-hard-donor 全新 hard states 上的成功轨迹 |
| easy anchors | 40% | H-001 的 easy/tradeoff/retry/abort 成功轨迹 |
| recovery/contract correction | 20% | 输入包含失败尝试，target 为经 verifier 验证的修复输出 |

约束：

- 主要 hard/easy/recovery 族各至少 50 个独立 source states；更细失败族 20 个只够 smoke，
  不得据此声明分族提升；
- 每个 source 最多保留 1--2 条行为真正不同的成功轨迹，训练 sampler 按 source 等权；
- 40/40/20 同时按独立 source 数和 supervised completion token 数控制，容差均为 +/-5pp；
- 不直接将 rejected response 当成 SFT target；
- 不保留只因系统注入 connector 才成功、但 action/evidence 错误的样本；
- 对成功轨迹做 action、证据、长度和来源分布检查，防止只学表面 cue；
- 保留 H-001 easy anchors，防止 hard specialization 覆盖通才行为。
- correction 输入只能包含该决策时刻真实可见的信息，禁止包含未来 observation、终局
  verifier 结果、gold action、reward 和隐藏 controller 字段；
- 所有 target 必须 100% 通过 allowed-action、evidence 可见性、终局 verifier、上下文
  一致性和 system-assembled-only 假成功审计。

## 6. H-005：确定性合同层

### 假设

如果 rationale 失败主要来自 action 对应的固定连接词，而动作与 grounding 已基本正确，则把该固定映射交给系统组装可以消除合同型失败，并且不改变模型语义决策。

### 实施顺序

1. 冻结 action 到 connector 的单一映射表；
2. 在生产 render path 中组装，不在 evaluator 中单独打补丁；
3. serve 与 eval 调用同一 render 函数；
4. 增加 action 授权、evidence grounding、语言一致性、verifier 文案泄露检查；
5. 对 H-001/H-003/H-004 的已有 320 份四采样输出离线重放；
6. 验证组装前后 action、evidence、tool trace 完全不变。

历史重放只用于验证系统变化，不用于训练。必须在同一 parser、render、inference seed 和
tool snapshot 下重算 `H-001+H-005 / H-004+H-005`；禁止拿 assembled candidate 与历史
raw baseline 比较。

### Go 条件

- schema/contract 100%；
- 越权 0；
- action/evidence 指标不得因组装变化；
- raw 与 assembled 指标均完整出现在报告中；
- 所有调用点使用同一实现，无 evaluator-only shortcut。

任一条件失败则不进入 H-006。

## 7. H-006：rejection-sampled hard SFT

### 假设

H-004 已经发现了有效 hard 策略；将 verifier 验证过的成功策略蒸馏回 H-001，并混入 easy anchors，可以保留 hard 增益且避免 GRPO 的 easy 精度回退。

### 开训前仍需冻结的完整主配置

```text
start_checkpoint = H-001 h48
tuning_mode      = 继续训练 H-001 原 adapter；不 merge、不 stack 新 adapter
tuner            = LoRA/QLoRA
lora_rank         = 16
lora_alpha        = 以 H-001 adapter_config 为准，预期 32；preflight 必须实读确认
lora_dropout      = 以 H-001 adapter_config 为准，预期 0.05；preflight 必须实读确认
target_modules    = 以 H-001 adapter_config 为准，预期 all-linear；不得静默改变
bias              = none
learning_rate     = 5e-6
epochs            = 1
micro_batch       = 1
gradient_accum    = 12
effective_batch   = 12 per data-parallel replica
optimizer         = AdamW；具体 Transformers 实现与版本须写入 manifest
warmup_ratio      = 0.05
weight_decay      = 0.0
lr_scheduler      = linear
max_grad_norm     = 1.0
max_sequence      = 6144；preflight 必须证明 0 truncation
completion_loss   = true
packing           = false
release_seed      = 20260930
robustness_seeds  = 20260931, 20260932
eval_protocol     = 4 rollouts/state
```

实际 optimizer steps 在数据 manifest 冻结后按样本数、world size 和 effective batch 精确计算，
并写入预登记；`eval_steps/save_steps` 均设为总步数约 1/4，至少 1，保存每个评测点和 final。
H-001 adapter hash、base model hash、tokenizer/chat-template hash 和 trainable parameter 清单必须
保存。当前训练脚本必须在 manifest 中显式写出 optimizer，禁止依赖未记录的库默认值。

不做参数网格。pilot 使用单独的 dev seed，且只看 train-shadow/internal-dev。dev gate 通过后，
固定上述 release seed 和两个 robustness seed；release 永远是预先指定的 20260930，不能从三个
seed 中挑最好。三个 seed 的 promotion 评测属于同一次冻结 campaign。

### Development Go 条件

只在 train-shadow/internal-dev 上执行，用于决定是否允许一次 promotion campaign。所有候选
与 H-001/H-004 均通过 H-005 同栈重算。

### Promotion Go 条件

- 同栈绝对 `full_success_mean_4 >= 0.90`；
- 相对 `H-001+H-005` 的 paired、source-state clustered bootstrap 95% CI 下界 > 0；
- hard 相对 `H-001+H-005` 点估计至少 +10pp，且 95% CI 下界 > 0；
- hard 相对 `H-004+H-005` 的非劣 CI 下界不低于 -2pp；
- easy 和 `pass^4` 相对 `H-001+H-005` 的非劣 CI 下界不低于 -2pp；
- 每一个训练 seed 均须 contract 100%、越权 0、easy 非劣不超过 2pp；
- release seed 固定为 20260930，其余 seed 只验证稳健性，不能替换 release seed。

如果统一 checkpoint 通过，优先采用统一 checkpoint，不再引入 router 复杂度。

## 8. H-007：安全 Router 兜底

仅当 H-006 未过总体门，但仍满足“hard 增益存在、easy 回退可定位”时启动。

候选：

- generalist：H-001；
- hard specialist：优先比较 H-004，H-003 只作消融；
- abstain/fallback：H-001。

Router 输入只能使用结果产生前可见的状态特征：

- violation 类型；
- 剩余 retry budget；
- 可行替代项数量；
- evidence coverage；
- 约束冲突数量；
- 是否 OOD/字段缺失。

禁止输入：gold action、最终 reward、任务是否成功、LLM judge 结果、任何 test 标签。

Router 只能使用 router-train 拟合，在 router-calibration 上确定阈值和 abstain；两者都必须
与 promotion/sealed 按 source lineage 隔离。禁止在 promotion-val 上调阈值。

### Go 条件

- 总体至少超过最佳单 checkpoint 3pp；
- hard 相对 H-004 回退不超过 2pp；
- easy 相对 H-001 回退不超过 2pp；
- easy 错送 specialist 的比例不超过 5%；
- abstain/fallback 成功率不低于 H-001；
- 三个固定评测 seed 方向一致。

任一失败则删除 router 候选，不把“两个模型拼接”包装成模型能力提升。

## 9. H-008：真实多轮 Agent RL

### 启动前提

只有失败分类显示主要剩余问题来自以下一项或多项时启动：

- 工具选择或调用时机错误；
- 工具 observation 后未修正计划；
- retry/tradeoff/abort 跨轮策略错误；
- 重复调用、过早回答或不能停止；
- 单步 action 已正确，但完整终局状态失败。

如果失败仍主要是连接词、schema 或表面文本，不启动 RL。

### 环境要求

- rollout 直接调用生产 ReAct loop；
- train/serve 共用 parser、tool registry、prompt builder、render、终止和预算逻辑；
- 每轮记录 state、allowed actions、model action、tool args、observation、verifier delta、cost、latency 和终局状态；
- 工具/检索 observation token 做 loss mask，只更新模型生成 token；
- 外部服务失败与模型失败分开记账，不把 judge/tool outage 记作模型零分。

如果宣称使用 turn-level credit assignment，learner 必须真正按 turn 切分 transition、return
和 advantage，并有集成测试证明不同 turn 得到不同、可追溯的 advantage。仅添加
`potential_delta`、但仍给整条 sequence 同一 advantage，只能称 reward shaping，不能称
turn-level credit assignment。

建议 trace schema：

```text
trajectory_id
turn_id
state_hash
prompt_hash
allowed_actions
action
tool_name
args_hash
observation_hash
verifier_before
verifier_after
terminal_status
reward_components
latency_ms
token_count
```

### 奖励合同

先经过硬门：

- schema 失败、越权、伪造工具结果、突破 controller-owned 逻辑：轨迹最大回报封顶为失败；
- 合同通过后才计算能力奖励。

能力奖励建议：

```text
R = 0.80 * terminal_state_reward
  + 0.15 * potential_delta
  + 0.05 * calibrated_language_quality
  - duplicate/over-budget/loop penalties
```

其中 `potential_delta` 采用状态势函数差：

```text
gamma * Phi(state_next) - Phi(state_current)
```

`Phi` 只包含可审计状态，例如 violation 数量减少、关键证据覆盖增加、可行解距离缩短。必须通过单元测试证明 shaping 不会奖励绕路、重复调用或改变终局最优动作。

LLM judge：

- 权重不超过 5%--10%；
- 只判断措辞、清晰度和用户沟通；
- 不覆盖 deterministic verifier；
- 用人工双盲样本校准；
- judge 不可用时该组件记 missing，不把整条轨迹记 0。

### Frontier dynamic sampling

- 预评估每个 state 的 group success；
- 只把成功率 15%--85% 的组送入 RL 更新；
- 全败组进入 teacher/RFT/SFT 队列；
- 全对组进入 replay anchor，不产生无效 GRPO step；
- pilot 至少 100 个独立 frontier states、至少 60 个非零 advantage groups；
- 每类主要失败至少 20 个独立 source states。

这里的成功率和方差只能由 deterministic terminal/verifier reward 计算；judge-only variance
不得使一个组获得 RL 资格。

### 保守 pilot 起点

```text
group_size       = 6; 显存不足时为 4
temperature      = 1.0
kl_beta           = 0.04
learning_rate     = 2e-6
max_updates       = 100 hard cap；预登记目标可为 60--100
internal_eval     = 每 10 updates，只能使用 train-shadow/internal-dev
max_turns         = min(production cap, successful trace P99 + 1)
```

这是预登记起点，不做同步网格搜索。监控：

- non-zero advantage group ratio；
- reward component 均值、方差和相关性；
- terminal success 与训练 reward 的相关性；
- KL、entropy、clip ratio、gradient norm；
- tool-call 数、重复率、loop rate、stop accuracy；
- easy/hard/contract 各族 paired 指标；
- latency 和 token cost。

### 立即停训条件

- 连续两个 internal evaluation checkpoint 没有正向 paired gain；
- easy 相对冻结基线回退超过 2pp；
- contract 低于 100%；
- 任一越权动作；
- reward 上升但 terminal success 不升：立即判定 reward hacking，不允许在同一 run 中换权重续跑；
- judge 与 deterministic verifier 分歧超过 10%；
- entropy 快速坍塌并伴随重复模式或 gradient spike；
- 训练 batch 的非零 advantage 组不足以支撑更新。

最近两个窗口的 deterministic non-zero advantage group ratio 必须高于预登记下限；低于门时
停止训练并回到 frontier mining，不用 judge 方差凑有效组。

## 10. 统一评测协议

### 10.1 主指标

- `full_success_mean_4`：每状态四次 rollout 的平均成功率；
- `pass^4`：同一状态四次全部成功的比例；
- terminal state/verifier success；
- action accuracy；
- evidence grounding；
- contract/schema；
- unauthorized action；
- tool loop/duplicate/stop accuracy；
- P50/P95 latency、token 和 tool cost。

### 10.2 统计

- 二元 paired 结果使用 McNemar；
- 连续 reward 使用 source-state clustered bootstrap；
- 统一报告 95% CI 和 effect size；
- 多任务族比较使用 Holm correction；
- 不以 `p > 0.05` 证明两模型等价；非劣效必须预登记 margin。

### 10.3 人工盲审

从 promotion-val 固定抽样，不看模型名，评估：

- 理由是否具体；
- 是否真实引用证据；
- 是否泄露内部实现；
- 是否清晰说明权衡；
- 是否存在不必要的过度思考。

LLM judge 必须先与人工标注校准；目标 agreement >= 90%，或 Cohen's kappa >= 0.70。未达标时，judge 只能作描述性指标。

## 11. Sealed-160 开启条件

只有以下条件全部满足时才允许打开：

1. 统一 checkpoint 或 router 已二选一；
2. checkpoint hash、adapter、prompt、parser、tool snapshot、router threshold 全部冻结；
3. 一次性 promotion campaign 通过总体、hard、easy、安全和成本门；
4. 三训练 seed 方向一致；
5. 人工盲审和 judge 校准通过；
6. shadow 回放无新的 P0/P1 故障；
7. 评测脚本、样本数和 rollout seed 已预登记；
8. 负责人签字确认 sealed 只打开一次。

sealed 结果不得用于继续选择 checkpoint 或调参。如果失败，同一候选永久 No-Go；禁止通过
另建 sealed 集让同一候选“复活”。后续新实验族必须有实质方法变化，并重新建立独立评测治理。

## 12. 训练前逐项检查

### 数据

- [ ] train/promotion/sealed source-state 血缘无交叉
- [ ] hard challenge/internal-dev 及其近亲均在禁止回流清单中
- [ ] manifest hash 已生成并归档
- [ ] action/任务族/难度/长度分布已报告
- [ ] system-assembled-only success 未混入模型成功数据
- [ ] 每类主要失败至少 20 个独立状态

### 环境

- [ ] train/serve parser parity 测试通过
- [ ] tool schema 与生产一致
- [ ] render、终止、预算与生产一致
- [ ] tool/judge outage 与模型失败分开处理
- [ ] 所有 controller-owned 字段不可被模型覆盖

### 训练

- [ ] 单一主配置已预登记
- [ ] adapter 继续/合并/堆叠方式和 trainable parameter 清单已冻结
- [ ] optimizer、warmup、batch、LoRA、max sequence、loss mask 和总更新步数已显式记录
- [ ] seed、checkpoint hash、代码 commit 已记录
- [ ] smoke run 不接触 promotion/sealed
- [ ] promotion look-budget 与 append-only 访问日志已启用
- [ ] zero-variance sampler 已启用
- [ ] easy anchor 防遗忘数据已加入
- [ ] stop rules 已自动化

### 评测

- [ ] 每状态至少四次 rollout
- [ ] raw model 与 assembled system 指标分开
- [ ] paired CI、McNemar、pass^4 已生成
- [ ] 分任务族和失败类型报告已生成
- [ ] 成本、延迟、循环和越权指标已生成

## 13. 训练产物和命名

每个 run 必须保存：

```text
run_manifest.json
data_manifest.json
config.yaml
git_commit.txt
checkpoint_hash.txt
train_metrics.jsonl
eval_metrics.json
failure_taxonomy.json
sampled_trajectories.jsonl
promotion_decision.md
```

建议命名：

```text
H005-contract-assembly-<date>-<seed>
H006-rft-hard-sft-from-h001-<date>-<seed>
H007-state-router-h001-h004-<date>-<seed>
H008-native-react-arlt-<date>-<seed>
```

`promotion_decision.md` 必须明确写出：Go、No-Go 或 Quarantine，以及触发的具体门槛。不能只写“效果不错”。

## 14. 文献原则映射

- Agent Lightning：训练与 Agent 执行解耦、MDP 化轨迹、层级信用分配和可观测性。  
  https://arxiv.org/abs/2508.03680
- VerlTool：真实多轮工具轨迹、统一工具接口和异步 rollout。  
  https://arxiv.org/abs/2509.01055
- Turn-Level Credit Assignment：多轮 Agent 使用 turn-level advantage，而不是粗粒度整轨迹分数。  
  https://arxiv.org/abs/2505.11821
- RAGEN：监控 Echo Trap、reward variance collapse、entropy drop 和 gradient spike。  
  https://arxiv.org/abs/2504.20073
- DAPO：dynamic sampling 丢弃零方差组，提高有效更新密度。  
  https://arxiv.org/abs/2503.14476
- DeepSeek-R1：cold-start SFT、RL、rejection sampling、再次 SFT 的分阶段路线。  
  https://www.nature.com/articles/s41586-025-09422-z
- ToolRL：工具学习需要比最终答案更细的奖励分解。  
  https://arxiv.org/abs/2504.13958
- ReTool：真实工具执行和 outcome-driven tool-use RL。  
  https://arxiv.org/abs/2504.11536
- tau-bench：以最终状态评估 Agent，并使用 pass^k 衡量重复运行可靠性。  
  https://arxiv.org/abs/2406.12045
- Hidden Cost of Structure：constrained decoding 能保证结构，但可能损害生成任务能力，因此只约束结构字段。  
  https://aclanthology.org/2025.ranlp-1.124/

## 15. 当前下一步

2026-09-04 纠错准备续记：`STEP-04-H006-CORRECTIVE-PREP.md`。规则教师候选 train
240 来源 / 420 示例，shadow 48 来源 / 84 示例；均明确标来源，不能算 H001 成功率。
source 等权 loss 已实现并通过聚焦测试，最终混合和实际 Trainer 预检尚未完成。
“允许先内部纠错训练、最终验收另设门”的流程调整仍待确认；未改旧 readiness，未开训。

进度：

- [x] Step 1：评分合同、promotion 访问预算与隔离骨架；
- [x] Step 2：H-005 生产组装、语义冲突检查与 raw/system 双指标；
- [x] Step 3：H-001/H-004 同栈重算与历史输出只读 replay；
- [ ] Step 4：新 hard donor 数据、easy anchors 与禁止回流审计；
- [ ] Step 5：H-006 训练、内部评测与晋升裁决。

Step 3 的严格同栈结论：H-001 semantic/system 为 0.94375，H-004 为 0.853125；
paired source-cluster 95% CI 为 [-0.146875, -0.040625]，H-004 明确淘汰。H-005 对合法
组装样本 100% 精确且越权为 0，但冻结 overall readiness gate 因上游错误动作覆盖不足仍记
FAIL；该失败阻断 promotion/sealed，不阻断从 H-001 开始 Step 4 的内部修复数据准备。

2026-09-04 Step 4 续记：hard donor 已保留 155 个独立训练候选来源；H001 boundary 经额外
训练质量检查保留 52 个来源，retry 暂无合格训练 target。旧 ordinary 候选存在材料不足，
现已重建为 216 个通过规则教师完整 ReAct 验证的独立 source，尚不等于 216 个 H001 成功
anchor。数据导出已隔离 teacher prefix，mix 配比通过也不能直接放行 GPU。具体证据与
阻塞项见 `STEP-04-H006-DATA.md`。Step 3 决策分数不得当作完整工具链能力证明。
可解普通题的后续小检查为 1/12 次成功，11 次因填写非模型权限参数而被拒绝，未达到
至少 2/3 个成功 source 的扩大采样门；下一步先准备基础工具合同/解释修复示范，禁止
降低 parser 权限门或将 shadow 失败输出回灌训练。

在启动任何 GPU 训练前，依次完成：

1. 新建 train-only `train-hard-donor`，完成禁止回流和 causal-visibility 审计；
2. 生成 H-006 数据，落实 source 等权、每 source 上限和 source/token 双配比；
3. 补全并冻结 H-006 optimizer、总更新步数、adapter hash 和三 seed；
4. 只在 train-shadow/internal-dev 完成 pilot 和失败尸检；
5. 通过 dev gate 后，才允许一次冻结的 promotion campaign。

## 16. 独立专家审查记录

审查方式：独立后训练专家 Agent，Sol/xhigh；未授权其修改代码或文档。  
审查裁决：**H-005 Conditional Go；H-006 No-Go；H-007/H-008 尚未授权。**

专家识别的前三项高风险：

1. 将旧 hard challenge、internal-dev 或其近亲回灌训练，造成隐性数据泄漏；
2. H-005 固定 connector 抬高 lexical 指标，但模型语义解释没有改善；
3. rejection sampling 让易成功 source 和长 hard trace 获得过高权重，再次造成 easy 遗忘。

本版已据此加入并冻结以下修改：

- train-hard-donor 和禁止回流清单；
- promotion look-budget、一次性 campaign 和访问日志；
- H-001/H-004/H-006 的 H-005 同栈比较；
- source 等权、每 source target 上限、source/token 双配比和 causal-visibility 审计；
- H-006 完整训练配置、固定 release seed 和不可挑 seed 规则；
- 绝对 0.90、相对 H-001/H-004 的 CI/非劣门；
- Router 独立 train/calibration；
- deterministic variance 与真实 per-turn advantage 的实现证明；
- sealed 失败后同一候选永久 No-Go。
