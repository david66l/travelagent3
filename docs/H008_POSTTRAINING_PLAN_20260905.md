# H008 后训练执行记录（2026-09-05）

> 后续工作以 [最新后训练方案 v3](POSTTRAINING_PLAN.md) 为准，目标是提升小模型自主驱动 Agent Loop 的决策能力。本文保留 H008 历史执行记录；其中不同运行时的成绩不能直接混算，最新同栈对照与职责边界见新方案。

## 目标

让 Agent 在真实多轮 ReAct 历史中稳定完成两类薄弱决策：

1. 根据用户意图选择正确的实时信息或交通工具，并提交正确参数；
2. 根据工具错误决定“修改参数重试”还是“原参数重试”，避免重复无效动作。

所有 H008 数据、训练、评测和日志均在云端服务器执行。测试集保持封存，不进入训练器。

## H007 基线与评分修正

- H007 r4 Core：5/10（50%）。
- H007 r4 Expanded：17/20（85%）；按修正后的外部失败安全终止合同离线复核约为 18/20（90%），仍需正式重跑确认。
- H007 r4 Recovery：17/32（53.12%）。
- 外部 provider 失败时，评分允许“有授权替代方案则 propose_tradeoff；没有授权替代方案则 grounded abort”。
- search_current_info 和 search_transport 仍必须满足意图参数合同，例如 opening_hours、restaurant、event、train、flight。

## H008 训练前诊断

### 外部信息链

- 28 个状态，每状态 4 次，共 112 次。
- 成功 0/112。
- 28/28 状态进入 sft_repair。
- policy output error 100/112，invalid action 30/112。
- 结论：属于冷启动全失败区，不能直接做 GRPO，必须先用 SFT/RFT 把正确动作放进模型概率支持区。

### 错误恢复链

- 16 个状态，每状态 4 次，共 64 次。
- 成功 1/64（1.56%）。
- 15 组进入 sft_repair，1 组进入 grpo_update。
- 修改参数恢复 0/32；原参数重试 1/32。
- policy output error 49/64，policy argument error 31/64，invalid action 29/64。
- 结论：绝大多数状态先做 SFT；仅保留真实混合成功组，待 warm-start 后重新挖掘 GRPO frontier。

## 数据

### 在线 RL 源环境

- 外部信息：train 560、validation 112、test 112，七个 family 各自平衡。
- 恢复：train 512、validation 64、test 128，场景、证据风格和目标位置正交覆盖。
- train/validation/test 的 task id、环境指纹和请求模板互不重叠。

### H008 warm-start SFT

- 数据版本：`h008-multiturn-warmstart-273404940cb0fcf7`。
- train 688、validation 128、test 0。
- 训练集组成：外部信息多轮纠偏 280、恢复多轮纠偏 64、H007 旧能力锚点 344。
- 验证集组成：外部信息多轮纠偏 56、恢复多轮纠偏 8、H007 旧能力锚点 64。
- 纠偏样本和旧能力锚点严格 1:1。
- 外部信息标签只要求目标动作、参数和真实工具 observation 通过；不把后续旧 solver 快照的成败混入动作标签。
- 恢复标签要求首轮真实失败、可见错误和第二轮恢复 observation 全部可验证。
- 冻结 test 未读取。

## 训练前门禁

- 816/816 模型可见载荷唯一。
- 最大序列 5626 token，p95 为 5327，6144 上限下 0 截断。
- 816/816 tool-call 终止边界正确。
- 依赖版本兼容，Git 源码快照为 `07aba12b910fa07c9a4f112f3e026b1db67b362a`。
- 修复 action executor 混用机器时钟和冻结评测时钟的问题；相关测试 28/28 通过。
- H008 相关与时间/执行器测试 32/32 通过。

## 第一轮 warm-start SFT

- 输入：H007 adapter。
- 输出：`h008-multiturn-warmstart-sft-v1-seed20260905`。
- 最佳保存点：`checkpoint-56`，eval loss 0.178511。
- 最终 train loss 0.159879，训练正常结束，无 OOM/NaN。
- checkpoint-56 adapter SHA256：`412e148752a7d2c7ab3c99fc7128734defcdaa452eb371e8b3180bcc3345a3dc`。

冻结测试集上的原生生成结果：

- 外部信息：目标动作 50.00%，目标参数 47.32%，目标 observation 24.11%。
- 恢复：目标第二动作 82.03%，目标参数 55.47%，目标 observation 54.69%。
- 两套决策边界语料均没有完整任务成功；这些语料只足以评价目标决策，不能冒充端到端成功率。

## 第二轮不重叠 SFT

- 数据版本：`h008-multiturn-warmstart-254f7c57d898f701`。
- train 816、validation 144、test 0；与第一轮 scenario id 零重叠。
- 训练集组成：外部信息 280、恢复 128、旧能力锚点 408。
- 输入：第一轮 `checkpoint-56`。
- 输出：`h008-multiturn-second-pass-sft-v2-seed20260905`。
- 最佳保存点：`checkpoint-64`，eval loss 0.168691。
- 最终 train loss 0.131134，训练正常结束，无 OOM/NaN。
- checkpoint-64 adapter SHA256：`63e7c71b692f475716c3f7a644652bd37183283a3314f6aa48599a25cd1ae8e8`。

冻结测试集上的原生生成结果：

- 外部信息：目标动作 55.36%，目标参数 51.79%，目标 observation 24.11%。
- 恢复：目标第二动作 82.81%，目标参数 55.47%，目标 observation 54.69%。
- 结论：验证损失下降，但真实目标执行基本没有提升，不能凭 loss 晋级，也不能直接进入 GRPO。

## 架构诊断与修正

错误轨迹显示两类系统问题：

1. 模型即使只剩一个合法工具，也会生成动态白名单之外的动作；
2. 用户已经明确给出的查询、日期和单一交通方式仍由模型重复生成，导致改写、错日期和 `both` 等错误。

已修正：

- 本地 checkpoint 默认使用状态级 `qwen_tool_envelope` 约束解码；仍可通过配置切回 native。
- `search_current_info` 的明确查询原文和开始日期由控制器强制注入。
- `search_transport` 的出发地、目的地、日期、返程日期，以及用户只指定一种方式时的 mode 由控制器强制注入。
- 模型继续负责信息类型选择和恢复关键词，不把真正需要判断的业务决策写死。
- 直接相关执行器测试 26/26 通过，Ruff 和 diff check 通过。
- 扩大回归中另有 5 个既有 verifier-repair replay 测试失败；其轨迹不经过本次修改的 current-info/transport 分支，单独登记为基线问题。

## 系统级冻结测试结果

模型、冻结测试集、seed 和采样次数不变，只启用状态级约束解码与可信参数注入：

### 外部信息（28 题 x 4，共 112 条）

- 目标动作：55.36% -> 98.21%。
- 目标参数核心匹配：51.79% -> 91.07%。
- 目标 observation：24.11% -> 91.07%。
- capability reward gate：7.14% -> 67.86%；该指标不是完整任务成功率。
- 原生非法动作率 38.39%，系统级非法动作率 2.68%。

### 错误恢复（32 题 x 4，共 128 条）

- 目标第二动作：82.81% -> 93.75%。
- 目标参数：55.47% -> 62.50%。
- 目标 observation：54.69% -> 61.72%。
- 原参数重试 observation：65.63% -> 79.69%。
- 修改参数 observation：仍为 43.75%。
- capability reward gate：3.13% -> 43.75%；该指标不是完整任务成功率。

## 下一步固定顺序

1. 提交本次约束解码默认值和可信参数注入，保留所有审计日志与报告。
2. 从未使用的 train/validation 状态构建第三轮恢复专用 SFT：重点覆盖诊断证据、关键词删减和原参数复用；继续排除前两轮 scenario id，test 不进入训练。
3. 第三轮只在独立 validation 提升后，才用冻结 recovery test 复测；目标是修改参数 observation 明显高于 43.75%，且原样重试不低于当前 79.69%。
4. GRPO 只使用训练集侧、有真实好坏差异的决策边界，并以目标决策即时奖励结束；不使用冻结 test，也不让无关的后续旧快照污染奖励。
5. 最后用完整 Core、Expanded、Recovery Agent Loop 回归检查端到端成功率、重复调用、约束遵守和旧能力退化；达到晋级门槛后才解除 quarantine。

## 状态

H008 两轮 SFT 和系统级决策边界评测已完成。普通外部信息目标 observation 达到 91.07%；下一阶段聚焦错误恢复中的参数修改能力，并补完整 Agent Loop 晋级评测。
