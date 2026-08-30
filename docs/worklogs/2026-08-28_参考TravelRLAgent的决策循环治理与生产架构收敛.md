# 2026-08-28 参考 TravelRLAgent 的决策循环治理与生产架构收敛

## 1. 本轮目标

对照 `D:\TravelRLAgent` 的开放式多轮工具 Agent，判断其调度器、数据和 GRPO
策略中哪些机制适合迁移到 TravelAgent2，并先完成无需重新训练的生产运行时改造。

本轮原则是保留现有 DAG、CP-SAT 和 Verifier，不把确定性求解与验证节点重新交给
语言模型；模型只负责搜索、澄清、失败恢复和约束冲突取舍等真实决策。

## 2. 对照项目结论

- 对照项目使用真正的开放工具循环，显式处理无工具输出、非法工具调用、重复调用、
  最大轮数和强制结束。
- 其 SFT 数据为 881 条，平均 5.84 次真实工具调用；RL 数据为 200 条，平均
  6.46 次调用。RL 数据中只有 4 条带明显工具失败标记，长轨迹覆盖优于失败恢复覆盖。
- 保存的 GRPO 配置使用 Qwen3-4B SFT 起点、每题 6 个 generation、最多 13 轮、
  `beta=0.04`、最多 200 step，并通过 vLLM Server 生成轨迹。
- 最终混合奖励默认有 60% 至 70% 来自 LLM 对标准答案的最终文本打分。该做法适合
  开放式旅游问答，但不能替代 TravelAgent2 的 CP-SAT 与硬约束 Verifier 奖励。

因此本项目只迁移“循环治理、输出自修复、重复无进展检测、逐轮可审计信号”，不迁移
“所有节点开放给模型”和“最终文本 Judge 主导奖励”。

## 3. 已实施的 P0 改造

### 3.1 生产默认改为 Controller First

- `AGENTIC_EXECUTION_MODE` 默认由 `policy_driven` 改为 `controller_first`。
- 显式 `policy_driven` 模式继续保留，用于全链路模型能力评测和训练研究。
- 在完整 9 步测试链路中，Controller First 由控制器执行 8 个固定节点，模型只处理
  `search_candidates`；显式 Policy Driven 仍由模型处理全部 9 个节点。

### 3.2 有界策略自修复

新增 `SelfRepairingAgentPolicy`：

- 只修复模型自身负责的输出错误，不重试 Provider 超时和控制器合同错误；
- 默认只允许 1 次修复，最大可配置为 2 次；
- 覆盖越权动作、参数 Schema 非法、本地工具调用格式错误以及重复无进展调用；
- 第二次请求会收到结构化 `policy_feedback`，要求针对具体错误修正；
- 失败调用与最终调用的 Token 会合并计入最终动作，避免低估成本。

### 3.3 重复与无进展保护

- `FailureRecord` 新增 `attempted_arguments`，不再只记录工具名称；
- 对 `QUERY_TOO_BROAD`、参数非法、快照不匹配等确定需要换参数的失败，禁止原样重放；
- 对普通外部错误允许一次同参重试，连续两次相同失败后要求改变策略；
- 不在运行时全局禁止同工具调用，避免误杀超时后的合法重试。

### 3.4 轨迹与奖励审计

- `PolicyAction` 记录 `repair_attempts` 和 `repair_error_codes`；
- Turn Reward 增加 `POLICY_SELF_REPAIRED` 信号；
- Episode 审计指标增加修复总次数和发生修复的决策步数；
- 本轮只记录信号，不直接修改奖励值，后续需通过消融实验决定惩罚或正向信用。

## 4. 验证结果

- Ruff 检查：通过。
- 本轮核心 Policy、Loop、Reward、Integration：64/64 通过。
- 全部 `backend/tests/unit/agentic`：347/347 通过（Windows 下设置
  `PYTHONUTF8=1`，避免旧测试按 GBK 读取 UTF-8 中文数据）。
- `git diff --check`：无空白错误；仅有仓库既有的 Windows CRLF 提示。

## 5. 下一阶段

1. 把搜索失败、非法参数、重复调用、Verifier 不通过等状态导出成新的决策节点数据。
2. 将课程从“调整奖励权重”升级为“格式 → 单工具恢复 → 跨工具恢复 → 验证失败重规划”。
3. 在 R2 奖励中加入逐轮修复、信息增益和无进展信用，但保持任务完成与硬约束至少
   80% 的主导权重。
4. 先运行 SFT 基线和规则策略消融，确认新增决策空间确实产生模型可学习差异，再租卡
   进行下一轮 GRPO。

## 6. 第二阶段已完成：语义对照恢复课程

继续审计现有 Turn Credit 后确认，其信用边界已经合理：模型非法动作固定为 `-1`，
合法调用遇到外部失败为 `0`，只有合法后续动作可以继承 Verifier 终局奖励。因此本轮
没有为了包装概念直接新建奖励公式，而是优先修复数据中缺少“同错误不同策略”的问题。

新增 `build_stage3_decision_loop_curriculum.py`，在相同搜索请求上构造两类平衡样本：

- `QUERY_TOO_BROAD`：原参数语义有问题，后续必须收窄关键词；
- `UPSTREAM_TIMEOUT`：参数本身有效，后续必须保持相同参数安全重试。

正式数据集生成在 `ml/agentic/datasets/stage3-decision-loop-curriculum-v1`：

| Split | 总数 | 改参数 | 原参数重试 |
|---|---:|---:|---:|
| train | 512 | 256 | 256 |
| validation | 64 | 32 | 32 |
| test | 128 | 64 | 64 |

训练与盲测错误文案无重合，task ID 与环境指纹跨 Split 无重合。GRPO preflight 为
`ready=true`，512 条训练任务、64 条验证任务，错误和警告均为 0。两个场景均通过
真实 TRL Snapshot 环境执行测试：第一轮返回预期错误，按错误语义执行对应恢复后第二轮
成功。

该课程的工程价值是让模型学习 `error semantics -> recovery action`，而不是把“失败后
永远换参数”或“失败后永远原样重试”当作捷径。

## 7. 语义难度升级与云端基线

首次在云端 RTX 4090 上对 v1 做 2 题 × 4 rollout 冒烟时，SFT v7 与 GRPO v11
均为 8/8。复核样本后确认，v1 的失败文案直接写出了“删除哪个关键词”或“保持参数
不变”，因此它适合作为协议回归，却不足以提供新的 RL 方差。

随后将正式课程升级为 `stage3-decision-loop-curriculum.v2`，新增两档证据：

- `explicit_instruction`：显式给出恢复办法，用于验证工具协议和多轮执行稳定性；
- `diagnostic_evidence`：只描述参数校验、召回噪声、关键词区分度或上游执行阶段，要求
  模型自行判断应该改参还是重试。

v2 仍为 train 512、validation 64、test 128；每个 split 同时在“改参数/同参重试”和
“显式指令/诊断证据”两个轴上严格平衡，训练与盲测文案、task ID 和环境指纹均无
重合，GRPO preflight 为 `ready=true`。

云端扩大评测前的 4 题 × 4 rollout 冒烟结果：

- SFT v7：14/16；
- GRPO v11：14/16；
- 两个 checkpoint 在相同 task、sample 和 seed 下逐条结果一致；
- 唯一有方差的组为 `change_arguments/diagnostic_evidence`，成功率 2/4；
- 显式改参、显式重试、诊断式重试三组均为 4/4。

这说明旧 GRPO 的恢复收益没有自然外推到新的诊断式表达；新的稳定瓶颈是“根据检索
诊断选择应保留的条件”，不是同参重试、工具格式或循环本身。已启动 32 题 × 4 rollout
扩大审计，正式训练只接收其中具有组内方差且通过环境合同的任务。

扩大审计完成后的配对结果为：SFT v7 `111/128（86.72%）`，GRPO v11
`115/128（89.84%）`，绝对增加 3.13 个百分点。配对分布为 GRPO 单独成功 5、SFT
单独成功 1、共同成功 110、共同失败 12；McNemar `p=0.21875`，按任务聚类 bootstrap
区间下界为 0，因此候选门禁拒绝，不能将该结果表述为新场景上的显著提升。

分层结果进一步确认主要瓶颈：

| 场景 | SFT v7 | GRPO v11 |
|---|---:|---:|
| 诊断式改参 | 18/32 | 20/32 |
| 显式改参 | 31/32 | 32/32 |
| 诊断式同参重试 | 31/32 | 31/32 |
| 显式同参重试 | 31/32 | 32/32 |

为避免在开发集上继续自适应后仍把它当盲测，另行冻结
`stage3-decision-loop-final-holdout-v1`：128 条、四个语义分层各 32 条，使用全新的
失败表达，和 v2 的 task ID、环境指纹、失败文案均零重合。该集合在训练与 checkpoint
选择期间不运行，只在候选确定后一次性验收。

## 8. Turn Credit 修复与 R1v2 训练

首次 R1 冒烟暴露出两个实现问题：TRL 会在工具循环后生成一段最终 assistant 文本，
旧实现把这段非工具输出误算成非法策略动作；同时，非法动作和外部失败动作在组内归一化
前被排除，导致合法动作所在的 turn bucket 经常只剩零方差，局部信用无法进入优化。

修复后：

- 无法和工具动作对齐的最终文本保留 B0 episode 信用，并单独计为
  `unmatched_model_turns`；
- 非法动作固定信用 `-1`、外部失败固定信用 `0`、合法成功动作保留正信用，三者共同参与
  同轮组内归一化；
- 非法动作仍禁止获得正信用，避免为了制造方差破坏安全边界。

修复后的 8 轨迹冒烟中，4 条获得有效局部信用、4 条非法动作、8 个外部失败 turn，
`invalid_action_positive_credit_count=0`，Turn Credit 门禁通过。

正式 R1v2 从旧 GRPO v11 的 checkpoint-5 继续训练，使用
`controller_first`、8 candidates/group、temperature 1.2、学习率 2e-6、beta 0.04、
10 optimizer steps。训练耗时 371.18 秒，共生成 80 条轨迹、296 个模型 turn；41 个 turn
获得非零局部信用，10 个被比较的 turn bucket 均非零方差，训练报告门禁无错误。

## 9. Checkpoint 选择与冻结集结论

在 v2 validation 的 16 题 × 4 rollout 上：

| 模型 | 成功数 | 成功率 |
|---|---:|---:|
| 旧 GRPO v11 | 54/64 | 84.38% |
| R1v2 step-5 | 56/64 | 87.50% |
| R1v2 step-10 | 56/64 | 87.50% |

step-5 与 step-10 同分，因此按“同效果选更少更新”的规则选择 step-5 进入一次性冻结集。
该 validation 增益只有 2/64（+3.13pp，McNemar p=0.5），只用于 checkpoint 选择，
不作为最终 RL 增益结论。

冻结集使用 32 个任务、每题 4 个固定 seed，共 128 条逐条配对轨迹：

| 指标 | 旧 GRPO v11 | R1v2 step-5 | 变化 |
|---|---:|---:|---:|
| 轨迹成功数 | 101/128 | 101/128 | 0 |
| 轨迹成功率 | 78.91% | 78.91% | 0pp |
| 诊断式改参 | 6/32 | 6/32 | 0pp |
| 显式改参 | 31/32 | 31/32 | 0pp |
| 诊断式同参重试 | 32/32 | 32/32 | 0pp |
| 显式同参重试 | 32/32 | 32/32 | 0pp |

配对分布为 candidate-only 0、baseline-only 0、共同成功 101、共同失败 27；McNemar
`p=1.0`，任务聚类 bootstrap 95% 区间为 `[0, 0]`。候选还新增 1 次非法参数/空动作，
因此晋级门禁以 `INSUFFICIENT_ABSOLUTE_GAIN`、`CANDIDATE_SUCCESS_BELOW_TARGET`、
`PAIRED_SIGNIFICANCE_NOT_REACHED` 和
`CLUSTER_BOOTSTRAP_INTERVAL_CROSSES_ZERO` 明确拒绝。本轮不再运行生产回归门禁，也不
替换旧 GRPO v11。

## 10. 根因与下一轮路线

共同失败 27 条中，26 条是输出格式合法但恢复关键词不符合不可变快照合同，1 条是参数
Schema 错误。主要瓶颈因此不是 JSON 格式，也不是 Agent Loop 没运行，而是模型无法把
全新的诊断措辞稳定映射到“保留哪个兴趣词”。

R1v2 的路由依据来自旧模型审计中 3 个有组内成功/失败方差的任务组，并据此把整个
`change_arguments/diagnostic_evidence` 分层扩展为 128 个唯一状态；它不是简单复制 3 个
task。但进一步审计发现，这 128 个状态全部来自南京、只有 1 种诊断模板，而且 10 步
训练实际只采到 10 个 task、3 组兴趣词组合。validation 又沿用同一诊断模板，因此开发集
的轻微上涨不能证明新措辞泛化。后续按以下顺序继续：

1. 从完整训练分区构造多城市、多兴趣组合、多诊断措辞的 verified repair SFT，先把
   “诊断证据 -> 正确改参”放进模型支持集；
2. 以该 SFT checkpoint 作为新基线，只把在线采样后有组内方差的状态送入 GRPO，保证
   简历中的增益比较是 `SFT` 对 `SFT+GRPO`；
3. 重新生成从未打开过的 final holdout v2，禁止再用本轮 v1 做 checkpoint 选择；
4. 只有新候选同时通过配对显著性、最差分层和原 Stage2 生产回归，才允许晋级。

失败分析器同步修正了建议路由：参数 Schema 错误进入状态级约束解码；快照正确时的语义
参数错误进入 verified repair SFT；不再把所有模型错误都错误归因成“加 JSON 约束”。

## 11. 双 Agent 复核后的新增 P0 问题

主 Agent 与独立审计 Agent 分别复核数据和训练日志后得到一致结论。validation 的两条
candidate-only success 实际来自同一个任务的两个随机采样；该任务与训练集同为南京、
同一诊断模板和同一兴趣分布，因此 `54/64 -> 56/64` 是局部边界变化，不是跨表达泛化。

旧 v2 生成器还存在因子取模混叠：

- `change/diagnostic` 永远落在南京；
- 128 个 routed train 状态只有 1 个诊断模板；
- 正确保留词永远位于两个初始关键词的第 0 位；
- 10 步训练只实际采到 10 个 task 和 3 组兴趣组合，其中“地标/历史”占 6 个。

Turn Credit 也存在作用域污染：80 条轨迹有 156 个环境 policy record，但 `tool_mask`
切出 296 个模型生成片段，其中 140 个无法和环境动作对齐。旧实现让 unmatched 片段继承
trajectory B0，成功轨迹的最终文本或终止后多余调用可能因此获得正信用。

## 12. Turn Credit v3 安全边界

本轮将 unmatched 模型片段改为零信用，因为生产中的 Controller First 只把工具选择交给
模型，工具完成后的最终文本不属于部署策略动作。新增对齐门禁：

- 正常轨迹最多只允许比环境 policy record 多 1 个最终 assistant 片段；
- 模型片段少于 policy record，或额外 unmatched 超过 1 个时，整条轨迹的局部投影清零；
- 报告新增 `alignment_rejected_trajectories` 和 `extra_unmatched_model_turns`；
- 任一指标大于 0 时，训练报告以 `TURN_TO_TOKEN_ALIGNMENT_NOT_PROVEN` 或
  `EXTRA_UNMATCHED_MODEL_TURNS` 拒绝，不得进入候选验收。

后续 decision-loop GRPO 将把 `max_tool_calling_iterations` 固定为 2：该任务只有“首次搜索
和失败后恢复”两次策略动作，第三次 TRL 工具执行只会发生在环境已终止之后。

## 13. 正交 V3 课程与 Verified SFT

新增 `stage3-decision-loop-curriculum.v3`，用分层内部轮换替代全局 ordinal 取模。数据仍为
train 512、validation 64、test 128，但每个分层现在同时满足：

- 4 个城市均衡覆盖；
- 改参目标位于初始关键词第 0/1 位各 50%；
- diagnostic-change 训练使用 8 个模板，validation 使用 4 个未见模板，test 使用 6 个
  进一步未见模板；
- train、validation、test 模板 ID 与环境指纹均无重合。

在此基础上生成 `stage3-decision-loop-sft-v3`：每个恢复标签都经过不可变快照真实执行与
终局 Verifier 验证，再按 1:1 混入 Stage2 原能力 replay。最终规模为 train 1024、
validation 128、test 256；1,408 条模型可见输入全部唯一，最大序列 1,493 tokens，工具
信封、终止 token 和长度门禁均通过。新的 SFT 正从纯 SFT v7 分叉训练；后续 GRPO 必须
从该 checkpoint 分叉，以保证最终比较严格为 `新 SFT` 对 `同一新 SFT + GRPO`。

## 14. SFT 训练状态与下一轮评测治理

V3 SFT 已在云端 RTX 4090 完成 64/64 optimizer steps，耗时约 457 秒。终端可见指标为
`train_loss=0.2783`、`eval_loss=0.03857`；最终评测执行结束后 SSH 连接中断，需在实例
恢复后核验 `training_report.json` 和 LoRA adapter 是否完整落盘。上述 loss 只证明训练
过程收敛，不等同于 Agent Loop 成功率，未完成真实 rollout 对照前不得表述为效果提升。

评测脚本新增 decision-loop 分解指标，不再只报告一个容易掩盖偏置的总成功率。每次审计
同时按以下维度报告成功率、平均策略动作数和策略输出错误率：

- 恢复策略：改参数 / 同参数重试；
- 证据类型：显式指令 / 诊断证据；
- 正确目标位置：第 0 位 / 第 1 位；
- 城市：北京 / 成都 / 广州 / 南京。

GRPO 路由也新增状态级 v2：只保留审计中 `0 < success_rate < 1`、有非零方差且明确
`eligible_for_update=true` 的原始环境状态，不再因为一个样本有方差就把整个语义分层
扩成训练集。默认至少需要 64 个不同的可学习状态，否则拒绝启动正式 GRPO。validation
继续使用完整 V3 验证集，以免只在被筛中的局部难例上自我证明。

本地累计复核结果为 27 个相关单测全部通过，新增及修改脚本 Ruff 检查通过。当前外部
阻塞为云实例 SSH 映射端口返回 `Connection refused`；实例恢复后首先核验 SFT 产物，
随后在相同任务、相同 seed 下跑纯 SFT v7 与 V3 SFT 的多轮 Agent Loop 配对评测。

## 15. V3 SFT 产物核验与平衡冒烟

云实例恢复后确认训练产物完整落盘：主目录和 `checkpoint-64` 均包含约 66 MB 的 LoRA
adapter，`training_report.json` 状态为 `trained`；训练前 1,408 行模型预检、终止边界
预检均为零错误。由此确认此前 SSH 中断发生在训练与报告写入之后，无需重训。

使用 `controller_first`、最多 2 次工具决策、temperature 1.2，在 V3 validation 上选择
16 个任务，每题 4 个固定 seed。样本同时覆盖四种恢复分层和正确目标位于初始关键词
第 0/1 位的情况：

| 模型 | 成功数 | 成功率 | 非法动作 | 空动作/输出错误 |
|---|---:|---:|---:|---:|
| 纯 SFT v7 | 56/64 | 87.50% | 6 | 2 |
| V3 verified SFT | 58/64 | 90.63% | 2 | 0 |

逐条配对为 candidate-only success 2、baseline-only success 0、共同成功 56、共同失败 6。
目标位于第 0 位时 V3 SFT 为 30/32，较旧 SFT 的 28/32 增加 2；目标位于第 1 位时双方
均为 28/32。所有新版轨迹平均策略动作数均为 2，证明评测实际执行了“首次搜索 -> 接收
错误观察 -> 再次搜索”的 Agent Loop，而不是单次模型调用。

仍有一类明确共同失败：北京“艺术/建筑”诊断样本指出“艺术造成召回边界扩散、建筑
匹配正常”，两者都在 4/4 采样中原样重复旧参数，未保留第二位的“建筑”。因此本轮只
能称为正向冒烟，不能作为正式 SFT 提升结论。已启动完整 validation 64 题 × 4 seed 的
配对评测；正式结果将按恢复方式、证据表达、目标位置和城市分层报告。

## 16. V3 SFT 完整配对结果

在全部 validation 64 题、每题 4 个相同 rollout seed 上完成 256 条逐条配对轨迹：

| 指标 | 纯 SFT v7 | V3 verified SFT | 变化 |
|---|---:|---:|---:|
| 成功数 | 211/256 | 221/256 | +10 |
| 成功率 | 82.42% | 86.33% | +3.91pp |
| 输出/参数 Schema 错误 | 12 | 1 | -11 |
| 未知参数错误 | 12 | 0 | -12 |

配对分布为 candidate-only 12、baseline-only 2、共同成功 209、共同失败 33。精确 McNemar
双侧检验 `p=0.01294`，按任务聚类 bootstrap 95% 区间为 `[+1.17pp, +6.64pp]`；候选
相对错误减少 22.22%。按最低 256 对、绝对增益至少 3pp、候选成功率至少 85%、
`p<=0.05` 且聚类区间下界大于 0 的 SFT 比较门禁，本轮正式通过。

但语义最差分层门禁未通过：

| 分层 | 纯 SFT v7 | V3 verified SFT |
|---|---:|---:|
| 诊断式改参，目标位置 0 | 14/32 | 17/32 |
| 诊断式改参，目标位置 1 | 14/32 | 15/32 |
| 显式改参，目标位置 0 | 30/32 | 32/32 |
| 显式改参，目标位置 1 | 29/32 | 29/32 |
| 诊断式同参重试，位置 0/1 | 31/32、31/32 | 32/32、32/32 |
| 显式同参重试，位置 0/1 | 31/32、31/32 | 32/32、32/32 |

V3 SFT 将核心诊断式改参从 28/64 提升到 32/64，但 50% 仍低于预设的 65% 最差分层
门槛。因此 SFT 被接受为下一阶段 warm start，不作为最终候选。当前已对训练分区的 128 个
诊断式改参状态启动每题 4 次在线采样；全成功状态只留作评测锚点，全失败状态回流 SFT
repair，只有组内同时出现成功和失败的 exact state 才进入 GRPO。

模型已完整备份到本地 `ml/agentic/checkpoints/stage3-decision-loop-sft-v3-from-sftv7-lr2e6-epoch1`，
主 adapter SHA-256 为
`f4337d76e002d61dac4e89e9202c403954fe0e8ffd3f320eb25d29c3dee466ac`，与云端一致。

## 17. 状态级 GRPO 难例路由

以 V3 SFT 为策略，在 V3 train 的全部 128 个“诊断式改参”状态上执行 4 个固定随机
采样，共 512 条真实两轮 Agent Loop。首轮路由结果：

| 路由 | 独立状态数 | 规则 |
|---|---:|---|
| `grpo_update` | 62 | 成功率 0.25 / 0.50 / 0.75，组内非零方差 |
| `sft_repair` | 57 | 0/4，全失败 |
| `evaluation` | 9 | 4/4，全成功 |

62 个 GRPO 状态中，成功率 0.25 有 23 个、0.50 有 19 个、0.75 有 20 个。512 条轨迹
总成功率只有 30.66%，再次确认该子集确实是模型的语义瓶颈，不是从易题上制造虚假 RL
收益。所有 rollout 均产生模型动作，策略输出/参数 Schema 错误为 0；失败主要是合法
`search_pois` 参数没有按诊断证据删除噪声词，属于 RL 可以比较的行为质量差异。

首轮 62 个 exact state 距离预设的 64 状态门槛差 2 个。没有降低门槛，也没有把 0/4
状态直接加入 GRPO；当前只对 57 个 `sft_repair` 状态使用全新的 rollout seed 补采 4 次。
状态级路由器已支持合并多份独立审计报告并按 task ID 去重：任一轮证明存在混合成败
方差即可入选，始终全失败的状态继续留在 SFT repair 支路。

补采 57 个状态后，21 个出现新方差、36 个继续全失败；与首轮合并并按 task ID 去重后，
最终得到 83 个 GRPO exact state。训练集覆盖正确目标位置 0/1 为 45/38，北京、南京、
广州、成都分别为 19/21/23/20；validation 保留完整 64 题。数据预检为 `ready=true`，
train/validation task ID 与环境指纹均无重合。

## 18. Turn-Credit GRPO 冒烟

从 V3 SFT adapter 分叉，以 `controller_first`、2 次工具决策、8 candidates/group、
temperature 1.2、constant LR 1e-6、beta 0.04、turn-credit blend 0.5 执行 2 optimizer
steps。训练耗时 57.52 秒，16 条轨迹、48 个模型生成片段对应 32 个环境策略动作和 16 个
最终 assistant 片段：

- 16 条轨迹全部通过多轮资格检查；
- 5 个动作获得有效非零局部信用；
- 1 个可比较 turn bucket 具有非零方差；
- 16 个 unmatched 最终文本片段全部为零信用；
- `alignment_rejected_trajectories=0`、`extra_unmatched_model_turns=0`；
- `invalid_action_positive_credit_count=0`；
- Turn Credit 训练证据门禁错误为空。

由此确认修正后的 R1 投影没有把最终文本或终止后输出混入工具策略奖励。该 2-step
checkpoint 仅用于行为回退冒烟，不用于效果声明；通过验证后再以相同超参数从原 V3 SFT
重新启动正式训练，避免将调试更新混入正式 run。

## 19. 首轮正式 R1 诊断与超参数对照

首轮正式 Turn-Credit R1 从 V3 SFT 原始 checkpoint 分叉，使用 83 个状态级可学习样本、
8 candidates/group、temperature 1.2、constant LR 1e-6、beta 0.04，训练 10 optimizer
steps并在第 5/10 步保存 checkpoint。训练数值稳定：总耗时 518.12 秒，train loss
`0.09032`，末段 KL 约 `0.00043`，completion clipped ratio 为 0；但该 run 只推进到
`epoch=0.12048`，实际只覆盖约 10/83 个训练状态。

训练证据门禁将该 run 标记为 `rejected`，原因是 80 条轨迹中有 1 条出现
`3 model spans / 1 policy record`：79 条轨迹均为 2 个可审计工具动作加 1 个正常 final
assistant 片段，异常轨迹则额外缺少 1 个环境动作记录。投影器已将该异常轨迹整条 advantage
清零，因此没有发生错误信用更新；但逐轮对齐证据不完整，不能把该 checkpoint 晋级或对外
宣称为正式 R1 结果。零容忍门禁保持不变。

在 V3 validation 的全部 16 个“诊断式改参”任务上，每题使用与 SFT 基线相同的 4 个
rollout seed，得到如下严格配对结果：

| 模型 | 成功数 | 成功率 | 相对 V3 SFT 的逐样本变化 |
|---|---:|---:|---|
| V3 SFT | 32/64 | 50.00% | 基线 |
| R1 checkpoint-5 | 32/64 | 50.00% | candidate-only 0，baseline-only 0 |
| R1 checkpoint-10 | 32/64 | 50.00% | candidate-only 0，baseline-only 0 |

因此首轮 10-step、LR 1e-6 的 R1 既未退化，也没有产生任何可观察的行为提升，不能作为
“RL 提升指标”的证据。代码已新增强制落盘的
`turn_credit_alignment_mismatches.jsonl`：对齐失败时记录 task/trajectory ID、模型片段
边界、credit records 与有界解码文本；日志写入失败会直接终止，异常轨迹仍保持全零且正式
门禁仍要求 mismatch 为 0。相关 Turn Credit 测试本地和云端均为 14/14 通过。

为区分“奖励/数据学不动”和“R1 更新过弱”，当前从同一 V3 SFT 分叉运行标准 trajectory
GRPO B0 超参数对照：LR 3e-6、20 steps，在第 10/20 步保存 checkpoint。该对照只用于
寻找有效训练强度；仍需通过同一核心难题逐样本评测后，才能决定下一轮完整 R1 的学习率、
步数和覆盖范围。

## 20. 标准 GRPO B0 超参数对照结果

B0 对照从同一 V3 SFT 分叉，以 LR 3e-6 训练 20 steps，耗时 726.02 秒，训练报告状态为
`trained`。训练推进到 `epoch=0.24096`，末段 KL 约 `0.00211`，全部 completion 均未
触顶截断。它的目的不是替代 Turn Credit，而是先验证状态级数据与六组件轨迹奖励是否能够
产生真实策略改进。

在 16 个核心“诊断式改参”任务、每题 4 个与 SFT 完全相同的 rollout seed 上：

| 模型 | 成功数 | 成功率 | 绝对变化 | 新增成功 / 回退 | McNemar p |
|---|---:|---:|---:|---:|---:|
| V3 SFT | 32/64 | 50.00% | - | - | - |
| B0 checkpoint-10 | 37/64 | 57.81% | +7.81pp | 5 / 0 | 0.0625 |
| B0 checkpoint-20 | 41/64 | 64.06% | +14.06pp | 9 / 0 | 0.00390625 |

checkpoint-20 已达到严格逐样本统计显著，非法动作也从 SFT 的 33/64 降至 24/64，证明
数据与奖励本身可以学动，首轮 R1 无变化主要来自训练覆盖和更新强度不足，而不是奖励完全
失效。提升仍不均衡：目标位于第 0 位从 17/32 提升至 25/32，目标位于第 1 位保持
16/32；因此尚未达到所有语义子分层都稳定提升的最终标准。

当前正在对 checkpoint-20 执行完整 V3 validation 64 题 × 4 seed 回归。只有确认 256 条
轨迹总体不退化、其他三个恢复分层不受损后，才会将该 checkpoint 作为 B0 候选，并据此
确定下一轮带强制 alignment 日志的完整 R1 训练配置。冻结的 final holdout v2 已生成
128 个任务，四分层、四城市和目标位置均衡，且 task、source task、failure message、模板
和环境指纹对开发集均为零重合；在 checkpoint 选择结束前不使用该盲测集。
