# Native ReAct 困难集模型对照与有界 GRPO 路由

日期：2026-08-31  
状态：Dev 模型选择完成；当前 GRPO 不满足全局晋升条件

## 目标

在不接触 160 条 sealed Test 的前提下，对 Qwen3-1.7B Base、SFT、GRPO 整体替换和 SFT+GRPO 有界路由进行同协议配对评测，并回答三个问题：

1. SFT 是否相对 Base 提升了完整 Agent Loop；
2. 当前 GRPO 是否能作为全局策略模型；
3. 只在训练目标状态调用 GRPO 专家，是否能获得稳定收益。

## 固定协议

- 数据：`evals/native-react-hard-v2` 的 40 条冻结 Dev；160 条 Test 未解封。
- Agent：生产 ReAct 任务图，模型负责检索、恢复、澄清和复杂审核决策，Controller 负责确定性门禁、CP-SAT、Verifier、组装与结束。
- 工具：正式 Dev 使用实时 Provider，并记录故障注入和降级证据。
- 推理：同一 vLLM 端点、`temperature=0`、按 case 固定 rollout seed。
- 统计：逐题配对迁移、exact McNemar、按场景族聚类 Bootstrap 95% CI。

## 结果

| Arm | 通过率 | Verifier 硬通过率 | 平均 Token | 平均策略调用 | 平均延迟 |
|---|---:|---:|---:|---:|---:|
| Base | 36/40（90.00%） | 29/30（96.67%） | 8117.5 | 3.65 | 16128.6 ms |
| SFT | 33/40（82.50%） | 27/32（84.38%） | 7852.2 | 3.58 | 20523.0 ms |
| GRPO 整体替换 | 35/40（87.50%） | 29/32（90.62%） | 7900.3 | 3.60 | 17274.7 ms |
| SFT + GRPO 有界路由 | 34/40（85.00%） | 25/29（86.21%） | 7993.5 | 3.60 | 18279.0 ms |

相对 Base 的配对结论：

- SFT：-7.5 pp，改善 2、回退 5，McNemar `p=0.4531`，95% CI `[-25.0,+10.0] pp`；
- GRPO 整体替换：-2.5 pp，改善 2、回退 3，`p=1.0`，95% CI `[-15.0,+12.5] pp`；
- 有界路由：-5.0 pp，改善 1、回退 3，`p=0.625`，95% CI `[-17.5,+5.0] pp`。

这些差异均没有统计显著性，也不支持“当前 RL 提升了完整 Agent Loop”的简历结论。

## 路由审计

有界路由只允许 GRPO checkpoint 处理 `get_poi_detail` 决策状态：

- 144/144 次策略调用都有路由证据；
- GRPO 专家调用 26 次，其余 118 次由 SFT generalist 处理；
- 专家成功执行的动作全部是 `get_poi_detail`；
- 3 次专家输出越界，被 `SPECIALIST_SCOPE_VIOLATION` 拦截并回退 SFT；
- 回退时同时累计专家失败调用和 generalist 调用的 Token，防止把路由成本低报。

路由前置条件进一步要求：已有城市知识和 POI 候选、尚无 POI 详情，并且天气、活动、交通或营业信息等意图要求的实时证据已经齐全。这样不会为了命中窄专家而打乱生产检索顺序。

## 当前瓶颈

1. 当前 GRPO checkpoint 只学过 `get_poi_detail` 单步状态，训练目标过窄；把它作为全局策略使用属于职责错配。
2. 生产 ReAct 的 `review_itinerary` 由模型选择接受、重算、补证据或安全终止，但现有 React GRPO 工具面没有覆盖这个节点。
3. 困难集主要剩余错误位于实时证据过期/不可用、Verifier 失败后的有界恢复和必要终止，而不是 POI 详情动作本身。
4. 正式 Dev 使用实时网络；Tavily 配额、超时和公共降级会使同一模型重复运行出现波动。因此它适合作为生产韧性 Canary，不适合作为唯一的模型选择环境。

## 下一阶段门禁

1. 补齐训练环境与生产 ReAct 的职责一致性，让 `review_itinerary` 决策真正进入训练工具面。
2. 用与困难集文案、城市和状态指纹隔离的合成快照构造：可修复失败、恢复预算耗尽、证据不可核验和无安全替代四类反事实状态。
3. 先用确定性 Snapshot 环境做可学习性审计，只把同组有成功有失败、Reward 非零方差的任务送入 GRPO；全失败任务先进入 SFT repair，全成功任务只作回放锚点。
4. 新候选必须相对同起点 SFT 在独立冻结 validation 上获得正向配对迁移，且旧 POI、澄清、搜索和正常规划能力不回退，才允许进入下一次完整 Dev Canary。
5. sealed Test 仍保持一次性使用，当前阶段禁止解封。

## 可诚实对外表达的结论

已完成生产 ReAct 困难集的四臂配对评测、GRPO 状态级路由、越界回退、成本归因和统计门禁；实验否决了“窄 GRPO 全局替换”的错误晋升方案，并将下一轮 RL 目标收敛到 Verifier 驱动的恢复与安全终止决策。当前不能声称 Qwen3-1.7B 的 RL 已提升全链路指标。

---

## 2026-08-31 14:45 暂停点：Verifier Repair P1 可学习性审计

### 本轮完成内容

1. 将 `constraint-flexibility.v1` 从 ORM 模型中拆成纯 Pydantic 合同，使离线训练链不再隐式加载 SQLAlchemy、pgvector 或数据库模型。
2. 补齐 Verifier 失败后的三个模型决策目标：
   - `retry_solve`：只在用户授权求解器调整且仍有重试预算时允许；
   - `propose_tradeoff`：存在用户明确允许的放宽选项时提出取舍；
   - `abort`：硬约束不可放宽、没有安全替代时停止。
3. 建立 P1 确定性快照语料和 smoke-only 均衡派生集：训练 3 条、Validation 12 条；派生过程不读取 sealed Test。
4. 增加浅层捷径审计和确定性 Controller 强基线。三个浅层规则在 Validation 上均为 `33.33%`，确定性路由为 `100%`。这说明标签合同一致，但也明确暴露出当前任务仍可被 Controller 解出，现阶段 GRPO 只能证明训练工程闭环，不能单独证明开放式策略智能。
5. 在全新隔离 venv 中通过 CUDA、BF16、NF4、TRL environment API、依赖完整性和相关单测门禁；未宣称整个 backend 全量测试通过。

### 冻结的 SFT Validation 基线

- 模型：`qwen3-1.7b-native-react-verifier-repair-semantic-sft-e05-lr1e5-v2`
- 协议：12 个状态，每个状态 4 个随机 rollout，共 48 次；`temperature=1.2`、ReAct、最多 2 次工具迭代、NF4、seed `20260909`。
- 总成功：`17/48 = 35.42%`；非法动作、解析错误和策略输出错误均为 0。
- `abort`：`16/16` 成功，已饱和，只保留为评测锚点。
- `propose_tradeoff`：`0/16` 完全成功，6 次获得可验证部分分，存在组内方差。
- `retry_solve`：`1/16` 完全成功，存在组内方差。
- 路由：4 组 `evaluation`、4 组 `sft_repair`、4 组 `grpo_update`。
- 证据目录：`artifacts/native-react-posttraining/verifier-repair-p1-sft-baseline-smoke12x4-seed20260909`
- `report.json` SHA256：`89f4769e7112ec4953a1158808e03a4b9e15eddbcc2cac97b4f13cfe15accfb8`
- `rollouts.jsonl` SHA256：`a7e5eddfcd9bcbef1faa5e4b4a189505eb17e7ed65adb65aee98a43406ed552e`

### 训练 3 题的可学习性审计

使用同一 SFT、同一 seed 和同一采样协议，各题生成 4 次，共 12 次。审计已正常完成，退出码为 0：

| 目标 | 成功 | Reward 分布/均值 | 路由 | 结论 |
|---|---:|---:|---|---|
| `abort` | 4/4 | 均为 1.0 | `evaluation` | 已学会，没有 GRPO 梯度 |
| `propose_tradeoff` | 0/4 完全成功 | 0.428571、0.714286、-1、-1；均值 -0.214286 | `grpo_update` | 虽未完全成功，但存在可验证部分分和非零方差，可学习 |
| `retry_solve` | 2/4 | 1、1、-1、-1；均值 0 | `grpo_update` | 好坏轨迹各半，是清晰的偏好信号 |

整体为 `6/12 = 50%` 完全成功，两个训练组满足非零方差门槛，因此审阅专家要求的“至少一个真实训练组可产生 GRPO 信号”已通过。该结论只是可学习性门禁，不是 RL 提升结论。

- 证据目录：`artifacts/native-react-posttraining/verifier-repair-p1-sft-train3x4-seed20260909`
- `report.json` SHA256：`1fa5c9a0defefec14522cc59de734d14042ce572719cd9d9d75940029831bd5e8`
- `rollouts.jsonl` SHA256：`4d8c6f704fb78b2308ec008b9b1c4eeb6f39b6f29e124fb4d41a55e320e3a2bc`
- `group_decisions.jsonl` SHA256：`90a60a0fe5aa82906a77122176ecf70404f8bf569fb2d44fd10f07bc68f9bfe5`

### GRPO smoke 失败及根因

首次三步 GRPO smoke 在 `0/3`、尚未执行任何 optimizer step 时失败，未产生模型更新或 checkpoint。失败目录保留为工程证据：

`artifacts/native-react-posttraining/verifier-repair-p1-grpo-smoke3step-g4-seed20260909`

表面错误是：目标动作是 `abort`，但 replay 后的运行态仍不允许 `abort`。进一步做了 raw row 与 Hugging Face `Dataset.from_list` 的逐字段对照，已确认根因：

- 原始 Python 字典逐题 reset 均能正确到达 `review_itinerary`；
- 经 Arrow/Hugging Face Dataset 往返后，三个状态均无法到达目标授权边界；
- Dataset 为异构嵌套字典推断联合 schema，并向每条样本缺失的 `constraint_flexibility.relaxation_options` 键自动补 `None`；
- 补出的 `None` 使约束灵活性合同解析失败，Controller 因而丢失用户授权，导致 replay 状态和目标标签不一致。

因此这不是模型能力失败、标签错误或显存问题，而是训练数据进入 Arrow 后发生了静默语义漂移。刚开始准备的本地修复补丁因文件上下文不匹配而整体失败，没有应用半成品修改。

### 恢复工作时的下一步

1. 在 HF Dataset 边界把 `task` 与 `snapshot` 编码为规范 JSON 标量，环境 reset 时再解码；保留模型 prompt 的结构化格式。这样避免 Arrow 联合 schema 改写不可变环境合同。
2. 增加三层回归：JSON 往返等价、真实 Dataset 往返等价、三个 verifier-repair 状态均能到达预期 review 授权动作。
3. 本地相关单测、Ruff 通过后同步云端，再用同一 seed 做 Dataset reset 复验；禁止通过更换 seed 掩盖问题。
4. 经后训练审阅专家复核后，使用全新输出目录启动 3-step、group 4 的 GRPO smoke；保留当前失败目录。
5. 报告必须同时记录：`scheduled_optimizer_steps=3`、非零方差 group 数、实际产生非零优势的有效更新 step 数。零方差 `abort` 不能算作有效 RL 更新。
6. 训练后按完全相同的 12 状态 × 4 rollout 协议与冻结 SFT 基线配对比较。只有独立 Validation 改善且旧能力不回退，才能写“RL 确实提升指标”。

### 暂停状态

- 云端仓库：`/root/autodl-tmp/TravelAgent2-verifier-repair`
- 干净训练环境：`/root/autodl-tmp/venvs/travelagent-grpo-smoke-p1`
- 当前无 screen 训练任务、无 GPU 计算进程。
- sealed Test 未解封。
- 当前仍不能对外宣称 RL 已提升指标。

---

## 2026-09-01：理由质量后训练与 DPO 隔离协议

### 当前结论

Agent/Controller 的权限边界、训练/评测数据隔离、共享 Chat Template、梯度累积回归测试和分阶段晋升门已经收口；当前未过线的是 Qwen3-1.7B 对 `abort`、`propose_tradeoff`、`retry_solve` 的“为什么此动作合适”这一理由能力。

V10、V11、V13 三条 SFT 尝试均已按预注册门槛判失败，不再继续 SFT 调参，也没有启动 GRPO。专家只批准一次 quarantine DPO：从结构和动作保持最好的 V11 step-12 起步，reference 为同一个 V11 的冻结副本；固定 24-step horizon，并在 step 12 强制停止评测。DPO 尚未启动，当前处于代码与数据硬审计阶段。

### 严格理由评测修正

旧指标只检查是否复制了 verifier evidence，导致“证据复读”也可能被算作成功。现已将理由拆成独立硬门：

- `grounding`：完整引用可见证据；
- `specificity`：保留日期、时长、金额等可见锚点；
- `language`：使用用户语言；
- `public_language`：不泄露 action、verifier、controller 等内部实现；
- `action_rationale`：必须说明为什么应重算、取舍或终止，而不是只复读冲突。

教师目标已改为“动作依据在前、完整 evidence 在后”。每个动作有 8 个公开理由前缀，避免旧 adapter 在 evidence 后提前 EOS 的前缀捷径。严格审计发现：V9 在旧口径下为 `66/80`，但严格理由完整成功为 `0/80`，因此旧结果不能再作为理由质量结论。

### 数据集 V11

路径：

`/root/autodl-tmp/corpora/verifier-reason-quality-sft-v11-rationale-first-balanced-p1-source-even`

- train / internal-dev / train-shadow：`300 / 80 / 80`；
- train 三动作精确 `100 / 100 / 100`；
- 40 个 optimization source，每个贡献 7 或 8 条，每个 source/action cell 为 2 或 3 条；
- internal-dev 与 train-shadow 的 source 完全隔离；
- 460 条模型可见 payload 全部唯一，无官方 validation/test 输入；
- 理由 grounding、strict quality、rationale-first、schema 均为 `100%`；
- `training_authorized=false`，真人 blind review 仍为 pending，所有训练结果只能标记 AI-provisional/quarantine。

### 已执行实验

| 实验 | 结果 | 判定 |
|---|---|---|
| Base step-0，80 条 internal-dev | structure `80/80`，action `70/80`，rationale `18/80`，grounding `21/80`，full `4/80`；retry action `0/10` | 基础模型有工具调用能力，但理由和 retry 不足 |
| V10 hard-case SFT | action `74/80`，严格 rationale `0/80`，full `0/80` | 失败，停止 |
| V11 unweighted SFT，horizon 24 / stop 12 | structure `80/80`，action `74/80`，grounding `43/80`，rationale `7/80`，full `1/80` | 未达到 rationale `20/80`，不恢复到 step 24 |
| V12 rationale-weight 3 | 发现 Transformers 5 自定义 loss 的梯度累积未除以 12；loss/gradient 被放大约 12 倍 | 协议失败，模型不做行为评测，不作证据 |
| V13 修复 grad-acc 后 rationale-weight 3，horizon 24 / stop 12 | structure `78/80`，action `69/80`，grounding `29/80`，rationale `10/80`，full `0/80`；retry action `1/10` | 有效运行但能力门失败，不恢复到 step 24 |

V11 checkpoint-12 adapter SHA-256：

`974adf643c1f62ff403182f517739b8d51295bc911d660df6420cba6d4b47ed3`

V13 checkpoint-12 adapter SHA-256：

`36aa1cb039af9ae2736c5a0c68c758af060cec7cdf59a051c8e6ef8861a99922`

### 零优化器信号诊断

在 V11 step-12 上使用 12 条均衡样本做了只前向/反向、不执行 optimizer 的诊断。adapter 前后 SHA 完全一致。

- rationale prefix 平均 token NLL：`4.510`；
- evidence 平均 token NLL：`0.05695`；
- chosen 相对 evidence-only 的平均 log-prob margin：`-1.161`；
- rationale 梯度贡献显著高于 evidence/action/EOS。

这证明问题不是“理由没有梯度”，而是当前模型稳定偏好 evidence-only + 立即 EOS。V11/V13 SFT 已失败且该偏好被零更新诊断确认，因此满足尝试一次最小 DPO 的必要条件。

### 唯一允许的 DPO 协议

新增：

- `scripts/build_verifier_reason_dpo_preferences.py`
- `ml/agentic/training/train_dpo.py` 的 train-only same-action reason-quality 合同
- `backend/tests/unit/agentic/test_verifier_reason_dpo_preferences.py`

数据协议固定为 300 个 optimization prompt、600 对偏好、每动作 200 对。每个 prompt 有两种 rejected：

1. 同 prompt/action/evidence，只输出 evidence 后立即 EOS；
2. 同 prompt/action/evidence，使用其他动作的语义错误理由，并用长度反向补偿第一类短负例。

构建器必须证明：300 个 context 各有 A/B 两对、40 个 source 且每 source 14/16 pairs、每 12 个顺序 micro-batch 为三动作各 4 条且 A/B 各 6 条、completion 长度均值差全局不超过 0.5 token/每动作不超过 1 token、长度单变量 AUC 在 `[0.45, 0.55]`、最大完整序列不超过 5120 token、共享模板 prompt/suffix 边界完全一致。

训练协议锁死为：V11 step-12 policy + 同 checkpoint 冻结 reference、`beta=0.1`、`lr=5e-6`、effective batch `12`、warmup `1` step、linear horizon `24`、max grad norm `1`、seed `20260923`、step 12 停止。禁止更换 seed、beta 或学习率进行 sweep。

step-12 继续门：structure `>=79/80`、action `>=72/80`、rationale `>=20/80`、grounding `>=40/80`、full `>=10/80`，三个 route 的 rationale 均非零，且 schema/controller-owned 违规为零。任一失败即退休 DPO 路线；只有全部通过才可从同一 checkpoint 恢复到 step 24。

### 当前暂停点

- 本地 DPO builder、trainer fail-closed 检查和单测已落盘；`py_compile` 与手工 600-row validator 通过。
- 已修复代码审查发现的 adapter 参数哈希运行时 import 问题，并要求 step-1 同时记录有限的 loss 与 gradient norm。
- Windows 本地 pytest 因缺少 `pytest_asyncio` 未启动；不在本机临时污染依赖，下一步在已有云端隔离 venv 运行相关 pytest。
- 真实 V11 tokenizer 的 600-pair 长度/AUC 审计尚未执行，DPO 训练尚未启动，GPU 当前空闲。
- internal-dev 仅用于 step-12 行为门；train-shadow、frozen test 均未解封。
- 即使 DPO 通过 step 24，也只能成为 quarantine candidate；真人 blind review 完成前不可 promotion/deploy，也不可对外声称“后训练已提升完整 Agent 能力”。
