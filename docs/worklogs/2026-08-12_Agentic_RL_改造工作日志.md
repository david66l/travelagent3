# TravelAgent2 Agentic RL 改造工作日志

> 日期：2026-08-12
> 项目：`E:\A_Louis\TravelAgent2`
> 分支：`feat/agentic-rl-long-horizon`
> 当前状态：按用户要求暂停开发，保留现场
> 目标：将现有确定性旅行规划系统改造成“确定性约束求解 + 可训练策略模型 + Agentic RL”的 Long-Horizon 出行智能体

## 1. 今天最终形成的整体逻辑

今天不是简单地给原项目加一个 RL 脚本，而是把系统拆成职责明确、可以闭环的四层：

1. **策略模型负责决策**：看当前任务、约束、证据、失败历史和剩余预算，只决定下一步动作。
2. **Controller 负责权限和流程**：维护任务图、允许动作、状态版本、预算、重试和终止条件。
3. **确定性工具负责事实与求解**：天气、POI、路线矩阵、VRP 求解器、硬约束验证器提供可验证结果。
4. **训练系统负责改进策略**：先用高质量轨迹做 SFT，再在隔离快照环境中用六类 Reward 做 GRPO。

核心闭环如下：

```text
用户需求
  → Goal Ledger（目标和约束）
  → Task Graph（长任务拆解）
  → Policy 选择一个动作
  → Controller 校验动作权限和参数
  → 工具/求解器执行
  → Verifier 校验该子任务是否真的完成
  → 更新可信事实、产物、失败记录和预算
  → 下一轮 Policy 决策
  → Completion Guard 确认硬约束通过后才允许结束
  → 轨迹进入评测、SFT 数据或 GRPO Rollout
```

这样，大模型不是替代所有业务逻辑，而是负责长任务中的策略选择；城市、约束、POI、矩阵、行程和验证结果仍由程序掌握，防止模型伪造成功。

## 2. 今日完成的主要工作

### 2.1 建立改造前质量基线

- 冻结改造前开发基线，建立独立开发分支。
- 增加 Phase 0 检查脚本和统一质量门禁。
- 在较早的完整回归中取得：
  - `693 passed`
  - `123 deselected`（依赖数据库的测试）
  - `25 warnings`
  - 用时约 `271.71s`
- 当时 lint、format、mypy、compile 和前端类型检查均通过；本机未安装 Go，因此 Go 检查跳过。

### 2.2 统一观测、验证和工具安全边界

- 新增版本化 `ObservationEnvelope`，统一表达：
  - 工具是否成功；
  - 数据来源；
  - 置信度；
  - 是否 fallback；
  - 错误码；
  - 快照版本和环境版本。
- 新增程序化行程验证器，输出稳定的硬约束违规码、软分和指标。
- 新增 Tool Guard 的 `off / shadow / enforce` 模式。
- 新增 Completion Guard，模型不能自己宣称“任务完成”或“硬约束通过”。
- 将 POI 搜索、路线矩阵、VRP 求解和验证器正式暴露为 Agent 可选择的动作。

### 2.3 建立 Long-Horizon Agent 状态与循环

- 新增 Goal Ledger，保存目标、硬约束、软偏好、缺失信息和能力判断。
- 新增 Task Graph，把完整旅行规划拆成有依赖关系的子任务。
- 新增版本化 Fact、Artifact、Failure、Plan Version 和 Budget Ledger。
- 新增 `BoundedAgentLoop`：
  - 每轮只向策略提供当前状态允许的动作；
  - 支持依赖调度和安全的并行只读任务；
  - 执行后必须经过 Verifier；
  - 失败可记录、重试或进入安全终止；
  - 受 step、tool、solver、token 和 latency 预算约束。
- 新增局部重规划与状态失效逻辑，目标变化后不重跑所有步骤，只失效受影响的事实和任务。

### 2.4 接入现有 TravelAgent 生产流程

- 增加 API Policy，让当前 DeepSeek/OpenAI-compatible API 可以先充当 Teacher 和线上策略。
- 增加 `TravelActionExecutor`，将 Agent 动作接到现有天气、POI、路线、求解器和验证器。
- 将可信大对象改为 Controller 注入：
  - destination；
  - constraints；
  - candidate POIs；
  - route matrix；
  - solver itinerary；
  - validation facts。
- Policy 只做有限选择，不允许复制或改写这些受保护载荷。
- 将 Agent Branch 正式接入 LangGraph；当前默认仍保持 shadow 配置，不直接替换稳定路径。
- Agent 生成的草稿只有在验证通过后才映射回原项目的 legacy itinerary 格式。

### 2.5 建立可回放、隐私安全的轨迹系统

- 每一步记录：
  - 决策上下文；
  - 动作和参数；
  - 工具观测；
  - 验证结果；
  - 状态前后哈希；
  - 环境、策略、验证器版本。
- 增加轨迹内容哈希和回放校验。
- 在持久化前递归脱敏手机号、邮箱、身份证等直接标识信息。
- 修复策略上下文缺少真实证据的问题：现在会把受限大小的事实和产物摘要投影给 Policy，而不只是给不可读的引用 ID。

### 2.6 建立 Agent 与原确定性链路的成对评测

- 同一场景可同时运行 deterministic 和 agent 两条路线。
- 增加归一化指标和 release gate：
  - hard pass；
  - validated success；
  - P95 latency；
  - tool calls；
  - solver calls；
  - token usage；
  - fallback 和失败原因。
- 默认正式门禁要求 300 个精确配对场景，并防止 Agent 在核心正确性指标上退化。

### 2.7 建立真实 SFT 数据与训练入口

- 新增 SFT Dataset Builder，对真实 Agent Episode 做三层审核：
  - L1：轨迹完整性、版本、隐私和授权；
  - L2：动作权限、参数边界、工具观测、重复调用和 grounding；
  - L3：结果是否为 validated plan、有效澄清或安全终止。
- 训练/验证/测试按 template family、city 或生产用户分组切分，防止泄漏。
- 增加 manifest、review 和 dataset version。
- 新增真实 TRL + PEFT QLoRA SFT 入口：
  - 默认 `Qwen/Qwen2.5-3B-Instruct`；
  - 4-bit NF4 double quant；
  - LoRA；
  - completion-only loss；
  - CUDA 和最小 3000 条训练样本门禁。
- 当前开发环境没有安装正式训练依赖，也没有合格规模数据，因此**没有声称已经完成模型训练**。

### 2.8 建立六类 Reward 和反 Reward Hacking 门禁

训练主 Reward 包含六类：

1. Task Reward：任务是否真实完成。
2. Constraint Reward：硬约束是否通过。
3. Format Reward：动作和观测格式是否有效。
4. Tool Reward：工具调用是否正确且产生信息增益。
5. Grounding Reward：参数是否来自可信上下文。
6. Efficiency Reward：是否避免重复、空转和浪费调用。

额外处理：

- 文案质量分暂时只审计，默认权重为 0，避免未经校准的 LLM Judge 主导训练。
- 安全违规、伪造事实、非法环境修改直接记为 `-1`。
- 硬约束失败却请求完成、或失败终局，Reward 上限被压到 `-0.25` 以下。
- 过程奖励总权重不超过 0.2，防止模型靠“调用很多看似正确的工具”刷分。
- 每一步都保存局部 Reward 信号，为后续 turn-level credit assignment 做准备。

### 2.9 建立隔离快照 Rollout 环境

- 每个 Rollout 使用独立环境和工具计数器。
- 同一 GRPO group 使用相同 task、snapshot、environment version 和初始状态指纹。
- 训练环境不访问实时供应商，所有工具返回来自不可变快照，保证可复现和避免成本污染。
- 快照会校验工具参数是否与预期一致。
- 已跑通完整 9 步旅行规划快照流程，相关环境测试 `5 passed`，实际测试执行约 `0.27s`。

### 2.10 建立 GRPO Group 审计和离线数据出口

- 新增 Group Auditor，检查：
  - group size；
  - task 和初始状态一致性；
  - snapshot/environment 一致性；
  - trajectory 唯一性；
  - Reward 是否有有效方差。
- 全成功或全失败的零方差 group 不进入参数更新，分别路由到评测、SFT 修复池或降难度队列。
- 新增 GRPO-B0 数据出口，明确只使用 trajectory-level advantage，并保留 turn-level process signal 供后续对照。

### 2.11 修复“训练格式与线上推理格式割裂”问题

这是今天后半段发现的最关键问题之一。

原状态：

- API Policy 输出 `{"action": ..., "arguments": ...}` JSON；
- 但 Hugging Face 官方工具训练格式要求 assistant 产生结构化 `tool_calls`，并为样本附带 `tools` Schema；
- 如果保持原样，SFT 训出来的模型无法无缝接到原生工具调用推理。

解决方案：

- API Teacher 暂时仍可使用结构化 JSON，保证现有 API 兼容性。
- SFT 和 GRPO 训练样本统一升级为原生 `tool_calls`。
- 新增 Policy-visible Action Schema，覆盖 Controller 动作和工具动作。
- 新增 `NativeToolAgentPolicy`，本地 SFT/GRPO checkpoint 和支持工具调用的 API 模型均可通过同一路径接回生产 Agent Loop。
- 新增配置：
  - `AGENTIC_POLICY_PROTOCOL=json|native_tool`
  - `AGENTIC_POLICY_MODEL=<served checkpoint>`
- 这一部分针对性测试 `26 passed`。

### 2.12 缩小模型权限和减少无效输出

发现原工具 Schema 要求模型生成 city、POIs、constraints、matrix、itinerary 等字段，但生产执行器其实会从可信状态注入。这既浪费 token，又允许模型伪造关键事实。

解决方案：建立独立的 Policy Action Contract：

- `get_weather`：模型最多决定 date，不决定 city。
- `search_pois`：模型最多决定 grounded keywords/category，不决定 destination。
- `get_poi_detail`：无模型参数，Controller 选择候选 POI。
- `get_route_matrix`：无模型参数，Controller 注入 POIs 和 constraints。
- `solve_itinerary`：模型只可选 `auto/cpsat/greedy`。
- `validate_itinerary`：无模型参数，Controller 注入 itinerary、constraints、facts。
- `compose_draft / finish / capability_check`：显式空参数 Schema。
- `ask_user / abort / propose_tradeoff`：只允许必要且可审计的文本参数。

API Policy、Native Tool Policy、SFT 数据和 GRPO 数据现在共用同一套参数校验。

### 2.13 关闭 Agent Policy 的无意义思考模式

- 原先只对 intent 和 output formatting 关闭 DeepSeek V4 thinking。
- 今天把 `agent_policy` 也加入关闭范围。
- 原因：每一步只是受限动作选择，不需要十几秒隐藏推理；程序验证器负责结果真假。
- 同时把 Native Tool Call 的默认生成上限缩到 256 token，并要求 exactly one tool call，减少延迟和格式漂移。

### 2.14 建立真正的 TRL 多轮环境适配器和 GRPO 启动脚本

- 根据 Hugging Face TRL 官方 `environment_factory` 契约实现 `TRLTravelEnvironment`。
- TRL 每个 rollout 使用独立环境实例，并在同一实例中多轮调用工具。
- 环境内部复用生产 `InteractiveAgentSession + BoundedAgentLoop + TravelActionExecutor + Verifier + Reward Engine`，没有另造一套 RL-only 状态机。
- 虽然 TRL 会暴露环境的全部 public tools，但每一步仍由生产 Controller 的 allowed actions 做门禁；越权动作失败、记录并扣分，不能写入可信状态。
- 新增 GRPO Corpus Preflight：
  - 默认至少 1000 个训练任务；
  - 必须有 validation；
  - task ID 和初始状态指纹不得跨 split 重叠；
  - 快照工具和响应必须完整；
  - 禁止 PII；
  - 检查训练依赖。
- 新增真实 QLoRA GRPO-B0 启动脚本，默认：
  - 8 generations；
  - 最多 16 轮工具调用；
  - 4-bit NF4；
  - LoRA；
  - trajectory-level hierarchical reward；
  - 可选 colocated vLLM。
- 训练报告强制注明 `credit_assignment_claim=trajectory-level only`，不能把 B0 描述成已解决长程信用分配。
- GRPO corpus/environment/training 针对性测试 `11 passed`。

## 3. 今天遇到的问题、难点和解决办法

| 问题 / 难点 | 根因 | 处理方式 | 当前结果 |
|---|---|---|---|
| 原项目确定性流程与 RL Agent 容易变成两套系统 | 训练环境若另写状态机，线上线下会漂移 | 训练、评测和线上共用 BoundedAgentLoop、Executor、Verifier、Reward | 已解决核心架构问题 |
| 模型容易伪造城市、约束、POI、矩阵和行程 | 直接暴露底层 Tool Schema，权限过大 | 增加 Policy-visible Schema；可信字段由 Controller 注入 | 已实现并测试 |
| Policy 只看到引用 ID，看不到可决策证据 | 状态投影过度精简 | 加入有上限的 fact/artifact 摘要，并移除随机 ID/时间戳 | 已实现并测试 |
| SFT 数据格式与本地工具推理不一致 | 旧样本训练 JSON，官方工具模型使用 tool_calls | 数据统一为原生 tool_calls + tools；增加 NativeToolAgentPolicy | 已实现并测试 |
| Reward 容易被“多调用工具、格式正确”钻空子 | 过程分可能盖过结果分 | 结果优先、过程分封顶、安全门禁、失败终局封顶 | 已实现并测试 |
| GRPO 同组初始状态可能不同 | 环境随机采样会污染相对优势 | 固定 task/snapshot/fingerprint，逐 rollout 隔离实例 | 已实现并测试 |
| 全成功/全失败 group 没有学习信号 | 组内 Reward 方差为 0 | Group Auditor 过滤并分流 | 已实现并测试 |
| 长任务不能只看终局分 | B0 会把同一总分分给整条轨迹 | B0 明确只做基线，同时记录 turn signals；R1 留作对照 | 基础数据已准备，R1 未实现 |
| Agent 每一步 LLM 可能再次变成 15 秒 | DeepSeek thinking 默认开启 | agent_policy 关闭 thinking；限制单次工具选择输出 | 已实现单元测试，待真实 API 压测 |
| 真实 Provider 会导致 RL 不可复现、昂贵 | 在线数据和延迟每次不同 | SnapshotToolExecutor，训练不访问实时 Provider | 已实现并测试 |
| 训练代码可能只是假入口 | 缺依赖、GPU、数据仍可能运行到一半才失败 | SFT/GRPO preflight 检查样本量、split、依赖、CUDA 和版本 | 已实现；尚未正式训练 |
| TRL 环境会暴露全部 public methods | 官方 environment_factory 的工具发现机制如此 | 保留完整工具集，但每步由生产 Controller 拒绝越权动作 | 已实现并测试 |
| Windows 输出出现乱码和 requests 依赖警告 | 当前终端编码与全局 Conda requests 依赖组合问题 | 不影响 venv 测试结果；记录为环境债务 | 尚未专项修复 |
| 最后一轮全量回归被中断 | 用户要求立即暂停并写日志 | 保留所有现场，明确未完成项 | 待恢复后第一步执行 |

## 4. 测试与验证记录

### 已完成的验证

- 较早完整质量基线：`693 passed, 123 deselected, 25 warnings`。
- Agent Policy、原生工具协议、SFT/GRPO 样本格式：`26 passed`。
- GRPO corpus、TRL environment、SFT preflight 组合：`11 passed`。
- TRL 快照完整流程环境：`5 passed`，测试体本身约 `0.27s`。
- 多轮交互驱动：先前 `3 passed`。
- token、tool、latency 计量相关目标测试：先前均通过。
- 代码格式化和针对本轮文件的 Ruff 检查均通过。

### 尚未完成的验证

- 最新未提交改动的全量 Agentic 测试被用户暂停指令中断。
- 最新未提交改动的完整 mypy 检查被中断。
- 尚未重新运行整个 `scripts/check_phase0.py`。
- 尚未使用真实 DeepSeek API 测 Native Tool Policy 的端到端延迟。
- 尚未安装 TRL/Transformers/Torch/PEFT/bitsandbytes 等训练依赖。
- 尚未在 CUDA GPU 上运行 SFT 或 GRPO smoke training。
- 尚无 3000+ SFT 样本和 1000+ GRPO 任务，因此没有正式 checkpoint 和训练对照结果。

## 5. 代码提交记录

今天已经完成并提交的提交如下：

| Commit | 内容 |
|---|---|
| `1f12668` | 冻结改造前开发基线 |
| `6674b32` | 建立 Phase 0 质量基线 |
| `7e31910` | 版本化 Observation 与行程验证器 |
| `e08ef9d` | shadow-mode 工具调用 Guard |
| `b8351ad` | validator tool 与 Completion Guard |
| `40b723f` | Bounded Long-Horizon Agent Loop 核心 |
| `9b3dc75` | shadow mode 初始化 Agent Ledger |
| `d64a27b` | 依赖感知局部重规划 |
| `917eb81` | 可回放、隐私安全 Episode |
| `c2100df` | POI、路线和 solver Agent Actions |
| `686dc1b` | API Policy 和可信 Action Executor |
| `3891636` | LangGraph 接入验证后的 Agent Draft |
| `7296799` | 成对 Agent Release Evaluation |
| `cb04ffc` | 将有界证据投影给 Agent Policy |
| `411e7a1` | 审核式 SFT Dataset Pipeline |
| `9b059c0` | 六类层级反作弊 Reward |
| `3645d5b` | 隔离快照 Rollout Environment |
| `559093d` | GRPO Rollout Group Auditor |
| `c87244d` | 真实 QLoRA SFT 训练入口 |
| `3f976fc` | 真实 token/tool/latency 预算计量 |
| `60215b3` | 用生产 Agent Loop 实现逐动作交互驱动 |

## 6. 当前未提交现场

暂停时，工作区仍有一组**未提交**改动。这些改动主要属于同一个主题：“原生工具协议 + GRPO-B0 正式训练入口”。

主要新增文件：

- `backend/src/agentic/policy_actions.py`
- `backend/src/agentic/grpo_dataset.py`
- `backend/src/agentic/grpo_training.py`
- `backend/src/agentic/trl_environment.py`
- `ml/agentic/training/train_grpo.py`
- `scripts/build_grpo_dataset.py`
- 对应的 Policy、SFT、GRPO、Environment 和 Training 测试

主要修改文件：

- `backend/src/agentic/policy.py`
- `backend/src/core/llm_client.py`
- `backend/src/agentic/sft_dataset.py`
- `backend/src/agentic/training.py`
- `backend/src/agentic/integration.py`
- `backend/src/core/settings.py`
- `backend/src/agentic/environment.py`
- `.env.example`
- `backend/pyproject.toml`

这些代码已经通过针对性测试，但因为全量回归被暂停，当前不应直接合并或宣称完成。

## 7. 当前进度判断

为了避免“代码写了很多 = 项目训练完成”的误解，进度分开计算：

| 领域 | 当前状态 | 说明 |
|---|---|---|
| Agent Loop 架构 | 基本完成 | Goal、Task Graph、Loop、Verifier、Budget、Termination 已形成闭环 |
| 原生产流程接入 | 基本完成 | Agent Branch 已接 LangGraph，默认 shadow，尚需更多真实场景验证 |
| 轨迹与评测基础设施 | 基本完成 | Replay、隐私、成对评测、Release Gate 已有 |
| SFT 数据工程 | 基本完成 | 审核、切分、原生 tool_calls、preflight 已有 |
| SFT 正式训练 | 未开始 | 缺合格规模数据、训练依赖和 CUDA 训练记录 |
| 六类 Reward | B0 完成 | 程序 Reward 已实现，质量分尚未人工校准 |
| GRPO 环境与审计 | B0 基本完成 | 快照、隔离、group audit、TRL adapter 和启动脚本已有 |
| GRPO 正式训练 | 未开始 | 缺 1000+ 任务、GPU 和训练运行记录 |
| 长程信用分配 R1 | 未实现 | 已记录 turn signal，但尚未做 return-to-go/逐轮信用实验 |
| 上线替换 API Policy | 未开始 | 需要训练 checkpoint、固定评测和灰度门禁通过 |

一句话结论：**工程地基和训练接口已经搭到可以进入数据生产与 GPU 实验阶段，但目前仍是“可训练系统已基本成型”，不是“SFT/GRPO 模型已经训练完成”。**

## 8. 恢复开发后的第一批动作

恢复时按以下顺序继续，避免破坏当前现场：

1. 先检查是否仍有后台测试进程，再重新查看 `git status`。
2. 对最新未提交改动运行 Ruff、mypy 和全部 Agentic 单元测试。
3. 运行完整 `scripts/check_phase0.py`，确认没有破坏旧功能。
4. 修复所有回归后，将“native tool protocol + GRPO training”作为一个或两个清晰 commit 提交。
5. 增加 GRPO task/snapshot corpus builder，生成首批可审计 smoke corpus。
6. 在独立 CUDA 环境安装固定版本训练依赖，先执行 SFT smoke，再执行 GRPO smoke。
7. 生成 Base、SFT、SFT+GRPO-B0 的同测试集结果。
8. 只有 B0 稳定后，再实现 GRPO-R1 的逐轮信用对照。

## 9. 不能夸大的部分

- 目前没有真实 SFT checkpoint。
- 目前没有真实 GRPO checkpoint。
- 目前没有证明本地模型优于 DeepSeek API。
- 目前没有证明 GRPO-B0 优于 SFT。
- 目前没有解决完整的 Long-Horizon credit assignment。
- 目前没有通过最新改动后的完整项目回归。
- 当前能准确表述的是：已经实现了可验证的 Long-Horizon Agent 工程闭环、可审计数据管线、六类 Reward、隔离 Rollout 环境，以及真实 SFT/GRPO 训练入口。

## 10. 本轮参考的官方接口

- Hugging Face TRL SFT Trainer：<https://huggingface.co/docs/trl/sft_trainer>
- Hugging Face TRL Dataset Formats：<https://huggingface.co/docs/trl/en/dataset_formats>
- Hugging Face TRL GRPO Trainer：<https://huggingface.co/docs/trl/en/grpo_trainer>
- Hugging Face Transformers Tool Use：<https://huggingface.co/docs/transformers/en/chat_extras>
- Hugging Face PEFT Quantization：<https://huggingface.co/docs/peft/v0.14.0/developer_guides/quantization>

---

记录说明：本日志依据当日 Git 提交、当前工作区差异和已执行测试整理。最后一次全量回归在开始后被主动暂停，因此所有“最新未提交改动”的状态均以“针对性测试通过、全量回归待执行”为准。
