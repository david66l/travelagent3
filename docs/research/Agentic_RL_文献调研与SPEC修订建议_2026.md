# TravelAgent2 Agentic RL 文献调研与 SPEC 修订建议

> 调研日期：2026-08-12  
> 目标：为 TravelAgent2 的 Agent Loop、SFT、GRPO 和六类 Reward 提供可复现、可落地的研究依据。  
> 结论边界：论文中的数学、代码或搜索结果不能直接外推为旅行规划效果；本文件只提取可迁移的方法，并要求在 TravelAgent2 上重新做对照和消融。

## 1. 结论先行

有必要调研，但不应该只找“最新顶刊”。Agentic RL 仍是快速演进方向，可靠证据主要来自三层：

1. 正式发表论文：用于确定相对成熟的结论；
2. 高相关预印本：用于发现最新方法，但必须标明证据等级；
3. TravelAgent2 自己的复现实验：这是最终能否写入简历、能否上线的唯一依据。

本轮调研后，对原 SPEC 做六项关键修正：

1. **SFT 以真实端到端工具轨迹为主**：Teacher 必须在同一个快照环境里真的调用工具、读 Observation、处理失败并结束；不能把几段文本拼成“伪轨迹”。
2. **普通单轮 GRPO 只能作为基线**：长任务若把一个终局总分平摊给所有步骤，会把错误动作和正确动作一起奖励。首版先保留标准 GRPO 作为可运行基线，同时记录 turn-level 信号；正式候选必须加入分步信用分配对照。
3. **六类 Reward 不再做无条件线性相加**：硬约束、安全和 grounding 是门禁，不能被格式分、文案分补回来；主奖励来自可验证终局结果，过程奖励只使用环境可验证信号且设置上限。
4. **训练任务按模型能力动态采样**：同一任务做一组 Rollout；全成功和全失败的组不进入该轮策略更新，优先保留成功率居中的任务，以维持可学习的组内差异。
5. **训练稳定性增加三类预警**：同时监控组内 Reward 方差、策略熵和梯度范数，避免多轮 RL 先变得机械重复、随后突然坍塌。
6. **追问用户必须进入训练环境**：旅行需求经常缺日期、预算或偏好。首版使用可复现的状态机用户模拟器，后续再用 LLM 用户做鲁棒性压力测试。

## 2. 文献选择与证据等级

| 文献 | 状态 | 主要问题 | 对本项目的用途 | 采用等级 |
|---|---|---|---|---|
| DeepSeekMath | 技术报告 / arXiv，2024 | GRPO 原始定义 | 建立标准 GRPO 基线 | A：基础方法 |
| DeepSeek-R1 | Nature，2025；下载的是作者完整技术版 | 冷启动、纯 RL、可验证奖励 | 设计 SFT→RL 多阶段路线；优先规则奖励 | A：核心依据 |
| ReTool | 预印本，2025 | RL 中实时调用工具 | 工具必须真的进入 Rollout；结果奖励优先 | B：高相关参考 |
| RAGEN | 预印本，2025 | 多轮 RL 坍塌与采样 | 训练预警、组内方差筛选、Rollout 新鲜度 | B：高相关参考 |
| Turn-Level Reward Design | 预印本，2025 | 多轮信用分配 | 记录逐步奖励并设置 turn-level 对照 | B：高相关参考 |
| Tool Zero | Findings of EMNLP 2025 | 工具 RL 的动态课程奖励 | 早期宽松探索、后期严格正确；工具名扰动测试 | A：正式发表 |
| Agent Lightning | 预印本，2025 | 线上 Agent 与训练解耦 | 将 LangGraph 执行转为 state/action/reward 事件，不重写线上图 | B：架构参考 |
| MUA-RL | arXiv；ICLR 2026 投稿已撤回 | 用户交互进入 RL Loop | 只作为用户模拟实验灵感，不作为已验证结论 | C：谨慎参考 |
| Demystifying Agentic RL | 预印本 / 投稿版，2025 | 数据、探索与工具效率经验研究 | 真实轨迹、模型感知采样、少而有效的工具调用 | B：实验参考 |
| AT²PO | ACL 2026 Long Paper | 树搜索、逐轮信用、逐轮优化 | 二阶段研究增强，不作为首个 MVP 前置条件 | A：正式发表 |

证据使用原则：A 级可以支撑架构决策；B 级需要在本项目复现后再下结论；C 级只生成实验假设。

## 3. 四个核心模块应如何设计

### 3.1 Agent Loop：执行环和训练环必须是同一套语义

Agent 每一步统一建模为：

```text
state_t
  -> policy 输出 action_t（工具名 + 参数 / ask_user / finish）
  -> guard 校验
  -> 快照环境真实执行
  -> observation_t+1
  -> validator / reward oracle 记录可验证信号
  -> 下一步或终止
```

必须满足：

- 训练和线上共享 Action、Observation、Tool Schema、Validator；只允许数据适配器不同；
- 一个 Episode 是真实连续执行，不是把多次互不相关的回答拼接成长对话；
- Tool error、空结果、超时、约束冲突和用户补充信息都成为明确 Observation；
- 每个 Rollout 的环境状态隔离，同一个 GRPO group 使用相同初始任务与环境版本；
- 线上 LangGraph 与训练系统解耦，通过统一 Trajectory Event 接口衔接；
- 不保存或监督模型私有思维链，只保存简短决策摘要、结构化 Action 和可验证环境事件。

### 3.2 SFT：教会“怎么做事”，不是教会“怎么写长答案”

数据优先级：

1. Teacher 在真实快照环境中成功完成的端到端轨迹；
2. 线上影子流量中经脱敏、回放、人工或规则验收的轨迹；
3. 真实失败后成功恢复的轨迹；
4. 文本拼接或只给工具调用结果的轨迹只能用于早期格式 smoke test，不进入核心训练集。

每条 SFT 轨迹至少验证：

- Tool Schema、参数来源和调用顺序合法；
- Observation 确实来自对应环境执行；
- 遇到缺信息时会 `ask_user`，而不是猜；
- 遇到失败时有重试、换策略或安全终止；
- `finish` 前实际调用 Validator 且满足终止条件；
- 没有测试集模板、城市组合或用户画像泄漏。

SFT 的作用是建立可用策略初始点。纯 RL 虽可能探索出能力，但 DeepSeek-R1-Zero 也展示了可读性和语言混合等行为问题；TravelAgent2 是受约束业务系统，首版不应跳过冷启动。

### 3.3 GRPO：先做可信基线，再比较多轮信用方案

建议分三步：

#### GRPO-B0：标准轨迹级基线

- 从 SFT Checkpoint 开始；
- 同一任务、同一环境快照生成 `G` 条完整轨迹；
- 以门禁后的终局可验证奖励形成组内相对优势；
- 用它证明训练管线、环境隔离和 Reward 可复现。

#### GRPO-B1：层级 Reward + 动态任务采样

- 收集每一步的格式、工具、grounding 和信息增益信号，但过程奖励总和设上限；
- 任务先小批量试跑，剔除组内全成功和全失败样本；
- 优先选择当前模型成功率约 20%–80% 的任务；
- 维持任务类型、城市、约束冲突和路径长度多样性。

#### GRPO-R1：逐轮信用分配对照

- 比较“同一终局优势给全部 token”与“将可验证过程信号归到对应 turn，并传播未来终局结果”；
- 可先实现 return-to-go / AIR 风格分解，再评估自定义 turn-level GRPO；
- MT-GRPO 的精确展开需要指数级 Rollout 且假定固定轮数，不适合作为本项目首版；
- AT²PO 的熵引导树扩展是后续增强项，只有基线稳定且有足够算力时再做。

### 3.4 六类 Reward：保留六个维度，改成层级结构

原来的六项业务含义仍然成立，但职责必须拆开：

| Reward | 作用层级 | 来源 | 首版是否进入策略奖励 |
|---|---|---|---|
| `R_format` | 过程信号 | Parser / JSON Schema | 是，但低权重且封顶 |
| `R_tool` | 过程信号 | 工具白名单、状态机、信息增益 | 是，但低权重且封顶 |
| `R_grounding` | 硬门禁 + 过程信号 | 参数溯源、POI/Observation ID | 是；严重伪造直接失败 |
| `R_efficiency` | 成功后的修正项 | 重复调用、无增益步骤、时延/Token | 是；失败轨迹不因“短”获益 |
| `R_constraint` | 核心终局结果 + 硬门禁 | Solver / Validator | 是，主奖励 |
| `R_quality` | 终局辅助 / 离线评测 | 规则特征 + 盲测 Judge | 首轮主要做评测；校准后再低权重加入 |

推荐计算顺序：

```text
第一层：安全、工具权限、事实伪造、环境完整性检查
        严重违规 => 立即失败，R_episode = -1

第二层：终局硬约束检查
        finish 且 hard_pass = false => 总奖励封顶为负

第三层：可验证终局目标
        R_terminal = 完成度 + 硬约束 + 软约束质量

第四层：受限过程修正
        R_process = clip(格式 + 工具选择 + grounding + 效率, -c, c)

最终：R_episode = gate(R_terminal + R_process)
```

过程奖励只能来自可验证事实，不能奖励“看起来很聪明的分析”。这兼顾了两类论文结论：ReTool 强调以结果奖励减少过程投机；多轮信用研究又表明完全稀疏的终局奖励会让长任务难学。因此本项目采用“**终局结果为主，少量环境可验证过程信号为辅**”。

## 4. Curriculum 不只是调权重

建议课程设计：

| 阶段 | 环境难度 | Reward 严格度 | 目标 |
|---|---|---|---|
| C0 | 单工具、参数明确 | Schema 合法和部分参数匹配即可得少量分 | 学会协议，建立探索 |
| C1 | 2–3 工具、有缺失信息 | 正确工具、参数来源和追问获得过程分 | 学会信息收集 |
| C2 | Solver + Validator | 主要看可执行结果，过程分开始降权 | 学会闭环完成 |
| C3 | 多约束冲突、工具失败 | 只有正确恢复并通过门禁才有高分 | 学会长程恢复 |
| C4 | 动态重规划、模拟用户改变需求 | 严格 AST/Schema/状态一致性和终局结果 | 学会鲁棒策略 |

Tool Zero 的核心启发不是“永远给部分分”，而是早期用较宽松的可验证相似度帮助探索，随后平滑过渡到严格的工具与参数正确性。

## 5. 用户模拟器

旅行规划不是静态问答。建议环境包含状态化用户：

```text
hidden_profile = 真实预算、日期弹性、必去点、不可接受项、偏好优先级
visible_request = 只暴露部分信息
ask_user(question) = 根据 hidden_profile 和对话历史返回确定性回答
```

首版使用规则/状态机模拟器，优点是便宜、可复现、Reward 稳定。LLM 模拟用户放到鲁棒性评测：它可以制造口语、省略、改口和矛盾，但不宜直接作为第一版训练环境的唯一真相来源。

## 6. 训练坍塌与 Reward Hacking 监控

每次更新至少记录：

- `reward_mean` 和 `reward_std_within_group`；
- `zero_variance_group_ratio`；
- `success_rate_by_task_family`；
- `policy_entropy`；
- `gradient_norm`；
- `duplicate_tool_call_rate`；
- `tool_information_gain`；
- `hard_constraint_pass_rate`；
- `format_only_high_score_count`；
- `high_reward_short_trajectory_count`。

预警规则不要只看平均 Reward：

- 组内方差连续下降，同时重复调用上升：可能进入 Echo Trap；
- 策略熵突然下降、梯度范数尖峰：暂停更新并回滚最近 Checkpoint；
- 平均 Reward 上升但硬约束通过率下降：说明 Reward 被钻空子；
- 工具调用显著减少但任务完成率不升：可能学会了提前结束；
- Judge 分数上升但程序指标不升：Judge 只能保留在离线评测。

阈值不在论文中照搬，应通过 100–300 个任务的 smoke run 建立本项目分位数基线。

## 7. 必做实验矩阵

| 实验 | 目的 | 最少对照 |
|---|---|---|
| E1：轨迹质量 | 验证真实端到端轨迹是否优于拼接轨迹 | Base、SFT-stitched、SFT-E2E |
| E2：训练收益 | 验证 RL 是否真的超过模仿学习 | Base、SFT、SFT+GRPO-B0 |
| E3：Reward 结构 | 验证门禁是否优于线性相加 | naive linear、hierarchical gate |
| E4：过程信号 | 验证稀疏与受限密集信号 | outcome-only、outcome+verified process |
| E5：任务采样 | 验证模型感知采样 | random、variance-aware |
| E6：信用分配 | 验证长任务提升来自逐轮归因 | trajectory-level、turn-aware |
| E7：课程学习 | 验证宽松→严格是否帮助探索 | fixed strict、curriculum |
| E8：追问能力 | 验证用户模拟训练 | 无模拟用户、状态机用户 |
| E9：Reward 消融 | 解释六维 Reward 的实际作用 | 每次去掉一个非门禁分量 |

所有结果按短链路、中链路、长链路分别报告，不能只给总平均。只有 TravelAgent2 自己的实验结果可以用于简历中的性能数字。

## 8. 不建议现在做的事

- 不直接实现完整 AT²PO 树搜索：工程和 Rollout 成本高，先证明普通 GRPO 基线；
- 不让 LLM Judge 主导训练奖励：旅行文案“好看”容易掩盖事实和约束错误；
- 不把私有思维链作为核心训练资产：保留结构化行为与环境证据即可；
- 不在在线 API 环境直接 RL：数据漂移、费用和不可复现会破坏训练；
- 不把全失败任务反复塞入 GRPO：它们没有组内学习信号，应先回到 SFT 或降低难度；
- 不把论文在数学/搜索域的百分比包装成旅行项目结果。

## 9. 实施优先级

### P0：训练地基

1. 统一 Transition / Trajectory Event Schema；
2. 快照环境、状态化用户模拟器、Validator；
3. 300 条固定评测任务和 30 条最小 Golden 轨迹；
4. API Teacher 真实跑 Agent Loop，产出可回放 E2E 轨迹；
5. Base 与 SFT-E2E 基线。

### P1：可信 GRPO

1. 标准 GRPO-B0；
2. 层级 Reward、动态任务采样、坍塌监控；
3. SFT 与 GRPO 的同环境对照；
4. Reward Hacking 审计和消融。

### P2：研究亮点

1. turn-aware credit assignment；
2. 状态机用户训练 + LLM 用户压力测试；
3. entropy-guided branch / AT²PO 小规模对照；
4. 将训练-执行解耦封装成通用 Agentic RL Runtime。

## 10. 本地文献库与来源

论文已下载到 `docs/research/agentic_rl_papers/`，并完成 PDF 文件头、页数、文本抽取和关键方法页视觉检查。下载 PDF 用于本项目离线研究；发表状态以官方期刊/会议页面为准。

- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1 / Nature](https://www.nature.com/articles/s41586-025-09422-z)
- [ReTool](https://arxiv.org/abs/2504.11536)
- [RAGEN](https://arxiv.org/abs/2504.20073)
- [Turn-Level Reward Design](https://arxiv.org/abs/2505.11821)
- [Tool Zero / Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.485/)
- [Agent Lightning](https://arxiv.org/abs/2508.03680)
- [MUA-RL](https://arxiv.org/abs/2508.18669)
- [Demystifying Agentic RL](https://arxiv.org/abs/2510.11701)
- [AT²PO / ACL 2026](https://aclanthology.org/2026.acl-long.1106/)

## 11. 最终判断

这轮论文调研不是为了把项目堆成“论文名合集”，而是为了删掉不可靠的设计。对 TravelAgent2 最有价值的研究亮点应是：

> 在共享确定性工具与 Validator 的 Long-Horizon 出行环境中，用真实端到端轨迹完成 SFT 冷启动，再通过层级可验证 Reward、模型感知采样和逐轮信用分配优化 Agent 策略，并用状态化用户模拟器训练需求澄清与动态重规划能力。

这比“给 LangGraph 接一个本地模型，再跑一次 GRPO”完整得多，也更容易通过代码、轨迹和实验指标在面试中自证。
