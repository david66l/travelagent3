# TravelAgent2 已有数据能否复用：云端审核结果

完成日期：2026-09-05。正式结论版本 existing-data-final.v3。只审核，没有训练或模型 API 调用。

## 审核结论

**可以继续使用原生主线的 205 个训练动作；六批旧数据共 6,624 条，全部禁止原样并入 v9 训练。旧数据保留为场景迁移资产，不删除。**

| 数据 | 本次审核 train 条目 | 结论 |
|---|---:|---|
| 首轮教师 SFT（v6） | 145 | 已按 v9 的参数契约和当前判分器复审通过，可作回放训练数据 |
| 最新教师纠错（v9） | 60 | 来源、上下文、执行结果和标签审计通过，可用于纠错 SFT |
| aligned-v4 | 2,144 | 旧阶段状态；1,992 条仅提供一个工具，主要是 search_pois；禁止直接使用 |
| formal-3804 导出 | 932 | 160 条 capability_check 已移除；其余仍在旧控制器状态下；禁止直接使用 |
| stage32 replay | 1,264 | 95 条目标含模型自写 options，957 条仅提供一个工具；需来源去重与重新采集 |
| stage3-loop | 1,024 | 旧阶段与两种消息结构；958 条仅提供一个工具；不能当作当前自由决策轨迹 |
| verifier-repair-semantic | 1,080 | 包含 300 条 retry_solve、300 条模型自写 options；状态及输入协议不同 |
| tradeoff-grounding | 180 | 包含 30 条 retry_solve、120 条模型自写 options；需重建当前授权语义 |

表内是动作/训练条目，不是完整任务或独立结构。旧数据数值是此次逐行扫描得到；另外扫描 aligned 的父版本 train 2,146 条及相关 split 用于来源核对，不重复算入六批旧数据。

## 205 个动作为什么判定可复用

- 24 条原教师 train 轨迹和 12 条纠错后缀均通过当前独立判分、终局 reward gate、轨迹重放完整性、工具/求解预算核对；该复审没有重新调用模型。
- 每个训练动作及参数都对应到原 episode 的明确 step；全部 205 条输入进一步逐一核对原 model-calls.jsonl 中教师真正看到的 state，输出对应实际 action_id，没有未记录的 wrapper 修复反馈。
- 系统提示、当前单一 travel_agent 状态、状态作用域工具 schema、参数权限全部吻合。没有观察到这些 train 输入与被扫描 heldout 的精确规范化输入重合；这不是对所有潜在同源泄漏的数学保证。
- 云端实际 TRL 预处理与 collator：205 行、输入屏蔽 562682 token、监督 6464 token；最大长度 3872，6144 上限下没有截断，EOS/padding 检查通过。仅用微型 CPU 模型初始化数据处理，没有前向、反向或更新权重；真实 4B 训练启动仍复查。
- 两份原数据保持独立、原文件不改写。native-verified-candidate-rows.jsonl 是本次核对副本，不是已经完成新训练的证明。

这批数据只覆盖 9/13 个动作，没有 ask_user、propose_tradeoff、search_transport、retrieve_city_knowledge 的监督目标；虽然包含正常规划和停止，但不足以训练/验证全部能力。24 个首轮任务和 12 个新任务续跑也不能宣称 36 个独立模板，仍沿用六个训练场景组。

## 旧数据不能直接使用的实证

**协议和执行责任已经变化。** 旧输入具有 search_candidates / capability_check 等阶段节点、旧 allowed_actions 或 policy_state 包装。部分样本几乎把下一工具固定在输入中。当前模型需要在统一 loop 中选择工具和停止时机，改字段名称无法补回自主决策过程。

**目标中存在当前已禁止的动作和权限。** 六批合计 490 行以已移除动作为目标；602 行让模型编写 propose_tradeoff.options。当前 options 由 Harness 按用户授权装配；不能把旧答案原样保留，也不能把 retry_solve 简单改名后假装发生过真实执行。

**版本间重复和 split 调整是真实存在的。** 6,624 行按完整规范化提示、工具 schema 与目标去重后为 6,043 个不同条目。548 个 train 条目与扫描范围内的历史 validation/test 提示重合，其中 275 条与历史 test 重合。数字按条目计数，不能相加或当作独立泄漏任务数。

aligned-v4 的 derivation.json 明确记录 test→train 125、validation→train 138 的拆分调整；实际逐行提示/状态/ID 交叉核对结果保存在 cross-version-overlap.json。不能同时沿用所有旧版本 train 与旧 test 而宣称隔离。此发现是跨版本污染风险，不等于每一份历史数据在其自己的新拆分内都存在重叠。

**计数不等于独立规模。** verifier-repair-semantic 的派生记录标明 train 来自 60 个 source state，扩成 1,080 条。这能提供变体，但不能当作 1,080 个独立场景。candidate_episodes=3,804 的 formal 文件，真正导出的 train 只有 932 动作，且包含控制器环境标记。

## 可以回收多少任务素材

从六批旧 train 提取原请求、硬约束、软偏好，按这三者去重，并排除与已扫描 heldout 的请求签名或 scenario_id 重合项，得到 **3,612 个迁移候选请求**。全部附原 dataset / scenario_id / example_id。

这些是可以继续审查的请求素材，不是 3,612 条合格 SFT，也不是已确认的独立任务组。候选仅通过此次显式重合排除；还要审查同源模板、场景可行性、事实来源、完整工具返回与原状态可还原性。投影后的 SFT context 不含全部 provider 状态和完整 ledger，不能直接据此宣称能重放完整任务。

下一步优先从候选中按来源与失败类型分层抽查约 100 条，检查可还原率；能恢复的 train 场景在 v9 下重新执行，再由学生自主成功轨迹或 GLM-5.3-Flash 纠错产生标签。观察到的旧验证/测试派生保持隔离，不因缺数据而回收进 train。教师 API 每批上限 100 元，包含失败和重试；本轮费用为 0 次模型 API 调用。

## 审核边界与复现

- 六批旧 train 全量逐行检查；验证和测试仅在云端参与哈希、来源和重合检查，没有进入训练目标或展示其答案。
- 多消息旧样本不强行作为“首个状态—最后动作”判错；保留每条提示消息作去重，单独标注非当前消息布局。工具已移除和保护参数问题独立于该布局判断。
- v9 的 resolved_precondition_failures 是不进入模型提示的内部字段。初版只用序列化 prompt 重放防护时误报四条纠错动作，已改为参数契约检查与真实源轨迹/模型调用核对；v1/v2 记录保留，最终以 v3 为准。
- 本次没有宣称旧标签语义已逐条人工验证，没有运行新的教师或学生 rollout，没有修改原数据、训练代码或模型权重。
- 云端根目录：`/root/autodl-tmp/travelagent-existing-data-audit-20260905`。输入冻结副本 inputs/；最终结果 results-v3/；两份旧审核中间结果保留用于追溯。
- 逐行问题：results-v3/*-row-audit.jsonl；跨版本重合：cross-version-overlap.json；真实教师上下文：actual-call-conditioning-audit.json；标签检查：token-mask-audit.json。
- 复用决策：reuse-decisions.json；候选请求：migration-request-candidates.jsonl。后者全部标记 training_ready=false 与 source_group_verified=false。
