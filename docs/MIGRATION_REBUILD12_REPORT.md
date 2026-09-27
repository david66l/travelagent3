# TravelAgent2：12 组旧场景迁移与学生自主运行结果

最终环境：evidence-progress-retry-v9 + legacy-provider-facts-migration.v2。执行记录目录为 20260905；本报告在用户暂停后恢复执行时完成。

## 已完成

从此前可进入重建的 62 条候选（57 个来源组）中按五类已恢复来源及不同动作/故障选取 12 个不重复源任务组。初选只按来源轮转，未执行前改为兼顾动作/故障分层；两个选择文件均留存，正式名单 selected12-v2.json。

**5 组通过真实路线矩阵、TravelVRPSolver、ItineraryValidator、任务判分、终局 reward gate、轨迹完整性和预期故障重放检查；7 组保留待补事实。** 这是脚本驱动的环境可执行性检查，不是模型成绩。

随后使用原 Qwen3-4B + 首轮 SFT adapter 自主运行通过检查的 5 组：**5/5 完成、31 次模型调用、31 个真实决策后状态核验通过、0 次解析/参数错误。** 模型没有脚本前缀或教师纠错提示，权重未更新。

| 场景 | 环境结果 | 主来源 |
|---|---|---|
| migration-01 | 通过真实执行检查 | formal3804 |
| migration-02 | 通过真实执行检查 | formal3804 |
| migration-03 | 通过真实执行检查 | formal3804 |
| migration-04 | 待补事实/约束 | repair_semantic |
| migration-05 | 待补事实/约束 | repair_semantic |
| migration-06 | 待补事实/约束 | repair_semantic |
| migration-07 | 待补事实/约束 | replay_stage32 |
| migration-08 | 待补事实/约束 | replay_stage32 |
| migration-09 | 通过真实执行检查 | stage3_loop |
| migration-10 | 通过真实执行检查 | stage3_loop |
| migration-11 | 待补事实/约束 | tradeoff |
| migration-12 | 待补事实/约束 | tradeoff |

## 为什么暂不作为新增高质量 SFT

1. 5 个成功场景的 preference_match 均为 0。旧 POI 标签稀疏，部分语料的未约束查询返回固定集合；通过硬约束不代表满足用户艺术、建筑、摄影、美食等偏好。不能为了让分数好看而把用户兴趣复制到 POI 标签。
2. 3 组配置搜索故障，学生仅在 migration-02 实际触发并恢复 TOOL_TIMEOUT；migration-09 与 migration-10 的查询没有命中原故障匹配条件。环境脚本能触发不等于学生测量时发生过，不能声称三类恢复能力均已验证。
3. 7 组包括关闭/预警、住宿成本、孕妇低疲劳、固定活动时窗和返程等场景。决定性事实有的只存在于旧求解/校验答案中，不能直接转换成当前可信工具证据。另有源 missing_slots 与用户明确预算的矛盾，需要单独核实。

当前**新增可直接训练样本 0 条**。31 次决策含一次失败工具动作，也未导出/清洗/做 token 标签审计，不能称为 31 条合格 SFT。原有 205 动作可复用结论保持不变。

## 本轮代码与核验

- evaluation/migration_fixture.py：只迁移真实存在于源快照的 provider 字段；保留缺失项作为阻断，不补造营业时间、价格、坐标、时长或餐厅。旧 solver/verifier 答案、私有 oracle、教师动作前缀不进入新 case。
- 保留原 SnapshotToolExecutor 的 exact / context_tolerant_keywords 语义，包括仅忽略已声明的城市、日期等词、无 expected_arguments 时的原始无约束匹配。未通过放宽故障判定来制造成功。
- 工具注册表关闭在线回退；路线矩阵、求解与校验沿用当前真实实现。批量 guard 一次处理整批，保留预算限制和重复调用检查；工具故障输出符合 observation.error.code 契约。
- scripts/collect_harness_migration.py：记录学生实际上下文/动作、工具记录与决策后状态，额外保留 provider counters 和 seen_fault_indices。后续恢复须用此迁移 backend；原纠错驱动不能不经适配直接使用这些状态。
- 当前冻结源码内相关回归 **70 passed**，包括 7 项新增迁移单测。这不是本轮跑过整个大仓库全部测试的声明。
- 每条学生轨迹通过源动作映射、状态 hash、预算计数、独立保存行程校验；所有证据、失败和早期版本保留云端。

## 两轮学生结果如何解释

student-v1 使用迁移 provider v1，结果 3/5，该轮是 27 次调用、22 个状态，4 次解析/参数错误。检查发现迁移器丢失原查询匹配契约，使某些合法查询变为空结果，部分故障也被错误跳过。

修正契约后重新运行 student-v2，结果 5/5、31 次调用/状态、0 次解析错误。**这是环境修正后的重新测量，不是模型训练收益，也不是同环境 A/B 对照。** 模型权重和解码未改变。此前 preflight-v1 的 3 项故障误报来自读取 observation.error_code，而正确字段为 observation.error.code；该统计已修正。

## 下一步

这 5 组保留为离线流程回归材料，7 组形成 blocked-source-work-queue.json，逐项列出必须补齐的证据。后续优先建设带完整属性、兴趣依据和真实失败条件的任务事实，再开展学生失败采集及 GLM-5.3-Flash 纠错。没有必要现在付费扩写同类弱事实模板。

教师每批 API 含失败/重试上限 100 元的约束不变；本轮 **0 次教师 API、0 次新训练、无新增租机**。全部计算、模型推理、测试、审计与压缩在现有云主机进行，本地仅轻量编辑与小报告同步。

云端根目录：/root/autodl-tmp/travelagent-migration-rebuild12-20260905。正式 preflight-v3、student-v2、student-audit-v2、runtime-v3；本报告及来源哈希位于 final-v1。证据包只在云端，未将模型权重复制到本地。
