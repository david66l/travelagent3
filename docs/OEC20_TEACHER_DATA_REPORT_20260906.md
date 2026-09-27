# TravelAgent2 具名餐厅与教师配对 SFT 数据先导

日期：2026-09-06。冻结环境 runtime-v2；场景 travel-named-dining20.v1；校验器 travel-validator.v3-named-dining。

具名餐厅已进入实际求解，真实学生检查点恢复已实现；20 个任务各进行一次初态教师示范和一次学生状态接手，共 40 次采集尝试，成功与失败均已归档。全部计算、测试、API 采集、审计和压缩在既有云主机运行。本地仅源码/文档编辑和 SSH/SCP。没有新增租机、没有更新模型权重。

## 结果

| 组别 | 轨迹 | 通过任务判据 | 质量候选 | 标签审核接收轨迹 | 导出动作 |
|---|---:|---:|---:|---:|---:|
| 既有学生重新自主运行 | 20 | 7 | 5 | 不作本批教师标签 | 0 |
| GLM 从初态完整示范 | 20 | 16 | 14 | 11 | 71 |
| GLM 从学生状态接手 | 20 | 17 | 11 | 10 | 54 |

按预期终态分别统计：{"demo": {"grounded_stop": {"trajectories": 4, "task_passed": 4, "quality_candidates": 4}, "plan": {"trajectories": 16, "task_passed": 12, "quality_candidates": 10}}, "correction": {"grounded_stop": {"trajectories": 4, "task_passed": 4, "quality_candidates": 4}, "plan": {"trajectories": 16, "task_passed": 13, "quality_candidates": 7}}}。任务判据包含有依据的替代方案/终止，不把它们称为生成有效行程。

本批导出 125 个监督动作，通过实际 TRL 预处理和 collator 标签检查；输入上下文 366477 token 全部屏蔽、监督 4763 token，最长 3927、截断 0。仅在云端初始化微型 CPU 模型检查数据处理，未做 forward/backward；实际 4B 训练启动仍须复查。学生错误前缀只作上下文，监督标签不含学生动作、私有 reasoning 或控制器代答。

接收轨迹的终态类别：{"validated_plan": 13, "safe_termination": 8}。25 条质量候选中另有 4 条因缺失工具 observation 或重复同参数调用过多而被基础 SFT 审核排除；本轮没有事后放宽规则来提高接收率。完整的 SFT 拒绝统计：{"PAIRED_INDEPENDENT_QUALITY_REJECTED": 15, "L2_TOOL_OBSERVATION_MISSING": 11, "L3_UNUSABLE_OUTCOME": 7, "L2_EXCESSIVE_IDENTICAL_RETRIES": 5}。

接手集合包含 15 个观察到错误的检查点和 5 个正常学生前缀检查点，分组结果：{"observed_error_continuation": {"trajectories": 15, "task_passed": 13, "quality_candidates": 7, "accepted_episodes": 6, "original_student_eventually_passed": 3}, "student_prefix_expert_takeover": {"trajectories": 5, "task_passed": 4, "quality_candidates": 4, "accepted_episodes": 4, "original_student_eventually_passed": 4}}。检查点出现错误不等于原学生整条任务最终失败；不能把全部 20 条都称为错误恢复，接手成功率也不等于学生模型提升。完整示范从初态运行，接手轨迹继承学生已有证据及已消耗的工具、步骤、solver/token 预算，仅扩大 API 等待时间；两者通过率不是等起点 A/B。

## 环境和审计

- 具名模式显式开启 `require_named_restaurants`，新 20 题全部开启。餐厅候选及详情进入路线矩阵，CP-SAT 按天/午晚餐选择真实实体，保留营业时间、人均费、源 ID 和相邻交通时间/费用；不输出通用餐食代替成功。旧默认通用用餐模式保留兼容，本次未证明所有入口均自动切换到具名模式。
- 校验器独立检查每餐实体、时段、时长、餐费、交通边和时间间隔，并用原始合成目录复核硬约束。具名源路线通过核对的教师轨迹 28。这仍是可复现合成事实，不是线上真实商家实时信息验证。
- 新环境脚本可达性预检 20/20、相关云端回归 147 passed。学生 201 个状态和全部 237 次实际输入 token 哈希核对，差异 0；学生净采集 67.73 秒，16 路 vLLM，原 adapter 未更新。
- 教师 311 个执行状态及 20 个学生检查点来源核对，原始 API 的系统提示、工具 schema、模型可见状态和目标参数逐项匹配导出标签。教师额外收到固定的“一次只调用一个工具”格式提醒，经白名单逐字核对，不含任务事实或下一动作提示；该冗余提醒不进入学生提示。因此是状态/工具契约一致，不是两个模型完整请求逐字相同。provider 故障计数、已触发 fault 状态和工具预算均恢复，未重放学生工具获取额外机会。
- 采集结束重启已确认跳过全部完成任务，无重复付费调用。API 异常和模型任务失败保留在原始证据中，未自动重新抽取成功轨迹。
- 当前 20 个任务源于 18 个既有 train 父模板，含两个组合变体，全部 train。按来源组管理，不能随机按行划 dev/test；未运行独立留出评估。
- 新环境学生 7/20 不能与旧通用餐食环境 976/1000 直接比较。原 1,000 条旧轨迹没有改写复用为新环境轨迹。
- 拒绝原因：{"PREFERENCE_MATCH_BELOW_0_5": 10, "TASK_CRITERION_FAILED": 7, "REWARD_GATE_FAILED": 7, "TEACHER_PROVIDER_ERROR": 4, "ORIGINAL_CATALOG_CONSTRAINT_CHECK_FAILED": 4, "CONFIGURED_FAULT_NOT_EXERCISED": 1, "RAW_POLICY_ERROR": 3}。导出动作分布：{"search_pois": 19, "get_poi_detail": 17, "search_current_info": 27, "get_route_matrix": 13, "solve_itinerary": 13, "validate_itinerary": 13, "finish": 13, "get_weather": 2, "abort": 6, "propose_tradeoff": 2}。精确重复导出 payload：0；不代表不存在模板相关性。
- 采集中已观察到“任务完成但偏好 1/3”的接手轨迹：三个景点只有一个符合艺术或科学兴趣。餐厅不计入偏好评分分母。现有 solver 以访问数量为主要目标，教师还受已继承候选和剩余预算限制；下一轮应验证能否通过合法动作修复，不能将此类拒绝一概解释为教师决策错误。筛选阈值本批不变。

## API 费用和速度

GLM-5.3-Flash 国内端点，4 条轨迹并发，每条内部按工具结果顺序运行。共新增 319 次请求，本批保守计提 5.932369 元（已知用量计提 1.623374 元＋未知用量预留 4.308995 元）；沿用原 100 元总账，累计 7.381303 元，剩余 92.618697 元。本批未知用量请求 5 次，全部保留最大预算预留，不按零费用处理。按原价输入 0.8/输出 2.8 元每百万 token 计提，未扣缓存优惠；不是供应商最终账单，既有云算力费另计。

本批未知用量包括 4 次供应商读取超时，以及最后一条组合接手任务触发 episode 等待上限后的在途请求取消。后一项是本地队列预算取消，不冒充供应商返回的用量。

教师首个新请求至最后一条轨迹落盘共 50.5 分钟，包含异常后暂停、排空在途队列和恢复等待，不含前置学生采集及后置标签审核。单轨迹墙钟中位数 221.3 秒，最大 600.4 秒；API 单请求中位数 17.9 秒，p95 84.7 秒。不可与学生 vLLM 吞吐互换。此次保持统一教师参数，未中途降低 reasoning 或修改并发以混合不同生成条件。下一轮优先比较 max/high 并改进队列对单次异常的处理；GLM-5.3-Flash 支持 max/high/low，不能关闭思考。[官方参数说明](https://docs.bigmodel.cn/cn/guide/capabilities/thinking)。具体对照设计见项目 `docs/TEACHER_SPEED_PLAN_20260906.md`，尚未实测新配置提速倍数。

## 下一阶段

先补与本批 18 父模板隔离的开发任务，并预注册同环境对照：既有学生、纯完整示范 SFT、示范＋真实学生状态接手 SFT。两类教师数据均通过标签审核的共同任务目前 5 个；后续要控制任务来源覆盖，不能把不同任务筛选的差异当成纠错效果。两个训练组从同一权重出发，匹配监督 token 和训练预算，保留共享基础能力样本，至少分开报告初态成功率、错误恢复、硬约束、工具/API 次数和延迟。当前数据是方法先导，不能作为规模验收或用教师通过率宣称训练收益；大规模扩充应由独立 dev 的差距和数据来源覆盖决定。尚未开启本轮新训练或 RL。

## 证据入口

云端根目录 `/root/autodl-tmp/travelagent-oec20-20260906`；冻结源码 `runtime-v2`，预检 `preflight-v1`，学生 `student-v1`，教师 `teacher-v1`，审计 `student-audit` / `teacher-audit-v1`，监督标签 `teacher-audit-v1/sft/train.jsonl`，复现 helper `repro`，本报告与摘要 `final-v1`。原始教师回复及大型证据留云端，dotenv 和缓存排除归档。最终 GPU 状态：0 %, 1 MiB。
