# TravelAgent2：100 条旧任务迁移候选的云端审核

完成日期：2026-09-05。最终结果目录 results-v3。全程云端数据扫描与独立校验，没有模型 API、推理、训练或新 rollout；本地只进行脚本/文档编辑和 SSH/SCP 同步。

## 结论

100 条中，**62 条能定位请求一致的完整旧环境快照，可进入迁移重建；5 条同源重合需要隔离；10 条同 ID 原请求不一致；23 条在本轮扫描来源中未找到精确对应场景。当前新增可直接训练的数据为 0 条。**

62 条对应 **57 个不同的已记录源任务组**，已形成按组去重的重建队列。这是来源标识去重，不代表 57 个语义独立模板，也不是 57 条已通过 v9 的训练任务。未找到来源不等于文件永久丢失。

| 主来源 | 抽样数 | 可进入重建 | 同源隔离 | 未解决来源问题 |
|---|---:|---:|---:|---:|
| tradeoff-grounding | 17 | 17 | 0 | 0 |
| verifier-repair-semantic | 17 | 12 | 5 | 0 |
| stage3-loop | 17 | 6 | 0 | 11 |
| stage32 replay | 17 | 11 | 0 | 6 |
| formal3804 | 16 | 16 | 0 | 0 |
| aligned-v4 | 16 | 0 | 0 | 16 |

## 抽样和追溯

从上一轮 3,612 个候选请求中，固定 seed=20260905，以六个主来源轮转，再按旧目标动作与可见 failure_summary 分层，选取 100 个不重复请求，覆盖 23 个层。多来源请求按固定优先级归属一个主来源，仍保留全部来源引用。no_visible_failure 仅指该字段中没有错误，不代表没有验证失败或业务故障。

本轮是偏向问题发现的均衡诊断抽样，不能把 62/100 外推为全池可用率。查阅五组源语料：curriculum-zh-v1、stage3-decision-loop-curriculum-v3、necessary-abort-repair-v2/corpus、verifier-repair-grpo-semantic-v2、verifier-repair-grpo-rl-challenge-v5。逐文件 SHA-256 和行号保存。历史 SFT validation/test 只用于来源匹配；未把留出答案用于训练、模型提示或新任务标签。

67 条找到精确 task_id 与 user_request 匹配，其中包含需隔离的 5 条；全部通过当前 EnvironmentTask / EnvironmentSnapshot 结构检查。10 条虽然 ID 一样，但原请求变化，禁止静默替换请求以伪造恢复成功。23 条主要涉及 priority-search、adaptive-recovery 等派生场景，仍需追溯其专属源快照。

## 比上一轮新增的同源风险

此前排除了相同请求签名或 scenario_id，本轮继续按源 task_id 和 source_content_hash 检查。样本中发现 5 条与历史留出派生同源，已隔离。将本轮已发现的来源关系用于整个候选池，标出 **365 条已观察到的同源重合风险请求**，写入 pool-known-group-quarantine.jsonl。

剩余 3,247 条仍只是未触发本次已知关系的候选，不是全部审核通过。来源追溯尚不完整；本结论涉及跨版本使用时的隔离风险，不能解释为每一份原始数据集都在自己的拆分内泄漏。旧数据原文件保持不变。

## 工具事实与场景可行性

- 67 条匹配源快照均有结构有效的路线矩阵；仅 4 条覆盖全部搜索候选，37 条覆盖全部非餐厅候选。矩阵可能只针对已选子集，不能将其直接判成原数据错误，但当前自由选择 POI 后必须检查对应路线覆盖。
- 67 条中，63 条搜索候选坐标齐全，33 条搜索候选的 open_time/close_time 齐全；其他字段可能在详情工具中，仍需迁移时按实体合并并核对。不能把空字段默认成已确认可营业。
- 对旧快照保存的行程重新运行当前 ItineraryValidator：66 条至少一个保存行程通过基础硬约束检查，1 条未通过。此处没有重新求解，也未证明自然语言中的全部约束、不可行原因、实时事实或故障机制正确；不能当作 Agent 成功率。
- 这些快照包含 built_in / fallback 等合成或兜底事实，以及预置 solve_itinerary / validate_itinerary 返回。只能作为离线环境素材，不能声称是真实实时旅行数据。迁移时必须移除旧求解、旧校验结论作为成功依据，改用当前真实求解器和独立判分。
- 请求、日期/天数和重叠约束字段做了核对；源 slots 缺少的候选约束字段另行标记，未自动丢弃。未声称当前协议已能无损承载所有旧场景。

## 已准备好的下一阶段输入

source-group-deduplicated-rebuild-queue.jsonl：57 个源任务组的重建队列，均带来源行号/哈希、请求及约束，training_ready=false、v9_rollout_ready=false。详细核对仍覆盖全部 100 条；原始 train 快照单独留存在云端，不把 hidden_test_facts 或旧目标动作放入模型上下文。

下一阶段先从队列中选 12 个覆盖不同来源/故障的组，构建 v9 环境，补齐事实和路线覆盖，验证真实求解/独立判分、输入权限和故障重放，再运行学生/教师采集。教师使用 GLM-5.3-Flash；每批 API 含失败重试上限 100 元，付费采集前落实价格核对和成本硬停止。当前未产生任何教师 API 费用，也没有新租算力。

原生 205 个已审核动作的可复用结论不变；本轮未进行第二轮 SFT。

## 复现与边界

云端根目录：/root/autodl-tmp/travelagent-migration-pilot-audit-20260905。sample-manifest.jsonl 为固定抽样名单，sample-audit.jsonl 为逐条结果，source-hashes.json 为来源版本，recovered-train-sources.jsonl 为 train 原始快照证据，summary.json 为摘要。

初版在混合格式的 stage2 SFT 文件上停止，保留 results-v1；v2 完成初审；v3 改用真实场景源、增加 necessary-abort 来源、矩阵实体别名核对和全池已知同源隔离。本报告以 v3 为准。原始输入未修改。
