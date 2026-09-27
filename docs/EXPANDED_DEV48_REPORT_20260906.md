# TravelAgent2：48题扩展验证

2026-09-06。原学生和两组固定第3轮SFT权重完成同批48题评估，全部计算、审计与归档在既有云主机。新增训练0次、教师API0次，原权重未改，未自动上线。

| 指标 | 原学生 | 仅示范SFT | 示范＋接手SFT |
|---|---:|---:|---:|
|质量通过 / 48|12|23|29|
|任务通过 / 48|12|23|31|
|普通规划 / 24|2|2|13|
|工具恢复 / 12|0|10|5|
|约束终态 / 12|10|11|11|
|原始策略错误次数|17|19|17|

本轮覆盖16类相关组合，每类3个城市/日期变体。普通规划24题、超时恢复12题、约束终态12题；重点包括两日文化/科学/自然/混合行程、三日混合行程、上午开放、闭馆与高价可选项、禁去场馆，以及搜索/详情/路线/天气超时。48题均在模型评估前通过固定策略的真实工具与求解器预检，恢复场景均实际触发配置故障。固定策略轨迹不作训练标签。

对旧20题训练、新96题训练和原12题dev进行ID、请求、来源组、实体ID/名称/URL去重，无重叠。新数据保持validation，train/test分区拒绝校验通过，旧分区3项回归测试通过。仅新增数据生成器和采集器版本路由；所有工具、守卫、求解器与质量标准保持前轮冻结源码。评测计划及案例SHA在三个模型首次调用前冻结。

三组同环境、seed42、greedy、vLLM并发16/batch4、384输出token、零自动重试，单任务时限120秒。所有任务一次尝试，失败保留；恢复启动检查全部跳过已完成任务，无重复推理。三个模型逐题初始输入token哈希一致；完成生成的实际输入token、已提交状态、轨迹回放、证据文件SHA及成功规划的原始场馆硬约束审核通过。

单列取消调用：原学生/仅示范/混合分别0/4/0次，均为120秒截止后取消，终态未提交批次中的CancelledError与调用上下文一致，无返回动作，任务保留失败。生成可能已经开始，但实际输入token哈希未记录，不能宣称这些取消调用的真实输入也完成了哈希验证；没有补造哈希或重跑以消除失败。

逐题配对差异：`{"baseline_vs_demo_only": {"both_pass": 10, "only_left": 2, "only_right": 13, "both_fail": 23}, "baseline_vs_demo_plus_correction": {"both_pass": 12, "only_left": 0, "only_right": 17, "both_fail": 19}, "demo_only_vs_demo_plus_correction": {"both_pass": 18, "only_left": 5, "only_right": 11, "both_fail": 14}}`。

这48题是针对已知能力薄弱处扩大覆盖的合成开发集，不是锁定的独立最终测试。16组之间仍共享工具技能、部分组合结构与训练相关，3个变体不是3种独立能力；单一训练种子不足以证明稳定优势。原12题不覆盖、不合并包装成同分布60题。这一轮结果优先用于确定新训练来源的技能需求，禁止直接把dev事实、状态或预检动作转换成训练标签。

## 失败证据

baseline: {"failed_cases": 36, "failed_without_solver": 35, "failure_step_errors": {"RESEARCH_EVIDENCE_INSUFFICIENT": 60, "REPEATED_NO_PROGRESS_ACTION": 131, "TOOL_TIMEOUT": 9, "CANDIDATES_REQUIRED": 3, "VALIDATION_NOT_PASSED": 1}}

demo_only: {"failed_cases": 25, "failed_without_solver": 24, "failure_step_errors": {"RESEARCH_EVIDENCE_INSUFFICIENT": 21, "REPEATED_NO_PROGRESS_ACTION": 30, "TOOL_TIMEOUT": 2, "TERMINAL_EVIDENCE_REFERENCE_INVALID": 5}}

demo_plus_correction: {"failed_cases": 19, "failed_without_solver": 17, "failure_step_errors": {"REPEATED_NO_PROGRESS_ACTION": 23, "RESEARCH_EVIDENCE_INSUFFICIENT": 14, "TOOL_TIMEOUT": 7, "ORIGIN_REQUIRED": 2, "TERMINAL_EVIDENCE_REFERENCE_INVALID": 2}}


后续应根据分组表现增补独立多日证据获取和工具恢复训练来源，先保证共同配对规划样本覆盖，再固定更大规模、多随机种子训练及锁定测试方案。当前不宣称达到大厂社招项目的正式规模验收，也不因小批得分自动提高纠错比例或替换模型。

原API总账仍681次，保守计提8.379063/100元、剩91.620937元，算力另计。最终GPU `0 %, 1 MiB`。云端证据根 `/root/autodl-tmp/travelagent-expanded-dev48-20260906`，包括冻结runtime-v1、preflight-v1、evaluation-plan.json、eval-*、audit-*、repro及final-v1；仅小型报告同步本地。

归档SHA `a6a38df538f335c32222c5679dcb59025aa8f78938bd622cd96d8e2f0db785d1`，2667文件逐项核验，含144个新模型episode。
