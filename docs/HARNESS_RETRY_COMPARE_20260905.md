# TravelAgent2 重试防护修复与同环境复测

记录时间：2026-09-05 19:03（Asia/Shanghai）。架构 `agent-harness-v1`，执行修订 `evidence-progress-retry-v8`。

本轮已完成检查点中的重试修复、完整 agentic + evaluation 回归、独立云端快照和基模/SFT dev 对照。v8 基模 5/8，原首轮 SFT adapter 8/8。本轮没有再次训练或更新模型权重；adapter 仍在 quarantine，未替换线上服务。

## 结果与口径

| 执行环境 | 基模任务通过 | SFT 任务通过 | 模型调用（基模 / SFT） | 前置拒绝（基模 / SFT） | 输出 token（基模 / SFT） |
|---|---:|---:|---:|---:|---:|
| v6（历史） | 5/8 | 6/8 | 65 / 38 | 19 / 4 | 1980 / 1219 |
| v7（候选重试修复） | 5/8 | 6/8 | 68 / 48 | 20 / 8 | 2019 / 1429 |
| v8（缺失研究证据重试修复） | 5/8 | 8/8 | 68 / 50 | 22 / 8 | 1977 / 1449 |

这是 8 道合成 dev 任务的一次固定 seed 推理对照，包含 4 道可行行程、2 道合理停止、2 道必要澄清；不是 8 道行程全部完成率。v8 内基模与 SFT 共用相同环境。v6→v8 的变化涉及 Harness，不能写成新一轮训练收益。开发题已用于诊断和修复，结果不代表锁定测试、真实 provider 或稳定泛化。

| v8 任务 | 基模 | SFT |
|---|---|---|
| validation-01-01 | 失败 | 通过 |
| validation-01-02 | 失败 | 通过 |
| validation-02-01 | 失败 | 通过 |
| validation-02-02 | 通过 | 通过 |
| validation-03-01 | 通过 | 通过 |
| validation-03-02 | 通过 | 通过 |
| validation-04-01 | 通过 | 通过 |
| validation-04-02 | 通过 | 通过 |

## 修复原因与范围

1. 上次未通过的 v7 测试使用了错误的成功状态名 `succeeded`。`DecisionRecord.outcome_status` 实际为 `completed`，导致成功搜索未被视为恢复进展。原端到端脚本还漏掉了题目要求的两次场馆开放时间查证；现已补齐真实工具路径，没有放宽验收。
2. 候选证据到达后，同一任务内的详情重试现在可以继续。无新进展、失败调用、无关天气查询、其他任务的成功均不会解除候选前置错误的重试防护；参数错误仍需改参数。
3. v7 真实复测保留了 5/8 对 6/8。SFT 的两道失败题已能取得详情，但又被历史 `MISSING_ARTIFACT:route_matrix` 拦截求解。v8 将同一规则扩展到 verifier 明确标记的缺失研究资料：仅相应资料生产动作的新进展能重开重试窗口。
4. 新证据之后如果同动作同参数再次失败两次，仍会拦截。模型仍自行选择工具、参数、顺序和终止；Harness 不替模型调用工具或生成答案。

本轮仅定义 `CANDIDATES_REQUIRED` 和 `RESEARCH_EVIDENCE_INSUFFICIENT` 中 `MISSING_ARTIFACT` 的恢复条件。其他错误继续沿用既有边界，没有把任何一次研究调用成功都视为可无限重试。判断使用现有的有界决策历史和进展标记。

## 验证与证据

- 本地完整 agentic + evaluation：628 passed、5 skipped。v7 前一轮为 622 passed、5 skipped，原 18:40 检查点中的 1 failed 已修复。
- 云端 v8 定向回归：`============================== 59 passed in 1.23s ==============================`。
- 修改的 policy 和新增测试均通过 Ruff lint、format 检查；policy diff 空白检查通过。
- v7、v8 各 16 条模型轨迹全部重新判分、哈希回放、预算计数核验。对已提交行程，从实际 solver 输入输出再次独立校验。
- 同一版本中两模型的基模权重、数据哈希、推理 seed、解码配置、工具/求解器/校验器源码、推理运行库均逐项匹配。与 v6 比较，已登记运行源码仅 `harness.py` 和 `policy_repair.py` 两项变化，评测脚本哈希相同。
- v7 快照保留部署时原文件；本地随后仅格式化，两份文件 AST 已核对一致。v8 使用格式化后的独立快照，所有源码归档哈希已核验。
- adapter SHA-256：`cab21208ad6df31e4a077b6437c3f9c22e21860ed18e66afff14f8de12f18c91`。
- dev cases SHA-256：`a9f3edf026f8c4302fbe63f5e84d938b5e27fed9a98ab08ba657988c7b913a3f`。
- v8 执行源码归档 SHA-256：`8fa7e0bd72c3f523b9b98df272bb5b744c70c4f9b12a7f46b37181c8dbb691d8`。
- v8 原始轨迹与源码证据包 SHA-256：`965aaf3be2537f40353ae376c8a6e946debd437c649b9119dae95d2bcb72be41`。

云端根目录 `/root/autodl-tmp/harness-sft-first-20260905/`；源码分别为 `runtime-v7`、`runtime-v8`，结果为 `qwen-dev-v7`、`sft-dev-v7`、`qwen-dev-v8`、`sft-dev-v8`。原 v6 代码仓、训练 adapter、原始结果均保留。locked test 未运行，未进入训练目标。证据包不含密钥、基础模型权重或优化器状态。

## 下一步

已整理 `student-error-inventory.json`，逐题保留真实决策错误、最终结果和 episode 哈希，仅作 dev 错误分析。即使任务最终通过，也应继续统计先取详情但缺候选、过早求解、重复获取相同资料等无效步骤。

下一阶段从独立的新 train 任务采集 SFT 自主运行轨迹，按上述错误类型让教师从学生实际失败状态续跑并执行验证；不要把这 8 道 dev 的状态或答案直接送入训练。随后扩大场景组合，并做等目标 token 的普通增量示范与定向纠错对照。当前没有完成 200–500 条扩充、第二轮 SFT 或 Agentic RL；这些是后续实验，不能作为已完成成果。

## 本地复现入口

项目根目录运行：

```powershell
.\backend\.venv\Scripts\python.exe -m pytest backend/tests/unit/agentic backend/tests/unit/evaluation -q
```

证据包包含云端执行源码、两轮模型完整轨迹、测试日志、审计结果和本轮部署/复测/审计脚本。旧检查点中 `work/sft-first` 的实际位置是 `C:/Users/Rain/Documents/Codex/2026-09-05/e-travelagent/work/sft-first`，不在项目根目录下。
