# Step 4：H-006 数据准备与质量审计

更新日期：2026-09-04。状态：进行中，未开训，禁止把本页的候选数量写成模型提升。

最新续记：见 `STEP-04-H006-CORRECTIVE-PREP.md`。已新增 240 个 train 来源的 420 条
显式规则教师纠错候选，另留 48 个 shadow 来源；新增 source 等权 loss 实现及测试。
最终混合、训练接口/预检和冻结 readiness 仍未完成，本次没有开训。

## 1. 白话结论

旧数据没有丢，但旧 H-001 的 300 条训练样本只来自 40 个 source states，且这 40 个
source 全部与禁止回流集合重合。保留旧 checkpoint 和历史证据，不直接回灌旧答案。

昨晚已经积累了一批有用的高难答案。不过普通工具链和重试解释仍有问题，原普通题库中
还有材料不足的题。必须先把题目质量、模型错误、数据导出错误分开处理，不能凑数开训。

此前 Step 3 的高分来自 verifier-repair 决策状态；它不能证明完整旅行工具链已经跑通。

## 2. 冻结 donor 的已完成结果

所有模型采样均为每 source 4 次，temperature 0.8，max_new_tokens 192，4-bit 加载。
checkpoint、tokenizer 和代码快照详见各运行的原始 `report.json`。

| 数据池 | 检查的 source 数 | 原始合格 source | 额外训练质量检查后 | 用途 |
|---|---:|---:|---:|---|
| H004 hard wave 1 | 180 | 101 | 101 | 训练候选 |
| H004 hard wave 2 | 60 | 54 | 54 | 训练候选 |
| H004 hard validation | 54 | 35 | 35 | train-shadow，不混入 train |
| H001 boundary train | 90 | 53 | 52 | 训练候选 |

两批 hard training 的 source 交集已检查为 0，合计 155；包含 abort 84、tradeoff 71，
没有合格 retry training source。H001 boundary 保留 abort 22、tradeoff 30，retry 0。

队列在 H001 boundary 只有 53/90 个 source 成功时退出，未达到原定 75 个；没有降低该门。
后续 boundary validation、普通任务大批采样和 GPU training 均没有自动启动。

### 重试失败归因

H001 的 30 道 retry 题共 120 次采样：64 次选中了 retry 并执行成功，但最终仅 1 次通过
原始全部语义门。61 次包含工具 schema 描述回声，例如直接输出英文 `Grounded failure detail`。
这不是具体旅行冲突解释。

原始通过的那 1 次又同时声称“无法修复”“需用户调整”并要求重算，存在矛盾，被额外的
train-only 保守筛选隔离。因此可入训练候选的 retry 数为 0，原始评测分数不追溯改写。

## 3. 普通题源头缺陷及修复

旧 `easy-anchor-candidates` 中，180 个 train source 有 149 个景点详情响应数量不足；
36 个 shadow source 有 29 个不足。典型一天行程仅有 2 份详情，但生产研究门至少要求 3 份。

旧普通 smoke 的 3 个 source、12 次 H001 输出均在路线工具上填入了不允许的
`candidate_poi_ids`，被 parser 拦截。随后 CPU 规则教师在同样 3 题上也因
`RESEARCH_EVIDENCE_INSUFFICIENT` 无法走到终局。因此这批 smoke 不能作为完整任务能力的
有效估计：同时存在模型参数错误和题目本身不可完成两个问题。

已修复 `build_h006_easy_anchor_sources.py`：

- 从生产 `infer_research_requirements` 读取材料要求，不降低 verifier 门；
- 排除材料库存不足的 source；
- 每个选中 source 必须先由规则教师通过真实 ReAct loop、终局 hard verifier 和 replay；
- 规则教师只证明题目可解，其轨迹单独保存，绝不算作 H001 模型自己做对；
- 拒绝覆盖已有输出目录，旧候选及失败证据全部保留。

新建 `fresh-easy-base-v2` 使用以前未使用的 seeds 11000–12599。
`easy-anchor-candidates-v2` 含 180 train + 36 shadow，每个 train 任务族 30 个。
216/216 个 source 已由规则教师完整跑通。

新的已知禁止回流审计：source ID、source hash、tool snapshot 均 0 交叉，prompt 最大
相似度 0.0326087；独立 source 数统计为 180/36。注意这不替代尚未完成的 protected registry
检查，也没有证明跨模板泛化。

新普通题上的 H001 小检查：`h001-easy-smoke-valid-sources-v2`，仍只检查前 3 个 shadow source，
每 source 4 次、同一 seed 20260925。结果为 1/12 次成功、1/3 个 source 至少成功一次，
另外 11 次均为 `UNEXPECTED_ARGUMENT:candidate_poi_ids`。没有达到原定至少 2 个成功 source
的小检查门，故不启动普通题大批采样。这只是三个开发 source 的诊断，不是全体任务成功率。
该唯一成功轨迹属于 shadow，不能混入 train；其余 teacher 可解性证明也不能冒称模型输出。

## 4. 数据导出与开训开关修复

`build_h006_sft_mix.py` 现在只导出实际有 checkpoint inference 记录的 sampled step。
以前 transported decision episode 中的 teacher prefix 也可能被 easy/recovery 分支导出；
现在按 `actions[].step_index` 精确筛选，并核对动作、原始参数、模型身份和 token 记录。

retry 的本步状态是 pending，不能伪造为成功后再导出；仅当完整终局 verifier 通过时才由
独立决策导出逻辑生成样本，保留原始轨迹不变。target 使用模型原始参数，不包含系统填入的
strategy、options 或固定 connector。

对现有合格记录实际执行了导出核验：285 + 153 + 97 + 166 = 701 个 sampled decisions，
均只导出 1 个实际模型步骤，原始 target 参数逐条相等。此数包含 shadow 与重复采样，不能
声称为 701 个独立训练 source。

混合脚本即使配比通过也不再直接授予 training_authorized。source 等权选样不等于训练
sampler 等权；当每 source 有 1–2 条样本时，实际 trainer 的 source 权重仍需实现/检查。

## 5. 当前开训阻塞项

- 普通完整工具链的 H001 成功 anchors 尚未证明足够；
- retry 的可用正确解释不足，不能用矛盾答案或 schema 回声凑数；
- 最终 40/40/20 的 source/token 双配比尚未完成；
- source 等权训练、0 truncation、adapter/optimizer/步数和代码快照 preflight 未完成；
- promotion 功效分析与 protected hash-only lineage registry 未完成；
- promotion/sealed payload 保持未打开，不能为开训便利而绕过这些门。

下一步顺序：先读可解普通题上的小检查；如果仍是同类工具合同错误，停止扩大 H001 采样，
准备来源可追溯的基础工具合同修复示范，并明确区分规则教师与模型 donor。任何数据来源或
训练阶段变更都要单独记录，不能把规则教师答案冒称 H001 成功 anchors，也不能静默改变
冻结的晋升门槛。

## 6. 证据路径

统一 artifact 根目录：`artifacts/native-react-posttraining/step4-h006-data-20260903/`。

- `donor-target-quality-20260904-v2.json`：原始 donor 与额外隔离结果，包含输入和审计器 hash；
- `hard-h004-wave1/`、`hard-h004-wave2/`、`hard-h004-validation/`、`h001-boundary-train/`：原始报告和全部 rollout；
- `h001-easy-smoke/`：旧题库 smoke 失败证据；
- `easy-anchor-candidates-v2/source_feasibility.jsonl`：216 个规则教师可解性证明；
- `easy-v2-lineage-audit-20260904.json`：已知禁止回流审计；
- `valid-easy-smoke-target-quality-20260904.json`：可解普通题的小检查质量报告；
- `recovery-sft/`：此前生成的 60 train + 12 shadow 可见工具失败恢复样本。

本次只修改数据准备/审计脚本及其测试；没有修改生产 parser、工具权限、reason assembler
或旧冻结评测指标，没有新增模型权重。

验证结果：全部 H006 数据相关测试与 SFT dataset 测试合计 41 passed；本次修改文件的
Ruff 检查通过。两次 GPU 小检查均已自然结束，旧队列保持停止，没有后台训练。

最终裁决：Step 4 继续准备；批量 H001 ordinary donor mining 暂不放行；H006 training
继续 Quarantine。下一次修复示范只能来自独立 train source，不能直接回灌这批 shadow
诊断输出，也不能通过允许模型写 `candidate_poi_ids` 绕过工具权限合同。
