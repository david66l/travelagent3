# H-001 canonical rationale SFT 重训（预登记假设卡）

- **日期**: 2026-09-01
- **状态**: 进行中（诊断阶段）
- **变更**: 教师 reason 前缀从 `sha256(action:evidence) % 8` 随机选择改为**每动作唯一 canonical 连接词**（`reason_quality.py::canonical_repair_rationale`）
- **设计降级说明**: 原计划 (action × violation_code) 18 槽查表，降级为每动作 1 条 —— 5 个调用点（warmstart/DPO builder/DPO trainer 校验/GRPO 探针/审计）都用此函数做相等校验，按码分叉要求全部站点同步取码，一致性风险大；动作级唯一已满足可学习性（LIMA：一致性 > 多样性），多样性由 evidence 承载

## 假设

教师 reason 的哈希随机前缀是 rationale 段 NLL 恒高（实测 4.51 vs 证据段 0.057）的主因；
改为确定性连接词后，前缀段变为可学习分布。

## 预期指标（vs v11 基线）

| 指标 | v11 基线 | 预期 |
|---|---|---|
| rationale_prefix 分段 NLL（零优化器诊断, 12 样本） | 4.510 | **< 0.5**（kill 门：> 2.0 即证伪停手） |
| reason_action_rationale_match 失败（internal-dev 80 题） | 56/60 (tradeoff) | < 10 |
| full_success_rate | 0.45 | ≥ 0.60 |
| action_accuracy | 0.81 | 不降（≥ 0.81） |

## 变量控制

只改前缀选择函数。训练超参全同 v11：lr 1e-5, 2 epoch, QLoRA r16, seed 20260923。

## 验证顺序（先诊断后训练）

1. 零优化器诊断：v11 checkpoint 对**新语料**算分段 NLL —— 不过 kill 门不开 GPU
2. 过线 → SFT 重训 → internal-dev 80 题评测 → 对照 v11 四项指标
3. 结果（无论正负）回写本卡

## 关联

- 根因分析: docs/worklogs/2026-09-01_后训练根因分析与80分改进路线图.md
- 同构性审计: 语料 vs 最新 loop 五层一致（prompt/schema/template/completion/版本），见对话记录
- 教师正确性审计: 300 行合同一致性 100%，三层正确性结论见对话记录
