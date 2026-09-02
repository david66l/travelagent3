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

## 诊断结果（2026-09-02 回写）

**训练+NLL 部分完成，趋势支持假设但未过原定 kill 门；行为评测被独立的评测基建 bug 阻塞（见下）。**

| 段落 NLL | v11@12 步（旧教材） | canonical@12 步 | canonical@24 步 |
|---|---|---|---|
| rationale_prefix | 4.510 | 3.178 | **2.342**（仍下降） |
| action | 0.045 | 0.040 | 0.023 |
| evidence | 0.057 | 0.041 | **0.013** |
| other（确定性信封对照） | 0.206 | 0.140 | **0.053** |

- 方法论修正：零优化器诊断（旧模型 × 新教材 = 3.93）无法回答"新教材可学性"——旧模型的概率质量在旧习惯上。正确仪器 = 复刻 v11 超参的探针训练（12/24 步）。
- 判读：对照组（同为确定性 token 的信封段）已压到 0.05，证明配置可学确定性目标；连接词是纯回忆段（prompt 从不出现），收敛更慢但斜率持续为负（−1.33 / −0.84）。24 步 2.34 优于 kill 门 2.0 但未过线——按预登记精神，**不启动正式训练**，先解决下面的评测阻塞并考虑延长训练步数预注册（h48）后重验。
- 训练健康度：train_loss 0.406 / eval_loss 0.658 / tok_acc 0.825（无 v8 式退化）。

## 发现的评测基建 bug（H-002 待立项）

`react.py:71` 的 `0 <= delta <= 10` 天气预报窗口以**墙钟**计算，而重放的是**冻结语料**：
9/1 评测时行程（2026-09-12 出发）距出发 11 天 → weather 非必需，3 动作前缀可到达 review；
9/2 起距出发 ≤10 天 → weather 变必需 → 所有冻结前缀不再充分 → `replayed decision state did not reach itinerary review`。
**影响面：全部 internal-dev/train-shadow 重放评测 + 未来在冻结语料上的 GRPO rollout（语料自带保质期）。**
修复方向：重放冻结状态时将环境参考时钟钉到语料构建时刻（frozen clock 注入 delta 与 freshness 计算）。

## 关联

- 根因分析: docs/worklogs/2026-09-01_后训练根因分析与80分改进路线图.md
- 同构性审计: 语料 vs 最新 loop 五层一致（prompt/schema/template/completion/版本）
- 教师正确性审计: 300 行合同一致性 100%
- 探针产物: 服务器 h001-canonical-probe-h24-stop12 / h001-canonical-probe-full24 + h001-zero-optimizer-diagnostic-v1/
