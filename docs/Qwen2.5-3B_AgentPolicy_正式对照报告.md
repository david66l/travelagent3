# Qwen2.5-3B Agent Policy：Base / SFT / SFT+GRPO 对照

> 范围：逐任务、逐样本配对的 NF4 随机验证审计。当前训练为受控工程实验，
> 不将结果表述为论文级 benchmark，也不宣称已完成逐轮信用优化。

## 评测契约

- 任务数：32；每任务采样：4；总 rollout：128。
- Seed：44；协议：`sha256-task-sample-v1`。
- 温度：0.8；量化：`nf4-double-quant`。
- Reward：`hierarchical-b0.v1`。
- validation 仅用于盲评，不参与 SFT/GRPO 任务挑选或梯度更新。

## 总体结果

| 模型 | 成功率 | 平均 Reward | 相对 Base 成功率 | 相对 Base Reward |
|---|---:|---:|---:|---:|
| Base 3B | 60.16% | 0.2301 | +0.00 pp | +0.0000 |
| SFT 3B | 80.47% | 0.5934 | +20.31 pp | +0.3633 |
| SFT+GRPO 3B | 82.81% | 0.6373 | +22.66 pp | +0.4072 |

## 分任务能力

| 模型 | 澄清 | 故障恢复 | 普通搜索 | 约束权衡 |
|---|---:|---:|---:|---:|
| Base 3B | 84.38% | 81.25% | 75.00% | 0.00% |
| SFT 3B | 87.50% | 93.75% | 84.38% | 56.25% |
| SFT+GRPO 3B | 90.62% | 93.75% | 84.38% | 62.50% |

## Reward 分量均值

六类策略 Reward 为 task、constraint、tool、grounding、efficiency、format；
quality 当前权重为 0，仅作审计字段。

| 模型 | task | constraint | tool | grounding | efficiency | format | quality |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base 3B | 0.2031 | 0.2031 | 0.1766 | 0.4219 | 0.3704 | 0.4219 | 0.0000 |
| SFT 3B | 0.5813 | 0.6094 | 0.3965 | 0.7188 | 0.5363 | 0.7188 | 0.0000 |
| SFT+GRPO 3B | 0.6250 | 0.6562 | 0.4316 | 0.7656 | 0.5593 | 0.7656 | 0.0000 |

## 结论与边界

- Base 已能处理部分澄清、搜索和恢复，但本评测中的权衡成功率为 0%。
- 环境对齐 SFT 主要补齐约束冲突时的合法工具决策，并显著提高总体成功率。
- 单步保守 GRPO 在不降低搜索/恢复的前提下，进一步提高澄清与权衡成功率。
- 当前结果支持“多轮环境中的轨迹级 GRPO 工程基线”；尚未完成逐轮信用对照、完整 Reward 消融或线上灰度，不能宣称已解决 Long-Horizon credit assignment。
- 每个 arm 的原始 `rollouts.jsonl` 保留失败终止原因、动作、独立 rollout seed 和 Reward 分量，可逐条复核。
