# TravelAgent Stage32：级联蒸馏与自适应路由 SPEC v1.0

> 日期：2026-08-15  
> 状态：执行中  
> 目标：把现有 Qwen3-4B / Qwen3-8B 的旅行工具决策能力压缩到 Qwen3-1.7B，并用可审计的升级路由获得真实推理收益。

## 1. 项目价值与边界

本阶段不把“训练了一个小模型”当作成果。完整成果必须同时证明：

1. **数据能力**：多教师采样、环境执行、验证器筛选、去重、冲突隔离、评测集防泄漏；
2. **后训练能力**：1.7B Base、SFT、SFT+DPO 三段严格对照；
3. **Agent 能力**：评价工具选择、参数、恢复、澄清、取舍和必要终止，而不是只评价自然语言相似度；
4. **系统能力**：1.7B → 4B → 8B 的置信度升级、超时回退、可观测性和并发压测；
5. **证据能力**：冻结评测、配对统计检验、训练与推理报告、数据和 checkpoint 哈希可复现。

当前实现属于**序列级行为蒸馏**：学生学习验证器认可的教师工具轨迹。除非后续保存并使用教师 token logits，否则不得声称已经完成白盒 logits KD。

## 2. 冻结基线

### 2.1 教师与现有候选

- 高频教师：`qwen3-4b-stage28-dpo-abort-diverse-v2`；
- 复杂教师：`Qwen3-8B`；
- 学生底模：`Qwen3-1.7B`；
- 现有 Stage29 外部模型评测：4B DPO `127/150`，8B `136/150`；
- 现有 4B+8B 路由回放：`143/150`；
- Stage31 运行时投影后的 4B 学生路径：`81/81`。

Stage27 Pilot、Stage29 Dev/Sealed Test、Stage19 Holdout 和 Stage32 Holdout 都是禁用训练语料。Stage31 的 300 条授权回放只可用于分布分析、在策略采样和回归，不可直接整批复制进训练集，也不可冒充真实流量或 Canary 证据。

## 3. 模型职责

| 模型 | 主要职责 | 默认流量 |
|---|---|---:|
| Qwen3-1.7B | 搜索、单字段澄清、可重试恢复、控制器已确定的终止 | 目标 70%～85% |
| Qwen3-4B Stage28 DPO | 多字段澄清、参数复杂搜索、恢复分歧、中等边界 | 目标 10%～25% |
| Qwen3-8B | 多约束取舍、不可行/不安全边界、长上下文重规划、教师裁决 | 目标 5%～15% |

路由只能读取生产可见状态、模型输出合法性、候选一致性和校准分数，禁止读取金标动作。

## 4. 数据构建

### 4.1 Pilot

- 128 个训练任务，澄清、搜索、恢复、取舍/终止各 32；
- 4B 每题 2 个候选；
- 8B 对取舍/终止、4B 全失败或 4B 候选分歧的题生成 2 个候选；
- 仅保留环境执行和验证器通过的 chosen；
- rejected 必须与 chosen 共享相同模型可见上下文，并且有明确的失败或效率原因码；
- Pilot 通过后才允许正式生成。

Pilot 门槛：任务至少 120 个有成功候选；四个任务族均不少于 28 个成功；标签冲突为 0；冻结语料精确/近重复为 0；4B 必须实际贡献不少于 40% 的 chosen。

### 4.2 Formal

- 目标 3,200 个唯一训练任务；
- 高频池约 2,400，优先由 4B 生成；
- 困难池约 800，由 4B 与 8B 联合生成并由验证器裁决；
- 每个模型可见 prompt 只允许一个 SFT 标签；冲突全部隔离，不做多数票硬合并；
- SFT train / validation / test 按任务模板、城市簇、约束指纹做 group split；
- DPO 至少 1,500 个唯一上下文对，四个能力族都必须覆盖；
- DPO 负例优先来自 1.7B Base/SFT 在训练任务上的在策略失败；教师 chosen 与学生 rejected 必须共享完全相同的模型可见上下文；
- 教师之间都成功且验证器无法给出明确优劣时，不为凑数量制造伪偏好；
- 另建 400 条 Stage32 frozen holdout，任务模板和约束组合与训练组隔离。

### 4.3 多教师裁决

1. verifier success 优先于模型身份；
2. hard constraint pass 优先；
3. 无非法动作、无重复调用、步骤更少、工具调用更少优先；
4. 高频题在质量等价时优先 4B；
5. 复杂题或 4B 分歧时优先 8B；
6. 教师仍然分歧且验证器无法区分时，样本隔离，不进入训练。

必须保存 teacher model、checkpoint、采样温度、候选序号、轨迹 ID、验证器版本、reward、选择原因和数据哈希。

## 5. 训练矩阵

| 运行 | 起点 | 数据 | 目的 |
|---|---|---|---|
| A | Qwen3-1.7B Base | 无 | 底模基线 |
| B | Qwen3-1.7B Base | verifier-chosen SFT | 行为蒸馏 |
| C | B | 在训练任务上采样学生失败 | 形成同上下文 on-policy rejected |
| D | B | verifier-grounded DPO | 偏好优化 |
| E（可选） | D | student on-policy SFT repair | 修复 DPO 后仍稳定复现的错误 |

统一使用 QLoRA 起步，固定 seed、tokenizer、最大长度和评测协议。SFT 与 DPO 均先 smoke 再 formal。若 DPO 不超过 SFT，则保留为负结果，不以训练集指标替代外部增益。

## 6. 评测与统计

质量评测至少包括：

- Stage32 frozen holdout；
- Stage29 150 条外部模型评测；
- 1.7B Base / SFT / DPO、4B、8B、三级 Router 的同协议比较；
- action accuracy、argument validity、hard-pass、abort recall、false abort、fallback；
- 按 clarification/search/recovery/tradeoff/abort 分桶；
- 配对 Bootstrap 95% CI 和 McNemar exact test；
- temperature=0 五轮稳定性与 temperature=0.2 三轮鲁棒性。

效率评测必须在同一 GPU、同一量化、同一 max length、同一并发矩阵下报告：

- P50/P95/TTFT；
- requests/s 和 tokens/s；
- 峰值/空载显存；
- 1.7B 独立服务与三级路由端到端开销；
- 每 1,000 请求的 4B/8B 升级次数和估算 GPU 时间。

## 7. 晋升门槛

### 7.1 1.7B checkpoint

- Stage29 总正确率不低于 4B Stage28 DPO 的 95% 相对水平，即至少 `121/150`；
- search、clarification 各不低于 90%；
- false abort 不超过 1%；
- 参数合同错误率不超过 1%；
- 相对 4B 的 P95 延迟至少降低 25%，或吞吐至少提高 40%；
- SFT/DPO 不得以 Stage29 调参，Stage29 只在冻结候选后运行。

### 7.2 三级 Router

- Stage29 至少 `143/150`，不得低于现有双模型 Router；
- 8B 调用比例相对现有 46% 至少下降 30%；
- HTTP 错误为 0；
- temperature=0 五轮动作稳定率至少 98%；
- 升级路由不可读取金标或 sealed metadata。

未通过门槛时，保留 1.7B 为实验 checkpoint，不进入生产默认路由。

## 8. 执行顺序

1. 冻结数据合同、禁用语料清单和基线哈希；
2. 实现多教师候选合并、裁决和 provenance；
3. 跑 128 题 Pilot，并审计教师贡献和污染；
4. 下载并验证 Qwen3-1.7B；
5. 扩大 Formal 数据，构建 SFT/DPO/holdout；
6. 训练并评测 Base、SFT；
7. 在训练任务上采样学生错误，构建 DPO 后训练并评测；
8. 对 DPO 后仍稳定复现的学生错误最多做一轮 SFT repair，避免反复污染 holdout；
9. 校准 1.7B→4B→8B Router；
10. 完成并发、显存、稳定性、统计显著性和消融；
11. 冻结报告、模型卡、数据卡、工作日志和复现命令。

## 9. 参考方法

- Sequence-Level Knowledge Distillation: https://arxiv.org/abs/1606.07947
- On-Policy Distillation / GKD: https://arxiv.org/abs/2306.13649
- MiniLLM（生成模型 reverse-KL KD；本阶段仅作为可选白盒扩展）: https://arxiv.org/abs/2306.08543

这些论文只支撑方法选择；项目中的任何质量、速度和显存结论必须来自本仓库实验。
