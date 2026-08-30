# TravelAgent Stage32 模型卡与数据卡 v1.0

日期：2026-08-15  
状态：SFT 生产默认；SFT-reference DPO v2 进入 Shadow；旧 DPO 拒绝

## 1. 系统定位

Stage32 将旅行 Agent 的“有界工具决策”从通用规划模型中拆出，训练 Qwen3-1.7B 专用学生。学生负责 `search_pois`、`ask_user`、`propose_tradeoff`、`abort` 及恢复动作；Qwen3-8B 仅负责学生课程之外的 `complex` 规划。求解、验证、完成门禁仍由确定性控制器持有。

这不是 logits 蒸馏，而是多教师、验证器引导的行为蒸馏：4B 提供高频策略候选，8B 提供复杂候选，控制器验证候选并记录教师来源，再以 SFT 和 DPO 完成后训练。

## 2. 模型谱系

| 角色 | 基座/来源 | 结论 |
|---|---|---|
| 学生基座 | Qwen3-1.7B | 冻结外部集模型裸能力 106/150 |
| SFT 学生 | Qwen3-1.7B + Stage32 SFT QLoRA | 135/150，生产默认 |
| 旧 DPO | SFT + 隐式 Base reference | 104/150、37 次 HTTP 流中断，拒绝 |
| 修正版 DPO v2 | SFT + 显式冻结 SFT reference | 137/150、0 HTTP 错误，Shadow 候选 |
| 既有学生教师 | Qwen3-4B Stage28 DPO | 133/150 |
| 复杂教师 | Qwen3-8B Base | 142/150 |

训练使用 NF4 double quantization、LoRA r=16/alpha=32。SFT 训练 1 epoch；DPO v2 使用 beta=0.1、学习率 2e-6、1 epoch，并显式加载名为 `ref` 的只读 SFT adapter。

## 3. 数据卡

### 3.1 Cascade Pilot

- 192 个唯一任务，768 条候选 rollout；
- 161 个任务通过验证器、SFT 可训练性和泄漏门禁；
- 4B chosen 98，8B chosen 63；
- 四类任务：search 39、clarification 36、recovery 48、tradeoff 38；
- 31 个与禁用语料精确重合的任务被隔离，保留数据重合为 0；
- 强教师候选全部成功，未伪造 success-over-failure 偏好对。

### 3.2 正式 SFT

- Stage28 已审计 SFT：1282 条；
- Stage32 Cascade：161 条；
- 合并后 1443 条，train/validation/test = 1264/105/74；
- 动作分布：search 1101、ask_user 156、propose_tradeoff 106、abort 80；
- 唯一模型输入 1443，无标签冲突，无 split group overlap；
- 最大序列 1015 token，P95 1007，无截断。

### 3.3 偏好数据与 on-policy 门禁

- 正式偏好集 train/validation/test = 727/182/182，共 1091 个唯一 pair；
- 覆盖 clarification、search、recovery、tradeoff 和 5 类必要终止；
- 学生在 1264 个训练域提示上进行 3 次随机采样，共 3792 次；
- 只产生 2 个唯一动作错误，数据集因多样性和验证切分不足被标记 `rejected`；
- 未把 2 条重复扩写成“300 条”，未启动第二轮 on-policy DPO；
- on-policy 数据使用冻结外部评测 0 条，训练提示与 Stage29 精确/近似重合均为 0。

## 4. 双口径评测

冻结外部集 150 题，temperature=0，并发 8，单轮工程测量。

### 4.1 模型裸能力：原始多动作空间

| 模型 | 正确 | 平均延迟 | 吞吐 |
|---|---:|---:|---:|
| 1.7B Base | 106/150 | 1952.8 ms | 3.865 req/s |
| 1.7B SFT | 135/150 | 3114.8 ms | 2.497 req/s |
| 1.7B SFT+DPO v2 | 137/150 | 3474.2 ms | 2.252 req/s |
| 4B Stage28 DPO | 133/150 | 5596.1 ms | 1.367 req/s |
| 8B Base | 142/150 | 2922.3 ms | 2.701 req/s |

SFT 相对 Base 提升 29 题，即 19.33 个百分点；修正版 DPO 再提升 2 题。DPO v2 对 SFT 的 McNemar 双侧 p=0.5，因此不能把 +2/150 宣称为确定收益，只进入 Shadow。

### 4.2 最新生产控制器合同

150 条中有 1 条冻结标签与当前运行时允许动作冲突；合同一致集为 149 条。

| 模型 | 合同一致正确 | 平均延迟 | 吞吐 |
|---|---:|---:|---:|
| 1.7B Base | 148/149 | 2061.3 ms | 3.713 req/s |
| 1.7B SFT | 148/149 | 2986.8 ms | 2.607 req/s |
| 1.7B SFT+DPO v2 | 148/149 | 3058.2 ms | 2.557 req/s |
| 4B Stage28 DPO | 149/149 | 4283.7 ms | 1.819 req/s |
| 8B Base | 145/149 | 2478.8 ms | 3.184 req/s |

生产合同把动作空间收窄后，Base/SFT/DPO 的差距被控制器覆盖。这说明模型后训练收益与 controller-first 系统收益必须分开报告。

## 5. DPO reference 故障复盘

旧训练代码把 `ref_model=None` 描述为“冻结 SFT”。但 TRL 1.9.2 对没有 `ref` adapter 的 PEFT policy 会进入 `disable_adapter()`，实际参考模型是 Base。旧 DPO 因此出现：

- 模型裸能力从 SFT 135/150 降至 104/150；
- 37 次 incomplete chunked stream；
- preference accuracy 仍为 100%，证明训练内指标不能替代外部行为验收。

修复后显式加载同一 SFT checkpoint 为只读 `ref` adapter，重新训练得到 137/150、0 HTTP 错误。训练报告中的 `reference_policy` 为 `frozen-sft-adapter:ref`。

## 6. 服务路由

- 四类有界动作全部路由 1.7B；
- 只有课程之外的 `complex` 动作路由 8B；
- 学生推理失败时允许一次 8B fallback；
- 当前冻结集不含课程外 complex 动作，因此教师占比为 0，不代表生产流量永远为 0；
- 路由回放为顺序离线回放，不冒充同卡双模型在线延迟。

## 7. 已知限制

- 延迟和吞吐是单轮工程测量，未给出多轮置信区间；
- DPO v2 的 +2/150 不具统计显著性，仍需 Shadow 稳定性证据；
- 8B 的复杂规划优势需在多轮/分布外 benchmark 中继续验证；
- Stage31 授权回放不是实际线上 Canary；
- 4-bit bitsandbytes 性能不等同于最终 TensorRT-LLM/AWQ 部署性能。

## 8. 可复现哈希

| 产物 | SHA256 |
|---|---|
| 1.7B SFT adapter | `cd958716d6329eac3b25e94ca51fce79631ceb1dc9188e2adb8494e5041fec6e` |
| 1.7B DPO v2 adapter | `a333dc1eb7dc8e797e04395fc47bceaf84557bdcbdb9294bc81f26a4d9c667f3` |
| 正式 SFT manifest | `0485f8a746fcfbd24576212e328bb8dcdc69882214e94674a94351b3f1a71430` |
| Cascade Pilot manifest | `418ec3257fd619b4c64daefda3e17aea906fa4f8bb7828416ee43399f0194495` |
| 最终评测 report | `a20caaa2598fa9b8c57984229f3a642bd4ca71e906d5f35ebe61141206b69e66` |

最终报告见 `ml/agentic/reports/stage32-cascade-distillation-final-v1/`。二进制 adapter 和凭据不得上传公开 Git；API key、SSH 密码和服务凭据均不属于模型卡或数据集内容。
