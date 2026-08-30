# 阶段 15：Recovery SFT R2 拒绝与均衡回放 R3 准备

日期：2026-08-14  
状态：R2、R3 均已完成并被正式晋升门拒绝；无效格式重试方案已撤销

## 1. 本阶段目标

阶段 14 的 Recovery SFT R1 在训练后从 `4/48` 退化到 `0/48`，并出现 `trusted_city`、`max_results` 等 schema 禁止参数。阶段 15 的目标是：

1. 降低 Recovery SFT 的更新强度；
2. 将旧能力回放比例从 `adaptive:replay = 1:1` 提高到 `1:3`；
3. 用 LoRA 线性混合控制 Recovery adapter 对原候选的影响；
4. 在自适应恢复集和原正式 curriculum 上进行配对评测；
5. 把未知参数率、受保护参数率加入 checkpoint 硬门禁；
6. 若 R2 仍有单类能力回退，构建按能力族均衡的 R3 数据，而不是放宽门禁。

## 2. 恢复现场与远端审计

本轮从本地未提交工作树、阶段 14 日志和 AutoDL 训练目录恢复上下文。远端状态：

- 工作目录：`/root/autodl-tmp/TravelAgent2`；
- GPU：NVIDIA GeForce RTX 4090，24564 MiB；
- 审计时显存占用约 1 MiB、利用率 0%，无遗留训练进程；
- 当前最佳旧候选仍为 `artifacts/grpo-qwen25-3b-aligned-learnable-step1`；
- R2、多个线性混合候选和配对评测实际上已经完成，但尚未写入本地工作日志。

本轮没有重复启动已经完成的训练。先回收远端报告，再根据真实结果决定后续工作。

## 3. R2 训练配置与结果

R2 从当前最佳 GRPO adapter 继续训练：

- 数据：`sft-adaptive-recovery-replay3-zh-v1`；
- 数据比例：`adaptive:replay = 1:3`；
- train / validation / test：1024 / 128 / 128；
- epoch：1；
- learning rate：`1e-5`；
- batch size：1；
- gradient accumulation：16；
- max length：2048；
- seed：42；
- 输出：`artifacts/sft-qwen25-3b-adaptive-recovery-r2`。

训练结果：

- 训练耗时：366.69 秒；
- train loss：0.069470；
- eval loss：0.324187；
- eval mean token accuracy：0.949670；
- epoch：1.0。

与 R1 相比，R2 已显著降低更新强度并增加旧能力回放，但训练 loss 不能代替行为评测和晋升门。

## 4. LoRA 线性混合实验

为了避免把完整 Recovery adapter 直接替换原候选，新增并使用 `ml/agentic/training/merge_lora_adapters.py`，对兼容的 LoRA adapter 做线性组合：

- 主 adapter：`grpo-qwen25-3b-aligned-learnable-step1`；
- 候选 adapter：`sft-qwen25-3b-adaptive-recovery-r2`；
- Recovery 权重：25%、10%、5%、2.5%；
- 混合产物只允许用于评测，不能未经门禁直接发布。

专项测试验证了 base model、PEFT 类型、任务类型、LoRA rank、alpha、`modules_to_save` 和 target modules 的兼容性；目标模块比较不依赖列表顺序。

## 5. 自适应恢复正式对照

评测协议：

- 数据：官方 GRPO validation 派生的独立自适应恢复集；
- 任务：12；
- 每题 rollout：4；
- 总 rollout：48；
- temperature：0.7；
- max tool iterations：4；
- seed：42；
- checkpoint 均使用相同 system prompt、tool schema 和配对 seed。

结果：

| Checkpoint | 成功 rollout | 成功率 | 失败/非法策略 rollout |
|---|---:|---:|---:|
| 原 GRPO-B0 + schema prompt | 10 / 48 | 20.83% | 38 / 48 |
| 10% Recovery R2 blend + schema prompt | 35 / 48 | 72.92% | 13 / 48 |

10% 混合在独立自适应恢复集上提升 `52.08` 个百分点，说明 R2 adapter 确实包含可用的证据驱动恢复能力；但仍有 27.08% rollout 不能完成恢复，因此只能进入原能力回归门，不能据此直接晋升。

## 6. 原正式 curriculum 配对评测

评测协议：

- 数据：正式 curriculum validation；
- 每个能力族 8 个任务；
- 能力族：clarification、recovery、search、tradeoff；
- 总任务：32；
- 每题 rollout：4；
- 总 rollout：128；
- family offset：2；
- temperature：0.8；
- seed：44；
- before / after 使用完全相同的任务、初始状态和 rollout seed。

结果：

| 能力族 | 原 GRPO-B0 | 10% Recovery R2 blend | 变化 |
|---|---:|---:|---:|
| clarification | 32 / 32，100% | 29 / 32，90.63% | -9.38 pp |
| recovery（旧超时恢复） | 32 / 32，100% | 32 / 32，100% | 0 |
| search | 32 / 32，100% | 32 / 32，100% | 0 |
| tradeoff | 6 / 32，18.75% | 32 / 32，100% | +81.25 pp |
| 总体 | 102 / 128，79.69% | 125 / 128，97.66% | +17.97 pp |

虽然总体成功率和平均 Reward 都显著提高，但 clarification 从 100% 降到 90.63%，超过既定的单能力族最大回退阈值 5%。正式晋升报告：

- `promoted = false`；
- gate error：`FAMILY_SUCCESS_REGRESSION:clarification:1.000000->0.906250`。

不能因为总体指标更高就忽略单类回退，也没有修改阈值迎合候选。

## 7. 降低混合权重后的诊断

继续检查 5% 和 2.5% Recovery 权重：

- clarification 均为 `30/32 = 93.75%`；
- 相对原候选仍下降 6.25 个百分点；
- 仍超过 5% 门禁，不能晋升。

两个固定失败样本的原始输出为合法意图但非法 JSON，例如：

```text
<tool_call>
{"name": "ask_user", "arguments": {"question": "请问您对行程预算是否有具体要求？"})
</tool_call>
```

失败原因是闭合符号输出为 `})`，解析器正确返回 `TOOL_CALL_PARSE_ERROR`。这不是 96 token 上限导致的截断，也不能通过放宽严格 JSON parser 来掩盖。说明即使很低的 Recovery adapter 权重仍会在部分采样 seed 上造成格式能力回退。

因此以下候选全部维持“实验产物、拒绝晋升”状态：

- `sft-qwen25-3b-adaptive-recovery-r2`；
- `blend-qwen25-3b-adaptive-r2-w25`；
- `blend-qwen25-3b-adaptive-r2-w10`；
- `blend-qwen25-3b-adaptive-r2-w05`；
- `blend-qwen25-3b-adaptive-r2-w025`。

## 8. 本轮新增的评测硬门禁

阶段 14 的 R1 schema 回退发生在模型输出解析/参数校验阶段，旧报告只会把它计为 `empty_action_rollout`，无法单独审计未知参数与受保护参数。

本轮扩展 `scripts/audit_model_curriculum.py`：

- 记录 policy output error 数量与比例；
- 单独记录 `POLICY_ARGUMENT_INVALID`；
- 从原始 tool call 中恢复参数名；
- 统计 unknown argument error；
- 统计 protected argument error；
- 当前受保护字段包括 `city`、`trusted_city`、`max_results`、`constraints`、`facts`、`matrices`、`itineraries`。

扩展 `scripts/compare_curriculum_audits.py`：

- 默认 unknown argument error rate 必须为 0；
- 默认 protected argument error rate 必须为 0；
- 任一非零即拒绝 checkpoint；
- 阈值和 after behavior gate 写入晋升报告，避免只看成功率。

这解决了“模型在执行前被 schema 拒绝，但晋升报告没有专门指标”的可观测性缺口。

## 9. R3 均衡回放数据方案

R2 的 1:3 回放虽然增加了旧数据，但仍从统一 replay pool 做稳定哈希抽样，不能保证 clarification、tradeoff 和普通 search 的训练量严格均衡。

本轮已修改 `scripts/build_adaptive_recovery_sft_dataset.py`：

1. 将旧能力回放明确分类为 clarification、tradeoff、search；
2. 每个 split 内按三类近似等额分配 replay；
3. adaptive recovery 自身作为第四类能力；
4. 各能力池使用稳定哈希排序；
5. train / validation / test 对每类使用独立、不重叠的 source task；
6. 仍禁止官方 GRPO validation 进入训练；
7. derivation manifest 新增 `replay_family_examples`，可以审计每类数量；
8. 如果某类 source row 不足，构建直接失败，不允许静默用其他类别补齐。

使用 `replay_ratio=3` 时，目标分布为：

```text
adaptive recovery : clarification : tradeoff : search = 1 : 1 : 1 : 1
```

R3 代码已完成，但为了先写日志，本轮尚未生成新数据目录、尚未同步 AutoDL、尚未启动 R3 训练。

## 10. 本轮验证

专项测试：

- adaptive recovery 数据与均衡分配；
- curriculum audit 行为门禁；
- checkpoint promotion gate；
- 共 `11 passed`。

Ruff：

- 本轮涉及的脚本和测试全部通过。

本轮没有重新运行完整后端 972 项回归；最近一次完整回归仍是阶段 14 的 `972 passed`。恢复开发后，在 R3 正式构建前至少运行 Agentic / Evaluation 专项回归，在阶段收口时再运行完整 unit + integration。

## 11. 证据位置

本地已回收的轻量报告：

- `ml/agentic/reports/adaptive-recovery-r2-audit/remote/artifacts/sft-qwen25-3b-adaptive-recovery-r2/training_report.json`；
- `ml/agentic/reports/adaptive-recovery-r2-audit/remote/artifacts/eval-adaptive-recovery-b0-schema-prompt-formal/report.json`；
- `ml/agentic/reports/adaptive-recovery-r2-audit/remote/artifacts/eval-adaptive-recovery-blend-w10-schema-prompt-formal/report.json`；
- `ml/agentic/reports/adaptive-recovery-r2-audit/remote/artifacts/eval-original-b0-schema-prompt-formal/report.json`；
- `ml/agentic/reports/adaptive-recovery-r2-audit/remote/artifacts/eval-original-blend-w10-schema-prompt-formal/report.json`；
- `ml/agentic/reports/adaptive-recovery-r2-audit/remote/artifacts/eval-original-blend-w10-schema-prompt-formal/promotion.json`。

完整 adapter、rollout JSONL 和诊断日志仍保存在 AutoDL 数据盘。本地只回收轻量报告，不复制模型权重，避免占用本机磁盘。

## 12. 下一步

恢复工作后按以下顺序继续：

1. 运行本轮 Agentic / Evaluation 专项回归；
2. 用均衡策略构建 `sft-adaptive-recovery-balanced-zh-v1`，检查四类数量、split overlap 和 official validation overlap；
3. 执行 SFT preflight；
4. 同步本轮代码、测试和数据到 AutoDL；
5. 从当前最佳 GRPO-B0 adapter 做更保守的 R3 小步 SFT；
6. 先跑 4 个任务的小门禁，失败则停止；
7. 通过后复测独立 adaptive recovery 12×4；
8. 再跑原 curriculum 32×4 配对评测；
9. unknown/protected argument error rate 必须为 0；
10. clarification 等任一能力族回退超过 5% 时继续拒绝，不进入 R1-v2 GRPO。

当前准确结论：R2 证明了 Recovery SFT 能显著提高自适应恢复，但所有 R2/混合候选仍因可复现的 clarification JSON 格式回退被拒绝；线上候选不变，下一步是均衡能力回放 R3，而不是继续降低混合权重或放宽 parser/晋升门。

## 13. 继续执行：R3 正式数据构建

恢复工作后先运行 Agentic + Evaluation 专项回归：

- 收集 194 项；
- `194 passed`；
- 仅有 1 条第三方 LangGraph pending deprecation warning。

随后使用均衡回放策略正式构建：

`ml/agentic/datasets/build/sft-adaptive-recovery-balanced-zh-v1`

构建结果：

- dataset version：`sft-adaptive-recovery-13c4ad426de067de`；
- candidate / accepted / exported：1280 / 1280 / 1280；
- rejected：0；
- train / validation / test：1024 / 128 / 128；
- adaptive recovery：320；
- clarification replay：320；
- tradeoff replay：320；
- search replay：320；
- internal source overlap：空；
- official validation overlap：空；
- 自适应恢复审计样本：320；
- preflight：`ready=true`，errors / warnings 均为空。

这次数据分布严格实现了四能力 `1:1:1:1`，不再依赖统一 replay pool 的近似随机分布。

同步包：`E:\A_Louis\travelagent_stage15_r3.tar`。本地与 AutoDL SHA-256 一致：

```text
661958249f50ec7c9097af4ef0345d3639c7e4554a54e7e4a8433571e89776c7
```

AutoDL 训练环境未安装 pytest，因此没有污染训练 venv 补装测试依赖。本地 194 项作为代码回归证据；云端执行 compileall 与真实训练 preflight。首次 preflight 因未设置 OpenAI key 在导入全局 LLM client 时失败，随后使用 `OPENAI_API_KEY=offline-training-placeholder` 明确标记离线训练进程，preflight 通过。该 placeholder 不用于任何外部 API 调用。

## 14. R3 训练

R3 从当前最佳 GRPO-B0 adapter 继续：

- model：`artifacts/grpo-qwen25-3b-aligned-learnable-step1`；
- dataset：`sft-adaptive-recovery-balanced-zh-v1`；
- output：`artifacts/sft-qwen25-3b-adaptive-recovery-r3-balanced`；
- epoch：0.5；
- learning rate：`5e-6`；
- batch size：1；
- gradient accumulation：16；
- max length：2048；
- seed：42；
- 正式优化 step：32。

训练结果：

- train runtime：211.05 秒；
- train loss：0.275032；
- eval loss：0.544861；
- eval mean token accuracy：0.890426；
- eval entropy：0.254467；
- run scope：formal；
- continued from adapter：true。

相比 R2 的 1 epoch、`1e-5` 和 token accuracy 0.9497，R3 更新更保守，没有再次快速拟合到接近满 token accuracy。

## 15. R3 小门结果

### 原能力固定 8 题配对小门

协议：四能力各 2 题、每题 4 rollout、seed 44、temperature 0.8。

| Checkpoint | 成功 | 成功率 | 平均 Reward |
|---|---:|---:|---:|
| 原 GRPO-B0 | 26 / 32 | 81.25% | 0.598164 |
| R3 | 28 / 32 | 87.50% | 0.713734 |

分能力：

- clarification：100% → 100%；
- 旧 recovery：100% → 100%；
- search：100% → 100%；
- tradeoff：25% → 50%；
- unknown argument error rate：0；
- protected argument error rate：0。

小门 `promoted=true`。

### 独立自适应恢复 4 题小门

- 15 / 16 成功；
- 成功率 93.75%；
- unknown / protected argument error rate 均为 0。

因此 R3 获准进入正式 12×4 和 32×4 评测。

## 16. R3 正式评测与拒绝

### 独立自适应恢复 12×4

- 成功：33 / 48；
- 成功率：68.75%；
- 原 GRPO-B0：10 / 48，20.83%；
- 提升：47.92 个百分点；
- policy output error：0；
- unknown argument error：0；
- protected argument error：0；
- 15 个失败为策略未正确适配快照反馈，而不是 schema 越权。

R3 显著好于原候选，但低于 R2 10% blend 的 35 / 48（72.92%）。

### 原 curriculum 32×4

| 能力族 | 原 GRPO-B0 | R3 | 变化 |
|---|---:|---:|---:|
| clarification | 32 / 32，100% | 30 / 32，93.75% | -6.25 pp |
| recovery | 32 / 32，100% | 32 / 32，100% | 0 |
| search | 32 / 32，100% | 32 / 32，100% | 0 |
| tradeoff | 6 / 32，18.75% | 16 / 32，50% | +31.25 pp |
| 总体 | 102 / 128，79.69% | 110 / 128，85.94% | +6.25 pp |

R3 总体成功率和平均 Reward 均提升，unknown / protected argument error rate 均为 0；但 clarification 回退 6.25 个百分点，超过既定 5% 阈值。

正式晋升报告：

- `promoted=false`；
- gate error：`FAMILY_SUCCESS_REGRESSION:clarification:1.000000->0.937500`。

两个失败仍是 R2 阶段已经观察到的固定任务和 seed，输出用 `})` 错误闭合 JSON。均衡回放、降低学习率和缩短 epoch 没有消除该问题，因此 R3 明确拒绝，不进入 R1-v2 GRPO。

## 17. 受限格式重生成实验及撤销

为了区分“模型核心决策错误”和“可恢复的序列化错误”，曾实现以下受限实验：

1. 只对 `TOOL_CALL_PARSE_ERROR` / `TOOL_CALL_SHAPE_ERROR` 重试一次；
2. 原 strict JSON parser 不变；
3. allowed action 与 Pydantic schema 校验不变；
4. unknown / protected 参数仍为硬拒绝；
5. 尝试记录成功 retry rate。

先测试同温度随机重生成，再测试带纠错上下文的 greedy fallback。两种方法在两个固定 clarification 失败上都再次产生完全相同的非法 `})` 输出：

- 成功格式重试：0；
- 原失败仍为 2 / 32；
- 没有获得任何有效性收益；
- 只会增加一次模型推理延迟。

因此该方案未保留。相关本地与 AutoDL 源码均已回滚，`format_retry` 标识已从生产代码、报告脚本和测试中清除。回滚后再次运行 Agentic + Evaluation：`194 passed`，Ruff 通过。没有通过放宽 parser 或修改晋升阈值迁就候选。

另外使用全新 offset 10、seed 46 的 8 题配对集做了独立诊断：R3 为 27 / 32（84.38%），原候选为 26 / 32（81.25%），四能力无回退；独立 adaptive offset 12 的 4 题为 10 / 16（62.5%）。该结果说明 R3 的总体改进不是完全由一个固定集合造成，但正式晋升仍必须服从原 32×4 clarification 回退门。

## 18. R3 证据与最新下一步

本地轻量证据目录：

- `ml/agentic/reports/adaptive-recovery-r3-audit/`；
- 压缩归档：`ml/agentic/reports/adaptive-r3-reports.tgz`。

AutoDL 保留完整 R3 adapter、rollout 和控制台日志；GPU 当前空闲，无遗留训练进程。

当前准确结论：

- R3 证明四能力均衡回放可以在更保守训练下保留大部分旧能力，并将自适应恢复从 20.83% 提升到 68.75%；
- 但 clarification 仍从 100% 降到 93.75%，正式门禁正确拒绝；
- retry 和 greedy fallback 对固定语法错误无效，已撤销；
- 当前线上候选继续保持原 GRPO-B0，不发布 R2/R3；
- 不启动后续 GRPO。

下一步不再重复改变 replay 比例。应单独研究并实现训练/推理阶段的真正结构化解码约束，或构建针对 JSON 闭合 token 的格式保持目标；任何方案都必须先在未见 offset/seed 上验证，再重新使用正式晋升集，且报告中必须区分“模型原生合法率”和“解码器约束后的系统合法率”。

## 19. 结构化解码调研与版本审计

继续执行后，先核对了 Hugging Face、Outlines 和 vLLM 的官方接口，而不是把 parser repair 当成结构化解码：

- Hugging Face `generate` 支持 logits processor / token-prefix 约束；
- Outlines 可以把 JSON Schema 编译成有限状态约束，并接入 Transformers 的逐 token logits masking；
- vLLM 的当前 structured outputs 与原生 tool calling 可以在服务端做 schema-constrained decoding；
- vLLM 对 Qwen2.5 工具协议使用 `hermes` tool-call parser；
- `tool_choice="required"` 的完整支持要求 vLLM 至少为 0.8.3。

官方资料：

- [Transformers text generation](https://huggingface.co/docs/transformers/main_classes/text_generation)；
- [Outlines generator](https://dottxt-ai.github.io/outlines/latest/features/core/generator/)；
- [Outlines output types](https://dottxt-ai.github.io/outlines/latest/features/core/output_types/)；
- [Outlines Transformers integration](https://dottxt-ai.github.io/outlines/latest/api_reference/models/transformers/)；
- [vLLM structured outputs](https://docs.vllm.ai/en/v0.17.0/examples/online_serving/structured_outputs/)；
- [vLLM tool calling](https://github.com/vllm-project/vllm/blob/main/docs/features/tool_calling.md)。

同时发现一个独立部署债务：`backend/src/core/llm_client.py` 已发送 `tool_choice="required"`，但 `Dockerfile.vllm` 仍固定在 `v0.6.3.post1`，低于该能力的官方最低版本；Qwen 部署参数也没有声明 `--tool-call-parser hermes`。`docker-compose.vllm.yml` 还使用未固定的 `latest`、Llama-3.2 基模以及与当前 Qwen adapter 不匹配的占位目录。本轮没有盲目改写生产镜像：必须先完成离线协议验证，再单独升级和做服务端集成测试。

## 20. 状态绑定的 JSON Schema 约束实现

新增 `policy_tool_call_json_schema(actions)`：

1. 使用 Draft 2020-12 JSON Schema；
2. 每个 allowed action 生成一个 `oneOf` 分支；
3. `name` 使用 action-specific `const`；
4. `arguments` 直接复用对应 Pydantic action model 的精确 schema；
5. action 名称与 arguments schema 在同一分支绑定，禁止 `ask_user` 名称搭配 `search_pois` 参数；
6. 顶层与参数对象继续禁止额外字段；
7. 空 allowed-action 集直接失败。

`LocalCheckpointAgentPolicy` 增加三种显式协议：

- `native`：原生 Qwen 生成，作为模型能力审计面；
- `json_schema`：从首 token 起只允许纯 JSON；
- `qwen_tool_envelope`：强制 `<tool_call>\n{...}\n</tool_call>`，同时只在内部 JSON 上应用状态绑定 schema。

旧布尔参数仅为已经产生的实验报告保留兼容映射，新 CLI 使用 `--structured-decoding-mode`。`scripts/audit_model_curriculum.py` 还新增逐 rollout latency，并在报告中写入 mean / p50 / p95。依赖固定在 agentic-training extra：`outlines>=1.3.3,<2.0.0` 与直接使用的 `outlines-core>=0.2.14,<0.3.0`。

AutoDL 没有污染正式训练 venv；Outlines 1.3.3、Outlines Core 0.2.14 及其少量依赖安装在独立目录 `/root/autodl-tmp/outlines-spike-1.3.3`，实验时通过独立 `PYTHONPATH` 加载。

## 21. 纯 JSON 约束实验：语法成功、策略失败

先对两个固定 clarification 失败使用完全相同的 task、rollout seed、NF4 和温度：

- `curriculum-00246-西安`，seed `306011568`；
- `curriculum-00496-上海`，seed `2098894813`。

首版虽然消除了非法 `})`，但解码文本尾部保留了 `<|im_end|>`。最终仅在纯 JSON 约束模式对 tokenizer control token 使用 `skip_special_tokens=True`；native 模式不变，strict parser 不变，也没有做 JSON repair。修正后两条均生成合法 `ask_user`，2 / 2 通过。

固定 curriculum 32×4 的纯 JSON 结果：

| Checkpoint | 成功 | 成功率 |
|---|---:|---:|
| 原 GRPO-B0 | 128 / 128 | 100% |
| R3 | 128 / 128 | 100% |

这只能证明 decoder 的系统收益，不能证明 R3 权重晋升。更关键的独立 adaptive recovery 正式结果为：

| Checkpoint | Native | 纯 JSON | 变化 |
|---|---:|---:|---:|
| 原 GRPO-B0 | 10 / 48（20.83%） | 3 / 48（6.25%） | -14.58 pp |
| R3 | 33 / 48（68.75%） | 26 / 48（54.17%） | -14.58 pp |

两边恰好都下降 14.58 个百分点，且 policy output / unknown / protected argument error 均为 0。失败来自合法但策略错误的动作，而不是格式错误。延迟：

- R3 纯 JSON：mean 2627.529 ms，p50 2575.285 ms，p95 3081.459 ms；
- B0 纯 JSON：mean 2698.318 ms，p50 2594.833 ms，p95 3174.609 ms。

因此纯 JSON 模式正式拒绝，不能部署。原因是训练和 Qwen chat template 都使用原生 `<tool_call>` 外壳，从首 token 强制 `{` 造成明显输出分布偏移。

## 22. Qwen 原生 tool-call 外壳约束

第二版先由 Outlines Core 从状态绑定 JSON Schema 构造内部正则，再把完整 Qwen 外壳纳入同一个生成时有限状态约束：

```text
<tool_call>\n(JSON_SCHEMA_REGEX)\n</tool_call>
```

它不是 prompt 提醒、retry、parser 放宽或事后修复；从第一个生成 token 起，外壳、action 名称和 arguments 都在同一个约束内。

分级验证结果：

1. 两个历史固定 clarification 失败：2 / 2 通过；
2. adaptive recovery 4 题×4 小门：15 / 16（93.75%）；
3. R3 adaptive recovery 正式集：33 / 48（68.75%），与 R3 native 完全一致；
4. B0 adaptive recovery 正式集：10 / 48（20.83%），与 B0 native 完全一致；
5. 两个正式集的 policy output / unknown / protected argument error 均为 0；
6. 同协议下 R3 相对 B0 的权重收益仍为 +47.92 pp，没有被 decoder 夸大；
7. R3 固定 curriculum 32×4：128 / 128，clarification / recovery / search / tradeoff 各 32 / 32。
8. B0 固定 curriculum 32×4：同样为 128 / 128，确认该集合上的满分属于 decoder 系统收益，而不是 R3 权重收益。

adaptive 延迟：

- R3 外壳约束：mean 3029.356 ms，p50 2934.488 ms，p95 3488.494 ms；
- B0 外壳约束：mean 3049.772 ms，p50 2961.393 ms，p95 3618.110 ms。

R3 固定集延迟：mean 2178.977 ms，p50 2233.306 ms，p95 3040.023 ms。当前结论是原生外壳约束同时取得“固定格式可靠性”和“adaptive 策略分布保持”，优于纯 JSON 模式。
B0 固定集延迟为 mean 2254.239 ms、p50 2376.383 ms、p95 3183.133 ms；与 R3 一样，128 个 rollout 无 policy/schema/未知参数/受保护参数错误。

## 23. 本轮代码验证

新增或扩展测试覆盖：

- action 名称与 arguments schema 分支绑定；
- 空 action 集拒绝；
- native Qwen envelope 与纯 JSON strict parsing；
- 纯 JSON backend 接收状态级 schema；
- Qwen envelope 正则包含完整外壳；
- 未启用 backend 时明确失败；
- rollout latency 的 mean / p50 / p95 插值计算。

Agentic + Evaluation 专项回归：`201 passed`，仅保留 1 条第三方 LangGraph pending deprecation warning。相关 Ruff 检查全部通过。

## 24. 当前晋升解释

本轮结果不能把 R3 重新解释为“模型原生晋升”：原生模型正式门仍记录 clarification 从 100% 到 93.75%，因此 R3 adapter 单独状态仍为 rejected，不启动 GRPO。

但已经得到一个新的系统候选：`R3 + qwen_tool_envelope`。它在当前本地 Transformers/Outlines 实现上消除了固定格式失败，保持 adaptive 原生成功率，并维持 unknown/protected argument error 为 0。该组合能否进入线上候选，必须经过下一阶段的 vLLM 版本升级、Hermes parser、真实 OpenAI-compatible tool call 集成测试和延迟预算门，不能直接把离线 Outlines 结果等同于生产发布。

## 25. 证据回收与运行状态

九组结构化解码报告已从 AutoDL 回收到：

- `ml/agentic/reports/structured-decoding-r1/`；
- 原始压缩归档：`ml/agentic/reports/structured-decoding-r1-reports.tgz`；
- 归档 SHA-256：`c31688671428e9ceef25e685e3dded4dc7c4e683c0670bcebbbfd23a46a18446`。

归档包含 B0/R3 的纯 JSON 固定集、纯 JSON adaptive 集、Qwen envelope smoke、Qwen envelope adaptive 正式集和 Qwen envelope 固定正式集。AutoDL 与本地归档哈希一致。

最终本地回归仍为 `201 passed`，Ruff 与 `git diff --check` 通过。AutoDL 最终代码同步归档 SHA-256 为 `95739f4bdd430108573553bb4265f7c3990f5bbce7cf16960181cc2d72c90d8e`，远端校验一致；远端没有残留 screen 会话或 GPU 计算进程。
