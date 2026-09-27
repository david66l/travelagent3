# Step 3：H-001 / H-004 同栈重算与历史只读回放

- 日期：2026-09-03
- 状态：已完成（v7 严格同栈比较通过；H-004 No-Go）
- 范围：仅 `internal_dev`；不训练、不调参、不访问 promotion-val 和 sealed test
- 远程隔离目录：`/root/autodl-tmp/TravelAgent2-h005-eval-20260903`
- 远程最终结果目录：`artifacts/native-react-posttraining/step3-h005-same-stack-v7-20260903`
- 远程排错保留目录：`artifacts/native-react-posttraining/step3-h005-same-stack-20260903`

## 1. 本步要回答的问题

H-005 把动作对应的固定连接词交给系统组装。本步必须把三件事分开：

1. 模型按旧词面合同能过多少：`raw_model_legacy_success`；
2. 模型不靠固定连接词时，动作、证据和语义本身能过多少：
   `raw_model_contract_success`；
3. 经过生产入口组装后的最终输出能过多少：`assembled_system_success`。

第三项是产品栈效果，不得冒充模型能力，也不得单独用于晋升。

## 2. 冻结的同栈配置

| 项目 | 固定值 |
|---|---|
| corpus | `verifier-reason-quality-sft-v14-canonical-p0/internal_dev_grpo.jsonl` |
| corpus SHA-256 | `fe0a06f604456f0e0d1c1725c7934b8bf95cca3ba1b9d70a30d28f36a069fc33` |
| 题数 | 80 |
| 每题采样 | 4 |
| 总 rollout | 每个 checkpoint 320 |
| seed | `20260910` |
| seed 派生 | `sha256-task-sample-v1` |
| temperature | `0.8` |
| max new tokens | `192` |
| quantization | 4-bit |
| render | `qwen3-agent-prefix-preserving.v1` |
| chat template SHA-256 | `dfe4e379b6439a9f01e881660c5f1ea57cd8026a8831553492723b38d15c9e63` |
| reason assembly | `repair-reason-assembly.v1` |

权重固定为：

- H-001：`h001-canonical-h48-seed20260923/checkpoint-48`，adapter SHA-256
  `734a7fcdea9509071fddb07ce7c9d3020bad1ed35690b81e6d46c92d98019924`；
- H-004：`h004-dpo-reason-repair-on-r2`，adapter SHA-256
  `f59f02f204ad117ff9d70f6fab3ecdcd9e9fa75d4b235454ba17741d896fb85b`。

关键 H-005 文件已从 E 盘逐字节同步到隔离目录；远程原训练目录保持不动。服务器为
RTX 4080 SUPER 32GB，开跑前 GPU 空闲。

## 3. 历史输出只读 replay

工具：`scripts/replay_reason_assembly.py`。

正式回放会强制检查：

- 历史 `report.json` 必须存在，corpus SHA 和 split 必须相同；
- 任务集合、来源状态、目标动作必须与 corpus 完全相同；
- `(task_id, sample_index, rollout_seed)` 不得重复；
- 每题采样数和 sample index 必须完整；
- 每条 seed 必须能由冻结公式重新算出；
- 输入 rollouts 在回放前后 SHA-256 必须相同。

| 历史输出 | rollout | raw legacy | raw semantic contract | assembled system | 严格 connector-only 挽救 |
|---|---:|---:|---:|---:|---:|
| H-001 旧单采样 | 80 | 0.8875 | 0.9500 | 0.9500 | 5 |
| H-004 旧四采样 | 320 | 0.828125 | 0.915625 | 0.915625 | 28 |

回放不变量：

- action、非 reason 参数、证据/数值锚点、状态指纹和原始输出哈希均未改变；
- 回放没有执行模型和工具；
- 旧 v4 artifact 没保存完整 tool trace，也没把 rollouts SHA 写回 source report，
  因此完整工具轨迹明确记为 `unverifiable`，历史 replay 只算反事实诊断，不能算晋升证据；
- H-004 有 8 条无可用参数、3 条含内部词，H-005 均 fail closed，没有用连接词掩盖；
- “connector-only 挽救”已收紧为：raw legacy 失败、raw semantic 成功、assembled system
  成功，而且动作、非 reason 参数和证据不变量全部通过。

## 4. 新的真实同栈四采样

执行命令使用同一个 evaluator，只替换 checkpoint。H-001 与 H-004 输出还会由
`scripts/compare_h005_same_stack.py` 强制检查相同 corpus、seed、采样数、temperature、
量化、模板、route schema、runtime、authority transport、状态指纹、实际模型输入 token
哈希、评测代码快照和逐条配对 key。

| checkpoint | raw legacy | raw semantic contract | assembled system | pass^4 |
|---|---:|---:|---:|---:|
| H-001 + H-005 | 0.821875 | 0.943750 | 0.943750 | 0.837500 |
| H-004 + H-005 | 0.709375 | 0.853125 | 0.853125 | 0.662500 |

## 5. 独立专家审查后的修正

独立审查给出的阻断项已处理：

- 严格收紧 connector-only recovery，不能把动作错、越权或语义错算成连接词收益；
- 历史 replay 必须有来源报告，并拒绝重复、缺失、错 seed、错来源状态；
- 不再把“没执行工具”写成“完整工具轨迹 100% 不变”；不可恢复的旧证据明确记为
  `unverifiable`；
- 新增同栈 comparator，强制逐条配对并生成 McNemar、source-state clustered bootstrap
  95% CI 和 `pass^4`；
- comparator 锁死正式 internal-dev 的 SHA、80 个任务、10 个独立 source 和每个 source
  的 `abort/propose_tradeoff/retry_solve = 1/6/1` 组成；
- checkpoint 身份不再只相信报告，比较时会重新哈希真实
  `adapter_model.safetensors`；
- 每条 rollout 必须有且只有一条有效推理计量；token/延迟缺失、负数或非有限值均拒绝；
- 每条保存实际模型输入 token 序列 SHA，报告保存 10 个关键 evaluator/Agent 文件的
  合并代码快照；两臂必须逐条、逐文件一致；
- assembled system 始终与 raw model 分栏，报告保持
  `production_promotion_eligible=false`。

## 6. v5/v6 预检发现的问题与处置

第一次 H-001 v5 运行完成 320 条后，严格预检发现 3 条解析失败 rollout 虽正确计为失败，
但 `PolicyOutputError` 分支没有保留已经发生的 GPU 推理计量。质量分没有因此变高，但证据链
不满足“每条请求均可审计”。第一次 H-004 随即停止，避免继续消耗 GPU。

修复方式：`LocalCheckpointAgentPolicy` 在生成结束、解析开始前就保存 inference metrics；
即使工具调用解析或参数校验失败，rollout 也会保留同一条计量。该修复先进入 v6；旧 v5
H-001 和未完成 H-004 只保留作排错证据，禁止进入最终 comparator。

第二轮 v6 启动前的专家复审又指出：只哈希 10 个关键文件仍不是完整代码闭包，只哈希
adapter 权重也没有绑定底模和 tokenizer。v6 因此在首题阶段主动停止，同样不得进入最终
比较。v7 的最终修复为：

- 代码快照覆盖 `backend/src/**/*.py` 与 `scripts/**/*.py`，远程共 392 个文件；
- 有效模型清单同时哈希 adapter config/weights、底模 config/全部本地权重和 tokenizer
  文件；
- 加载生产 chat template 后，再从运行时 tokenizer 计算 backend tokenizer、special
  tokens、added vocab 和最终 chat template 的身份；
- H-001/H-004 远程预检确认底模相同、有效 tokenizer 相同、adapter 不同；
- prompt SHA 必须是严格 64 位十六进制，代码快照漂移和跨臂 prompt 漂移均有拒绝测试。

独立专家最终结论为 `Go for Step 3 internal diagnostic`；前提是只使用当前 v7 两臂重跑
并由 comparator 全量通过，不能把该结论解释为 promotion。

v7 运行记录：

- H-001 PID：`53746`；
- H-004 自动启动 watcher PID：`53748`；
- evaluator 远程代码快照：
  `0f6fd5f9d5f3020510914b213f56589d3f14db12ff84216346e504621ab9dff0`；
- 共同底模 manifest：
  `130a9c1b23d88ff41cb3b938083455ff880d555b414260042806006550e4e35f`；
- 共同有效 tokenizer manifest：
  `e81f6adc78d0e15020897abdd37f1cb654999bacf2aa3c62fe32641a95bcc7e1`；
- 最终仍以两个 v7 报告内记录并由 comparator 重新计算的快照为准。

## 7. Step 3 退出条件

- 两个 checkpoint 的 320-rollout 新评测都完整结束；
- 同栈 comparator 的所有合同与逐条配对检查通过；
- raw / semantic / assembled / pass^4 和配对 CI 全部落盘；
- 日志与失败分类写入本记录；
- 本步只形成 internal-dev 诊断结论，不触发训练或晋升。

以上五项均已满足。严格 comparator 返回成功，`contract_equality` 与所有
`pair_invariants` 全部为 `true`，320 个 rollout、80 个任务逐条配对完整。

## 8. 最终统计结论

所有 delta 均为 `H-004 - H-001`。正式区间使用 10 个 source state 的 paired
source-cluster bootstrap；McNemar 只作描述性辅助。

| 指标 | H-001 | H-004 | paired delta | 95% CI |
|---|---:|---:|---:|---:|
| raw legacy success | 0.821875 | 0.709375 | -0.112500 | [-0.193750, -0.034375] |
| raw semantic contract | 0.943750 | 0.853125 | -0.090625 | [-0.146875, -0.040625] |
| assembled system | 0.943750 | 0.853125 | -0.090625 | [-0.146875, -0.040625] |
| semantic/system pass^4 | 0.837500 | 0.662500 | -0.175000 | [-0.325000, -0.037500] |

H-004 的语义/系统成功率少 29/320，pass^4 少 14/80；总体区间完全低于 0，不能解释为
随机波动。分动作看：

| target | H-004 - H-001 semantic/system | 95% CI | 结论 |
|---|---:|---:|---|
| abort | +0.050000 | [-0.050000, 0.175000] | 不确定 |
| propose_tradeoff | -0.083333 | [-0.150000, -0.029167] | 明确退步 |
| retry_solve | -0.275000 | [-0.450000, -0.100000] | 明确大幅退步 |

H-001 的 18 条 system failure 中，6 条为错误动作、12 条为正确动作但 grounding/具体性
不足；H-004 的 47 条中，13 条为错误动作、33 条为 grounding/具体性不足、1 条为语言问题。
H-004 不能作为统一 checkpoint 或 H-006 起点。

H-005 分别严格挽救了 39 条 H-001 和 46 条 H-004 的固定连接词失败。H-004 挽救数更高
不是模型更强，而是它原始词面失败更多。raw semantic 与 assembled system 完全相等，说明
系统组装只修固定表达，没有掩盖动作或语义错误。

成本仅作运行顺序相关的描述性观测：H-004 平均少 2.025 completion tokens，但平均慢
134.844 ms；不得据此单独判断模型优劣。

## 9. Gate 拆分与裁决

冻结的原始 `h005_safety_gate` 必须如实保留为 `FAIL`，不得在看到结果后改分：

- H-001 assembly coverage 0.981250，target-row hydration 0.978571；
- H-004 assembly coverage 0.959375，target-row hydration 0.953571；
- 两臂 model contract compliance 均为 1.0、override 均为 0、eligible assembly exact
  均为 1.0。

缺口来自模型选择 `get_poi_detail` 等非 repair target，而不是 connector 组装错误。系统若为
凑 100% 覆盖而把错误动作强改成目标动作，反而会越权。经两位独立专家复核，裁决分三层：

- `overall frozen readiness gate = FAIL`：当前结果不能 production promotion；
- `H005 connector component = Conditional Go`：合法进入组装的样本 100% 精确，且不越权；
- `H004 checkpoint = No-Go`；`Step 4 internal preparation from H001 = Go`。

该拆分只用于失败归因和预登记后续评测，不追溯修改本次冻结总门结果。promotion 与 sealed
继续保持关闭。

## 10. 产物与完整性

远程目录：
`/root/autodl-tmp/TravelAgent2-h005-eval-20260903/artifacts/native-react-posttraining/step3-h005-same-stack-v7-20260903`

E 盘已同步 `comparison.json`、两臂 `report.json` 和三份运行日志到：
`artifacts/native-react-posttraining/step3-h005-same-stack-v7-20260903`。完整 rollout
保留在远程目录，未在本地重复存放。

| 产物 | SHA-256 |
|---|---|
| `h001/report.json` | `ff8b91c284b3395be4608d898fabedccb07dd35de0ff6070c46b682f5733ead7` |
| `h001/rollouts.jsonl` | `999160f4a3be83d34d9c1c119c1e6c2dabd86d7651712550f15619b12786d628` |
| `h004/report.json` | `e91d2d1ff868bf2e858cad2cde55f2f266cbdd611f2413ebdf939ae00f94defb` |
| `h004/rollouts.jsonl` | `457c3d565e8cdf001e700bf849ab5905762cdccdee7db136e163a63adba7716f` |
| `comparison.json` | `ae39800c95f7150c0ff7a673fa69d011abc7e74ba301d0b71c712533aaaee358` |

两个 v7 报告的代码快照同为
`0f6fd5f9d5f3020510914b213f56589d3f14db12ff84216346e504621ab9dff0`，覆盖 392 个
Python 文件；共同底模和有效 tokenizer 身份分别为 `130a9c...e35f` 与 `e81f6a...cc7e1`。
两臂各 320/320 条均有且只有一份 inference metrics、合法 prompt SHA，且 generation audit
与 inference metrics 逐条相等。

本地冻结代码验证：142 passed，2 skipped；跳过项均为可选依赖条件，不是失败。
