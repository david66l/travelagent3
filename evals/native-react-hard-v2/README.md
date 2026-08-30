# Native ReAct Hard Benchmark v2

这是 TravelAgent 生产 ReAct 协议的 200 题冻结困难评测集。v2 是在 v1 的 Dev 冒烟审计后发布的作者修订版；v1 不得再用于模型比较。

## 为什么发布 v2

v1 的 40 题 Dev 首跑发现 6 个作者合同问题：普通题预算意外触发不可行、两道题缺少旅行天数、过期数据题没有显式时效查询、修改字段被标在错误约束域，以及英文“核实”被误标为“必去”。v2 修正题面和标签，不通过放宽评分掩盖模型错误。

## 数据划分

- `dev.jsonl`：40 题，用于评测工具开发、错误定位和 checkpoint 选择。
- `test.jsonl`：160 题，只允许最终冻结评测使用一次。
- `manifest.json`：文件哈希、生成代码版本、分层数量和训练污染审计。

Dev 使用北京、上海、广州、成都；Test 使用另外 16 个城市，城市没有重合。

## 场景矩阵

10 个能力族各 20 题：澄清、POI 事实检索、时效信息、交通班次、硬约束冲突、无障碍需求、工具恢复、用户修改、安全终止和噪声/多语言输入。共覆盖 20 个城市、40 个语义簇；难度分布为 L2 40、L3 100、L4 60。

## 评测纪律

- 运行前必须校验 split 文件 SHA-256；字节变化即拒绝执行。
- 默认只打开 Dev；Test 必须显式传入 `--allow-frozen-test`。
- SFT、DPO、GRPO 数据构建脚本不得读取本目录。
- 同题重复采样只表示随机稳定性，不增加独立样本量。
- 报告必须给出任务级结果和按 `cluster_id` 重采样的 Bootstrap 置信区间。

这些任务由确定性程序组合生成，尚未完成双人独立标注。可以声称“200 个不同冻结任务、20 城市、40 语义簇、完成训练污染审计”，不能声称“200 道独立人工题”。

## Dev 参考基线（2026-08-30）

使用已配置的 `deepseek-v4-flash` 作为远程策略模型，在生产 Native ReAct 全链路上完成 40 题 Dev 评测：

- 任务通过率：39/40（97.5%），按语义簇 Bootstrap 的 95% CI 为 92.5%–100%。
- 意图合同通过率：40/40；规划类任务最终成功率：35/36（97.22%）。
- 30 题实际进入 Verifier，其中 27 题最终硬通过（90%）；另外 2 题以有依据的安全终止正确收口，Verifier 场景解决率为 29/30（96.67%）。
- 故障注入：4/4 成功触发，4/4 在约定步数内恢复，覆盖超时、限流、空结果和过期数据。
- 平均每题 3.775 次策略模型调用、13.3 次逻辑工具调用、11,797.4 Token；平均延迟 9.42 秒，P95 17.31 秒。
- 唯一失败是英文复杂请求在 `MUST_VISIT_MISSING` 后错误触发全局重检索，导致 POI 详情由 8 个降为 5 个，最终以证据不足安全终止。该失败保留，作为 Verifier 失败后局部修复策略的后训练靶点。

该结果只用于验证评测器和建立远程模型参考线，不是 Qwen Base/SFT/GRPO 的效果，也不是最终 Test 成绩。完整运行报告保存在本地忽略目录 `outputs/native-react-hard/deepseek-v4-flash-dev-v2-frozen.json`；160 题 Test 尚未解封。

运行命令：

```powershell
$env:LANGSMITH_TRACING='false'
backend\.venv\Scripts\python.exe scripts\evaluate_native_react_hard.py `
  --output outputs/native-react-hard/deepseek-v4-flash-dev-v2-frozen.json `
  --split dev `
  --policy-model deepseek-v4-flash `
  --bootstrap-samples 10000 `
  --resume
```
