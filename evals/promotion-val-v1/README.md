# Promotion Val v1

这是 H-005/H-006 后训练的独立晋升协议目录。当前只冻结了规则，**没有生成或打开任何
晋升题**，因此 H-006 GPU 训练仍是 No-Go。

## 白话规则

1. 平时调模型只能看 `train-shadow` 和 `internal-dev`。
2. `promotion-val-v1` 每个实验族只能统一运行一次，运行前模型、代码、参数、seed 和门槛
   必须全部锁定。
3. H-001、H-004、H-006 必须使用相同的 H-005 parser/render/tool snapshot 重跑，不能拿
   新规则下的候选对比旧报告里的基线。
4. 每题跑四次，但统计样本量仍按独立 `source_state_id` 算，不能把一次题的四次回答冒充
   四道题。
5. 旧 hard challenge、所有 internal-dev、当前 promotion 和 sealed-160 及其同源改写都
   不得进入训练。
6. promotion 失败后，不允许根据结果修改同一实验族再看一次。
7. sealed-160 仍由 `evals/native-react-hard-v2/test.jsonl` 承担，继续封存；本目录中的审计
   工具只读取公开协议和 hash/lineage 元数据，不读取 sealed 内容。

## 已冻结的晋升门

- 四采样总体成功率至少 0.90；
- 相对 `H-001+H-005` 的配对 95% CI 下界必须大于 0；
- hard 相对 H-001 至少提升 10pp，且 CI 下界大于 0；
- hard 相对 `H-004+H-005` 最多退 2pp；
- easy 和 `pass^4` 相对 `H-001+H-005` 最多退 2pp；
- contract 必须 100%，越权动作必须为 0；
- LLM judge 只作辅助描述，不负责最终晋升。

## 数据冻结前必须完成

- 用 internal-dev 的 paired discordance 做功效分析，确定最终独立状态数量；200 只是暂定值；
- 每个状态具备 `source_state_id/source_hash/template_lineage/tool_snapshot_hash` 等完整血缘；
- 与所有 train、train-hard-donor、train-shadow、internal-dev、旧 hard challenge 和 sealed
  做血缘隔离审计；
- 建立 hash-chain access log；
- 记录数据文件 SHA-256，但审计报告不得泄露 protected case 文本。

## 文件

- `protocol.json`：冻结评分、统计、数据隔离和访问规则。
- `protocol.sha256`：协议文件字节哈希；修改协议必须显式升级版本，不能覆盖后继续沿用 v1。
- `readiness.json`：当前准备状态；可以随证据推进更新，但必须始终引用冻结的协议哈希。
- `access-log.jsonl`：第一次 promotion campaign 前由授权工具创建；当前不存在表示零次访问。
- `lineage.jsonl`：数据完成后生成；当前不存在，因此 `dataset_ready=false`。

## 审计

只检查已冻结协议：

```powershell
backend\.venv\Scripts\python.exe scripts\audit_posttraining_promotion_protocol.py
```

要求数据也已经准备好：

```powershell
backend\.venv\Scripts\python.exe scripts\audit_posttraining_promotion_protocol.py `
  --lineage evals\promotion-val-v1\lineage.jsonl `
  --access-log evals\promotion-val-v1\access-log.jsonl `
  --require-dataset-ready
```

未完成 lineage 和功效分析前，第二条命令必须失败，这是预期的 fail-closed 行为。
