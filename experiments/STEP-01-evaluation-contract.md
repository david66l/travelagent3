# 五步计划 Step 1：评分规则与验证集隔离

> 状态：完成  
> 日期：2026-09-03  
> 范围：只冻结协议与保护机制；未生成 promotion 题，未读取 sealed-160，未启动训练

## 白话结论

评分规则已经锁住，程序现在会阻止：

1. 训练数据和验证题来自同一个原始状态；
2. 同一个实验族第二次打开 promotion 集；
3. 规则或配置没冻结就打开 promotion；
4. 数据、功效分析、血缘检查没完成就误授权 H-006；
5. 把系统补出来的格式分冒充模型能力分。

当前项目已有 `native-react-hard-v2`：40 道 Dev 已用于开发，160 道 Test 继续封存。
新的 `promotion-val-v1` 是夹在 Dev 和最终 Test 之间的一次性晋升层。目前只建立了规范，
还没有生成题目，所以 H-006 训练开关仍然关闭。

## 冻结规则

- 主指标：`full_success_mean_4 >= 0.90`；
- 每个独立 `source_state_id` 固定 4 次 rollout；四次不算四个独立样本；
- 相对 `H-001+H-005` 的配对 95% CI 下界必须大于 0；
- hard 相对 H-001 至少 +10pp，且不能比 `H-004+H-005` 低超过 2pp；
- easy 和 `pass^4` 不能比 `H-001+H-005` 低超过 2pp；
- contract 100%，越权动作 0；
- H-001/H-004/H-006 必须用同一 parser/render/tool snapshot 重跑；
- raw model 与 assembled system 分开报告；
- LLM judge 不能决定晋升；
- 每个实验族只能打开 promotion 一次；
- sealed-160 总共只能做一次最终验收。

## 数据隔离

训练侧与保护侧必须按以下字段检查：

```text
source_state_id
source_hash
template_lineage
tool_snapshot_hash
generator_model
generator_prompt_hash
parent_trajectory_id
```

永久禁止回流训练：

- H-003/H-004 已用 hard challenge；
- 所有 internal-dev；
- promotion-val-v1；
- sealed-160；
- 上述数据的同源改写、槽位替换和失败轨迹派生数据。

审计只输出冲突对象的 hash，不输出保护题文本和 case id。

## 已实现文件

- `evals/promotion-val-v1/protocol.json`：冻结指标、统计和访问规则；
- `evals/promotion-val-v1/protocol.sha256`：协议字节哈希；
- `evals/promotion-val-v1/readiness.json`：当前准备状态，引用冻结协议哈希；
- `evals/promotion-val-v1/README.md`：白话使用说明；
- `backend/src/evaluation/posttraining_promotion_protocol.py`：协议、血缘、访问链审计；
- `scripts/audit_posttraining_promotion_protocol.py`：只读审计入口；
- `scripts/record_promotion_access.py`：授权后才可写入 hash-chain 访问事件；
- `backend/tests/unit/evaluation/test_posttraining_promotion_protocol.py`：协议防护测试。

冻结协议文件 SHA-256：

```text
f0d8f53dd7014ed5c8f59313b76f9888c1a62798a3d73157c413f098819435c0
```

## 验证结果

```text
py_compile: passed
unit tests: 7 passed
protocol audit: passed
protected_case_payloads_read: false
dataset_ready: false
h006_gpu_training_authorized: false
--require-dataset-ready: exit 3 (预期的 fail-closed)
未授权 promotion 打开测试: exit 1，且没有生成 access log
```

## 尚未完成但不属于本步的内容

- promotion-val-v1 题目尚未制作；
- paired-discordance 功效分析尚未完成；
- lineage.jsonl 尚未生成；
- H-005 connector assembly 尚未实现；
- H-001/H-004 尚未在 H-005 同栈重测；
- H-006 没有获得 GPU 训练授权。

下一步进入五步计划 Step 2：固定连接词由系统统一组装，并把模型原始分和系统最终分分开。
