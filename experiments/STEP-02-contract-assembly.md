# 五步计划 Step 2：固定连接词组装与双指标

> 状态：完成  
> 日期：2026-09-03  
> 范围：只实现 H-005 合同层和评测记账；未评测 H-001/H-004，未启动训练，未读取 promotion/sealed 数据

## 白话结论

模型以后不用死记三段固定开头。模型仍然负责三件真正有能力含量的事：

1. 选对 `retry_solve / propose_tradeoff / abort`；
2. 说清楚当前真实证据；
3. 不能在解释里主张另一个动作。

程序只根据已经选定的动作补固定开头。模型原话和程序组装后的话同时保留、分别打分，
因此程序补一句正确套话不能把错误动作、错误证据或冲突解释伪装成模型提升。

## 生产链路

唯一组装顺序如下：

```text
模型 action + 原始 reason
        ↓
参数与 controller 权限校验
        ↓
私有实现词 / 明确动作冲突检查
        ↓
action -> canonical connector
        ↓
最终 arguments.reason + controller-owned 字段
```

组装发生在 Agent Loop 的中央授权入口，不在 evaluator 里临时改答案。线上执行和离线
TRL 评测都会经过同一实现。历史模型如果已经生成正确固定开头，组装函数是幂等的：不会
重复添加；若生成了另一个动作的固定开头，则直接失败。

冻结版本：

```text
repair-reason-assembly.v1
```

## 三类分数

- `raw_model_legacy_full_success`：旧口径，模型原话连固定连接词也必须生成；只用于和历史报告连续比较。
- `raw_model_contract_success`：新模型职责口径，要求动作、证据、具体信息、语言、公开措辞和语义一致；不要求模型背固定开头。
- `assembled_system_success`：最终系统口径，检查中央授权、controller 注入和系统组装后的完整结果。

正式报告还记录 `reason_assembly_exact_rate`。该值不是模型分数，只证明最终文本确实由冻结
函数从模型原话生成，没有 evaluator-only shortcut。

## 防止“套话假提升”

系统连接词不能改变以下内容：

- action；
- 模型原始 reason；
- 证据短语和日期、金额、时长等具体锚点；
- tool trace；
- controller 授权的 options/strategy。

以下情况不会因为补了连接词而通过：

- 没引用要求的可见证据；
- 漏掉第几天、多少分钟、多少元等具体信息；
- 中文证据却主要用英文包装；
- 泄露 `retry_solve`、验证器、控制器等内部实现词；
- 选择重算却明确写“必须停止”，或选择终止却明确要求用户做取舍。

## 已实现文件

- `backend/src/agentic/reason_quality.py`：单一组装函数、幂等处理、私有词和语义冲突检查、模型语义评分。
- `backend/src/agentic/policy_actions.py`：三类 repair action 在中央授权入口统一组装，原始参数留在 `model_arguments`。
- `backend/src/agentic/policy_prompts.py`：提示模型只写有证据的事实细节，不再要求生成固定套话。
- `backend/src/agentic/trl_environment.py`：同一次 rollout 分开计算 raw legacy、raw contract 和 assembled system。
- `scripts/evaluate_tradeoff_grounding_sft.py`：报告 schema 升级到 v5，输出三类分数和组装一致率。
- 对应单元测试：reason、权限边界、Agent Loop、TRL reward、评测报告。

## 验证结果

```text
相关回归测试：105 passed, 1 skipped
ruff：passed
git diff --check：passed
新验证：evidence-only raw reason -> raw legacy fail / raw contract pass / assembled system pass
新验证：ungrounded raw reason -> assembled system 仍失败
新验证：cross-action reason -> 中央授权前拒绝
新验证：legacy canonical reason -> 幂等，不重复连接词
```

扩大到全部 `backend/tests/unit/agentic` 后结果为 `583 passed, 4 skipped, 11 failed`。
其中 10 个失败已用未修改的 HEAD 代码单独复现，属于进入本步前已有的历史测试/数据合同漂移；
剩余 1 个依赖未进入 Git archive 的本地生成数据，无法在隔离 HEAD 副本中做等价复现。
本步相关的 5 个测试文件全部通过。第二步没有为清零历史红灯而顺手修改旧数据生成流程。

## 本步没有做的事

- 没有运行 H-001/H-004 checkpoint；
- 没有读取 `promotion-val-v1` 题目；
- 没有读取 sealed-160；
- 没有启动远程 GPU 或训练；
- 没有把 assembled 分数当成模型原生能力。

下一步是 Step 3：用同一套 prompt、parser、组装函数、采样参数和工具快照，重新评测
`H-001+H-005` 与 `H-004+H-005`，并对已有输出做只读离线 replay。
