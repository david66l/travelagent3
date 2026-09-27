# H006 内部同栈评测运行记录

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent run mode
- Mode: run
- Date: 2026-09-04
- Verification Status: COMPLETED H006 ONLY / EXIT 0
- Version Label: h006-internal-agent-eval-v1-20260904
- Decision: SELECT H006 checkpoint-32；NO-GO H006 step-34

## 白话目的

先做动作决策层的严格同栈考试。按用户要求不重跑已经评测过的 H001，本轮只跑
H006 checkpoint-32 和 H006 step-34。两个 checkpoint 使用同一批 80 道 internal-dev
题、同一随机种子和同一套 H005 系统组装，各跑 4 次，每个模型共 320 次 rollout。
这里不打开 promotion-val 或 sealed-160。

这一轮选出值得进入完整多轮 Agent Loop 的 H006 checkpoint；它本身不是最终晋升证明。

## 冻结输入

- corpus：`/root/autodl-tmp/corpora/verifier-reason-quality-sft-v14-canonical-p0/internal_dev_grpo.jsonl`
- corpus SHA256：`fe0a06f604456f0e0d1c1725c7934b8bf95cca3ba1b9d70a30d28f36a069fc33`
- 当前评测代码快照：`ca035e5ce8e70906b36d055c94ed92c88247d10e72d8e5ab8526905ebf5e86b6`，407 个 Python 文件
- H001 adapter SHA256：`734a7fcdea9509071fddb07ce7c9d3020bad1ed35690b81e6d46c92d98019924`
- H006 checkpoint-32 SHA256：`7a26396eb6d8a4ce477431d78be39d1362bb7173bd29685634936e02fddc25cd`
- H006 step-34 SHA256：`3aabbe1f65c3ff962e659798ffd4ba2569736b1e0a2a692ec061b391a85bb515`
- GPU 预检：RTX 4080 SUPER 32 GB，空闲
- CLI 预检：evaluator 和 comparator 均可加载
- 聚焦测试：35 passed，1 skipped；跳过项为可选条件

## 冻结评测配置

- split：`internal_dev`
- 80 tasks × 4 samples = 320 rollouts / model
- seed：`20260910`，逐题 seed 使用 `sha256-task-sample-v1`
- temperature：0.8
- max new tokens：192
- quantization：4-bit
- evaluator：`scripts/evaluate_tradeoff_grounding_sft.py`
- 每个模型硬超时：30 分钟
- 运行顺序：H006 checkpoint-32 → H006 step-34
- 两臂完成前禁止改动 `backend/src/**/*.py` 和 `scripts/**/*.py`

## 已确认的评测命令

用户在 2026-09-04 明确要求复用已完成的 H001 历史评测，不重跑 H001。本轮只执行
H006 checkpoint-32 与 H006 step-34。由于旧 H001 的完整代码快照不同，旧结果只能作
历史参照，不能冒充本轮同批次严格对照；H006 两个 checkpoint 之间仍可严格逐条配对。

共同环境：

```bash
cd /root/autodl-tmp/TravelAgent2-h005-eval-20260903
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/autodl-tmp/TravelAgent2-h005-eval-20260903/backend/src:/root/autodl-tmp/TravelAgent2-h005-eval-20260903
mkdir -p artifacts/native-react-posttraining/h006-internal-agent-eval-v1-20260904
```

H001（本轮不执行，仅保留原预案）：

```bash
/root/miniconda3/bin/python scripts/evaluate_tradeoff_grounding_sft.py \
  --checkpoint /root/autodl-tmp/TravelAgent2-verifier-repair/artifacts/native-react-posttraining/h001-canonical-h48-seed20260923/checkpoint-48 \
  --internal-dev-corpus /root/autodl-tmp/corpora/verifier-reason-quality-sft-v14-canonical-p0/internal_dev_grpo.jsonl \
  --splits internal_dev \
  --output-dir artifacts/native-react-posttraining/h006-internal-agent-eval-v1-20260904/h001 \
  --samples-per-task 4 --temperature 0.8 --max-new-tokens 192 --seed 20260910 --load-in-4bit \
  2>&1 | tee artifacts/native-react-posttraining/h006-internal-agent-eval-v1-20260904/h001.log
```

H006 checkpoint-32：

```bash
/root/miniconda3/bin/python scripts/evaluate_tradeoff_grounding_sft.py \
  --checkpoint artifacts/native-react-posttraining/h006-internal-corrective-sft-v2-seed20260930/checkpoint-32 \
  --internal-dev-corpus /root/autodl-tmp/corpora/verifier-reason-quality-sft-v14-canonical-p0/internal_dev_grpo.jsonl \
  --splits internal_dev \
  --output-dir artifacts/native-react-posttraining/h006-internal-agent-eval-v1-20260904/h006-checkpoint-32 \
  --samples-per-task 4 --temperature 0.8 --max-new-tokens 192 --seed 20260910 --load-in-4bit \
  2>&1 | tee artifacts/native-react-posttraining/h006-internal-agent-eval-v1-20260904/h006-checkpoint-32.log
```

H006 step-34：

```bash
/root/miniconda3/bin/python scripts/evaluate_tradeoff_grounding_sft.py \
  --checkpoint artifacts/native-react-posttraining/h006-internal-corrective-sft-v2-seed20260930/checkpoint-34 \
  --internal-dev-corpus /root/autodl-tmp/corpora/verifier-reason-quality-sft-v14-canonical-p0/internal_dev_grpo.jsonl \
  --splits internal_dev \
  --output-dir artifacts/native-react-posttraining/h006-internal-agent-eval-v1-20260904/h006-step-34 \
  --samples-per-task 4 --temperature 0.8 --max-new-tokens 192 --seed 20260910 --load-in-4bit \
  2>&1 | tee artifacts/native-react-posttraining/h006-internal-agent-eval-v1-20260904/h006-step-34.log
```

每个命令单独监控 PID、日志增长、GPU、rollout 进度和输出文件；异常不自动重试，
只有单臂超过 30 分钟才终止。两臂均成功后运行严格 paired comparator，比较 raw model、
semantic contract、assembled system、pass^4、分动作表现和 source-clustered 95% CI。

## 运行结果

- checkpoint-32：2026-09-04 12:40:13 +08:00 启动，PID 104111；13:06:12 完成，退出码 0，约 26 分钟。
- step-34：约 13:06:30 启动，PID 105520；13:31:27 完成，退出码 0，约 25 分钟。
- 严格比较：13:32:21 完成，退出码 0。
- 两臂均为 320/320 条 rollout；运行期间无报错、无超时、无自动重试。
- 评测契约的 16 项相等性检查全部通过；320 条逐条配对的 6 项不变量全部通过。

| 指标 | checkpoint-32 | step-34 | 结论 |
|---|---:|---:|---|
| 系统完整成功率 | 94.6875% | 93.4375% | checkpoint-32 高 1.25pp |
| pass^4（同题 4 次全成功） | 85.0% | 82.5% | checkpoint-32 高 2.5pp |
| 旧版 raw 完整成功率 | 76.5625% | 75.0% | checkpoint-32 高 1.5625pp |
| 旧版 raw pass^4 | 48.75% | 38.75% | checkpoint-32 高 10pp |
| 动作准确率 | 99.0625% | 98.4375% | checkpoint-32 更高 |
| policy 输出错误率 | 0.625% | 1.25% | checkpoint-32 更低 |
| 合同合规率 | 100% | 100% | 都通过 |
| 控制器越权率 | 0% | 0% | 都通过 |
| abort 系统成功率 | 85.0% | 87.5% | step-34 高 2.5pp |
| tradeoff 系统成功率 | 97.0833% | 95.0% | checkpoint-32 高 2.0833pp |
| retry 系统成功率 | 90.0% | 90.0% | 持平 |

严格比较中的差值均为“step-34 减 checkpoint-32”：

- 系统完整成功率差值 -1.25pp，source-state 聚类 bootstrap 95% CI 为 [-3.4375pp, +0.9375pp]。
- pass^4 差值 -2.5pp，95% CI 为 [-10pp, +5pp]。
- 旧版 raw pass^4 差值 -10pp，95% CI 为 [-20.03125pp, -1.25pp]，这项明确支持 checkpoint-32。
- tradeoff 差值 -2.0833pp，95% CI 为 [-4.1667pp, 0pp]；step-34 的主要退步来自这里。
- step-34 平均单次请求快约 202 ms，但质量下降，不能用这点速度收益换掉稳定性。

因此本轮选择 **H006 checkpoint-32** 进入下一关，H006 step-34 判为 **No-Go**。
这里的选择是“两个 H006 checkpoint 谁更好”，不等于已经允许上线。

## H001 历史结果怎么用

旧 H001 不重跑，只作历史参照。它的系统完整成功率为 94.375%，pass^4 为 83.75%，
abort / tradeoff / retry 分别为 92.5% / 97.5% / 77.5%。checkpoint-32 的对应点估计为
94.6875%、85.0%、85.0% / 97.0833% / 90.0%：整体和 pass^4 略高，retry 明显更好，
但 abort 低 7.5pp，是下一关必须重点检查的风险。

这组 H001 与 H006 数字不能当严格 A/B：旧 H001 的评测代码快照为
`0f6fd5f9d5f3020510914b213f56589d3f14db12ff84216346e504621ab9dff0`，本轮为
`ca035e5ce8e70906b36d055c94ed92c88247d10e72d8e5ab8526905ebf5e86b6`。审计发现 4 个既有
Python 文件发生变化、15 个文件新增；其中 `backend/src/agentic/local_policy.py` 属于实际评测
运行路径，所以严格 comparator 拒绝跨快照比较是正确行为。

## 结果文件与校验

- `h006-checkpoint-32/report.json`：`838a15f033d729b703dc9ad8dbc459e4a94165c8dde74d8d7f7aa821f2000596`
- `h006-checkpoint-32/rollouts.jsonl`：`d34046da3d3310e562567933f4965e341aab407ae7aaa09e67eb61c5f7c26fa5`
- `h006-step-34/report.json`：`db458fed4a0cd5f6a74f16a35529132af8dd59bbf9d1494ead9e13c2cc22c774`
- `h006-step-34/rollouts.jsonl`：`068c9b72676d919d7127ae07f81e9d53abb2c8384af15d77791b2fb4e1359717`
- `comparison-checkpoint32-vs-step34.json`：`b9867af592eabf511a62e12027bdb9fa287b790640ef9fff36147d56c51fee41`
- `h006-checkpoint-32.log`：`4267ff59de2a408f965e00d57c2d1dfb80a8a2157b524ed1e6c406e61864e5fe`
- `h006-step-34.log`：`fc0acc75303571b7c0d3e7718ca40fbf43473e586e5260edd002fed96ca358bf`

## 下一道门

下一步只让 checkpoint-32 参加真实多轮工具执行的完整 Agent Loop 评测，重点看 abort、
重试后能否收敛、工具错误恢复、预算耗尽和是否会做越权动作。当前 production promotion
仍为 false；原因是冻结的 H005 安全门要求组装覆盖和 hydration 都达到 100%，而本轮
checkpoint-32 在 comparator 口径下分别约为 99.0625% 和 98.9286%，不能把“接近 100%”写成“已经通过”。
只有同栈决策层和多轮工具层都通过，才允许准备 promotion campaign。
