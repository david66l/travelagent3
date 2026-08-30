# TravelAgent Stage33：真实 Shadow 与 Canary 晋级 SPEC v1.0

日期：2026-08-15  
状态：执行中  
前置阶段：Stage32 级联蒸馏与自适应路由

## 1. 目标

将 Stage32 的离线模型结论转化为可上线证据：Qwen3-1.7B SFT 作为冠军学生，
Qwen3-1.7B DPO SFT-reference v2 作为挑战者，Qwen3-8B 继续负责课程外复杂决策和
学生失败后的单次回退。在不影响用户结果的条件下，积累同输入、同控制器、同服务版本的
成对数据，最终决定 DPO 是晋级、继续 Shadow，还是淘汰。

Stage33 不再以训练 loss 或单次正确率作为主要成果，而以数据来源治理、成对统计、服务
稳定性、回滚能力和真实流量证据作为验收对象。

## 2. 当前冻结结论

- 生产冠军：`qwen3-1.7b-stage32-sft-replay-formal-v1`；
- Shadow 挑战者：`qwen3-1.7b-stage32-dpo-sftref-formal-v2`；
- 复杂教师：Qwen3-8B Base；
- 外部冻结集合同一致口径：SFT 与 DPO 均为 148/149；
- DPO 相对 SFT 的 P95 延迟比为 1.032x，HTTP 错误均为 0；
- 149 对的非劣效 95% 区间为 `[-2.51pp, +2.51pp]`，无法证明 1pp 非劣效；
- 冻结集不是线上流量，当前不具备 Canary 资格。

## 3. 运行拓扑

1. 用户可见结果继续走稳定确定性主链，Stage33 Agent 只运行在独立 Shadow worker；
2. 对每个进入模型决策的有界上下文，同时请求 SFT 冠军与 DPO 挑战者；
3. 只执行 SFT 冠军动作，DPO 动作、延迟、错误码与路由证据写入 `shadow_trace`；
4. DPO 超时、格式错误或服务不可用必须 fail-open，不得改变冠军动作；
5. 冠军失败仍按原有策略失败处理，禁止用挑战者静默掩盖冠军故障；
6. `complex` 决策仍由 8B 执行；学生推理失败最多回退 8B 一次；
7. 挑战者与冠军共用同一 1.7B Base 服务的两个 LoRA adapter，生产 Canary 前必须完成
   容量隔离或显式队列上限验证。

## 4. 三层证据

### 4.1 冻结外部评测

用于检验模型能力、控制器合同和历史可比性。不得写成 `live_shadow`，不得单独授权 Canary。

### 4.2 授权回放

使用无个人信息、可审计的完整 Agent 场景验证端到端行为。它可以验证系统质量门禁，但
`canary_evidence=false`。

### 4.3 真实 Shadow

只接收字段白名单和 PII 脱敏后的真实规划输入。每条用于晋级的记录必须同时具备：

- `evaluation_source=live_shadow`；
- `release_gate_eligible=true`；
- 冠军与挑战者输入哈希一致；
- `outcome_observed=true`，即动作已在隔离环境执行并经过确定性验证器或人工裁决；
- 相同 controller、tool schema、模型 adapter、推理参数与部署版本。

仅旁路生成但未执行的 DPO 动作只能用于分歧挖掘，不能作为效果正确率或 Canary 证据。

## 5. 晋级门禁

至少 300 个合同一致、完整成对且结果可观测的真实 Shadow 决策，并同时满足：

- DPO 合同正确率 ≥ 98%；
- DPO-SFT 成对正确率差的 95% 置信区间下界 ≥ -1pp；
- DPO HTTP/推理错误率 ≤ 1%；
- DPO P95 推理延迟 ≤ SFT 的 1.25 倍；
- route trace 覆盖率 100%；
- 学生到 8B 的推理失败回退率 ≤ 2%；
- 高风险 `tradeoff/abort` 单独审计，不允许被总体高分掩盖；
- 真实 Shadow 来源校验通过。

300 是开始判定的最小样本，而不是保证通过的固定样本。若非劣效区间仍跨越 -1pp，则继续
采集至区间收敛，最多 2000 对；仍不收敛则保持 SFT 冠军，不为追求结论重复或扩写样本。

## 6. 分阶段放量

- 0%：配置与单测，只验证挑战者故障不影响冠军；
- 5%：至少 300 个真实 Shadow 配对，独立 worker 并发 1；
- 20%：5% 门禁通过后观察队列、显存、P95 和失败率至少一个完整业务周期；
- 100% Shadow：只扩大观察面，仍不改变用户动作；
- 1% Canary：必须经本 SPEC 的真实证据门禁通过并保留一键回滚；
- 5%/20%/50% Canary：每一级重新检查硬约束、错误率、延迟和人工投诉；
- 100%：只有连续两个观察周期无回退才允许。

每次只能改变一个变量：采样率、模型版本或路由策略三者不可同时变化。

## 7. 回滚

Shadow 异常时先将 `AGENTIC_CHALLENGER_SHADOW_ENABLED=false`；Agent Shadow 整体异常时再将
`AGENTIC_SHADOW_SAMPLE_RATE=0`。任何回滚不得删除失败记录。Canary 期间出现硬约束回退、
错误率超门槛或 P95 超门槛，立即把用户流量恢复到 SFT 冠军。

## 8. 本阶段交付物

- `backend/src/evaluation/policy_shadow.py`：成对统计和晋级门禁；
- `scripts/build_stage33_policy_shadow_report.py`：可复现报告构建器；
- `ShadowComparingAgentPolicy` 与 `PolicyShadowTrace`：运行时旁路双推理；
- `scripts/export_stage33_policy_shadow_observations.py`：隐私最小化的分歧观测导出；
- `deploy/stage33-shadow.env`：不含凭据的部署合同；
- `ml/agentic/reports/stage33-policy-shadow-baseline-v1`：当前冻结基线；
- Stage33 工作日志、测试结果和内容哈希。

## 9. 当前决策

Stage33 基线已经证明 DPO 在生产控制器下没有显著退化，延迟也处于门槛内；但由于只有
149 个合同一致样本、1pp 非劣效区间未通过且不存在真实 Shadow 结果，DPO 继续保持
Shadow。SFT 仍是唯一冠军学生，不允许提前切换。
