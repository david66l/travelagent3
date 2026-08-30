# 阶段 33：真实 Shadow 与 Canary 晋级工作日志

日期：2026-08-15  
状态：模拟用户 300 对回放完成；5% Shadow 继续等待真实流量证据

## 本轮目标

将 Stage32 的 1.7B SFT 冠军、DPO v2 挑战者与 8B 复杂教师接入可审计 Shadow，补齐
成对统计、非劣效判断、隐私最小化观测、部署镜像、队列隔离和回滚边界。

## 已完成

- 新增 Stage33 SPEC，冻结三层证据、最小 300 对、1pp 非劣效、错误率和延迟门槛；
- 实现 `policy-shadow-comparison.v1` 成对晋级器；
- 支持 McNemar 精确检验、成对差值 95% CI、P95 比值、HTTP 错误和动作族切片；
- 强制真实晋级数据包含 `live_shadow`、`release_gate_eligible=true` 与
  `outcome_observed=true`，仅修改 CLI 标签不能伪造 Canary 证据；
- 新增 `PolicyShadowTrace` 和 `ShadowComparingAgentPolicy`；
- SFT 与 DPO 并行推理，但只执行 SFT；DPO 失败不影响冠军，冠军失败不被挑战者掩盖；
- Agent 评测增加挑战者调用、失败率和动作分歧率；
- 新增隐私最小化观测导出器，不导出原始上下文、轨迹 ID 或挑战者参数；
- 修复 smoke 脚本误标 `live_shadow`，现在固定为 `synthetic_smoke` 且不参与发布门禁；
- 构建 `travelagent2-backend:stage33` 不可变镜像并通过 Compose 配置校验；
- 云端 8004 同时加载 1.7B Base、SFT、DPO v2，8002 保留 8B Base；
- SFT、DPO、8B 三个模型共 9 次实际请求全部正确，HTTP 错误为 0；
- 完整 Agent synthetic smoke 通过：硬约束通过，DPO 调用 1、失败 0、分歧 0；
- 授权回放 v2 完成 10/10：硬通过/有效草案/任务完成均 100%，DPO 调用 10、失败 0、
  分歧 0，Agent P95 5,098 ms。

## 发现并修复的问题

### 1. Smoke 来源污染

人工 smoke 原先使用默认 `live_shadow` provenance，可能污染 Canary 统计。现已显式改为
`synthetic_smoke`、`release_gate_eligible=false`。

### 2. Shadow 队列被普通 worker 抢占

Stage33 compose 首版没有限制普通 worker 的队列。授权回放 v1 前两条通过，第 3 条被默认
环境 worker 抢走并失败。现已固定普通 worker 监听 `default,planning`，独立 Shadow worker
只监听 `shadow`。v1 失败记录保留；新 batch v2 10/10 通过。

### 3. API policy identity 潜在异常

旧代码假设 `ApiAgentPolicy` 存在 `model` 字段。现改为依次安全读取 policy、client、settings，
并增加回归测试。

## 当前证据

### 冻结基线

- 合同一致配对 149，冲突排除 1；
- SFT/DPO 均为 148/149，独赢为 0/0，动作分歧 0；
- DPO/SFT P95 比值 1.032x；
- 95% CI 为 `[-2.51pp, +2.51pp]`；
- 样本不足且不是线上来源，继续 Shadow。

### 授权回放

- v2 10/10 完成，质量指标全部通过；
- DPO 旁路只证明结构、稳定性和分歧，不代表其动作已执行；
- 授权回放不具备 Canary 资格。

## 运行状态

- 本地 Stage33 Docker 栈在线，backend、gateway、frontend、Postgres、Redis、VRP 均健康；
- Shadow worker 并发 1，采样率 5%；
- 本机隧道仅绑定 `127.0.0.1:18004/18002`；
- 云端 1.7B 双 LoRA 与 8B 教师在线；
- Stage33 镜像 digest：
  `sha256:0ea4bd954d62cc0cc4cf8c8d7b9118a9affa1716e1c8e115362c54c28abe4e16`。

## 下一里程碑

等待真实规划请求进入 5% Shadow。累计至少 300 个完整、合同一致、结果可观测的配对后，
重新生成 Stage33 成对报告。若 95% 非劣效区间下界仍低于 -1pp，继续采集而不是扩写或重复
样本；达到门禁后才讨论 1% Canary。

## 2026-08-16：300 个模拟真实用户授权回放

- 完成批次 `stage33-simulated-users-v1`：300/300 请求执行完成，超时或执行失败 0；
- 数据库复核得到 600 条 completed 记录和 300 个严格配对，无 pending/running；
- Agent 硬通过率 99.33%、有效草案率 99.33%、任务完成率 99.52%、兜底率 0.67%；
- Agent P50/P95 为 4,211/13,658 ms，确定性基线 P95 为 15,605 ms；
- SFT 主策略调用 300 次，DPO 旁路调用 300 次，DPO 失败 0、动作分歧 0；
- 8B 教师未被调用：本批全部属于信息完整的 `search` 高频动作，没有复杂路由场景；
- 两个 Agent 未硬通过场景为苏州和丽江，根因是检索返回的 POI 类型不能满足下游景点
  详情/路线工具契约，不是 SFT/DPO 选错动作；
- 反复观察到向量检索关闭和 POI 覆盖不足，当前仅能评价结构化/关键词 RAG 链路；
- 质量门禁通过，但来源为 `authorized_replay`，报告保持
  `release_eligible=false`、`canary_evidence=false`，真实证据仍为 0/300；
- 原始报告保存于 `ml/agentic/reports/stage33-simulated-users-v1/report.json`，SHA-256：
  `2F8F3EE45C0A1E7712E9BC199B8A5F56861727F323185842D7AD0402B6C83EA0`；
- 中文结论保存于 `ml/agentic/reports/stage33-simulated-users-v1/REPORT.md`。

## 2026-08-16：POI 合同故障后续修复

- 已确认苏州、丽江两条失败不是策略动作错误，而是错误类别 POI 被工具层当成景点供给成功；
- 执行器现会校验请求类别，并为苏州、丽江提供受控景点兜底；
- POI/动作执行器定向测试 41/41 通过；原失败场景端到端回归 2/2 硬通过、0 兜底；
- 定点报告见 `ml/agentic/reports/stage33-poi-contract-regression-v1/`，报告 JSON SHA-256：
  `44B5D28C5BD5929B5716BB00B97623CE737A5A6517080D9F7FE8E6DD80C10A35`；
- 该回归只证明故障已修复，不改变真实 Shadow 仍为 0/300 的发布边界。
