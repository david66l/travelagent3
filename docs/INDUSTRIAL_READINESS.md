# TravelAgent2 工业化验收说明

更新时间：2026-08-30

## 结论

当前仓库达到“可发布的生产候选（production candidate）”水平：核心 Agent、异步任务、数据一致性、可观测性、安全扫描、容器和 GitOps 发布均有自动门禁；本地真实依赖与真实模型链路已经跑通。

这不等于“已经被真实大流量证明”。Kubernetes 多节点部署、24 小时以上稳定性压测、数据层自动故障转移和更大规模真实用户评测仍需要目标云环境。面试或比赛中应使用“工业化生产候选”，不能写成“已支撑百万用户”或“生产零故障”。

## 当前生产链路

```text
User
  -> Go Gateway：认证、限流、熔断、请求追踪
  -> FastAPI：同一事务保存 Message + PlanningJob
  -> Celery/Redis：异步派发；失败由事务 Outbox 语义重新派发
  -> LangGraph Checkpoint + Agent Ledger
       结构化 LLM 意图识别 / 受控用户修改
       -> ReAct：模型逐轮选择只读工具
       -> Guard：动作白名单、参数注入、预算、幂等与终止边界
       -> Evidence Store + ResearchSufficiencyVerifier
       -> CP-SAT/Greedy：硬约束排程
       -> ItineraryValidator：事实、时间窗、路线间隔硬校验
       -> 补搜索 / 重求解 / 询问 / 权衡 / 安全终止
       -> 仅从验证产物生成回答
  -> PostgreSQL PlanningJobEvent
  -> SSE：last_event_id 断线重放
  -> 用户确认；拒绝时解析原因并生成新 Goal/Plan Version
```

LLM 决定“下一步做什么、调用哪个工具、如何理解自然语言”；Controller 决定“这个动作是否允许、预算是否足够、状态如何持久化”；CP-SAT 决定“在硬约束下怎样排程”；Verifier 决定“结果是否真的可交付”。模型不能直接写任务终态，也不能自行宣布验证通过。

生产默认模型与真实 ReAct 验收保持一致，为 `deepseek-v4-flash`。历史 Qwen 学生模型继续保留为离线/Shadow 候选；旧任务 DAG 上的高分不能直接冒充新 ReAct 协议已经晋级。

## 可复核证据

| 能力 | 当前证据 | 验收结论 |
|---|---|---|
| 原生 ReAct 完整闭环 | `artifacts/full-agent-loop-v2/formal-10-final-measured.json` | 10/10 场景得到正确业务终态；6 个生成或修改行程的场景全部 Verifier 硬通过，5 个 CP-SAT optimal、1 个受控 fallback；其余为 3 个安全补信息和 1 个前置澄清 |
| 多轮工具决策 | 同一报告 | 行程场景平均 3.6 次策略调用、13.6 次工具调用；不是单次 LLM 后由代码假装循环 |
| 故障恢复 | `artifacts/full-agent-loop-v1/recovery-smoke.json` | 3/3 恢复场景硬通过，平均 14.33 次工具调用 |
| 用户修改 | 完整闭环中的 `fal-v2-revise-shanghai` | 修改后 Goal/Plan 版本递增，重新求解并硬通过 |
| 可验证规划价值 | `ml/agentic/reports/stage40-policy-driven-runtime-vs-pure-react-v1/REPORT.md` | 历史 30 对消融中，逐步决策 + Solver + Verifier 硬通过率 96.67%，相对纯 ReAct Token 降低 79.51%；该结果只用于架构消融，不冒充当前原生 ReAct 大样本结果 |
| 行为蒸馏/SFT | `ml/agentic/reports/stage32-cascade-distillation-final-v1/README.md` | Qwen3-1.7B 从 Base 106/150 提升到 SFT 135/150，即 70.67% -> 90.00%，+19.33 个百分点 |
| 偏好优化 | 同上 | 冻结 SFT reference 的 DPO 从 135/150 到 137/150，即 +1.33 个百分点；增益不足以直接上线，因此只进 Shadow |
| 学生推理效率 | 同上 | 1.7B SFT 平均 3114.8 ms，对比 4B SFT+DPO 5596.1 ms，延迟降低 44.34%；吞吐 2.497 vs 1.367 req/s，提升 82.66% |
| 后端回归 | 2026-08-30 本地全量测试 | 1368 passed，3 skipped，0 failed；包含新增的评测汇总口径回归测试 |
| 控制面容量 | `artifacts/performance/k6-local-summary.json` | 20 VU、15 秒、261 完整迭代、783 HTTP 请求、0 失败；chat admission 平均 121.96 ms、P95 491.60 ms、最大 529.88 ms |
| 并发幂等 | PostgreSQL advisory lock 集成验证 | 同一 Idempotency-Key 12 路并发只生成 1 个 Job 和 1 条消息，12 个请求均得到可重试一致响应 |
| 数据/队列故障 | Redis/PostgreSQL fault-injection 测试 | Redis 通知丢失可由 PostgreSQL 事件重放恢复；数据库或 Broker 故障不伪装成功，任务进入明确重试或终态 |
| 数据库迁移 | 空库与长期库双路径 | 从空库执行到 Alembic head 后 `alembic check` 无漂移；迁移改为单一 Sync Hook Job，消除多副本 initContainer 竞态 |
| K8s 静态验收 | kubeconform 0.8.0 + KubeLinter 0.8.3 | 49 个资源 schema 全部有效，0 lint error；14 个工作负载检查无 root/提权/capability/latest-tag 问题 |
| 凭据与依赖 | Gitleaks、仓库 secret scanner、pip-audit、npm audit、Bandit、Trivy | Git 历史和受控文件未发现凭据；Python/npm 已知依赖漏洞为 0；Bandit 高/中危为 0；三类生产镜像 fixable 高危/严重漏洞为 0 |

说明：模型评测、恢复 smoke 和单机 k6 的样本量不同，不能把这些数字合并成一个“总准确率”。

## 发布门禁

唯一自动发布入口是 `.github/workflows/deploy.yml`，不存在旁路 `kubectl apply` 工作流。

1. `CI` 必须整体成功：Python/TypeScript/Go 检查、全量测试、迁移、依赖审计、Git 历史密钥扫描、IaC 扫描、K8s schema/lint、控制面 k6、生产镜像启动健康检查。
2. Deploy 只接受通过 CI 的 `main` 提交或受 GitHub Environment 审批的手动 production 发布。
3. 构建 commit-SHA 镜像；不推送 `latest`。
4. 对待发布镜像执行 Trivy fixable High/Critical 阻断扫描。
5. 发布前检查 `main` 仍等于本次构建 SHA，拒绝并发流程中的过期构建。
6. GitOps 只提交不可变 SHA 标签，提交带 `[skip ci]`，避免发布机器人形成递归部署。
7. ArgoCD 先部署数据层（wave -2），再执行 Redis 幂等初始化与单实例 Alembic 迁移（wave -1），最后部署应用（wave 0）。迁移失败时应用不升级。
8. ArgoCD CLI 固定版本并校验官方 checksum；成功后等待 Application Healthy。

生产 Secret 不在 `k8s/` 中，也不存在可被 ArgoCD误应用的占位 Secret。集群必须通过外部密钥系统创建 `travel-agent-secrets`；示例只位于 `deploy/examples/travel-agent-secret.yaml.example`。

## 运行时可靠性设计

- `PlanningJob` 是任务真相源，并承担轻量事务 Outbox；事务提交后才发布 Broker 消息。
- Worker 使用数据库租约和幂等状态迁移，重复派发不会重复完成任务。
- 公开 SSE 事件先持久化，再用 Redis 唤醒连接；Redis 不是关键事实源。
- 同一会话更新串行化，避免两个用户请求互相覆盖 Agent Ledger。
- 重试只覆盖声明过的临时故障；业务拒绝、参数错误和硬约束不满足不会盲目重试。
- 搜索“没有结果”和搜索服务“不可用”是两个不同 Observation，后者不能被当作事实不存在。
- 所有动作受 token、工具调用、重复无进展和最大步数预算约束，预算耗尽进入可解释的安全终止。
- 行程只有在 Verifier 硬通过后才可进入待用户确认状态。

## 安全与隔离

- Production 配置缺少 JWT 或隐私加密密钥时 fail closed。
- 生产容器非 root、禁止提权、drop ALL capabilities、默认只读根文件系统；PostgreSQL 等确需写盘的容器有显式例外说明。
- Namespace 启用 Restricted Pod Security；所有工作负载使用 RuntimeDefault seccomp。
- NetworkPolicy 默认拒绝入站/出站，只开放 Gateway、服务依赖、DNS 和外部 HTTPS；外部 HTTPS 排除内网、loopback 和 link-local 地址，降低 SSRF 横向移动范围。
- PostgreSQL 主写 Service 只选择 `postgres-0`，只读 Service 只选择 `postgres-1`，避免请求随机写入 standby。
- Redis 集群初始化使用独立、幂等的 Sync Hook Job，消除 initContainer 等待尚未启动 Redis 进程的部署死锁。
- 数据库和 Redis 基础镜像使用 digest；应用镜像使用 commit SHA。

## 监控与告警

业务和可靠性指标至少覆盖：请求量/错误率/延迟、PlanningJob 状态、任务排队与租约、重试/DLQ、LLM token 与延迟、工具错误、Solver 状态、Verifier 硬通过率、Agent 终止原因和成本熔断。Prometheus 告警规则与 Grafana 面板位于 `monitoring/`。

需要把以下信号设为发布后观察窗口的强制项：

- `chat_acceptance` P95 < 600 ms、P99 < 1000 ms、错误率 < 1%。
- 可求解行程 Verifier 硬通过率 >= 95%；安全澄清/拒绝单独统计，不能算失败也不能算硬通过。
- 重复任务完成数为 0；租约过期、DLQ、budget exhausted 和 fallback 比例持续可见。
- 模型版本、prompt/contract 版本和评测集 hash 必须随每条评测和线上 episode 保存。

## 回滚与数据恢复

- 应用回滚：把 GitOps 镜像标签回退到上一个 commit SHA，由 ArgoCD 同步。
- 数据库迁移采用 expand/contract；发布 Job 只前滚。破坏性 schema 删除必须至少跨两个版本，不能依赖自动 downgrade。
- 本地 PostgreSQL 镜像升级前已生成 `artifacts/backups/local-pre-pgvector-upgrade.dump`；该目录被 Git 忽略。
- PostgreSQL 备份 CronJob 使用独立 PVC。真正生产还必须增加异地对象存储、加密、保留策略和定期恢复演练。

## 尚未冒充完成的生产证明

以下项目需要真实云资源或更多线上样本，当前只能列为发布前条件：

1. 在目标 Kubernetes/CNI/StorageClass/GPU 环境执行真实部署、NetworkPolicy 连通性和 ArgoCD hook 顺序验收。
2. 至少 200 个冻结、去污染、多城市多轮场景，按场景分层并给出置信区间；当前原生 ReAct 正式集只有 10 个场景。
3. 24 小时以上 soak、跨节点故障、Worker 被杀、Redis master 故障、PostgreSQL 主库切换和 SSE 重连风暴。
4. 当前自建 PostgreSQL standby 没有自动主从切换；正式业务应使用云托管 PostgreSQL 或 CloudNativePG/Patroni。Redis 同理优先使用托管集群或成熟 Operator。
5. 原生 ReAct 正式集平均约 10.7k Token、P95 约 16.2 秒，证明了完整循环而非低延迟；进一步优化应依赖缓存、并行只读工具、上下文压缩与通过同协议门禁的学生模型。
6. 后端 Debian 基础镜像仍有上游暂未提供修复的 Perl 系统包漏洞报告（2 Critical、2 High）。发布门禁已阻断所有可修复 High/Critical，但该风险例外仍需跟随基础镜像更新。

因此，当前最准确的对外表述是：**完成了具备完整 Agent Loop、确定性求解与验证、可恢复异步执行、可观测与安全发布门禁的工业化生产候选，并通过单机真实依赖和小规模真实模型评测；大规模生产证明待目标集群与线上流量完成。**
