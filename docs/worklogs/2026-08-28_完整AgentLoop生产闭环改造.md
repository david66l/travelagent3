# 2026-08-28 完整 Agent Loop 生产闭环改造

## 目标

把原来“LangGraph 中一个节点内部一次跑完整条链”的 Agent 分支，改造成生产可恢复、训练与推理同构、能真实执行多轮决策的 Agent Loop。

## 本次完成

- 将生产循环改为逐批次运行：每个串行动作或安全的并行只读批次结束后返回 LangGraph 并保存 checkpoint，下一轮从同一 Ledger 和 trajectory 继续。
- 新增显式候选审核：`search_pois → 观察候选 → accept_candidates / 改关键词继续搜索 / ask_user`，不再把一次搜索直接当成任务完成。
- 新增显式行程审核：`validate_itinerary → accept_itinerary / retry_solve / 回到搜索`。局部或全局重规划会提升 `plan_version`，失效旧 Solver/Verifier 产物并重开必要任务子图。
- Agent Ledger 增加 Decision History，并对“相同动作、相同参数、相同 Observation”做无进展检测，阻止重复搜索和死循环；默认 Agent 预算提升到 24 步/工具调用。
- 修复 `ask_user` 中断前问题产物未落盘的问题；用户回答后恢复原阻塞任务、预算和轨迹，而不是重新初始化整条链。
- FastAPI 与 Celery Worker 统一使用 PostgreSQL LangGraph checkpointer；修复 SSE 断线后快速重连仍可能被延迟清理任务取消的问题。
- 移除 Agent 失败时静默回退旧 Planner 的路径；确认后的事实核验冲突也会显式停止，避免旧 Planner 覆盖已验证的 Agent 行程。
- 将 SFT/GRPO 环境和数据脚本同步到新的 11 轮动作协议，默认 GRPO 工具轮次由 12 提升到 16，为恢复动作保留余量。
- 正式默认配置切换为 `policy_driven + enforce guards`；`controller_first` 和 5% Shadow 作为显式性能/灰度对照保留在独立配置中。

## 关键边界

- LLM 决定“下一步做什么”和需要学习的搜索/恢复参数。
- Controller 决定当前状态允许哪些动作、预算是否耗尽、哪些事实可信、任务如何迁移。
- CP-SAT 决定在时间、预算、营业时间、路线等硬约束下怎样排程。
- Verifier 决定子任务和最终行程是否真正通过；LLM 不能绕过。

## 验证

- 新增/更新测试覆盖：逐 checkpoint 延续同一 trajectory、候选二次决策、问题产物持久化、重复无进展终止、Verifier 失败后的版本化重规划、Agent 失败不回退旧 Planner、训练脚本遵循新动作契约。
- 本地 `backend/tests/unit/agentic + backend/tests/unit/graph` 最终回归：`413 passed`。
- Worker、WebSocket、SSE 与任务派发专项回归：`36 passed`。
- 相关 Python 模块通过 `compileall`，本次涉及文件通过 Ruff 检查。
- 当前 Windows 本地未启动带 PostgreSQL/Redis/真实模型端点的完整 Docker 故障注入，因此数据库级跨进程恢复仍需在集成环境做一次 E2E 验收。

## 后续可选增强

- 对未来真实预订/支付类写操作增加业务级 idempotency key；当前 Agent Loop 内主要是只读检索与可重算规划。
- 在带 PostgreSQL、Redis、真实模型端点的 Docker 环境执行故障注入 E2E，验证 Worker 中途退出后从最后一个 checkpoint 恢复。
