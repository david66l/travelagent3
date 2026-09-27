# TravelAgent2 本地代码审查报告

审查范围：本地工作树（`feat/agentic-rl-long-horizon`，HEAD `e51e896`），非某个已提交版本。
所有结论均带 `文件:行号` 证据；标注「已实测」的项为本次真实执行结果。
本文件由审查生成，可直接删除，不属于项目产物。

---

## 修复记录

验证结果：`pytest tests/unit/` → **1452 passed, 0 failed**（修复前为 1432 passed / 2 failed / 2 collection errors）。
剩余 126 个 error 全部是 `ConnectionRefusedError: ('127.0.0.1', 5432)`，即本机未启动 Postgres，与代码无关。
前端 `tsc --noEmit` exit 0、`npm run lint` 无警告。

| # | 问题 | 处理 |
|---|---|---|
| P0-2 | validator 版本漂移导致 2 个测试失败 | 测试改为引用 `VALIDATOR_VERSION` 常量（`test_termination.py`、`test_tool_executor.py`），版本再升级不会误报 |
| P0-3 | `fcntl` 无条件导入，Windows 下收集失败 | `core/api_budget.py` 改为双后端锁（POSIX `flock` / Windows `msvcrt.locking`），目录 fsync 在 Windows 跳过。**同时消除了 `tests/unit/core` 对 `test_planning_worker` 的顺序依赖** |
| P0-5a | CORS 缺 `Idempotency-Key`，跨域发消息必失败 | 加入默认头列表；并把源/头改为可配置（`GATEWAY_CORS_ORIGINS` / `GATEWAY_CORS_ALLOW_HEADERS`），`.env.example` 已补文档 |
| P0-5b | `pending_approval` 在 SSE 投影中丢失，确认永远失败 | 后端两处投影（持久化路径 `chat_runtime.py:752`、进程内路径 `:548`）补回该字段；前端改为「缺 key 则不覆盖」；更新并加固了 `test_websocket.py` 的契约测试 |
| P1-1 | 死掉的 shadow/routing 门禁（永远通不过） | 删除 `PolicyRouteTrace`/`PolicyShadowTrace` 及其字段与消费方（`integration.py` 的 `summarize_policy_routing`、`agent_policy_routing` 全链路、`agentic_eval.py` 的 7 个指标字段 + 3 项门禁 + 2 个配置项），删除孤儿脚本 `export_stage33_policy_shadow_observations.py`、`smoke_shadow_pair.py`、`replay_shadow_pairs.py` 与相关测试，`TopBar` 移除「智能路由」徽标 |
| P1-2a | `validate_checkpoint_candidate` 无阈值 | 增加绝对下限（默认 ≥30 任务、成功率 ≥0.5、平均奖励 ≥0.0，可 CLI 覆盖），缺失/非数值指标 fail-closed；新增 2 个回归测试 |
| P1-2b | `behavior_gate` 缺 key 时 fail-open | 缺 key 或缺指标现在硬失败（`BEHAVIOR_GATE_MISSING` / `BEHAVIOR_GATE_METRIC_MISSING`）；新增 2 个回归测试 |
| P1-4 | 奖励偏好「不解决问题」（实测 12 步以上澄清胜出） | 两处修正：efficiency 不再惩罚原始轨迹长度（改为只惩罚重复调用与零信息步），并把 clarification 的任务分设为 0.9 < validated_plan 1.0 以保证严格次序；奖励配置版本升至 `hierarchical-b0.v3`；新增跨 9/12/16 步的次序回归测试。**注意：这改变了训练信号，需重建语料后重训** |
| P2 | redis pubsub 连接泄漏 | `chat_runtime.py` 的 `_redis_sub` finally 补 `await pubsub.aclose()` |
| P2 | k8s 上 SSE 30 秒断流 | `k8s/configmap.yaml` 的 `GATEWAY_WRITE_TIMEOUT` 改为 `0s` 并注明原因 |
| P0-4 | compose 中浏览器绕过网关 | 按选择**不改拓扑**，改为在 README 加显著警告（需设 `NEXT_PUBLIC_API_URL` 指向网关、不要对外发布 8000、用 `GATEWAY_CORS_ORIGINS` 列真实源） |

**尚未处理（需你决定或工作量较大）**：P0-1（94 个删除的文件收尾）、P1-3（`ml/training` 自评 M4 门禁）、
P1-5（`git_commit: "unknown"`、contamination root 记录、sealed 封印脚本）、P2 中后端其余诸项
（`planning_job.status` 写阶段名、重复 drain、流式路由占用 DB 连接、`users.py`/`webhooks.py` 鉴权、
配额死代码、迁移链与 `create_all` 冲突）、前端其余诸项（登录跳 404 路由、token 永久缓存、错误信封）。
`gateway` 的 Go 代码改动（CORS 配置化）**未经编译验证**——本机未安装 Go，需在 CI 或装有 Go 的机器上确认。

---

## 0. 总览

| 维度 | 结论 |
|---|---|
| 代码质量 | 分层清晰、注释解释「为什么」、失败路径大多 fail-closed。属于明显高于平均水平的工程 |
| 最大问题 | **工作树不可验证**：94 个文件被删除（含 47 个测试）、102 个修改，全部未提交 |
| 第二问题 | **重构未收尾**：审计/门禁/评估仍在依赖已被删除的 shadow / routing 机制 |
| 第三问题 | **Gateway 在默认配置下不生效**：compose 里浏览器直连 backend，K8s 里 SSE 会在 30s 断流 |
| 第四问题 | **若干「门禁」不可失败**：奖励函数甚至更偏好「不解决问题」 |

规模参考：backend Python 264 文件 / 50,936 行（不含 `ml/`、`models/` 的 4.4GB / 3.9GB 实验资产）。

---

## 1. P0 — 必须优先处理

### P0-1 工作树与 HEAD 不一致，CI 形态的验证在此树无法完成

- `git status --porcelain`：**94 个 D（删除）、102 个 M（修改）、157 个未跟踪**；分支领先远端 29 个提交。
- 47 个被删文件是测试；`scripts/evaluate_native_react_hard.py`（唯一包含 sealed-test 守卫的脚本）也在其中。
- 实测：`pytest tests/unit/` → **2 failed, 1432 passed, 5 skipped, 126 errors**（10 分 45 秒）。
- `scripts/travelctl.py list repair` 输出为空且 **exit 0**：`SUPPORTED_VERBS` 声明了 `repair`（`travelctl.py:50`），但 `repair_*.py` 已被删除，`_print_index` 遇到空列表直接 `continue`。

**影响**：README 中「测试通过 / 门禁成立」的表述与当前代码不是同一个 revision。审查结论对当前树有效，对 HEAD 未必有效。

### P0-2 2 个真实断言失败：validator 版本升到 v3，测试仍断言 v2

```
backend/src/evaluation/validator.py:19   VALIDATOR_VERSION = "travel-validator.v3-named-dining"
backend/tests/unit/agentic/test_termination.py:42        == "travel-validator.v2"   # 失败
backend/tests/unit/tools/test_tool_executor.py:275       == "travel-validator.v2"   # 失败
```

注意 `docs/AGENT_HARNESS_ARCHITECTURE.md:130` 仍写着「新的运行使用 `travel-validator.v2`、`terminal-contract-v3`」——文档、代码、测试三者互不一致。

### P0-3 另有 127 个本地失败与 1 个顺序依赖

- `fcntl` 在 Windows 不存在，但 `backend/src/core/api_budget.py:8` 无条件 `import fcntl`，导致 `test_teacher_queue.py`、`test_api_budget.py` **收集阶段**就报错。CI 跑 ubuntu 看不出这个问题，本地 Windows 开发必踩。同文件 `os.open(str(self.path.parent), os.O_RDONLY)`（`:55`）对目录 fsync，同样是 POSIX-only。
- 79 个 error 是 `ConnectionRefused`（需要 Postgres/Redis），未 skip 而是 error。
- 实测顺序依赖：`pytest tests/unit/core tests/unit/worker/test_planning_worker.py` 会失败，而 `pytest tests/unit/worker/` 单独跑 39 passed。即 `tests/unit/core` 中存在污染全局状态的测试。

### P0-4 Gateway 在「推荐」部署方式下不参与请求链路

- `frontend/next.config.js:2`：`INTERNAL_API_URL || 'http://localhost:8000'`；`docker-compose.yml:98,103` 设 `INTERNAL_API_URL=http://backend:8000`、`NEXT_PUBLIC_API_URL=""`。
- 结果：浏览器 `/api/v1/*` 被 Next 直接重写到 `backend:8000`，**绕过 gateway 的 JWT / 限流 / 熔断**；同时 `docker-compose.yml:53` 还把 `8000:8000` 发布到宿主机。

这不是「gateway 有 bug」，而是「gateway 的安全边界被部署拓扑架空」，比单个漏洞更值得处理。

### P0-5 前端两个 blocker（默认配置下浏览器无法发消息 / 无法确认行程）

1. **CORS 缺 `Idempotency-Key`**：`frontend/src/lib/api.ts:198-201` 发送该头部，`gateway/cmd/gateway/main.go:90` 的 `AllowHeaders` 未列入。`frontend/.env.local` 明确设 `NEXT_PUBLIC_API_URL=http://localhost:8080`，属跨域请求 → 预检拒绝 → **每次发消息都失败**。已核对两侧源码确认。
2. **`pending_approval` 在 SSE 投影中丢失**：`graph/runner.py:740` 的持久化 payload 含 `pending_approval`，但 `api/chat_runtime.py:758-765` 投影 `awaiting_confirm` 时只取 `itinerary` / `warnings` / `agent_policy_routing`；`frontend/src/lib/chatEvents.ts:263-265` 再把它强制置为 `null`；`useChat.ts:158` 于是提交 `approval: null`；`graph/approval.py:90-94` 抛 `APPROVAL_REQUIRED`。默认 `planning_executor="celery"`（`core/settings.py:177`）走的就是这条路径 → **「确认并完善」永远失败**。

---

## 2. P1 — 审计与验证体系已失效（本次最值得看的发现）

这一组的共同点是：**shadow / routing 机制已作为「决策包装」被有意删除，但读取它的门禁、指标、接口全部留下，且无人清理。**

### P1-1 发布门禁要求一个运行时永远不会产生的指标

- `backend/src/agentic/loop.py:78-99` 仍定义 `PolicyRouteTrace` / `PolicyShadowTrace`，但**全 `src/` 无任何地方构造它们**（仅存在于测试里）。
- 消费方却还在：`agentic/integration.py:237` 读 `step.action.route_trace` → 永远 `None` → `summarize_policy_routing` 返回 `None`。
- `evaluation/agentic_eval.py:439-441`：`policy_route_calls / policy_decisions` → **0.0**。
- `evaluation/agentic_eval.py:106`：`minimum_policy_route_trace_rate` 默认 **1.0** → `POLICY_ROUTE_TRACE_RATE` 门禁永远 `0.0 >= 1.0` = False。
- `evaluation/agentic_eval.py:105`：`minimum_routed_policy_decisions` 默认 1 → `MINIMUM_ROUTED_POLICY_DECISIONS` 同样永远失败。
- 而 `api/v1/agentic_evaluation.py:24,63` 还要求 `evaluation_source="live_shadow"`——**生产者 `agentic/shadow.py`、`worker/shadow_tasks.py` 已被删除**。

**结论：整条 shadow → 配对评估 → 晋升 链路结构性死亡，且是「永远通不过」而非「永远通过」。** 相关单测（`tests/unit/evaluation/test_agentic_eval.py`）之所以能过，是因为它们**自己构造 `PolicyRouteTrace` 并注入**——验证的是测试写的 fixture，不是运行时行为。

### P1-2 两个「晋升门禁」实际上都不设阈值

- `scripts/validate_checkpoint_candidate.py:60-63` 只比较「manifest 自报指标」与「对比报告指标」是否一致，**没有任何下限**。子代理实测：`tasks=1, success_rate=0.0, hard_pass=False` → `valid = True`。
- `scripts/compare_curriculum_audits.py:47-61` 的 `behavior_gate` 用 `.get(...) or 0.0`：**缺 key 时判定为 0 违规 → 通过**（fail-open）。子代理实测：无 `behavior_gate` 键 → `promoted=True`；有该键且 1% 违规 → 拒绝。
- `evaluation/posttraining_promotion_protocol.py:204-256` 的 0.90 / 10pp 等阈值，只校验协议 JSON **自己写对了数字**；全仓库 grep 显示没有任何测量值流入这些门禁。

### P1-3 `ml/training` 的「M4 LoRA 评测门禁」是自评循环

`ml/training/eval_lora.py:31-33` 比较的是**数据集自己的 `prediction` 字段与自己的 `completion` 字段**；`--adapter`（`:23`）只用于打印和 mlflow 参数，**从不加载任何模型**。实测：`ml/training/data/train.jsonl` 50 行、`prediction == completion` 50/50、恰好等于 `--min-samples 50` 阈值 → `match_rate = 1.0` 通过。而 `scripts/verify_milestones.sh:19-25` 把这个当作 M4 门禁执行；`train_lora.py:68-69` 自己写明是 placeholder，不产出权重。

### P1-4 奖励函数偏好「不解决问题」（已实测）

用真实 `HierarchicalRewardEngine` 打分：

```
A) 只问一句澄清（1 步 ask_user，无行程、无校验报告）   total = 0.969125
B) 完整验证行程（9 步，hard_pass=True）                total = 0.959569
→ clarification 胜出 +0.009556
```

机制：`agentic/reward.py:611-613` 对 `clarification` / `safe_termination` 直接给 `constraint = 1.0`（**不要求任何校验报告**）；`:646-648` 的 efficiency 项单调惩罚「步骤更多」的轨迹。由于该奖励就是 GRPO/DPO 的信号（`trl_environment.py:413-433` → `train_grpo.py:392`），组内相对优势会系统性偏向早停。

另：`reward.py:696-699` 的 grounding 是裸子串匹配且**无最小长度保护**，而同类函数 `_grounded_phrase_match`（`:713-718`）有 `min(len) < 3` 保护——单字符叶子即可通过。

### P1-5 训练/评测资产的可复现性与封印

- `ml/agentic/checkpoints/*/training_report.json` 中 `"git_commit": "unknown"`，且无 `provenance` 字段（）——checkpoint 无法关联到代码版本，与 README「`training_report.json` 必须携带可解析的 `git_commit`」的纪律相矛盾。
- `scripts/build_native_react_hard_benchmark.py:42-43` 对不存在的 contamination root 直接 `continue`，manifest 只记录扫描**数量**、不记录**扫描了哪些 root** → 写错路径仍报 `passed: true`。
- 无 `--allow-frozen-test` 守卫（该 flag 只存在于已删除的脚本中），sealed-160 的封印目前仅剩文档描述。
- `scripts/compare_native_react_hard_arms.py:48-51` 按 `family`（10 个）而非文档要求的 `cluster_id`（40 个）重采样，CI 区间被低估。

---

## 3. P2 — 运行时缺陷

### Gateway（Go）

| 位置 | 问题 |
|---|---|
| `k8s/configmap.yaml:30` | `GATEWAY_WRITE_TIMEOUT: "30s"` 覆盖 Go 默认的 `0`（`internal/config/config.go:53`，其注释明确说「有限值会截断正常 SSE 流」）。Go 的 `WriteTimeout` 覆盖整个响应（`cmd/gateway/main.go:119-124`），**规划流会在 30 秒被切断**；`.env.example:86` 与 `start.sh:286` 都坚持 `0s`，ingress 设 `3600`。仅 K8s 错。 |
| `internal/middleware/auth.go:50-53,69-72` | Redis 出错时 `if err == nil && blacklisted > 0` 丢弃错误 → **fail-open**，已吊销/封禁 token 仍可用；`device_fingerprint` 为空则完全跳过设备绑定。 |
| `internal/middleware/rate_limit.go:21-25,47-58` | 同上 fail-open，仅打日志、无指标无告警；且 `/health` 是静态 JSON，不检查任何依赖 → **Redis 挂掉时网关「看起来健康」却什么都不拦**。 |
| `internal/middleware/circuit_breaker.go:16-37` + `internal/proxy/proxy.go:33-36` | 代理直接写 ResponseWriter 并 `return nil`，熔断器的 503 分支不可达；且是**全局单实例**，一个坏端点会拖垮整个 API。 |
| `internal/config/config.go:54` | `JWT_SECRET` 默认 `dev-secret-change-me` 且无校验，而后端 `core/settings.py:283-287` 对同一输入直接拒绝启动；`docker-compose.prod.yml` 全文没有 `JWT_SECRET`。 |
| `internal/proxy/proxy.go:23,45` | 未设置 `.Transport`（用 `http.DefaultTransport`），无 `ResponseHeaderTimeout`/拨号超时；`config.go:51-52` 声称「per-request upstream timeouts 保护服务」——该超时**不存在**。 |
| `cmd/gateway/main.go:161` vs `monitoring/grafana/dashboards/overview.json:25` | 指标 `status` 用 `http.StatusText`（`"OK"`），面板查 `status=~"5.."` → 5xx 面板永远为空；且无 gateway 侧 5xx 告警。 |

`docker-compose.prod.yml` 实测 `docker compose config` **exit 1**：`container_name` + `deploy.replicas` 冲突（`:237-254`、`:266-283`），文档所称的「生产级单机部署」无法启动。`docker-compose.observability.yml:97-108` 用 nginx 占了同名 `gateway` 服务/容器名/端口，按 README 执行会把 Go 网关替换成 nginx。

### Backend

| 位置 | 问题 |
|---|---|
| `api/chat_runtime.py:830-862` | `pubsub = redis_client._client.pubsub()` 后 **只 `unsubscribe`，从不 `aclose()`** → 每条 SSE 流永久占用共享 Redis 池的一个连接（池上限 100）→ 累计约 100 条流后整个 state-Redis 层报 `MaxConnectionsError`。同仓库 `worker/planning_worker.py:421` 是正确写法（`await pubsub.aclose()`）。**一行修复，收益最大。** |
| `repositories/planning_job.py:482` + `:307-312` + `worker/planning_tasks.py:92` | `update_stage` 把**图阶段名写进 `status` 列**；而重认领只接受 `pending`/`retrying`/`running`。worker 中途崩溃（kill/OOM/发版）后 `status='writing'` 之类**永远无法被回收**：SSE 客户端挂到 1800s 超时，无终态、无 DLQ。 |
| `api/chat_runtime.py:798-821` | `_drain` 被 `_redis_sub`(:852) 与 `_poll`(:866) **并发调用并共享 nonlocal `last_event_id`** → 同一批事件可能重复下发（重复渲染、重复正文）。 |
| `api/v1/chat.py:179` | 流式路由依赖 `get_current_user` → `Depends(get_db)`，FastAPI 要等响应结束才拆依赖栈 → **一条 Postgres 连接被占用最长 1800 秒**（池 20+10）。约 30 条并发流即可打满 API 连接池。 |
| `api/v1/users.py:58-67` | `GET /users/{user_id}` **完全没有鉴权**，返回 email/phone。 |
| `api/v1/webhooks.py:11-36` | webhook 无鉴权、无 admin 限制，含 `DELETE /webhooks/events/clear`。 |
| `api/v1/agent_chat.py:280-288` | `repo.create(user_id=..., content=...)` 与 `BaseRepository.create(obj)` 签名不符 → `TypeError` 被 `except Exception: pass` 吞掉 → 该路径**行程永不落库**。 |
| `api/v1/chat.py:320-328` + `monitoring/rate_limit_controller.py:193-196` | `estimated_tokens=0` → 令牌/外部 API 配额检查被跳过；`core/guest_policy.py:14 ensure_guest_can_plan` **零调用方** → 「游客 1 条行程」限制是死代码。 |
| `core/database.py:80-82` + `api/main.py:45` | 启动时 `Base.metadata.create_all`，且从不写 `alembic_version`。`create_all` 不会 ALTER，因此 app 引导过的库会永久缺列，之后 `alembic upgrade head` 撞 `DuplicateTable`。迁移链本身干净（15 revision、单 head、列级 diff 无漂移）。 |

### Frontend

`Idempotency-Key` CORS 与 `pending_approval` 见 P0-5。其余：

- `components/LoginForm.tsx:90-92,106` `router.push("/chat")` / `router.push("/admin")`——构建产物只有 `/`、`/login`、`/_not-found` 三条路由（`npm run build` 已确认）→ **登录后落到 404**。
- `hooks/useChat.ts:14-26` token 缓存在 ref 中永不过期；`lib/api.ts` 中 `postChatMessage`/`postChatAction` 无 401 恢复（只有 `createConversation` 有）；前端**从不调用** `/api/v1/auth/refresh`，`getStoredRefreshToken` 导出后无人使用 → 退出登录或 token 过期后无法自愈。
- `lib/api.ts:121-122,144-146` 读 `json.message`，而后端错误信封是 `{error:{code,message}}`（`core/responses.py:59-73`）→ **服务端诊断信息永远被丢弃**。
- 正向确认：`tsc --noEmit` exit 0、`npm run build` exit 0、`npm run lint` 无警告；SSE 解析器（多字节安全、keepalive、退避、429 `Retry-After`）写得扎实。

---

## 4. 文档与代码的偏差

| 文档 | 实际 |
|---|---|
| `CLAUDE.md:3`「LangGraph 编排（16 类节点 + 10 个职能 Agent 模块）+ Bounded Agent Loop（Policy/Controller/Executor/Verifier 四权分立）」 | `graph/graph.py:29-36` 只有 6 个顶层节点（另加 gathering 子图）；`backend/src/agents/` 只有 **8** 个模块（`hallucination_detector.py`、`rag_retrieval.py` 已被删除）；`policy_controller.py` 仍在但 controller 代决策机制已被删除，`policy_shadow.py`/`shadow.py`/`replanner.py` 不存在。「四权分立」描述的是已废弃架构。 |
| `README.md` 目录结构称 `planner/` 为「行程规划 DAG」 | 该 DAG 已删除（`docs/AGENT_HARNESS_ARCHITECTURE.md:99` 自述已移除），`planner/core/dag.py` 不存在。 |
| `README.md:231`「travelctl 列出 140 个脚本」 | `scripts/*.py` 实际 **135** 个。 |
| `README.md:142`「Go 1.26 toolchain」 | `gateway/go.mod:3` 写 `go 1.25.0`（Dockerfile/CI 用 1.26）。 |
| `docs/AGENT_HARNESS_ARCHITECTURE.md:130`「新的运行使用 `travel-validator.v2`」 | 代码是 `travel-validator.v3-named-dining`（`evaluation/validator.py:19`）。 |
| `README.md:131` 可观测性启动命令 | 需要叠加 base compose（`-f docker-compose.yml -f docker-compose.observability.yml`），且会替换掉 Go 网关。 |
| `README.md:232`「所有后训练 checkpoint 处于 quarantine，无一晋升生产路由」 | 与代码一致且是诚实的：`train_sft.py:847-848` 硬编码 `promotion_eligible: False`。 |

---

## 5. 建议的修复顺序

1. **收尾工作树**：确认那 94 个删除是否有意；提交或恢复，并修掉引用已删模块的测试。在此之前任何「已验证」的说法都不成立。
2. **补齐重构缺口**：删除 `PolicyRouteTrace`/`PolicyShadowTrace` 及其消费方与门禁，或恢复生产者；`minimum_policy_route_trace_rate` 若保留则必须能被真实运行满足。
3. **改掉 3 个 fail-open 门禁**（`validate_checkpoint_candidate` 无阈值、`behavior_gate` 缺 key 即通过、promotion protocol 无测量流入），并删除或修好 `ml/training` 的自评 M4 门禁。
4. **修奖励反转**（`reward.py:611-613`、`:646-648`、`:696-699`）——它直接决定 GRPO 学到什么。
5. **`chat_runtime.py:857-860` 加 `await pubsub.aclose()`**（一行，堵住 Redis 连接泄漏）。
6. **`k8s/configmap.yaml:30` 改为 `0s`**，否则 K8s 上规划流 30 秒必断。
7. **修前端两个 blocker**：gateway `AllowHeaders` 加 `Idempotency-Key`；`chat_runtime.py:758-765` 补 `pending_approval`，前端改为「缺 key 则不覆盖」。
8. **安全项**：`users.py:58` 与 `webhooks.py` 加鉴权；网关 fail-closed；生产 compose 补 `APP_ENV`/`JWT_SECRET`/`PRIVACY_ENCRYPTION_KEY`（目前 prod 会走 dev 分支随机生成密钥，重启即失会话且历史密文不可解）。
9. **可用性项**：`planning_job.status` 不要写阶段名 + 增加租约回收；SSE 路由不要持有 DB 会话；`_drain` 单所有者。
10. **文档同步**：更新 `CLAUDE.md` 与 README 的架构描述，使其匹配 `agent-harness-v1`。

---

## 6. 值得肯定的部分

- `agentic/harness.py` 的 `preflight_action` / `invalidate_derived_evidence`：把「证据版本失效」「前置条件」做成显式且可测的边界，注释解释了每条拒绝的理由。
- `agentic/loop.py` 的预算扣费语义（先检查后扣、拒绝的提案不扣工具额度）与 `_record_decision` 的重复观测检测（对 `solve_time_ms` 做排除，避免「同样的求解结果」被当成新证据）设计得当。
- `agentic/episode_accounting.py` 的对账验证器（预算单调、动作 ID 唯一、执行器已排空）是真实的独立校验。
- `reward.py:189-221` 有 5 道 fail-closed 闸门，`tests/unit/agentic/test_reward.py` 有 15+ 个真实拒绝用例。
- `limit/limiter.go` 的 Redis 有序集合滑动窗口 Lua 脚本实现正确。
- `scripts/travelctl.py` 的按文件名派发**结构上不可能产生悬空 verb**（这正是 `repair` 暴露删除问题的方式）。
- Alembic 迁移链干净：15 个 revision 线性、单 head，逐列对比模型无漂移。
- 前端 SSE 解析与「流式正文 vs 最终正文」去重（`chatEvents.ts:83-109`）避免了经典的正文重复追加 bug。
