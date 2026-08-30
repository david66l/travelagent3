# Stage 31 Shadow 运行手册

## 目的与边界

Shadow 会保留当前确定性流程作为唯一用户可见结果，同时异步复制经过字段白名单和 PII 脱敏的规划输入，运行 4B/8B 路由 Agent。Shadow 不发送消息、不写用户记忆、不执行预订，也不改变当前请求结果。

Stage 30 只授权进入本阶段。未满足本手册中的准出门前，不允许把 `AGENTIC_POLICY_MODE` 改成 `agent`。

## 部署合同

1. 将 `deploy/stage31-shadow.env` 的非敏感变量合并到真实部署环境。
2. 通过 Secret Manager 注入推理端点凭据；仓库和 `.env` 模板不得保存真实密钥。
3. 设置 `AGENTIC_STUDENT_BASE_URL` 和 `AGENTIC_TEACHER_BASE_URL`。当前本地 Docker 配置通过宿主机的 `18000/18002` SSH 转发访问云端两个 loopback 服务；两个服务必须暴露模板中的 served model name。
4. 使用 `docker-compose.stage31-shadow.yml` 将 Shadow 队列放到独立 Celery worker，避免模型推理占用主规划 worker。
5. 初始采样率保持 5%。确认队列积压、数据库写入和两个推理端点稳定后，按 5% → 20% → 100% 提升；每次只改采样率。

Stage 31 已证明 4-bit 4B/8B 可以在一张 24 GB GPU 上同时常驻，但两个端点共享算力：各并发 4 时 P95 超过 5 秒，量化 4B 的 False Abort 也超过门槛。因此该拓扑只用于并发 1、5% 采样的探索性 Shadow；Canary/生产仍必须使用容量隔离端点，或在重新评测通过后使用具备显式队列上限的共享网关。

## 每日检查

- Shadow 失败不得改变确定性响应状态或延迟路径。
- `agentic_evaluation_records` 中 pending/running 不应持续积压。
- Agent 与 deterministic 必须按同一 `scenario_id` 成对完成。
- 管理接口 `/api/v1/admin/agentic-evaluation/report` 必须显示 route trace 覆盖率、学生回退率、教师占比和动作族分布。
- 单次学生失败只允许升级教师一次；教师失败必须记为 Shadow 失败，不能静默吞掉。
- 样本只保留脱敏字段，训练划分使用不可逆用户分区哈希。

## Canary 准出门

至少积累 300 个完整配对样本，并同时满足：

- Agent hard pass ≥ 98%；
- validated draft ≥ 95%；
- autonomous task completion ≥ 98%；
- Agent episode fallback ≤ 5%；
- hard pass 和 validated draft 相对确定性基线零回退；
- P95 延迟不超过确定性路径的 1.5 倍；
- 平均工具调用 ≤ 16；
- 至少存在 1 次真实路由决策，policy route trace 覆盖率为 100%；
- 学生转教师的推理失败回退率 ≤ 2%。

教师占比是成本与流量分布指标，先观察不设硬阈值；若显著偏离 Stage 29 的 46%，必须按 action family 审计原因，不能直接据此晋升或拒绝模型。

## 回滚

任何异常都先把 `AGENTIC_SHADOW_SAMPLE_RATE` 设为 `0`，停止新增 Shadow 作业；确定性用户路径保持不变。保留已完成的配对记录用于复盘，确认无在途任务后再停止独立 Shadow worker。不要删除失败记录。

## SSH 双端点隧道

使用 `scripts/start_stage31_tunnel.ps1` 建立本机到两个远端 loopback vLLM 端点的转发。脚本固定绑定本机 `127.0.0.1:18000/18002`，启用 SSH keepalive 和转发失败即退出；`-Reconnect` 会在连接退出后每 5 秒重试。

远端地址、端口、用户名和身份文件均通过参数传入，不写入仓库。示例：

```powershell
.\scripts\start_stage31_tunnel.ps1 `
  -RemoteHost <host> `
  -RemotePort <port> `
  -RemoteUser <user> `
  -IdentityFile <private-key-path> `
  -Reconnect
```

没有身份文件时 OpenSSH 会交互式请求密码，因此不能做到无人值守重连；禁止把密码写进脚本、命令行、Compose、环境文件或日志。要实现系统启动后自动恢复，应单独配置最小权限 SSH key，再由 Windows 任务计划程序启动上述脚本。

隧道验证：

```powershell
Invoke-RestMethod http://127.0.0.1:18000/v1/models
Invoke-RestMethod http://127.0.0.1:18002/v1/models
```

## 离线不可变 Stage31 镜像

Docker Hub 鉴权在当前网络上仍错误选择不可达的 IPv6 地址。`backend/Dockerfile.stage31` 因此以本机已经验证的 `travelagent2-backend:latest` 作为依赖基座，把当前源码、迁移和数据固化为 `travelagent2-backend:stage31`：

```powershell
docker build -f backend/Dockerfile.stage31 -t travelagent2-backend:stage31 backend
docker compose -f docker-compose.yml -f docker-compose.stage31-shadow.yml up -d --no-build
```

Stage31 overlay 中所有 Python 服务均使用该标签且不再挂载宿主机源码。网络恢复后仍应使用规范的多阶段 `backend/Dockerfile` 全量重建，以重新解析并锁定依赖基座。
