# 运维手册

## 健康检查

| 端点 | 说明 |
|------|------|
| `GET /api/health` | 存活探针 |
| `GET /api/ready` | 就绪探针（DB + Redis） |
| `GET /api/v1/metrics` | Prometheus 指标 |

## 部署

### Docker Compose（开发）

```bash
docker compose up -d
```

### Kubernetes

```bash
bash scripts/k8s-apply.sh
kubectl -n travel-agent get pods
```

## 监控与告警

启动可观测栈：

```bash
docker compose -f docker-compose.observability.yml up -d
```

告警规则：`monitoring/prometheus/alerts.yml`

| 告警 | 条件 |
|------|------|
| HighErrorRate | 5xx > 5% |
| HighP99Latency | HTTP P99 > 2s |
| CostCircuitBreakerActive | 成本熔断开启 |
| DailyTokenBudgetHigh | 日 Token > 4000 万 |
| DailyExternalApiCostHigh | 日 API 成本 > 800 CNY |

Grafana 大盘：

- `travel-agent-overview.json` — 服务概览
- `travel-agent-cost.json` — 成本与 LLM 延迟

## 成本熔断

环境变量（见 `.env.example`）：

- `COST_CIRCUIT_BREAKER_ENABLED`
- `COST_CIRCUIT_BREAKER_DAILY_TOKENS`
- `COST_CIRCUIT_BREAKER_DAILY_API_COST_CNY`
- `COST_CIRCUIT_BREAKER_HOURLY_GPU_COST_CNY`

熔断激活后，`model_router` 自动将请求路由到 `SMALL_MODEL`。

手动熔断（Redis）：

```bash
redis-cli SET cost:circuit:manual 1 EX 3600
```

解除：`redis-cli DEL cost:circuit:manual`

## 常见故障

| 现象 | 处理 |
|------|------|
| SSE 断开 | 检查 `rate_limit_max_concurrent_sse`、网关超时 |
| 规划卡住 | 查看 Celery worker 日志、`planning_jobs_failed_total` |
| Redis 不可用 | `/api/ready` 失败，恢复 Redis 集群 |
| 成本熔断误触 | 调高阈值或清除 `cost:day:*` 计数（谨慎） |

## 备份

K8s CronJob：`k8s/backup-cronjob.yaml`（Postgres 逻辑备份）。
