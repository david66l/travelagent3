# 性能与压测

## PRD 目标（§5.1）

| 指标 | 目标 |
|------|------|
| 首响应（HTTP 202） | < 100ms |
| 草稿行程（缓存命中） | < 500ms |
| 完整行程 P99 | < 8s |
| SSE 首事件 | P99 < 500ms |
| 并发 SSE / 实例 | ≥ 1000 |
| Celery 吞吐 | ≥ 50 task/s/worker |

## 调优手段（M6）

1. **模型分层**：`model_router` 将 intent/chat 路由到 `SMALL_MODEL`
2. **Prompt 压缩**：`prompt_compress_max_messages` / `prompt_compress_max_chars`
3. **工具缓存**：L1 本地 + Redis L2，指标 `cache_hits_total` / `cache_misses_total`
4. **成本熔断**：超预算时强制小模型，保护 GPU/API 支出

## 压测

### k6（推荐）

```bash
# 安装 k6 后
k6 run scripts/load/k6_sse.js
k6 run -e VUS=100 -e DURATION=5m scripts/load/k6_sse.js
```

### 性能门禁脚本

```bash
bash scripts/run_performance_gate.sh
```

CI：`/.github/workflows/performance.yml`

### pytest 性能标记

```bash
cd backend
uv run pytest -m performance
```

## 混沌 / 降级测试

```bash
cd backend
uv run pytest tests/chaos -m chaos
```

覆盖：成本熔断触发后模型降级、游客强制小模型等场景。

## 指标观察

Prometheus 查询示例：

```promql
histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))
histogram_quantile(0.99, sum(rate(llm_request_duration_seconds_bucket[5m])) by (le, model))
sum(rate(cache_hits_total[5m])) / (sum(rate(cache_hits_total[5m])) + sum(rate(cache_misses_total[5m])))
```

Grafana：导入 `monitoring/grafana/dashboards/travel-agent-cost.json`。
