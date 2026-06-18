# 系统架构

## 概览

TravelAgent2 采用前后端分离 + 异步任务架构：

```text
Browser → Nginx Gateway → FastAPI Backend → PostgreSQL
                              ↓
                         Redis (cache / queue / state)
                              ↓
                         Celery Workers (planning / memory)
```

## 后端分层

| 层 | 目录 | 职责 |
|----|------|------|
| API | `backend/src/api/` | HTTP/SSE 入口、鉴权、限流 |
| Services | `backend/src/services/` | 业务编排 |
| Agents / Planner | `backend/src/agents/`, `planner/` | 意图识别、行程规划 DAG |
| Skills / Tools | `backend/src/skills/`, `tools/` | 外部数据（搜索、天气、POI） |
| Core | `backend/src/core/` | 配置、Redis、LLM、安全、成本 |

## 核心数据流（聊天 + 规划）

1. 用户通过 `POST /api/v1/chat/message` 提交消息
2. `process_chat_message` 识别意图，必要时创建 `PlanningJob`
3. Celery worker 执行规划 DAG，事件写入 Redis
4. 客户端通过 `GET /api/v1/chat/stream` SSE 接收阶段进度与结果

## M6 成本优化模块

| 模块 | 作用 |
|------|------|
| `cost_circuit_breaker` | 全局日 Token / API / GPU 成本熔断 |
| `user_tier` + `token_quota` | 游客/免费/会员分级配额 |
| `model_router` | 简单意图与熔断时走小模型 |
| `prompt_compress` | 长对话截断，降低 Token |
| `external_api_tracker` | Tavily 等外部调用计数与成本估算 |

## 部署拓扑（K8s）

见 `k8s/`：backend、celery、gateway、frontend、Redis、Postgres、HPA、PDB。

可观测栈见 `monitoring/` + `docker-compose.observability.yml`。
