# TravelAgent2

AI 旅行规划助手，基于 SSE 流式对话 + 异步行程规划 + Redis/Celery 任务队列。支持游客/会员分级配额、成本熔断、模型分层路由、Prometheus 指标与 K8s 部署。

当前架构、指标口径、发布门禁和未完成的生产证明见 [`docs/INDUSTRIAL_READINESS.md`](docs/INDUSTRIAL_READINESS.md)。该文档区分了“已验证”“历史消融”和“仍需真实集群验证”，避免把离线实验写成线上能力。

## 项目概述

TravelAgent2 接收用户的自然语言旅行需求。FastAPI 先持久化 `PlanningJob`，再由 Celery Worker 执行可恢复的 LangGraph Agent Loop，并以 Server-Sent Events（SSE）实时推送进度。Agent 会逐轮选择工具、读取 Observation、更新 Agent Ledger，并在搜索不足或 Verifier 失败时继续搜索或重规划；CP-SAT 负责硬约束求解，模型不能自行宣布行程有效。关键事件写入 PostgreSQL，Redis 用于队列、热状态与实时通知，因此客户端可以通过 `job_id + last_event_id` 断线续传。系统包含：

- **前端**：Next.js 15 + React 19 + TypeScript + Tailwind CSS
- **后端**：FastAPI + SQLAlchemy 2（asyncpg）+ Pydantic v2
- **任务队列**：Celery + Redis
- **网关**：Go + Echo（JWT 鉴权、限流、熔断、路由）
- **可观测性**：Prometheus + Grafana + Loki + OpenTelemetry

## 架构图

```text
                          ┌─────────────────┐
                          │   Browser/CLI   │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │  Nginx / Ingress │
                          └────────┬────────┘
                                   │
┌──────────────────────────────────┼──────────────────────────────────┐
│                          Kubernetes Cluster                         │
│                                                                     │
│   ┌──────────────┐              ┌──────────────┐                    │
│   │   Gateway    │──────────────│   Frontend   │                    │
│   │  (Go/Echo)   │              │  (Next.js)   │                    │
│   │ JWT/限流/熔断 │              └──────────────┘                    │
│   └──────┬───────┘                                                  │
│          │                                                          │
│          ▼                                                          │
│   ┌──────────────┐         ┌──────────────┐    ┌──────────────┐     │
│   │    Backend   │◄───────►│    Redis     │    │  PostgreSQL  │     │
│   │  (FastAPI)   │ 通知/热态│  cache/queue │    │ jobs/events  │     │
│   └──────┬───────┘         └──────┬───────┘    └──────────────┘     │
│          │                        │                                 │
│          ▼                        ▼                                 │
│   ┌──────────────┐         ┌──────────────┐                         │
│   │ Celery Worker│◄────────│ Celery Beat  │                         │
│   │   planning   │         │   scheduler  │                         │
│   └──────────────┘         └──────────────┘                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

主请求链路为：`POST 消息 → PlanningJob 事务提交 → Celery/LangGraph → PlanningJobEvent → SSE 重放`。FastAPI 进程内不保存任务完成状态；Redis Pub/Sub 丢失时，SSE 仍可从 PostgreSQL 事件日志恢复。

核心规划闭环为：`结构化意图识别/补槽 → ReAct 模型逐轮选工具 → Guard/工具执行 → Observation/Facts/Artifacts → 证据充分性检查 → CP-SAT/Greedy → 硬 Verifier → 继续搜索、重试、重规划、询问或安全终止 → 用户确认/修改`。每个串行动作或安全的并行只读批次都形成可恢复 checkpoint；同一 Agent Ledger 保存预算、失败、动作历史和版本化证据，避免失败后无界循环或偷偷切回旧 Planner。旧 `controller_first`/`policy_driven` DAG 只保留为消融基线，不是生产默认入口。

K8s 部署清单位于 [`k8s/`](k8s/) 目录：

| 资源 | 文件 | 说明 |
|------|------|------|
| Gateway | [`k8s/gateway-deployment.yaml`](k8s/gateway-deployment.yaml) | JWT 鉴权、限流、熔断、路由，副本数 3 |
| Backend | [`k8s/backend-deployment.yaml`](k8s/backend-deployment.yaml) | FastAPI 业务 API；数据库迁移由单一 Sync Hook Job 执行 |
| Frontend | [`k8s/frontend-deployment.yaml`](k8s/frontend-deployment.yaml) | Next.js 静态/SSR 服务 |
| Celery Worker | [`k8s/celery-worker-deployment.yaml`](k8s/celery-worker-deployment.yaml) | 异步规划任务执行 |
| Celery Beat | [`k8s/celery-beat-deployment.yaml`](k8s/celery-beat-deployment.yaml) | 定时任务调度 |
| Redis | [`k8s/redis-statefulset.yaml`](k8s/redis-statefulset.yaml) + [`k8s/redis-cluster-init-job.yaml`](k8s/redis-cluster-init-job.yaml) | 6 节点缓存/队列；独立幂等 Job 初始化集群 |
| PostgreSQL | [`k8s/postgres-statefulset.yaml`](k8s/postgres-statefulset.yaml) + [`k8s/postgres-service.yaml`](k8s/postgres-service.yaml) | pgvector 持久化；主写和只读 Service 分离 |
| Migration | [`k8s/migration-job.yaml`](k8s/migration-job.yaml) | ArgoCD wave -1 单实例 Alembic 门禁 |
| NetworkPolicy | [`k8s/network-policies.yaml`](k8s/network-policies.yaml) | 默认拒绝，仅开放必要服务依赖和外部 HTTPS |
| HPA | [`k8s/hpa.yaml`](k8s/hpa.yaml) | 水平自动扩缩容 |
| PDB | [`k8s/pdb.yaml`](k8s/pdb.yaml) |  Pod 中断预算 |

## 快速开始

### 方式一：Docker Compose（推荐）

```bash
# 1. 克隆仓库并进入目录
cd TravelAgent2

# 2. 准备环境变量
cp .env.example .env
# 编辑 .env，至少配置所选模型 Provider Key / JWT_SECRET / PRIVACY_ENCRYPTION_KEY

# 3. 一键启动全部服务
docker compose up -d

# 4. 查看服务状态
docker compose ps

# 5. 冒烟测试（需服务全部就绪）
python3 scripts/e2e_smoke.py
```

服务入口：

| 服务 | 地址 |
|------|------|
| Gateway | http://127.0.0.1:8080 |
| Backend | http://127.0.0.1:8000 |
| Frontend | http://127.0.0.1:3000 |

### 方式二：本地开发

```bash
# 依赖服务
docker compose up -d postgres redis

# 后端（使用 uv）
cd backend
uv pip install -e ".[dev]"
uv run alembic upgrade head
uv run uvicorn api.main:app --reload --port 8000

# 前端
cd frontend
npm install
npm run dev

# Celery Worker
cd backend
uv run celery -A core.celery_app worker -Q default,planning -l info
```

### 可观测性

```bash
docker compose -f docker-compose.observability.yml up -d
```

Grafana 大盘位于 [`monitoring/grafana/dashboards/`](monitoring/grafana/dashboards/)。

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Next.js 15, React 19, TypeScript 5, Tailwind CSS, Zustand, Lucide React |
| 后端 | FastAPI, SQLAlchemy 2 (asyncpg), Pydantic v2, Celery, Structlog |
| 网关 | Go 1.26 toolchain, Echo v4, JWT, go-redis, Prometheus client |
| 数据库 | PostgreSQL 15（Compose 兼容卷）/ PostgreSQL 16 + pgvector（K8s） |
| 缓存/队列 | Redis 7 |
| 模型/AI | DeepSeek/OpenAI-compatible API, Tavily Search, 自定义模型路由；Qwen 学生模型离线/Shadow 候选 |
| 可观测性 | Prometheus, Grafana, Loki, OpenTelemetry, Alertmanager |
| 部署 | Docker, Docker Compose, Kubernetes, ArgoCD, Argo Rollouts |
| 测试 | pytest (async), Playwright, Go testing, k6 |

## 目录结构

```text
TravelAgent2/
├── backend/              # FastAPI 后端
│   ├── src/
│   │   ├── api/          # HTTP/SSE 路由与入口
│   │   ├── core/         # 配置、Redis、LLM、安全、熔断、成本
│   │   ├── services/     # 业务编排
│   │   ├── agents/       # 意图识别、实时查询
│   │   ├── planner/      # 行程规划 DAG
│   │   ├── skills/       # 外部数据（搜索、天气、POI、价格）
│   │   ├── worker/       # Celery 任务
│   │   └── models/       # SQLAlchemy 模型
│   ├── tests/            # 单元/集成/混沌/安全测试
│   ├── migrations/       # Alembic 数据库迁移
│   └── pyproject.toml    # Python 依赖与工具配置
├── frontend/             # Next.js 前端
│   ├── src/app/          # App Router
│   ├── src/components/   # React 组件
│   ├── src/hooks/        # 自定义 Hooks
│   ├── src/stores/       # Zustand 状态管理
│   └── package.json
├── gateway/              # Go 网关
│   ├── cmd/gateway/      # 入口
│   └── internal/         # auth、limit、breaker、proxy、middleware
├── k8s/                  # Kubernetes 部署清单
├── monitoring/           # Prometheus/Grafana/Loki/OTel 配置
├── scripts/              # 冒烟测试、压测、部署脚本
├── ml/                   # 模型适配与训练
└── docs/                 # 架构、运维、性能文档
```

## API 文档

启动 Backend 后访问：

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

主要接口：

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/auth/guest` | 游客登录 |
| POST | `/api/v1/auth/register` | 用户注册 |
| POST | `/api/v1/auth/login` | 用户登录 |
| GET  | `/api/v1/conversations` | 会话列表 |
| POST | `/api/v1/conversations` | 创建会话 |
| POST | `/api/v1/chat/message` | 发送消息 |
| GET  | `/api/v1/chat/stream` | SSE 流式进度 |
| GET  | `/api/v1/planning-jobs/{job_id}` | 查询规划任务 |
| GET  | `/api/v1/itineraries` | 行程列表 |
| GET  | `/api/v1/metrics` | Prometheus 指标 |
| GET  | `/api/health` | 健康检查 |
| GET  | `/api/ready` | 就绪探针 |

## 测试

```bash
# 后端测试
cd backend
source .venv/bin/activate
pytest -q

# 前端构建/类型检查
cd frontend
npm install
npm run build

# Go 网关测试
cd gateway
go test ./...
```

## 更多文档

| 文档 | 说明 |
|------|------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 系统架构与模块边界 |
| [`docs/OPERATIONS.md`](docs/OPERATIONS.md) | 部署、监控、告警与故障处理 |
| [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) | 性能目标、压测与调优 |
| [`docs/interview/TravelAgent_竞赛经历与面试讲解稿.md`](docs/interview/TravelAgent_竞赛经历与面试讲解稿.md) | 竞赛项目定位、架构讲解、简历要点与面试追问 |
| [`docs/interview/TravelAgent_主张证据账本.json`](docs/interview/TravelAgent_主张证据账本.json) | 可陈述指标、证据来源与表达边界 |
| [`PRD_AI全栈高并发改造.md`](PRD_AI全栈高并发改造.md) | 完整改造 PRD |
