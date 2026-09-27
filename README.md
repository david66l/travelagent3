# TravelAgent2

AI 旅行规划助手，基于 SSE 流式对话 + 异步行程规划 + Redis/Celery 任务队列。支持游客/会员分级配额、成本熔断、模型接入、Prometheus 指标与 K8s 部署。

当前架构、指标口径、发布门禁和未完成的生产证明见 [`docs/INDUSTRIAL_READINESS.md`](docs/INDUSTRIAL_READINESS.md)。该文档区分了“已验证”“历史消融”和“仍需真实集群验证”，避免把离线实验写成线上能力。

最新后训练方案见 [`docs/POSTTRAINING_PLAN.md`](docs/POSTTRAINING_PLAN.md)：以提升小模型在 Agent Loop 中的决策能力为目标，按“职责盘点 → 基线评测 → 示范与纠偏 SFT → 条件满足后的 GRPO → 独立验收”推进，包含白话解释、任务清单和验收条件。

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

核心规划闭环统一为 **Agent Loop + Harness (`agent-harness-v1`)**：模型每轮决定一个动作，harness 检查权限、参数、预算和前置证据，执行后把结果交回模型。求解、校验、失败恢复、追问与提交时机均由模型选择。在线与 TRL 训练共用同一循环；外层 LangGraph 只负责对话、checkpoint、展示和用户确认。旧 DAG、控制器代决策、specialist/shadow 包装及专用训练环境已从工作代码删除。

当前架构的职责边界、13 个动作和多轮交互见 [`docs/AGENT_HARNESS_ARCHITECTURE.md`](docs/AGENT_HARNESS_ARCHITECTURE.md)。旧实验报告保留为历史证据；旧运行 checkpoint 需要新建会话，旧教师前缀语料需要重建。新的模型能力基线尚待测量。

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

> **⚠️ 生产环境必须让浏览器流量走网关。** 默认 compose 把 `NEXT_PUBLIC_API_URL` 留空，由 Next.js 的 rewrite 直接发往 `backend:8000`（`frontend/next.config.js`），同时 `docker-compose.yml` 还把 `8000:8000` 发布到宿主机。这条路径**绕过了网关的 JWT 鉴权、限流与熔断**——后端自身仍会校验 Bearer token，但边缘策略不生效。生产部署请设置 `NEXT_PUBLIC_API_URL` 指向网关（如 `https://api.example.com`），并**不要**对外发布后端端口；同时用 `GATEWAY_CORS_ORIGINS` 列出真实的前端源（默认仅 `localhost:3000`，且因启用凭据不接受通配符）。

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
├── scripts/              # 冒烟测试、压测、语料构建/审计/评测/晋升门脚本
├── ml/                   # 后训练（SFT/GRPO/DPO）训练入口与训练库
├── experiments/          # 实验总账（每个 run 一行：指标与裁决）
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

## 后训练与实验治理

Qwen 学生模型的后训练（SFT / GRPO / DPO，Qwen3-1.7B + QLoRA 单卡）与全链路使用同一套参数合同与 chat template（SHA 锁定），训练入口在 [`ml/agentic/training/`](ml/agentic/training/)，语料构建/审计/评测/晋升门脚本在 [`scripts/`](scripts/)。

- **实验总账**：[`experiments/index.md`](experiments/index.md) —— 每个 run 一行（假设来源、关键指标、裁决），当前 107 个受控 run。
- **统一工具入口**：`python scripts/travelctl.py list` 列出全部语料构建/审计/评测/晋升脚本（140 个，按动词族组织）；`travelctl <verb> <subject> [args...]` 派发执行，旧脚本入口全部保留。
- **当前状态（2026-09）**：所有后训练 checkpoint 处于 quarantine，无一晋升生产路由；160 题 sealed test 未解封。历史已验证的晋升案例：终止边界 SFT（多动作 0/150 → 150/150）、级联蒸馏（Base 106/150 → SFT 135/150）。
- **可复现性纪律**：正式训练运行于 git 仓库内，`training_report.json` 必须携带可解析的 `git_commit`（无法解析时训练入口直接拒绝开跑）。

## 更多文档

| 文档 | 说明 |
|------|------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 系统架构与模块边界 |
| [`docs/OPERATIONS.md`](docs/OPERATIONS.md) | 部署、监控、告警与故障处理 |
| [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) | 性能目标、压测与调优 |
| [`docs/interview/TravelAgent_竞赛经历与面试讲解稿.md`](docs/interview/TravelAgent_竞赛经历与面试讲解稿.md) | 竞赛项目定位、架构讲解、简历要点与面试追问 |
| [`docs/interview/TravelAgent_主张证据账本.json`](docs/interview/TravelAgent_主张证据账本.json) | 可陈述指标、证据来源与表达边界 |
| [`PRD_AI全栈高并发改造.md`](PRD_AI全栈高并发改造.md) | 完整改造 PRD |
