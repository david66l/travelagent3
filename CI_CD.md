# CI/CD 配置说明

## 工作流概览

| 工作流 | 触发条件 | 任务 |
|--------|---------|------|
| `backend.yml` | `backend/**` 变更 | Lint → Type Check → Test (pytest) → Security Scan |
| `frontend.yml` | `frontend/**` 变更 | Lint → Type Check → Build |
| `docker.yml` | 全栈变更 | Build Backend Image → Build Frontend Image → Compose Up + Health Check |
| `dependabot.yml` | 定时 | Python/npm/GHA 依赖自动更新 PR |

## 快速开始

### 本地验证（提交前）

```bash
# 后端检查
cd backend
ruff check src/ && ruff format --check src/
mypy src/ --ignore-missing-imports
pytest --cov=src --cov-report=term

# 前端检查
cd frontend
npm run lint
npx tsc --noEmit
npm run build
```

### Docker 本地启动

```bash
# 1. 复制环境变量模板
cp .env.example .env
# 编辑 .env，填入真实的 OPENAI_API_KEY 和 TAVILY_API_KEY

# 2. 启动全部服务
docker compose up -d --build

# 3. 验证
open http://localhost:3000      # 前端
curl http://localhost:8000/api/health  # 后端健康检查
```

## 环境变量

| 变量 | 必需 | 说明 |
|------|------|------|
| `OPENAI_API_KEY` | ✅ | LLM API 密钥 |
| `TAVILY_API_KEY` | ❌ | 搜索 API 密钥（不填则回退 DuckDuckGo） |
| `DATABASE_URL` | ✅ | PostgreSQL 连接串 |
| `REDIS_URL` | ✅ | Redis 连接串 |

生产部署时请使用安全的密钥管理服务，**切勿将真实密钥提交到仓库**。

## 镜像说明

| 镜像 | Dockerfile | 基础镜像 | 大小优化 |
|------|-----------|---------|---------|
| Backend | `backend/Dockerfile` | `python:3.13-slim` | 多阶段构建 + uv |
| Frontend | `frontend/Dockerfile` | `node:20-alpine` | 多阶段构建 + standalone |

## 覆盖率门禁

当前阈值：**80%**（实际 82.40%）

如需调整，修改 `backend/pyproject.toml`：
```toml
[tool.coverage.report]
fail_under = 80
```

## 安全扫描

- **Bandit**：Python 代码安全漏洞扫描
- **pip-audit**：依赖包已知漏洞扫描

扫描结果不会阻断 CI（`|| true`），但会在 PR 中展示报告供人工审查。
