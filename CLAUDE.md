# TravelAgent2

AI全栈自主规划旅游Agent。6 Agent LangGraph架构 + OR-Tools CP-SAT + pgvector。

## 项目结构

- `gateway/` — Go + Echo 网关
- `backend/` — Python FastAPI + Celery
- `frontend/` — Next.js 15

## 常用命令

```bash
# 启动后端
cd backend && .venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# 启动网关
set -a && . .env && ./gateway/bin/gateway

# 测试
cd backend && .venv/bin/python -m pytest tests/unit/ -q

# 构建前端
cd frontend && npm run build
```

## 可用 Skills (Claude Code)

- `design-md` — 用 Google DESIGN.md 规范创建/校验设计系统文件
