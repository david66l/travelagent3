#!/usr/bin/env bash
# 重启 TravelAgent2 全部本地服务（后台守护，不依赖交互式终端）
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }

stop_all() {
  log "停止旧进程..."
  for port in 8000 3000 8080 8081 8001; do
    lsof -ti:"$port" 2>/dev/null | xargs kill -9 2>/dev/null || true
  done
  pkill -9 -f "uvicorn api.main:app" 2>/dev/null || true
  pkill -9 -f "uvicorn vrp_solver_service.main:app" 2>/dev/null || true
  pkill -9 -f "celery -A core.celery_app" 2>/dev/null || true
  pkill -9 -f "next dev" 2>/dev/null || true
  pkill -9 -f "llama-server" 2>/dev/null || true
  pkill -9 -f "gateway/bin/gateway" 2>/dev/null || true
  sleep 2
}

start_all() {
  source "$PROJECT_ROOT/backend/.venv/bin/activate"
  export PYTHONPATH="$PROJECT_ROOT/backend/src"
  set -a && source "$PROJECT_ROOT/.env" && set +a

  log "启动 Qwen LLM (8081)..."
  nohup bash "$PROJECT_ROOT/scripts/start_local_llm.sh" >> logs/llama.log 2>&1 &
  echo $! > logs/llama.pid

  log "启动 VRP Solver (8001)..."
  nohup python3 -m uvicorn vrp_solver_service.main:app \
    --host 0.0.0.0 --port 8001 --app-dir "$PROJECT_ROOT/backend/src" \
    >> logs/vrp.log 2>&1 &
  echo $! > logs/vrp.pid

  log "启动 Backend (8000)..."
  nohup python3 -m uvicorn api.main:app \
    --host 0.0.0.0 --port 8000 --app-dir "$PROJECT_ROOT/backend/src" \
    >> logs/backend.log 2>&1 &
  echo $! > logs/backend.pid

  log "启动 Celery worker + beat..."
  cd "$PROJECT_ROOT/backend/src"
  nohup python3 -m celery -A core.celery_app worker \
    -Q default,memory,planning -l info \
    >> "$PROJECT_ROOT/logs/celery_worker.log" 2>&1 &
  echo $! > "$PROJECT_ROOT/logs/celery_worker.pid"
  nohup python3 -m celery -A core.celery_app beat -l info \
    >> "$PROJECT_ROOT/logs/celery_beat.log" 2>&1 &
  echo $! > "$PROJECT_ROOT/logs/celery_beat.pid"
  cd "$PROJECT_ROOT"

  log "启动 Frontend (3000)..."
  cd "$PROJECT_ROOT/frontend"
  nohup npm run dev >> "$PROJECT_ROOT/logs/frontend.log" 2>&1 &
  echo $! > "$PROJECT_ROOT/logs/frontend.pid"
  cd "$PROJECT_ROOT"

  log "启动 Gateway (8080)..."
  nohup "$PROJECT_ROOT/gateway/bin/gateway" >> logs/gateway.log 2>&1 &
  echo $! > logs/gateway.pid
}

health_check() {
  local ok=0
  curl -sf http://127.0.0.1:8081/v1/models >/dev/null 2>&1 && ok=$((ok+1)) || log "FAIL Qwen :8081"
  curl -sf http://127.0.0.1:8000/api/health >/dev/null 2>&1 && ok=$((ok+1)) || log "FAIL Backend :8000"
  curl -sf http://127.0.0.1:8001/health >/dev/null 2>&1 && ok=$((ok+1)) || log "FAIL VRP :8001"
  curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && ok=$((ok+1)) || log "FAIL Gateway :8080"
  curl -sf http://127.0.0.1:3000 >/dev/null 2>&1 && ok=$((ok+1)) || log "FAIL Frontend :3000"
  pgrep -f "celery -A core.celery_app worker" >/dev/null 2>&1 && ok=$((ok+1)) || log "FAIL Celery"
  log "健康检查: $ok/6 通过"
  [ "$ok" -ge 6 ]
}

stop_all
start_all

log "等待服务就绪..."
for i in $(seq 1 60); do
  if health_check; then
    log "全部服务已就绪"
    log "  前端   http://localhost:3000"
    log "  后端   http://localhost:8000"
    log "  网关   http://localhost:8080"
    log "  Qwen   http://localhost:8081"
    exit 0
  fi
  sleep 1
done

log "部分服务未就绪，请查看 logs/ 目录"
exit 1
