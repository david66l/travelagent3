#!/usr/bin/env bash
# TravelAgent 一键启动脚本
# 启动基础设施(PostgreSQL + Redis) + 后端 + 前端

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# 颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "${BLUE}[STEP]${NC}  $1"; }

# ===== 1. 检查依赖 =====
log_step "检查系统依赖..."

missing=()
command -v python3      >/dev/null 2>&1 || missing+=("python3")
command -v npm          >/dev/null 2>&1 || missing+=("npm")

if [ ${#missing[@]} -gt 0 ]; then
    log_error "缺少以下依赖: ${missing[*]}"
    echo "请安装后再试。"
    exit 1
fi

# Docker 可选（优先本地数据库）
if ! command -v docker >/dev/null 2>&1; then
    log_warn "Docker 未安装，将尝试使用本地 PostgreSQL/Redis"
fi

log_info "系统依赖检查通过"

# ===== 2. 启动基础设施 (优先本地服务， fallback Docker) =====
log_step "检查 PostgreSQL + Redis..."

# 检测本地 PostgreSQL（brew 安装）
pg_ready_local() {
    pg_isready -h localhost -p 5432 -U travelagent -d travel_agent >/dev/null 2>&1
}

# 检测本地 Redis（brew 安装）
redis_ready_local() {
    redis-cli -h localhost -p 6379 ping 2>/dev/null | grep -q PONG
}

# 检测 Docker PostgreSQL
pg_ready_docker() {
    docker exec travel_agent_postgres pg_isready -U travelagent -d travel_agent >/dev/null 2>&1
}

# 检测 Docker Redis
redis_ready_docker() {
    docker exec travel_agent_redis redis-cli ping 2>/dev/null | grep -q PONG
}

USE_DOCKER=false

if pg_ready_local && redis_ready_local; then
    log_info "检测到本地 PostgreSQL 和 Redis 已运行，跳过 Docker"
elif pg_ready_docker && redis_ready_docker; then
    log_info "检测到 Docker PostgreSQL 和 Redis 已在运行"
    USE_DOCKER=true
else
    # 尝试启动本地服务
    if command -v pg_isready >/dev/null 2>&1 && command -v redis-cli >/dev/null 2>&1; then
        log_warn "本地 PostgreSQL/Redis 未运行，尝试启动..."
        if brew services start postgresql@16 >/dev/null 2>&1 || brew services start postgresql >/dev/null 2>&1; then
            sleep 2
        fi
        if brew services start redis >/dev/null 2>&1; then
            sleep 1
        fi

        # 再次检测
        if pg_ready_local && redis_ready_local; then
            log_info "本地服务启动成功"
        else
            USE_DOCKER=true
        fi
    else
        USE_DOCKER=true
    fi
fi

# 如需 Docker，尝试启动
if [ "$USE_DOCKER" = true ]; then
    if ! command -v docker >/dev/null 2>&1; then
        log_error "未检测到本地 PostgreSQL/Redis，且 Docker 未安装"
        echo ""
        echo "请执行以下命令安装本地数据库："
        echo "  brew install postgresql@16 redis"
        echo "  brew services start postgresql@16"
        echo "  brew services start redis"
        echo "  createuser -s travelagent"
        echo "  createdb -O travelagent travel_agent"
        echo ""
        exit 1
    fi

    start_container() {
        local name=$1
        local svc=$2
        if docker ps -q --filter "name=$name" | grep -q .; then
            log_warn "$svc 已在运行，跳过"
            return 0
        fi
        if ! docker compose up -d "$svc" 2>/dev/null; then
            return 1
        fi
        return 0
    }

    if ! start_container "travel_agent_postgres" "PostgreSQL"; then
        echo ""
        log_error "拉取 PostgreSQL 镜像失败"
        echo ""
        echo -e "${YELLOW}解决方案（选其一）：${NC}"
        echo ""
        echo "1. 配置 Docker 镜像加速"
        echo "   Docker Desktop → Settings → Docker Engine → registry-mirrors"
        echo ""
        echo "2. 用 Homebrew 安装本地数据库（推荐）"
        echo "   brew install postgresql@16 redis"
        echo "   brew services start postgresql@16"
        echo "   brew services start redis"
        echo "   createuser -s travelagent"
        echo "   createdb -O travelagent travel_agent"
        echo ""
        exit 1
    fi

    if ! start_container "travel_agent_redis" "Redis"; then
        log_error "拉取 Redis 镜像失败"
        exit 1
    fi
fi

# ===== 3. 等待服务就绪 =====
log_step "等待数据库就绪..."
RETRIES=30
for i in $(seq 1 $RETRIES); do
    if pg_ready_local || pg_ready_docker; then
        log_info "PostgreSQL 就绪"
        break
    fi
    if [ $i -eq $RETRIES ]; then
        log_error "PostgreSQL 启动超时"
        exit 1
    fi
    sleep 1
done

log_step "等待 Redis 就绪..."
for i in $(seq 1 $RETRIES); do
    if redis_ready_local || redis_ready_docker; then
        log_info "Redis 就绪"
        break
    fi
    if [ $i -eq $RETRIES ]; then
        log_error "Redis 启动超时"
        exit 1
    fi
    sleep 1
done

# ===== 4. Python 虚拟环境 =====
VENV_DIR="$PROJECT_ROOT/backend/.venv"
if [ ! -d "$VENV_DIR" ]; then
    log_step "创建 Python 虚拟环境..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ===== 5. 安装后端依赖 =====
log_step "检查并安装后端依赖..."
cd "$PROJECT_ROOT/backend"
if ! pip show travel-agent >/dev/null 2>&1; then
    pip install -e . >/dev/null 2>&1
    log_info "后端依赖安装完成"
else
    log_info "后端依赖已安装"
fi

# ===== 6. 创建日志目录 =====
mkdir -p "$PROJECT_ROOT/logs"

# ===== 7. 启动 VRP 求解服务 (port 8001) =====
# 路线求解跑在独立服务里，主后端通过 VRP_SOLVER_URL 调它。必须随后端一起启动，
# 且带 --reload —— 否则改了 solver 代码这个服务不会热加载，会出现「后端是新代码、
# 求解服务还是旧代码」的半新半旧状态（改 .env 仍需手动重启，--reload 不重载环境变量）。
log_step "启动 VRP 求解服务 (port 8001)..."

if lsof -ti:8001 >/dev/null 2>&1; then
    log_warn "端口 8001 已被占用，尝试释放..."
    kill $(lsof -ti:8001) 2>/dev/null || true
    sleep 1
fi

export PYTHONPATH="$PROJECT_ROOT/backend/src"
cd "$PROJECT_ROOT/backend/src"

python3 -m uvicorn vrp_solver_service.main:app \
    --host 0.0.0.0 \
    --port 8001 \
    --reload \
    --timeout-graceful-shutdown 10 \
    > "$PROJECT_ROOT/logs/vrp_solver.log" 2>&1 &

VRP_PID=$!

log_step "等待 VRP 求解服务启动..."
for i in $(seq 1 20); do
    if curl -s http://127.0.0.1:8001/health >/dev/null 2>&1; then
        log_info "VRP 求解服务启动成功 (PID: $VRP_PID)"
        break
    fi
    if [ $i -eq 20 ]; then
        log_warn "VRP 求解服务启动较慢，请查看 logs/vrp_solver.log"
    fi
    sleep 1
done

# ===== 8. 启动后端 =====
log_step "启动后端服务 (port 8000)..."

# 如果已有后端在运行，先停止
if lsof -ti:8000 >/dev/null 2>&1; then
    log_warn "端口 8000 已被占用，尝试释放..."
    kill $(lsof -ti:8000) 2>/dev/null || true
    sleep 1
fi

export PYTHONPATH="$PROJECT_ROOT/backend/src"
cd "$PROJECT_ROOT/backend/src"

# 使用 python3 直接启动 uvicorn，捕获 PID
# --timeout-graceful-shutdown 10: 前端的 WebSocket 长连接会让 --reload 卡在
# "Waiting for connections to close" 永不重启（改代码热加载就把后端挂死）。限时
# 10s 后强制断连，热加载就不会再被 WS 卡住。
python3 -m uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --timeout-graceful-shutdown 10 \
    > "$PROJECT_ROOT/logs/backend.log" 2>&1 &

BACKEND_PID=$!

# 等待后端健康检查
log_step "等待后端启动..."
for i in $(seq 1 20); do
    if curl -s http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
        log_info "后端启动成功 (PID: $BACKEND_PID)"
        break
    fi
    if [ $i -eq 20 ]; then
        log_warn "后端启动较慢，请查看 logs/backend.log"
    fi
    sleep 1
done

# ===== 9. 启动 Go 网关 (port 8080) =====
# 边缘鉴权/限流/熔断 + SSE 流式透传；前端通过它访问后端。两点关键：
#  • 子 shell 里加载 .env，拿到与后端一致的 JWT_SECRET，否则网关会拒掉所有 token；
#  • WriteTimeout=0，否则默认 30s 会掐断 SSE 长连接（流式输出中途断流）。
if [ -x "$PROJECT_ROOT/gateway/bin/gateway" ]; then
    log_step "启动 Go 网关 (port 8080)..."
    if lsof -ti:8080 >/dev/null 2>&1; then
        kill $(lsof -ti:8080) 2>/dev/null || true
        sleep 1
    fi
    (
        set -a
        [ -f "$PROJECT_ROOT/.env" ] && . "$PROJECT_ROOT/.env"
        set +a
        export GATEWAY_PORT=8080 GATEWAY_WRITE_TIMEOUT=0s
        export BACKEND_URL="http://localhost:8000" FRONTEND_URL="http://localhost:3000"
        exec "$PROJECT_ROOT/gateway/bin/gateway"
    ) > "$PROJECT_ROOT/logs/gateway.log" 2>&1 &
    GATEWAY_PID=$!
    for i in $(seq 1 15); do
        if curl -s http://127.0.0.1:8080/health >/dev/null 2>&1; then
            log_info "网关启动成功 (PID: $GATEWAY_PID)"
            break
        fi
        if [ $i -eq 15 ]; then
            log_warn "网关启动较慢，请查看 logs/gateway.log"
        fi
        sleep 1
    done
else
    GATEWAY_PID=""
    log_warn "未找到 gateway/bin/gateway，跳过网关（cd gateway && make build 构建）"
fi

# ===== 10. 启动 Celery Worker / Beat =====
log_step "启动 Celery worker + beat..."

cd "$PROJECT_ROOT/backend/src"

# --concurrency=2: 默认 prefork 会按 CPU 核数(8) 起满 worker，叠加 Postgres+Redis+
# 两个 uvicorn+llama(mlock 4.4G)+Next.js，16GB 必然 swap，反过来拖慢所有推理。
# 规划是低频任务，2 个并发足够，省下的内存直接喂给 LLM/求解。
python3 -m celery -A core.celery_app worker \
    -Q default,memory,planning -l info \
    --concurrency=2 \
    > "$PROJECT_ROOT/logs/celery_worker.log" 2>&1 &
CELERY_WORKER_PID=$!

python3 -m celery -A core.celery_app beat \
    -l info \
    > "$PROJECT_ROOT/logs/celery_beat.log" 2>&1 &
CELERY_BEAT_PID=$!

log_info "Celery worker 启动成功 (PID: $CELERY_WORKER_PID)"
log_info "Celery beat 启动成功 (PID: $CELERY_BEAT_PID)"

# ===== 11. 启动前端 =====
log_step "启动前端服务 (port 3000)..."

if lsof -ti:3000 >/dev/null 2>&1; then
    log_warn "端口 3000 已被占用，尝试释放..."
    kill $(lsof -ti:3000) 2>/dev/null || true
    sleep 1
fi

cd "$PROJECT_ROOT/frontend"
npm run dev > "$PROJECT_ROOT/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!

log_info "前端启动成功 (PID: $FRONTEND_PID)"

# ===== 12. 保存 PID 文件 =====
[ -n "$GATEWAY_PID" ] && echo "$GATEWAY_PID" > "$PROJECT_ROOT/logs/gateway.pid"
echo "$VRP_PID" > "$PROJECT_ROOT/logs/vrp_solver.pid"
echo "$BACKEND_PID" > "$PROJECT_ROOT/logs/backend.pid"
echo "$FRONTEND_PID" > "$PROJECT_ROOT/logs/frontend.pid"
echo "$CELERY_WORKER_PID" > "$PROJECT_ROOT/logs/celery_worker.pid"
echo "$CELERY_BEAT_PID" > "$PROJECT_ROOT/logs/celery_beat.pid"

# ===== 13. 输出访问信息 =====
sleep 2
clear || true

cat <<'EOF'
╔════════════════════════════════════════════════════════════╗
║              🧳 TravelAgent 启动成功！                      ║
╠════════════════════════════════════════════════════════════╣
EOF

echo -e "║  ${GREEN}前端页面${NC}   http://localhost:3000                            ║"
echo -e "║  ${GREEN}API 网关${NC}   http://localhost:8080  (前端经此访问后端)        ║"
echo -e "║  ${GREEN}后端 API${NC}   http://localhost:8000                            ║"
echo -e "║  ${GREEN}求解服务${NC}   http://localhost:8001                            ║"
echo -e "║  ${GREEN}API 文档${NC}   http://localhost:8000/docs                       ║"
echo -e "║  ${GREEN}WebSocket${NC}  ws://localhost:8000/ws/chat/{session_id}         ║"
echo    "╠════════════════════════════════════════════════════════════╣"
echo    "║  日志文件:                                                  ║"
echo -e "║    后端  → logs/backend.log                                ║"
echo -e "║    前端  → logs/frontend.log                               ║"
echo    "╠════════════════════════════════════════════════════════════╣"
echo    "║  操作:                                                      ║"
echo -e "║    查看后端日志  ${YELLOW}tail -f logs/backend.log${NC}                  ║"
echo -e "║    查看前端日志  ${YELLOW}tail -f logs/frontend.log${NC}                 ║"
echo -e "║    停止所有服务  ${YELLOW}./stop.sh${NC}                                 ║"
echo    "╚════════════════════════════════════════════════════════════╝"

# ===== 14. 优雅关闭 =====
cleanup() {
    echo ""
    log_step "正在停止服务..."
    kill $FRONTEND_PID 2>/dev/null || true
    kill $BACKEND_PID 2>/dev/null || true
    [ -n "$GATEWAY_PID" ] && kill $GATEWAY_PID 2>/dev/null || true
    kill $VRP_PID 2>/dev/null || true
    kill $CELERY_WORKER_PID 2>/dev/null || true
    kill $CELERY_BEAT_PID 2>/dev/null || true
    wait 2>/dev/null || true
    rm -f "$PROJECT_ROOT/logs/"*.pid
    log_info "所有服务已停止"
    exit 0
}
trap cleanup INT TERM

# 保持脚本运行，用户按 Ctrl+C 触发 cleanup
wait
