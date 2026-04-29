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

# ===== 7. 启动后端 =====
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
python3 -m uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    > "$PROJECT_ROOT/logs/backend.log" 2>&1 &

BACKEND_PID=$!

# 等待后端健康检查
log_step "等待后端启动..."
for i in $(seq 1 20); do
    if curl -s http://localhost:8000/api/health >/dev/null 2>&1; then
        log_info "后端启动成功 (PID: $BACKEND_PID)"
        break
    fi
    if [ $i -eq 20 ]; then
        log_warn "后端启动较慢，请查看 logs/backend.log"
    fi
    sleep 1
done

# ===== 8. 启动前端 =====
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

# ===== 9. 保存 PID 文件 =====
echo "$BACKEND_PID" > "$PROJECT_ROOT/logs/backend.pid"
echo "$FRONTEND_PID" > "$PROJECT_ROOT/logs/frontend.pid"

# ===== 10. 输出访问信息 =====
sleep 2
clear || true

cat <<'EOF'
╔════════════════════════════════════════════════════════════╗
║              🧳 TravelAgent 启动成功！                      ║
╠════════════════════════════════════════════════════════════╣
EOF

echo -e "║  ${GREEN}前端页面${NC}   http://localhost:3000                            ║"
echo -e "║  ${GREEN}后端 API${NC}   http://localhost:8000                            ║"
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

# ===== 11. 优雅关闭 =====
cleanup() {
    echo ""
    log_step "正在停止服务..."
    kill $FRONTEND_PID 2>/dev/null || true
    kill $BACKEND_PID 2>/dev/null || true
    wait 2>/dev/null || true
    rm -f "$PROJECT_ROOT/logs/"*.pid
    log_info "所有服务已停止"
    exit 0
}
trap cleanup INT TERM

# 保持脚本运行，用户按 Ctrl+C 触发 cleanup
wait
