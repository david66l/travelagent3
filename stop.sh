#!/usr/bin/env bash
# TravelAgent 一键停止脚本

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PID_DIR="$PROJECT_ROOT/logs"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

stopped=0

# 从 PID 文件停止
if [ -f "$PID_DIR/backend.pid" ]; then
    pid=$(cat "$PID_DIR/backend.pid")
    if kill "$pid" 2>/dev/null; then
        log_info "已停止后端服务 (PID: $pid)"
        stopped=1
    else
        log_warn "后端进程 $pid 未运行"
    fi
    rm -f "$PID_DIR/backend.pid"
fi

if [ -f "$PID_DIR/frontend.pid" ]; then
    pid=$(cat "$PID_DIR/frontend.pid")
    if kill "$pid" 2>/dev/null; then
        log_info "已停止前端服务 (PID: $pid)"
        stopped=1
    else
        log_warn "前端进程 $pid 未运行"
    fi
    rm -f "$PID_DIR/frontend.pid"
fi

# 兜底：按端口停止
if lsof -ti:8000 >/dev/null 2>&1; then
    kill $(lsof -ti:8000) 2>/dev/null || true
    log_info "已释放端口 8000"
    stopped=1
fi

if lsof -ti:3000 >/dev/null 2>&1; then
    kill $(lsof -ti:3000) 2>/dev/null || true
    log_info "已释放端口 3000"
    stopped=1
fi

# 询问是否停止 Docker
if docker ps -q --filter "name=travel_agent_" | grep -q .; then
    echo ""
    read -p "是否同时停止 PostgreSQL 和 Redis? [y/N] " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        cd "$PROJECT_ROOT"
        docker compose down
        log_info "Docker 服务已停止"
    else
        log_info "Docker 服务保持运行"
    fi
fi

if [ $stopped -eq 0 ]; then
    log_warn "没有找到正在运行的服务"
else
    echo ""
    log_info "TravelAgent 已停止"
fi
