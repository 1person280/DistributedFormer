#!/bin/bash
# DistributedFormer 云端部署脚本 (Linux/macOS/Git Bash)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     DistributedFormer 云端部署脚本                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 解析参数
MODE="${1:-up}"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

cd "$PROJECT_DIR"

case "$MODE" in
    up)
        echo "🚀 启动 DistributedFormer 服务..."
        docker-compose -f "$COMPOSE_FILE" up -d redis
        echo "⏳ 等待 Redis 就绪..."
        sleep 3
        docker-compose -f "$COMPOSE_FILE" up -d pulse-network-1 pulse-network-2
        echo "✅ 服务已启动!"
        echo ""
        echo "  Redis:      localhost:6379"
        echo "  Pulse-1:    docker container df-pulse-1"
        echo "  Pulse-2:    docker container df-pulse-2"
        echo ""
        docker-compose -f "$COMPOSE_FILE" ps
        ;;
    
    down)
        echo "🛑 停止所有服务..."
        docker-compose -f "$COMPOSE_FILE" down
        echo "✅ 服务已停止"
        ;;
    
    restart)
        echo "🔄 重启服务..."
        docker-compose -f "$COMPOSE_FILE" restart
        echo "✅ 服务已重启"
        ;;
    
    logs)
        echo "📋 查看日志..."
        docker-compose -f "$COMPOSE_FILE" logs -f --tail=100
        ;;
    
    status)
        echo "📊 服务状态:"
        docker-compose -f "$COMPOSE_FILE" ps
        echo ""
        echo "📈 Redis 信息:"
        docker exec df-redis redis-cli info stats 2>/dev/null | grep -E "keyspace_hits|keyspace_misses" || echo "  Redis 未运行"
        ;;
    
    monitor)
        echo "📡 启动监控模式..."
        docker-compose -f "$COMPOSE_FILE" --profile monitoring up -d
        echo "  Prometheus: http://localhost:9090"
        ;;
    
    build)
        echo "🔨 构建镜像..."
        docker-compose -f "$COMPOSE_FILE" build --no-cache
        echo "✅ 构建完成"
        ;;
    
    clean)
        echo "🧹 清理数据..."
        docker-compose -f "$COMPOSE_FILE" down -v
        echo "✅ 数据已清理"
        ;;
    
    *)
        echo "用法: $0 {up|down|restart|logs|status|monitor|build|clean}"
        echo ""
        echo "  up       - 启动服务"
        echo "  down     - 停止服务"
        echo "  restart  - 重启服务"
        echo "  logs     - 查看日志"
        echo "  status   - 查看状态"
        echo "  monitor  - 启动监控"
        echo "  build    - 构建镜像"
        echo "  clean    - 清理数据卷"
        exit 1
        ;;
esac
