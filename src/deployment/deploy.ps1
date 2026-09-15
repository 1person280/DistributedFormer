# DistributedFormer 云端部署脚本 (Windows PowerShell)
param(
    [Parameter()]
    [ValidateSet("up", "down", "restart", "logs", "status", "monitor", "build", "clean")]
    [string]$Mode = "up"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$ComposeFile = Join-Path $ScriptDir "docker-compose.yml"

Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     DistributedFormer 云端部署脚本 (PowerShell)         ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

Set-Location $ProjectDir

switch ($Mode) {
    "up" {
        Write-Host "🚀 启动 DistributedFormer 服务..." -ForegroundColor Green
        docker-compose -f $ComposeFile up -d redis
        Write-Host "⏳ 等待 Redis 就绪..." -ForegroundColor Yellow
        Start-Sleep -Seconds 3
        docker-compose -f $ComposeFile up -d pulse-network-1 pulse-network-2
        Write-Host "✅ 服务已启动!" -ForegroundColor Green
        Write-Host ""
        Write-Host "  Redis:   localhost:6379"
        Write-Host "  Pulse-1: docker container df-pulse-1"
        Write-Host "  Pulse-2: docker container df-pulse-2"
        docker-compose -f $ComposeFile ps
    }
    "down" {
        Write-Host "🛑 停止所有服务..." -ForegroundColor Red
        docker-compose -f $ComposeFile down
        Write-Host "✅ 服务已停止" -ForegroundColor Green
    }
    "restart" {
        Write-Host "🔄 重启服务..." -ForegroundColor Yellow
        docker-compose -f $ComposeFile restart
        Write-Host "✅ 服务已重启" -ForegroundColor Green
    }
    "logs" {
        Write-Host "📋 查看日志..." -ForegroundColor Cyan
        docker-compose -f $ComposeFile logs -f --tail=100
    }
    "status" {
        Write-Host "📊 服务状态:" -ForegroundColor Cyan
        docker-compose -f $ComposeFile ps
    }
    "monitor" {
        Write-Host "📡 启动监控模式..." -ForegroundColor Magenta
        docker-compose -f $ComposeFile --profile monitoring up -d
        Write-Host "  Prometheus: http://localhost:9090"
    }
    "build" {
        Write-Host "🔨 构建镜像..." -ForegroundColor Yellow
        docker-compose -f $ComposeFile build --no-cache
        Write-Host "✅ 构建完成" -ForegroundColor Green
    }
    "clean" {
        Write-Host "🧹 清理数据..." -ForegroundColor Red
        docker-compose -f $ComposeFile down -v
        Write-Host "✅ 数据已清理" -ForegroundColor Green
    }
}
