# run.ps1 - one-click start of the DistributedFormer web console + open browser
# Usage:
#   .\run.ps1                  # default port 8011, start service and open browser
#   .\run.ps1 -Port 9000       # custom port
#   .\run.ps1 -NoBrowser       # start service only, no browser
#   .\run.ps1 -Stop            # stop the previously started service process
#   .\run.ps1 -Restart         # force-stop ALL residual ui processes on the port, then start fresh
#   .\run.ps1 -Restart -NoBrowser  # restart without opening a browser

param(
    [int]$Port = 8011,
    [switch]$NoBrowser,
    [switch]$Stop,
    [switch]$Restart
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$host_ = '127.0.0.1'
$url = "http://${host_}:${Port}/workflow"
$pidFile = Join-Path $root '.run.ps1.pid'

# ---- Stop mode ----
if ($Stop) {
    if (Test-Path $pidFile) {
        $old = Get-Content $pidFile | ForEach-Object { [int]$_ } | Where-Object { $_ -gt 0 }
        if ($old) {
            Get-Process -Id $old -ErrorAction SilentlyContinue | Stop-Process -Force
            Remove-Item $pidFile -ErrorAction SilentlyContinue
            Write-Host "[run.ps1] stopped service PID=$old" -ForegroundColor DarkYellow
        }
    }
    exit 0
}

# ---- Restart mode: force-clear ALL residual ui processes on the port, then start fresh ----
if ($Restart) {
    Write-Host "[run.ps1] restart: force-clearing residual ui processes on port $Port ..." -ForegroundColor Cyan
    $ids = New-Object System.Collections.Generic.HashSet[int]
    # 1) stop the service process recorded by this script
    if (Test-Path $pidFile) {
        foreach ($old in (Get-Content $pidFile)) {
            $n = 0; [int]::TryParse($old, [ref]$n) | Out-Null
            if ($n -gt 0) { [void]$ids.Add($n) }
        }
        Remove-Item $pidFile -ErrorAction SilentlyContinue
    }
    # 2) stop any process LISTENING on the target port (old procs hold the port, new one can't bind)
    foreach ($p in Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        [void]$ids.Add($p.OwningProcess)
    }
    # 3) fallback: stop any python running this UI (-m src.cli ui + target port)
    $pat = '--port\s+' + [regex]::Escape([string]$Port) + '(\s|$)'
    foreach ($pr in Get-CimInstance Win32_Process -Filter "Name like '%python%'" -ErrorAction SilentlyContinue) {
        if ($pr.CommandLine -match '-m src\.cli ui' -and $pr.CommandLine -match $pat) {
            [void]$ids.Add([int]$pr.ProcessId)
        }
    }
    foreach ($id in $ids) {
        Get-Process -Id $id -ErrorAction SilentlyContinue | Stop-Process -Force
    }
    Start-Sleep -Milliseconds 500
    $still = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($still) {
        Write-Host "[run.ps1] port $Port still held, aborting. Manually free it then retry." -ForegroundColor Red
        exit 1
    }
    Write-Host "[run.ps1] port $Port cleared, starting fresh ..." -ForegroundColor Green
}

# ---- Locate Python ----
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command py -ErrorAction SilentlyContinue }
if (-not $py) { Write-Error "no python/py found, please install Python 3.9+" }
$pyExe = $py.Source

if (-not (Test-Path (Join-Path $root 'pyproject.toml'))) {
    Write-Host "[run.ps1] run from the repository root" -ForegroundColor Red
    exit 1
}

# Reuse an already running service recorded by this script
if (Test-Path $pidFile) {
    $old = Get-Content $pidFile | ForEach-Object { [int]$_ } | Where-Object { $_ -gt 0 }
    if ($old -and (Get-Process -Id $old -ErrorAction SilentlyContinue)) {
        Write-Host "[run.ps1] service already running (PID=$old), port $Port" -ForegroundColor Green
        if (-not $NoBrowser) { Start-Process $url }
        Write-Host "[run.ps1] reusing existing service; keep this window open (Ctrl+C to stop viewing)." -ForegroundColor DarkGray
        # 别 exit 0: 保持窗口常驻, 让 run.cmd 不会一闪而过。
        # 前台等待, 服务被关闭(或窗口被关)才退出。
        Wait-Process -Id $old
        exit 0
    }
}

Write-Host "[run.ps1] starting: $pyExe -m src.cli ui --host $host_ --port $Port" -ForegroundColor Cyan
Write-Host "[run.ps1] waiting for port $Port ..."

$spArgs = @('-m', 'src.cli', 'ui', '--host', "$host_", '--port', "$Port")

# Launch python in the CURRENT console window (-NoNewWindow) so its logs stream
# live here; Ctrl+C / closing this window stops the service (not a hidden daemon).
$proc = Start-Process -FilePath $pyExe -ArgumentList $spArgs -WorkingDirectory $root -PassThru -NoNewWindow
Set-Content -Path $pidFile -Value $proc.Id

# ---- Wait for the port to become ready ----
$ready = $false
$tries = 0
while ($tries -lt 60 -and -not $ready) {
    $tries = $tries + 1
    if ($proc.HasExited) { Write-Error "service exited early (exit=$($proc.ExitCode)), check the log" }
    try {
        $r = Invoke-WebRequest -Uri "http://${host_}:${Port}/" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -ge 200) { $ready = $true }
    } catch {
        Start-Sleep -Milliseconds 500
    }
}
if (-not $ready) {
    Write-Host "[run.ps1] port $Port not ready, process alive, continuing to open browser" -ForegroundColor DarkYellow
} else {
    Write-Host "[run.ps1] service ready  http://${host_}:${Port}/" -ForegroundColor Green
}

# ---- Open browser ----
if (-not $NoBrowser) { Start-Process $url }

Write-Host ""
Write-Host "  * console:   http://${host_}:${Port}/" -ForegroundColor White
Write-Host "  * workflow:  $url" -ForegroundColor White
Write-Host "  * stop:      .\run.ps1 -Stop   (or end PID=$($proc.Id))" -ForegroundColor DarkGray
Write-Host "  * restart:   .\run.ps1 -Restart   (force-clear residual ui processes + start)" -ForegroundColor DarkGray
Write-Host ""

# Foreground wait; closing the window stops the service
Write-Host "[run.ps1] service running (Ctrl+C or close window to exit) ..."
Wait-Process -Id $proc.Id