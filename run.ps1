# run.ps1 - one-click start of the DistributedFormer web console + open browser
# Usage:
#   .\run.ps1                  # default port 8011, start service and open browser
#   .\run.ps1 -Port 9000       # custom port
#   .\run.ps1 -NoBrowser       # start service only, no browser
#   .\run.ps1 -Stop            # stop the previously started service process

param(
    [int]$Port = 8011,
    [switch]$NoBrowser,
    [switch]$Stop
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
        exit 0
    }
}

Write-Host "[run.ps1] starting: $pyExe -m src.cli ui --host $host_ --port $Port" -ForegroundColor Cyan
Write-Host "[run.ps1] waiting for port $Port ..."

$spArgs = @('-m', 'src.cli', 'ui', '--host', "$host_", '--port', "$Port")

$proc = Start-Process -FilePath $pyExe -ArgumentList $spArgs -WorkingDirectory $root -PassThru -WindowStyle Hidden
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
Write-Host ""

# Foreground wait; closing the window stops the service
Write-Host "[run.ps1] service running (Ctrl+C or close window to exit) ..."
Wait-Process -Id $proc.Id