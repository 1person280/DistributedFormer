@echo off
rem run.cmd - one-click launcher for the DistributedFormer web console
rem This file MUST stay pure ASCII (English only): cmd.exe parses .cmd files
rem with the OEM code page, and any non-ASCII byte (e.g. UTF-8 Chinese) would
rem be mis-decoded and corrupt the command lines, making the window crash or
rem close instantly. Keep all Chinese text in run.ps1 / the web UI instead.
setlocal EnableDelayedExpansion
title DistributedFormer Console
cd /d "%~dp0"

echo [run.cmd] starting DistributedFormer...
echo   args: %*
echo.
echo   Live logs below. Close this window (or Ctrl+C) to stop the service.
echo.

rem Run the PowerShell script in the foreground of THIS window. The service
rem runs attached to this console so logs stream live and closing the window
rem / Ctrl+C stops the service.
rem -Restart: force-clear residual ui processes on port 8011 first, so a
rem            stale process can never hold the port and kill the new one.

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" -Restart %*

endlocal