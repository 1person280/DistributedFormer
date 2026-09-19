@echo off
rem run.cmd - double-click launcher for the DistributedFormer web console
rem Opens a terminal window, launches run.ps1 (bypassing execution policy),
rem and keeps the window open until the service stops.
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
pause