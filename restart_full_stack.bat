@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
cd /d "%ROOT%"
set "PS=powershell.exe"
set "PS_ARGS=-NoProfile -ExecutionPolicy Bypass"
set "KEEP_OPEN=1"
if /I "%~1"=="--no-pause" set "KEEP_OPEN=0"

echo ============================================================
echo  GPT Crypto Bot - Full Stack Restart
echo  Workspace: %ROOT%
echo  Build: menu_build_v2 / 2026-04-13
echo ============================================================
echo.

echo [0/7] Running tests...
%PS% %PS_ARGS% -File "%ROOT%run_tests.ps1" || goto :fail

echo [1/7] Restarting trading bot...
%PS% %PS_ARGS% -File "%ROOT%start_bot_bg.ps1" -ForceRestart || goto :fail

echo [2/7] Restarting RL worker and training pipeline...
%PS% %PS_ARGS% -File "%ROOT%start_rl_worker_bg.ps1" -ForceRestart || goto :fail

echo [3/7] Restarting market agent...
%PS% %PS_ARGS% -File "%ROOT%start_market_agent_bg.ps1" -ForceRestart || goto :fail

echo [4/7] Waiting for heartbeats...
timeout /t 5 /nobreak >nul

echo [5/7] Bot status:
%PS% %PS_ARGS% -File "%ROOT%bot_status.ps1"
%PS% %PS_ARGS% -File "%ROOT%bot_status.ps1" -FailIfNotRunning || goto :fail

echo [6/7] RL worker status:
%PS% %PS_ARGS% -File "%ROOT%rl_worker_status.ps1"
%PS% %PS_ARGS% -File "%ROOT%rl_worker_status.ps1" -FailIfNotRunning || goto :fail

echo [7/7] Market agent status:
%PS% %PS_ARGS% -File "%ROOT%market_agent_status.ps1"
%PS% %PS_ARGS% -File "%ROOT%market_agent_status.ps1" -FailIfNotRunning || goto :fail

echo.
echo Logs:
echo   Bot:          %ROOT%bot_stderr.log
echo   RL worker:    %ROOT%.runtime\rl_worker_wrapper_stderr.log
echo   Market agent: %ROOT%agent_stderr.log
echo.
echo Full stack restart finished.
if "%KEEP_OPEN%"=="1" pause
exit /b 0

:fail
echo.
echo Full stack restart failed with exit code %errorlevel%.
echo.
echo --- Bot stderr tail ---
%PS% %PS_ARGS% -Command "if (Test-Path '%ROOT%bot_stderr.log') { Get-Content '%ROOT%bot_stderr.log' -Tail 40 } else { 'bot_stderr.log not found' }"
echo.
echo --- RL worker stderr tail ---
%PS% %PS_ARGS% -Command "if (Test-Path '%ROOT%.runtime\rl_worker_wrapper_stderr.log') { Get-Content '%ROOT%.runtime\rl_worker_wrapper_stderr.log' -Tail 40 } else { 'rl_worker_wrapper_stderr.log not found' }"
echo.
echo --- Market agent stderr tail ---
%PS% %PS_ARGS% -Command "if (Test-Path '%ROOT%agent_stderr.log') { Get-Content '%ROOT%agent_stderr.log' -Tail 40 } else { 'agent_stderr.log not found' }"
echo.
echo Logs:
echo   Bot:          %ROOT%bot_stderr.log
echo   RL worker:    %ROOT%.runtime\rl_worker_wrapper_stderr.log
echo   Market agent: %ROOT%agent_stderr.log
if "%KEEP_OPEN%"=="1" pause
exit /b %errorlevel%
