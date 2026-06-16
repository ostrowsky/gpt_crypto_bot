@echo off
cd /d "%~dp0"
if exist "%~dp0.runtime\rl_worker.stop" del /f /q "%~dp0.runtime\rl_worker.stop"
wscript.exe //nologo "%~dp0run_rl_headless.vbs"
