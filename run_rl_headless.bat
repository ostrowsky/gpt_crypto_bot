@echo off
cd /d "%~dp0"
start "GPT Crypto RL" powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0headless_loop.ps1"
