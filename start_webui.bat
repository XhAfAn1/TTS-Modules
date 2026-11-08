@echo off
title AAC TTS Comparison WebUI
cd /d "%~dp0"
echo Starting AAC TTS Comparison WebUI on http://127.0.0.1:7860 ...

if exist .venv\Scripts\python.exe (
    .\.venv\Scripts\python.exe webui\server.py
) else (
    python webui\server.py
)
pause
