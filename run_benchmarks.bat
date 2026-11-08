@echo off
title AAC TTS Benchmark Suite
cd /d "%~dp0"
echo Running automated TTS benchmarks...

if exist .venv\Scripts\python.exe (
    .\.venv\Scripts\python.exe benchmark.py %*
) else (
    python benchmark.py %*
)
pause
