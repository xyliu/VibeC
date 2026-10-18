@echo off
cd /d "%~dp0"

echo ========================================================
echo     VibeC - AMD NPU / DirectML Accelerated Assistant
echo ========================================================
echo [*] Hardware Backend: AMD NPU / DirectML + Zen4 CPU
echo [*] Starting demo_qwen3_asr.py with --amd-npu ...
echo [*] Press F8 to record / stop.
echo.

"%~dp0.venv\Scripts\python.exe" -u "%~dp0demo_qwen3_asr.py" --amd-npu

pause
