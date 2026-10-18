@echo off
cd /d "%~dp0"

echo ========================================================
echo         VibeC - Quick Speech-to-Text Assistant
echo ========================================================
echo [*] Starting demo_qwen3_asr.py ...
echo [*] Default Hotkey: F8 or Win+Shift+H
echo [*] The floating bar will appear at the TOP-CENTER of your active screen.
echo.

"%~dp0.venv\Scripts\python.exe" -u "%~dp0demo_qwen3_asr.py"

pause
