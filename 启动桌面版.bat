@echo off
rem ============================================================
rem  Precision Part Surface Defect Detection System 1.0
rem  Desktop (PyQt5) launcher - windowed mode, no console.
rem  Keep this file next to main.py and the bundled "python".
rem ============================================================
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONDONTWRITEBYTECODE=1"

if not exist "%~dp0python\pythonw.exe" goto no_python

start "" "%~dp0python\pythonw.exe" "%~dp0main.py" %*
exit /b 0

:no_python
echo [ERROR] Bundled Python not found: "%~dp0python\pythonw.exe"
echo Please keep the "python" folder next to this script.
pause
exit /b 1
