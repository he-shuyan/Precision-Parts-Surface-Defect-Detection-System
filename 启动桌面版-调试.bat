@echo off
rem ============================================================
rem  Desktop version - DEBUG launcher (keeps the console window
rem  so that Python tracebacks are visible). Use this one when
rem  the silent launcher seems to do nothing.
rem ============================================================
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not exist "%~dp0python\python.exe" goto no_python

"%~dp0python\python.exe" "%~dp0main.py" %*
set "RC=%ERRORLEVEL%"
echo.
echo Exit code: %RC%
pause
exit /b %RC%

:no_python
echo [ERROR] Bundled Python not found: "%~dp0python\python.exe"
echo Please keep the "python" folder next to this script.
pause
exit /b 1
