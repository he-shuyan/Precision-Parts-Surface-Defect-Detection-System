@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem ===== 1) python bundled inside this project (full deep-learning env) =====
if exist "%~dp0python\python.exe" goto use_local

rem ===== 2) system python =====
python --version >nul 2>nul
if not errorlevel 1 goto use_sys

rem ===== 3) fallback runtime already present on this PC =====
if exist "C:\Program Files\MySQL\MySQL Workbench 8.0\python.exe" goto use_mysql

echo [ERROR] No Python runtime found.
echo Please install Python 3.10/3.11 64-bit, or keep the bundled "python" folder.
pause
exit /b 1

:use_sys
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
python "%~dp0app.py" %*
goto finished

:use_local
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
"%~dp0python\python.exe" "%~dp0app.py" %*
goto finished

:use_mysql
set "PYTHONHOME=C:\Program Files\MySQL\MySQL Workbench 8.0\python"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
"C:\Program Files\MySQL\MySQL Workbench 8.0\python.exe" "%~dp0app.py" %*
goto finished

:finished
pause
