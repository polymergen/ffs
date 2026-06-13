@echo off
setlocal
cd /d "%~dp0"
call "%~dp0ffs_launch_env.bat"
if errorlevel 1 (
    pause
    exit /b 1
)
"%PYTHON%" "%~dp0main.py" %*
