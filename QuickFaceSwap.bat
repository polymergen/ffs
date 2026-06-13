@echo off
setlocal
cd /d "%~dp0"
call "%~dp0ffs_launch_env.bat"
if errorlevel 1 (
    pause
    exit /b 1
)

"%PYTHON%" "%~dp0quick_image_faceswap.py"
set "ERR=%ERRORLEVEL%"
if not "%ERR%"=="0" pause
exit /b %ERR%
