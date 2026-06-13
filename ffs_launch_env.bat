@echo off
rem Conda environment used by FastFaceSwap launchers
set "FFS_CONDA_ENV=ffs-gold-2024-09"
set "SCRIPT_DIR=%~dp0"
set "PYTHON="

if exist "%SCRIPT_DIR%venv\Scripts\python.exe" set "PYTHON=%SCRIPT_DIR%venv\Scripts\python.exe"
if not defined PYTHON if exist "D:\Installed\Anaconda\envs\%FFS_CONDA_ENV%\python.exe" set "PYTHON=D:\Installed\Anaconda\envs\%FFS_CONDA_ENV%\python.exe"
if not defined PYTHON if exist "%USERPROFILE%\anaconda3\envs\%FFS_CONDA_ENV%\python.exe" set "PYTHON=%USERPROFILE%\anaconda3\envs\%FFS_CONDA_ENV%\python.exe"
if not defined PYTHON if exist "%USERPROFILE%\miniconda3\envs\%FFS_CONDA_ENV%\python.exe" set "PYTHON=%USERPROFILE%\miniconda3\envs\%FFS_CONDA_ENV%\python.exe"
if not defined PYTHON if exist "C:\ProgramData\anaconda3\envs\%FFS_CONDA_ENV%\python.exe" set "PYTHON=C:\ProgramData\anaconda3\envs\%FFS_CONDA_ENV%\python.exe"

if not defined PYTHON (
    echo Could not find conda env "%FFS_CONDA_ENV%" python.exe
    echo Looked for venv\Scripts\python.exe, D:\Installed\Anaconda\envs, and common user conda paths.
    echo Edit ffs_launch_env.bat if your install location differs, or set FFS_PYTHON.
    exit /b 1
)

exit /b 0
