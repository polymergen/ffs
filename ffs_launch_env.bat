@echo off
rem Conda environment used by FastFaceSwap launchers
set "FFS_CONDA_ENV=ffs-gold-2024-09"
set "PYTHON="

if exist "D:\Installed\Anaconda\envs\%FFS_CONDA_ENV%\python.exe" set "PYTHON=D:\Installed\Anaconda\envs\%FFS_CONDA_ENV%\python.exe"
if not defined PYTHON if exist "%USERPROFILE%\anaconda3\envs\%FFS_CONDA_ENV%\python.exe" set "PYTHON=%USERPROFILE%\anaconda3\envs\%FFS_CONDA_ENV%\python.exe"
if not defined PYTHON if exist "%USERPROFILE%\miniconda3\envs\%FFS_CONDA_ENV%\python.exe" set "PYTHON=%USERPROFILE%\miniconda3\envs\%FFS_CONDA_ENV%\python.exe"
if not defined PYTHON if exist "C:\ProgramData\anaconda3\envs\%FFS_CONDA_ENV%\python.exe" set "PYTHON=C:\ProgramData\anaconda3\envs\%FFS_CONDA_ENV%\python.exe"

if not defined PYTHON (
    echo Could not find conda env "%FFS_CONDA_ENV%" python.exe
    echo Looked under D:\Installed\Anaconda\envs and common user conda paths.
    echo Edit ffs_launch_env.bat if your install location differs.
    exit /b 1
)

exit /b 0
