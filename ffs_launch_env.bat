@echo off
rem Activate conda env used by FastFaceSwap launchers.
set "FFS_CONDA_ENV=PyEditorFromFFS"
set "SCRIPT_DIR=%~dp0"
set "PYTHON="

if defined FFS_PYTHON if exist "%FFS_PYTHON%" set "PYTHON=%FFS_PYTHON%"

if not defined PYTHON (
    for %%C in (
        "D:\Installed\Anaconda"
        "%USERPROFILE%\anaconda3"
        "%USERPROFILE%\miniconda3"
        "%USERPROFILE%\mambaforge"
        "C:\ProgramData\anaconda3"
    ) do (
        if not defined PYTHON if exist %%~C\Scripts\activate.bat if exist %%~C\envs\%FFS_CONDA_ENV%\python.exe (
            call %%~C\Scripts\activate.bat "%FFS_CONDA_ENV%"
            set "PYTHON=%%~C\envs\%FFS_CONDA_ENV%\python.exe"
        )
    )
)

if not defined PYTHON (
    for %%C in (
        "D:\Installed\Anaconda"
        "%USERPROFILE%\anaconda3"
        "%USERPROFILE%\miniconda3"
        "%USERPROFILE%\mambaforge"
        "C:\ProgramData\anaconda3"
    ) do (
        if not defined PYTHON if exist %%~C\envs\%FFS_CONDA_ENV%\Scripts\activate.bat (
            call %%~C\envs\%FFS_CONDA_ENV%\Scripts\activate.bat
            set "PYTHON=%%~C\envs\%FFS_CONDA_ENV%\python.exe"
        )
    )
)

if not defined PYTHON if exist "D:\Installed\Anaconda\envs\%FFS_CONDA_ENV%\python.exe" set "PYTHON=D:\Installed\Anaconda\envs\%FFS_CONDA_ENV%\python.exe"
if not defined PYTHON if exist "%USERPROFILE%\anaconda3\envs\%FFS_CONDA_ENV%\python.exe" set "PYTHON=%USERPROFILE%\anaconda3\envs\%FFS_CONDA_ENV%\python.exe"
if not defined PYTHON if exist "%USERPROFILE%\miniconda3\envs\%FFS_CONDA_ENV%\python.exe" set "PYTHON=%USERPROFILE%\miniconda3\envs\%FFS_CONDA_ENV%\python.exe"
if not defined PYTHON if exist "C:\ProgramData\anaconda3\envs\%FFS_CONDA_ENV%\python.exe" set "PYTHON=C:\ProgramData\anaconda3\envs\%FFS_CONDA_ENV%\python.exe"

if not defined PYTHON (
    echo Could not find or activate conda env "%FFS_CONDA_ENV%".
    echo Run install_conda_windows.cmd once to create it.
    echo Or set FFS_PYTHON to your python.exe path.
    exit /b 1
)

exit /b 0
