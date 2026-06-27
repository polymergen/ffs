@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "FFS_CONDA_ENV=PyEditorFromFFS"
set "CONDA_EXE="
set "CONDA_BASE="

for %%C in (
    "D:\Installed\Anaconda"
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\mambaforge"
    "C:\ProgramData\anaconda3"
) do (
    if not defined CONDA_EXE if exist %%~C\Scripts\conda.exe (
        set "CONDA_EXE=%%~C\Scripts\conda.exe"
        set "CONDA_BASE=%%~C"
    )
)

if not defined CONDA_EXE (
    echo Could not find conda. Install Miniconda/Anaconda first:
    echo   https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

echo Using conda: %CONDA_EXE%

"%CONDA_EXE%" env list | findstr /I /C:" %FFS_CONDA_ENV% " >nul 2>&1
if errorlevel 1 (
    echo Creating conda env "%FFS_CONDA_ENV%" ...
    "%CONDA_EXE%" env create -f environment.yml
    if errorlevel 1 (
        echo conda env create failed. Trying: conda create -n %FFS_CONDA_ENV% python=3.9 -y
        "%CONDA_EXE%" create -n %FFS_CONDA_ENV% python=3.9 -y
        if errorlevel 1 exit /b 1
    )
) else (
    echo Conda env "%FFS_CONDA_ENV%" already exists.
)

call "%CONDA_BASE%\Scripts\activate.bat" %FFS_CONDA_ENV%
if errorlevel 1 (
    echo Could not activate %FFS_CONDA_ENV%
    exit /b 1
)

python -m pip install --upgrade pip wheel

echo Installing PyTorch (CUDA 11.8 wheels)...
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

echo Installing ONNX Runtime GPU + TensorFlow...
pip install onnxruntime-gpu==1.17.0
pip install tensorflow-gpu==2.10.1
pip install protobuf==3.20.2

echo Installing application dependencies...
pip install -r requirements-pip.txt
pip install python-magic-bin

pip uninstall opencv-python opencv-headless-python opencv-contrib-python -q -y 2>nul
pip install opencv-python==4.9.0.80

call "%~dp0download_models.cmd"

echo.
echo Done. Conda env: %FFS_CONDA_ENV%
echo Run: fastfaceswap.bat  or  QuickFaceSwap.bat
exit /b 0
