@echo off
setlocal
cd /d "%~dp0"
set "MODEL_BASE=https://github.com/RichardErkhov/FastFaceSwap/releases/download/model"
for %%F in (inswapper_128.onnx GFPGANv1.4.pth complex_256_v7_stage3_12999.h5 GFPGANv1.4.onnx) do (
    if exist "%%F" (
        echo Already present: %%F
    ) else (
        echo Downloading %%F ...
        curl --location --remote-header-name --remote-name "%MODEL_BASE%/%%F"
        if errorlevel 1 exit /b 1
    )
)
exit /b 0
