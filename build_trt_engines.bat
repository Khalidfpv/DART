@echo off
REM Build TensorRT FP16 engines for DART (one-time, ~5 minutes)
REM This must be done once per GPU model / TensorRT version

set PYTHONIOENCODING=utf-8
call conda activate dartsam3
cd /d "%~dp0"

echo.
echo =============================================
echo   Building DART TensorRT Engines
echo   This is a one-time process (~5 min)
echo =============================================
echo.

echo [1/3] Exporting HuggingFace backbone to TRT FP16...
python scripts/export_hf_backbone.py --imgsz 1008
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Backbone export failed.
    pause
    exit /b 1
)

echo.
echo [2/3] Exporting encoder-decoder to ONNX...
python -m sam3.trt.export_enc_dec --checkpoint sam3.pt --output enc_dec.onnx --max-classes 4 --imgsz 1008
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Encoder-decoder ONNX export failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Building encoder-decoder TRT engine...
python -m sam3.trt.build_engine --onnx enc_dec.onnx --output enc_dec_fp16.engine --fp16 --mixed-precision none
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Encoder-decoder engine build failed.
    pause
    exit /b 1
)

echo.
echo =============================================
echo   TRT engines built successfully!
echo   - hf_backbone_fp16.engine
echo   - enc_dec_fp16.engine
echo   You can now use run_webcam.bat or run_video.bat
echo =============================================
pause
