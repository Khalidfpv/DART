@echo off
REM DART Setup & Run — Execute this once Meta approves your SAM3 access

set PYTHONIOENCODING=utf-8
call conda activate dartsam3
cd /d "%~dp0"

echo.
echo =============================================
echo   DART Setup - Checking SAM3 Access...
echo =============================================
echo.

REM Check if access is granted
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='facebook/sam3', filename='config.json'); print('ACCESS GRANTED!')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: SAM3 access still pending.
    echo Check: https://huggingface.co/facebook/sam3
    echo Meta needs to approve your request.
    pause
    exit /b 1
)

echo.
echo [1/3] Building TRT backbone engine...
python scripts/export_hf_backbone.py --imgsz 1008
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Backbone export failed.
    pause
    exit /b 1
)

echo.
echo [2/3] Building TRT encoder-decoder engine...
python -m sam3.trt.export_enc_dec --checkpoint sam3.pt --output enc_dec.onnx --max-classes 4 --imgsz 1008
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Enc-dec export failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Building TRT encoder-decoder engine...
python -m sam3.trt.build_engine --onnx enc_dec.onnx --output enc_dec_fp16.engine --fp16 --mixed-precision none
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Enc-dec build failed.
    pause
    exit /b 1
)

echo.
echo [4/4] Launching live webcam detection!
echo Press 'q' to quit, 'p' to pause, 's' for screenshot
echo.
python live_detect.py --source 0 --classes person chair laptop cup bottle phone --trt hf_backbone_fp16.engine --trt-enc-dec enc_dec_fp16.engine --confidence 0.3

pause
