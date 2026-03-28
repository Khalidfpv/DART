@echo off
REM DART Live Webcam Detection
REM Uses full ViT-H backbone with TensorRT acceleration
REM Edit the --classes line to change what objects to detect

set PYTHONIOENCODING=utf-8

call conda activate dartsam3

echo.
echo =============================================
echo   DART Live Webcam Detection
echo   Press 'q' to quit, 'p' to pause
echo =============================================
echo.

REM If TRT engines exist, use them for max speed
if exist "hf_backbone_fp16.engine" (
    if exist "enc_dec_fp16.engine" (
        echo Using TensorRT acceleration...
        python live_detect.py ^
            --source 0 ^
            --classes person chair laptop cup bottle phone ^
            --trt hf_backbone_fp16.engine ^
            --trt-enc-dec enc_dec_fp16.engine ^
            --confidence 0.3
        goto :end
    )
)

REM Fallback: torch.compile mode (slower but no TRT build needed)
echo TRT engines not found. Using torch.compile mode...
echo To build TRT engines, run: build_trt_engines.bat
python live_detect.py ^
    --source 0 ^
    --classes person chair laptop cup bottle phone ^
    --compile default ^
    --confidence 0.3

:end
pause
