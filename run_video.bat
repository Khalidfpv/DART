@echo off
REM DART Video Detection
REM Usage: Drag and drop a video file onto this .bat, or run from command line:
REM   run_video.bat path\to\video.mp4

set PYTHONIOENCODING=utf-8

call conda activate dartsam3

if "%~1"=="" (
    echo Usage: run_video.bat path\to\video.mp4
    echo   Or drag and drop a video file onto this .bat
    pause
    exit /b 1
)

echo.
echo =============================================
echo   DART Video Detection
echo   Input: %~1
echo   Press 'q' to quit, 'p' to pause
echo =============================================
echo.

REM If TRT engines exist, use them for max speed
if exist "hf_backbone_fp16.engine" (
    if exist "enc_dec_fp16.engine" (
        echo Using TensorRT acceleration...
        python live_detect.py ^
            --source "%~1" ^
            --classes person chair laptop cup bottle car bicycle dog cat ^
            --trt hf_backbone_fp16.engine ^
            --trt-enc-dec enc_dec_fp16.engine ^
            --confidence 0.3 ^
            --track ^
            --save "%~dpn1_detected.mp4"
        goto :end
    )
)

REM Fallback: torch.compile mode
echo TRT engines not found. Using torch.compile mode...
python live_detect.py ^
    --source "%~1" ^
    --classes person chair laptop cup bottle car bicycle dog cat ^
    --compile default ^
    --confidence 0.3 ^
    --track ^
    --save "%~dpn1_detected.mp4"

:end
pause
