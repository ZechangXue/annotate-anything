@echo off
REM Change to the directory of this script so relative paths work
cd /d "%~dp0"

REM Run the interactive SAM2 annotator
python sam_interactive_segmentation.py

REM If Python exits with an error, keep the window open
if errorlevel 1 (
    echo.
    echo Python exited with an error. Press any key to close...
    pause >nul
)

