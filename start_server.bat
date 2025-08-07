@echo off
REM Windows batch script to start vLLM server for Cosmos-Reason1-7B

echo Starting vLLM server for NVIDIA Cosmos-Reason1-7B...

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if required packages are installed
python -c "import vllm" >nul 2>&1
if errorlevel 1 (
    echo Error: vLLM is not installed
    echo Please install requirements: pip install -r requirements_vllm.txt
    pause
    exit /b 1
)

REM Start the server
python start_vllm_server.py %*

pause
