#!/bin/bash
# vLLM Server Startup Script with proper CUDA library path
# This script sets the necessary environment variables and starts the vLLM server

# Detect virtual environment and set CUDA library path
if [ -n "$VIRTUAL_ENV" ]; then
    export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
fi

# Start the vLLM server
python start_vllm_server.py "$@"


#chmod +x start_server.sh