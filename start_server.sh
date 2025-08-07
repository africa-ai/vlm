#!/bin/bash
# vLLM Server Startup Script with proper CUDA library path
# This script sets the necessary environment variables and starts the vLLM server

# Set CUDA library path for vLLM
export LD_LIBRARY_PATH="/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"

# Start the vLLM server
python start_vllm_server.py "$@"


#chmod +x start_server.sh