#!/usr/bin/env python
"""
vLLM Server Startup Script for NVIDIA Cosmos-Reason1-7B
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set CUDA library path to help vLLM find CUDA runtime
# Detect virtual environment and use appropriate paths
venv_path = os.environ.get("VIRTUAL_ENV")
if venv_path:
    base_path = f"{venv_path}/lib/python3.11/site-packages/nvidia"
else:
    base_path = "/usr/local/lib/python3.11/site-packages/nvidia"

cuda_lib_paths = [
    f"{base_path}/cuda_runtime/lib",
    f"{base_path}/cudnn/lib", 
    f"{base_path}/cublas/lib",
    f"{base_path}/cufft/lib",
    f"{base_path}/curand/lib",
    f"{base_path}/cusolver/lib",
    f"{base_path}/cusparse/lib",
    f"{base_path}/nccl/lib",
    f"{base_path}/nvtx/lib"
]

# Add existing LD_LIBRARY_PATH
existing_path = os.environ.get("LD_LIBRARY_PATH", "")
all_paths = [p for p in cuda_lib_paths if Path(p).exists()]
if existing_path:
    all_paths.append(existing_path)

if all_paths:
    os.environ["LD_LIBRARY_PATH"] = ":".join(all_paths)
    print(f"Set LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")

try:
    from vllm_server.config import VLLMServerConfig, create_vllm_env_file
    from vllm_server.server import VLLMServer
except ImportError as e:
    print(f"Error importing vLLM dependencies: {e}")
    print("\nPlease install vLLM dependencies:")
    print("pip install -r requirements_vllm.txt")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Start vLLM server for Cosmos-Reason1-7B")
    
    # Server configuration
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    # Model configuration
    parser.add_argument("--model", default="nvidia/Cosmos-Reason1-7B", 
                       help="Model name or path")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                       help="Tensor parallel size for multi-GPU")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                       help="GPU memory utilization fraction")
    parser.add_argument("--max-model-len", type=int, default=4096,
                       help="Maximum model sequence length")
    parser.add_argument("--max-num-seqs", type=int, default=4,
                       help="Maximum number of sequences to process in parallel")
    
    # Performance tuning
    parser.add_argument("--disable-cudagraph", action="store_true",
                       help="Disable CUDA graph for multi-GPU stability")
    parser.add_argument("--nccl-p2p-disable", action="store_true",
                       help="Disable NCCL P2P (set NCCL_P2P_DISABLE=1)")
    parser.add_argument("--nccl-ib-disable", action="store_true",
                       help="Disable NCCL IB (set NCCL_IB_DISABLE=1)")
    parser.add_argument("--nccl-socket-ifname", type=str, default=None,
                       help="Set NCCL_SOCKET_IFNAME (e.g., eth0)")
    parser.add_argument("--nccl-shm-disable", action="store_true",
                       help="Disable NCCL shared memory (NCCL_SHM_DISABLE=1)")
    parser.add_argument("--cuda-visible-devices", type=str, default=None,
                       help="Set CUDA_VISIBLE_DEVICES (e.g., 0,1)")
    parser.add_argument("--pytorch-cuda-alloc-conf", type=str, default=None,
                       help="Set PYTORCH_CUDA_ALLOC_CONF (e.g., expandable_segments:True)")
    
    # API configuration
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--served-model-name", default="cosmos-reason-vlm",
                       help="Model name to serve in API")
    
    # Environment setup
    parser.add_argument("--create-env", action="store_true",
                       help="Create environment configuration file")
    parser.add_argument("--env-file", default=".env.vllm",
                       help="Environment configuration file path")
    
    args = parser.parse_args()
    
    # Create environment file if requested
    if args.create_env:
        result = create_vllm_env_file(args.env_file)
        print(result)
        return
    
    # Create server configuration
    config = VLLMServerConfig(
        model_name=args.model,
        host=args.host,
        port=args.port,
        workers=args.workers,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        api_key=args.api_key,
        served_model_name=args.served_model_name,
        enable_cudagraph=not args.disable_cudagraph,
        nccl_p2p_disable=args.nccl_p2p_disable,
    nccl_ib_disable=args.nccl_ib_disable,
    nccl_socket_ifname=args.nccl_socket_ifname,
    nccl_shm_disable=args.nccl_shm_disable,
    cuda_visible_devices=args.cuda_visible_devices,
    pytorch_cuda_alloc_conf=args.pytorch_cuda_alloc_conf,
    )
    
    print("Starting vLLM server with configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Server: {config.host}:{config.port}")
    print(f"  Workers: {config.workers}")
    print(f"  Tensor Parallel: {config.tensor_parallel_size}")
    print(f"  GPU Memory: {config.gpu_memory_utilization * 100}%")
    print(f"  Max Model Length: {config.max_model_len}")
    print(f"  Max Sequences: {config.max_num_seqs}")
    print(f"  NCCL_P2P_DISABLE: {config.nccl_p2p_disable}")
    print(f"  NCCL_IB_DISABLE: {config.nccl_ib_disable}")
    print(f"  NCCL_SOCKET_IFNAME: {config.nccl_socket_ifname}")
    print(f"  NCCL_SHM_DISABLE: {config.nccl_shm_disable}")
    print(f"  CUDA_VISIBLE_DEVICES: {config.cuda_visible_devices}")
    print(f"  PYTORCH_CUDA_ALLOC_CONF: {config.pytorch_cuda_alloc_conf}")
    print(f"  CUDA Graphs Enabled: {config.enable_cudagraph and config.tensor_parallel_size == 1}")
    
    # Create and run server
    try:
        server = VLLMServer(config)
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
