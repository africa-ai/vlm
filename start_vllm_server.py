#!/usr/bin/env python
"""
vLLM Server Startup Script for NVIDIA Cosmos-Reason1-7B
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

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
    parser.add_argument("--host", default="localhost", help="Server host")
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
        served_model_name=args.served_model_name
    )
    
    print("Starting vLLM server with configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Server: {config.host}:{config.port}")
    print(f"  Workers: {config.workers}")
    print(f"  Tensor Parallel: {config.tensor_parallel_size}")
    print(f"  GPU Memory: {config.gpu_memory_utilization * 100}%")
    print(f"  Max Model Length: {config.max_model_len}")
    print(f"  Max Sequences: {config.max_num_seqs}")
    
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
