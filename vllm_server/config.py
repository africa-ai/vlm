"""
vLLM Server Configuration for NVIDIA Cosmos-Reason1-7B
Handles model serving with optimized inference for visual language model processing
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class VLLMServerConfig:
    """Configuration for vLLM server hosting Cosmos-Reason1-7B"""
    
    # Model configuration
    model_name: str = "nvidia/Cosmos-Reason1-7B"
    model_path: Optional[str] = None  # Local model path if available
    
    # Server configuration
    host: str = "localhost"
    port: int = 8000
    workers: int = 1
    
    # vLLM configuration
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    max_num_seqs: int = 4
    
    # Vision model specific
    image_input_type: str = "pixel_values"
    image_token_id: int = 0
    image_input_shape: str = "1,3,224,224"
    
    # Performance tuning
    disable_log_stats: bool = False
    disable_log_requests: bool = False
    max_log_len: int = 200
    enable_cudagraph: bool = True  # Enable CUDA graph for performance (disable for multi-GPU stability)
    # NCCL / multi-GPU tuning
    nccl_p2p_disable: bool = False  # Disable NCCL P2P (useful in virtualized/container envs)
    nccl_ib_disable: bool = False   # Disable NCCL IB (when IB not available)
    nccl_socket_ifname: Optional[str] = None  # Bind NCCL sockets to specific interface, e.g., eth0
    nccl_shm_disable: bool = False  # Disable NCCL SHM if shared memory is constrained
    cuda_visible_devices: Optional[str] = None  # e.g., "0,1"
    pytorch_cuda_alloc_conf: Optional[str] = None  # e.g., "expandable_segments:True"

    # API configuration
    api_key: Optional[str] = None
    served_model_name: Optional[str] = "cosmos-reason-vlm"

    # Resource limits
    max_concurrent_requests: int = 10
    request_timeout: float = 300.0
    
    @classmethod
    def from_env(cls) -> "VLLMServerConfig":
        """Load configuration from environment variables"""
        return cls(
            model_name=os.getenv("VLLM_MODEL_NAME", "nvidia/Cosmos-Reason1-7B"),
            model_path=os.getenv("VLLM_MODEL_PATH"),
            host=os.getenv("VLLM_HOST", "localhost"),
            port=int(os.getenv("VLLM_PORT", "8000")),
            workers=int(os.getenv("VLLM_WORKERS", "1")),
            tensor_parallel_size=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
            gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85")),
            max_model_len=int(os.getenv("VLLM_MAX_MODEL_LEN", "4096")),
            max_num_seqs=int(os.getenv("VLLM_MAX_NUM_SEQS", "4")),
            api_key=os.getenv("VLLM_API_KEY"),
            served_model_name=os.getenv("VLLM_SERVED_MODEL_NAME"),
            max_concurrent_requests=int(os.getenv("VLLM_MAX_CONCURRENT_REQUESTS", "10")),
            request_timeout=float(os.getenv("VLLM_REQUEST_TIMEOUT", "300.0")),
            enable_cudagraph=os.getenv("VLLM_ENABLE_CUDAGRAPH", "false").lower() == "true",
            nccl_p2p_disable=os.getenv("VLLM_NCCL_P2P_DISABLE", "false").lower() == "true",
            nccl_ib_disable=os.getenv("VLLM_NCCL_IB_DISABLE", "false").lower() == "true",
            nccl_socket_ifname=os.getenv("VLLM_NCCL_SOCKET_IFNAME"),
            nccl_shm_disable=os.getenv("VLLM_NCCL_SHM_DISABLE", "false").lower() == "true",
            cuda_visible_devices=os.getenv("VLLM_CUDA_VISIBLE_DEVICES"),
            pytorch_cuda_alloc_conf=os.getenv("VLLM_PYTORCH_CUDA_ALLOC_CONF"),
        )
    
    def to_vllm_args(self) -> Dict[str, Any]:
        """Convert to vLLM engine arguments"""
        args = {
            "model": self.model_path or self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "disable_log_stats": self.disable_log_stats,
            "disable_log_requests": self.disable_log_requests,
            "enable_cudagraph": self.enable_cudagraph
        }
        
        # Add vision-specific parameters
        if "vision" in self.model_name.lower() or "vlm" in self.model_name.lower():
            args.update({
                "image_input_type": self.image_input_type,
                "image_token_id": self.image_token_id,
                "image_input_shape": self.image_input_shape
            })
        
        return args
    
    def to_server_args(self) -> Dict[str, Any]:
        """Convert to vLLM OpenAI API server arguments"""
        return {
            "host": self.host,
            "port": self.port,
            "served_model_name": self.served_model_name or self.model_name,
            "api_key": self.api_key,
            "max_log_len": self.max_log_len
        }


def create_vllm_env_file(config_path: str = ".env.vllm"):
    """Create environment file with vLLM configuration"""
    env_content = """# vLLM Server Configuration for Cosmos-Reason1-7B

# Model Configuration
VLLM_MODEL_NAME=nvidia/Cosmos-Reason1-7B
# VLLM_MODEL_PATH=/path/to/local/model  # Optional: local model path

# Server Configuration
VLLM_HOST=localhost
VLLM_PORT=8000
VLLM_WORKERS=1

# vLLM Engine Configuration
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_MAX_MODEL_LEN=4096
VLLM_MAX_NUM_SEQS=4

# API Configuration
# VLLM_API_KEY=your-api-key-here  # Optional: API authentication
VLLM_SERVED_MODEL_NAME=cosmos-reason-vlm

# Performance Configuration
VLLM_MAX_CONCURRENT_REQUESTS=10
VLLM_REQUEST_TIMEOUT=300.0

# Logging
VLLM_DISABLE_LOG_STATS=false
VLLM_DISABLE_LOG_REQUESTS=false
"""
    
    config_file = Path(config_path)
    if not config_file.exists():
        config_file.write_text(env_content)
        return f"Created vLLM environment configuration at {config_path}"
    else:
        return f"Environment configuration already exists at {config_path}"


if __name__ == "__main__":
    # Create example configuration
    config = VLLMServerConfig.from_env()
    print("vLLM Server Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Server: {config.host}:{config.port}")
    print(f"  GPU Memory: {config.gpu_memory_utilization * 100}%")
    print(f"  Max Sequences: {config.max_num_seqs}")
    
    # Create environment file
    result = create_vllm_env_file()
    print(f"\n{result}")
