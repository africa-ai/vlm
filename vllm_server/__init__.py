"""
vLLM Server Package for NVIDIA Cosmos-Reason1-7B
"""

from .config import VLLMServerConfig, create_vllm_env_file
from .client import VLLMClient, SyncVLLMClient, test_vllm_connection

try:
    from .server import VLLMServer
except ImportError:
    # vLLM dependencies not available
    VLLMServer = None

__all__ = [
    "VLLMServerConfig",
    "VLLMClient", 
    "SyncVLLMClient",
    "VLLMServer",
    "create_vllm_env_file",
    "test_vllm_connection"
]

__version__ = "1.0.0"
