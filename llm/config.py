"""
Configuration settings for VLM processing
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import torch

@dataclass
class VLMConfig:
    """Configuration class for Visual Language Model settings"""
    
    # Model settings
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_cache_dir: str = "./models"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype: torch.dtype = torch.float16
    trust_remote_code: bool = True
    
    # Generation parameters
    max_new_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = True
    
    # Processing settings
    batch_size: int = 4
    max_image_size: int = 1024
    image_format: str = "RGB"
    
    # GPU settings
    gpu_memory_fraction: float = 0.8
    use_flash_attention: bool = True
    
    # Output settings
    output_dir: str = "./output"
    save_json: bool = True
    save_csv: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "vlm_processing.log"
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Validate device
        if self.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            self.device = "cpu"
            
        # Adjust batch size for CPU
        if self.device == "cpu" and self.batch_size > 1:
            print("Warning: Reducing batch size to 1 for CPU processing")
            self.batch_size = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VLMConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get model loading arguments"""
        return {
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": self.trust_remote_code,
            "device_map": "auto" if self.device == "cuda" else None,
            "cache_dir": self.model_cache_dir,
        }
    
    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Get text generation arguments"""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
        }

# Default configuration
DEFAULT_CONFIG = VLMConfig()

# Environment-based configuration overrides
def load_config_from_env() -> VLMConfig:
    """Load configuration with environment variable overrides"""
    config = VLMConfig()
    
    # Model settings
    if "VLM_MODEL_NAME" in os.environ:
        config.model_name = os.environ["VLM_MODEL_NAME"]
    if "VLM_CACHE_DIR" in os.environ:
        config.model_cache_dir = os.environ["VLM_CACHE_DIR"]
    if "VLM_DEVICE" in os.environ:
        config.device = os.environ["VLM_DEVICE"]
    
    # Processing settings
    if "VLM_BATCH_SIZE" in os.environ:
        config.batch_size = int(os.environ["VLM_BATCH_SIZE"])
    if "VLM_MAX_TOKENS" in os.environ:
        config.max_new_tokens = int(os.environ["VLM_MAX_TOKENS"])
    if "VLM_TEMPERATURE" in os.environ:
        config.temperature = float(os.environ["VLM_TEMPERATURE"])
    
    # Output settings
    if "VLM_OUTPUT_DIR" in os.environ:
        config.output_dir = os.environ["VLM_OUTPUT_DIR"]
    
    return config
