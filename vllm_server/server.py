"""
vLLM Server for NVIDIA Cosmos-Reason1-7B
Provides OpenAI-compatible API for vision-language model inference
"""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator
import uvicorn
from fastapi import FastAPI, HTTPException
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from .config import VLLMServerConfig
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VLLMServer:
    """vLLM Server wrapper for Cosmos-Reason1-7B"""
    
    def __init__(self, config: VLLMServerConfig):
        self.config = config
        self.engine: Optional[AsyncLLMEngine] = None
        self.app: Optional[FastAPI] = None
        
    async def create_engine(self):
        """Create vLLM engine from configuration"""
        # Multi-GPU hardening: set NCCL env flags if configured
        if self.config.nccl_p2p_disable:
            os.environ["NCCL_P2P_DISABLE"] = "1"
        if self.config.nccl_ib_disable:
            os.environ["NCCL_IB_DISABLE"] = "1"
        if self.config.nccl_socket_ifname:
            os.environ["NCCL_SOCKET_IFNAME"] = self.config.nccl_socket_ifname
        if self.config.nccl_shm_disable:
            os.environ["NCCL_SHM_DISABLE"] = "1"
        if self.config.cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices
        if self.config.pytorch_cuda_alloc_conf:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = self.config.pytorch_cuda_alloc_conf
        # If running tensor parallel across multiple GPUs, CUDA graphs can cause instability in some envs
        effective_enable_cudagraph = self.config.enable_cudagraph and self.config.tensor_parallel_size == 1
        if not effective_enable_cudagraph and self.config.enable_cudagraph and self.config.tensor_parallel_size > 1:
            logger.warning("Disabling CUDA graphs because tensor_parallel_size > 1 for stability.")

        engine_args = AsyncEngineArgs(
            model=self.config.model_name,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            enforce_eager=not effective_enable_cudagraph,
        )
        logger.info(
            "Engine args: model=%s, tp=%s, gpu_mem=%.2f, max_len=%s, enforce_eager=%s, NCCL_P2P_DISABLE=%s, NCCL_IB_DISABLE=%s, NCCL_SOCKET_IFNAME=%s, NCCL_SHM_DISABLE=%s, CUDA_VISIBLE_DEVICES=%s, PYTORCH_CUDA_ALLOC_CONF=%s",
            self.config.model_name,
            self.config.tensor_parallel_size,
            self.config.gpu_memory_utilization,
            self.config.max_model_len,
            (not effective_enable_cudagraph),
            os.environ.get("NCCL_P2P_DISABLE", "0"),
            os.environ.get("NCCL_IB_DISABLE", "0"),
            os.environ.get("NCCL_SOCKET_IFNAME", ""),
            os.environ.get("NCCL_SHM_DISABLE", "0"),
            os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        )
        
        # Create async engine
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        logger.info("vLLM engine initialized successfully")
        return engine
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        """FastAPI lifespan manager"""
        # Startup
        logger.info("Starting vLLM server...")
        self.engine = await self.create_engine()
        logger.info(f"Server ready on {self.config.host}:{self.config.port}")
        
        yield
        
        # Shutdown
        logger.info("Shutting down vLLM server...")
        if self.engine:
            # Clean shutdown of engine
            pass
    
    def create_app(self) -> FastAPI:
        """Create FastAPI application"""
        app = FastAPI(
            title="Cosmos-Reason1-7B vLLM Server",
            description="OpenAI-compatible API for NVIDIA Cosmos-Reason1-7B Vision Language Model",
            version="1.0.0",
            lifespan=self.lifespan
        )
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint"""
            if self.engine is None:
                raise HTTPException(status_code=503, detail="Engine not initialized")
            return {"status": "healthy", "model": self.config.model_name}
        
        @app.get("/v1/models")
        async def list_models():
            """List available models (OpenAI compatible)"""
            return {
                "object": "list",
                "data": [{
                    "id": self.config.served_model_name or self.config.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "nvidia"
                }]
            }
        
        @app.post("/v1/chat/completions")
        async def create_chat_completion(request: dict):
            """Create chat completion (OpenAI compatible)"""
            if self.engine is None:
                raise HTTPException(status_code=503, detail="Engine not initialized")
            
            try:
                # Extract parameters
                messages = request.get("messages", [])
                max_tokens = request.get("max_tokens", 512)
                temperature = request.get("temperature", 0.7)
                top_p = request.get("top_p", 0.9)
                
                # Convert messages to prompt
                prompt = self._messages_to_prompt(messages)
                
                # Create sampling parameters
                sampling_params = SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    use_beam_search=False
                )
                
                # Generate request ID
                request_id = random_uuid()
                
                # Generate response
                async for output in self.engine.generate(prompt, sampling_params, request_id):
                    if output.finished:
                        generated_text = output.outputs[0].text
                        
                        return {
                            "id": request_id,
                            "object": "chat.completion",
                            "created": 0,
                            "model": self.config.served_model_name or self.config.model_name,
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": generated_text
                                },
                                "finish_reason": "stop"
                            }],
                            "usage": {
                                "prompt_tokens": len(prompt.split()),
                                "completion_tokens": len(generated_text.split()),
                                "total_tokens": len(prompt.split()) + len(generated_text.split())
                            }
                        }
                
            except Exception as e:
                logger.error(f"Error in chat completion: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/v1/vision/analyze")
        async def analyze_image(request: dict):
            """Custom endpoint for vision analysis"""
            if self.engine is None:
                raise HTTPException(status_code=503, detail="Engine not initialized")
            
            try:
                # Extract image and prompt
                image_data = request.get("image")
                prompt = request.get("prompt", "Analyze this image.")
                max_tokens = request.get("max_tokens", 1024)
                
                # Create vision prompt
                vision_prompt = f"<image>{image_data}</image>\n{prompt}"
                
                # Create sampling parameters
                sampling_params = SamplingParams(
                    max_tokens=max_tokens,
                    temperature=0.1,
                    top_p=0.9
                )
                
                # Generate request ID
                request_id = random_uuid()
                
                # Generate response
                async for output in self.engine.generate(vision_prompt, sampling_params, request_id):
                    if output.finished:
                        return {
                            "id": request_id,
                            "analysis": output.outputs[0].text,
                            "model": self.config.served_model_name or self.config.model_name
                        }
                        
            except Exception as e:
                logger.error(f"Error in vision analysis: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        self.app = app
        return app
    
    def _messages_to_prompt(self, messages: list) -> str:
        """Convert OpenAI messages format to prompt"""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n".join(prompt_parts) + "\nAssistant:"
    
    def run(self):
        """Run the vLLM server"""
        app = self.create_app()
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start server
        uvicorn.run(
            app,
            host=self.config.host,
            port=self.config.port,
            workers=self.config.workers,
            access_log=True,
            log_config=None,
            loop="asyncio",
            server_header=False
        )


async def main():
    """Main entry point"""
    # Load configuration
    config = VLLMServerConfig.from_env()
    
    # Create and run server
    server = VLLMServer(config)
    server.run()


if __name__ == "__main__":
    asyncio.run(main())
