"""
Clean vLLM Client for OCR + Text Processing Pipeline
Only contains what we need: text completion via vLLM server
"""

import json
import logging
from typing import Optional, Dict, Any, List
import asyncio
import aiohttp

logger = logging.getLogger(__name__)


class VLLMClient:
    """Clean async client for vLLM text completion"""
    
    def __init__(self, server_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> bool:
        """Check if vLLM server is healthy"""
        try:
            if not self.session:
                raise RuntimeError("Client session not initialized")
            
            async with self.session.get(
                f"{self.server_url}/health",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def complete_text(self, prompt: str, max_tokens: int = 1500,
                          temperature: float = 0.0) -> Dict[str, Any]:
        """
        Complete text using vLLM server via chat completions endpoint
        Optimized for A10G performance with better error handling
        """
        if not self.session:
            raise RuntimeError("Client session not initialized")
        
        try:
            # Optimized payload for speed with thinking disabled
            payload = {
                "model": "Qwen/Qwen3-8B",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "stop": ["</json>", "\n\n---", "```", "Human:", "Assistant:"],
                "stream": False,
                "enable_thinking": False,  # Disable Qwen3 thinking mode for speed
            }
            
            logger.info(f"Sending request to {self.server_url}/v1/chat/completions")
            logger.info(f"Payload model: {payload['model']}, max_tokens: {max_tokens}")
            logger.info(f"Prompt length: {len(prompt)} characters")
            
            async with self.session.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=60)  # Reduced from 120
            ) as response:
                
                response_text = await response.text()
                logger.info(f"Response status: {response.status}")
                logger.info(f"Response length: {len(response_text)} characters")
                
                if response.status == 200:
                    try:
                        result = await response.json()
                        
                        # Extract completion text from chat format
                        completion = ""
                        if "choices" in result and len(result["choices"]) > 0:
                            choice = result["choices"][0]
                            if "message" in choice and "content" in choice["message"]:
                                completion = choice["message"]["content"]
                            elif "text" in choice:
                                completion = choice["text"]
                        
                        logger.info(f"Successfully extracted completion: {len(completion)} characters")
                        
                        return {
                            "status": "success",
                            "completion": completion,
                            "model": result.get("model", "Qwen/Qwen3-8B"),
                            "usage": result.get("usage", {}),
                        }
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode failed: {e}")
                        logger.error(f"Raw response (first 500 chars): {response_text[:500]}")
                        return {
                            "status": "error",
                            "error": f"JSON decode error: {e}. Response: {response_text[:200]}"
                        }
                        
                else:
                    logger.error(f"HTTP error {response.status}")
                    logger.error(f"Error response: {response_text}")
                    return {
                        "status": "error",
                        "error": f"HTTP {response.status}: {response_text[:200]}"
                    }
                    
        except asyncio.TimeoutError:
            logger.error("Request timed out after 60 seconds")
            return {
                "status": "error",
                "error": "Request timed out after 60 seconds"
            }
        except Exception as e:
            logger.error(f"Text completion failed with exception: {e}")
            return {
                "status": "error",
                "error": f"Exception: {str(e)}"
            }


class SyncVLLMClient:
    """Synchronous wrapper for VLLMClient"""
    
    def __init__(self, server_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.client = VLLMClient(server_url, api_key)
    
    def health_check(self) -> bool:
        """Synchronous health check"""
        async def _check():
            async with self.client as client:
                return await client.health_check()
        
        return asyncio.run(_check())
    
    def complete_text(self, prompt: str, max_tokens: int = 1500,  # Reduced from 2000
                     temperature: float = 0.0) -> Dict[str, Any]:  # Set to 0 for faster generation
        """Synchronous text completion - optimized for speed"""
        async def _complete():
            async with self.client as client:
                return await client.complete_text(prompt, max_tokens, temperature)
        
        return asyncio.run(_complete())


# Convenience function for testing
def test_vllm_connection(server_url: str = "http://localhost:8000") -> bool:
    """Test connection to vLLM server"""
    client = SyncVLLMClient(server_url)
    return client.health_check()


if __name__ == "__main__":
    # Test the client
    import sys
    
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print(f"Testing vLLM server connection to {server_url}...")
    if test_vllm_connection(server_url):
        print("✅ vLLM server is accessible")
        
        # Test text completion
        client = SyncVLLMClient(server_url)
        result = client.complete_text("The capital of France is")
        
        if result.get("status") == "success":
            print(f"✅ Text completion works: {result['completion'][:100]}...")
        else:
            print(f"❌ Text completion failed: {result.get('error')}")
    else:
        print("❌ vLLM server is not accessible")
        print("Make sure the server is running with: python start_vllm_server.py")
