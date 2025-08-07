"""
vLLM Client for Cosmos-Reason1-7B Dictionary Processing
Connects to vLLM server for distributed inference
"""

import asyncio
import base64
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import httpx
import aiohttp
from PIL import Image
from .config import VLLMServerConfig

logger = logging.getLogger(__name__)


class VLLMClient:
    """Client for connecting to vLLM server"""
    
    def __init__(self, server_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Headers for requests
        self.headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),
            headers=self.headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> bool:
        """Check if vLLM server is healthy"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30),
                    headers=self.headers
                )
            
            async with self.session.get(f"{self.server_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Server healthy: {data}")
                    return True
                else:
                    logger.error(f"Health check failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False
    
    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image to base64 string"""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load and process image
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Encode to base64
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/jpeg;base64,{image_data}"
    
    async def analyze_dictionary_image(self, 
                                     image_path: Union[str, Path],
                                     custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze dictionary image with Cosmos-Reason1-7B
        
        Args:
            image_path: Path to the dictionary page image
            custom_prompt: Optional custom prompt for analysis
            
        Returns:
            Dictionary with analysis results
        """
        if not self.session:
            raise RuntimeError("Client session not initialized. Use async context manager.")
        
        # Default reasoning prompt for Kalenjin dictionary
        default_prompt = """<reasoning>
I need to analyze this Kalenjin dictionary page systematically to extract dictionary entries.

Let me examine:
1. The overall layout and structure of the page
2. Individual dictionary entries with their components
3. The format pattern for each entry (grapheme -> IPA -> English meaning)
4. Any special formatting or organization

For each dictionary entry, I should identify:
- The Kalenjin word/grapheme (original text)
- The IPA phonetic transcription (if present)
- The English translation/meaning
- Any additional linguistic information

I'll organize these into structured data.
</reasoning>

Analyze this Kalenjin dictionary page image. Extract all dictionary entries with their:
1. Original Kalenjin grapheme/word
2. IPA phonetic transcription (if present)
3. English meaning/translation

Format the response as a structured analysis with clear identification of each entry."""
        
        prompt = custom_prompt or default_prompt
        
        try:
            # Encode image
            image_data = self._encode_image(image_path)
            
            # Prepare request payload
            payload = {
                "image": image_data,
                "prompt": prompt,
                "max_tokens": 2048
            }
            
            # Send request to vLLM server
            async with self.session.post(
                f"{self.server_url}/v1/vision/analyze",
                json=payload
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Successfully analyzed image: {image_path}")
                    return {
                        "image_path": str(image_path),
                        "analysis": result.get("analysis", ""),
                        "model": result.get("model", ""),
                        "request_id": result.get("id", ""),
                        "status": "success"
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"Analysis failed: {response.status} - {error_text}")
                    return {
                        "image_path": str(image_path),
                        "error": f"HTTP {response.status}: {error_text}",
                        "status": "error"
                    }
                    
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            return {
                "image_path": str(image_path),
                "error": str(e),
                "status": "error"
            }
    
    async def chat_completion(self, 
                            messages: List[Dict[str, str]], 
                            max_tokens: int = 1024,
                            temperature: float = 0.1) -> Dict[str, Any]:
        """
        Send chat completion request to vLLM server
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Chat completion response
        """
        if not self.session:
            raise RuntimeError("Client session not initialized. Use async context manager.")
        
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9
        }
        
        try:
            async with self.session.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload
            ) as response:
                
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Chat completion failed: {response.status} - {error_text}")
                    return {
                        "error": f"HTTP {response.status}: {error_text}",
                        "status": "error"
                    }
                    
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return {"error": str(e), "status": "error"}
    
    async def batch_process_images(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Process multiple images in parallel
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of processing results
        """
        logger.info(f"Starting batch processing of {len(image_paths)} images")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
        
        async def process_single_image(image_path):
            async with semaphore:
                return await self.analyze_dictionary_image(image_path)
        
        # Process all images concurrently
        tasks = [process_single_image(path) for path in image_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "image_path": str(image_paths[i]),
                    "error": str(result),
                    "status": "error"
                })
            else:
                processed_results.append(result)
        
        successful = sum(1 for r in processed_results if r.get("status") == "success")
        logger.info(f"Batch processing completed: {successful}/{len(image_paths)} successful")
        
        return processed_results


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
    
    def analyze_dictionary_image(self, image_path: Union[str, Path], 
                               custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous image analysis"""
        async def _analyze():
            async with self.client as client:
                return await client.analyze_dictionary_image(image_path, custom_prompt)
        
        return asyncio.run(_analyze())
    
    def batch_process_images(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """Synchronous batch processing"""
        async def _process():
            async with self.client as client:
                return await client.batch_process_images(image_paths)
        
        return asyncio.run(_process())


# Convenience function for quick testing
def test_vllm_connection(server_url: str = "http://localhost:8000") -> bool:
    """Test connection to vLLM server"""
    client = SyncVLLMClient(server_url)
    return client.health_check()


if __name__ == "__main__":
    # Test the client
    import sys
    
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print(f"Testing connection to vLLM server at {server_url}...")
    if test_vllm_connection(server_url):
        print("✅ Connection successful!")
    else:
        print("❌ Connection failed!")
