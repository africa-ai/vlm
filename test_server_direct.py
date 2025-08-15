#!/usr/bin/env python
"""
Direct vLLM Server Test - Simple diagnostic tool
"""
import asyncio
import aiohttp
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_server_direct():
    """Test vLLM server directly with minimal overhead"""
    
    server_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        # 1. Test health endpoint
        try:
            logger.info("🔍 Testing health endpoint...")
            async with session.get(f"{server_url}/health", timeout=aiohttp.ClientTimeout(total=10)) as response:
                health_status = response.status
                health_text = await response.text()
                logger.info(f"Health Status: {health_status}")
                logger.info(f"Health Response: {health_text}")
                
                if health_status != 200:
                    logger.error("❌ Server health check failed!")
                    return
                    
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return
        
        # 2. Test models endpoint
        try:
            logger.info("🔍 Testing models endpoint...")
            async with session.get(f"{server_url}/v1/models", timeout=aiohttp.ClientTimeout(total=10)) as response:
                models_status = response.status
                models_text = await response.text()
                logger.info(f"Models Status: {models_status}")
                logger.info(f"Models Response: {models_text}")
                
        except Exception as e:
            logger.error(f"⚠️  Models endpoint failed: {e}")
        
        # 3. Test simple completion
        try:
            logger.info("🔍 Testing simple completion...")
            
            payload = {
                "model": "Qwen/Qwen3-8B",
                "messages": [
                    {"role": "user", "content": "Hello, just say 'Hi' back"}
                ],
                "max_tokens": 5,
                "temperature": 0.0,
                "stream": False
            }
            
            logger.info(f"Sending payload: {json.dumps(payload, indent=2)}")
            
            async with session.post(
                f"{server_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                completion_status = response.status
                completion_text = await response.text()
                
                logger.info(f"Completion Status: {completion_status}")
                logger.info(f"Completion Response Length: {len(completion_text)}")
                logger.info(f"Completion Response: {completion_text}")
                
                if completion_status == 200:
                    try:
                        result = json.loads(completion_text)
                        choices = result.get("choices", [])
                        if choices:
                            message = choices[0].get("message", {})
                            content = message.get("content", "")
                            logger.info(f"✅ Completion successful: '{content}'")
                        else:
                            logger.warning("⚠️  No choices in response")
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ JSON decode failed: {e}")
                else:
                    logger.error(f"❌ Completion failed with status {completion_status}")
                    
        except asyncio.TimeoutError:
            logger.error("❌ Completion timed out after 30 seconds")
        except Exception as e:
            logger.error(f"❌ Completion failed: {e}")

if __name__ == "__main__":
    print("🚀 Testing vLLM server directly...")
    asyncio.run(test_server_direct())
