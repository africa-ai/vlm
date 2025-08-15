#!/usr/bin/env python
"""
Quick vLLM Server Diagnostic and Fix
"""

import asyncio
import aiohttp
import json
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_server_step_by_step():
    """Test server with progressively complex requests"""
    
    server_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Health check
        print("1️⃣ Testing server health...")
        try:
            async with session.get(f"{server_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    print("✅ Server is healthy")
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return
        
        # Test 2: Simple completion
        print("\n2️⃣ Testing simple completion...")
        simple_payload = {
            "model": "Qwen/Qwen3-8B",
            "messages": [{"role": "user", "content": "Say 'Hello'"}],
            "max_tokens": 5,
            "temperature": 0.0
        }
        
        try:
            start_time = time.time()
            async with session.post(
                f"{server_url}/v1/chat/completions",
                json=simple_payload,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                end_time = time.time()
                
                if response.status == 200:
                    result = await response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    print(f"✅ Simple completion works: '{content}' ({end_time-start_time:.2f}s)")
                else:
                    error_text = await response.text()
                    print(f"❌ Simple completion failed: {response.status} - {error_text}")
                    return
        except Exception as e:
            print(f"❌ Simple completion failed: {e}")
            return
        
        # Test 3: JSON extraction (medium complexity)
        print("\n3️⃣ Testing JSON extraction...")
        json_payload = {
            "model": "Qwen/Qwen3-8B",
            "messages": [{"role": "user", "content": 'Extract from "hello n. /helo/ greeting" as JSON: [{"word": "hello", "pos": "n", "ipa": "/helo/", "meaning": "greeting"}]'}],
            "max_tokens": 100,
            "temperature": 0.0
        }
        
        try:
            start_time = time.time()
            async with session.post(
                f"{server_url}/v1/chat/completions",
                json=json_payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                end_time = time.time()
                
                if response.status == 200:
                    result = await response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    print(f"✅ JSON extraction works ({end_time-start_time:.2f}s)")
                    print(f"   Response: {content[:100]}...")
                else:
                    error_text = await response.text()
                    print(f"❌ JSON extraction failed: {response.status} - {error_text}")
        except Exception as e:
            print(f"❌ JSON extraction failed: {e}")
        
        # Test 4: Check token limits and model config
        print("\n4️⃣ Testing model info...")
        try:
            async with session.get(f"{server_url}/v1/models") as response:
                if response.status == 200:
                    models = await response.json()
                    print("✅ Available models:")
                    for model in models.get("data", []):
                        print(f"   - {model.get('id', 'unknown')}")
                else:
                    print("⚠️ Could not get model info")
        except Exception as e:
            print(f"⚠️ Model info failed: {e}")

if __name__ == "__main__":
    print("🔍 vLLM Server Diagnostic")
    print("=" * 40)
    asyncio.run(test_server_step_by_step())
