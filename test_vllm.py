#!/usr/bin/env python
"""
Test vLLM Server Connection and Processing
"""

import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from vllm_server.client import test_vllm_connection, SyncVLLMClient
from llm.vllm_processor import VLLMServerProcessor
from llm.config import load_config_from_env


def test_server_connection(server_url: str = "http://localhost:8000"):
    """Test basic server connection"""
    print(f"Testing connection to vLLM server at {server_url}...")
    
    if test_vllm_connection(server_url):
        print("✅ Server connection successful!")
        return True
    else:
        print("❌ Server connection failed!")
        return False


def test_image_processing(server_url: str = "http://localhost:8000", 
                         image_path: str = None):
    """Test image processing through vLLM server"""
    if not image_path:
        print("No test image provided, skipping image processing test")
        return
    
    print(f"\nTesting image processing with: {image_path}")
    
    try:
        # Create processor
        config = load_config_from_env()
        processor = VLLMServerProcessor(config, server_url)
        
        # Check connection
        if not processor.check_server_connection():
            print("❌ Failed to connect to server")
            return
        
        # Process image
        result = processor.process_image(image_path)
        
        if result.get("status") == "success":
            print("✅ Image processing successful!")
            print(f"   Found {len(result.get('entries', []))} dictionary entries")
            
            # Show first entry as example
            entries = result.get('entries', [])
            if entries:
                print(f"   Example entry: {entries[0]}")
        else:
            print(f"❌ Image processing failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error during image processing: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test vLLM server")
    parser.add_argument("--server-url", default="http://localhost:8000",
                       help="vLLM server URL")
    parser.add_argument("--test-image", help="Test image path")
    parser.add_argument("--wait-for-server", action="store_true",
                       help="Wait for server to become available")
    
    args = parser.parse_args()
    
    # Wait for server if requested
    if args.wait_for_server:
        print("Waiting for server to become available...")
        max_attempts = 30
        for attempt in range(max_attempts):
            if test_server_connection(args.server_url):
                break
            print(f"Attempt {attempt + 1}/{max_attempts} failed, retrying in 2 seconds...")
            time.sleep(2)
        else:
            print("❌ Server did not become available within timeout")
            sys.exit(1)
    
    # Test connection
    if not test_server_connection(args.server_url):
        print("\nIs the vLLM server running? To start it:")
        print("python start_vllm_server.py")
        sys.exit(1)
    
    # Test image processing if image provided
    if args.test_image:
        test_image_processing(args.server_url, args.test_image)
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()
