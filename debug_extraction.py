#!/usr/bin/env python
"""
Debug the extraction process to see what the model is responding with
"""

import sys
import asyncio
import base64
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from vllm_server.client import SyncVLLMClient
from llm.parser.main import DictionaryParser
from llm.parser.prompts import PromptTemplates
from llm.vllm_processor import VLLMServerProcessor
from llm.config import load_config_from_env

def debug_single_image():
    """Debug a single image extraction"""
    # Load config and create processor
    config = load_config_from_env()
    processor = VLLMServerProcessor(config)
    
    print(f"🔧 Using VLLMProcessor with preprocessing enabled")
    
    # Find the first image to test
    images_dir = Path("results/images")
    if not images_dir.exists():
        print("❌ No images directory found")
        return
        
    image_files = list(images_dir.glob("*.png"))[:1]  # Just test the first image
    
    if not image_files:
        print("❌ No images found")
        return
        
    image_path = image_files[0]
    print(f"🔍 Testing image: {image_path.name}")
    
    try:
        # First, let's see what prompt is being used
        from llm.parser.prompts import PromptTemplates
        actual_prompt = PromptTemplates.get_extraction_prompt("complete")
        print(f"📝 Prompt being used (first 800 chars):")
        print(actual_prompt[:800])
        print("\n" + "="*50 + "\n")
        
        # Process with the full pipeline (includes preprocessing)
        print("🤖 Processing with VLLMProcessor (includes preprocessing)...")
        result = processor.process_image(str(image_path))
        
        print(f"✅ Processor Result type: {type(result)}")
        print(f"✅ Result content:")
        if isinstance(result, dict):
            for key, value in result.items():
                if key == 'entries':
                    print(f"   {key}: {len(value) if isinstance(value, list) else value} entries")
                    if isinstance(value, list) and value:
                        print(f"      Sample: {value[0]}")
                elif key == 'raw_responses':
                    print(f"   {key}: {len(value) if isinstance(value, list) else 'N/A'} responses")
                    if isinstance(value, list) and value:
                        print(f"      Full response: {value[0][:1000]}...")
                else:
                    print(f"   {key}: {str(value)[:200]}...")
        else:
            print(f"   {str(result)[:500]}...")
        print("\n" + "="*50 + "\n")
        
        # Result should already be parsed by the processor
        if isinstance(result, dict) and 'entries' in result:
            entries = result['entries']
            print(f"📊 Found {len(entries)} entries from processor:")
            for i, entry in enumerate(entries[:3]):  # Show first 3
                print(f"   Entry {i+1}: {entry}")
        else:
            print(f"❓ Unexpected result format: {result}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    debug_single_image()

if __name__ == "__main__":
    main()
