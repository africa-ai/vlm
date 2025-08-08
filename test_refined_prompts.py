#!/usr/bin/env python
"""
Test refined prompts with a sample dictionary page
"""

import sys
import asyncio
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm.vllm_processor import VLLMServerProcessor
from llm.config import load_config_from_env
from llm.parser.prompts import KalenjinPrompts


async def test_refined_prompts():
    """Test the refined prompts with sample dictionary processing"""
    
    # Load configuration
    config = load_config_from_env()
    processor = VLLMServerProcessor(config)
    
    # Test the refined prompts formatting
    print("=== Testing Refined Prompt Templates ===")
    
    # Test system prompt
    system_prompt = KalenjinPrompts.get_system_message()
    print(f"System Prompt Length: {len(system_prompt)} characters")
    print(f"System Prompt Preview: {system_prompt[:200]}...")
    
    # Test extraction prompt
    extraction_prompt = KalenjinPrompts.get_extraction_prompt(focus="complete")
    print(f"\nExtraction Prompt Length: {len(extraction_prompt)} characters")
    print(f"Extraction Prompt Preview: {extraction_prompt[:200]}...")
    
    # Test focused prompts
    for focus_type in ["grapheme", "ipa", "meaning"]:
        focused_prompt = KalenjinPrompts.get_extraction_prompt(focus=focus_type)
        print(f"\n{focus_type.upper()} Focus Prompt Length: {len(focused_prompt)} characters")
        print(f"{focus_type.upper()} Focus Preview: {focused_prompt[:200]}...")
    
    # Test batch prompt
    batch_prompt = KalenjinPrompts.get_extraction_prompt(focus="complete", page_number=1)
    print(f"\nBatch Prompt Length: {len(batch_prompt)} characters")
    print(f"Batch Prompt Preview: {batch_prompt[:200]}...")
    
    # Test conversation formatting
    conversation = KalenjinPrompts.format_conversation(
        user_prompt="Analyze this dictionary page and extract all Kalenjin entries.",
        image_path="sample.jpg"
    )
    print(f"\nConversation Messages: {len(conversation)} messages")
    for i, msg in enumerate(conversation):
        print(f"Message {i+1}: {msg['role']} - {len(msg['content'])} chars")
    
    print("\n=== All prompt templates validated successfully! ===")
    
    # Check if server is running for a live test
    try:
        await processor.test_connection()
        print("\n=== Server is running - ready for live processing! ===")
        
        # Find a sample image to test with
        sample_images = list(Path(".").glob("*.jpg")) + list(Path(".").glob("*.png"))
        if sample_images:
            print(f"Sample images available: {[img.name for img in sample_images[:3]]}")
            print("You can now run the full processing pipeline with improved prompts!")
        
    except Exception as e:
        print(f"\n=== Server not running: {e} ===")
        print("Start the server first: python start_vllm_server.py")


if __name__ == "__main__":
    asyncio.run(test_refined_prompts())
