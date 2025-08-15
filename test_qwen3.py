#!/usr/bin/env python
"""
Test script for Qwen3-8B OCR + LLM pipeline
Verify performance improvements and thinking disabled
"""

import time
from pathlib import Path
from llm.config import load_config_from_env
from llm.ocr_processor import OCRProcessor

def test_qwen3_performance():
    """Test Qwen3-8B performance with thinking disabled"""
    print("🚀 Testing Qwen3-8B OCR + LLM Pipeline (Thinking Disabled)")
    print("=" * 60)
    
    # Load config
    config = load_config_from_env()
    processor = OCRProcessor(config, server_url="http://localhost:8000")
    
    # Check server connection
    print("🔍 Checking server connection...")
    if not processor.check_server_connection():
        print("❌ vLLM server not accessible at http://localhost:8000")
        print("Please start the server with: python start_vllm_server.py")
        return
    
    print("✅ Server connection successful!")
    
    # Test with sample image
    test_image = "c:\\Users\\sugutt\\Downloads\\kalenjin_dictionary_page_003.png"
    
    if not Path(test_image).exists():
        print(f"❌ Test image not found: {test_image}")
        print("Please provide a dictionary page image to test with.")
        return
    
    print(f"📖 Processing test image: {test_image}")
    
    # Time the processing
    start_time = time.time()
    
    try:
        # Process image
        result = processor.process_image(test_image)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Display results
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        
        if result.get('status') == 'success':
            entries = result.get('entries', [])
            print(f"✅ Successfully extracted {len(entries)} entries")
            
            # Show sample entries
            print("\n📝 Sample entries:")
            for i, entry in enumerate(entries[:3]):  # Show first 3
                grapheme = entry.get('grapheme', 'N/A')
                meaning = entry.get('english_meaning', 'N/A')
                ipa = entry.get('ipa', 'N/A')
                
                print(f"  {i+1}. {grapheme} - {meaning}")
                if ipa and ipa != 'N/A':
                    print(f"     IPA: {ipa}")
            
            if len(entries) > 3:
                print(f"     ... and {len(entries) - 3} more entries")
            
            # Performance analysis
            print(f"\n📊 Performance Analysis:")
            print(f"   • Processing speed: {len(entries)/processing_time:.1f} entries/second")
            
            if processing_time < 30:
                print("   🚀 EXCELLENT: Under 30 seconds!")
            elif processing_time < 60:
                print("   ✅ GOOD: Under 1 minute")
            else:
                print("   ⚠️  SLOW: Over 1 minute - check optimization")
                
        else:
            error = result.get('error', 'Unknown error')
            print(f"❌ Processing failed: {error}")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")

if __name__ == "__main__":
    test_qwen3_performance()
