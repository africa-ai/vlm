#!/usr/bin/env python3
"""
Performance Test Script
Test the optimized OCR + LLM pipeline
"""

import time
from pathlib import Path
from llm.config import load_config_from_env
from llm.ocr_processor import OCRProcessor

def test_single_image_performance():
    """Test performance on a single image"""
    print("🚀 Testing Optimized OCR + LLM Performance...")
    
    # Setup
    config = load_config_from_env()
    processor = OCRProcessor(config)
    
    # Check server connection
    if not processor.check_server_connection():
        print("❌ vLLM server not available")
        return
    
    # Test image path
    test_image = Path("c:/Users/sugutt/Downloads/kalenjin_dictionary_page_003.png")
    
    if not test_image.exists():
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"📄 Processing: {test_image.name}")
    print("⏱️  Starting performance test...")
    
    # Time the processing
    start_time = time.time()
    
    result = processor.process_image(test_image)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Results
    print(f"\n🏁 Performance Results:")
    print(f"   ⏱️  Processing Time: {processing_time:.2f} seconds")
    print(f"   📊 Entries Found: {len(result.get('entries', []))}")
    print(f"   🎯 Status: {result.get('status')}")
    print(f"   🔧 Method: {result.get('method', 'N/A')}")
    
    if processing_time < 60:
        print(f"   🚀 EXCELLENT! Under 1 minute (target achieved)")
    elif processing_time < 120:
        print(f"   ✅ GOOD! Under 2 minutes (significant improvement)")
    else:
        print(f"   ⚠️  Still slow, needs more optimization")
    
    # Show sample entries
    entries = result.get('entries', [])
    if entries:
        print(f"\n📝 Sample Entries:")
        for i, entry in enumerate(entries[:3]):  # Show first 3
            print(f"   {i+1}. {entry.get('grapheme', 'N/A')} -> {entry.get('english_meaning', 'N/A')[:50]}...")
    
    return processing_time, len(entries)

if __name__ == "__main__":
    test_single_image_performance()
