#!/usr/bin/env python3
"""
Test Qwen2.5-8B-Instruct Performance vs Cosmos
Quick performance comparison for OCR + LLM pipeline
"""
import time
import logging
from pathlib import Path

from llm.config import load_config_from_env
from llm.ocr_processor import OCRProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_qwen_performance():
    """Test Qwen2.5-8B performance on sample dictionary page"""
    
    logger.info("🚀 Testing Qwen2.5-8B-Instruct Performance")
    logger.info("=" * 60)
    
    # Load config and create processor
    config = load_config_from_env()
    processor = OCRProcessor(config, "http://localhost:8000")
    
    # Test server connection
    logger.info("📡 Checking vLLM server connection...")
    if not processor.check_server_connection():
        logger.error("❌ vLLM server not accessible! Start with: python start_vllm_server.py")
        return
    
    logger.info("✅ vLLM server connected successfully!")
    
    # Test sample images
    test_images = [
        "c:\\Users\\sugutt\\Downloads\\kalenjin_dictionary_page_003.png",
        "results/images/kalenjin_dictionary_page_001.png",
        "results/images/kalenjin_dictionary_page_002.png"
    ]
    
    # Find existing test image
    test_image = None
    for image_path in test_images:
        if Path(image_path).exists():
            test_image = image_path
            break
    
    if not test_image:
        logger.error("❌ No test images found. Please ensure dictionary pages are available.")
        return
    
    logger.info(f"🔍 Testing with: {Path(test_image).name}")
    
    # Performance test
    start_time = time.time()
    
    result = processor.process_image(test_image)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Results
    logger.info("📊 PERFORMANCE RESULTS")
    logger.info("-" * 40)
    logger.info(f"⏱️  Processing Time: {processing_time:.2f} seconds")
    
    if result.get("status") == "success":
        entries = result.get("entries", [])
        logger.info(f"📚 Entries Extracted: {len(entries)}")
        logger.info(f"🎯 Method: {result.get('method', 'OCR + LLM')}")
        
        # Show some sample entries
        if entries:
            logger.info("\n🔍 SAMPLE ENTRIES:")
            for i, entry in enumerate(entries[:3]):  # Show first 3
                grapheme = entry.get('grapheme', '')
                meaning = entry.get('english_meaning', '')
                ipa = entry.get('ipa', '')
                logger.info(f"   {i+1}. {grapheme} → {meaning[:50]}...")
                if ipa:
                    logger.info(f"      IPA: {ipa}")
        
        # Speed comparison
        logger.info("\n⚡ SPEED COMPARISON:")
        if processing_time < 30:
            logger.info(f"   🚀 EXCELLENT! {processing_time:.1f}s is much faster than 2+ minutes!")
        elif processing_time < 60:
            logger.info(f"   ✅ GOOD! {processing_time:.1f}s is a significant improvement!")
        else:
            logger.info(f"   ⚠️  Still slow at {processing_time:.1f}s - may need further optimization")
            
    else:
        logger.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
    
    logger.info("=" * 60)
    logger.info("Test completed!")

if __name__ == "__main__":
    test_qwen_performance()
