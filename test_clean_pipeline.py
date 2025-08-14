"""
Test the clean PDF → Images → OCR → vLLM → JSON pipeline
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from llm.ocr_processor import OCRProcessor
from llm.config import load_config_from_env


def test_clean_pipeline():
    """Test our clean OCR + LLM pipeline"""
    print("🧪 Testing Clean PDF → Images → OCR → vLLM → JSON Pipeline")
    print("=" * 60)
    
    # Load config
    config = load_config_from_env()
    processor = OCRProcessor(config)
    
    # Test server connection
    print("1. Testing vLLM server connection...")
    if processor.check_server_connection():
        print("   ✅ vLLM server is accessible")
    else:
        print("   ❌ vLLM server not available")
        print("   Please start the server: python start_vllm_server.py")
        return
    
    # Find test images
    images_dir = Path("results/images")
    if images_dir.exists():
        image_paths = list(images_dir.glob("*.png"))[:1]  # Test with one image
        
        if image_paths:
            print(f"2. Testing OCR + LLM processing on: {image_paths[0].name}")
            
            try:
                # Process single image
                results = processor.process_images([str(image_paths[0])])
                
                if results and results[0]['status'] == 'success':
                    entries = results[0]['entries']
                    print(f"   ✅ Successfully extracted {len(entries)} entries")
                    
                    # Show sample entries
                    for i, entry in enumerate(entries[:3]):
                        kalenjin = entry.get('grapheme', 'N/A')
                        english = entry.get('english_meaning', 'N/A')
                        print(f"      {i+1}. {kalenjin} → {english}")
                    
                    if len(entries) > 3:
                        print(f"      ... and {len(entries)-3} more entries")
                        
                else:
                    print("   ❌ Processing failed")
                    print(f"   Error: {results[0].get('error', 'Unknown')}")
                    
            except Exception as e:
                print(f"   ❌ Processing error: {e}")
        else:
            print("   ❌ No test images found in results/images/")
    else:
        print("   ❌ No results/images directory found")
    
    print("\n🎉 Clean pipeline test completed!")
    print("\nTo process your PDF:")
    print("  python main.py pipeline kalenjin_dictionary.pdf --output ./results")


if __name__ == "__main__":
    test_clean_pipeline()
