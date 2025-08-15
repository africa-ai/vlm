"""
Test optimized chunked processing to resolve timeout issues
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from llm.config import load_config_from_env
from llm.ocr_processor import OCRProcessor


async def test_chunked_processing():
    """Test the enhanced OCR processor with chunking"""
    
    print("🔧 Testing Enhanced OCR Processor with Chunking...")
    
    # Load config
    config = load_config_from_env()
    
    # Create processor
    processor = OCRProcessor(config)
    
    # Test image
    test_image = "kalenjin_dictionary_page_002.png"
    
    if not Path(test_image).exists():
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"📷 Processing {test_image}...")
    
    try:
        # This should now use chunked processing for the 4,466 char text
        result = processor.process_image(test_image)
        
        print("\n✅ Processing Complete!")
        print(f"📊 Status: {result.get('status')}")
        print(f"📝 Dictionary Entries: {len(result.get('entries', []))}")
        print(f"📝 Example Sentences: {len(result.get('examples', []))}")
        print(f"🔧 Method: {result.get('method')}")
        
        # Show sample entries
        entries = result.get('entries', [])
        if entries:
            print(f"\n📋 Sample entries:")
            for i, entry in enumerate(entries[:3]):
                print(f"  {i+1}. {entry.get('grapheme', 'N/A')} ({entry.get('grammar', 'N/A')}) - {entry.get('definition', 'N/A')[:50]}...")
        
        # Show sample sentences
        examples = result.get('examples', [])
        if examples:
            print(f"\n💬 Sample sentences:")
            for i, example in enumerate(examples[:2]):
                print(f"  {i+1}. {example.get('kalenjin', 'N/A')} = {example.get('english', 'N/A')}")
        
        if result.get('error'):
            print(f"⚠️ Error: {result['error']}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_chunked_processing())
