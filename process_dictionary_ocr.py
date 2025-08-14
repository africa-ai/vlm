#!/usr/bin/env python
"""
Main script for OCR + LLM dictionary extraction
Simple, fast, reliable approach
"""

import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm.ocr_processor import create_ocr_processor


def main():
    """Main function for OCR + LLM processing"""
    parser = argparse.ArgumentParser(description="Extract dictionary entries using OCR + LLM")
    
    parser.add_argument("--images-dir", type=str, default="results/images",
                       help="Directory containing dictionary images")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000",
                       help="LLM server URL")
    parser.add_argument("--single-image", type=str,
                       help="Process single image instead of batch")
    parser.add_argument("--test", action="store_true",
                       help="Run test on first image only")
    
    args = parser.parse_args()
    
    print("🚀 OCR + LLM Dictionary Extraction")
    print("📋 Pipeline: Image → OCR → LLM → JSON")
    print("⚡ Much faster than vision models!")
    print()
    
    # Create processor
    try:
        processor = create_ocr_processor(server_url=args.server_url)
        print(f"✅ OCR processor initialized")
        print(f"🔗 LLM server: {args.server_url}")
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        return 1
    
    # Process images
    if args.single_image:
        # Single image processing
        image_path = Path(args.single_image)
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return 1
            
        print(f"🔍 Processing single image: {image_path.name}")
        result = processor.process_image(image_path)
        
        if result.get('status') == 'success':
            entries_count = len(result.get('entries', []))
            print(f"✅ Success! Extracted {entries_count} entries")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
            
    elif args.test:
        # Test mode - first image only
        images_dir = Path(args.images_dir)
        if not images_dir.exists():
            print(f"❌ Images directory not found: {images_dir}")
            return 1
            
        image_files = list(images_dir.glob("*.png"))[:1]
        if not image_files:
            print(f"❌ No images found in {images_dir}")
            return 1
            
        print(f"🧪 Test mode: Processing {image_files[0].name}")
        result = processor.process_image(image_files[0])
        
        if result.get('status') == 'success':
            entries = result.get('entries', [])
            print(f"✅ Test successful! Found {len(entries)} entries")
            
            # Show sample entries
            for i, entry in enumerate(entries[:3], 1):
                kalenjin = entry.get('grapheme', '')
                english = entry.get('english_meaning', '')
                print(f"  {i}. {kalenjin} → {english}")
                
        else:
            print(f"❌ Test failed: {result.get('error', 'Unknown error')}")
            
    else:
        # Batch processing
        images_dir = Path(args.images_dir)
        if not images_dir.exists():
            print(f"❌ Images directory not found: {images_dir}")
            return 1
            
        image_files = list(images_dir.glob("*.png"))
        if not image_files:
            print(f"❌ No images found in {images_dir}")
            return 1
            
        print(f"📁 Found {len(image_files)} images in {images_dir}")
        print(f"🔄 Starting batch processing...")
        
        results = processor.batch_process_images(image_files)
        
        # Summary
        successful = sum(1 for r in results if r.get('status') == 'success')
        total_entries = len(processor.all_entries)
        
        print(f"\n🏁 Batch processing complete!")
        print(f"   ✅ Processed: {successful}/{len(image_files)} images")
        print(f"   📚 Total entries: {total_entries}")
        print(f"   📄 Results saved to: {processor.live_results_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
