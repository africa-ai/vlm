#!/usr/bin/env python
"""
Test the new OCR + LLM approach
Clean pipeline: Image/PDF → OCR → vLLM → JSON
"""

import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ocr_llm_processor import OCRLLMProcessor


def test_single_image():
    """Test OCR + LLM on a single image"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Find first image to test
    images_dir = Path("results/images")
    if not images_dir.exists():
        print("❌ No images directory found")
        print("Run pdf extraction first or provide image files")
        return
    
    image_files = list(images_dir.glob("*.png"))
    if not image_files:
        print("❌ No PNG images found in results/images/")
        return
    
    image_path = image_files[0]
    print(f"🔍 Testing OCR + LLM pipeline on: {image_path.name}")
    
    # Initialize processor
    processor = OCRLLMProcessor()
    
    # Check server connection
    if not processor.client.health_check():
        print("❌ vLLM server is not running!")
        print("Start the server first with: python start_vllm_server.py")
        return
    
    print("✅ vLLM server is running")
    
    # Process single page
    result = processor.process_single_page(str(image_path), 1)
    
    # Display results
    print("\n" + "="*60)
    print("📊 RESULTS SUMMARY")
    print("="*60)
    
    if result["status"] == "success":
        entries = result["entries"]
        print(f"✅ Success! Extracted {len(entries)} entries")
        
        print(f"\n📝 Sample entries:")
        for i, entry in enumerate(entries[:5], 1):
            grapheme = entry.get('grapheme', 'Unknown')
            meaning = entry.get('english_meaning', 'No meaning')
            pos = entry.get('part_of_speech', 'Unknown')
            print(f"   {i}. {grapheme} ({pos}) - {meaning[:50]}...")
        
        if len(entries) > 5:
            print(f"   ... and {len(entries)-5} more entries")
            
        # Show some OCR text
        ocr_sample = result.get("raw_ocr_text", "")[:200]
        print(f"\n📄 OCR Text Sample:")
        print(f"   {ocr_sample}...")
        
    else:
        print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    print("\n🎯 This approach is much simpler than the image-based VLM!")
    print("   • No image preprocessing headaches")
    print("   • No token limit issues") 
    print("   • Direct text processing")
    print("   • Much faster and more reliable")


def test_pdf():
    """Test OCR + LLM on PDF"""
    if len(sys.argv) < 2:
        print("Usage for PDF: python test_ocr_llm.py <pdf_path> [max_pages]")
        return
    
    pdf_path = sys.argv[1]
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 3  # Limit to 3 pages for testing
    
    if not Path(pdf_path).exists():
        print(f"❌ PDF not found: {pdf_path}")
        return
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    print(f"🔍 Testing OCR + LLM pipeline on PDF: {pdf_path}")
    print(f"📄 Processing first {max_pages} pages")
    
    # Initialize processor
    processor = OCRLLMProcessor()
    
    # Check server connection
    if not processor.client.health_check():
        print("❌ vLLM server is not running!")
        print("Start the server first with: python start_vllm_server.py")
        return
    
    print("✅ vLLM server is running")
    
    # Process PDF
    results = processor.process_pdf(pdf_path, max_pages)
    
    # Save results
    output_file = processor.save_results(results, "test_ocr_llm_results.json")
    
    # Display summary
    successful = sum(1 for r in results if r["status"] == "success")
    total_entries = sum(len(r["entries"]) for r in results if r["status"] == "success")
    
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY")
    print("="*60)
    print(f"✅ Pages processed: {successful}/{len(results)}")
    print(f"📚 Total entries extracted: {total_entries}")
    print(f"💾 Results saved to: {output_file}")
    
    if total_entries > 0:
        print(f"\n🎯 Success! The OCR + LLM approach extracted {total_entries} entries")
        print("   This is much more reliable than the complex VLM image processing!")


def main():
    """Main test function"""
    
    if len(sys.argv) > 1 and sys.argv[1].endswith('.pdf'):
        # PDF test
        test_pdf()
    else:
        # Single image test
        test_single_image()


if __name__ == "__main__":
    main()
