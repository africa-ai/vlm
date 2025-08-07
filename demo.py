"""
Demo script for Kalenjin Dictionary Processing Framework

This script demonstrates how to use the framework to process
a Kalenjin dictionary PDF and extract linguistic data.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def demo_conversion():
    """Demo: Convert PDF to images"""
    print("=== Demo: PDF to Images Conversion ===")
    
    from scripts.pdf_to_images import PDFToImageConverter
    
    # Initialize converter with high DPI for better OCR
    converter = PDFToImageConverter(dpi=300, image_format='PNG')
    
    pdf_path = "kalenjin_dictionary.pdf"
    output_dir = "./demo_images"
    
    if not os.path.exists(pdf_path):
        print(f"⚠️  PDF file {pdf_path} not found. Please add your dictionary PDF.")
        return []
    
    try:
        # Convert PDF to images
        print(f"📄 Converting {pdf_path} to high-resolution images...")
        image_paths = converter.convert_pdf_to_images(pdf_path, output_dir)
        
        print(f"✅ Successfully converted {len(image_paths)} pages")
        for i, path in enumerate(image_paths[:3], 1):  # Show first 3
            print(f"   📸 Page {i}: {path}")
        
        if len(image_paths) > 3:
            print(f"   ... and {len(image_paths) - 3} more pages")
        
        return image_paths
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return []

def demo_vlm_processing(image_paths):
    """Demo: Process images with VLM"""
    if not image_paths:
        print("⚠️  No images to process. Run PDF conversion first.")
        return []
    
    print("\n=== Demo: VLM Processing ===")
    
    try:
        from llm.main import VLMProcessor
        from llm.config import VLMConfig
        
        # Configure VLM with Cosmos-Reason1-7B
        config = VLMConfig(
            model_name="nvidia/Cosmos-Reason1-7B",
            device="cuda" if os.environ.get("CUDA_AVAILABLE", "false").lower() == "true" else "cpu",
            batch_size=1,  # Conservative for demo
            max_new_tokens=2048,  # Higher for reasoning model
            temperature=0.05,     # Very low for precise extraction
            output_dir="./demo_results"
        )
        
        print(f"🤖 Initializing Cosmos-Reason1-7B: {config.model_name}")
        print(f"💻 Using device: {config.device}")
        print("🧠 This model uses advanced reasoning for accurate extraction")
        
        print(f"🤖 Initializing VLM: {config.model_name}")
        print(f"💻 Using device: {config.device}")
        
        # Initialize processor
        processor = VLMProcessor(config)
        
        print("📥 Loading model (this may take a while on first run)...")
        processor.load_model()
        
        # Process first few images for demo
        demo_images = image_paths[:2]  # Just 2 images for demo
        print(f"🔍 Processing {len(demo_images)} images...")
        
        results = processor.batch_process_images([str(p) for p in demo_images])
        
        # Save results
        processor.save_results(results, "demo_extraction")
        
        # Print summary
        successful = sum(1 for r in results if r['status'] == 'success')
        total_entries = sum(len(r['entries']) for r in results if r['status'] == 'success')
        
        print(f"✅ Processing completed!")
        print(f"   📊 Successfully processed: {successful}/{len(results)} images")
        print(f"   📝 Total entries extracted: {total_entries}")
        
        # Show sample entries
        print("\n📋 Sample extracted entries:")
        for result in results[:1]:  # Show first result
            if result['status'] == 'success' and result['entries']:
                for entry in result['entries'][:3]:  # Show first 3 entries
                    grapheme = entry.get('grapheme', 'N/A')
                    ipa = entry.get('ipa', 'N/A')
                    meaning = entry.get('english_meaning', 'N/A')
                    print(f"   🔤 {grapheme} → {ipa} → '{meaning}'")
                break
        
        return results
        
    except ImportError as e:
        print(f"❌ VLM dependencies not available: {e}")
        print("💡 Install with: pip install torch transformers accelerate")
        return []
    except Exception as e:
        print(f"❌ VLM processing failed: {e}")
        return []

def demo_analysis(results):
    """Demo: Analyze results"""
    if not results:
        print("⚠️  No results to analyze.")
        return
    
    print("\n=== Demo: Results Analysis ===")
    
    from utils import calculate_extraction_stats, export_for_analysis
    
    # Calculate statistics
    stats = calculate_extraction_stats(results)
    
    print(f"📈 Extraction Statistics:")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    print(f"   Entries per image: {stats['entries_per_image']:.1f}")
    print(f"   Average confidence: {stats['average_confidence']:.2f}")
    print(f"   Grapheme coverage: {stats['grapheme_coverage']:.1%}")
    print(f"   IPA coverage: {stats['ipa_coverage']:.1%}")
    print(f"   Meaning coverage: {stats['meaning_coverage']:.1%}")
    
    # Export for analysis
    try:
        export_for_analysis(results, "./demo_analysis")
        print("📁 Analysis files exported to ./demo_analysis/")
    except Exception as e:
        print(f"⚠️  Analysis export failed: {e}")

def main():
    """Main demo function"""
    print("🌟 Kalenjin Dictionary Processing Framework Demo\n")
    
    # Check requirements
    missing_deps = []
    try:
        import fitz  # PyMuPDF
    except ImportError:
        missing_deps.append("PyMuPDF")
    
    try:
        from PIL import Image
    except ImportError:
        missing_deps.append("Pillow")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("💡 Install with: pip install -r requirements.txt")
        return
    
    # Demo pipeline
    try:
        # Step 1: PDF to Images
        image_paths = demo_conversion()
        
        # Step 2: VLM Processing (optional, requires more setup)
        if input("\n🤖 Run VLM processing demo? (y/N): ").lower().startswith('y'):
            results = demo_vlm_processing(image_paths)
            
            # Step 3: Analysis
            if results:
                demo_analysis(results)
        else:
            print("⏭️  Skipping VLM processing (requires model download)")
        
        print("\n🎉 Demo completed!")
        print("💡 To run the full pipeline, use: python main.py pipeline kalenjin_dictionary.pdf")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")

if __name__ == "__main__":
    main()
