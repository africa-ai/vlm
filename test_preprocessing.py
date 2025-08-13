#!/usr/bin/env python
"""
Test image preprocessing to see what the model actually sees
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm.image_preprocessor import optimize_image_for_vlm

def test_preprocessing():
    """Test preprocessing and save the result for visual inspection"""
    
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
    print(f"🔍 Testing preprocessing on: {image_path.name}")
    print(f"📄 Original file: {image_path}")
    
    try:
        # Load original for comparison
        from PIL import Image
        original = Image.open(image_path)
        orig_width, orig_height = original.size
        orig_pixels = orig_width * orig_height
        orig_tokens = int(orig_pixels * 0.65)
        
        print(f"📊 Original: {orig_width}x{orig_height} ({orig_pixels:,} pixels, ~{orig_tokens:,} tokens)")
        
        # Process image with current settings
        optimized_images = optimize_image_for_vlm(str(image_path), target_tokens=100000)
        
        print(f"✅ Preprocessing complete!")
        print(f"📊 Generated {len(optimized_images)} image part(s)")
        
        # Save the preprocessed images for inspection
        output_dir = Path("debug_output")
        output_dir.mkdir(exist_ok=True)
        
        total_tokens = 0
        for i, img in enumerate(optimized_images):
            output_path = output_dir / f"preprocessed_part_{i+1}.png"
            img.save(output_path)
            width, height = img.size
            pixels = width * height
            estimated_tokens = int(pixels * 0.65)  # Same estimation as used internally
            total_tokens += estimated_tokens
            
            print(f"   Part {i+1}: {width}x{height} pixels ({pixels:,} total)")
            print(f"           Estimated tokens: {estimated_tokens:,}")
            print(f"           Saved to: {output_path}")
        
        # Calculate reduction
        reduction_factor = orig_tokens / total_tokens if total_tokens > 0 else 0
        print(f"\n📈 Token reduction: {orig_tokens:,} → {total_tokens:,} ({reduction_factor:.1f}x smaller)")
        print(f"\n🔍 You can now visually inspect the preprocessed images in {output_dir}")
        print("   Check if the text is readable and the full page is captured")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_preprocessing()

def main():
    """Main test function"""
    
    if len(sys.argv) != 2:
        print("Usage: python test_preprocessing.py <image_path>")
        print("Example: python test_preprocessing.py kalenjin_dictionary.pdf.page_1.png")
        return
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Run tests
    try:
        result = optimize_image_for_vlm(image_path, target_tokens=100000)
        if result:
            print(f"\n✅ Successfully optimized image into {len(result)} parts")
            print("Ready for VLM processing with reduced token usage!")
        else:
            print("\n❌ Image preprocessing failed")
    except Exception as e:
        print(f"\n❌ Error during image preprocessing: {e}")

if __name__ == "__main__":
    main()
