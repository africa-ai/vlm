#!/usr/bin/env python
"""
Test Image Preprocessing for VLM Processing
Tests the image optimization approach to reduce token usage
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm.image_preprocessor import optimize_image_for_vlm, ImagePreprocessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_image_preprocessing(image_path: str):
    """Test image preprocessing on a sample image"""
    
    logger.info(f"Testing image preprocessing on: {image_path}")
    
    # Test 1: Basic preprocessing
    preprocessor = ImagePreprocessor(max_width=1024, max_height=1024)
    
    try:
        # Load original image
        from PIL import Image
        original_image = Image.open(image_path)
        original_size = original_image.size
        
        # Estimate original token count
        original_tokens = preprocessor.estimate_token_count(original_image)
        
        logger.info(f"Original image: {original_size}, estimated tokens: {original_tokens}")
        
        # Test preprocessing
        processed_image = preprocessor.preprocess_image(image_path)
        processed_tokens = preprocessor.estimate_token_count(processed_image)
        
        logger.info(f"Processed image: {processed_image.size}, estimated tokens: {processed_tokens}")
        logger.info(f"Token reduction: {original_tokens / processed_tokens:.1f}x")
        
        # Test splitting if needed
        if processed_tokens > 30000:
            splits = preprocessor.split_image_vertically(processed_image, 4)
            logger.info(f"Split into {len(splits)} parts:")
            for i, split in enumerate(splits):
                split_tokens = preprocessor.estimate_token_count(split)
                logger.info(f"  Part {i+1}: {split.size}, estimated tokens: {split_tokens}")
        
        # Test full optimization pipeline
        logger.info("\nTesting full optimization pipeline:")
        optimized_images = optimize_image_for_vlm(image_path, target_tokens=30000)
        
        total_optimized_tokens = sum(preprocessor.estimate_token_count(img) for img in optimized_images)
        logger.info(f"Optimized to {len(optimized_images)} parts, total tokens: {total_optimized_tokens}")
        logger.info(f"Overall token reduction: {original_tokens / total_optimized_tokens:.1f}x")
        
        return optimized_images
        
    except Exception as e:
        logger.error(f"Error testing preprocessing: {e}")
        return None

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
    result = test_image_preprocessing(image_path)
    
    if result:
        print(f"\n✅ Successfully optimized image into {len(result)} parts")
        print("Ready for VLM processing with reduced token usage!")
    else:
        print("\n❌ Image preprocessing failed")

if __name__ == "__main__":
    main()
