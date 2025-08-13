"""
Image Preprocessing Module for VLM Processing
Optimizes images to reduce token count while maintaining quality for dictionary analysis.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Preprocesses images for optimal VLM processing by:
    1. Resizing to reasonable dimensions
    2. Enhancing contrast and sharpness for text readability
    3. Converting to optimal format
    4. Reducing file size while maintaining quality
    """
    
    def __init__(self, 
                 max_width: int = 800,  # Increased for better text readability
                 max_height: int = 1000,  # Increased for dictionary pages
                 quality: int = 85,  # Higher quality
                 enhance_text: bool = True):
        """
        Initialize image preprocessor with balanced defaults for readability vs. token count.
        
        Args:
            max_width: Maximum image width in pixels (balanced for readability)
            max_height: Maximum image height in pixels (balanced for readability)
            quality: JPEG quality (1-100)
            enhance_text: Whether to enhance text readability
        """
        self.max_width = max_width
        self.max_height = max_height
        self.quality = quality
        self.enhance_text = enhance_text
        
    def resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: PIL Image object
            
        Returns:
            Resized PIL Image
        """
        width, height = image.size
        
        # Calculate scaling factor
        width_scale = self.max_width / width
        height_scale = self.max_height / height
        scale = min(width_scale, height_scale, 1.0)  # Don't upscale
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            
        return image
    
    def enhance_for_text(self, image: Image.Image) -> Image.Image:
        """
        Enhance image for better text recognition.
        
        Args:
            image: PIL Image object
            
        Returns:
            Enhanced PIL Image
        """
        if not self.enhance_text:
            return image
            
        try:
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
        except Exception as e:
            logger.warning(f"Enhancement failed: {e}")
            
        return image
    
    def preprocess_image(self, image_path: str) -> Image.Image:
        """
        Complete image preprocessing pipeline.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed PIL Image
        """
        try:
            # Load image
            image = Image.open(image_path)
            original_size = image.size
            
            # Resize image
            image = self.resize_image(image)
            
            # Enhance for text recognition
            image = self.enhance_for_text(image)
            
            new_size = image.size
            size_reduction = (original_size[0] * original_size[1]) / (new_size[0] * new_size[1])
            
            logger.info(f"Preprocessed {image_path}: {original_size} -> {new_size} "
                       f"(reduction: {size_reduction:.1f}x)")
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to preprocess image {image_path}: {e}")
            # Return original image as fallback
            return Image.open(image_path)
    
    def split_image_vertically(self, image: Image.Image, num_splits: int = 2) -> list[Image.Image]:
        """
        Split image vertically into multiple parts for processing.
        Useful for dictionary pages with multiple columns.
        
        Args:
            image: PIL Image object
            num_splits: Number of vertical splits
            
        Returns:
            List of PIL Image objects
        """
        width, height = image.size
        split_height = height // num_splits
        
        splits = []
        for i in range(num_splits):
            top = i * split_height
            bottom = (i + 1) * split_height if i < num_splits - 1 else height
            
            split_image = image.crop((0, top, width, bottom))
            splits.append(split_image)
            
        logger.info(f"Split image into {len(splits)} parts of size {splits[0].size}")
        return splits
    
    def estimate_token_count(self, image: Image.Image) -> int:
        """
        Estimate approximate token count for an image based on actual VLM behavior.
        Based on observed data: 364x600 (218K pixels) = 136K actual tokens.
        
        Args:
            image: PIL Image object
            
        Returns:
            Estimated token count (realistic for VLM models)
        """
        width, height = image.size
        pixels = width * height
        
        # Based on actual observations: 
        # 364x600 = 218,400 pixels generated 136,019 tokens
        # This gives us ~0.62 tokens per pixel
        tokens_per_pixel = 0.65  # Slightly conservative
        
        estimated_tokens = int(pixels * tokens_per_pixel)
        
        # Add overhead for text tokens in the prompt
        estimated_tokens += 2000
        
        return estimated_tokens

def optimize_image_for_vlm(image_path: str, 
                          target_tokens: int = 80000,  # Lower target to encourage splits
                          max_splits: int = 4,  # Allow more splits
                          force_split: bool = True) -> list[Image.Image]:  # Force splitting for dictionary pages
    """
    Optimize image for VLM processing with balanced token control and readability.
    For dictionary pages, force splitting for better systematic processing.
    
    Args:
        image_path: Path to input image
        target_tokens: Target token count per image (lower to encourage splits)
        max_splits: Maximum number of splits to try
        force_split: Whether to force splitting even if under target tokens
        
    Returns:
        List of optimized PIL Images ready for VLM processing
    """
    # Use balanced preprocessing settings (no override)
    preprocessor = ImagePreprocessor()  # Use default balanced settings
    
    # Load and preprocess image
    image = preprocessor.preprocess_image(image_path)
    
    # Estimate token count with accurate calculation
    estimated_tokens = preprocessor.estimate_token_count(image)
    
    logger.info(f"Estimated tokens for {image_path}: {estimated_tokens}")
    
    # Force split for dictionary pages OR if over target tokens
    if force_split or estimated_tokens > target_tokens:
        # For dictionary pages, always split into at least 2 parts (top/bottom)
        # This helps the model focus on smaller sections systematically
        min_splits = 2 if force_split else 1
        splits_needed = max(min_splits, min(max_splits, (estimated_tokens // target_tokens) + 1))
        logger.info(f"Splitting image into {splits_needed} parts (forced: {force_split})")
        images = preprocessor.split_image_vertically(image, splits_needed)
    else:
        images = [image]
        logger.info("Processing as single image (no splitting needed)")
    
    # Log final token estimates and validate
    valid_images = []
    for i, img in enumerate(images):
        tokens = preprocessor.estimate_token_count(img)
        logger.info(f"Part {i+1}: {img.size}, estimated tokens: {tokens}")
        
        # Strict validation - must be under target
        if tokens <= target_tokens:
            valid_images.append(img)
        else:
            logger.warning(f"Part {i+1} still too large ({tokens} tokens), skipping")
    
    return valid_images if valid_images else [images[0]]  # Return at least one image
