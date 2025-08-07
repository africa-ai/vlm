"""
Main Visual Language Model Processor for Kalenjin Dictionary
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import csv
from datetime import datetime

try:
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from PIL import Image
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .config import VLMConfig, load_config_from_env
try:
    from .parser.main import DictionaryParser
except ImportError:
    # Fallback if parser module is not available
    DictionaryParser = None

# Setup logging
logger = logging.getLogger(__name__)

class VLMProcessor:
    """Main processor for Visual Language Model operations"""
    
    def __init__(self, config: Optional[VLMConfig] = None):
        """
        Initialize VLM Processor
        
        Args:
            config: VLMConfig instance or None to use default/env config
        """
        self.config = config or load_config_from_env()
        self.model = None
        self.processor = None
        if DictionaryParser:
            self.parser = DictionaryParser(self.config)
        else:
            logger.warning("DictionaryParser not available")
            self.parser = None
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=self.config.log_file
        )
        
    def load_model(self):
        """Load the QWEN 2.5 VL model"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required. Install with: "
                "pip install torch transformers pillow"
            )
        
        logger.info(f"Loading model: {self.config.model_name}")
        
        try:
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
                cache_dir=self.config.model_cache_dir
            )
            
            # Load model
            model_kwargs = self.config.get_model_kwargs()
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if model_kwargs.get("device_map") is None:
                self.model = self.model.to(self.config.device)
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def process_image(self, image_path: str, prompt: str) -> str:
        """
        Process a single image with the VLM
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt for the model
            
        Returns:
            Generated text response
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Load and process image
            image = Image.open(image_path).convert(self.config.image_format)
            
            # Prepare inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.config.get_generation_kwargs()
                )
            
            # Decode response
            response = self.processor.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Remove the input prompt from response
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            raise
    
    def batch_process_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple images in batches
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of processing results
        """
        results = []
        
        for i in range(0, len(image_paths), self.config.batch_size):
            batch_paths = image_paths[i:i + self.config.batch_size]
            
            logger.info(f"Processing batch {i//self.config.batch_size + 1} "
                       f"({len(batch_paths)} images)")
            
            batch_results = []
            for image_path in batch_paths:
                try:
                    # Parse dictionary entries from image
                    entries = self.parser.parse_image(image_path)
                    batch_results.append({
                        "image_path": image_path,
                        "entries": entries,
                        "status": "success",
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.error(f"Failed to process {image_path}: {e}")
                    batch_results.append({
                        "image_path": image_path,
                        "entries": [],
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
            
            results.extend(batch_results)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_prefix: str = "dictionary_extraction"):
        """
        Save processing results to JSON and CSV files
        
        Args:
            results: List of processing results
            output_prefix: Prefix for output filenames
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        if self.config.save_json:
            json_path = Path(self.config.output_dir) / f"{output_prefix}_{timestamp}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to JSON: {json_path}")
        
        # Save CSV
        if self.config.save_csv:
            csv_path = Path(self.config.output_dir) / f"{output_prefix}_{timestamp}.csv"
            self._save_csv(results, csv_path)
            logger.info(f"Results saved to CSV: {csv_path}")
    
    def _save_csv(self, results: List[Dict[str, Any]], csv_path: Path):
        """Save results to CSV format"""
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'image_path', 'grapheme', 'ipa', 'english_meaning', 
                'status', 'timestamp'
            ])
            
            # Write data
            for result in results:
                if result['status'] == 'success':
                    for entry in result['entries']:
                        writer.writerow([
                            result['image_path'],
                            entry.get('grapheme', ''),
                            entry.get('ipa', ''),
                            entry.get('english_meaning', ''),
                            result['status'],
                            result['timestamp']
                        ])
                else:
                    writer.writerow([
                        result['image_path'], '', '', '',
                        result['status'], result['timestamp']
                    ])

def main():
    """Main entry point for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Kalenjin dictionary with VLM")
    parser.add_argument("image_dir", help="Directory containing dictionary images")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("--model", help="Model name override")
    parser.add_argument("--batch-size", type=int, help="Batch size for processing")
    parser.add_argument("--device", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Create config with overrides
    config = load_config_from_env()
    if args.output:
        config.output_dir = args.output
    if args.model:
        config.model_name = args.model
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.device:
        config.device = args.device
    
    # Initialize processor
    processor = VLMProcessor(config)
    processor.load_model()
    
    # Get image paths
    image_dir = Path(args.image_dir)
    image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    
    if not image_paths:
        logger.error(f"No images found in {image_dir}")
        return
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Process images
    results = processor.batch_process_images([str(p) for p in image_paths])
    
    # Save results
    processor.save_results(results)
    
    # Print summary
    successful = sum(1 for r in results if r['status'] == 'success')
    total_entries = sum(len(r['entries']) for r in results if r['status'] == 'success')
    
    print(f"Processing completed:")
    print(f"  Successfully processed: {successful}/{len(results)} images")
    print(f"  Total dictionary entries extracted: {total_entries}")

if __name__ == "__main__":
    main()
