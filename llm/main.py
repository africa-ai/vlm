"""
Core OCR processing functions for the clean pipeline.
Handles PDF → Images → OCR → vLLM → JSON workflow.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

from .ocr_processor import OCRProcessor
from .config import load_config_from_env


logger = logging.getLogger(__name__)


def run_ocr_pipeline(
    images_dir: str,
    output_dir: str,
    max_workers: int = 4,
    batch_size: int = 8
) -> Dict[str, Any]:
    """
    Run OCR + LLM processing on a directory of images.
    
    Args:
        images_dir: Directory containing PNG images
        output_dir: Directory to save results
        max_workers: Number of concurrent workers
        batch_size: Images per batch
        
    Returns:
        Processing summary with counts and timing
    """
    logger.info(f"Starting OCR pipeline: {images_dir} → {output_dir}")
    
    # Setup paths
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if not images_path.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    
    # Find image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(images_path.glob(ext))
    
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Setup processor
    config = load_config_from_env()
    processor = OCRProcessor(config)
    
    # Check server connection
    if not processor.check_server_connection():
        raise ConnectionError("vLLM server not available. Start with: python start_vllm_server.py")
    
    # Process images
    image_paths = [str(img) for img in sorted(image_files)]
    results = processor.process_images(
        image_paths=image_paths,
        output_dir=output_dir,
        max_workers=max_workers,
        batch_size=batch_size
    )
    
    # Summary
    total_entries = sum(len(r.get('entries', [])) for r in results if r['status'] == 'success')
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    summary = {
        'total_images': len(image_files),
        'successful': successful,
        'failed': failed,
        'total_entries': total_entries,
        'output_dir': str(output_path)
    }
    
    logger.info(f"Pipeline complete: {successful}/{len(image_files)} images processed, {total_entries} entries extracted")
    return summary


def process_single_image(image_path: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Process a single image with OCR + LLM.
    
    Args:
        image_path: Path to image file
        output_dir: Optional output directory
        
    Returns:
        Processing result with entries or error
    """
    logger.info(f"Processing single image: {image_path}")
    
    # Setup processor
    config = load_config_from_env()
    processor = OCRProcessor(config)
    
    # Check server
    if not processor.check_server_connection():
        raise ConnectionError("vLLM server not available")
    
    # Process
    results = processor.process_images([image_path], output_dir=output_dir)
    
    if results and results[0]['status'] == 'success':
        entries = results[0]['entries']
        logger.info(f"Extracted {len(entries)} entries from {Path(image_path).name}")
        return results[0]
    else:
        error = results[0]['error'] if results else 'Unknown error'
        logger.error(f"Failed to process {image_path}: {error}")
        return {'status': 'error', 'error': error}


def get_processing_stats(results_dir: str) -> Dict[str, Any]:
    """
    Get statistics from processing results directory.
    
    Args:
        results_dir: Directory containing result files
        
    Returns:
        Statistics summary
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        return {'error': 'Results directory not found'}
    
    # Count result files
    json_files = list(results_path.glob('*.json'))
    jsonl_files = list(results_path.glob('*.jsonl'))
    live_files = list(results_path.glob('live_*.json*'))
    
    stats = {
        'results_directory': str(results_path),
        'json_files': len(json_files),
        'jsonl_files': len(jsonl_files),
        'live_files': len(live_files),
        'total_files': len(list(results_path.iterdir()))
    }
    
    return stats
