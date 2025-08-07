"""
Utility functions and helpers for the dictionary processing framework
"""

import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Union
import re

logger = logging.getLogger(__name__)

def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate JSON data against a schema
    
    Args:
        data: Data to validate
        schema: JSON schema
        
    Returns:
        True if valid, False otherwise
    """
    try:
        import jsonschema
        jsonschema.validate(data, schema)
        return True
    except ImportError:
        logger.warning("jsonschema not installed, skipping validation")
        return True
    except jsonschema.ValidationError as e:
        logger.error(f"Schema validation failed: {e}")
        return False

def merge_results(results_files: List[str], output_file: str) -> None:
    """
    Merge multiple result files into one
    
    Args:
        results_files: List of paths to result JSON files
        output_file: Path for merged output file
    """
    all_results = []
    
    for file_path in results_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
                all_results.extend(results)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
    
    # Save merged results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Merged {len(results_files)} files into {output_file}")

def convert_json_to_csv(json_file: str, csv_file: str) -> None:
    """
    Convert JSON results to CSV format
    
    Args:
        json_file: Path to JSON file
        csv_file: Path for CSV output
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'image_path', 'grapheme', 'ipa', 'english_meaning', 
            'part_of_speech', 'context', 'confidence_score',
            'status', 'timestamp'
        ])
        
        # Write data
        for result in results:
            if result.get('status') == 'success':
                for entry in result.get('entries', []):
                    writer.writerow([
                        result.get('image_path', ''),
                        entry.get('grapheme', ''),
                        entry.get('ipa', ''),
                        entry.get('english_meaning', ''),
                        entry.get('part_of_speech', ''),
                        entry.get('context', ''),
                        entry.get('confidence_score', ''),
                        result.get('status', ''),
                        result.get('timestamp', '')
                    ])

def clean_extracted_text(text: str) -> str:
    """
    Clean and normalize extracted text
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.strip().split())
    
    # Remove common OCR artifacts
    text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\/\[\]]+', '', text)
    
    # Fix common OCR mistakes (customize as needed)
    replacements = {
        'rn': 'm',  # Common OCR error
        'vv': 'w',
        '0': 'o',   # Only in specific contexts
        '1': 'l',   # Only in specific contexts
    }
    
    # Apply replacements cautiously
    for old, new in replacements.items():
        # Only replace if it makes sense in context
        pass  # Implement context-aware replacements
    
    return text

def calculate_extraction_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics for extraction results
    
    Args:
        results: List of extraction results
        
    Returns:
        Dictionary with statistics
    """
    total_images = len(results)
    successful_images = sum(1 for r in results if r.get('status') == 'success')
    total_entries = sum(len(r.get('entries', [])) for r in results)
    
    # Calculate confidence statistics
    all_confidences = []
    for result in results:
        if result.get('status') == 'success':
            for entry in result.get('entries', []):
                if 'confidence_score' in entry:
                    all_confidences.append(entry['confidence_score'])
    
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    
    # Count entries by type
    grapheme_count = 0
    ipa_count = 0
    meaning_count = 0
    
    for result in results:
        if result.get('status') == 'success':
            for entry in result.get('entries', []):
                if entry.get('grapheme'):
                    grapheme_count += 1
                if entry.get('ipa'):
                    ipa_count += 1
                if entry.get('english_meaning'):
                    meaning_count += 1
    
    return {
        'total_images': total_images,
        'successful_images': successful_images,
        'success_rate': successful_images / total_images if total_images > 0 else 0,
        'total_entries': total_entries,
        'entries_per_image': total_entries / successful_images if successful_images > 0 else 0,
        'average_confidence': avg_confidence,
        'grapheme_coverage': grapheme_count / total_entries if total_entries > 0 else 0,
        'ipa_coverage': ipa_count / total_entries if total_entries > 0 else 0,
        'meaning_coverage': meaning_count / total_entries if total_entries > 0 else 0,
    }

def export_for_analysis(results: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Export results in various formats for analysis
    
    Args:
        results: Extraction results
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Export statistics
    stats = calculate_extraction_stats(results)
    with open(output_path / "statistics.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    # Export unique graphemes
    unique_graphemes = set()
    for result in results:
        if result.get('status') == 'success':
            for entry in result.get('entries', []):
                if entry.get('grapheme'):
                    unique_graphemes.add(entry['grapheme'])
    
    with open(output_path / "unique_graphemes.txt", 'w', encoding='utf-8') as f:
        for grapheme in sorted(unique_graphemes):
            f.write(f"{grapheme}\n")
    
    # Export entries with missing IPA
    missing_ipa = []
    for result in results:
        if result.get('status') == 'success':
            for entry in result.get('entries', []):
                if entry.get('grapheme') and not entry.get('ipa'):
                    missing_ipa.append(entry['grapheme'])
    
    with open(output_path / "missing_ipa.txt", 'w', encoding='utf-8') as f:
        for grapheme in sorted(set(missing_ipa)):
            f.write(f"{grapheme}\n")
    
    # Export entries with missing meanings
    missing_meaning = []
    for result in results:
        if result.get('status') == 'success':
            for entry in result.get('entries', []):
                if entry.get('grapheme') and not entry.get('english_meaning'):
                    missing_meaning.append(entry['grapheme'])
    
    with open(output_path / "missing_meanings.txt", 'w', encoding='utf-8') as f:
        for grapheme in sorted(set(missing_meaning)):
            f.write(f"{grapheme}\n")
    
    logger.info(f"Analysis files exported to {output_dir}")

def validate_model_availability(model_name: str) -> bool:
    """
    Check if a model is available from Hugging Face
    
    Args:
        model_name: Name of the model
        
    Returns:
        True if available, False otherwise
    """
    try:
        from huggingface_hub import model_info
        model_info(model_name)
        return True
    except Exception as e:
        logger.error(f"Model {model_name} not available: {e}")
        return False

def estimate_processing_time(num_images: int, batch_size: int = 4) -> Dict[str, float]:
    """
    Estimate processing time based on image count and settings
    
    Args:
        num_images: Number of images to process
        batch_size: Batch size for processing
        
    Returns:
        Time estimates in minutes
    """
    # These are rough estimates - adjust based on your hardware
    time_per_image = {
        'cpu': 30.0,      # seconds per image on CPU
        'gpu': 5.0,       # seconds per image on GPU
        'gpu_optimized': 2.0  # with flash attention, etc.
    }
    
    num_batches = (num_images + batch_size - 1) // batch_size
    
    estimates = {}
    for device, time_per_img in time_per_image.items():
        total_seconds = num_images * time_per_img
        estimates[device] = total_seconds / 60  # Convert to minutes
    
    return estimates

if __name__ == "__main__":
    # Example usage
    print("Dictionary processing utilities")
    print("Import this module to use the utility functions")
