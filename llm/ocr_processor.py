"""
OCR + LLM Processor for Dictionary Extraction
Simple, clean pipeline: Image → OCR → LLM → JSON
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pytesseract
from PIL import Image

from llm.parser.main import DictionaryParser
from llm.config import VLMConfig
from vllm_server.client import SyncVLLMClient

logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    Simple OCR + LLM processor for dictionary extraction
    Much faster and more reliable than vision models
    """
    
    def __init__(self, config: VLMConfig, server_url: str = "http://localhost:8000"):
        self.config = config
        self.server_url = server_url
        self.client = SyncVLLMClient(server_url, config.api_key if hasattr(config, 'api_key') else None)
        self.parser = DictionaryParser()
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup continuous saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.live_results_file = self.output_dir / f"ocr_results_{timestamp}.jsonl"
        self.live_entries_file = self.output_dir / f"ocr_entries_{timestamp}.json"
        self.processed_count = 0
        self.all_entries = []
        
        logger.info(f"OCRProcessor initialized with server: {server_url}")
        logger.info(f"Live results will be saved to: {self.live_results_file}")
    
    def check_server_connection(self) -> bool:
        """
        Check if vLLM server is accessible
        
        Returns:
            True if server is accessible, False otherwise
        """
        try:
            # Try to get health status
            response = self.client.health_check()
            if response:
                logger.info("vLLM server is healthy")
                return True
            else:
                logger.warning("vLLM server health check failed")
                return False
        except Exception as e:
            logger.error(f"Cannot connect to vLLM server at {self.server_url}: {e}")
            return False
    
    def extract_text_from_image(self, image_path: Union[str, Path]) -> str:
        """
        Extract raw text from image using OCR - copies exactly what it sees
        
        Args:
            image_path: Path to image file
            
        Returns:
            Raw text extracted from image (exactly as seen)
        """
        try:
            logger.info(f"Extracting text from: {image_path}")
            
            # Load image
            img = Image.open(image_path)
            
            # OCR configuration for exact text extraction (no interpretation)
            # PSM 6: Uniform block of text (good for dictionary pages)
            # OEM 3: Default OCR Engine Mode (best accuracy)
            # No character whitelist - extract everything as seen
            custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
            
            # Extract text exactly as OCR sees it
            raw_text = pytesseract.image_to_string(img, config=custom_config)
            
            logger.info(f"Extracted {len(raw_text)} characters from {Path(image_path).name}")
            logger.info(f"Text preview: {raw_text[:200]}...")
            
            # Return raw text without any cleaning or interpretation
            return raw_text
            
        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {e}")
            return ""
    
    def process_text_with_llm(self, raw_text: str, image_path: str = "") -> Dict[str, Any]:
        """
        Process OCR text with LLM to extract dictionary entries
        
        Args:
            raw_text: Raw text from OCR
            image_path: Original image path (for reference)
            
        Returns:
            Processing result with entries
        """
        if not raw_text.strip():
            return {
                "image_path": image_path,
                "entries": [],
                "error": "No text extracted from OCR",
                "status": "error"
            }
        
        try:
            logger.info("Sending OCR text to LLM for parsing...")
            
            # Use our clean parser to process the OCR text
            entries = self.parser.parse_ocr_text(raw_text, self.client)
            
            logger.info(f"LLM extracted {len(entries)} entries from OCR text")
            
            return {
                "image_path": image_path,
                "entries": entries,
                "raw_text": raw_text,
                "method": "OCR + LLM",
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
                
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            return {
                "image_path": image_path,
                "entries": [],
                "error": str(e),
                "status": "error"
            }
    
    def process_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Complete pipeline: Image → OCR → LLM → JSON
        
        Args:
            image_path: Path to dictionary image
            
        Returns:
            Processing result with extracted entries
        """
        logger.info(f"Processing {Path(image_path).name} with OCR + LLM pipeline")
        
        # Step 1: OCR extraction
        raw_text = self.extract_text_from_image(image_path)
        
        if not raw_text:
            return {
                "image_path": str(image_path),
                "entries": [],
                "error": "OCR failed to extract text",
                "status": "error"
            }
        
        # Step 2: LLM processing
        result = self.process_text_with_llm(raw_text, str(image_path))
        
        return result
    
    def process_images(self, image_paths: List[Union[str, Path]], 
                      output_dir: Optional[str] = None,
                      max_workers: int = 4, 
                      batch_size: int = 8) -> List[Dict[str, Any]]:
        """
        Process multiple images with OCR + LLM pipeline (main interface)
        
        Args:
            image_paths: List of image file paths
            output_dir: Output directory for results (optional)
            max_workers: Number of concurrent workers (not used in current implementation)
            batch_size: Batch size for processing (not used in current implementation)
            
        Returns:
            List of processing results
        """
        # Set output directory if provided
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Update live result files for new output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.live_results_file = self.output_dir / f"ocr_results_{timestamp}.jsonl"
            self.live_entries_file = self.output_dir / f"ocr_entries_{timestamp}.json"
            logger.info(f"Updated live results location: {self.live_results_file}")
        
        # Use batch processing method
        return self.batch_process_images(image_paths)
    
    def save_live_result(self, result: Dict[str, Any]) -> None:
        """Save individual processing result immediately"""
        try:
            # Append to JSONL file
            with open(self.live_results_file, 'a', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
            
            # Update running totals
            if result.get("status") == "success":
                entries = result.get("entries", [])
                self.all_entries.extend(entries)
                
                # Update live entries file
                with open(self.live_entries_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "total_entries": len(self.all_entries),
                        "processed_images": self.processed_count,
                        "last_updated": datetime.now().isoformat(),
                        "processing_method": "OCR + LLM",
                        "entries": self.all_entries
                    }, f, indent=2, ensure_ascii=False)
                
                # Log samples
                if entries:
                    logger.info(f"📝 Sample entries from {Path(result['image_path']).name}:")
                    for i, entry in enumerate(entries[:3]):
                        kalenjin = entry.get("grapheme", "")
                        english = entry.get("english_meaning", "")
                        logger.info(f"  {i+1}. {kalenjin} → {english}")
                    if len(entries) > 3:
                        logger.info(f"  ... and {len(entries)-3} more entries")
        
        except Exception as e:
            logger.error(f"Failed to save live result: {e}")
    
    def batch_process_images(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Process multiple images with OCR + LLM pipeline
        """
        logger.info(f"Starting OCR + LLM processing of {len(image_paths)} images")
        logger.info(f"🔄 Results will be saved to: {self.live_results_file}")
        
        results = []
        
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"\n🔍 Processing image {i}/{len(image_paths)}: {Path(image_path).name}")
            
            try:
                result = self.process_image(image_path)
                results.append(result)
                self.processed_count += 1
                
                # Save immediately
                self.save_live_result(result)
                
                # Progress log
                if result.get("status") == "success":
                    entries_count = len(result.get("entries", []))
                    total_entries = len(self.all_entries)
                    logger.info(f"✅ Image {i} processed successfully")
                    logger.info(f"   📊 This image: {entries_count} entries")
                    logger.info(f"   📊 Total so far: {total_entries} entries from {self.processed_count} images")
                else:
                    logger.error(f"❌ Image {i} failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                error_result = {
                    "image_path": str(image_path),
                    "entries": [],
                    "error": f"Processing failed: {str(e)}",
                    "status": "error",
                    "timestamp": datetime.now().isoformat()
                }
                results.append(error_result)
                self.processed_count += 1
                self.save_live_result(error_result)
                logger.error(f"❌ Image {i} failed with exception: {e}")
        
        # Final summary
        successful = sum(1 for r in results if r.get("status") == "success")
        total_entries = len(self.all_entries)
        
        logger.info(f"\n🏁 OCR + LLM Processing completed!")
        logger.info(f"   ✅ Successful: {successful}/{len(image_paths)} images")
        logger.info(f"   📚 Total entries extracted: {total_entries}")
        logger.info(f"   ⚡ Method: OCR + LLM (much faster than vision models)")
        logger.info(f"   📄 Results saved to: {self.live_results_file}")
        
        return results


def create_ocr_processor(config: Optional[VLMConfig] = None, 
                        server_url: str = "http://localhost:8000") -> OCRProcessor:
    """
    Factory function to create OCR processor
    
    Args:
        config: VLM configuration
        server_url: LLM server URL
        
    Returns:
        OCRProcessor instance
    """
    if config is None:
        from llm.config import load_config_from_env
        config = load_config_from_env()
    
    return OCRProcessor(config, server_url)


if __name__ == "__main__":
    # Test OCR processor
    from llm.config import load_config_from_env
    
    config = load_config_from_env()
    processor = OCRProcessor(config)
    
    print("🔧 Testing OCR + LLM pipeline...")
    
    # Test on single image
    test_image = "results/images/kalenjin_dictionary_page_001.png"
    if Path(test_image).exists():
        result = processor.process_image(test_image)
        print(f"✅ Processed {test_image}")
        print(f"📊 Found {len(result.get('entries', []))} entries")
        print(f"📝 Status: {result.get('status')}")
    else:
        print(f"❌ Test image not found: {test_image}")
