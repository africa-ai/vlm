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
        
        # Setup output directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create separate folders for entries and examples
        self.entries_dir = self.output_dir / "dictionary_entries"
        self.examples_dir = self.output_dir / "example_sentences"
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup continuous saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.live_results_file = self.output_dir / f"ocr_results_{timestamp}.jsonl"
        self.live_entries_file = self.entries_dir / f"dictionary_entries_{timestamp}.json"
        self.live_examples_file = self.examples_dir / f"example_sentences_{timestamp}.json"
        
        self.processed_count = 0
        self.all_entries = []
        self.all_examples = []
        
        logger.info(f"OCRProcessor initialized with server: {server_url}")
        logger.info(f"📚 Dictionary entries will be saved to: {self.entries_dir}")
        logger.info(f"🗣️  Example sentences will be saved to: {self.examples_dir}")
        logger.info(f"📊 Live results will be saved to: {self.live_results_file}")
    
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
        Extract raw text from image using high accuracy OCR configuration
        Optimized for maximum dictionary entry detection (7 headwords vs 4)
        """
        try:
            logger.info(f"Extracting text from: {image_path}")
            
            # Configure Tesseract path for Windows
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            
            # Load image
            img = Image.open(image_path)
            
            # Convert to grayscale for better OCR performance
            if img.mode != 'L':
                img = img.convert('L')
            
            # HIGH ACCURACY configuration (best from comparison tests)
            # OEM 1: Neural LSTM engine (most accurate)
            # PSM 3: Fully automatic page segmentation (preserves dictionary layout)
            # preserve_interword_spaces: Maintains proper word spacing
            high_accuracy_config = r'--oem 1 --psm 3 -c preserve_interword_spaces=1'
            
            raw_text = pytesseract.image_to_string(img, config=high_accuracy_config)
            
            logger.info(f"Extracted {len(raw_text)} characters from {Path(image_path).name}")
            logger.info(f"Using HIGH ACCURACY config - expect ~7 headwords per page")
            logger.info(f"Text preview: {raw_text[:200]}...")
            
            return raw_text
            
        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {e}")
            return ""
    
    def _process_ocr_structure(self, raw_text: str, ocr_data: dict) -> str:
        """
        Process OCR data to better preserve dictionary structure
        
        Args:
            raw_text: Raw OCR text
            ocr_data: Detailed OCR data with bounding boxes
            
        Returns:
            Improved text with better structure
        """
        try:
            # Get text elements with their positions
            texts = ocr_data.get('text', [])
            confs = ocr_data.get('conf', [])
            lefts = ocr_data.get('left', [])
            tops = ocr_data.get('top', [])
            widths = ocr_data.get('width', [])
            heights = ocr_data.get('height', [])
            
            # Filter out low-confidence and empty text
            elements = []
            for i, text in enumerate(texts):
                if text.strip() and confs[i] > 30:  # Confidence threshold
                    elements.append({
                        'text': text,
                        'conf': confs[i],
                        'left': lefts[i],
                        'top': tops[i],
                        'width': widths[i],
                        'height': heights[i],
                        'right': lefts[i] + widths[i],
                        'bottom': tops[i] + heights[i]
                    })
            
            # Sort by position (top to bottom, left to right)
            elements.sort(key=lambda x: (x['top'], x['left']))
            
            # Reconstruct text with better spacing and line breaks
            result_lines = []
            current_line = []
            current_top = -1
            line_height_threshold = 10  # Pixels
            
            for elem in elements:
                # Check if this element is on a new line
                if current_top == -1 or abs(elem['top'] - current_top) > line_height_threshold:
                    # New line
                    if current_line:
                        result_lines.append(' '.join(current_line))
                    current_line = [elem['text']]
                    current_top = elem['top']
                else:
                    # Same line
                    current_line.append(elem['text'])
            
            # Add the last line
            if current_line:
                result_lines.append(' '.join(current_line))
            
            return '\n'.join(result_lines)
            
        except Exception as e:
            logger.debug(f"Structure processing failed: {e}")
            return raw_text
    
    def process_text_with_llm(self, raw_text: str, image_path: str = "") -> Dict[str, Any]:
        """
        Process OCR text with LLM to extract dictionary entries
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
            
            # Pre-filter OCR text to reduce token count
            cleaned_text = self._prefilter_ocr_text(raw_text)
            logger.info(f"Cleaned text length: {len(cleaned_text)} characters")
            
            # If text is very long, chunk it
            if len(cleaned_text) > 3000:  # ~750 tokens roughly
                logger.info("Text is long, using chunked processing...")
                return self._process_text_in_chunks(cleaned_text, image_path)
            
            # Single fast processing for smaller texts
            logger.info("Processing text with enhanced LLM parser...")
            result = self.parser.parse_ocr_text(cleaned_text, self.client)
            
            # Result now contains both entries and examples
            entries = result.get("entries", [])
            examples = result.get("examples", [])
            
            logger.info(f"LLM extracted {len(entries)} dictionary entries and {len(examples)} example sentences from OCR text")
            
            return {
                "image_path": image_path,
                "entries": entries,
                "examples": examples,
                "raw_text": raw_text,
                "cleaned_text": cleaned_text,
                "method": "OCR + LLM (Enhanced with Examples)",
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
                
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "image_path": image_path,
                "entries": [],
                "error": f"LLM processing failed: {str(e)}",
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
        """Save individual processing result immediately with dual extraction support"""
        try:
            # Append to main JSONL file
            with open(self.live_results_file, 'a', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
            
            # Update running totals for entries and examples
            if result.get("status") == "success":
                entries = result.get("entries", [])
                examples = result.get("examples", [])
                
                # Extend collections
                self.all_entries.extend(entries)
                self.all_examples.extend(examples)
                
                # Save dictionary entries to separate file if any found
                if entries:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    entries_file = os.path.join(self.entries_dir, f"entries_{timestamp}.json")
                    with open(entries_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "entries": entries,
                            "source_image": result["image_path"],
                            "timestamp": result["timestamp"],
                            "method": result["method"]
                        }, f, ensure_ascii=False, indent=2)
                
                # Save example sentences to separate file if any found
                if examples:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    examples_file = os.path.join(self.examples_dir, f"examples_{timestamp}.json")
                    with open(examples_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "examples": examples,
                            "source_image": result["image_path"],
                            "timestamp": result["timestamp"],
                            "method": result["method"]
                        }, f, ensure_ascii=False, indent=2)
                
                # Update comprehensive live entries file
                with open(self.live_entries_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "total_entries": len(self.all_entries),
                        "total_examples": len(self.all_examples),
                        "processed_images": self.processed_count,
                        "last_updated": datetime.now().isoformat(),
                        "processing_method": "OCR + LLM (Enhanced with Examples)",
                        "entries": self.all_entries,
                        "examples": self.all_examples
                    }, f, indent=2, ensure_ascii=False)
                
                # Log samples for both entries and examples
                if entries:
                    logger.info(f"📝 Dictionary entries from {Path(result['image_path']).name}:")
                    for i, entry in enumerate(entries[:3]):
                        kalenjin = entry.get("grapheme", "")
                        english = entry.get("english_meaning", "")
                        logger.info(f"  {i+1}. {kalenjin} → {english}")
                    if len(entries) > 3:
                        logger.info(f"  ... and {len(entries)-3} more dictionary entries")
                
                if examples:
                    logger.info(f"🗣️  Example sentences from {Path(result['image_path']).name}:")
                    for i, example in enumerate(examples[:2]):
                        kalenjin = example.get("kalenjin_sentence", "")
                        english = example.get("english_translation", "")
                        logger.info(f"  {i+1}. {kalenjin} → {english}")
                    if len(examples) > 2:
                        logger.info(f"  ... and {len(examples)-2} more example sentences")
        
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

    def _prefilter_ocr_text(self, text: str) -> str:
        """
        Pre-filter OCR text to reduce token count and improve processing speed
        """
        lines = text.split('\n')
        
        # Keep only lines that likely contain dictionary entries
        filtered_lines = []
        for line in lines:
            line = line.strip()
            if len(line) < 3:  # Skip very short lines
                continue
            if line.lower().startswith(('page', 'nandi', 'english')):  # Skip headers
                continue
            if len(line.split()) > 20:  # Skip very long lines (likely examples)
                continue
            if any(pattern in line.lower() for pattern in ['/ke:', '/ak', '/al', 'v.', 'n.', 'a.', 'c.']):
                filtered_lines.append(line)
            elif len(line.split()) <= 10 and any(c.islower() for c in line):  # Potential dictionary entry
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)

    def _process_text_in_chunks(self, text: str, image_path: str) -> Dict[str, Any]:
        """
        Process large OCR text in chunks for faster processing
        """
        lines = text.split('\n')
        chunk_size = 50  # lines per chunk
        all_entries = []
        
        chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
        
        logger.info(f"Processing {len(chunks)} chunks for faster performance...")
        
        for i, chunk in enumerate(chunks):
            chunk_text = '\n'.join(chunk)
            if not chunk_text.strip():
                continue
                
            try:
                entries = self.parser.parse_ocr_text(chunk_text, self.client)
                all_entries.extend(entries)
                logger.info(f"Chunk {i+1}/{len(chunks)}: {len(entries)} entries")
            except Exception as e:
                logger.warning(f"Chunk {i+1} failed: {e}")
                continue
        
        return {
            "image_path": image_path,
            "entries": all_entries,
            "method": "OCR + LLM (Chunked)",
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }


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
