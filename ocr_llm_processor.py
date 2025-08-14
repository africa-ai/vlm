"""
OCR + LLM Processor for Dictionary Extraction
Clean pipeline: PDF → Images → OCR → vLLM → JSON
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import fitz  # PyMuPDF for PDF processing
import pytesseract
from PIL import Image
import io

from vllm_server.client import SyncVLLMClient
from llm.parser.main import DictionaryParser

logger = logging.getLogger(__name__)

class OCRLLMProcessor:
    """
    Simple OCR + LLM processor for dictionary extraction
    Much simpler than the image-based VLM approach
    """
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.client = SyncVLLMClient(server_url)
        self.parser = DictionaryParser()
        
        # Setup output directory
        self.output_dir = Path("results")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"OCRLLMProcessor initialized with vLLM server: {server_url}")
    
    def pdf_to_images(self, pdf_path: str, output_dir: Optional[str] = None) -> List[str]:
        """
        Convert PDF to images for OCR processing
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save images (optional)
            
        Returns:
            List of image file paths
        """
        if output_dir is None:
            output_dir = self.output_dir / "images"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        pdf_document = fitz.open(pdf_path)
        image_paths = []
        
        logger.info(f"Converting {len(pdf_document)} pages from PDF to images...")
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Convert to image with high DPI for better OCR
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            
            # Save as PNG
            image_path = output_dir / f"page_{page_num+1:03d}.png"
            pix.save(str(image_path))
            image_paths.append(str(image_path))
            
            logger.info(f"Saved page {page_num+1} as {image_path}")
        
        pdf_document.close()
        logger.info(f"✅ Converted {len(image_paths)} pages to images")
        return image_paths
    
    def ocr_image(self, image_path: str) -> str:
        """
        Extract text from image using OCR
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text
        """
        try:
            img = Image.open(image_path)
            
            # OCR with better configuration for dictionary pages
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()[]{}/-_/\' '
            
            text = pytesseract.image_to_string(img, config=custom_config)
            
            logger.info(f"OCR extracted {len(text)} characters from {Path(image_path).name}")
            return text
            
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            return ""
    
    def llm_parse_text(self, ocr_text: str, page_num: int = None) -> Dict[str, Any]:
        """
        Parse OCR'd text using vLLM to extract dictionary entries
        
        Args:
            ocr_text: Raw text from OCR
            page_num: Page number (for context)
            
        Returns:
            Parsed dictionary entries
        """
        # Simple, direct prompt for text parsing
        prompt = f"""Extract all Kalenjin dictionary entries from this OCR text as JSON.

OCR Text:
{ocr_text}

Extract each dictionary entry with this exact format:
[
  {{
    "grapheme": "kalenjin_word",
    "ipa": "/pronunciation/",
    "english_meaning": "definition",
    "part_of_speech": "v.t.",
    "context": "usage example",
    "confidence_score": 0.9
  }}
]

Be thorough and extract every entry you can identify from the text above.
"""
        
        try:
            # Send to vLLM server for text completion (not image analysis!)
            response = self.client.complete_text(prompt)
            
            # Parse the response
            entries = self.parser.parse_vlm_response(response)
            
            logger.info(f"LLM extracted {len(entries)} entries from page {page_num or 'unknown'}")
            
            return {
                "page_number": page_num,
                "entries": [entry.to_dict() if hasattr(entry, 'to_dict') else entry for entry in entries],
                "raw_ocr_text": ocr_text,
                "llm_response": response,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"LLM parsing failed for page {page_num}: {e}")
            return {
                "page_number": page_num,
                "entries": [],
                "error": str(e),
                "status": "error",
                "timestamp": datetime.now().isoformat()
            }
    
    def process_single_page(self, image_path: str, page_num: int = None) -> Dict[str, Any]:
        """
        Process a single page: OCR → LLM → JSON
        
        Args:
            image_path: Path to page image
            page_num: Page number
            
        Returns:
            Processing result
        """
        logger.info(f"📄 Processing page {page_num}: {Path(image_path).name}")
        
        # Step 1: OCR
        ocr_text = self.ocr_image(image_path)
        
        if not ocr_text.strip():
            logger.warning(f"No text extracted from {image_path}")
            return {
                "page_number": page_num,
                "entries": [],
                "error": "No text extracted by OCR",
                "status": "error",
                "timestamp": datetime.now().isoformat()
            }
        
        # Step 2: LLM Parse
        result = self.llm_parse_text(ocr_text, page_num)
        result["image_path"] = str(image_path)
        
        return result
    
    def process_pdf(self, pdf_path: str, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process entire PDF: PDF → Images → OCR → LLM → JSON
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum pages to process (optional)
            
        Returns:
            List of processing results
        """
        logger.info(f"🚀 Starting PDF processing: {pdf_path}")
        
        # Step 1: PDF to Images
        image_paths = self.pdf_to_images(pdf_path)
        
        if max_pages:
            image_paths = image_paths[:max_pages]
            logger.info(f"Limited to first {max_pages} pages")
        
        # Step 2: Process each page
        results = []
        total_entries = 0
        
        for i, image_path in enumerate(image_paths, 1):
            result = self.process_single_page(image_path, i)
            results.append(result)
            
            if result["status"] == "success":
                page_entries = len(result["entries"])
                total_entries += page_entries
                logger.info(f"✅ Page {i}: {page_entries} entries extracted")
            else:
                logger.error(f"❌ Page {i}: {result.get('error', 'Unknown error')}")
        
        # Final summary
        successful_pages = sum(1 for r in results if r["status"] == "success")
        logger.info(f"\n🏁 PDF Processing Complete!")
        logger.info(f"   📄 Pages processed: {successful_pages}/{len(results)}")
        logger.info(f"   📚 Total entries: {total_entries}")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], filename: str = "dictionary_results.json"):
        """Save results to JSON file"""
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Results saved to: {output_file}")
        return str(output_file)


def main():
    """Test the OCR + LLM processor"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ocr_llm_processor.py <pdf_path> [max_pages]")
        return
    
    pdf_path = sys.argv[1]
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Process PDF
    processor = OCRLLMProcessor()
    results = processor.process_pdf(pdf_path, max_pages)
    
    # Save results
    output_file = processor.save_results(results)
    
    print(f"\n✅ Processing complete! Results saved to: {output_file}")


if __name__ == "__main__":
    main()
