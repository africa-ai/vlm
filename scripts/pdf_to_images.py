"""
PDF to High-Resolution Images Converter
Converts PDF pages to high-resolution images for VLM processing
"""

import os
import sys
import io
from pathlib import Path
from typing import List, Optional
import fitz  # PyMuPDF
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFToImageConverter:
    def __init__(self, dpi: int = 300, image_format: str = 'PNG'):
        """
        Initialize PDF to Image converter
        
        Args:
            dpi: Resolution for image conversion (default: 300)
            image_format: Output image format (default: PNG)
        """
        self.dpi = dpi
        self.image_format = image_format.upper()
        
    def convert_pdf_to_images(
        self, 
        pdf_path: str, 
        output_dir: str, 
        page_range: Optional[tuple] = None
    ) -> List[str]:
        """
        Convert PDF pages to high-resolution images
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save images
            page_range: Tuple of (start_page, end_page) or None for all pages
            
        Returns:
            List of paths to generated images
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        doc = fitz.open(str(pdf_path))
        image_paths = []
        
        # Determine page range
        start_page = page_range[0] if page_range else 0
        end_page = page_range[1] if page_range else doc.page_count
        
        logger.info(f"Converting pages {start_page} to {end_page-1} from {pdf_path.name}")
        
        try:
            for page_num in range(start_page, min(end_page, doc.page_count)):
                page = doc[page_num]
                
                # Create transformation matrix for high DPI
                zoom = self.dpi / 72  # 72 is default DPI
                mat = fitz.Matrix(zoom, zoom)
                
                # Render page to pixmap
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Save image
                image_filename = f"{pdf_path.stem}_page_{page_num + 1:03d}.{self.image_format.lower()}"
                image_path = output_dir / image_filename
                
                img.save(str(image_path), format=self.image_format)
                image_paths.append(str(image_path))
                
                logger.info(f"Converted page {page_num + 1} -> {image_filename}")
                
        finally:
            doc.close()
        
        logger.info(f"Successfully converted {len(image_paths)} pages")
        return image_paths
    
    def batch_convert(self, pdf_directory: str, output_base_dir: str) -> dict:
        """
        Convert multiple PDFs to images
        
        Args:
            pdf_directory: Directory containing PDF files
            output_base_dir: Base directory for output images
            
        Returns:
            Dictionary mapping PDF names to their image paths
        """
        pdf_dir = Path(pdf_directory)
        output_base = Path(output_base_dir)
        
        results = {}
        
        for pdf_file in pdf_dir.glob("*.pdf"):
            pdf_output_dir = output_base / pdf_file.stem
            try:
                image_paths = self.convert_pdf_to_images(
                    str(pdf_file), 
                    str(pdf_output_dir)
                )
                results[pdf_file.name] = image_paths
            except Exception as e:
                logger.error(f"Failed to convert {pdf_file.name}: {e}")
                results[pdf_file.name] = []
        
        return results

def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert PDF to high-resolution images")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("-o", "--output", default="./images", help="Output directory")
    parser.add_argument("--dpi", type=int, default=300, help="Image resolution (DPI)")
    parser.add_argument("--format", default="PNG", help="Image format")
    parser.add_argument("--start-page", type=int, help="Start page (1-indexed)")
    parser.add_argument("--end-page", type=int, help="End page (1-indexed)")
    
    args = parser.parse_args()
    
    converter = PDFToImageConverter(dpi=args.dpi, image_format=args.format)
    
    page_range = None
    if args.start_page and args.end_page:
        page_range = (args.start_page - 1, args.end_page)  # Convert to 0-indexed
    
    try:
        image_paths = converter.convert_pdf_to_images(
            args.pdf_path, 
            args.output, 
            page_range
        )
        print(f"Successfully converted {len(image_paths)} pages")
        for path in image_paths:
            print(f"  -> {path}")
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
