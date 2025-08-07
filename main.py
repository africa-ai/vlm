"""
Main entry point for Kalenjin Dictionary Processing Framework
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from scripts.pdf_to_images import PDFToImageConverter
from llm.main import VLMProcessor
from llm.config import VLMConfig, load_config_from_env

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Kalenjin Dictionary Processing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert PDF to images only
  python main.py convert kalenjin_dictionary.pdf --output ./images
  
  # Process images with VLM
  python main.py process ./images --output ./results
  
  # Full pipeline: PDF -> Images -> VLM Processing
  python main.py pipeline kalenjin_dictionary.pdf --output ./results
  
  # Batch process multiple PDFs
  python main.py batch ./pdfs --output ./results
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Convert command (PDF to images)
    convert_parser = subparsers.add_parser('convert', help='Convert PDF to high-res images')
    convert_parser.add_argument('pdf_path', help='Path to PDF file')
    convert_parser.add_argument('-o', '--output', default='./images', help='Output directory for images')
    convert_parser.add_argument('--dpi', type=int, default=300, help='Image resolution (DPI)')
    convert_parser.add_argument('--format', default='PNG', help='Image format')
    
    # Process command (VLM processing of images)
    process_parser = subparsers.add_parser('process', help='Process images with VLM')
    process_parser.add_argument('image_dir', help='Directory containing images')
    process_parser.add_argument('-o', '--output', default='./output', help='Output directory')
    process_parser.add_argument('--model', help='Model name override')
    process_parser.add_argument('--batch-size', type=int, help='Batch size')
    process_parser.add_argument('--device', help='Device (cuda/cpu)')
    
    # Pipeline command (full processing pipeline)
    pipeline_parser = subparsers.add_parser('pipeline', help='Full pipeline: PDF -> Images -> VLM')
    pipeline_parser.add_argument('pdf_path', help='Path to PDF file')
    pipeline_parser.add_argument('-o', '--output', default='./results', help='Output directory')
    pipeline_parser.add_argument('--dpi', type=int, default=300, help='Image resolution (DPI)')
    pipeline_parser.add_argument('--model', help='Model name override')
    pipeline_parser.add_argument('--batch-size', type=int, help='Batch size')
    pipeline_parser.add_argument('--device', help='Device (cuda/cpu)')
    pipeline_parser.add_argument('--keep-images', action='store_true', help='Keep intermediate images')
    
    # Batch command (multiple PDFs)
    batch_parser = subparsers.add_parser('batch', help='Process multiple PDFs')
    batch_parser.add_argument('pdf_dir', help='Directory containing PDF files')
    batch_parser.add_argument('-o', '--output', default='./batch_results', help='Output base directory')
    batch_parser.add_argument('--dpi', type=int, default=300, help='Image resolution (DPI)')
    batch_parser.add_argument('--model', help='Model name override')
    batch_parser.add_argument('--batch-size', type=int, help='Batch size')
    batch_parser.add_argument('--device', help='Device (cuda/cpu)')
    
    # Global options
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'convert':
            run_convert(args)
        elif args.command == 'process':
            run_process(args)
        elif args.command == 'pipeline':
            run_pipeline(args)
        elif args.command == 'batch':
            run_batch(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1
    
    return 0

def run_convert(args):
    """Run PDF to image conversion"""
    logger = logging.getLogger(__name__)
    logger.info(f"Converting PDF {args.pdf_path} to images")
    
    converter = PDFToImageConverter(dpi=args.dpi, image_format=args.format)
    image_paths = converter.convert_pdf_to_images(args.pdf_path, args.output)
    
    logger.info(f"Successfully converted {len(image_paths)} pages")
    for path in image_paths:
        print(f"  -> {path}")

def run_process(args):
    """Run VLM processing on images"""
    logger = logging.getLogger(__name__)
    logger.info(f"Processing images from {args.image_dir}")
    
    # Load configuration
    config = load_config_from_env()
    if args.output:
        config.output_dir = args.output
    if args.model:
        config.model_name = args.model
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.device:
        config.device = args.device
    
    # Initialize and run processor
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

def run_pipeline(args):
    """Run full processing pipeline"""
    logger = logging.getLogger(__name__)
    logger.info(f"Running full pipeline for {args.pdf_path}")
    
    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    
    # Step 1: Convert PDF to images
    logger.info("Step 1: Converting PDF to images")
    converter = PDFToImageConverter(dpi=args.dpi, image_format="PNG")
    image_paths = converter.convert_pdf_to_images(args.pdf_path, str(images_dir))
    
    # Step 2: Process with VLM
    logger.info("Step 2: Processing with VLM")
    config = load_config_from_env()
    config.output_dir = str(output_dir)
    if args.model:
        config.model_name = args.model
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.device:
        config.device = args.device
    
    processor = VLMProcessor(config)
    processor.load_model()
    
    results = processor.batch_process_images(image_paths)
    processor.save_results(results)
    
    # Step 3: Cleanup (if requested)
    if not args.keep_images:
        logger.info("Step 3: Cleaning up intermediate images")
        import shutil
        shutil.rmtree(images_dir)
    
    # Print summary
    successful = sum(1 for r in results if r['status'] == 'success')
    total_entries = sum(len(r['entries']) for r in results if r['status'] == 'success')
    
    print(f"Pipeline completed:")
    print(f"  Successfully processed: {successful}/{len(results)} pages")
    print(f"  Total dictionary entries extracted: {total_entries}")
    print(f"  Results saved to: {output_dir}")

def run_batch(args):
    """Run batch processing on multiple PDFs"""
    logger = logging.getLogger(__name__)
    logger.info(f"Batch processing PDFs from {args.pdf_dir}")
    
    pdf_dir = Path(args.pdf_dir)
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Initialize components
    converter = PDFToImageConverter(dpi=args.dpi, image_format="PNG")
    
    config = load_config_from_env()
    if args.model:
        config.model_name = args.model
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.device:
        config.device = args.device
    
    processor = VLMProcessor(config)
    processor.load_model()
    
    all_results = []
    
    for pdf_file in pdf_files:
        logger.info(f"Processing {pdf_file.name}")
        
        try:
            # Create output directory for this PDF
            pdf_output_dir = Path(args.output) / pdf_file.stem
            images_dir = pdf_output_dir / "images"
            
            # Convert to images
            image_paths = converter.convert_pdf_to_images(str(pdf_file), str(images_dir))
            
            # Process with VLM
            config.output_dir = str(pdf_output_dir)
            processor.config = config
            
            results = processor.batch_process_images(image_paths)
            processor.save_results(results, f"{pdf_file.stem}_extraction")
            
            all_results.extend(results)
            
            # Cleanup images
            import shutil
            shutil.rmtree(images_dir)
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_file.name}: {e}")
    
    # Print final summary
    successful = sum(1 for r in all_results if r['status'] == 'success')
    total_entries = sum(len(r['entries']) for r in all_results if r['status'] == 'success')
    
    print(f"Batch processing completed:")
    print(f"  Processed {len(pdf_files)} PDF files")
    print(f"  Successfully processed: {successful}/{len(all_results)} pages")
    print(f"  Total dictionary entries extracted: {total_entries}")
    print(f"  Results saved to: {args.output}")

if __name__ == "__main__":
    exit(main())
