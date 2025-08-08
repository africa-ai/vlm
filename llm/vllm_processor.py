"""
VLM Processor with vLLM Server Integration
Updated to use distributed vLLM serving for better performance and scalability
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import asyncio
import io
import base64

from llm.parser.main import DictionaryParser
from llm.parser.schemas import DictionaryEntry
from llm.config import load_config_from_env, VLMConfig
from llm.image_preprocessor import optimize_image_for_vlm
from vllm_server.client import VLLMClient, SyncVLLMClient

logger = logging.getLogger(__name__)


class VLLMServerProcessor:
    """
    VLM Processor using vLLM server for Cosmos-Reason1-7B inference
    Provides better performance and scalability compared to local model loading
    """
    
    def __init__(self, config: VLMConfig, server_url: str = "http://localhost:8000"):
        self.config = config
        self.server_url = server_url
        self.client = SyncVLLMClient(server_url, config.api_key if hasattr(config, 'api_key') else None)
        self.parser = DictionaryParser()
        self.is_connected = False
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"VLLMServerProcessor initialized with server: {server_url}")
    
    def check_server_connection(self) -> bool:
        """Check connection to vLLM server"""
        try:
            self.is_connected = self.client.health_check()
            if self.is_connected:
                logger.info("✅ Successfully connected to vLLM server")
            else:
                logger.error("❌ Failed to connect to vLLM server")
            return self.is_connected
        except Exception as e:
            logger.error(f"Error checking server connection: {e}")
            self.is_connected = False
            return False
    
    def process_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a single dictionary image using vLLM server with preprocessing
        
        Args:
            image_path: Path to the dictionary page image
            
        Returns:
            Dictionary with processing results including entries and reasoning
        """
        if not self.is_connected:
            if not self.check_server_connection():
                raise RuntimeError("vLLM server is not available. Please start the server first.")
        
        try:
            logger.info(f"Processing image with preprocessing: {image_path}")
            
            # Preprocess and potentially split the image (optimized for speed)
            optimized_images = optimize_image_for_vlm(str(image_path), target_tokens=75000)
            
            logger.info(f"Image split into {len(optimized_images)} part(s) for processing")
            
            all_entries = []
            all_responses = []
            all_reasoning = []
            
            # Process each optimized image part
            for i, image in enumerate(optimized_images):
                logger.info(f"Processing image part {i+1}/{len(optimized_images)}")
                
                # Convert PIL image to base64 for API
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                
                # Send image to vLLM server for analysis
                result = self.client.analyze_dictionary_image_base64(img_base64)
                
                if result.get("status") == "success":
                    # Parse the response to extract structured entries
                    analysis_text = result.get("analysis", "")
                    
                    # Use parser to structure the response
                    entries = self.parser.parse_vlm_response(analysis_text)
                    all_entries.extend(entries)
                    all_responses.append(analysis_text)
                    all_reasoning.append(self._extract_reasoning(analysis_text))
                else:
                    logger.error(f"Failed to process image part {i+1}: {result.get('error', 'Unknown error')}")
            
            return {
                "image_path": str(image_path),
                "entries": [entry.to_dict() if hasattr(entry, 'to_dict') else entry 
                          for entry in all_entries],
                "raw_responses": all_responses,
                "reasoning": all_reasoning,
                "model": result.get("model", "cosmos-reason-vlm"),
                "parts_processed": len(optimized_images),
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return {
                "image_path": str(image_path),
                "entries": [],
                "error": str(e),
                "status": "error",
                "timestamp": datetime.now().isoformat()
            }
    
    def batch_process_images(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Process multiple images one by one using vLLM server (sequential processing to manage memory)
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of processing results
        """
        if not self.is_connected:
            if not self.check_server_connection():
                raise RuntimeError("vLLM server is not available. Please start the server first.")
        
        logger.info(f"Starting sequential processing of {len(image_paths)} images via vLLM server")
        
        processed_results = []
        
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"Processing image {i}/{len(image_paths)}: {Path(image_path).name}")
            
            try:
                # Process single image using the existing process_image method
                result = self.process_image(image_path)
                processed_results.append(result)
                
                # Log progress
                if result.get("status") == "success":
                    entries_count = len(result.get("entries", []))
                    logger.info(f"✓ Image {i} processed successfully - {entries_count} entries found")
                else:
                    logger.error(f"✗ Image {i} failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"✗ Image {i} failed with exception: {e}")
                processed_results.append({
                    "image_path": str(image_path),
                    "entries": [],
                    "error": f"Processing failed: {str(e)}",
                    "status": "error",
                    "timestamp": datetime.now().isoformat()
                })
        
        # Calculate final statistics
        successful = sum(1 for r in processed_results if r.get("status") == "success")
        total_entries = sum(len(r.get("entries", [])) for r in processed_results if r.get("status") == "success")
        
        logger.info(f"Sequential processing completed: {successful}/{len(image_paths)} successful, "
                   f"{total_entries} total entries extracted")
        
        return processed_results
    
    def _extract_reasoning(self, response_text: str) -> str:
        """Extract reasoning section from Cosmos response"""
        try:
            if "<reasoning>" in response_text and "</reasoning>" in response_text:
                start = response_text.find("<reasoning>") + len("<reasoning>")
                end = response_text.find("</reasoning>")
                return response_text[start:end].strip()
            return ""
        except Exception:
            return ""
    
    def save_results(self, results: List[Dict[str, Any]], 
                    filename_prefix: str = "dictionary_extraction") -> Dict[str, str]:
        """
        Save processing results to JSON and CSV files
        
        Args:
            results: List of processing results
            filename_prefix: Prefix for output filenames
            
        Returns:
            Dictionary with saved file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_file = self.output_dir / f"{filename_prefix}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Prepare data for CSV
        csv_data = []
        for result in results:
            if result.get("status") == "success":
                for entry in result.get("entries", []):
                    csv_data.append({
                        "image_path": result["image_path"],
                        "kalenjin_word": entry.get("kalenjin_word", ""),
                        "ipa_transcription": entry.get("ipa_transcription", ""),
                        "english_meaning": entry.get("english_meaning", ""),
                        "additional_info": entry.get("additional_info", ""),
                        "confidence": entry.get("confidence", 0.0),
                        "timestamp": result["timestamp"]
                    })
            else:
                csv_data.append({
                    "image_path": result["image_path"],
                    "kalenjin_word": "",
                    "ipa_transcription": "",
                    "english_meaning": "",
                    "additional_info": f"Error: {result.get('error', 'Processing failed')}",
                    "confidence": 0.0,
                    "timestamp": result["timestamp"]
                })
        
        # Save CSV
        if csv_data:
            import pandas as pd
            df = pd.DataFrame(csv_data)
            csv_file = self.output_dir / f"{filename_prefix}_{timestamp}.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
        else:
            csv_file = None
        
        saved_files = {
            "json": str(json_file),
            "csv": str(csv_file) if csv_file else None
        }
        
        logger.info(f"Results saved to:")
        logger.info(f"  JSON: {saved_files['json']}")
        if saved_files['csv']:
            logger.info(f"  CSV: {saved_files['csv']}")
        
        return saved_files
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get information about the connected vLLM server"""
        if not self.is_connected:
            self.check_server_connection()
        
        return {
            "server_url": self.server_url,
            "is_connected": self.is_connected,
            "model_name": "nvidia/Cosmos-Reason1-7B",
            "served_model": "cosmos-reason-vlm",
            "client_type": "vLLM Server"
        }


# Factory function for backward compatibility
def create_vlm_processor(config: Optional[VLMConfig] = None, 
                        use_vllm_server: bool = True,
                        server_url: str = "http://localhost:8000") -> Union[VLLMServerProcessor, 'VLMProcessor']:
    """
    Factory function to create appropriate VLM processor
    
    Args:
        config: VLM configuration
        use_vllm_server: Whether to use vLLM server (recommended)
        server_url: vLLM server URL
        
    Returns:
        VLM processor instance
    """
    if config is None:
        config = load_config_from_env()
    
    if use_vllm_server:
        return VLLMServerProcessor(config, server_url)
    else:
        # Import local processor for fallback
        from .main import VLMProcessor
        return VLMProcessor(config)


if __name__ == "__main__":
    # Test the vLLM server processor
    config = load_config_from_env()
    processor = VLLMServerProcessor(config)
    
    print("Testing vLLM server connection...")
    if processor.check_server_connection():
        print("✅ vLLM server is ready for processing!")
        print(f"Server info: {processor.get_server_info()}")
    else:
        print("❌ vLLM server is not available. Please start the server first.")
        print(f"Expected server URL: {processor.server_url}")
        print("\nTo start the server, run:")
        print("python -m vllm_server.server")
