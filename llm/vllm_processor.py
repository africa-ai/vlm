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
        
        # Setup continuous saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.live_results_file = self.output_dir / f"live_results_{timestamp}.jsonl"
        self.live_entries_file = self.output_dir / f"live_entries_{timestamp}.json"
        self.processed_count = 0
        self.all_entries = []
        
        logger.info(f"VLLMServerProcessor initialized with server: {server_url}")
        logger.info(f"Live results will be saved to: {self.live_results_file}")
        logger.info(f"Live entries will be saved to: {self.live_entries_file}")
    
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
    
    def save_live_result(self, result: Dict[str, Any]) -> None:
        """Save individual processing result immediately"""
        try:
            # Append to JSONL file (one result per line)
            with open(self.live_results_file, 'a', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
            
            # If successful, add entries to running collection
            if result.get("status") == "success":
                entries = result.get("entries", [])
                self.all_entries.extend(entries)
                
                # Update the live entries file
                with open(self.live_entries_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "total_entries": len(self.all_entries),
                        "processed_images": self.processed_count,
                        "last_updated": datetime.now().isoformat(),
                        "entries": self.all_entries
                    }, f, indent=2, ensure_ascii=False)
                
                # Log sample of new entries found
                if entries:
                    logger.info(f"📝 Sample entries from this image:")
                    for i, entry in enumerate(entries[:3]):  # Show first 3 entries
                        kalenjin = entry.get("kalenjin_word", entry.get("term", entry.get("grapheme", "")))
                        english = entry.get("english_meaning", entry.get("definition", entry.get("english", "")))
                        logger.info(f"  {i+1}. {kalenjin} → {english}")
                    if len(entries) > 3:
                        logger.info(f"  ... and {len(entries)-3} more entries")
        
        except Exception as e:
            logger.error(f"Failed to save live result: {e}")
    
    
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
            
            # Split image for systematic processing (force split for better coverage)
            optimized_images = optimize_image_for_vlm(str(image_path), target_tokens=80000, force_split=True)
            
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
                
                # Send image to vLLM server for analysis using our refined prompts
                from llm.parser.prompts import PromptTemplates
                custom_prompt = PromptTemplates.get_extraction_prompt("complete")
                result = self.client.analyze_dictionary_image_base64(img_base64, custom_prompt=custom_prompt)
                
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
        Process multiple images one by one with continuous saving
        """
        if not self.is_connected:
            if not self.check_server_connection():
                raise RuntimeError("vLLM server is not available. Please start the server first.")
        
        logger.info(f"Starting sequential processing of {len(image_paths)} images via vLLM server")
        logger.info(f"🔄 Results will be continuously saved to:")
        logger.info(f"   📄 Live results: {self.live_results_file}")
        logger.info(f"   📚 Live entries: {self.live_entries_file}")
        
        processed_results = []
        
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"\n🔍 Processing image {i}/{len(image_paths)}: {Path(image_path).name}")
            
            try:
                # Process single image using the existing process_image method
                result = self.process_image(image_path)
                processed_results.append(result)
                self.processed_count += 1
                
                # Save result immediately
                self.save_live_result(result)
                
                # Log progress with stats
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
                processed_results.append(error_result)
                self.processed_count += 1
                self.save_live_result(error_result)
                logger.error(f"❌ Image {i} failed with exception: {e}")
        
        # Final summary
        successful = sum(1 for r in processed_results if r.get("status") == "success")
        total_entries = len(self.all_entries)
        
        logger.info(f"\n🏁 Processing completed!")
        logger.info(f"   ✅ Successful: {successful}/{len(image_paths)} images")
        logger.info(f"   📚 Total entries extracted: {total_entries}")
        logger.info(f"   📄 Results saved to: {self.live_results_file}")
        logger.info(f"   📚 Entries saved to: {self.live_entries_file}")
        
        return processed_results
    
    def save_results(self, results: List[Dict[str, Any]], filename_prefix: str = "final_results") -> None:
        """
        Save final results to files (compatibility method)
        
        Args:
            results: List of processing results
            filename_prefix: Prefix for result files
        """
        try:
            # Save final compiled results
            final_results_file = self.output_dir / f"{filename_prefix}.json"
            
            # Extract all entries from successful results
            all_entries = []
            successful_results = []
            
            for result in results:
                if result.get("status") == "success":
                    successful_results.append(result)
                    entries = result.get("entries", [])
                    all_entries.extend(entries)
            
            # Save comprehensive final results
            final_data = {
                "summary": {
                    "total_images_processed": len(results),
                    "successful_images": len(successful_results),
                    "failed_images": len(results) - len(successful_results),
                    "total_entries_extracted": len(all_entries),
                    "processing_completed": datetime.now().isoformat()
                },
                "entries": all_entries,
                "detailed_results": results
            }
            
            with open(final_results_file, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📁 Final results saved to: {final_results_file}")
            logger.info(f"📊 Summary: {len(all_entries)} entries from {len(successful_results)}/{len(results)} images")
            
        except Exception as e:
            logger.error(f"Failed to save final results: {e}")
    
    
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
