"""
Dictionary Parser for OCR + LLM extraction.
Simple parser for extracting dictionary entries from OCR'd text.
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional

from .prompts import get_extraction_prompt, get_system_prompt

logger = logging.getLogger(__name__)


class DictionaryParser:
    """Simple parser for extracting dictionary entries from OCR text"""
    
    def __init__(self, config=None):
        """
        Initialize dictionary parser
        
        Args:
            config: Configuration instance (optional)
        """
        self.config = config
        
        # Regex patterns for cleaning and validation
        self.patterns = {
            'ipa_pattern': r'(/[^/]+/|\[[^\]]+\]|_[^_]+_)',
            'kalenjin_word': r'\b[a-zA-Z-]{2,}\b',
            'english_text': r'[a-zA-Z\s\-\',\.;:]+',
            'artifact_pattern': r'^[^a-zA-Z]*$|^\d+$|^[IV]+\.$',  # Roman numerals, numbers, etc.
        }
    
    def parse_ocr_text(self, ocr_text: str, vllm_client) -> List[Dict[str, Any]]:
        """
        Parse dictionary entries from OCR-extracted text using vLLM
        
        Args:
            ocr_text: Raw text extracted from OCR
            vllm_client: vLLM client for text completion
            
        Returns:
            List of extracted dictionary entries
        """
        if not ocr_text or not ocr_text.strip():
            logger.warning("Empty OCR text provided")
            return []
        
        # Clean OCR text
        cleaned_text = self._clean_ocr_text(ocr_text)
        
        if not cleaned_text:
            logger.warning("No valid text after cleaning OCR output")
            return []
        
        # Get extraction prompt - combine system and user prompt
        system_prompt = get_system_prompt()
        user_prompt = get_extraction_prompt(cleaned_text)
        
        # Combine prompts for single completion call
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        try:
            # Process with vLLM - simple completion call
            response = vllm_client.complete_text(
                prompt=full_prompt,
                max_tokens=4000,
                temperature=0.1
            )
            
            if not response or response.get('status') != 'success':
                error_msg = response.get('error', 'Unknown error') if response else 'No response'
                logger.error(f"vLLM completion failed: {error_msg}")
                return []
            
            # Extract the completion content from vLLM response
            completion = response.get('completion', '')
            
            if not completion:
                logger.error("Empty completion from vLLM")
                return []
            
            # Parse JSON response
            entries = self._parse_json_response(completion)
            
            # Validate and clean entries
            valid_entries = []
            for entry in entries:
                cleaned_entry = self._validate_entry(entry)
                if cleaned_entry:
                    valid_entries.append(cleaned_entry)
            
            logger.info(f"Extracted {len(valid_entries)} valid entries from OCR text")
            return valid_entries
            
        except Exception as e:
            logger.error(f"Error processing OCR text with vLLM: {e}")
            return []
    
    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean OCR text to improve LLM processing
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Basic cleaning
        cleaned = text.strip()
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove common OCR artifacts
        cleaned = re.sub(r'[^\w\s\-/\[\]().,;:\'"]', ' ', cleaned)
        
        # Remove very short lines (likely artifacts)
        lines = cleaned.split('\n')
        clean_lines = [line.strip() for line in lines if len(line.strip()) > 2]
        
        return '\n'.join(clean_lines)
    
    def _parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse JSON response from vLLM, handling various formats
        
        Args:
            response: Raw response text from vLLM
            
        Returns:
            List of dictionary entries
        """
        entries = []
        
        # Clean response text
        response = response.strip()
        
        # Try to find JSON array or objects
        json_patterns = [
            r'\[.*\]',  # JSON array
            r'\{.*\}',  # Single JSON object
        ]
        
        json_text = None
        for pattern in json_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                json_text = match.group(0)
                break
        
        if not json_text:
            logger.warning("No JSON found in response")
            return []
        
        try:
            # Parse JSON
            data = json.loads(json_text)
            
            # Handle different formats
            if isinstance(data, list):
                entries = data
            elif isinstance(data, dict):
                entries = [data]
            else:
                logger.warning(f"Unexpected JSON format: {type(data)}")
                return []
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            # Try to extract individual JSON objects
            entries = self._extract_individual_objects(response)
        
        return entries
    
    def _extract_individual_objects(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract individual JSON objects from malformed response
        
        Args:
            text: Response text that may contain multiple JSON objects
            
        Returns:
            List of extracted objects
        """
        entries = []
        
        # Find all JSON-like objects
        object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(object_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                entry = json.loads(match)
                entries.append(entry)
            except json.JSONDecodeError:
                continue
        
        return entries
    
    def _validate_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validate and clean a dictionary entry
        
        Args:
            entry: Raw dictionary entry
            
        Returns:
            Cleaned entry or None if invalid
        """
        if not isinstance(entry, dict):
            return None
        
        # Required fields
        grapheme = entry.get('grapheme', '').strip()
        english_meaning = entry.get('english_meaning', '').strip()
        
        if not grapheme or not english_meaning:
            return None
        
        # Skip artifacts
        if re.match(self.patterns['artifact_pattern'], grapheme):
            return None
        
        # Clean IPA if present
        ipa = entry.get('ipa', '')
        if ipa:
            ipa = ipa.strip()
            if not re.search(self.patterns['ipa_pattern'], ipa):
                ipa = None
        else:
            ipa = None
        
        # Build clean entry
        clean_entry = {
            'grapheme': grapheme,
            'english_meaning': english_meaning,
            'ipa': ipa,
            'confidence': self._calculate_confidence(grapheme, english_meaning, ipa)
        }
        
        return clean_entry
    
    def _calculate_confidence(self, grapheme: str, english_meaning: str, ipa: Optional[str]) -> float:
        """
        Calculate confidence score for an entry
        
        Args:
            grapheme: Kalenjin word
            english_meaning: English translation
            ipa: IPA transcription
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        score = 0.6  # Base score
        
        # Grapheme quality
        if len(grapheme) > 2 and re.match(r'^[a-zA-Z-]+$', grapheme):
            score += 0.1
        
        # English meaning quality
        if len(english_meaning) > 3 and any(word in english_meaning.lower() for word in ['to', 'a', 'the', 'of', 'in', 'and']):
            score += 0.1
        
        # IPA presence
        if ipa and re.search(self.patterns['ipa_pattern'], ipa):
            score += 0.1
        
        # Length reasonableness
        if 2 <= len(grapheme) <= 20 and 3 <= len(english_meaning) <= 100:
            score += 0.1
        
        return min(1.0, score)
