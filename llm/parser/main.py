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
            'ipa_pattern': r'(/[^/]+/)',  # Specifically look for /content/ pattern
            'kalenjin_word': r'\b[a-zA-Z-]{2,}\b',
            'english_text': r'[a-zA-Z\s\-\',\.;:]+',
            'artifact_pattern': r'^[^a-zA-Z]*$|^\d+$|^[IV]+\.$',
            'grammar_markers': r'\b(c\.|p\.|a\.|v\.|n\.|adj\.|adv\.|prep\.|conj\.|interj\.|v\.t\.|v\.i\.|v\.itv\.|v\.app\.|v\.refl\.|v\.ven\.|v\.ins\.)\b',
            'headword_pattern': r'^[a-zA-Z][a-zA-Z-]*[a-zA-Z]?$',  # Valid headword pattern
            'example_indicators': r'\b(The|He|She|It|I|You|We|They|There|What|Where|When|How)\b'  # Sentence starters to avoid
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
        Parse JSON response from vLLM, handling various formats with improved extraction
        
        Args:
            response: Raw response text from vLLM
            
        Returns:
            List of dictionary entries
        """
        entries = []
        
        # Clean response text
        response = response.strip()
        
        # Find the JSON array more precisely - look for opening [ and closing ]
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_text = response[json_start:json_end]
            
            try:
                # Parse the extracted JSON
                data = json.loads(json_text)
                if isinstance(data, list):
                    entries = data
                elif isinstance(data, dict):
                    entries = [data]
                else:
                    logger.warning(f"Unexpected JSON format: {type(data)}")
                    return []
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse extracted JSON: {e}")
                # Fallback to individual object extraction
                entries = self._extract_individual_objects(json_text)
        else:
            logger.warning("No JSON array found in response")
            # Fallback to pattern matching
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
        Validate and clean a dictionary entry - optimized for Kalenjin dictionary structure
        
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
        
        # Skip artifacts and OCR errors
        if re.match(self.patterns['artifact_pattern'], grapheme):
            return None
            
        # Validate headword pattern (should look like: ak, akwai, ke-al, alamaliet)
        if not re.match(self.patterns['headword_pattern'], grapheme):
            logger.debug(f"Invalid headword pattern: {grapheme}")
            return None
            
        # Skip entries with numbers in headword (numbers only for definitions)
        if re.search(r'[0-9]', grapheme):
            logger.debug(f"Skipping entry with numbers in headword: {grapheme}")
            return None
            
        # Skip example sentences (look for sentence indicators)
        if re.search(self.patterns['example_indicators'], english_meaning):
            logger.debug(f"Skipping example sentence: {english_meaning}")
            return None
            
        # Skip very long "definitions" that are likely examples or sentences
        if len(english_meaning) > 150:  # Dictionary definitions should be concise
            logger.debug(f"Skipping overly long definition: {english_meaning[:50]}...")
            return None
            
        # Validate and clean IPA - must be in /forward slashes/
        ipa = entry.get('ipa', '')
        if ipa:
            ipa = ipa.strip()
            # Only accept IPA in /forward slashes/ format
            ipa_match = re.search(self.patterns['ipa_pattern'], ipa)
            if ipa_match:
                ipa = ipa_match.group(1)  # Extract just the /content/
            else:
                logger.debug(f"Invalid IPA format, ignoring: {ipa}")
                ipa = None
        else:
            ipa = None
        
        # Clean grapheme (remove ke- prefixes for validation if needed)
        base_grapheme = grapheme
        if grapheme.startswith('ke-'):
            base_grapheme = grapheme[3:]  # Remove ke- prefix for some validations
            
        # Ensure reasonable lengths
        if len(base_grapheme) < 2 or len(english_meaning) < 3:
            return None
        
        # Handle numbered definitions (merge multiple meanings)
        definitions = self._merge_numbered_definitions(english_meaning)
        
        # Build clean entry
        clean_entry = {
            'grapheme': grapheme,
            'english_meaning': english_meaning,
            'ipa': ipa,
            'definitions': definitions,  # Add parsed definitions
            'confidence': self._calculate_confidence(grapheme, english_meaning, ipa)
        }
        
        return clean_entry
    
    def _merge_numbered_definitions(self, definition: str) -> list:
        """
        Merge numbered definitions (1., 2., 3.) into a list.
        
        Args:
            definition: Definition text that may contain numbered meanings
            
        Returns:
            List of definition objects with number and text
        """
        import re
        
        # Split on numbered patterns like "1.", "2.", "3."
        pattern = r'\s*(\d+)\.\s*'
        parts = re.split(pattern, definition)
        
        definitions = []
        if len(parts) > 2:  # Has numbered definitions
            # Remove empty first element and pair numbers with definitions
            parts = [p.strip() for p in parts[1:] if p.strip()]  # Skip first empty part
            for i in range(0, len(parts), 2):
                if i + 1 < len(parts):
                    number = parts[i]
                    def_text = parts[i + 1].strip()
                    if def_text and number.isdigit():
                        definitions.append({
                            'number': int(number),
                            'definition': def_text
                        })
        
        if not definitions:
            # Single definition (no numbers) - remove any leading numbers if present
            clean_def = re.sub(r'^\s*\d+\.\s*', '', definition).strip()
            definitions.append({
                'number': 1,
                'definition': clean_def
            })
        
        return definitions
    
    def _calculate_confidence(self, grapheme: str, english_meaning: str, ipa: Optional[str]) -> float:
        """
        Calculate confidence score for an entry - rewards IPA transcriptions
        
        Args:
            grapheme: Kalenjin word
            english_meaning: English translation
            ipa: IPA transcription
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        score = 0.5  # Base score
        
        # Grapheme quality
        if len(grapheme) > 2 and re.match(r'^[a-zA-Z-]+$', grapheme):
            score += 0.15
        
        # English meaning quality
        if len(english_meaning) > 3 and any(word in english_meaning.lower() for word in ['to', 'a', 'the', 'of', 'in', 'and', 'v.', 'n.', 'adj.']):
            score += 0.15
        
        # IPA presence - MAJOR bonus for finding phonetic transcriptions
        if ipa and re.search(self.patterns['ipa_pattern'], ipa):
            score += 0.2  # Increased bonus for IPA
            logger.debug(f"IPA found for '{grapheme}': {ipa}")
        
        # Length reasonableness
        if 2 <= len(grapheme) <= 20 and 3 <= len(english_meaning) <= 100:
            score += 0.1
        
        return min(1.0, score)
