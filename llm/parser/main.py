"""
Main Dictionary Parser for VLM-based extraction
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re

from .schemas import DictionaryEntry, ExtractionResult
from .prompts import PromptTemplates, PROMPT_COMBINATIONS

logger = logging.getLogger(__name__)

class DictionaryParser:
    """Parser for extracting dictionary entries from images using VLM"""
    
    def __init__(self, config=None):
        """
        Initialize dictionary parser
        
        Args:
            config: VLMConfig instance for settings
        """
        self.config = config
        self.prompt_templates = PromptTemplates()
        
        # Regex patterns for Kalenjin dictionary format
        self.patterns = {
            'entry_line': r'^([a-zA-Z-]+)\s+(v\.t\.|v\.i\.|v\.|n\.|adj\.|adv\.|prep\.|conj\.|interj\.)\s*(/[^/]+/)?\s*(.+)$',
            'ipa_pattern': r'/([^/]+)/',
            'pos_pattern': r'\b(v\.t\.|v\.i\.|v\.|n\.|adj\.|adv\.|prep\.|conj\.|interj\.)\b',
            'kalenjin_word': r'\b[a-zA-Z-]{2,}\b',
            'cross_reference': r'[A-Z][a-z]+\.',
            'usage_example': r'[A-Z][a-z]+[^.]*\?',  # Questions like "Ingoro aba?"
        }
    
    def parse_image(self, image_path: str, method: str = "standard") -> List[Dict[str, Any]]:
        """
        Parse dictionary entries from a single image
        
        Args:
            image_path: Path to the image file
            method: Parsing method ("standard", "with_examples", "batch", "quality_focused")
            
        Returns:
            List of extracted dictionary entries
        """
        start_time = time.time()
        
        try:
            # Get the appropriate prompt
            prompt_config = PROMPT_COMBINATIONS.get(method, PROMPT_COMBINATIONS["standard_extraction"])
            user_prompt = prompt_config["user"]
            
            # Process with VLM (this would be called from the main VLM processor)
            response = self._process_with_vlm(image_path, user_prompt)
            
            # Parse the VLM response
            entries = self._parse_vlm_response(response)
            
            # Post-process and validate entries
            validated_entries = self._validate_entries(entries)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Extracted {len(validated_entries)} entries from {image_path} "
                       f"in {processing_time:.2f}s")
            
            return validated_entries
            
        except Exception as e:
            logger.error(f"Failed to parse image {image_path}: {e}")
            return []
    
    def _process_with_vlm(self, image_path: str, prompt: str) -> str:
        """
        Process image with VLM (placeholder - called from main processor)
        
        Args:
            image_path: Path to image
            prompt: Text prompt
            
        Returns:
            VLM response text
        """
        # This method is a placeholder and will be called by the main VLM processor
        # In practice, this would interface with the loaded model
        
        if hasattr(self, '_vlm_processor') and self._vlm_processor:
            return self._vlm_processor.process_image(image_path, prompt)
        else:
            # Fallback for testing or when VLM is not available
            logger.warning("VLM processor not available, using fallback parsing")
            return self._fallback_parse(image_path)
    
    def _fallback_parse(self, image_path: str) -> str:
        """
        Fallback parsing method (for testing without VLM)
        
        Args:
            image_path: Path to image file
            
        Returns:
            Mock JSON response
        """
        # This is a simple fallback for testing
        return """[
            {
                "grapheme": "example_word",
                "ipa": "/ɪɡˈzæmpəl/",
                "english_meaning": "a sample or illustration",
                "part_of_speech": "noun",
                "context": "used as an example",
                "confidence_score": 0.8
            }
        ]"""
    
    def _parse_vlm_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse VLM response and extract structured data
        
        Args:
            response: Raw VLM response text
            
        Returns:
            List of dictionary entry dictionaries
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                json_str = json_match.group()
                entries = json.loads(json_str)
                return entries if isinstance(entries, list) else [entries]
            
            # If no JSON found, try to parse as plain text
            return self._parse_text_response(response)
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return self._parse_text_response(response)
        except Exception as e:
            logger.error(f"Error parsing VLM response: {e}")
            return []
    
    def _parse_text_response(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse plain text response using regex patterns
        
        Args:
            text: Plain text response
            
        Returns:
            List of parsed entries
        """
        entries = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue
            
            entry = self._parse_entry_line(line)
            if entry and entry.get('grapheme'):
                entries.append(entry)
        
        return entries
    
    def _parse_entry_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single dictionary entry line
        
        Args:
            line: Text line containing dictionary entry
            
        Returns:
            Parsed entry dictionary or None
        """
        try:
            # Extract IPA transcription
            ipa_match = re.search(self.patterns['ipa_pattern'], line)
            ipa = ipa_match.group(1) if ipa_match else None
            
            # Extract part of speech
            pos_match = re.search(self.patterns['pos_pattern'], line)
            pos = pos_match.group(1) if pos_match else None
            
            # Extract grapheme (first word, typically)
            words = line.split()
            if not words:
                return None
            
            grapheme = words[0]
            
            # Remove known elements to get meaning
            clean_line = line
            if ipa_match:
                clean_line = clean_line.replace(ipa_match.group(), '').strip()
            if pos_match:
                clean_line = clean_line.replace(pos_match.group(), '').strip()
            
            # Get remaining text as meaning
            meaning_parts = clean_line.split()
            if meaning_parts and meaning_parts[0] == grapheme:
                meaning_parts = meaning_parts[1:]  # Remove grapheme
            
            english_meaning = ' '.join(meaning_parts).strip() if meaning_parts else None
            
            return {
                'grapheme': grapheme,
                'ipa': ipa,
                'english_meaning': english_meaning,
                'part_of_speech': pos,
                'context': None,
                'confidence_score': 0.7  # Lower confidence for regex parsing
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse line '{line}': {e}")
            return None
    
    def _validate_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and clean extracted entries
        
        Args:
            entries: List of raw extracted entries
            
        Returns:
            List of validated entries
        """
        validated = []
        
        for entry_data in entries:
            try:
                # Create DictionaryEntry object for validation
                entry = DictionaryEntry(
                    grapheme=entry_data.get('grapheme', '').strip(),
                    ipa=self._clean_ipa(entry_data.get('ipa')),
                    english_meaning=self._clean_text(entry_data.get('english_meaning')),
                    part_of_speech=self._normalize_pos(entry_data.get('part_of_speech')),
                    context=self._clean_text(entry_data.get('context')),
                    confidence_score=float(entry_data.get('confidence_score', 1.0))
                )
                
                # Only include valid entries
                if entry.is_valid():
                    validated.append(entry.to_dict())
                else:
                    logger.debug(f"Skipping invalid entry: {entry_data}")
                    
            except Exception as e:
                logger.warning(f"Failed to validate entry {entry_data}: {e}")
        
        return validated
    
    def _clean_ipa(self, ipa: Optional[str]) -> Optional[str]:
        """Clean and validate IPA transcription"""
        if not ipa:
            return None
        
        ipa = ipa.strip()
        
        # Add slashes if missing
        if ipa and not (ipa.startswith('/') and ipa.endswith('/')):
            if not (ipa.startswith('[') and ipa.endswith(']')):
                ipa = f"/{ipa}/"
        
        return ipa if ipa else None
    
    def _clean_text(self, text: Optional[str]) -> Optional[str]:
        """Clean and normalize text fields"""
        if not text:
            return None
        
        # Remove extra whitespace and clean up
        text = ' '.join(text.strip().split())
        
        # Remove common artifacts
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^[^\w]+|[^\w]+$', '', text)
        
        return text if text else None
    
    def _normalize_pos(self, pos: Optional[str]) -> Optional[str]:
        """Normalize part-of-speech tags for Kalenjin dictionary"""
        if not pos:
            return None
        
        pos = pos.strip().lower()
        
        # Standardize Kalenjin dictionary abbreviations
        pos_mapping = {
            'v.t.': 'verb transitive',
            'v.i.': 'verb intransitive', 
            'v.': 'verb',
            'n.': 'noun',
            'adj.': 'adjective',
            'adv.': 'adverb',
            'prep.': 'preposition',
            'conj.': 'conjunction',
            'interj.': 'interjection',
            'pron.': 'pronoun'
        }
        
        return pos_mapping.get(pos, pos)
    
    def batch_parse_images(self, image_paths: List[str], method: str = "batch") -> List[ExtractionResult]:
        """
        Parse multiple images in batch
        
        Args:
            image_paths: List of image file paths
            method: Parsing method to use
            
        Returns:
            List of ExtractionResult objects
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {Path(image_path).name}")
            
            start_time = time.time()
            
            try:
                # Parse entries from image
                entries_data = self.parse_image(image_path, method)
                
                # Convert to DictionaryEntry objects
                entries = [DictionaryEntry.from_dict(data) for data in entries_data]
                
                # Create result
                result = ExtractionResult(
                    source_image=image_path,
                    entries=entries,
                    page_number=i + 1,
                    processing_time=time.time() - start_time,
                    model_confidence=self._calculate_batch_confidence(entries),
                    extraction_method=method
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                
                error_result = ExtractionResult(
                    source_image=image_path,
                    entries=[],
                    page_number=i + 1,
                    processing_time=time.time() - start_time,
                    model_confidence=0.0,
                    extraction_method=method
                )
                error_result.add_error(str(e))
                results.append(error_result)
        
        return results
    
    def _calculate_batch_confidence(self, entries: List[DictionaryEntry]) -> float:
        """Calculate overall confidence for a batch of entries"""
        if not entries:
            return 0.0
        
        total_confidence = sum(entry.confidence_score for entry in entries)
        return total_confidence / len(entries)
    
    def set_vlm_processor(self, processor):
        """Set the VLM processor instance"""
        self._vlm_processor = processor
