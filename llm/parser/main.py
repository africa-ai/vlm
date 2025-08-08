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
            'entry_line': r'^([a-zA-Z-]+)\s+(v\.t\.|v\.i\.|v\.|n\.|adj\.|adv\.|prep\.|conj\.|interj\.)\s*(/[^/]+/|_[^_]+_)?\s*(.+)$',
            'ipa_pattern': r'(/[^/]+/|_[^_]+_|\[[^\]]+\])',
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
        Process image with VLM (connected from main processor)
        
        Args:
            image_path: Path to image
            prompt: Text prompt
            
        Returns:
            VLM response text
        """
        if hasattr(self, '_vlm_processor') and self._vlm_processor:
            try:
                logger.info(f"Processing {image_path} with Cosmos-Reason1-7B VLM")
                return self._vlm_processor.process_image(image_path, prompt)
            except Exception as e:
                logger.error(f"VLM processing failed: {e}")
                logger.info("Falling back to mock parsing")
                return self._fallback_parse(image_path)
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
    
    def parse_vlm_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Public method to parse VLM response and extract structured data.
        This is the interface used by VLLMServerProcessor.
        
        Args:
            response: Raw VLM response text
            
        Returns:
            List of dictionary entry dictionaries
        """
        return self._parse_vlm_response(response)
    
    def calculate_confidence_score(self, entry: Dict[str, Any]) -> float:
        """
        Calculate confidence score based on extraction quality indicators
        
        Args:
            entry: Dictionary entry to evaluate
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 0.6  # Base score
        
        # Grapheme quality (20 points)
        grapheme = entry.get("grapheme", "").strip()
        if grapheme and len(grapheme) >= 2:
            score += 0.15
            if grapheme.isalpha() or '-' in grapheme:  # Valid Kalenjin word patterns
                score += 0.05
        
        # IPA presence and quality (15 points)
        ipa = entry.get("ipa", "")
        if ipa:
            score += 0.10
            if any(marker in str(ipa) for marker in ['/', '_', '[', ']']):  # Proper IPA markers
                score += 0.05
        
        # English meaning quality (20 points)
        meaning = entry.get("english_meaning", "").strip()
        if meaning and len(meaning) >= 5:
            score += 0.10
            if not any(artifact in meaning.lower() for artifact in ['"', 'null', 'json', '{']):
                score += 0.10  # No JSON artifacts
        
        # Part of speech (10 points)
        pos = entry.get("part_of_speech", "")
        if pos and pos.strip() in ['v.t.', 'v.i.', 'n.', 'adj.', 'adv.', 'v.', 'prep.']:
            score += 0.10
        
        # Context presence (10 points)
        context = entry.get("context", "")
        if context and len(str(context).strip()) > 3:
            score += 0.10
        
        # Penalize obvious errors (-20 points)
        if any(bad in str(entry).lower() for bad in ['entries":', '```', '"ipa":', 'null']):
            score -= 0.20
        
        # Ensure score stays in valid range
        return max(0.0, min(1.0, score))
    
    def _parse_vlm_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse VLM response and extract structured data with improved JSON handling
        
        Args:
            response: Raw VLM response text
            
        Returns:
            List of dictionary entry dictionaries
        """
        try:
            logger.debug(f"Parsing VLM response: {response[:200]}...")
            
            # Clean up the response text first
            cleaned_response = response.strip()
            
            # Try multiple JSON extraction strategies
            json_data = None
            
            # Strategy 1: Look for array in ```json blocks
            json_block_match = re.search(r'```json\s*(\[[\s\S]*?\])\s*```', cleaned_response, re.MULTILINE)
            if json_block_match:
                json_str = json_block_match.group(1)
                try:
                    json_data = json.loads(json_str)
                    logger.debug("Successfully parsed JSON from code block")
                except json.JSONDecodeError:
                    pass
            
            # Strategy 2: Look for any array structure
            if json_data is None:
                array_match = re.search(r'\[[\s\S]*\]', cleaned_response)
                if array_match:
                    json_str = array_match.group()
                    try:
                        json_data = json.loads(json_str)
                        logger.debug("Successfully parsed JSON from array match")
                    except json.JSONDecodeError:
                        pass
            
            # Strategy 3: Look for individual objects and combine them
            if json_data is None:
                object_matches = re.findall(r'\{[^{}]*\}', cleaned_response)
                if object_matches:
                    objects = []
                    for obj_str in object_matches:
                        try:
                            obj = json.loads(obj_str)
                            objects.append(obj)
                        except json.JSONDecodeError:
                            continue
                    if objects:
                        json_data = objects
                        logger.debug(f"Successfully parsed {len(objects)} JSON objects")
            
            # If we found JSON data, validate and return it
            if json_data:
                if isinstance(json_data, list):
                    # Filter out invalid entries (like JSON syntax fragments)
                    valid_entries = []
                    for entry in json_data:
                        if self._is_valid_entry(entry):
                            # Calculate real confidence score
                            entry["confidence_score"] = self.calculate_confidence_score(entry)
                            valid_entries.append(entry)
                    
                    logger.info(f"Extracted {len(valid_entries)} valid entries from JSON")
                    return valid_entries
                elif isinstance(json_data, dict):
                    if self._is_valid_entry(json_data):
                        json_data["confidence_score"] = self.calculate_confidence_score(json_data)
                        return [json_data]
            
            # If JSON parsing failed, try intelligent text parsing
            logger.warning("JSON parsing failed, attempting intelligent text parsing")
            return self._parse_text_response_intelligent(cleaned_response)
            
        except Exception as e:
            logger.error(f"Error parsing VLM response: {e}")
            return []
    
    def _is_valid_entry(self, entry: Dict[str, Any]) -> bool:
        """
        Check if an entry is a valid dictionary entry (not JSON syntax)
        
        Args:
            entry: Dictionary entry to validate
            
        Returns:
            True if valid dictionary entry
        """
        if not isinstance(entry, dict):
            return False
        
        # Check for JSON syntax artifacts
        grapheme = entry.get('grapheme', entry.get('term', entry.get('kalenjin_word', '')))
        if not grapheme or len(grapheme) < 2:
            return False
        
        # Reject JSON syntax fragments
        json_artifacts = ['"entries":', '"ipa":', '"englishMeaning":', '"originalGrapheme":', 
                         '```', '{', '}', '[', ']', '",', 'null']
        
        if any(artifact in str(grapheme) for artifact in json_artifacts):
            return False
        
        # Must have some meaningful content
        english = entry.get('english_meaning', entry.get('definition', entry.get('english', '')))
        if not english or len(english) < 2:
            return False
        
        # Reject if english meaning contains JSON artifacts
        if any(artifact in str(english) for artifact in json_artifacts):
            return False
        
        # Extract and clean IPA if present
        if 'ipa' in entry and entry['ipa']:
            ipa = str(entry['ipa']).strip()
            # Clean common IPA formatting variations
            if ipa and ipa != 'null' and not any(artifact in ipa for artifact in json_artifacts):
                # Ensure IPA has proper formatting
                if not (ipa.startswith('/') or ipa.startswith('_') or ipa.startswith('[')):
                    # Try to extract IPA from the text
                    ipa_match = re.search(self.patterns['ipa_pattern'], ipa)
                    if ipa_match:
                        entry['ipa'] = ipa_match.group()
        
        return True
    
    def _parse_text_response_intelligent(self, text: str) -> List[Dict[str, Any]]:
        """
        Intelligent plain text response parsing for dictionary entries
        
        Args:
            text: Plain text response
            
        Returns:
            List of parsed entries
        """
        entries = []
        lines = text.strip().split('\n')
        
        current_entry = {}
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue
            
            # Skip JSON artifacts
            if any(artifact in line for artifact in ['"entries":', '```', '{', '}', 'null']):
                continue
            
            # Look for dictionary entry patterns
            # Pattern 1: "kalenjin_word - english_meaning"
            if ' - ' in line:
                parts = line.split(' - ', 1)
                if len(parts) == 2 and len(parts[0].strip()) > 1:
                    if current_entry:
                        entries.append(current_entry)
                    
                    # Extract IPA from kalenjin part if present
                    kalenjin_part = parts[0].strip()
                    ipa_match = re.search(self.patterns['ipa_pattern'], kalenjin_part)
                    ipa = ipa_match.group() if ipa_match else None
                    
                    # Clean grapheme (remove IPA if found)
                    grapheme = kalenjin_part
                    if ipa:
                        grapheme = grapheme.replace(ipa, '').strip()
                    
                    current_entry = {
                        'grapheme': grapheme.split()[0] if grapheme.split() else grapheme,
                        'ipa': ipa,
                        'english_meaning': parts[1].strip(),
                        'confidence_score': 0.8
                    }
            
            # Pattern 2: "word: definition"
            elif ':' in line and not line.startswith('"'):
                parts = line.split(':', 1)
                if len(parts) == 2 and len(parts[0].strip()) > 1:
                    if current_entry:
                        entries.append(current_entry)
                    
                    # Extract IPA from kalenjin part if present
                    kalenjin_part = parts[0].strip()
                    ipa_match = re.search(self.patterns['ipa_pattern'], kalenjin_part)
                    ipa = ipa_match.group() if ipa_match else None
                    
                    # Clean grapheme
                    grapheme = kalenjin_part
                    if ipa:
                        grapheme = grapheme.replace(ipa, '').strip()
                    
                    current_entry = {
                        'grapheme': grapheme.split()[0] if grapheme.split() else grapheme,
                        'ipa': ipa,
                        'english_meaning': parts[1].strip(),
                        'confidence_score': 0.8
                    }
            
            # Pattern 3: Traditional dictionary format "word pos /ipa/ definition"
            else:
                entry = self._parse_entry_line(line)
                if entry and entry.get('grapheme') and len(entry.get('grapheme', '')) > 1:
                    if current_entry:
                        entries.append(current_entry)
                    current_entry = entry
        
        # Add the last entry
        if current_entry and current_entry.get('grapheme'):
            entries.append(current_entry)
        
        # Filter out any remaining artifacts and add confidence scores
        valid_entries = []
        for entry in entries:
            if self._is_valid_entry(entry):
                entry["confidence_score"] = self.calculate_confidence_score(entry)
                valid_entries.append(entry)
        
        # Filter out very low confidence entries (< 0.5)
        high_confidence_entries = [
            entry for entry in valid_entries 
            if entry.get("confidence_score", 0) >= 0.5
        ]
        
        logger.info(f"Text parsing: {len(entries)} total entries, {len(high_confidence_entries)} high-confidence entries kept")
        return high_confidence_entries

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
        Parse a single dictionary entry line with enhanced IPA extraction
        
        Args:
            line: Text line containing dictionary entry
            
        Returns:
            Parsed entry dictionary or None
        """
        try:
            # Extract IPA transcription (multiple patterns)
            ipa_match = re.search(self.patterns['ipa_pattern'], line)
            ipa = ipa_match.group() if ipa_match else None
            
            # Extract part of speech
            pos_match = re.search(self.patterns['pos_pattern'], line)
            pos = pos_match.group() if pos_match else None
            
            # Extract grapheme (first word, typically)
            words = line.split()
            if not words:
                return None
            
            grapheme = words[0].rstrip('.,;:')
            
            # Extract English meaning (everything after IPA and POS)
            meaning_text = line
            
            # Remove grapheme from beginning
            if grapheme in meaning_text:
                meaning_text = meaning_text[meaning_text.index(grapheme) + len(grapheme):].strip()
            
            # Remove POS if present
            if pos and pos in meaning_text:
                meaning_text = meaning_text.replace(pos, '', 1).strip()
            
            # Remove IPA if present  
            if ipa and ipa in meaning_text:
                meaning_text = meaning_text.replace(ipa, '', 1).strip()
            
            # Clean up the meaning
            meaning_text = meaning_text.strip('.,;: ')
            
            if not meaning_text or len(meaning_text) < 2:
                return None
            
            return {
                'grapheme': grapheme,
                'ipa': ipa,
                'english_meaning': meaning_text,
                'part_of_speech': pos,
                'context': None,
                'confidence_score': 0.8  # Will be recalculated by confidence scorer
            }
            
        except Exception as e:
            logger.debug(f"Error parsing line '{line}': {e}")
            return None
            
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
