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
        
        # Enhanced regex patterns based on actual Kalenjin dictionary analysis
        self.patterns = {
            # Real IPA patterns from dictionary: /_abus-é:t, _abus-0/, /_apus-nat-é:t/
            'ipa_pattern': r'(/[_a-zA-Z:\-é,\s]+/)',
            
            # Real Kalenjin headwords: abuset, abusnatet, abutanut, ach, acha, etc.
            'headword_pattern': r'^[a-zA-Z][a-zA-Z-]*[a-zA-Z]?$',
            
            # Complete grammar markers from actual dictionary
            'grammar_markers': r'\b(n\.|v\.|v\.caus\.|v\.t\.|v\.i\.|pn\.|adv\.|i\.|num\.|p\.)\b',
            
            # Artifacts to skip
            'artifact_pattern': r'^[^a-zA-Z]*$|^\d+$|^[IV]+\.$|^Nandi|^English|^\d+\s*$',
            
            # Example sentence indicators to skip (including real sentence patterns)
            'example_indicators': r'\b(The|He|She|It|I|You|We|They|There|What|Where|When|How|Koaget|Konu|Give|Owendi|Acha|mete)\b',
            
            # Example sentence pattern: "Kalenjin. English. /ipa/"
            'example_sentence_pattern': r'^[A-Z][a-zA-Z\s]+\.\s+[A-Z][a-zA-Z\s\(\)]+\.\s*/[^/]+/$',
            
            # Real Kalenjin words (3-8 chars typical)
            'kalenjin_word': r'\b[a-zA-Z-]{3,12}\b',
            
            # Cross-reference pattern: "(see yai)", "[< Swa. hela tano]"
            'cross_reference': r'\([^)]*see\s+[^)]*\)|\[[^\]]*<[^\]]*\]'
        }
    
    def parse_ocr_text(self, ocr_text: str, vllm_client) -> Dict[str, Any]:
        """
        Parse both dictionary entries AND example sentences from OCR text
        
        Args:
            ocr_text: Raw text extracted from OCR
            vllm_client: vLLM client for text completion
            
        Returns:
            Dictionary with both entries and example sentences
        """
        if not ocr_text or not ocr_text.strip():
            logger.warning("Empty OCR text provided")
            return {"entries": [], "examples": []}
        
        # Clean OCR text but preserve example sentences
        cleaned_text = self._clean_ocr_text_preserve_examples(ocr_text)
        
        if not cleaned_text:
            logger.warning("No valid text after cleaning OCR output")
            return {"entries": [], "examples": []}
        
        # Extract example sentences first (they're gold!)
        example_sentences = self._extract_example_sentences(cleaned_text)
        
        # Extract dictionary entries
        dictionary_entries = self._extract_dictionary_entries(cleaned_text, vllm_client)
        
        logger.info(f"Extracted {len(dictionary_entries)} dictionary entries and {len(example_sentences)} example sentences")
        
        return {
            "entries": dictionary_entries,
            "examples": example_sentences
        }
    
    def _extract_example_sentences(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract example sentences: Kalenjin sentence + English translation + IPA
        
        Args:
            text: Cleaned OCR text
            
        Returns:
            List of example sentence objects
        """
        examples = []
        lines = text.split('\n')
        
        # Pattern: "Kalenjin sentence. English translation. /ipa_pronunciation/"
        example_patterns = [
            r'^([A-Z][a-zA-Z\s]+)\.\s+([A-Z][a-zA-Z\s\(\)\']+)\.\s+(/[^/]+/)$',
            r'^([A-Z][a-zA-Z\s,!]+!?)\s+([A-Z][a-zA-Z\s\(\),!\']+!?)\s+(/[^/]+/)$'
        ]
        
        for line in lines:
            line = line.strip()
            if len(line) < 10:
                continue
            
            for pattern in example_patterns:
                match = re.match(pattern, line)
                if match:
                    kalenjin_sentence = match.group(1).strip()
                    english_translation = match.group(2).strip()
                    ipa_pronunciation = match.group(3).strip()
                    
                    example = {
                        'kalenjin': kalenjin_sentence,
                        'english': english_translation,
                        'ipa': ipa_pronunciation,
                        'source_line': line,
                        'confidence': 0.9  # High confidence for pattern matches
                    }
                    
                    examples.append(example)
                    logger.debug(f"Found example sentence: {kalenjin_sentence} → {english_translation}")
                    break
        
        logger.info(f"Extracted {len(examples)} example sentences")
        return examples
    
    def _extract_dictionary_entries(self, text: str, vllm_client) -> List[Dict[str, Any]]:
        """
        Extract dictionary entries using vLLM
        
        Args:
            text: Cleaned OCR text
            vllm_client: vLLM client
            
        Returns:
            List of dictionary entries
        """
        # Filter out example sentences for dictionary extraction
        filtered_text = self._filter_examples_for_dictionary(text)
        
        # Get extraction prompt
        system_prompt = get_system_prompt()
        user_prompt = get_extraction_prompt(filtered_text)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        try:
            # Process with vLLM
            response = vllm_client.complete_text(
                prompt=full_prompt,
                max_tokens=4000,
                temperature=0.1
            )
            
            if not response or response.get('status') != 'success':
                error_msg = response.get('error', 'Unknown error') if response else 'No response'
                logger.error(f"vLLM completion failed: {error_msg}")
                return []
            
            completion = response.get('completion', '')
            if not completion:
                logger.error("Empty completion from vLLM")
                return []
            
            # Parse JSON response
            entries = self._parse_json_response(completion)
            
            # Validate entries
            valid_entries = []
            for entry in entries:
                cleaned_entry = self._validate_entry(entry)
                if cleaned_entry:
                    valid_entries.append(cleaned_entry)
            
            return valid_entries
            
        except Exception as e:
            logger.error(f"Dictionary extraction failed: {e}")
            return []
    
    def _clean_ocr_text_preserve_examples(self, text: str) -> str:
        """
        Clean OCR text while preserving example sentences (they're gold!)
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text with example sentences preserved
        """
        if not text:
            return ""
        
        cleaned = text.strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Light cleaning - preserve structure
        lines = cleaned.split('\n')
        clean_lines = [line.strip() for line in lines if len(line.strip()) > 2]
        
        return '\n'.join(clean_lines)
    
    def _filter_examples_for_dictionary(self, text: str) -> str:
        """
        Filter out example sentences when extracting dictionary entries
        """
        lines = text.split('\n')
        clean_lines = []
        
        example_patterns = [
            r'^[A-Z][a-zA-Z\s]+\.\s+[A-Z].*\.\s*/.*/$',  # "Koaget tuga. The cattle grazed. /ipa/"
            r'^\([A-Z].*\)\s+.*\.\s*/.*/$',              # "(S/he) grazed the cattle. /ipa/"
            r'^Give\s+\(me\).*\.\s*/.*/$',               # "Give (me) 3 ten cent pieces. /ipa/"
            r'^[A-Z][a-zA-Z\s,!]+!?\s+[A-Z].*!?\s*/.*/$' # "Acha, mete! No, leave (it)! /ipa/"
        ]
        
        for line in lines:
            line = line.strip()
            if len(line) < 3:
                continue
            
            # Skip if matches example sentence pattern
            if any(re.match(pattern, line) for pattern in example_patterns):
                continue
            
            clean_lines.append(line)
        
        return '\n'.join(clean_lines)
    
    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean OCR text to improve LLM processing, removing example sentences
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text with example sentences filtered out
        """
        if not text:
            return ""
        
        # Basic cleaning
        cleaned = text.strip()
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove common OCR artifacts
        cleaned = re.sub(r'[^\w\s\-/\[\]().,;:\'"]', ' ', cleaned)
        
        # Split into lines and filter
        lines = cleaned.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) < 3:  # Skip very short lines
                continue
            
            # Skip example sentences with pattern "Kalenjin. English. /ipa/"
            if re.match(self.patterns['example_sentence_pattern'], line):
                logger.debug(f"Filtered example sentence: {line}")
                continue
            
            # Skip lines that are clearly example sentences
            example_patterns = [
                r'^[A-Z][a-zA-Z\s]+\.\s+[A-Z].*\.\s*/.*/$',  # "Koaget tuga. The cattle grazed. /ipa/"
                r'^\([A-Z].*\)\s+.*\.\s*/.*/$',              # "(S/he) grazed the cattle. /ipa/"
                r'^Give\s+\(me\).*\.\s*/.*/$',               # "Give (me) 3 ten cent pieces. /ipa/"
            ]
            
            if any(re.match(pattern, line) for pattern in example_patterns):
                logger.debug(f"Filtered example by pattern: {line}")
                continue
            
            clean_lines.append(line)
        
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
        Enhanced validation based on actual Kalenjin dictionary patterns
        """
        if not isinstance(entry, dict):
            return None
        
        # Get fields (support both old and new formats)
        grapheme = entry.get('grapheme', '').strip()
        grammar = entry.get('grammar', '').strip()
        ipa = entry.get('ipa', '').strip()
        definition = entry.get('definition', entry.get('english_meaning', '')).strip()
        
        if not grapheme or not definition:
            logger.debug(f"Missing required fields: grapheme={grapheme}, definition={definition}")
            return None
        
        # Skip artifacts using real patterns
        if re.match(self.patterns['artifact_pattern'], grapheme):
            logger.debug(f"Skipping artifact: {grapheme}")
            return None
            
        # Validate real Kalenjin headword patterns (3-12 chars, from actual data)
        if not (3 <= len(grapheme) <= 12) or not re.match(self.patterns['headword_pattern'], grapheme):
            logger.debug(f"Invalid headword: {grapheme}")
            return None
            
        # Skip if headword contains numbers (only definitions should have numbers)
        if re.search(r'[0-9]', grapheme):
            logger.debug(f"Skipping headword with numbers: {grapheme}")
            return None
            
        # Skip obvious example sentences using real sentence patterns
        if re.search(self.patterns['example_indicators'], definition):
            logger.debug(f"Skipping example sentence: {definition[:50]}")
            return None
        
        # Skip example sentences with pattern "Kalenjin. English. /ipa/"
        if re.match(self.patterns['example_sentence_pattern'], definition):
            logger.debug(f"Skipping example sentence pattern: {definition}")
            return None
        
        # Skip if definition starts with common sentence patterns
        if definition.startswith(('The ', 'He ', 'She ', 'It ', 'I ', 'You ', 'We ', 'They ')):
            logger.debug(f"Skipping sentence starting definition: {definition[:50]}")
            return None
            
        # Validate grammar marker against real patterns
        if grammar and not re.search(self.patterns['grammar_markers'], grammar):
            logger.debug(f"Invalid grammar marker: {grammar}")
            # Don't reject, just clear invalid grammar
            grammar = ""
            
        # Enhanced IPA validation using real patterns
        clean_ipa = None
        if ipa:
            ipa_match = re.search(self.patterns['ipa_pattern'], ipa)
            if ipa_match:
                clean_ipa = ipa_match.group(1)
                logger.debug(f"Valid IPA found: {clean_ipa}")
            else:
                logger.debug(f"Invalid IPA format: {ipa}")
        
        # Skip definitions that are clearly too long (examples vs definitions)
        if len(definition) > 200:
            logger.debug(f"Definition too long, likely example: {definition[:50]}...")
            return None
            
        # Handle cross-references and etymology
        clean_definition = re.sub(self.patterns['cross_reference'], '', definition).strip()
        if not clean_definition:
            return None
            
        # Build validated entry with new structure
        clean_entry = {
            'grapheme': grapheme,
            'grammar': grammar,
            'ipa': clean_ipa,
            'definition': clean_definition,
            'confidence': self._calculate_enhanced_confidence(grapheme, grammar, clean_ipa, clean_definition)
        }
        
        return clean_entry
    
    def _calculate_enhanced_confidence(self, grapheme: str, grammar: str, ipa: str, definition: str) -> float:
        """
        Enhanced confidence calculation based on real dictionary patterns
        """
        score = 0.3  # Base score
        
        # Headword quality (real patterns: abuset, acha, aget)
        if 3 <= len(grapheme) <= 8 and re.match(r'^[a-zA-Z-]+$', grapheme):
            score += 0.2
            
        # Grammar marker presence (real markers: n., v., v.caus., etc.)
        if grammar and re.search(self.patterns['grammar_markers'], grammar):
            score += 0.2
            
        # IPA presence (major quality indicator)
        if ipa and re.search(self.patterns['ipa_pattern'], ipa):
            score += 0.3  # High value for IPA presence
            
        # Definition quality
        if 5 <= len(definition) <= 100 and any(word in definition.lower() for word in ['to', 'of', 'the', 'a', 'in']):
            score += 0.1
            
        # Bonus for common Kalenjin patterns
        if grapheme.startswith(('ab', 'ag', 'ach', 'ak', 'ai')):  # Common prefixes from real data
            score += 0.05
            
        return min(1.0, score)
    
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
