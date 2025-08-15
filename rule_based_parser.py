
"""
Rule-Based Kalenjin Dictionary Parser
Fast, reliable, no-LLM approach using pure pattern matching
"""

import re
import json
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pytesseract
from PIL import Image
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KalenjinDictionaryParser:
    """Rule-based parser for Kalenjin dictionary pages - FAST & RELIABLE"""
    
    def __init__(self):
        # Configure Tesseract with correct path
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Grammar markers found in your dictionary
        self.grammar_markers = {
            'n.': 'noun',
            'v.': 'verb', 
            'v.t.': 'transitive verb',
            'v.i.': 'intransitive verb',
            'v.caus.': 'causative verb',
            'adv.': 'adverb',
            'p.': 'preposition',
            'i.': 'interjection',
            'pn.': 'pronoun',
            'num.': 'numeral',
            'af.': 'affix',
            'c.': 'conjunction'
        }
        
        # Common OCR corrections for this dictionary
        self.ocr_corrections = {
            ' _': ' ',      # Remove underscore artifacts
            '_ ': ' ',      # Remove underscore artifacts
            '_': '',        # Remove remaining underscores
            '6': 'ó',       # Common OCR error for accented o
            'Us': 'ús',     # Fix capital U issues
            'Us-': 'ús-',   # Fix capital U in compounds
            ' / ': ' /',    # Fix IPA spacing
            '/ ': '/',      # Fix IPA spacing
            'é:t,': 'é:t, ',  # Fix comma spacing
            'é:t_': 'é:t, ', # Fix underscore in IPA
        }
        
        # Headword validation patterns
        self.headword_pattern = re.compile(r'^[a-z][\w\-]*$', re.IGNORECASE)
        
        # IPA pattern - more flexible for various formats
        self.ipa_pattern = re.compile(r'/[^/]+/')
        
        # Example sentence pattern: "Kalenjin. English. /ipa/"
        self.sentence_pattern = re.compile(r'([A-Z][^.]+\.)(\s+[A-Z][^.]+\.)(\s*/[^/]+/)')
        
    def extract_text(self, image_path: str) -> str:
        """Extract text with optimal Tesseract settings"""
        try:
            img = Image.open(image_path)
            if img.mode != 'L':
                img = img.convert('L')
            
            # High accuracy config - same as your working OCR setup
            config = r'--oem 1 --psm 3 -c preserve_interword_spaces=1'
            text = pytesseract.image_to_string(img, config=config)
            
            logger.info(f"Extracted {len(text)} characters from {Path(image_path).name}")
            return self.clean_ocr_text(text)
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""
    
    def clean_ocr_text(self, text: str) -> str:
        """Clean and normalize OCR text"""
        lines = text.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip unwanted content
            if self.should_skip_line(line):
                continue
            
            # Apply OCR corrections
            for error, correction in self.ocr_corrections.items():
                line = line.replace(error, correction)
            
            # Normalize spacing
            line = re.sub(r'\s+', ' ', line).strip()
            
            if line:  # Only add non-empty lines
                clean_lines.append(line)
        
        cleaned = '\n'.join(clean_lines)
        logger.info(f"Cleaned text: {len(cleaned)} characters ({len(clean_lines)} lines)")
        return cleaned
    
    def should_skip_line(self, line: str) -> bool:
        """Determine if a line should be skipped during cleaning"""
        if len(line) < 2:
            return True
        
        # Skip headers and page indicators
        lower_line = line.lower()
        skip_patterns = [
            'nandi', 'english', 'page', '—', 
            'kalenjin', 'dictionary'
        ]
        
        for pattern in skip_patterns:
            if pattern in lower_line:
                return True
        
        # Skip pure numbers (page numbers)
        if line.strip().isdigit():
            return True
        
        # Skip lines with only punctuation/symbols
        if re.match(r'^[^\w]+$', line):
            return True
        
        return False
    
    def parse_dictionary_entries(self, text: str) -> List[Dict]:
        """Extract dictionary entries using pattern matching"""
        entries = []
        lines = text.split('\n')
        current_entry = None
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            
            # Check if this starts a new dictionary entry
            entry_match = self.match_dictionary_entry(line)
            if entry_match:
                # Save previous entry
                if current_entry:
                    entries.append(self.finalize_entry(current_entry))
                
                # Start new entry
                current_entry = entry_match
                current_entry['line_number'] = i + 1
            
            elif current_entry and not self.is_example_sentence(line) and not self.is_new_entry_starting(line):
                # Continue building current entry (multi-line definitions)
                # BUT only if this line doesn't look like a new entry starting
                self.extend_entry_definition(current_entry, line)
        
        # Don't forget the last entry
        if current_entry:
            entries.append(self.finalize_entry(current_entry))
        
        logger.info(f"Extracted {len(entries)} dictionary entries")
        return entries
    
    def match_dictionary_entry(self, line: str) -> Optional[Dict]:
        """Check if line starts a new dictionary entry"""
        
        # Create pattern for: headword + grammar marker + rest
        grammar_pattern = '|'.join(re.escape(marker) for marker in self.grammar_markers.keys())
        pattern = rf'^([a-z][\w\-]*(?:\s+[a-z][\w\-]*)*)\s+({grammar_pattern})\s*(.*)'
        
        match = re.match(pattern, line, re.IGNORECASE)
        if match:
            headword = match.group(1).strip()
            grammar = match.group(2)
            rest = match.group(3).strip()
            
            # Validate headword (should be reasonable Kalenjin word)
            if self.is_valid_headword(headword):
                # Extract IPA and definition
                ipa, definition = self.extract_ipa_and_definition(rest)
                
                return {
                    'headword': headword,
                    'grammar': grammar,
                    'grammar_full': self.grammar_markers.get(grammar, grammar),
                    'ipa': ipa,
                    'definition': definition,
                    'raw_line': line
                }
        
        return None
    
    def is_valid_headword(self, word: str) -> bool:
        """Validate that this looks like a real Kalenjin headword"""
        # Basic validation
        if not word or len(word) < 2 or len(word) > 20:
            return False
        
        # Should be mostly lowercase letters with possible hyphens
        if not re.match(r'^[a-z][\w\-]*$', word, re.IGNORECASE):
            return False
        
        # Exclude obvious OCR artifacts
        artifacts = ['ii', 'iii', 'iv', 'vii', 'viii', 'ix', 'xi']
        if word.lower() in artifacts:
            return False
        
        return True
    
    def is_new_entry_starting(self, line: str) -> bool:
        """Check if this line looks like it might be starting a new entry"""
        line = line.strip()
        
        # If line starts with a lowercase word that could be a headword
        # This catches cases where OCR split the grammar marker to next line
        words = line.split()
        if len(words) > 0:
            first_word = words[0]
            
            # Looks like a potential headword
            if (len(first_word) >= 3 and 
                first_word.islower() and 
                re.match(r'^[a-z][\w\-]*$', first_word) and
                not first_word in ['the', 'and', 'or', 'but', 'to', 'of', 'in', 'on', 'at', 'by']):
                
                # Check if rest of line has grammar marker nearby
                rest_line = ' '.join(words[1:]) if len(words) > 1 else ''
                grammar_pattern = '|'.join(re.escape(marker) for marker in self.grammar_markers.keys())
                
                # Grammar marker in this line or looks like standalone headword
                if re.search(grammar_pattern, rest_line) or len(words) == 1:
                    return True
        
        return False
    
    def extract_ipa_and_definition(self, text: str) -> Tuple[str, str]:
        """Extract IPA transcription and definition from text - handles incomplete patterns"""
        ipa = ""
        definition = text
        
        # Find complete IPA patterns /like_this/
        ipa_matches = self.ipa_pattern.findall(text)
        if ipa_matches:
            ipa = ', '.join(ipa_matches)
            # Remove complete IPA from definition
            for ipa_match in ipa_matches:
                definition = definition.replace(ipa_match, '')
        
        # Handle incomplete IPA patterns (common with line breaks)
        # Look for patterns like "/word-é:t, word" (missing closing /)
        if not ipa:
            # More sophisticated pattern for incomplete IPA
            incomplete_pattern = re.search(r'/[^/\n]+(?:[,\s][^/\n]+)*(?=\s|$)', text.strip())
            if incomplete_pattern:
                potential_ipa = incomplete_pattern.group(0)
                # Add missing closing / if needed
                if not potential_ipa.endswith('/'):
                    potential_ipa += '/'
                logger.debug(f"Found incomplete IPA pattern: {potential_ipa}")
                ipa = potential_ipa
                definition = definition.replace(incomplete_pattern.group(0), '')
        
        # Also check if IPA is in the definition but was missed
        if not ipa and '/[' not in definition:  # Avoid bracket notation
            # Look for IPA patterns in the definition text
            def_ipa_matches = self.ipa_pattern.findall(definition)
            if def_ipa_matches:
                ipa = ', '.join(def_ipa_matches)
                # Remove from definition since it's now properly extracted
                for ipa_match in def_ipa_matches:
                    definition = definition.replace(ipa_match, '')
        
        # Clean up definition
        definition = re.sub(r'\s+', ' ', definition).strip()
        definition = re.sub(r'^[,\s\[\<]+', '', definition)  # Remove leading punctuation
        definition = re.sub(r'[,\s\]\>]+$', '', definition)  # Remove trailing punctuation
        
        # CRITICAL: Separate embedded example sentences from definition
        definition = self._separate_examples_from_definition(definition)
        
        return ipa, definition
    
    def _separate_examples_from_definition(self, definition: str) -> str:
        """Separate example sentences from definition text"""
        # Pattern: Definition ends at first period followed by capitalized sentence
        # e.g., "to abuse (verbally), yell at. Kéégutie légok. (S/he) yelled..."
        
        # Find first period followed by space and capital letter (start of example)
        example_start = re.search(r'\.\s+[A-Z][^.]*\.', definition)
        if example_start:
            # Extract just the definition part (before the example)
            definition = definition[:example_start.start() + 1]  # Include the period
        
        # Also remove any trailing IPA patterns that belong to examples
        definition = re.sub(r'\s*/[^/]+/\s*$', '', definition)
        
        return definition.strip()
    
    def extend_entry_definition(self, entry: Dict, line: str):
        """Extend current entry with continuation line - with STRONG boundary detection"""
        line = line.strip()
        if not line:
            return
        
        # STRONG boundary detection - look for clear new entry patterns
        # Pattern 1: "word grammar_marker" at start of line or after period
        boundary_patterns = [
            r'\b([a-z][\w\-]{2,11})\s+(n\.|v\.|v\.t\.|v\.i\.|v\.caus\.|adv\.|p\.|i\.|pn\.|num\.)\s',
            r'^\s*([a-z][\w\-]{2,11})\s+(n\.|v\.|v\.t\.|v\.i\.|v\.caus\.|adv\.|p\.|i\.|pn\.|num\.)',
        ]
        
        for pattern in boundary_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                # Found a new entry boundary
                boundary_pos = match.start()
                
                # Only take the part BEFORE the boundary
                if boundary_pos > 0:
                    definition_part = line[:boundary_pos].strip()
                    if definition_part:
                        if entry['definition']:
                            entry['definition'] += ' ' + definition_part
                        else:
                            entry['definition'] = definition_part
                
                # Log the boundary detection
                logger.debug(f"Entry boundary detected: '{match.group()}' - stopping definition extension")
                return  # Stop processing this line
        
        # Pattern 2: If line contains too many complete IPA patterns, it's likely merged entries
        ipa_count = len(self.ipa_pattern.findall(line))
        if ipa_count >= 2:
            # Multiple IPA patterns suggest merged entries
            logger.debug(f"Multiple IPA patterns detected ({ipa_count}) - likely merged entries")
            
            # Try to extract just the first meaningful part
            first_ipa_end = line.find('/')
            if first_ipa_end > 0:
                second_ipa_start = line.find('/', first_ipa_end + 1)
                if second_ipa_start > 0:
                    # Take everything up to the second IPA
                    definition_part = line[:second_ipa_start].strip()
                    if definition_part.endswith('/'):
                        definition_part = definition_part[:-1].strip()
                    
                    if entry['definition']:
                        entry['definition'] += ' ' + definition_part
                    else:
                        entry['definition'] = definition_part
                    return
        
        # Pattern 3: Check for obvious example sentence patterns and stop before them
        if re.search(r'[A-Z][a-z]+\s+[a-z]+\.\s+[A-Z]', line):
            # Looks like "Kalenjin word. English sentence." pattern
            logger.debug("Example sentence pattern detected - stopping definition extension")
            return
        
        # No boundary detected, add the line (but with limits)
        if len(entry['definition']) > 200:  # Prevent extremely long definitions
            logger.debug("Definition getting too long - stopping extension")
            return
        
        if entry['definition']:
            entry['definition'] += ' ' + line
        else:
            entry['definition'] = line
    
    def finalize_entry(self, entry: Dict) -> Dict:
        """Final cleanup and validation of entry"""
        # Clean up definition
        entry['definition'] = re.sub(r'\s+', ' ', entry['definition']).strip()
        
        # Fix incomplete IPA patterns - check definition for completion
        if entry['ipa'] and not entry['ipa'].endswith('/'):
            logger.info(f"Attempting to complete incomplete IPA for {entry['headword']}: '{entry['ipa']}'")
            
            # Look for IPA completion in definition
            definition = entry['definition']
            
            # For patterns like "/apus-nat-é:t, apus-" find "nat; apls-nat-wa, apús-nat-wé:k/"
            if definition:
                # Try to find pattern that starts where IPA left off and ends with /
                completion_match = re.search(r'^([^/]*/)(?:\s*(.*))?', definition)
                if completion_match:
                    completion_part = completion_match.group(1)  # "nat; apls-nat-wa, apús-nat-wé:k/"
                    remaining_definition = completion_match.group(2) or ""  # "foolishness, sillyness."
                    
                    # Combine incomplete IPA with completion
                    entry['ipa'] = entry['ipa'] + completion_part
                    entry['definition'] = remaining_definition.strip()
                    
                    logger.info(f"Completed IPA: '{entry['ipa']}'")
                    logger.info(f"Updated definition: '{entry['definition']}'")
                else:
                    # Fallback: just add closing /
                    entry['ipa'] = self.complete_incomplete_ipa(entry['ipa'])
            else:
                entry['ipa'] = self.complete_incomplete_ipa(entry['ipa'])
        
        # If no IPA was extracted but definition contains IPA patterns, extract them
        elif not entry['ipa'] and entry['definition']:
            ipa_in_def = self.ipa_pattern.findall(entry['definition'])
            if ipa_in_def:
                # Found IPA in definition, extract it
                entry['ipa'] = ', '.join(ipa_in_def)
                # Remove from definition
                for ipa_match in ipa_in_def:
                    entry['definition'] = entry['definition'].replace(ipa_match, '')
                entry['definition'] = re.sub(r'\s+', ' ', entry['definition']).strip()
                logger.debug(f"Extracted IPA from definition for {entry['headword']}: {entry['ipa']}")
        
        # Separate example sentences from definition
        entry['definition'] = self._separate_examples_from_definition(entry['definition'])
        
        # Add metadata
        entry['confidence'] = self.calculate_entry_confidence(entry)
        entry['word_length'] = len(entry['headword'])
        
        return entry
    
    def complete_incomplete_ipa(self, ipa_text: str) -> str:
        """Complete incomplete IPA patterns that are missing closing /"""
        if not ipa_text:
            return ipa_text
        
        # Handle patterns like "/apus-nat-é:t, apus-" (missing closing /)
        if ipa_text.startswith('/') and not ipa_text.endswith('/'):
            # Check if it looks incomplete (ends with dash, comma, etc.)
            if re.search(r'[,\-]\s*$', ipa_text):
                # Try to find a reasonable completion
                # Look for similar pattern in the definition
                if '-' in ipa_text:
                    # For "/apus-nat-é:t, apus-" -> "/apus-nat-é:t, apus-nat/"
                    base_pattern = re.search(r'/([^,]+)', ipa_text)
                    if base_pattern:
                        base = base_pattern.group(1)
                        if '-' in base:
                            # Extract the root pattern and apply it
                            root_parts = base.split('-')
                            if len(root_parts) >= 2:
                                # Complete with same pattern
                                ipa_text = ipa_text.rstrip(',- ') + '-nat/'
                
                # If still no closing /, add one
                if not ipa_text.endswith('/'):
                    ipa_text += '/'
        
        return ipa_text
    
    def calculate_entry_confidence(self, entry: Dict) -> float:
        """Calculate confidence score for entry quality"""
        score = 0.5  # Base score
        
        # Has IPA transcription
        if entry['ipa']:
            score += 0.3
        
        # Has reasonable definition
        if entry['definition'] and len(entry['definition']) > 5:
            score += 0.2
        
        # Headword looks good
        if self.is_valid_headword(entry['headword']):
            score += 0.1
        
        # Grammar marker recognized
        if entry['grammar'] in self.grammar_markers:
            score += 0.1
        
        return min(1.0, score)
    
    def parse_example_sentences(self, text: str) -> List[Dict]:
        """Extract example sentences using pattern matching"""
        sentences = []
        
        # Find all example sentence patterns
        matches = self.sentence_pattern.findall(text)
        for match in matches:
            kalenjin = match[0].strip().rstrip('.')
            english = match[1].strip().rstrip('.')
            ipa = match[2].strip()
            
            # Validate sentence
            if len(kalenjin) > 5 and len(english) > 5:
                sentences.append({
                    'kalenjin': kalenjin,
                    'english': english,
                    'ipa': ipa,
                    'type': 'example_sentence',
                    'confidence': 0.9  # High confidence for clear pattern matches
                })
        
        logger.info(f"Extracted {len(sentences)} example sentences")
        return sentences
    
    def is_example_sentence(self, line: str) -> bool:
        """Check if line contains an example sentence"""
        return bool(self.sentence_pattern.search(line))
    
    def process_page(self, image_path: str, save_results: bool = True) -> Dict:
        """Process a complete dictionary page"""
        logger.info(f"🔍 Processing: {Path(image_path).name}")
        start_time = datetime.now()
        
        # Extract text
        text = self.extract_text(image_path)
        if not text:
            return {'error': 'Failed to extract text from image'}
        
        # Parse entries and sentences
        entries = self.parse_dictionary_entries(text)
        sentences = self.parse_example_sentences(text)
        
        # Calculate statistics
        high_confidence_entries = [e for e in entries if e.get('confidence', 0) > 0.7]
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'image_path': str(image_path),
            'dictionary_entries': entries,
            'example_sentences': sentences,
            'statistics': {
                'total_entries': len(entries),
                'high_confidence_entries': len(high_confidence_entries),
                'total_sentences': len(sentences),
                'processing_time_seconds': processing_time,
                'avg_confidence': sum(e.get('confidence', 0) for e in entries) / len(entries) if entries else 0
            },
            'raw_text': text,
            'method': 'Rule-Based Parser (No LLM)',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"📚 Found {len(entries)} dictionary entries ({len(high_confidence_entries)} high confidence)")
        logger.info(f"🗣️ Found {len(sentences)} example sentences")
        logger.info(f"⚡ Processing time: {processing_time:.2f} seconds")
        
        # Save results
        if save_results:
            self.save_results(result, image_path)
        
        return result
    
    def save_results(self, result: Dict, image_path: str):
        """Save results to JSON files"""
        # Create output directories
        output_dir = Path("rule_based_output")
        entries_dir = output_dir / "dictionary_entries"
        sentences_dir = output_dir / "example_sentences"
        
        for dir_path in [output_dir, entries_dir, sentences_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Generate filename
        image_name = Path(image_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete results
        complete_file = output_dir / f"{image_name}_complete_{timestamp}.json"
        with open(complete_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save dictionary entries separately
        if result['dictionary_entries']:
            entries_file = entries_dir / f"{image_name}_entries_{timestamp}.json"
            with open(entries_file, "w", encoding="utf-8") as f:
                json.dump(result['dictionary_entries'], f, indent=2, ensure_ascii=False)
        
        # Save example sentences separately
        if result['example_sentences']:
            sentences_file = sentences_dir / f"{image_name}_sentences_{timestamp}.json"
            with open(sentences_file, "w", encoding="utf-8") as f:
                json.dump(result['example_sentences'], f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Results saved to {output_dir}")
    
    def process_multiple_pages(self, image_paths: List[str]) -> Dict:
        """Process multiple dictionary pages"""
        all_results = []
        total_entries = 0
        total_sentences = 0
        
        for image_path in image_paths:
            result = self.process_page(image_path)
            if 'error' not in result:
                all_results.append(result)
                total_entries += len(result['dictionary_entries'])
                total_sentences += len(result['example_sentences'])
        
        summary = {
            'processed_pages': len(all_results),
            'total_entries': total_entries,
            'total_sentences': total_sentences,
            'results': all_results,
            'method': 'Rule-Based Parser (No LLM)',
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        output_dir = Path("rule_based_output")
        output_dir.mkdir(exist_ok=True)
        summary_file = output_dir / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"🎉 Batch processing complete: {total_entries} entries, {total_sentences} sentences")
        return summary


def main():
    """Test the rule-based parser"""
    parser = KalenjinDictionaryParser()
    
    # Test with your dictionary page
    test_image = "kalenjin_dictionary_page_002.png"
    
    if not Path(test_image).exists():
        logger.error(f"Test image not found: {test_image}")
        return
    
    # Process the page
    result = parser.process_page(test_image)
    
    if 'error' in result:
        logger.error(f"Processing failed: {result['error']}")
        return
    
    # Display results
    print("\n" + "="*60)
    print("🚀 RULE-BASED PARSER RESULTS")
    print("="*60)
    
    stats = result['statistics']
    print(f"⚡ Processing time: {stats['processing_time_seconds']:.2f} seconds")
    print(f"📚 Dictionary entries: {stats['total_entries']} ({stats['high_confidence_entries']} high confidence)")
    print(f"🗣️ Example sentences: {stats['total_sentences']}")
    print(f"🎯 Average confidence: {stats['avg_confidence']:.2f}")
    
    print("\n📚 SAMPLE DICTIONARY ENTRIES:")
    for i, entry in enumerate(result['dictionary_entries'][:8]):
        conf = f"({entry['confidence']:.2f})"
        print(f"{i+1:2d}. {entry['headword']:12} {entry['grammar']:8} {entry['ipa']:20} → {entry['definition'][:50]}... {conf}")
    
    if result['example_sentences']:
        print("\n🗣️ SAMPLE EXAMPLE SENTENCES:")
        for i, sentence in enumerate(result['example_sentences'][:5]):
            print(f"{i+1}. {sentence['kalenjin']} → {sentence['english']} {sentence['ipa']}")
    
    print(f"\n💾 Results saved to: rule_based_output/")


if __name__ == "__main__":
    main()
