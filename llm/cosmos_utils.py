"""
Cosmos-Reason1-7B specific utilities for enhanced dictionary processing
"""

import logging
from typing import Dict, List, Any, Optional
import json
import re

logger = logging.getLogger(__name__)

class CosmosReasoningProcessor:
    """Enhanced processing utilities for Cosmos-Reason1-7B model"""
    
    def __init__(self):
        self.reasoning_templates = {
            'systematic_analysis': self._create_systematic_prompt,
            'verification': self._create_verification_prompt,
            'completion': self._create_completion_prompt
        }
    
    def _create_systematic_prompt(self, base_prompt: str) -> str:
        """Create a systematic reasoning prompt"""
        return f"""<reasoning>
I need to approach this Kalenjin dictionary extraction task systematically:

1. VISUAL ANALYSIS: First, I'll examine the overall image structure
   - Identify the two-column layout
   - Locate entry boundaries and separators
   - Note any formatting patterns

2. ENTRY IDENTIFICATION: For each visible entry, I'll identify:
   - The main Kalenjin word (grapheme)
   - Part of speech indicators (v.t., v.i., n., adj.)
   - IPA transcriptions (in /slashes/ or _underscores_)
   - English definitions and translations
   - Any contextual examples or cross-references

3. STRUCTURED EXTRACTION: I'll organize findings into JSON format
   - Ensure each entry has all available fields
   - Maintain accuracy over completeness
   - Assign confidence scores based on clarity

4. QUALITY VERIFICATION: Finally, I'll review for:
   - Completeness of extraction
   - Consistency in formatting
   - Accuracy of transcriptions
</reasoning>

{base_prompt}

Please proceed with systematic analysis and extraction."""
    
    def _create_verification_prompt(self, extracted_data: str) -> str:
        """Create a verification prompt for extracted data"""
        return f"""<reasoning>
I need to verify the accuracy and completeness of this extracted dictionary data:

1. COMPLETENESS CHECK: Are all visible entries captured?
2. ACCURACY VERIFICATION: Are the transcriptions and meanings correct?
3. FORMAT CONSISTENCY: Is the JSON structure proper and consistent?
4. CONFIDENCE ASSESSMENT: How confident am I in each extraction?
</reasoning>

Please review and verify this extracted data from the Kalenjin dictionary:

{extracted_data}

Provide corrections, improvements, or confirmation of accuracy. Focus on:
- Missing entries that should be included
- Incorrect IPA transcriptions
- Inaccurate English meanings
- Formatting issues

Return the verified/corrected JSON array."""
    
    def _create_completion_prompt(self, partial_data: str) -> str:
        """Create a completion prompt for partially extracted data"""
        return f"""<reasoning>
I have partial dictionary data that may be incomplete. I need to:

1. ANALYZE GAPS: Identify what information might be missing
2. CROSS-REFERENCE: Look for patterns in existing data
3. COMPLETE ENTRIES: Fill in missing fields where possible
4. MAINTAIN ACCURACY: Only add information I'm confident about
</reasoning>

Here is partially extracted Kalenjin dictionary data:

{partial_data}

Please complete any missing information and ensure all entries are properly formatted. Focus on:
- Adding missing IPA transcriptions where visible
- Completing partial English definitions
- Adding context/examples where available
- Ensuring proper confidence scoring

Return the completed JSON array."""
    
    def enhance_prompt_with_reasoning(self, prompt: str, reasoning_type: str = 'systematic_analysis') -> str:
        """Enhance a regular prompt with reasoning capabilities"""
        
        if reasoning_type in self.reasoning_templates:
            return self.reasoning_templates[reasoning_type](prompt)
        else:
            # Default reasoning enhancement
            return f"""<reasoning>
Let me approach this task methodically and think through each step carefully.
I'll analyze the image systematically to extract accurate dictionary entries.
</reasoning>

{prompt}

Please provide detailed, accurate results with step-by-step reasoning."""
    
    def extract_reasoning_from_response(self, response: str) -> Dict[str, str]:
        """Extract reasoning and final answer from model response"""
        
        # Look for reasoning tags
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL | re.IGNORECASE)
        
        # Extract JSON data
        json_match = re.search(r'\[[\s\S]*\]', response)
        
        result = {
            'reasoning': reasoning_match.group(1).strip() if reasoning_match else '',
            'extracted_data': json_match.group(0) if json_match else '',
            'full_response': response
        }
        
        return result
    
    def validate_reasoning_quality(self, reasoning_text: str) -> Dict[str, Any]:
        """Evaluate the quality of reasoning provided"""
        
        quality_indicators = {
            'systematic_approach': bool(re.search(r'step|systematic|method|approach', reasoning_text, re.IGNORECASE)),
            'visual_analysis': bool(re.search(r'visual|image|examine|look', reasoning_text, re.IGNORECASE)),
            'linguistic_awareness': bool(re.search(r'ipa|phonetic|grapheme|linguistic', reasoning_text, re.IGNORECASE)),
            'accuracy_focus': bool(re.search(r'accura|precis|careful|thorough', reasoning_text, re.IGNORECASE)),
            'completeness_check': bool(re.search(r'complete|all|every|missing', reasoning_text, re.IGNORECASE))
        }
        
        quality_score = sum(quality_indicators.values()) / len(quality_indicators)
        
        return {
            'quality_score': quality_score,
            'indicators': quality_indicators,
            'reasoning_length': len(reasoning_text.split()),
            'has_structure': bool(re.search(r'\d+\.|•|-', reasoning_text))
        }
    
    def create_reasoning_chain(self, image_path: str, base_prompt: str) -> List[Dict[str, str]]:
        """Create a chain of reasoning prompts for complex extraction"""
        
        chain = [
            {
                'step': 'analysis',
                'prompt': self.enhance_prompt_with_reasoning(base_prompt, 'systematic_analysis'),
                'purpose': 'Initial systematic analysis and extraction'
            },
            {
                'step': 'verification', 
                'prompt': 'PLACEHOLDER_FOR_VERIFICATION',  # Will be filled with results from step 1
                'purpose': 'Verify and correct initial extraction'
            },
            {
                'step': 'completion',
                'prompt': 'PLACEHOLDER_FOR_COMPLETION',  # Will be filled with results from step 2
                'purpose': 'Complete any missing information'
            }
        ]
        
        return chain
    
    def format_cosmos_response(self, raw_response: str, image_path: str) -> Dict[str, Any]:
        """Format Cosmos model response for the framework"""
        
        # Extract reasoning and data
        parsed = self.extract_reasoning_from_response(raw_response)
        
        # Evaluate reasoning quality
        reasoning_quality = self.validate_reasoning_quality(parsed['reasoning']) if parsed['reasoning'] else {}
        
        # Try to parse JSON data
        extracted_entries = []
        if parsed['extracted_data']:
            try:
                extracted_entries = json.loads(parsed['extracted_data'])
                if not isinstance(extracted_entries, list):
                    extracted_entries = [extracted_entries]
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from response: {e}")
        
        return {
            'image_path': image_path,
            'entries': extracted_entries,
            'reasoning': parsed['reasoning'],
            'reasoning_quality': reasoning_quality,
            'raw_response': raw_response,
            'success': len(extracted_entries) > 0
        }

# Global instance for easy access
cosmos_processor = CosmosReasoningProcessor()
