"""
Prompt templates for OCR text processing.
Simple prompts for extracting dictionary entries from OCR'd text.
"""

# Main extraction prompt for OCR text
EXTRACTION_PROMPT = """You are processing OCR-extracted text from a Kalenjin dictionary page. 

Extract dictionary entries as a clean JSON array. Return ONLY the JSON, no additional text.

OCR Text:
{ocr_text}

CRITICAL REQUIREMENTS:
1. Your response must be ONLY a valid JSON array - no explanations, comments, or extra text
2. ALWAYS extract phonetic transcriptions (IPA) when present - this is MANDATORY
3. Look carefully for pronunciation guides in /slashes/, [brackets], or _underscores_
4. If no clear phonetic transcription is visible, set ipa to null

Instructions for extraction:
- Extract clean Kalenjin words as graphemes
- Extract corresponding English meanings/translations  
- MANDATORY: Find and extract IPA phonetic transcriptions (usually in /slashes/, [brackets], or _underscores_)
- Include part of speech markers (v.t., v.i., n., adj., etc.) in the english_meaning
- Skip headers, page numbers, and non-dictionary content

Output format - return ONLY this JSON structure:
[
  {{
    "grapheme": "kalenjin_word",
    "english_meaning": "english translation with grammar info",
    "ipa": "/phonetic_transcription/"
  }}
]

REMINDER: Extract phonetic transcriptions whenever they appear in the text. Look for patterns like:
- /ke:-apus/ 
- [pronunciation]
- _phonetic_guide_

JSON ARRAY ONLY:"""


def get_extraction_prompt(ocr_text: str) -> str:
    """
    Get the main extraction prompt with OCR text inserted.
    
    Args:
        ocr_text: Raw OCR-extracted text from dictionary page
        
    Returns:
        Complete prompt for LLM processing
    """
    return EXTRACTION_PROMPT.format(ocr_text=ocr_text)


def get_system_prompt() -> str:
    """
    Get the system prompt for the LLM.
    
    Returns:
        System prompt defining the assistant's role
    """
    return """You are an expert linguistic assistant specializing in dictionary data extraction. 
You process OCR-extracted text and identify dictionary entries with high accuracy. 
You output clean, structured JSON and ignore non-dictionary content like headers, page numbers, and formatting artifacts."""


# Validation prompt for checking extracted entries
VALIDATION_PROMPT = """Review these extracted dictionary entries for accuracy:

{entries}

Check for:
1. Valid Kalenjin words (graphemes) 
2. Meaningful English translations
3. Proper IPA formatting (if present)
4. No OCR artifacts or formatting errors

Return only the valid, clean entries in the same JSON format."""


def get_validation_prompt(entries: str) -> str:
    """
    Get validation prompt with entries to check.
    
    Args:
        entries: JSON string of extracted entries
        
    Returns:
        Validation prompt
    """
    return VALIDATION_PROMPT.format(entries=entries)
