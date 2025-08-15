"""
Prompt templates for OCR text processing with Qwen3-8B.
Optimized for fast, direct extraction (thinking disabled).
"""

# Main extraction prompt for OCR text
# Main extraction prompt optimized for Qwen3-8B (thinking disabled)
EXTRACTION_PROMPT = """Extract dictionary entries from OCR text. Return only JSON.

DICTIONARY STRUCTURE:
- HEADWORDS: Kalenjin words (ak, akwai, ke-al) - NO NUMBERS
- GRAMMAR: c., p., a., v., n., adj., v.t., v.i.
- IPA: In /forward slashes/ like /ák/, /ákwá:y/
- DEFINITIONS: English meanings, preserve numbers (1., 2., 3.)

OCR TEXT:
{ocr_text}

RULES:
1. Extract main entries only, skip examples
2. Headwords = pure Kalenjin (no numbers)
3. Preserve numbered definitions (1. fat. 2. rich.)
4. IPA must be in /slashes/

JSON FORMAT:
[{{"grapheme": "headword", "english_meaning": "grammar + definition", "ipa": "/pronunciation/"}}]

JSON ONLY:"""


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
