"""
Prompt templates for OCR text processing with Qwen2.5-8B-Instruct.
Optimized prompts for extracting dictionary entries from OCR'd text.
"""

# Main extraction prompt for OCR text
# Main extraction prompt optimized for Qwen2.5-8B-Instruct
EXTRACTION_PROMPT = """<|im_start|>system
You are an expert Kalenjin dictionary digitization assistant. Extract dictionary entries from OCR text with perfect accuracy.

DICTIONARY STRUCTURE:
- HEADWORDS: Pure Kalenjin words (ak, akwai, ke-al, alamaliet)
- GRAMMAR: c., p., a., v., n., adj., v.t., v.i., v.itv.
- IPA: Always in /forward slashes/ like /ák/, /ákwá:y/, /ke:-ál/
- DEFINITIONS: English meanings, may have numbered variants (1., 2., 3.)

EXTRACTION RULES:
1. Extract ONLY main dictionary entries, skip examples/cross-references
2. Preserve numbered definitions (1., 2., 3.) for multiple meanings
3. IPA must be extracted from /forward slashes/
4. NO numbers in headwords (only in definitions)
<|im_end|>
<|im_start|>user
Extract dictionary entries from this OCR text:

{ocr_text}

Return valid JSON array only:
[
  {{
    "grapheme": "headword_only",
    "english_meaning": "grammar_marker complete_definition_with_numbers",
    "ipa": "/exact_pronunciation/"
  }}
]
<|im_end|>
<|im_start|>assistant
"""


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
