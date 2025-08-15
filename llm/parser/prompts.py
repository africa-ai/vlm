"""
Prompt templates for OCR text processing.
Simple prompts for extracting dictionary entries from OCR'd text.
"""

# Main extraction prompt for OCR text
EXTRACTION_PROMPT = """You are processing OCR-extracted text from a Kalenjin dictionary page. 

Your task is to identify and extract dictionary entries in JSON format. Each entry should contain:
- grapheme: The Kalenjin word/term
- english_meaning: The English translation or definition  
- ipa: International Phonetic Alphabet transcription (if present)

OCR Text:
{ocr_text}

Instructions:
1. Look for dictionary entry patterns (word followed by definition)
2. Extract clean Kalenjin words as graphemes
3. Extract corresponding English meanings/translations
4. Include IPA transcriptions when present (usually in /slashes/ or [brackets])
5. Skip headers, page numbers, and non-dictionary content
6. Return only valid dictionary entries

Output format - JSON array:
[
  {{
    "grapheme": "kalenjin_word",
    "english_meaning": "english translation or definition",
    "ipa": "/phonetic_transcription/" (if available, otherwise null)
  }}
]

Extract dictionary entries from the OCR text above:"""


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
