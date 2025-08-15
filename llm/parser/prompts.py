"""
Prompts for Kalenjin Dictionary Extraction
Optimized for Qwen3-8B with compact, effective prompts to prevent timeouts
"""

# Ultra-compact extraction prompt for speed
EXTRACTION_PROMPT = """Extract dictionary entries as JSON:

{ocr_text}

JSON: [{{"word": "headword", "pos": "n.", "ipa": "/ipa/", "def": "meaning"}}]"""

# Compact sentence extraction prompt
SENTENCE_EXTRACTION_PROMPT = """Extract example sentences only. Return JSON only.

PATTERN: "Kalenjin sentence. English translation. /ipa_pronunciation/"

EXAMPLES:
• "Koaget tuga. The cattle grazed yesterday. /_ko:-aké:t _tu:-ka/"
• "Konu abutanosiek somok. Give (me) 3 ten cent pieces. /ko:n-U: _aputa:n-o:s-yék/"
• "Owendi agine. I'm going, too. /a-wé:n-ti: _akiné/"

TEXT:
{ocr_text}

FORMAT: [{{"kalenjin": "sentence", "english": "translation", "ipa": "/ipa/"}}]

JSON:"""


def get_extraction_prompt(ocr_text: str) -> str:
    """
    Generate the main extraction prompt with OCR text inserted.
    
    Args:
        ocr_text: OCR-extracted text to be processed
        
    Returns:
        Complete prompt for LLM processing
    """
    return EXTRACTION_PROMPT.format(ocr_text=ocr_text)


def get_sentence_extraction_prompt(ocr_text: str) -> str:
    """
    Generate the sentence extraction prompt with OCR text inserted.
    
    Args:
        ocr_text: OCR-extracted text to be processed
        
    Returns:
        Complete prompt for sentence extraction
    """
    return SENTENCE_EXTRACTION_PROMPT.format(ocr_text=ocr_text)


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
    Generate validation prompt for extracted entries.
    
    Args:
        entries: JSON string of extracted entries
        
    Returns:
        Complete validation prompt
    """
    return VALIDATION_PROMPT.format(entries=entries)
