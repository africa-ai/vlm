"""
Prompt templates for OCR text processing with Qwen3-8B.
Optimized for fast, direct extraction (thinking disabled).
"""

# Main extraction prompt for OCR text
# Main extraction prompt optimized for Qwen3-8B based on actual dictionary analysis
EXTRACTION_PROMPT = """Extract dictionary entries from this Kalenjin-English dictionary text. Use the EXACT patterns shown below.

KALENJIN DICTIONARY STRUCTURE (from actual page):
• HEADWORDS: abuset, abusnatet, abutanut, ach, acha, achek, agai, age, agenge, aget, agine, agutaniet, agutie, aita, aiya, aiyeb, ak
• GRAMMAR MARKERS: n. (noun), v. (verb), v.caus. (causative verb), v.t. (transitive verb), pn. (pronoun), adv. (adverb), i. (interjection), num. (numeral), p. (preposition)
• IPA FORMAT: /_abus-é:t, _abus-0/, /_apus-nat-é:t/, /_a:c/, /_a:ca, _a:cica/, /_acé:k/, /_akay/, /_aké/, /_aké:nke/, /_kw-aké:t/, /_ki:-aké:t/, /_akiné/, /_akuta:n-yét/, /ke:-aku:tyé:/, /_ay-ta/, /_a:yya/, /_ayya/, /_ayye:p/

IMPORTANT - EXAMPLE SENTENCES (DO NOT EXTRACT AS ENTRIES):
Example sentences appear within entries following pattern: KALENJIN_SENTENCE. English translation. /ipa_pronunciation/
• "Koaget tuga. The cattle grazed yesterday. /_ko:-aké:t _tu:-ka/"
• "Konu abutanosiek somok. Give (me) 3 ten cent pieces. /ko:n-U: _aputa:n-o:s-yék/"
• "Owendi agine. I'm going, too. /a-wé:n-ti: _akiné/"
• "Acha, mete! No, leave (it)! /_a:ca meté:/"

OCR TEXT TO PROCESS:
{ocr_text}

EXTRACTION RULES:
1. HEADWORDS are standalone words at line start: "abuset", "acha", "aget", etc.
2. GRAMMAR immediately follows headword: "abuset n.", "ach p.", "aget v."  
3. IPA in /slashes/ with underscores and colons: /_abus-é:t, _abus-0/
4. DEFINITIONS follow IPA, may include examples
5. SKIP: page numbers (26), headers (Nandi — English), cross-references in parentheses
6. SKIP: Example sentences with pattern "Kalenjin. English. /ipa/"
7. HANDLE VARIANTS: Some entries have multiple forms (kw-aget, ki-aget)

EXACT FORMAT FROM REAL DICTIONARY ENTRIES (NOT EXAMPLES):
- "abuset n. /_abus-é:t, _abus-0/ [< ke-abus to fool] conning."
- "ach p. /_a:c/ without."
- "acha i. /_a:ca, _a:cica, _a:cicica/ no!"
- "kw-aget v. /_kw-aké:t/ to graze (of livestock)."
- "ki-aget v.caus. /_ki:-aké:t/ to graze (tr.)."

JSON OUTPUT:
[{{"grapheme": "headword", "grammar": "part_of_speech", "ipa": "/pronunciation/", "definition": "meaning"}}]

Extract ONLY main dictionary entries, NOT example sentences. Return ONLY JSON array:"""


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
