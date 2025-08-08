"""
Prompt templates for dictionary extraction using VLM
"""

from typing import Dict, List, Optional
import json

class PromptTemplates:
    """Collection of prompt templates for different extraction tasks"""
    
    # Base system prompt for dictionary extraction with reasoning
    SYSTEM_PROMPT = """You are Cosmos-Reason1-7B, a specialized vision-language model expert in analyzing dictionary pages and extracting linguistic data with high precision.

Your expertise includes:
- Recognizing dictionary formatting conventions and layouts
- Extracting Kalenjin words with accurate spelling
- Identifying IPA transcriptions in various formats
- Understanding grammatical abbreviations and linguistic notation
- Preserving semantic relationships and contextual information

<reasoning>
For dictionary analysis, I will:
1. Scan the entire image systematically from top to bottom, left to right
2. Identify column boundaries and entry separations
3. Parse each entry's components: headword → grammar → phonetics → definition → examples
4. Validate extracted data for linguistic consistency
5. Format results in structured JSON with complete field coverage
</reasoning>

Quality standards:
- Extract EVERY visible dictionary entry, no matter how small
- Preserve exact Kalenjin spelling and capitalization
- Capture all IPA variants: /slashes/, _underscores_, [brackets]
- Include complete English definitions, not fragments
- Maintain grammatical accuracy for parts of speech"""
    
    # Main extraction prompt optimized for Cosmos-Reason1-7B
    EXTRACTION_PROMPT = """<reasoning>
Analyzing this Kalenjin dictionary page systematically:

STEP 1: Layout Analysis
- Dictionary pages typically have 2 columns with alphabetically ordered entries
- Each entry follows a consistent pattern: HEADWORD + GRAMMAR + PHONETICS + DEFINITION + EXAMPLES
- Text hierarchy: headwords are prominent, definitions follow, examples in context

STEP 2: Component Recognition Patterns
- Kalenjin headwords: Often start entries, may include hyphens (ke-word, -word)
- Grammar codes: v.t. (transitive verb), v.i. (intransitive verb), n. (noun), adj. (adjective), adv. (adverb)
- IPA transcriptions: /forward-slashes/, _underscores_, [square-brackets], or (parentheses)
- English definitions: Follow grammar codes, may be lengthy explanations
- Cross-references: Capitalized related words (Kogiabus, Keabus)
- Examples: Often questions ("Ingoro aba?") or usage sentences

STEP 3: Quality Validation
- Verify each entry has a clear Kalenjin headword (2+ characters)
- Ensure English definitions are meaningful (5+ characters)
- Check phonetic transcriptions for proper formatting
- Confirm grammar codes match standard abbreviations
</reasoning>

**TASK**: Extract ALL dictionary entries from this Kalenjin dictionary page image.

**EXTRACTION RULES**:
1. **Completeness**: Find every single entry, including partial ones at margins
2. **Accuracy**: Preserve exact spelling, capitalization, and punctuation
3. **Structure**: Maintain the relationship between headword and definition
4. **Context**: Include usage examples, cross-references, and grammatical notes

**KALENJIN DICTIONARY FORMAT GUIDE**:
- **Headword**: Main Kalenjin word (abus, ke-abus, -aita)
- **Grammar**: v.t., v.i., n., adj., adv., prep., conj., interj.
- **Phonetics**: /ke:-apus/, /_apa_/, [kεapus], (ke-apus)
- **Definition**: Complete English translation or explanation
- **Examples**: Usage sentences, questions, cross-references

**REQUIRED JSON FORMAT**:
```json
[
  {
    "grapheme": "abus",
    "ipa": "/ke:-apus/",
    "english_meaning": "to give bad advice, to fool, make a fool of someone",
    "part_of_speech": "v.t.",
    "context": "Kogiabus - (S/he) was ill-advised, fooled. Usage: 'Abus kobo?' - 'Are you fooling them?'",
    "confidence_score": 0.95
  },
  {
    "grapheme": "ke-abus", 
    "ipa": "/_kεapus_/",
    "english_meaning": "trickster, one who gives bad advice",
    "part_of_speech": "n.",
    "context": "Related to 'abus' (to fool). Example: 'Keabus ago' - 'That trickster'",
    "confidence_score": 0.90
  }
]
```

**CRITICAL**: Extract EVERY visible entry. Do not skip entries due to unclear text - include them with lower confidence scores. Scan both columns completely from top to bottom."""
    
    # Focused extraction for specific fields
    GRAPHEME_FOCUS_PROMPT = """**TASK**: Extract ONLY the Kalenjin headwords from this dictionary page.

<reasoning>
Focus exclusively on identifying the main Kalenjin words that serve as dictionary headwords:
1. Scan for words at the beginning of each entry
2. Look for Kalenjin morphological patterns (prefixes like ke-, ko-, suffixes like -aita)  
3. Ignore English definitions and grammatical annotations
4. Include variant forms and inflected entries
</reasoning>

**KALENJIN WORD PATTERNS**:
- Simple words: abus, angul, chebo
- Prefixed words: ke-abus, ko-angul  
- Suffixed words: -aita, -en, -un
- Compound words: che-abus, ngo-something

**EXTRACTION RULES**:
- Extract the EXACT spelling as shown in the dictionary
- Include hyphens and morphological markers
- Preserve capitalization patterns
- Skip English words and abbreviations

Return as clean JSON array:
```json
[
  {"grapheme": "abus"},
  {"grapheme": "ke-abus"},
  {"grapheme": "abusiot"},
  {"grapheme": "-aita"}
]
```"""
    
    IPA_FOCUS_PROMPT = """**TASK**: Extract phonetic transcriptions (IPA) from this Kalenjin dictionary page.

<reasoning>
Phonetic transcriptions in Kalenjin dictionaries appear in multiple formats:
1. Standard IPA in forward slashes: /ke:-apus/
2. Underscored variants: /_apa_/, _kεapus_
3. Square brackets: [kεapus], [ke:-apus]
4. Parenthetical forms: (ke-apus)
5. Mixed notation with tone markers: /kè:àpùs/

I need to identify all these patterns and associate them with their corresponding headwords.
</reasoning>

**PHONETIC NOTATION PATTERNS**:
- Forward slashes: `/ke:-apus/`, `/apa/`, `/keεn/`
- Underscores: `_apa_`, `_kεapus_`, `_keεn_`
- Square brackets: `[kεapus]`, `[ke:-apus]`
- Parentheses: `(ke-apus)`, `(apa)`
- Tone markers: `è`, `à`, `ò`, `ù` (grave accent = low tone)
- Length markers: `:` (long vowel)

**EXTRACTION RULES**:
- Capture the COMPLETE phonetic transcription including all markers
- Preserve tone and length notation exactly as shown
- Associate each IPA with its corresponding Kalenjin headword
- Include partial transcriptions if visible

Return as JSON array with headword-IPA pairs:
```json
[
  {"grapheme": "abus", "ipa": "/ke:-apus/"},
  {"grapheme": "apa", "ipa": "_àpà_"},
  {"grapheme": "chebo", "ipa": "[tʃèbò]"},
  {"grapheme": "keen", "ipa": "(kè:èn)"}
]
```"""
    
    MEANING_FOCUS_PROMPT = """**TASK**: Extract English definitions and meanings for Kalenjin words from this dictionary page.

<reasoning>
English definitions in dictionaries follow specific patterns:
1. They appear after the headword and grammatical information
2. They may be comprehensive explanations or concise translations
3. They often include multiple senses separated by semicolons or numbered
4. They may contain usage examples and contextual information
5. Cross-references to related words may be included
</reasoning>

**DEFINITION EXTRACTION PATTERNS**:
- Primary meanings: "to give bad advice", "water", "house"
- Multiple senses: "1. to fool; 2. to deceive; 3. to mislead"
- Contextual usage: "used when referring to...", "especially in..."
- Comparative forms: "more/less than...", "similar to..."
- Cultural context: "traditional term for...", "ceremonial..."

**EXTRACTION RULES**:
- Capture COMPLETE definitions, not fragments
- Include all numbered or separated senses
- Preserve explanatory context and usage notes
- Skip grammatical abbreviations (v.t., n., etc.)
- Include cross-references when they clarify meaning

Return as JSON array with headword-definition pairs:
```json
[
  {"grapheme": "abus", "english_meaning": "to give bad advice, to fool, make a fool of someone"},
  {"grapheme": "ke-abus", "english_meaning": "trickster, one who gives bad advice; a deceiver"},
  {"grapheme": "apa", "english_meaning": "water (general term); any liquid for drinking"},
  {"grapheme": "got", "english_meaning": "house, dwelling place; home (traditional structure)"}
]
```"""
    
    # Quality check and validation prompt
    VALIDATION_PROMPT = """**TASK**: Review and validate the extracted dictionary entries for accuracy and completeness.

<reasoning>
Quality validation requires checking:
1. Linguistic accuracy - proper Kalenjin orthography and English translations
2. Structural completeness - all required fields present
3. Data consistency - formatting matches dictionary conventions
4. Content integrity - no OCR errors or misinterpretations
</reasoning>

**VALIDATION CHECKLIST**:
- ✅ Kalenjin graphemes use correct spelling and morphological markers
- ✅ IPA transcriptions follow proper phonetic notation
- ✅ English definitions are complete and culturally appropriate
- ✅ Parts of speech match standard grammatical abbreviations
- ✅ Context includes relevant usage examples or cross-references
- ✅ No JSON artifacts or formatting errors remain

**EXTRACTED DATA TO VALIDATE**:
{extracted_data}

**INSTRUCTIONS**:
1. Review each entry for accuracy and completeness
2. Correct any obvious errors (OCR mistakes, incomplete definitions)
3. Standardize formatting for consistency
4. Add missing information where possible
5. Remove any invalid or artifact entries

Return the validated entries in the same JSON format with confidence scores adjusted based on quality."""
    
    # Batch processing prompt
    BATCH_PROMPT = """**DICTIONARY PAGE EXTRACTION** - Page {page_number}

<reasoning>
This is part of a larger dictionary digitization project. I need to:
1. Maintain consistency with previous pages
2. Extract every visible entry regardless of clarity
3. Preserve alphabetical ordering information
4. Note any page-specific formatting variations
</reasoning>

**PAGE CONTEXT**: This is page {page_number} of a comprehensive Kalenjin-English dictionary.

**EXTRACTION REQUIREMENTS**:
- Extract ALL dictionary entries visible on this page
- Maintain strict accuracy for linguistic data
- Include partial entries that continue from previous/next pages
- Note any special formatting or annotations
- Preserve alphabetical sequence for quality control

**STANDARD ENTRY FORMAT**:
```json
{{
  "grapheme": "kalenjin_headword",
  "ipa": "/phonetic_transcription/",
  "english_meaning": "complete English definition with all senses",
  "part_of_speech": "grammatical_category",
  "context": "usage examples, cross-references, cultural notes",
  "confidence_score": 0.85,
  "page_number": {page_number},
  "entry_position": "approximate location on page"
}}
```

**QUALITY STANDARDS**:
- Minimum 95% accuracy for headwords and definitions
- Complete phonetic transcriptions where visible
- Comprehensive context for cultural or technical terms
- Clear indication of confidence level for each entry

Extract every visible entry from this page systematically.

Return as JSON array with all entries found on this page."""
    
    @classmethod
    def get_extraction_prompt(
        cls, 
        focus: str = "complete",
        page_number: Optional[int] = None,
        custom_instructions: Optional[str] = None
    ) -> str:
        """
        Get appropriate extraction prompt based on focus area
        
        Args:
            focus: Type of extraction ("complete", "grapheme", "ipa", "meaning")
            page_number: Page number if processing in batch
            custom_instructions: Additional custom instructions
            
        Returns:
            Formatted prompt string
        """
        base_prompts = {
            "complete": cls.EXTRACTION_PROMPT,
            "grapheme": cls.GRAPHEME_FOCUS_PROMPT,
            "ipa": cls.IPA_FOCUS_PROMPT,
            "meaning": cls.MEANING_FOCUS_PROMPT
        }
        
        prompt = base_prompts.get(focus, cls.EXTRACTION_PROMPT)
        
        if page_number:
            prompt = cls.BATCH_PROMPT.format(page_number=page_number)
        
        if custom_instructions:
            prompt += f"\n\nAdditional instructions: {custom_instructions}"
        
        return prompt
    
    @classmethod
    def get_validation_prompt(cls, extracted_data: List[Dict]) -> str:
        """
        Get validation prompt for extracted data
        
        Args:
            extracted_data: List of extracted dictionary entries
            
        Returns:
            Formatted validation prompt
        """
        data_json = json.dumps(extracted_data, indent=2, ensure_ascii=False)
        return cls.VALIDATION_PROMPT.format(extracted_data=data_json)
    
    @classmethod
    def get_system_message(cls) -> str:
        """Get the system prompt for model initialization"""
        return cls.SYSTEM_PROMPT
    
    @classmethod
    def format_conversation(
        cls,
        user_prompt: str,
        image_path: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Format conversation for VLM input
        
        Args:
            user_prompt: User's prompt text
            image_path: Path to image (if applicable)
            
        Returns:
            Formatted conversation messages
        """
        messages = [
            {
                "role": "system",
                "content": cls.get_system_message()
            },
            {
                "role": "user", 
                "content": user_prompt
            }
        ]
        
        return messages
    
    @classmethod
    def create_few_shot_examples(cls) -> str:
        """
        Create few-shot learning examples for better performance
        
        Returns:
            Example prompt with sample inputs/outputs
        """
        return """Here are examples of correct dictionary entry extraction for this Kalenjin dictionary format:

Example 1:
Dictionary shows: "abus v.t. /ke:-apus/ to give bad advice, to fool, make a fool of. Kogiabus. (S/he) was ill-advised, fooled."
Output:
```json
[{
  "grapheme": "abus",
  "ipa": "/ke:-apus/",
  "english_meaning": "to give bad advice, to fool, make a fool of",
  "part_of_speech": "v.t.",
  "context": "Kogiabus. (S/he) was ill-advised, fooled",
  "confidence_score": 0.95
}]
```

Example 2:
Dictionary shows: "aba n. /_apa/, _apa (nom.)/ father, paternal uncle. Ingoro aba? Where is father?"
Output:
```json
[{
  "grapheme": "aba",
  "ipa": "/_apa/",
  "english_meaning": "father, paternal uncle",
  "part_of_speech": "n.",
  "context": "Ingoro aba? Where is father?",
  "confidence_score": 0.90
}]
```

Example 3:
Dictionary shows: "abaita n. /_apay-ta/, _apay; apay-wa, apay-we:k/ act of going for milk to obtain a praise name for a new-born infant."
Output:
```json
[{
  "grapheme": "abaita", 
  "ipa": "/_apay-ta/",
  "english_meaning": "act of going for milk to obtain a praise name for a new-born infant",
  "part_of_speech": "n.",
  "context": "apay-wa, apay-we:k alternate forms",
  "confidence_score": 0.88
}]
```

Now extract entries from the provided image following this exact format:"""

# Pre-defined prompt combinations for common use cases
PROMPT_COMBINATIONS = {
    "standard_extraction": {
        "system": PromptTemplates.SYSTEM_PROMPT,
        "user": PromptTemplates.EXTRACTION_PROMPT
    },
    
    "with_examples": {
        "system": PromptTemplates.SYSTEM_PROMPT,
        "user": PromptTemplates.create_few_shot_examples() + "\n\n" + PromptTemplates.EXTRACTION_PROMPT
    },
    
    "batch_processing": {
        "system": PromptTemplates.SYSTEM_PROMPT,
        "user": PromptTemplates.BATCH_PROMPT
    },
    
    "quality_focused": {
        "system": PromptTemplates.SYSTEM_PROMPT + "\n\nPrioritize accuracy over completeness. Only extract entries you are highly confident about.",
        "user": PromptTemplates.EXTRACTION_PROMPT
    }
}
