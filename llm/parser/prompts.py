"""
Prompt templates for dictionary extraction using VLM
"""

from typing import Dict, List, Optional
import json

class PromptTemplates:
    """Collection of prompt templates for different extraction tasks"""
    
    # Base system prompt for dictionary extraction with reasoning
    SYSTEM_PROMPT = """You are Cosmos-Reason1-7B, a powerful reasoning model specialized in visual analysis and linguistic extraction. 

Your task is to analyze Kalenjin dictionary images with systematic reasoning:

<reasoning>
I will approach this task step by step:
1. First, I'll scan the image to identify the overall structure and layout
2. Then, I'll identify individual dictionary entries
3. For each entry, I'll extract the components: grapheme, IPA, meaning, part of speech
4. Finally, I'll format the results in structured JSON
</reasoning>

Focus on extracting:
- Grapheme: The original Kalenjin word/spelling
- IPA: International Phonetic Alphabet transcription (in /slashes/)
- English meaning: Translation or definition in English
- Part of speech: Grammatical category (v.t., v.i., n., adj., etc.)
- Context: Usage examples or contextual information

Be systematic, accurate, and thorough in your analysis."""
    
    # Main extraction prompt optimized for Cosmos-Reason1-7B
    EXTRACTION_PROMPT = """<reasoning>
Let me systematically analyze this Kalenjin dictionary page:

1. I'll examine the layout - typically two columns with alphabetical entries
2. Each entry follows the pattern: Kalenjin_word + grammar_info + IPA + English_definition + examples
3. I need to identify IPA transcriptions (in /forward slashes/ or _underscores_)
4. Grammar abbreviations include v.t. (transitive verb), v.i. (intransitive verb), n. (noun), etc.
5. I'll extract cross-references and usage examples for context
</reasoning>

Please analyze this Kalenjin dictionary page image and extract ALL visible dictionary entries.

The dictionary format includes:
- **Entry structure**: `grapheme + part_of_speech + IPA + definition + context`
- **IPA notations**: Forward slashes `/ke:-apus/` and underscores `/_apa/`
- **POS abbreviations**: `v.t.` (transitive verb), `v.i.` (intransitive verb), `n.` (noun), etc.
- **Cross-references**: Capitalized related entries (e.g., "Kogiabus")
- **Usage examples**: Questions and contextual sentences

For each entry, extract:
- **Grapheme**: The main Kalenjin word (e.g., "abus", "abaita")
- **IPA**: Phonetic transcription (e.g., "/ke:-apus/", "/_apay-ta/")
- **English meaning**: The English definition or translation
- **Part of speech**: Grammar category (v.t., v.i., n., adj., etc.)
- **Context**: Any usage examples, cross-references, or additional notes

Return a comprehensive JSON array:
```json
[
  {
    "grapheme": "abus",
    "ipa": "/ke:-apus/",
    "english_meaning": "to give bad advice, to fool, make a fool of",
    "part_of_speech": "v.t.",
    "context": "Kogiabus. (S/he) was ill-advised, fooled",
    "confidence_score": 0.95
  }
]
```

Extract ALL visible entries from both columns. Use systematic reasoning to ensure completeness and accuracy."""
    
    # Focused extraction for specific fields
    GRAPHEME_FOCUS_PROMPT = """Focus specifically on identifying Kalenjin words (graphemes) in this dictionary image. 

Extract only the original Kalenjin words/terms, ignoring English text unless it's part of a definition.

Return as JSON array:
```json
[{"grapheme": "word1"}, {"grapheme": "word2"}]
```"""
    
    IPA_FOCUS_PROMPT = """Look for International Phonetic Alphabet (IPA) transcriptions in this dictionary image.

IPA transcriptions are typically enclosed in forward slashes /.../ or square brackets [...].

Return as JSON array:
```json
[{"grapheme": "kalenjin_word", "ipa": "/ipa_transcription/"}]
```"""
    
    MEANING_FOCUS_PROMPT = """Extract English meanings and definitions for Kalenjin words in this dictionary image.

Focus on the English translation/definition portions of each entry.

Return as JSON array:
```json
[{"grapheme": "kalenjin_word", "english_meaning": "English definition"}]
```"""
    
    # Quality check and validation prompt
    VALIDATION_PROMPT = """Review the following extracted dictionary entries for accuracy and completeness:

{extracted_data}

Check for:
1. Correct Kalenjin spelling (graphemes)
2. Accurate IPA transcriptions (if present)
3. Appropriate English meanings
4. Consistent formatting

Return the corrected/validated entries in the same JSON format, or confirm if no changes are needed."""
    
    # Batch processing prompt
    BATCH_PROMPT = """This is page {page_number} of a Kalenjin dictionary. Please extract all dictionary entries visible on this page.

Follow the standard format for each entry:
- Grapheme (Kalenjin word)
- IPA transcription (if available)  
- English meaning
- Part of speech
- Context/examples

Return as JSON array with all entries found on this page."""
    
    @classmethod
    def get_extraction_prompt(
        self, 
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
            "complete": self.EXTRACTION_PROMPT,
            "grapheme": self.GRAPHEME_FOCUS_PROMPT,
            "ipa": self.IPA_FOCUS_PROMPT,
            "meaning": self.MEANING_FOCUS_PROMPT
        }
        
        prompt = base_prompts.get(focus, self.EXTRACTION_PROMPT)
        
        if page_number:
            prompt = self.BATCH_PROMPT.format(page_number=page_number)
        
        if custom_instructions:
            prompt += f"\n\nAdditional instructions: {custom_instructions}"
        
        return prompt
    
    @classmethod
    def get_validation_prompt(self, extracted_data: List[Dict]) -> str:
        """
        Get validation prompt for extracted data
        
        Args:
            extracted_data: List of extracted dictionary entries
            
        Returns:
            Formatted validation prompt
        """
        data_json = json.dumps(extracted_data, indent=2, ensure_ascii=False)
        return self.VALIDATION_PROMPT.format(extracted_data=data_json)
    
    @classmethod
    def get_system_message(self) -> str:
        """Get the system prompt for model initialization"""
        return self.SYSTEM_PROMPT
    
    @classmethod
    def format_conversation(
        self,
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
                "content": self.get_system_message()
            },
            {
                "role": "user", 
                "content": user_prompt
            }
        ]
        
        return messages
    
    @classmethod
    def create_few_shot_examples(self) -> str:
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
