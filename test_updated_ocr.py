#!/usr/bin/env python
"""
Test Updated OCR Processor with High Accuracy Configuration
"""

import logging
from pathlib import Path
from llm.config import load_config_from_env
from llm.ocr_processor import OCRProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_updated_ocr():
    """Test the updated OCR processor with high accuracy config"""
    print("🚀 Testing Updated OCR Processor (High Accuracy)")
    print("=" * 60)
    
    # Load config and create processor
    config = load_config_from_env()
    processor = OCRProcessor(config, server_url="http://localhost:8000")
    
    # Test image
    test_image = "c:\\Users\\sugutt\\Downloads\\kalenjin_dictionary_page_002.png"
    
    if not Path(test_image).exists():
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"📖 Testing OCR extraction on: {test_image}")
    
    # Extract text only (no LLM processing needed for this test)
    raw_text = processor.extract_text_from_image(test_image)
    
    if raw_text:
        print(f"✅ OCR extraction successful!")
        print(f"📝 Text length: {len(raw_text)} characters")
        print(f"📄 Lines: {len(raw_text.splitlines())}")
        
        # Count dictionary patterns
        import re
        
        ipa_patterns = re.findall(r'/[^/]+/', raw_text)
        grammar_patterns = re.findall(r'\b(?:v\.|n\.|adj\.|adv\.|c\.|p\.)\b', raw_text)
        
        print(f"🔤 IPA patterns found: {len(ipa_patterns)}")
        print(f"📖 Grammar markers found: {len(grammar_patterns)}")
        
        # Show first few lines as preview
        lines = raw_text.split('\n')
        print(f"\n📄 First 10 lines preview:")
        for i, line in enumerate(lines[:10], 1):
            if line.strip():
                print(f"  {i:2d}. {line.strip()}")
        
        # Show some IPA examples
        if ipa_patterns:
            print(f"\n🔤 IPA examples:")
            for i, ipa in enumerate(ipa_patterns[:5], 1):
                print(f"  {i}. {ipa}")
        
        # Show grammar examples
        if grammar_patterns:
            print(f"\n📖 Grammar markers:")
            unique_grammar = list(set(grammar_patterns))
            for i, marker in enumerate(unique_grammar, 1):
                print(f"  {i}. {marker}")
        
        print(f"\n🎯 Expected results:")
        print(f"   • Should find ~37 IPA patterns")
        print(f"   • Should find ~7 potential headwords")
        print(f"   • Should preserve dictionary structure")
        
    else:
        print("❌ OCR extraction failed!")

if __name__ == "__main__":
    test_updated_ocr()
