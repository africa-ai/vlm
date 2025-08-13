#!/usr/bin/env python
"""
Detailed debugging to understand why the model isn't extracting all entries
"""

import sys
import base64
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from vllm_server.client import SyncVLLMClient
from llm.parser.prompts import PromptTemplates
from llm.image_preprocessor import optimize_image_for_vlm
import io

def debug_model_reasoning():
    """Debug exactly what the model sees and how it reasons"""
    client = SyncVLLMClient()
    
    # Find the first image to test
    images_dir = Path("results/images")
    image_files = list(images_dir.glob("*.png"))[:1]
    image_path = image_files[0]
    
    print(f"🔍 Testing model reasoning on: {image_path.name}")
    
    # Get preprocessed image
    print("🖼️  Preprocessing image...")
    optimized_images = optimize_image_for_vlm(str(image_path), target_tokens=100000)
    image = optimized_images[0]
    
    # Convert to base64
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    
    # Test with a more explicit prompt
    explicit_prompt = """<reasoning>
I need to systematically scan this dictionary page and extract EVERY single entry visible.

SCANNING STRATEGY:
1. Start from top-left, scan left column from top to bottom
2. Then scan right column from top to bottom  
3. Look for ALL headwords (Kalenjin words that start entries)
4. Don't stop until I've covered the entire page
5. Include partial entries at margins
6. Count entries as I go to ensure completeness
</reasoning>

**CRITICAL TASK**: Extract EVERY dictionary entry from this page. Do not stop until you have scanned the entire image systematically.

**SYSTEMATIC SCANNING REQUIRED**:
- Scan LEFT COLUMN: top to bottom completely
- Scan RIGHT COLUMN: top to bottom completely 
- Look for Kalenjin headwords at the start of each entry
- Include ALL visible entries, even partial ones
- Aim for 20+ entries minimum (typical dictionary page has many entries)

**REQUIRED OUTPUT**: JSON array with ALL entries found. If you find less than 15 entries, re-scan the image more carefully.

JSON format:
```json
[
  {"grapheme": "word1", "ipa": "/ipa1/", "english_meaning": "def1", "part_of_speech": "pos1", "context": "context1", "confidence_score": 0.95},
  {"grapheme": "word2", "ipa": "/ipa2/", "english_meaning": "def2", "part_of_speech": "pos2", "context": "context2", "confidence_score": 0.92}
]
```

**IMPORTANT**: Extract ALL entries systematically. Count as you go."""
    
    print("🤖 Sending explicit scanning prompt to model...")
    try:
        result = client.analyze_dictionary_image_base64(img_base64, custom_prompt=explicit_prompt)
        
        if result.get("status") == "success":
            response = result.get("analysis", "")
            print("✅ Model Response:")
            print("="*80)
            print(response)
            print("="*80)
            
            # Try to count JSON entries in response
            import json
            import re
            
            # Look for JSON arrays
            json_matches = re.findall(r'\[[\s\S]*?\]', response)
            if json_matches:
                print(f"\n📊 Found {len(json_matches)} potential JSON arrays")
                for i, match in enumerate(json_matches[:2]):  # Show first 2
                    try:
                        parsed = json.loads(match)
                        if isinstance(parsed, list):
                            print(f"   Array {i+1}: {len(parsed)} entries")
                            for j, entry in enumerate(parsed[:5]):  # Show first 5
                                if isinstance(entry, dict) and 'grapheme' in entry:
                                    print(f"      {j+1}. {entry['grapheme']} → {entry.get('english_meaning', 'N/A')}")
                            if len(parsed) > 5:
                                print(f"      ... and {len(parsed)-5} more")
                    except:
                        print(f"   Array {i+1}: Failed to parse")
            else:
                print("❌ No JSON arrays found in response")
        else:
            print(f"❌ Model failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_reasoning()
