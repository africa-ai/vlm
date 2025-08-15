#!/usr/bin/env python
"""
Test OCR extraction on dictionary page to verify accuracy
"""

import logging
from pathlib import Path
from PIL import Image
import pytesseract
import re
import platform

# Configure Tesseract path for Windows
if platform.system() == "Windows":
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Tesseract-OCR\tesseract.exe"
    ]
    
    for path in tesseract_paths:
        if Path(path).exists():
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"✅ Configured Tesseract: {path}")
            break
    else:
        print("❌ Tesseract not found in common Windows locations")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ocr_extraction(image_path: str):
    """Test OCR extraction with the updated method"""
    
    print(f"🔍 Testing OCR extraction on: {image_path}")
    
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        print(f"📷 Image loaded: {img.size} pixels, mode: {img.mode}")
        
        # Test 1: Original complex config (what we had before)
        print("\n1️⃣ Testing ORIGINAL OCR config...")
        complex_config = (
            r'--oem 1 --psm 6 '
            r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            r'àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğĠġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁłŃńŅņŇňŉŊŋŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽž'
            r'0123456789.,;:!?()[]{}/-_=+*|\\\'\" \n\t'
            r' -c preserve_interword_spaces=1'
            r' -c textord_tabfind_show_vlines=1'
            r' -c textord_tabfind_show_tabs=1'
            r' -c textord_heavy_nr=1'
        )
        
        original_text = pytesseract.image_to_string(img, config=complex_config)
        print(f"📝 Original OCR length: {len(original_text)} characters")
        print(f"📄 Original preview: {original_text[:300]}...")
        
        # Test 2: New simplified config
        print("\n2️⃣ Testing NEW SIMPLIFIED OCR config...")
        simple_config = (
            r'--oem 1 --psm 6 '
            r'-c preserve_interword_spaces=1 '
            r'-c textord_tabfind_show_vlines=1 '
            r'-c textord_heavy_nr=1'
        )
        
        simple_text = pytesseract.image_to_string(img, config=simple_config)
        print(f"📝 Simplified OCR length: {len(simple_text)} characters")
        print(f"📄 Simplified preview: {simple_text[:300]}...")
        
        # Test 3: Apply spacing cleanup
        print("\n3️⃣ Testing spacing cleanup...")
        def clean_ocr_spacing(raw_text: str) -> str:
            """Clean OCR text to fix spacing issues"""
            if not raw_text:
                return ""
            
            # Split into lines and process each
            lines = raw_text.split('\n')
            clean_lines = []
            
            for line in lines:
                line = line.strip()
                if len(line) < 2:
                    continue
                
                # Fix common OCR spacing issues
                # Add space between lowercase and uppercase
                line = re.sub(r'([a-z])([A-Z])', r'\1 \2', line)
                # Add space before and after forward slashes (IPA markers)
                line = re.sub(r'(\w)(/)(\w)', r'\1 \2\3', line)
                # Add space between word and number
                line = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', line)
                # Add space between number and word
                line = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', line)
                # Ensure space after common grammar markers
                line = re.sub(r'([.])([a-zA-Z])', r'\1 \2', line)
                # Clean up multiple spaces
                line = re.sub(r'\s+', ' ', line)
                
                clean_lines.append(line)
            
            return '\n'.join(clean_lines)
        
        cleaned_text = clean_ocr_spacing(simple_text)
        print(f"📝 Cleaned OCR length: {len(cleaned_text)} characters")
        print(f"📄 Cleaned preview: {cleaned_text[:300]}...")
        
        # Compare quality
        print("\n📊 COMPARISON:")
        print(f"Original length: {len(original_text)}")
        print(f"Simplified length: {len(simple_text)}")
        print(f"Cleaned length: {len(cleaned_text)}")
        
        # Look for dictionary patterns
        print("\n🔍 Dictionary pattern detection:")
        
        def count_patterns(text, name):
            ipa_patterns = len(re.findall(r'/[^/]+/', text))
            grammar_patterns = len(re.findall(r'\b[a-z]\.\s', text))
            headwords = len(re.findall(r'^[a-z]+\s', text, re.MULTILINE))
            
            print(f"  {name}:")
            print(f"    - IPA patterns (/word/): {ipa_patterns}")
            print(f"    - Grammar markers (v., n., etc.): {grammar_patterns}")
            print(f"    - Potential headwords: {headwords}")
        
        count_patterns(original_text, "Original")
        count_patterns(cleaned_text, "Cleaned")
        
        # Sample specific entries
        print("\n📚 Sample dictionary entries found:")
        
        def find_entries(text):
            lines = text.split('\n')
            entries = []
            for line in lines:
                line = line.strip()
                # Look for lines that might be dictionary entries
                if (len(line) > 10 and 
                    (re.search(r'/[^/]+/', line) or  # Has IPA
                     re.search(r'\b[a-z]\.\s', line))):  # Has grammar marker
                    entries.append(line)
            return entries[:5]  # First 5
        
        sample_entries = find_entries(cleaned_text)
        for i, entry in enumerate(sample_entries, 1):
            print(f"  {i}. {entry}")
        
    except Exception as e:
        logger.error(f"OCR test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    image_path = r"c:\Users\sugutt\Downloads\kalenjin_dictionary_page_002.png"
    
    if Path(image_path).exists():
        test_ocr_extraction(image_path)
    else:
        print(f"❌ Image not found: {image_path}")
        print("Please make sure the image path is correct.")
