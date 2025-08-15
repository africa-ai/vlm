#!/usr/bin/env python
"""
OCR Accuracy Comparison Tool
Saves OCR results and compares with original document for quality assessment
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from PIL import Image
import pytesseract
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Tesseract
def configure_tesseract():
    """Configure Tesseract path for Windows"""
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Tesseract-OCR\tesseract.exe"
    ]
    
    for path in tesseract_paths:
        if Path(path).exists():
            pytesseract.pytesseract.tesseract_cmd = path
            logger.info(f"✅ Configured Tesseract: {path}")
            return True
    
    logger.error("❌ Tesseract not found in standard locations")
    return False

def extract_with_different_configs(image_path: str) -> dict:
    """Extract text using different OCR configurations"""
    
    img = Image.open(image_path)
    if img.mode != 'L':
        img = img.convert('L')
    
    results = {}
    
    # 1. Basic OCR (default)
    try:
        basic_text = pytesseract.image_to_string(img)
        results['basic'] = {
            'text': basic_text,
            'length': len(basic_text),
            'config': 'Default Tesseract'
        }
    except Exception as e:
        results['basic'] = {'error': str(e)}
    
    # 2. Dictionary-optimized config
    try:
        dict_config = r'--oem 1 --psm 6 -c preserve_interword_spaces=1'
        dict_text = pytesseract.image_to_string(img, config=dict_config)
        results['dictionary'] = {
            'text': dict_text,
            'length': len(dict_text),
            'config': 'Dictionary optimized (OEM 1, PSM 6)'
        }
    except Exception as e:
        results['dictionary'] = {'error': str(e)}
    
    # 3. High accuracy config
    try:
        accuracy_config = r'--oem 1 --psm 3'
        accuracy_text = pytesseract.image_to_string(img, config=accuracy_config)
        results['high_accuracy'] = {
            'text': accuracy_text,
            'length': len(accuracy_text),
            'config': 'High accuracy (OEM 1, PSM 3)'
        }
    except Exception as e:
        results['high_accuracy'] = {'error': str(e)}
    
    # 4. With text cleaning
    try:
        cleaned_text = clean_ocr_text(results['dictionary']['text'])
        results['cleaned'] = {
            'text': cleaned_text,
            'length': len(cleaned_text),
            'config': 'Dictionary + text cleaning'
        }
    except Exception as e:
        results['cleaned'] = {'error': str(e)}
    
    return results

def clean_ocr_text(raw_text: str) -> str:
    """Clean OCR text to fix spacing and formatting issues"""
    if not raw_text:
        return ""
    
    lines = raw_text.split('\n')
    clean_lines = []
    
    for line in lines:
        line = line.strip()
        if len(line) < 2:
            continue
        
        # Fix common OCR spacing issues
        line = re.sub(r'([a-z])([A-Z])', r'\1 \2', line)  # camelCase
        line = re.sub(r'(\w)(/)(\w)', r'\1 \2\3', line)   # word/word
        line = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', line)  # word1
        line = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', line)  # 1word
        line = re.sub(r'([.])([a-zA-Z])', r'\1 \2', line) # .word
        line = re.sub(r'\s+', ' ', line)                   # multiple spaces
        
        clean_lines.append(line)
    
    return '\n'.join(clean_lines)

def analyze_dictionary_patterns(text: str) -> dict:
    """Analyze text for dictionary-specific patterns"""
    
    # IPA patterns (text in /forward slashes/)
    ipa_patterns = re.findall(r'/[^/]+/', text)
    
    # Grammar markers
    grammar_markers = re.findall(r'\b(n\.|v\.|a\.|adj\.|adv\.|prep\.|conj\.|interj\.|caus\.|tr\.|intr\.|p\.)\b', text)
    
    # Potential headwords (words at start of lines, followed by grammar)
    lines = text.split('\n')
    headwords = []
    for line in lines:
        line = line.strip()
        # Look for pattern: word followed by grammar marker
        match = re.match(r'^([a-zA-Z-]+)\s+(n\.|v\.|a\.|adj\.)', line)
        if match:
            headwords.append(match.group(1))
    
    # Word count and character analysis
    word_count = len(text.split())
    char_count = len(text)
    
    return {
        'ipa_patterns': {
            'count': len(ipa_patterns),
            'examples': ipa_patterns[:5]  # First 5 examples
        },
        'grammar_markers': {
            'count': len(grammar_markers),
            'types': list(set(grammar_markers))
        },
        'headwords': {
            'count': len(headwords),
            'examples': headwords[:10]  # First 10 examples
        },
        'text_stats': {
            'word_count': word_count,
            'char_count': char_count,
            'line_count': len(lines)
        }
    }

def save_comparison_results(image_path: str, results: dict, analysis: dict):
    """Save comparison results to files"""
    
    # Create output directory
    output_dir = Path("ocr_comparison_results")
    output_dir.mkdir(exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = Path(image_path).stem
    
    # Save detailed results
    results_file = output_dir / f"{image_name}_ocr_comparison_{timestamp}.json"
    
    full_results = {
        'source_image': str(image_path),
        'timestamp': datetime.now().isoformat(),
        'ocr_results': results,
        'pattern_analysis': analysis
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    
    # Save best OCR text for manual review
    best_config = 'cleaned' if 'cleaned' in results else 'dictionary'
    if best_config in results and 'text' in results[best_config]:
        text_file = output_dir / f"{image_name}_best_ocr_{timestamp}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(f"OCR Results for: {image_path}\n")
            f.write(f"Configuration: {results[best_config]['config']}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n\n")
            f.write(results[best_config]['text'])
    
    logger.info(f"💾 Results saved to: {results_file}")
    logger.info(f"📄 Best OCR text saved to: {text_file}")
    
    return results_file, text_file

def generate_quality_report(analysis: dict) -> str:
    """Generate a quality assessment report"""
    
    report = []
    report.append("📊 OCR QUALITY ASSESSMENT REPORT")
    report.append("=" * 50)
    
    for config_name, data in analysis.items():
        if 'error' in data:
            report.append(f"\n❌ {config_name.upper()}: FAILED")
            report.append(f"   Error: {data['error']}")
            continue
        
        patterns = data.get('patterns', {})
        stats = patterns.get('text_stats', {})
        
        report.append(f"\n✅ {config_name.upper()}:")
        report.append(f"   📝 Length: {stats.get('char_count', 0)} characters")
        report.append(f"   📚 Words: {stats.get('word_count', 0)}")
        report.append(f"   📄 Lines: {stats.get('line_count', 0)}")
        
        ipa = patterns.get('ipa_patterns', {})
        report.append(f"   🔤 IPA patterns: {ipa.get('count', 0)}")
        if ipa.get('examples'):
            report.append(f"      Examples: {', '.join(ipa['examples'][:3])}")
        
        grammar = patterns.get('grammar_markers', {})
        report.append(f"   📖 Grammar markers: {grammar.get('count', 0)}")
        if grammar.get('types'):
            report.append(f"      Types: {', '.join(grammar['types'])}")
        
        headwords = patterns.get('headwords', {})
        report.append(f"   🎯 Potential headwords: {headwords.get('count', 0)}")
        if headwords.get('examples'):
            report.append(f"      Examples: {', '.join(headwords['examples'][:5])}")
    
    return '\n'.join(report)

def main():
    """Main comparison function"""
    
    # Configure Tesseract
    if not configure_tesseract():
        print("❌ Tesseract configuration failed. Please install Tesseract OCR.")
        return
    
    # Test image
    image_path = r"c:\Users\sugutt\Downloads\kalenjin_dictionary_page_002.png"
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"🔍 Processing: {image_path}")
    print(f"📷 Image: {Path(image_path).name}")
    
    # Extract text with different configurations
    print("\n📝 Extracting text with different OCR configurations...")
    results = extract_with_different_configs(image_path)
    
    # Analyze each result for dictionary patterns
    print("🔍 Analyzing dictionary patterns...")
    analysis = {}
    for config_name, data in results.items():
        if 'text' in data:
            patterns = analyze_dictionary_patterns(data['text'])
            analysis[config_name] = {
                'config': data['config'],
                'length': data['length'],
                'patterns': patterns
            }
        else:
            analysis[config_name] = data
    
    # Save results
    print("\n💾 Saving comparison results...")
    results_file, text_file = save_comparison_results(image_path, results, analysis)
    
    # Generate and display quality report
    print("\n" + generate_quality_report(analysis))
    
    print(f"\n🎉 Comparison complete!")
    print(f"📁 Results saved in: ocr_comparison_results/")
    print(f"📄 Best OCR text: {text_file.name}")
    print(f"📊 Full comparison: {results_file.name}")

if __name__ == "__main__":
    main()
