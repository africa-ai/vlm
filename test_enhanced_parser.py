#!/usr/bin/env python
"""
Test Enhanced Parser with Real Dictionary Data
Compare performance before/after improvements
"""

import json
import logging
from pathlib import Path
from llm.parser.main import DictionaryParser
from vllm_server.client import SyncVLLMClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_parser():
    """Test the enhanced parser with real dictionary data"""
    
    # Load the high accuracy OCR text from our comparison
    json_file = "ocr_comparison_results/kalenjin_dictionary_page_002_ocr_comparison_20250815_140228.json"
    
    if not Path(json_file).exists():
        logger.error(f"Comparison file not found: {json_file}")
        return
    
    with open(json_file, 'r', encoding='utf-8') as f:
        comparison_data = json.load(f)
    
    # Get the high accuracy OCR text
    high_accuracy_text = comparison_data['ocr_results']['high_accuracy']['text']
    
    logger.info("🧪 Testing Enhanced Parser")
    logger.info("=" * 50)
    logger.info(f"📄 Text length: {len(high_accuracy_text)} characters")
    
    # Initialize parser and client
    parser = DictionaryParser()
    client = SyncVLLMClient("http://localhost:8000")
    
    # Check server connection
    if not client.health_check():
        logger.error("❌ vLLM server not accessible. Please start the server first.")
        return
    
    logger.info("✅ vLLM server connected")
    
    # Test parsing
    try:
        logger.info("🔄 Processing with enhanced parser...")
        entries = parser.parse_ocr_text(high_accuracy_text, client)
        
        logger.info(f"📊 RESULTS: {len(entries)} entries extracted")
        
        if entries:
            logger.info("📝 Sample entries:")
            for i, entry in enumerate(entries[:5]):
                grapheme = entry.get('grapheme', 'N/A')
                grammar = entry.get('grammar', 'N/A')
                ipa = entry.get('ipa', 'N/A')
                definition = entry.get('definition', 'N/A')
                confidence = entry.get('confidence', 0.0)
                
                logger.info(f"  {i+1}. {grapheme} ({grammar}) {ipa}")
                logger.info(f"     Definition: {definition[:80]}...")
                logger.info(f"     Confidence: {confidence:.2f}")
            
            if len(entries) > 5:
                logger.info(f"     ... and {len(entries) - 5} more entries")
        
        # Analyze results
        logger.info("🔍 QUALITY ANALYSIS:")
        
        # Count entries with IPA
        with_ipa = sum(1 for e in entries if e.get('ipa'))
        logger.info(f"   📚 Entries with IPA: {with_ipa}/{len(entries)} ({with_ipa/len(entries)*100:.1f}%)")
        
        # Count entries with grammar
        with_grammar = sum(1 for e in entries if e.get('grammar'))
        logger.info(f"   📖 Entries with grammar: {with_grammar}/{len(entries)} ({with_grammar/len(entries)*100:.1f}%)")
        
        # Average confidence
        avg_confidence = sum(e.get('confidence', 0) for e in entries) / len(entries) if entries else 0
        logger.info(f"   🎯 Average confidence: {avg_confidence:.3f}")
        
        # Most confident entries
        sorted_entries = sorted(entries, key=lambda x: x.get('confidence', 0), reverse=True)
        logger.info(f"   🏆 Top entries by confidence:")
        for i, entry in enumerate(sorted_entries[:3]):
            grapheme = entry.get('grapheme', 'N/A')
            conf = entry.get('confidence', 0)
            logger.info(f"      {i+1}. {grapheme} ({conf:.3f})")
        
        # Save results
        output_file = "enhanced_parser_test_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "test_timestamp": "2025-08-15T14:30:00",
                "source_text_length": len(high_accuracy_text),
                "entries_extracted": len(entries),
                "entries_with_ipa": with_ipa,
                "entries_with_grammar": with_grammar,
                "average_confidence": avg_confidence,
                "entries": entries
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Parser test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    test_enhanced_parser()
