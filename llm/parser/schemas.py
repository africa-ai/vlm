"""
Data schemas for dictionary entry extraction
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json

@dataclass
class DictionaryEntry:
    """Schema for a single dictionary entry"""
    
    grapheme: str  # Original Kalenjin script/spelling
    ipa: Optional[str] = None  # International Phonetic Alphabet representation
    english_meaning: Optional[str] = None  # English translation/meaning
    part_of_speech: Optional[str] = None  # Grammatical category (noun, verb, etc.)
    context: Optional[str] = None  # Usage context or example
    etymology: Optional[str] = None  # Word origin information
    confidence_score: float = 1.0  # Extraction confidence (0.0 to 1.0)
    page_number: Optional[int] = None  # Source page number
    bbox: Optional[List[float]] = None  # Bounding box coordinates [x1, y1, x2, y2]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'grapheme': self.grapheme,
            'ipa': self.ipa,
            'english_meaning': self.english_meaning,
            'part_of_speech': self.part_of_speech,
            'context': self.context,
            'etymology': self.etymology,
            'confidence_score': self.confidence_score,
            'page_number': self.page_number,
            'bbox': self.bbox
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DictionaryEntry':
        """Create instance from dictionary"""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'DictionaryEntry':
        """Create instance from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def is_valid(self) -> bool:
        """Check if the entry has minimum required data"""
        return bool(self.grapheme and self.grapheme.strip())
    
    def merge_with(self, other: 'DictionaryEntry') -> 'DictionaryEntry':
        """Merge with another entry, filling in missing fields"""
        merged = DictionaryEntry(
            grapheme=self.grapheme,
            ipa=self.ipa or other.ipa,
            english_meaning=self.english_meaning or other.english_meaning,
            part_of_speech=self.part_of_speech or other.part_of_speech,
            context=self.context or other.context,
            etymology=self.etymology or other.etymology,
            confidence_score=max(self.confidence_score, other.confidence_score),
            page_number=self.page_number or other.page_number,
            bbox=self.bbox or other.bbox
        )
        return merged

@dataclass
class ExtractionResult:
    """Schema for extraction results from a page/image"""
    
    source_image: str  # Path to source image
    entries: List[DictionaryEntry]  # Extracted dictionary entries
    page_number: Optional[int] = None  # Page number if known
    processing_time: float = 0.0  # Processing time in seconds
    model_confidence: float = 1.0  # Overall model confidence
    extraction_method: str = "vlm"  # Method used for extraction
    errors: List[str] = None  # List of errors encountered
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata
    
    def __post_init__(self):
        """Post-initialization setup"""
        if self.errors is None:
            self.errors = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'source_image': self.source_image,
            'entries': [entry.to_dict() for entry in self.entries],
            'page_number': self.page_number,
            'processing_time': self.processing_time,
            'model_confidence': self.model_confidence,
            'extraction_method': self.extraction_method,
            'errors': self.errors,
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionResult':
        """Create instance from dictionary"""
        # Convert entry dictionaries back to DictionaryEntry objects
        entries = [DictionaryEntry.from_dict(entry) for entry in data.get('entries', [])]
        data_copy = data.copy()
        data_copy['entries'] = entries
        return cls(**data_copy)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ExtractionResult':
        """Create instance from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_valid_entries(self) -> List[DictionaryEntry]:
        """Get only valid entries"""
        return [entry for entry in self.entries if entry.is_valid()]
    
    def get_entry_count(self) -> int:
        """Get count of valid entries"""
        return len(self.get_valid_entries())
    
    def has_errors(self) -> bool:
        """Check if there were extraction errors"""
        return len(self.errors) > 0
    
    def add_error(self, error: str):
        """Add an error message"""
        self.errors.append(error)
    
    def merge_with(self, other: 'ExtractionResult') -> 'ExtractionResult':
        """Merge with another extraction result"""
        merged_entries = self.entries + other.entries
        merged_errors = self.errors + other.errors
        
        # Merge metadata
        merged_metadata = self.metadata.copy()
        merged_metadata.update(other.metadata)
        
        return ExtractionResult(
            source_image=f"{self.source_image},{other.source_image}",
            entries=merged_entries,
            page_number=self.page_number or other.page_number,
            processing_time=self.processing_time + other.processing_time,
            model_confidence=min(self.model_confidence, other.model_confidence),
            extraction_method=f"{self.extraction_method},{other.extraction_method}",
            errors=merged_errors,
            metadata=merged_metadata
        )

# JSON Schema definitions for validation
DICTIONARY_ENTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "grapheme": {"type": "string"},
        "ipa": {"type": ["string", "null"]},
        "english_meaning": {"type": ["string", "null"]},
        "part_of_speech": {"type": ["string", "null"]},
        "context": {"type": ["string", "null"]},
        "etymology": {"type": ["string", "null"]},
        "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "page_number": {"type": ["integer", "null"]},
        "bbox": {
            "type": ["array", "null"],
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4
        }
    },
    "required": ["grapheme"],
    "additionalProperties": False
}

EXTRACTION_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "source_image": {"type": "string"},
        "entries": {
            "type": "array",
            "items": DICTIONARY_ENTRY_SCHEMA
        },
        "page_number": {"type": ["integer", "null"]},
        "processing_time": {"type": "number", "minimum": 0.0},
        "model_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "extraction_method": {"type": "string"},
        "errors": {
            "type": "array",
            "items": {"type": "string"}
        },
        "metadata": {"type": "object"}
    },
    "required": ["source_image", "entries"],
    "additionalProperties": False
}
