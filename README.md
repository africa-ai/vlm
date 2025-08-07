# Kalenjin Dictionary Processing Framework

A comprehensive framework for extracting linguistic data from Kalenjin dictionary PDFs using **NVIDIA Cosmos-Reason1-7B** Vision Language Model. This advanced reasoning model systematically converts PDF pages to high-resolution images and processes them to extract:

- **Graphemes** (original Kalenjin words/spelling)
- **IPA** (International Phonetic Alphabet transcriptions)
- **English meanings** (translations and definitions)

## Project Structure

```
ocr/
├── scripts/
│   └── pdf_to_images.py          # PDF to high-res image conversion
├── llm/
│   ├── __init__.py               # Package initialization
│   ├── config.py                 # VLM configuration settings
│   ├── main.py                   # Main VLM processor
│   └── parser/
│       ├── __init__.py           # Parser package init
│       ├── main.py               # Dictionary entry parser
│       ├── prompts.py            # VLM prompt templates
│       └── schemas.py            # Data schemas and validation
├── output/                       # Generated results (JSON/CSV)
├── main.py                       # Main CLI entry point
├── utils.py                      # Utility functions
├── requirements.txt              # Python dependencies
├── .env                          # Environment configuration
├── kalenjin_dictionary.pdf       # Source dictionary
└── README.md                     # This file
```

## Dictionary Format Support

This framework is specifically optimized for Kalenjin dictionary format with:

### Structure Recognition
- **Two-column layout** with alphabetical entries
- **Entry pattern**: `grapheme + part-of-speech + IPA + definition + context`
- **IPA notations**: Forward slashes `/ke:-apus/` and underscores `/_apa/`
- **POS abbreviations**: `v.t.` (transitive verb), `v.i.` (intransitive verb), `n.` (noun), etc.

### Sample Entry Processing
```
Input: "abus v.t. /ke:-apus/ to give bad advice, to fool, make a fool of. Kogiabus. (S/he) was ill-advised, fooled."

Extracted:
{
  "grapheme": "abus",
  "ipa": "/ke:-apus/",
  "english_meaning": "to give bad advice, to fool, make a fool of",
  "part_of_speech": "v.t.",
  "context": "Kogiabus. (S/he) was ill-advised, fooled"
}
```

### Special Features Handled
- **Cross-references**: Capitalized related entries (e.g., "Kogiabus")
- **Usage examples**: Questions and statements ("Ingoro aba? Where is father?")
- **Multiple IPA variants**: Complex notations like `/_apa/, _apa (nom.)/`
- **Alternate forms**: Listed variations (e.g., "apay-wa, apay-we:k")

## Features

### 🔄 PDF Processing
- High-resolution PDF to image conversion (300+ DPI)
- Batch processing of multiple PDFs
- Support for PNG, JPEG formats

### 🤖 VLM Processing
- **NVIDIA Cosmos-Reason1-7B** model integration with advanced reasoning
- Systematic step-by-step analysis of dictionary entries  
- GPU acceleration with CUDA support
- Configurable batch processing
- Enhanced accuracy through reasoning capabilities

### 📊 Data Extraction
- Structured extraction of dictionary entries
- Support for graphemes, IPA, and English meanings
- Confidence scoring for extractions
- Error handling and logging

### 💾 Output Formats
- JSON format for structured data
- CSV format for analysis
- Comprehensive statistics and reports

## Installation

1. **Clone or create the project structure**:
```bash
git clone <repository-url>  # or create the directory structure manually
cd ocr
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment** (optional):
```bash
cp .env.example .env
# Edit .env with your preferred settings
```

4. **Download the VLM model** (optional - will auto-download on first use):
```bash
python -c "from transformers import Qwen2VLForConditionalGeneration; Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')"
```

## Usage

### Command Line Interface

The framework provides several commands for different use cases:

#### 1. Convert PDF to Images Only
```bash
python main.py convert kalenjin_dictionary.pdf --output ./images --dpi 300
```

#### 2. Process Images with VLM
```bash
python main.py process ./images --output ./results --batch-size 4
```

#### 3. Full Pipeline (PDF → Images → VLM)
```bash
python main.py pipeline kalenjin_dictionary.pdf --output ./results --dpi 300 --batch-size 4
```

#### 4. Batch Process Multiple PDFs
```bash
python main.py batch ./pdf_directory --output ./batch_results
```

### Python API Usage

```python
from scripts.pdf_to_images import PDFToImageConverter
from llm.main import VLMProcessor
from llm.config import VLMConfig

# Convert PDF to images
converter = PDFToImageConverter(dpi=300)
image_paths = converter.convert_pdf_to_images("dictionary.pdf", "./images")

# Configure VLM
config = VLMConfig(
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    device="cuda",
    batch_size=4,
    output_dir="./results"
)

# Process with VLM
processor = VLMProcessor(config)
processor.load_model()
results = processor.batch_process_images(image_paths)
processor.save_results(results)
```

## Configuration

### Environment Variables (.env)
```bash
VLM_MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
VLM_DEVICE=cuda
VLM_BATCH_SIZE=4
VLM_MAX_TOKENS=2048
VLM_TEMPERATURE=0.1
IMAGE_DPI=300
LOG_LEVEL=INFO
```

### Model Configuration
```python
from llm.config import VLMConfig

config = VLMConfig(
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    device="cuda",                    # or "cpu"
    batch_size=4,                     # Adjust based on GPU memory
    max_new_tokens=2048,
    temperature=0.1,                  # Low for consistent extraction
    output_dir="./results"
)
```

## Output Format

### JSON Structure
```json
[
  {
    "image_path": "page_001.png",
    "entries": [
      {
        "grapheme": "laibartut",
        "ipa": "/laɪ'bartʊt/",
        "english_meaning": "to speak, to talk",
        "part_of_speech": "verb",
        "context": "laibartutab he/she speaks",
        "confidence_score": 0.95
      }
    ],
    "status": "success",
    "timestamp": "2024-01-15T10:30:00"
  }
]
```

### CSV Format
```csv
image_path,grapheme,ipa,english_meaning,part_of_speech,context,confidence_score,status,timestamp
page_001.png,laibartut,/laɪ'bartʊt/,"to speak, to talk",verb,"laibartutab he/she speaks",0.95,success,2024-01-15T10:30:00
```

## Hardware Requirements

### Minimum Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 10GB+ free space
- **GPU**: Optional but recommended (4GB+ VRAM)

### Recommended for Optimal Performance
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **GPU**: NVIDIA RTX 3080+ (8GB+ VRAM)
- **Storage**: SSD with 20GB+ free space

## Performance Tips

1. **GPU Acceleration**: Use CUDA-compatible GPU for 5-10x speed improvement
2. **Batch Size**: Increase batch size if you have more GPU memory
3. **Image Resolution**: Balance quality vs. processing time (300 DPI recommended)
4. **Flash Attention**: Enable for additional GPU optimization
5. **Model Caching**: Keep models in local cache to avoid re-downloading

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size
python main.py process ./images --batch-size 1

# Or use CPU
python main.py process ./images --device cpu
```

#### Model Download Issues
```bash
# Set cache directory
export TRANSFORMERS_CACHE=/path/to/large/storage
python main.py process ./images
```

#### Poor Extraction Quality
- Increase image DPI (--dpi 400)
- Check image quality and contrast
- Verify dictionary format matches expected structure

### Logs and Debugging
```bash
# Enable debug logging
python main.py process ./images --log-level DEBUG --log-file debug.log

# Check extraction statistics
python -c "
from utils import calculate_extraction_stats
import json
with open('results.json') as f:
    results = json.load(f)
stats = calculate_extraction_stats(results)
print(json.dumps(stats, indent=2))
"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- **QWEN Team** for the excellent VL model
- **Hugging Face** for the transformers library
- **PyMuPDF** for PDF processing capabilities

## Contact

[Add contact information or support channels]

---

**Note**: This framework is specifically designed for Kalenjin dictionary processing but can be adapted for other linguistic extraction tasks by modifying the prompt templates and schemas.
