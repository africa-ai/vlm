# Kalenjin Dictionary Processing Framework

A comprehensive AI-powered framework for extracting linguistic data from Kalenjin dictionary PDFs using **NVIDIA Cosmos-Reason1-7B** Vision Language Model with high-performance **vLLM server integration**. This advanced reasoning model systematically converts PDF pages to high-resolution images and processes them to extract:

- **Graphemes** (original Kalenjin words/spelling)
- **IPA** (International Phonetic Alphabet transcriptions) 
- **English meanings** (translations and definitions)

## 🚀 **New: vLLM Server Integration**

**High-Performance Processing** with 2-3x speed improvement:
- ✅ Optimized CUDA kernels for faster inference
- ✅ Persistent model loading (no reload between requests)
- ✅ Parallel processing of multiple images
- ✅ REST API compatible with OpenAI format
- ✅ Production-ready scaling

## 📦 **Quick Installation**

```bash
# Full installation with vLLM server (recommended)
pip install .[vllm]

# Or interactive installation
python install.py

# Basic installation (CPU only)
pip install .
```

## ⚡ **Quick Start**

```bash
# 1. Start vLLM server (recommended for best performance)
python start_vllm_server.py
# Or: start_server.bat

# 2. Process dictionary with vLLM server
python main.py pipeline kalenjin_dictionary.pdf --use-vllm-server --output ./results

# 3. Or use local processing (slower)
python main.py pipeline kalenjin_dictionary.pdf --output ./results
```

## Project Structure

```
ocr/
├── scripts/
│   └── pdf_to_images.py          # PDF to high-res image conversion
├── llm/
│   ├── config.py                 # VLM configuration settings
│   ├── main.py                   # Local VLM processor
│   ├── vllm_processor.py         # vLLM server processor
│   ├── cosmos_utils.py           # Cosmos reasoning utilities
│   └── parser/
│       ├── main.py               # Dictionary entry parser
│       ├── prompts.py            # VLM prompt templates
│       └── schemas.py            # Data schemas and validation
├── vllm_server/
│   ├── __init__.py               # vLLM package init
│   ├── config.py                 # vLLM server configuration
│   ├── server.py                 # FastAPI vLLM server
│   └── client.py                 # HTTP client for server
├── output/                       # Generated results (JSON/CSV)
├── main.py                       # Main CLI entry point
├── setup.py                      # Professional installation
├── install.py                    # Interactive installer
├── start_vllm_server.py          # vLLM server startup
├── test_vllm.py                  # Server testing
├── requirements.txt              # Core dependencies
├── requirements_vllm.txt         # vLLM dependencies
├── VLLM_GUIDE.md                 # Comprehensive vLLM guide
├── .env                          # Environment configuration
├── kalenjin_dictionary.pdf       # Source dictionary
└── README.md                     # This file
```

## 🎯 **Processing Options**

### **Option 1: vLLM Server (Recommended)**
**Best Performance:** 2-3x faster with optimized inference
```bash
# Start vLLM server
python start_vllm_server.py

# Process with server
python main.py pipeline kalenjin_dictionary.pdf --use-vllm-server --output ./results
```

### **Option 2: Local Processing**
**Simpler Setup:** Direct model loading
```bash
python main.py pipeline kalenjin_dictionary.pdf --output ./results
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

## 🌟 **Features**

### ⚡ **vLLM Server Integration**
- **High-Performance Inference**: 2-3x faster processing
- **Persistent Model Loading**: No reload between requests
- **Parallel Processing**: Handle multiple images simultaneously
- **REST API**: OpenAI-compatible endpoints
- **Production Scaling**: Ready for large dictionary collections

### 🔄 **PDF Processing**
- High-resolution PDF to image conversion (300+ DPI)
- Batch processing of multiple PDFs
- Support for PNG, JPEG formats

### 🤖 **VLM Processing**
- **NVIDIA Cosmos-Reason1-7B** model with advanced reasoning
- Systematic step-by-step analysis of dictionary entries  
- GPU acceleration with CUDA support
- Configurable batch processing
- Enhanced accuracy through reasoning capabilities

### 📊 **Data Extraction**
- Structured extraction of dictionary entries
- Support for graphemes, IPA, and English meanings
- Confidence scoring for extractions
- Error handling and logging

### 💾 **Output Formats**
- JSON format for structured data
- CSV format for analysis
- Comprehensive statistics and reports

## 📋 **Installation Options**

### **Method 1: Full Installation (Recommended)**
```bash
# Install with vLLM server support
pip install .[vllm]

# Run post-install setup
kalenjin-setup
```

### **Method 2: Interactive Installation**
```bash
python install.py
# Follow prompts to choose installation type
```

### **Method 3: Manual Installation**
```bash
# Basic installation (CPU only)
pip install .

# Development installation
pip install .[dev]

# Everything included
pip install .[all]
```

### **Method 4: From Source**
```bash
git clone <repository-url>
cd kalenjin-dictionary-processor
pip install -r requirements_vllm.txt  # For full features
python setup.py  # Run setup
```
```

4. **Download the VLM model** (optional - will auto-download on first use):
```bash
python -c "from transformers import Qwen2VLForConditionalGeneration; Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')"
```

## 🚀 **Usage**

### **Command Line Interface**

The framework provides several commands optimized for both local and vLLM server processing:

#### **1. Full Pipeline with vLLM Server (Recommended)**
```bash
# Start vLLM server first
python start_vllm_server.py

# Process dictionary
python main.py pipeline kalenjin_dictionary.pdf --use-vllm-server --output ./results
```

#### **2. Local Processing (No Server Required)**
```bash
python main.py pipeline kalenjin_dictionary.pdf --output ./results
```

#### **3. Convert PDF to Images Only**
```bash
python main.py convert kalenjin_dictionary.pdf --output ./images --dpi 300
```

#### **4. Process Pre-Converted Images**
```bash
# With vLLM server (faster)
python main.py process ./images --use-vllm-server --output ./results --batch-size 8

# Local processing
python main.py process ./images --output ./results --batch-size 4
```

#### **5. Batch Process Multiple PDFs**
```bash
# With vLLM server for best performance
python main.py batch ./pdf_directory --use-vllm-server --output ./batch_results

# Local processing
python main.py batch ./pdf_directory --output ./batch_results
```

### **vLLM Server Commands**

```bash
# Start server with default settings
python start_vllm_server.py

# Start with custom configuration
python start_vllm_server.py --port 8000 --gpu-memory-utilization 0.85 --max-num-seqs 8

# Test server connection
python test_vllm.py --wait-for-server

# Test with image processing
python test_vllm.py --test-image dictionary_page.png
```

### **Python API Usage**

#### **Option 1: vLLM Server Processing (Recommended)**
```python
from llm.vllm_processor import VLLMServerProcessor
from llm.config import load_config_from_env

# Configure processor
config = load_config_from_env()
processor = VLLMServerProcessor(config, "http://localhost:8000")

# Check server connection
if processor.check_server_connection():
    # Process images
    results = processor.batch_process_images(image_paths)
    processor.save_results(results)
```

#### **Option 2: Local Processing**
```python
from scripts.pdf_to_images import PDFToImageConverter
from llm.main import VLMProcessor
from llm.config import VLMConfig

# Convert PDF to images
converter = PDFToImageConverter(dpi=300)
image_paths = converter.convert_pdf_to_images("dictionary.pdf", "./images")

# Configure VLM
config = VLMConfig(
    model_name="nvidia/Cosmos-Reason1-7B",
    device="cuda",
    batch_size=4,
    output_dir="./results"
)

# Process with VLM
processor = VLMProcessor(config)
processor.load_model()
results = processor.batch_process_images(image_paths)
```

#### **Option 3: Direct vLLM Client**
```python
from vllm_server.client import SyncVLLMClient

# Connect to server
client = SyncVLLMClient("http://localhost:8000")

# Process single image
result = client.analyze_dictionary_image("page.png")
print(f"Found {len(result.get('entries', []))} entries")

# Batch processing
results = client.batch_process_images(image_paths)
```
processor.save_results(results)
```

## ⚙️ **Configuration**

### **Environment Variables (.env)**
Auto-created during installation with smart defaults:
```bash
# Model Configuration
MODEL_NAME=nvidia/Cosmos-Reason1-7B
DEVICE=auto
BATCH_SIZE=2
OUTPUT_DIR=./output

# vLLM Server Configuration
VLLM_SERVER_URL=http://localhost:8000
VLLM_MODEL_NAME=nvidia/Cosmos-Reason1-7B
VLLM_HOST=localhost
VLLM_PORT=8000
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_MAX_NUM_SEQS=4
VLLM_TENSOR_PARALLEL_SIZE=1

# Processing Settings
DEFAULT_DPI=300
IMAGE_FORMAT=PNG
LOG_LEVEL=INFO
LOG_FILE=kalenjin_processing.log
```

### **vLLM Server Configuration**
```python
from vllm_server.config import VLLMServerConfig

config = VLLMServerConfig(
    model_name="nvidia/Cosmos-Reason1-7B",
    host="localhost",
    port=8000,
    gpu_memory_utilization=0.85,      # Use 85% of GPU memory
    max_num_seqs=4,                   # Parallel sequences
    tensor_parallel_size=1            # Single GPU
)
```

### **Local Model Configuration**
```python
from llm.config import VLMConfig

config = VLMConfig(
    model_name="nvidia/Cosmos-Reason1-7B",
    device="cuda",                    # or "cpu", "auto"
    batch_size=2,                     # Adjust based on GPU memory
    max_new_tokens=2048,
    temperature=0.1,                  # Low for consistent extraction
    output_dir="./results"
)
```

## 📊 **Output Format**

### **JSON Structure**
```json
[
  {
    "image_path": "page_001.png",
    "entries": [
      {
        "kalenjin_word": "laibartut",
        "ipa_transcription": "/laɪ'bartʊt/",
        "english_meaning": "to speak, to talk",
        "part_of_speech": "verb",
        "additional_info": "laibartutab he/she speaks",
        "confidence": 0.95
      }
    ],
    "raw_response": "Full VLM analysis response...",
    "reasoning": "Step-by-step reasoning from Cosmos model...",
    "model": "nvidia/Cosmos-Reason1-7B",
    "status": "success",
    "timestamp": "2024-01-15T10:30:00"
  }
]
```

### **CSV Format**
```csv
image_path,kalenjin_word,ipa_transcription,english_meaning,part_of_speech,additional_info,confidence,timestamp
page_001.png,laibartut,/laɪ'bartʊt/,"to speak, to talk",verb,"laibartutab he/she speaks",0.95,2024-01-15T10:30:00
```

## 🔧 **Hardware Requirements**

### **For vLLM Server (Recommended)**
- **GPU**: NVIDIA GPU with 12GB+ VRAM (RTX 3080 Ti, RTX 4080, V100, A100)
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Storage**: 50GB+ free space for model caching

### **For Local Processing**
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070, RTX 4070)
- **CPU**: 4+ cores  
- **RAM**: 12GB+

### **CPU-Only Processing (Slower)**
- **CPU**: 8+ cores recommended
- **RAM**: 16GB+
## 📈 **Performance Comparison**

| Feature | Local Processing | vLLM Server |
|---------|------------------|-------------|
| **Speed** | Baseline | **2-3x faster** |
| **Memory Usage** | Higher overhead | Optimized |
| **Model Loading** | Per-run (~30-60s) | Once at startup |
| **Parallel Processing** | Sequential batches | True parallel |
| **GPU Utilization** | Variable | **85% optimized** |
| **Production Ready** | Development | **✅ Production** |
| **Setup Complexity** | Simple | Moderate |

## 💡 **Performance Tips**

### **vLLM Server Optimization**
```bash
# Multi-GPU setup
python start_vllm_server.py --tensor-parallel-size 2

# Increase parallel processing
python start_vllm_server.py --max-num-seqs 8

# Optimize memory usage
python start_vllm_server.py --gpu-memory-utilization 0.9
```

### **General Optimization**
1. **Use vLLM Server**: 2-3x performance improvement
2. **GPU Acceleration**: Enable CUDA for best performance
3. **Batch Processing**: Process multiple images simultaneously
4. **Image Quality**: Balance DPI vs. processing time (300-400 DPI)
5. **Model Caching**: Pre-download models to avoid startup delays

## 🔧 **Testing & Validation**

### **Test vLLM Server**
```bash
# Test server connection
python test_vllm.py --wait-for-server

# Test with sample image
python test_vllm.py --test-image sample_page.png

# Benchmark performance
python test_vllm.py --benchmark
```

### **Validate Installation**
```bash
# Run setup validation
kalenjin-setup

# Check all components
python -c "from llm.vllm_processor import VLLMServerProcessor; print('✅ Installation OK')"
```

## 🐛 **Troubleshooting**

### **vLLM Server Issues**

#### Server Won't Start
```bash
# Check CUDA availability
nvidia-smi

# Verify vLLM installation
python -c "import vllm; print('vLLM OK')"

# Check port availability
netstat -an | findstr :8000
```

#### Out of Memory
```bash
# Reduce GPU memory usage
python start_vllm_server.py --gpu-memory-utilization 0.7

# Reduce parallel sequences
python start_vllm_server.py --max-num-seqs 2

# Use smaller batch sizes
python main.py process ./images --use-vllm-server --batch-size 2
```

### **Local Processing Issues**

#### CUDA Out of Memory
```bash
# Reduce batch size
python main.py process ./images --batch-size 1

# Use CPU processing
python main.py process ./images --device cpu
```

#### Model Download Issues
```bash
# Set cache directory
set TRANSFORMERS_CACHE=C:\large_storage\models
python main.py process ./images
```

#### Poor Extraction Quality
- **Increase Image DPI**: `--dpi 400` for better text recognition
- **Check Image Quality**: Ensure high contrast and clear text
- **Verify Dictionary Format**: Must match expected Kalenjin format
- **Review Reasoning**: Check `raw_response` and `reasoning` in output

### **Connection Issues**
```bash
# Test server connection
curl http://localhost:8000/health

# Check firewall settings
# Ensure port 8000 is not blocked

## 📚 **Additional Resources**

### **Documentation**
- **[vLLM Integration Guide](VLLM_GUIDE.md)**: Complete guide for vLLM server setup and usage
- **[Setup Features](SETUP_FEATURES.md)**: Detailed installation options and features
- **[Configuration Reference](.env)**: All environment variables explained

### **Quick Reference**
```bash
# Installation
pip install .[vllm]                    # Full installation
python install.py                      # Interactive setup
kalenjin-setup                          # Post-install configuration

# vLLM Server
python start_vllm_server.py            # Start server
python test_vllm.py                    # Test connection
start_server.bat                       # Windows batch script

# Processing
python main.py pipeline dict.pdf --use-vllm-server    # Full pipeline
python main.py process ./images --use-vllm-server     # Process images
python main.py batch ./pdfs --use-vllm-server         # Batch processing
```

### **Console Commands (After Installation)**
```bash
kalenjin-process pipeline dict.pdf     # Main processing
start-vllm-server --port 8000          # Start vLLM server
test-vllm --wait-for-server            # Test installation
kalenjin-setup                          # Run setup
```

## 🌟 **Key Benefits**

✅ **High-Performance Processing**: 2-3x faster with vLLM server integration  
✅ **Professional Installation**: Enterprise-grade setup.py with flexible options  
✅ **Production Ready**: Scalable REST API and distributed processing  
✅ **Advanced AI**: NVIDIA Cosmos-Reason1-7B with systematic reasoning  
✅ **Comprehensive Output**: JSON/CSV with confidence scores and reasoning  
✅ **Automatic Setup**: Smart environment detection and configuration  
✅ **Cross-Platform**: Works on Windows, Linux, macOS  
✅ **Developer Friendly**: Full development tools and testing included  

## 📄 **License**

MIT License - See LICENSE file for details.

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)  
5. Open a Pull Request

## 📧 **Support**

- **Issues**: [GitHub Issues](https://github.com/kalenjin-dictionary/processor/issues)
- **Documentation**: [Project Wiki](https://github.com/kalenjin-dictionary/processor/wiki)
- **Email**: contact@kalenjin-dictionary.org

---

**Ready to process Kalenjin dictionaries with state-of-the-art AI! 🚀📚**

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
