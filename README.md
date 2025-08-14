# Kalenjin Dictionary OCR & Extraction

A clean and efficient pipeline to extract dictionary entries from PDF documents using OCR and language models.

## Architecture

**Simple Pipeline: PDF → Images → OCR → vLLM → JSON**

1. **PDF to Images**: Convert PDF pages to PNG images using PyMuPDF
2. **OCR**: Extract raw text using pytesseract (Tesseract OCR)
3. **vLLM Server**: Process extracted text with NVIDIA Cosmos-Reason1-7B to structure dictionary entries
4. **JSON Output**: Clean dictionary entries with translations and metadata

## Why This Approach?

- **Simple & Effective**: OCR extracts text cleanly, LLM structures it perfectly
- **Fast Processing**: No complex image preprocessing or vision model overhead
- **Reliable Results**: OCR is mature technology, LLM excels at text structuring
- **Resource Efficient**: Uses GPU only for LLM text processing, not image analysis

## Quick Start

### 1. Install Dependencies
```bash
python install.py
```
This installs:
- pytesseract + tesseract OCR engine
- vLLM server with CUDA support
- PDF processing utilities

### 2. Start vLLM Server
```bash
python start_vllm_server.py
```
Starts NVIDIA Cosmos-Reason1-7B model server with:
- Multi-GPU tensor parallelism (if available)
- OpenAI-compatible API endpoints
- Optimized CUDA kernels for fast inference

### 3. Process Dictionary
```bash
# Full pipeline (recommended)
python main.py pipeline kalenjin_dictionary.pdf --output ./results

# Or step by step
python main.py images kalenjin_dictionary.pdf --output ./results
python main.py ocr ./results/images --output ./results
```

## Features

### Core Capabilities
- **Clean Text Extraction**: pytesseract OCR for reliable text extraction
- **Intelligent Structuring**: vLLM server processes OCR text into structured entries
- **Batch Processing**: Handle multiple pages efficiently
- **Real-time Progress**: Monitor extraction with live result files

### Technical Features
- **GPU Accelerated**: vLLM server with multi-GPU support (2×A10G tested)
- **Memory Optimized**: Smart batching prevents OOM errors
- **Error Handling**: Robust processing with detailed error reporting
- **Configurable**: Environment-based configuration for different setups

## Output Format

Dictionary entries are extracted as clean JSON:

```json
{
  "grapheme": "amun",
  "english_meaning": "to drink water or other liquids", 
  "ipa": "/a.mun/",
  "confidence": 0.85,
  "page": 15,
  "metadata": {
    "ocr_confidence": 0.92,
    "processing_time": 1.2
  }
}
```

## Configuration

### Environment Variables
```bash
# vLLM Server
export VLLM_SERVER_URL="http://localhost:8000"
export MODEL_NAME="nvidia/Cosmos-Reason1-7B-Instruct"

# Processing
export MAX_WORKERS=4
export BATCH_SIZE=8

# Multi-GPU (if needed)
export TENSOR_PARALLEL_SIZE=2
export GPU_MEMORY_UTILIZATION=0.8
```

### Files
- `llm/config.py`: Processing configuration
- `vllm_server/config.py`: Server configuration
- Environment variables override defaults

## Performance

### Typical Results (2×NVIDIA A10G)
- **Speed**: 30-60 seconds per page (including OCR + LLM processing)
- **Extraction**: 15-40 dictionary entries per page
- **Accuracy**: 85-95% confidence for clear dictionary entries
- **Memory**: ~16GB GPU memory with tensor parallelism

### Optimization Tips
1. **Use vLLM Server**: 3-5x faster than direct model loading
2. **Enable Multi-GPU**: Use `--tensor-parallel-size 2` for dual GPUs
3. **Batch Processing**: Process multiple images together for efficiency
4. **Monitor Progress**: Watch live result files during processing

## Testing

Test the complete pipeline:
```bash
python test_clean_pipeline.py
```

This verifies:
- vLLM server connectivity
- OCR functionality  
- End-to-end processing
- Sample output quality

## Requirements

### System Requirements
- **Python**: 3.8+ 
- **GPU**: CUDA-capable (16GB+ VRAM recommended)
- **OS**: Linux/Windows with CUDA support
- **Storage**: ~5GB for model and dependencies

### Key Dependencies
- **pytesseract**: OCR text extraction
- **vLLM**: High-performance LLM server
- **PyMuPDF**: PDF to image conversion
- **aiohttp**: Async HTTP client for server communication

## Troubleshooting

### Common Issues

#### 1. vLLM Server Won't Start
```bash
# Check GPU availability
nvidia-smi

# Verify CUDA installation  
python -c "import torch; print(torch.cuda.is_available())"

# Check server logs
tail -f vllm_server.log
```

#### 2. OCR Not Working
```bash
# Install tesseract (Ubuntu/Debian)
sudo apt-get install tesseract-ocr

# Install tesseract (Windows)
# Download from: https://github.com/UB-Mannheim/tesseract/wiki

# Test OCR
tesseract --version
```

#### 3. Memory Issues
- Reduce `gpu_memory_utilization` (try 0.7)
- Lower `max_model_len` in server config
- Process fewer images per batch

#### 4. Multi-GPU Issues
```bash
# Set NCCL environment variables
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
```

## Project Structure

```
kalenjin-dictionary-ocr/
├── main.py                    # CLI entry point
├── install.py                 # Installation script
├── start_vllm_server.py      # vLLM server launcher
├── test_clean_pipeline.py    # Pipeline testing
├── llm/
│   ├── config.py             # Configuration management
│   ├── ocr_processor.py      # OCR + LLM processor
│   ├── main.py               # Processing functions
│   └── parser/               # Text parsing utilities
├── vllm_server/              # vLLM server implementation
├── utils/                    # PDF and utility functions
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Development

### Adding Features
1. **New Output Formats**: Extend `llm/parser/` modules
2. **Custom Models**: Update model configurations in `llm/config.py`
3. **Processing Options**: Add new CLI commands in `main.py`
4. **Server Endpoints**: Extend `vllm_server/server.py` APIs

### Contributing
- Follow clean architecture principles
- Add tests for new functionality
- Update documentation
- Maintain OCR → LLM → JSON pipeline simplicity

---

**Clean, simple, effective dictionary extraction using the power of OCR + Language Models.**

*Built with NVIDIA Cosmos-Reason1-7B and vLLM for production-ready performance.*
