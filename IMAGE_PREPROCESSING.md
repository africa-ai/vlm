# Image Preprocessing for VLM Processing

## Problem: Large Token Usage

When processing high-resolution dictionary images, the VLM model can generate over 1 million tokens per image, which exceeds the model's context window (128K tokens max). This causes processing failures.

## Solution: Image Preprocessing & Splitting

### 1. **Automatic Image Optimization**
- Resize images to optimal dimensions (1024x1024 max)
- Enhance contrast and sharpness for better text recognition
- Reduce file size while maintaining quality
- Estimate token usage before processing

### 2. **Intelligent Image Splitting**
- Split large images vertically into manageable parts
- Target ~30K tokens per part (well within model limits)
- Process each part independently
- Combine results from all parts

### 3. **Updated Processing Flow**
```
Original Image (e.g., 4000x6000, ~1.1M tokens)
    ↓
Resize & Optimize (1024x1536, ~150K tokens)
    ↓
Split Vertically (4 parts of 1024x384, ~40K tokens each)
    ↓
Process Each Part with VLM
    ↓
Combine All Results
```

## Usage

### Tomorrow's Multi-GPU Setup
```bash
# Start vLLM server on 2 GPUs
LD_LIBRARY_PATH="/home/coder/vlm/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH" \
python start_vllm_server.py --tensor-parallel-size 2 --max-model-len 32768

# Process with preprocessing (automatic)
python main.py pipeline kalenjin_dictionary.pdf --use-vllm-server --output ./results
```

### Test Preprocessing
```bash
# Test on a sample image
python test_preprocessing.py kalenjin_dictionary.pdf.page_1.png
```

## Benefits

1. **Fits Model Limits**: Images split to stay within 32K token context
2. **Better Performance**: Smaller images process faster
3. **Higher Quality**: Enhanced text recognition
4. **Scalable**: Can handle any size dictionary
5. **Multi-GPU Ready**: Works with tensor parallelism

## Technical Details

### Image Preprocessing (`llm/image_preprocessor.py`)
- `ImagePreprocessor`: Core preprocessing class
- `optimize_image_for_vlm()`: Complete optimization pipeline
- Token estimation based on image dimensions
- Automatic splitting when needed

### Updated VLM Processor (`llm/vllm_processor.py`)
- Automatic preprocessing for all images
- Base64 image handling for split parts
- Combined results from multiple parts
- Enhanced error handling and logging

### Enhanced Client (`vllm_server/client.py`)
- New `analyze_dictionary_image_base64()` method
- Support for processed image parts
- Better error handling for large images

This approach transforms the 1.1M token problem into manageable 30K token chunks, making your dictionary processing both reliable and efficient! 🚀
