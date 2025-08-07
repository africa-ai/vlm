"""
vLLM Server Setup and Usage Guide for Cosmos-Reason1-7B
"""

print("=== vLLM Server Integration for Cosmos-Reason1-7B ===\n")

print("🚀 QUICK START:")
print("1. Install vLLM dependencies:")
print("   pip install -r requirements_vllm.txt")
print()

print("2. Start the vLLM server:")
print("   python start_vllm_server.py")
print("   # Or use the batch script:")
print("   start_server.bat")
print()

print("3. Process images with vLLM server:")
print("   python main.py process ./images --use-vllm-server --output ./results")
print("   # Or full pipeline:")
print("   python main.py pipeline kalenjin_dictionary.pdf --use-vllm-server --output ./results")
print()

print("📋 BENEFITS OF vLLM SERVER:")
print("✅ Better GPU memory management")
print("✅ Faster inference with optimized kernels")
print("✅ Support for parallel processing")
print("✅ Persistent model loading (no reload between requests)")
print("✅ REST API compatible with OpenAI format")
print("✅ Better scalability for production use")
print()

print("🔧 CONFIGURATION:")
print("Default server settings:")
print("- Model: nvidia/Cosmos-Reason1-7B")
print("- Host: localhost:8000") 
print("- GPU Memory: 85%")
print("- Max Sequences: 4")
print("- Tensor Parallel: 1 (single GPU)")
print()

print("🧪 TEST THE SERVER:")
print("   python test_vllm.py --wait-for-server")
print("   # With test image:")
print("   python test_vllm.py --test-image path/to/dictionary/page.png")
print()

print("🌟 Features now available:")
print("• High-performance vLLM serving")
print("• Distributed inference")
print("• Production-ready scaling")
print("• OpenAI-compatible API")
print("• Advanced reasoning with Cosmos-Reason1-7B")
print()

print("Ready to process Kalenjin dictionaries with vLLM! 🎉")
