"""
Setup and installation helper for Kalenjin Dictionary Processing Framework
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_pip():
    """Check if pip is available"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        return False

def install_requirements():
    """Install requirements from requirements.txt"""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    print("📦 Installing Python packages...")
    try:
        # Install basic requirements first
        basic_cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
        subprocess.run(basic_cmd, check=True)
        
        # Install remaining requirements
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        subprocess.run(cmd, check=True)
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False

def check_gpu_support():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"✅ CUDA available: {device_count} GPU(s)")
            print(f"   Primary GPU: {device_name}")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed - cannot check GPU support")
        return False

def create_directories():
    """Create necessary directories"""
    dirs = ["output", "models", "temp"]
    
    for dirname in dirs:
        dir_path = Path(dirname)
        dir_path.mkdir(exist_ok=True)
        print(f"📁 Created directory: {dirname}")
    
    return True

def download_model(model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    """Download the VLM model"""
    print(f"🤖 Checking model availability: {model_name}")
    
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        
        # Check if model exists
        print("📥 Downloading model (this may take a while)...")
        model_path = snapshot_download(repo_id=model_name, cache_dir="./models")
        print(f"✅ Model downloaded to: {model_path}")
        
        return True
    except ImportError:
        print("⚠️  huggingface_hub not available - model will download on first use")
        return True
    except Exception as e:
        print(f"⚠️  Model download failed: {e}")
        print("💡 Model will be downloaded automatically on first use")
        return True

def test_installation():
    """Test if installation works"""
    print("\n🧪 Testing installation...")
    
    # Test PDF processing
    try:
        from scripts.pdf_to_images import PDFToImageConverter
        converter = PDFToImageConverter()
        print("✅ PDF processing module works")
    except Exception as e:
        print(f"❌ PDF processing test failed: {e}")
        return False
    
    # Test VLM imports
    try:
        from llm.config import VLMConfig
        from llm.main import VLMProcessor
        config = VLMConfig()
        print("✅ VLM modules work")
    except Exception as e:
        print(f"❌ VLM modules test failed: {e}")
        return False
    
    # Test parser
    try:
        from llm.parser.schemas import DictionaryEntry
        entry = DictionaryEntry(grapheme="test")
        print("✅ Parser modules work")
    except Exception as e:
        print(f"❌ Parser modules test failed: {e}")
        return False
    
    print("✅ All tests passed!")
    return True

def main():
    """Main setup function"""
    print("🚀 Kalenjin Dictionary Processing Framework Setup\n")
    
    # Check prerequisites
    if not check_python_version():
        return 1
    
    if not check_pip():
        return 1
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install packages
    print("\n📦 Installing dependencies...")
    if not install_requirements():
        return 1
    
    # Check GPU support
    print("\n🖥️  Checking hardware support...")
    has_gpu = check_gpu_support()
    
    # Update environment file with GPU setting
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, "r") as f:
            content = f.read()
        
        if has_gpu:
            content = content.replace("VLM_DEVICE=cpu", "VLM_DEVICE=cuda")
            content = content.replace("CUDA_AVAILABLE=false", "CUDA_AVAILABLE=true")
        else:
            content = content.replace("VLM_DEVICE=cuda", "VLM_DEVICE=cpu")
        
        with open(env_file, "w") as f:
            f.write(content)
    
    # Download model (optional)
    if input("\n🤖 Download VLM model now? (y/N): ").lower().startswith('y'):
        download_model()
    else:
        print("⏭️  Skipping model download (will download on first use)")
    
    # Test installation
    if not test_installation():
        print("\n❌ Setup completed with errors")
        return 1
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Place your Kalenjin dictionary PDF in the project root")
    print("2. Run demo: python demo.py")
    print("3. Or run full pipeline: python main.py pipeline your_dictionary.pdf")
    
    return 0

if __name__ == "__main__":
    exit(main())
