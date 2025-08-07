"""
Setup script for Kalenjin Dictionary Processing Framework with vLLM Integration
Provides installation, configuration, and environment setup for the framework
"""

from setuptools import setup, find_packages
from pathlib import Path
import subprocess
import sys
import os

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read version from package
__version__ = "1.0.0"

# Core dependencies
INSTALL_REQUIRES = [
    # Core ML dependencies  
    "torch>=2.0.0",
    "transformers>=4.35.0",
    "accelerate>=0.20.0",
    
    # Image processing
    "pillow>=9.0.0",
    "opencv-python>=4.7.0",
    
    # PDF processing
    "PyMuPDF>=1.23.0",
    "pdf2image>=1.16.0",
    
    # Data processing
    "pandas>=1.5.0",
    "numpy>=1.24.0",
    
    # Utilities
    "jsonschema>=4.17.0",
    "tqdm>=4.65.0",
    "python-dotenv>=1.0.0",
    "huggingface-hub>=0.17.0",
    "regex>=2023.0.0",
    "typing-extensions>=4.5.0",
    
    # HTTP clients for vLLM integration
    "httpx>=0.25.0",
    "aiohttp>=3.9.0",
]

# vLLM server dependencies (optional)
VLLM_REQUIRES = [
    "vllm>=0.6.0",
    "uvicorn>=0.24.0", 
    "fastapi>=0.104.0",
    "pydantic>=2.5.0",
    "python-multipart>=0.0.6",
]

# Development dependencies
DEV_REQUIRES = [
    "pytest>=7.0.0",
    "black>=23.0.0", 
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
]

# Extra dependencies
EXTRAS_REQUIRE = {
    "vllm": VLLM_REQUIRES,
    "dev": DEV_REQUIRES,
    "all": VLLM_REQUIRES + DEV_REQUIRES,
}

def post_install_setup():
    """Run post-installation setup"""
    print("\n🚀 Running post-installation setup...")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create environment file
    create_environment_file()
    
    # Setup directories
    setup_directories()
    
    # Check CUDA after installation
    check_cuda_availability()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Review and update .env file with your preferences")
    print("2. For vLLM server: pip install .[vllm]")
    print("3. Start processing: python main.py --help")
    print("4. Test installation: python test_vllm.py")

def create_environment_file():
    """Create .env file with default configuration"""
    env_file = Path(".env")
    if env_file.exists():
        print("✅ Environment file (.env) already exists")
        return
    
    env_content = """# Kalenjin Dictionary Processing Framework Configuration

# Model Configuration
MODEL_NAME=nvidia/Cosmos-Reason1-7B
DEVICE=auto
BATCH_SIZE=2
OUTPUT_DIR=./output

# vLLM Server Configuration (optional)
VLLM_SERVER_URL=http://localhost:8000
VLLM_MODEL_NAME=nvidia/Cosmos-Reason1-7B
VLLM_HOST=localhost
VLLM_PORT=8000
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_MAX_NUM_SEQS=4
VLLM_TENSOR_PARALLEL_SIZE=1

# Logging
LOG_LEVEL=INFO
LOG_FILE=kalenjin_processing.log

# PDF Processing  
DEFAULT_DPI=300
IMAGE_FORMAT=PNG
"""
    
    env_file.write_text(env_content)
    print("✅ Created environment configuration file (.env)")

def setup_directories():
    """Create necessary directories"""
    directories = [
        "output",
        "images", 
        "results",
        "logs",
        "cache",
    ]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_name}")

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_cuda_availability():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print(f"✅ CUDA available: {gpu_count} GPU(s) - {gpu_name}")
            return True
        else:
            print("⚠️  CUDA not available - CPU processing only")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed - cannot check CUDA")
        return False

# Setup configuration
setup(
    name="kalenjin-dictionary-processor",
    version=__version__,
    description="AI-powered framework for processing Kalenjin dictionary PDFs using vision-language models with vLLM integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kalenjin Dictionary Project",
    author_email="contact@kalenjin-dictionary.org",
    url="https://github.com/kalenjin-dictionary/processor",
    
    # Package configuration
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    # Entry points
    entry_points={
        "console_scripts": [
            "kalenjin-process=main:main",
            "start-vllm-server=start_vllm_server:main",
            "test-vllm=test_vllm:main",
            "kalenjin-setup=setup:post_install_setup",
        ],
    },
    
    # Package data
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json", "*.yaml", "*.yml", "*.cfg", "*.ini"],
        "llm": ["*.json", "*.yaml", "*.yml"],
        "vllm_server": ["*.json", "*.yaml", "*.yml"],
        "scripts": ["*.py"],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research", 
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent", 
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
    ],
    
    # Keywords
    keywords=[
        "kalenjin", "dictionary", "nlp", "vision-language-model", 
        "pdf-processing", "linguistics", "ai", "vllm", "cuda",
    ],
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/kalenjin-dictionary/processor/issues",
        "Source": "https://github.com/kalenjin-dictionary/processor",
        "Documentation": "https://github.com/kalenjin-dictionary/processor/blob/main/README.md",
        "vLLM Guide": "https://github.com/kalenjin-dictionary/processor/blob/main/VLLM_GUIDE.md",
    },
    
    # Additional metadata
    license="MIT",
    platforms=["any"],
    zip_safe=False,
)

if __name__ == "__main__":
    post_install_setup()
