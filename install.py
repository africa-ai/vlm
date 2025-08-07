"""
Installation and Setup Script for Kalenjin Dictionary Processing Framework
Run this after installing the package to complete setup
"""

import os
import sys
import subprocess
from pathlib import Path

def install_kalenjin_processor():
    """Install the Kalenjin Dictionary Processor"""
    print("🚀 Installing Kalenjin Dictionary Processing Framework\n")
    
    print("📋 Installation Options:")
    print("1. Basic installation (CPU only)")
    print("2. Full installation with vLLM server (GPU recommended)")
    print("3. Development installation")
    
    choice = input("\nSelect installation type (1-3): ").strip()
    
    if choice == "1":
        # Basic installation
        cmd = [sys.executable, "-m", "pip", "install", "."]
        print("📦 Installing basic package...")
        
    elif choice == "2":
        # Full installation with vLLM
        cmd = [sys.executable, "-m", "pip", "install", ".[vllm]"]
        print("📦 Installing with vLLM server support...")
        
    elif choice == "3":
        # Development installation
        cmd = [sys.executable, "-m", "pip", "install", "-e", ".[all]"]
        print("📦 Installing in development mode...")
        
    else:
        print("❌ Invalid choice. Defaulting to basic installation.")
        cmd = [sys.executable, "-m", "pip", "install", "."]
    
    try:
        # Install PyTorch with CUDA first
        print("\n🔥 Installing PyTorch with CUDA support...")
        torch_cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
        subprocess.run(torch_cmd, check=True)
        
        # Install the package
        print("📦 Installing package...")
        subprocess.run(cmd, check=True)
        
        print("✅ Installation completed successfully!")
        
        # Run post-install setup
        print("\n🔧 Running post-installation setup...")
        from setup import post_install_setup
        post_install_setup()
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        sys.exit(1)
    except ImportError:
        print("⚠️  Could not run post-install setup. Run: python setup.py")

def main():
    """Main installation function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Kalenjin Dictionary Processing Framework Installation")
        print("\nUsage:")
        print("  python install.py          # Interactive installation")
        print("  python install.py --basic  # Basic installation")
        print("  python install.py --full   # Full installation with vLLM")
        print("  python install.py --dev    # Development installation")
        return
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--basic":
            cmd = [sys.executable, "-m", "pip", "install", "."]
        elif arg == "--full":
            cmd = [sys.executable, "-m", "pip", "install", ".[vllm]"]
        elif arg == "--dev":
            cmd = [sys.executable, "-m", "pip", "install", "-e", ".[all]"]
        else:
            print(f"❌ Unknown argument: {arg}")
            return
            
        print(f"📦 Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print("✅ Installation completed!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Installation failed: {e}")
    else:
        install_kalenjin_processor()

if __name__ == "__main__":
    main()
