"""
Installation and Setup Script for Kalenjin Dictionary Processing Framework
Simple OCR + LLM pipeline - no complex vision model dependencies
"""

import os
import sys
import subprocess
from pathlib import Path

def install_kalenjin_processor():
    """Install the Kalenjin Dictionary Processor"""
    print("🚀 Installing Kalenjin Dictionary Processing Framework")
    print("📋 Simple OCR + LLM Pipeline\n")
    
    print("📋 Installation Options:")
    print("1. Basic installation (OCR + vLLM client)")
    print("2. Development installation")
    
    choice = input("\nSelect installation type (1-2): ").strip()
    
    if choice == "1":
        # Basic installation
        cmd = [sys.executable, "-m", "pip", "install", "."]
        print("📦 Installing basic OCR + LLM package...")
        
    elif choice == "2":
        # Development installation
        cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
        print("📦 Installing in development mode...")
        
    else:
        print("❌ Invalid choice. Defaulting to basic installation.")
        cmd = [sys.executable, "-m", "pip", "install", "."]
    
    try:
        # Install the package
        print("📦 Installing package...")
        subprocess.run(cmd, check=True)
        
        print("✅ Installation completed successfully!")
        print("\n📋 Next steps:")
        print("1. Start the vLLM server: python start_vllm_server.py")
        print("2. Process a PDF: python main.py pipeline your_dictionary.pdf")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        sys.exit(1)

def main():
    """Main installation function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Kalenjin Dictionary Processing Framework Installation")
        print("Simple OCR + LLM Pipeline")
        print("\nUsage:")
        print("  python install.py          # Interactive installation")
        print("  python install.py --basic  # Basic installation")
        print("  python install.py --dev    # Development installation")
        return
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--basic":
            cmd = [sys.executable, "-m", "pip", "install", "."]
        elif arg == "--dev":
            cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
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
