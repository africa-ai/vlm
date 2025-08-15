"""
Installation and Setup Script for Kalenjin Dictionary Processing Framework
Simple OCR + LLM pipeline - no complex vision model dependencies
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_tesseract():
    """Check if Tesseract OCR is installed"""
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Tesseract OCR found: {result.stdout.split()[1] if result.stdout else 'installed'}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_tesseract():
    """Install Tesseract OCR based on the operating system"""
    system = platform.system().lower()
    
    print("📦 Installing Tesseract OCR...")
    
    if system == "linux":
        # Linux (Ubuntu/Debian)
        try:
            subprocess.run(['sudo', 'apt-get', 'update'], check=True)
            subprocess.run(['sudo', 'apt-get', 'install', '-y', 'tesseract-ocr'], check=True)
            print("✅ Tesseract installed via apt-get")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install via apt-get. Please install manually:")
            print("   Ubuntu/Debian: sudo apt-get install tesseract-ocr")
            print("   CentOS/RHEL: sudo yum install tesseract")
            return False
    
    elif system == "darwin":  # macOS
        try:
            subprocess.run(['brew', 'install', 'tesseract'], check=True)
            print("✅ Tesseract installed via Homebrew")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install via Homebrew. Please install manually:")
            print("   macOS: brew install tesseract")
            return False
    
    elif system == "windows":
        print("❌ Windows Tesseract installation requires manual setup:")
        print("   1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   2. Install to C:\\Program Files\\Tesseract-OCR\\")
        print("   3. Add to PATH or set TESSERACT_CMD environment variable")
        print("   4. Restart terminal after installation")
        return False
    
    else:
        print(f"❌ Unsupported OS: {system}")
        print("   Please install Tesseract OCR manually for your system")
        return False

def install_kalenjin_processor():
    """Install the Kalenjin Dictionary Processor"""
    print("🚀 Installing Kalenjin Dictionary Processing Framework")
    print("📋 Simple OCR + LLM Pipeline\n")
    
    # Check Tesseract first
    print("🔍 Checking Tesseract OCR...")
    if not check_tesseract():
        print("❌ Tesseract OCR not found")
        
        install_choice = input("📦 Install Tesseract automatically? (y/n): ").strip().lower()
        if install_choice == 'y':
            if not install_tesseract():
                print("❌ Please install Tesseract manually and rerun this script")
                return False
        else:
            print("❌ Tesseract is required for OCR processing")
            print("📋 Manual installation instructions:")
            system = platform.system().lower()
            if system == "linux":
                print("   Ubuntu/Debian: sudo apt-get install tesseract-ocr")
                print("   CentOS/RHEL: sudo yum install tesseract")
            elif system == "darwin":
                print("   macOS: brew install tesseract")
            elif system == "windows":
                print("   Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
            return False
    
    print("\n📋 Installation Options:")
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
