#!/usr/bin/env python
"""
Tesseract OCR Installation Helper for Windows
"""
import os
import subprocess
import sys
from pathlib import Path

def check_tesseract_installed():
    """Check if Tesseract is already installed"""
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Tesseract is already installed!")
            print(f"Version info: {result.stdout.split()[1]}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    return False

def find_tesseract_executable():
    """Try to find Tesseract executable in common locations"""
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe", 
        r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', '')),
        r"C:\tools\tesseract\tesseract.exe",
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            print(f"✅ Found Tesseract at: {path}")
            return path
    
    return None

def setup_tesseract_path():
    """Setup Tesseract path for pytesseract"""
    tesseract_path = find_tesseract_executable()
    
    if tesseract_path:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        print(f"✅ Configured pytesseract to use: {tesseract_path}")
        return True
    else:
        print("❌ Tesseract executable not found in common locations")
        return False

def test_tesseract():
    """Test Tesseract with a simple operation"""
    try:
        import pytesseract
        from PIL import Image
        import numpy as np
        
        # Create a simple test image with text
        test_image = Image.new('RGB', (200, 50), color='white')
        
        # This is a minimal test - just check if tesseract can run
        result = pytesseract.image_to_string(test_image, config='--psm 6')
        print("✅ Tesseract test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Tesseract test failed: {e}")
        return False

def main():
    print("🔍 Tesseract OCR Installation Check")
    print("=" * 50)
    
    # Check if already available via PATH
    if check_tesseract_installed():
        if test_tesseract():
            print("\n🎉 Tesseract is working correctly!")
            return
    
    print("\n🔍 Searching for Tesseract installation...")
    
    # Try to find and configure Tesseract
    if setup_tesseract_path():
        if test_tesseract():
            print("\n🎉 Tesseract configured and working!")
            return
    
    print("\n❌ Tesseract OCR is not installed or not found.")
    print("\n📥 INSTALLATION INSTRUCTIONS:")
    print("1. Go to: https://github.com/UB-Mannheim/tesseract/wiki")
    print("2. Download: tesseract-ocr-w64-setup-v5.3.3.20231005.exe")
    print("3. Install to: C:\\Program Files\\Tesseract-OCR\\")
    print("4. Add to PATH or restart this script")
    print("\n🔄 Alternative: Use package manager:")
    print("   choco install tesseract    (if you have Chocolatey)")
    print("   scoop install tesseract    (if you have Scoop)")

if __name__ == "__main__":
    main()
