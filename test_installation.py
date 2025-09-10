#!/usr/bin/env python3
"""
Test script to verify all dependencies are properly installed
"""

def test_imports():
    """Test that all required packages can be imported"""
    try:
        import streamlit as st
        print("✓ Streamlit imported successfully")
    except ImportError as e:
        print(f"✗ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import pdfplumber
        print("✓ pdfplumber imported successfully")
    except ImportError as e:
        print(f"✗ pdfplumber import failed: {e}")
        return False
    
    try:
        from docx import Document
        print("✓ python-docx imported successfully")
    except ImportError as e:
        print(f"✗ python-docx import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ Pillow imported successfully")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False
    
    try:
        import pytesseract
        print("✓ pytesseract imported successfully")
    except ImportError as e:
        print(f"✗ pytesseract import failed: {e}")
        return False
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    return True

def test_tesseract():
    """Test if Tesseract OCR is available"""
    try:
        import pytesseract
        # Try to get version info
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract OCR available (version: {version})")
        return True
    except Exception as e:
        print(f"✗ Tesseract OCR not available: {e}")
        print("  Install with: brew install tesseract")
        return False

def test_opencv():
    """Test OpenCV functionality"""
    try:
        import cv2
        import numpy as np
        
        # Test basic OpenCV operations
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        print("✓ OpenCV image processing functions working")
        return True
    except Exception as e:
        print(f"✗ OpenCV functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing QC Cancellations App Dependencies")
    print("=" * 50)
    
    imports_ok = test_imports()
    tesseract_ok = test_tesseract()
    opencv_ok = test_opencv()
    
    print("\n" + "=" * 50)
    if imports_ok and tesseract_ok and opencv_ok:
        print("✓ All dependencies are properly installed!")
        print("You can now run: streamlit run app.py")
    else:
        print("✗ Some dependencies are missing.")
        print("Install missing packages with: pip install -r requirements.txt")
        if not tesseract_ok:
            print("Install Tesseract with: brew install tesseract")
        if not opencv_ok:
            print("OpenCV installation may need additional system dependencies")
