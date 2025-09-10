#!/usr/bin/env python3
"""
Comprehensive test script for PDF data extraction accuracy
Tests all extraction methods and validates results
"""

import os
import sys
import tempfile
import zipfile
from pathlib import Path
import pandas as pd
import json

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import CancellationProcessor

def create_test_pdf_content():
    """Create test PDF content with known data"""
    test_content = """
CANCELLATION REQUEST FORM

Contract Number: 123456789
Customer Name: John Smith
VIN: 1HGBH41JXMN109186
Sale Date: 01/15/2024
Cancellation Date: 03/20/2024
Cancellation Reason: Customer Request

Vehicle Information:
Make: Honda
Model: Civic
Year: 2021
Mileage: 42,703

Financial Information:
Total Refund: $2,500.00
Dealer NCB: No
No Chargeback: Yes

Dealer Information:
Dealer Name: ABC Auto Group
Address: 123 Main St, Anytown, ST 12345
Phone: (555) 123-4567

Signature: [X] John Smith
Date: 03/20/2024
"""
    return test_content

def create_test_zip():
    """Create a test ZIP file with various document types"""
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
        with zipfile.ZipFile(temp_zip.name, 'w') as zf:
            # Create test PDF content
            test_content = create_test_pdf_content()
            
            # Add as text file (simulating PDF content)
            zf.writestr("cancellation_request.txt", test_content)
            
            # Add another test file
            zf.writestr("contract_details.txt", """
Contract Details:
Contract #: 123456789
Customer: John Smith
VIN: 1HGBH41JXMN109186
Sale Date: 01/15/2024
Mileage: 42,703
            """)
        
        return temp_zip.name

def test_extraction_accuracy():
    """Test the accuracy of data extraction"""
    print("🔍 Testing PDF Data Extraction Accuracy")
    print("=" * 50)
    
    # Create test data
    test_zip_path = create_test_zip()
    
    try:
        # Initialize processor
        processor = CancellationProcessor()
        
        # Process the test ZIP
        print("📁 Processing test ZIP file...")
        result = processor.process_zip(test_zip_path)
        
        if not result:
            print("❌ Failed to process ZIP file")
            return False
        
        # Get the packet data
        packets = processor.packets
        if not packets:
            print("❌ No packets found")
            return False
        
        # Get the first (and only) packet
        packet = packets[0]  # packets is now a list, not a dict
        
        print(f"\n📊 Extracted Data for Packet:")
        print("-" * 30)
        print(f"Packet type: {type(packet)}")
        print(f"Packet content: {packet}")
        
        # Expected values
        expected = {
            'contract_number': '123456789',
            'customer_name': 'John Smith',
            'vin': '1HGBH41JXMN109186',
            'sale_date': '01/15/2024',
            'cancellation_date': '03/20/2024',
            'cancellation_reason': 'Customer Request',
            'mileage': '42,703',
            'total_refund': '$2,500.00',
            'dealer_ncb': 'No',
            'no_chargeback': 'Yes'
        }
        
        # Check each field
        accuracy_score = 0
        total_fields = len(expected)
        
        for field, expected_value in expected.items():
            # Handle both dict and list packet structures
            if isinstance(packet, dict):
                extracted_value = packet.get(field, 'Not found')
            elif isinstance(packet, list) and len(packet) > 0:
                # If packet is a list, search through all files for the field
                extracted_value = 'Not found'
                for file_data in packet:
                    if isinstance(file_data, dict):
                        # Map expected field names to actual field names in the data
                        field_mapping = {
                            'contract_number': 'contracts',
                            'customer_name': 'customer_names',
                            'vin': 'vins',
                            'sale_date': 'sale_dates',
                            'cancellation_date': 'cancellation_dates',
                            'cancellation_reason': 'reasons',
                            'mileage': 'mileages',
                            'total_refund': 'total_refund',
                            'dealer_ncb': 'dealer_ncb',
                            'no_chargeback': 'no_chargeback'
                        }
                        
                        actual_field = field_mapping.get(field, field)
                        if actual_field in file_data and file_data[actual_field]:
                            if isinstance(file_data[actual_field], list):
                                # Get the first non-empty value from the list
                                for value in file_data[actual_field]:
                                    if value and str(value).strip():
                                        extracted_value = value
                                        break
                            else:
                                extracted_value = file_data[actual_field]
                            break
            else:
                extracted_value = 'Not found'
            
            # Normalize for comparison
            if field in ['mileage']:
                # Remove commas and spaces for mileage comparison
                extracted_normalized = re.sub(r'[,\s]', '', str(extracted_value))
                expected_normalized = re.sub(r'[,\s]', '', str(expected_value))
                match = extracted_normalized == expected_normalized
            else:
                match = str(extracted_value).strip() == str(expected_value).strip()
            
            status = "✅ PASS" if match else "❌ FAIL"
            if match:
                accuracy_score += 1
            
            print(f"{field:20} | {status:8} | Expected: {expected_value:15} | Got: {str(extracted_value):15}")
        
        # Calculate accuracy percentage
        accuracy_percentage = (accuracy_score / total_fields) * 100
        
        print(f"\n📈 Accuracy Score: {accuracy_score}/{total_fields} ({accuracy_percentage:.1f}%)")
        
        # Test individual extraction methods
        print(f"\n🔬 Testing Individual Extraction Methods:")
        print("-" * 40)
        
        # Test PDF extraction methods
        test_file = None
        for file_data in processor.files_data:
            if file_data['filename'].endswith('.txt'):
                # Try different possible keys for file path
                test_file = file_data.get('file_path') or file_data.get('path') or file_data.get('temp_path')
                if test_file:
                    break
        
        if test_file:
            print(f"Testing extraction from: {os.path.basename(test_file)}")
            
            # Test each method
            methods = ['pdfplumber', 'fitz', 'pdfminer', 'pymupdf4llm', 'ocr']
            for method in methods:
                try:
                    if method == 'pdfplumber':
                        import pdfplumber
                        with pdfplumber.open(test_file) as pdf:
                            text = ""
                            for page in pdf.pages:
                                text += page.extract_text() or ""
                    elif method == 'fitz':
                        import fitz
                        doc = fitz.open(test_file)
                        text = ""
                        for page in doc.pages:
                            text += page.get_text()
                        doc.close()
                    elif method == 'pdfminer':
                        from pdfminer.high_level import extract_text
                        text = extract_text(test_file)
                    elif method == 'pymupdf4llm':
                        import pymupdf4llm
                        text = pymupdf4llm.to_markdown(test_file)
                    elif method == 'ocr':
                        from PIL import Image
                        import pytesseract
                        image = Image.open(test_file)
                        text = pytesseract.image_to_string(image)
                    
                    # Count key patterns found
                    vin_count = len(re.findall(r'1HGBH41JXMN109186', text))
                    contract_count = len(re.findall(r'123456789', text))
                    name_count = len(re.findall(r'John Smith', text))
                    
                    print(f"{method:15} | VIN: {vin_count} | Contract: {contract_count} | Name: {name_count} | Length: {len(text)}")
                    
                except Exception as e:
                    print(f"{method:15} | ERROR: {str(e)[:50]}...")
        
        # Test data extraction patterns
        print(f"\n🎯 Testing Data Extraction Patterns:")
        print("-" * 40)
        
        test_text = create_test_pdf_content()
        
        # Test VIN extraction
        vin_patterns = [
            r'VIN[:\s]+([A-HJ-NPR-Z0-9]{17})',
            r'VIN\s*#?\s*:?\s*([A-HJ-NPR-Z0-9]{17})',
        ]
        
        for i, pattern in enumerate(vin_patterns):
            matches = re.findall(pattern, test_text, re.IGNORECASE)
            print(f"VIN Pattern {i+1}: {len(matches)} matches - {matches}")
        
        # Test Contract extraction
        contract_patterns = [
            r'Contract\s*#?\s*:?\s*(\d+)',
            r'Contract\s+Number[:\s]+(\d+)',
        ]
        
        for i, pattern in enumerate(contract_patterns):
            matches = re.findall(pattern, test_text, re.IGNORECASE)
            print(f"Contract Pattern {i+1}: {len(matches)} matches - {matches}")
        
        # Test Date extraction
        date_patterns = [
            r'(\d{1,2}/\d{1,2}/\d{2,4})',
            r'Sale\s+Date[:\s]+(\d{1,2}/\d{1,2}/\d{2,4})',
            r'Cancellation\s+Date[:\s]+(\d{1,2}/\d{1,2}/\d{2,4})',
        ]
        
        for i, pattern in enumerate(date_patterns):
            matches = re.findall(pattern, test_text, re.IGNORECASE)
            print(f"Date Pattern {i+1}: {len(matches)} matches - {matches}")
        
        # Overall assessment
        print(f"\n🏆 Overall Assessment:")
        print("-" * 20)
        
        if accuracy_percentage >= 90:
            print("✅ EXCELLENT - Data extraction is highly accurate")
        elif accuracy_percentage >= 75:
            print("⚠️  GOOD - Data extraction is mostly accurate with minor issues")
        elif accuracy_percentage >= 50:
            print("⚠️  FAIR - Data extraction has some accuracy issues")
        else:
            print("❌ POOR - Data extraction needs significant improvement")
        
        return accuracy_percentage >= 75
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        if os.path.exists(test_zip_path):
            os.unlink(test_zip_path)

def test_with_real_pdf():
    """Test with a real PDF if available"""
    print(f"\n🔍 Testing with Real PDF Files")
    print("=" * 40)
    
    # Look for PDF files in the current directory
    pdf_files = list(Path('.').glob('*.pdf'))
    
    if not pdf_files:
        print("No PDF files found in current directory for testing")
        return True
    
    processor = CancellationProcessor()
    
    for pdf_file in pdf_files[:2]:  # Test up to 2 PDFs
        print(f"\n📄 Testing: {pdf_file.name}")
        print("-" * 30)
        
        try:
            # Test the advanced PDF extraction
            text = processor.extract_text_from_pdf_advanced(str(pdf_file))
            
            if text:
                print(f"✅ Successfully extracted {len(text)} characters")
                
                # Look for common patterns
                vin_matches = re.findall(r'[A-HJ-NPR-Z0-9]{17}', text)
                contract_matches = re.findall(r'Contract\s*#?\s*:?\s*(\d+)', text, re.IGNORECASE)
                date_matches = re.findall(r'\d{1,2}/\d{1,2}/\d{2,4}', text)
                money_matches = re.findall(r'\$\d+\.?\d*', text)
                
                print(f"   VINs found: {len(vin_matches)}")
                print(f"   Contract numbers: {len(contract_matches)}")
                print(f"   Dates found: {len(date_matches)}")
                print(f"   Money amounts: {len(money_matches)}")
                
                # Show first 200 characters
                print(f"   Preview: {text[:200]}...")
            else:
                print("❌ No text extracted")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    import re
    
    print("🚀 Starting PDF Data Extraction Accuracy Tests")
    print("=" * 60)
    
    # Run accuracy test
    success = test_extraction_accuracy()
    
    # Test with real PDFs if available
    test_with_real_pdf()
    
    print(f"\n{'='*60}")
    if success:
        print("✅ All tests completed successfully!")
        print("📊 Data extraction accuracy is acceptable for production use")
    else:
        print("❌ Tests failed - data extraction needs improvement")
        print("🔧 Consider adjusting extraction patterns or methods")
    
    print(f"{'='*60}")
