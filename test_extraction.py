#!/usr/bin/env python3
"""
Test script to verify extraction accuracy
"""

import sys
import os
from app import SimpleTextProcessor
import tempfile
import zipfile

def create_test_zip():
    """Create a test ZIP file with sample cancellation documents"""
    
    # Sample text content for different file types
    test_files = {
        "contract_1.txt": """
        CONTRACT CANCELLATION REQUEST
        
        Contract Number: ABC123456789
        Customer Name: John Smith
        VIN: 1HGBH41JXMN109186
        Cancellation Date: 12/15/2024
        Sale Date: 08/20/2024
        Reason: Customer Request
        Mileage: 45,000 miles
        Total Refund: $1,250.00
        Dealer NCB: Yes
        No Chargeback: Yes
        
        This is a test cancellation request.
        """,
        
        "contract_2.txt": """
        CANCELLATION FORM
        
        Contract #: ABC123456789
        Name: John Smith
        Vehicle ID: 1HGBH41JXMN109186
        Date: 12/15/2024
        Contract Date: 08/20/2024
        Why: Customer Request
        Odometer: 45,000 mi
        Refund Amount: $1,250.00
        NCB: Yes
        Chargeback: No
        
        Additional information here.
        """,
        
        "lender_letter.txt": """
        LENDER LETTER
        
        Policy Number: ABC123456789
        Client Name: John Smith
        VIN: 1HGBH41JXMN109186
        Cancellation Reason: Customer Request
        Effective Date: 12/15/2024
        Original Sale: 08/20/2024
        Miles: 45,000
        Refund: $1,250.00
        Dealer NCB: Yes
        No Chargeback: Yes
        
        This is the lender letter.
        """,
        
        "screenshot.txt": """
        PCMI SCREENSHOT
        
        Contract: ABC123456789
        Customer: John Smith
        VIN: 1HGBH41JXMN109186
        Sale Date: 08/20/2024
        Mileage: 45,000
        Total Refund: $1,250.00
        NCB Status: Yes
        Chargeback: No
        
        Screenshot data here.
        """
    }
    
    # Create temporary ZIP file
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
        with zipfile.ZipFile(temp_zip.name, 'w') as zip_file:
            for filename, content in test_files.items():
                zip_file.writestr(filename, content)
        
        return temp_zip.name

def test_extraction():
    """Test the extraction accuracy"""
    print("🧪 Testing Extraction Accuracy")
    print("=" * 50)
    
    # Create test ZIP
    test_zip_path = create_test_zip()
    print(f"✅ Created test ZIP: {test_zip_path}")
    
    try:
        # Initialize processor
        processor = SimpleTextProcessor()
        
        # Process the test ZIP
        print("\n📁 Processing test files...")
        with open(test_zip_path, 'rb') as f:
            all_data, files_processed = processor.process_zip(f)
        
        # Evaluate QC checklist
        print("\n📋 Evaluating QC checklist...")
        qc_results = processor.evaluate_qc_checklist(all_data, files_processed)
        
        # Display results
        print("\n📊 EXTRACTION RESULTS")
        print("=" * 50)
        
        # Data extraction summary
        print("\n📈 Data Extraction Summary:")
        print(f"  VINs Found: {len(all_data['vin'])}")
        print(f"  Contracts Found: {len(all_data['contract_number'])}")
        print(f"  Customer Names: {len(all_data['customer_name'])}")
        print(f"  Cancellation Dates: {len(all_data['cancellation_date'])}")
        print(f"  Sale Dates: {len(all_data['sale_date'])}")
        print(f"  Reasons: {len(all_data['reason'])}")
        print(f"  Mileages: {len(all_data['mileage'])}")
        print(f"  Total Refunds: {len(all_data['total_refund'])}")
        print(f"  Dealer NCB: {len(all_data['dealer_ncb'])}")
        print(f"  No Chargeback: {len(all_data['no_chargeback'])}")
        
        # Show all extracted values
        print("\n🔍 All Extracted Values:")
        for field, values in all_data.items():
            if values:
                print(f"  {field}: {values}")
        
        # QC Results
        print("\n📋 QC Checklist Results:")
        print("=" * 50)
        
        for field, result in qc_results.items():
            status_icon = {'PASS': '✅', 'FAIL': '❌', 'INFO': 'ℹ️'}
            print(f"\n{field.replace('_', ' ').title()}:")
            print(f"  Status: {status_icon[result['status']]} {result['status']}")
            print(f"  Value: {result['value']}")
            print(f"  Reason: {result['reason']}")
        
        # Calculate accuracy
        print("\n📊 ACCURACY ANALYSIS")
        print("=" * 50)
        
        # Expected values
        expected = {
            'vin': ['1HGBH41JXMN109186'],
            'contract_number': ['ABC123456789'],
            'customer_name': ['John Smith'],
            'cancellation_date': ['12/15/2024'],
            'sale_date': ['08/20/2024'],
            'reason': ['Customer Request'],
            'mileage': ['45000'],
            'total_refund': ['1250.00'],
            'dealer_ncb': ['Yes'],
            'no_chargeback': ['Yes']
        }
        
        total_fields = len(expected)
        correct_fields = 0
        
        print("\nField-by-Field Accuracy:")
        for field, expected_values in expected.items():
            if field in all_data:
                extracted_values = all_data[field]
                unique_extracted = list(set(extracted_values))
                
                # Check if we found the expected values
                found_expected = any(exp_val in unique_extracted for exp_val in expected_values)
                
                if found_expected:
                    correct_fields += 1
                    print(f"  ✅ {field}: Found expected values")
                else:
                    print(f"  ❌ {field}: Expected {expected_values}, got {unique_extracted}")
            else:
                print(f"  ❌ {field}: Not found in extracted data")
        
        accuracy = (correct_fields / total_fields) * 100
        print(f"\n🎯 Overall Accuracy: {accuracy:.1f}% ({correct_fields}/{total_fields} fields)")
        
        if accuracy >= 90:
            print("🎉 EXCELLENT! Extraction is working very well!")
        elif accuracy >= 80:
            print("👍 GOOD! Extraction is working well with minor issues.")
        elif accuracy >= 70:
            print("⚠️  FAIR! Extraction needs improvement.")
        else:
            print("❌ POOR! Extraction needs significant improvement.")
        
        return accuracy
        
    finally:
        # Clean up
        if os.path.exists(test_zip_path):
            os.unlink(test_zip_path)
            print(f"\n🧹 Cleaned up test file: {test_zip_path}")

if __name__ == "__main__":
    accuracy = test_extraction()
    sys.exit(0 if accuracy >= 80 else 1)