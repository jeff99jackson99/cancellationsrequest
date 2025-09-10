#!/usr/bin/env python3
"""
Comprehensive test script to verify extraction accuracy with real-world scenarios
"""

import sys
import os
from app import QCProcessor
import tempfile
import zipfile

def create_comprehensive_test_zip():
    """Create a comprehensive test ZIP with various document types and formats"""
    
    test_files = {
        "cancellation_request.pdf": """
        CANCELLATION REQUEST FORM
        
        Contract Number: GAP10137957
        Customer Name: Sarah Johnson
        VIN: 1FTFW1ET5DFC12345
        Cancellation Date: 11/30/2024
        Sale Date: 03/15/2024
        Reason: Customer Request
        Mileage: 67,500 miles
        Total Refund: $2,450.00
        Dealer NCB: No
        No Chargeback: Yes
        
        Additional notes: Customer moving out of state.
        """,
        
        "lender_letter.docx": """
        LENDER CANCELLATION LETTER
        
        Policy #: GAP10137957
        Client: Sarah Johnson
        Vehicle ID: 1FTFW1ET5DFC12345
        Effective Date: 11/30/2024
        Original Sale: 03/15/2024
        Cancellation Reason: Customer Request
        Odometer Reading: 67,500 mi
        Refund Amount: $2,450.00
        NCB Status: No
        Chargeback: Yes
        
        This cancellation is approved.
        """,
        
        "pcmi_screenshot.png": """
        PCMI SYSTEM SCREENSHOT
        
        Contract: GAP10137957
        Customer: Sarah Johnson
        VIN: 1FTFW1ET5DFC12345
        Sale Date: 03/15/2024
        Current Mileage: 67,500
        Total Refund: $2,450.00
        Dealer NCB: No
        No Chargeback: Yes
        
        System generated data.
        """,
        
        "contract_agreement.txt": """
        SERVICE CONTRACT AGREEMENT
        
        Agreement Number: GAP10137957
        Purchaser: Sarah Johnson
        Vehicle Identification: 1FTFW1ET5DFC12345
        Contract Date: 03/15/2024
        Vehicle Sale Date: 03/15/2024
        Current Mileage: 67,500 miles
        Refund Calculation: $2,450.00
        NCB Eligible: No
        Chargeback Required: Yes
        
        Terms and conditions apply.
        """,
        
        "inconsistent_data.txt": """
        PARTIAL CANCELLATION FORM
        
        Contract: GAP10137957
        Name: Sarah Johnson
        VIN: 1FTFW1ET5DFC12345
        Date: 11/30/2024
        Why: Customer Request
        Miles: 67,500
        Amount: $2,450.00
        NCB: No
        Chargeback: Yes
        
        This file has slightly different formatting.
        """
    }
    
    # Create temporary ZIP file
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
        with zipfile.ZipFile(temp_zip.name, 'w') as zip_file:
            for filename, content in test_files.items():
                zip_file.writestr(filename, content)
        
        return temp_zip.name

def test_comprehensive_extraction():
    """Test comprehensive extraction scenarios"""
    print("🧪 Comprehensive Extraction Test")
    print("=" * 60)
    
    # Create test ZIP
    test_zip_path = create_comprehensive_test_zip()
    print(f"✅ Created comprehensive test ZIP: {test_zip_path}")
    
    try:
        # Initialize processor
        processor = QCProcessor()
        
        # Process the test ZIP
        print("\n📁 Processing comprehensive test files...")
        with open(test_zip_path, 'rb') as f:
            all_data, files_processed = processor.process_zip(f)
        
        # Evaluate QC checklist
        print("\n📋 Evaluating QC checklist...")
        qc_results = processor.evaluate_qc_checklist(all_data, files_processed)
        
        # Display results
        print("\n📊 COMPREHENSIVE EXTRACTION RESULTS")
        print("=" * 60)
        
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
                unique_values = list(set(values))
                print(f"  {field}: {unique_values}")
        
        # QC Results
        print("\n📋 QC Checklist Results:")
        print("=" * 60)
        
        for field, result in qc_results.items():
            status_icon = {'PASS': '✅', 'FAIL': '❌', 'INFO': 'ℹ️'}
            print(f"\n{field.replace('_', ' ').title()}:")
            print(f"  Status: {status_icon[result['status']]} {result['status']}")
            print(f"  Value: {result['value']}")
            print(f"  Reason: {result['reason']}")
        
        # Calculate accuracy
        print("\n📊 COMPREHENSIVE ACCURACY ANALYSIS")
        print("=" * 60)
        
        # Expected values
        expected = {
            'vin': ['1FTFW1ET5DFC12345'],
            'contract_number': ['GAP10137957'],
            'customer_name': ['Sarah Johnson'],
            'cancellation_date': ['11/30/2024'],
            'sale_date': ['03/15/2024'],
            'reason': ['Customer Request'],
            'mileage': ['67500'],
            'total_refund': ['2450.00'],
            'dealer_ncb': ['No'],
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
        
        # Test specific scenarios
        print("\n🔍 SCENARIO TESTING")
        print("=" * 60)
        
        # Test 1: VIN consistency across files
        unique_vins = list(set(all_data['vin']))
        if len(unique_vins) == 1:
            print("✅ VIN Consistency: All files have the same VIN")
        else:
            print(f"❌ VIN Consistency: Found {len(unique_vins)} different VINs: {unique_vins}")
        
        # Test 2: Contract number consistency
        unique_contracts = list(set(all_data['contract_number']))
        if len(unique_contracts) == 1:
            print("✅ Contract Consistency: All files have the same contract number")
        else:
            print(f"❌ Contract Consistency: Found {len(unique_contracts)} different contracts: {unique_contracts}")
        
        # Test 3: Customer name consistency
        unique_customers = list(set(all_data['customer_name']))
        if len(unique_customers) == 1:
            print("✅ Customer Consistency: All files have the same customer name")
        else:
            print(f"❌ Customer Consistency: Found {len(unique_customers)} different customers: {unique_customers}")
        
        # Test 4: Mileage consistency
        unique_mileages = list(set(all_data['mileage']))
        if len(unique_mileages) == 1:
            print("✅ Mileage Consistency: All files have the same mileage")
        else:
            print(f"❌ Mileage Consistency: Found {len(unique_mileages)} different mileages: {unique_mileages}")
        
        # Test 5: 90+ days calculation
        if qc_results['ninety_days']['status'] == 'PASS':
            print("✅ 90+ Days Rule: Cancellation is more than 90 days after sale")
        elif qc_results['ninety_days']['status'] == 'FAIL':
            print(f"❌ 90+ Days Rule: {qc_results['ninety_days']['reason']}")
        else:
            print(f"ℹ️ 90+ Days Rule: {qc_results['ninety_days']['reason']}")
        
        if accuracy >= 95:
            print("\n🎉 EXCELLENT! Extraction is working exceptionally well!")
        elif accuracy >= 90:
            print("\n👍 VERY GOOD! Extraction is working very well!")
        elif accuracy >= 80:
            print("\n✅ GOOD! Extraction is working well with minor issues.")
        elif accuracy >= 70:
            print("\n⚠️  FAIR! Extraction needs improvement.")
        else:
            print("\n❌ POOR! Extraction needs significant improvement.")
        
        return accuracy
        
    finally:
        # Clean up
        if os.path.exists(test_zip_path):
            os.unlink(test_zip_path)
            print(f"\n🧹 Cleaned up test file: {test_zip_path}")

if __name__ == "__main__":
    accuracy = test_comprehensive_extraction()
    sys.exit(0 if accuracy >= 90 else 1)
