#!/usr/bin/env python3
"""
Comparison test between old extraction method and new simple text method
"""

import sys
import os
from app import QCProcessor  # Old method
from app_simple_text import SimpleTextProcessor  # New method
import tempfile
import zipfile

def create_test_zip():
    """Create a test ZIP file with sample cancellation documents"""
    
    test_files = {
        "contract_1.pdf": """
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
        
        "contract_2.pdf": """
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
        
        "lender_letter.docx": """
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
        
        "screenshot.png": """
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

def test_comparison():
    """Compare old method vs new simple text method"""
    print("🔬 EXTRACTION METHOD COMPARISON")
    print("=" * 80)
    
    # Create test ZIP
    test_zip_path = create_test_zip()
    print(f"✅ Created test ZIP: {test_zip_path}")
    
    try:
        # Test old method
        print("\n📊 OLD METHOD (Complex Patterns)")
        print("-" * 40)
        old_processor = QCProcessor()
        
        with open(test_zip_path, 'rb') as f:
            old_all_data, old_files_processed = old_processor.process_zip(f)
        
        old_qc_results = old_processor.evaluate_qc_checklist(old_all_data, old_files_processed)
        
        # Test new method
        print("\n📊 NEW METHOD (Simple Text Conversion)")
        print("-" * 40)
        new_processor = SimpleTextProcessor()
        
        with open(test_zip_path, 'rb') as f:
            new_all_data, new_files_processed = new_processor.process_zip(f)
        
        new_qc_results = new_processor.evaluate_qc_checklist(new_all_data, new_files_processed)
        
        # Comparison table
        print("\n📋 COMPARISON RESULTS")
        print("=" * 80)
        
        fields_to_compare = [
            'vin', 'contract_number', 'customer_name', 'cancellation_date', 
            'sale_date', 'reason', 'mileage', 'total_refund', 'dealer_ncb', 'no_chargeback'
        ]
        
        print(f"{'Field':<20} {'Old Method':<30} {'New Method':<30}")
        print("-" * 80)
        
        for field in fields_to_compare:
            old_values = old_all_data.get(field, [])
            new_values = new_all_data.get(field, [])
            
            old_unique = list(set(old_values))
            new_unique = list(set(new_values))
            
            old_str = f"{len(old_values)} found: {old_unique[:2]}{'...' if len(old_unique) > 2 else ''}"
            new_str = f"{len(new_values)} found: {new_unique[:2]}{'...' if len(new_unique) > 2 else ''}"
            
            print(f"{field:<20} {old_str:<30} {new_str:<30}")
        
        # QC Status Comparison
        print("\n📋 QC STATUS COMPARISON")
        print("=" * 80)
        
        qc_fields = ['contract_number', 'customer_name', 'vin_match', 'mileage_match', 'ninety_days']
        
        print(f"{'Field':<20} {'Old Status':<15} {'New Status':<15} {'Improvement':<15}")
        print("-" * 80)
        
        for field in qc_fields:
            old_status = old_qc_results.get(field, {}).get('status', 'N/A')
            new_status = new_qc_results.get(field, {}).get('status', 'N/A')
            
            if old_status == new_status:
                improvement = "Same"
            elif new_status == 'PASS' and old_status != 'PASS':
                improvement = "✅ Better"
            elif new_status == 'FAIL' and old_status == 'PASS':
                improvement = "❌ Worse"
            else:
                improvement = "Different"
            
            print(f"{field:<20} {old_status:<15} {new_status:<15} {improvement:<15}")
        
        # Accuracy calculation
        print("\n📊 ACCURACY COMPARISON")
        print("=" * 80)
        
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
        
        def calculate_accuracy(data, expected):
            correct = 0
            total = len(expected)
            
            for field, expected_values in expected.items():
                if field in data:
                    extracted_values = data[field]
                    unique_extracted = list(set(extracted_values))
                    found_expected = any(exp_val in unique_extracted for exp_val in expected_values)
                    if found_expected:
                        correct += 1
            
            return (correct / total) * 100
        
        old_accuracy = calculate_accuracy(old_all_data, expected)
        new_accuracy = calculate_accuracy(new_all_data, expected)
        
        print(f"Old Method Accuracy:  {old_accuracy:.1f}%")
        print(f"New Method Accuracy:  {new_accuracy:.1f}%")
        print(f"Improvement:          {new_accuracy - old_accuracy:+.1f}%")
        
        # Summary
        print("\n📋 SUMMARY")
        print("=" * 80)
        
        if new_accuracy > old_accuracy:
            print("✅ NEW METHOD IS BETTER!")
            print(f"   - {new_accuracy - old_accuracy:.1f}% more accurate")
            print("   - Simpler text-based extraction")
            print("   - More reliable pattern matching")
        elif new_accuracy == old_accuracy:
            print("🤔 BOTH METHODS ARE EQUAL")
            print("   - Same accuracy level")
            print("   - New method is simpler")
        else:
            print("❌ OLD METHOD IS BETTER")
            print(f"   - {old_accuracy - new_accuracy:.1f}% more accurate")
            print("   - Complex patterns work better")
        
        return new_accuracy > old_accuracy
        
    finally:
        # Clean up
        if os.path.exists(test_zip_path):
            os.unlink(test_zip_path)
            print(f"\n🧹 Cleaned up test file: {test_zip_path}")

if __name__ == "__main__":
    success = test_comparison()
    sys.exit(0 if success else 1)
