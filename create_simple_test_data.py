#!/usr/bin/env python3
"""
Create simple test data for the QC Cancellations app
"""

import zipfile
import os
from datetime import datetime, timedelta

def create_sample_documents():
    """Create sample document content for testing"""
    
    # Sample document 1 - Complete cancellation packet
    doc1_content = """
CANCELLATION REQUEST FORM

Contract Number: GAP12345678
VIN: 1HGBH41JXMN109186
Customer: John Doe
Vehicle: 2021 Honda Civic

Cancellation Date: 12/15/2023
Contract Sale Date: 08/15/2023

Reason for Cancellation: Vehicle Traded

Agent NCB Fee: No Chargeback
Dealer NCB Fee: No Chargeback

This is an Autohouse Contract.

Signature: _________________

Please send refund to:
123 Main Street
Anytown, ST 12345
"""
    
    # Sample document 2 - Lender letter
    doc2_content = """
LENDER LETTER

Contract Number: GAP12345678
VIN: 1HGBH41JXMN109186

Payoff Letter addressed to Ascent

Cancellation Effective Date: 12/15/2023

Remit refund to:
456 Oak Avenue
Different City, ST 54321

This is a customer direct cancellation.
"""
    
    # Sample document 3 - PCMI Screenshot description
    doc3_content = """
PCMI SCREENSHOT

NCB Fee Buckets:
- Agent NCB: $0.00
- Dealer NCB: $0.00

Mileage: 45,000

Diversicare Contract
"""
    
    # Sample document 4 - Bucket screenshot text (simulated)
    doc4_content = """
PCMI NCB Fee Buckets
Contract: GAP12345678
VIN: 1HGBH41JXMN109186

Fee Type          Amount    Status
Agent NCB         $0.00     No Chargeback
Dealer NCB        $0.00     No Chargeback
Total Fees        $0.00     Complete
Processing Fee    $25.00    Applied
Refund Amount     $1,250.00 Pending

Mileage: 45,000
Cancellation Date: 12/15/2023
Reason: Vehicle Traded
"""
    
    return [
        ("cancellation_form.txt", doc1_content),
        ("lender_letter.txt", doc2_content),
        ("pcmi_screenshot.txt", doc3_content),
        ("bucket_screenshot.txt", doc4_content)
    ]

def create_test_zip():
    """Create a test ZIP file with sample documents"""
    documents = create_sample_documents()
    
    with zipfile.ZipFile("test_cancellation_packet.zip", "w") as zipf:
        for filename, content in documents:
            zipf.writestr(filename, content)
    
    print("Created test_cancellation_packet.zip with sample documents")
    print("Files included:")
    for filename, _ in documents:
        print(f"  - {filename}")
    print("\nYou can now upload this ZIP file to test the Streamlit app!")

if __name__ == "__main__":
    create_test_zip()
