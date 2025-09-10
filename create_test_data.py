#!/usr/bin/env python3
"""
Create sample test data for the QC Cancellations app including bucket screenshots
"""

import zipfile
import os
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_bucket_screenshot():
    """Create a sample bucket screenshot image"""
    # Create a white background
    width, height = 800, 600
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 18)
        font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw header
    draw.text((50, 50), "PCMI NCB Fee Buckets", fill='black', font=font_large)
    draw.text((50, 90), "Contract: GAP12345678", fill='black', font=font_medium)
    draw.text((50, 120), "VIN: 1HGBH41JXMN109186", fill='black', font=font_medium)
    
    # Draw table headers
    y_start = 180
    draw.text((100, y_start), "Fee Type", fill='black', font=font_medium)
    draw.text((300, y_start), "Amount", fill='black', font=font_medium)
    draw.text((500, y_start), "Status", fill='black', font=font_medium)
    
    # Draw table rows
    rows = [
        ("Agent NCB", "$0.00", "No Chargeback"),
        ("Dealer NCB", "$0.00", "No Chargeback"),
        ("Total Fees", "$0.00", "Complete"),
        ("Processing Fee", "$25.00", "Applied"),
        ("Refund Amount", "$1,250.00", "Pending")
    ]
    
    for i, (fee_type, amount, status) in enumerate(rows):
        y = y_start + 40 + (i * 30)
        draw.text((100, y), fee_type, fill='black', font=font_small)
        draw.text((300, y), amount, fill='black', font=font_small)
        draw.text((500, y), status, fill='black', font=font_small)
    
    # Add some additional text
    draw.text((50, 400), "Mileage: 45,000", fill='black', font=font_medium)
    draw.text((50, 430), "Cancellation Date: 12/15/2023", fill='black', font=font_medium)
    draw.text((50, 460), "Reason: Vehicle Traded", fill='black', font=font_medium)
    
    return img

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
    
    return [
        ("cancellation_form.txt", doc1_content),
        ("lender_letter.txt", doc2_content),
        ("pcmi_screenshot.txt", doc3_content)
    ]

def create_test_zip():
    """Create a test ZIP file with sample documents and bucket screenshot"""
    documents = create_sample_documents()
    
    with zipfile.ZipFile("test_cancellation_packet.zip", "w") as zipf:
        # Add text documents
        for filename, content in documents:
            zipf.writestr(filename, content)
        
        # Add bucket screenshot
        bucket_img = create_bucket_screenshot()
        img_buffer = io.BytesIO()
        bucket_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        zipf.writestr("bucket_screenshot.png", img_buffer.getvalue())
    
    print("Created test_cancellation_packet.zip with sample documents and bucket screenshot")
    print("Files included:")
    for filename, _ in documents:
        print(f"  - {filename}")
    print("  - bucket_screenshot.png (PCMI NCB Fee Buckets)")

if __name__ == "__main__":
    create_test_zip()
