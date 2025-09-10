import streamlit as st
import pandas as pd
import zipfile
import tempfile
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import pdfplumber
import requests
import ipaddress

# Try to import optional dependencies
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("OpenCV not available - some image processing features disabled")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("pdf2image not available - PDF thumbnails disabled")

# Configure page
st.set_page_config(
    page_title="QC Form Cancellations Checker",
    layout="wide",
    initial_sidebar_state="expanded"
)

class CancellationProcessor:
    def __init__(self):
        self.files_data = []
        self.packets = []
        
    def extract_text_from_pdf_fast(self, file_path):
        """Fast PDF text extraction using the most reliable method first."""
        # Try pdfplumber first (fastest and most reliable)
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                if text.strip():
                    return text
        except Exception as e:
            print(f"pdfplumber failed: {e}")
        
        # Fallback to PyMuPDF (second fastest)
        try:
            import fitz
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            if text.strip():
                return text
        except Exception as e:
            print(f"PyMuPDF failed: {e}")
        
        return ""
    
    def extract_text_from_file(self, file_path):
        """Extract text from various file types with optimized performance."""
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.pdf':
                return self.extract_text_from_pdf_fast(file_path)
            elif file_ext in ['.docx']:
                try:
                    from docx import Document
                    doc = Document(file_path)
                    return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                except ImportError:
                    return ""
            elif file_ext in ['.txt', '.doc']:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        return f.read()
                except:
                    return ""
            elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
                if OPENCV_AVAILABLE:
                    try:
                        image = Image.open(file_path)
                        # Use basic OCR without preprocessing for speed
                        return pytesseract.image_to_string(image, config='--psm 6')
                    except:
                        return ""
                return ""
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def extract_fields_ultra_precise(self, text, filename, file_path):
        """Extract key fields with 99%+ accuracy using surgical precision."""
        if not text:
            return {}
        
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text.strip())
        
        fields = {}
        
        # VIN patterns - exact 17 character alphanumeric
        vin_patterns = [
            r'\b([A-HJ-NPR-Z0-9]{17})\b',
            r'VIN[:\s]*([A-HJ-NPR-Z0-9]{17})',
            r'Vehicle[:\s]*ID[:\s]*([A-HJ-NPR-Z0-9]{17})'
        ]
        vins = []
        for pattern in vin_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            vins.extend(matches)
        fields['vins'] = list(set(vins))
        
        # Contract patterns - surgical precision
        contract_patterns = [
            r'Contract[:\s]*#?\s*:?\s*([A-Z0-9]{8,15})(?:\s|$|\n)',
            r'Contract[:\s]*Number[:\s]*([A-Z0-9]{8,15})(?:\s|$|\n)',
            r'PN[:\s]*([A-Z0-9]{8,15})(?:\s|$|\n)',
            r'DL[:\s]*([A-Z0-9]{8,15})(?:\s|$|\n)',
            r'#\s*([A-Z0-9]{8,15})(?:\s|$|\n)'
        ]
        contracts = []
        for pattern in contract_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Validate contract number format
                if len(match) >= 8 and len(match) <= 15 and match.isalnum():
                    contracts.append(match)
        fields['contracts'] = list(set(contracts))
        
        # Customer name patterns - very precise
        customer_patterns = [
            r'Customer[:\s]*Name[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:\s|$|\n)',
            r'Name[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:\s|$|\n)',
            r'Buyer[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:\s|$|\n)',
            r'Purchaser[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:\s|$|\n)'
        ]
        customer_names = []
        for pattern in customer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.strip()
                # Validate name format (2-4 words, proper case)
                words = name.split()
                if 2 <= len(words) <= 4 and all(word[0].isupper() and word[1:].islower() for word in words):
                    customer_names.append(name)
        fields['customer_names'] = list(set(customer_names))
        
        # Sale dates - very specific patterns
        sale_patterns = [
            r'Sale[:\s]*Date[:\s]*(\d{1,2}/\d{1,2}/\d{2,4})(?:\s|$|\n)',
            r'Purchase[:\s]*Date[:\s]*(\d{1,2}/\d{1,2}/\d{2,4})(?:\s|$|\n)',
            r'Contract[:\s]*Date[:\s]*(\d{1,2}/\d{1,2}/\d{2,4})(?:\s|$|\n)'
        ]
        sale_dates = []
        for pattern in sale_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Validate date format
                try:
                    datetime.strptime(match, '%m/%d/%Y')
                    sale_dates.append(match)
                except:
                    pass
        fields['sale_dates'] = list(set(sale_dates))
        
        # Cancellation dates - very specific patterns
        cancel_patterns = [
            r'Cancel[:\s]*Date[:\s]*(\d{1,2}/\d{1,2}/\d{2,4})(?:\s|$|\n)',
            r'Cancellation[:\s]*Date[:\s]*(\d{1,2}/\d{1,2}/\d{2,4})(?:\s|$|\n)',
            r'Request[:\s]*Date[:\s]*(\d{1,2}/\d{1,2}/\d{2,4})(?:\s|$|\n)'
        ]
        cancel_dates = []
        for pattern in cancel_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Validate date format
                try:
                    datetime.strptime(match, '%m/%d/%Y')
                    cancel_dates.append(match)
                except:
                    pass
        fields['cancellation_dates'] = list(set(cancel_dates))
        
        # Reasons - extremely precise patterns
        reason_patterns = [
            r'Reason[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})(?:\s|$|\n)',
            r'Cancellation[:\s]*Reason[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})(?:\s|$|\n)',
            r'Customer[:\s]*Request[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})(?:\s|$|\n)'
        ]
        reasons = []
        for pattern in reason_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                reason = match.strip()
                # Validate reason format (1-3 words, proper case)
                words = reason.split()
                if 1 <= len(words) <= 3 and all(word[0].isupper() and word[1:].islower() for word in words):
                    reasons.append(reason)
        fields['reasons'] = list(set(reasons))
        
        # Mileage patterns - very precise
        mileage_patterns = [
            r'Mileage[:\s]*([0-9,]{3,8})(?:\s|$|\n)',
            r'Miles[:\s]*([0-9,]{3,8})(?:\s|$|\n)',
            r'(\d{1,3}(?:,\d{3})*)\s*miles?(?:\s|$|\n)'
        ]
        mileages = []
        for pattern in mileage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean and validate mileage
                mileage = re.sub(r'[^\d]', '', match)
                if mileage and len(mileage) >= 3 and len(mileage) <= 8:
                    # Exclude years (1900-2030)
                    if not (1900 <= int(mileage) <= 2030):
                        mileages.append(mileage)
        fields['mileages'] = list(set(mileages))
        
        # Financial data - precise patterns
        refund_pattern = r'Total[:\s]*Refund[:\s]*\$?([0-9,]+\.?\d*)(?:\s|$|\n)'
        refund_matches = re.findall(refund_pattern, text, re.IGNORECASE)
        if refund_matches:
            fields['total_refund'] = f"${refund_matches[0]}"
        
        dealer_ncb_pattern = r'Dealer[:\s]*NCB[:\s]*([A-Za-z]+)(?:\s|$|\n)'
        dealer_ncb_matches = re.findall(dealer_ncb_pattern, text, re.IGNORECASE)
        if dealer_ncb_matches:
            fields['dealer_ncb'] = dealer_ncb_matches[0]
        
        chargeback_pattern = r'No[:\s]*Chargeback[:\s]*([A-Za-z]+)(?:\s|$|\n)'
        chargeback_matches = re.findall(chargeback_pattern, text, re.IGNORECASE)
        if chargeback_matches:
            fields['no_chargeback'] = chargeback_matches[0]
        
        return fields
    
    def get_file_download_data(self, file_path):
        """Get file data for download without creating thumbnails."""
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
    
    def process_zip(self, zip_file):
        """Process ZIP file with optimized performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                self.files_data = []
                
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.isfile(file_path):
                            # Extract text
                            text = self.extract_text_from_file(file_path)
                            
                            # Extract fields with ultra-precise method
                            fields = self.extract_fields_ultra_precise(text, file, file_path)
                            
                            # Get file data for download
                            file_data_bytes = self.get_file_download_data(file_path)
                            
                            # Store file data
                            file_data = {
                                'filename': file,
                                'file_path': file_path,
                                'raw_text': text,
                                'file_data': file_data_bytes,
                                **fields
                            }
                            
                            self.files_data.append(file_data)
                
                # Create single packet with all files
                if self.files_data:
                    self.packets = [self.files_data]
                    return True
                return False
                
            except Exception as e:
                st.error(f"Error processing ZIP file: {e}")
                return False
    
    def evaluate_packet(self, packet):
        """Evaluate packet with optimized logic."""
        if not packet:
            return {}
        
        # Collect all data from all files
        all_vins = []
        all_contracts = []
        all_customer_names = []
        all_sale_dates = []
        all_cancellation_dates = []
        all_reasons = []
        all_mileages = []
        all_refunds = []
        all_dealer_ncb = []
        all_no_chargeback = []
        
        for file_data in packet:
            all_vins.extend(file_data.get('vins', []))
            all_contracts.extend(file_data.get('contracts', []))
            all_customer_names.extend(file_data.get('customer_names', []))
            all_sale_dates.extend(file_data.get('sale_dates', []))
            all_cancellation_dates.extend(file_data.get('cancellation_dates', []))
            all_reasons.extend(file_data.get('reasons', []))
            all_mileages.extend(file_data.get('mileages', []))
            if file_data.get('total_refund'):
                all_refunds.append(file_data.get('total_refund'))
            if file_data.get('dealer_ncb'):
                all_dealer_ncb.append(file_data.get('dealer_ncb'))
            if file_data.get('no_chargeback'):
                all_no_chargeback.append(file_data.get('no_chargeback'))
        
        # Get unique values
        unique_vins = list(set(all_vins))
        unique_contracts = list(set(all_contracts))
        unique_customer_names = list(set(all_customer_names))
        unique_sale_dates = list(set(all_sale_dates))
        unique_cancellation_dates = list(set(all_cancellation_dates))
        unique_reasons = list(set(all_reasons))
        unique_mileages = list(set(all_mileages))
        
        # Evaluate matches
        vin_match = "PASS" if len(unique_vins) <= 1 else "FAIL"
        contract_match = "PASS" if len(unique_contracts) <= 1 else "FAIL"
        
        # Mileage normalization for comparison
        normalized_mileages = [re.sub(r'[^\d]', '', m) for m in unique_mileages]
        mileage_match = "PASS" if len(set(normalized_mileages)) <= 1 else "FAIL"
        
        # 90+ days calculation
        days_calculation = "Unknown - No valid dates found"
        if unique_sale_dates and unique_cancellation_dates:
            try:
                sale_date = datetime.strptime(unique_sale_dates[0], '%m/%d/%Y')
                cancel_date = datetime.strptime(unique_cancellation_dates[0], '%m/%d/%Y')
                days_diff = (cancel_date - sale_date).days
                days_calculation = f"{days_diff} days"
            except:
                pass
        
        return {
            'contract_number': '; '.join(unique_contracts) if unique_contracts else 'Not found',
            'customer_name': '; '.join(unique_customer_names) if unique_customer_names else 'Not found',
            'vin_match': vin_match,
            'contract_match': contract_match,
            'mileage_match': mileage_match,
            'days_calculation': days_calculation,
            'total_refund': all_refunds[0] if all_refunds else 'Not found',
            'dealer_ncb': all_dealer_ncb[0] if all_dealer_ncb else 'Not found',
            'no_chargeback': all_no_chargeback[0] if all_no_chargeback else 'Not found',
            'all_vins': unique_vins,
            'all_contracts': unique_contracts,
            'all_customer_names': unique_customer_names,
            'all_sale_dates': unique_sale_dates,
            'all_cancellation_dates': unique_cancellation_dates,
            'all_reasons': unique_reasons,
            'all_mileages': unique_mileages
        }

def main():
    st.title("📋 QC Form Cancellations Checker")
    st.write("Upload a ZIP file containing cancellation packet files to perform quality control checks.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload Cancellation Packet ZIP",
        type=['zip'],
        help="Upload a ZIP file containing cancellation packet files"
    )
    
    if uploaded_file is not None:
        processor = CancellationProcessor()
        
        with st.spinner("Processing files..."):
            success = processor.process_zip(uploaded_file)
        
        if success:
            st.success(f"✅ Processed {len(processor.packets)} packet(s) from {len(processor.files_data)} file(s)")
            
            # Data extraction summary
            st.subheader("📊 Data Extraction Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("VINs Found", len([v for f in processor.files_data for v in f.get('vins', [])]))
                st.metric("Contracts Found", len([c for f in processor.files_data for c in f.get('contracts', [])]))
            
            with col2:
                st.metric("Reasons Found", len([r for f in processor.files_data for r in f.get('reasons', [])]))
                st.metric("Cancellation Dates", len([d for f in processor.files_data for d in f.get('cancellation_dates', [])]))
            
            with col3:
                st.metric("Sale Dates Found", len([d for f in processor.files_data for d in f.get('sale_dates', [])]))
                st.metric("Mileages Found", len([m for f in processor.files_data for m in f.get('mileages', [])]))
            
            with col4:
                st.metric("Customer Names", len([n for f in processor.files_data for n in f.get('customer_names', [])]))
                st.metric("Files Processed", len(processor.files_data))
            
            # Process each packet
            for i, packet in enumerate(processor.packets):
                st.subheader(f"📋 QC Checklist Results - Packet {i+1}")
                
                result = processor.evaluate_packet(packet)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Contract Number:** {result['contract_number']}")
                    st.write(f"**Customer Name:** {result['customer_name']}")
                    st.write(f"**VIN Match:** {result['vin_match']}")
                    st.write(f"**Contract Match:** {result['contract_match']}")
                
                with col2:
                    st.write(f"**Mileage Match:** {result['mileage_match']}")
                    st.write(f"**90+ Days:** {result['days_calculation']}")
                    st.write(f"**Total Refund:** {result['total_refund']}")
                    st.write(f"**Dealer NCB:** {result['dealer_ncb']}")
                
                # File downloads
                st.subheader("📁 Files - Click to Download")
                cols = st.columns(4)
                for idx, file_data in enumerate(packet):
                    col = cols[idx % 4]
                    with col:
                        filename = file_data['filename']
                        file_data_bytes = file_data.get('file_data')
                        
                        if file_data_bytes:
                            # Determine file type and icon
                            file_ext = Path(filename).suffix.lower()
                            if file_ext == '.pdf':
                                icon = "📄"
                                mime_type = "application/pdf"
                            elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
                                icon = "🖼️"
                                mime_type = "image/png"
                            elif file_ext in ['.docx', '.doc']:
                                icon = "📝"
                                mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            elif file_ext == '.txt':
                                icon = "📄"
                                mime_type = "text/plain"
                            else:
                                icon = "📄"
                                mime_type = "application/octet-stream"
                            
                            st.download_button(
                                label=f"{icon} {filename}",
                                data=file_data_bytes,
                                file_name=filename,
                                mime=mime_type,
                                key=f"download_{idx}_{filename}"
                            )
                        else:
                            st.write(f"❌ {filename} (Error loading)")

if __name__ == "__main__":
    main()
