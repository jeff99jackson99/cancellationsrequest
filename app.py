import streamlit as st
import pandas as pd
import zipfile
import tempfile
import os
import re
import json
from datetime import datetime, timedelta
from pathlib import Path
import pdfplumber
from docx import Document
from PIL import Image
import pytesseract
import io
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    # Fallback for environments where OpenCV is not available
    import numpy as np

# Page configuration
st.set_page_config(
    page_title="QC Form Cancellations Checker",
    page_icon="📋",
    layout="wide"
)

# Title and description
st.title("📋 QC Form Cancellations Checker")
st.markdown("Upload a ZIP file containing cancellation packet files to perform quality control checks. You can also upload individual bucket screenshots for NCB fee analysis.")

# Check for OpenCV availability
if not OPENCV_AVAILABLE:
    st.warning("⚠️ OpenCV is not available. Screenshot processing will use basic OCR without image preprocessing.")

class ScreenshotProcessor:
    """Enhanced screenshot processing for bucket files"""
    
    def __init__(self):
        self.ncb_patterns = {
            'agent_ncb': [
                r'agent\s*ncb[:\s]*\$?([0-9,]+\.?[0-9]*)',
                r'agent\s*no\s*chargeback[:\s]*\$?([0-9,]+\.?[0-9]*)',
                r'agent\s*fee[:\s]*\$?([0-9,]+\.?[0-9]*)',
                r'agent[:\s]*\$?([0-9,]+\.?[0-9]*)'
            ],
            'dealer_ncb': [
                r'dealer\s*ncb[:\s]*\$?([0-9,]+\.?[0-9]*)',
                r'dealer\s*no\s*chargeback[:\s]*\$?([0-9,]+\.?[0-9]*)',
                r'dealer\s*fee[:\s]*\$?([0-9,]+\.?[0-9]*)',
                r'dealer[:\s]*\$?([0-9,]+\.?[0-9]*)'
            ],
            'total_fees': [
                r'total[:\s]*\$?([0-9,]+\.?[0-9]*)',
                r'amount[:\s]*\$?([0-9,]+\.?[0-9]*)',
                r'fee[:\s]*\$?([0-9,]+\.?[0-9]*)'
            ]
        }
    
    def preprocess_image(self, image):
        """Preprocess image for better OCR"""
        if not OPENCV_AVAILABLE:
            # Fallback: return original image if OpenCV is not available
            return image
            
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to PIL
        return Image.fromarray(thresh)
    
    def extract_ncb_data(self, image_path):
        """Extract NCB fee data from bucket screenshot"""
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            processed_image = self.preprocess_image(image)
            
            # Extract text using OCR
            text = pytesseract.image_to_string(processed_image, config='--psm 6')
            
            # Also try with different OCR settings
            text_alt = pytesseract.image_to_string(image, config='--psm 3')
            
            # Combine both results
            combined_text = text + "\n" + text_alt
            
            # Extract NCB data
            ncb_data = {
                'agent_ncb_amount': None,
                'dealer_ncb_amount': None,
                'total_amount': None,
                'has_agent_ncb': False,
                'has_dealer_ncb': False,
                'raw_text': combined_text
            }
            
            # Extract Agent NCB
            for pattern in self.ncb_patterns['agent_ncb']:
                matches = re.findall(pattern, combined_text, re.IGNORECASE)
                if matches:
                    try:
                        amount = float(matches[0].replace(',', ''))
                        ncb_data['agent_ncb_amount'] = amount
                        ncb_data['has_agent_ncb'] = amount > 0
                        break
                    except ValueError:
                        continue
            
            # Extract Dealer NCB
            for pattern in self.ncb_patterns['dealer_ncb']:
                matches = re.findall(pattern, combined_text, re.IGNORECASE)
                if matches:
                    try:
                        amount = float(matches[0].replace(',', ''))
                        ncb_data['dealer_ncb_amount'] = amount
                        ncb_data['has_dealer_ncb'] = amount > 0
                        break
                    except ValueError:
                        continue
            
            # Extract total amount
            for pattern in self.ncb_patterns['total_fees']:
                matches = re.findall(pattern, combined_text, re.IGNORECASE)
                if matches:
                    try:
                        amount = float(matches[0].replace(',', ''))
                        ncb_data['total_amount'] = amount
                        break
                    except ValueError:
                        continue
            
            return ncb_data
            
        except Exception as e:
            st.warning(f"Error processing screenshot {image_path}: {e}")
            return {
                'agent_ncb_amount': None,
                'dealer_ncb_amount': None,
                'total_amount': None,
                'has_agent_ncb': False,
                'has_dealer_ncb': False,
                'raw_text': ''
            }

class CancellationProcessor:
    def __init__(self):
        self.files_data = []
        self.packets = {}
        self.screenshot_processor = ScreenshotProcessor()
        
    def extract_text_from_file(self, file_path):
        """Extract text from various file types"""
        text = ""
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.pdf':
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
            elif file_ext in ['.docx']:
                doc = Document(file_path)
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
            elif file_ext in ['.doc']:
                # Fallback for .doc files
                with open(file_path, 'rb') as f:
                    text = f.read().decode('utf-8', errors='ignore')
            elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff']:
                # Check if it's a bucket screenshot
                if self.is_bucket_screenshot(file_path):
                    ncb_data = self.screenshot_processor.extract_ncb_data(file_path)
                    text = ncb_data['raw_text']
                else:
                    try:
                        image = Image.open(file_path)
                        text = pytesseract.image_to_string(image)
                    except Exception as e:
                        st.warning(f"OCR failed for {file_path}: {e}")
            elif file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
        except Exception as e:
            st.warning(f"Error extracting text from {file_path}: {e}")
            
        return text
    
    def is_bucket_screenshot(self, file_path):
        """Determine if image is likely a bucket screenshot"""
        try:
            image = Image.open(file_path)
            # Quick OCR to check for bucket-related keywords
            text = pytesseract.image_to_string(image, config='--psm 6')
            bucket_keywords = ['ncb', 'bucket', 'fee', 'agent', 'dealer', 'chargeback', 'pcmi']
            return any(keyword in text.lower() for keyword in bucket_keywords)
        except:
            return False
    
    def extract_fields(self, text, filename, file_path=None):
        """Extract all relevant fields from text using regex patterns"""
        fields = {
            'filename': filename,
            'vins': [],
            'contracts': [],
            'reasons': [],
            'cancellation_dates': [],
            'sale_dates': [],
            'refund_addresses': [],
            'mileages': [],
            'has_agent_ncb': False,
            'has_dealer_ncb': False,
            'is_autohouse': False,
            'is_customer_direct': False,
            'is_diversicare': False,
            'has_signature': False,
            'has_pcmi_hint': False,
            'is_lender_letter': False,
            'agent_ncb_amount': None,
            'dealer_ncb_amount': None,
            'total_ncb_amount': None
        }
        
        # VIN extraction
        vin_pattern = r'\b([A-HJ-NPR-Z0-9]{17})\b'
        fields['vins'] = re.findall(vin_pattern, text.upper())
        
        # Contract ID extraction
        gap_pattern = r'\bGAP\d{8}\b'
        common_prefix_pattern = r'\b(?:PN|EL|AD|OP|PT|PPM|EW|TW)\d{8}\b'
        generic_contract_pattern = r'(?:Contract\s*(?:No\.?|Number)?|CN|Contract#)\s*[:#]?\s*([A-Z]{1,3}\d{6,})'
        
        contracts = []
        contracts.extend(re.findall(gap_pattern, text.upper()))
        contracts.extend(re.findall(common_prefix_pattern, text.upper()))
        contracts.extend(re.findall(generic_contract_pattern, text.upper()))
        fields['contracts'] = list(set(contracts))  # Remove duplicates
        
        # Reason extraction
        reason_pattern = r'(?:vehicle\s+traded|trade\s*in|sold|repossession|total\s*loss|dealer\s*buyback|customer\s*request)'
        fields['reasons'] = re.findall(reason_pattern, text.lower())
        
        # Date extraction
        date_pattern = r'([0-1]?\d[\/\-][0-3]?\d[\/\-](?:20)?\d{2})'
        cancellation_date_pattern = r'(?:cancellation|cancel|effective)\s*date[ :\-]*' + date_pattern
        sale_date_pattern = r'(?:contract\s*sale|sale)\s*date[ :\-]*' + date_pattern
        
        fields['cancellation_dates'] = re.findall(cancellation_date_pattern, text, re.IGNORECASE)
        fields['sale_dates'] = re.findall(sale_date_pattern, text, re.IGNORECASE)
        
        # Refund address extraction
        address_pattern = r'(?:remit|send|mail)\s+(?:refund|check)\s+to[: ]\s*(.+?)(?:\n|$)'
        fields['refund_addresses'] = re.findall(address_pattern, text, re.IGNORECASE)
        
        # Mileage extraction
        mileage_pattern = r'(?:mileage|odom(?:eter)?)\s*[:#]?\s*([0-9]{1,6}(?:,[0-9]{3})?)'
        fields['mileages'] = re.findall(mileage_pattern, text, re.IGNORECASE)
        
        # Flag detection with partial-word matching
        fields['has_agent_ncb'] = bool(re.search(r'Agent\s+NCB|No\s*Chargeback', text, re.IGNORECASE))
        fields['has_dealer_ncb'] = bool(re.search(r'Dealer\s+NCB|No\s*Chargeback', text, re.IGNORECASE))
        fields['is_autohouse'] = bool(re.search(r'auto\s*house|autohouse', text, re.IGNORECASE))  # matches autohouse.com, AutoHouseLLC, Auto House, etc.
        fields['is_customer_direct'] = bool(re.search(r'customer\s+direct|dealer\s+out\s+of\s+business|\bFF\b', text, re.IGNORECASE))
        fields['is_diversicare'] = bool(re.search(r'diversicare', text, re.IGNORECASE))  # matches Diversicare, Diversicare-*, etc.
        fields['has_signature'] = bool(re.search(r'signature', text, re.IGNORECASE))
        fields['has_pcmi_hint'] = bool(re.search(r'pcmi|ncb\s*(fee|bucket)', text, re.IGNORECASE))
        fields['is_lender_letter'] = bool(re.search(r'lender\s+letter|payoff\s+letter|addressed\s+to\s+ascent', text, re.IGNORECASE))
        
        # Enhanced NCB amount extraction for bucket screenshots
        if file_path and self.is_bucket_screenshot(file_path):
            ncb_data = self.screenshot_processor.extract_ncb_data(file_path)
            fields['agent_ncb_amount'] = ncb_data['agent_ncb_amount']
            fields['dealer_ncb_amount'] = ncb_data['dealer_ncb_amount']
            fields['total_ncb_amount'] = ncb_data['total_amount']
            fields['has_agent_ncb'] = ncb_data['has_agent_ncb']
            fields['has_dealer_ncb'] = ncb_data['has_dealer_ncb']
        
        return fields
    
    def group_files_into_packets(self):
        """Group files into packets based on Contract ID or VIN"""
        for file_data in self.files_data:
            packet_key = None
            
            # Try Contract ID first
            if file_data['contracts']:
                packet_key = f"CONTRACT_{file_data['contracts'][0]}"
            # Fall back to VIN
            elif file_data['vins']:
                packet_key = f"VIN_{file_data['vins'][0]}"
            else:
                packet_key = "UNCLASSIFIED"
            
            if packet_key not in self.packets:
                self.packets[packet_key] = []
            
            self.packets[packet_key].append(file_data)
    
    def normalize_reason(self, reason):
        """Normalize reason to canonical form"""
        reason_map = {
            'vehicle traded': 'Vehicle Traded',
            'trade in': 'Trade In',
            'sold': 'Sold',
            'repossession': 'Repossession',
            'total loss': 'Total Loss',
            'dealer buyback': 'Dealer Buyback',
            'customer request': 'Customer Request'
        }
        return reason_map.get(reason.lower(), reason.title())
    
    def parse_date(self, date_str):
        """Parse date string in various formats"""
        formats = ['%m/%d/%Y', '%m-%d-%Y', '%m/%d/%y', '%m-%d-%y']
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None
    
    def reconcile_values(self, values):
        """Reconcile multiple values and return status and canonical value"""
        # Remove empty values and duplicates while preserving order
        clean_values = list(dict.fromkeys([v for v in values if v and v.strip()]))
        
        if len(clean_values) == 0:
            return 'INFO', ''
        elif len(clean_values) == 1:
            return 'PASS', clean_values[0]
        else:
            return 'FAIL', '; '.join(clean_values)
    
    def choose_cxl_date_with_lender_preference(self, files, all_cxl_dates):
        """
        If any record looks like a lender letter file, prefer its cancellation date.
        Otherwise fall back to reconcile(all_cxl_dates).
        Returns (status, chosen_value)
        """
        # Heuristic: lender-letter record has is_lender_letter True
        lender_dates = []
        for file_data in files:
            if file_data.get("is_lender_letter") and file_data.get("cancellation_dates"):
                lender_dates.extend(file_data["cancellation_dates"])

        lender_dates = [d for d in lender_dates if d and d.strip()]
        lender_dates = list(dict.fromkeys(lender_dates))  # de-dup, order-preserving

        if len(lender_dates) == 1:
            return ("PASS", lender_dates[0])
        elif len(lender_dates) > 1:
            # Conflicting dates even within lender letters
            return ("FAIL", "; ".join(lender_dates))

        # No lender-letter dates; use normal reconcile across all dates
        return self.reconcile_values(all_cxl_dates)
    
    def evaluate_packet(self, packet_key, files):
        """Evaluate a packet against QC checklist"""
        # Initialize result
        result = {
            'Packet Key': packet_key,
            'Files': ', '.join([f['filename'] for f in files])
        }
        
        # Collect all values from all files in packet
        all_vins = []
        all_contracts = []
        all_reasons = []
        all_cancellation_dates = []
        all_sale_dates = []
        all_refund_addresses = []
        all_mileages = []
        
        has_agent_ncb = False
        has_dealer_ncb = False
        is_autohouse = False
        is_customer_direct = False
        is_diversicare = False
        has_signature = False
        has_pcmi_hint = False
        is_lender_letter = False
        
        # NCB amounts from screenshots
        agent_ncb_amounts = []
        dealer_ncb_amounts = []
        total_ncb_amounts = []
        
        for file_data in files:
            all_vins.extend(file_data['vins'])
            all_contracts.extend(file_data['contracts'])
            all_reasons.extend(file_data['reasons'])
            all_cancellation_dates.extend(file_data['cancellation_dates'])
            all_sale_dates.extend(file_data['sale_dates'])
            all_refund_addresses.extend(file_data['refund_addresses'])
            all_mileages.extend(file_data['mileages'])
            
            has_agent_ncb = has_agent_ncb or file_data['has_agent_ncb']
            has_dealer_ncb = has_dealer_ncb or file_data['has_dealer_ncb']
            is_autohouse = is_autohouse or file_data['is_autohouse']
            is_customer_direct = is_customer_direct or file_data['is_customer_direct']
            is_diversicare = is_diversicare or file_data['is_diversicare']
            has_signature = has_signature or file_data['has_signature']
            has_pcmi_hint = has_pcmi_hint or file_data['has_pcmi_hint']
            is_lender_letter = is_lender_letter or file_data['is_lender_letter']
            
            # Collect NCB amounts
            if file_data['agent_ncb_amount'] is not None:
                agent_ncb_amounts.append(file_data['agent_ncb_amount'])
            if file_data['dealer_ncb_amount'] is not None:
                dealer_ncb_amounts.append(file_data['dealer_ncb_amount'])
            if file_data['total_ncb_amount'] is not None:
                total_ncb_amounts.append(file_data['total_ncb_amount'])
        
        # Remove duplicates and empty values
        all_vins = list(set([v for v in all_vins if v.strip()]))
        all_contracts = list(set([c for c in all_contracts if c.strip()]))
        all_reasons = list(set([r for r in all_reasons if r.strip()]))
        all_cancellation_dates = list(set([d for d in all_cancellation_dates if d.strip()]))
        all_sale_dates = list(set([d for d in all_sale_dates if d.strip()]))
        all_refund_addresses = list(set([a for a in all_refund_addresses if a.strip()]))
        all_mileages = list(set([m for m in all_mileages if m.strip()]))
        
        # VIN evaluation
        vin_status, vin_value = self.reconcile_values(all_vins)
        result['VIN match on all forms'] = vin_status
        result['VIN (canonical)'] = vin_value
        
        # Contract evaluation
        contract_status, contract_value = self.reconcile_values(all_contracts)
        result['Contract match on all forms and Google sheet'] = contract_status
        result['Contract (canonical)'] = contract_value
        
        # Reason evaluation
        reason_status, reason_value = self.reconcile_values(all_reasons)
        result['Reason match across all forms'] = reason_status
        if reason_value:
            result['Reason (canonical)'] = '; '.join([self.normalize_reason(r) for r in reason_value.split('; ')])
        else:
            result['Reason (canonical)'] = reason_value
        
        # Cancellation date evaluation with lender letter preference
        cxl_status, cxl_value = self.choose_cxl_date_with_lender_preference(files, all_cancellation_dates)
        result['Cancellation date match across all forms (favor lender letter if applicable)'] = cxl_status
        result['Cancellation Effective Date'] = cxl_value
        
        # Sale date
        if len(all_sale_dates) == 0:
            result['Sale Date'] = ''
        else:
            result['Sale Date'] = all_sale_dates[0]  # Take first one
        
        # 90-day check
        cxl_date = self.parse_date(result['Cancellation Effective Date'])
        sale_date = self.parse_date(result['Sale Date'])
        
        if cxl_date and sale_date:
            days_diff = (cxl_date - sale_date).days
            result['Is the cancellation effective date past 90 days from contract sale date?'] = 'Yes' if days_diff > 90 else 'No'
        else:
            result['Is the cancellation effective date past 90 days from contract sale date?'] = 'Unknown'
        
        # Enhanced NCB flags with amounts
        if agent_ncb_amounts:
            result['Is there an Agent NCB Fee?'] = f'Yes (${sum(agent_ncb_amounts):.2f})' if sum(agent_ncb_amounts) > 0 else 'No'
        else:
            result['Is there an Agent NCB Fee?'] = 'Yes' if has_agent_ncb else 'No'
            
        if dealer_ncb_amounts:
            result['Is there a Dealer NCB Fee?'] = f'Yes (${sum(dealer_ncb_amounts):.2f})' if sum(dealer_ncb_amounts) > 0 else 'No'
        else:
            result['Is there a Dealer NCB Fee?'] = 'Yes' if has_dealer_ncb else 'No'
        
        # Refund address
        if is_lender_letter and all_refund_addresses:
            result['Is there a different address to send the refund? (only if lender letter addressed to Ascent)'] = 'Yes'
            result['Alt Refund Address (if any)'] = '; '.join(all_refund_addresses[:2])  # First 2 addresses
        else:
            result['Is there a different address to send the refund? (only if lender letter addressed to Ascent)'] = 'No'
            result['Alt Refund Address (if any)'] = ''
        
        # Signatures
        result['All necessary signatures collected?'] = "Likely (detected 'Signature' text)" if has_signature else "Needs manual check"
        
        # Contract types
        result['Is this an Autohouse Contract?'] = 'Yes' if is_autohouse else 'No'
        result['Is this a customer direct cancellation? (Dealer Out of Business or FF contract)'] = 'Yes' if is_customer_direct else 'No'
        result['Is this a Diversicare contract?'] = 'Yes' if is_diversicare else 'No'
        
        # PCMI Screenshot with enhanced detection
        if has_pcmi_hint or agent_ncb_amounts or dealer_ncb_amounts:
            result['PCMI Screenshot (Of NCB fee buckets)'] = 'Present (hint detected)'
            if total_ncb_amounts:
                result['Total NCB Amount'] = f'${sum(total_ncb_amounts):.2f}'
        else:
            result['PCMI Screenshot (Of NCB fee buckets)'] = 'Not found'
            result['Total NCB Amount'] = ''
        
        # Mileage
        result['Mileage values found'] = '; '.join(all_mileages[:3]) if all_mileages else ''
        
        return result
    
    def process_zip(self, zip_file):
        """Process uploaded ZIP file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract ZIP
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Process all files
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.isfile(file_path):
                        text = self.extract_text_from_file(file_path)
                        if text.strip():  # Only process files with content
                            fields = self.extract_fields(text, file, file_path)
                            self.files_data.append(fields)
            
            # Group files into packets
            self.group_files_into_packets()
            
            # Evaluate each packet
            results = []
            for packet_key, files in self.packets.items():
                result = self.evaluate_packet(packet_key, files)
                results.append(result)
            
            return results

# Main app
def main():
    processor = CancellationProcessor()
    
    # Create two columns for uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Upload Cancellation Packet ZIP")
        uploaded_file = st.file_uploader(
            "Upload a ZIP file containing cancellation packet files",
            type=['zip'],
            help="Upload a ZIP file containing PDF, DOCX, DOC, image, or text files. Screenshots of bucket files will be automatically processed for NCB fee data.",
            key="zip_uploader"
        )
    
    with col2:
        st.subheader("📸 Upload Bucket Screenshot")
        uploaded_screenshot = st.file_uploader(
            "Upload a bucket screenshot for NCB fee analysis",
            type=['png', 'jpg', 'jpeg', 'tiff'],
            help="Upload a screenshot of the PCMI NCB fee buckets to extract fee amounts.",
            key="screenshot_uploader"
        )
    
    # Process individual screenshot if uploaded
    screenshot_data = None
    if uploaded_screenshot is not None:
        with st.spinner("Processing bucket screenshot..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_screenshot.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_screenshot.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Process the screenshot
                screenshot_data = processor.screenshot_processor.extract_ncb_data(tmp_file_path)
                
                # Clean up temp file
                os.unlink(tmp_file_path)
                
                st.success("Screenshot processed successfully!")
                
                # Display screenshot results
                st.subheader("📊 Screenshot Analysis Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Agent NCB Amount", f"${screenshot_data['agent_ncb_amount']:.2f}" if screenshot_data['agent_ncb_amount'] else "N/A")
                
                with col2:
                    st.metric("Dealer NCB Amount", f"${screenshot_data['dealer_ncb_amount']:.2f}" if screenshot_data['dealer_ncb_amount'] else "N/A")
                
                with col3:
                    st.metric("Total Amount", f"${screenshot_data['total_amount']:.2f}" if screenshot_data['total_amount'] else "N/A")
                
                # Show extracted text
                with st.expander("View Extracted Text"):
                    st.text(screenshot_data['raw_text'])
                    
            except Exception as e:
                st.error(f"Error processing screenshot: {str(e)}")
    
    # Process ZIP file if uploaded
    if uploaded_file is not None:
        with st.spinner("Processing files..."):
            try:
                results = processor.process_zip(uploaded_file)
                
                if results:
                    # Create DataFrame
                    df = pd.DataFrame(results)
                    
                    # Display results
                    st.success(f"✅ Processed {len(results)} packet(s) from {len(processor.files_data)} file(s)")
                    
                    # Summary statistics - more prominent
                    st.subheader("📊 QC Analysis Summary")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Total Packets", len(results))
                    
                    with col2:
                        pass_count = len([r for r in results if r['VIN match on all forms'] == 'PASS'])
                        st.metric("VIN Matches (PASS)", pass_count)
                    
                    with col3:
                        fail_count = len([r for r in results if r['VIN match on all forms'] == 'FAIL'])
                        st.metric("VIN Conflicts (FAIL)", fail_count)
                    
                    with col4:
                        screenshot_count = len([r for r in results if 'Present (hint detected)' in str(r.get('PCMI Screenshot (Of NCB fee buckets)', ''))])
                        st.metric("Screenshots Processed", screenshot_count)
                    
                    with col5:
                        info_count = len([r for r in results if r['VIN match on all forms'] == 'INFO'])
                        st.metric("Needs Review (INFO)", info_count)
                    
                    # Main results table - more prominent
                    st.subheader("📋 QC Checklist Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download buttons
                    st.subheader("💾 Download Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📄 Download CSV",
                            data=csv,
                            file_name=f"qc_cancellations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        json_data = df.to_json(orient='records', indent=2)
                        st.download_button(
                            label="📄 Download JSON",
                            data=json_data,
                            file_name=f"qc_cancellations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    # Show screenshot processing details
                    screenshot_files = [f for f in processor.files_data if f.get('agent_ncb_amount') is not None or f.get('dealer_ncb_amount') is not None]
                    if screenshot_files:
                        st.subheader("📸 Screenshot Processing Details")
                        screenshot_data = []
                        for file_data in screenshot_files:
                            screenshot_data.append({
                                'File': file_data['filename'],
                                'Agent NCB Amount': f"${file_data['agent_ncb_amount']:.2f}" if file_data['agent_ncb_amount'] else 'N/A',
                                'Dealer NCB Amount': f"${file_data['dealer_ncb_amount']:.2f}" if file_data['dealer_ncb_amount'] else 'N/A',
                                'Total Amount': f"${file_data['total_ncb_amount']:.2f}" if file_data['total_ncb_amount'] else 'N/A'
                            })
                        
                        if screenshot_data:
                            st.dataframe(pd.DataFrame(screenshot_data), use_container_width=True)
                    
                    # Detailed analysis for each packet
                    st.subheader("🔍 Detailed Packet Analysis")
                    for i, result in enumerate(results):
                        with st.expander(f"Packet {i+1}: {result['Packet Key']} ({result['Files']})"):
                            # Create a more detailed view
                            detail_cols = st.columns(2)
                            
                            with detail_cols[0]:
                                st.write("**Basic Information:**")
                                st.write(f"• VIN: {result.get('VIN (canonical)', 'N/A')}")
                                st.write(f"• Contract: {result.get('Contract (canonical)', 'N/A')}")
                                st.write(f"• Reason: {result.get('Reason (canonical)', 'N/A')}")
                                st.write(f"• Cancellation Date: {result.get('Cancellation Effective Date', 'N/A')}")
                                st.write(f"• Sale Date: {result.get('Sale Date', 'N/A')}")
                            
                            with detail_cols[1]:
                                st.write("**QC Status:**")
                                st.write(f"• VIN Match: {result.get('VIN match on all forms', 'N/A')}")
                                st.write(f"• Contract Match: {result.get('Contract match on all forms and Google sheet', 'N/A')}")
                                st.write(f"• Reason Match: {result.get('Reason match across all forms', 'N/A')}")
                                st.write(f"• Date Match: {result.get('Cancellation date match across all forms (favor lender letter if applicable)', 'N/A')}")
                                st.write(f"• 90+ Days: {result.get('Is the cancellation effective date past 90 days from contract sale date?', 'N/A')}")
                            
                            st.write("**Fees & Flags:**")
                            st.write(f"• Agent NCB: {result.get('Is there an Agent NCB Fee?', 'N/A')}")
                            st.write(f"• Dealer NCB: {result.get('Is there a Dealer NCB Fee?', 'N/A')}")
                            st.write(f"• Refund Address: {result.get('Is there a different address to send the refund? (only if lender letter addressed to Ascent)', 'N/A')}")
                            st.write(f"• Signatures: {result.get('All necessary signatures collected?', 'N/A')}")
                            st.write(f"• Autohouse: {result.get('Is this an Autohouse Contract?', 'N/A')}")
                            st.write(f"• Customer Direct: {result.get('Is this a customer direct cancellation? (Dealer Out of Business or FF contract)', 'N/A')}")
                            st.write(f"• Diversicare: {result.get('Is this a Diversicare contract?', 'N/A')}")
                            st.write(f"• PCMI Screenshot: {result.get('PCMI Screenshot (Of NCB fee buckets)', 'N/A')}")
                            
                            if result.get('Mileage values found'):
                                st.write(f"• Mileage: {result.get('Mileage values found', 'N/A')}")
                
                else:
                    st.warning("No valid files found in the ZIP archive.")
                    
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()
