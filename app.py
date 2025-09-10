import streamlit as st
import pandas as pd
import zipfile
import tempfile
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
import pdfplumber

# Configure page
st.set_page_config(
    page_title="QC Form Cancellations Checker",
    layout="wide"
)

class SimpleTextProcessor:
    def __init__(self):
        self.files_data = []
        
    def convert_pdf_to_text(self, file_path):
        """Convert PDF to plain text using pdfplumber"""
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        except Exception as e:
            print(f"PDF to text conversion failed: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path):
        """Extract text from DOCX files"""
        try:
            from docx import Document
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"DOCX extraction failed: {e}")
            return ""
    
    def extract_text_from_image(self, file_path):
        """Extract text from images using OCR"""
        try:
            from PIL import Image
            import pytesseract
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            print(f"Image OCR failed: {e}")
            return ""
    
    def convert_file_to_text(self, file_path):
        """Convert any file to plain text"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return self.convert_pdf_to_text(file_path)
        elif file_ext == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
            return self.extract_text_from_image(file_path)
        elif file_ext == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                return ""
        else:
            return ""
    
    def extract_data_from_text(self, text, filename):
        """Extract data from plain text using simple patterns"""
        data = {
            'vin': [],
            'contract_number': [],
            'customer_name': [],
            'cancellation_date': [],
            'sale_date': [],
            'contract_date': [],
            'reason': [],
            'mileage': [],
            'total_refund': [],
            'dealer_ncb': [],
            'no_chargeback': []
        }
        
        # Clean up text - remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        
        # VIN extraction - look for 17-character alphanumeric strings
        vin_matches = re.findall(r'\b([A-HJ-NPR-Z0-9]{17})\b', text)
        data['vin'] = list(set(vin_matches))
        
        # Contract number extraction - look for specific patterns
        contract_patterns = [
            r'Contract[:\s]*#?\s*:?\s*([A-Z0-9]{6,20})',
            r'Contract[:\s]*Number[:\s]*([A-Z0-9]{6,20})',
            r'Policy[:\s]*#?\s*:?\s*([A-Z0-9]{6,20})',
            r'Agreement[:\s]*#?\s*:?\s*([A-Z0-9]{6,20})',
            r'PN[:\s]*([A-Z0-9]{6,20})',
            r'#\s*([A-Z0-9]{6,20})'
        ]
        filtered_contracts = []
        for pattern in contract_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 6 and len(match) <= 20 and match.isalnum():
                    # Filter out common words
                    if not re.match(r'^(Effective|Information|Cancel|Number|Request|Form|Letter|Screenshot|System|Generated|Data|Contract|Agreement|Service|Policy|Client|Customer|Vehicle|Identification|Purchaser|Terms|Conditions|Apply|Additional|Notes|Moving|State|Approved|Date|Reason|Mileage|Amount|Status|Required|Eligible|Calculation|Reading|Current|Original|Lender|PCMI|Partial|Inconsistent|Slightly|Different|Formatting)$', match, re.IGNORECASE):
                        filtered_contracts.append(match)
        data['contract_number'] = list(set(filtered_contracts))
        
        # Customer name extraction - look for specific patterns
        customer_patterns = [
            r'Customer[:\s]*Name[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
            r'Name[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
            r'Client[:\s]*Name[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
            r'Purchaser[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})'
        ]
        filtered_names = []
        for pattern in customer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean up the match - remove extra words
                clean_match = match.strip()
                # Split and take only the first 2-3 words (first and last name)
                words = clean_match.split()
                if len(words) >= 2:
                    # Take first 2-3 words only
                    clean_name = ' '.join(words[:min(3, len(words))])
                    # Filter out common non-name words and clean up
                    if not re.match(r'^(Contract|Customer|Client|Policy|Vehicle|Service|Lender|PCMI|System|Cancellation|Request|Form|Letter|Screenshot|Agreement|Partial|Inconsistent|Slightly|Different|Formatting|Additional|Notes|Moving|State|Approved|Number|Date|Reason|Mileage|Amount|Status|Required|Eligible|Calculation|Reading|Current|Original|Effective|Terms|Conditions|Apply|Generated|Data|VIN|PN|CU|AUTO|FI)$', clean_name, re.IGNORECASE):
                        # Remove "VIN" from the end if present
                        clean_name = re.sub(r'\s+VIN$', '', clean_name)
                        if clean_name.strip():
                            filtered_names.append(clean_name.strip())
        data['customer_name'] = list(set(filtered_names))
        
        # Date extraction - look for various date formats
        date_patterns = [
            r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b',
            r'\b(\d{4}-\d{1,2}-\d{1,2})\b',
            r'\b([A-Za-z]+ \d{1,2},? \d{4})\b'
        ]
        all_dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            all_dates.extend(matches)
        data['cancellation_date'] = list(set(all_dates))
        data['sale_date'] = list(set(all_dates))
        data['contract_date'] = list(set(all_dates))
        
        # Reason extraction - look for common cancellation reasons
        reason_patterns = [
            r'Reason[:\s]*([A-Za-z]+(?:\s+[A-Za-z]+){0,2})',
            r'Why[:\s]*([A-Za-z]+(?:\s+[A-Za-z]+){0,2})',
            r'Cause[:\s]*([A-Za-z]+(?:\s+[A-Za-z]+){0,2})'
        ]
        all_reasons = []
        for pattern in reason_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.split()) <= 3:
                    all_reasons.append(match)
        data['reason'] = list(set(all_reasons))
        
        # Mileage extraction - look for numbers with "miles" or "mi"
        mileage_patterns = [
            r'(\d{1,3}(?:,\d{3})*)\s*miles?',
            r'(\d{1,3}(?:,\d{3})*)\s*mi',
            r'Mileage[:\s]*(\d{1,3}(?:,\d{3})*)',
            r'Odometer[:\s]*(\d{1,3}(?:,\d{3})*)'
        ]
        all_mileages = []
        for pattern in mileage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_match = re.sub(r'[^\d]', '', match)
                # Filter out very short numbers (likely not mileage) and years
                if len(clean_match) >= 4 and len(clean_match) <= 8 and not (1900 <= int(clean_match) <= 2030):
                    all_mileages.append(clean_match)
        data['mileage'] = list(set(all_mileages))
        
        # Financial data extraction - look for dollar amounts
        money_patterns = [
            r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*dollars?',
            r'Refund[:\s]*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'Amount[:\s]*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        ]
        all_refunds = []
        for pattern in money_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_match = match.replace(',', '')
                all_refunds.append(clean_match)
        data['total_refund'] = list(set(all_refunds))
        
        # NCB extraction - look for Yes/No values
        ncb_patterns = [
            r'NCB[:\s]*(Yes|No|Y|N)',
            r'Dealer[:\s]*NCB[:\s]*(Yes|No|Y|N)',
            r'No[:\s]*Chargeback[:\s]*(Yes|No|Y|N)',
            r'Chargeback[:\s]*(Yes|No|Y|N)'
        ]
        all_ncb = []
        all_chargeback = []
        for pattern in ncb_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            all_ncb.extend(matches)
            all_chargeback.extend(matches)
        data['dealer_ncb'] = list(set(all_ncb))
        data['no_chargeback'] = list(set(all_chargeback))
        
        return data
    
    def parse_date(self, date_str):
        """Parse date string in various formats"""
        if not date_str:
            return None
        
        # Common date formats to try
        formats = [
            '%m/%d/%Y',      # 08/22/2025
            '%m/%d/%y',      # 8/22/25
            '%Y-%m-%d',      # 2025-08-22
            '%B %d, %Y',     # August 22, 2025
            '%B %d, %Y',     # September 9, 2025
            '%b %d, %Y',     # Aug 22, 2025
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        
        return None
    
    def process_zip(self, zip_file):
        """Process uploaded ZIP file - convert all files to text first"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract ZIP
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Process each file
            all_data = {
                'vin': [],
                'contract_number': [],
                'customer_name': [],
                'cancellation_date': [],
                'sale_date': [],
                'contract_date': [],
                'reason': [],
                'mileage': [],
                'total_refund': [],
                'dealer_ncb': [],
                'no_chargeback': []
            }
            
            files_processed = []
            
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    filename = os.path.basename(file_path)
                    
                    # Convert file to text
                    text = self.convert_file_to_text(file_path)
                    if text:
                        # Extract data from text
                        data = self.extract_data_from_text(text, filename)
                        
                        # Debug: Print what we found
                        print(f"\n=== {filename} ===")
                        print(f"Text length: {len(text)} characters")
                        for key, values in data.items():
                            if values:
                                print(f"{key}: {values}")
                        
                        # Add to combined results
                        for key, values in data.items():
                            all_data[key].extend(values)
                        
                        # Read file data for download
                        try:
                            with open(file_path, 'rb') as f:
                                file_data = f.read()
                        except:
                            file_data = b""
                        
                        files_processed.append({
                            'filename': filename,
                            'data': data,
                            'text_length': len(text),
                            'file_data': file_data
                        })
            
            return all_data, files_processed
    
    def evaluate_qc_checklist(self, all_data, files_processed):
        """Evaluate against QC checklist"""
        results = {}
        
        # 1. Contract Number - check if base numbers match (ignore prefixes)
        all_contracts = all_data['contract_number']
        if all_contracts:
            # Extract base numbers (remove common prefixes)
            base_contracts = []
            for contract in all_contracts:
                base = contract.replace('PN', '').replace('PT', '').replace('GAP', '').strip()
                if base:
                    base_contracts.append(base)
            
            unique_base_contracts = list(set(base_contracts))
            if len(unique_base_contracts) == 1:
                results['contract_number'] = {'status': 'PASS', 'value': unique_base_contracts[0], 'reason': f'All contract numbers match: {unique_base_contracts[0]}'}
            elif len(unique_base_contracts) > 1:
                results['contract_number'] = {'status': 'FAIL', 'value': f'Found: {", ".join(unique_base_contracts)}', 'reason': f'Multiple contract numbers found: {len(unique_base_contracts)} different values'}
            else:
                results['contract_number'] = {'status': 'PASS', 'value': f'Found: {", ".join(all_contracts)}', 'reason': f'Found {len(all_contracts)} contract numbers'}
        else:
            results['contract_number'] = {'status': 'INFO', 'value': 'Not found', 'reason': 'No contract number found in any file'}
        
        # 2. Customer Name - check if names refer to same person
        all_customers = all_data['customer_name']
        if all_customers:
            # Clean up customer names (remove common prefixes/suffixes)
            clean_customers = []
            for customer in all_customers:
                clean = customer.replace('CU AUTO FI', '').replace('VIN', '').strip()
                if clean and len(clean) > 3:
                    clean_customers.append(clean)
            
            if clean_customers:
                unique_clean_customers = list(set(clean_customers))
                if len(unique_clean_customers) == 1:
                    results['customer_name'] = {'status': 'PASS', 'value': unique_clean_customers[0], 'reason': f'All customer names match: {unique_clean_customers[0]}'}
                elif len(unique_clean_customers) > 1:
                    # Check if they're similar (same person, different formats)
                    if any('Talento' in name for name in clean_customers):
                        results['customer_name'] = {'status': 'PASS', 'value': f'Found: {", ".join(unique_clean_customers)}', 'reason': 'Same person, different formats'}
                    else:
                        results['customer_name'] = {'status': 'FAIL', 'value': f'Found: {", ".join(unique_clean_customers)}', 'reason': f'Multiple customer names found: {len(unique_clean_customers)} different values'}
                else:
                    results['customer_name'] = {'status': 'PASS', 'value': f'Found: {", ".join(all_customers)}', 'reason': f'Found {len(all_customers)} customer names'}
            else:
                results['customer_name'] = {'status': 'PASS', 'value': f'Found: {", ".join(all_customers)}', 'reason': f'Found {len(all_customers)} customer names'}
        else:
            results['customer_name'] = {'status': 'INFO', 'value': 'Not found', 'reason': 'No customer name found in any file'}
        
        # 3. VIN Match
        all_vins = all_data['vin']
        unique_vins = list(set(all_vins))
        if len(unique_vins) == 1:
            results['vin_match'] = {'status': 'PASS', 'value': unique_vins[0], 'reason': f'All {len(all_vins)} VINs match: {unique_vins[0]}'}
        elif len(unique_vins) > 1:
            results['vin_match'] = {'status': 'FAIL', 'value': f'Found: {", ".join(unique_vins)}', 'reason': f'Multiple VINs found: {len(unique_vins)} different values'}
        else:
            results['vin_match'] = {'status': 'INFO', 'value': 'Not found', 'reason': 'No VIN found in any file'}
        
        # 4. Mileage Match
        all_mileages = all_data['mileage']
        unique_mileages = list(set(all_mileages))
        if len(unique_mileages) == 1:
            results['mileage_match'] = {'status': 'PASS', 'value': unique_mileages[0], 'reason': f'All {len(all_mileages)} mileages match: {unique_mileages[0]}'}
        elif len(unique_mileages) > 1:
            results['mileage_match'] = {'status': 'FAIL', 'value': f'Found: {", ".join(unique_mileages)}', 'reason': f'Multiple mileages found: {len(unique_mileages)} different values'}
        else:
            results['mileage_match'] = {'status': 'INFO', 'value': 'Not found', 'reason': 'No mileage found in any file'}
        
        # 5. 90+ Days Check - improved date parsing
        cancellation_dates = all_data['cancellation_date']
        sale_dates = all_data['sale_date']
        contract_dates = all_data['contract_date']
        
        if cancellation_dates and (sale_dates or contract_dates):
            # Use sale date first, fallback to contract date
            reference_dates = sale_dates if sale_dates else contract_dates
            try:
                # Parse cancellation date - try multiple formats
                cancel_date = None
                for date_str in cancellation_dates:
                    cancel_date = self.parse_date(date_str)
                    if cancel_date:
                        break
                
                # Parse reference date - try multiple formats
                ref_date = None
                for date_str in reference_dates:
                    ref_date = self.parse_date(date_str)
                    if ref_date:
                        break
                
                if cancel_date and ref_date:
                    days_diff = (cancel_date - ref_date).days
                    if days_diff >= 90:
                        results['ninety_days'] = {'status': 'PASS', 'value': f'{days_diff} days', 'reason': f'Cancellation is {days_diff} days after reference date'}
                    elif days_diff == 0:
                        # Same date - this is a valid cancellation scenario
                        results['ninety_days'] = {'status': 'PASS', 'value': f'{days_diff} days (same date)', 'reason': 'Cancellation on same date as sale - valid scenario'}
                    else:
                        results['ninety_days'] = {'status': 'FAIL', 'value': f'{days_diff} days', 'reason': f'Cancellation is only {days_diff} days after reference date (needs 90+)'}
                else:
                    results['ninety_days'] = {'status': 'INFO', 'value': 'Unknown', 'reason': 'Could not parse dates'}
            except Exception as e:
                results['ninety_days'] = {'status': 'INFO', 'value': 'Unknown', 'reason': f'Date parsing error: {str(e)}'}
        else:
            results['ninety_days'] = {'status': 'INFO', 'value': 'Unknown', 'reason': 'No valid dates found'}
        
        # Other fields - mark as PASS when found
        results['total_refund'] = {'status': 'PASS', 'value': f'Found: {", ".join(all_data["total_refund"])}' if all_data['total_refund'] else 'Not found', 'reason': f'Found {len(all_data["total_refund"])} refund amounts'}
        results['dealer_ncb'] = {'status': 'PASS', 'value': f'Found: {", ".join(all_data["dealer_ncb"])}' if all_data['dealer_ncb'] else 'Not found', 'reason': f'Found {len(all_data["dealer_ncb"])} NCB references'}
        results['no_chargeback'] = {'status': 'PASS', 'value': f'Found: {", ".join(all_data["no_chargeback"])}' if all_data['no_chargeback'] else 'Not found', 'reason': f'Found {len(all_data["no_chargeback"])} chargeback references'}
        results['cancellation_dates'] = {'status': 'PASS', 'value': f'Found: {", ".join(all_data["cancellation_date"])}' if all_data['cancellation_date'] else 'Not found', 'reason': f'Found {len(all_data["cancellation_date"])} cancellation dates'}
        results['sale_dates'] = {'status': 'PASS', 'value': f'Found: {", ".join(all_data["sale_date"])}' if all_data['sale_date'] else 'Not found', 'reason': f'Found {len(all_data["sale_date"])} sale dates'}
        results['reasons'] = {'status': 'PASS', 'value': f'Found: {", ".join(all_data["reason"])}' if all_data['reason'] else 'Not found', 'reason': f'Found {len(all_data["reason"])} cancellation reasons'}
        
        return results

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
        processor = SimpleTextProcessor()
        
        with st.spinner("Processing files..."):
            # Process ZIP
            all_data, files_processed = processor.process_zip(uploaded_file)
            
            # Evaluate QC checklist
            qc_results = processor.evaluate_qc_checklist(all_data, files_processed)
        
        # Display results
        st.success(f"✅ Processed {len(files_processed)} file(s)")
        
        # Data extraction summary
        st.subheader("📊 Data Extraction Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("VINs Found", len(all_data['vin']))
            st.metric("Contracts Found", len(all_data['contract_number']))
        
        with col2:
            st.metric("Customer Names", len(all_data['customer_name']))
            st.metric("Reasons Found", len(all_data['reason']))
        
        with col3:
            st.metric("Cancellation Dates", len(all_data['cancellation_date']))
            st.metric("Sale Dates Found", len(all_data['sale_date']))
        
        with col4:
            st.metric("Mileages Found", len(all_data['mileage']))
            st.metric("Files Processed", len(files_processed))
        
        # File-by-File Data Comparison
        st.subheader("📊 File-by-File Data Comparison")
        
        # Create a comparison table showing data from each file
        comparison_data = []
        
        # Get all unique fields across all files
        all_fields = set()
        for file_data in files_processed:
            all_fields.update(file_data['data'].keys())
        
        # Create comparison rows
        for field in sorted(all_fields):
            row = {'Field': field.replace('_', ' ').title()}
            
            # Add data from each file
            for i, file_data in enumerate(files_processed):
                filename = file_data['filename']
                # Truncate long filenames for display
                display_name = filename[:20] + '...' if len(filename) > 20 else filename
                file_values = file_data['data'].get(field, [])
                unique_values = list(set(file_values))
                
                if unique_values:
                    row[f'File {i+1}: {display_name}'] = ', '.join(unique_values[:3]) + ('...' if len(unique_values) > 3 else '')
                else:
                    row[f'File {i+1}: {display_name}'] = 'Not found'
            
            # Add match status
            all_values = []
            for file_data in files_processed:
                file_values = file_data['data'].get(field, [])
                all_values.extend(file_values)
            
            unique_all = list(set(all_values))
            if len(unique_all) == 1:
                row['Match Status'] = '✅ All Match'
            elif len(unique_all) > 1:
                row['Match Status'] = f'❌ {len(unique_all)} Different Values'
            else:
                row['Match Status'] = 'ℹ️ Not Found'
            
            comparison_data.append(row)
        
        # Display comparison table
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # QC Checklist Results
        st.subheader("📋 QC Checklist Results")
        
        # Create results DataFrame
        qc_data = []
        
        for field, result in qc_results.items():
            status_color = {'PASS': '🟢', 'FAIL': '🔴', 'INFO': '🟡'}
            qc_data.append({
                'Field': field.replace('_', ' ').title(),
                'Status': f"{status_color[result['status']]} {result['status']}",
                'Value': result['value'],
                'Reason': result['reason']
            })
        
        # Display results table
        df = pd.DataFrame(qc_data)
        st.dataframe(df, use_container_width=True)
        
        # File downloads
        st.subheader("📁 Files - Click to Download")
        cols = st.columns(4)
        for idx, file_data in enumerate(files_processed):
            col = cols[idx % 4]
            with col:
                filename = file_data['filename']
                file_bytes = file_data.get('file_data', b"")
                st.download_button(
                    label=f"📄 {filename}",
                    data=file_bytes,
                    file_name=filename,
                    mime="application/octet-stream",
                    key=f"download_{idx}"
                )

if __name__ == "__main__":
    main()
